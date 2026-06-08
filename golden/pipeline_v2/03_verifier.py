"""STEP 3 — Verifier.

gpt-5.4-mini (medium reasoning, flex) independently re-grades each of the 400
selected queries on MU/PU/ET, seeing ONLY the query text + the three rubrics —
no cluster context, no sight of the picker's label. A different model + a
"classify-only" framing makes picker↔verifier disagreement a meaningful signal
(same model + same prompt would agree ~always). Disagreement + any
`axis_applicable=false` flag drives the IAA-hard bucket in step 5.

Run (from repo root):
    python golden/pipeline_v2/03_verifier.py
    python golden/pipeline_v2/03_verifier.py --status

Outputs:
    golden/cache/golden_set_v1/verifier_400.parquet
    golden/cache/golden_set_v1/agreement_400.parquet
    golden/cache/golden_set_v1/state/03_verifier_state.json
    golden/cache/golden_set_v1/state/03_verifier_errors.jsonl
    golden/cache/golden_set_v1/reports/03_verifier_report.html
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from golden.pipeline_v2 import _io, _report  # noqa: E402
from golden.pipeline_v2._llm import (  # noqa: E402
    VERIFIER_MODEL, estimate_cost_usd, run_concurrent, verifier_client,
)
from golden.pipeline_v2.schemas import AXES, VerifierResponse  # noqa: E402

TEMPLATE_VERSION = "cluster_verifier_v1"
STATE_PATH = _io.STATE_DIR / "03_verifier_state.json"
ERRORS_PATH = _io.STATE_DIR / "03_verifier_errors.jsonl"
REPORT_PATH = _io.REPORTS_DIR / "03_verifier_report.html"

VERIFIER_COLUMNS = [
    "query_key", "query_text",
    "verifier_mu_grade", "verifier_pu_grade", "verifier_et_grade",
    "verifier_mu_axis_applicable", "verifier_pu_axis_applicable", "verifier_et_axis_applicable",
    "verifier_mu_confidence", "verifier_pu_confidence", "verifier_et_confidence",
    "verifier_notes_cs", "verifier_model", "verifier_run_at",
]


def build_prompt() -> tuple[str, str]:
    system, user = _io.load_prompt(_io.VERIFIER_PROMPT)
    for k, f in (("mu_rubric", "MU.md"), ("pu_rubric", "PU.md"), ("et_rubric", "ET.md")):
        system = system.replace("{" + k + "}", (_io.RUBRICS_DIR / f).read_text(encoding="utf-8").strip())
    return system, user


def make_worker(client, system: str, user_tmpl: str, run_at: str):
    async def worker(row: dict):
        user = user_tmpl.replace("{query_text}", row["query_text"])
        parsed, usage = await client.call_structured(
            phase="verifier", template_version=TEMPLATE_VERSION,
            system=system, user=user, schema_model=VerifierResponse,
            reasoning_effort="medium",
        )
        return [{
            "query_key": row["query_key"], "query_text": row["query_text"],
            "verifier_mu_grade": parsed.mu_grade, "verifier_pu_grade": parsed.pu_grade,
            "verifier_et_grade": parsed.et_grade,
            "verifier_mu_axis_applicable": parsed.mu_axis_applicable,
            "verifier_pu_axis_applicable": parsed.pu_axis_applicable,
            "verifier_et_axis_applicable": parsed.et_axis_applicable,
            "verifier_mu_confidence": parsed.mu_confidence,
            "verifier_pu_confidence": parsed.pu_confidence,
            "verifier_et_confidence": parsed.et_confidence,
            "verifier_notes_cs": parsed.notes_cs,
            "verifier_model": VERIFIER_MODEL, "verifier_run_at": run_at,
        }], usage
    return worker


def compute_agreement(verifier_df: pd.DataFrame, picker: pd.DataFrame) -> pd.DataFrame:
    """Join verifier grades to picker labels and derive agreement signals."""
    pk = picker[["query_key", "query_text", "clinical_relevance", "bucket",
                 "mu_grade", "pu_grade", "et_grade",
                 "mu_confidence", "pu_confidence", "et_confidence"]]
    df = verifier_df.merge(pk, on="query_key", suffixes=("", "_pk"))
    for ax in AXES:
        a = ax.lower()
        df[f"{a}_agree"] = df[f"{a}_grade"] == df[f"verifier_{a}_grade"]
    df["n_axes_agreement"] = df[[f"{ax.lower()}_agree" for ax in AXES]].sum(axis=1)
    df["any_not_applicable"] = ~(
        df["verifier_mu_axis_applicable"] & df["verifier_pu_axis_applicable"]
        & df["verifier_et_axis_applicable"])
    # Contradiction = verifier marks an axis N/A where the picker assigned a
    # NON-zero grade — a genuine "picker may have over-labeled" signal. (Plain
    # axis-N/A on a grade-0 axis is the normal case and carries no signal.)
    contra = None
    for ax in AXES:
        a = ax.lower()
        c = (df[f"{a}_grade"] != f"{ax}0") & (~df[f"verifier_{a}_axis_applicable"])
        contra = c.astype(int) if contra is None else contra + c.astype(int)
    df["contradiction_count"] = contra
    # IAA-hard = any grade disagreement (<=2/3 axes agree) OR a contradiction.
    # NOT "any axis N/A" — that fired on ~97% of queries and carried no signal.
    df["is_iaa_hard_candidate"] = (df["n_axes_agreement"] <= 2) | (df["contradiction_count"] > 0)
    return df


async def run(args) -> int:
    _io.ensure_dirs()
    if not _io.GOLDEN_400_PICKED.exists():
        print("[verifier] golden_400_picked.parquet missing — run step 2 first.", file=sys.stderr)
        return 2
    picked = pd.read_parquet(_io.GOLDEN_400_PICKED)
    picker = picked[picked["bucket"] != "spill_400_buffer"].reset_index(drop=True)

    system, user_tmpl = build_prompt()
    prompt_hash = _io.sha256_text(system + "\n@@@USER@@@\n" + user_tmpl)
    prev = _io.read_state(STATE_PATH)
    if prev and prev.get("prompt_hash_sha256") not in (None, prompt_hash) and not args.force:
        print("[verifier] prompt changed; refusing to resume (use --force).", file=sys.stderr)
        return 2

    prior_rows: list[dict] = []
    done: set = set()
    if _io.VERIFIER_400.exists():
        pdf = pd.read_parquet(_io.VERIFIER_400)
        prior_rows = pdf.to_dict("records")
        done = set(pdf["query_key"])
    todo = [r for r in picker.to_dict("records") if r["query_key"] not in done]
    print(f"[verifier] {len(picker)} queries, {len(done)} done, {len(todo)} to verify "
          f"(concurrency={args.concurrency})", file=sys.stderr)

    run_at = _io.now_iso()
    started_at = (prev or {}).get("started_at") or run_at
    client = verifier_client(concurrency=args.concurrency)
    worker = make_worker(client, system, user_tmpl, run_at)

    def write_artifacts(rows, stats, status):
        vdf = pd.DataFrame(prior_rows + rows, columns=VERIFIER_COLUMNS)
        _io.write_parquet_atomic(vdf, _io.VERIFIER_400)
        agree = compute_agreement(vdf, picker)
        _io.write_parquet_atomic(agree, _io.AGREEMENT_400)
        est = estimate_cost_usd(VERIFIER_MODEL, stats["prompt_tokens"], stats["completion_tokens"])
        state = {
            "started_at": started_at, "last_checkpoint_at": _io.now_iso(),
            "completed": len(vdf), "total": len(picker),
            "failed": stats.get("failed", 0), "current_status": status,
            "prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
            "est_cost_usd": round(est, 2),
            "prompt_hash_sha256": prompt_hash, "verifier_model": VERIFIER_MODEL,
        }
        _io.write_state(state, STATE_PATH)
        REPORT_PATH.write_text(render_report(agree, stats, state), encoding="utf-8")
        return est

    def on_checkpoint(rows, stats, n_done, n_total):
        write_artifacts(rows, stats, "running" if n_done < n_total else "completed")
        print(f"[verifier] checkpoint {n_done}/{n_total} | report refreshed", file=sys.stderr)

    t0 = time.monotonic()
    rows, stats = await run_concurrent(
        todo, worker, checkpoint_every=args.checkpoint_every,
        on_checkpoint=on_checkpoint, errors_path=ERRORS_PATH,
        item_id=lambda r: r["query_key"],
    )
    est = write_artifacts(rows, stats, "completed")
    elapsed = time.monotonic() - t0

    agree = pd.read_parquet(_io.AGREEMENT_400)
    full = int((agree["n_axes_agreement"] == 3).sum())
    hard = int(agree["is_iaa_hard_candidate"].sum())
    print(f"\n[verifier] DONE. verified={len(agree)}/{len(picker)} failed={stats['failed']} "
          f"full-agree={full} ({100*full/max(len(agree),1):.0f}%) iaa-hard={hard} "
          f"~${est:.2f} wall={elapsed:.0f}s", file=sys.stderr)
    print(f"[verifier] report: {REPORT_PATH}", file=sys.stderr)
    return 0


def render_report(agree: pd.DataFrame, stats: dict, state: dict) -> str:
    R = _report
    rng = random.Random(42)
    n = len(agree)

    # 1. agreement summary
    dist = agree["n_axes_agreement"].value_counts().to_dict() if n else {}
    summ = R.kv({
        "Verified": f"{n} / {state['total']}",
        "Full agreement (3/3 axes)": f"{dist.get(3,0)} ({100*dist.get(3,0)/max(n,1):.0f}%)",
        "2 / 3 axes": dist.get(2, 0), "1 / 3 axes": dist.get(1, 0), "0 / 3 axes": dist.get(0, 0),
        "IAA-hard candidates (disagree ≤2/3 OR contradiction)": int(agree["is_iaa_hard_candidate"].sum()) if n else 0,
        "Contradictions (picker non-zero, verifier N/A)": int((agree["contradiction_count"] > 0).sum()) if n else 0,
        "Any axis-not-applicable (informational only)": int(agree["any_not_applicable"].sum()) if n else 0,
        "Failed calls": stats.get("failed", 0),
        "Est. cost (flex)": f"${state['est_cost_usd']}",
    })

    # 2. per-axis confusion matrices (picker rows × verifier cols)
    confs = ""
    grades = {"MU": ["MU0", "MU1", "MU2"], "PU": ["PU0", "PU1", "PU2"], "ET": ["ET0", "ET1", "ET2"]}
    for ax in AXES:
        a = ax.lower()
        gs = grades[ax]
        head = "<tr><th>picker ↓ / verifier →</th>" + "".join(f"<th>{g}</th>" for g in gs) + "</tr>"
        body = ""
        for pg in gs:
            cells = ""
            for vg in gs:
                c = int(((agree[f"{a}_grade"] == pg) & (agree[f"verifier_{a}_grade"] == vg)).sum()) if n else 0
                style = ' style="background:#dff7df"' if pg == vg and c else ""
                cells += f"<td{style}>{c}</td>"
            body += f'<tr><td class="rowhdr">{pg}</td>{cells}</tr>'
        confs += f"<h3>{ax}</h3><table class='grid3'><thead>{head}</thead><tbody>{body}</tbody></table>"

    # 3. disagreement examples (sample)
    dis = agree[agree["n_axes_agreement"] <= 2] if n else agree
    idx = list(dis.index); rng.shuffle(idx)
    drows = []
    for i in idx[:30]:
        r = agree.loc[i]
        drows.append({
            "query_text": r["query_text"],
            "picker MU/PU/ET": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
            "verifier MU/PU/ET": f"{r['verifier_mu_grade']}/{r['verifier_pu_grade']}/{r['verifier_et_grade']}",
            "agree": int(r["n_axes_agreement"]), "verifier_notes_cs": r["verifier_notes_cs"],
        })
    dis_tbl = R.table(drows, ["query_text", "picker MU/PU/ET", "verifier MU/PU/ET", "agree", "verifier_notes_cs"])

    # 4. contradictions: verifier marks an axis N/A where the picker gave a
    # non-zero grade (possible picker over-labeling — the actionable N/A signal).
    na = agree[agree["contradiction_count"] > 0] if n else agree
    narows = []
    for _, r in na.head(40).iterrows():
        flags = [ax for ax in AXES
                 if (r[f"{ax.lower()}_grade"] != f"{ax}0") and not r[f"verifier_{ax.lower()}_axis_applicable"]]
        narows.append({"query_text": r["query_text"], "contradicted axes (picker≠0, verifier N/A)": ",".join(flags),
                       "picker MU/PU/ET": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
                       "verifier_notes_cs": r["verifier_notes_cs"]})
    na_tbl = R.table(narows, ["query_text", "contradicted axes (picker≠0, verifier N/A)", "picker MU/PU/ET", "verifier_notes_cs"])

    # 5. confidence comparison
    conf_rows = []
    if n:
        for label, mask in [("agreeing (3/3)", agree["n_axes_agreement"] == 3),
                            ("disagreeing (≤2)", agree["n_axes_agreement"] <= 2)]:
            sub = agree[mask]
            if len(sub):
                conf_rows.append({
                    "subset": label, "n": len(sub),
                    "picker mean conf": round(sub[["mu_confidence", "pu_confidence", "et_confidence"]].mean().mean(), 2),
                    "verifier mean conf": round(sub[["verifier_mu_confidence", "verifier_pu_confidence", "verifier_et_confidence"]].mean().mean(), 2),
                })
    conf_tbl = R.table(conf_rows, ["subset", "n", "picker mean conf", "verifier mean conf"])

    prov = R.provenance({
        "verifier_model": VERIFIER_MODEL,
        "prompt": _io.VERIFIER_PROMPT.name,
        "input": _io.GOLDEN_400_PICKED.name + " (selected 400)",
        "completed_at": state["last_checkpoint_at"],
    })

    return R.page(
        "Step 3 — Verifier report",
        R.section("1. Agreement summary", summ),
        R.section("2. Per-axis confusion matrix (picker × verifier)", confs),
        R.section("3. Disagreement examples (sample of ≤2/3 agreement)", dis_tbl),
        R.section("4. Contradictions (verifier N/A where picker assigned non-zero)", na_tbl),
        R.section("5. Confidence: picker vs verifier", conf_tbl),
        prov,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3 — verifier (gpt-5.4-mini flex)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()
    if args.status:
        _io.print_status(STATE_PATH)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
