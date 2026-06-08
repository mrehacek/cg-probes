"""STEP 6 — gpt-5.4 LLM reference annotator (regular tier, flex OFF).

Blind-grades all 400 final queries (golden_400_filled) on MU/PU/ET with gpt-5.4
medium reasoning on the REGULAR service tier (per user: flex OFF) — using the
same query-alone verifier prompt. This is the single, uniform gpt-5.4 reference
label set uploaded to the annotation app as a virtual annotator (llm_reference),
so model-vs-human shows up in the app's IAA tabs.

Output is keyed by `source` (golden-v1:<query_key>); 07_upload.py resolves
source -> queryId after the packages are imported and posts to
/admin/annotations/bulk.

Run:
    python golden/pipeline_v2/06_reference.py
    python golden/pipeline_v2/06_reference.py --status

Outputs:
    golden/cache/golden_set_v1/golden_gpt5.4_reference.parquet
    golden/cache/golden_set_v1/state/06_reference_state.json
    golden/cache/golden_set_v1/state/06_reference_errors.jsonl
    golden/cache/golden_set_v1/reports/06_reference_report.html
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from golden.pipeline_v2 import _io, _report  # noqa: E402
from golden.pipeline_v2._llm import (  # noqa: E402
    PICKER_MODEL, estimate_cost_usd, reference_client, run_concurrent,
)
from golden.pipeline_v2.schemas import AXES, VerifierResponse  # noqa: E402

TEMPLATE_VERSION = "cluster_verifier_v1"
ANNOTATOR_ID = "llm_reference"
STATE_PATH = _io.STATE_DIR / "06_reference_state.json"
ERRORS_PATH = _io.STATE_DIR / "06_reference_errors.jsonl"
REPORT_PATH = _io.REPORTS_DIR / "06_reference_report.html"
REFERENCE = _io.OUT_DIR / "golden_gpt5.4_reference.parquet"
MANIFEST = _io.OUT_DIR / "packs_manifest.parquet"

COLUMNS = ["source", "query_key", "query_text",
           "mu_grade", "pu_grade", "et_grade",
           "mu_confidence", "pu_confidence", "et_confidence",
           "notes_cs", "annotator_id", "model", "run_at"]


def build_prompt() -> tuple[str, str]:
    system, user = _io.load_prompt(_io.VERIFIER_PROMPT)
    for k, f in (("mu_rubric", "MU.md"), ("pu_rubric", "PU.md"), ("et_rubric", "ET.md")):
        system = system.replace("{" + k + "}", (_io.RUBRICS_DIR / f).read_text(encoding="utf-8").strip())
    return system, user


def make_worker(client, system, user_tmpl, run_at):
    async def worker(row: dict):
        user = user_tmpl.replace("{query_text}", row["query_text"])
        parsed, usage = await client.call_structured(
            phase="reference", template_version=TEMPLATE_VERSION, system=system, user=user,
            schema_model=VerifierResponse, reasoning_effort="medium")
        return [{
            "source": row["source"], "query_key": row["query_key"], "query_text": row["query_text"],
            "mu_grade": parsed.mu_grade, "pu_grade": parsed.pu_grade, "et_grade": parsed.et_grade,
            "mu_confidence": parsed.mu_confidence, "pu_confidence": parsed.pu_confidence,
            "et_confidence": parsed.et_confidence, "notes_cs": parsed.notes_cs,
            "annotator_id": ANNOTATOR_ID, "model": PICKER_MODEL, "run_at": run_at,
        }], usage
    return worker


async def run(args) -> int:
    _io.ensure_dirs()
    if not MANIFEST.exists():
        print("[reference] packs_manifest.parquet missing — run step 5 first.", file=sys.stderr)
        return 2
    man = pd.read_parquet(MANIFEST)[["source", "query_key", "query_text"]]

    prior_rows, done = [], set()
    if REFERENCE.exists():
        pdf = pd.read_parquet(REFERENCE)
        prior_rows, done = pdf.to_dict("records"), set(pdf["source"])
    todo = [r for r in man.to_dict("records") if r["source"] not in done]
    print(f"[reference] {len(man)} queries, {len(done)} done, {len(todo)} to grade "
          f"(gpt-5.4 medium, REGULAR tier, concurrency={args.concurrency})", file=sys.stderr)

    run_at = _io.now_iso()
    client = reference_client(concurrency=args.concurrency)
    worker = make_worker(client, *build_prompt(), run_at)

    def write_artifacts(rows, stats, status):
        df = pd.DataFrame(prior_rows + rows, columns=COLUMNS)
        _io.write_parquet_atomic(df, REFERENCE)
        est = estimate_cost_usd(PICKER_MODEL, stats["prompt_tokens"], stats["completion_tokens"], flex=False)
        state = {"run_at": run_at, "last_checkpoint_at": _io.now_iso(),
                 "completed": len(df), "total": len(man), "failed": stats.get("failed", 0),
                 "current_status": status, "annotator_id": ANNOTATOR_ID, "model": PICKER_MODEL,
                 "service_tier": "regular", "reasoning_effort": "medium",
                 "prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
                 "est_cost_usd": round(est, 2)}
        _io.write_state(state, STATE_PATH)
        REPORT_PATH.write_text(render_report(df, state), encoding="utf-8")
        return est

    def on_checkpoint(rows, stats, n, total):
        write_artifacts(rows, stats, "running" if n < total else "completed")
        print(f"[reference] checkpoint {n}/{total} | report refreshed", file=sys.stderr)

    t0 = time.monotonic()
    rows, stats = await run_concurrent(todo, worker, checkpoint_every=args.checkpoint_every,
                                       on_checkpoint=on_checkpoint, errors_path=ERRORS_PATH,
                                       item_id=lambda r: r["source"])
    est = write_artifacts(rows, stats, "completed")
    print(f"\n[reference] DONE. graded={len(prior_rows)+len(rows)}/{len(man)} failed={stats['failed']} "
          f"~${est:.2f} wall={time.monotonic()-t0:.0f}s", file=sys.stderr)
    print(f"[reference] report: {REPORT_PATH}", file=sys.stderr)
    return 0


def render_report(df, state) -> str:
    R = _report
    per_axis = {ax: {g: int((df[ax.lower() + "_grade"] == g).sum()) for g in (f"{ax}0", f"{ax}1", f"{ax}2")}
                for ax in AXES} if len(df) else {ax: {} for ax in AXES}
    head = R.kv({
        "Annotator id": state["annotator_id"], "Model": state["model"],
        "Tier / reasoning": f"{state['service_tier']} / {state['reasoning_effort']}",
        "Graded": f"{state['completed']} / {state['total']}", "Failed": state["failed"],
        "Est. cost (regular tier)": f"${state['est_cost_usd']}",
    })
    # compare reference vs our pipeline labels (from manifest) if available
    cmp_html = ""
    if MANIFEST.exists() and len(df):
        man = pd.read_parquet(MANIFEST)
        m = df.merge(man[["source", "mu_grade", "pu_grade", "et_grade"]], on="source", suffixes=("_ref", "_ours"))
        rows = []
        for ax in AXES:
            a = ax.lower()
            agree = float((m[f"{a}_grade_ref"] == m[f"{a}_grade_ours"]).mean())
            rows.append({"axis": ax, "reference vs pipeline-label agreement": f"{100*agree:.0f}%"})
        cmp_html = R.table(rows, ["axis", "reference vs pipeline-label agreement"])
    return R.page(
        "Step 6 — gpt-5.4 reference annotator",
        R.section("1. Run summary", head),
        R.section("2. Reference grade distribution", R.grade_grid(per_axis)),
        R.section("3. Reference vs pipeline-label agreement", cmp_html or "<p class='muted'>n/a</p>"),
        R.provenance({"prompt": _io.VERIFIER_PROMPT.name, "completed_at": state["last_checkpoint_at"]}),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 6 — gpt-5.4 reference annotator (regular tier)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    args = ap.parse_args()
    if args.status:
        _io.print_status(STATE_PATH)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
