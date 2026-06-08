"""STEP 4 — Backfill synthesis (conditional).

Triggered when a safety-critical cell (MU2 / PU2) is short after step 2. For
each short cell, synthesize realistic Czech patient queries from topically
relevant clusters (MU2 from MU1-bearing clusters; PU2 from PU1-bearing), then
INDEPENDENTLY VERIFY each synthetic with gpt-5.4-mini and keep only those the
verifier confirms at the target grade — this is the plausibility filter the PLAN
asks for, and it supplies the other two axis labels. Kept synthetics are swapped
in for the lowest-confidence abundant (MU0/PU0) picks so the set stays at 400
and no real safety content is dropped.

Per the user decision (2026-06-05) the PLAN's 40% synthetic cap is waived: cells
are filled toward target (MU2=80, PU2=60), so these cells become
majority-synthetic — unavoidable given ~0 real supply. Every synthetic is marked
source=synthesized and listed in the report for review.

Run (from repo root):
    python golden/pipeline_v2/04_backfill.py
    python golden/pipeline_v2/04_backfill.py --status
    python golden/pipeline_v2/04_backfill.py --max-clusters 30 --per-cluster 4

Outputs:
    golden/cache/golden_set_v1/golden_400_filled.parquet
    golden/cache/golden_set_v1/state/04_backfill_state.json
    golden/cache/golden_set_v1/state/04_backfill_errors.jsonl
    golden/cache/golden_set_v1/reports/04_backfill_report.html
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

from contrastive.textnorm import normalize  # noqa: E402
from golden.pipeline_v2 import _io, _report  # noqa: E402
from golden.pipeline_v2._llm import (  # noqa: E402
    BACKFILL_MODEL, VERIFIER_MODEL, backfill_client, estimate_cost_usd,
    run_concurrent, verifier_client,
)
from golden.pipeline_v2.schemas import (  # noqa: E402
    AXES, SynthesizerResponse, VerifierResponse,
)

SYNTH_TEMPLATE = "backfill_synthesizer_v1"
VERIFY_TEMPLATE = "cluster_verifier_v1"
STATE_PATH = _io.STATE_DIR / "04_backfill_state.json"
ERRORS_PATH = _io.STATE_DIR / "04_backfill_errors.jsonl"
REPORT_PATH = _io.REPORTS_DIR / "04_backfill_report.html"

# cell -> (target axis, anchor grade used to find topically-relevant clusters)
BACKFILL_CELLS = {"MU2": ("MU", "MU1"), "PU2": ("PU", "PU1")}
TARGET = {"MU2": 80, "PU2": 60}


def load_rubric_system(prompt_path: Path) -> tuple[str, str]:
    system, user = _io.load_prompt(prompt_path)
    for k, f in (("mu_rubric", "MU.md"), ("pu_rubric", "PU.md"), ("et_rubric", "ET.md")):
        system = system.replace("{" + k + "}", (_io.RUBRICS_DIR / f).read_text(encoding="utf-8").strip())
    return system, user


def relevant_clusters(picks: pd.DataFrame, anchor_grade: str, axis: str, max_clusters: int):
    """Clusters that contain anchor-grade picks, ranked by anchor count × weight."""
    col = axis.lower() + "_grade"
    anchored = picks[picks[col] == anchor_grade]
    if anchored.empty:
        return []
    g = (anchored.groupby("supercluster_id")
         .agg(n_anchor=("query_text", "size"),
              clinical_relevance=("clinical_relevance", "first"),
              czech_label=("czech_label", "first"))
         .reset_index())
    wmap = {"oncology-core": 1.0, "oncology-adjacent": 0.7, "navigational": 0.3, "non-clinical": 0.0}
    g["w"] = g["clinical_relevance"].map(wmap).fillna(0.0)
    g = g.sort_values(["w", "n_anchor"], ascending=False).head(max_clusters)
    return g["supercluster_id"].tolist()


async def synthesize(picks, summaries, cells, args):
    """Generate synthetic queries for each (short cell, relevant cluster)."""
    system, user_tmpl = load_rubric_system(_io.SYNTH_PROMPT)
    # {n_requested} appears in BOTH the SYSTEM rules and the USER message; it's a
    # constant, so resolve it in the system prefix here (worker resolves the user copy).
    system = system.replace("{n_requested}", str(args.per_cluster))
    client = backfill_client(concurrency=args.concurrency)

    tasks = []  # (cell, axis, grade, cluster_id)
    for cell, (axis, anchor) in cells.items():
        for cid in relevant_clusters(picks, anchor, axis, args.max_clusters):
            tasks.append((cell, axis, cell, cid))

    def anchors_for(cid, axis, anchor):
        col = axis.lower() + "_grade"
        sub = picks[(picks["supercluster_id"] == cid) & (picks[col] == anchor)]
        return sub["query_text"].head(10).tolist()

    async def worker(task):
        cell, axis, grade, cid = task
        anchor = cells[cell][1]
        examples = anchors_for(cid, axis, anchor)
        meta = summaries.get(cid, {})
        user = (user_tmpl
                .replace("{target_axis}", axis)
                .replace("{target_grade}", grade)
                .replace("{n_requested}", str(args.per_cluster))
                .replace("{cluster_czech_label}", str(meta.get("czech_label", "")))
                .replace("{cluster_summary_cs}", str(meta.get("summary_cs", "")))
                .replace("{example_queries}", "\n".join(f'- "{q}"' for q in examples)))
        parsed, usage = await client.call_structured(
            phase="backfill", template_version=SYNTH_TEMPLATE, system=system, user=user,
            schema_model=SynthesizerResponse, reasoning_effort="medium")
        rows = [{
            "query_text": g.query_text, "target_grade": g.target_grade, "target_axis": g.target_axis,
            "synthesis_cluster_id": cid, "clinical_relevance": meta.get("clinical_relevance", ""),
            "czech_label": meta.get("czech_label", ""),
            "synthesis_rationale_cs": g.rationale_cs, "synthesis_plausibility_cs": g.clinical_plausibility_cs,
        } for g in parsed.generated]
        return rows, usage

    rows, stats = await run_concurrent(tasks, worker, checkpoint_every=10_000,
                                       errors_path=ERRORS_PATH, item_id=lambda t: f"{t[0]}:{t[3]}")
    return rows, stats


async def verify_synthetics(cands, args):
    """Blind-grade each synthetic; keep those confirmed at the target grade."""
    system, user_tmpl = load_rubric_system(_io.VERIFIER_PROMPT)
    client = verifier_client(concurrency=args.concurrency)
    run_at = _io.now_iso()

    async def worker(c):
        user = user_tmpl.replace("{query_text}", c["query_text"])
        parsed, usage = await client.call_structured(
            phase="verifier", template_version=VERIFY_TEMPLATE, system=system, user=user,
            schema_model=VerifierResponse, reasoning_effort="medium")
        vg = {"MU": parsed.mu_grade, "PU": parsed.pu_grade, "ET": parsed.et_grade}
        confirmed = vg[c["target_axis"]] == c["target_grade"]
        return [{
            **c, "mu_grade": parsed.mu_grade, "pu_grade": parsed.pu_grade, "et_grade": parsed.et_grade,
            "mu_confidence": parsed.mu_confidence, "pu_confidence": parsed.pu_confidence,
            "et_confidence": parsed.et_confidence, "verifier_notes_cs": parsed.notes_cs,
            "confirmed": confirmed, "verify_run_at": run_at,
        }], usage

    rows, stats = await run_concurrent(cands, worker, checkpoint_every=10_000,
                                       errors_path=ERRORS_PATH, item_id=lambda c: c["query_text"][:40])
    return rows, stats


def build_filled(picker400: pd.DataFrame, kept: list[dict]) -> pd.DataFrame:
    """Swap kept synthetics in for lowest-confidence abundant (MU0/PU0) picks."""
    base = picker400.copy()
    base["source"] = "picked"
    for c in ("synthesis_cluster_id", "synthesis_rationale_cs", "synthesis_plausibility_cs"):
        base[c] = None

    n = len(kept)
    if "min_conf" not in base.columns:
        base["min_conf"] = base[["mu_confidence", "pu_confidence", "et_confidence"]].min(axis=1)
    if "category_weight" not in base.columns:
        base["category_weight"] = 0.0
    # removable = pure-abundant (MU0 & PU0), never a rare safety cell. The MU0&PU0
    # pool happens to hold only ET0 + ET2 picks (ET1 sits with MU1 queries). The
    # synthetics already re-supply ET2 (the PU2 distress queries verify as ET2) but
    # almost no ET0, so we drop ET2 from the pool FIRST and protect ET0 (which has
    # no synthetic source) — otherwise ET0 collapses to near zero. Lowest conf first.
    et_rank = {"ET2": 0, "ET1": 1, "ET0": 2}
    removable = base[(base["mu_grade"] == "MU0") & (base["pu_grade"] == "PU0")].copy()
    removable["_et_rank"] = removable["et_grade"].map(et_rank).fillna(1)
    removable = removable.sort_values(["_et_rank", "min_conf", "category_weight"], ascending=True)
    drop_keys = set(removable["query_key"].head(n))
    base = base[~base["query_key"].isin(drop_keys)].copy()

    syn_rows = []
    for c in kept:
        syn_rows.append({
            "supercluster_id": c["synthesis_cluster_id"],
            "clinical_relevance": c.get("clinical_relevance", ""),
            "czech_label": c.get("czech_label", ""),
            "query_text": c["query_text"], "query_key": normalize(c["query_text"]),
            "mu_grade": c["mu_grade"], "pu_grade": c["pu_grade"], "et_grade": c["et_grade"],
            "mu_confidence": c["mu_confidence"], "pu_confidence": c["pu_confidence"],
            "et_confidence": c["et_confidence"],
            "notes_cs": c.get("verifier_notes_cs", ""),
            "bucket": "synthetic", "source": "synthesized",
            "synthesis_cluster_id": c["synthesis_cluster_id"],
            "synthesis_rationale_cs": c["synthesis_rationale_cs"],
            "synthesis_plausibility_cs": c["synthesis_plausibility_cs"],
        })
    syn = pd.DataFrame(syn_rows)
    out = pd.concat([base, syn], ignore_index=True)
    return out, drop_keys


async def run(args) -> int:
    _io.ensure_dirs()
    if not _io.GOLDEN_400_PICKED.exists():
        print("[backfill] golden_400_picked.parquet missing — run step 2 first.", file=sys.stderr)
        return 2
    shortfall = _io.read_state(_io.STATE_DIR / "02_shortfall.json") or {}
    picked = pd.read_parquet(_io.GOLDEN_400_PICKED)
    picker400 = picked[picked["bucket"] != "spill_400_buffer"].reset_index(drop=True)
    picks_raw = pd.read_parquet(_io.PICKS_RAW)

    # trigger
    needed = {}
    for cell, (axis, _) in BACKFILL_CELLS.items():
        sf = shortfall.get(axis, {}).get(cell, 0)
        if sf > 0:
            needed[cell] = sf
    if not needed:
        print("[backfill] no MU2/PU2 shortfall — pass-through (copy picked -> filled).", file=sys.stderr)
        out = picker400.copy(); out["source"] = "picked"
        for c in ("synthesis_cluster_id", "synthesis_rationale_cs", "synthesis_plausibility_cs"):
            out[c] = None
        _io.write_parquet_atomic(out, _io.GOLDEN_400_FILLED)
        _io.write_state({"current_status": "completed", "triggered": False,
                         "completed_at": _io.now_iso()}, STATE_PATH)
        return 0
    print(f"[backfill] triggered. needed: {needed}", file=sys.stderr)

    # cluster summaries for the synth prompt
    summaries = {c["supercluster_id"]: c for c in _io.load_superclusters()}

    t0 = time.monotonic()
    gen_rows, gstats = await synthesize(picks_raw, summaries, {k: BACKFILL_CELLS[k] for k in needed}, args)
    # dedup synthetics vs each other and vs real picks
    real_keys = set(picks_raw["query_key"])
    seen, cands = set(), []
    for r in gen_rows:
        k = normalize(r["query_text"])
        if k in real_keys or k in seen:
            continue
        seen.add(k); cands.append(r)
    print(f"[backfill] generated {len(gen_rows)} -> {len(cands)} unique candidates; verifying...", file=sys.stderr)

    ver_rows, vstats = await verify_synthetics(cands, args)
    confirmed = [r for r in ver_rows if r["confirmed"]]
    # keep up to `needed` per cell, highest target-axis confidence first
    kept = []
    for cell, want in needed.items():
        axis = BACKFILL_CELLS[cell][0]
        pool = [r for r in confirmed if r["target_grade"] == cell]
        pool.sort(key=lambda r: r[f"{axis.lower()}_confidence"], reverse=True)
        kept.extend(pool[:want])
    print(f"[backfill] confirmed {len(confirmed)}/{len(cands)}; keeping {len(kept)}", file=sys.stderr)

    filled, dropped = build_filled(picker400, kept)
    _io.write_parquet_atomic(filled, _io.GOLDEN_400_FILLED)

    elapsed = time.monotonic() - t0
    pt = gstats["prompt_tokens"] + vstats["prompt_tokens"]
    ct = gstats["completion_tokens"] + vstats["completion_tokens"]
    est = (estimate_cost_usd(BACKFILL_MODEL, gstats["prompt_tokens"], gstats["completion_tokens"])
           + estimate_cost_usd(VERIFIER_MODEL, vstats["prompt_tokens"], vstats["completion_tokens"]))

    kept_per_cell = {cell: sum(1 for r in kept if r["target_grade"] == cell) for cell in needed}
    state = {
        "completed_at": _io.now_iso(), "current_status": "completed", "triggered": True,
        "needed": needed, "generated": len(gen_rows), "unique_candidates": len(cands),
        "confirmed": len(confirmed), "kept": len(kept), "kept_per_cell": kept_per_cell,
        "dropped_picks": len(dropped), "final_n": len(filled),
        "synth_models": {"generator": BACKFILL_MODEL, "verifier": VERIFIER_MODEL},
        "prompt_tokens": pt, "completion_tokens": ct, "est_cost_usd": round(est, 2),
        "wall_sec": round(elapsed, 1),
    }
    _io.write_state(state, STATE_PATH)
    REPORT_PATH.write_text(render_report(filled, kept, ver_rows, state), encoding="utf-8")

    print(f"\n[backfill] DONE. kept {kept_per_cell}, dropped {len(dropped)} picks, "
          f"final={len(filled)} ~${est:.2f} wall={elapsed:.0f}s", file=sys.stderr)
    print(f"[backfill] report: {REPORT_PATH}", file=sys.stderr)
    return 0


def render_report(filled, kept, ver_rows, state) -> str:
    R = _report
    syn = filled[filled["source"] == "synthesized"]

    trig = R.kv({
        "Triggered": state["triggered"], "Shortfall (needed)": str(state["needed"]),
        "Generated": state["generated"], "Unique candidates": state["unique_candidates"],
        "Verifier-confirmed at target": state["confirmed"], "Kept": state["kept"],
        "Kept per cell": str(state["kept_per_cell"]), "Picks demoted": state["dropped_picks"],
        "Final set size": state["final_n"],
        "Models": f"gen={state['synth_models']['generator']} / verify={state['synth_models']['verifier']}",
        "Est. cost (flex)": f"${state['est_cost_usd']}", "Wall": f"{state['wall_sec']}s",
    })

    # real vs synthetic per cell, post-backfill
    sumrows = []
    for ax in AXES:
        col = ax.lower() + "_grade"
        for g in (f"{ax}0", f"{ax}1", f"{ax}2"):
            cell = filled[filled[col] == g]
            n_syn = int((cell["source"] == "synthesized").sum())
            sumrows.append({"cell": g, "total": len(cell), "real": len(cell) - n_syn,
                            "synthetic": n_syn,
                            "synthetic %": f"{100*n_syn/max(len(cell),1):.0f}%"})
    sum_tbl = R.table(sumrows, ["cell", "total", "real", "synthetic", "synthetic %"])

    # EVERY synthetic query — the critical eyeball
    synrows = [{
        "target": r["target_grade"], "query_text": r["query_text"],
        "MU/PU/ET (verifier)": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
        "source cluster": r["czech_label"],
        "rationale_cs": r["synthesis_rationale_cs"],
        "plausibility_cs": r["synthesis_plausibility_cs"],
    } for r in kept]
    syn_tbl = R.table(synrows, ["target", "query_text", "MU/PU/ET (verifier)", "source cluster",
                                "rationale_cs", "plausibility_cs"])

    # rejected synthetics (verifier did NOT confirm target) — transparency
    rej = [r for r in ver_rows if not r["confirmed"]]
    rejrows = [{"target": r["target_grade"], "query_text": r["query_text"],
                "verifier MU/PU/ET": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
                "verifier_notes_cs": r["verifier_notes_cs"]} for r in rej[:40]]
    rej_tbl = R.table(rejrows, ["target", "query_text", "verifier MU/PU/ET", "verifier_notes_cs"])

    # final distribution vs target
    distrows = []
    TGT = {"MU": {"MU0": 160, "MU1": 160, "MU2": 80}, "PU": {"PU0": 200, "PU1": 140, "PU2": 60},
           "ET": {"ET0": 80, "ET1": 180, "ET2": 140}}
    for ax in AXES:
        col = ax.lower() + "_grade"
        for g in TGT[ax]:
            got = int((filled[col] == g).sum()); tgt = TGT[ax][g]
            distrows.append({"cell": g, "target": tgt, "achieved": got,
                             "status": "OK" if got >= tgt else f"SHORT {tgt-got}"})
    dist_tbl = R.table(distrows, ["cell", "target", "achieved", "status"])

    prov = R.provenance({
        "generator": state["synth_models"]["generator"], "verifier": state["synth_models"]["verifier"],
        "synth_prompt": _io.SYNTH_PROMPT.name, "completed_at": state["completed_at"],
    })

    return R.page(
        "Step 4 — Backfill synthesis report",
        R.section("1. Trigger decision", trig),
        R.section("2. Real vs synthetic per cell (post-backfill)", sum_tbl),
        R.section("3. EVERY kept synthetic query — READ THESE (annotator-visible)", syn_tbl),
        R.section("4. Rejected synthetics (verifier did not confirm target)", rej_tbl),
        R.section("5. Final per-axis distribution vs target", dist_tbl),
        prov,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 4 — backfill synthesis (conditional)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--max-clusters", type=int, default=30, help="relevant clusters per short cell")
    ap.add_argument("--per-cluster", type=int, default=4, help="queries requested per cluster")
    args = ap.parse_args()
    if args.status:
        _io.print_status(STATE_PATH)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
