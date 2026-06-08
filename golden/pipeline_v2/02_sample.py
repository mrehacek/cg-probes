"""STEP 2 — Sampling (local, no LLM).

Greedy constraint sampler: pick 400 distinct queries from picks_raw that best
match the per-axis marginal targets (PLAN §5), under a ≤5-per-supercluster cap,
preferring oncology-core and high-confidence picks. Because every row carries a
grade on all three axes simultaneously, the three marginals cannot all be hit
exactly — scarce cells (MU2, PU2, PU1 given the picker supply) are filled
maximally and the residual shortfall is recorded for step 4 backfill.

The 400 are split into core_300 (higher confidence, protected) and
iaa_reservoir_100 (lowest confidence — ambiguous, demotable when step 4 swaps in
synthetic MU2/PU2). A small spill buffer of extra candidates is also emitted.

Run (from repo root):
    python golden/pipeline_v2/02_sample.py
    python golden/pipeline_v2/02_sample.py --status

Outputs:
    golden/cache/golden_set_v1/golden_400_picked.parquet
    golden/cache/golden_set_v1/state/02_shortfall.json
    golden/cache/golden_set_v1/state/02_sample_state.json
    golden/cache/golden_set_v1/reports/02_sampling_report.html
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from golden.pipeline_v2 import _io, _report  # noqa: E402
from golden.pipeline_v2.schemas import AXES  # noqa: E402

STATE_PATH = _io.STATE_DIR / "02_sample_state.json"
SHORTFALL_PATH = _io.STATE_DIR / "02_shortfall.json"
REPORT_PATH = _io.REPORTS_DIR / "02_sampling_report.html"

# Per-axis marginal targets (PLAN §5). Each sums to 400; selecting 400 rows sums
# to 400 per axis, so these are simultaneous marginal goals on one row-set.
TARGETS = {
    "MU": {"MU0": 160, "MU1": 160, "MU2": 80},
    "PU": {"PU0": 200, "PU1": 140, "PU2": 60},
    "ET": {"ET0": 80, "ET1": 180, "ET2": 140},
}
CATEGORY_WEIGHT = {
    "oncology-core": 1.0, "oncology-adjacent": 0.7,
    "navigational": 0.3, "non-clinical": 0.0,
}

N_TOTAL = 400
N_RESERVOIR = 100          # of the 400, the lowest-confidence = IAA-hard reservoir
N_CORE = N_TOTAL - N_RESERVOIR
CLUSTER_CAP = 5            # ≤5 picks per supercluster (diversity)
BUFFER = 100              # extra candidates beyond the 400, for step 4/5 flexibility
OVERFILL_PENALTY = 0.5    # discourage overshooting an already-met cell


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived sampling columns; dedupe to one row per query_key (best kept)."""
    df = df.copy()
    df["category_weight"] = df["clinical_relevance"].map(CATEGORY_WEIGHT).fillna(0.0)
    df["min_conf"] = df[["mu_confidence", "pu_confidence", "et_confidence"]].min(axis=1)
    df["conf_factor"] = (df["min_conf"] >= 3).map({True: 1.0, False: 0.6})
    # one row per distinct query: keep the highest category_weight, then confidence
    df = (df.sort_values(["category_weight", "min_conf"], ascending=False)
            .drop_duplicates(subset="query_key", keep="first")
            .reset_index(drop=True))
    return df


def greedy_select(df: pd.DataFrame, n: int, remaining: dict, cluster_count: Counter,
                  chosen_keys: set) -> list[int]:
    """Greedily pick up to n row-indices maximizing cell-need × weight × confidence."""
    rows = df.to_dict("records")
    picked: list[int] = []
    # precompute the three grade cells per row
    cells = [(r["mu_grade"], r["pu_grade"], r["et_grade"]) for r in rows]
    while len(picked) < n:
        best_i, best_score = -1, -1e18
        for i, r in enumerate(rows):
            if r["query_key"] in chosen_keys:
                continue
            if cluster_count[r["supercluster_id"]] >= CLUSTER_CAP:
                continue
            need = 0.0
            for ax, cell in zip(AXES, cells[i]):
                tgt = TARGETS[ax][cell]
                rem = remaining[cell]
                if rem > 0:
                    need += rem / tgt
                else:
                    need -= OVERFILL_PENALTY * (1.0 / tgt)  # mild push off full cells
            quality = r["category_weight"] * r["conf_factor"]
            score = need * quality + 0.01 * quality  # baseline orders overfill picks
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0:
            break  # no feasible candidate left
        r = rows[best_i]
        picked.append(best_i)
        chosen_keys.add(r["query_key"])
        cluster_count[r["supercluster_id"]] += 1
        for ax, cell in zip(AXES, cells[best_i]):
            remaining[cell] -= 1
    return picked


def run(args) -> int:
    _io.ensure_dirs()
    if not _io.PICKS_RAW.exists():
        print("[sample] picks_raw.parquet missing — run step 1 first.", file=sys.stderr)
        return 2
    raw = pd.read_parquet(_io.PICKS_RAW)
    df = prep(raw)
    print(f"[sample] {len(raw)} picks -> {len(df)} distinct queries", file=sys.stderr)

    remaining = {cell: t for ax in TARGETS for cell, t in TARGETS[ax].items()}
    cluster_count: Counter = Counter()
    chosen_keys: set = set()

    # --- Phase 1: scarcity-first. Cells whose supply is below target (the rare
    # safety-critical content — MU2/PU1/PU2 given the picker yield) are precious;
    # grab every row touching such a cell before greedy-filling abundant cells,
    # so we never drop a rare urgent/distress query. ---
    recs = df.to_dict("records")
    available = {cell: int((df[ax.lower() + "_grade"] == cell).sum())
                 for ax in AXES for cell in TARGETS[ax]}
    scarce_cells = {cell for ax in AXES for cell in TARGETS[ax]
                    if available[cell] < TARGETS[ax][cell]}
    order = sorted(range(len(recs)),
                   key=lambda i: (-recs[i]["category_weight"], -recs[i]["min_conf"]))
    phase1: list[int] = []
    for i in order:
        if len(phase1) >= N_TOTAL:
            break
        r = recs[i]
        if r["query_key"] in chosen_keys:
            continue
        # NOTE: scarce safety rows are exempt from the ≤5 cluster cap — never drop
        # a rare urgent/distress query for diversity; the cap still binds phase 2.
        cells = (r["mu_grade"], r["pu_grade"], r["et_grade"])
        if any(c in scarce_cells and remaining[c] > 0 for c in cells):
            phase1.append(i)
            chosen_keys.add(r["query_key"])
            cluster_count[r["supercluster_id"]] += 1
            for ax, c in zip(AXES, cells):
                remaining[c] -= 1
    print(f"[sample] phase 1 (scarce-first): {len(phase1)} rows "
          f"touching {sorted(scarce_cells)}", file=sys.stderr)

    # --- Phase 2: greedy-fill the rest of the 400, then a spill buffer ---
    sel_idx = phase1 + greedy_select(df, N_TOTAL - len(phase1), remaining,
                                     cluster_count, chosen_keys)
    buf_idx = greedy_select(df, BUFFER, remaining, cluster_count, chosen_keys)

    sel = df.iloc[sel_idx].copy()
    # core vs reservoir: lowest-confidence 100 -> reservoir (ambiguous / demotable)
    sel = sel.sort_values(["min_conf", "category_weight"], ascending=True).reset_index(drop=True)
    sel["bucket"] = ["iaa_reservoir_100"] * min(N_RESERVOIR, len(sel)) + \
                    ["core_300"] * max(0, len(sel) - N_RESERVOIR)
    buf = df.iloc[buf_idx].copy()
    buf["bucket"] = "spill_400_buffer"
    out = pd.concat([sel, buf], ignore_index=True)

    # cluster_diversity_score: 1 / (#selected-400 rows from the same cluster)
    in400 = sel["supercluster_id"].value_counts()
    out["cluster_diversity_score"] = out["supercluster_id"].map(
        lambda c: round(1.0 / in400.get(c, 1), 3))

    _io.write_parquet_atomic(out, _io.GOLDEN_400_PICKED)

    # achieved vs target + shortfall (over the 400 only)
    achieved = {ax: {g: int((sel[ax.lower() + "_grade"] == g).sum())
                     for g in TARGETS[ax]} for ax in AXES}
    shortfall = {ax: {g: max(0, TARGETS[ax][g] - achieved[ax][g]) for g in TARGETS[ax]}
                 for ax in AXES}
    _io.write_json_atomic(shortfall, SHORTFALL_PATH)

    state = {
        "completed_at": _io.now_iso(),
        "current_status": "completed",
        "n_selected": len(sel),
        "n_core": int((sel["bucket"] == "core_300").sum()),
        "n_reservoir": int((sel["bucket"] == "iaa_reservoir_100").sum()),
        "n_buffer": len(buf),
        "by_category": sel["clinical_relevance"].value_counts().to_dict(),
        "achieved": achieved,
        "shortfall": shortfall,
        "input_picks": len(raw),
    }
    _io.write_state(state, STATE_PATH)

    REPORT_PATH.write_text(render_report(sel, buf, achieved, shortfall, state), encoding="utf-8")

    # console summary
    short_cells = [f"{g}:{n}" for ax in AXES for g, n in shortfall[ax].items() if n > 0]
    print(f"[sample] DONE. selected={len(sel)} (core={state['n_core']} "
          f"reservoir={state['n_reservoir']}) buffer={len(buf)}", file=sys.stderr)
    print(f"[sample] shortfalls: {', '.join(short_cells) or 'none'}", file=sys.stderr)
    print(f"[sample] report: {REPORT_PATH}", file=sys.stderr)
    return 0


def render_report(sel, buf, achieved, shortfall, state) -> str:
    R = _report
    rng = random.Random(42)

    head = R.kv({
        "Selected": f"{len(sel)} (core {state['n_core']} + reservoir {state['n_reservoir']})",
        "Spill buffer": len(buf),
        "By category": ", ".join(f"{k}={v}" for k, v in state["by_category"].items()),
        "Input distinct queries": state["input_picks"],
    })

    # achieved vs target per axis
    dist_rows = []
    for ax in AXES:
        for g in TARGETS[ax]:
            tgt, got = TARGETS[ax][g], achieved[ax][g]
            dist_rows.append({
                "cell": g, "target": tgt, "achieved": got,
                "delta": got - tgt,
                "status": "OK" if got >= tgt else f"SHORT {tgt - got}",
            })
    dist_tbl = R.table(dist_rows, ["cell", "target", "achieved", "delta", "status"])

    # shortfall callout
    short_cells = {f"{g}": n for ax in AXES for g, n in shortfall[ax].items() if n > 0}
    if short_cells:
        items = "".join(f"<li>{R.status_badge(False, f'{c}: needs {n} more')}</li>"
                        for c, n in short_cells.items())
        backfillable = [c for c in short_cells if c in ("MU2", "PU2")]
        call = (f"<ul>{items}</ul><p>Step 4 backfill targets "
                f"<b>{', '.join(backfillable) or 'none'}</b> (MU2/PU2 only); other "
                f"shortfalls are accepted and reported.</p>")
    else:
        call = f"<p>{R.status_badge(True, 'all cells met target')}</p>"

    # cluster diversity histogram (picks per cluster within the 400)
    per_cluster = sel["supercluster_id"].value_counts()
    hist = Counter(per_cluster.values)
    div = "".join(R.bar_row(f"{k} pick(s)/cluster", hist.get(k, 0), len(per_cluster))
                  for k in range(1, CLUSTER_CAP + 1))
    div += f"<p class='muted'>{len(per_cluster)} distinct clusters in the 400; cap is {CLUSTER_CAP}.</p>"

    # sample queries per cell (5 each)
    srows = []
    for ax in AXES:
        col = ax.lower() + "_grade"
        for g in TARGETS[ax]:
            sub = sel[sel[col] == g]
            idx = list(sub.index)
            rng.shuffle(idx)
            for i in idx[:5]:
                r = sel.loc[i]
                srows.append({
                    "cell": g, "query_text": r["query_text"],
                    "MU/PU/ET": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
                    "category": r["clinical_relevance"], "bucket": r["bucket"],
                })
    samp_tbl = R.table(srows, ["cell", "query_text", "MU/PU/ET", "category", "bucket"])

    prov = R.provenance({
        "input": _io.GOLDEN_400_PICKED.name + " from " + _io.PICKS_RAW.name,
        "targets": str(TARGETS),
        "cluster_cap": CLUSTER_CAP,
        "completed_at": state["completed_at"],
    })

    return R.page(
        "Step 2 — Sampling report",
        R.section("1. Headline", head),
        R.section("2. Per-axis distribution: achieved vs target", dist_tbl),
        R.section("3. Shortfall callout", call),
        R.section("4. Cluster diversity (picks per cluster in the 400)", div),
        R.section("5. Sample queries per cell (5 random each)", samp_tbl),
        prov,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2 — greedy constraint sampler (local)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        _io.print_status(STATE_PATH)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
