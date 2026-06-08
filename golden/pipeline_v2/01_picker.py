"""STEP 1 — Cluster picker.

For each of the 1,889 superclusters, ask gpt-5.4 (medium reasoning, flex) to
SELECT verbatim member queries carrying non-zero MU/PU/ET signal and LABEL each
on all three axes. Resumable (skips already-processed clusters), checkpoints
every N completed clusters to parquet + state + HTML report.

Run (from repo root, by path — module names can't start with a digit):
    python golden/pipeline_v2/01_picker.py                  # full run / resume
    python golden/pipeline_v2/01_picker.py --status         # print state, exit
    python golden/pipeline_v2/01_picker.py --limit 20       # smoke test
    python golden/pipeline_v2/01_picker.py --force           # ignore prompt-hash gate

Outputs:
    golden/cache/golden_set_v1/picks_raw.parquet
    golden/cache/golden_set_v1/state/01_picker_state.json
    golden/cache/golden_set_v1/state/01_picker_errors.jsonl
    golden/cache/golden_set_v1/reports/01_picker_report.html
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# --- bootstrap repo root onto sys.path (works when run by path) --------------
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from contrastive.textnorm import normalize  # noqa: E402
from golden.pipeline_v2 import _io, _report  # noqa: E402
from golden.pipeline_v2._llm import (  # noqa: E402
    PICKER_MODEL,
    estimate_cost_usd,
    picker_client,
    run_concurrent,
)
from golden.pipeline_v2.schemas import AXES, PickerResponse  # noqa: E402

TEMPLATE_VERSION = "cluster_picker_v1"
STATE_PATH = _io.STATE_DIR / "01_picker_state.json"
ERRORS_PATH = _io.STATE_DIR / "01_picker_errors.jsonl"
REPORT_PATH = _io.REPORTS_DIR / "01_picker_report.html"

# PU2/MU2 supply targets for the green/red headline (PLAN §4 checkpoint).
SUPPLY_TARGETS = {"MU2": 80, "PU2": 60}

# Per-category processing fraction (user decision 2026-06-05): navigational
# clusters are mostly phone-number/name junk that wastes calls and gets
# mislabeled ET1/ET2, so we sample ~25% of them for ET0/administrative coverage;
# non-clinical is skipped entirely. Sampling is deterministic on supercluster_id
# (hash bucket) so it's reproducible and resume-safe. --nav-fraction overrides.
CATEGORY_FRACTION = {
    "oncology-core": 1.0,
    "oncology-adjacent": 1.0,
    "navigational": 0.25,
    "non-clinical": 0.0,
}


def select_clusters(clusters: list[dict], fractions: dict[str, float]):
    """Filter clusters by per-category fraction. Returns (selected, excluded_counter)."""
    selected: list[dict] = []
    excluded: Counter = Counter()
    for c in clusters:
        cat = c.get("clinical_relevance", "")
        frac = fractions.get(cat, 1.0)
        if frac >= 1.0:
            selected.append(c)
        elif frac <= 0.0:
            excluded[cat] += 1
        else:
            bucket = int(_io.sha256_text(str(c["supercluster_id"])), 16) % 1000
            if bucket < round(frac * 1000):
                selected.append(c)
            else:
                excluded[cat] += 1
    return selected, excluded

PICK_COLUMNS = [
    "supercluster_id", "clinical_relevance", "czech_label", "query_text",
    "query_key", "mu_grade", "pu_grade", "et_grade",
    "mu_confidence", "pu_confidence", "et_confidence",
    "notes_cs", "verbatim_match", "picker_model", "picker_run_at",
]


# --- prompt assembly ---------------------------------------------------------

def build_prompt() -> tuple[str, str]:
    """Resolve the picker prompt: inject the three rubrics into [SYSTEM]."""
    system, user = _io.load_prompt(_io.PICKER_PROMPT)
    rubrics = {
        "mu_rubric": (_io.RUBRICS_DIR / "MU.md").read_text(encoding="utf-8").strip(),
        "pu_rubric": (_io.RUBRICS_DIR / "PU.md").read_text(encoding="utf-8").strip(),
        "et_rubric": (_io.RUBRICS_DIR / "ET.md").read_text(encoding="utf-8").strip(),
    }
    for k, v in rubrics.items():
        system = system.replace("{" + k + "}", v)
    return system, user


def render_member_queries(texts: list[str]) -> str:
    return "\n".join(f'- "{t}"' for t in texts)


def query_key_lookup(cluster: dict) -> dict[str, str]:
    """Map verbatim + normalized member-query text -> query_key for join-back."""
    lut: dict[str, str] = {}
    for q in cluster.get("queries", []):
        key = q.get("query_key", "")
        surface = q.get("query", "")
        lut.setdefault(surface, key)
        lut.setdefault(normalize(surface), key)
    return lut


# --- worker ------------------------------------------------------------------

def make_worker(client, system: str, user_tmpl: str, run_at: str,
                dead_clusters: list[dict]):
    async def worker(cluster: dict):
        member_texts = _io.cluster_member_queries(cluster)  # all, most-clicked first
        user = (
            user_tmpl
            .replace("{supercluster_id}", str(cluster["supercluster_id"]))
            .replace("{clinical_relevance}", str(cluster.get("clinical_relevance", "")))
            .replace("{czech_label}", str(cluster.get("czech_label", "")))
            .replace("{summary_cs}", str(cluster.get("summary_cs", "")))
            .replace("{n_queries}", str(cluster.get("n_queries", len(member_texts))))
            .replace("{member_queries}", render_member_queries(member_texts))
        )
        parsed, usage = await client.call_structured(
            phase="picker",
            template_version=TEMPLATE_VERSION,
            system=system,
            user=user,
            schema_model=PickerResponse,
            reasoning_effort="medium",
        )
        lut = query_key_lookup(cluster)
        rows: list[dict] = []
        for pk in parsed.picks:
            qt = pk.query_text
            verbatim = qt in lut
            qkey = lut.get(qt) or lut.get(normalize(qt)) or normalize(qt)
            rows.append({
                "supercluster_id": cluster["supercluster_id"],
                "clinical_relevance": cluster.get("clinical_relevance", ""),
                "czech_label": cluster.get("czech_label", ""),
                "query_text": qt,
                "query_key": qkey,
                "mu_grade": pk.mu_grade, "pu_grade": pk.pu_grade, "et_grade": pk.et_grade,
                "mu_confidence": pk.mu_confidence,
                "pu_confidence": pk.pu_confidence,
                "et_confidence": pk.et_confidence,
                "notes_cs": pk.notes_cs,
                "verbatim_match": verbatim,
                "picker_model": PICKER_MODEL,
                "picker_run_at": run_at,
            })
        if not parsed.picks:
            dead_clusters.append({
                "supercluster_id": cluster["supercluster_id"],
                "clinical_relevance": cluster.get("clinical_relevance", ""),
                "czech_label": cluster.get("czech_label", ""),
                "skipped_reason_cs": parsed.skipped_reason_cs,
            })
        return rows, usage

    return worker


# --- report ------------------------------------------------------------------

def render_report(df: pd.DataFrame, dead_clusters: list[dict], stats: dict,
                  meta: dict) -> str:
    rng = random.Random(42)
    R = _report
    n_clusters_with_picks = df["supercluster_id"].nunique() if len(df) else 0
    n_dead = len(dead_clusters)
    n_processed = n_clusters_with_picks + n_dead
    total_clusters = meta["total_clusters"]

    # 1. Run summary
    mean_picks = (len(df) / n_clusters_with_picks) if n_clusters_with_picks else 0
    excluded = meta.get("excluded", {})
    excl_str = ", ".join(f"{v} {k}" for k, v in excluded.items()) or "none"
    summary = R.kv({
        "Clusters processed": f"{n_processed} / {total_clusters} selected",
        "Excluded by category policy": f"{excl_str} (policy {meta.get('fractions', {})})",
        "Clusters failed": stats.get("failed", 0),
        "Total picks": len(df),
        "Mean picks / picking cluster": f"{mean_picks:.1f}",
        "Clusters returning 0 picks": f"{n_dead} ({100*n_dead/max(n_processed,1):.0f}%)",
        "Prompt tokens": f"{stats.get('prompt_tokens', 0):,}",
        "Completion tokens": f"{stats.get('completion_tokens', 0):,}",
        "Est. cost (flex, upper bound)": f"${meta['est_cost']:.2f}",
    })

    # 2. Per-category breakdown
    cat_rows = []
    if len(df):
        by_cat_picks = df.groupby("clinical_relevance").size()
    else:
        by_cat_picks = pd.Series(dtype=int)
    dead_by_cat = Counter(d["clinical_relevance"] for d in dead_clusters)
    picking_by_cat = df.groupby("clinical_relevance")["supercluster_id"].nunique() if len(df) else pd.Series(dtype=int)
    cats = sorted(set(by_cat_picks.index) | set(dead_by_cat) | set(picking_by_cat.index))
    for c in cats:
        cat_rows.append({
            "category": c,
            "clusters w/ picks": int(picking_by_cat.get(c, 0)),
            "dead clusters": int(dead_by_cat.get(c, 0)),
            "picks": int(by_cat_picks.get(c, 0)),
        })
    cat_tbl = R.table(cat_rows, ["category", "clusters w/ picks", "dead clusters", "picks"])

    # 3. Cell counts (3x3)
    per_axis = {ax: {} for ax in AXES}
    for ax in AXES:
        col = f"{ax.lower()}_grade"
        counts = df[col].value_counts().to_dict() if len(df) else {}
        per_axis[ax] = {g: int(counts.get(g, 0)) for g in (f"{ax}0", f"{ax}1", f"{ax}2")}
    grid = R.grade_grid(per_axis)

    # 4. PU2/MU2 supply check
    supply_rows = []
    for cell, target in SUPPLY_TARGETS.items():
        ax = cell[:2]
        got = per_axis[ax].get(cell, 0)
        supply_rows.append(
            f"<li>{R.status_badge(got >= target, f'{cell}: {got} picks (target {target})')}</li>"
        )
    supply = "<ul>" + "".join(supply_rows) + "</ul>"

    # 5. Confidence histogram per axis
    conf_html = ""
    for ax in AXES:
        col = f"{ax.lower()}_confidence"
        vc = df[col].value_counts().to_dict() if len(df) else {}
        bars = "".join(R.bar_row(f"conf {i}", int(vc.get(i, 0)), len(df)) for i in range(1, 6))
        conf_html += f"<h3>{ax}</h3>{bars}"

    # 6. Sample picks: 5 random per (axis, grade)
    sample_rows = []
    for ax in AXES:
        col = f"{ax.lower()}_grade"
        for g in (f"{ax}0", f"{ax}1", f"{ax}2"):
            sub = df[df[col] == g] if len(df) else df
            idx = list(sub.index)
            rng.shuffle(idx)
            for i in idx[:5]:
                r = df.loc[i]
                sample_rows.append({
                    "cell": g, "query_text": r["query_text"],
                    "MU/PU/ET": f"{r['mu_grade']}/{r['pu_grade']}/{r['et_grade']}",
                    "czech_label": r["czech_label"], "notes_cs": r["notes_cs"],
                })
    sample_tbl = R.table(sample_rows, ["cell", "query_text", "MU/PU/ET", "czech_label", "notes_cs"])

    # 7. Dead clusters (oncology-core flagged)
    core_dead = [d for d in dead_clusters if d["clinical_relevance"] == "oncology-core"]
    dead_tbl = R.table(
        [{"supercluster_id": d["supercluster_id"], "category": d["clinical_relevance"],
          "czech_label": d["czech_label"], "skipped_reason_cs": d["skipped_reason_cs"]}
         for d in sorted(dead_clusters, key=lambda d: d["clinical_relevance"])][:200],
        ["supercluster_id", "category", "czech_label", "skipped_reason_cs"],
    )
    dead_note = (
        f"<p>{R.status_badge(len(core_dead) == 0, f'{len(core_dead)} oncology-core clusters returned 0 picks')}"
        " — these are worth eyeballing; navigational/non-clinical dead clusters are expected.</p>"
    )

    # 8. Errors / non-verbatim
    n_nonverbatim = int((~df["verbatim_match"]).sum()) if len(df) else 0
    err_html = R.kv({
        "Failed LLM calls (see errors.jsonl)": stats.get("failed", 0),
        "Non-verbatim picks (model didn't match input exactly)": n_nonverbatim,
    })

    # 9. Provenance
    prov = R.provenance({
        "model": PICKER_MODEL,
        "prompt": f"{_io.PICKER_PROMPT.name} (sha256 {meta['prompt_hash'][:12]})",
        "input": f"{_io.SUPERCLUSTERS_JSONL.name} (sha256 {meta['input_hash'][:12]})",
        "started_at": meta["started_at"],
        "report_at": meta["report_at"],
    })

    return R.page(
        "Step 1 — Cluster picker report",
        R.section("1. Run summary", summary),
        R.section("2. Per-category breakdown", cat_tbl),
        R.section("3. Cell counts (3 axes × 3 grades)", grid),
        R.section("4. PU2 / MU2 supply check", supply),
        R.section("5. Confidence histogram per axis", conf_html),
        R.section("6. Sample picks (5 random per cell)", sample_tbl),
        R.section("7. Dead clusters (0 picks)", dead_note, dead_tbl),
        R.section("8. Errors / retries", err_html),
        prov,
    )


# --- main --------------------------------------------------------------------

async def run(args) -> int:
    _io.ensure_dirs()
    system, user_tmpl = build_prompt()
    prompt_hash = _io.sha256_text(system + "\n@@@USER@@@\n" + user_tmpl)
    input_hash = _io.sha256_file(_io.SUPERCLUSTERS_JSONL)

    # prompt-hash gating
    prev = _io.read_state(STATE_PATH)
    if prev and prev.get("prompt_hash_sha256") not in (None, prompt_hash) and not args.force:
        print("[picker] prompt changed since last run; refusing to resume. "
              "Re-run with --force to override (mixes old + new prompt outputs).",
              file=sys.stderr)
        return 2

    clusters = _io.load_superclusters()
    if args.limit:
        clusters = clusters[: args.limit]

    fractions = dict(CATEGORY_FRACTION)
    if args.nav_fraction is not None:
        fractions["navigational"] = args.nav_fraction
    if args.include_non_clinical:
        fractions["non-clinical"] = 1.0
    clusters, excluded = select_clusters(clusters, fractions)
    if args.sample:
        pool = list(clusters)
        random.Random(args.sample_seed).shuffle(pool)
        clusters = pool[: args.sample]
        print(f"[picker] --sample {args.sample}: representative draw across "
              f"{Counter(c['clinical_relevance'] for c in clusters)}", file=sys.stderr)
    total_clusters = len(clusters)
    if excluded:
        print(f"[picker] category policy {fractions} -> excluded "
              + ", ".join(f"{v} {k}" for k, v in excluded.items())
              + f"; processing {total_clusters} clusters", file=sys.stderr)

    # resume: load prior picks + dead clusters, skip done ids
    prior_rows: list[dict] = []
    done_ids: set = set()
    if _io.PICKS_RAW.exists():
        prior_df = pd.read_parquet(_io.PICKS_RAW)
        prior_rows = prior_df.to_dict("records")
        done_ids |= set(prior_df["supercluster_id"].unique())
    dead_clusters: list[dict] = list((prev or {}).get("dead_clusters", []))
    done_ids |= {d["supercluster_id"] for d in dead_clusters}

    todo = [c for c in clusters if c["supercluster_id"] not in done_ids]
    print(f"[picker] {total_clusters} clusters, {len(done_ids)} already done, "
          f"{len(todo)} to process (concurrency={args.concurrency})", file=sys.stderr)
    if not todo:
        print("[picker] nothing to do — already complete.", file=sys.stderr)

    run_at = _io.now_iso()
    started_at = (prev or {}).get("started_at") or run_at
    client = picker_client(concurrency=args.concurrency)
    worker = make_worker(client, system, user_tmpl, run_at, dead_clusters)
    t0 = time.monotonic()

    def on_checkpoint(rows, stats, n_done, n_total):
        all_rows = prior_rows + rows
        df = pd.DataFrame(all_rows, columns=PICK_COLUMNS) if all_rows else pd.DataFrame(columns=PICK_COLUMNS)
        _io.write_parquet_atomic(df, _io.PICKS_RAW)
        completed_ids = sorted(done_ids | set(df["supercluster_id"].unique())
                               | {d["supercluster_id"] for d in dead_clusters})
        est_cost = estimate_cost_usd(PICKER_MODEL, stats["prompt_tokens"], stats["completion_tokens"])
        state = {
            "started_at": started_at,
            "last_checkpoint_at": _io.now_iso(),
            "completed_clusters": len(completed_ids),
            "total_clusters": total_clusters,
            "failed_clusters": stats.get("failed", 0),
            "dead_cluster_count": len(dead_clusters),
            "current_status": "running" if n_done < n_total else "completed",
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "est_cost_usd": round(est_cost, 2),
            "input_hash_sha256": input_hash,
            "prompt_path_used": str(_io.PICKER_PROMPT.relative_to(REPO)),
            "prompt_hash_sha256": prompt_hash,
            "completed_cluster_ids": completed_ids,
            "dead_clusters": dead_clusters,
        }
        _io.write_state(state, STATE_PATH)
        # Render the HTML report at every checkpoint so a long run is always
        # inspectable mid-flight (not only at completion).
        meta = {
            "total_clusters": total_clusters, "est_cost": est_cost,
            "prompt_hash": prompt_hash, "input_hash": input_hash,
            "started_at": started_at, "report_at": _io.now_iso(),
            "excluded": dict(excluded), "fractions": fractions,
        }
        REPORT_PATH.write_text(render_report(df, dead_clusters, stats, meta), encoding="utf-8")
        print(f"[picker] checkpoint {n_done}/{n_total} done | picks={len(df)} "
              f"dead={len(dead_clusters)} ~${est_cost:.2f} | report refreshed", file=sys.stderr)

    rows, stats = await run_concurrent(
        todo, worker,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=on_checkpoint,
        errors_path=ERRORS_PATH,
        item_id=lambda c: c["supercluster_id"],
    )

    # final artifacts
    df = pd.read_parquet(_io.PICKS_RAW) if _io.PICKS_RAW.exists() else pd.DataFrame(columns=PICK_COLUMNS)
    est_cost = estimate_cost_usd(PICKER_MODEL, stats["prompt_tokens"], stats["completion_tokens"])
    meta = {
        "total_clusters": total_clusters,
        "est_cost": est_cost,
        "prompt_hash": prompt_hash,
        "input_hash": input_hash,
        "started_at": started_at,
        "report_at": _io.now_iso(),
        "excluded": dict(excluded),
        "fractions": fractions,
    }
    REPORT_PATH.write_text(render_report(df, dead_clusters, stats, meta), encoding="utf-8")

    elapsed = time.monotonic() - t0
    n_new = len(todo)
    rate = n_new / elapsed if elapsed > 0 and n_new else 0
    n_processed = df["supercluster_id"].nunique() + len(dead_clusters)
    coverage = n_processed / total_clusters if total_clusters else 0
    mu2 = int((df["mu_grade"] == "MU2").sum()) if len(df) else 0
    pu2 = int((df["pu_grade"] == "PU2").sum()) if len(df) else 0
    print(f"\n[picker] DONE. picks={len(df)} clusters_processed={n_processed}/{total_clusters} "
          f"({coverage:.1%}) MU2={mu2} PU2={pu2} failed={stats['failed']} ~${est_cost:.2f}",
          file=sys.stderr)
    if n_new:
        print(f"[picker] wall={elapsed:.1f}s for {n_new} new calls "
              f"= {rate:.2f} clusters/s ({60*rate:.0f}/min)", file=sys.stderr)
    print(f"[picker] report: {REPORT_PATH}", file=sys.stderr)
    if coverage < 0.95:
        print(f"[picker] WARNING: cluster coverage {coverage:.1%} < 95% — review before step 2.",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 1 — cluster picker (gpt-5.4 flex)")
    ap.add_argument("--status", action="store_true", help="print state file and exit")
    ap.add_argument("--force", action="store_true", help="ignore prompt-hash gate on resume")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="process only first N clusters (debug)")
    ap.add_argument("--sample", type=int, default=0,
                    help="process a seeded representative draw of N selected clusters (test)")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--nav-fraction", type=float, default=None,
                    help="fraction of navigational clusters to process (default 0.25)")
    ap.add_argument("--include-non-clinical", action="store_true",
                    help="also process non-clinical clusters (default: skip)")
    args = ap.parse_args()
    if args.status:
        _io.print_status(STATE_PATH)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
