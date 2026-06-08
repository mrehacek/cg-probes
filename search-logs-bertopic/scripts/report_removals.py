"""Generate a markdown relevance-audit report of P1-05 cluster classifications.

Reads `cluster_manifest.jsonl` and writes a human-readable report listing every
cluster the LLM labelled `navigational` or `non-clinical` (i.e. the labels that
trigger downstream routing decisions other than probe training), with top
words, top-10 queries by clicks, and the LLM's reasoning. NOTE: this is a
LABELLING audit, not a removal — every cluster + query stays in the dataset.
Downstream stages consume the labels:
  - `oncology-core` / `oncology-adjacent` → P2 contrastive / P3 golden set / P4 probes
  - `navigational` → routed to a hospital-info handler in the RAG safety layer
  - `non-clinical` → filtered out (spam, SQL injection, true off-topic noise)

Output:
  cache/clusters/relevance_audit.md

Usage:
  python search-logs-bertopic/scripts/report_removals.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_DIR = ROOT / "cache" / "clusters"
MANIFEST_JSONL = CLUSTERS_DIR / "cluster_manifest.jsonl"
OVERVIEW_JSONL = CLUSTERS_DIR / "cluster_overview_qwen3_minc10_leaf.jsonl"
PREPROCESS_PARQUET = ROOT / "cache" / "preprocess" / "embed_input.parquet"
OUT_MD = CLUSTERS_DIR / "relevance_audit.md"

# Buckets surfaced for human review. `oncology-core` / `oncology-adjacent` flow
# into probe training and aren't audited here. `navigational` is *kept* but
# routed differently; `non-clinical` is what gets fully filtered downstream.
AUDIT_LEVELS = {"non-clinical", "navigational"}

ACTION_BY_REL = {
    "oncology-core":      "→ P2/P3/P4 probe training (kept)",
    "oncology-adjacent":  "→ P2/P3/P4 probe training (kept)",
    "navigational":       "→ kept; routed to hospital-info handler in RAG layer",
    "non-clinical":       "→ filtered downstream (spam / SQL / true off-topic)",
}


def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _slug(s: str) -> str:
    """GitHub-style markdown anchor slug for a heading."""
    out = []
    for ch in s.lower():
        if ch.isalnum() or ch in " -_":
            out.append(ch)
    return "".join(out).strip().replace(" ", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST_JSONL)
    ap.add_argument("--out", type=Path, default=OUT_MD)
    ap.add_argument("--top-q", type=int, default=10, help="queries per cluster to show")
    args = ap.parse_args()

    if not args.manifest.exists():
        sys.exit(f"missing: {args.manifest} (run cluster_cards.py first)")
    if not OVERVIEW_JSONL.exists():
        sys.exit(f"missing: {OVERVIEW_JSONL}")
    if not PREPROCESS_PARQUET.exists():
        sys.exit(f"missing: {PREPROCESS_PARQUET}")

    manifest = _load_jsonl(args.manifest)
    overview = {int(r["cluster_id"]): r for r in _load_jsonl(OVERVIEW_JSONL)}

    pp = pd.read_parquet(PREPROCESS_PARQUET, columns=["query_key", "query", "clicks"])
    key_to_query = dict(zip(pp["query_key"].astype(str), pp["query"].astype(str)))
    key_to_clicks = dict(zip(pp["query_key"].astype(str), pp["clicks"].fillna(0).astype(int)))

    total_clusters = len(manifest)
    audit = [c for c in manifest if c["clinical_relevance"] in AUDIT_LEVELS]

    # Group by bucket
    by_bucket: dict[str, list[dict]] = {b: [] for b in AUDIT_LEVELS}
    for c in audit:
        by_bucket[c["clinical_relevance"]].append(c)
    for b in by_bucket:
        by_bucket[b].sort(key=lambda c: -c["n_queries"])

    n_queries_total = int(pp.shape[0])

    lines: list[str] = []
    lines.append(f"# Cluster relevance audit\n")
    lines.append(
        f"P1-05 LLM (gpt-5.4-mini, prompt `cluster_card_v3`) classified each of the "
        f"**{total_clusters:,}** clusters by `clinical_relevance`. **Nothing is being "
        f"deleted** — this audit lets you verify the LABEL is correct. Downstream stages "
        f"consume the labels:\n\n"
    )
    for rel, action in ACTION_BY_REL.items():
        lines.append(f"- **`{rel}`** {action}")
    lines.append("")
    lines.append(
        "This document lists clusters labelled `non-clinical` and `navigational` so you "
        "can spot misclassifications. If you find any, send me the cluster IDs and the "
        "intended label and I will patch the manifest.\n"
    )
    lines.append("## Summary\n")
    counts = pd.Series([c["clinical_relevance"] for c in manifest]).value_counts()
    rows_by_relevance = (
        pd.DataFrame(
            [{"clinical_relevance": c["clinical_relevance"], "n_queries": c["n_queries"]} for c in manifest]
        ).groupby("clinical_relevance")["n_queries"].sum().to_dict()
    )
    lines.append("| relevance | clusters | queries | downstream |")
    lines.append("|---|---:|---:|---|")
    for rel in ("oncology-core", "oncology-adjacent", "navigational", "non-clinical"):
        n_clu = int(counts.get(rel, 0))
        n_q = int(rows_by_relevance.get(rel, 0))
        action = ACTION_BY_REL[rel]
        lines.append(f"| `{rel}` | {n_clu} | {n_q:,} | {action} |")
    lines.append("")

    # Per-bucket TOC
    lines.append("## Table of contents\n")
    for bucket in ("non-clinical", "navigational"):
        items = by_bucket.get(bucket, [])
        if not items:
            continue
        lines.append(f"### {bucket} ({len(items)} clusters)")
        for c in items:
            anchor = _slug(f"cluster-{c['topic_id']}-{c['czech_label']}")
            lines.append(
                f"- [`#{c['topic_id']}` — {c['czech_label']}  ·  n={c['n_queries']}](#{anchor})"
            )
        lines.append("")
    lines.append("---\n")

    # Per-cluster sections
    for bucket in ("non-clinical", "navigational"):
        items = by_bucket.get(bucket, [])
        if not items:
            continue
        lines.append(f"## Clusters labelled `{bucket}`  ·  downstream: {ACTION_BY_REL[bucket]}\n")
        for c in items:
            cid = int(c["topic_id"])
            ov = overview.get(cid, {})
            members = ov.get("member_query_keys", [])
            top_words = ov.get("top_words", [])
            total_clicks = ov.get("total_clicks", 0)
            heading = f"Cluster {cid} — {c['czech_label']}  ·  n={c['n_queries']}"
            lines.append(f"### {heading}\n")
            lines.append(
                f"- **relevance:** `{c['clinical_relevance']}`  ·  "
                f"**total_clicks:** {total_clicks:,}  ·  "
                f"**U:** {c['urgency_potential']}  ·  "
                f"**E:** {c['emotional_potential']}"
            )
            if c.get("suspected_anchor_levels"):
                lvls = " ".join(c["suspected_anchor_levels"])
                lines.append(f"- **suspected anchor levels:** {lvls}")
            lines.append(f"- **summary:** {c['summary_cs']}")
            lines.append(f"- **why this label:** {c['rationale_cs']}")
            if top_words:
                tw_inline = " ".join(top_words[:15])
                lines.append(f"- **top_words:** `{tw_inline}`")

            # Top-N queries by clicks
            clicked = sorted(
                ((key_to_clicks.get(k, 0), k) for k in members),
                key=lambda x: (-x[0], x[1]),
            )
            shown = 0
            lines.append("")
            lines.append(f"**Top {args.top_q} queries by clicks:**\n")
            lines.append("| clicks | query |")
            lines.append("|---:|---|")
            for clicks, k in clicked:
                q = key_to_query.get(k)
                if q is None:
                    continue
                q_safe = q.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {clicks} | {q_safe} |")
                shown += 1
                if shown >= args.top_q:
                    break
            lines.append("")
            lines.append("")  # spacer
        lines.append("---\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}  ({len(lines):,} lines)")
    by_rel = pd.Series([c["clinical_relevance"] for c in manifest]).value_counts().to_dict()
    print(f"labels: {by_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
