#!/usr/bin/env python3
"""
merge_level_report.py

Simulate BERTopic hierarchical merge (levels 1 and 2) from centroid embeddings.
Produces a self-contained HTML report with collapsible cluster/query views.

Usage:
    python scripts/merge_level_report.py [--cutoff1 F] [--cutoff2 F] [--out PATH]
"""
from __future__ import annotations

import argparse
import html as html_module
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_distances

# ─── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
CENTROIDS_PARQUET = ROOT / "cache/clusters/centroids_qwen3.parquet"
MANIFEST_JSONL    = ROOT / "cache/clusters/cluster_manifest.jsonl"
OVERVIEW_JSONL    = ROOT / "cache/clusters/cluster_overview_qwen3_minc10_leaf.jsonl"
ASSIGNMENTS_CSV   = ROOT / "cache/clusters/bertopic_assignments_qwen3_minc10_leaf.csv"
OUT_DIR           = ROOT / "cache/reports"

# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BERTopic merge-level HTML report")
    p.add_argument("--cutoff1", type=float, default=None,
                   help="Cosine distance cutoff for Level 1 (default: 5%% of distance range)")
    p.add_argument("--cutoff2", type=float, default=None,
                   help="Cosine distance cutoff for Level 2 (default: 10%% of distance range)")
    p.add_argument("--out", type=Path, default=OUT_DIR / "merge_level_1_report.html",
                   help="Output HTML path")
    return p.parse_args()

# ─── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    print("Loading centroids...", flush=True)
    cent_df = pd.read_parquet(CENTROIDS_PARQUET)
    cluster_ids = cent_df["cluster_id"].values.astype(int)
    C = np.array(cent_df["centroid"].tolist(), dtype=np.float32)
    idx_by_cid = {int(cid): i for i, cid in enumerate(cluster_ids)}
    print(f"  {len(cluster_ids)} clusters, {C.shape[1]}-d centroids", flush=True)

    print("Loading manifest...", flush=True)
    manifest: dict[int, dict] = {}
    with open(MANIFEST_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                manifest[int(d["topic_id"])] = d
    print(f"  {len(manifest)} cluster cards", flush=True)

    print("Loading overview...", flush=True)
    overview: dict[int, dict] = {}
    with open(OVERVIEW_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                overview[int(d["cluster_id"])] = d
    print(f"  {len(overview)} cluster overviews", flush=True)

    print("Loading assignments...", flush=True)
    assign_df = pd.read_csv(ASSIGNMENTS_CSV)
    cluster_queries: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for _, row in assign_df.iterrows():
        cluster_queries[int(row["cluster_id"])].append(
            (str(row["query"]), int(row["clicks"]))
        )
    for cid in cluster_queries:
        cluster_queries[cid].sort(key=lambda x: -x[1])
    print(f"  {len(assign_df):,} query assignments across {len(cluster_queries)} clusters",
          flush=True)

    return cluster_ids, C, idx_by_cid, manifest, overview, cluster_queries

# ─── Linkage & merging ─────────────────────────────────────────────────────────

def compute_linkage(C: np.ndarray):
    print("Computing cosine distances...", flush=True)
    D = cosine_distances(C.astype(np.float64))
    n = len(D)
    condensed = D[np.triu_indices(n, k=1)]
    print("Computing linkage (average/UPGMA, matches BERTopic)...", flush=True)
    Z = linkage(condensed, method="average")
    dists = Z[:, 2]
    print(f"  Distance range: [{dists.min():.5f}, {dists.max():.5f}]", flush=True)
    return Z, dists


def detect_cutoffs(dists: np.ndarray, c1_override=None, c2_override=None):
    d_min, d_max = float(dists.min()), float(dists.max())
    c1 = c1_override if c1_override is not None else d_min + 0.05 * (d_max - d_min)
    c2 = c2_override if c2_override is not None else d_min + 0.10 * (d_max - d_min)
    print(f"Level 1 cutoff: {c1:.5f}", flush=True)
    print(f"Level 2 cutoff: {c2:.5f}", flush=True)
    return c1, c2


def apply_cut(
    Z: np.ndarray, cutoff: float, cluster_ids: np.ndarray
) -> tuple[dict[int, list[int]], list[int]]:
    labels = fcluster(Z, t=cutoff, criterion="distance")
    super_to_orig: dict[int, list[int]] = defaultdict(list)
    for row_i, super_id in enumerate(labels):
        super_to_orig[int(super_id)].append(int(cluster_ids[row_i]))
    merged = {sid: cids for sid, cids in super_to_orig.items() if len(cids) > 1}
    singletons = [cids[0] for cids in super_to_orig.values() if len(cids) == 1]
    return merged, sorted(singletons)


def group_max_dist(cids: list[int], C: np.ndarray, idx_by_cid: dict) -> float:
    if len(cids) == 1:
        return 0.0
    idxs = [idx_by_cid[c] for c in cids if c in idx_by_cid]
    if len(idxs) < 2:
        return 0.0
    vecs = C[idxs].astype(np.float64)
    return float(cosine_distances(vecs).max())


def build_knee_rows(dists: np.ndarray, cutoff1: float, cutoff2: float) -> list[dict]:
    n_orig = len(dists) + 1
    d_min, d_max = float(dists.min()), float(dists.max())
    # 30 thresholds covering the first 25% of the range (where level 1/2 merges occur)
    # plus a few wider ones for context
    thresholds_focused = np.linspace(d_min, d_min + 0.25 * (d_max - d_min), 20)
    thresholds_wide = np.linspace(d_min + 0.25 * (d_max - d_min), d_max, 10)[1:]
    thresholds = np.concatenate([thresholds_focused, thresholds_wide])

    rows = []
    for t in thresholds:
        n_merges = int(np.sum(dists <= t))
        n_clusters = n_orig - n_merges
        rows.append({
            "threshold": float(t),
            "n_clusters": n_clusters,
            "n_merges": n_merges,
            "pct_reduction": 100.0 * n_merges / (n_orig - 1),
            "is_l1": abs(t - cutoff1) == min(abs(thresholds - cutoff1)),
            "is_l2": abs(t - cutoff2) == min(abs(thresholds - cutoff2)),
        })
    return rows

# ─── HTML helpers ──────────────────────────────────────────────────────────────

def h(text: Any) -> str:
    return html_module.escape(str(text))


BADGE_CSS = {
    "oncology-core":     ("badge-core",       "#d1ecf1", "#0c5460"),
    "oncology-adjacent": ("badge-adjacent",   "#d4edda", "#155724"),
    "navigational":      ("badge-navigational","#fff3cd", "#856404"),
    "non-clinical":      ("badge-nonclinical","#f8d7da", "#721c24"),
}


def badge(relevance: str) -> str:
    cls, bg, color = BADGE_CSS.get(relevance, ("badge-unknown", "#e2e3e5", "#383d41"))
    return (
        f'<span class="badge {cls}" '
        f'style="background:{bg};color:{color}">'
        f'{h(relevance)}</span>'
    )


def urgency_badge(level: str) -> str:
    colors = {"high": ("#dc3545", "white"), "moderate": ("#fd7e14", "white"), "low": ("#6c757d", "white")}
    bg, fg = colors.get(level, ("#6c757d", "white"))
    return f'<span class="urg-badge" style="background:{bg};color:{fg};padding:1px 5px;border-radius:3px;font-size:0.75em">{h(level)}</span>'


def render_query_list(queries: list[tuple[str, int]], max_show: int = 0) -> str:
    show = queries if max_show == 0 else queries[:max_show]
    items = "".join(
        f'<li>{h(q)} <span style="color:#6c757d;font-size:0.8em">({c:,}×)</span></li>'
        for q, c in show
    )
    return f'<ol class="query-list">{items}</ol>'


def render_cluster_card(
    cid: int,
    manifest: dict,
    overview: dict,
    cluster_queries: dict,
    indent: int = 0,
) -> str:
    m = manifest.get(cid, {})
    ov = overview.get(cid, {})
    label = m.get("czech_label", f"Cluster {cid}")
    n_q = m.get("n_queries", ov.get("size", 0))
    cr = m.get("clinical_relevance", "unknown")
    urgency = m.get("urgency_potential", "")
    emotional = m.get("emotional_potential", "")
    top_words = ov.get("top_words", [])[:12]
    queries = cluster_queries.get(cid, [])

    tw_str = " · ".join(h(w) for w in top_words) if top_words else "—"
    indent_style = f"margin-left:{indent}px;" if indent else ""

    summary_html = (
        f'<b>#{cid}</b> {h(label)} '
        f'&nbsp;|&nbsp; <b>{n_q:,}</b> q '
        f'&nbsp;|&nbsp; {badge(cr)} '
        f'&nbsp;|&nbsp; U:{urgency_badge(urgency)} '
        f'E:{urgency_badge(emotional)}'
    )

    query_section = (
        f'<details class="query-list-details">'
        f'<summary style="cursor:pointer;color:#0d6efd;font-size:0.85em">'
        f'All queries ({len(queries):,})</summary>'
        f'{render_query_list(queries)}'
        f'</details>'
    )

    body_html = (
        f'<div style="font-size:0.82em;color:#495057;margin:4px 0 8px 0">'
        f'<b>Top words:</b> {tw_str}</div>'
        f'{query_section}'
    )

    return (
        f'<details class="cluster-card" style="{indent_style}">'
        f'<summary>{summary_html}</summary>'
        f'<div class="card-body">{body_html}</div>'
        f'</details>\n'
    )


def render_merged_card(
    group_num: int,
    cids: list[int],
    manifest: dict,
    overview: dict,
    cluster_queries: dict,
    C: np.ndarray,
    idx_by_cid: dict,
) -> str:
    n_total = sum(manifest.get(c, {}).get("n_queries", overview.get(c, {}).get("size", 0)) for c in cids)
    relevances = [manifest.get(c, {}).get("clinical_relevance", "unknown") for c in cids]
    dominant_rel = Counter(relevances).most_common(1)[0][0]
    max_dist = group_max_dist(cids, C, idx_by_cid)

    # Combined queries (union, sorted by clicks)
    all_queries: list[tuple[str, int]] = []
    for cid in cids:
        all_queries.extend(cluster_queries.get(cid, []))
    all_queries.sort(key=lambda x: -x[1])

    # Sort constituent clusters by size desc
    cids_sorted = sorted(
        cids,
        key=lambda c: -manifest.get(c, {}).get("n_queries", overview.get(c, {}).get("size", 0))
    )
    cid_labels = ", ".join(f"#{c}" for c in cids_sorted)

    summary_html = (
        f'<b>Group #{group_num}</b> '
        f'&nbsp;|&nbsp; <b>{len(cids)}</b> clusters merged '
        f'&nbsp;|&nbsp; max dist: <b>{max_dist:.4f}</b> '
        f'&nbsp;|&nbsp; <b>{n_total:,}</b> queries '
        f'&nbsp;|&nbsp; {badge(dominant_rel)}'
    )

    # Constituent cluster cards (indented)
    constituent_cards = "".join(
        render_cluster_card(c, manifest, overview, cluster_queries, indent=16)
        for c in cids_sorted
    )

    combined_section = (
        f'<details class="query-list-details" style="margin-top:8px">'
        f'<summary style="cursor:pointer;color:#0d6efd;font-weight:600">'
        f'Combined queries ({len(all_queries):,})</summary>'
        f'{render_query_list(all_queries)}'
        f'</details>'
    )

    body_html = (
        f'<div style="font-size:0.82em;color:#495057;margin:4px 0 8px 0">'
        f'Clusters: {h(cid_labels)}</div>'
        f'<div style="font-weight:600;margin:8px 0 4px 0;font-size:0.9em">'
        f'Constituent clusters:</div>'
        f'{constituent_cards}'
        f'{combined_section}'
    )

    return (
        f'<details class="merged-card">'
        f'<summary>{summary_html}</summary>'
        f'<div class="card-body" style="padding-left:8px">{body_html}</div>'
        f'</details>\n'
    )

# ─── HTML sections ──────────────────────────────────────────────────────────────

def render_dendrogram_table(rows: list[dict]) -> str:
    header = (
        '<table class="info-table">'
        '<thead><tr>'
        '<th>Distance Threshold</th>'
        '<th># Super-clusters</th>'
        '<th># Merges done</th>'
        '<th>% Reduction</th>'
        '<th>Note</th>'
        '</tr></thead><tbody>'
    )
    body_rows = []
    for r in rows:
        note = ""
        row_style = ""
        if r["is_l1"]:
            note = "← Level 1 cutoff"
            row_style = ' style="background:#fff3cd;font-weight:600"'
        elif r["is_l2"]:
            note = "← Level 2 cutoff"
            row_style = ' style="background:#d4edda;font-weight:600"'
        body_rows.append(
            f'<tr{row_style}>'
            f'<td>{r["threshold"]:.5f}</td>'
            f'<td>{r["n_clusters"]:,}</td>'
            f'<td>{r["n_merges"]:,}</td>'
            f'<td>{r["pct_reduction"]:.1f}%</td>'
            f'<td>{note}</td>'
            f'</tr>'
        )
    return header + "".join(body_rows) + "</tbody></table>"


def render_relevance_breakdown(
    all_cids: list[int],
    merged_groups: dict[int, list[int]],
    manifest: dict,
) -> str:
    merged_cids = set(c for cids in merged_groups.values() for c in cids)
    cats = ["oncology-core", "oncology-adjacent", "navigational", "non-clinical"]
    total_by_cat: Counter = Counter()
    merged_by_cat: Counter = Counter()
    for cid in all_cids:
        cr = manifest.get(cid, {}).get("clinical_relevance", "unknown")
        total_by_cat[cr] += 1
        if cid in merged_cids:
            merged_by_cat[cr] += 1

    rows = []
    for cat in cats:
        tot = total_by_cat[cat]
        mrg = merged_by_cat[cat]
        pct = 100 * mrg / tot if tot > 0 else 0
        bar = (
            f'<div style="background:#e9ecef;border-radius:3px;height:14px;width:120px;display:inline-block">'
            f'<div style="background:#0d6efd;border-radius:3px;height:14px;width:{pct:.0f}%"></div>'
            f'</div>'
        )
        rows.append(
            f'<tr><td>{badge(cat)}</td>'
            f'<td>{tot:,}</td>'
            f'<td>{mrg:,}</td>'
            f'<td>{tot - mrg:,}</td>'
            f'<td>{pct:.1f}% {bar}</td></tr>'
        )
    header = (
        '<table class="info-table"><thead><tr>'
        '<th>Category</th><th>Total</th><th>Merged</th><th>Singleton</th><th>% Merged</th>'
        '</tr></thead><tbody>'
    )
    return header + "".join(rows) + "</tbody></table>"


def render_level_section(
    level_num: int,
    all_cids: list[int],
    merged_groups: dict[int, list[int]],
    singletons: list[int],
    manifest: dict,
    overview: dict,
    cluster_queries: dict,
    C: np.ndarray,
    idx_by_cid: dict,
    cutoff: float,
) -> str:
    parts: list[str] = []
    n_super = len(merged_groups) + len(singletons)
    n_merged_orig = sum(len(v) for v in merged_groups.values())

    parts.append(f'<section id="level-{level_num}">')

    if level_num == 0:
        parts.append(f'<h2>Level 0 — Original Clusters ({len(all_cids):,})</h2>')
        parts.append('<p style="color:#6c757d">The raw BERTopic HDBSCAN leaf-level output. No merging applied.</p>')
    else:
        parts.append(
            f'<h2>Level {level_num} — Merge (cutoff = {cutoff:.5f})</h2>'
        )
        parts.append(
            f'<p style="color:#6c757d">'
            f'{len(merged_groups):,} merged groups absorbing {n_merged_orig:,} original clusters '
            f'→ <b>{n_super:,} super-clusters</b> total '
            f'(down from {len(all_cids):,}).'
            f'</p>'
        )

    # Filter input
    parts.append(
        f'<div style="margin:12px 0">'
        f'<input type="search" class="level-filter" data-level="{level_num}" '
        f'placeholder="Filter by label, cluster ID, or query text..." '
        f'style="width:420px;padding:6px 10px;border:1px solid #ced4da;border-radius:4px;font-size:0.9em">'
        f'</div>'
    )

    if level_num == 0:
        # All 2,023 clusters sorted by n_queries desc
        cids_sorted = sorted(
            all_cids,
            key=lambda c: -manifest.get(c, {}).get("n_queries", overview.get(c, {}).get("size", 0))
        )
        parts.append(
            f'<details id="level-0-list">'
            f'<summary style="cursor:pointer;font-weight:600;padding:8px;'
            f'background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px">'
            f'Show all {len(cids_sorted):,} clusters (sorted by query count desc)</summary>'
            f'<div id="level-0-cards" style="margin-top:8px">'
        )
        for cid in cids_sorted:
            parts.append(render_cluster_card(cid, manifest, overview, cluster_queries))
        parts.append('</div></details>')

    else:
        # Merged groups section
        if merged_groups:
            # Sort merged groups by total n_queries desc
            groups_sorted = sorted(
                merged_groups.items(),
                key=lambda kv: -sum(
                    manifest.get(c, {}).get("n_queries", overview.get(c, {}).get("size", 0))
                    for c in kv[1]
                )
            )
            parts.append(f'<h3>Merged Groups ({len(merged_groups):,})</h3>')
            parts.append(
                f'<details id="level-{level_num}-merged" open>'
                f'<summary style="cursor:pointer;font-weight:600;padding:8px;'
                f'background:#fff3cd;border:1px solid #ffc107;border-radius:4px">'
                f'Show all {len(merged_groups):,} merged groups</summary>'
                f'<div id="level-{level_num}-merged-cards" style="margin-top:8px">'
            )
            for gnum, (_, cids) in enumerate(groups_sorted, 1):
                parts.append(
                    render_merged_card(gnum, cids, manifest, overview, cluster_queries,
                                       C, idx_by_cid)
                )
            parts.append('</div></details>')

            # Relevance breakdown for merged
            parts.append('<h4 style="margin-top:16px">Clinical Relevance in Merged Groups</h4>')
            parts.append(render_relevance_breakdown(all_cids, merged_groups, manifest))

        # Singletons section
        if singletons:
            singletons_sorted = sorted(
                singletons,
                key=lambda c: -manifest.get(c, {}).get("n_queries", overview.get(c, {}).get("size", 0))
            )
            parts.append(
                f'<h3 style="margin-top:24px">Unchanged Clusters ({len(singletons_sorted):,})</h3>'
            )
            parts.append(
                f'<details id="level-{level_num}-singletons">'
                f'<summary style="cursor:pointer;font-weight:600;padding:8px;'
                f'background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px">'
                f'Show all {len(singletons_sorted):,} unchanged clusters</summary>'
                f'<div id="level-{level_num}-singleton-cards" style="margin-top:8px">'
            )
            for cid in singletons_sorted:
                parts.append(render_cluster_card(cid, manifest, overview, cluster_queries))
            parts.append('</div></details>')

    parts.append('</section>')
    return "".join(parts)


# ─── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5;
    color: #212529;
    font-size: 14px;
}
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
h2 { font-size: 1.25rem; margin: 28px 0 10px 0; border-bottom: 2px solid #dee2e6; padding-bottom: 6px; }
h3 { font-size: 1.05rem; margin: 20px 0 8px 0; color: #495057; }
h4 { font-size: 0.95rem; margin: 12px 0 6px 0; color: #6c757d; }
.subtitle { color: #6c757d; margin-bottom: 20px; font-size: 0.9rem; }
nav { margin: 12px 0 24px 0; }
nav a { color: #0d6efd; text-decoration: none; margin-right: 16px; font-size: 0.9rem; }
nav a:hover { text-decoration: underline; }
section { margin-bottom: 40px; }

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.stat-card {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 14px;
    text-align: center;
}
.stat-card .stat-label { font-size: 0.8rem; color: #6c757d; margin-bottom: 4px; }
.stat-card .stat-value { font-size: 1.4rem; font-weight: 700; color: #212529; }

.badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 600;
    white-space: nowrap;
}

details.cluster-card {
    border: 1px solid #dee2e6;
    border-radius: 5px;
    margin: 4px 0;
    background: white;
}
details.cluster-card > summary {
    padding: 7px 12px;
    cursor: pointer;
    list-style: none;
    user-select: none;
    line-height: 1.5;
}
details.cluster-card > summary:hover { background: #f8f9fa; }
details.cluster-card > summary::-webkit-details-marker { display: none; }
details.cluster-card > summary::before {
    content: "▶ ";
    font-size: 0.7em;
    color: #adb5bd;
}
details.cluster-card[open] > summary::before { content: "▼ "; }

.card-body { padding: 8px 14px 12px 14px; border-top: 1px solid #f0f0f0; }

details.merged-card {
    border: 2px solid #ffc107;
    border-radius: 6px;
    margin: 6px 0;
    background: #fffdf5;
}
details.merged-card > summary {
    padding: 9px 14px;
    cursor: pointer;
    list-style: none;
    user-select: none;
    line-height: 1.5;
    font-size: 0.95rem;
}
details.merged-card > summary:hover { background: #fff8e1; }
details.merged-card > summary::-webkit-details-marker { display: none; }
details.merged-card > summary::before {
    content: "▶ ";
    font-size: 0.7em;
    color: #ffc107;
}
details.merged-card[open] > summary::before { content: "▼ "; }

details.query-list-details { margin: 4px 0; }
details.query-list-details > summary {
    cursor: pointer;
    list-style: none;
    padding: 3px 0;
}
details.query-list-details > summary::-webkit-details-marker { display: none; }

ol.query-list {
    margin: 6px 0 0 20px;
    padding: 0;
    font-size: 0.82rem;
    line-height: 1.6;
    columns: 2;
    column-gap: 24px;
    max-height: 400px;
    overflow-y: auto;
    background: #fafafa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    padding: 8px 8px 8px 28px;
}
@media (max-width: 900px) { ol.query-list { columns: 1; } }

.info-table {
    border-collapse: collapse;
    width: 100%;
    background: white;
    border-radius: 6px;
    overflow: hidden;
    font-size: 0.88rem;
    margin: 8px 0;
}
.info-table th {
    background: #343a40;
    color: white;
    padding: 7px 12px;
    text-align: left;
    font-weight: 600;
}
.info-table td { padding: 6px 12px; border-bottom: 1px solid #f0f0f0; }
.info-table tr:last-child td { border-bottom: none; }
.info-table tr:hover td { background: #f8f9fa; }

.cid { font-family: monospace; font-size: 0.85em; color: #6c757d; }

.level-comparison { width: 100%; border-collapse: collapse; background: white;
    border-radius: 6px; overflow: hidden; }
.level-comparison th { background: #495057; color: white; padding: 10px 16px; text-align: left; }
.level-comparison td { padding: 10px 16px; border-bottom: 1px solid #dee2e6; }
.level-comparison tr:last-child td { border-bottom: none; }

.hidden-by-filter { display: none !important; }
"""

# ─── JavaScript ──────────────────────────────────────────────────────────────────

JS = """
(function() {
    // Filter logic for each level section
    document.querySelectorAll('.level-filter').forEach(function(input) {
        input.addEventListener('input', function() {
            var val = this.value.toLowerCase().trim();
            var levelId = this.dataset.level;
            var section = document.getElementById('level-' + levelId);
            if (!section) return;

            // When filtering, open parent details so results are visible
            if (val) {
                section.querySelectorAll('details').forEach(function(d) {
                    // Only open top-level container details
                    if (d.id && d.id.startsWith('level-')) d.open = true;
                });
            }

            // Filter cluster-card and merged-card elements
            section.querySelectorAll('details.cluster-card, details.merged-card').forEach(function(card) {
                // Only filter top-level cards (not nested cluster-cards inside merged-cards)
                var parent = card.parentElement;
                var isNested = parent && parent.closest('details.merged-card') !== null
                               && card.classList.contains('cluster-card');
                if (isNested) return;

                if (!val) {
                    card.classList.remove('hidden-by-filter');
                    return;
                }
                var text = card.querySelector('summary').textContent.toLowerCase();
                // Also search inside query lists if available
                var queryText = '';
                var ql = card.querySelector('ol.query-list');
                if (ql) queryText = ql.textContent.toLowerCase();
                var match = text.includes(val) || queryText.includes(val);
                card.classList.toggle('hidden-by-filter', !match);
            });
        });
    });
})();
"""

# ─── Main HTML assembly ──────────────────────────────────────────────────────────

def render_html(
    cluster_ids: np.ndarray,
    C: np.ndarray,
    idx_by_cid: dict,
    manifest: dict,
    overview: dict,
    cluster_queries: dict,
    merged1: dict,
    singletons1: list,
    merged2: dict,
    singletons2: list,
    cutoff1: float,
    cutoff2: float,
    knee_rows: list[dict],
) -> str:
    all_cids = list(map(int, cluster_ids))
    n0 = len(all_cids)
    n1 = len(merged1) + len(singletons1)
    n2 = len(merged2) + len(singletons2)
    n_merged_orig1 = sum(len(v) for v in merged1.values())
    n_merged_orig2 = sum(len(v) for v in merged2.values())
    gen_date = str(date.today())

    parts: list[str] = []

    parts.append(f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BERTopic Merge Level Report</title>
  <style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>BERTopic Cluster Merge Report</h1>
<p class="subtitle">
  Generated: {gen_date} &nbsp;|&nbsp;
  Level 1 cutoff: <code>{cutoff1:.5f}</code> (5% of distance range) &nbsp;|&nbsp;
  Level 2 cutoff: <code>{cutoff2:.5f}</code> (10% of distance range)
</p>

<nav>
  <a href="#overview">Overview</a>
  <a href="#dendrogram">Dendrogram</a>
  <a href="#level-0">Level 0 (Original)</a>
  <a href="#level-1">Level 1 (First merge)</a>
  <a href="#level-2">Level 2 (Second merge)</a>
</nav>

<section id="overview">
<h2>Level Comparison</h2>
<table class="level-comparison">
  <thead>
    <tr>
      <th>Level</th>
      <th># Super-clusters</th>
      <th>Merged groups</th>
      <th>Clusters absorbed</th>
      <th>Singletons</th>
      <th>% Reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>0</b> — Original</td>
      <td><b>{n0:,}</b></td>
      <td>—</td>
      <td>—</td>
      <td>{n0:,}</td>
      <td>0%</td>
    </tr>
    <tr>
      <td><b>1</b> — First merge (cutoff {cutoff1:.5f})</td>
      <td><b>{n1:,}</b></td>
      <td>{len(merged1):,}</td>
      <td>{n_merged_orig1:,}</td>
      <td>{len(singletons1):,}</td>
      <td>{100*(n0-n1)/n0:.1f}%</td>
    </tr>
    <tr>
      <td><b>2</b> — Second merge (cutoff {cutoff2:.5f})</td>
      <td><b>{n2:,}</b></td>
      <td>{len(merged2):,}</td>
      <td>{n_merged_orig2:,}</td>
      <td>{len(singletons2):,}</td>
      <td>{100*(n0-n2)/n0:.1f}%</td>
    </tr>
  </tbody>
</table>

<div class="stats-grid" style="margin-top:16px">
  <div class="stat-card">
    <div class="stat-label">Level 0 clusters</div>
    <div class="stat-value">{n0:,}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Level 1 super-clusters</div>
    <div class="stat-value">{n1:,}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Level 2 super-clusters</div>
    <div class="stat-value">{n2:,}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">L1 merged groups</div>
    <div class="stat-value">{len(merged1):,}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">L2 merged groups</div>
    <div class="stat-value">{len(merged2):,}</div>
  </div>
</div>
</section>

<section id="dendrogram">
<h2>Dendrogram: Cluster Count vs. Distance Threshold</h2>
<p style="color:#6c757d;margin-bottom:8px;font-size:0.88rem">
  First 25% + upper portion of the distance range shown. Each merge reduces cluster count by 1.
  Highlighted rows mark the chosen Level 1 and Level 2 cutoffs.
</p>
{render_dendrogram_table(knee_rows)}
</section>
""")

    # Level 0, 1, 2 sections
    parts.append(render_level_section(
        0, all_cids, {}, all_cids,
        manifest, overview, cluster_queries, C, idx_by_cid, 0.0
    ))
    parts.append(render_level_section(
        1, all_cids, merged1, singletons1,
        manifest, overview, cluster_queries, C, idx_by_cid, cutoff1
    ))
    parts.append(render_level_section(
        2, all_cids, merged2, singletons2,
        manifest, overview, cluster_queries, C, idx_by_cid, cutoff2
    ))

    parts.append(f"""
</div><!-- /container -->
<script>{JS}</script>
</body>
</html>""")

    return "".join(parts)

# ─── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_ids, C, idx_by_cid, manifest, overview, cluster_queries = load_data()

    Z, dists = compute_linkage(C)
    cutoff1, cutoff2 = detect_cutoffs(dists, args.cutoff1, args.cutoff2)

    print("Applying Level 1 cut...", flush=True)
    merged1, singletons1 = apply_cut(Z, cutoff1, cluster_ids)
    print(f"  Merged groups: {len(merged1)}, singletons: {len(singletons1)}", flush=True)

    print("Applying Level 2 cut...", flush=True)
    merged2, singletons2 = apply_cut(Z, cutoff2, cluster_ids)
    print(f"  Merged groups: {len(merged2)}, singletons: {len(singletons2)}", flush=True)

    print("Building dendrogram table...", flush=True)
    knee_rows = build_knee_rows(dists, cutoff1, cutoff2)

    print("Rendering HTML...", flush=True)
    html_content = render_html(
        cluster_ids, C, idx_by_cid,
        manifest, overview, cluster_queries,
        merged1, singletons1,
        merged2, singletons2,
        cutoff1, cutoff2,
        knee_rows,
    )

    print(f"Writing {out_path}...", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done! {size_mb:.1f} MB -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
