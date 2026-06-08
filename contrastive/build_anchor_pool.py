"""P2 step 3 — build the per-axis anchor pool from the golden picker output.

Reads `golden/cache/golden_set_v1/picks_raw.parquet` (10,128 graded picks, one
row per query, graded on all three safety axes), melts it into one row per
(axis, query, grade), and **excludes every query / supercluster used by the
golden-400 eval set** so the probe training pairs stay cluster-disjoint from the
benchmark.

Exclusion is by BOTH:
  * `query_key` — the 290 real golden picks (synthetic golden items aren't in
    picks_raw, but we union their keys defensively).
  * `supercluster_id` — every cluster the golden-400 draws from, including the
    `synthesis_cluster_id` topic-source clusters of the 110 synthetic items
    (those clusters seeded synthetic eval queries → must not seed train anchors).

Output: `contrastive/cache/anchors_safety.parquet`
    columns: axis, query_key, query_text, grade (0/1/2 int), grade_token,
             supercluster_id, clinical_relevance, source="picker"
Plus an HTML sample/sparsity report beside it (house rule).

Run from the cikm-ds repo root with either venv (pandas only):
    python -m contrastive.build_anchor_pool
"""

from __future__ import annotations

import html
import sys

import pandas as pd

from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES, AXIS_GRADE_COL

GOLDEN_DIR = REPO / "golden" / "cache" / "golden_set_v1"
PICKS_RAW = GOLDEN_DIR / "picks_raw.parquet"
GOLDEN_400 = GOLDEN_DIR / "golden_400_filled.parquet"

REPORT_HTML = p2_io.P2_CACHE / "anchors_safety_report.html"


def _golden_exclusions() -> tuple[set[str], set[str]]:
    """(query_keys, supercluster_ids) covered by the golden-400 eval set."""
    g = pd.read_parquet(GOLDEN_400)
    keys = set(g["query_key"].dropna().astype(str))
    clusters: set[str] = set(g["supercluster_id"].dropna().astype(str))
    # Synthetic items' topic-source clusters also seed the eval set.
    if "synthesis_cluster_id" in g.columns:
        clusters |= set(g["synthesis_cluster_id"].dropna().astype(str))
    return keys, clusters


def build() -> pd.DataFrame:
    picks = pd.read_parquet(PICKS_RAW)
    excl_keys, excl_clusters = _golden_exclusions()

    n_before = len(picks)
    picks = picks.copy()
    picks["query_key"] = picks["query_key"].astype(str)
    picks["supercluster_id"] = picks["supercluster_id"].astype(str)

    drop_mask = picks["query_key"].isin(excl_keys) | picks["supercluster_id"].isin(excl_clusters)
    kept = picks.loc[~drop_mask].copy()
    print(
        f"[exclude] golden keys={len(excl_keys)} clusters={len(excl_clusters)} "
        f"-> dropped {drop_mask.sum()}/{n_before} picks, {len(kept)} remain"
    )

    rows: list[dict] = []
    for axis in SAFETY_AXES:
        col = AXIS_GRADE_COL[axis]
        for _, r in kept.iterrows():
            token = r[col]
            if not isinstance(token, str) or not token.startswith(axis):
                continue  # missing / malformed grade for this axis
            rows.append(
                {
                    "axis": axis,
                    "query_key": r["query_key"],
                    "query_text": r["query_text"],
                    "grade": p2_io.token_to_int(token),
                    "grade_token": token,
                    "supercluster_id": r["supercluster_id"],
                    "clinical_relevance": r.get("clinical_relevance"),
                    "source": "picker",
                }
            )
    anchors = pd.DataFrame(rows)

    # Hard invariants: zero overlap with the eval set.
    assert anchors["query_key"].nunique() == len(set(anchors["query_key"]))
    assert not (set(anchors["query_key"]) & excl_keys), "anchor/golden query_key overlap!"
    assert not (set(anchors["supercluster_id"]) & excl_clusters), "anchor/golden cluster overlap!"

    return anchors


def sparsity_table(anchors: pd.DataFrame) -> pd.DataFrame:
    """Per-(axis, grade) counts of queries + distinct clusters."""
    g = (
        anchors.groupby(["axis", "grade"])
        .agg(n_queries=("query_key", "nunique"), n_clusters=("supercluster_id", "nunique"))
        .reset_index()
    )
    return g


def _spanning_clusters(anchors: pd.DataFrame) -> dict[str, int]:
    """Per axis: # clusters that contain ≥2 distinct grades (real contrast supply)."""
    out: dict[str, int] = {}
    for axis in SAFETY_AXES:
        sub = anchors[anchors["axis"] == axis]
        per_cluster = sub.groupby("supercluster_id")["grade"].nunique()
        out[axis] = int((per_cluster >= 2).sum())
    return out


def write_report(anchors: pd.DataFrame, spars: pd.DataFrame, spanning: dict[str, int]) -> None:
    p2_io.ensure_cache()
    pivot = spars.pivot(index="axis", columns="grade", values="n_queries").fillna(0).astype(int)
    pivot = pivot.reindex(SAFETY_AXES)

    parts: list[str] = [
        "<html><head><meta charset='utf-8'><title>P2 anchor pool</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}"
        "th{background:#f0f0f0}td.t{text-align:left;max-width:560px}</style></head><body>",
        "<h1>P2 contrastive anchor pool — sparsity</h1>",
        f"<p>Total anchor rows: {len(anchors)} "
        f"({anchors['query_key'].nunique()} distinct queries, "
        f"{anchors['supercluster_id'].nunique()} clusters). "
        "Cluster-disjoint from golden-400.</p>",
        "<h2>Queries per (axis, grade)</h2>",
        pivot.to_html(),
        "<h2>Clusters spanning ≥2 grades (real contrast supply)</h2>",
        "<ul>" + "".join(
            f"<li><b>{ax}</b>: {n} clusters</li>" for ax, n in spanning.items()
        ) + "</ul>",
    ]
    # Sample queries per (axis, grade).
    parts.append("<h2>Sample queries</h2>")
    for axis in SAFETY_AXES:
        parts.append(f"<h3>{axis} — {p2_io.AXIS_LONG_NAMES_EN[axis]}</h3><table>"
                     "<tr><th>grade</th><th class='t'>query</th><th class='t'>cluster</th></tr>")
        sub = anchors[anchors["axis"] == axis]
        for grade in (0, 1, 2):
            samp = sub[sub["grade"] == grade].head(6)
            for _, r in samp.iterrows():
                parts.append(
                    f"<tr><td>{grade}</td><td class='t'>{html.escape(str(r['query_text']))}</td>"
                    f"<td class='t'>{html.escape(str(r['supercluster_id']))}</td></tr>"
                )
        parts.append("</table>")
    parts.append("</body></html>")
    REPORT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"[report] {REPORT_HTML}")


def main() -> int:
    p2_io.ensure_cache()
    anchors = build()
    spars = sparsity_table(anchors)
    spanning = _spanning_clusters(anchors)

    print("\n[sparsity] queries per (axis, grade):")
    print(spars.to_string(index=False))
    print("\n[spanning] clusters with >=2 grades:", spanning)

    from golden.pipeline_v2._io import write_parquet_atomic
    write_parquet_atomic(anchors, p2_io.ANCHORS_SAFETY)
    print(f"\n[write] {p2_io.ANCHORS_SAFETY}  ({len(anchors)} rows)")
    write_report(anchors, spars, spanning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
