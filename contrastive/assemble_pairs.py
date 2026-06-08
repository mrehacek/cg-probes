"""P2 step 5 — assemble per-axis probe datasets from anchors + synth variants.

Produces, per axis, a long labeled dataset the P4 probe consumes:
`contrastive/cache/probe_dataset_{axis}.parquet` with columns
    text, grade (0/1/2), axis, supercluster_id, source ("real"|"synth"),
    query_key, pair_key, anchor_grade, split ("train"|"dev"|"test")

Design:
  * **Positives** (grade 2; PU also grade 1) = verifier-confirmed synthetic
    variants (MU/PU). ET positives are **real** ET2.
  * **Negatives** (grade 0) are **cluster-matched** to the positives — drawn from
    the SAME superclusters as the grade-2 positives — so the difference-in-means
    direction is topic-controlled (the PU<->ET collinearity fix; complements the
    probe's within-anchor paired difference). Real, capped to a few× positives.
  * **Grade 1** carried for the landing analysis (real MU1; real ET1; synthetic
    confirmed PU1).
  * `pair_key` links a synthetic variant to its real anchor (same topic); the
    probe forms paired differences from it. For real rows pair_key == query_key.
  * **Cluster-disjoint split** `sha256(supercluster_id) % 5`: test (==0), dev
    (==1), train (else). No supercluster crosses splits — asserted.

The evaluation-400 set is held out entirely upstream (build_anchor_pool already
excluded its queries + clusters), so this contrastive test split is a clean,
human-label-free separability benchmark.

Run from the repo root:
    python -m contrastive.assemble_pairs
"""

from __future__ import annotations

import hashlib
import html
import sys

import pandas as pd

from contrastive import p2_io
from contrastive.p2_io import SAFETY_AXES, AXIS_GRADE_COL
from golden.pipeline_v2._io import write_parquet_atomic

NEG_RATIO = 3        # grade-0 negatives per grade-2 positive (cap)
NEG_CAP_PER_CLUSTER = 6
G1_CAP = 400         # cap grade-1 rows per axis (landing analysis)
ET_CAP_PER_GRADE = 900


def _split(scid) -> str:
    h = int(hashlib.sha256(str(scid).encode("utf-8")).hexdigest(), 16) % 5
    return "test" if h == 0 else "dev" if h == 1 else "train"


def _real_anchors() -> pd.DataFrame:
    a = pd.read_parquet(p2_io.ANCHORS_SAFETY)
    a["query_key"] = a["query_key"].astype(str)
    a["supercluster_id"] = a["supercluster_id"].astype(str)
    return a


def _confirmed_synth(axis: str) -> pd.DataFrame:
    path = p2_io.synth_variants_path(axis)
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(path)
    d = d[d["verifier_confirmed"]].copy()
    d["supercluster_id"] = d["supercluster_id"].astype(str)
    d["anchor_query_key"] = d["anchor_query_key"].astype(str)
    return d


def _cluster_matched_negatives(real_g0: pd.DataFrame, pos_clusters: set,
                               n_target: int) -> pd.DataFrame:
    """Real grade-0 rows drawn from the positives' superclusters, capped."""
    in_cluster = real_g0[real_g0["supercluster_id"].isin(pos_clusters)]
    capped = in_cluster.groupby("supercluster_id", group_keys=False).head(NEG_CAP_PER_CLUSTER)
    capped = capped.sort_values("query_key")
    if len(capped) > n_target:
        capped = capped.head(n_target)
    return capped


def _row(text, grade, axis, scid, source, query_key, pair_key, anchor_grade) -> dict:
    return {
        "text": text, "grade": int(grade), "axis": axis,
        "supercluster_id": str(scid), "source": source,
        "query_key": str(query_key), "pair_key": str(pair_key),
        "anchor_grade": (int(anchor_grade) if anchor_grade is not None else None),
    }


def _build_axis(axis: str, anchors: pd.DataFrame) -> pd.DataFrame:
    col = AXIS_GRADE_COL[axis]
    real_axis = anchors  # anchors_safety rows are per (axis, query) already
    real = real_axis[real_axis["axis"] == axis]
    g0 = real[real["grade"] == 0]
    g1 = real[real["grade"] == 1]
    g2 = real[real["grade"] == 2]

    rows: list[dict] = []

    if axis == "ET":
        # All real. Cap each grade; pair_key == query_key.
        for gdf, g in ((g0, 0), (g1, 1), (g2, 2)):
            take = gdf.sort_values("query_key").head(ET_CAP_PER_GRADE)
            for _, r in take.iterrows():
                rows.append(_row(r["query_text"], g, axis, r["supercluster_id"],
                                 "real", r["query_key"], r["query_key"], g))
        return pd.DataFrame(rows)

    # MU / PU: synthetic positives + cluster-matched real grade-0 negatives.
    synth = _confirmed_synth(axis)
    pos2 = synth[synth["target_grade"] == 2]
    pos1 = synth[synth["target_grade"] == 1]  # PU only (MU has no grade-1 synth)

    pos_clusters = set(pos2["supercluster_id"]) | set(pos1["supercluster_id"])

    # grade-2 positives (synthetic)
    for _, r in pos2.iterrows():
        rows.append(_row(r["text"], 2, axis, r["supercluster_id"], "synth",
                         r["anchor_query_key"] + ":v2", r["anchor_query_key"], r["anchor_grade"]))
    # grade-1: synthetic (PU) or real (MU)
    if len(pos1):
        for _, r in pos1.head(G1_CAP).iterrows():
            rows.append(_row(r["text"], 1, axis, r["supercluster_id"], "synth",
                             r["anchor_query_key"] + ":v1", r["anchor_query_key"], r["anchor_grade"]))
    else:
        for _, r in g1.sort_values("query_key").head(G1_CAP).iterrows():
            rows.append(_row(r["query_text"], 1, axis, r["supercluster_id"],
                             "real", r["query_key"], r["query_key"], 1))

    # grade-0 negatives: cluster-matched real, capped to NEG_RATIO x positives
    n_target = NEG_RATIO * len(pos2)
    negs = _cluster_matched_negatives(g0, pos_clusters, n_target)
    for _, r in negs.iterrows():
        rows.append(_row(r["query_text"], 0, axis, r["supercluster_id"],
                         "real", r["query_key"], r["query_key"], 0))
    return pd.DataFrame(rows)


def _write_report(axis: str, df: pd.DataFrame, path) -> None:
    piv = df.pivot_table(index="split", columns="grade", values="text",
                         aggfunc="count", fill_value=0)
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:4px 10px}"
        "td.t{text-align:left;max-width:520px}</style></head><body>",
        f"<h1>P2 probe dataset — {axis} ({p2_io.AXIS_LONG_NAMES_EN[axis]})</h1>",
        f"<p>{len(df)} rows · sources {df['source'].value_counts().to_dict()} · "
        f"grades {df['grade'].value_counts().sort_index().to_dict()}</p>",
        "<h3>rows per split × grade (clusters are disjoint across splits)</h3>",
        piv.to_html(),
        "<h3>sample grade-2 vs grade-0 (topic-matched by cluster)</h3><table>"
        "<tr><th>grade</th><th>src</th><th class='t'>text</th><th>cluster</th></tr>",
    ]
    for _, r in df.sort_values(["supercluster_id", "grade"]).head(30).iterrows():
        parts.append(f"<tr><td>{r['grade']}</td><td>{r['source']}</td>"
                     f"<td class='t'>{html.escape(str(r['text']))}</td>"
                     f"<td>{html.escape(str(r['supercluster_id']))}</td></tr>")
    parts.append("</table></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    p2_io.ensure_cache()
    anchors = _real_anchors()
    summary = []
    for axis in SAFETY_AXES:
        df = _build_axis(axis, anchors)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df["split"] = df["supercluster_id"].map(_split)

        # cluster-disjoint invariant
        per_cluster_splits = df.groupby("supercluster_id")["split"].nunique()
        assert (per_cluster_splits == 1).all(), "a supercluster crosses splits!"

        out = p2_io.probe_dataset_path(axis)
        write_parquet_atomic(df, out)
        report = out.with_name(f"probe_dataset_{axis}_report.html")
        _write_report(axis, df, report)

        gc = df.groupby(["split", "grade"]).size().unstack(fill_value=0)
        print(f"[{axis}] {len(df)} rows -> {out}")
        print(gc.to_string())
        print(f"      report {report}\n")
        summary.append((axis, len(df), df["grade"].value_counts().sort_index().to_dict()))

    print("=== P2 probe datasets assembled ===")
    for ax, n, g in summary:
        print(f"  {ax}: {n} rows, grades {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
