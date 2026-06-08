"""
Prepare classification dataset from category-constrained level-2 merge.

Each clinical_relevance category (oncology-core, oncology-adjacent, navigational,
non-clinical) is merged independently via UPGMA average-linkage on cosine distances
between L2-normalized centroids. Cross-category merges are impossible by construction.

Outputs (under cache/classification/ by default):
  superclusters_l2_constrained.jsonl  -- one supercluster per line (class definitions)
  queries_classified_l2.csv           -- flat table: one query per row with supercluster_id
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path(__file__).resolve().parents[1]
CENTROIDS_PARQUET = ROOT / "cache/clusters/centroids_qwen3.parquet"
MANIFEST_JSONL    = ROOT / "cache/clusters/cluster_manifest.jsonl"
OVERVIEW_JSONL    = ROOT / "cache/clusters/cluster_overview_qwen3_minc10_leaf.jsonl"
ASSIGNMENTS_CSV   = ROOT / "cache/clusters/bertopic_assignments_qwen3_minc10_leaf.csv"
DEFAULT_OUT_DIR   = ROOT / "cache/classification"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cutoff", type=float, default=0.10,
        help="Fraction of within-category distance range to use as merge cutoff (default 0.10)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory (default: cache/classification/)",
    )
    return p.parse_args()


def load_data():
    print("Loading centroids...", flush=True)
    cent_df = pd.read_parquet(CENTROIDS_PARQUET)
    cluster_ids = cent_df["cluster_id"].to_numpy(dtype=int)
    C = np.array(cent_df["centroid"].tolist(), dtype=np.float32)
    idx_by_cid = {int(cid): i for i, cid in enumerate(cluster_ids)}
    print(f"  {len(cluster_ids)} centroids, shape {C.shape}", flush=True)

    print("Loading manifest...", flush=True)
    manifest = {}
    with open(MANIFEST_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            manifest[int(rec["topic_id"])] = rec
    print(f"  {len(manifest)} manifest entries", flush=True)

    print("Loading cluster overview...", flush=True)
    overview = {}
    with open(OVERVIEW_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            overview[int(rec["cluster_id"])] = rec

    print("Loading query assignments...", flush=True)
    assign_df = pd.read_csv(ASSIGNMENTS_CSV)
    cluster_queries: dict[int, list[dict]] = defaultdict(list)
    for _, row in assign_df.iterrows():
        cid = int(row["cluster_id"])
        cluster_queries[cid].append({
            "query_key": str(row["query_key"]),
            "query":     str(row["query"]),
            "clicks":    int(row["clicks"]),
            "cluster_id": cid,
        })
    for cid in cluster_queries:
        cluster_queries[cid].sort(key=lambda q: -q["clicks"])
    print(f"  {len(assign_df)} query assignments across {len(cluster_queries)} clusters", flush=True)

    return cluster_ids, C, idx_by_cid, manifest, overview, cluster_queries


def constrained_merge(
    cluster_ids: np.ndarray,
    C: np.ndarray,
    idx_by_cid: dict,
    manifest: dict,
    cutoff_pct: float,
) -> list[tuple[str, list[int]]]:
    """
    Per-category UPGMA merge. Returns list of (clinical_relevance, [cluster_ids]).
    Each group is guaranteed to be within one category.
    """
    cat_to_cids: dict[str, list[int]] = defaultdict(list)
    for cid in cluster_ids:
        rel = manifest.get(int(cid), {}).get("clinical_relevance", "unknown")
        cat_to_cids[rel].append(int(cid))

    superclusters: list[tuple[str, list[int]]] = []

    for cat in sorted(cat_to_cids):
        cids = cat_to_cids[cat]
        print(f"  {cat}: {len(cids)} clusters", flush=True)

        if len(cids) == 1:
            superclusters.append((cat, cids))
            continue

        idxs = [idx_by_cid[cid] for cid in cids]
        C_sub = C[idxs].astype(np.float64)
        n = len(cids)
        condensed = cosine_distances(C_sub)[np.triu_indices(n, k=1)]
        Z = linkage(condensed, method="average")
        dists = Z[:, 2]
        d_min, d_max = float(dists.min()), float(dists.max())
        cutoff = d_min + cutoff_pct * (d_max - d_min)
        print(
            f"    distance range [{d_min:.5f}, {d_max:.5f}], cutoff={cutoff:.5f}",
            flush=True,
        )

        labels = fcluster(Z, t=cutoff, criterion="distance")
        super_to_orig: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            super_to_orig[int(lbl)].append(cids[i])

        n_merged = sum(1 for g in super_to_orig.values() if len(g) > 1)
        n_sing = sum(1 for g in super_to_orig.values() if len(g) == 1)
        print(
            f"    -> {len(super_to_orig)} superclusters "
            f"({n_merged} merged, {n_sing} singletons)",
            flush=True,
        )

        for group_cids in super_to_orig.values():
            superclusters.append((cat, sorted(group_cids)))

    return superclusters


def build_supercluster_records(
    superclusters: list[tuple[str, list[int]]],
    manifest: dict,
    overview: dict,
    cluster_queries: dict,
) -> list[dict]:
    """
    Assign IDs (category-scoped, ordered by n_queries desc) and build full records.
    """
    # Sort within each category by total query count desc, then assign IDs
    cat_groups: dict[str, list[tuple[list[int], int]]] = defaultdict(list)
    for cat, cids in superclusters:
        n_q = sum(manifest.get(cid, {}).get("n_queries", 0) for cid in cids)
        cat_groups[cat].append((cids, n_q))

    records = []
    for cat in sorted(cat_groups):
        groups = sorted(cat_groups[cat], key=lambda x: -x[1])
        for seq, (cids, n_q) in enumerate(groups, start=1):
            sc_id = f"{cat}_{seq:04d}"

            # Representative from largest constituent by n_queries
            anchor = max(
                cids,
                key=lambda cid: manifest.get(cid, {}).get("n_queries", 0),
            )
            anchor_m = manifest.get(anchor, {})

            # Merge top_words: anchor first, then others, deduplicated
            seen_words: set[str] = set()
            merged_words: list[str] = []
            for cid in [anchor] + [c for c in cids if c != anchor]:
                for w in overview.get(cid, {}).get("top_words", []):
                    if w not in seen_words:
                        seen_words.add(w)
                        merged_words.append(w)

            # Merge queries: union sorted by clicks desc
            all_queries: list[dict] = []
            for cid in cids:
                all_queries.extend(cluster_queries.get(cid, []))
            all_queries.sort(key=lambda q: -q["clicks"])

            records.append({
                "supercluster_id":       sc_id,
                "clinical_relevance":    cat,
                "constituent_cluster_ids": cids,
                "n_constituent_clusters": len(cids),
                "n_queries":             n_q,
                "czech_label":           anchor_m.get("czech_label", ""),
                "summary_cs":            anchor_m.get("summary_cs", ""),
                "urgency_potential":     anchor_m.get("urgency_potential", ""),
                "emotional_potential":   anchor_m.get("emotional_potential", ""),
                "top_words":             merged_words,
                "queries":               all_queries,
            })

    return records


def print_summary(records: list[dict]) -> None:
    cat_stats: dict[str, dict] = {}
    for rec in records:
        cat = rec["clinical_relevance"]
        if cat not in cat_stats:
            cat_stats[cat] = {"superclusters": 0, "merged": 0, "singletons": 0}
        cat_stats[cat]["superclusters"] += 1
        if rec["n_constituent_clusters"] > 1:
            cat_stats[cat]["merged"] += 1
        else:
            cat_stats[cat]["singletons"] += 1

    orig_counts = {
        "oncology-core": 647,
        "oncology-adjacent": 821,
        "navigational": 517,
        "non-clinical": 38,
    }

    print(
        f"\n{'Category':<20} {'orig':>6} {'superclusters':>13} "
        f"{'merged_groups':>13} {'singletons':>10}",
        flush=True,
    )
    print("-" * 66, flush=True)
    total_orig = total_sc = 0
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        orig = orig_counts.get(cat, "?")
        print(
            f"{cat:<20} {orig:>6} {s['superclusters']:>13} "
            f"{s['merged']:>13} {s['singletons']:>10}",
            flush=True,
        )
        total_orig += orig if isinstance(orig, int) else 0
        total_sc += s["superclusters"]
    print("-" * 66, flush=True)
    print(
        f"{'TOTAL':<20} {total_orig:>6} {total_sc:>13}",
        flush=True,
    )
    reduction = 100 * (1 - total_sc / total_orig) if total_orig else 0
    print(f"\nCluster count reduction: {reduction:.1f}%", flush=True)


def write_outputs(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSONL ---
    jsonl_path = out_dir / "superclusters_l2_constrained.jsonl"
    print(f"\nWriting {jsonl_path}...", flush=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_mb = jsonl_path.stat().st_size / 1_048_576
    print(f"  {len(records)} superclusters, {size_mb:.1f} MB", flush=True)

    # --- CSV (flat) ---
    csv_path = out_dir / "queries_classified_l2.csv"
    print(f"Writing {csv_path}...", flush=True)
    rows = []
    for rec in records:
        sc_id = rec["supercluster_id"]
        cat = rec["clinical_relevance"]
        for q in rec["queries"]:
            rows.append({
                "supercluster_id":    sc_id,
                "clinical_relevance": cat,
                "query_key":          q["query_key"],
                "query":              q["query"],
                "clicks":             q["clicks"],
                "original_cluster_id": q["cluster_id"],
            })
    flat_df = pd.DataFrame(rows)
    flat_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  {len(flat_df)} rows", flush=True)


def main() -> int:
    args = parse_args()

    cluster_ids, C, idx_by_cid, manifest, overview, cluster_queries = load_data()

    print(f"\nRunning per-category constrained merge (cutoff={args.cutoff:.0%})...", flush=True)
    superclusters = constrained_merge(cluster_ids, C, idx_by_cid, manifest, args.cutoff)

    print("\nBuilding supercluster records...", flush=True)
    records = build_supercluster_records(superclusters, manifest, overview, cluster_queries)

    print_summary(records)
    write_outputs(records, args.out_dir)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
