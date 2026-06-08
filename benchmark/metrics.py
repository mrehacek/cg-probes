"""P4 — aggregate probe results, cross-axis collinearity, and cost/latency.

Reads `results_probe.json` (from eval_golden) and the saved direction vectors,
and produces:
  * a tidy per-cell table (separability + golden macro-F1 ±synthetic);
  * per-(embedder, mode) **cross-axis direction cosine** matrices (MU/PU/ET) —
    the geometric-redundancy / PU↔ET-collinearity check;
  * a probe inference cost/latency figure (dot product vs a per-call LLM).

Human-vs-probe κ / quadratic-weighted κ are added once clinician labels are
pulled (annotations_pull). Run with the embeddings venv.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from contrastive.p2_io import REPO, SAFETY_AXES

CACHE = REPO / "benchmark" / "cache"
RESULTS = CACHE / "results_probe.json"
DIRDIR = CACHE / "directions"
OUT = CACHE / "results.json"


def load_results() -> list[dict]:
    if not RESULTS.exists():
        sys.exit(f"no results yet: {RESULTS} (run eval_golden first)")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def tidy(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        s, b = r["separability"], r["benchmark"]
        rows.append({
            "embedder": r["embedder"], "mode": r["mode"], "axis": r["axis"],
            "dim": r["dim"], "n_train": r["n_train"],
            "sep_f1": round(s["macro_f1"], 3),
            "sep_f1_extreme": round(s["macro_f1_extreme"], 3),
            "sep_f1_ovr": round(s["macro_f1_ovr"], 3),
            "gold_f1_all": round(b["macro_f1_all"], 3),
            "gold_f1_real": (round(b["macro_f1_real_only"], 3)
                             if b["macro_f1_real_only"] is not None else None),
            "gold_cov": round(b["coverage"], 2),
        })
    return pd.DataFrame(rows).sort_values(["embedder", "mode", "axis"]).reset_index(drop=True)


def collinearity(results: list[dict]) -> dict:
    """Per (embedder, mode): 3x3 cosine matrix of the MU/PU/ET directions."""
    cells = {(r["embedder"], r["mode"]) for r in results}
    out = {}
    for emb, mode in sorted(cells):
        dirs = {}
        for ax in SAFETY_AXES:
            p = DIRDIR / f"{emb}__{mode}__{ax}.npy"
            if p.exists():
                dirs[ax] = np.load(p)
        if len(dirs) < 2:
            continue
        axes = [a for a in SAFETY_AXES if a in dirs]
        M = pd.DataFrame(index=axes, columns=axes, dtype=float)
        for a in axes:
            for b in axes:
                da, db = dirs[a], dirs[b]
                M.loc[a, b] = float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12))
        out[f"{emb}__{mode}"] = M.round(3)
    return out


def cost_latency(dim: int = 4096, n: int = 100_000) -> dict:
    """Probe = one dot product per query. Time it vs a nominal LLM call."""
    rng = np.random.RandomState(0)
    X = rng.randn(n, dim).astype(np.float32)
    w = rng.randn(dim).astype(np.float32)
    t0 = time.time()
    _ = X @ w
    dt = time.time() - t0
    return {"probe_us_per_query": round(dt / n * 1e6, 3), "dim": dim,
            "note": "vs a per-call LLM classification at ~0.3-3 s + token cost"}


def main() -> int:
    results = load_results()
    df = tidy(results)
    print("=== per-cell probe results ===")
    print(df.to_string(index=False))

    print("\n=== cross-axis direction cosine (geometric redundancy) ===")
    coll = collinearity(results)
    for cell, M in coll.items():
        print(f"\n[{cell}]")
        print(M.to_string())

    cost = cost_latency()
    print(f"\n=== cost ===\nprobe: {cost['probe_us_per_query']} us/query (dim {cost['dim']}); {cost['note']}")

    OUT.write_text(json.dumps({
        "cells": df.to_dict("records"),
        "collinearity": {k: M.to_dict() for k, M in coll.items()},
        "cost": cost,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
