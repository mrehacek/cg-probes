"""P4 — fit probes per cell and produce the two headline evaluations.

For each cell (embedder, instruction-mode, axis):
  1. Fit the **paired-difference DoM** direction on the contrastive TRAIN split.
  2. Tune two thresholds on the contrastive DEV split (max macro-F1).
  3. Evaluate on:
       (a) contrastive TEST split  -> SEPARABILITY (no human labels needed);
       (b) golden-400              -> BENCHMARK, macro-F1 with & without synthetic.
  4. Robustness: extreme-DoM and OvR-DoM directions on the same split.
  5. Save the direction vector (for the cross-axis collinearity analysis).

Outputs:
  benchmark/cache/results_probe.json          (all cells' metrics)
  benchmark/cache/directions/{cell}.npy        (axis directions for metrics.py)

Run with the embeddings venv (sklearn). Cells whose embedding parquet is missing
are skipped with a notice (so gemini/openai can be evaluated before the HF
endpoints land).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from benchmark import embedders as E
from benchmark import probe as P
from benchmark import _gold
from benchmark.run_embed import _cell_path, _tid
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int

DIRDIR = REPO / "benchmark" / "cache" / "directions"
GOLDEN_400 = REPO / "golden" / "cache" / "golden_set_v1" / "golden_400_filled.parquet"


def _results_path(gold_source: str):
    tag = "" if gold_source == "silver" else f"__{gold_source}"
    return REPO / "benchmark" / "cache" / f"results_probe{tag}.json"


def _emb_lookup(embedder: str, mode: str, axis: str | None) -> dict | None:
    path = _cell_path(embedder, mode, axis if mode in E.PER_AXIS_MODES else None)
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return {r["text_id"]: np.asarray(r["embedding"], dtype=np.float64) for _, r in d.iterrows()}


def _vecs(texts, emb: dict):
    """(X, keep_mask) — rows for texts present in the embedding lookup."""
    keep, rows = [], []
    for t in texts:
        v = emb.get(_tid(str(t)))
        keep.append(v is not None)
        if v is not None:
            rows.append(v)
    X = np.vstack(rows) if rows else np.zeros((0, 1))
    return X, np.asarray(keep)


def _golden_axis(axis: str, gold_source: str = "silver") -> pd.DataFrame:
    """Per-axis gold via the shared loader (silver=pipeline-400, human=clinician)."""
    return _gold.golden_axis(axis, gold_source)


def eval_cell(embedder: str, mode: str, axis: str, gold_source: str = "silver") -> dict | None:
    emb = _emb_lookup(embedder, mode, axis)
    if emb is None:
        return None

    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X_all, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X_all):
        ds = ds.iloc[:len(X_all)].reset_index(drop=True)

    tr = (ds["split"] == "train").to_numpy()
    dev = (ds["split"] == "dev").to_numpy()
    te = (ds["split"] == "test").to_numpy()
    y = ds["grade"].to_numpy()

    # PRIMARY: paired-difference DoM (fit on train rows)
    w = P.fit_paired_dom(X_all[tr], ds[tr].reset_index(drop=True))
    s_dev, s_te = P.project(X_all[dev], w), P.project(X_all[te], w)
    t1, t2 = P.tune_two_thresholds(s_dev, y[dev])

    sep = {
        "macro_f1": P.macro_f1(y[te], P.predict_3class(s_te, t1, t2)),
        "n_test": int(te.sum()),
        "grade1_landing": P.grade1_landing(s_te, y[te], t1, t2),
        "thresholds": [t1, t2],
    }
    # robustness directions
    w_ext = P.fit_extreme_dom(X_all[tr], y[tr])
    s_dev_e, s_te_e = P.project(X_all[dev], w_ext), P.project(X_all[te], w_ext)
    te1, te2 = P.tune_two_thresholds(s_dev_e, y[dev])
    sep["macro_f1_extreme"] = P.macro_f1(y[te], P.predict_3class(s_te_e, te1, te2))
    dirs_ovr = P.fit_ovr_dom(X_all[tr], y[tr])
    sep["macro_f1_ovr"] = P.macro_f1(y[te], P.predict_ovr(X_all[te], dirs_ovr))

    # BENCHMARK on the gold set with the PRIMARY direction + thresholds
    gold = _golden_axis(axis, gold_source)
    Xg, keepg = _vecs(gold["text"], emb)
    gold = gold[keepg].reset_index(drop=True)
    sg = P.project(Xg, w)
    pg = P.predict_3class(sg, t1, t2)
    yg = gold["grade"].to_numpy()
    real = ~gold["synthetic"].to_numpy()
    bench = {
        "macro_f1_all": P.macro_f1(yg, pg),
        "macro_f1_real_only": (P.macro_f1(yg[real], pg[real]) if real.any() else None),
        "n_all": int(len(gold)), "n_real": int(real.sum()),
        "coverage": float(keepg.mean()),
    }

    DIRDIR.mkdir(parents=True, exist_ok=True)
    np.save(DIRDIR / f"{embedder}__{mode}__{axis}.npy", w)

    return {"embedder": embedder, "mode": mode, "axis": axis,
            "dim": int(X_all.shape[1]), "n_train": int(tr.sum()),
            "separability": sep, "benchmark": bench}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="*", default=E.REGISTRY)
    ap.add_argument("--modes", nargs="*",
                    default=["none", "generic", "per_axis", "per_axis_pos"])
    ap.add_argument("--gold-source", choices=["silver", "human"], default="silver",
                    help="benchmark against the pipeline-400 (silver) or clinician-200 (human) gold")
    a = ap.parse_args()
    RESULTS = _results_path(a.gold_source)

    results = []
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
    # index existing by cell to allow re-runs to overwrite
    idx = {(r["embedder"], r["mode"], r["axis"]): i for i, r in enumerate(results)}

    for embedder in a.embedders:
        for mode in a.modes:
            if mode not in E.INSTR_MODES.get(embedder, []):
                continue
            for axis in SAFETY_AXES:
                r = eval_cell(embedder, mode, axis, a.gold_source)
                if r is None:
                    print(f"[skip] {embedder}/{mode}/{axis} — no embeddings yet", flush=True)
                    continue
                key = (embedder, mode, axis)
                if key in idx:
                    results[idx[key]] = r
                else:
                    idx[key] = len(results); results.append(r)
                s, b = r["separability"], r["benchmark"]
                print(f"[{embedder}/{mode}/{axis}] SEP macro-F1={s['macro_f1']:.3f} "
                      f"(ext {s['macro_f1_extreme']:.3f}, ovr {s['macro_f1_ovr']:.3f}) | "
                      f"GOLD all={b['macro_f1_all']:.3f} real={b['macro_f1_real_only']} "
                      f"cov={b['coverage']:.2f}", flush=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {RESULTS} ({len(results)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
