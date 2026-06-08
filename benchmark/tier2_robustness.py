"""Tier-2 robustness tests: linearity sufficiency + selectivity control.

Both run on cached embeddings, no clinician labels.

  1. LINEAR-vs-MLP (Pimentel et al.; Hewitt & Liang).
     The paper claims the safety axes are *linearly* recoverable. To earn that we
     must show a non-linear probe of the same inputs does NOT do meaningfully
     better. We fit three heads on the contrastive TRAIN split and score the
     held-out contrastive TEST split:
       - dom      : the paper's 1-D paired-difference direction + tuned thresholds
       - linear   : multinomial logistic on the full embedding (linear, multi-dim)
       - mlp      : one hidden layer (non-linear) on the full embedding
     A small mlp-minus-dom gap => linear recovery is sufficient (the headline).
     logistic-minus-dom isolates how much the 1-D DoM compression costs vs a full
     linear read-out.

  2. SELECTIVITY (Hewitt & Liang 2019).
     A probe that scores well could be exploiting embedding structure unrelated to
     the concept. We refit the SAME DoM probe on **grade labels shuffled within the
     train split** (a control task with identical label marginals but no real
     signal) and measure macro-F1 on TEST. Selectivity = real - control; a large
     positive gap means the probe reads the concept, not an artifact. Averaged over
     several shuffles for a stable control estimate.

Output: benchmark/cache/tier2.json. Run with the embeddings venv.
  python -m benchmark.tier2_robustness
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from benchmark import embedders as E
from benchmark import probe as P
from benchmark.eval_golden import _emb_lookup, _vecs
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES

OUT = REPO / "benchmark" / "cache" / "tier2.json"
N_SHUFFLES = 5
SEED = 0


def _load_cell(embedder: str, mode: str, axis: str):
    emb = _emb_lookup(embedder, mode, axis)
    if emb is None:
        return None
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    return X, ds


def linear_vs_mlp(X: np.ndarray, ds: pd.DataFrame) -> dict:
    tr = (ds["split"] == "train").to_numpy()
    dev = (ds["split"] == "dev").to_numpy()
    te = (ds["split"] == "test").to_numpy()
    y = ds["grade"].to_numpy()

    # paper's 1-D DoM
    w = P.fit_paired_dom(X[tr], ds[tr].reset_index(drop=True))
    t1, t2 = P.tune_two_thresholds(P.project(X[dev], w), y[dev])
    f1_dom = P.macro_f1(y[te], P.predict_3class(P.project(X[te], w), t1, t2))

    # full-embedding linear read-out
    lin = LogisticRegression(max_iter=2000, C=1.0)
    lin.fit(X[tr], y[tr])
    f1_lin = P.macro_f1(y[te], lin.predict(X[te]))

    # one-hidden-layer non-linear read-out
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                        early_stopping=True, random_state=SEED)
    mlp.fit(X[tr], y[tr])
    f1_mlp = P.macro_f1(y[te], mlp.predict(X[te]))

    return {
        "f1_dom": float(f1_dom),
        "f1_linear_full": float(f1_lin),
        "f1_mlp": float(f1_mlp),
        "mlp_minus_dom": float(f1_mlp - f1_dom),
        "mlp_minus_linear": float(f1_mlp - f1_lin),
        "linear_minus_dom": float(f1_lin - f1_dom),
    }


def selectivity(X: np.ndarray, ds: pd.DataFrame) -> dict:
    tr = (ds["split"] == "train").to_numpy()
    dev = (ds["split"] == "dev").to_numpy()
    te = (ds["split"] == "test").to_numpy()
    y = ds["grade"].to_numpy()

    w = P.fit_paired_dom(X[tr], ds[tr].reset_index(drop=True))
    t1, t2 = P.tune_two_thresholds(P.project(X[dev], w), y[dev])
    f1_real = P.macro_f1(y[te], P.predict_3class(P.project(X[te], w), t1, t2))

    rng = np.random.RandomState(SEED)
    controls = []
    tr_idx = np.where(tr)[0]
    for _ in range(N_SHUFFLES):
        y_sh = y.copy()
        perm = rng.permutation(tr_idx)
        y_sh[tr_idx] = y[perm]
        ds_sh = ds.copy()
        ds_sh["grade"] = y_sh
        # shuffled extreme-DoM (paired needs pair structure; extreme matches the
        # control-task convention and the shuffled labels break the pairing anyway)
        try:
            w_c = P.fit_extreme_dom(X[tr], y_sh[tr])
            t1c, t2c = P.tune_two_thresholds(P.project(X[dev], w_c), y_sh[dev])
            controls.append(P.macro_f1(y[te], P.predict_3class(P.project(X[te], w_c), t1c, t2c)))
        except Exception:
            continue
    f1_ctrl = float(np.mean(controls)) if controls else None
    return {
        "f1_real": float(f1_real),
        "f1_control_mean": f1_ctrl,
        "f1_control_std": float(np.std(controls)) if controls else None,
        "selectivity": (float(f1_real - f1_ctrl) if f1_ctrl is not None else None),
        "n_shuffles": len(controls),
    }


def run(embedders: list[str], modes: list[str]) -> list[dict]:
    rows = []
    for embedder in embedders:
        for mode in modes:
            if mode not in E.INSTR_MODES.get(embedder, []):
                continue
            for axis in SAFETY_AXES:
                cell = _load_cell(embedder, mode, axis)
                if cell is None:
                    continue
                X, ds = cell
                lvm = linear_vs_mlp(X, ds)
                sel = selectivity(X, ds)
                rows.append({"embedder": embedder, "mode": mode, "axis": axis,
                             "linear_vs_mlp": lvm, "selectivity": sel})
                print(f"[{embedder}/{mode}/{axis}] "
                      f"dom={lvm['f1_dom']:.3f} lin={lvm['f1_linear_full']:.3f} "
                      f"mlp={lvm['f1_mlp']:.3f} (mlp-dom={lvm['mlp_minus_dom']:+.3f}) | "
                      f"sel={sel['selectivity']:+.3f} "
                      f"(real {sel['f1_real']:.3f} vs ctrl {sel['f1_control_mean']:.3f})",
                      flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="*", default=E.REGISTRY)
    ap.add_argument("--modes", nargs="*",
                    default=["none", "generic", "per_axis", "per_axis_pos"])
    a = ap.parse_args()
    rows = run(a.embedders, a.modes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {OUT} ({len(rows)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
