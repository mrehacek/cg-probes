"""Probe-head comparison on the clinician gold: does a richer (still-linear) head
rescue ET, which the 1-D difference-in-means (DoM) direction handles poorly?

For each axis we fit several heads on the contrastive TRAIN(+DEV) split and
evaluate on the **human gold** (real-only) — the same split the headline uses.
Heads:
  * dom_paired   — the paper's 1-D paired-difference direction + 2 tuned thresholds
  * dom_extreme  — 1-D class-mean (g2-g0) direction + 2 tuned thresholds
  * linear       — multinomial logistic on the FULL embedding (linear, multi-dim)
  * linear_bal   — same, class_weight='balanced' (helps skewed ET)
  * mlp          — 1 hidden layer (non-linear upper bound)

Head selection is by **contrastive TEST** (no human-gold tuning → no leakage);
the human-gold column is reported for every head for transparency. Run per the
best cell/axis (by human real-only QWK) unless --all-cells.

  python -m benchmark.probe_methods                 # best cell per axis
  python -m benchmark.probe_methods --axis ET --all-cells
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.neural_network import MLPClassifier

from benchmark import _gold
from benchmark import embedders as E
from benchmark import probe as P
from benchmark.eval_golden import _emb_lookup, _vecs
from benchmark.figures import best_cells
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES

CACHE = REPO / "benchmark" / "cache"
DIRDIR = CACHE / "directions"
TIER1_HUMAN = CACHE / "tier1__human.json"
SEED = 0


def _qwk(y, p):
    return float(cohen_kappa_score(y, p, labels=[0, 1, 2], weights="quadratic"))


def _macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def _train_arrays(embedder, mode, axis):
    """Contrastive (X, ds) aligned; plus the cell's emb lookup for reuse."""
    emb = _emb_lookup(embedder, mode, axis)
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    return emb, X, ds


def _human_arrays(emb, axis):
    ga = _gold.golden_axis(axis, "human")
    Xg, keep = _vecs(ga["text"], emb)
    ga = ga[keep].reset_index(drop=True)
    real = ~ga["synthetic"].to_numpy()
    return Xg[real], ga["grade"].to_numpy()[real]


def evaluate_cell(embedder, mode, axis) -> dict:
    emb, X, ds = _train_arrays(embedder, mode, axis)
    tr = (ds["split"] == "train").to_numpy()
    dev = (ds["split"] == "dev").to_numpy()
    te = (ds["split"] == "test").to_numpy()
    y = ds["grade"].to_numpy()
    trdev = tr | dev
    Xh, yh = _human_arrays(emb, axis)

    out = {}

    def _record(name, pred_te, pred_h):
        out[name] = {
            "ctest_qwk": _qwk(y[te], pred_te), "ctest_macro": _macro(y[te], pred_te),
            "human_qwk": _qwk(yh, pred_h), "human_macro": _macro(yh, pred_h)}

    # 1-D DoM heads (threshold tuned on dev)
    for name, wfn in (("dom_paired", lambda: P.fit_paired_dom(X[tr], ds[tr].reset_index(drop=True))),
                      ("dom_extreme", lambda: P.fit_extreme_dom(X[tr], y[tr]))):
        w = wfn()
        t1, t2 = P.tune_two_thresholds(P.project(X[dev], w), y[dev])
        _record(name, P.predict_3class(P.project(X[te], w), t1, t2),
                P.predict_3class(P.project(Xh, w), t1, t2))

    # full-embedding linear + balanced + mlp (argmax; fit on train+dev)
    lin = LogisticRegression(max_iter=3000, C=1.0).fit(X[trdev], y[trdev])
    _record("linear", lin.predict(X[te]), lin.predict(Xh))
    linb = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X[trdev], y[trdev])
    _record("linear_bal", linb.predict(X[te]), linb.predict(Xh))
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
                        random_state=SEED).fit(X[trdev], y[trdev])
    _record("mlp", mlp.predict(X[te]), mlp.predict(Xh))
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=SAFETY_AXES, default=None)
    ap.add_argument("--all-cells", action="store_true",
                    help="sweep every embedder×mode (else just the best cell/axis)")
    a = ap.parse_args()
    best = best_cells(json.loads(TIER1_HUMAN.read_text(encoding="utf-8")))
    axes = [a.axis] if a.axis else SAFETY_AXES

    for axis in axes:
        if a.all_cells:
            cells = [(e, m) for e in E.REGISTRY for m in E.INSTR_MODES[e]
                     if (DIRDIR / f"{e}__{m}__{axis}.npy").exists()]
        else:
            b = best[axis]; cells = [(b["embedder"], b["mode"])]
        print(f"\n===== {axis} =====", flush=True)
        for emb, mode in cells:
            res = evaluate_cell(emb, mode, axis)
            print(f"[{emb}/{mode}]", flush=True)
            print(f"   {'head':12s} {'cTEST-QWK':>9s} {'human-QWK':>9s} {'human-macroF1':>13s}", flush=True)
            for name, d in res.items():
                star = "  <-- current" if name == "dom_paired" else ""
                print(f"   {name:12s} {d['ctest_qwk']:9.3f} {d['human_qwk']:9.3f} "
                      f"{d['human_macro']:13.3f}{star}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
