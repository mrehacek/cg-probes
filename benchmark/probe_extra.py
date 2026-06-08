"""Extra probe-head experiments on the clinician gold (cached vectors, no new cost):
ordinal (cumulative-link) head, multi-embedder concatenation, and PCA-whitening —
to see whether anything beats the class-balanced full-linear head before we invest
in heavier ET-specific contrastive pairs.

All heads are fit on the contrastive TRAIN+DEV split and evaluated on the human
gold (real-only QWK/macro-F1). Reference = full-linear class-balanced on the best
single embedder/axis (the current headline candidate).

  python -m benchmark.probe_extra
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score

from benchmark import _gold
from benchmark import embedders as E
from benchmark.eval_golden import _emb_lookup, _vecs
from benchmark.figures import best_cells
from benchmark.run_embed import _tid
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES

CACHE = REPO / "benchmark" / "cache"
TIER1_HUMAN = CACHE / "tier1__human.json"
CONCAT_EMBEDDERS = ["qwen8b", "gemini", "harrier27b", "openai3large"]
SEED = 0


def _qwk(y, p):
    return float(cohen_kappa_score(y, p, labels=[0, 1, 2], weights="quadratic"))


def _macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def _cell_xy(embedder, mode, axis):
    emb = _emb_lookup(embedder, mode, axis)
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    trdev = ds["split"].isin(["train", "dev"]).to_numpy()
    ga = _gold.golden_axis(axis, "human")
    Xg, keepg = _vecs(ga["text"], emb)
    ga = ga[keepg].reset_index(drop=True)
    real = ~ga["synthetic"].to_numpy()
    return X[trdev], ds["grade"].to_numpy()[trdev], Xg[real], ga["grade"].to_numpy()[real]


def _concat_xy(axis):
    """Concatenate the generic-mode vectors of all embedders (texts present in all)."""
    looks = {e: _emb_lookup(e, "generic", axis if "generic" in E.PER_AXIS_MODES else None)
             for e in CONCAT_EMBEDDERS}
    looks = {e: l for e, l in looks.items() if l is not None}
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    ga = _gold.golden_axis(axis, "human")

    def _stack(texts):
        rows, keep = [], []
        for t in texts:
            vs = [looks[e].get(_tid(str(t))) for e in looks]
            ok = all(v is not None for v in vs)
            keep.append(ok)
            if ok:
                rows.append(np.concatenate(vs))
        return (np.vstack(rows) if rows else np.zeros((0, 1))), np.asarray(keep)

    Xtr, ktr = _stack(ds["text"])
    ds = ds[ktr].reset_index(drop=True)
    trdev = ds["split"].isin(["train", "dev"]).to_numpy()
    Xg, kg = _stack(ga["text"])
    ga = ga[kg].reset_index(drop=True)
    real = ~ga["synthetic"].to_numpy()
    return Xtr[trdev], ds["grade"].to_numpy()[trdev], Xg[real], ga["grade"].to_numpy()[real]


def _linear(Xtr, ytr, Xev):
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    return clf.predict(Xev)


def _ordinal(Xtr, ytr, Xev):
    """Frank & Hall cumulative-link ordinal from two balanced binary logistics."""
    c1 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, (ytr >= 1).astype(int))
    c2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, (ytr >= 2).astype(int))
    p1 = c1.predict_proba(Xev)[:, 1]
    p2 = np.minimum(c2.predict_proba(Xev)[:, 1], p1)
    probs = np.vstack([1 - p1, p1 - p2, p2]).T
    return probs.argmax(axis=1)


def _whiten(Xtr, ytr, Xev, k=256):
    pca = PCA(n_components=min(k, Xtr.shape[1], Xtr.shape[0] - 1), whiten=True,
              random_state=SEED).fit(Xtr)
    return _linear(pca.transform(Xtr), ytr, pca.transform(Xev))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    best = best_cells(json.loads(TIER1_HUMAN.read_text(encoding="utf-8")))
    print(f"{'axis':4s} {'head':22s} {'human-QWK':>9s} {'human-macroF1':>13s}", flush=True)
    summary = {}
    for axis in SAFETY_AXES:
        b = best[axis]
        Xtr, ytr, Xev, yev = _cell_xy(b["embedder"], b["mode"], axis)
        rows = {
            "linear (best emb)": _linear(Xtr, ytr, Xev),
            "ordinal (best emb)": _ordinal(Xtr, ytr, Xev),
            "whiten+linear": _whiten(Xtr, ytr, Xev),
        }
        Xc_tr, yc_tr, Xc_ev, yc_ev = _concat_xy(axis)
        rows["concat-linear (4 emb)"] = (_linear(Xc_tr, yc_tr, Xc_ev), yc_ev)
        rows["concat-ordinal (4 emb)"] = (_ordinal(Xc_tr, yc_tr, Xc_ev), yc_ev)

        print(f"--- {axis}  (best={b['embedder']}/{b['mode']}, concat n={len(yc_ev)} vs {len(yev)}) ---",
              flush=True)
        summary[axis] = {}
        for name, pred in rows.items():
            yy = yc_ev if isinstance(pred, tuple) else yev
            pp = pred[0] if isinstance(pred, tuple) else pred
            q, m = _qwk(yy, pp), _macro(yy, pp)
            summary[axis][name] = {"qwk": q, "macro": m}
            print(f"{axis:4s} {name:22s} {q:9.3f} {m:13.3f}", flush=True)
    (CACHE / "probe_extra.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[write] {CACHE/'probe_extra.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
