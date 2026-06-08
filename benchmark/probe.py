"""P4 — 3-class ordinal difference-in-means probe (the computational core).

One axis direction `w` per (axis, embedder, instruction-mode); project queries
onto it; two tuned thresholds split the 1-D projection into the ordinal grades
0/1/2. Operates on L2-normalized embeddings.

Direction estimators
  * `fit_paired_dom`  — PRIMARY. Mean of **within-anchor differences**
    `emb(grade2) − emb(matched grade0)`, where the matched grade-0 is the
    variant's own real anchor (same topic) when available, else the same
    supercluster's mean grade-0, else the global grade-0 mean. Cancels the
    topic / ET component → controls the PU↔ET confound.
  * `fit_extreme_dom` — class-mean grade2 − grade0 (robustness / comparison).
  * `fit_ovr_dom`     — one-vs-rest per grade, argmax (robustness).

Thresholding
  * `tune_two_thresholds` — quantile grid over t1<t2 maximizing dev macro-F1.
  * `predict_3class`, `grade1_landing`.

Collinearity
  * `direction_cosines`, `residualize` (Gram-Schmidt one direction out of another).

Pure numpy/sklearn — no embedding or LLM dependency. Run with the embeddings
venv (has sklearn). `python -m benchmark.probe --selftest` validates the math on
synthetic separable data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + EPS)


# --- direction estimators ----------------------------------------------------

def fit_paired_dom(X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Paired-difference direction on the TRAIN rows in `df` (aligned to X rows).

    `df` columns used: grade, supercluster_id, query_key, pair_key. Each grade-2
    row is matched to a grade-0 reference (its anchor via pair_key, else cluster
    mean, else global mean); the unit-normalized mean of the differences is `w`.
    """
    grade = df["grade"].to_numpy()
    qkey = df["query_key"].astype(str).to_numpy()
    pkey = df["pair_key"].astype(str).to_numpy()
    scid = df["supercluster_id"].astype(str).to_numpy()

    idx_by_key = {k: i for i, k in enumerate(qkey)}
    g0_mask = grade == 0
    global_g0 = X[g0_mask].mean(axis=0) if g0_mask.any() else X.mean(axis=0)
    # per-cluster grade-0 mean
    cluster_g0: dict[str, np.ndarray] = {}
    if g0_mask.any():
        g0_df = pd.DataFrame({"scid": scid[g0_mask]})
        for c, sub in g0_df.groupby("scid"):
            cluster_g0[c] = X[g0_mask][sub.index.to_numpy()].mean(axis=0)

    diffs = []
    for i in np.where(grade == 2)[0]:
        pk = pkey[i]
        if pk in idx_by_key and grade[idx_by_key[pk]] == 0:
            neg = X[idx_by_key[pk]]
        else:
            neg = cluster_g0.get(scid[i], global_g0)
        diffs.append(X[i] - neg)
    if not diffs:
        raise ValueError("no grade-2 rows to fit a paired direction")
    return _unit(np.mean(diffs, axis=0))


def fit_extreme_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class-mean grade-2 minus grade-0 (ignores grade-1)."""
    return _unit(X[y == 2].mean(axis=0) - X[y == 0].mean(axis=0))


def fit_ovr_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-vs-rest centroid direction per grade. Returns (3, d)."""
    dirs = np.zeros((3, X.shape[1]), dtype=np.float64)
    for g in (0, 1, 2):
        if (y == g).any():
            dirs[g] = _unit(X[y == g].mean(axis=0) - X[y != g].mean(axis=0))
    return dirs


def project(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


# --- thresholding ------------------------------------------------------------

def tune_two_thresholds(scores: np.ndarray, y: np.ndarray,
                        n_grid: int = 60) -> tuple[float, float]:
    """Grid over t1<t2 (score quantiles) maximizing 3-class macro-F1 on (scores, y)."""
    qs = np.quantile(scores, np.linspace(0.02, 0.98, n_grid))
    qs = np.unique(qs)
    best, best_f1 = (float(np.median(scores)), float(np.median(scores))), -1.0
    for a in range(len(qs)):
        for b in range(a + 1, len(qs)):
            t1, t2 = qs[a], qs[b]
            pred = predict_3class(scores, t1, t2)
            f1 = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best = f1, (float(t1), float(t2))
    return best


def predict_3class(scores: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(scores >= t2, 2, np.where(scores >= t1, 1, 0))


def predict_ovr(X: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    return np.argmax(X @ dirs.T, axis=1)


def grade1_landing(scores: np.ndarray, y: np.ndarray, t1: float, t2: float) -> dict:
    """Where do true grade-1 items land under the thresholds?"""
    g1 = predict_3class(scores[y == 1], t1, t2)
    n = max(len(g1), 1)
    return {f"pred_{k}": int((g1 == k).sum()) for k in (0, 1, 2)} | {"n": int((y == 1).sum())}


# --- ordinal-logistic robustness (on the 1-D projection) ---------------------

def fit_score_logistic(scores_tr: np.ndarray, y_tr: np.ndarray) -> LogisticRegression:
    """Multinomial logistic on the scalar projection — a smoother ordinal head
    than hard thresholds, used as a robustness row."""
    lr = LogisticRegression(multi_class="multinomial", max_iter=1000)
    lr.fit(scores_tr.reshape(-1, 1), y_tr)
    return lr


# --- collinearity ------------------------------------------------------------

def direction_cosines(dirs: dict[str, np.ndarray]) -> pd.DataFrame:
    axes = list(dirs)
    M = pd.DataFrame(index=axes, columns=axes, dtype=float)
    for a in axes:
        for b in axes:
            M.loc[a, b] = float(np.dot(_unit(dirs[a]), _unit(dirs[b])))
    return M


def residualize(w_target: np.ndarray, w_other: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: component of w_target orthogonal to w_other."""
    o = _unit(w_other)
    return _unit(w_target - np.dot(w_target, o) * o)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))


# --- self-test ---------------------------------------------------------------

def _selftest() -> int:
    rng = np.random.RandomState(0)
    d, n = 64, 600
    # latent axis = first coordinate; grade increases along it. Plus topic noise.
    w_true = np.zeros(d); w_true[0] = 1.0
    grades = rng.randint(0, 3, n)
    X = rng.randn(n, d) * 0.5
    X[:, 0] += grades * 2.0  # separable along axis 0
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    df = pd.DataFrame({
        "grade": grades,
        "supercluster_id": rng.randint(0, 20, n).astype(str),
        "query_key": [f"q{i}" for i in range(n)],
        "pair_key": [f"q{i}" for i in range(n)],
    })
    tr = slice(0, 400); te = slice(400, n)
    w = fit_extreme_dom(X[tr], grades[tr])
    s_tr, s_te = project(X[tr], w), project(X[te], w)
    t1, t2 = tune_two_thresholds(s_tr, grades[tr])
    f1 = macro_f1(grades[te], predict_3class(s_te, t1, t2))
    wp = fit_paired_dom(X[tr], df.iloc[tr].reset_index(drop=True))
    f1p = macro_f1(grades[te], predict_3class(project(X[te], wp),
                   *tune_two_thresholds(project(X[tr], wp), grades[tr])))
    print(f"[selftest] extreme-DoM macro-F1={f1:.3f}  paired-DoM macro-F1={f1p:.3f}  "
          f"cos(extreme,paired)={float(np.dot(_unit(w), _unit(wp))):.3f}")
    assert f1 > 0.8 and f1p > 0.8, "probe failed to recover a planted separable axis"
    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("usage: python -m benchmark.probe --selftest")
