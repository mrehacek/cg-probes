"""Tier-1 construct-validity tests for the CG-Probes benchmark.

These three checks back claims the abstract already makes and need NO clinician
labels — they run entirely on the cached embeddings + saved probe directions.

  1. TOPIC-ORTHOGONALITY (the load-bearing claim).
     A probe could secretly be a *topic* detector ("fever queries") rather than a
     clinical-safety axis. For each axis direction `w` we build BERTopic
     supercluster centroid directions in the SAME embedder space (centroid minus
     global mean, unit) and report cos(w, topic_dir). Pre-registered reading:
       |cos| < 0.40  near-orthogonal (good) · 0.40-0.60 partial · >0.60 entangled.
     Secondary: eta^2 = fraction of the 1-D projection's variance explained by
     supercluster identity (one-way ANOVA) — how topic-driven the score is.

  2. ORDINAL / CLINICAL metrics on the golden-400 (macro-F1 alone hides the
     safety-critical behaviour): quadratic-weighted kappa (QWK, the ordinal
     agreement metric the clinicians flagged as primary), per-grade F1 (esp. the
     grade-2 emergency class), threshold-free Spearman rho between the probe
     projection and the ordinal grade, and AUROC for the >=1 and >=2 binary
     collapses (deployable triage operating points).

  3. ET ORDINALITY. ET may be nominal-with-rough-order, not truly 1-D ordinal.
     We test whether mean(proj | grade) is monotone 0<1<2 on the held-out
     contrastive TEST split and compare ordinal-threshold macro-F1 against a
     nominal one-vs-rest macro-F1; a large nominal>ordinal gap means "report ET
     as 3-way nominal".

Output: benchmark/cache/tier1.json  (+ figures embedded later by make_report).
Run with the embeddings venv (sklearn/scipy):
  python -m benchmark.tier1_validity
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from benchmark import embedders as E
from benchmark import probe as P
from benchmark.eval_golden import _emb_lookup, _golden_axis, _vecs
from benchmark.run_embed import _tid
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES

RESULTS_PROBE = REPO / "benchmark" / "cache" / "results_probe.json"
DIRDIR = REPO / "benchmark" / "cache" / "directions"
OUT = REPO / "benchmark" / "cache" / "tier1.json"

MIN_CLUSTER_MEMBERS = 5  # centroids from fewer labeled texts are too noisy to trust
ORTH_BANDS = {"near_orthogonal": 0.40, "partial": 0.60}  # |cos| thresholds


def _load_dir(embedder: str, mode: str, axis: str) -> np.ndarray | None:
    p = DIRDIR / f"{embedder}__{mode}__{axis}.npy"
    return np.load(p) if p.exists() else None


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _eta_squared(scores: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of variance in `scores` explained by categorical `groups` (one-way
    ANOVA eta^2). 0 = topic-independent projection, 1 = fully topic-determined."""
    grand = scores.mean()
    ss_tot = float(((scores - grand) ** 2).sum()) + 1e-12
    ss_between = 0.0
    for g in pd.unique(groups):
        sel = scores[groups == g]
        ss_between += len(sel) * (sel.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def topic_orthogonality(embedder: str, mode: str, axis: str,
                        w: np.ndarray) -> dict:
    """cos(axis direction, supercluster centroid direction) distribution + eta^2.

    The topic centroid of a cluster is built from its **grade-0 members only**
    (when >=3 exist) so the centroid captures the *topic*, not the safety signal —
    the clusters were selected for grade contrast, so an all-grade centroid would
    leak grade into the 'topic' direction and inflate apparent entanglement. We
    lead with mean|cos| (max over hundreds of noisy per-cluster estimates is
    upward-biased) and keep max as a worst-case. `cosines` is returned for the
    report's orthogonality strip figure.
    """
    emb = _emb_lookup(embedder, mode, axis)
    if emb is None:
        return {}
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    scid = ds["supercluster_id"].astype(str).to_numpy()
    grade = ds["grade"].to_numpy()
    global_mean = X.mean(axis=0)
    wu = _unit(w)

    cosines, n_allgrade = [], 0
    for c in pd.unique(scid):
        in_c = scid == c
        g0 = in_c & (grade == 0)
        if g0.sum() >= 3:                       # pure-topic centroid
            members = X[g0]
        elif in_c.sum() >= MIN_CLUSTER_MEMBERS:  # fallback: all grades
            members = X[in_c]; n_allgrade += 1
        else:
            continue
        topic_dir = _unit(members.mean(axis=0) - global_mean)
        cosines.append(float(np.dot(wu, topic_dir)))
    proj = X @ wu
    if not cosines:
        return {"n_clusters_used": 0}
    acos = np.abs(np.asarray(cosines))
    eta2 = _eta_squared(proj, scid)
    # η² (variance of the projection explained by topic-cluster membership) is the
    # decisive "is it a topic detector?" metric: a probe can have low cosine to any
    # single centroid yet still be largely predicted by which cluster a query is in
    # (ET does exactly this). So a high η² overrides a low mean-cosine verdict.
    if eta2 > 0.50:
        verdict = "topic_confounded"
    elif acos.mean() < ORTH_BANDS["near_orthogonal"]:
        verdict = "near_orthogonal"
    elif acos.mean() < ORTH_BANDS["partial"]:
        verdict = "partial"
    else:
        verdict = "entangled"
    return {
        "n_clusters_used": int(len(cosines)),
        "n_fallback_allgrade": int(n_allgrade),
        "mean_abs_cos": float(acos.mean()),
        "p95_abs_cos": float(np.quantile(acos, 0.95)),
        "max_abs_cos": float(acos.max()),
        "eta2_projection": eta2,
        "cosines": [round(c, 4) for c in cosines],  # for the strip figure
        "verdict": verdict,
    }


def golden_ordinal_metrics(embedder: str, mode: str, axis: str, w: np.ndarray,
                           t1: float, t2: float, gold_source: str = "silver") -> dict:
    emb = _emb_lookup(embedder, mode, axis)
    if emb is None:
        return {}
    gold = _golden_axis(axis, gold_source)
    Xg, keepg = _vecs(gold["text"], emb)
    gold = gold[keepg].reset_index(drop=True)
    s = Xg @ _unit(w)
    y = gold["grade"].to_numpy()
    pred = P.predict_3class(s, t1, t2)
    real = ~gold["synthetic"].to_numpy()

    def _block(mask) -> dict:
        yy, pp, ss = y[mask], pred[mask], s[mask]
        if len(yy) < 5 or len(np.unique(yy)) < 2:
            return {}
        per_class = f1_score(yy, pp, labels=[0, 1, 2], average=None, zero_division=0)
        rho = spearmanr(ss, yy).statistic
        out = {
            "n": int(len(yy)),
            "qwk": float(cohen_kappa_score(yy, pp, labels=[0, 1, 2], weights="quadratic")),
            "macro_f1": float(f1_score(yy, pp, labels=[0, 1, 2], average="macro", zero_division=0)),
            "f1_per_grade": [round(float(x), 3) for x in per_class],
            "spearman_rho": float(rho) if rho == rho else None,
        }
        if (yy >= 1).any() and (yy < 1).any():
            out["auroc_ge1"] = float(roc_auc_score(yy >= 1, ss))
        if (yy >= 2).any() and (yy < 2).any():
            out["auroc_ge2"] = float(roc_auc_score(yy >= 2, ss))
        return out

    return {"all": _block(np.ones(len(y), bool)), "real_only": _block(real)}


def et_ordinality(embedder: str, mode: str, axis: str, w: np.ndarray,
                  t1: float, t2: float) -> dict:
    """Monotone-projection + ordinal-vs-nominal on the held-out contrastive TEST."""
    emb = _emb_lookup(embedder, mode, axis)
    if emb is None:
        return {}
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    te = (ds["split"] == "test").to_numpy()
    tr = (ds["split"] == "train").to_numpy()
    y = ds["grade"].to_numpy()
    s = X @ _unit(w)
    means = [float(s[(y == g) & te].mean()) if ((y == g) & te).any() else None
             for g in (0, 1, 2)]
    mono = all(m is not None for m in means) and means[0] < means[1] < means[2]
    rho = spearmanr(s[te], y[te]).statistic
    # ordinal (threshold) vs nominal (one-vs-rest argmax) macro-F1 on TEST
    f1_ord = P.macro_f1(y[te], P.predict_3class(s[te], t1, t2))
    dirs_ovr = P.fit_ovr_dom(X[tr], y[tr])
    f1_nom = P.macro_f1(y[te], P.predict_ovr(X[te], dirs_ovr))
    return {
        "mean_proj_by_grade": means,
        "monotone_0_1_2": bool(mono),
        "spearman_rho_test": float(rho) if rho == rho else None,
        "macro_f1_ordinal": float(f1_ord),
        "macro_f1_nominal_ovr": float(f1_nom),
        "nominal_minus_ordinal": float(f1_nom - f1_ord),
    }


def run(embedders: list[str], modes: list[str], gold_source: str = "silver") -> list[dict]:
    thresholds = {}
    if RESULTS_PROBE.exists():
        for r in json.loads(RESULTS_PROBE.read_text(encoding="utf-8")):
            thresholds[(r["embedder"], r["mode"], r["axis"])] = r["separability"]["thresholds"]

    rows = []
    for embedder in embedders:
        for mode in modes:
            if mode not in E.INSTR_MODES.get(embedder, []):
                continue
            for axis in SAFETY_AXES:
                w = _load_dir(embedder, mode, axis)
                if w is None:
                    continue
                t1, t2 = thresholds.get((embedder, mode, axis), (None, None))
                if t1 is None:
                    print(f"[skip] {embedder}/{mode}/{axis} — no thresholds", flush=True)
                    continue
                rec = {
                    "embedder": embedder, "mode": mode, "axis": axis,
                    "topic_orthogonality": topic_orthogonality(embedder, mode, axis, w),
                    "golden": golden_ordinal_metrics(embedder, mode, axis, w, t1, t2, gold_source),
                    "ordinality": et_ordinality(embedder, mode, axis, w, t1, t2),
                }
                rows.append(rec)
                to = rec["topic_orthogonality"]
                g = rec["golden"].get("real_only", {})
                print(f"[{embedder}/{mode}/{axis}] "
                      f"orth mean|cos|={to.get('mean_abs_cos', float('nan')):.2f} "
                      f"max={to.get('max_abs_cos', float('nan')):.2f} "
                      f"({to.get('verdict', '?')}) | "
                      f"QWK={g.get('qwk', float('nan')):.3f} "
                      f"rho={g.get('spearman_rho', float('nan')):.3f} "
                      f"mono={rec['ordinality'].get('monotone_0_1_2')}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="*", default=E.REGISTRY)
    ap.add_argument("--modes", nargs="*",
                    default=["none", "generic", "per_axis", "per_axis_pos"])
    ap.add_argument("--gold-source", choices=["silver", "human"], default="silver")
    a = ap.parse_args()
    rows = run(a.embedders, a.modes, a.gold_source)
    out = OUT if a.gold_source == "silver" else OUT.with_name(f"tier1__{a.gold_source}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {out} ({len(rows)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
