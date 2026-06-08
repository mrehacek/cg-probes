"""Final human-gold benchmark: every system vs the clinician gold, positioned
against the human-agreement ceiling, with a silver-vs-human sensitivity table.

Systems scored on the 200-item clinician gold (package "Golden v1 — Core"):
  * probe (best cell/axis, by human real-only QWK)
  * gpt-oss-120b, gpt-oss-safeguard-20b   (cached triage predictions, re-scored)
  * gpt-5.4                                (frontier reference annotator)
Ceiling = human–human QWK + Krippendorff α (from annotations_pull -> human_iaa.json).
Metrics per axis (all + real-only): QWK (lead), macro-F1, per-grade F1, AUROC≥2,
each with a 95% bootstrap CI on QWK. Also a silver→human delta for the probe.

Reads cached artifacts only (no model calls); the safeguard column refreshes
automatically when its prediction parquet is re-run. Output: results_human.json.
  python -m benchmark.human_benchmark
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from benchmark import _gold
from benchmark import probe as P
from benchmark.eval_golden import _emb_lookup, _vecs
from benchmark.figures import best_cells
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int

CACHE = REPO / "benchmark" / "cache"
TIER1_HUMAN = CACHE / "tier1__human.json"
TIER1_SILVER = CACHE / "tier1.json"
PROBE_HUMAN = CACHE / "results_probe__human.json"
DIRDIR = CACHE / "directions"
BASELINE_DIR = CACHE / "baselines"
GPT54_REF = REPO / "golden" / "cache" / "golden_set_v1" / "golden_gpt5.4_reference.parquet"
IAA = CACHE / "human_iaa.json"
OUT = CACHE / "results_human.json"

BASELINES = ["gpt-oss-120b", "gpt-oss-safeguard-20b"]

# Which cached triage parquet each baseline is scored from. The 2026-06-07
# reasoning sweep (see safeguard_latency.json / report) found gpt-oss-safeguard-20b
# is best AND cheapest at reasoning=low — medium/high *lower* its MU accuracy and
# emergency recall — so its deployable column uses the low-reasoning predictions.
# gpt-oss-120b stays on its default-reasoning run.
PRED_VARIANT = {"gpt-oss-safeguard-20b": "triage_low", "gpt-oss-120b": "triage"}


def _qwk(y, p):
    return float(cohen_kappa_score(y, p, labels=[0, 1, 2], weights="quadratic"))


def _macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def _boot_qwk(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [_qwk(y[i], p[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(n))
            if len(np.unique(y[i])) > 1]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (None, None)


def _scores(y, p, s=None) -> dict:
    """Full metric block for true y, pred p, and (optional) continuous score s."""
    y, p = np.asarray(y), np.asarray(p)
    lo, hi = _boot_qwk(y, p)
    out = {"n": int(len(y)), "qwk": _qwk(y, p), "qwk_lo": lo, "qwk_hi": hi,
           "macro_f1": _macro(y, p),
           "f1_per_grade": [round(float(x), 3) for x in
                            f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)]}
    if s is not None and (y >= 2).any() and (y < 2).any():
        out["auroc_ge2"] = float(roc_auc_score(y >= 2, np.asarray(s)))
    return out


def _human_axis(axis: str) -> pd.DataFrame:
    return _gold.golden_axis(axis, "human")  # text, grade, synthetic, query_key


def probe_system(axis: str, cell: dict, thr: dict) -> dict:
    emb = _emb_lookup(cell["embedder"], cell["mode"], axis)
    w = np.load(DIRDIR / f"{cell['embedder']}__{cell['mode']}__{axis}.npy")
    w = w / (np.linalg.norm(w) + 1e-12)
    t1, t2 = thr[(cell["embedder"], cell["mode"], axis)]
    ga = _human_axis(axis)
    Xg, keep = _vecs(ga["text"], emb)
    ga = ga[keep].reset_index(drop=True)
    s = Xg @ w
    pred = P.predict_3class(s, t1, t2)
    y = ga["grade"].to_numpy(); real = ~ga["synthetic"].to_numpy()
    out = {"cell": f"{cell['embedder']}/{cell['mode']}", "all": _scores(y, pred, s)}
    if real.any():
        out["real_only"] = _scores(y[real], pred[real], s[real])
    return out


def probe_linear_system(axis: str, cell: dict, gold_source: str = "human") -> dict:
    """Headline probe = class-balanced FULL-LINEAR head (best per probe_extra.py:
    ordinal/whiten/concat all underperform). Fit on contrastive train+dev, eval on
    `gold_source`. AUROC≥2 from predict_proba[:,2]."""
    from sklearn.linear_model import LogisticRegression
    emb = _emb_lookup(cell["embedder"], cell["mode"], axis)
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    trdev = ds["split"].isin(["train", "dev"]).to_numpy()
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[trdev],
                                                                         ds["grade"].to_numpy()[trdev])
    i2 = int(np.where(clf.classes_ == 2)[0][0]) if 2 in clf.classes_ else None
    ga = _gold.golden_axis(axis, gold_source)
    Xg, keepg = _vecs(ga["text"], emb)
    ga = ga[keepg].reset_index(drop=True)
    pred = clf.predict(Xg)
    s = clf.predict_proba(Xg)[:, i2] if i2 is not None else None
    y = ga["grade"].to_numpy(); real = ~ga["synthetic"].to_numpy()
    out = {"cell": f"{cell['embedder']}/{cell['mode']}", "head": "linear",
           "all": _scores(y, pred, s)}
    if real.any():
        out["real_only"] = _scores(y[real], pred[real], None if s is None else s[real])
    return out


def _pred_map(model: str) -> pd.DataFrame:
    variant = PRED_VARIANT.get(model, "triage")
    path = BASELINE_DIR / f"{model}__{variant}.parquet"
    if not path.exists():  # fall back to the default triage run
        path = BASELINE_DIR / f"{model}__triage.parquet"
    p = pd.read_parquet(path)
    p["query_key"] = p["query_key"].astype(str)
    return p


def llm_system(axis: str, pred_df: pd.DataFrame, grade_col: str) -> dict:
    ga = _human_axis(axis)
    truth = ga[["query_key", "grade", "synthetic"]].copy()
    truth["query_key"] = truth["query_key"].astype(str)
    pp = pred_df[["query_key", grade_col]].rename(columns={grade_col: "pred"})
    m = truth.merge(pp, on="query_key").dropna(subset=["pred"])
    if m.empty:
        return {}
    m["pred"] = m["pred"].astype(int)
    y, p = m["grade"].to_numpy(), m["pred"].to_numpy()
    real = ~m["synthetic"].to_numpy()
    out = {"all": _scores(y, p)}
    if real.any():
        out["real_only"] = _scores(y[real], p[real])
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not (TIER1_HUMAN.exists() and PROBE_HUMAN.exists()):
        sys.exit("run eval_golden + tier1_validity with --gold-source human first")
    tier1_h = json.loads(TIER1_HUMAN.read_text(encoding="utf-8"))
    probe_rows = json.loads(PROBE_HUMAN.read_text(encoding="utf-8"))
    thr = {(r["embedder"], r["mode"], r["axis"]): r["separability"]["thresholds"] for r in probe_rows}
    best = best_cells(tier1_h)
    iaa = json.loads(IAA.read_text(encoding="utf-8")) if IAA.exists() else {}

    # baseline + gpt-5.4 prediction frames (query_key -> per-axis int grade)
    preds = {m: _pred_map(m) for m in BASELINES if (BASELINE_DIR / f"{m}__triage.parquet").exists()}
    ref = pd.read_parquet(GPT54_REF)
    ref["query_key"] = ref["query_key"].astype(str)
    for ax in SAFETY_AXES:
        ref[f"{ax.lower()}_grade_int"] = ref[f"{ax.lower()}_grade"].map(token_to_int)

    results = {"axes": {}, "ceiling": {}, "silver_delta": {}}
    silver_t1 = {(" ".join([r["embedder"], r["mode"]]), r["axis"]):
                 r["golden"]["real_only"].get("qwk")
                 for r in json.loads(TIER1_SILVER.read_text(encoding="utf-8"))} if TIER1_SILVER.exists() else {}

    from benchmark.probe_methods import evaluate_cell as _heads
    results["probe_heads"] = {}
    for ax in SAFETY_AXES:
        cell = best.get(ax)
        sysd = {}
        if cell:
            # Headline probe = 1-D difference-in-means direction (the method; interpretable,
            # and tied with any heavier head on MU/PU). The full-linear head is retained for
            # ET, where the single direction underfits (capacity table §0.4/F10).
            sysd["probe"] = probe_system(ax, cell, thr)
            sysd["probe_linear"] = probe_linear_system(ax, cell, "human")  # full-linear (ET improver)
            # probe-head capacity: 1-D DoM (interpretable primary) vs a class-balanced
            # full-LINEAR head vs an MLP, all on human gold. ET needs the richer linear
            # subspace; MU/PU are already near-optimal as a single direction.
            heads = _heads(cell["embedder"], cell["mode"], ax)
            results["probe_heads"][ax] = {
                "cell": f"{cell['embedder']}/{cell['mode']}",
                "dom": heads["dom_paired"], "linear": heads["linear_bal"], "mlp": heads["mlp"]}
        for m, pdf in preds.items():
            sysd[m] = llm_system(ax, pdf, f"{ax.lower()}_grade")
        sysd["gpt-5.4"] = llm_system(ax, ref, f"{ax.lower()}_grade_int")
        results["axes"][ax] = sysd

        # ceiling from IAA. `qwk_adj` = codebook-reconciled inter-rater QWK — for ET
        # this applies the documented somatic-urgency->ET1 fix (the same codebook rule
        # the adjudicated gold uses), so the ceiling and the system scores share one
        # basis; for MU/PU it equals the raw value. The raw QWK is always retained.
        hh = (iaa.get("human_human_qwk", {}) or {}).get(ax, {})
        al = (iaa.get("axes", {}) or {}).get(ax, {})
        raw_q = hh.get("qwk")
        adj_q = raw_q
        if ax == "ET" and (iaa.get("et_error_analysis") or {}).get("qwk_after_somatic_reconcile") is not None:
            adj_q = iaa["et_error_analysis"]["qwk_after_somatic_reconcile"]
        results["ceiling"][ax] = {
            "human_human_qwk": raw_q, "human_human_qwk_raw": raw_q,
            "human_human_qwk_adj": adj_q, "reconciled": (ax == "ET" and adj_q != raw_q),
            "raw_agreement": hh.get("raw_agreement"),
            "alpha_human": al.get("alpha_human"), "alpha_with_llm": al.get("alpha_with_llm")}

        # silver→human delta for the HEADLINE (1-D DoM) probe, real-only QWK — both
        # sides on the same DoM direction (tier1 silver uses the same 1-D probe), so the
        # comparison is apples-to-apples.
        if cell and "probe" in sysd and "real_only" in sysd["probe"]:
            human_q = sysd["probe"]["real_only"]["qwk"]
            silver_q = silver_t1.get((f"{cell['embedder']} {cell['mode']}", ax))
            results["silver_delta"][ax] = {
                "cell": sysd["probe"]["cell"], "silver_qwk": silver_q,
                "human_qwk": human_q,
                "delta": (round(human_q - silver_q, 3) if silver_q is not None else None)}

    if iaa.get("et_error_analysis"):
        results["et_error_analysis"] = iaa["et_error_analysis"]

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # console summary
    print("=== Human-gold benchmark (real-only QWK; ceiling = human–human QWK) ===", flush=True)
    for ax in SAFETY_AXES:
        c = results["ceiling"][ax]
        cq = c["human_human_qwk"]; cqs = f"{cq:.3f}" if cq is not None else "—"
        ca = f"{c['alpha_human']:.2f}" if c["alpha_human"] is not None else "—"
        print(f"\n[{ax}]  human ceiling QWK={cqs}  alpha(human)={ca}", flush=True)
        for name, d in results["axes"][ax].items():
            blk = d.get("real_only") or d.get("all") or {}
            if blk:
                ci = f"[{blk.get('qwk_lo'):.2f},{blk.get('qwk_hi'):.2f}]" if blk.get("qwk_lo") is not None else ""
                print(f"   {name:24s} QWK={blk['qwk']:.3f} {ci}  macroF1={blk['macro_f1']:.3f}  "
                      f"AUROC>=2={blk.get('auroc_ge2', float('nan')):.3f}  n={blk['n']}", flush=True)
        sd = results["silver_delta"].get(ax)
        if sd and sd.get("silver_qwk") is not None:
            print(f"   silver→human probe QWK: {sd['silver_qwk']:.3f} → {sd['human_qwk']:.3f} "
                  f"(Δ {sd['delta']:+.3f})", flush=True)
    print(f"\n[write] {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
