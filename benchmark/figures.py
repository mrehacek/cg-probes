"""Build the CG-Probes figure set from the cached result JSONs + embeddings.

Each figure is a paper-ready SVG/PDF/PNG (via _viz.emit) plus an inline PNG for
the HTML report. `build_all()` returns {fig_name: {"img": data_uri, "caption":
str}} for make_report to drop in, and writes nothing else.

Headline (paper): F1 separability, F2 probe-vs-LLM (QWK), F3 cost Pareto.
Support: F4 cross-axis cosine, F5 topic-orthogonality strip, F6 embedder x axis
QWK, F7 selectivity, F8 linear-vs-MLP. Report-only: confusion + entanglement
scatter for the best cell per axis.

Run with the embeddings venv:  python -m benchmark.figures
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from benchmark import _viz
from benchmark import probe as P
from benchmark.eval_golden import _emb_lookup, _golden_axis, _vecs
from contrastive import p2_io
from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int

CACHE = REPO / "benchmark" / "cache"
RESULTS_PROBE = CACHE / "results_probe.json"
TIER1 = CACHE / "tier1.json"
TIER2 = CACHE / "tier2.json"
CF = CACHE / "counterfactual" / "results.json"
DIRDIR = CACHE / "directions"
GOLDEN_400 = REPO / "golden" / "cache" / "golden_set_v1" / "golden_400_filled.parquet"
BASELINES = ["gpt-oss-120b", "gpt-oss-safeguard-20b"]

# Representative end-to-end latency (ms/query). Probe = measured embed throughput
# (meta wall/n) + negligible dot. safeguard-20b = MEASURED single-query latency on
# the serving GPU (Nvidia L40S 48GB, HFIE/vLLM, drained cluster, reasoning=low —
# its deployable config; see cache/baselines/safeguard_latency.json). 120b/gpt-5.4
# remain nominal (not benchmarked here).
LLM_LATENCY_MS = {"gpt-oss-safeguard-20b": 2878, "gpt-oss-120b": 2500, "gpt-5.4": 4000}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _qwk(y, p):
    return float(cohen_kappa_score(y, p, labels=[0, 1, 2], weights="quadratic"))


def _macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def _boot_qwk(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(_qwk(y[idx], p[idx]))
    if not vals:
        return (None, None)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# --- best-cell selection (by golden real-only QWK from tier1) ----------------

def best_cells(tier1: list[dict]) -> dict:
    best = {}
    for ax in SAFETY_AXES:
        cand = [r for r in tier1
                if r["axis"] == ax and r.get("golden", {}).get("real_only", {}).get("qwk") is not None]
        if not cand:
            continue
        best[ax] = max(cand, key=lambda r: r["golden"]["real_only"]["qwk"])
    return best


def _cell_proj(embedder, mode, axis, w, split_df=None):
    """Return (s_test, y_test, t1, t2) on the contrastive TEST split for a cell."""
    emb = _emb_lookup(embedder, mode, axis)
    ds = pd.read_parquet(p2_io.probe_dataset_path(axis))
    X, keep = _vecs(ds["text"], emb)
    ds = ds[keep].reset_index(drop=True)
    if len(ds) != len(X):
        ds = ds.iloc[:len(X)].reset_index(drop=True)
    te = (ds["split"] == "test").to_numpy()
    s = X @ (w / (np.linalg.norm(w) + 1e-12))
    return s[te], ds["grade"].to_numpy()[te]


def _golden_proj(embedder, mode, axis, w, t1, t2):
    emb = _emb_lookup(embedder, mode, axis)
    gold = _golden_axis(axis)
    Xg, keepg = _vecs(gold["text"], emb)
    gold = gold[keepg].reset_index(drop=True)
    s = Xg @ (w / (np.linalg.norm(w) + 1e-12))
    pred = P.predict_3class(s, t1, t2)
    return s, gold["grade"].to_numpy(), pred, (~gold["synthetic"].to_numpy())


def _baseline_grades(model, axis):
    g = pd.read_parquet(GOLDEN_400)
    g["synthetic"] = g["source"].astype(str) == "synthesized"
    g["query_key"] = g["query_key"].astype(str)
    # safeguard's deployable column = low reasoning (reasoning sweep: best+cheapest).
    variant = {"gpt-oss-safeguard-20b": "triage_low"}.get(model, "triage")
    bp = CACHE / "baselines" / f"{model}__{variant}.parquet"
    if not bp.exists():
        bp = CACHE / "baselines" / f"{model}__triage.parquet"
    pred = pd.read_parquet(bp)
    pred["query_key"] = pred["query_key"].astype(str)
    truth = g.assign(grade=g[f"{axis.lower()}_grade"].map(token_to_int))[
        ["query_key", "grade", "synthetic"]]
    pp = pred[["query_key", f"{axis.lower()}_grade"]].rename(
        columns={f"{axis.lower()}_grade": "pred"})
    m = truth.merge(pp, on="query_key").dropna(subset=["pred"])
    r = m[~m["synthetic"]]
    return r["grade"].to_numpy(), r["pred"].to_numpy().astype(int)


# --- figure builders ----------------------------------------------------------

def build_all() -> dict:
    probe = _load(RESULTS_PROBE)
    tier1 = _load(TIER1)
    tier2 = _load(TIER2)
    cf = _load(CF)
    thr = {(r["embedder"], r["mode"], r["axis"]): r["separability"]["thresholds"] for r in probe}
    best = best_cells(tier1)
    figs: dict = {}

    # ---- F1: projection separability (best cell per axis) + ordinality box ----
    f1_imgs = []
    for ax in SAFETY_AXES:
        b = best[ax]; emb, mode = b["embedder"], b["mode"]
        w = np.load(DIRDIR / f"{emb}__{mode}__{ax}.npy")
        t1, t2 = thr[(emb, mode, ax)]
        s_te, y_te = _cell_proj(emb, mode, ax, w)
        title = f"{ax} — {emb}/{mode}  (TEST macro-F1={b['ordinality']['macro_f1_ordinal']:.2f})"
        f1_imgs.append(_viz.projection_hist(s_te, y_te, t1, t2, title, f"F1_proj_{ax}"))
        _viz.grade_box(s_te, y_te, f"{ax} ordinality", f"F1b_box_{ax}")
    figs["F1_separability"] = {
        "imgs": f1_imgs,
        "caption": "F1. Held-out contrastive-TEST projection onto each axis probe "
                   "direction, by true grade (best embedder/mode by golden QWK). "
                   "Dashed lines: tuned thresholds. Visible grade ordering 0<1<2 is "
                   "the linear-recoverability claim made directly visible."}

    # ---- F2: probe vs LLM, QWK (real-only) with bootstrap CI ----
    data = {}
    for ax in SAFETY_AXES:
        b = best[ax]; emb, mode = b["embedder"], b["mode"]
        w = np.load(DIRDIR / f"{emb}__{mode}__{ax}.npy")
        t1, t2 = thr[(emb, mode, ax)]
        s, yg, pg, real = _golden_proj(emb, mode, ax, w, t1, t2)
        yq, pq = yg[real], pg[real]
        lo, hi = _boot_qwk(yq, pq)
        row = {"probe (best)": (_qwk(yq, pq), lo, hi)}
        for model in BASELINES:
            yb, pb = _baseline_grades(model, ax)
            blo, bhi = _boot_qwk(yb, pb)
            row[model] = (_qwk(yb, pb), blo, bhi)
        data[ax] = row
    figs["F2_probe_vs_llm"] = {
        "img": _viz.probe_vs_llm(data, "quadratic-weighted κ", "F2_probe_vs_llm"),
        "caption": "F2. Per-axis quadratic-weighted κ (ordinal agreement) of the best "
                   "probe vs deployable LLM baselines on the real (non-synthetic) golden "
                   "items. Bars = 95% bootstrap CI. Higher is better.",
        "data": {ax: {k: round(v[0], 3) for k, v in row.items()} for ax, row in data.items()}}

    # ---- F3: cost-performance Pareto ----
    points = []
    # probe amortized embed latency from meta wall/n (representative)
    for ax in SAFETY_AXES:
        b = best[ax]
        meta = _load(CACHE / "emb" / f"{b['embedder']}__{b['mode'] if b['mode'] in ('none','generic') else 'generic'}.meta.json")
        ms = (meta["wall_sec"] / meta["n_texts"] * 1000) if meta else 20.0
        points.append({"label": f"probe·{ax}", "cost_ms": max(ms, 1.0),
                       "score": data[ax]["probe (best)"][0], "kind": "probe"})
    for model in BASELINES:
        qs = [_qwk(*_baseline_grades(model, ax)) for ax in SAFETY_AXES]
        points.append({"label": model, "cost_ms": LLM_LATENCY_MS[model],
                       "score": float(np.mean(qs)), "kind": "llm"})
    figs["F3_pareto"] = {
        "img": _viz.pareto(points, "mean QWK (real-only)", "F3_pareto"),
        "caption": "F3. Cost–performance. x = representative ms/query (probe = measured "
                   "embedding throughput + dot product; LLM = nominal reasoning-call "
                   "latency, NOT load-benchmarked). Probes sit ~2 orders of magnitude "
                   "cheaper at comparable QWK."}

    # ---- F4: cross-axis cosine heatmap (best embedder, generic) ----
    # pick the embedder of the MU best cell, generic mode for a clean comparison
    emb0 = best["MU"]["embedder"]
    mode0 = "generic" if (DIRDIR / f"{emb0}__generic__MU.npy").exists() else best["MU"]["mode"]
    dirs = {ax: np.load(DIRDIR / f"{emb0}__{mode0}__{ax}.npy") for ax in SAFETY_AXES}
    M = np.array([[float(dirs[a] @ dirs[b] / (np.linalg.norm(dirs[a]) * np.linalg.norm(dirs[b]) + 1e-12))
                   for b in SAFETY_AXES] for a in SAFETY_AXES])
    figs["F4_cross_axis"] = {
        "img": _viz.heatmap(M, SAFETY_AXES, SAFETY_AXES,
                            f"Cross-axis direction cosine ({emb0}/{mode0})", "F4_cross_axis"),
        "caption": "F4. Cosine between the three axis directions (one embedder). Off-"
                   "diagonal magnitude = geometric redundancy; PU–ET is the entangled pair."}

    # ---- F5: topic-orthogonality strip (best cell per axis) ----
    by_axis = {ax: best[ax]["topic_orthogonality"]["cosines"] for ax in SAFETY_AXES}
    figs["F5_orthogonality"] = {
        "img": _viz.orthogonality_strip(by_axis, "F5_orthogonality"),
        "caption": "F5. |cos(axis direction, supercluster grade-0 centroid)| across "
                   "clusters (best cell/axis). Black bar = mean. Most mass left of 0.4 "
                   "(near-orthogonal): probes are not merely topic detectors."}

    # ---- F6: embedder x axis QWK matrix (generic/none cells only, real-only) ----
    rows_emb = ["qwen8b", "harrier27b", "openai3large", "gemini"]
    grid = np.full((len(rows_emb), 3), np.nan)
    for i, e in enumerate(rows_emb):
        for j, ax in enumerate(SAFETY_AXES):
            cands = [r for r in tier1 if r["embedder"] == e and r["axis"] == ax
                     and r["mode"] in ("generic", "none")
                     and r.get("golden", {}).get("real_only", {}).get("qwk") is not None]
            if cands:
                grid[i, j] = max(c["golden"]["real_only"]["qwk"] for c in cands)
    figs["F6_embedder_axis"] = {
        "img": _viz.heatmap(grid, rows_emb, SAFETY_AXES, "Golden QWK by embedder × axis",
                            "F6_embedder_axis", cmap="viridis", vmin=0, vmax=0.8, fmt="{:.2f}"),
        "caption": "F6. Best golden real-only QWK per embedder × axis (generic/none "
                   "modes). Stability across embedders supports an embedder-agnostic claim."}

    # ---- F7: selectivity bars (best cell per axis) ----
    real_f1, ctrl_f1 = [], []
    for ax in SAFETY_AXES:
        b = best[ax]
        t2row = next(r for r in tier2 if r["embedder"] == b["embedder"]
                     and r["mode"] == b["mode"] and r["axis"] == ax)
        real_f1.append(t2row["selectivity"]["f1_real"])
        ctrl_f1.append(t2row["selectivity"]["f1_control_mean"])
    figs["F7_selectivity"] = {
        "img": _viz.grouped_bar(SAFETY_AXES, {"real probe": real_f1, "shuffled control": ctrl_f1},
                                "macro-F1 (contrastive TEST)", "Selectivity (Hewitt & Liang)",
                                "F7_selectivity", colors=[_viz.CB["green"], _viz.CB["grey"]],
                                hline=0.333, hline_label="chance (0.33)"),
        "caption": "F7. Real probe vs a probe fit on grade-shuffled labels (control task, "
                   "same marginals). The large gap = the probe reads the concept, not an "
                   "embedding artifact."}

    # ---- F8: linear-vs-MLP bars (best cell per axis) ----
    dom, lin, mlp = [], [], []
    for ax in SAFETY_AXES:
        b = best[ax]
        t2row = next(r for r in tier2 if r["embedder"] == b["embedder"]
                     and r["mode"] == b["mode"] and r["axis"] == ax)
        lvm = t2row["linear_vs_mlp"]
        dom.append(lvm["f1_dom"]); lin.append(lvm["f1_linear_full"]); mlp.append(lvm["f1_mlp"])
    figs["F8_linear_mlp"] = {
        "img": _viz.grouped_bar(SAFETY_AXES, {"1-D DoM": dom, "linear (full)": lin, "MLP": mlp},
                                "macro-F1 (contrastive TEST)", "Linear sufficiency",
                                "F8_linear_mlp"),
        "caption": "F8. MLP ≈ full-linear everywhere (non-linearity buys ~nothing → "
                   "'linearly recoverable' is earned). The 1-D DoM is near-optimal for "
                   "MU/PU but lossy for ET (ET needs >1 linear direction → topic-based)."}

    # ---- report-only: confusion + entanglement scatter ----
    extras = []
    for ax in SAFETY_AXES:
        b = best[ax]; emb, mode = b["embedder"], b["mode"]
        w = np.load(DIRDIR / f"{emb}__{mode}__{ax}.npy")
        t1, t2 = thr[(emb, mode, ax)]
        s, yg, pg, real = _golden_proj(emb, mode, ax, w, t1, t2)
        extras.append(_viz.confusion(yg[real], pg[real], f"{ax} golden (real) confusion", f"R_conf_{ax}"))
    figs["R_confusion"] = {"imgs": extras,
                           "caption": "Confusion of the best probe on real golden items per axis."}

    # PU vs ET entanglement scatter on a shared cell (generic of PU's best embedder)
    embE = best["PU"]["embedder"]
    modeE = "generic" if (DIRDIR / f"{embE}__generic__PU.npy").exists() else best["PU"]["mode"]
    emb = _emb_lookup(embE, modeE, "PU")
    if emb is not None:
        gold = _golden_axis("PU")  # any; need both PU & ET grades -> load golden raw
        graw = pd.read_parquet(GOLDEN_400)
        texts = graw["query_text"].astype(str).tolist()
        Xg, keepg = _vecs(texts, emb)
        graw = graw[keepg].reset_index(drop=True)
        wpu = np.load(DIRDIR / f"{embE}__{modeE}__PU.npy")
        wet = np.load(DIRDIR / f"{embE}__{modeE}__ET.npy")
        sx = Xg @ (wpu / np.linalg.norm(wpu)); sy = Xg @ (wet / np.linalg.norm(wet))
        ypu = graw["pu_grade"].map(token_to_int).to_numpy()
        figs["R_entangle"] = {
            "img": _viz.entanglement_scatter(sx, sy, ypu, "PU direction", "ET direction",
                                             f"PU vs ET projection ({embE}/{modeE}), colored by PU grade",
                                             "R_entangle"),
            "caption": "Golden queries projected on PU and ET directions, colored by PU "
                       "grade. PU separation along x with ET spread along y illustrates the "
                       "partial PU–ET entanglement geometrically."}
    return figs


def build_human() -> dict:
    """Human-gold figures: F2h (systems vs human gold + ceiling), F3h (Pareto on
    human QWK), F9 (human ceiling bars + ET reconcile). Reads results_human.json."""
    rh = _load(CACHE / "results_human.json")
    iaa = _load(CACHE / "human_iaa.json")
    if rh is None:
        return {}
    figs: dict = {}
    order = ["probe", "gpt-oss-safeguard-20b", "gpt-oss-120b", "gpt-5.4"]

    # F2h — systems vs human gold (real-only QWK) + human ceiling (ET reconciled)
    data, ceiling, ceiling_raw = {}, {}, {}
    for ax in SAFETY_AXES:
        sysd = rh["axes"].get(ax, {})
        row = {}
        for name in order:
            blk = (sysd.get(name) or {}).get("real_only") or (sysd.get(name) or {}).get("all")
            if blk:
                row[name] = (blk["qwk"], blk.get("qwk_lo"), blk.get("qwk_hi"))
        if row:
            data[ax] = row
            c = rh["ceiling"][ax]
            ceiling[ax] = c.get("human_human_qwk_adj", c.get("human_human_qwk"))
            if c.get("reconciled"):
                ceiling_raw[ax] = c.get("human_human_qwk_raw")
    if data:
        figs["F2h_vs_human"] = {
            "img": _viz.probe_vs_llm(data, "quadratic-weighted κ",  # detail in caption
                                     "F2h_vs_human", ceiling=ceiling, ceiling_raw=ceiling_raw,
                                     title=None),  # title lives in the LaTeX caption
            "caption": "F2h. Per-axis QWK of each system against the 200-item clinician gold "
                       "(real-only), 95% bootstrap CI. Dashed line = inter-oncologist QWK ceiling. "
                       "The ET ceiling is shown AFTER reconciling a documented codebook ambiguity "
                       "(somatic urgency miscoded as high emotional load), so the ceiling and the "
                       "system scores share one codebook basis; the faint dotted 'raw' tick is the "
                       "pre-correction inter-rater κ (0.32). See report §3.2. A system at/above the "
                       "dashed line matches reconciled human–human agreement."}

    # F3h — cost–performance on human QWK (mean real-only across axes). LLMs share a
    # square marker, distinct colors; probes are green circles.
    LLM_COLOR = {"gpt-oss-safeguard-20b": _viz.CB["blue"], "gpt-oss-120b": _viz.CB["orange"],
                 "gpt-5.4": _viz.CB["purple"]}
    points = []
    for ax in SAFETY_AXES:
        sd = rh["axes"].get(ax, {}).get("probe", {})
        blk = sd.get("real_only") or sd.get("all")
        if blk:
            points.append({"label": f"probe·{ax}", "cost_ms": 20.0, "score": blk["qwk"],
                           "kind": "probe"})
    for m, lat in LLM_LATENCY_MS.items():
        qs = [((rh["axes"].get(ax, {}).get(m, {}) or {}).get("real_only")
               or (rh["axes"].get(ax, {}).get(m, {}) or {}).get("all") or {}).get("qwk")
              for ax in SAFETY_AXES]
        qs = [q for q in qs if q is not None]
        if qs:
            points.append({"label": m, "cost_ms": lat, "score": float(np.mean(qs)),
                           "kind": "llm", "color": LLM_COLOR.get(m)})
    if points:
        figs["F3h_pareto"] = {
            "img": _viz.pareto(points, "mean QWK (clinician gold, real-only)", "F3h_pareto"),
            "caption": "F3h. Cost vs clinician-gold QWK. Probes (green circles, ~20 ms/query) sit "
                       "~2 orders of magnitude cheaper than the reasoning LLMs (squares) at "
                       "comparable agreement; safeguard latency is measured on the L40S, the "
                       "others are nominal."}

    # F10 — probe-head capacity on clinician gold (does a richer LINEAR head help?)
    ph = rh.get("probe_heads", {})
    if ph:
        dom = [ph[ax]["dom"]["human_qwk"] for ax in SAFETY_AXES]
        lin = [ph[ax]["linear"]["human_qwk"] for ax in SAFETY_AXES]
        mlp = [ph[ax]["mlp"]["human_qwk"] for ax in SAFETY_AXES]
        ceil = [rh["ceiling"][ax].get("human_human_qwk_adj") for ax in SAFETY_AXES]
        figs["F10_probe_heads"] = {
            "img": _viz.grouped_bar(
                SAFETY_AXES,
                {"1-D DoM (single direction)": dom, "linear (full subspace)": lin, "MLP": mlp},
                "QWK (clinician gold, real-only)", "Probe head: does ET need more than one direction?",
                "F10_probe_heads", colors=[_viz.CB["green"], _viz.CB["sky"], _viz.CB["grey"]],
                markers=ceil, marker_label="human ceiling"),
            "caption": "F10. Probe-head capacity on clinician gold. The 1-D difference-in-means "
                       "direction is near-optimal for MU/PU (a single interpretable axis), but "
                       "underfits ET; a class-balanced FULL-LINEAR head lifts ET QWK 0.49→0.58 — and "
                       "an MLP does no better, so ET is a linear *subspace*, not a single direction. "
                       "Head chosen on the cluster-disjoint contrastive TEST (no human-gold tuning)."}

    # F9 — human ceiling per axis (+ ET reconcile annotation)
    hh = [rh["ceiling"][ax].get("human_human_qwk") or 0 for ax in SAFETY_AXES]
    al = [rh["ceiling"][ax].get("alpha_human") or 0 for ax in SAFETY_AXES]
    figs["F9_ceiling"] = {
        "img": _viz.grouped_bar(SAFETY_AXES, {"human–human QWK": hh, "Krippendorff α (interval)": al},
                                "agreement", "Inter-oncologist agreement (human ceiling)",
                                "F9_ceiling", colors=[_viz.CB["green"], _viz.CB["blue"]]),
        "caption": "F9. Inter-oncologist agreement per axis (the ceiling any system is measured "
                   "against). ET is low (α=0.19) due to a single codebook ambiguity — somatic "
                   "urgency coded as high emotional load; reconciling it lifts ET QWK to ~0.82 "
                   "(see report §IAA)."}
    return figs


def main() -> int:
    figs = build_all()
    print(f"[figures] built {len(figs)} figure groups -> {_viz.FIG_DIR}")
    for k, v in figs.items():
        n = len(v.get("imgs", [])) or 1
        print(f"  {k}: {n} image(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
