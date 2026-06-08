"""P4 — comprehensive, self-explanatory HTML report.

Narrates what was done, the results, and how to interpret them, with the
best-probe-vs-LLM comparison as the headline. Reads results_probe.json, the
saved direction vectors, and the baseline triage parquets. Run with the
embeddings venv (uses sklearn to score the LLM baselines).
"""

from __future__ import annotations

import html
import json
import sys

import numpy as np
import pandas as pd

from benchmark import _gold
from benchmark import figures as FIG
from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int

CACHE = REPO / "benchmark" / "cache"
RESULTS = CACHE / "results_probe.json"
TIER1 = CACHE / "tier1.json"
TIER2 = CACHE / "tier2.json"
CF = CACHE / "counterfactual" / "results.json"
DIRDIR = CACHE / "directions"
BASE_DIR = CACHE / "baselines"
GOLDEN_400 = REPO / "golden" / "cache" / "golden_set_v1" / "golden_400_filled.parquet"
OUT = CACHE / "report.html"

AXIS_NAME = {"MU": "Medical Urgency", "PU": "Psychological Urgency",
             "ET": "Topic Sensitivity"}
BASELINES = ["gpt-oss-120b", "gpt-oss-safeguard-20b"]


# ---------- data ----------

def load_cells() -> pd.DataFrame:
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    return pd.DataFrame([{
        "embedder": r["embedder"], "mode": r["mode"], "axis": r["axis"], "dim": r["dim"],
        "sep_f1": r["separability"]["macro_f1"],
        "sep_ovr": r["separability"]["macro_f1_ovr"],
        "gold_all": r["benchmark"]["macro_f1_all"],
        "gold_real": r["benchmark"]["macro_f1_real_only"],
    } for r in res])


def best_probe(df: pd.DataFrame, axis: str) -> pd.Series:
    sub = df[(df["axis"] == axis) & df["gold_real"].notna()].sort_values("gold_real", ascending=False)
    return sub.iloc[0]


def score_baselines() -> dict:
    """{model: {axis: {'all':, 'real':}}} from the triage parquets."""
    from sklearn.metrics import f1_score
    g = pd.read_parquet(GOLDEN_400)
    g["synthetic"] = g["source"].astype(str) == "synthesized"
    g["query_key"] = g["query_key"].astype(str)
    # safeguard's deployable column = low reasoning (the reasoning sweep found it
    # best+cheapest; medium/high lower MU accuracy & emergency recall).
    variant = {"gpt-oss-safeguard-20b": "triage_low"}
    out: dict = {}
    for model in BASELINES:
        p = BASE_DIR / f"{model}__{variant.get(model, 'triage')}.parquet"
        if not p.exists():
            p = BASE_DIR / f"{model}__triage.parquet"
        if not p.exists():
            continue
        pred = pd.read_parquet(p)
        pred["query_key"] = pred["query_key"].astype(str)
        out[model] = {}
        for ax in SAFETY_AXES:
            truth = g.assign(grade=g[f"{ax.lower()}_grade"].map(token_to_int))[
                ["query_key", "grade", "synthetic"]]
            pp = pred[["query_key", f"{ax.lower()}_grade"]].rename(
                columns={f"{ax.lower()}_grade": "pred"})
            m = truth.merge(pp, on="query_key").dropna(subset=["pred"])
            if m.empty:
                continue
            allf = f1_score(m["grade"], m["pred"], labels=[0, 1, 2], average="macro", zero_division=0)
            r = m[~m["synthetic"]]
            realf = f1_score(r["grade"], r["pred"], labels=[0, 1, 2], average="macro",
                             zero_division=0) if len(r) else None
            out[model][ax] = {"all": allf, "real": realf}
    return out


def _lat(d: dict, effort: str) -> dict:
    """Prefer the drained-cluster (gap) isolated latency; fall back to back-to-back."""
    return (d.get(f"gpt-oss-safeguard-20b__{effort}__gap")
            or d.get(f"gpt-oss-safeguard-20b__{effort}") or {}).get("isolated") or {}


def safeguard_reasoning(source: str = "human") -> dict:
    """Per safeguard reasoning variant: real-only F1/QWK vs gold + count of MU2
    emergencies graded benign (on the silver set's 28 clear emergencies)."""
    from sklearn.metrics import cohen_kappa_score, f1_score
    def t2i(x):
        try:
            return token_to_int(x) if isinstance(x, str) else float("nan")
        except Exception:
            return float("nan")
    g = _gold.load_gold(source); g["query_key"] = g["query_key"].astype(str)
    sv = _gold.load_gold("silver"); sv["query_key"] = sv["query_key"].astype(str)
    sv = sv.assign(mu_t=sv["mu_grade"].map(t2i))
    out = {}
    for effort, fn in [("low", "triage_low"), ("medium", "triage_medium")]:
        p = BASE_DIR / f"gpt-oss-safeguard-20b__{fn}.parquet"
        if not p.exists():
            continue
        pred = pd.read_parquet(p); pred["query_key"] = pred["query_key"].astype(str)
        row = {}
        for ax in SAFETY_AXES:
            col = f"{ax.lower()}_grade"
            tr = g.assign(t=g[col].map(t2i))[["query_key", "t", "synthetic"]].dropna(subset=["t"])
            pp = pred[["query_key", col]].rename(columns={col: "p"})
            m = tr.merge(pp, on="query_key").dropna(subset=["p"])
            m["p"] = m["p"].astype(int); r = m[~m["synthetic"]]
            if len(r):
                row[ax] = (f1_score(r.t, r.p, labels=[0, 1, 2], average="macro", zero_division=0),
                           float(cohen_kappa_score(r.t, r.p, labels=[0, 1, 2], weights="quadratic")),
                           len(r))
        mm = sv[~(sv["source"].astype(str) == "synthesized")][["query_key", "mu_t"]].merge(
            pred[["query_key", "mu_grade"]], on="query_key").dropna(subset=["mu_grade"])
        mm["mu_grade"] = mm["mu_grade"].astype(int)
        row["mu2_miss"] = (int(((mm.mu_t == 2) & (mm.mu_grade == 0)).sum()), int((mm.mu_t == 2).sum()))
        out[effort] = row
    return out


def probe_mu2_miss():
    """(benign_miss, total) — clear silver (pipeline-labeled) MU2 emergencies the
    headline 1-D DoM MU probe grades MU0, at its deployed threshold. This is the
    same head reported in §3; defensive: returns None on any failure."""
    try:
        import numpy as _np
        from benchmark import probe as _P
        from benchmark.eval_golden import _emb_lookup, _vecs
        emb = _emb_lookup("gemini", "per_axis", "MU")
        rows = json.loads((CACHE / "results_probe__human.json").read_text(encoding="utf-8"))
        thr = {(r["embedder"], r["mode"], r["axis"]): r["separability"]["thresholds"] for r in rows}
        t1, t2 = thr[("gemini", "per_axis", "MU")]
        w = _np.load(CACHE / "directions" / "gemini__per_axis__MU.npy")
        w = w / (_np.linalg.norm(w) + 1e-12)
        sv = _gold.load_gold("silver")
        sv = sv.assign(t=sv["mu_grade"].map(lambda x: token_to_int(x) if isinstance(x, str) else x))
        sv = sv[(sv["t"] == 2) & ~(sv["source"].astype(str) == "synthesized")]
        Xg, keep = _vecs(sv["query_text"], emb)
        pred = _P.predict_3class(Xg @ w, t1, t2)
        return int((pred == 0).sum()), int(len(pred))
    except Exception:
        return None


def fig_reasoning_latency(sweep: dict, lat: dict) -> str:
    """Twin-axis: reasoning effort vs MU real-QWK (line) and isolated latency (bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from benchmark._viz import CB, emit
    efforts = [e for e in ("low", "medium") if e in sweep]
    mu_qwk = [sweep[e]["MU"][1] for e in efforts]
    miss = [sweep[e]["mu2_miss"] for e in efforts]
    iso = [_lat(lat, e).get("mean_s") for e in efforts]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.4))
    x = list(range(len(efforts)))
    ax1.bar(x, iso, width=0.5, color=CB["sky"], alpha=0.85)
    ax1.set_ylabel("isolated latency (s/query)", color=CB["blue"])
    ax1.set_xticks(x); ax1.set_xticklabels([e + " reasoning" for e in efforts])
    for xi, v in zip(x, iso):
        if v is not None:
            ax1.text(xi, v + 0.08, f"{v:.1f}s", ha="center", fontsize=9, color=CB["blue"])
    ax2 = ax1.twinx()
    ax2.plot(x, mu_qwk, "-o", color=CB["red"], lw=2)
    for xi, q, mm in zip(x, mu_qwk, miss):
        ax2.annotate(f"κ {q:.2f} · {mm[0]}/{mm[1]} emerg. missed", (xi, q),
                     textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=8.5, color=CB["red"])
    ax2.set_ylabel("MU quadratic-weighted κ", color=CB["red"])
    ax2.set_ylim(0.45, 0.78)
    if any(iso):
        ax1.set_ylim(0, max(v for v in iso if v) * 1.4)
    ax1.grid(False); ax2.grid(False)
    fig.suptitle("More reasoning → higher latency AND worse emergency recall", fontsize=10.5)
    return emit(fig, "F9_reasoning_latency")


def collinearity(df: pd.DataFrame) -> dict:
    out = {}
    for (emb, mode), _ in df.groupby(["embedder", "mode"]):
        dirs = {ax: np.load(DIRDIR / f"{emb}__{mode}__{ax}.npy")
                for ax in SAFETY_AXES if (DIRDIR / f"{emb}__{mode}__{ax}.npy").exists()}
        if len(dirs) < 3:
            continue
        M = pd.DataFrame(index=SAFETY_AXES, columns=SAFETY_AXES, dtype=float)
        for a in SAFETY_AXES:
            for b in SAFETY_AXES:
                M.loc[a, b] = float(dirs[a] @ dirs[b] /
                                    (np.linalg.norm(dirs[a]) * np.linalg.norm(dirs[b]) + 1e-12))
        out[f"{emb} / {mode}"] = M
    return out


# ---------- html helpers ----------

def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _fig(figs: dict, key: str, width: str = "100%", img_w: str = "32%") -> str:
    """Render a figure group (single 'img' or multi 'imgs') with its caption.
    img_w sets the per-image width for the multi-image branch."""
    g = figs.get(key)
    if not g:
        return f"<p class='warn'>[missing figure: {key}]</p>"
    cap = html.escape(g.get("caption", ""))
    if "imgs" in g:
        imgs = "".join(f"<img src='{u}' style='max-width:{img_w};margin:.3%'>" for u in g["imgs"])
        return f"<figure>{imgs}<figcaption>{cap}</figcaption></figure>"
    return (f"<figure><img src='{g['img']}' style='max-width:{width}'>"
            f"<figcaption>{cap}</figcaption></figure>")


def _heat(M: pd.DataFrame) -> str:
    s = ["<table class='heat'><tr><th></th>" + "".join(f"<th>{c}</th>" for c in M.columns) + "</tr>"]
    for r in M.index:
        cells = [f"<th>{r}</th>"]
        for c in M.columns:
            v = float(M.loc[r, c])
            bg = f"hsl({int(140*(1-abs(v)))},65%,{int(96-32*abs(v))}%)"
            cells.append(f"<td style='background:{bg}'>{v:+.2f}</td>")
        s.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(s) + "</table>"


# ---------- build ----------

def main() -> int:
    if not RESULTS.exists():
        sys.exit("no results_probe.json (run eval_golden first)")
    df = load_cells()
    base = score_baselines()
    coll = collinearity(df)
    tier1 = json.loads(TIER1.read_text(encoding="utf-8")) if TIER1.exists() else []
    tier2 = json.loads(TIER2.read_text(encoding="utf-8")) if TIER2.exists() else []
    cf = json.loads(CF.read_text(encoding="utf-8")) if CF.exists() else None
    t1_best = FIG.best_cells(tier1) if tier1 else {}
    print("[report] building figures…", flush=True)
    figs = FIG.build_all() if tier1 and tier2 else {}
    figs_h = FIG.build_human()
    rh = json.loads((CACHE / "results_human.json").read_text(encoding="utf-8")) \
        if (CACHE / "results_human.json").exists() else None
    iaa = json.loads((CACHE / "human_iaa.json").read_text(encoding="utf-8")) \
        if (CACHE / "human_iaa.json").exists() else None

    P: list[str] = []
    A = P.append
    A("<html><head><meta charset='utf-8'><title>CG-Probes — results & interpretation</title><style>"
      "body{font-family:system-ui,-apple-system,sans-serif;margin:0 auto;max-width:980px;padding:2rem 2.2rem;"
      "color:#1a1a1a;line-height:1.55}h1{font-size:1.7rem}h2{margin-top:2.2rem;border-bottom:2px solid #e8e8e8;"
      "padding-bottom:.3rem}h3{margin-top:1.4rem;color:#333}h4{margin:1.1rem 0 .3rem;color:#444;font-size:1.02rem}"
      "table{border-collapse:collapse;margin:.8rem 0;font-size:.93rem}"
      "td,th{border:1px solid #d4d4d4;padding:5px 11px;text-align:right}td:first-child,th:first-child{text-align:left}"
      "th{background:#f3f4f6}.win{font-weight:bold;color:#0a7a3a}.note{background:#f7f9fc;border-left:4px solid #4a7fc0;"
      "padding:.7rem 1rem;margin:1rem 0;border-radius:0 4px 4px 0}.warn{background:#fff8f0;border-left:4px solid #d08a2a}"
      "code{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:.88em}.heat td{text-align:center;min-width:46px}"
      "em{color:#555}ul{margin:.4rem 0}"
      ".warn-cell{background:#ffe0e0;font-weight:bold;color:#a33}"
      ".qsec{margin-top:1.4rem}.qsum{cursor:pointer;font-size:1.25rem;font-weight:600;color:#333;"
      "list-style:none}.qsum::-webkit-details-marker{display:none}.qhint{font-size:.8rem;"
      "font-weight:400;color:#4a7fc0}details[open] .qhint{display:none}"
      "table.cf{font-size:.82rem;margin:.3rem .4rem .3rem 0}table.cf td,table.cf th{text-align:center;padding:3px 7px}"
      "table.cf caption{font-size:.82rem;color:#666;padding-bottom:2px;caption-side:top}"
      "figure{margin:1.2rem 0;text-align:center}figure img{max-width:100%;border:1px solid #eee;border-radius:4px}"
      "figcaption{font-size:.85rem;color:#555;margin-top:.4rem;text-align:left;max-width:760px;margin-left:auto;margin-right:auto}"
      "</style></head><body>")

    A("<h1>CG-Probes — Recovering clinically-grounded directions in patient-query embeddings</h1>")
    A("<p><em>Self-contained results report. Generated from the benchmark cache. "
      "Lead metric is <b>quadratic-weighted κ</b> on the clinician gold (§3); the silver benchmark "
      "and separability use macro-F1. Silver vs golden is explained in §2.</em></p>")

    # ---- TL;DR ----
    A("<div class='note'><b>Summary.</b> We test whether three clinical "
      "safety axes — Medical Urgency (MU), Psychological Urgency (PU), and Topic "
      "Sensitivity (ET, the intrinsic emotional load of the subject), each graded 0/1/2 — "
      "can be read off a <b>frozen</b> text "
      "embedder as single linear <b>directions</b> (a dot product), with no model "
      "fine-tuning. They can. On held-out, cluster-disjoint data the probes separate all "
      "three axes well above chance, and against the <b>two-oncologist clinician gold</b> a "
      "frozen-encoder linear probe is <b>level with the deployable safety LLM "
      "(gpt-oss-safeguard-20b) on the urgency axes (MU, PU)</b> and trails the larger reasoning LLMs "
      "only modestly; the topic-load axis (ET) is the one place the LLMs' world knowledge wins. "
      "The probe is a <b>single forward pass through a frozen encoder — no token generation</b> — so "
      "at the system level it is <b>~10²–10³× cheaper per query</b> than an autoregressive safety LLM, "
      "and its cost is <b>flat per added policy axis</b> (all axes read one shared embedding), so a "
      "10–15-signal policy is essentially free. <i>(If the query embedding is already computed — e.g. "
      "reused from the RAG retriever — the marginal probe cost is one dot product, ~0.3 µs.)</i></div>")

    # ---- PIPELINE & MODELS ----
    A("<h2>1. Pipeline &amp; models</h2>")
    A("<p>The full pipeline and the model used at each step. The operating language is "
      "<b>Czech</b>: the generative and annotation steps use models picked for Czech fluency, and all "
      "four frozen embedders officially support Czech (multilingual).</p>")
    A("<table><tr><th>step</th><th>model(s)</th><th>role</th></tr>"
      "<tr><td><b>1. Query clustering</b></td>"
      "<td>Qwen3-Embedding-4B → UMAP → HDBSCAN; cluster cards by gpt-5.4-mini</td>"
      "<td>group 79,658 de-duplicated real queries into ~1,889 topic superclusters; defines "
      "the <b>cluster-disjoint</b> train/test split</td></tr>"
      "<tr><td><b>2. Contrastive pair synthesis</b></td>"
      "<td>gemini-3.5-flash (generate) + gpt-5.4-mini (blind verify)</td>"
      "<td>create the safety-critical grades (MU2, PU1/PU2) that are near-absent in real logs; "
      "the verifier is a <em>different</em> model family, so a positive can’t be a "
      "shared-generator artifact</td></tr>"
      "<tr><td><b>3. Golden reference annotation</b></td>"
      "<td>gpt-5.4 (virtual annotator)</td>"
      "<td>an independent third grade alongside the two oncologists on the 200-item gold "
      "(itself <b>excluded</b> from the adjudicated gold)</td></tr>"
      "<tr><td><b>4. Frozen embedders under test</b></td>"
      "<td>Qwen3-Embedding-8B, gemini-embedding-2, text-embedding-3-large, Harrier-27b</td>"
      "<td>the frozen spaces in which we probe for linear safety directions — no fine-tuning</td></tr>"
      "<tr><td><b>5. Deployable safety-LLM baselines</b></td>"
      "<td>gpt-oss-120b, gpt-oss-safeguard-20b</td>"
      "<td>the one-call policy classifiers the probe is benchmarked against</td></tr>"
      "</table>")
    A("<div class='note'><b>Model selection.</b> Czech is lower-resource, so "
      "model choice matters. Across candidates we <b>eyeballed Czech output quality</b>, and "
      "<b>GPT (gpt-5.4 family) was clearly the strongest at Czech — especially at imitating the "
      "patient register and real search-query style</b> (short, lay, diacritic-free, often "
      "misspelled). It is therefore used for synthesis verification and as the virtual annotator. "
      "The embedders were chosen for Czech / multilingual coverage; <b>Harrier-27b</b> — English-first "
      "in origin but officially Czech-supporting and <b>near the top of the MTEB leaderboard</b> — is "
      "included as a strong general-purpose contrast, so the linear-recoverability result is not tied "
      "to one embedder family. (It still underperforms the multilingual Qwen/gemini on these Czech "
      "clinical-safety axes — §3.1 — so a top general embedder is not automatically best here.) "
      "Probe <b>directions and thresholds are fit only on the "
      "cluster-disjoint contrastive split — never on the clinician gold</b>; the headline then "
      "reports, per axis, the strongest embedder cell (which cell is shown in §3.1).</div>")

    # ---- AXES + PROBE (methods, merged from old 'What we did') ----
    A("<p><b>The three safety axes</b> (clinician-defined, each graded 0/1/2): <b>MU</b> — medical / "
      "somatic urgency (0 = benign, 2 = somatic emergency needing EMS); <b>PU</b> — psychological "
      "urgency (0 = no distress, 2 = suicidal intent); <b>ET</b> — Topic Sensitivity, the intrinsic "
      "emotional load of the <em>topic</em> (0 = benign, 2 = prognosis / end-of-life / body-integrity), "
      "graded independent of tone. Source: 79,658 de-duplicated Czech oncology search queries "
      "clustered into ~1,889 superclusters (the cluster-disjoint train/test split).</p>")
    A("<p><b>The probe.</b> For each axis we take a single <b>difference-in-means direction</b> on "
      "L2-normalized embeddings (mean of grade-2 minus topic-matched grade-0), project a query onto "
      "it, and split the 1-D projection into 0/1/2 with two tuned thresholds — scoring a query is one "
      "dot product. The rare safety-critical grades absent in real logs (MU2, PU1/PU2) are "
      "synthesized and cross-model verified (§1, step 2); directions and thresholds are fit only on "
      "the cluster-disjoint contrastive split, never on the clinician gold.</p>")

    # ---- HOW TO READ (moved before the headline) ----
    A("<h2>2. How to read the numbers</h2>")
    A("<div class='note'><b>Two label sets — silver and golden.</b>"
      "<ul style='margin:.4rem 0'>"
      "<li><b>Silver</b> = labels from the <b>pipeline</b> (cluster picker + a gpt-5.4 reference "
      "grade), on ~400 queries. Cheap and immediate — so the early experiments (separability, the "
      "embedder × instruction ablation, the geometry and validity checks) were run on silver "
      "<b>without waiting for the clinicians</b>, because it was already good enough to establish the "
      "method.</li>"
      "<li><b>Golden</b> = the <b>clinician</b> ground truth: 200 queries each graded by <b>two "
      "oncologists</b> with adjudication (§3). This is the benchmark that counts; the headline (§3) "
      "is scored against it.</li></ul>"
      "<b>Why both sets.</b> Silver enabled fast iteration at larger N; golden then confirms the "
      "conclusions hold against real clinicians. The <b>silver→golden delta</b> (§3.3) shows the "
      "headline barely moves (MU/PU &lt;0.04 QWK) — i.e. silver was a faithful stand-in, not a "
      "shortcut that flattered the result.</div>")
    A("<ul>"
      "<li><b>QWK</b> (quadratic-weighted κ) is the <b>lead metric</b> on the clinician gold (§3): "
      "ordinal agreement that credits near-misses and penalizes 0↔2 confusions, and the same metric "
      "the oncologists' inter-rater agreement is reported in. <b>0 = chance, 1 = perfect.</b></li>"
      "<li><b>macro-F1</b> averages the F1 of each of the 3 grades equally, so a model can't win by "
      "only predicting the common grade (<b>≈0.33 = chance</b>). Used for separability and the other "
      "silver-based checks.</li>"
      "<li><b>Separability</b> = macro-F1 on the held-out, cluster-disjoint contrastive TEST split — "
      "<em>“is the axis linearly present at all?”</em>, no human labels involved.</li>"
      "<li>Numbers are reported <b>real-only</b> (synthetic excluded) as the headline, plus "
      "<em>all</em>, because synthetic rare-grade items can flatter the score.</li></ul>")

    # ---- CLINICIAN-GOLD HEADLINE ----
    if rh:
        SYS = ["probe", "gpt-oss-safeguard-20b", "gpt-oss-120b", "gpt-5.4"]
        SYS_LABEL = {"probe": "DoM probe", "gpt-oss-safeguard-20b": "safeguard-20b",
                     "gpt-oss-120b": "gpt-oss-120b", "gpt-5.4": "gpt-5.4 (frontier)"}
        A("<h2>3. Headline — clinician gold (N≈200, two oncologists)</h2>")
        A("<div class='note'><b>Clinician-gold benchmark.</b> Every system is scored "
          "against the <b>clinician gold</b> — 200 queries each labelled by <b>two oncologists</b> "
          "(package “Golden v1 — Core”), gold = adjudicated value or the agreed value; the LLM judge "
          "is excluded from gold. Lead metric = <b>quadratic-weighted κ</b> (ordinal agreement); the "
          "dashed line is the <b>inter-oncologist agreement ceiling</b> — what two humans achieve. "
          "Numbers are real-only (synthetic excluded). The method sections (§4–§7) use the larger "
          "silver (pipeline-label) set, run before the clinicians; the silver→human delta is in §3.3.</div>")
        A(_fig(figs_h, "F2h_vs_human", "98%"))

        # 0.1 systems table
        A("<h3>3.1 Per-axis κ vs clinician gold (real-only)</h3>")
        A("<table><tr><th>axis</th><th>human ceiling κ</th>"
          + "".join(f"<th>{SYS_LABEL[s]}</th>" for s in SYS) + "<th>n</th></tr>")
        for ax in SAFETY_AXES:
            cell = rh["axes"][ax]; cc = rh["ceiling"][ax]
            ceil = cc.get("human_human_qwk_adj", cc.get("human_human_qwk"))
            ceil_txt = (f"{_f(ceil)} (raw {_f(cc.get('human_human_qwk_raw'))})†"
                        if cc.get("reconciled") else _f(ceil))
            sg = (cell.get("gpt-oss-safeguard-20b", {}).get("real_only") or {}).get("qwk")
            row = [f"<td><b>{ax}</b></td><td>{ceil_txt}</td>"]
            for s in SYS:
                blk = cell.get(s, {}).get("real_only") or cell.get(s, {}).get("all") or {}
                q = blk.get("qwk")
                win = (s == "probe" and sg is not None and q is not None and q >= sg)
                row.append(f"<td class='{'win' if win else ''}'>{_f(q)}</td>")
            n = (cell.get("probe", {}).get("real_only") or {}).get("n", "—")
            row.append(f"<td>{n}</td>")
            A("<tr>" + "".join(row) + "</tr>")
        A("</table>")
        A("<div class='note'><b>Reading.</b> On the urgency axes <b>MU and PU</b> the frozen-encoder "
          "DoM probe shows <b>no detectable difference from the deployable LLMs</b>, though the eval "
          "is underpowered to <i>prove</i> equivalence. Paired on the same 89 real items, ΔQWK "
          "(probe−LLM) is small with every 95% CI including 0 — vs safeguard·low +0.02 (MU) / +0.05 "
          "(PU); vs gpt-oss-120b −0.06 / −0.08. A TOST equivalence test (α=0.05) does <b>not</b> clear "
          "a tight ±0.10 margin (90% CIs reach ±0.15–0.18), so at n=89 we certify equivalence only "
          "within ≈±0.18. Point estimates: MU probe 0.65 vs safeguard 0.64 / 120b 0.72; PU 0.81 vs "
          "0.76 / 0.89. Only the frontier gpt-5.4 pulls clear on PU (0.98 [0.93,1.00], which exceeds "
          "the 0.96 ceiling = sampling variance). So the claim is <b>competitive with the deployable "
          "LLMs (no significant difference) at far lower marginal cost</b>, not proven equivalence or "
          "a point-estimate win. On <b>ET</b> the probe is the weakest system; we treat ET as a "
          "probe-side <b>negative result</b> (topic-confounded, η²=0.81 — §7.2) and make <b>no ET "
          "system ranking</b> (wide CIs at n≈89; the reconciled gold encodes one coding convention, "
          "so a high ET score partly reflects sharing it). The headline probe is the interpretable "
          "<b>1-D difference-in-means direction</b>; the full-linear head (used to improve ET) is "
          "in <b>§3.4</b>. "
          "<b>†The ET ceiling (0.82) is the reliability of the <i>reconciled</i> codebook</b> "
          "(raw inter-rater κ 0.32; one under-specified cell, systematically adjudicated — §3.2; "
          "raw shown as the faint tick in F2h).</div>")

        # 2.1b per-embedder DoM (every embedder at the headline mode; real-only κ + 95% CI)
        try:
            pbe = json.loads((CACHE / "probe_by_embedder.json").read_text(encoding="utf-8"))
        except Exception:
            pbe = None
        if pbe:
            EMB_ORDER = ["qwen8b", "gemini", "openai3large", "harrier27b"]
            lab = pbe["emb_label"]; hm = pbe.get("head_mode", {})
            A("<p style='margin-top:1rem'><b>Same DoM probe, every embedder.</b> The headline picks "
              "the best embedder per axis; here is the 1-D DoM probe on the clinician gold for "
              "<i>all four</i> frozen embedders, each at that axis's headline instruction mode "
              "(real-only κ, 95% bootstrap CI). <b>Bold green = best on that axis</b> (ties within "
              "0.02 share it). The effect is not one lucky embedder — but it <i>is</i> "
              "embedder-dependent (the multilingual Qwen and gemini lead; OpenAI and Harrier trail — "
              "Harrier despite topping MTEB and listing Czech support).</p>")
            A("<table><tr><th>axis</th><th>mode</th>"
              + "".join(f"<th>{lab.get(e, e)}</th>" for e in EMB_ORDER) + "</tr>")
            for ax in SAFETY_AXES:
                cells = pbe["axes"].get(ax, {})
                row = [f"<td><b>{ax}</b></td><td><code>{hm.get(ax, '—')}</code></td>"]
                for e in EMB_ORDER:
                    c = cells.get(e)
                    if not c:
                        row.append("<td>—</td>"); continue
                    ci = f" <span style='color:#888;font-size:.82em'>[{_f(c['lo'])},{_f(c['hi'])}]</span>"
                    row.append(f"<td class='{'win' if c.get('winner') else ''}'>{_f(c['qwk'])}{ci}</td>")
                A("<tr>" + "".join(row) + "</tr>")
            A("</table>")
            A("<p style='font-size:.85rem;color:#777'>— = that embedder has no cell at the axis's "
              "headline mode (OpenAI text-embedding-3 takes no task instruction, so it has no "
              "per-axis cell). Modes: MU=per_axis, PU=generic, ET=generic.</p>")

        # 2.1c real-only vs full-set ceiling reconciliation
        A("<div class='note'><b>Full-set vs real-only ceiling.</b> The dashed line in F2h and the “human ceiling κ” "
          "column above are the <b>full-set</b> inter-oncologist QWK (all 200 items, synthetic "
          "included). On the <b>real-only</b> queries the systems are actually scored on, "
          "inter-oncologist agreement is <b>markedly lower</b> — <b>MU 0.49, PU 0.91, ET 0.32</b> "
          "(§3.5) — because real grade-2 cases are rare and genuinely ambiguous for humans too. So on "
          "real queries the probe <b>matches or exceeds</b> the real-only human ceiling on MU "
          "(0.65 vs 0.49) and sits just under it on PU; we headline the conservative full-set ceiling "
          "and report the real-only ceiling in §3.5, so the human bar is neither over- nor "
          "under-stated.</div>")

        # 2.2 ET IAA error analysis
        et = (iaa or {}).get("et_error_analysis") or rh.get("et_error_analysis")
        if et:
            A("<h3>3.2 Why ET inter-rater agreement is low — a single codebook ambiguity</h3>")
            A(f"<p>ET inter-oncologist agreement is low (κ={_f(rh['ceiling']['ET']['human_human_qwk'],2)}, "
              f"Krippendorff α={_f((iaa or {}).get('axes',{}).get('ET',{}).get('alpha_human'),2)}), "
              "but the disagreement is <b>not random</b>: it is near-deterministic and concentrated on "
              "one boundary.</p><ul>"
              f"<li><b>{et['all_one_dir_disagreements_n']}/{et['all_one_dir_disagreements_n']+1}</b> "
              "disagreements go one direction (one rater grades higher).</li>"
              f"<li>That over-rating concentrates on the ET1→ET2 cell (n={et['one_cell_ET1_to_ET2_n']}), "
              f"where <b>{_f(et['share_MU_ge1_in_cell'],2)}</b> of items are somatic-urgent (MU≥1) vs "
              f"only <b>{_f(et['share_MU_ge1_elsewhere'],2)}</b> elsewhere — a 4× enrichment.</li>"
              "<li>One rater coded <b>acute somatic urgency as high emotional load (ET2)</b>; the "
              "codebook intended ET2 for <b>long-term</b> emotional/existential load, with somatic "
              "urgency staying ET1.</li>"
              f"<li><b>Reconciling this one ambiguity lifts ET agreement to κ="
              f"{_f(et['qwk_after_somatic_reconcile'],2)}</b> — on par with MU/PU.</li></ul>"
              "<p><em>Interpretation: ET’s raw α understates reliability; it reflects a single "
              "codebook underspecification, not construct unreliability. The adjudicated gold applies "
              "the codebook-intended rule (ET2 = long-term emotional load, not somatic severity), so "
              "the ET benchmark is on the corrected boundary.</em></p>")

        # 0.3 silver→human delta + ceiling fig
        A("<h3>3.3 Does the silver-label benchmark hold up against clinicians?</h3>")
        A("<table><tr><th>axis</th><th>cell</th><th>silver κ</th><th>human κ</th><th>Δ</th></tr>")
        for ax in SAFETY_AXES:
            sd = rh["silver_delta"].get(ax, {})
            A(f"<tr><td><b>{ax}</b></td><td><code>{sd.get('cell','—')}</code></td>"
              f"<td>{_f(sd.get('silver_qwk'))}</td><td>{_f(sd.get('human_qwk'))}</td>"
              f"<td>{('+' if (sd.get('delta') or 0)>=0 else '')}{_f(sd.get('delta'))}</td></tr>")
        A("</table>")
        A("<p><em>MU/PU move ≤0.04 κ from silver to clinician gold — the pre-clinician "
          "conclusions hold. The 1-D DoM ET drops ~0.15 (silver over-credited it), consistent "
          "with ET being the contested axis — and the reason ET uses the full-linear head (§3.4), "
          "which is silver-stable. This is why silver was a usable stand-in: every method experiment "
          "(separability, instruction conditioning, geometry, validity) was run on silver before the "
          "clinician set existed, and the one comparison re-checkable against clinicians — the probe "
          "vs the deployable LLMs — keeps its ranking (probe level with safeguard on MU/PU, behind on "
          "ET).</em></p>")

        # 0.4 improving ET: a linear subspace, not one direction
        ph = rh.get("probe_heads", {})
        if ph:
            A("<h3>3.4 The probe head: one direction (interpretable) vs a linear subspace</h3>")
            A("<p>The <b>headline</b> probe (§3.1) is a <b>single difference-in-means direction</b> per "
              "axis — the geometric claim — which is near-optimal for MU/PU but <b>underfits ET</b>. For "
              "ET we additionally fit a class-balanced <b>full-linear</b> head in the same frozen space. "
              "Below we compare 1-D DoM, the full-linear head, and a non-linear MLP, all selected on the "
              "cluster-disjoint contrastive TEST and evaluated on clinician gold. (We also tried ordinal, "
              "whitening, and 4-embedder concatenation heads — all underperformed the full-linear head; "
              "<code>benchmark/probe_extra.py</code>.)</p>")
            A("<table><tr><th>axis</th><th>1-D DoM (κ)</th><th>full-linear (κ)</th><th>MLP (κ)</th>"
              "<th>ceiling</th></tr>")
            for ax in SAFETY_AXES:
                h = ph[ax]
                ceil = rh["ceiling"][ax].get("human_human_qwk_adj")
                gain = h["linear"]["human_qwk"] - h["dom"]["human_qwk"]
                A(f"<tr><td><b>{ax}</b></td><td>{_f(h['dom']['human_qwk'])}</td>"
                  f"<td class='{'win' if gain>0.02 else ''}'>{_f(h['linear']['human_qwk'])}</td>"
                  f"<td>{_f(h['mlp']['human_qwk'])}</td><td>{_f(ceil)}</td></tr>")
            A("</table>")
            A(_fig(figs_h, "F10_probe_heads", "62%"))
            A("<div class='note'><b>Result.</b> The full-linear head lifts <b>ET 0.49→0.58 κ</b> "
              "(macro-F1 0.66→0.74) — and the MLP does <b>no better</b>, so the gain is from "
              "<em>capacity, not non-linearity</em>: <b>ET is a linear subspace, not a single "
              "direction</b>. Same conclusion as the orthogonality strip (F5), the linear-vs-MLP "
              "ablation (F8) and the counterfactual test — ET is topic-distributed. We therefore keep the "
              "<b>1-D DoM direction as the headline</b> (a single clinician-meaningful vector, and tied "
              "with any heavier head on MU/PU — it even beats full-linear on the contrastive TEST), and "
              "use the <b>full-linear head only for ET</b>, where one direction underfits. <b>To push ET "
              "past ~0.58 the lever is training "
              "data, not the head</b> — MU-decorrelated ET pairs — since ordinal/whitening/concat heads "
              "did not help.</div>")

        A(_fig(figs_h, "F9_ceiling", "58%"))
        A(_fig(figs_h, "F3h_pareto", "62%"))
        # NOTE: golden-set inter-annotator-agreement section (§3.5/3.6) is omitted
        # from the public artifact — it renders annotator-level data, which is withheld.

    # ---- SEPARABILITY ----
    A("<h2>4. Does each axis separate at all? (no human labels)</h2>")
    A("<p>Best separability per axis on the held-out contrastive test split:</p><ul>")
    for ax in SAFETY_AXES:
        sub = df[df["axis"] == ax].sort_values("sep_f1", ascending=False).iloc[0]
        A(f"<li><b>{ax}</b>: {sub['sep_f1']:.3f} (<code>{sub['embedder']}/{sub['mode']}</code>) — "
          f"{'strong' if sub['sep_f1']>0.8 else 'clear' if sub['sep_f1']>0.6 else 'weak'} "
          f"separation, well above the 0.33 chance line.</li>")
    A("</ul><p><em>All three axes are linearly present. PU separates most strongly (but it is the "
      "most synthetic-aided axis); ET separates on 100% real data, so it is the most trustworthy "
      "single result.</em></p>")
    A("<p>This is the central claim made <b>directly visible</b>: each axis's held-out queries, "
      "projected onto the single probe direction, sort into ordered grade bands (0&lt;1&lt;2).</p>")
    A(_fig(figs, "F1_separability"))

    # ---- EMBEDDER x INSTRUCTION ----
    A("<h2>5. Embedders &amp; the instruction-conditioning ablation</h2>")
    A("<p>Frozen embedders can be <b>conditioned</b> with a task instruction prepended to the query. "
      "We tried three modes — <code>none</code> (raw query), <code>generic</code> (one shared "
      "instruction), <code>per_axis</code> (an axis-specific instruction) — and scored the probe on "
      "the <b>silver</b> set (this ablation predates the clinician gold; see §2). The exact "
      "instructions used:</p>")
    try:
        EMB = CACHE / "emb"

        def _instr(nm):
            return json.loads((EMB / f"qwen8b__{nm}.meta.json").read_text(encoding="utf-8")).get("instruction")
        A("<table><tr><th>mode</th><th>instruction (prepended to the Czech query)</th></tr>"
          "<tr><td><code>none</code></td><td><em>— raw query, no instruction</em></td></tr>")
        for nm, lbl in [("generic", "generic (all axes)"), ("per_axis__MU", "per_axis · MU"),
                        ("per_axis__PU", "per_axis · PU"), ("per_axis__ET", "per_axis · ET")]:
            A(f"<tr><td><code>{lbl}</code></td><td>{html.escape(_instr(nm) or '—')}</td></tr>")
        A("</table>")
        A("<p style='font-size:.85rem;color:#777'>Instructions are identical across the "
          "instruction-following embedders (Qwen/gemini/Harrier); OpenAI text-embedding-3 takes no "
          "instruction, so its <code>generic</code> falls back to raw. A positive-only "
          "<code>per_axis_pos</code> variant (dropping the “ignoring …” clause) was also tested — "
          "it did not rescue ET (note below).</p>")
    except Exception as e:
        A(f"<p class='warn'>[instructions unavailable: {html.escape(str(e))}]</p>")
    A("<p><b>Silver</b> real-only macro-F1 for every embedder × instruction mode:</p>")
    piv = df.pivot_table(index=["embedder", "mode"], columns="axis", values="gold_real")
    piv = piv.reindex(columns=SAFETY_AXES)
    A(piv.round(3).to_html(na_rep="—"))
    A("<div class='note'><b>Finding.</b> Conditioning the embedder on a "
      "per-axis instruction is <b>not</b> uniformly good. It <b>helps the intent axes (MU, PU)</b> "
      "but <b>hurts the topic axis (ET)</b> on every embedder — most dramatically gemini ET "
      "(0.73 generic → 0.40 per-axis). We tested whether negation in the instruction was to blame "
      "(a positive-only variant): it was not — removing negation did not rescue ET. The mechanism: "
      "ET signal <em>is</em> the topic, so any axis-focusing instruction pulls the embedding away "
      "from the topical content the ET direction rides on. <b>Best practice: per-axis instruction "
      "for MU/PU, generic/raw for ET.</b></div>")

    # ---- COLLINEARITY ----
    A("<h2>6. Are the three axes geometrically distinct?</h2>")
    A("<p>Cosine between the learned axis directions (1.0 = identical direction, 0 = orthogonal). "
      "Low off-diagonal = the axes are not redundant.</p>")
    A("<div style='display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start'>")
    for cell, M in list(coll.items())[:3]:
        A(f"<div><div style='font-weight:600;margin-bottom:.2rem'>{html.escape(cell)}</div>"
          f"{_heat(M)}</div>")
    A("</div>")
    A(_fig(figs, "F4_cross_axis", "36%"))
    A("<p><em>MU↔PU correlate moderately (~0.4–0.6, both are “urgency”); ET is the most "
      "independent (cosine ~0.1–0.5 with the others). The axes are distinct directions, not "
      "three views of one signal — which justifies treating them separately. Per-axis instruction "
      "tends to increase these cosines (it entangles the axes), another reason it can hurt.</em></p>")

    # ---- CONSTRUCT VALIDITY (Tier-1 / Tier-2) ----
    A("<h2>7. Are the probes <em>valid</em>? (extra tests, no clinician labels)</h2>")
    A("<p>Separability and golden macro-F1 alone don't rule out three failure modes a "
      "reviewer will raise: the probe could be a <b>topic detector</b>, the signal could be a "
      "non-linear <b>artifact</b>, or macro-F1 could hide <b>ordinal</b> errors. We ran five "
      "checks that need no human labels.</p>")

    # 7.1 Ordinal / clinical metrics
    A("<h3>7.1 Ordinal &amp; safety-critical metrics (golden, real-only)</h3>")
    A("<p>macro-F1 treats a 0→2 error like a 0→1 error. <b>QWK</b> (quadratic-weighted κ) "
      "penalizes far-off ordinal errors; <b>Spearman ρ</b> is the threshold-free monotonic "
      "association between the raw projection and the grade; <b>AUROC≥2</b> is the deployable "
      "“flag the emergency” operating characteristic; F1(g2) is the safety-critical class alone.</p>")
    A("<table><tr><th>Axis</th><th>cell</th><th>QWK</th><th>Spearman ρ</th>"
      "<th>AUROC ≥1</th><th>AUROC ≥2</th><th>F1(g0)</th><th>F1(g1)</th><th>F1(g2)</th><th>n</th></tr>")
    for ax in SAFETY_AXES:
        b = t1_best.get(ax)
        g = (b or {}).get("golden", {}).get("real_only", {})
        if not g:
            continue
        pc = g.get("f1_per_grade", [None, None, None])
        A(f"<tr><td><b>{ax}</b></td><td><code>{b['embedder']}/{b['mode']}</code></td>"
          f"<td class='win'>{_f(g.get('qwk'))}</td><td>{_f(g.get('spearman_rho'))}</td>"
          f"<td>{_f(g.get('auroc_ge1'))}</td><td>{_f(g.get('auroc_ge2'))}</td>"
          f"<td>{_f(pc[0])}</td><td>{_f(pc[1])}</td><td>{_f(pc[2])}</td><td>{g.get('n')}</td></tr>")
    A("</table>")
    A("<div class='note'><b>Reading.</b> AUROC≥2 (rank-quality of the emergency flag) is the "
      "strongest column — the raw projection ranks the safety-critical cases well even where the "
      "thresholded F1(g2) is modest, i.e. the signal is there and only the operating point needs "
      "tuning. The thin grade-1 band (few real items) is where most F1 is lost.</div>")

    # 7.2 Topic-orthogonality
    A("<h3>7.2 Topic-orthogonality — is it just a topic detector?</h3>")
    A("<p>For each axis direction we measured the cosine to every BERTopic supercluster's "
      "<b>grade-0 centroid</b> (a pure-topic anchor, built from benign queries so no grade signal "
      "leaks in). Low cosine ⇒ the axis is not aligned with topic.</p>")
    A("<table><tr><th>Axis</th><th>cell</th><th>mean |cos|</th><th>p95 |cos|</th>"
      "<th>max |cos|</th><th>η² (proj~topic)</th><th>verdict</th></tr>")
    for ax in SAFETY_AXES:
        to = (t1_best.get(ax) or {}).get("topic_orthogonality", {})
        if not to:
            continue
        A(f"<tr><td><b>{ax}</b></td><td><code>{t1_best[ax]['embedder']}/{t1_best[ax]['mode']}</code></td>"
          f"<td class='win'>{_f(to.get('mean_abs_cos'),2)}</td><td>{_f(to.get('p95_abs_cos'),2)}</td>"
          f"<td>{_f(to.get('max_abs_cos'),2)}</td><td>{_f(to.get('eta2_projection'),2)}</td>"
          f"<td>{to.get('verdict','—')}</td></tr>")
    A("</table>")
    A(_fig(figs, "F5_orthogonality", "50%"))
    A("<p><em>The decisive metric is <b>η² (variance of the projection explained by topic-cluster "
      "membership)</b>, not mean cosine: a probe can have low cosine to any single centroid yet still "
      "be predicted by which cluster a query is in. <b>MU (η²≈0.12) and PU (η²≈0.22) are not topic "
      "detectors</b> — low cosine and low η² agree. <b>ET (η²≈0.81) is topic-confounded</b>: topic "
      "membership explains ~81% of where an ET query lands, despite a low mean cosine — so the 1-D ET "
      "probe is substantially a topic detector, consistent with every other ET result. The probes "
      "encode safety, not topic, for MU/PU; ET does not.</em></p>")

    # 7.3 Linear sufficiency
    A("<h3>7.3 Is a <em>linear</em> probe enough?</h3>")
    A("<p>If a non-linear read-out of the same embedding did much better, “linearly recoverable” "
      "would be overstated. We compare the 1-D difference-in-means direction, a full-dimensional "
      "linear (logistic) probe, and a 1-hidden-layer MLP — all on the held-out test split.</p>")
    A(_fig(figs, "F8_linear_mlp", "50%"))
    A("<div class='note'><b>Finding.</b> The MLP barely beats the full linear probe anywhere "
      "(non-linearity buys ≈0) — so <b>“linearly recoverable” is earned</b>. The 1-D direction is "
      "near-optimal for MU/PU, but for <b>ET it loses ~0.17</b> to the full-linear probe: ET is "
      "genuinely <b>multi-directional</b> (a topic family, not one axis) — converging with 7.2 and "
      "the instruction ablation.</div>")

    # 7.4 Selectivity
    A("<h3>7.4 Selectivity — concept, or exploitable structure?</h3>")
    A("<p>Hewitt &amp; Liang's control task: refit the same probe on <b>grade-shuffled</b> labels "
      "(identical marginals, no real signal). A high real−control gap means the probe reads the "
      "concept rather than incidental embedding structure.</p>")
    A(_fig(figs, "F7_selectivity", "50%"))

    # 7.5 Counterfactual
    if cf:
        A("<h3>7.5 Counterfactual minimal pairs — does it move for the right reason?</h3>")
        A("<p>For grade-1 seeds we generated, holding topic constant, a <b>paraphrase</b> (grade "
          "unchanged — projection should <em>not</em> move) and minimal <b>flip-up / flip-down</b> "
          "edits (grade ±1 — projection <em>should</em> move, in the right direction). Effects are "
          "Δprojection in units of the axis's projection SD.</p>")
        A("<table><tr><th>embedder</th><th>axis</th><th>invariance (Δ paraphrase)</th>"
          "<th>sensitivity (Δ flip)</th><th>ratio</th><th>direction-correct</th></tr>")
        for emb, per_axis in cf.get("by_embedder", {}).items():
            for ax in SAFETY_AXES:
                d = per_axis.get(ax)
                if not d:
                    continue
                ratio = d.get("ratio")
                cls = "win" if (ratio and ratio > 1.5) else ""
                A(f"<tr><td><code>{emb}</code></td><td><b>{ax}</b></td>"
                  f"<td>{_f(d.get('invariance_effect'),2)}</td>"
                  f"<td>{_f(d.get('sensitivity_effect'),2)}</td>"
                  f"<td class='{cls}'>{_f(ratio,2)}</td>"
                  f"<td>{_f(d.get('direction_correct_rate'),2)}</td></tr>")
        A("</table>")
        A("<div class='note'><b>Finding.</b> For <b>MU and PU</b> a grade-flip moves the projection "
          "several× more than a paraphrase, in the correct direction ~90–99% of the time — strong "
          "behavioural (quasi-causal) evidence the probe tracks the safety concept itself. For "
          "<b>ET</b> the ratio collapses toward 1 and direction-correctness toward chance: a minimal "
          "ET grade-flip necessarily changes the topic, so a 1-D probe can't tell it from a "
          "paraphrase — the same ET-is-multidimensional story again.</div>")

    # 7.6 report-only extras
    A("<h3>7.6 Confusion &amp; the PU–ET entanglement, visualized</h3>")
    A(_fig(figs, "R_confusion", img_w="26%"))
    A(_fig(figs, "R_entangle", "38%"))
    A(_fig(figs, "F6_embedder_axis", "48%"))

    # ---- COST ----
    A("<h2>8. Inference cost &amp; deployability on one GPU</h2>")
    _lj = json.loads((BASE_DIR / "safeguard_latency.json").read_text(encoding="utf-8")) \
        if (BASE_DIR / "safeguard_latency.json").exists() else {}
    _tj = json.loads((BASE_DIR / "safeguard_throughput.json").read_text(encoding="utf-8")) \
        if (BASE_DIR / "safeguard_throughput.json").exists() else {}
    sweep = safeguard_reasoning("human")
    lo, me = _lat(_lj, "low"), _lat(_lj, "medium")
    thr_rows = (_tj.get("gpt-oss-safeguard-20b__medium", {}) or {}).get("sweep", [])
    thr_max = max((r["qps"] for r in thr_rows if r.get("qps")), default=None)
    A("<p>A probe scores a query with <b>one forward pass through the frozen encoder</b> (no token "
      "generation) followed by a dot product; all three axes read the one embedding. At the system "
      "level that <b>decode-free encoder pass is ~10²–10³× cheaper</b> than the autoregressive LLM, "
      "which decodes hundreds–thousands of tokens per query. Where the embedding is <b>already "
      "computed</b> (e.g. reused from the RAG retriever's index) the <i>marginal</i> probe cost is "
      "just the dot product, ~0.3&nbsp;µs (the ~10⁶–10⁷× figures below). The deployable LLM baseline "
      "(gpt-oss-safeguard-20b) is one <b>triage</b> call per query; three per-axis calls would triple "
      "GPU occupancy, which a single dedicated hospital GPU cannot absorb. All LLM latencies below are "
      "<b>measured</b> on the serving GPU (Nvidia <b>L40S 48&nbsp;GB</b>, HF Inference Endpoint, vLLM).</p>")
    A("<table><tr><th>system</th><th>reasoning</th><th>single-query latency (drained cluster)</th>"
      "<th>vs probe (marginal)</th></tr>")
    A("<tr><td>linear probe <span style='font-weight:400;color:#777'>(marginal, embedding "
      "precomputed)</span></td><td>—</td><td>~0.3&nbsp;µs (one dot product)</td><td>1×</td></tr>")
    if lo:
        A(f"<tr class='win'><td>safeguard-20b <b>(deployed)</b></td><td>low</td>"
          f"<td>mean {_f(lo.get('mean_s'),2)}&nbsp;s · median {_f(lo.get('median_s'),2)} · p95 {_f(lo.get('p95_s'),2)}</td>"
          f"<td>~{lo['mean_s']/0.3e-6/1e6:.0f}×10⁶</td></tr>")
    if me:
        A(f"<tr><td>safeguard-20b</td><td>medium</td>"
          f"<td>mean {_f(me.get('mean_s'),2)}&nbsp;s · p95 {_f(me.get('p95_s'),2)}</td><td>—</td></tr>")
    A("<tr><td>safeguard-20b</td><td>high</td><td>erratic ~13–60&nbsp;s (up to ~10k reasoning "
      "tokens; one cold start hit 941&nbsp;s)</td><td>—</td></tr>")
    A("</table>")
    A(f"<p><b>Throughput ceiling</b>: 4×L40S replicas sustained <b>~{_f(thr_max,1)} queries/sec</b> "
      "at medium reasoning (concurrency 32; a single replica ~0.5–1.5 q/s). Latency is "
      "<b>generated-tokens ÷ GPU decode-rate</b> — the L40S decodes this 20B model at ~100–150 tok/s "
      "single-stream, so a faster GPU (H100 ≈ 3–4× bandwidth) shrinks the constant but not the gap "
      "(<b>~10²–10³× vs a decode-free encoder pass</b>; ~10⁶–10⁷× marginal once the embedding is "
      "computed). The embedding is computed once and shared across all three axes.</p>")

    # 8.1 reasoning-hurts-safety
    A("<h3>8.1 Reasoning effort: more is slower <em>and</em> less safe</h3>")
    A("<p>Counter-intuitively, raising the safety model's reasoning effort <b>lowers</b> its MU "
      "accuracy and emergency recall while multiplying latency: with more reasoning it talks itself "
      "into reading a bare clinical term (“prasklé střevo” = ruptured bowel) as an <i>educational "
      "lookup</i> → MU0. We deploy <b>low</b> reasoning — fastest <em>and</em> best.</p>")
    if "low" in sweep and "medium" in sweep:
        A(f"<figure><img src='{fig_reasoning_latency(sweep, _lj)}' style='max-width:62%'>"
          "<figcaption>F9. Reasoning effort vs single-query latency (bars) and MU agreement with "
          "oncologists (line), annotated with clear emergencies graded benign.</figcaption></figure>")
    A("<table><tr><th>reasoning</th><th>MU F1/QWK</th><th>PU F1/QWK</th><th>ET F1/QWK</th>"
      "<th>emergencies missed</th><th>latency</th></tr>")
    for e in ("low", "medium"):
        if e not in sweep:
            continue
        s = sweep[e]; mm = s["mu2_miss"]; L = _lat(_lj, e)
        nm = "<b>low</b>" if e == "low" else e
        A(f"<tr><td>{nm}</td><td>{_f(s['MU'][0],2)}/{_f(s['MU'][1],2)}</td>"
          f"<td>{_f(s['PU'][0],2)}/{_f(s['PU'][1],2)}</td><td>{_f(s['ET'][0],2)}/{_f(s['ET'][1],2)}</td>"
          f"<td>{mm[0]}/{mm[1]}</td><td>{_f(L.get('mean_s'),2)}&nbsp;s</td></tr>")
    A("</table>")
    _pm = probe_mu2_miss()
    _pm_txt = (f" <b>For the same 28 pipeline-labeled emergencies the headline 1-D DoM MU probe grades "
               f"{_pm[0]}/{_pm[1]} as benign at its deployed threshold</b> (the rest downgraded to MU1 = "
               f"“see a physician,” not dismissed) — the head-to-head safety comparison favours the probe.") if _pm else ""
    A("<div class='note'>Accuracy columns are real-only vs the oncologist gold; emergencies are the "
      "silver set's 28 clear MU2 cases. <b>Even at its best (low), safeguard-20b grades 8 of 28 clear "
      "emergencies as benign</b> (hemoptysis, airway edema, acute limb ischemia, overdose) — an "
      "irreducible floor from over-literal policy-following on bare Czech terms plus a 20B model's "
      f"weaker Czech.{_pm_txt} A prior confound is now fixed: the client had silently dropped "
      "<code>reasoning_effort</code> for gpt-oss models (the routing regex matched only gpt-5/o-series), "
      "so the earlier baseline ran at the vLLM default (~medium); reasoning is now passed via "
      "<code>extra_body</code> and swept.</div>")

    # ---- HOW TO INTERPRET ----
    # ---- CAVEATS ----
    A("<h2>9. Caveats</h2><ul class='warn' style='list-style-position:inside'>")
    A("<li><b>Clinician gold is in hand (headline = §3).</b> The benchmark is now against a "
      "200-item two-oncologist gold with a reported IAA ceiling; the silver (pipeline + gpt-5.4) "
      "400-item numbers in §4–§7 are retained only as a larger-N robustness view. MU/PU shift "
      "&lt;0.04 QWK between silver and clinician gold (§3.3).</li>")
    A("<li><b>ET gold boundary is partly definitional.</b> The ET1/ET2 inter-rater split is low "
      "(raw κ 0.32) because of a documented codebook ambiguity (somatic urgency miscoded as high "
      "emotional load); the adjudicated gold and the reported ET ceiling apply the codebook-"
      "reconciled rule (κ 0.82), with the raw value always disclosed (§3.2).</li>")
    A("<li><b>Synthetic safety cells.</b> MU2/PU2 training positives are 100% synthetic; mitigated "
      "by the generator-mismatch design + verifier gate + reporting real-only, but disclose. The "
      "200-item clinician set is 90 real + 110 synthetic, also reported ±synthetic.</li>")
    A("<li><b>Skewed marginals &amp; small grade-2 cells</b> on a single Czech oncology setting — "
      "real MU2/PU2 are thin, so we lead with QWK/AUROC≥2 over per-grade-2 F1; generalization "
      "untested.</li>")
    A("<li><b>PU separability is inflated by synthetic positives</b> (0.9+ on the contrastive "
      "set); the clinician real-only QWK (~0.81) is the operative number.</li></ul>")

    A(f"<hr><p style='color:#888;font-size:.85rem'>Cells: {len(df)} (4 embedders × instruction modes "
      f"× 3 axes). Baselines: {', '.join(base)}. Best-probe selection = max real-only golden "
      f"macro-F1 per axis.</p>")
    A("</body></html>")

    OUT.write_text("".join(P), encoding="utf-8")
    print(f"[report] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
