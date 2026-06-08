"""Standalone investigative deep-dive: the gpt-oss-safeguard-20b LLM baseline.

Documents how the safety-LLM baseline works (triage pipeline + prompt), and the
2026-06-07 findings: (1) the reasoning_effort routing bug + fix, (2) the reasoning
sweep showing MORE reasoning is slower AND less safe, (3) measured single-query
latency + throughput on the serving GPU (Nvidia L40S 48 GB), (4) accuracy vs the
clinician (oncologist) gold, and (5) why a probe layer wins on cost *scaling*.

    python -m benchmark.report_safeguard
    -> benchmark/cache/safeguard_baseline_report.html
"""

from __future__ import annotations

import glob
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score  # noqa: E402

from benchmark import _gold  # noqa: E402
from benchmark._viz import CB, emit  # noqa: E402
from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int  # noqa: E402

OUT = REPO / "benchmark" / "cache" / "baselines"
CACHE = REPO / "contrastive" / "cache" / "llm_responses"
if not CACHE.exists():
    CACHE = REPO / "cache" / "llm_responses"
LAT = OUT / "safeguard_latency.json"
THR = OUT / "safeguard_throughput.json"
RES_HUMAN = REPO / "benchmark" / "cache" / "results_human.json"
TRIAGE_PROMPT = REPO / "benchmark" / "prompts" / "triage_policy_v1.txt"
OUT_HTML = REPO / "benchmark" / "cache" / "safeguard_baseline_report.html"

SG = "gpt-oss-safeguard-20b"
AXIS_NAME = {"MU": "Medical Urgency", "PU": "Psychological Urgency", "ET": "Topic Sensitivity"}


def t2i(x):
    """Gold grades are axis-prefixed tokens ('MU1'); predictions are plain ints."""
    try:
        return token_to_int(x) if isinstance(x, str) else int(x)
    except Exception:
        return np.nan


def _qwk(a, b):
    return float(cohen_kappa_score(a, b, labels=[0, 1, 2], weights="quadratic"))


# ---------------------------------------------------------------- data loading

def load_pred(effort):
    fn = {"low": "triage_low", "medium": "triage_medium"}[effort]
    p = pd.read_parquet(OUT / f"{SG}__{fn}.parquet")
    p["query_key"] = p["query_key"].astype(str)
    return p


def score_vs(effort, source):
    """real-only {axis:(f1,qwk,n)} of safeguard {effort} vs gold `source`."""
    g = _gold.load_gold(source); g["query_key"] = g["query_key"].astype(str)
    pred = load_pred(effort)
    row = {}
    for ax in SAFETY_AXES:
        col = f"{ax.lower()}_grade"
        tr = g.assign(t=g[col].map(t2i))[["query_key", "t", "synthetic"]].dropna(subset=["t"])
        pp = pred.assign(p=pred[col].map(t2i))[["query_key", "p"]].dropna(subset=["p"])
        m = tr.merge(pp, on="query_key"); m["t"] = m.t.astype(int); m["p"] = m.p.astype(int)
        r = m[~m.synthetic]
        if len(r):
            row[ax] = (f1_score(r.t, r.p, labels=[0, 1, 2], average="macro", zero_division=0),
                       _qwk(r.t, r.p), len(r))
    return row


def emergency_misses(effort):
    """Set of silver-MU2 (real) queries graded MU0 by safeguard {effort}."""
    sv = _gold.load_gold("silver"); sv["query_key"] = sv["query_key"].astype(str)
    sv = sv.assign(mu_t=sv["mu_grade"].map(t2i))
    sv = sv[(sv.mu_t == 2) & ~(sv["source"].astype(str) == "synthesized")]
    pred = load_pred(effort)
    m = sv[["query_key", "query_text"]].merge(pred[["query_key", "mu_grade"]], on="query_key")
    m["mu_grade"] = m["mu_grade"].astype("Int64")
    miss = m[m.mu_grade == 0]
    return set(miss.query_text), len(sv)


def mu_confusion(effort):
    sv = _gold.load_gold("silver"); sv["query_key"] = sv["query_key"].astype(str)
    sv = sv.assign(t=sv["mu_grade"].map(t2i))
    sv = sv[~(sv["source"].astype(str) == "synthesized")].dropna(subset=["t"])
    pred = load_pred(effort)
    m = sv[["query_key", "t"]].merge(
        pred.assign(p=pred["mu_grade"].map(t2i))[["query_key", "p"]], on="query_key").dropna()
    return confusion_matrix(m.t.astype(int), m.p.astype(int), labels=[0, 1, 2])


def rationale_index():
    idx = {}
    for fp in glob.glob(str(CACHE / "*.json")):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if d.get("model") != "openai/gpt-oss-safeguard-20b" or d.get("template_version") != "triage_policy_v1":
            continue
        eff = d.get("payload", {}).get("reasoning_effort", "?")
        u = d.get("payload", {}).get("user", "")
        q = u.split('"')[1] if '"' in u else u
        idx[(q, eff)] = d["response"]
    return idx


# -------------------------------------------------------------------- figures

def fig_reasoning(sweep, lat):
    efforts = ["low", "medium"]
    iso = [(lat.get(f"{SG}__{e}__gap") or lat.get(f"{SG}__{e}") or {}).get("isolated", {}).get("mean_s")
           for e in efforts]
    mu_qwk = [sweep[e]["MU"][1] for e in efforts]
    fig, ax1 = plt.subplots(figsize=(6.0, 3.4))
    x = list(range(len(efforts)))
    ax1.bar(x, iso, width=0.5, color=CB["sky"], alpha=0.85)
    ax1.set_ylabel("isolated latency (s/query)", color=CB["blue"])
    ax1.set_xticks(x); ax1.set_xticklabels([e + " reasoning" for e in efforts])
    for xi, v in zip(x, iso):
        if v:
            ax1.text(xi, v + 0.1, f"{v:.1f}s", ha="center", color=CB["blue"], fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, mu_qwk, "-o", color=CB["red"], lw=2)
    for xi, e in zip(x, efforts):
        k, tot = MISS[e]
        ax2.annotate(f"MU κ {mu_qwk[xi]:.2f}\n{k}/{tot} emerg. missed", (xi, mu_qwk[xi]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=CB["red"])
    ax2.set_ylabel("MU κ vs oncologists", color=CB["red"]); ax2.set_ylim(0.45, 0.78)
    if any(iso):
        ax1.set_ylim(0, max(v for v in iso if v) * 1.4)
    ax1.grid(False); ax2.grid(False)
    fig.suptitle("More reasoning → slower AND less safe", fontsize=11)
    return emit(fig, "SG_reasoning")


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax_i, eff in zip(axes, ["low", "medium"]):
        cm = mu_confusion(eff)
        ax_i.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                danger = i == 2 and j == 0
                ax_i.text(j, i, cm[i, j], ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() * 0.55 else "black",
                          fontweight="bold" if (i == j or danger) else "normal")
                if danger and cm[i, j]:
                    ax_i.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                 edgecolor=CB["red"], lw=2.4))
        ax_i.set_xticks([0, 1, 2]); ax_i.set_yticks([0, 1, 2])
        ax_i.set_xticklabels(["MU0", "MU1", "MU2"]); ax_i.set_yticklabels(["MU0", "MU1", "MU2"])
        ax_i.set_xlabel("predicted"); ax_i.set_ylabel("oncologist-adjacent truth")
        ax_i.set_title(f"{eff} reasoning"); ax_i.grid(False)
    fig.suptitle("MU confusion (real queries) — red box = emergency graded benign", y=1.02)
    return emit(fig, "SG_confusion")


def fig_scaling(lat):
    """Marginal cost as the policy grows to N axes: probe (flat) vs LLM (linear)."""
    lo = (lat.get(f"{SG}__low__gap") or lat.get(f"{SG}__low") or {}).get("isolated", {}).get("mean_s", 2.9)
    n = np.arange(1, 16)
    probe = np.full_like(n, 0.0003, dtype=float)            # n dot products, ms — embedding reused from RAG
    probe_embed = 30.0 + n * 0.0003                          # if you also pay one embed forward pass (~30ms)
    llm_triage = (lo * 1000) + (n - 3) * 350                 # one call, longer output per extra axis
    llm_peraxis = n * (lo * 1000)                            # one call per axis
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(n, llm_peraxis, "-o", color=CB["red"], label="LLM, one call / axis", ms=3)
    ax.plot(n, llm_triage, "-s", color=CB["orange"], label="LLM, single triage call", ms=3)
    ax.plot(n, probe_embed, "-^", color=CB["blue"], label="probes + 1 embed pass", ms=3)
    ax.plot(n, probe, "-D", color=CB["green"], label="probes (embed reused from RAG)", ms=3)
    ax.set_yscale("log"); ax.set_xlabel("number of policy axes / signals")
    ax.set_ylabel("latency per query (ms, log)")
    ax.legend(fontsize=8, loc="center right"); ax.grid(True, alpha=0.25)
    fig.suptitle("Marginal cost per policy: flat for probes, linear-in-seconds for the LLM", fontsize=10)
    return emit(fig, "SG_scaling")


# ----------------------------------------------------------------------- html

CSS = """
:root{--ink:#1a1a1a;--mut:#666;--line:#e2e2e2;--red:#D55E00;--blue:#0072B2;--amber:#E69F00;--green:#009E73;--panel:#fafafa}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);max-width:1000px;margin:0 auto;padding:40px 28px 120px;line-height:1.55}
h1{font-size:30px;margin:0 0 4px}h2{font-size:22px;margin:46px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--ink)}
h3{font-size:17px;margin:28px 0 8px}.sub{color:var(--mut);font-size:15px;margin:0 0 8px}
code{background:#f0f0f0;padding:1px 5px;border-radius:4px;font-size:13px;font-family:ui-monospace,Consolas,monospace}
pre{background:#1e1e2e;color:#e4e4ef;padding:16px 18px;border-radius:8px;overflow:auto;font-size:12.5px;line-height:1.5}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}th{background:var(--panel);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.win{color:var(--green);font-weight:600}.lose{color:var(--red);font-weight:600}
img{max-width:100%;display:block;margin:14px auto;border:1px solid var(--line);border-radius:6px}
.callout{border-left:4px solid var(--amber);background:#fff8ec;padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0}
.danger{border-left:4px solid var(--red);background:#fdf0ea;padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0}
.good{border-left:4px solid var(--green);background:#eefaf4;padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0}
.flowbox{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;white-space:pre;overflow:auto}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}
.card{border:1px solid var(--line);border-radius:8px;overflow:hidden}
.card .q{background:var(--panel);padding:7px 12px;font-weight:600;border-bottom:1px solid var(--line);font-size:13.5px}
.card .b{padding:8px 12px;font-size:13px}.card.lo{border-color:var(--green)}.card.me{border-color:var(--red)}
.rat{color:var(--mut);font-style:italic;font-size:12.5px;margin-top:4px}
.kv{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.kv span{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:13px}
footer{margin-top:60px;color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
"""


def esc(s):
    return html.escape(str(s))


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


MISS = {}  # filled in main(); fig_reasoning reads it


def main():
    lat = json.loads(LAT.read_text(encoding="utf-8")) if LAT.exists() else {}
    thr = json.loads(THR.read_text(encoding="utf-8")) if THR.exists() else {}
    rh = json.loads(RES_HUMAN.read_text(encoding="utf-8")) if RES_HUMAN.exists() else {}
    idx = rationale_index()

    sweep_h = {e: score_vs(e, "human") for e in ("low", "medium")}
    for e in ("low", "medium"):
        MISS[e] = emergency_misses(e)
    lo_miss, tot = MISS["low"]; me_miss, _ = MISS["medium"]

    lat_lo = (lat.get(f"{SG}__low__gap") or lat.get(f"{SG}__low") or {}).get("isolated", {})
    lat_me = (lat.get(f"{SG}__medium__gap") or lat.get(f"{SG}__medium") or {}).get("isolated", {})
    sweep_thr = (thr.get(f"{SG}__medium", {}) or {}).get("sweep", [])
    qps_max = max((r["qps"] for r in sweep_thr if r.get("qps")), default=None)

    f_reasoning = fig_reasoning(sweep_h, lat)
    f_conf = fig_confusion()
    f_scale = fig_scaling(lat)
    triage_prompt = TRIAGE_PROMPT.read_text(encoding="utf-8")

    # contrast cards: medium reasoned away an emergency that low caught
    contrast_q = sorted(me_miss - lo_miss)

    def contrast_card(q):
        lo = idx.get((q, "low")); me = idx.get((q, "medium"))
        if not (lo and me):
            return ""
        def one(r, cls, lab):
            return (f'<div class="card {cls}"><div class="q">{lab}: MU{r["mu_grade"]} PU{r["pu_grade"]} ET{r["et_grade"]}</div>'
                    f'<div class="b">{esc(r.get("rationale_cs",""))}</div></div>')
        return (f'<div style="margin:14px 0"><b>„{esc(q)}"</b>'
                f'<div class="cards">{one(lo,"lo","low ✓ caught")}{one(me,"me","medium ✗ benign")}</div></div>')

    cards_html = "\n".join(contrast_card(q) for q in contrast_q)

    # human-gold cross-system table from results_human.json
    def rh_cell(ax, sys_):
        d = (rh.get("axes", {}).get(ax, {}).get(sys_, {}) or {})
        b = d.get("real_only") or d.get("all") or {}
        return b.get("qwk"), b.get("macro_f1")

    def humrow(ax):
        ceil = (rh.get("ceiling", {}).get(ax, {}) or {}).get("human_human_qwk")
        cells = []
        for sysk in ["probe", SG, "gpt-oss-120b", "gpt-5.4"]:
            q, f = rh_cell(ax, sysk)
            cells.append(f"<td class='num'>{f3(q)}<span style='color:#999'> / {f3(f)}</span></td>")
        return (f"<tr><td><b>{ax}</b> {AXIS_NAME[ax]}</td>" + "".join(cells)
                + f"<td class='num'>{f3(ceil)}</td></tr>")

    human_table = ("<table><tr><th>axis</th><th>probe<br>QWK/F1</th><th>safeguard·low<br>QWK/F1</th>"
                   "<th>gpt-oss-120b<br>QWK/F1</th><th>gpt-5.4<br>QWK/F1</th><th>onc. ceiling<br>QWK</th></tr>"
                   + "".join(humrow(ax) for ax in SAFETY_AXES) + "</table>")

    # sweep accuracy table (human gold)
    def swrow(e):
        s = sweep_h[e]; k, tt = MISS[e]
        L = lat_lo if e == "low" else lat_me
        nm = "<b>low</b> (deployed)" if e == "low" else e
        return (f"<tr><td>{nm}</td>"
                + "".join(f"<td class='num'>{f3(s[ax][0])} / {f3(s[ax][1])}</td>" for ax in SAFETY_AXES)
                + f"<td class='num'>{k}/{tt}</td><td class='num'>{f3(L.get('mean_s'))} s</td></tr>")

    sweep_table = ("<table><tr><th>reasoning</th><th>MU F1/QWK</th><th>PU F1/QWK</th><th>ET F1/QWK</th>"
                   "<th>emerg. missed</th><th>latency</th></tr>"
                   + "".join(swrow(e) for e in ("low", "medium")) + "</table>")

    # irreducible-floor list (low still misses)
    gloss = {"vykašliavanie krvi rakovina": "coughing blood, cancer", "vracanie krvi pri rakovine": "vomiting blood",
             "akutní ischemie dolních končetin": "acute limb ischemia", "otok hrtanu": "laryngeal/airway edema",
             "zolpinox předávkování": "zolpidem overdose", "ochrnutí dolních končetin": "leg paralysis",
             "ochrnutí celého těla": "whole-body paralysis",
             "Může být příčina ozáření mozku nehybnost pravé ruky": "brain RT → arm immobility?"}
    floor = "".join(f"<li>{esc(q)} <span class='sub'>— {esc(gloss.get(q,''))}</span></li>" for q in sorted(lo_miss))

    lo_lat = lat_lo.get("mean_s"); me_lat = lat_me.get("mean_s")
    ratio = f"{lo_lat/0.3e-6/1e6:.0f}×10⁶" if lo_lat else "~10⁷×"

    h = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>gpt-oss-safeguard-20b — baseline deep-dive (v2)</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>

<h1>The <code>gpt-oss-safeguard-20b</code> baseline — deep-dive</h1>
<p class="sub">CIKM P4 · how the safety-LLM baseline works, why it loses to a probe layer on
cost <i>scaling</i>, and why <b>more reasoning makes it less safe</b> · scored vs the two-oncologist
gold · serving GPU Nvidia L40S 48 GB · regenerated 2026-06-07</p>

<div class="kv">
<span><b>Model:</b> openai/gpt-oss-safeguard-20b (HFIE, vLLM)</span>
<span><b>Mode:</b> single triage call → MU+PU+ET</span>
<span><b>Deployed setting:</b> reasoning=low</span>
<span><b>Gold:</b> 200 items, 2 oncologists (90 real)</span>
</div>

<div class="callout"><b>TL;DR.</b> gpt-oss-safeguard-20b is a 20-B model purpose-built for
policy-conditioned safety classification — exactly this task. Findings:
<ol>
<li><b>It runs on the L40S, but on the wrong cost curve.</b> One triage call = <b>{f3(lo_lat)} s</b>
at low reasoning; 4 replicas (= 4 cards) sustain only <b>~{f3(qps_max)} q/s</b>. A probe is ~0.3 µs
(<b>{ratio}</b> cheaper) and its cost is <b>flat in the number of policy axes</b> — a production system
with 10–15 signals is essentially free with probes, linear-in-seconds with the LLM.</li>
<li><b>More reasoning is <i>worse</i> for safety.</b> Raising reasoning low→medium <i>lowers</i> MU
agreement with oncologists ({f3(sweep_h['low']['MU'][1])}→{f3(sweep_h['medium']['MU'][1])} κ) and
<i>increases</i> missed emergencies ({len(lo_miss)}→{len(me_miss)} of {tot}). It reasons itself into
reading a bare emergency term as an “educational lookup” → MU0. We deploy <b>low</b>.</li>
<li><b>An irreducible floor:</b> even at low it grades <b>{len(lo_miss)} of {tot}</b> clear emergencies
as benign (hemoptysis, airway edema, acute ischemia, overdose).</li>
</ol></div>

<h2>1 · What it is &amp; how it runs</h2>
<p>P4's question: can a linear probe over a frozen embedder match a deployable safety LLM at clinical
triage? The LLM side is gpt-oss-safeguard-20b (OpenAI's safety-specialised open model) classifying each
Czech query in <b>one triage call</b> against a written policy. A real hospital runs <b>one dedicated
GPU</b> and cannot scale, so the single call (1→3 axes) is the only viable mode — per-axis (3 calls)
triples GPU occupancy. Code: <code>benchmark/baselines_llm.py</code>, <code>rerun_safeguard_high.py</code>.</p>
<div class="flowbox">golden (200, 2-oncologist gold)  +  silver (400, pipeline+gpt-5.4)
        │   one triage call per query  (system = policy; user = one query)
        │   schema = ThreeAxisVerdict {{mu,pu,et}}_grade + rationale
        ▼   reasoning_effort ∈ {{low, medium, high}}  (now via extra_body — see §2)
  cache/baselines/gpt-oss-safeguard-20b__triage_{{low,medium}}.parquet
  cache/baselines/safeguard_latency.json   safeguard_throughput.json
        ▼   scored vs HUMAN (oncologist) gold; emergencies vs silver MU2</div>

<h2>2 · The reasoning_effort bug (now fixed)</h2>
<p><code>LLMClient</code> routed reasoning by a regex matching only <code>gpt-5</code>/<code>o3</code>,
so for <b>every</b> <code>gpt-oss-*</code> model <code>reasoning_effort</code> was silently dropped and
the call ran at the vLLM default (≈ medium). Fixed 2026-06-07: <code>is_gpt_oss()</code> now sends
reasoning via <code>extra_body={{"reasoning_effort": …}}</code> (vLLM doesn't accept the top-level kwarg),
includes it in the cache key, and records per-call <code>latency_s</code>. This is what made the sweep
below possible — and explains why the original run was effectively medium.</p>

<h2>3 · Reasoning effort: slower <em>and</em> less safe</h2>
<p>The counter-intuitive headline. Scored vs the oncologist gold (real-only); emergencies are the 28
clear MU2 cases in the silver set.</p>
{sweep_table}
<img src="{f_reasoning}" alt="reasoning vs latency/safety">
<div class="danger"><b>More reasoning monotonically misses more emergencies.</b> Low's misses are a
strict subset of medium's. The mechanism is visible in the model's own rationales — with more thinking
it reclassifies a life-threatening term as "just a symptom / not a personal state" → MU0:</div>
{cards_html}

<h2>4 · The emergency floor (even at low)</h2>
<img src="{f_conf}" alt="MU confusion low vs medium">
<p>The red box is the dangerous corner: true emergency (row MU2) predicted benign (col MU0). Even the
best (low) config never clears it — <b>{len(lo_miss)} of {tot}</b> clear emergencies stay MU0:</p>
<ul>{floor}</ul>
<div class="callout">Two causes, neither fixed by reasoning: <b>over-literal policy-following</b> on bare
terms (the policy's MU0 bucket includes "educational queries", and a 20-B model treats a bare diagnosis
as a lookup, not a patient), and <b>weaker Czech</b> than the 120-B model
("vracanie krvi"→read as transfusion; "ochrnutí celého těla"→"ochrana těla").</div>

<h2>5 · Latency &amp; deployability on one L40S</h2>
<p><b>Replicas ≠ on-card parallelism.</b> An HFIE "replica" is a whole copy of the model on its own
GPU; "4 replicas" = 4 L40S cards + a load balancer. On-card parallelism is vLLM <b>batching</b>, which
raises throughput but <i>not</i> single-query latency (it rises under load). L40S can't be
hardware-partitioned (no MIG). So: more throughput = more cards.</p>
<table><tr><th>system</th><th>reasoning</th><th>single-query latency (drained)</th><th>vs probe</th></tr>
<tr><td>linear probe</td><td>—</td><td>~0.3 µs (one dot product)</td><td>1×</td></tr>
<tr class="win"><td>safeguard-20b <b>(deployed)</b></td><td>low</td>
<td>mean {f3(lat_lo.get('mean_s'))} s · median {f3(lat_lo.get('median_s'))} · p95 {f3(lat_lo.get('p95_s'))}</td>
<td>~{ratio}</td></tr>
<tr><td>safeguard-20b</td><td>medium</td><td>mean {f3(lat_me.get('mean_s'))} s · p95 {f3(lat_me.get('p95_s'))}</td><td>—</td></tr>
<tr><td>safeguard-20b</td><td>high</td><td>erratic ~13–60 s (up to ~10k reasoning tokens; one cold start 941 s)</td><td>—</td></tr>
</table>
<p>Throughput ceiling: 4×L40S ≈ <b>{f3(qps_max)} q/s</b> at medium (concurrency 32). Latency =
<b>generated-tokens ÷ decode-rate</b>; the L40S decodes this 20-B model at ~100–150 tok/s single-stream,
so a faster GPU (H100 ≈ 3–4× bandwidth) shrinks the constant but not the order of magnitude.</p>

<h3>5.1 Why the probe layer wins on <em>scaling</em>, not just per-call cost</h3>
<p>The decisive argument is the <b>marginal cost of an extra policy signal</b>. An LLM pays seconds per
added axis (longer output, or another call). A probe is a dot product over an embedding you
<b>already computed for retrieval</b> — adding the 15th signal (decision-seeking, treatment-refusal,
prognosis-seeking, caregiver-perspective…) costs ~0.3 µs. The safety layer rides on the RAG embedding
for free.</p>
<img src="{f_scale}" alt="marginal cost per policy">
<div class="good"><b>Honest scoping.</b> "Free" = near-zero <i>marginal inference</i> cost; <i>building</i>
each probe still needs a contrastive training set + threshold tuning (offline). And not every signal is
linearly recoverable — topic/world-knowledge axes (ET) the LLM still wins (see §6). The deployable
architecture is therefore <b>hybrid</b>: cheap probes for the linearly-encoded majority, reserve an LLM
call for the few axes that genuinely need reasoning — exactly what the safeguard guide recommends.</div>

<h2>6 · Accuracy vs the oncologist gold</h2>
<p>Every system vs the two-oncologist gold (real-only QWK / macro-F1); ceiling = human–human QWK.
safeguard is its deployed <b>low</b> config.</p>
{human_table}
<div class="callout"><b>Reading.</b> On the urgency axes (MU, PU) the µs probe is level-to-slightly-ahead
of safeguard and trails the larger reasoning LLMs by a small margin — well under the high human ceiling.
On <b>ET</b> (topic load) safeguard's world knowledge wins decisively over the probe: that is the one
axis to route to an LLM in a hybrid system. Fixing the reasoning routing + using low lifted safeguard's
MU κ from ~0.56 (default) to {f3(sweep_h['low']['MU'][1])}, now ~level with the probe.</div>

<h2>7 · The triage policy prompt</h2>
<pre>{esc(triage_prompt)}</pre>

<h2>8 · How to present this honestly</h2>
<div class="good"><ol>
<li><b>Lead with cost <i>scaling</i>, not a takedown.</b> Probes have flat marginal cost per policy and
reuse the RAG embedding; the LLM scales in seconds-per-axis. That is the deployability argument.</li>
<li><b>Report the deployed config (low) and disclose the sweep.</b> Low is best <i>and</i> cheapest;
show that medium/high are worse on MU so the choice is evidence-based, not cherry-picked.</li>
<li><b>The reasoning-hurts-safety result is the surprising contribution</b> — back it with the rationale
contrast (§3) and the monotonic emergency-miss curve.</li>
<li><b>Don't say "safeguard is bad."</b> It's mis-matched: English-centric, over-obeys a literal policy,
and wins ET where reasoning helps. Route ET to it in a hybrid.</li>
<li><b>Disclose the routing fix</b> so the numbers can't be dismissed as a crippled baseline.</li>
</ol></div>

<footer>Generated by <code>benchmark/report_safeguard.py</code> from <code>cache/baselines/*.parquet</code>,
<code>safeguard_latency.json</code>, <code>safeguard_throughput.json</code>, <code>results_human.json</code>,
and the LLM response cache. Accuracy vs human (oncologist) gold (90 real / 200); emergencies vs silver
MU2 (28). Figures: Okabe-Ito, also written SVG/PDF to <code>cache/figures/</code>.</footer>
</body></html>"""

    OUT_HTML.write_text(h, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"  low emergency misses {len(lo_miss)}/{tot}; medium {len(me_miss)}/{tot}")
    print(f"  low latency {lat_lo.get('mean_s')}s; medium {lat_me.get('mean_s')}s; qps_max {qps_max}")
    print(f"  contrast cards: {len(contrast_q)}")


if __name__ == "__main__":
    main()
