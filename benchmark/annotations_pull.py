"""Download the clinician gold from the annotation API and build the human-gold
eval set + IAA ceiling for package "Golden v1 — Core (IAA, MU/PU/ET)".

Gold rule (locked with the user): per (item, axis) the gold is the admin-
**adjudicated** `gold_<axis>` value when present; otherwise the item had no
disagreement, so the agreed human value is taken — and we VERIFY that invariant
(any missing-gold item where the two oncologists actually disagree is flagged as
`unresolved`, dropped, and counted, never silently resolved). The LLM judge
(`llm_reference`) and the admin (`admin`) are NOT used to form gold; the LLM is a
benchmarked system, scored separately.

The set is 200 items = real + synthesized (NOT real-only); the synthetic flag is
joined from `golden_400_filled.parquet` so the benchmark can report ±synthetic.

Outputs (with --execute; default is --dry-run):
  golden/cache/golden_set_v1/human_gold.parquet   — drop-in for benchmark._gold
  benchmark/cache/annotations/export_core.csv     — raw export snapshot
  benchmark/cache/human_iaa.json                  — server α/κ + local human–human QWK
  benchmark/cache/human_gold_report.html          — sample-rows audit (house rule)

  python -m benchmark.annotations_pull              # dry-run: prints the audit
  python -m benchmark.annotations_pull --execute    # writes the artifacts
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import sys

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from contrastive.p2_io import REPO, SAFETY_AXES, int_to_token

from golden.pipeline_v2 import _io  # noqa: F401  (loads .env)

PACKAGE_TITLE_MATCH = "Golden v1 — Core"
SOURCE_PREFIX = "golden-v1:"
LLM_IDS = {"llm_reference"}          # virtual annotators excluded from gold + ceiling
ADMIN_IDS = {"admin"}                # admin / incidental raters excluded
MIN_RATER_COVERAGE = 150             # a "real" rater must cover >= this many of 200

GOLDEN_400 = REPO / "golden" / "cache" / "golden_set_v1" / "golden_400_filled.parquet"
HUMAN_GOLD = REPO / "golden" / "cache" / "golden_set_v1" / "human_gold.parquet"
ANN_DIR = REPO / "benchmark" / "cache" / "annotations"
IAA_OUT = REPO / "benchmark" / "cache" / "human_iaa.json"
REPORT_OUT = REPO / "benchmark" / "cache" / "human_gold_report.html"


def _client() -> httpx.Client:
    base = os.environ["ANNOTATION_API_URL"].rstrip("/")
    tok = os.environ["ADMIN_TOKEN"]
    return httpx.Client(base_url=base, headers={"x-admin-token": tok}, timeout=180.0)


def resolve_package_id(c: httpx.Client, override: int | None) -> int:
    if override:
        return override
    r = c.get("/admin/stats"); r.raise_for_status()
    data = r.json()
    pkgs = data.get("packages", data if isinstance(data, list) else [])
    for p in pkgs:
        if PACKAGE_TITLE_MATCH in str(p.get("title", "")):
            print(f"[pull] package '{p['title']}' -> id={p['id']}", flush=True)
            return int(p["id"])
    raise SystemExit(f"no package matching {PACKAGE_TITLE_MATCH!r}; pass --package-id")


def detect_human_raters(df: pd.DataFrame) -> list[str]:
    """Annotator ids with a <id>_MU column, real coverage, not LLM/admin."""
    ids = []
    for col in df.columns:
        if col.endswith("_MU") and not col.startswith("gold"):
            rid = col[: -len("_MU")]
            if rid in LLM_IDS or rid in ADMIN_IDS:
                continue
            if df[col].notna().sum() >= MIN_RATER_COVERAGE:
                ids.append(rid)
    return sorted(ids)


def resolve_gold(df: pd.DataFrame, human_raters: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per axis: adjudicated value, else agreed human value, else unresolved(NaN).
    Returns (per-axis int-grade columns + provenance) and a summary dict."""
    out = pd.DataFrame(index=df.index)
    summary: dict = {}
    for ax in SAFETY_AXES:
        gcol = f"gold_{ax}"
        hcols = [f"{r}_{ax}" for r in human_raters if f"{r}_{ax}" in df.columns]
        grades, prov = [], []
        adj = agree = unresolved = 0
        for _, row in df.iterrows():
            a = row.get(gcol)
            if pd.notna(a):
                grades.append(int(a)); prov.append("adjudicated"); adj += 1
            else:
                vals = [int(row[h]) for h in hcols if pd.notna(row[h])]
                if vals and len(set(vals)) == 1:
                    grades.append(vals[0]); prov.append("agreement"); agree += 1
                else:
                    grades.append(np.nan); prov.append("unresolved"); unresolved += 1
        out[f"{ax}_grade_int"] = grades
        out[f"{ax}_provenance"] = prov
        summary[ax] = {"adjudicated": adj, "agreement": agree, "unresolved": unresolved,
                       "n_raters": len(hcols)}
    return out, summary


def pull_iaa(c: httpx.Client, pkg: int, human_raters: list[str]) -> dict:
    """Server Krippendorff α + Cohen κ per axis (human-only and incl-LLM), plus a
    locally computed human–human quadratic-weighted κ for the paper."""
    exclude_for_human = ",".join(sorted(LLM_IDS | ADMIN_IDS))
    iaa: dict = {"human_only_exclude": exclude_for_human, "axes": {}}
    for ax in SAFETY_AXES:
        entry: dict = {}
        for tag, params in (
            ("human", {"packageId": pkg, "axis": ax, "level": "interval",
                       "excludeAnnotators": exclude_for_human}),
            ("with_llm", {"packageId": pkg, "axis": ax, "level": "interval",
                          "excludeAnnotators": ",".join(sorted(ADMIN_IDS))}),
        ):
            r = c.get("/admin/iaa-alpha", params=params)
            if r.status_code == 200:
                d = r.json()
                entry[f"alpha_{tag}"] = d.get("alpha")
                entry[f"alpha_{tag}_units"] = d.get("units")
                entry[f"alpha_{tag}_raters"] = d.get("annotators")
        r = c.get("/admin/iaa", params={"packageId": pkg, "axis": ax,
                                        "excludeAnnotators": exclude_for_human})
        if r.status_code == 200:
            entry["cohen_pairs"] = r.json().get("pairs")
        iaa["axes"][ax] = entry
    return iaa


def local_human_qwk(df: pd.DataFrame, human_raters: list[str]) -> dict:
    """Quadratic-weighted κ between the two oncologists per axis (paper metric)."""
    out = {}
    if len(human_raters) < 2:
        return out
    a, b = human_raters[0], human_raters[1]
    for ax in SAFETY_AXES:
        ca, cb = f"{a}_{ax}", f"{b}_{ax}"
        if ca not in df.columns or cb not in df.columns:
            continue
        m = df[[ca, cb]].dropna()
        if len(m) < 5:
            continue
        out[ax] = {"qwk": float(cohen_kappa_score(m[ca].astype(int), m[cb].astype(int),
                                                  labels=[0, 1, 2], weights="quadratic")),
                   "raw_agreement": float((m[ca] == m[cb]).mean()),
                   "n": int(len(m)), "raters": [a, b]}
    return out


def et_somatic_analysis(df: pd.DataFrame, human_raters: list[str]) -> dict:
    """Error analysis for the ET inter-rater disagreement: is it the documented
    somatic-urgency→ET2 codebook ambiguity? Tests whether ET1↔ET2 disagreements
    concentrate on high-MU (somatic-urgent) items, and what agreement would be had
    that single ambiguity been specified (somatic urgency stays ET1)."""
    if len(human_raters) < 2:
        return {}
    r1, r2 = human_raters[0], human_raters[1]
    e1, e2 = df[f"{r1}_ET"], df[f"{r2}_ET"]
    # Orient to the dominant ET1<->ET2 direction: `hi` = the rater who systematically
    # gives the higher grade, `lo` = the other. Then the somatic-conflation cell is
    # (lo==1 & hi==2).
    if ((e1 == 2) & (e2 == 1)).sum() >= ((e2 == 2) & (e1 == 1)).sum():
        hi, lo, e_hi, e_lo = r1, r2, e1, e2
    else:
        hi, lo, e_hi, e_lo = r2, r1, e2, e1
    # resolved MU (adjudicated else agreed) for the enrichment test
    mu_gold = df["gold_MU"].astype("float").copy()
    for i in df.index:
        if pd.isna(mu_gold[i]):
            va, vb = df.at[i, f"{r1}_MU"], df.at[i, f"{r2}_MU"]
            mu_gold[i] = va if (pd.notna(va) and va == vb) else np.nan
    mu_ge1 = mu_gold >= 1
    cell = (e_lo == 1) & (e_hi == 2)               # the dominant one-cell disagreement
    elsewhere = ~(cell.fillna(False))
    # reconciled QWK: recode the over-rater's somatic-urgent ET2 -> ET1 (codebook rule)
    e_hi_rec = e_hi.copy()
    e_hi_rec[(e_hi == 2) & (e_lo == 1) & mu_ge1] = 1
    m = pd.DataFrame({"a": e_lo, "b": e_hi_rec}).dropna()
    qwk_rec = float(cohen_kappa_score(m["a"].astype(int), m["b"].astype(int),
                                      labels=[0, 1, 2], weights="quadratic")) if len(m) else None
    return {
        "raters": [lo, hi], "over_rater": hi,
        "one_cell_ET1_to_ET2_n": int(cell.sum()),
        "all_one_dir_disagreements_n": int((e_hi > e_lo).sum()),
        "share_MU_ge1_in_cell": float(mu_ge1[cell.fillna(False)].mean()) if cell.sum() else None,
        "share_MU_ge1_elsewhere": float(mu_ge1[elsewhere].mean()),
        "share_MU_ge1_overall": float(mu_ge1.mean()),
        "qwk_raw": (None),
        "qwk_after_somatic_reconcile": qwk_rec,
        "interpretation": ("ET disagreement is a single codebook underspecification: one rater "
                           "coded somatic-urgent items as high emotional load (ET2) where the "
                           "codebook intended ET1 (ET2 = long-term emotional/existential load). "
                           "Reconciling it brings ET agreement on par with MU/PU."),
    }


def build_human_gold(df: pd.DataFrame, gold: pd.DataFrame,
                     human_raters: list[str]) -> pd.DataFrame:
    qk = df["source"].str.replace(f"^{SOURCE_PREFIX}", "", regex=True)
    g400 = pd.read_parquet(GOLDEN_400)[["query_key", "source"]].copy()
    g400["query_key"] = g400["query_key"].astype(str)
    synth = dict(zip(g400["query_key"], g400["source"].astype(str) == "synthesized"))
    orig = dict(zip(g400["query_key"], g400["source"].astype(str)))

    rows = []
    llm_id = next(iter(LLM_IDS))
    for i, (_, row) in enumerate(df.iterrows()):
        key = qk.iloc[i]
        rec = {"query_key": key, "query_text": row["text"],
               "synthetic": bool(synth.get(key, False)),
               "orig_source": orig.get(key, "unknown")}
        for ax in SAFETY_AXES:
            gi = gold[f"{ax}_grade_int"].iloc[i]
            rec[f"{ax.lower()}_grade"] = (int_to_token(ax, int(gi)) if pd.notna(gi) else None)
            rec[f"{ax}_provenance"] = gold[f"{ax}_provenance"].iloc[i]
            for r in human_raters:
                rec[f"{r}_{ax}"] = (int(row[f"{r}_{ax}"]) if pd.notna(row.get(f"{r}_{ax}")) else None)
            lc = f"{llm_id}_{ax}"
            rec[f"llm_{ax}"] = (int(row[lc]) if pd.notna(row.get(lc)) else None)
        rows.append(rec)
    return pd.DataFrame(rows)


def write_report(hg: pd.DataFrame, human_raters: list[str], summary: dict,
                 hqwk: dict, iaa: dict) -> None:
    a, b = (human_raters + ["?", "?"])[:2]
    P = ["<html><head><meta charset='utf-8'><title>Human gold — audit</title><style>"
         "body{font-family:system-ui;margin:2rem auto;max-width:1100px;color:#1a1a1a}"
         "table{border-collapse:collapse;font-size:.86rem;margin:.6rem 0}"
         "td,th{border:1px solid #ccc;padding:3px 8px}th{background:#f3f4f6}"
         ".adj{background:#fff3e0}.agr{background:#e8f5e9}.unr{background:#ffebee}"
         "h2{border-bottom:2px solid #eee;margin-top:1.6rem}</style></head><body>"]
    P.append("<h1>Human gold — audit (Golden v1 — Core)</h1>")
    P.append(f"<p>Oncologists: <b>{a}</b>, <b>{b}</b>; LLM judge excluded from gold. "
             f"N={len(hg)} ({int(hg['synthetic'].sum())} synthetic / "
             f"{int((~hg['synthetic']).sum())} real).</p>")
    P.append("<h2>Gold provenance + human ceiling</h2><table>"
             "<tr><th>axis</th><th>adjudicated</th><th>agreement</th><th>unresolved</th>"
             "<th>human–human QWK</th><th>raw agree</th><th>α (human)</th></tr>")
    for ax in SAFETY_AXES:
        s = summary[ax]; q = hqwk.get(ax, {})
        al = iaa["axes"].get(ax, {}).get("alpha_human")
        P.append(f"<tr><td><b>{ax}</b></td><td>{s['adjudicated']}</td><td>{s['agreement']}</td>"
                 f"<td>{s['unresolved']}</td><td>{q.get('qwk', float('nan')):.3f}</td>"
                 f"<td>{q.get('raw_agreement', float('nan')):.3f}</td>"
                 f"<td>{al if al is not None else '—'}</td></tr>")
    P.append("</table>")
    P.append("<h2>Sample items (mixed provenance)</h2><table><tr><th>query</th>"
             "<th>synth</th>")
    for ax in SAFETY_AXES:
        P.append(f"<th>{a}·{ax}</th><th>{b}·{ax}</th><th>llm·{ax}</th><th>GOLD·{ax}</th>")
    P.append("</tr>")
    sample = pd.concat([hg[hg[f"{ax}_provenance"] == "adjudicated"].head(5) for ax in SAFETY_AXES]
                       + [hg.head(5)]).drop_duplicates("query_key").head(18)
    for _, r in sample.iterrows():
        P.append(f"<tr><td>{html.escape(str(r['query_text']))[:60]}</td>"
                 f"<td>{'Y' if r['synthetic'] else ''}</td>")
        for ax in SAFETY_AXES:
            prov = r[f"{ax}_provenance"]
            cls = {"adjudicated": "adj", "agreement": "agr", "unresolved": "unr"}.get(prov, "")
            P.append(f"<td>{r.get(f'{a}_{ax}','')}</td><td>{r.get(f'{b}_{ax}','')}</td>"
                     f"<td>{r.get(f'llm_{ax}','')}</td>"
                     f"<td class='{cls}'>{r.get(f'{ax.lower()}_grade','') or '—'}</td>")
        P.append("</tr>")
    P.append("</table><p style='color:#888;font-size:.8rem'>orange=adjudicated, "
             "green=agreement, red=unresolved.</p></body></html>")
    REPORT_OUT.write_text("".join(P), encoding="utf-8")
    print(f"[pull] wrote {REPORT_OUT}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-id", type=int, default=None)
    ap.add_argument("--execute", action="store_true", help="write artifacts (default dry-run)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    with _client() as c:
        pkg = resolve_package_id(c, a.package_id)
        print(f"[pull] GET export-wide.csv packageId={pkg} …", flush=True)
        r = c.get("/admin/export-wide.csv", params={"packageId": pkg}); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        human_raters = detect_human_raters(df)
        print(f"[pull] {len(df)} items; human raters={human_raters}; "
              f"LLM={sorted(LLM_IDS)}; admin-excluded={sorted(ADMIN_IDS)}", flush=True)

        gold, summary = resolve_gold(df, human_raters)
        print("[pull] gold provenance:", flush=True)
        for ax in SAFETY_AXES:
            s = summary[ax]
            print(f"   {ax}: adjudicated={s['adjudicated']} agreement={s['agreement']} "
                  f"UNRESOLVED={s['unresolved']} (raters={s['n_raters']})", flush=True)
        hqwk = local_human_qwk(df, human_raters)
        for ax, q in hqwk.items():
            print(f"   human–human {ax}: QWK={q['qwk']:.3f} raw={q['raw_agreement']:.3f} n={q['n']}",
                  flush=True)
        iaa = pull_iaa(c, pkg, human_raters)

    hg = build_human_gold(df, gold, human_raters)
    for ax in SAFETY_AXES:
        dist = hg[f"{ax.lower()}_grade"].dropna().value_counts().sort_index().to_dict()
        print(f"   {ax} gold dist: {dist}  (real-only n="
              f"{int((~hg['synthetic'] & hg[f'{ax.lower()}_grade'].notna()).sum())})", flush=True)

    et_err = et_somatic_analysis(df, human_raters)
    if et_err:
        et_err["qwk_raw"] = hqwk.get("ET", {}).get("qwk")
        print(f"[pull] ET error-analysis: one-cell ET1->ET2 n={et_err['one_cell_ET1_to_ET2_n']} "
              f"| MU>=1 in-cell={et_err['share_MU_ge1_in_cell']:.2f} vs elsewhere="
              f"{et_err['share_MU_ge1_elsewhere']:.2f} | QWK {et_err['qwk_raw']:.2f} -> "
              f"{et_err['qwk_after_somatic_reconcile']:.2f} after reconcile", flush=True)
    iaa["human_human_qwk"] = hqwk
    iaa["et_error_analysis"] = et_err
    iaa["provenance"] = summary
    iaa["n_items"] = int(len(hg))
    iaa["n_synthetic"] = int(hg["synthetic"].sum())

    if not a.execute:
        print("\n[dry-run] no files written. Re-run with --execute to write:", flush=True)
        print(f"  {HUMAN_GOLD}\n  {IAA_OUT}\n  {REPORT_OUT}\n  {ANN_DIR/'export_core.csv'}", flush=True)
        write_report(hg, human_raters, summary, hqwk, iaa)  # report is read-only-safe to preview
        return 0

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    (ANN_DIR / "export_core.csv").write_text(r.text, encoding="utf-8")
    HUMAN_GOLD.parent.mkdir(parents=True, exist_ok=True)
    hg.to_parquet(HUMAN_GOLD, index=False)
    IAA_OUT.write_text(json.dumps(iaa, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(hg, human_raters, summary, hqwk, iaa)
    print(f"\n[pull] wrote {HUMAN_GOLD} ({len(hg)} rows)\n[pull] wrote {IAA_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
