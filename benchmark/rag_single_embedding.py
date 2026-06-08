"""Standard-RAG single-embedding probe test (gemini, no custom instruction-conditioning).

The headline gemini MU result uses PER-AXIS instruction-conditioning: the query is
re-embedded with an axis-specific instruction. In a real RAG system the query is
embedded ONCE for retrieval; re-encoding it per safety axis is a deployment cost.

This script asks: on gemini, can the three safety probes (MU/PU/ET) run on a SINGLE
pre-retrieval embedding instead? Two single-embedding candidates vs the conditioned
baseline, all scored on the clinician golden (real-only) from cached artifacts only
(no model calls):

  * none      - raw query embedding, NO instruction  -> the plain "embed once" vector
  * generic   - one shared generic instruction        -> still a single embedding
  * per_axis  - axis-specific instruction (3 embeddings) = the conditioned baseline

The probe stays axis-specific in every mode (three directions w_MU/w_PU/w_ET applied
to the SAME vector under `none`), so this measures exactly "move the conditioning from
the embedder to the probe", not "drop the per-axis signal".

    x:\\projects\\onkoradce\\embeddings\\.venv\\Scripts\\python.exe -m benchmark.rag_single_embedding
        -> benchmark/cache/rag_single_embedding.{json,html}
"""
from __future__ import annotations

import html
import json
import sys

from benchmark.human_benchmark import probe_system, probe_linear_system
from contrastive.p2_io import REPO, SAFETY_AXES

CACHE = REPO / "benchmark" / "cache"
PROBE_HUMAN = CACHE / "results_probe__human.json"
RESULTS_HUMAN = CACHE / "results_human.json"
OUT_JSON = CACHE / "rag_single_embedding.json"
OUT_HTML = CACHE / "rag_single_embedding.html"

EMBEDDER = "gemini"
MODES = ["none", "generic", "per_axis"]
BASELINE = "per_axis"                 # the conditioned mode we compare the single-embedding modes against
SINGLE_EMB = ["none", "generic"]      # "embed once" candidates
TIE = 0.05                            # QWK gap below this is "not meaningfully worse"

MODE_LABEL = {
    "none": "none (raw, no instruction)",
    "generic": "generic (one shared instruction)",
    "per_axis": "per_axis (conditioned baseline)",
}


def _rl(d: dict) -> dict:
    """Pull the real-only block (fallback to all)."""
    return (d.get("real_only") or d.get("all") or {}) if d else {}


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _ci(blk):
    lo, hi = blk.get("qwk_lo"), blk.get("qwk_hi")
    return f"[{lo:.2f},{hi:.2f}]" if lo is not None and hi is not None else ""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = json.loads(PROBE_HUMAN.read_text(encoding="utf-8"))
    thr = {(r["embedder"], r["mode"], r["axis"]): r["separability"]["thresholds"] for r in rows}
    rh = json.loads(RESULTS_HUMAN.read_text(encoding="utf-8")) if RESULTS_HUMAN.exists() else {"ceiling": {}}
    ceiling = {ax: (rh.get("ceiling", {}).get(ax, {}) or {}).get("human_human_qwk") for ax in SAFETY_AXES}

    out = {"embedder": EMBEDDER, "modes": MODES, "baseline": BASELINE, "tie": TIE,
           "mode_label": MODE_LABEL, "ceiling": ceiling, "axes": {}}

    print(f"=== Single pre-retrieval embedding test  (embedder={EMBEDDER}; clinician golden, real-only QWK) ===\n",
          flush=True)
    for ax in SAFETY_AXES:
        cells = {}
        for mode in MODES:
            if (EMBEDDER, mode, ax) not in thr:
                print(f"  [skip] {EMBEDDER}/{mode}/{ax}: no cached threshold", flush=True)
                continue
            cell = {"embedder": EMBEDDER, "mode": mode}
            dom = _rl(probe_system(ax, cell, thr))
            lin = _rl(probe_linear_system(ax, cell, "human"))
            cells[mode] = {"dom": dom, "linear": lin}

        base_dom = cells.get(BASELINE, {}).get("dom", {})
        base_q = base_dom.get("qwk")
        cap = ceiling.get(ax)
        for mode, c in cells.items():
            dq = c["dom"].get("qwk")
            lq = c["linear"].get("qwk")
            c["delta_dom_vs_baseline"] = (round(dq - base_q, 3) if dq is not None and base_q is not None else None)
            c["delta_linear_vs_baseline"] = (
                round(lq - cells.get(BASELINE, {}).get("linear", {}).get("qwk", float("nan")), 3)
                if lq is not None and cells.get(BASELINE, {}).get("linear", {}).get("qwk") is not None else None)
            lo = c["dom"].get("qwk_lo")
            # best single-embedding head per cell (max of DoM / linear), used for the verdict
            best_q = max([q for q in (dq, lq) if q is not None], default=None)
            c["best_qwk"] = best_q
            c["signal_real"] = (lo is not None and lo > 0)
            c["works_vs_baseline"] = (best_q is not None and base_q is not None and best_q >= base_q - TIE)
            c["within_human_band"] = (best_q is not None and cap is not None and best_q >= cap - TIE)

        # per-axis verdict: does ANY single-embedding mode hold up vs the conditioned baseline?
        ok = {m: cells[m] for m in SINGLE_EMB if m in cells}
        holds = [m for m, c in ok.items() if c["works_vs_baseline"] and c["dom"].get("qwk_lo", -1) is not None]
        verdict = {
            "holds_modes": holds,
            "best_single_mode": max(ok, key=lambda m: ok[m]["best_qwk"] or -1) if ok else None,
            "baseline_dom_qwk": base_q,
            "ceiling": cap,
        }
        out["axes"][ax] = {"cells": cells, "verdict": verdict}

        # console block
        capf = _fmt(cap, 2)
        print(f"[{ax}]  human ceiling QWK={capf}   baseline {BASELINE} DoM QWK={_fmt(base_q)}", flush=True)
        for mode in MODES:
            if mode not in cells:
                continue
            c = cells[mode]
            d, l = c["dom"], c["linear"]
            mark = ""
            if mode in SINGLE_EMB:
                mark = "  OK" if c["works_vs_baseline"] else "  below"
            print(f"   {MODE_LABEL[mode]:<34} DoM={_fmt(d.get('qwk'))} {_ci(d):<13} "
                  f"lin={_fmt(l.get('qwk'))} {_ci(l):<13} "
                  f"macroF1={_fmt(d.get('macro_f1'),2)} AUROC>=2={_fmt(d.get('auroc_ge2'),2)} "
                  f"n={d.get('n','?')}{mark}", flush=True)
        bsm = verdict["best_single_mode"]
        print(f"   -> single-embedding holds: {holds or 'NONE'} "
              f"(best single mode: {bsm})\n", flush=True)

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(out)
    print(f"[write] {OUT_JSON}\n[write] {OUT_HTML}", flush=True)
    return 0


def _write_html(out: dict) -> None:
    P = []
    A = P.append
    A("<!doctype html><meta charset='utf-8'><title>Single-embedding probe test</title>")
    A("<style>body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;"
      "margin:2rem auto;padding:0 1rem;color:#1a1a1a}"
      "table{border-collapse:collapse;width:100%;margin:.6rem 0 1.4rem;font-size:14px}"
      "th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:right}"
      "th:first-child,td:first-child{text-align:left}"
      "thead th{background:#f3f4f6}tr.base td{background:#fafafa;font-style:italic}"
      ".ok{color:#127a2b;font-weight:600}.no{color:#b00020;font-weight:600}"
      ".ci{color:#888;font-size:12px}h1{font-size:1.5rem}h2{font-size:1.15rem;margin-top:1.6rem}"
      ".v{background:#f7f9fc;border-left:3px solid #4063c4;padding:.5rem .8rem;margin:.4rem 0}"
      "small{color:#666}</style>")
    A("<h1>Standard-RAG single-embedding probe test — gemini</h1>")
    A("<p>Can the MU/PU/ET safety probes run on a <b>single pre-retrieval gemini embedding</b> "
      "(no custom per-axis instruction-conditioning) and still work? Scored on the clinician golden "
      "set, <b>real-only QWK</b>, from cached artifacts (no model calls). The probe stays axis-specific "
      "in every mode — three directions applied to the same vector under <code>none</code> — so this "
      "isolates moving the conditioning from the <i>embedder</i> (a re-encode) to the <i>probe</i> "
      "(a free dot product).</p>")
    A(f"<p><small>Baseline = <b>{BASELINE}</b> (the conditioned mode, 3 embeddings). "
      f"&quot;Holds&quot; = best single-embedding head QWK ≥ baseline − {TIE}. "
      "Ceiling = inter-oncologist QWK.</small></p>")

    for ax in SAFETY_AXES:
        blk = out["axes"].get(ax)
        if not blk:
            continue
        cells, verdict = blk["cells"], blk["verdict"]
        cap = verdict["ceiling"]
        A(f"<h2>{ax} &nbsp;<small>human ceiling QWK = {_fmt(cap,2)} · "
          f"baseline DoM QWK = {_fmt(verdict['baseline_dom_qwk'])}</small></h2>")
        A("<table><thead><tr><th>mode</th><th>DoM QWK</th><th>linear QWK</th>"
          "<th>Δ vs baseline (best)</th><th>macro-F1</th><th>AUROC≥2</th><th>n</th><th>verdict</th></tr></thead><tbody>")
        for mode in MODES:
            c = cells.get(mode)
            if not c:
                continue
            d, l = c["dom"], c["linear"]
            base_row = (mode == BASELINE)
            dq = f"{_fmt(d.get('qwk'))} <span class='ci'>{_ci(d)}</span>"
            lq = f"{_fmt(l.get('qwk'))} <span class='ci'>{_ci(l)}</span>"
            if base_row:
                delta = "—"; vcell = "<small>baseline</small>"
            else:
                bq = c.get("best_qwk"); bb = verdict["baseline_dom_qwk"]
                dv = (bq - bb) if (bq is not None and bb is not None) else None
                delta = (f"{dv:+.3f}" if dv is not None else "—")
                if mode in SINGLE_EMB:
                    good = c["works_vs_baseline"]
                    extra = " · within human band" if c.get("within_human_band") else ""
                    vcell = (f"<span class='ok'>holds{extra}</span>" if good
                             else "<span class='no'>below</span>")
                else:
                    vcell = ""
            A(f"<tr class='{ 'base' if base_row else ''}'><td>{html.escape(MODE_LABEL[mode])}</td>"
              f"<td>{dq}</td><td>{lq}</td><td>{delta}</td>"
              f"<td>{_fmt(d.get('macro_f1'),2)}</td><td>{_fmt(d.get('auroc_ge2'),2)}</td>"
              f"<td>{d.get('n','?')}</td><td>{vcell}</td></tr>")
        A("</tbody></table>")
        holds = verdict["holds_modes"]
        msg = (f"A single pre-retrieval embedding <b>holds up</b> on {ax}: "
               f"{', '.join(holds)} ≥ conditioned baseline − {TIE}."
               if holds else
               f"On {ax}, the single-embedding modes fall below the conditioned baseline by more than {TIE} "
               f"— per-axis re-encoding still helps here.")
        A(f"<div class='v'>{msg}</div>")

    A("<p><small>QWK = quadratic-weighted Cohen κ; 95% bootstrap CI in brackets. "
      "DoM = 1-D difference-in-means probe (interpretable primary); linear = class-balanced "
      "full-linear head. Thin real grade-2 counts on ~200 items widen the CIs — read small "
      "gaps cautiously.</small></p>")
    OUT_HTML.write_text("".join(P), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
