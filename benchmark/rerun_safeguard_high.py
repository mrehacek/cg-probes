"""P4 — reasoning-effort sweep for the deployable safety-LLM triage baseline,
with per-query classification latency measured on the actual serving GPU.

Background. The earlier baseline silently dropped `reasoning_effort` (the
LLMClient routing regex matched only gpt-5/o-series, not gpt-oss-*), so safeguard
ran at the vLLM default (≈medium). The client now routes gpt-oss reasoning via
extra_body. A real hospital deployment uses ONE triage call per query (not 3
per-axis calls) on a single dedicated GPU (Nvidia L40S 48 GB), so latency/cost is
dominated by generated-token count ÷ GPU decode rate. This script measures that.

Per (model, effort) it writes:
  cache/baselines/{model}__triage_{effort}.parquet   (predictions, --full only)
  cache/baselines/safeguard_latency.json             (timing, always)

Latency, two ways:
  * isolated  — concurrency=1, clean single-query wall time (vs the probe's 0.3us);
                timer is the server round-trip (excludes local queueing).
  * under_load + throughput — concurrency=N, full run wall-clock -> queries/sec
                (what one GPU can actually sustain).

    # latency curve, no full prediction run (fast):
    python -m benchmark.rerun_safeguard_high --effort low,medium,high --iso 12
    # deployable accuracy at low+medium on all 400 + throughput:
    python -m benchmark.rerun_safeguard_high --effort low,medium --full --iso 12
    # does high rescue emergencies? run high only on a query subset:
    python -m benchmark.rerun_safeguard_high --effort high --subset mu2 --iso 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
import pandas as pd

from benchmark.baselines_llm import GOLDEN_400, OUT_DIR, TRIAGE_PROMPT, ThreeAxisVerdict, _make_client
from contrastive import p2_io
from contrastive.p2_io import token_to_int

LATENCY_JSON = OUT_DIR / "safeguard_latency.json"
RUN_CONCURRENCY = 8
CALL_TIMEOUT = 400.0


def _summary(lats):
    a = np.array(lats, dtype=float)
    if not len(a):
        return None
    return {"n": int(len(a)), "mean_s": round(float(a.mean()), 3),
            "median_s": round(float(np.median(a)), 3),
            "p95_s": round(float(np.percentile(a, 95)), 3),
            "min_s": round(float(a.min()), 3), "max_s": round(float(a.max()), 3)}


def _items(limit, subset):
    g = pd.read_parquet(GOLDEN_400)
    if subset == "mu2":  # emergency cases — the safety-critical recall test
        g = g[g["mu_grade"].map(token_to_int) == 2]
    if limit:
        g = g.head(limit)
    return [(str(r["query_key"]), str(r["query_text"])) for _, r in g.iterrows()]


async def _one(client, system, user_tmpl, effort, qkey, text, cache=True):
    user = user_tmpl.format(query_text=text)
    try:
        v, usage = await asyncio.wait_for(client.call_structured(
            phase=f"triage_{effort}_openai/gpt-oss-safeguard-20b",
            template_version="triage_policy_v1", system=system, user=user,
            schema_model=ThreeAxisVerdict, reasoning_effort=effort,
            temperature=0.0, cache=cache), timeout=CALL_TIMEOUT)
    except Exception as e:
        print(f"  ! {text[:38]!r}: {type(e).__name__}", flush=True)
        return ({"query_key": qkey, "text": text, "mu_grade": None,
                 "pu_grade": None, "et_grade": None}, None, None)
    row = {"query_key": qkey, "text": text, "mu_grade": v.mu_grade,
           "pu_grade": v.pu_grade, "et_grade": v.et_grade}
    return row, usage.get("latency_s"), usage.get("completion_tokens")


async def run(model, effort, iso_n, full, subset, gap=0.0):
    system, user_tmpl = p2_io.split_prompt_file(TRIAGE_PROMPT)
    items = _items(0, subset)
    tag = f"{model}/{effort}" + (f"/{subset}" if subset else "")

    # ---- isolated latency (concurrency=1, cache=False = true cold calls). With
    # gap>0 we sleep between calls so the cluster fully drains — the truest
    # single-query latency, no request overlap / batching from a prior call.
    iso_client = _make_client(model)
    iso_client.sem = asyncio.Semaphore(1)
    n = min(iso_n, len(items))
    print(f"[{tag}] isolated latency: {n} sequential cold calls (gap={gap}s)", flush=True)
    iso_lat, iso_tok = [], []
    for i, (qk, tx) in enumerate(items[:n], 1):
        _, lat, tok = await _one(iso_client, system, user_tmpl, effort, qk, tx, cache=False)
        if lat is not None:
            iso_lat.append(lat); iso_tok.append(tok)
        print(f"  [{tag}] {i}/{n}  {lat}s  {tok}tok", flush=True)
        if gap and i < n:
            await asyncio.sleep(gap)

    summ = {"model": model, "reasoning_effort": effort, "subset": subset,
            "isolated": _summary(iso_lat),
            "isolated_mean_completion_tokens": (round(float(np.mean(iso_tok)), 1) if iso_tok else None),
            "endpoint_host": iso_client._endpoint_host}

    # ---- full predictions + throughput
    if full:
        client = _make_client(model)
        client.sem = asyncio.Semaphore(RUN_CONCURRENCY)
        out_path = OUT_DIR / f"{model}__triage_{effort}.parquet"
        done, acc = set(), []
        if out_path.exists():
            prev = pd.read_parquet(out_path)
            acc = prev.to_dict("records"); done = set(prev["query_key"].astype(str))
        jobs = [(qk, tx) for qk, tx in items if qk not in done]
        print(f"[{tag}] full run: {len(jobs)} fresh @ conc={RUN_CONCURRENCY} ({len(done)} cached)", flush=True)
        run_lat, t0 = [], time.perf_counter()
        for i in range(0, len(jobs), 60):
            chunk = jobs[i:i + 60]
            res = await asyncio.gather(*(_one(client, system, user_tmpl, effort, qk, tx) for qk, tx in chunk))
            for row, lat, _ in res:
                acc.append(row)
                if lat is not None:
                    run_lat.append(lat)
            pd.DataFrame(acc).to_parquet(out_path, index=False)
            ok = sum(1 for r, _, _ in res if r["mu_grade"] is not None)
            print(f"[{tag}] {min(i+60,len(jobs))}/{len(jobs)} | {ok}/{len(chunk)} parsed | ckpt", flush=True)
        wall = time.perf_counter() - t0
        summ["under_load"] = _summary(run_lat)
        summ["throughput_qps"] = (round(len(jobs) / wall, 3) if wall > 0 and jobs else None)
        summ["run_concurrency"] = RUN_CONCURRENCY
        summ["fresh_queries"] = len(jobs)
        summ["wall_sec"] = round(wall, 2)
        summ["out_parquet"] = out_path.name

    print(f"[{tag}] DONE iso={summ['isolated']} qps={summ.get('throughput_qps')}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-safeguard-20b")
    ap.add_argument("--effort", default="medium", help="comma list: low,medium,high")
    ap.add_argument("--iso", type=int, default=12, help="isolated-latency sample size")
    ap.add_argument("--full", action="store_true", help="also run all 400 for predictions+throughput")
    ap.add_argument("--subset", default="", choices=["", "mu2"], help="restrict items (mu2 = emergencies)")
    ap.add_argument("--gap", type=float, default=0.0, help="sleep (s) between isolated calls for a drained-cluster latency")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = json.loads(LATENCY_JSON.read_text(encoding="utf-8")) if LATENCY_JSON.exists() else {}
    for effort in [e.strip() for e in a.effort.split(",") if e.strip()]:
        key = f"{a.model}__{effort}" + (f"__{a.subset}" if a.subset else "") + ("__gap" if a.gap else "")
        out[key] = asyncio.run(run(a.model, effort, a.iso, a.full, a.subset, a.gap))
        LATENCY_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[latency] wrote {LATENCY_JSON}  key={key}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
