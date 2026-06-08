"""P4 — saturated-throughput sweep for the safety-LLM triage endpoint.

Per-query latency is replica-independent; THROUGHPUT (queries/sec) scales with
replicas only if enough requests are in flight. This fires the golden queries at
increasing concurrency and reports sustained qps, so we can state what the actual
serving cluster (N×L40S) delivers — the deployment-cost number for the paper.

    python -m benchmark.throughput_probe --effort medium --conc 8,16,32 --n 64
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

OUT_JSON = OUT_DIR / "safeguard_throughput.json"


PER_CALL_CAP = 60.0  # abandon pathological long-reasoning calls so they don't
                     # monopolise a concurrency slot and distort the qps estimate.


async def _call(client, system, user_tmpl, text, effort):
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(client.call_structured(
            phase=f"thru_{effort}", template_version="triage_policy_v1",
            system=system, user=user_tmpl.format(query_text=text),
            schema_model=ThreeAxisVerdict, reasoning_effort=effort,
            temperature=0.0, cache=False), timeout=PER_CALL_CAP)
        return time.perf_counter() - t0
    except Exception:
        return None


async def sweep(model, effort, concs, n):
    g = pd.read_parquet(GOLDEN_400).head(n)
    texts = [str(r["query_text"]) for _, r in g.iterrows()]
    system, user_tmpl = p2_io.split_prompt_file(TRIAGE_PROMPT)
    results = []
    for conc in concs:
        client = _make_client(model)
        client.sem = asyncio.Semaphore(conc)
        t0 = time.perf_counter()
        lats = await asyncio.gather(*(_call(client, system, user_tmpl, t, effort) for t in texts))
        wall = time.perf_counter() - t0
        ok = [x for x in lats if x is not None]
        qps = len(ok) / wall if wall > 0 else None
        row = {"concurrency": conc, "n": len(texts), "ok": len(ok),
               "wall_sec": round(wall, 2), "qps": round(qps, 3) if qps else None,
               "mean_latency_s": round(float(np.mean(ok)), 2) if ok else None,
               "p95_latency_s": round(float(np.percentile(ok, 95)), 2) if ok else None}
        results.append(row)
        print(f"[thru {effort}] conc={conc:3d}  qps={row['qps']}  mean_lat={row['mean_latency_s']}s  "
              f"p95={row['p95_latency_s']}s  ({len(ok)}/{len(texts)} ok in {wall:.1f}s)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-safeguard-20b")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--conc", default="8,16,32")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    concs = [int(c) for c in a.conc.split(",")]
    out = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    res = asyncio.run(sweep(a.model, a.effort, concs, a.n))
    out[f"{a.model}__{a.effort}"] = {"sweep": res, "n_queries": a.n}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[thru] wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
