"""P4 — deployable LLM baselines on the golden-400 (probe-vs-LLM comparison).

Zero-shot, rubric-inlined classification of each golden query on MU/PU/ET, using
the same blind axis-judge prompt the synthesis verifier uses (query + rubric
only). Baselines are **deployable open models**:
  * gpt-oss-120b           (self-hosted OpenAI-compatible endpoint)
  * gpt-oss-safeguard-20b  (OpenAI API)
gpt-5.4 is the frontier reference (already computed in golden step 6), not re-run.

Chunked, checkpointed, concurrent (per the incremental-progress rule). Output:
  benchmark/cache/baselines/{model}.parquet  (query_key, text, axis, pred_grade)
Scored against the golden labels by metrics/eval. Smoke a model first:
  python -m benchmark.baselines_llm --model gpt-oss-120b --limit 6
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from contrastive import p2_io
from contrastive.llm_client import LLMClient
from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int
from golden.llm_annotator.pipeline._schema import AxisJudgeVerdict
from golden.pipeline_v2._io import write_parquet_atomic

GOLDEN_400 = REPO / "golden" / "cache" / "golden_set_v1" / "golden_400_filled.parquet"
JUDGE_PROMPT = REPO / "golden" / "llm_annotator" / "prompts" / "axis_judge.txt"
TRIAGE_PROMPT = REPO / "benchmark" / "prompts" / "triage_policy_v1.txt"
OUT_DIR = REPO / "benchmark" / "cache" / "baselines"

CALL_TIMEOUT = 120.0
CHUNK = 60  # queries per checkpoint


class ThreeAxisVerdict(BaseModel):
    """Single-call 3-axis classification (gpt-oss-safeguard policy prompt)."""
    mu_grade: int = Field(..., ge=0, le=2)
    pu_grade: int = Field(..., ge=0, le=2)
    et_grade: int = Field(..., ge=0, le=2)
    rationale_cs: str = Field("", max_length=300)

    @field_validator("mu_grade", "pu_grade", "et_grade")
    @classmethod
    def _in_range(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"grade must be 0/1/2, got {v}")
        return v


def _hfie_served_model(base: str, token: str, fallback: str) -> str:
    """vLLM needs the request `model` to match the served name — read it from
    /v1/models on the HF Inference Endpoint."""
    try:
        r = httpx.get(base.rstrip("/") + "/models",
                      headers={"Authorization": f"Bearer {token}"}, timeout=20)
        data = r.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return fallback


def _make_client(model: str) -> LLMClient:
    """gpt-oss-120b -> OpenAI-compatible endpoint; gpt-oss-safeguard-20b -> HFIE vLLM."""
    if model == "gpt-oss-120b":
        return LLMClient.for_openai_compatible(model=model, concurrency=12)
    if model == "gpt-oss-safeguard-20b":
        url = os.environ.get("HF_GPT_SAFEGUARD_20B_ENDPOINT_URL")
        token = os.environ.get("HF_TOKEN")
        if not url or not token:
            raise RuntimeError("HF_GPT_SAFEGUARD_20B_ENDPOINT_URL and HF_TOKEN must be set")
        base = url.rstrip("/") + ("" if url.rstrip("/").endswith("/v1") else "/v1")
        served = _hfie_served_model(base, token, model)
        return LLMClient(model=served, api_key=token, base_url=base, concurrency=12)
    return LLMClient(model=model, concurrency=12)


async def _bounded(coro):
    try:
        return await asyncio.wait_for(coro, timeout=CALL_TIMEOUT)
    except Exception:
        return None


async def classify(model: str, limit: int | None) -> int:
    g = pd.read_parquet(GOLDEN_400)
    if limit:
        g = g.head(limit)
    system, user_tmpl = p2_io.split_prompt_file(JUDGE_PROMPT)
    client = _make_client(model)
    rubrics = {ax: p2_io.load_axis_rubric(ax) for ax in SAFETY_AXES}

    out_path = OUT_DIR / f"{model}.parquet"
    done = set()
    acc: list[dict] = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        acc = prev.to_dict("records")
        done = {(r["query_key"], r["axis"]) for r in acc}

    jobs = [(str(r["query_key"]), str(r["query_text"]), ax)
            for _, r in g.iterrows() for ax in SAFETY_AXES
            if (str(r["query_key"]), ax) not in done]
    print(f"[{model}] {len(jobs)} (query,axis) judgements to do ({len(done)} cached)", flush=True)

    async def one(qkey, text, axis):
        user = user_tmpl.format(axis_short_code=axis, axis_long_name=p2_io.AXIS_LONG_NAMES_CS[axis],
                                axis_grades_csv=p2_io.axis_grades_csv(axis),
                                axis_rubric=rubrics[axis], query_text=text)
        res = await _bounded(client.call_structured(
            phase=f"baseline_{model}_{axis}", template_version="axis_judge_v1",
            system=system, user=user, schema_model=AxisJudgeVerdict,
            reasoning_effort="low", temperature=0.0))
        grade = None
        if res is not None:
            tok = res[0].grade
            grade = token_to_int(tok) if tok.startswith(axis) else None
        return {"query_key": qkey, "text": text, "axis": axis, "pred_grade": grade}

    for i in range(0, len(jobs), CHUNK):
        chunk = jobs[i:i + CHUNK]
        rows = await asyncio.gather(*(one(*j) for j in chunk))
        acc.extend(rows)
        write_parquet_atomic(pd.DataFrame(acc), out_path)
        ok = sum(1 for r in rows if r["pred_grade"] is not None)
        print(f"[{model}] {min(i+CHUNK,len(jobs))}/{len(jobs)} | {ok}/{len(chunk)} parsed | checkpointed",
              flush=True)

    print(f"[{model}] done -> {out_path}", flush=True)
    return 0


async def classify_triage(model: str, limit: int | None) -> int:
    """Single inference per query → all three axes (policy prompt + few-shot)."""
    g = pd.read_parquet(GOLDEN_400)
    if limit:
        g = g.head(limit)
    system, user_tmpl = p2_io.split_prompt_file(TRIAGE_PROMPT)
    client = _make_client(model)

    out_path = OUT_DIR / f"{model}__triage.parquet"
    done, acc = set(), []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        acc = prev.to_dict("records")
        done = set(prev["query_key"].astype(str))
    jobs = [(str(r["query_key"]), str(r["query_text"]))
            for _, r in g.iterrows() if str(r["query_key"]) not in done]
    print(f"[{model}/triage] {len(jobs)} queries to classify ({len(done)} cached)", flush=True)

    async def one(qkey, text):
        user = user_tmpl.format(query_text=text)
        res = await _bounded(client.call_structured(
            phase=f"triage_{model}", template_version="triage_policy_v1",
            system=system, user=user, schema_model=ThreeAxisVerdict,
            reasoning_effort="medium", temperature=0.0))
        row = {"query_key": qkey, "text": text,
               "mu_grade": None, "pu_grade": None, "et_grade": None}
        if res is not None:
            v = res[0]
            row.update(mu_grade=v.mu_grade, pu_grade=v.pu_grade, et_grade=v.et_grade)
        return row

    for i in range(0, len(jobs), CHUNK):
        rows = await asyncio.gather(*(one(*j) for j in jobs[i:i + CHUNK]))
        acc.extend(rows)
        write_parquet_atomic(pd.DataFrame(acc), out_path)
        ok = sum(1 for r in rows if r["mu_grade"] is not None)
        print(f"[{model}/triage] {min(i+CHUNK,len(jobs))}/{len(jobs)} | {ok}/{len(rows)} parsed | checkpointed",
              flush=True)
    print(f"[{model}/triage] done -> {out_path}", flush=True)
    return 0


def score_triage(model: str) -> None:
    from sklearn.metrics import f1_score
    g = pd.read_parquet(GOLDEN_400)
    g["synthetic"] = g["source"].astype(str) == "synthesized"
    pred = pd.read_parquet(OUT_DIR / f"{model}__triage.parquet")
    pred["query_key"] = pred["query_key"].astype(str)
    g["query_key"] = g["query_key"].astype(str)
    print(f"\n[{model}/triage] golden macro-F1 (single-call 3-axis):")
    for ax in SAFETY_AXES:
        truth = g.assign(grade=g[f"{ax.lower()}_grade"].map(token_to_int))[
            ["query_key", "grade", "synthetic"]]
        p = pred[["query_key", f"{ax.lower()}_grade"]].rename(columns={f"{ax.lower()}_grade": "pred"})
        m = truth.merge(p, on="query_key").dropna(subset=["pred"])
        if m.empty:
            continue
        f_all = f1_score(m["grade"], m["pred"], labels=[0, 1, 2], average="macro", zero_division=0)
        r = m[~m["synthetic"]]
        f_real = (f1_score(r["grade"], r["pred"], labels=[0, 1, 2], average="macro",
                           zero_division=0) if len(r) else None)
        rr = f"{f_real:.3f}" if f_real is not None else "—"
        print(f"  {ax}: all={f_all:.3f} real={rr} (n={len(m)})")


def score(model: str) -> None:
    """Per-axis macro-F1 of the baseline vs golden labels (all + real-only)."""
    from sklearn.metrics import f1_score
    g = pd.read_parquet(GOLDEN_400)
    g["synthetic"] = g["source"].astype(str) == "synthesized"
    pred = pd.read_parquet(OUT_DIR / f"{model}.parquet")
    print(f"\n[{model}] golden macro-F1:")
    for ax in SAFETY_AXES:
        truth = g.assign(grade=g[f"{ax.lower()}_grade"].map(token_to_int))[
            ["query_key", "grade", "synthetic"]]
        p = pred[pred["axis"] == ax][["query_key", "pred_grade"]]
        m = truth.merge(p, on="query_key").dropna(subset=["pred_grade"])
        if m.empty:
            continue
        f_all = f1_score(m["grade"], m["pred_grade"], labels=[0, 1, 2], average="macro", zero_division=0)
        r = m[~m["synthetic"]]
        f_real = (f1_score(r["grade"], r["pred_grade"], labels=[0, 1, 2], average="macro",
                           zero_division=0) if len(r) else None)
        rr = f"{f_real:.3f}" if f_real is not None else "—"
        print(f"  {ax}: all={f_all:.3f} real={rr} (n={len(m)})")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["triage", "per_axis"], default="triage",
                    help="triage = single call, 3 axes (policy prompt); per_axis = 3 calls")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if a.mode == "triage":
        if not a.score_only:
            asyncio.run(classify_triage(a.model, a.limit))
        score_triage(a.model)
    else:
        if not a.score_only:
            asyncio.run(classify(a.model, a.limit))
        score(a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
