"""Flex-tier LLM wrappers + a concurrent runner for the picker/verifier/synth.

Thin layer over `contrastive.llm_client.LLMClient`, which already provides the
asyncio.Semaphore concurrency cap, Tenacity retry (12 attempts, exp backoff),
flex-tier 900s timeout, reasoning-model auto-routing, strict json_schema, and a
sha256 per-call disk cache. We do NOT re-implement any of that (PLAN §12 is
superseded). The only new code here is `run_concurrent`, which adds the
step-level checkpoint / state / errors.jsonl layer on top of the client.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

from contrastive.llm_client import LLMClient

from golden.pipeline_v2._io import append_jsonl

# Model + tier choices come straight from PLAN §1.
PICKER_MODEL = "gpt-5.4"
VERIFIER_MODEL = "gpt-5.4-mini"
BACKFILL_MODEL = "gpt-5.4"
FLEX = "flex"

# USD per 1M tokens (list price; from the archived estimate_openai_cost.py).
# flex/batch tiers are ~half price (PLAN §1) — applied via FLEX_DISCOUNT.
PRICING = {
    "gpt-5.4": {"input": 1.25, "output": 10.0},
    "gpt-5.4-mini": {"input": 0.375, "output": 3.0},
}
FLEX_DISCOUNT = 0.5


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, *, flex: bool = True
) -> float:
    """Upper-bound USD estimate for a token spend.

    Prices every prompt token at the input rate (ignores OpenAI server-side
    prompt-cache discounts, which the usage dict doesn't break out), so the real
    bill is typically lower. Returns 0.0 for unknown models.
    """
    p = PRICING.get(model)
    if not p:
        return 0.0
    disc = FLEX_DISCOUNT if flex else 1.0
    return (
        prompt_tokens / 1e6 * p["input"] * disc
        + completion_tokens / 1e6 * p["output"] * disc
    )


def picker_client(concurrency: int = 100) -> LLMClient:
    return LLMClient(model=PICKER_MODEL, concurrency=concurrency, service_tier=FLEX)


def verifier_client(concurrency: int = 100) -> LLMClient:
    return LLMClient(model=VERIFIER_MODEL, concurrency=concurrency, service_tier=FLEX)


def backfill_client(concurrency: int = 30) -> LLMClient:
    return LLMClient(model=BACKFILL_MODEL, concurrency=concurrency, service_tier=FLEX)


def reference_client(concurrency: int = 50) -> LLMClient:
    """gpt-5.4 on the REGULAR tier (flex OFF, per user) for the uploaded LLM
    reference annotator — avoids flex-tail latency at full price."""
    return LLMClient(model=PICKER_MODEL, concurrency=concurrency, service_tier=None)


# worker(item) -> (rows, usage): rows is a list[dict] to accumulate into the
# step's parquet; usage is the {prompt_tokens, completion_tokens} dict returned
# by call_structured (empty {} on cache hit). The worker may raise; the failure
# is logged to errors.jsonl and that item is skipped.
Worker = Callable[[object], Awaitable[tuple[list[dict], dict]]]
Checkpoint = Callable[[list[dict], dict, int, int], None]


async def run_concurrent(
    items: Sequence[object],
    worker: Worker,
    *,
    checkpoint_every: int = 50,
    on_checkpoint: Checkpoint | None = None,
    errors_path: Path | None = None,
    item_id: Callable[[object], object] = lambda x: x,
) -> tuple[list[dict], dict]:
    """Run `worker` over `items` concurrently, streaming results as they finish.

    Concurrency is bounded by the client's own semaphore (set when you build the
    client), so we launch every task up front and consume via as_completed.
    Calls `on_checkpoint(rows_so_far, stats, n_completed, n_total)` every
    `checkpoint_every` finished items and once more at the end. Failures are
    appended to `errors_path` and counted in stats but never abort the run.

    Returns (all_rows, stats) where stats has completed / failed / prompt_tokens
    / completion_tokens.
    """
    rows: list[dict] = []
    stats = {"completed": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}

    async def _wrap(it: object):
        try:
            r, usage = await worker(it)
            return it, r, usage, None
        except Exception as exc:  # noqa: BLE001 — capture every failure mode
            return it, None, None, exc

    tasks = [asyncio.ensure_future(_wrap(it)) for it in items]
    total = len(tasks)
    n = 0
    for fut in asyncio.as_completed(tasks):
        it, r, usage, err = await fut
        n += 1
        if err is not None:
            stats["failed"] += 1
            if errors_path is not None:
                append_jsonl(errors_path, {"item": item_id(it), "error": repr(err)})
        else:
            stats["completed"] += 1
            rows.extend(r or [])
            stats["prompt_tokens"] += (usage or {}).get("prompt_tokens", 0)
            stats["completion_tokens"] += (usage or {}).get("completion_tokens", 0)
        if on_checkpoint and checkpoint_every > 0 and n % checkpoint_every == 0:
            on_checkpoint(rows, stats, n, total)
    if on_checkpoint:
        on_checkpoint(rows, stats, n, total)
    return rows, stats
