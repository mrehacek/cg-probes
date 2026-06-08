"""P1-05 — LLM cluster cards over qwen3 BERTopic output.

For each cluster:
  * Pull 10 most-clicked queries + 10 closest-to-centroid queries (deduplicated).
  * Call gpt-5.4-mini with the cluster_card_v3 prompt.
  * Parse a ClusterCard (Czech label, summary, clinical_relevance, urgency/
    emotional potential, suspected anchor levels, rationale).

Output:
  cache/clusters/cluster_manifest.jsonl       (one row per cluster, full schema)
  cache/clusters/cluster_manifest_summary.csv (flat per-cluster summary for review)

Reuses contrastive/llm_client.py — sha256 prompt cache, async, tenacity retry.

CLI:
  python search-logs-bertopic/cluster_cards.py --pilot 10
  python search-logs-bertopic/cluster_cards.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from contrastive.llm_client import LLMClient  # noqa: E402
from contrastive.schemas import ClusterCard  # noqa: E402

CLUSTERS_DIR = ROOT / "cache" / "clusters"
OVERVIEW_JSONL = CLUSTERS_DIR / "cluster_overview_qwen3_minc10_leaf.jsonl"
PREPROCESS_PARQUET = ROOT / "cache" / "preprocess" / "embed_input.parquet"

MANIFEST_JSONL = CLUSTERS_DIR / "cluster_manifest.jsonl"
SUMMARY_CSV = CLUSTERS_DIR / "cluster_manifest_summary.csv"

PROMPT_PATH = REPO / "contrastive" / "prompts" / "cluster_card_v3.txt"
TEMPLATE_VERSION = "cluster_card_v3"

N_CLICK_EXEMPLARS = 10
N_CENTROID_EXEMPLARS = 10


def _split_prompt(text: str) -> tuple[str, str]:
    sys_marker, usr_marker = "[SYSTEM]", "[USER]"
    if sys_marker not in text or usr_marker not in text:
        sys.exit(f"prompt missing [SYSTEM]/[USER] markers: {PROMPT_PATH}")
    sys_part, rest = text.split(sys_marker, 1)[1].split(usr_marker, 1)
    return sys_part.strip(), rest.strip()


def _load_overview() -> list[dict]:
    rows: list[dict] = []
    with open(OVERVIEW_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_exemplars(
    member_keys: list[str],
    centroid_keys: list[str],
    key_to_query: dict[str, str],
    key_to_clicks: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Return (top_clicks_queries, centroid_queries) — each a list of unique
    query strings, deduplicated within each list but allowing overlap across."""
    # Top by clicks within members
    clicked = sorted(
        ((key_to_clicks.get(k, 0), k) for k in member_keys),
        key=lambda x: (-x[0], x[1]),
    )
    top_clicks_queries = []
    seen_in_clicks: set[str] = set()
    for _, k in clicked:
        q = key_to_query.get(k)
        if q is None or q in seen_in_clicks:
            continue
        seen_in_clicks.add(q)
        top_clicks_queries.append(q)
        if len(top_clicks_queries) >= N_CLICK_EXEMPLARS:
            break

    # Centroid exemplars in the order BERTopic already ranked them
    centroid_queries: list[str] = []
    seen_centroid: set[str] = set()
    for k in centroid_keys[:N_CENTROID_EXEMPLARS]:
        q = key_to_query.get(k)
        if q is None or q in seen_centroid:
            continue
        seen_centroid.add(q)
        centroid_queries.append(q)

    return top_clicks_queries, centroid_queries


async def _process(
    client: LLMClient,
    *,
    row: dict,
    key_to_query: dict[str, str],
    key_to_clicks: dict[str, int],
    sys_prompt: str,
    user_template: str,
) -> tuple[int, ClusterCard | None, str | None, dict]:
    cluster_id = int(row["cluster_id"])
    n_queries = int(row["size"])
    top_clicks_q, centroid_q = _build_exemplars(
        row["member_query_keys"],
        row["exemplar_query_keys"],
        key_to_query,
        key_to_clicks,
    )
    top_words = " ".join(row.get("top_words") or [])[:600]
    user = user_template.format(
        topic_id=cluster_id,
        n_queries=n_queries,
        top_words=top_words,
        top_clicks_queries="\n".join(f"- {q}" for q in top_clicks_q) or "(žádné)",
        centroid_queries="\n".join(f"- {q}" for q in centroid_q) or "(žádné)",
    )
    # gpt-5.4-mini doesn't accept reasoning_effort='minimal' (only none/low/medium/high/xhigh).
    # Triage prompts are simple enough that 'low' is plenty.
    effort = "low"
    try:
        card, usage = await client.call_structured(
            phase="label_and_rate",
            template_version=TEMPLATE_VERSION,
            system=sys_prompt,
            user=user,
            schema_model=ClusterCard,
            reasoning_effort=effort,
        )
        # Override topic_id to make sure it matches cluster_id from BERTopic
        # (the LLM sometimes echoes back the wrong value).
        card = card.model_copy(update={"topic_id": cluster_id, "n_queries": n_queries})
        return cluster_id, card, None, usage
    except Exception as e:
        return cluster_id, None, f"{type(e).__name__}: {e}", {}


async def run(*, pilot: int, concurrency: int, model: str) -> int:
    if not OVERVIEW_JSONL.exists():
        sys.exit(f"missing: {OVERVIEW_JSONL} (run P1-04 first)")
    if not PREPROCESS_PARQUET.exists():
        sys.exit(f"missing: {PREPROCESS_PARQUET} (run P1-02 first)")

    print(f"[load] preprocess parquet …")
    pp = pd.read_parquet(PREPROCESS_PARQUET, columns=["query_key", "query", "clicks"])
    key_to_query = dict(zip(pp["query_key"].astype(str), pp["query"].astype(str)))
    key_to_clicks = dict(zip(pp["query_key"].astype(str), pp["clicks"].fillna(0).astype(int)))

    overview = _load_overview()
    overview.sort(key=lambda r: int(r["cluster_id"]))
    if pilot > 0:
        overview = overview[:pilot]
    print(f"[run] {len(overview):,} clusters → {model}  concurrency={concurrency}  pilot={bool(pilot)}")

    sys_prompt, user_template = _split_prompt(PROMPT_PATH.read_text(encoding="utf-8"))
    client = LLMClient(model=model, concurrency=concurrency)

    t0 = time.time()
    tasks = [
        _process(
            client,
            row=row,
            key_to_query=key_to_query,
            key_to_clicks=key_to_clicks,
            sys_prompt=sys_prompt,
            user_template=user_template,
        )
        for row in overview
    ]

    cards: list[dict] = []
    errors: list[tuple[int, str]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    n_done = 0

    for coro in asyncio.as_completed(tasks):
        cid, card, err, usage = await coro
        n_done += 1
        if err:
            errors.append((cid, err))
        else:
            cards.append(card.model_dump())
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
        if n_done % 50 == 0 or n_done == len(tasks):
            dt = time.time() - t0
            rps = n_done / dt if dt > 0 else 0.0
            eta_min = (len(tasks) - n_done) / rps / 60 if rps > 0 else float("inf")
            print(f"  [{n_done}/{len(tasks)}]  {rps:.1f} cards/s  ETA {eta_min:.1f} min  errors={len(errors)}")

    cards.sort(key=lambda r: int(r["topic_id"]))

    out_jsonl = MANIFEST_JSONL.with_name(
        MANIFEST_JSONL.stem + ("_pilot" if pilot else "") + ".jsonl"
    )
    out_csv = SUMMARY_CSV.with_name(
        SUMMARY_CSV.stem + ("_pilot" if pilot else "") + ".csv"
    )

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[write] {len(cards):,} cards → {out_jsonl}")

    flat = pd.DataFrame(cards)
    flat = flat[[
        "topic_id", "n_queries", "czech_label", "summary_cs",
        "clinical_relevance", "urgency_potential", "emotional_potential",
        "suspected_anchor_levels", "rationale_cs",
    ]]
    flat["suspected_anchor_levels"] = flat["suspected_anchor_levels"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else ""
    )
    flat.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[write] summary CSV → {out_csv}")

    if errors:
        print(f"[errors] {len(errors)} clusters failed (showing first 10):")
        for cid, msg in errors[:10]:
            print(f"  cluster {cid}: {msg}")

    # Relevance distribution
    if not flat.empty:
        print("\n[summary] clinical_relevance distribution:")
        print(flat["clinical_relevance"].value_counts().to_string())

    print(f"\n[tokens] prompt={total_prompt_tokens:,}  completion={total_completion_tokens:,}")
    print(f"[wall] {time.time() - t0:.1f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", type=int, default=0,
                    help="Process only first N clusters (sanity check before paying for full run).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.4-mini")
    args = ap.parse_args()
    return asyncio.run(run(pilot=args.pilot, concurrency=args.concurrency, model=args.model))


if __name__ == "__main__":
    raise SystemExit(main())
