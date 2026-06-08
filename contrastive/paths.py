"""Filesystem layout for this repo's local cache.

Self-contained: the BERTopic clustering pipeline lives in `search-logs-bertopic/`
(see `search-logs-bertopic/bertopic_query_clusters.ipynb`) and writes its outputs
to `search-logs-bertopic/cache/`. The contrastive stage reads those and writes its
own outputs under `cache/`.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Outputs (this repo's cache)
CACHE = Path("cache")
LLM_RESPONSES = CACHE / "llm_responses"
DIVERSE_TOP_K = CACHE / "diverse_top_k.parquet"
ANCHOR_CANDIDATES_URGENCY = CACHE / "anchor_candidates_urgency.csv"
ANCHOR_CANDIDATES_EMOTIONAL = CACHE / "anchor_candidates_emotional.csv"
MEETING_PACK_URGENCY = CACHE / "meeting_pack_urgency.csv"
CLUSTER_CARDS = CACHE / "cluster_cards.parquet"
ANCHORS_LOCKED_URGENCY = CACHE / "anchors_locked_urgency.parquet"
ANCHORS_LOCKED_EMOTIONAL = CACHE / "anchors_locked_emotional.parquet"
