"""Audit what `SpacyPersonFilter` (xx_ent_wiki_sm) drops on real queries.

Runs the multilingual NER over the aggregated query CSV, keeps every query
where spaCy flagged a PER entity, then writes a random 5000 to disk along
with the matched entity text + label so you can eyeball precision on Czech.

Output: scripts/out/spacy_person_filtered_sample.csv
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language

REPO = Path(__file__).resolve().parents[1]
CSV_IN = REPO / "data" / "search_queries_unified_2025-2026_aggregated.csv"
OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SAMPLE = OUT_DIR / "spacy_person_filtered_sample.csv"
OUT_ALL = OUT_DIR / "spacy_person_filtered_all.csv"

SAMPLE_N = 5000
SEED = 42


def collect_person_hits(nlp: Language, queries: list[str]) -> list[dict]:
    """Yield rows for every query where spaCy detected at least one PER entity."""
    hits: list[dict] = []
    # spaCy's nlp.pipe is much faster than calling nlp(q) in a loop.
    for q, doc in zip(queries, nlp.pipe(queries, batch_size=512, n_process=1)):
        pers = [ent for ent in doc.ents if ent.label_ == "PER"]
        if not pers:
            continue
        hits.append({
            "query": q,
            "matched_entities": " | ".join(e.text for e in pers),
            "labels": " | ".join(e.label_ for e in pers),
            "all_entities": " | ".join(f"{e.text}::{e.label_}" for e in doc.ents),
        })
    return hits


def main() -> int:
    if not CSV_IN.exists():
        print(f"missing input: {CSV_IN}", file=sys.stderr)
        return 1

    print(f"loading {CSV_IN.name} ...")
    df = pd.read_csv(CSV_IN, usecols=["query"])
    df = df.dropna(subset=["query"]).drop_duplicates(subset=["query"]).reset_index(drop=True)
    queries = df["query"].astype(str).tolist()
    print(f"  unique queries: {len(queries):,}")

    print("loading spaCy xx_ent_wiki_sm ...")
    nlp = spacy.load("xx_ent_wiki_sm")

    print("running NER ...")
    hits = collect_person_hits(nlp, queries)
    print(f"  PER hits: {len(hits):,}  ({len(hits) / len(queries):.2%} of unique queries)")

    pd.DataFrame(hits).to_csv(OUT_ALL, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"  wrote full list -> {OUT_ALL}")

    rng = random.Random(SEED)
    sample = hits if len(hits) <= SAMPLE_N else rng.sample(hits, SAMPLE_N)
    pd.DataFrame(sample).to_csv(OUT_SAMPLE, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"  wrote random sample (n={len(sample)}) -> {OUT_SAMPLE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
