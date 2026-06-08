# `data/` — released data artifacts

| File | What it is | Notes |
|------|------------|-------|
| `cluster_manifest.jsonl` | 2,023 BERTopic clusters: `topic_id`, `n_queries`, Czech `czech_label` / `summary_cs`, `clinical_relevance`, urgency/emotional potential, `suspected_anchor_levels`, `rationale_cs` | **No member queries** — cluster-level metadata only. |
| `example_search_logs.csv` | ~80-row filtered subsample of the (withheld) search-log corpus, for demonstrating the P1 pipeline | **`NEEDS-MANUAL-PII-REVIEW`** — the author must hand-clear every row before any public push. |

Other released data lives next to the code that consumes it (so the pipeline runs
in place):
- `contrastive/cache/probe_dataset_{MU,PU,ET}.parquet` — probe training pairs.
- `golden/cache/golden_set_v1/` — the 200-item golden evaluation set (+ silver
  pipeline gold + the LLM reference), annotators anonymized.
- `benchmark/cache/emb/` — headline-embedder embeddings (float16, text removed).
- `benchmark/cache/directions/` — trained probe direction vectors (`.npy`).
- `results/` — aggregate metrics, figures, and the HTML report (reader copies).

The **full** corpus, raw analytics/XLS exports, employee list, full-corpus
embeddings, and the LLM response cache are **withheld** (see `../DATA_STATEMENT.md`).