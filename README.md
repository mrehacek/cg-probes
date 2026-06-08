# CG-Probes — recovering guardrail directions from patient-query embeddings

> Code & data artifact for the CIKM short paper *CG-Probes: Recovering Guardrail
> Directions from Patient Query Embeddings*.
>
> **Anonymized submission.** Author, institution, and annotator identities are
> intentionally withheld.

## What this is

Pre-retrieval safety triage of patient questions, done **in embedding space**.
Instead of calling a reasoning LLM on every query, we recover **linear "guardrail"
directions** from a frozen text embedder and read three ordinal safety axes off a
single dot product per axis:

| Code | Axis | 0 → 1 → 2 |
|------|------|-----------|
| **MU** | Medical Urgency | benign → possibly urgent → urgent (EMS) |
| **PU** | Psychological Urgency | benign → distress → suicidal intent |
| **ET** * | Topic Sensitivity | low-load → mild-load → high-load topic |

The clinician-grounded rubrics (with Czech anchors) are the core contribution and
ship in full under [`docs/`](docs/) (with `*.EN.md` English translations). The headline result: a one-dimensional
difference-of-means probe approaches deployable-LLM agreement at ~2 orders of
magnitude lower cost. 

*ET was renamed in paper to TS.

## Pipeline (stages)

```
search-logs-bertopic/  P1  preprocess → embed → BERTopic cluster → cluster cards
                           (clustering in bertopic_query_clusters.ipynb; self-contained)
contrastive/           P2  cluster anchors → synth variants → graded contrastive pairs
golden/pipeline_v2/    P3  pick → sample → verify → backfill → packs → LLM reference
benchmark/             P4  fit probes → LLM baselines → metrics → figures → report
```

## Layout

```
docs/                 full MU / PU / Topic-Sensitivity rubrics + snapshot (+ EN translations)   ← the contribution
REPRODUCE.md          step-by-step reproduction recipe
data/                 cluster manifest (no member queries) + sample search logs
contrastive/cache/    probe_dataset_{MU,PU,ET}.parquet (training pairs)
golden/cache/         200-item golden set (90 real + 110 synthetic), annotators anonymized
benchmark/cache/      headline embeddings (fp16), probe directions, result JSONs, figures, report.html
results/              reader copies: results.json, figures/, report.html
```

## Quickstart

```
uv sync                                   # Python >= 3.12
python -m benchmark.eval_golden --embedders gemini   # regenerate the probe results, offline
```

The probe path needs **no API keys** — the headline embedder's embeddings ship on
disk. See [`REPRODUCE.md`](REPRODUCE.md) for the full recipe and what each
tier reproduces.

The full experiment report is also deployed and browsable at
**<https://cg-probes-report.onrender.com/>** — or open the local copy
[`results/report.html`](results/report.html).

## What is included

Trained, derived, or carefully-subsetted artifacts only: the axis rubrics; the
BERTopic cluster manifest (labels/top-words, **no member queries**); the graded
contrastive pairs; the 200-item golden evaluation set (with the real/synthetic
flag); the headline embedder's embeddings (float16, **query text removed** — kept
by `text_id` hash); the trained probe direction vectors; and the aggregate metrics,
figures, and HTML report.

## What is withheld (data-leakage policy)

Patient search queries are free-text → treated as PII-bearing and not released.

- the full search-log corpus (raw analytics/GA4 exports, the on-site search XLS,
  the unified ~80k-query table);
- hospital employee list that was used to remove people lookups in search logs;
- the full-corpus per-query embeddings, and the LLM response cache (its payloads
  embed raw query text);
- all secrets (`.env` API keys)
- some models were used on university infrastructure

Numbers in the paper are computed on the **full** sets; this artifact ships a safe,
reproducible subset.

## Changes due to annonymization

- institution names and known website/project names were removed from all data by regex,
  hospital name was generalized to "komplexní onkologické centrum" / patient portal
- university infrastructure was used for some of the model inference, so it was replaced

## Status

- [ ] **License** — to be chosen (currently none).

See [`DATA_STATEMENT.md`](DATA_STATEMENT.md) for data origin, de-identification, and
allowed uses.