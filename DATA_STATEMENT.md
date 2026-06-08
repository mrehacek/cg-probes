# Data statement

> **Anonymized submission.** Author, institution, and annotator identities are
> withheld. Bracketed `[...]` items are to be finalized for the camera-ready.

## Origin
The underlying corpus is the **search-query log of a Czech comprehensive cancer
centre's patient-facing web portals** (public information site + on-site search),
collected over a ~16-month window. Queries are short, free-text, Czech-language,
and typed by patients, carers, and the public — i.e. they are about oncology care
(symptoms, treatments, logistics, prognosis, distress).

## What is released here (and why it is safe)
This artifact ships **only derived or carefully-subsetted** data:
- **Axis rubrics** (`docs/`) — clinician/LLM-authored definitions; no patient data.
- **Cluster manifest** (`data/cluster_manifest.jsonl`) — BERTopic cluster ids, Czech
  labels, summaries, and top-words **only**; no member queries.
- **Contrastive pairs** (`contrastive/cache/probe_dataset_*.parquet`) — the probe
  training data: real cluster-anchor queries + LLM-synthesized variants, graded 0/1/2.
- **Golden set** (`golden/cache/golden_set_v1/`) — the 200-item evaluation set
  (90 real + 110 synthetic), with the real/synthetic flag retained.
- **Embeddings** of the above (headline embedder only; **query text removed**, kept
  by `text_id` hash), probe direction vectors, and aggregate result JSONs/figures.
- **`data/example_search_logs.csv`** — a small filtered subsample illustrating P1.

## What is withheld (data-leakage policy)
The **full search-log corpus**, raw analytics exports, the on-site search XLS, the
employee list, all per-query embeddings of the full corpus, and the LLM response
cache are **never released** (they carry, or are derived 1:1 from, raw patient
free-text). See the repository `README.md` → *What is withheld*.

## De-identification
Search queries are free-text and are treated as PII-bearing until scrubbed. The
released text columns were passed through (a) the project's spam/navigational/
employee-name filters during corpus construction, and (b) a release-time scrub for
emails, phone numbers, national-id (rodné číslo), long digit-runs, and personal
names. **The 90 real golden queries and `example_search_logs.csv` additionally
receive a manual hand-review by the author before any public push.**

## Annotators
Two oncologists each rated the 200-item golden set; the gold label is the
adjudicated value (else the agreed value). An LLM reference annotator is reported
separately and is **not** part of the human gold. All annotator identities are
anonymized to `annotator_A` / `annotator_B` / `llm_reference`.

## Allowed uses
Research and educational use related to clinical-safety triage of patient queries.
Not for clinical decision-making. [License to be added once institutional sign-off
is obtained — see `README.md`.]