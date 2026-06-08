# Axis definitions (the contribution)

CG-Probes recovers three **ordinal safety axes** from patient-query embeddings.
Each axis is graded **0 / 1 / 2** and is defined for the **oncology setting**
(drug names, comorbidities, TNM staging, etc. are all assumed in-scope). The
clinician/LLM-authored rubrics in this folder are the core contribution and ship
verbatim — Czech anchor phrases included.

| Code | Axis (paper name)        | What it grades | 0 → 1 → 2 |
|------|--------------------------|----------------|-----------|
| **MU** | Medical Urgency        | somatic emergency level of the reported issue, ignoring psychological burden | benign → possibly urgent → urgent (EMS) |
| **PU** | Psychological Urgency  | distress / existential concern / suicidal intent expressed in the text | benign → distress → emergency (suicidal intent) |
| **ET** | Topic Sensitivity      | *intrinsic* psychological weight of the subject matter, independent of tone | low-load → mild-load → high-load topic |

> `ET`'s code symbol is the heritage abbreviation ("Emotional Distress,
> topic-based"); the paper-facing name is **Topic Sensitivity**.

## Files
- [`MU.md`](MU.md), [`PU.md`](PU.md), [`ET.md`](ET.md) — full rubrics with operational
  definitions, annotation rules, and the complete Czech anchor sets (verbatim).
- [`axes-snapshot.md`](axes-snapshot.md) — the canonical single-file snapshot that
  defines all three axes together (2025-06-04 spec).

## How grades become tokens
Grades are carried as **axis-prefixed tokens** throughout the code and data
(`MU0`/`MU1`/`MU2`, `PU0…`, `ET0…`); the bare ordinal the probes consume is just
the trailing digit (`token_to_int("MU2") == 2`). See `contrastive/p2_io.py`.