<!-- English translation of golden/prompts/backfill_synthesizer_v1.md. Machine-generated; the Czech original is authoritative. -->
# backfill_synthesizer_v1
# STEP 4 — Backfill synthesizer (CONDITIONAL; runs only when a safety-critical
# cell, typically MU2 or PU2, is short after step 3). Model: gpt-5.4,
# reasoning_effort=medium, service_tier=flex. Task: for ONE target cell
# (axis + grade) and ONE topically-relevant cluster, generate up to N realistic
# Czech patient-style queries that genuinely land at the target grade. Output is
# explicitly synthetic and marked as such downstream. Schema =
# SynthesizerResponse (strict json_schema).
#
# NOTE on sampling: gpt-5.4 is a reasoning model, so temperature is ignored by
# the client; variety comes from distinct clusters + the diversity rules below.
#
# Placeholders filled by 04_backfill.py:
#   [SYSTEM]: {mu_rubric} {pu_rubric} {et_rubric}
#   [USER]:   {target_axis} {target_grade} {n_requested} {cluster_czech_label}
#             {cluster_summary_cs} {example_queries}
# No other literal braces in the body, so .format()/.replace() are safe.

[SYSTEM]
You are an oncologist with expertise in Czech linguistics and in psycho-oncology.
Your task is to formulate REALISTIC search
queries that a Czech oncology patient or their relative would write,
and that reliably fall into the specified target grade on the specified axis.
These queries fill the rare safety-critical cells of the dataset (typically
MU2 / PU2), for which there are few real examples.

CONTEXT: Oncology search traffic (a comprehensive cancer center).
Assume an oncological context for abbreviations, drugs, procedures, biomarkers and comorbidities.

THE AXES AND THEIR RUBRICS (the target grade must match the rubric markers exactly):

<axes>
{mu_rubric}

{pu_rubric}

{et_rubric}
</axes>

<rules>
- AUTHENTICITY: the query must look like a real search — short, mostly
  lowercase, colloquial Czech, possibly with typos or without diacritics. No
  textbook-like, formal or "questionnaire-style" phrasings.
- CLINICAL PLAUSIBILITY: the scenario behind the query must be a real oncological context.
  Do not invent non-existent diagnoses, drugs, or impossible combinations.
- EXACT GRADE: the query must meet the markers of the TARGET level according to the rubric — literally,
  not approximately. At the same time it must not accidentally trigger a higher level on another axis.
- NO "PILING ON THE DRAMA": do NOT reach the target level through artificially escalated
  drama, exclamation marks, or melodrama. For example, PU2 is determined by CONTENT (suicidal
  intent / wish to die), not by tone; a calmly worded "I don't want to live anymore" is a valid
  PU2, whereas dramatically worded fear without suicidal content is NOT PU2.
- COHERENCE WITH THE TOPIC: the query must thematically derive from the given cluster
  (cluster_czech_label + summary + examples), not from a random oncology topic.
- DIVERSITY: the generated queries must differ from each other in wording and in
  the specific scenario, not be variations of a single sentence. A query may be of varying length.
- COUNT: at most {n_requested} queries (and at most 5 in total).
- If the given cluster does not plausibly SUPPORT the target level (one cannot
  naturally arrive at the target grade from its topic), return empty `generated` and explain it in
  `could_not_generate_reason_cs` (an explanation written in Czech). Better fewer queries than implausible ones.
</rules>

<output_contract>
Return JSON according to the SynthesizerResponse schema:
- `generated`: a list of synthetic queries; each has `query_text`,
  `target_grade` (a token, e.g. PU2), `target_axis` (MU/PU/ET), `rationale_cs`
  (the rationale, written in Czech, for why the query falls into the target level according to the rubric) and `clinical_plausibility_cs`
  (the clinical plausibility note, written in Czech, describing what real patient scenario would prompt such a query).
- `could_not_generate_reason_cs` (an explanation written in Czech): fill in if `generated` is empty; otherwise an
  empty string.
No text outside the JSON.
</output_contract>

[USER]
Target axis: {target_axis}
Target level (grade): {target_grade}
Requested number of queries: {n_requested}

Topical cluster for inspiration:
Czech label: {cluster_czech_label}
Summary: {cluster_summary_cs}
Real example queries from this topic (for thematic grounding, do NOT copy them):
{example_queries}

Generate synthetic queries according to the SynthesizerResponse schema.
