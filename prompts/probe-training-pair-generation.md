SYSTEM:
You are a clinical NLP research assistant generating synthetic Czech medical
search queries to train interpretability probes for a Czech oncology patient
assistant. Your queries must:
(a) stay strictly within the given topic cluster's domain,
(b) accurately represent the specified grade of the target axis through the
    required linguistic markers — not through topical content,
(c) match the register, length distribution, and stylistic features of real
    search queries (including occasional misspellings, missing diacritics,
    and Czech/Slovak code-switching where present in the cluster sample),
(d) be diverse in surface realization — vary lexical choices, syntactic frames,
    and question forms within each grade.

Output strict JSON only, no commentary.

==================================================
USER:

## Target axis: {AXIS_NAME}

{AXIS_SHORT_DESCRIPTION}

## Grade definitions

{For each grade, include:}
[GRADE_CODE] — [GRADE_LABEL]
Operational definition: ...
Required linguistic markers (≥1 must appear in each generated query): ...
Disqualifying markers (presence forces escalation/demotion): ...

## Topic cluster

Cluster keywords: {CLUSTER_KEYWORDS}

Sample real queries from this cluster (study these for register, length,
spelling patterns, and topical scope — your generated queries must look
like they could appear in this sample):
{20-40 SAMPLE_REAL_QUERIES, copy-pasted verbatim including misspellings}

## Generation task

For each grade of {AXIS_NAME} other than the safety flag (handled separately),
generate {N_PER_GRADE} = 12 synthetic queries that:

1. Stay strictly within the topic domain implied by the cluster keywords and
   sample queries. Do not drift to other clinical areas.

2. Express the target grade through the required linguistic markers from the
   grade definition. Do not import markers from other axes (e.g., do not add
   urgency markers when varying affect; do not add affect markers when varying
   urgency).

3. Match the length distribution of the real cluster sample. As a guideline:
   - ~50% of queries: 2–4 tokens (telegraphic, search-engine style)
   - ~35% of queries: 5–8 tokens (compact natural language)
   - ~12% of queries: 9–13 tokens (elaborated)
   - ~3% of queries: 14–20 tokens (long-form, rarest)
   The actual cluster sample takes precedence — if it skews longer or shorter,
   match that.

4. Vary surface form. Do not reuse the same hedge word, the same affect word,
   or the same syntactic frame more than 3 times within a grade. Use Czech
   synonyms and morphological variation; include occasional Slovak forms only
   if the cluster sample contains them.

5. Reflect realistic search-query features: occasional missing diacritics
   ("rakovina" → "rakovina" or "rakovina"), occasional minor typos, missing
   punctuation, occasional fragmented syntax. Mirror the rate observed in
   the cluster sample.

If a grade is genuinely incompatible with this topic cluster (e.g., the topic
is purely procedural and admits no concern-level affect), output an empty
array for that grade and explain in the "skipped" field. Do not force a grade
that doesn't fit.

## Output schema

{
  "topic_id": "{TOPIC_ID}",
  "axis": "{AXIS_NAME}",
  "generated": {
    "<grade_code>": [
      {
        "query": "...",
        "length_tokens": N,
        "markers_used": ["marker-id from grade definition, e.g. (a)", "..."]
      },
      ...
    ],
    ...
  },
  "skipped": [
    { "grade": "<grade_code>", "reason": "..." }
  ]
}

## Few-shot example (different cluster, different axis)

Target axis: U (Clinical Urgency)
Cluster keywords: antidepresiva nezadouci ucinky ssri vedlejsi
Sample real queries: "antidepresiva nezadouci ucinky", "ssri vedlejsi
ucinky", "antidepresiva pribirani", "antidepresiva libido", "antidepresiva
unava", "kdy zacnou ucinkovat antidepresiva", "antidepresiva potence",
"antidepresiva alkohol", "ssri prujem", "antidepresiva spavost"

Output:
{
  "topic_id": "ex_001",
  "axis": "U",
  "generated": {
    "U0": [
      {"query": "antidepresiva nezadouci ucinky", "length_tokens": 3, "markers_used": ["navigational"]},
      {"query": "ssri ucinkovani doba", "length_tokens": 3, "markers_used": ["navigational"]},
      {"query": "kdy zacnou ucinkovat antidepresiva", "length_tokens": 4, "markers_used": ["informational"]},
      {"query": "ssri vs snri rozdil", "length_tokens": 4, "markers_used": ["informational"]}
    ],
    "U1": [
      {"query": "antidepresiva pribirani jak resit", "length_tokens": 4, "markers_used": ["routine-symptom"]},
      {"query": "ssri snizena libido je to bezne", "length_tokens": 6, "markers_used": ["routine-symptom + concern modifier"]},
      {"query": "antidepresiva unava ranni jak dlouho", "length_tokens": 5, "markers_used": ["routine-symptom + duration"]},
      {"query": "ssri prujem druhy tyden", "length_tokens": 4, "markers_used": ["mild-symptom + duration"]}
    ],
    "U2": [
      {"query": "ssri palpitace 3 dny", "length_tokens": 4, "markers_used": ["concerning-symptom + persistent"]},
      {"query": "po antidepresivech otok jazyka", "length_tokens": 4, "markers_used": ["concerning-symptom requiring same-day contact"]},
      {"query": "ssri tepla vlna a tras", "length_tokens": 5, "markers_used": ["concerning-symptom-cluster"]}
    ],
    "U3": [
      {"query": "po ssri nemuzu dychat", "length_tokens": 4, "markers_used": ["acute-respiratory + temporal-immediacy"]},
      {"query": "antidepresiva otok krku dusim se", "length_tokens": 5, "markers_used": ["anaphylaxis-like emergency"]}
    ]
  },
  "skipped": []
}

Now generate for the target cluster and axis specified above.