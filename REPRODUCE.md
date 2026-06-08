# Reproducing the CG-Probes results

This artifact ships the trained derivatives so the headline numbers can be
re-derived **offline, with no API keys**. Two tiers:

- **Tier A — inspect (no compute).** Open [`results/report.html`](results/report.html)
  (self-contained), [`results/results.json`](results/results.json), and
  [`results/figures/`](results/figures/). These are the paper's numbers and plots.
- **Tier B — recompute the probe results** from the shipped embeddings + contrastive
  pairs + golden set (below).
- **Tier C — full recompute** (re-embed every embedder, re-run LLM baselines): needs
  API keys; see the end.

## 0. Install
```
uv sync          # Python >= 3.12; installs pandas/numpy/scikit-learn/openai/...
```
(Any environment with `scikit-learn`, `pandas`, `numpy`, `pyarrow` works for Tier B;
the report/figures in §2 additionally need `matplotlib`.)

## 1. Tier B — recompute probe results (offline)
The **headline embedder is `gemini`**, and its slim embeddings ship under
`benchmark/cache/emb/` (float16, query text removed — lookups are by `text_id`
hash). This regenerates every `gemini` cell of `results.json`:

```
python -m benchmark.eval_golden --embedders gemini
```
- Reads: `contrastive/cache/probe_dataset_{MU,PU,ET}.parquet` (train/dev/test splits),
  `benchmark/cache/emb/gemini__*.parquet`, `golden/cache/golden_set_v1/golden_400_filled.parquet`.
- Writes: `benchmark/cache/results_probe.json` (+ probe directions to
  `benchmark/cache/directions/`).
- The probe is the **paired-difference-of-means** direction fit on the
  cluster-disjoint contrastive train split; two thresholds are tuned on dev; it is
  evaluated on the held-out contrastive test split (separability) and on the golden
  set (benchmark, macro-F1 with and without synthetic items).

**Expected:** the regenerated `gemini` cells match the shipped `results.json`
**to 3 decimals** (the shipped numbers were computed in float64; the shipped
embeddings are float16 — the difference is below the reported precision).

Benchmark against the clinician (human) gold instead of the silver pipeline gold:
```
python -m benchmark.eval_golden --embedders gemini --gold-source human
```

## 2. Headline table, figures, and the HTML report (offline)
The headline clinician-gold result and the report regenerate **offline** from the
shipped embeddings. The headline best-cells are `gemini` and `qwen8b/generic`
(PU's best cell), and **both ship** under `benchmark/cache/emb/` (`qwen8b/generic`
via Git LFS). The LLM-baseline columns come from cached per-query prediction
parquets in `benchmark/cache/baselines/` (re-scored offline by scikit-learn — no
API calls).

```
python -m benchmark.human_benchmark   # headline clinician-gold QWK table (real n=89)
python -m benchmark.make_report       # rebuilds benchmark/cache/report.html
python -m benchmark.figures           # rebuilds the figures
```
Requires `matplotlib` (included in `uv sync`). Only the **non-headline** cells
(`openai3large`, `harrier27b`, and non-`generic` `qwen8b` modes) are not shipped;
they are not needed for the headline result. Re-embed those specifically (Tier C)
if you want to rebuild them.

## 3. Tier C — full recompute (needs API keys)
Only if you want to rebuild everything from raw texts:
```
# re-embed any embedder/mode (writes benchmark/cache/emb/*.parquet)
python -m benchmark.run_embed --embedder gemini --instr-mode per_axis
python -m benchmark.run_embed --embedder qwen8b --instr-mode generic   # etc.
# LLM baselines (HF endpoint for gpt-oss / safeguard)
python -m benchmark.baselines_llm ...
```
Set the relevant keys in `.env` (see `.env.example`). Re-embedding the full
corpus and re-running the LLM baselines incurs API cost; cached LLM responses are
**not** shipped by design (they embed raw query text).

## Determinism
Fixed seeds for sampling/partitioning; **cluster-disjoint** train/test split
(`hash(supercluster_id)`), so no query leaks between splits; pinned deps via
`uv.lock`.