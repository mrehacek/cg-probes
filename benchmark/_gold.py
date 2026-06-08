"""Single source of truth for "the gold" labels the benchmark evaluates against.

Two sources, identical schema, so silver<->human is a one-argument switch across
every consumer (eval_golden, tier1_validity, baselines_llm, figures, make_report):

  * "silver" — `golden_400_filled.parquet`: pipeline picker + gpt-5.4 reference
    labels (400 items, 110 synthetic). The pre-clinician headline.
  * "human"  — `human_gold.parquet` (built by annotations_pull): the clinician
    gold for package "Golden v1 — Core" (200 items = 90 real + 110 synthetic,
    each doubly-rated by two oncologists; gold = adjudicated value, else the
    agreed value). gpt-5.4 / LLM judges are NOT in this gold.

Grades are carried as axis-prefixed **tokens** ("MU2") in both, so downstream
`token_to_int` works unchanged. `synthetic` is the real-vs-synthesized flag (both
sources carry it; the human set is NOT real-only).
"""

from __future__ import annotations

import pandas as pd

from contrastive.p2_io import REPO, SAFETY_AXES, token_to_int

GOLD_DIR = REPO / "golden" / "cache" / "golden_set_v1"
SILVER = GOLD_DIR / "golden_400_filled.parquet"
HUMAN = GOLD_DIR / "human_gold.parquet"

GRADE_COLS = [f"{ax.lower()}_grade" for ax in SAFETY_AXES]


def load_gold(source: str = "silver") -> pd.DataFrame:
    """Uniform gold frame: query_key, query_text, source, {mu,pu,et}_grade tokens,
    synthetic(bool). `source` in {"silver","human"}."""
    if source == "silver":
        g = pd.read_parquet(SILVER)
        out = g[["query_key", "query_text", *GRADE_COLS]].copy()
        out["source"] = g["source"].astype(str)
        out["synthetic"] = out["source"] == "synthesized"
    elif source == "human":
        if not HUMAN.exists():
            raise FileNotFoundError(
                f"{HUMAN} not found — run `python -m benchmark.annotations_pull --execute` first")
        g = pd.read_parquet(HUMAN)
        out = g[["query_key", "query_text", *GRADE_COLS, "synthetic"]].copy()
        out["source"] = g.get("orig_source", pd.Series(["human"] * len(g)))
    else:
        raise ValueError(f"unknown gold source {source!r} (silver|human)")
    out["query_key"] = out["query_key"].astype(str)
    return out


def golden_axis(axis: str, source: str = "silver") -> pd.DataFrame:
    """Per-axis gold in the shape eval_golden expects: {text, grade(int), synthetic}.

    Rows whose grade token is missing (e.g. an unresolved human item) are dropped.
    """
    g = load_gold(source)
    col = f"{axis.lower()}_grade"
    sub = g[g[col].notna()].copy()
    return pd.DataFrame({
        "text": sub["query_text"].astype(str),
        "grade": sub[col].map(token_to_int).astype(int),
        "synthetic": sub["synthetic"].astype(bool),
        "query_key": sub["query_key"].astype(str),
    }).reset_index(drop=True)
