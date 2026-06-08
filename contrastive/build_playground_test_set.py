"""P2 — build a playground test set for hand-eyeballing the
synth_variant_v2.txt prompt before committing to a Batch API run.

Output (under `contrastive/cache/p2_playground/`):
  * test_set.csv — 20 anchors per axis (random stratified across grades), with:
      axis, anchor_grade, anchor_query, topic_id, czech_label, target_grades,
      input  (fully-rendered USER message),
      model, rater_quality, rater_target_grade_match, rater_topic_preserved,
      rater_register_preserved, rater_notes      (empty for user to fill in)
  * system_prompt.md — the SYSTEM message, copy-paste once into the playground.
  * README.md — how to use the test set in the OpenAI playground evals UI.

CLI:
  python -m contrastive.build_playground_test_set \\
      --picks golden/cache/axis_grade_candidates__qwen3.5-122b.parquet \\
      --per-axis 20 --cp-fix

  python -m contrastive.build_playground_test_set --skip-axes CP   # skip until re-run
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from contrastive.llm_client import _strictify  # noqa: E402
from contrastive.schemas import AXIS_GRADES, SynthVariantSet  # noqa: E402
from golden.pipeline._io import (  # noqa: E402
    AXIS_LONG_NAMES_CS,
    CACHE_DIR as GOLDEN_CACHE_DIR,
    load_axis_rubric,
    load_manifest,
    split_prompt,
)

PLAYGROUND_DIR = REPO / "contrastive" / "cache" / "p2_playground"
PROMPT_PATH = REPO / "contrastive" / "prompts" / "synth_variant_v2.txt"

_CP_SWAP = {"CP0": "CP0", "CP1": "CP2", "CP2": "CP1"}

ANNOTATION_COLS = [
    "model",
    "rater_quality",            # 1..5
    "rater_target_grade_match", # all-correct / partial / none
    "rater_topic_preserved",    # yes / drift / off
    "rater_register_preserved", # yes / drift / off
    "rater_notes",
]


def render_user_message(
    *, axis: str, anchor_query: str, anchor_grade: str,
    user_template: str,
) -> tuple[str, list[str]]:
    grades = list(AXIS_GRADES[axis])
    target_grades = [g for g in grades if g != anchor_grade]
    target_block = "\n".join(f"  - {g}" for g in target_grades)
    user = user_template.format(
        axis_short_code=axis,
        axis_long_name=AXIS_LONG_NAMES_CS.get(axis, axis),
        axis_grades_csv=", ".join(grades),
        axis_rubric=load_axis_rubric(axis),
        anchor_query=anchor_query,
        anchor_grade=anchor_grade,
        target_grades_block=target_block,
    )
    return user, target_grades


def sample_per_axis(
    picks: pd.DataFrame, axis: str, *, per_axis: int, rng: random.Random,
) -> pd.DataFrame:
    sub = picks[picks["axis"] == axis]
    if sub.empty:
        return sub
    grades = list(AXIS_GRADES[axis])
    per_grade = max(1, per_axis // len(grades))
    parts = []
    for g in grades:
        cell = sub[sub["suggested_grade"] == g]
        if cell.empty:
            continue
        n = min(per_grade, len(cell))
        parts.append(cell.sample(n=n, random_state=rng.randint(0, 10_000)))
    if not parts:
        return sub.iloc[0:0]
    out = pd.concat(parts, ignore_index=True)
    # If short of `per_axis`, top up from any cell
    short = per_axis - len(out)
    if short > 0:
        leftover = sub[~sub.index.isin(out.index)]
        if not leftover.empty:
            out = pd.concat([out, leftover.sample(n=min(short, len(leftover)),
                                                  random_state=rng.randint(0, 10_000))],
                            ignore_index=True)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks",
                    default=str(GOLDEN_CACHE_DIR / "axis_grade_candidates__qwen3.5-122b.parquet"))
    ap.add_argument("--per-axis", type=int, default=20)
    ap.add_argument("--skip-axes", nargs="*", default=[])
    ap.add_argument("--cp-fix", action="store_true",
                    help="remap pre-swap CP1<->CP2 before sampling")
    ap.add_argument("--rng-seed", type=int, default=20260520)
    args = ap.parse_args()

    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
    sys_part, user_template = split_prompt(PROMPT_PATH.read_text(encoding="utf-8"))

    picks_path = Path(args.picks)
    if not picks_path.exists():
        sys.exit(f"missing picks parquet: {picks_path}")
    picks = pd.read_parquet(picks_path)

    if args.cp_fix:
        m = picks["axis"] == "CP"
        picks.loc[m, "suggested_grade"] = picks.loc[m, "suggested_grade"].map(_CP_SWAP)

    if args.skip_axes:
        picks = picks[~picks["axis"].isin(args.skip_axes)].copy()

    # add cluster label
    mf = load_manifest()[["topic_id", "czech_label"]]
    picks = picks.merge(mf, on="topic_id", how="left")

    rng = random.Random(args.rng_seed)
    rows_out = []
    for axis in AXIS_GRADES.keys():
        if axis in set(args.skip_axes or []):
            continue
        sample = sample_per_axis(picks, axis, per_axis=args.per_axis, rng=rng)
        if sample.empty:
            print(f"  [{axis:<4}] no anchors available — skipped")
            continue
        for r in sample.itertuples(index=False):
            user_msg, target_grades = render_user_message(
                axis=axis, anchor_query=r.query_text, anchor_grade=r.suggested_grade,
                user_template=user_template,
            )
            row = {
                "axis": axis,
                "anchor_grade": r.suggested_grade,
                "anchor_query": r.query_text,
                "topic_id": int(r.topic_id),
                "czech_label": getattr(r, "czech_label", "") or "",
                "target_grades": ", ".join(target_grades),
                "input": user_msg,
            }
            for col in ANNOTATION_COLS:
                row[col] = ""
            rows_out.append(row)
        print(f"  [{axis:<4}] {len(sample):>3} anchors sampled")

    if not rows_out:
        sys.exit("no anchors found; partial-Stage-2 parquet may not yet cover requested axes")

    df = pd.DataFrame(rows_out)
    csv_path = PLAYGROUND_DIR / "test_set.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"\n[csv] {len(df):,} rows → {csv_path.relative_to(REPO)}")

    schema_path = PLAYGROUND_DIR / "response_schema.json"
    schema = _strictify(SynthVariantSet.model_json_schema())
    import json as _json
    schema_path.write_text(
        _json.dumps({"name": "SynthVariantSet", "strict": True, "schema": schema},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[json] {schema_path.relative_to(REPO)}")

    sys_path = PLAYGROUND_DIR / "system_prompt.md"
    sys_path.write_text(
        "# SYSTEM prompt — paste this once into the OpenAI playground\n\n"
        "Source: `contrastive/prompts/synth_variant_v2.txt`\n\n"
        "Paste the block below verbatim into the system role. Then load `test_set.csv` "
        "and use the `input` column for each user message. Annotation columns at the "
        "right of the CSV are for your post-hoc evaluation.\n\n"
        "```\n" + sys_part + "\n```\n",
        encoding="utf-8",
    )
    print(f"[md ] {sys_path.relative_to(REPO)}")

    readme = PLAYGROUND_DIR / "README.md"
    counts_per_axis = df.groupby("axis").size().to_dict()
    counts_per_axis_grade = df.groupby(["axis", "anchor_grade"]).size().to_dict()
    cells_lines = "\n".join(
        f"- {ax} {gr}: {n}" for (ax, gr), n in sorted(counts_per_axis_grade.items())
    )
    readme.write_text(
        f"""# P2 synth-variant playground test set

Built {pd.Timestamp.now().date()} from `{picks_path.name}` ({len(picks):,} picks
in source parquet at time of build).

## Files

- `test_set.csv` — {len(df):,} anchors across {len(counts_per_axis)} axes/attributes.
  Columns:
    * `axis`, `anchor_grade`, `anchor_query`, `topic_id`, `czech_label` — context.
    * `target_grades` — which grades the LLM is asked to generate variants for.
    * `input` — the fully-rendered USER message (paste into the playground user role).
    * annotation columns ({", ".join(ANNOTATION_COLS)}) — fill in by hand.

- `system_prompt.md` — copy-paste SYSTEM message (same across all rows).

## How to run in the OpenAI playground evals UI

1. Open <https://platform.openai.com/playground> → Evals.
2. Create a new eval → import dataset → upload `test_set.csv`. The `input`
   column is the user-role message.
3. Paste the contents of `system_prompt.md` (between the ``` fences) into
   the system role.
4. Set:
   - model = `gpt-5.4-mini` (and a separate run for `gpt-5.4` for comparison)
   - response_format = `json_schema` strict (SynthVariantSet schema below)
   - `reasoning_effort` = `minimal`
   - `max_completion_tokens` = 600
5. Run the eval. Each row produces 1 JSON response (a `SynthVariantSet`).
6. Eyeball-score in the annotation columns. Rate quality 1–5; flag axes that
   need prompt revision before the Batch API run.

## Sample distribution per (axis, grade)

{cells_lines}

## Pydantic schema (for `response_format = json_schema`)

```python
class SynthVariant(BaseModel):
    target_grade: str   # must be in AXIS_GRADES[axis]
    text: str           # 1..2000 chars
    justification_cs: str  # 1..400 chars

class SynthVariantSet(BaseModel):
    variants: list[SynthVariant]   # max 4
```

When configuring the playground, use the strict-mode JSON schema built by
`contrastive.llm_client._strictify(SynthVariantSet.model_json_schema())`.

## After playground evaluation

Once you're happy with the prompt + model on the test set, run the production
Batch API path:

```
python -m contrastive.synth_variants_batch prepare ...
python -m contrastive.synth_variants_batch costreport
python -m contrastive.synth_variants_batch submit
# wait — internet not needed during processing
python -m contrastive.synth_variants_batch status
python -m contrastive.synth_variants_batch download
```
""",
        encoding="utf-8",
    )
    print(f"[md ] {readme.relative_to(REPO)}")


if __name__ == "__main__":
    main()
