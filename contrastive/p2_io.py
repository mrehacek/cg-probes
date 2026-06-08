"""P2 safety-axis helpers — MU / PU / ET only.

The P2 contrastive-pair stage targets the three locked safety axes of the
golden set (Medical Urgency, Psychological Urgency, Emotional Distress
topic-based), each ordinal 0/1/2. This module is the single place that knows:

  * which axes are in scope (`SAFETY_AXES`)
  * their Czech long names (for the synth prompt header)
  * how to load the canonical rubric markdown (`golden/prompts/rubrics_v2/`)
  * how to convert between the axis-prefixed grade tokens used in
    `picks_raw.parquet` / `schemas.AXIS_GRADES` (e.g. "MU2") and the bare
    ordinal ints the probes consume (0/1/2).

It deliberately reuses `golden.pipeline_v2._io.split_prompt` and the
`schemas.AXIS_GRADES` token tuples rather than redefining either, so the P2
pipeline stays consistent with the golden-set construction it builds on.
"""

from __future__ import annotations

import sys
from pathlib import Path

# contrastive/p2_io.py -> parents[1] == repo root (cikm-ds/)
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from contrastive.schemas import AXIS_GRADES  # noqa: E402
from golden.pipeline_v2._io import split_prompt  # noqa: E402

# --- Axis registry -----------------------------------------------------------

SAFETY_AXES: list[str] = ["MU", "PU", "ET"]

AXIS_LONG_NAMES_CS: dict[str, str] = {
    "MU": "Lékařská naléhavost",
    "PU": "Psychická naléhavost",
    "ET": "Emoční zátěž tématu",
}

# Per-axis short English construct names (for reports / English-language probes).
AXIS_LONG_NAMES_EN: dict[str, str] = {
    "MU": "Medical Urgency",
    "PU": "Psychological Urgency",
    "ET": "Topic Sensitivity",
}

# Column in picks_raw.parquet holding each axis's grade token.
AXIS_GRADE_COL: dict[str, str] = {axis: f"{axis.lower()}_grade" for axis in SAFETY_AXES}

RUBRICS_DIR = REPO / "golden" / "prompts" / "rubrics_v2"


def grades_for(axis: str) -> tuple[str, ...]:
    """The ordered grade-token tuple for an axis, e.g. ('MU0','MU1','MU2')."""
    if axis not in SAFETY_AXES:
        raise ValueError(f"{axis!r} not a safety axis; expected one of {SAFETY_AXES}")
    return AXIS_GRADES[axis]


def axis_grades_csv(axis: str) -> str:
    """Comma-joined grade tokens for the `{axis_grades_csv}` prompt placeholder."""
    return ", ".join(grades_for(axis))


def token_to_int(token: str) -> int:
    """'MU2' -> 2. The ordinal is the trailing digit of the axis-prefixed token."""
    digit = token[-1]
    if not digit.isdigit():
        raise ValueError(f"grade token {token!r} does not end in an ordinal digit")
    return int(digit)


def int_to_token(axis: str, grade: int) -> str:
    """(axis='MU', grade=2) -> 'MU2'. Validates against AXIS_GRADES."""
    token = f"{axis}{grade}"
    if token not in grades_for(axis):
        raise ValueError(f"{token!r} not a valid grade for axis {axis!r}")
    return token


def load_axis_rubric(axis: str) -> str:
    """Full markdown rubric text for one safety axis (verbatim, incl. anchors)."""
    path = RUBRICS_DIR / f"{axis}.md"
    if not path.exists():
        raise FileNotFoundError(f"rubric not found for axis {axis!r}: {path}")
    return path.read_text(encoding="utf-8").strip()


def split_prompt_file(path: Path) -> tuple[str, str]:
    """(system, user) from a `[SYSTEM]/[USER]` prompt file."""
    return split_prompt(path.read_text(encoding="utf-8"))


def load_synth_prompt() -> tuple[str, str]:
    """(system, user) templates for the axis-agnostic synth_variant_v2 prompt."""
    path = REPO / "contrastive" / "prompts" / "synth_variant_v2.txt"
    return split_prompt_file(path)


# --- P2 cache layout (absolute, REPO-anchored so scripts run from anywhere) ---

P2_CACHE = REPO / "contrastive" / "cache"
ANCHORS_SAFETY = P2_CACHE / "anchors_safety.parquet"


def synth_variants_path(axis: str) -> Path:
    return P2_CACHE / f"synth_variants_{axis}.parquet"


def probe_dataset_path(axis: str) -> Path:
    return P2_CACHE / f"probe_dataset_{axis}.parquet"


def ensure_cache() -> None:
    P2_CACHE.mkdir(parents=True, exist_ok=True)
