"""Pydantic v2 schemas for the MU/PU/ET golden-set pipeline.

These are the strict-JSON-schema contracts for the three LLM stages:
  * PickerResponse      — step 1 (cluster picker, gpt-5.4)
  * VerifierResponse    — step 3 (independent verifier, gpt-5.4-mini)
  * SynthesizerResponse — step 4 (conditional backfill, gpt-5.4)

Grade fields use `Literal` enums so OpenAI strict mode constrains the model to a
valid token at decode time (no post-hoc grade typos to clean up). Free-text
fields are bounded but allow empty strings, because strict mode marks every
property `required` — the model must emit the key even when it has nothing to
say (e.g. `skipped_reason_cs` when `picks` is non-empty).

This is a fresh token set: the new axes are MU/PU/ET, NOT the retired 9-axis
ontology in `contrastive.schemas.AXIS_GRADES` (do not import that here).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- Grade alphabets ---------------------------------------------------------

MU_GRADES: tuple[str, ...] = ("MU0", "MU1", "MU2")
PU_GRADES: tuple[str, ...] = ("PU0", "PU1", "PU2")
ET_GRADES: tuple[str, ...] = ("ET0", "ET1", "ET2")

AXES: tuple[str, ...] = ("MU", "PU", "ET")
AXIS_GRADES: dict[str, tuple[str, ...]] = {
    "MU": MU_GRADES,
    "PU": PU_GRADES,
    "ET": ET_GRADES,
}
ALL_GRADES: tuple[str, ...] = MU_GRADES + PU_GRADES + ET_GRADES

MUGrade = Literal["MU0", "MU1", "MU2"]
PUGrade = Literal["PU0", "PU1", "PU2"]
ETGrade = Literal["ET0", "ET1", "ET2"]
TargetGrade = Literal[
    "MU0", "MU1", "MU2", "PU0", "PU1", "PU2", "ET0", "ET1", "ET2"
]
TargetAxis = Literal["MU", "PU", "ET"]

Confidence = int  # validated 1..5 via Field(ge=1, le=5)


# --- Step 1: picker ----------------------------------------------------------

class PickItem(BaseModel):
    """One picked, verbatim query graded on all three axes."""

    query_text: str = Field(..., min_length=1, max_length=2000)
    mu_grade: MUGrade
    pu_grade: PUGrade
    et_grade: ETGrade
    mu_confidence: int = Field(..., ge=1, le=5)
    pu_confidence: int = Field(..., ge=1, le=5)
    et_confidence: int = Field(..., ge=1, le=5)
    notes_cs: str = Field("", max_length=800)


class PickerResponse(BaseModel):
    """All picks for one cluster. `picks` may be empty (then explain why)."""

    picks: list[PickItem] = Field(default_factory=list, max_length=20)
    skipped_reason_cs: str = Field("", max_length=600)


# --- Step 3: verifier --------------------------------------------------------

class VerifierResponse(BaseModel):
    """Independent re-grade of ONE query (no cluster context, no picker label)."""

    mu_grade: MUGrade
    pu_grade: PUGrade
    et_grade: ETGrade
    mu_confidence: int = Field(..., ge=1, le=5)
    pu_confidence: int = Field(..., ge=1, le=5)
    et_confidence: int = Field(..., ge=1, le=5)
    mu_axis_applicable: bool
    pu_axis_applicable: bool
    et_axis_applicable: bool
    notes_cs: str = Field("", max_length=800)


# --- Step 4: backfill synthesizer (conditional) ------------------------------

class SynthItem(BaseModel):
    """One synthetic Czech patient-style query targeting a specific cell."""

    query_text: str = Field(..., min_length=1, max_length=2000)
    target_grade: TargetGrade
    target_axis: TargetAxis
    rationale_cs: str = Field(..., min_length=1, max_length=600)
    clinical_plausibility_cs: str = Field(..., min_length=1, max_length=600)


class SynthesizerResponse(BaseModel):
    """Synthetic queries for one (cluster, target cell). May be empty."""

    generated: list[SynthItem] = Field(default_factory=list, max_length=5)
    could_not_generate_reason_cs: str = Field("", max_length=600)
