"""LLM-judge schema for the llm-annotator subproject."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from contrastive.schemas import ALL_GRADE_TOKENS


class AxisJudgeVerdict(BaseModel):
    """One LLM verdict for a (query, axis) pair.

    `grade` is the prefixed token from `AXIS_GRADES[axis]` (e.g. "MU1", "PU2",
    "ET0"). The runner maps it to the annotation-app's integer value (0..max)
    once the axis is known. `confidence` is reported by the model itself —
    treat it as a soft signal, not calibrated.
    """

    grade: str = Field(..., min_length=1, max_length=8)
    confidence: int = Field(..., ge=1, le=5)
    note_cs: str = Field(..., min_length=1, max_length=400)

    @field_validator("grade")
    @classmethod
    def _grade_known_alphabet(cls, v: str) -> str:
        if v not in ALL_GRADE_TOKENS:
            raise ValueError(f"unknown grade token: {v}; allowed={ALL_GRADE_TOKENS}")
        return v