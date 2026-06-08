"""Pydantic v2 schemas — the single source of truth for prompt-output JSON
schemas and final parquet column types.

Levels are kept as `str` (not `Literal`) so prompt-injected typos surface as
verifier-stage rejects rather than Pydantic ValidationErrors that lose the raw
LLM output. Validators below enforce the allowed values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Legacy meeting-pack constants — kept only for the head-nurse meeting flow
# (anchor_curate.py, meeting_pack_urgency.py). The CIKM pipeline below uses
# the v4 axis tokens in AXIS_GRADES exclusively.
URGENCY_LEVELS = ("L0", "L1", "L2", "L3")
EMOTIONAL_LEVELS = ("E0", "E1", "E2")
ALL_LEVELS = URGENCY_LEVELS + EMOTIONAL_LEVELS

# P3 golden-set pipeline — v4 axes (axes-snapshot-20260520.md).
# Grades map 1:1 to annotation-app goldValue integers (level index 0..N-1).
#
# Migration notes vs v3 (20260519 snapshot):
#   * U: 4 grades → 3 (U3 merged into U2 per head-nurse triage feedback).
#   * EV/ET/EC: 4 grades → 3 (E3 suicide tier moved to new PU attribute).
#     Grade tokens are now AXIS-PREFIXED (EV0/EV1/EV2, ET0/ET1/ET2, EC0/EC1/EC2)
#     instead of the shared E0/E1/E2/E3 alphabet, so the same token cannot be
#     re-used across emotional axes — fixes ambiguous-grade cache collisions.
#   * PC axis (3-level Caregiver Perspective) → renamed CP attribute. Initial
#     v4 mapping was CP0 impersonal / CP1 caregiver / CP2 self; on 2026-05-20
#     CP1 and CP2 were SWAPPED (annotation-app migration
#     20260520120000_swap_c_axis_self_caregiver) so the canonical encoding now
#     matches studio.ts AXES['C']: CP0 undetermined / CP1 self / CP2 caregiver.
#     Any cached parquet outputs from before 2026-05-20 hold the pre-swap
#     encoding and must be re-run before assembling new packages.
#   * NEW PU attribute (Psychological Urgency, PU0/PU1/PU2) — absorbs the
#     suicide-risk / extreme-distress content that v3 placed in E3.
#   * NEW EHR attribute (EHR-personalization requirement, EHR0/EHR1/EHR2).
#
# Schema-level the seven "axes" + the three "attributes" are interchangeable:
# all are graded ordinal classifications. Listed together so the Stage 1/2
# picker pipeline treats them uniformly.
AXIS_GRADES: dict[str, tuple[str, ...]] = {
    "U":   ("U0", "U1", "U2"),
    "EV":  ("EV0", "EV1", "EV2"),
    "ET":  ("ET0", "ET1", "ET2"),
    "EC":  ("EC0", "EC1", "EC2"),
    "P":   ("P0", "P1", "P2"),
    "A":   ("A0", "A1", "A2"),
    "CP":  ("CP0", "CP1", "CP2"),
    "PU":  ("PU0", "PU1", "PU2"),
    "EHR": ("EHR0", "EHR1", "EHR2"),
    # MU (Medical Urgency, somatic) — the P2/P4 safety-axis pivot. v4 used "U";
    # the golden-set + benchmark pipeline grades MU/PU/ET (MU0/MU1/MU2).
    "MU":  ("MU0", "MU1", "MU2"),
}
ALL_GRADE_TOKENS = tuple(sorted({g for grades in AXIS_GRADES.values() for g in grades}))

ClinicalRelevance = Literal[
    "oncology-core", "oncology-adjacent", "navigational", "non-clinical"
]
Potential = Literal["low", "medium", "high"]
BaseSource = Literal["anchor", "real_search", "real_iec"]
DatasetSource = Literal["anchor", "real_search", "real_iec", "synthetic", "adversarial"]
Axis = Literal["urgency", "emotional"]
Split = Literal["train", "test"]
GoldenAxis = Literal["U", "EV", "ET", "EC", "P", "A", "CP", "PU", "EHR"]
# Legacy binary caregiver flag kept for backward-compat with the Stage 3
# voice classifier (03_caregiver_voice.py). The official replacement is the
# 3-level CP attribute (see CP.md) — routed through the standard picker.
QueryVoice = Literal["patient", "caregiver", "ambiguous"]
ClusterVoice = Literal["patient", "caregiver", "mixed"]


class ClusterCard(BaseModel):
    """Internal-only cluster triage card. Never shown to clinicians (per
    user decision §2.1 — clinicians see raw queries, not LLM ratings)."""

    topic_id: int
    n_queries: int
    czech_label: str = Field(..., max_length=80)
    summary_cs: str
    clinical_relevance: ClinicalRelevance
    urgency_potential: Potential
    emotional_potential: Potential
    suspected_anchor_levels: list[str] = Field(default_factory=list)
    rationale_cs: str

    @field_validator("suspected_anchor_levels")
    @classmethod
    def _check_levels(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in ALL_LEVELS]
        if bad:
            raise ValueError(f"unknown level(s): {bad}; allowed={ALL_LEVELS}")
        return v


class AnchorCandidate(BaseModel):
    """One candidate row in the meeting CSV — clinician fills `rater_*`."""

    candidate_id: str
    source: Literal["search", "iec"]
    text: str
    topic_id: int | None = None
    czech_label: str | None = None
    heuristic_match: str
    proposed_level: str
    clicks: int | None = None
    char_len: int
    rater_level: str | None = None
    rater_confidence: int | None = None
    rater_notes: str | None = None


class Variant(BaseModel):
    text: str
    target_level: str
    justification_cs: str

    @field_validator("target_level")
    @classmethod
    def _check_level(cls, v: str) -> str:
        if v not in ALL_LEVELS:
            raise ValueError(f"unknown level: {v}; allowed={ALL_LEVELS}")
        return v


class ContrastivePair(BaseModel):
    pair_id: str
    topic_id: int
    base_text: str
    base_level: str
    base_source: BaseSource
    variants: list[Variant]

    @field_validator("base_level")
    @classmethod
    def _check_level(cls, v: str) -> str:
        if v not in ALL_LEVELS:
            raise ValueError(f"unknown base_level: {v}; allowed={ALL_LEVELS}")
        return v


class UrgencyVariantSet(BaseModel):
    """Pre-meeting variant pack: for one base query at a proposed urgency
    level, the LLM emits 3 sibling variants covering the 3 *other* levels.
    Used by `meeting_pack_urgency.py` to scaffold the head-nurse session."""

    variants: list[Variant] = Field(..., min_length=3, max_length=3)

    @field_validator("variants")
    @classmethod
    def _three_distinct_urgency(cls, v: list[Variant]) -> list[Variant]:
        levels = [x.target_level for x in v]
        bad = [l for l in levels if l not in URGENCY_LEVELS]
        if bad:
            raise ValueError(f"non-urgency target_level(s): {bad}")
        if len(set(levels)) != 3:
            raise ValueError(f"variants must cover 3 distinct levels, got: {levels}")
        return v


class JudgeVerdict(BaseModel):
    """Second-pass LLM judge labels each item without seeing the synthesizer's
    target_level — agreement is the soft-label noise estimator."""

    item_id: str
    judged_level: str
    judge_confidence: int = Field(..., ge=1, le=5)
    judge_notes_cs: str

    @field_validator("judged_level")
    @classmethod
    def _check_level(cls, v: str) -> str:
        if v not in ALL_LEVELS:
            raise ValueError(f"unknown judged_level: {v}; allowed={ALL_LEVELS}")
        return v


# --- P3 golden-set schemas ---------------------------------------------------


class ClusterAxisSpan(BaseModel):
    """Stage 1: per (cluster, axis) — does this cluster plausibly host queries
    spanning multiple grades on this axis without breaking topic coherence?

    The axis identity is NOT in the schema: each call is single-axis, and the
    pipeline records the axis alongside the result. `predicted_grades_present`
    members must be drawn from AXIS_GRADES[axis] — validated downstream where
    the axis is known.
    """

    spans_grades: bool
    predicted_grades_present: list[str] = Field(..., max_length=4)
    justification_cs: str = Field(..., min_length=1, max_length=600)

    @field_validator("predicted_grades_present")
    @classmethod
    def _grades_known_alphabet(cls, v: list[str]) -> list[str]:
        bad = [g for g in v if g not in ALL_GRADE_TOKENS]
        if bad:
            raise ValueError(f"unknown grade token(s): {bad}; allowed={ALL_GRADE_TOKENS}")
        return v


class ClusterCaregiverVoice(BaseModel):
    """Stage 2 cluster-level pre-pass: is this cluster mostly patient-voice,
    caregiver-voice, or mixed? Cheap pre-filter — only `mixed`/`caregiver`
    clusters get the per-query pass."""

    majority_voice: ClusterVoice
    mix_estimate: float = Field(..., ge=0.0, le=1.0,
                                description="Estimated caregiver fraction in [0,1].")
    notes_cs: str = Field(..., max_length=400)


class QueryCaregiverVoice(BaseModel):
    """Stage 2 query-level: per-query voice classification."""

    voice: QueryVoice
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale_cs: str = Field(..., max_length=300)


class AxisGradePick(BaseModel):
    """One picked query for a (cluster, axis) tuple at a specific grade.

    `suggested_grade` must lie in AXIS_GRADES[axis] — validated by the
    pipeline once the axis is known.
    """

    query_text: str = Field(..., min_length=1, max_length=2000)
    suggested_grade: str
    justification_cs: str = Field(..., min_length=1, max_length=400)

    @field_validator("suggested_grade")
    @classmethod
    def _grade_known_alphabet(cls, v: str) -> str:
        if v not in ALL_GRADE_TOKENS:
            raise ValueError(f"unknown grade token: {v}; allowed={ALL_GRADE_TOKENS}")
        return v


class AxisGradeCandidate(BaseModel):
    """Stage 3 output for one (cluster, axis): the LLM's picked queries
    spanning the axis's grades. `picks` may be empty if the cluster has no
    grade-distinguishing material despite the span-flag (rare). Per-axis
    deduplication of query_text is enforced downstream."""

    picks: list[AxisGradePick] = Field(..., max_length=16)


# --- P2 contrastive-pair schemas ---------------------------------------------


class SynthVariant(BaseModel):
    """One LLM-rewritten variant of a source anchor query at a *different* grade
    on the same axis. Topic preserved (organ / diagnosis / procedure / modality);
    register preserved (Czech patient search-query style, lowercase, typos OK).
    The axis identity is NOT in the schema — each call is single-axis and the
    pipeline records the axis alongside the result. `target_grade` must lie in
    AXIS_GRADES[axis] — validated downstream where the axis is known."""

    target_grade: str
    text: str = Field(..., min_length=1, max_length=2000)
    justification_cs: str = Field(..., min_length=1, max_length=400)

    @field_validator("target_grade")
    @classmethod
    def _grade_known_alphabet(cls, v: str) -> str:
        if v not in ALL_GRADE_TOKENS:
            raise ValueError(f"unknown grade token: {v}; allowed={ALL_GRADE_TOKENS}")
        return v


class SynthVariantSet(BaseModel):
    """P2 Stage B output for one (anchor_query, axis): a list of variant
    rewrites at grades other than the anchor's own grade. `variants` may be
    shorter than the number of other grades if some are not achievable while
    preserving topic — better to drop than to force."""

    variants: list[SynthVariant] = Field(..., max_length=4)
