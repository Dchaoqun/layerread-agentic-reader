"""Validated data contract for LayerRead v0.2 model output."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects fields outside the documented contract."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


ReferenceId = Annotated[str, Field(min_length=1, max_length=20)]
CompactListItem = Annotated[str, Field(min_length=1, max_length=400)]


class CitedText(StrictModel):
    text: str = Field(min_length=1, max_length=800)
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=12)


class InferredText(StrictModel):
    """AI-derived content that may have no direct source paragraph."""

    text: str = Field(min_length=1, max_length=800)
    paragraph_refs: list[ReferenceId] = Field(default_factory=list, max_length=12)


class CriterionScore(StrictModel):
    score: int = Field(ge=0, le=100)
    explanation: str = Field(min_length=1, max_length=600)
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=12)


class ReadingDecision(StrictModel):
    recommendation: Literal[
        "值得深读",
        "建议重点阅读",
        "阅读总结即可",
        "快速浏览",
        "建议跳过",
    ]
    overall_value: CriterionScore
    information_density: CriterionScore
    evidence_quality: CriterionScore
    original_analysis: CriterionScore
    goal_relevance: CriterionScore
    promotion_likelihood: CriterionScore
    fluff_likelihood: CriterionScore
    confidence: CriterionScore
    recommended_approach: str = Field(min_length=1, max_length=600)


class PromotionAssessment(StrictModel):
    probability: int = Field(ge=0, le=100)
    potential_targets: list[CompactListItem] = Field(default_factory=list, max_length=5)
    rationale: str = Field(min_length=1, max_length=800)
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=12)
    valuable_parts: list[CitedText] = Field(default_factory=list, max_length=5)


class FluffAssessment(StrictModel):
    effective_point_count: int = Field(ge=0)
    repetition: str = Field(min_length=1, max_length=600)
    title_overstates_content: str = Field(min_length=1, max_length=600)
    compressibility: int = Field(ge=0, le=100)
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=12)
    skippable_parts: list[CitedText] = Field(default_factory=list, max_length=8)


class StructureItem(StrictModel):
    order: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=600)
    # A structural section can legitimately span far more paragraphs than a
    # single claim or score. Keep a bounded response while allowing long-form
    # articles to cite the complete section instead of failing validation.
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=64)


class KnowledgePoint(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    definition: str = Field(min_length=1, max_length=600)
    plain_explanation: str = Field(min_length=1, max_length=800)
    example: str = Field(min_length=1, max_length=600)
    applicability: str = Field(min_length=1, max_length=600)
    common_misconceptions: list[CompactListItem] = Field(
        default_factory=list,
        max_length=5,
    )
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=12)


class ClaimAnalysis(StrictModel):
    claim: str = Field(min_length=1, max_length=700)
    classification: Literal[
        "事实",
        "观点",
        "假设",
        "推断",
        "预测",
        "案例",
        "营销主张",
        "行动建议",
    ]
    evidence: str = Field(min_length=1, max_length=1_000)
    evidence_type: str = Field(min_length=1, max_length=200)
    evidence_strength: Literal["强", "中", "弱", "无直接证据"]
    logical_leaps: list[CompactListItem] = Field(default_factory=list, max_length=5)
    paragraph_refs: list[ReferenceId] = Field(min_length=1, max_length=12)


class CoreAnalysis(StrictModel):
    core_question: str = Field(min_length=1, max_length=400)
    one_sentence_conclusion: str = Field(min_length=1, max_length=500)
    author_intended_view: str = Field(min_length=1, max_length=800)
    structure: list[StructureItem] = Field(min_length=1, max_length=12)
    knowledge_points: list[KnowledgePoint] = Field(min_length=3, max_length=6)
    claims: list[ClaimAnalysis] = Field(min_length=1, max_length=10)
    limitations: list[CitedText] = Field(min_length=1, max_length=6)
    counterarguments: list[InferredText] = Field(default_factory=list, max_length=6)
    implicit_assumptions: list[InferredText] = Field(default_factory=list, max_length=6)
    possible_counterexamples: list[InferredText] = Field(default_factory=list, max_length=6)
    applicability_boundaries: list[InferredText] = Field(default_factory=list, max_length=8)
    items_to_verify: list[InferredText] = Field(default_factory=list, max_length=8)


class Explanation(StrictModel):
    kind: Literal["AI解释", "AI推断"]
    text: str = Field(min_length=1, max_length=1_000)
    paragraph_refs: list[ReferenceId] = Field(default_factory=list, max_length=12)


class ReferenceDiagnosticItem(StrictModel):
    """One model-supplied reference entry and its normalization outcome."""

    location: str
    original: str
    normalized_refs: list[str] = Field(default_factory=list)
    valid_refs: list[str] = Field(default_factory=list)
    invalid_refs: list[str] = Field(default_factory=list)
    statement: str = ""
    status: Literal["valid", "auto_corrected", "partially_invalid", "invalid"]
    category: Literal[
        "exact",
        "format_corrected",
        "split_list",
        "expanded_range",
        "partially_invalid",
        "nonexistent",
        "invalid_range",
        "unparseable",
    ]
    reason: str


class ReferenceDiagnostics(StrictModel):
    """Aggregate reference quality after conservative local normalization."""

    total_mentions: int = Field(ge=0)
    exact_valid_mentions: int = Field(ge=0)
    corrected_valid_mentions: int = Field(ge=0)
    invalid_mentions: int = Field(ge=0)
    nonexistent_mentions: int = Field(ge=0)
    unparseable_mentions: int = Field(ge=0)
    affected_locations: int = Field(ge=0)
    zero_valid_locations: int = Field(ge=0)
    unique_valid_references: list[str] = Field(default_factory=list)
    auto_corrections: dict[str, list[str]] = Field(default_factory=dict)
    invalid_references: list[str] = Field(default_factory=list)
    items: list[ReferenceDiagnosticItem] = Field(default_factory=list)
    risk_level: Literal["none", "low", "medium", "high"]
    risk_summary: str

    @property
    def invalid_rate(self) -> float:
        if self.total_mentions == 0:
            return 0.0
        return self.invalid_mentions / self.total_mentions


class AnalysisPayload(StrictModel):
    """The exact JSON object requested from the model."""

    reading_decision: ReadingDecision
    promotion_assessment: PromotionAssessment
    fluff_assessment: FluffAssessment
    core_analysis: CoreAnalysis
    explanations: list[Explanation] = Field(default_factory=list, max_length=8)


class Analysis(StrictModel):
    """Stored analysis record, including provenance and validation state."""

    article_id: str
    prompt_version: str
    model_name: str
    reading_decision: ReadingDecision
    core_analysis: CoreAnalysis
    explanations: list[Explanation]
    promotion_assessment: PromotionAssessment
    fluff_assessment: FluffAssessment
    raw_response: str
    validation_status: Literal[
        "valid",
        "valid_with_auto_corrections",
        "repaired",
        "valid_with_invalid_references",
    ]
    invalid_references: list[str] = Field(default_factory=list)
    initial_reference_diagnostics: ReferenceDiagnostics | None = None
    reference_diagnostics: ReferenceDiagnostics | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
