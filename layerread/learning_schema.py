"""Validated data contracts for LayerRead v0.3 active learning."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from layerread.analysis_schema import ReferenceDiagnostics, StrictModel


QuestionType = Literal["核心回忆", "Teach-back", "应用题", "批判性问题"]
LearningObjective = Literal[
    "文章问题",
    "核心观点",
    "证据链",
    "概念讲授",
    "知识迁移",
    "批判性评估",
]
MasteryStatus = Literal["已掌握", "基本理解", "存在缺口", "需要复习"]
CriticalDimension = Literal["隐含假设", "薄弱证据", "可能反例", "事实与预测"]


class LearningQuestion(StrictModel):
    question_id: str = Field(min_length=1)
    question_type: QuestionType
    learning_objective: LearningObjective
    prompt: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    evaluation_points: list[str] = Field(min_length=1)
    focus_concept: str = ""
    scenario: str = ""
    critical_dimensions: list[CriticalDimension] = Field(default_factory=list)
    paragraph_refs: list[str] = Field(min_length=1)


class LearningQuestionSet(StrictModel):
    recall_questions: list[LearningQuestion] = Field(min_length=3)
    teach_back_question: LearningQuestion
    application_question: LearningQuestion
    critical_question: LearningQuestion

    @model_validator(mode="after")
    def validate_question_roles_and_ids(self) -> "LearningQuestionSet":
        if any(question.question_type != "核心回忆" for question in self.recall_questions):
            raise ValueError("recall_questions 只能包含核心回忆题。")
        recall_objectives = {
            question.learning_objective for question in self.recall_questions
        }
        if recall_objectives != {"文章问题", "核心观点", "证据链"}:
            raise ValueError("核心回忆题必须完整覆盖文章问题、核心观点和证据链。")
        expected = (
            (self.teach_back_question, "Teach-back", "概念讲授"),
            (self.application_question, "应用题", "知识迁移"),
            (self.critical_question, "批判性问题", "批判性评估"),
        )
        for question, expected_type, expected_objective in expected:
            if question.question_type != expected_type:
                raise ValueError(f"{question.question_id} 必须是 {expected_type}。")
            if question.learning_objective != expected_objective:
                raise ValueError(
                    f"{question.question_id} 的学习目标必须是 {expected_objective}。"
                )

        if not self.teach_back_question.focus_concept:
            raise ValueError("Teach-back 必须指定要讲授的核心概念。")
        if not self.teach_back_question.scenario:
            raise ValueError("Teach-back 必须提供向陌生同事讲解的场景。")
        if not self.application_question.scenario:
            raise ValueError("应用题必须提供知识迁移场景。")
        if set(self.critical_question.critical_dimensions) != {
            "隐含假设",
            "薄弱证据",
            "可能反例",
            "事实与预测",
        }:
            raise ValueError("批判性问题必须覆盖假设、证据、反例和事实/预测辨析。")

        identifiers = [question.question_id for question in self.all_questions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("学习题目的 question_id 必须唯一。")
        return self

    @property
    def all_questions(self) -> list[LearningQuestion]:
        return [
            *self.recall_questions,
            self.teach_back_question,
            self.application_question,
            self.critical_question,
        ]


class QuestionFeedback(StrictModel):
    question_id: str = Field(min_length=1)
    answer_excerpt: str = Field(min_length=1)
    correct_parts: list[str] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)
    inaccuracies: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(min_length=1)
    concept_accuracy: str = ""
    causal_reasoning: str = ""
    mastery_status: MasteryStatus
    paragraph_refs: list[str] = Field(min_length=1)


class ConceptMastery(StrictModel):
    concept_title: str = Field(min_length=1)
    status: MasteryStatus
    evidence: str = Field(min_length=1)
    review_advice: str = Field(min_length=1)
    paragraph_refs: list[str] = Field(min_length=1)


class ReviewCard(StrictModel):
    question: str = Field(min_length=1)
    short_answer: str = Field(min_length=1)
    detailed_explanation: str = Field(min_length=1)
    paragraph_refs: list[str] = Field(min_length=1)


class LearningFeedback(StrictModel):
    question_feedback: list[QuestionFeedback] = Field(min_length=1)
    mastery_status: list[ConceptMastery] = Field(min_length=1)
    review_cards: list[ReviewCard] = Field(min_length=1)
    overall_review_advice: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_feedback_ids(self) -> "LearningFeedback":
        identifiers = [item.question_id for item in self.question_feedback]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("每道题只能有一份反馈。")
        concepts = [item.concept_title for item in self.mastery_status]
        if len(concepts) != len(set(concepts)):
            raise ValueError("每个核心概念只能有一份掌握状态。")
        return self


class LearningSession(StrictModel):
    article_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    question_prompt_version: str = Field(min_length=1)
    feedback_prompt_version: str = Field(min_length=1)
    questions: LearningQuestionSet
    user_answers: dict[str, str] = Field(default_factory=dict)
    feedback: LearningFeedback | None = None
    mastery_status: list[ConceptMastery] = Field(default_factory=list)
    review_cards: list[ReviewCard] = Field(default_factory=list)
    question_raw_response: str
    feedback_raw_response: str = ""
    question_reference_diagnostics: ReferenceDiagnostics | None = None
    feedback_reference_diagnostics: ReferenceDiagnostics | None = None
    feedback_answer_fingerprint: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
