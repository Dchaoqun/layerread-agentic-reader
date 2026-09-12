"""Validated data contracts for LayerRead v0.4 article chat."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from layerread.analysis_schema import StrictModel


InformationType = Literal["原文内容", "AI解释", "AI推断", "无法确认"]
MessageRole = Literal["user", "assistant"]


class ChatReplyPayload(StrictModel):
    """The exact structured reply requested from the model."""

    content: str = Field(min_length=1)
    source_paragraphs: list[str] = Field(default_factory=list)
    information_types: list[InformationType] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_label(self) -> "ChatReplyPayload":
        if "原文内容" in self.information_types and not self.source_paragraphs:
            raise ValueError("标记为原文内容的回答必须提供段落引用。")
        return self


class ChatMessage(StrictModel):
    id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    role: MessageRole
    content: str = Field(min_length=1)
    source_paragraphs: list[str] = Field(default_factory=list)
    information_types: list[InformationType] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SavedInsight(StrictModel):
    id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    source_message_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    user_note: str = ""
    selected_for_export: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationDigest(StrictModel):
    new_understandings: list[str]
    user_focus_questions: list[str]
    corrected_understandings: list[str]
    unresolved_questions: list[str]


class ChatSession(StrictModel):
    article_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)
    saved_insights: list[SavedInsight] = Field(default_factory=list)
    digest: ConversationDigest | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
