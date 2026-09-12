"""Article-grounded chat, insight management, and conversation refinement."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
import re
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import ValidationError

from layerread.analysis_schema import Analysis, StrictModel
from layerread.analyzer import ChatClient, OpenAICompatibleClient
from layerread.article import ArticleDraft
from layerread.chat_schema import (
    ChatMessage,
    ChatReplyPayload,
    ChatSession,
    ConversationDigest,
    SavedInsight,
)
from layerread.config import ModelSettings
from layerread.learning_schema import LearningSession
from layerread.prompts import active_prompt_version, prompt_text
from layerread.references import normalize_reference_values


CHAT_PROMPT_VERSION = "v0.4.0-chat"
DIGEST_PROMPT_VERSION = "v0.4.0-digest"
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARACTERS = 12_000
_INLINE_REFERENCE_PATTERN = re.compile(r"\[P\d+\]")
SchemaT = TypeVar("SchemaT", bound=StrictModel)


class ChatError(RuntimeError):
    """Safe, user-facing article-chat failure."""


def create_chat_session(analysis: Analysis, settings: ModelSettings) -> ChatSession:
    return ChatSession(
        article_id=analysis.article_id,
        model_name=settings.model,
        prompt_version=active_prompt_version(CHAT_PROMPT_VERSION),
    )


def _allowed_references(article: ArticleDraft) -> list[str]:
    return [paragraph.partition(" ")[0] for paragraph in article.numbered_paragraphs]


def _chat_schema(article: ArticleDraft) -> dict[str, Any]:
    schema = ChatReplyPayload.model_json_schema()
    schema["properties"]["source_paragraphs"]["items"] = {
        "type": "string",
        "enum": _allowed_references(article),
    }
    return schema


def _analysis_context(analysis: Analysis) -> dict[str, Any]:
    core = analysis.core_analysis
    return {
        "core_question": core.core_question,
        "one_sentence_conclusion": core.one_sentence_conclusion,
        "author_intended_view": core.author_intended_view,
        "knowledge_points": [
            point.model_dump(mode="json") for point in core.knowledge_points
        ],
        "claims": [claim.model_dump(mode="json") for claim in core.claims],
        "limitations": [item.model_dump(mode="json") for item in core.limitations],
        "counterarguments": [
            item.model_dump(mode="json") for item in core.counterarguments
        ],
        "items_to_verify": [
            item.model_dump(mode="json") for item in core.items_to_verify
        ],
    }


def _learning_context(session: LearningSession | None) -> dict[str, Any]:
    if session is None:
        return {}
    question_map = {
        question.question_id: question.prompt
        for question in session.questions.all_questions
    }
    answered = [
        {
            "question": question_map.get(question_id, question_id),
            "answer": answer,
        }
        for question_id, answer in session.user_answers.items()
        if answer.strip()
    ]
    feedback = (
        session.feedback.model_dump(mode="json")
        if session.feedback is not None
        else None
    )
    return {"user_answers": answered, "ai_feedback": feedback}


def _recent_history(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    selected: list[ChatMessage] = []
    used_characters = 0
    for message in reversed(messages):
        if len(selected) >= MAX_HISTORY_MESSAGES:
            break
        content_length = len(message.content)
        if selected and used_characters + content_length > MAX_HISTORY_CHARACTERS:
            break
        selected.append(message)
        used_characters += content_length
    selected.reverse()
    return [
        {
            "role": message.role,
            "content": message.content,
            "source_paragraphs": message.source_paragraphs,
            "information_types": message.information_types,
        }
        for message in selected
    ]


def build_chat_messages(
    article: ArticleDraft,
    analysis: Analysis,
    learning_session: LearningSession | None,
    session: ChatSession,
    question: str,
) -> list[dict[str, str]]:
    """Build a bounded, injection-resistant prompt for the next question."""

    question = question.strip()
    if not question:
        raise ChatError("请输入问题后再发送。")
    if session.article_id != analysis.article_id:
        raise ChatError("当前 Chat 与文章分析不一致，请重新开始对话。")

    context = {
        "article_analysis": _analysis_context(analysis),
        "learning": _learning_context(learning_session),
        "saved_insights": [
            {
                "content": insight.content,
                "user_note": insight.user_note,
            }
            for insight in session.saved_insights
        ],
        "recent_chat_history": _recent_history(session.messages),
    }
    return [
        {
            "role": "system",
            "content": prompt_text("chat"),
        },
        {
            "role": "user",
            "content": (
                f"合法段落编号：{', '.join(_allowed_references(article))}\n\n"
                "JSON Schema：\n"
                f"{json.dumps(_chat_schema(article), ensure_ascii=False)}\n\n"
                "受限上下文：\n"
                f"{json.dumps(context, ensure_ascii=False)}\n\n"
                f"编号正文：\n{article.numbered_text}\n\n"
                f"用户的新问题：\n{question}"
            ),
        },
    ]


def _extract_json(raw_response: str) -> Any:
    text = raw_response.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _parse_schema(raw_response: str, schema_type: type[SchemaT]) -> SchemaT:
    try:
        return schema_type.model_validate(_extract_json(raw_response))
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise ChatError("模型输出未通过结构校验。") from exc


def _normalize_chat_reply(
    payload: ChatReplyPayload,
    article: ArticleDraft,
) -> ChatReplyPayload:
    source_result = normalize_reference_values(payload.source_paragraphs, article)
    inline_refs = list(dict.fromkeys(_INLINE_REFERENCE_PATTERN.findall(payload.content)))
    inline_result = normalize_reference_values(inline_refs, article)
    invalid = list(dict.fromkeys([*source_result.invalid, *inline_result.invalid]))
    if invalid:
        raise ChatError("回答包含不存在或无法解析的引用：" + ", ".join(invalid))

    normalized_sources = list(source_result.valid)
    missing_inline = [ref for ref in inline_result.valid if ref not in normalized_sources]
    if missing_inline:
        raise ChatError(
            "回答正文使用了未列入 source_paragraphs 的引用："
            + ", ".join(missing_inline)
        )
    try:
        return ChatReplyPayload.model_validate(
            {
                **payload.model_dump(mode="python"),
                "source_paragraphs": normalized_sources,
                "information_types": list(dict.fromkeys(payload.information_types)),
            }
        )
    except ValidationError as exc:
        raise ChatError("回答的信息类型与引用不一致。") from exc


def _repair_messages(
    messages: list[dict[str, str]],
    invalid_response: str,
    issue: str,
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": invalid_response},
        {
            "role": "user",
            "content": prompt_text("chat_repair", issue=issue),
        },
    ]


def _generate_chat_payload(
    client: ChatClient,
    messages: list[dict[str, str]],
    article: ArticleDraft,
) -> ChatReplyPayload:
    raw_response = client.complete(messages)
    try:
        return _normalize_chat_reply(
            _parse_schema(raw_response, ChatReplyPayload),
            article,
        )
    except ChatError as exc:
        repaired_response = client.complete(
            _repair_messages(messages, raw_response, str(exc))
        )
        try:
            return _normalize_chat_reply(
                _parse_schema(repaired_response, ChatReplyPayload),
                article,
            )
        except ChatError as repair_exc:
            raise ChatError(
                "模型回答在一次自动修复后仍未通过 Chat 校验。"
                f"具体原因：{repair_exc}"
            ) from None


def generate_chat_reply(
    article: ArticleDraft,
    analysis: Analysis,
    learning_session: LearningSession | None,
    session: ChatSession,
    question: str,
    settings: ModelSettings,
    client_factory: Callable[[ModelSettings], ChatClient] = OpenAICompatibleClient,
) -> ChatMessage:
    client = client_factory(settings)
    payload = _generate_chat_payload(
        client,
        build_chat_messages(article, analysis, learning_session, session, question),
        article,
    )
    return ChatMessage(
        id=uuid4().hex,
        article_id=analysis.article_id,
        role="assistant",
        content=payload.content,
        source_paragraphs=payload.source_paragraphs,
        information_types=payload.information_types,
    )


def append_exchange(
    session: ChatSession,
    question: str,
    reply: ChatMessage,
) -> ChatSession:
    question = question.strip()
    if not question:
        raise ChatError("不能保存空问题。")
    if reply.role != "assistant" or reply.article_id != session.article_id:
        raise ChatError("模型回答与当前 Chat 会话不一致。")
    user_message = ChatMessage(
        id=uuid4().hex,
        article_id=session.article_id,
        role="user",
        content=question,
    )
    return ChatSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "messages": [*session.messages, user_message, reply],
            "updated_at": datetime.now(UTC),
        }
    )


def save_message_as_insight(session: ChatSession, message_id: str) -> ChatSession:
    if any(item.source_message_id == message_id for item in session.saved_insights):
        return session
    source = next(
        (
            message
            for message in session.messages
            if message.id == message_id and message.role == "assistant"
        ),
        None,
    )
    if source is None:
        raise ChatError("找不到要保存的 AI 回答。")
    insight = SavedInsight(
        id=uuid4().hex,
        article_id=session.article_id,
        source_message_id=source.id,
        content=source.content,
    )
    return ChatSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "saved_insights": [*session.saved_insights, insight],
            "updated_at": datetime.now(UTC),
        }
    )


def update_insight(
    session: ChatSession,
    insight_id: str,
    *,
    content: str,
    user_note: str,
    selected_for_export: bool,
) -> ChatSession:
    content = content.strip()
    if not content:
        raise ChatError("洞察内容不能为空。")
    found = False
    updated: list[SavedInsight] = []
    for insight in session.saved_insights:
        if insight.id != insight_id:
            updated.append(insight)
            continue
        found = True
        updated.append(
            SavedInsight.model_validate(
                {
                    **insight.model_dump(mode="python"),
                    "content": content,
                    "user_note": user_note.strip(),
                    "selected_for_export": selected_for_export,
                    "updated_at": datetime.now(UTC),
                }
            )
        )
    if not found:
        raise ChatError("找不到要更新的洞察。")
    return ChatSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "saved_insights": updated,
            "updated_at": datetime.now(UTC),
        }
    )


def delete_insight(session: ChatSession, insight_id: str) -> ChatSession:
    retained = [item for item in session.saved_insights if item.id != insight_id]
    if len(retained) == len(session.saved_insights):
        raise ChatError("找不到要删除的洞察。")
    return ChatSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "saved_insights": retained,
            "updated_at": datetime.now(UTC),
        }
    )


def build_digest_messages(session: ChatSession) -> list[dict[str, str]]:
    if not session.messages:
        raise ChatError("当前还没有对话可以提炼。")
    schema = ConversationDigest.model_json_schema()
    history = _recent_history(session.messages[-20:])
    return [
        {
            "role": "system",
            "content": prompt_text("digest"),
        },
        {
            "role": "user",
            "content": (
                f"JSON Schema：\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"最近对话：\n{json.dumps(history, ensure_ascii=False)}\n\n"
                "已保存洞察：\n"
                f"{json.dumps([item.model_dump(mode='json') for item in session.saved_insights], ensure_ascii=False)}"
            ),
        },
    ]


def refine_conversation(
    session: ChatSession,
    settings: ModelSettings,
    client_factory: Callable[[ModelSettings], ChatClient] = OpenAICompatibleClient,
) -> ChatSession:
    messages = build_digest_messages(session)
    client = client_factory(settings)
    raw_response = client.complete(messages)
    try:
        digest = _parse_schema(raw_response, ConversationDigest)
    except ChatError as exc:
        repaired_response = client.complete(
            _repair_messages(messages, raw_response, str(exc))
        )
        try:
            digest = _parse_schema(repaired_response, ConversationDigest)
        except ChatError as repair_exc:
            raise ChatError(
                "对话提炼在一次自动修复后仍未通过结构校验。"
                f"具体原因：{repair_exc}"
            ) from None
    return ChatSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "digest": digest,
            "updated_at": datetime.now(UTC),
        }
    )
