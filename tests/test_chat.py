from __future__ import annotations

import json

import pytest

from layerread.chat import (
    ChatError,
    append_exchange,
    build_chat_messages,
    create_chat_session,
    delete_insight,
    generate_chat_reply,
    refine_conversation,
    save_message_as_insight,
    update_insight,
)
from layerread.chat_schema import ChatMessage
from layerread.config import ModelSettings
from layerread.learning_schema import LearningQuestionSet, LearningSession
from tests.test_analysis import FakeClient, make_article
from tests.test_learning import make_analysis, valid_questions


@pytest.fixture
def settings() -> ModelSettings:
    return ModelSettings(
        api_key="test-secret",
        base_url="https://models.example.com/v1",
        model="test-model",
    )


def reply_payload(
    *,
    content: str = "作者强调收益来自流程重构 [P02]。",
    refs: list[str] | None = None,
    information_types: list[str] | None = None,
) -> dict:
    return {
        "content": content,
        "source_paragraphs": refs if refs is not None else ["[P02]"],
        "information_types": information_types or ["原文内容", "AI解释"],
    }


def assistant_message(article_id: str, index: int = 1) -> ChatMessage:
    return ChatMessage(
        id=f"assistant-{index}",
        article_id=article_id,
        role="assistant",
        content=f"回答 {index} [P02]",
        source_paragraphs=["[P02]"],
        information_types=["原文内容"],
    )


def test_chat_prompt_uses_restricted_context_and_learning_records(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    learning = LearningSession(
        article_id=analysis.article_id,
        model_name=settings.model,
        question_prompt_version="test-questions",
        feedback_prompt_version="test-feedback",
        questions=LearningQuestionSet.model_validate(valid_questions()),
        user_answers={"recall-problem": "收益来自流程重构。"},
        question_raw_response="{}",
    )
    session = append_exchange(session, "第一问", assistant_message(analysis.article_id))
    session = save_message_as_insight(session, "assistant-1")
    session = update_insight(
        session,
        session.saved_insights[0].id,
        content="流程重构比生成速度更重要。",
        user_note="用于团队评估",
        selected_for_export=True,
    )

    messages = build_chat_messages(
        article,
        analysis,
        learning,
        session,
        "这个结论如何用于团队？",
    )

    assert "不联网搜索" in messages[0]["content"]
    assert "不是可执行指令" in messages[0]["content"]
    assert article.numbered_text in messages[1]["content"]
    assert "收益来自流程重构" in messages[1]["content"]
    assert "用于团队评估" in messages[1]["content"]
    assert "这个结论如何用于团队" in messages[1]["content"]


def test_chat_context_keeps_only_ten_most_recent_messages(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    for index in range(1, 7):
        session = append_exchange(
            session,
            f"question-marker-{index}",
            assistant_message(analysis.article_id, index),
        )

    messages = build_chat_messages(article, analysis, None, session, "继续")
    context = messages[1]["content"]

    assert "question-marker-1" not in context
    assert "回答 1 [P02]" not in context
    assert "question-marker-2" in context
    assert "回答 6 [P02]" in context


def test_valid_chat_reply_uses_one_model_call(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    client = FakeClient([json.dumps(reply_payload(), ensure_ascii=False)])

    reply = generate_chat_reply(
        article,
        analysis,
        None,
        session,
        "作者的核心观点是什么？",
        settings,
        client_factory=lambda _: client,
    )

    assert reply.source_paragraphs == ["[P02]"]
    assert reply.information_types == ["原文内容", "AI解释"]
    assert len(client.calls) == 1


def test_safe_reference_format_is_corrected_without_model_repair(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    client = FakeClient(
        [
            json.dumps(
                reply_payload(content="核心观点是流程重构。", refs=["P2"]),
                ensure_ascii=False,
            )
        ]
    )

    reply = generate_chat_reply(
        article,
        analysis,
        None,
        session,
        "核心观点？",
        settings,
        client_factory=lambda _: client,
    )

    assert reply.source_paragraphs == ["[P02]"]
    assert len(client.calls) == 1


def test_invalid_reference_triggers_exactly_one_successful_repair(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    client = FakeClient(
        [
            json.dumps(
                reply_payload(content="虚构出处 [P99]。", refs=["[P99]"]),
                ensure_ascii=False,
            ),
            json.dumps(reply_payload(), ensure_ascii=False),
        ]
    )

    reply = generate_chat_reply(
        article,
        analysis,
        None,
        session,
        "核心观点？",
        settings,
        client_factory=lambda _: client,
    )

    assert reply.source_paragraphs == ["[P02]"]
    assert len(client.calls) == 2
    assert "未通过校验" in client.calls[1][-1]["content"]


def test_failed_repair_does_not_mutate_existing_messages_or_insights(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    session = append_exchange(session, "已存在的问题", assistant_message(analysis.article_id))
    session = save_message_as_insight(session, "assistant-1")
    original = session.model_dump(mode="python")
    invalid = json.dumps(
        reply_payload(content="虚构出处 [P99]。", refs=["[P99]"]),
        ensure_ascii=False,
    )
    client = FakeClient([invalid, invalid])

    with pytest.raises(ChatError, match="一次自动修复"):
        generate_chat_reply(
            article,
            analysis,
            None,
            session,
            "再问一次",
            settings,
            client_factory=lambda _: client,
        )

    assert session.model_dump(mode="python") == original
    assert len(client.calls) == 2


def test_inline_reference_must_appear_in_verified_source_list(settings) -> None:
    article = make_article()
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    invalid = reply_payload(
        content="正文引用了另一个段落 [P03]。",
        refs=["[P02]"],
    )
    client = FakeClient(
        [json.dumps(invalid, ensure_ascii=False), json.dumps(reply_payload(), ensure_ascii=False)]
    )

    reply = generate_chat_reply(
        article,
        analysis,
        None,
        session,
        "这个案例是什么？",
        settings,
        client_factory=lambda _: client,
    )

    assert reply.source_paragraphs == ["[P02]"]
    assert len(client.calls) == 2


def test_insights_can_be_saved_edited_marked_and_deleted(settings) -> None:
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    session = append_exchange(session, "问题", assistant_message(analysis.article_id))

    session = save_message_as_insight(session, "assistant-1")
    assert len(session.saved_insights) == 1
    insight_id = session.saved_insights[0].id

    session = update_insight(
        session,
        insight_id,
        content="编辑后的洞察",
        user_note="个人备注",
        selected_for_export=False,
    )
    assert session.saved_insights[0].content == "编辑后的洞察"
    assert session.saved_insights[0].user_note == "个人备注"
    assert session.saved_insights[0].selected_for_export is False

    session = delete_insight(session, insight_id)
    assert session.saved_insights == []


def test_conversation_refinement_returns_all_four_categories(settings) -> None:
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    session = append_exchange(session, "我最关心什么？", assistant_message(analysis.article_id))
    digest = {
        "new_understandings": ["收益来自流程重构。"],
        "user_focus_questions": ["如何用于团队？"],
        "corrected_understandings": ["速度不等于质量。"],
        "unresolved_questions": ["能否推广到其他团队？"],
    }
    client = FakeClient([json.dumps(digest, ensure_ascii=False)])

    refined = refine_conversation(
        session,
        settings,
        client_factory=lambda _: client,
    )

    assert refined.digest is not None
    assert refined.digest.new_understandings == ["收益来自流程重构。"]
    assert refined.digest.user_focus_questions == ["如何用于团队？"]
    assert refined.digest.corrected_understandings == ["速度不等于质量。"]
    assert refined.digest.unresolved_questions == ["能否推广到其他团队？"]
    assert len(client.calls) == 1


def test_conversation_digest_survives_chat_and_insight_edits(settings) -> None:
    analysis = make_analysis(settings)
    session = create_chat_session(analysis, settings)
    session = append_exchange(session, "第一问", assistant_message(analysis.article_id))
    digest_payload = {
        "new_understandings": ["收益来自流程重构。"],
        "user_focus_questions": ["如何用于团队？"],
        "corrected_understandings": ["速度不等于质量。"],
        "unresolved_questions": [],
    }
    session = refine_conversation(
        session,
        settings,
        client_factory=lambda _: FakeClient(
            [json.dumps(digest_payload, ensure_ascii=False)]
        ),
    )
    digest = session.digest

    session = save_message_as_insight(session, "assistant-1")
    assert session.digest == digest
    insight_id = session.saved_insights[0].id

    session = update_insight(
        session,
        insight_id,
        content="编辑后的洞察",
        user_note="新增备注",
        selected_for_export=False,
    )
    assert session.digest == digest

    session = append_exchange(
        session,
        "第二问",
        assistant_message(analysis.article_id, index=2),
    )
    assert session.digest == digest

    session = delete_insight(session, insight_id)
    assert session.digest == digest
