from __future__ import annotations

from layerread.chat import (
    append_exchange,
    create_chat_session,
    save_message_as_insight,
    update_insight,
)
from layerread.chat_schema import ChatMessage, ChatSession, ConversationDigest
from layerread.config import ModelSettings
from layerread.learning_schema import LearningFeedback, LearningQuestionSet, LearningSession
from tests.test_learning import valid_feedback, valid_questions


def make_learning_session(analysis, settings: ModelSettings) -> LearningSession:
    answers = {
        "recall-problem": "文章认为收益来自流程重构，而不是只提高生成速度。"
    }
    feedback = LearningFeedback.model_validate(valid_feedback(answers))
    return LearningSession(
        article_id=analysis.article_id,
        model_name=settings.model,
        question_prompt_version="test-v0.5-questions",
        feedback_prompt_version="test-v0.5-feedback",
        questions=LearningQuestionSet.model_validate(valid_questions()),
        user_answers=answers,
        feedback=feedback,
        mastery_status=feedback.mastery_status,
        review_cards=feedback.review_cards,
        question_raw_response="{}",
        feedback_raw_response="{}",
    )


def make_chat_session(analysis, settings: ModelSettings) -> ChatSession:
    session = create_chat_session(analysis, settings)
    reply = ChatMessage(
        id="v05-assistant-message",
        article_id=analysis.article_id,
        role="assistant",
        content="流程重构决定 AI 能否真正改善质量 [P02]。",
        source_paragraphs=["[P02]"],
        information_types=["原文内容", "AI解释"],
    )
    session = append_exchange(session, "这篇文章最重要的新理解是什么？", reply)
    session = save_message_as_insight(session, reply.id)
    session = update_insight(
        session,
        session.saved_insights[0].id,
        content="真正收益来自流程重构。",
        user_note="应用到团队评估清单。",
        selected_for_export=True,
    )
    return ChatSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "digest": ConversationDigest(
                new_understandings=["速度提升不等于质量提升。"],
                user_focus_questions=["如何用于团队？"],
                corrected_understandings=["不能把工具能力等同于流程收益。"],
                unresolved_questions=["单一案例能否推广到其他团队？"],
            ),
        }
    )
