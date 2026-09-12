"""Streamlit presentation for the LayerRead v0.3 active-learning workflow."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from layerread.analysis_schema import Analysis
from layerread.analyzer import AnalysisError
from layerread.article import ArticleDraft
from layerread.config import ModelSettings
from layerread.demo import DemoLimiter
from layerread.learning import (
    LearningError,
    answers_fingerprint,
    evaluate_learning_session,
    generate_learning_session,
    update_learning_answers,
)
from layerread.learning_schema import (
    LearningQuestion,
    LearningSession,
    MasteryStatus,
)
from layerread.references import article_paragraph_map, normalize_reference_values


_STATUS_HELP: dict[MasteryStatus, str] = {
    "已掌握": "回答准确、完整，并能正确说明关键关系。",
    "基本理解": "核心方向正确，但仍有少量遗漏或表达不够精确。",
    "存在缺口": "理解了部分内容，但关键环节尚未建立。",
    "需要复习": "当前回答暴露出基础概念混淆或主要内容缺失。",
}

_LEARNING_BUSY_PHASES = {
    "questions_queued",
    "questions_running",
    "feedback_queued",
    "feedback_running",
}


def learning_is_busy() -> bool:
    """Return whether a long-running learning model request is active or queued."""

    return st.session_state.get("learning_phase", "idle") in _LEARNING_BUSY_PHASES


def recover_interrupted_learning() -> None:
    """Unlock learning after a rerun interrupted an in-flight model request."""

    phase = st.session_state.get("learning_phase", "idle")
    if phase == "questions_running":
        st.session_state.learning_error = (
            "上一次主动学习题目生成因页面切换、刷新或连接中断而停止；"
            "原文章和分析仍然保留，请重新生成。模型供应商仍可能记录已经发送的请求。"
        )
    elif phase == "feedback_running":
        st.session_state.learning_error = (
            "上一次学习反馈生成因页面切换、刷新或连接中断而停止；"
            "已填写的答案仍然保留，请重新获取反馈。模型供应商仍可能记录已经发送的请求。"
        )
    else:
        return
    st.session_state.learning_phase = "idle"


def _widget_prefix(session: LearningSession) -> str:
    return f"learning_{int(session.started_at.timestamp() * 1_000_000)}"


def _answer_key(session: LearningSession, question_id: str) -> str:
    return f"{_widget_prefix(session)}_answer_{question_id}"


def _render_list(label: str, values: Iterable[str], empty_text: str) -> None:
    st.markdown(f"**{label}**")
    rows = list(values)
    if rows:
        for value in rows:
            st.write(f"- {value}")
    else:
        st.caption(empty_text)


def _render_safe_references(refs: list[str], article: ArticleDraft) -> None:
    normalized = normalize_reference_values(refs, article)
    if not normalized.valid:
        st.caption("没有通过存在性校验的原文出处。")
        return

    paragraph_map = article_paragraph_map(article)
    st.caption("原文出处：" + " ".join(normalized.valid))
    with st.expander("核对对应原文"):
        for reference in normalized.valid:
            st.markdown(f"**{reference}**")
            st.text(paragraph_map[reference])


def _question_heading(index: int, question: LearningQuestion) -> str:
    return f"{index}. {question.question_type} · {question.learning_objective}"


def _collect_answers(session: LearningSession) -> dict[str, str]:
    return {
        question.question_id: st.session_state.get(
            _answer_key(session, question.question_id),
            "",
        )
        for question in session.questions.all_questions
    }


def _initialize_answer_widgets(session: LearningSession) -> None:
    for question in session.questions.all_questions:
        key = _answer_key(session, question.question_id)
        if key not in st.session_state:
            st.session_state[key] = session.user_answers.get(question.question_id, "")


def _render_question_form(session: LearningSession) -> tuple[bool, bool]:
    _initialize_answer_widgets(session)
    feedback_is_current = (
        session.feedback is not None
        and session.feedback_answer_fingerprint
        == answers_fingerprint(session.user_answers)
    )
    feedback_label = (
        "重新评价当前答案（再次调用模型）"
        if feedback_is_current
        else "提交已作答内容并获取反馈（调用模型）"
    )
    with st.form(f"{_widget_prefix(session)}_form", clear_on_submit=False):
        for index, question in enumerate(session.questions.all_questions, start=1):
            with st.container(border=True):
                st.markdown(f"#### {_question_heading(index, question)}")
                st.write(question.prompt)
                if question.question_type == "Teach-back":
                    st.caption(f"讲解对象：{question.scenario} · 核心概念：{question.focus_concept}")
                elif question.scenario:
                    st.caption(f"应用场景：{question.scenario}")
                st.text_area(
                    "你的回答",
                    key=_answer_key(session, question.question_id),
                    height=150,
                    placeholder="先用自己的话回答；可以只回答这一题，也可以继续完成全部题目。",
                )

        left, right = st.columns(2)
        with left:
            save_clicked = st.form_submit_button(
                "保存当前答案",
                width="stretch",
            )
        with right:
            feedback_clicked = st.form_submit_button(
                feedback_label,
                type="primary",
                width="stretch",
            )
    return save_clicked, feedback_clicked


def _render_feedback(
    session: LearningSession,
    article: ArticleDraft,
) -> None:
    feedback = session.feedback
    if feedback is None:
        return

    if session.feedback_answer_fingerprint != answers_fingerprint(session.user_answers):
        st.warning("答案已更新，下面的旧反馈已隐藏。请重新提交以获取与当前答案一致的反馈。")
        return

    feedback_by_id = {
        item.question_id: item for item in feedback.question_feedback
    }
    st.markdown("### 针对你的回答")
    for question in session.questions.all_questions:
        item = feedback_by_id.get(question.question_id)
        if item is None:
            continue
        with st.container(border=True):
            st.markdown(f"#### {question.question_type} · {question.learning_objective}")
            st.write(question.prompt)
            st.caption(f"掌握判断：{item.mastery_status}")
            st.markdown("**反馈引用的你的原话**")
            st.markdown(f"> {item.answer_excerpt}")
            _render_list("回答正确的部分", item.correct_parts, "暂未识别出明确正确点。")
            _render_list("遗漏的部分", item.omissions, "没有发现明显遗漏。")
            _render_list("混淆或不准确之处", item.inaccuracies, "没有发现明显不准确之处。")
            _render_list("改进建议", item.improvement_suggestions, "暂无建议。")
            if question.question_type == "Teach-back":
                st.markdown("**概念准确性**")
                st.write(item.concept_accuracy)
                st.markdown("**因果关系**")
                st.write(item.causal_reasoning)
            _render_safe_references(item.paragraph_refs, article)

            with st.expander("查看参考答案与评分要点"):
                st.write(question.reference_answer)
                _render_list("评分要点", question.evaluation_points, "暂无评分要点。")
                _render_safe_references(question.paragraph_refs, article)

    st.markdown("### 核心概念掌握状态")
    for item in feedback.mastery_status:
        with st.container(border=True):
            st.markdown(f"**{item.concept_title} · {item.status}**")
            st.caption(_STATUS_HELP[item.status])
            st.write(item.evidence)
            st.write(f"复习建议：{item.review_advice}")
            _render_safe_references(item.paragraph_refs, article)

    st.markdown("### 复习卡")
    for index, card in enumerate(feedback.review_cards, start=1):
        with st.container(border=True):
            st.markdown(f"**卡片 {index}：{card.question}**")
            st.write(card.short_answer)
            with st.expander("查看详细解释"):
                st.write(card.detailed_explanation)
                _render_safe_references(card.paragraph_refs, article)
    st.info(feedback.overall_review_advice)


def _generate_questions(
    article: ArticleDraft,
    analysis: Analysis,
    settings: ModelSettings,
) -> bool:
    st.session_state.learning_error = None
    with st.status("正在生成主动学习题目…", expanded=True) as status:
        st.write("系统会生成核心回忆、Teach-back、应用和批判性问题，并校验原文引用。")
        try:
            st.session_state.learning_session = generate_learning_session(
                article,
                analysis,
                settings,
            )
        except (LearningError, AnalysisError) as exc:
            st.session_state.learning_error = str(exc)
            status.update(label="题目生成失败", state="error")
            return False
        except Exception:
            st.session_state.learning_error = "题目生成过程中发生未预期错误，请稍后重试。"
            status.update(label="题目生成失败", state="error")
            return False
        status.update(label="主动学习题目已生成", state="complete", expanded=False)
        return True


def _run_queued_learning_operation(
    article: ArticleDraft,
    analysis: Analysis,
    settings: ModelSettings | None,
    limiter: DemoLimiter | None,
) -> None:
    """Run one queued learning request after app navigation has been locked."""

    phase = st.session_state.get("learning_phase", "idle")
    if phase not in {"questions_queued", "feedback_queued"}:
        return
    if settings is None:
        st.session_state.learning_phase = "idle"
        st.session_state.learning_error = (
            "模型配置在学习请求开始前失效，请重新检查配置后再试。"
        )
        st.rerun(scope="app")

    assert settings is not None
    quota_action = (
        "learning_questions" if phase == "questions_queued" else "learning_feedback"
    )
    if limiter is not None:
        decision = limiter.claim(quota_action)
        if not decision.allowed:
            st.session_state.learning_phase = "idle"
            st.session_state.learning_error = decision.message
            st.rerun(scope="app")

    if phase == "questions_queued":
        st.session_state.learning_phase = "questions_running"
        _generate_questions(article, analysis, settings)
    else:
        session: LearningSession | None = st.session_state.learning_session
        if session is None:
            st.session_state.learning_phase = "idle"
            st.session_state.learning_error = "学习会话已经失效，请重新生成题目。"
            st.rerun(scope="app")
        assert session is not None
        st.session_state.learning_phase = "feedback_running"
        with st.status("正在评价已作答内容…", expanded=True) as status:
            st.write("系统会核对概念、因果关系、知识迁移和证据意识。")
            try:
                st.session_state.learning_session = evaluate_learning_session(
                    session,
                    article,
                    analysis,
                    settings,
                )
            except (LearningError, AnalysisError) as exc:
                st.session_state.learning_error = str(exc)
                status.update(label="反馈生成失败，答案已保留", state="error")
            except Exception:
                st.session_state.learning_error = (
                    "反馈生成过程中发生未预期错误，答案已保留，请稍后重试。"
                )
                status.update(label="反馈生成失败，答案已保留", state="error")
            else:
                st.session_state.learning_error = None
                status.update(
                    label="学习反馈已生成",
                    state="complete",
                    expanded=False,
                )
    st.session_state.learning_phase = "idle"
    st.rerun(scope="app")


def render_learning_tab(
    analysis: Analysis,
    article: ArticleDraft,
    settings: ModelSettings | None,
    limiter: DemoLimiter | None = None,
) -> None:
    """Render the optional v0.3 learning loop after article analysis."""

    st.markdown("### 主动学习")
    st.write("先主动回忆和应用，再根据你的具体回答获得反馈。题目可按任意顺序完成。")
    st.caption(
        "答案仅在当前浏览器会话中保留；获取反馈时，已作答内容会发送给你配置的模型服务。"
    )
    st.info(
        "生成题目或反馈时会暂时锁定应用内导航。请保持当前 LayerRead 页面打开，"
        "不要刷新或关闭页面；模型调用完成后导航会自动恢复。"
    )

    _run_queued_learning_operation(article, analysis, settings, limiter)

    session: LearningSession | None = st.session_state.learning_session
    if session is not None and session.article_id != analysis.article_id:
        st.session_state.learning_session = None
        session = None
        st.warning("当前分析已变化，旧学习会话已清除，请重新生成题目。")

    if session is None:
        generation_status = (
            limiter.status("learning_questions") if limiter is not None else None
        )
        if settings is None:
            st.warning("完成模型配置后才能生成主动学习题目。")
        if st.button(
            "生成主动学习题目（调用模型）",
            type="primary",
            width="stretch",
            disabled=(
                settings is None
                or (generation_status is not None and not generation_status.allowed)
            ),
        ):
            assert settings is not None
            st.session_state.learning_error = None
            st.session_state.learning_phase = "questions_queued"
            st.rerun(scope="app")
        if generation_status is not None and not generation_status.allowed:
            st.caption(generation_status.message)
        if st.session_state.learning_error:
            st.error(st.session_state.learning_error)
        return

    if limiter is None:
        with st.expander("重新生成题目"):
            st.warning("重新生成会清空当前题目、答案、反馈和复习卡，并再次调用模型。")
            confirmation_key = f"{_widget_prefix(session)}_regenerate_confirm"
            confirmed = st.checkbox("我确认清空当前学习会话并重新生成。", key=confirmation_key)
            if st.button(
                "确认重新生成（调用模型）",
                disabled=settings is None or not confirmed,
                width="stretch",
            ):
                assert settings is not None
                st.session_state.learning_error = None
                st.session_state.learning_phase = "questions_queued"
                st.rerun(scope="app")
    else:
        st.caption("在线体验每篇文章只生成一次主动学习题目，不开放重新生成。")

    save_clicked, feedback_clicked = _render_question_form(session)
    if save_clicked or feedback_clicked:
        try:
            updated = update_learning_answers(session, _collect_answers(session))
            st.session_state.learning_session = updated
            st.session_state.learning_error = None
        except LearningError as exc:
            st.session_state.learning_error = str(exc)
        else:
            if feedback_clicked:
                if settings is None:
                    st.session_state.learning_error = "模型配置不可用，答案已保存但尚未获取反馈。"
                else:
                    st.session_state.learning_phase = "feedback_queued"
            st.rerun(scope="app")

    if st.session_state.learning_error:
        st.error(st.session_state.learning_error)

    saved_count = len(session.user_answers)
    st.caption(f"当前会话已保存 {saved_count}/{len(session.questions.all_questions)} 道回答。")
    _render_feedback(session, article)
