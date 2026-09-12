"""Streamlit presentation for LayerRead v0.4 article chat."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from layerread.analysis_schema import Analysis
from layerread.analyzer import AnalysisError
from layerread.article import ArticleDraft
from layerread.chat import (
    ChatError,
    append_exchange,
    create_chat_session,
    delete_insight,
    generate_chat_reply,
    refine_conversation,
    save_message_as_insight,
    update_insight,
)
from layerread.chat_schema import ChatMessage, ChatSession, ConversationDigest
from layerread.config import ModelSettings
from layerread.demo import DemoLimiter
from layerread.learning_schema import LearningSession
from layerread.references import article_paragraph_map, normalize_reference_values


_QUICK_QUESTIONS = (
    "解释文章最关键的概念",
    "再举一个不同场景的例子",
    "检查文章论证中的逻辑漏洞",
    "说明如何把文章观点应用到实践",
    "找出还需要进一步核查的部分",
)

_INFORMATION_TYPE_HELP = {
    "原文内容": "来自当前文章并附有已验证段落编号",
    "AI解释": "基于原文所作的解释性转述",
    "AI推断": "由 AI 推导、并非作者直接陈述",
    "无法确认": "当前文章和会话材料不足以确认",
}

_CHAT_BUSY_PHASES = {
    "reply_queued",
    "reply_running",
    "digest_queued",
    "digest_running",
}


def chat_is_busy() -> bool:
    """Return whether a long-running Chat request is active or queued."""

    return st.session_state.get("chat_phase", "idle") in _CHAT_BUSY_PHASES


def recover_interrupted_chat() -> None:
    """Unlock Chat after an in-flight model request was interrupted by a rerun."""

    phase = st.session_state.get("chat_phase", "idle")
    if phase == "reply_running":
        st.session_state.chat_error = (
            "上一次 Chat 回答因页面切换、刷新或连接中断而停止；"
            "你的问题仍然保留，可以重新获取回答。模型供应商仍可能记录已经发送的请求。"
        )
    elif phase == "digest_running":
        st.session_state.chat_error = (
            "上一次对话提炼因页面切换、刷新或连接中断而停止；"
            "既有对话和洞察仍然保留，可以重新提炼。"
        )
    else:
        return
    st.session_state.chat_phase = "idle"


def _pending_question() -> str:
    value = st.session_state.get("chat_pending_question", "")
    return value.strip() if isinstance(value, str) else ""


def _session_prefix(session: ChatSession) -> str:
    return f"chat_{int(session.created_at.timestamp() * 1_000_000)}"


def _render_list(values: Iterable[str], empty_text: str) -> None:
    rows = list(values)
    if not rows:
        st.caption(empty_text)
        return
    for value in rows:
        st.write(f"- {value}")


def _render_sources(message: ChatMessage, article: ArticleDraft) -> None:
    normalized = normalize_reference_values(message.source_paragraphs, article)
    if not normalized.valid:
        if "无法确认" in message.information_types:
            st.caption("当前材料不足以确认，未提供原文出处。")
        return
    paragraph_map = article_paragraph_map(article)
    st.caption("原文出处：" + " ".join(normalized.valid))
    with st.expander("核对对应原文"):
        for reference in normalized.valid:
            st.markdown(f"**{reference}**")
            st.text(paragraph_map[reference])


def _render_information_types(message: ChatMessage) -> None:
    if not message.information_types:
        return
    labels = " · ".join(message.information_types)
    help_text = "；".join(
        f"{label}：{_INFORMATION_TYPE_HELP[label]}"
        for label in message.information_types
    )
    st.caption(f"信息类型：{labels}", help=help_text)


def _render_history(
    session: ChatSession,
    article: ArticleDraft,
) -> bool:
    changed = False
    saved_message_ids = {
        insight.source_message_id for insight in session.saved_insights
    }
    for message in session.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.role != "assistant":
                continue
            _render_information_types(message)
            _render_sources(message, article)
            if message.id in saved_message_ids:
                st.caption("已保存为关键洞察")
            elif st.button(
                "保存为关键洞察",
                icon=":material/bookmark_add:",
                key=f"save_insight_{message.id}",
            ):
                st.session_state.chat_session = save_message_as_insight(
                    session,
                    message.id,
                )
                st.session_state.chat_error = None
                changed = True
                break
    return changed


def _render_digest(digest: ConversationDigest) -> None:
    st.markdown("##### 最近一次提炼结果")
    sections = (
        ("新产生的关键理解", digest.new_understandings, "尚未形成新的关键理解。"),
        ("你最关心的问题", digest.user_focus_questions, "尚未识别出稳定关注问题。"),
        ("对原文理解的修正", digest.corrected_understandings, "尚未出现明确理解修正。"),
        ("尚未解决的问题", digest.unresolved_questions, "当前没有明确的未解决问题。"),
    )
    for title, values, empty_text in sections:
        st.markdown(f"**{title}**")
        _render_list(values, empty_text)


def _render_insight_manager(session: ChatSession) -> bool:
    with st.expander(f"关键洞察（{len(session.saved_insights)}）", expanded=False):
        if not session.saved_insights:
            st.caption("尚未保存洞察。可在任意 AI 回答下点击“保存为关键洞察”。")
            return False

        for index, insight in enumerate(session.saved_insights, start=1):
            st.markdown(f"#### 洞察 {index}")
            with st.form(f"insight_form_{insight.id}", clear_on_submit=False):
                content = st.text_area(
                    "洞察内容",
                    value=insight.content,
                    height=140,
                    key=f"insight_content_{insight.id}",
                )
                note = st.text_area(
                    "个人备注",
                    value=insight.user_note,
                    height=90,
                    placeholder="补充这条洞察为什么重要，或准备如何使用。",
                    key=f"insight_note_{insight.id}",
                )
                selected_for_export = st.checkbox(
                    "选择用于后续导出",
                    value=insight.selected_for_export,
                    key=f"insight_export_{insight.id}",
                )
                left, right = st.columns(2)
                save_clicked = left.form_submit_button(
                    "保存修改",
                    type="primary",
                    width="stretch",
                )
                delete_clicked = right.form_submit_button(
                    "删除洞察",
                    width="stretch",
                )
            try:
                if save_clicked:
                    st.session_state.chat_session = update_insight(
                        session,
                        insight.id,
                        content=content,
                        user_note=note,
                        selected_for_export=selected_for_export,
                    )
                    st.session_state.chat_error = None
                    return True
                if delete_clicked:
                    st.session_state.chat_session = delete_insight(session, insight.id)
                    st.session_state.chat_error = None
                    return True
            except ChatError as exc:
                st.session_state.chat_error = str(exc)
        return False


def _generate_reply(
    question: str,
    session: ChatSession,
    analysis: Analysis,
    article: ArticleDraft,
    learning_session: LearningSession | None,
    settings: ModelSettings,
) -> bool:
    st.session_state.chat_error = None
    with st.status("正在围绕当前文章回答…", expanded=True) as status:
        st.write("系统只使用当前文章、分析、学习记录、已保存洞察和有限对话历史。")
        try:
            reply = generate_chat_reply(
                article,
                analysis,
                learning_session,
                session,
                question,
                settings,
            )
            st.session_state.chat_session = append_exchange(
                session,
                question,
                reply,
            )
        except (ChatError, AnalysisError) as exc:
            st.session_state.chat_error = str(exc)
            status.update(label="回答失败，既有对话和洞察已保留", state="error")
            return False
        except Exception:
            st.session_state.chat_error = "回答过程中发生未预期错误，既有内容已保留，请稍后重试。"
            status.update(label="回答失败，既有对话和洞察已保留", state="error")
            return False
        status.update(label="回答已生成", state="complete", expanded=False)
        return True


def _refine_chat(session: ChatSession, settings: ModelSettings) -> bool:
    st.session_state.chat_error = None
    with st.status("正在提炼本次对话…", expanded=True) as status:
        try:
            st.session_state.chat_session = refine_conversation(session, settings)
        except (ChatError, AnalysisError) as exc:
            st.session_state.chat_error = str(exc)
            status.update(label="提炼失败，既有内容已保留", state="error")
            return False
        except Exception:
            st.session_state.chat_error = "提炼过程中发生未预期错误，既有内容已保留。"
            status.update(label="提炼失败，既有内容已保留", state="error")
            return False
        status.update(label="对话提炼完成", state="complete", expanded=False)
        return True


def _run_queued_chat_operation(
    session: ChatSession,
    analysis: Analysis,
    article: ArticleDraft,
    learning_session: LearningSession | None,
    settings: ModelSettings | None,
    limiter: DemoLimiter | None,
) -> None:
    """Run one queued Chat request after the app navigation has been locked."""

    phase = st.session_state.get("chat_phase", "idle")
    if phase not in {"reply_queued", "digest_queued"}:
        return
    if settings is None:
        st.session_state.chat_phase = "idle"
        st.session_state.chat_error = "模型配置在 Chat 请求开始前失效，请检查配置后重试。"
        st.rerun(scope="app")

    assert settings is not None
    quota_action = "chat" if phase == "reply_queued" else "chat_digest"
    if limiter is not None:
        decision = limiter.claim(quota_action)
        if not decision.allowed:
            st.session_state.chat_phase = "idle"
            st.session_state.chat_error = decision.message
            st.rerun(scope="app")

    if phase == "reply_queued":
        question = _pending_question()
        if not question:
            st.session_state.chat_phase = "idle"
            st.session_state.chat_error = "待回答的问题已经失效，请重新输入。"
            st.rerun(scope="app")
        st.session_state.chat_phase = "reply_running"
        succeeded = _generate_reply(
            question,
            session,
            analysis,
            article,
            learning_session,
            settings,
        )
        if succeeded:
            st.session_state.chat_pending_question = ""
    else:
        st.session_state.chat_phase = "digest_running"
        _refine_chat(session, settings)

    st.session_state.chat_phase = "idle"
    st.rerun(scope="app")


@st.fragment
def render_chat_tab(
    analysis: Analysis,
    article: ArticleDraft,
    learning_session: LearningSession | None,
    settings: ModelSettings | None,
    limiter: DemoLimiter | None = None,
) -> None:
    """Render article-grounded chat and saved-insight management."""

    st.markdown("### 文章 Chat")
    st.write("继续追问当前文章，并把真正重要的新理解保存为可编辑洞察。")
    st.caption(
        "Chat 默认不联网；每次只发送当前文章相关材料、已保存洞察和最近 10 条历史消息。"
    )
    st.info(
        "生成回答或提炼对话期间会暂时锁定应用内导航。请保持当前 LayerRead 页面打开，"
        "不要刷新或关闭页面；模型调用完成后导航会自动恢复。"
    )
    if settings is None:
        if chat_is_busy():
            st.session_state.chat_phase = "idle"
            st.session_state.chat_error = (
                "模型配置在 Chat 请求开始前失效；问题仍然保留，请检查配置后重试。"
            )
            st.rerun(scope="app")
        st.warning("完成模型配置后才能使用文章 Chat。")
        return

    session: ChatSession | None = st.session_state.chat_session
    if session is None or session.article_id != analysis.article_id:
        session = create_chat_session(analysis, settings)
        st.session_state.chat_session = session
        st.session_state.chat_error = None
        st.session_state.chat_pending_question = ""
        st.session_state.chat_phase = "idle"

    _run_queued_chat_operation(
        session,
        analysis,
        article,
        learning_session,
        settings,
        limiter,
    )

    if _render_insight_manager(session):
        st.rerun()
    session = st.session_state.chat_session

    chat_status = limiter.status("chat") if limiter is not None else None
    if chat_status is not None:
        st.caption(f"在线体验今日还可追问 {chat_status.remaining} 轮。")

    if session.messages:
        if _render_history(session, article):
            st.rerun()
    else:
        st.info("可以先使用快捷问题，也可以直接输入你仍然没有解决的疑问。")

    pending_question = _pending_question()
    if pending_question:
        with st.chat_message("user"):
            st.markdown(pending_question)
            st.caption("这条问题尚未获得回答。")

    if st.session_state.chat_error:
        st.error(st.session_state.chat_error)

    if pending_question:
        with st.container(border=True):
            st.markdown("#### 继续处理未回答问题")
            st.caption("问题已保留。可以重新调用模型，或取消等待后输入其他问题。")
            with st.container(horizontal=True):
                if st.button(
                    "重新获取回答",
                    type="primary",
                    icon=":material/refresh:",
                    key=f"{_session_prefix(session)}_retry_pending",
                ):
                    st.session_state.chat_error = None
                    st.session_state.chat_phase = "reply_queued"
                    st.rerun(scope="app")
                if st.button(
                    "取消等待",
                    icon=":material/close:",
                    key=f"{_session_prefix(session)}_cancel_pending",
                ):
                    st.session_state.chat_pending_question = ""
                    st.session_state.chat_error = None
                    st.rerun(scope="app")
    else:
        quick_question: str | None = None
        with st.container(border=True):
            st.markdown("#### 继续追问当前文章")
            st.caption("直接在下方输入问题并按 Enter 发送，也可以先选择一个快捷问题。")
            st.markdown("**快捷问题**")
            with st.container(horizontal=True):
                for index, label in enumerate(_QUICK_QUESTIONS):
                    if st.button(
                        label,
                        key=f"{_session_prefix(session)}_quick_{index}",
                    ):
                        quick_question = label

            typed_question = st.chat_input(
                "在这里输入追问，例如：这个结论如何应用到我的项目？",
                key=f"{_session_prefix(session)}_input",
                max_chars=4000,
                submit_mode="disable",
            )
        question = quick_question or typed_question
        if question:
            st.session_state.chat_pending_question = question.strip()
            st.session_state.chat_error = None
            st.session_state.chat_phase = "reply_queued"
            st.rerun(scope="app")

    session = st.session_state.chat_session
    digest_status = limiter.status("chat_digest") if limiter is not None else None
    with st.container(border=True):
        st.markdown("#### 提炼本次对话")
        st.caption("提炼结果会显示在按钮下方并自动保存；继续对话或编辑洞察后可再次更新。")
        if st.button(
            "提炼本次对话（调用模型）",
            icon=":material/summarize:",
            width="stretch",
            disabled=(
                not session.messages
                or bool(pending_question)
                or (digest_status is not None and not digest_status.allowed)
            ),
            key=f"{_session_prefix(session)}_digest",
        ):
            st.session_state.chat_error = None
            st.session_state.chat_phase = "digest_queued"
            st.rerun(scope="app")

        if digest_status is not None and not digest_status.allowed:
            st.caption(digest_status.message)

        session = st.session_state.chat_session
        if session.digest is not None:
            _render_digest(session.digest)
            st.caption("这是最近一次已保存的提炼结果，不会因页面编辑或新增追问自动消失。")
        elif session.messages:
            st.caption("当前对话尚未提炼。")
