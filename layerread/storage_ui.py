"""Streamlit orchestration for local snapshot restore and autosave."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from layerread.storage import (
    LocalSnapshot,
    StorageError,
    database_path,
    initialize_database,
    list_articles,
    load_snapshot,
    save_snapshot,
)
from layerread.connector import is_connector_extraction_method
from layerread.runtime import read_runtime_profile
from layerread.visualization import build_visualizations


def _start_new_article() -> None:
    """Clear the active browser session without deleting local history."""

    reset_values = {
        "article_input": "",
        "reading_goal": "",
        "familiarity": "入门",
        "focus_area": "通用",
        "media_dependency": False,
        "main_section": "导入与判断",
        "judgment_tab": "导入文章",
        "article_analysis_tab": "深度解析",
        "article_learning_tab": "主动学习",
        "evidence_tab": "论证与证据",
        "article_draft": None,
        "analysis": None,
        "analysis_error": None,
        "learning_session": None,
        "learning_error": None,
        "chat_session": None,
        "chat_error": None,
        "chat_phase": "idle",
        "chat_pending_question": "",
        "visualizations": None,
        "current_record_id": None,
        "storage_last_saved_at": None,
        "import_tab": (
            "Chrome 一键导入"
            if read_runtime_profile().connector_enabled
            else "URL 导入"
        ),
        "url_input": "",
        "url_extraction_result": None,
        "url_title_input": "",
        "url_account_input": "",
        "url_author_input": "",
        "url_published_at_input": "",
        "url_source_input": "",
        "url_body_input": "",
        "connector_import_result": None,
        "connector_import_error": None,
        "connector_import_notice": None,
        "connector_ack_import_token": None,
        "connector_routed_import_token": None,
        "connector_title_input": "",
        "connector_account_input": "",
        "connector_author_input": "",
        "connector_published_at_input": "",
        "connector_source_input": "",
        "connector_body_input": "",
    }
    for key, value in reset_values.items():
        st.session_state[key] = value
    st.session_state.pop("layerread_connector_component", None)
    st.session_state.pop("deepread_connector_component", None)
    st.query_params.pop("layerread_import", None)
    st.query_params.pop("deepread_import", None)
    profile = read_runtime_profile()
    st.session_state.storage_notice = (
        "已清除当前会话中的文章、分析和对话。"
        if not profile.long_term_storage_enabled
        else "已开始新文章；之前的内容仍保留在本地文章库中。"
    )


def _apply_snapshot(snapshot: LocalSnapshot) -> None:
    article = snapshot.article
    st.session_state.article_input = article.raw_text
    st.session_state.reading_goal = article.reading_goal
    st.session_state.familiarity = article.familiarity
    st.session_state.focus_area = article.focus_area
    st.session_state.media_dependency = article.media_dependency
    st.session_state.article_draft = article
    st.session_state.analysis = snapshot.analysis
    st.session_state.analysis_error = None
    st.session_state.learning_session = snapshot.learning_session
    st.session_state.learning_error = None
    st.session_state.chat_session = snapshot.chat_session
    st.session_state.chat_error = None
    st.session_state.chat_phase = "idle"
    st.session_state.chat_pending_question = ""
    st.session_state.visualizations = snapshot.visualizations
    st.session_state.current_record_id = snapshot.record_id
    st.session_state.storage_error = None
    if is_connector_extraction_method(article.extraction_method):
        st.session_state.import_tab = (
            "Chrome 一键导入"
            if read_runtime_profile().connector_enabled
            else "URL 导入"
        )
        st.session_state.connector_import_result = None
        st.session_state.connector_import_error = None
        st.session_state.connector_title_input = article.source_title
        st.session_state.connector_account_input = article.account
        st.session_state.connector_author_input = article.author
        st.session_state.connector_published_at_input = article.published_at
        st.session_state.connector_source_input = article.source_url
        st.session_state.connector_body_input = article.raw_text
    st.session_state.main_section = "导入与判断"
    st.session_state.judgment_tab = (
        "阅读决策" if snapshot.analysis is not None else "处理结果"
    )


def initialize_local_storage() -> None:
    """Initialize SQLite once without opening any previous article."""

    if st.session_state.local_storage_initialized:
        return
    try:
        initialize_database()
    except StorageError as exc:
        st.session_state.storage_error = str(exc)
    finally:
        st.session_state.local_storage_initialized = True


def autosave_current_snapshot() -> None:
    """Persist current state without clearing any in-memory data on failure."""

    article = st.session_state.article_draft
    if article is None:
        return
    analysis = st.session_state.analysis
    visualizations = st.session_state.visualizations
    if analysis is not None and (
        visualizations is None or visualizations.article_id != analysis.article_id
    ):
        visualizations = build_visualizations(article, analysis)
        st.session_state.visualizations = visualizations
    try:
        chat_session = st.session_state.chat_session
        persisted_chat = (
            chat_session
            if chat_session is not None
            and (
                chat_session.messages
                or chat_session.saved_insights
                or chat_session.digest is not None
            )
            else None
        )
        summary = save_snapshot(
            article,
            analysis,
            st.session_state.learning_session,
            persisted_chat,
            visualizations,
        )
    except StorageError as exc:
        st.session_state.storage_error = str(exc)
        return
    st.session_state.current_record_id = summary.record_id
    st.session_state.storage_last_saved_at = summary.updated_at
    st.session_state.storage_error = None


def _record_label(record) -> str:
    assets = ["分析" if record.has_analysis else "仅正文"]
    if record.has_learning:
        assets.append("学习")
    if record.has_chat:
        assets.append("Chat")
    local_time = record.updated_at.astimezone().strftime("%m-%d %H:%M")
    return f"{record.title} · {'/'.join(assets)} · {local_time}"


def render_local_library() -> None:
    """Render the single-user local article library in the sidebar."""

    with st.sidebar:
        st.divider()
        st.header("本地文章")
        st.button(
            "开始新文章",
            icon=":material/note_add:",
            width="stretch",
            on_click=_start_new_article,
            key="start_new_article",
        )
        try:
            records = list_articles()
        except StorageError as exc:
            st.error(str(exc))
            return
        if not records:
            st.caption("处理文章后会自动保存到本地 SQLite。")
            st.caption(f"数据库：{database_path()}")
            return

        record_map = {record.record_id: record for record in records}
        selected_id = st.selectbox(
            "选择本地记录",
            options=list(record_map),
            format_func=lambda record_id: _record_label(record_map[record_id]),
            key="local_article_selector",
        )
        if st.button(
            "打开所选文章",
            icon=":material/folder_open:",
            width="stretch",
        ):
            try:
                snapshot = load_snapshot(selected_id)
            except StorageError as exc:
                st.session_state.storage_error = str(exc)
            else:
                _apply_snapshot(snapshot)
                st.session_state.storage_notice = f"已打开本地文章：{snapshot.title}"
                st.rerun()

        current_id = st.session_state.current_record_id
        if current_id in record_map:
            current = record_map[current_id]
            st.caption(f"当前记录：{current.title}")
        saved_at: datetime | None = st.session_state.storage_last_saved_at
        if saved_at is not None:
            st.caption(f"最近自动保存：{saved_at.astimezone().strftime('%H:%M:%S')}")
        st.caption(f"数据库：{database_path()}")


def render_session_workspace() -> None:
    """Render a no-persistence workspace control for the hosted demo."""

    with st.sidebar:
        st.divider()
        st.header("当前会话")
        st.button(
            "清除并开始新文章",
            icon=":material/delete_sweep:",
            width="stretch",
            on_click=_start_new_article,
            key="start_new_demo_article",
        )
        st.caption("关闭页面或会话失效后，当前文章、图片、分析和对话将无法恢复。")


def render_storage_messages() -> None:
    if st.session_state.storage_notice:
        st.info(st.session_state.storage_notice)
        st.session_state.storage_notice = None
    if st.session_state.storage_error:
        st.warning(st.session_state.storage_error)
