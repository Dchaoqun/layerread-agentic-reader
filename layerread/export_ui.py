"""Streamlit UI for selectable Markdown export."""

from __future__ import annotations

from datetime import datetime
import os

import streamlit as st

from layerread.analysis_schema import Analysis
from layerread.article import ArticleDraft
from layerread.chat_schema import ChatSession
from layerread.export import (
    ExportError,
    ExportSectionId,
    SECTION_LABELS,
    available_export_sections,
    build_export_document,
    markdown_filename,
    render_markdown,
)
from layerread.learning_schema import LearningSession
from layerread.notion import (
    NotionClient,
    NotionError,
    NotionPartialExportError,
    normalize_notion_id,
)
from layerread.storage import (
    StorageError,
    article_record_id,
    list_notion_exports,
    record_notion_export,
    save_snapshot,
)
from layerread.visualization import VisualizationBundle


def _initialize_notion_state() -> None:
    defaults = {
        "notion_token": os.environ.get("NOTION_API_TOKEN", ""),
        "notion_target_id": os.environ.get("NOTION_TARGET_ID", ""),
        "notion_target_kind": os.environ.get("NOTION_TARGET_TYPE", "page"),
        "notion_connection_result": None,
        "notion_export_result": None,
        "notion_export_error": None,
        "notion_partial_page": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _record_export_safely(
    article: ArticleDraft,
    target_id: str,
    sections: list[ExportSectionId],
    *,
    status: str,
    page_id: str = "",
    page_url: str = "",
    error_message: str = "",
) -> None:
    try:
        record_notion_export(
            article_record_id(article),
            target_id,
            list(sections),
            status=status,
            notion_page_id=page_id,
            notion_page_url=page_url,
            error_message=error_message,
        )
    except StorageError as exc:
        st.warning(f"Notion 操作已完成，但本地导出记录保存失败：{exc}")


def _render_notion_export(
    document,
    article: ArticleDraft,
    analysis: Analysis | None,
    learning: LearningSession | None,
    chat: ChatSession | None,
    visualizations: VisualizationBundle | None,
    selection: list[ExportSectionId],
) -> None:
    _initialize_notion_state()
    st.divider()
    st.markdown("### Notion 导出")
    st.write("连接你自己的 Notion Integration，并把同一份所选内容创建为新页面。")
    st.caption(
        "需要先在 Notion 中把目标页面或数据库共享给 Integration。"
        "Token 只用于本次 API 请求，不会写入本地数据库、普通日志或导出内容。"
    )
    if article.images:
        st.caption(
            f"本次会自动上传并保存 {len(article.images)} 个原文重要图片或长图分片，"
            "保留现有清晰度，不调用模型。"
        )

    st.text_input(
        "Integration Token",
        type="password",
        key="notion_token",
        autocomplete="off",
        placeholder="ntn_… 或 secret_…",
    )
    target_columns = st.columns([1, 2])
    with target_columns[0]:
        st.selectbox(
            "目标类型",
            options=["page", "database"],
            format_func=lambda value: "Page" if value == "page" else "Database / Data source",
            key="notion_target_kind",
        )
    with target_columns[1]:
        st.text_input(
            "目标 Page ID、Database ID 或 Notion 链接",
            key="notion_target_id",
            placeholder="粘贴 ID 或完整链接",
        )

    with st.container(horizontal=True):
        test_clicked = st.button(
            "测试连接",
            icon=":material/cable:",
            key="test_notion_connection",
        )
        export_clicked = st.button(
            "导出到 Notion",
            type="primary",
            icon=":material/upload:",
            key="export_to_notion",
        )

    if test_clicked:
        st.session_state.notion_connection_result = None
        try:
            client = NotionClient(st.session_state.notion_token)
            target = client.test_connection(
                st.session_state.notion_target_id,
                st.session_state.notion_target_kind,
            )
        except NotionError as exc:
            st.session_state.notion_connection_result = ("error", str(exc))
        else:
            st.session_state.notion_connection_result = (
                "success",
                f"连接成功：{target.display_name}（{target.target_type}）",
            )

    connection_result = st.session_state.notion_connection_result
    if connection_result is not None:
        level, message = connection_result
        st.success(message) if level == "success" else st.error(message)

    if export_clicked:
        st.session_state.notion_export_result = None
        st.session_state.notion_export_error = None
        st.session_state.notion_partial_page = None
        target_for_record = st.session_state.notion_target_id.strip()
        try:
            target_for_record = normalize_notion_id(target_for_record)
            # Ensure the article exists locally before the export outcome is logged.
            save_snapshot(article, analysis, learning, chat, visualizations)
            client = NotionClient(st.session_state.notion_token)
            with st.status("正在创建 Notion 页面…", expanded=True) as status:
                st.write("正在验证目标并创建页面。")
                if article.images:
                    st.write(f"正在上传 {len(article.images)} 个原文重要图片或长图分片。")
                result = client.export_document(
                    document,
                    st.session_state.notion_target_id,
                    st.session_state.notion_target_kind,
                    images=article.images,
                )
                st.write(f"已分批写入 {result.appended_blocks} 个 Block。")
                if result.uploaded_images:
                    st.write(f"已保存 {result.uploaded_images} 个原文图片或长图分片。")
                status.update(label="Notion 导出完成", state="complete", expanded=False)
        except NotionPartialExportError as exc:
            message = str(exc)
            st.session_state.notion_export_error = message
            st.session_state.notion_partial_page = {
                "page_id": exc.page_id,
                "url": exc.page_url,
            }
            _record_export_safely(
                article,
                target_for_record,
                selection,
                status="failed",
                page_id=exc.page_id,
                page_url=exc.page_url,
                error_message=message,
            )
        except (NotionError, StorageError) as exc:
            message = str(exc)
            st.session_state.notion_export_error = message
            _record_export_safely(
                article,
                target_for_record,
                selection,
                status="failed",
                error_message=message,
            )
        else:
            exported_at = datetime.now().astimezone()
            st.session_state.notion_export_result = {
                "title": result.page_title,
                "url": result.page_url,
                "page_id": result.page_id,
                "exported_at": exported_at,
                "uploaded_images": result.uploaded_images,
            }
            _record_export_safely(
                article,
                target_for_record,
                selection,
                status="success",
                page_id=result.page_id,
                page_url=result.page_url,
            )

    if st.session_state.notion_export_error:
        st.error(
            "Notion 导出失败，当前文章和学习数据均已保留，可修正配置后重试："
            + st.session_state.notion_export_error
        )
        st.info("上方 Markdown 下载仍然可用。")
        partial_page = st.session_state.notion_partial_page
        if partial_page and partial_page["url"]:
            st.link_button(
                "检查已创建的部分页面",
                partial_page["url"],
                icon=":material/open_in_new:",
                width="stretch",
            )
    if st.session_state.notion_export_result:
        result = st.session_state.notion_export_result
        st.success(
            f"已导出《{result['title']}》 · "
            f"{result['exported_at'].strftime('%Y-%m-%d %H:%M:%S')} · "
            f"原文图片 {result.get('uploaded_images', 0)} 个"
        )
        if result["url"]:
            st.link_button(
                "打开 Notion 页面",
                result["url"],
                icon=":material/open_in_new:",
                width="stretch",
            )

    try:
        history = list_notion_exports(article_record_id(article))
    except StorageError:
        history = []
    if history:
        with st.expander("最近 Notion 导出记录"):
            for item in history[:5]:
                local_time = item.exported_at.astimezone().strftime("%Y-%m-%d %H:%M")
                label = "成功" if item.status == "success" else "失败"
                st.write(f"{local_time} · {label} · {len(item.exported_sections)} 个章节")


def render_export_tab(
    article: ArticleDraft,
    analysis: Analysis | None,
    learning: LearningSession | None,
    chat: ChatSession | None,
    visualizations: VisualizationBundle | None,
    *,
    notion_enabled: bool = True,
) -> None:
    st.markdown("### 选择导出内容")
    if notion_enabled:
        st.write("同一份选择会同时用于 Markdown 与 Notion；只生成你勾选的章节。")
    else:
        st.write("选择需要写入 Markdown 的章节；只生成你勾选的内容。")
    st.caption(
        "导出完全在本地完成，不会调用模型。只有标记为“选择用于后续导出”的 Chat 洞察和笔记会进入文件。"
    )

    available = available_export_sections(
        analysis,
        learning,
        chat,
        visualizations,
    )
    selection = st.multiselect(
        "选择导出模块",
        options=available,
        default=available,
        format_func=lambda section_id: SECTION_LABELS[section_id],
        key=f"export_sections_{analysis.article_id if analysis else 'article'}",
        placeholder="请选择至少一个模块",
    )
    if not selection:
        st.warning("请至少选择一个导出模块。")
        return

    try:
        document = build_export_document(
            article,
            analysis,
            learning,
            chat,
            visualizations,
            list(selection),
            source_url=article.source_url,
        )
        markdown = render_markdown(document)
    except (ExportError, TypeError, ValueError) as exc:
        st.error(f"Markdown 生成失败，当前文章和学习资产未受影响：{exc}")
        return

    st.markdown("### Markdown 导出")
    metrics = st.columns(3)
    metrics[0].metric("已选模块", len(document.sections))
    metrics[1].metric("导出字符", f"{len(markdown):,}")
    metrics[2].metric("文件格式", "Markdown")

    st.download_button(
        "下载 Markdown",
        data=markdown.encode("utf-8"),
        file_name=markdown_filename(document),
        mime="text/markdown; charset=utf-8",
        type="primary",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",
    )
    with st.expander("预览 Markdown 源码"):
        st.code(markdown, language="markdown", wrap_lines=True)

    if notion_enabled:
        _render_notion_export(
            document,
            article,
            analysis,
            learning,
            chat,
            visualizations,
            list(selection),
        )
