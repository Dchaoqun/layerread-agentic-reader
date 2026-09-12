"""叠读 LayerRead v0.8 development Streamlit entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from layerread.analysis_schema import (
    Analysis,
    CitedText,
    CriterionScore,
    InferredText,
    ReferenceDiagnostics,
)
from layerread.analyzer import (
    AnalysisError,
    analyze_article,
    parse_analysis,
)
from layerread.article import ArticleValidationError, process_article, upgrade_article_draft
from layerread.chat_ui import chat_is_busy, recover_interrupted_chat, render_chat_tab
from layerread.config import (
    ModelConfigResult,
    ModelSettings,
    read_model_configs,
    read_portfolio_model_config,
)
from layerread.connector import (
    CONNECTOR_EXTRACTION_METHOD,
    ConnectorImportError,
    is_connector_extraction_method,
    parse_connector_payload,
)
from layerread.connector_package import (
    build_online_connector_archive,
    connector_version,
)
from layerread.demo import DemoLimiter, limit_images
from layerread.demo_ui import build_demo_limiter, render_demo_sidebar
from layerread.export_ui import render_export_tab
from layerread.learning_ui import (
    learning_is_busy,
    recover_interrupted_learning,
    render_learning_tab,
)
from layerread.references import (
    article_paragraph_map,
    normalize_reference_values,
    normalize_references,
    validate_references,
)
from layerread.runtime import RuntimeProfile, filter_model_configs, read_runtime_profile
from layerread.storage_ui import (
    autosave_current_snapshot,
    initialize_local_storage,
    render_local_library,
    render_session_workspace,
    render_storage_messages,
)
from layerread.visualization import build_visualizations
from layerread.visualization_ui import render_visualization_tab
from layerread.url_import import ArticleExtractionResult, extract_article_from_url

if TYPE_CHECKING:
    from layerread.article import ArticleDraft
    from layerread.connector import ConnectorArticleImportResult


st.set_page_config(
    page_title="叠读 LayerRead",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from layerread.connector_ui import render_connector_bridge


_REFERENCE_CATEGORY_LABELS = {
    "exact": "格式正确",
    "format_corrected": "单编号格式修正",
    "split_list": "合并引用拆分",
    "expanded_range": "引用范围展开",
    "partially_invalid": "部分编号不存在",
    "nonexistent": "编号不存在",
    "invalid_range": "无效引用范围",
    "unparseable": "无法安全解析",
}

_REFERENCE_LOCATION_LABELS = {
    "reading_decision": "阅读决策",
    "promotion_assessment": "推广判断",
    "fluff_assessment": "水文判断",
    "core_analysis": "核心分析",
    "explanations": "AI补充",
    "overall_value": "整体价值",
    "information_density": "信息密度",
    "evidence_quality": "证据质量",
    "original_analysis": "原创分析",
    "goal_relevance": "目标相关性",
    "promotion_likelihood": "推广可能性",
    "fluff_likelihood": "水文可能性",
    "confidence": "判断置信度",
    "valuable_parts": "仍有价值部分",
    "skippable_parts": "可跳过部分",
    "structure": "文章结构",
    "knowledge_points": "知识点",
    "claims": "观点与证据",
    "limitations": "局限",
    "counterarguments": "反方观点",
    "implicit_assumptions": "隐含假设",
    "possible_counterexamples": "可能反例",
    "applicability_boundaries": "适用边界",
    "items_to_verify": "待验证内容",
    "paragraph_refs": "引用",
}

_CONNECTOR_TAB_LABEL = "Chrome 一键导入"
_DEMO_PUBLIC_ORIGIN = "https://layerread-agentic-reader-demo.streamlit.app"
_CONNECTOR_VERSION = connector_version()
_MAIN_SECTIONS = (
    "导入与判断",
    "文章分析",
    "文章学习",
    "导出",
    "证据与核验",
)
_JUDGMENT_TABS = ("导入文章", "处理结果", "阅读决策", "推广与水文")
_ARTICLE_ANALYSIS_TABS = ("深度解析", "知识点", "可视化")
_ARTICLE_LEARNING_TABS = ("主动学习", "文章 Chat")
_EVIDENCE_TABS = ("论证与证据", "引用诊断", "引用原文核对")


def _connector_import_token_from_query() -> str:
    for name in ("layerread_import", "deepread_import"):
        value = st.query_params.get(name)
        if isinstance(value, list):
            value = value[-1] if value else ""
        if isinstance(value, str) and value:
            return value
    return ""


def _reference_location_label(location: str) -> str:
    label = location
    for key, replacement in _REFERENCE_LOCATION_LABELS.items():
        label = label.replace(key, replacement)
    return label


def initialize_state(profile: RuntimeProfile | None = None) -> None:
    """Create the session keys used by the article-analysis workflow."""

    profile = profile or read_runtime_profile()
    default_import_tab = (
        _CONNECTOR_TAB_LABEL if profile.connector_enabled else "URL 导入"
    )

    defaults = {
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
        "import_tab": default_import_tab,
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
        "article_draft": None,
        "analysis": None,
        "analysis_error": None,
        "analysis_phase": "idle",
        "analysis_reservation_id": "",
        "learning_session": None,
        "learning_error": None,
        "learning_phase": "idle",
        "chat_session": None,
        "chat_error": None,
        "chat_phase": "idle",
        "chat_pending_question": "",
        "visualizations": None,
        "local_storage_initialized": False,
        "current_record_id": None,
        "storage_error": None,
        "storage_notice": None,
        "storage_last_saved_at": None,
        "demo_privacy_accepted": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    allowed_import_tabs = {"URL 导入", "粘贴正文"}
    if profile.connector_enabled:
        allowed_import_tabs.add(_CONNECTOR_TAB_LABEL)
    if st.session_state.import_tab not in allowed_import_tabs:
        st.session_state.import_tab = default_import_tab

    navigation_defaults = {
        "main_section": (_MAIN_SECTIONS, "导入与判断"),
        "judgment_tab": (_JUDGMENT_TABS, "导入文章"),
        "article_analysis_tab": (_ARTICLE_ANALYSIS_TABS, "深度解析"),
        "article_learning_tab": (_ARTICLE_LEARNING_TABS, "主动学习"),
        "evidence_tab": (_EVIDENCE_TABS, "论证与证据"),
    }
    for key, (options, fallback) in navigation_defaults.items():
        pending_key = f"pending_{key}"
        pending_value = st.session_state.pop(pending_key, None)
        if pending_value in options:
            st.session_state[key] = pending_value
        elif st.session_state.get(key) not in options:
            st.session_state[key] = fallback

    connector_import_token = None
    if profile.connector_enabled:
        connector_import_token = _connector_import_token_from_query()
    else:
        st.query_params.pop("layerread_import", None)
        st.query_params.pop("deepread_import", None)
    if (
        connector_import_token
        and connector_import_token
        != st.session_state.connector_routed_import_token
    ):
        st.session_state.main_section = "导入与判断"
        st.session_state.judgment_tab = "导入文章"
        st.session_state.import_tab = _CONNECTOR_TAB_LABEL
        st.session_state.connector_import_notice = None
        st.session_state.connector_routed_import_token = connector_import_token

    legacy_draft = st.session_state.article_draft
    if legacy_draft is not None:
        try:
            st.session_state.article_draft = upgrade_article_draft(legacy_draft)
        except ArticleValidationError as exc:
            st.session_state.article_draft = None
            st.session_state.analysis = None
            st.session_state.learning_session = None
            st.session_state.chat_session = None
            st.session_state.chat_phase = "idle"
            st.session_state.chat_pending_question = ""
            st.session_state.visualizations = None
            st.session_state.storage_error = str(exc)


def render_header(profile: RuntimeProfile) -> None:
    st.title("叠读 LayerRead")
    st.markdown(
        "把一篇文章变成一次可追溯、需要主动思考、最终能够沉淀的深度学习过程。"
    )
    st.caption("v0.8.0-dev · Agentic 阅读、主动学习、引用核验与 Markdown 导出")
    if profile.is_demo:
        st.info(
            "当前为在线体验模式：支持 Chrome Connector、图片和视觉分析；"
            "不开放 Notion，也不会把文章、图片、分析或对话保存到文章库。"
        )
        st.caption(
            "点击模型功能后，当前文章及相关图片可能发送到 LayerRead 服务器和模型供应商。"
            "请勿提交机密、个人隐私或无权处理的内容。"
        )
    elif profile.is_portfolio:
        st.info(
            "当前为 Portfolio Edition：模型配置从本机 .env 读取；"
            "API Key 不在网页中输入，也不会写入 SQLite 或导出文件。"
        )
        st.caption(
            "Copyright © 2026 Dchaoqun · AGPL-3.0-only · 不提供任何担保 · "
            "[查看许可证与源代码](https://github.com/Dchaoqun/layerread-agentic-reader)"
        )


def render_model_configuration(
    results: tuple[ModelConfigResult, ...],
    *,
    portfolio_env: bool = False,
) -> ModelConfigResult:
    """Select a ready provider and show status without exposing credentials."""

    with st.sidebar:
        st.header("模型")
        ready_results = [result for result in results if result.is_ready]
        if ready_results:
            result_by_id = {result.provider_id: result for result in ready_results}
            if len(ready_results) > 1:
                selected_provider = st.segmented_control(
                    "选择模型",
                    options=list(result_by_id),
                    default=ready_results[0].provider_id,
                    required=True,
                    format_func=lambda provider_id: (
                        result_by_id[provider_id].settings.model
                        if result_by_id[provider_id].settings is not None
                        else provider_id
                    ),
                    key="model_provider",
                    width="stretch",
                )
                result = result_by_id[str(selected_provider)]
            else:
                result = ready_results[0]

            settings = result.settings
            assert settings is not None
            st.caption(f"当前模型：{settings.model}")
            if portfolio_env:
                st.caption("配置来源：项目根目录的本机 .env（该文件已被 Git 忽略）。")
                if settings.supports_vision:
                    st.warning(
                        "已通过 LLM_SUPPORTS_VISION=true 启用图片输入。"
                        "主动点击分析后，文章图片可能发送给模型供应商并产生额外费用。"
                    )
                else:
                    st.caption(
                        "图片输入未启用：已导入图片只保存在本地，"
                        "不会进入模型请求。"
                    )
            return result

        if portfolio_env:
            st.warning(
                "模型尚未配置。请停止应用，在项目根目录将 .env.example "
                "复制为 .env，填写 LLM_API_KEY、LLM_BASE_URL 和 LLM_MODEL，"
                "然后重新启动。"
            )
            for error in results[0].errors:
                st.caption(error)
        else:
            st.warning("当前没有可用模型，请联系管理员检查后端配置。")
        return results[0]


def _switch_to_manual_import() -> None:
    st.session_state.import_tab = "粘贴正文"


def _run_url_extraction() -> None:
    result = extract_article_from_url(st.session_state.url_input)
    st.session_state.url_extraction_result = result
    if result.is_success:
        st.session_state.url_title_input = result.title
        st.session_state.url_account_input = result.account
        st.session_state.url_author_input = result.author
        st.session_state.url_published_at_input = result.published_at
        st.session_state.url_source_input = result.source_url
        st.session_state.url_body_input = result.body


def _populate_connector_preview(result: ConnectorArticleImportResult) -> None:
    st.session_state.connector_import_result = result
    st.session_state.connector_title_input = result.title
    st.session_state.connector_account_input = result.account
    st.session_state.connector_author_input = result.author
    st.session_state.connector_published_at_input = result.published_at
    st.session_state.connector_source_input = result.source_url
    st.session_state.connector_body_input = result.body


def _confirmed_connector_draft() -> ArticleDraft | None:
    """Return the processed Connector article that can safely refill the editor."""

    draft = st.session_state.article_draft
    if (
        draft is not None
        and is_connector_extraction_method(draft.extraction_method)
        and draft.extraction_status == "confirmed"
    ):
        return draft
    return None


def _restore_connector_editor_from_draft() -> None:
    """Rebuild transient Connector widget values from the canonical article.

    Streamlit may stop rendering these widgets while the user visits another
    main section. The processed ArticleDraft is the durable in-session source
    of truth, so an empty editor is restored from it without replacing a
    non-empty, potentially unsaved edit.
    """

    draft = _confirmed_connector_draft()
    if draft is None or str(st.session_state.get("connector_body_input", "")).strip():
        return
    st.session_state.connector_title_input = draft.source_title
    st.session_state.connector_account_input = draft.account
    st.session_state.connector_author_input = draft.author
    st.session_state.connector_published_at_input = draft.published_at
    st.session_state.connector_source_input = draft.source_url
    st.session_state.connector_body_input = draft.raw_text


def _run_connector_import(payload: object) -> bool:
    st.session_state.connector_import_error = None
    st.session_state.connector_import_notice = None
    st.session_state.connector_import_result = None
    try:
        result = parse_connector_payload(payload)
    except ConnectorImportError as exc:
        st.session_state.connector_import_error = str(exc)
        return False
    _populate_connector_preview(result)
    if isinstance(payload, dict):
        import_id = payload.get("import_id")
        if isinstance(import_id, str):
            st.session_state.connector_ack_import_token = import_id
    st.query_params.pop("layerread_import", None)
    st.query_params.pop("deepread_import", None)
    return True


def _clear_connector_query() -> None:
    st.query_params.pop("layerread_import", None)
    st.query_params.pop("deepread_import", None)


def _record_connector_unavailable() -> None:
    _clear_connector_query()
    st.session_state.connector_import_notice = (
        "Connector 已连接，但当前没有待接收文章。请打开微信公众号文章，"
        "点击扩展图标，并等待高清图片处理完成后自动返回这里。"
    )


def _handle_connector_ack(component_result: object) -> None:
    acknowledged = getattr(component_result, "acknowledged", None)
    if (
        isinstance(acknowledged, str)
        and acknowledged == st.session_state.connector_ack_import_token
    ):
        st.session_state.connector_ack_import_token = None


@st.dialog(
    "正在接收微信文章",
    width="medium",
    dismissible=False,
    icon="spinner",
)
def _render_connector_receive_dialog(import_token: str) -> None:
    st.write("请保持本页面打开，LayerRead 正在领取正文和处理后的高清图片。")
    st.warning(
        "微信公众号图片可能采用懒加载。如果刚才没有先等待原文页面加载完成、"
        "从顶部滚动到底部并确认所有图片都已显示，本次接收的图片可能不完整。"
        "可以取消接收，完成上述步骤后重新点击 Connector。"
    )
    st.caption(
        "通常会在 5–20 秒内完成。文章较长或图片较多时可能更久；"
        "如果连接中断，下面会显示可重试的具体状态。"
    )
    component_result = render_connector_bridge(
        key="layerread_connector_receive_dialog",
        import_token=import_token,
        ack_import_token=st.session_state.connector_ack_import_token or "",
    )
    _handle_connector_ack(component_result)
    connector_payload = getattr(component_result, "received", None)
    if connector_payload is not None:
        if _run_connector_import(connector_payload):
            st.rerun()
        st.error(
            "文章包已经到达，但未通过安全校验："
            f"{st.session_state.connector_import_error}"
        )
    if getattr(component_result, "unavailable", None):
        _record_connector_unavailable()
        st.rerun()
    st.caption("完成接收后弹窗会自动关闭，并在导入页显示文章预览。")
    if st.button(
        "取消本次接收",
        icon=":material/close:",
        width="stretch",
    ):
        _clear_connector_query()
        st.session_state.connector_import_notice = (
            "已取消本次接收。请回到微信公众号文章，重新点击 Connector。"
        )
        st.rerun()


def _render_connector_import_status(result: ConnectorArticleImportResult) -> None:
    st.success("文章已从 Chrome 安全接收。请检查并编辑内容，确认后才会进入正式分析。")
    st.caption(f"来源：{result.filename} · 接收方式：{result.extraction_method}")
    if result.media_detected:
        if result.images:
            st.success(
                f"检测到 {result.image_reference_count} 张正文图片，"
                f"已接收 {len(result.images)} 个高清图片或长图分片。"
            )
            st.caption(
                "图片保持文字可读性优先；只有模型配置明确启用视觉能力时，"
                "点击分析才会把图片发送给模型供应商。"
            )
            preview_images = list(result.images[:6])
            st.image(
                [image.data_url for image in preview_images],
                caption=[image.image_id for image in preview_images],
                width=180,
            )
            if len(result.images) > len(preview_images):
                st.caption(
                    f"另有 {len(result.images) - len(preview_images)} 个图片分片未在此处展开。"
                )
        else:
            st.warning(
                f"检测到 {result.image_reference_count} 个图片引用或图片文件，"
                "但没有接收到可安全分析的图片本体。"
            )
    st.info(
        "请对照微信原文核对关键图片和图片数量。若有缺失，先不要确认文章；"
        "返回原文刷新页面，从顶部滚动到底部，待所有图片显示后重新点击 Connector。"
    )
    for note in result.notes:
        st.caption(note)


def _render_connector_confirmation_form(form_key: str) -> bool:
    with st.form(form_key, clear_on_submit=False):
        metadata_columns = st.columns(2)
        with metadata_columns[0]:
            st.text_input(
                "标题", key="connector_title_input", persist_state="session"
            )
            st.text_input(
                "公众号/来源",
                key="connector_account_input",
                persist_state="session",
            )
            st.text_input(
                "作者", key="connector_author_input", persist_state="session"
            )
        with metadata_columns[1]:
            st.text_input(
                "发布时间",
                key="connector_published_at_input",
                persist_state="session",
            )
            st.text_input(
                "原文链接",
                key="connector_source_input",
                persist_state="session",
            )
        st.text_area(
            "导入正文（确认前可编辑）",
            key="connector_body_input",
            height=420,
            persist_state="session",
        )
        return st.form_submit_button(
            "确认内容并处理文章",
            type="primary",
            width="stretch",
        )


def _render_url_extraction_status(result: ArticleExtractionResult) -> None:
    if result.is_success:
        st.success("正文提取成功。请检查并编辑元数据和正文，确认后才会进入正式分析。")
    else:
        failure = result.attempts[-1].message if result.attempts else "未能提取正文。"
        st.warning(f"URL 导入失败：{failure}")
        st.info("你可以重试，或切换到“粘贴正文”使用完整的手动导入流程。")
        st.button(
            "改用正文粘贴",
            icon=":material/content_paste:",
            on_click=_switch_to_manual_import,
            key="fallback_to_manual_import",
        )
    if result.attempts:
        with st.expander("查看提取过程"):
            for attempt in result.attempts:
                status = "成功" if attempt.success else "未通过"
                st.write(f"{attempt.adapter} · {status}：{attempt.message}")


def render_input_form(profile: RuntimeProfile) -> tuple[bool, bool, bool]:
    st.subheader("导入文章")
    if profile.connector_enabled:
        st.write(
            "推荐从 Chrome 一键导入公众号文章；也可以从链接提取或直接粘贴正文。"
        )
        st.info(
            "Chrome 一键导入会自动保留正文中的重要图片，包括 PPT 截图、信息长图、"
            "流程图和图表；其他导入方式仍可能只有文字。"
        )
        if profile.is_portfolio:
            st.caption(
                "Connector 导入的正文和图片保存在本机 SQLite。图片会增加数据库占用；"
                "只有在侧边栏明确启用视觉能力并点击分析后，才会发送给你的模型供应商。"
            )
        elif profile.is_demo:
            st.caption(
                "在线 Demo 会把你主动发送的正文和图片传到服务器；内容仅保留在当前页面会话。"
                "点击分析后，相关内容可能继续发送给模型供应商，并受每日额度限制。"
            )
    else:
        st.write("当前运行配置支持从链接提取或直接粘贴正文。")
        st.caption("此运行配置未启用浏览器扩展入口。")

    preference_columns = st.columns([2, 1, 1])
    with preference_columns[0]:
        st.text_input(
            "阅读目的（选填）",
            key="reading_goal",
            placeholder="例如：理解这项技术是否适合当前项目",
            persist_state="session",
        )
    with preference_columns[1]:
        st.selectbox(
            "主题熟悉度",
            options=["入门", "进阶", "专业"],
            key="familiarity",
            persist_state="session",
        )
    with preference_columns[2]:
        st.selectbox(
            "关注方向",
            options=["通用", "技术", "商业", "投资", "实践"],
            key="focus_area",
            persist_state="session",
        )
    if profile.image_features_enabled:
        st.checkbox(
            "这篇文章的关键内容依赖图片、图表或视频",
            key="media_dependency",
            help="选中后，如果当前导入方式没有带入图片，系统会提醒分析结果可能不完整。",
            persist_state="session",
        )
    else:
        st.session_state.media_dependency = False

    connector_tab = None
    if profile.connector_enabled:
        _restore_connector_editor_from_draft()
        connector_tab, url_tab, manual_tab = st.tabs(
            [_CONNECTOR_TAB_LABEL, "URL 导入", "粘贴正文"],
            key="import_tab",
            on_change="rerun",
        )
    else:
        url_tab, manual_tab = st.tabs(
            ["URL 导入", "粘贴正文"],
            key="import_tab",
            on_change="rerun",
        )
    manual_submitted = False
    url_confirmed = False
    connector_confirmed = False
    if manual_tab.open:
        with manual_tab:
            st.caption("手动粘贴始终可用，不依赖目标网站是否允许自动访问。")
            with st.form("article_form", clear_on_submit=False):
                st.text_area(
                    "文章正文",
                    key="article_input",
                    height=360,
                    placeholder="在这里粘贴完整文章正文。建议至少包含 100 个字和多个段落。",
                    persist_state="session",
                )
                manual_submitted = st.form_submit_button(
                    "处理文章",
                    type="primary",
                    width="stretch",
                )
    if connector_tab is not None and connector_tab.open:
        with connector_tab:
            st.warning(
                "导入前的必要步骤：先等待微信文章加载完成，再从顶部滚动到底部一次，"
                "确认正文中的图片、PPT、图表和长图都已显示，然后再点击 LayerRead "
                "Connector。微信图片可能懒加载，未滚动到的图片可能无法导入。"
            )
            st.caption(
                "点击 Connector 后等待高清图片处理完成，扩展会自动打开这里；"
                "最后对照原文检查预览并确认文章。"
            )
            connector_import_token = _connector_import_token_from_query()
            if connector_import_token:
                _render_connector_receive_dialog(connector_import_token)
            else:
                connector_component = render_connector_bridge(
                    key="layerread_connector_component",
                    ack_import_token=(
                        st.session_state.connector_ack_import_token or ""
                    ),
                )
                _handle_connector_ack(connector_component)
                connector_payload = getattr(connector_component, "received", None)
                if connector_payload is not None:
                    _run_connector_import(connector_payload)
                if getattr(connector_component, "unavailable", None):
                    _record_connector_unavailable()

            connector_error = st.session_state.connector_import_error
            connector_notice = st.session_state.connector_import_notice
            connector_result = st.session_state.connector_import_result
            confirmed_draft = _confirmed_connector_draft()
            if connector_error:
                st.error(f"Chrome 一键导入失败：{connector_error}")
            if connector_notice:
                st.info(connector_notice)
            if (
                connector_result is not None
                and is_connector_extraction_method(connector_result.extraction_method)
            ):
                _render_connector_import_status(connector_result)
                connector_confirmed = _render_connector_confirmation_form(
                    "connector_confirmation_form"
                )
            elif confirmed_draft is not None:
                st.success("当前显示已处理文章；可以检查或修改内容后重新确认。")
                st.caption(
                    f"已保留 {confirmed_draft.paragraph_count} 个正文段落和 "
                    f"{len(confirmed_draft.images)} 张重要图片。"
                )
                connector_confirmed = _render_connector_confirmation_form(
                    "connector_confirmation_form"
                )
            elif not connector_import_token:
                with st.expander("首次安装 LayerRead Connector"):
                    if profile.is_demo:
                        st.warning(
                            "本地源码版 Connector 只连接 localhost，不能被在线 Demo 检测；"
                            "在线版 0.4.0 也无法连接 Streamlit Cloud 的内嵌应用页面。"
                        )
                        st.download_button(
                            f"下载在线 Demo Connector {_CONNECTOR_VERSION}",
                            data=build_online_connector_archive(_DEMO_PUBLIC_ORIGIN),
                            file_name=(
                                f"layerread-online-connector-{_CONNECTOR_VERSION}.zip"
                            ),
                            mime="application/zip",
                            icon=":material/download:",
                            width="stretch",
                        )
                        st.markdown(
                            "1. 下载并解压压缩包。\n"
                            "2. 在 Chrome 打开 `chrome://extensions/`。\n"
                            "3. 开启“开发者模式”，点击“加载已解压的扩展程序”。\n"
                            "4. 选择解压后的 `layerread-online-connector` 目录。\n"
                            "5. 在扩展详情中打开“扩展程序选项”，选择“在线 Demo”并保存。\n"
                            "6. 刷新本页面，再点击“重新检测”。\n"
                            "7. 每次导入前，在微信原文中从顶部滚动到底部，确认全部图片已显示。"
                        )
                        st.caption(
                            f"当前在线包版本：{_CONNECTOR_VERSION}。它只额外授权当前 "
                            "LayerRead Demo 的精确 HTTPS 域名。"
                        )
                    else:
                        extension_path = (
                            Path(__file__).resolve().parent
                            / "browser_extension"
                            / "layerread_connector"
                        )
                        st.markdown(
                            "1. 在 Chrome 打开 `chrome://extensions/`。\n"
                            "2. 开启“开发者模式”，点击“加载已解压的扩展程序”。\n"
                            "3. 选择下面的目录，并把扩展固定到工具栏。"
                        )
                        st.code(str(extension_path), language=None)
                    st.caption("这是一次性设置；以后每篇文章只需点击一次扩展图标。")
    if url_tab.open:
        with url_tab:
            st.caption("先提取、再预览编辑；未确认的网页内容不会进入分析。")
            with st.form("url_import_form", clear_on_submit=False):
                st.text_input(
                    "公众号文章或网页链接",
                    key="url_input",
                    placeholder="https://mp.weixin.qq.com/s/…",
                    icon=":material/link:",
                    persist_state="session",
                )
                extract_clicked = st.form_submit_button(
                    "提取正文",
                    type="primary",
                    width="stretch",
                )
            if extract_clicked:
                with st.status("正在安全访问网页并提取正文…", expanded=True) as status:
                    _run_url_extraction()
                    result = st.session_state.url_extraction_result
                    if result.is_success:
                        status.update(label="正文提取完成", state="complete", expanded=False)
                    else:
                        status.update(label="URL 导入未完成", state="error", expanded=False)

            result = st.session_state.url_extraction_result
            if result is not None:
                _render_url_extraction_status(result)
            if result is not None and result.is_success:
                with st.form("url_confirmation_form", clear_on_submit=False):
                    metadata_columns = st.columns(2)
                    with metadata_columns[0]:
                        st.text_input(
                            "标题", key="url_title_input", persist_state="session"
                        )
                        st.text_input(
                            "公众号/来源",
                            key="url_account_input",
                            persist_state="session",
                        )
                        st.text_input(
                            "作者", key="url_author_input", persist_state="session"
                        )
                    with metadata_columns[1]:
                        st.text_input(
                            "发布时间",
                            key="url_published_at_input",
                            persist_state="session",
                        )
                        st.text_input(
                            "原文链接",
                            key="url_source_input",
                            persist_state="session",
                        )
                    st.text_area(
                        "提取正文（确认前可编辑）",
                        key="url_body_input",
                        height=420,
                        persist_state="session",
                    )
                    url_confirmed = st.form_submit_button(
                        "确认内容并处理文章",
                        type="primary",
                        width="stretch",
                    )
    return manual_submitted, url_confirmed, connector_confirmed


def handle_submission(
    manual_submitted: bool,
    url_confirmed: bool,
    connector_confirmed: bool,
    profile: RuntimeProfile,
    limiter: DemoLimiter | None,
) -> None:
    if not manual_submitted and not url_confirmed and not connector_confirmed:
        return

    try:
        media_dependency = st.session_state.media_dependency
        imported_images = ()
        detected_image_count = 0
        if connector_confirmed:
            result = st.session_state.connector_import_result
            confirmed_draft = _confirmed_connector_draft()
            extraction_method = (
                result.extraction_method
                if result is not None
                else (
                    confirmed_draft.extraction_method
                    if confirmed_draft is not None
                    else CONNECTOR_EXTRACTION_METHOD
                )
            )
            raw_text = st.session_state.connector_body_input
            metadata = {
                "source_title": st.session_state.connector_title_input,
                "source_url": st.session_state.connector_source_input,
                "account": st.session_state.connector_account_input,
                "author": st.session_state.connector_author_input,
                "published_at": st.session_state.connector_published_at_input,
                "extraction_method": extraction_method,
                "extraction_status": "confirmed",
            }
            media_dependency = media_dependency or bool(
                result is not None and result.media_detected
            )
            if result is not None:
                imported_images = result.images
                detected_image_count = result.image_reference_count
            elif confirmed_draft is not None:
                imported_images = confirmed_draft.images
                detected_image_count = confirmed_draft.image_count_detected
        elif url_confirmed:
            result = st.session_state.url_extraction_result
            extraction_method = result.extraction_method if result is not None else "url"
            raw_text = st.session_state.url_body_input
            metadata = {
                "source_title": st.session_state.url_title_input,
                "source_url": st.session_state.url_source_input,
                "account": st.session_state.url_account_input,
                "author": st.session_state.url_author_input,
                "published_at": st.session_state.url_published_at_input,
                "extraction_method": extraction_method,
                "extraction_status": "confirmed",
            }
        else:
            raw_text = st.session_state.article_input
            metadata = {}
        image_notice = ""
        if profile.is_demo:
            if limiter is None:
                st.error("在线体验限制服务暂不可用，请稍后重试。")
                return
            if len(raw_text) > limiter.policy.max_article_chars:
                st.error(
                    f"在线 Demo 单篇最多处理 {limiter.policy.max_article_chars:,} 个字符；"
                    "请缩短正文后重试。"
                )
                return
            limited_images = limit_images(imported_images, limiter.policy)
            imported_images = limited_images.images
            if limited_images.dropped_count:
                image_notice = (
                    f"为控制在线体验成本，已保留前 {len(imported_images)} 个图片或分片，"
                    f"忽略 {limited_images.dropped_count} 个超出限制的图片或分片。"
                )
        new_draft = process_article(
            raw_text=raw_text,
            reading_goal=st.session_state.reading_goal,
            familiarity=st.session_state.familiarity,
            focus_area=st.session_state.focus_area,
            media_dependency=media_dependency,
            images=imported_images,
            image_count_detected=detected_image_count,
            **metadata,
        )
    except ArticleValidationError as exc:
        st.error(str(exc))
        return

    if url_confirmed or connector_confirmed:
        st.session_state.article_input = raw_text
    previous = st.session_state.article_draft
    st.session_state.article_draft = new_draft
    if previous is None or (
        previous.numbered_text != new_draft.numbered_text
        or previous.reading_goal != new_draft.reading_goal
        or previous.familiarity != new_draft.familiarity
        or previous.focus_area != new_draft.focus_area
        or previous.media_dependency != new_draft.media_dependency
        or previous.images != new_draft.images
        or previous.source_url != new_draft.source_url
    ):
        st.session_state.analysis = None
        st.session_state.analysis_error = None
        st.session_state.learning_session = None
        st.session_state.learning_error = None
        st.session_state.chat_session = None
        st.session_state.chat_error = None
        st.session_state.chat_phase = "idle"
        st.session_state.chat_pending_question = ""
        st.session_state.visualizations = None
        st.session_state.current_record_id = None
        st.session_state.storage_last_saved_at = None

    st.session_state.storage_notice = " ".join(
        item
        for item in (
            "文章已完成清洗和段落编号。请检查结果，必要时可返回导入文章修改。",
            image_notice,
        )
        if item
    )
    st.session_state.pending_main_section = "导入与判断"
    st.session_state.pending_judgment_tab = "处理结果"
    st.rerun()


def render_result() -> None:
    draft = st.session_state.article_draft
    if draft is None:
        st.info("尚未处理文章。请先通过 Chrome、URL 或正文粘贴导入文章。")
        return

    if st.session_state.article_input != draft.raw_text:
        st.warning("输入框中的正文已经修改。请再次点击“处理文章”以更新下方结果。")

    st.divider()
    st.subheader("处理结果")

    metric_columns = st.columns(4)
    metric_columns[0].metric("原始字数", f"{draft.raw_char_count:,}")
    metric_columns[1].metric("清洗后字数", f"{draft.clean_char_count:,}")
    metric_columns[2].metric("有效段落", draft.paragraph_count)
    metric_columns[3].metric("移除段落", len(draft.removed_paragraphs))

    preference_summary = (
        f"阅读目的：{draft.reading_goal or '未填写'} ｜ "
        f"熟悉度：{draft.familiarity} ｜ 关注方向：{draft.focus_area}"
    )
    st.caption(preference_summary)

    if draft.source_url:
        st.markdown(f"**标题：** {draft.source_title or '未提取'}")
        source_details = [
            value
            for value in (
                f"来源：{draft.account}" if draft.account else "",
                f"作者：{draft.author}" if draft.author else "",
                f"发布时间：{draft.published_at}" if draft.published_at else "",
            )
            if value
        ]
        if source_details:
            st.caption(" ｜ ".join(source_details))
        st.link_button(
            "打开原文链接",
            draft.source_url,
            icon=":material/open_in_new:",
        )

    if draft.images:
        message = (
            f"已保留 {len(draft.images)} 个高清图片或长图分片，"
            "模型配置明确启用视觉能力后会结合正文分析。"
        )
        if draft.media_completeness == "images_included":
            st.success(message)
        else:
            st.warning(message + "部分正文图片因安全或容量限制未导入。")
    elif draft.media_dependency:
        st.warning(
            "你已标记这篇文章依赖图片、图表或视频。当前结果仅包含文字，"
            "后续分析可能遗漏媒体承载的信息。"
        )

    if draft.removed_paragraphs:
        with st.expander("查看清洗时移除的内容"):
            for paragraph in draft.removed_paragraphs:
                st.write(f"- {paragraph}")

    with st.expander("查看清洗后的编号正文"):
        st.code(draft.numbered_text, language=None, wrap_lines=True)


def _render_reference_summary(references: list[str], draft) -> None:
    """Show only existing paragraph IDs beside an AI-generated statement."""

    result = normalize_reference_values(references, draft)
    if result.valid:
        st.caption("模型引用（编号已验证，请核读语义是否支持）：" + " ".join(result.valid))
    if result.invalid:
        st.caption("⚠️ 存在未通过校验的引用，已隐藏。")


def _render_criterion(label: str, criterion: CriterionScore, draft) -> None:
    st.markdown(f"**{label} · {criterion.score}/100**")
    st.text(criterion.explanation)
    _render_reference_summary(criterion.paragraph_refs, draft)


def render_decision_card(analysis: Analysis, draft) -> None:
    decision = analysis.reading_decision
    st.subheader("阅读决策")
    st.success(decision.recommendation)
    st.markdown("**推荐阅读方式**")
    st.text(decision.recommended_approach)

    first_row = st.columns(4)
    first_row[0].metric("整体价值", decision.overall_value.score)
    first_row[1].metric("信息密度", decision.information_density.score)
    first_row[2].metric("证据质量", decision.evidence_quality.score)
    first_row[3].metric("目标相关性", decision.goal_relevance.score)
    second_row = st.columns(4)
    second_row[0].metric("原创分析", decision.original_analysis.score)
    second_row[1].metric("推广可能", f"{decision.promotion_likelihood.score}%")
    second_row[2].metric("水文可能", f"{decision.fluff_likelihood.score}%")
    second_row[3].metric("判断置信度", f"{decision.confidence.score}%")

    with st.expander("查看各项判断依据"):
        criteria = [
            ("整体价值", decision.overall_value),
            ("信息密度", decision.information_density),
            ("证据质量", decision.evidence_quality),
            ("原创分析", decision.original_analysis),
            ("目标相关性", decision.goal_relevance),
            ("推广可能性", decision.promotion_likelihood),
            ("水文可能性", decision.fluff_likelihood),
            ("判断置信度", decision.confidence),
        ]
        for label, criterion in criteria:
            _render_criterion(label, criterion, draft)
            st.divider()


def _render_cited_items(
    title: str,
    items: list[CitedText | InferredText],
    draft,
) -> None:
    st.markdown(f"#### {title}")
    if not items:
        st.caption("模型未列出相关内容。")
        return
    for item in items:
        st.text(f"• {item.text}")
        _render_reference_summary(item.paragraph_refs, draft)


def _get_reference_diagnostics(
    analysis: Analysis,
    draft,
) -> ReferenceDiagnostics | None:
    """Read stored diagnostics or derive them locally for an older session result."""

    diagnostics = getattr(analysis, "reference_diagnostics", None)
    if diagnostics is not None:
        return diagnostics
    try:
        payload = parse_analysis(analysis.raw_response)
        _, diagnostics = normalize_references(payload, draft)
    except (AnalysisError, AttributeError, TypeError, ValueError):
        return None
    return diagnostics


def _render_risk_banner(diagnostics: ReferenceDiagnostics, label: str) -> None:
    message = f"{label}：{diagnostics.risk_summary}"
    if diagnostics.risk_level == "high":
        st.error(message)
    elif diagnostics.risk_level == "medium":
        st.warning(message)
    elif diagnostics.risk_level == "low":
        st.info(message)
    else:
        st.success(message)


def render_reference_diagnostics(analysis: Analysis, draft) -> None:
    diagnostics = _get_reference_diagnostics(analysis, draft)
    if diagnostics is None:
        st.error("当前会话的分析结果无法重新解析，请重新分析文章以生成引用诊断。")
        return

    st.caption(
        "诊断先在本地修正安全可判定的格式问题，再统计真正不存在或无法解析的引用；"
        "不会因为格式修正额外调用模型。"
    )
    metrics = st.columns(5)
    metrics[0].metric("引用总次数", diagnostics.total_mentions)
    metrics[1].metric("原格式有效", diagnostics.exact_valid_mentions)
    metrics[2].metric("格式自动修正", diagnostics.corrected_valid_mentions)
    metrics[3].metric("编号不存在", diagnostics.nonexistent_mentions)
    metrics[4].metric("无法安全解析", diagnostics.unparseable_mentions)
    _render_risk_banner(diagnostics, "最终展示结果")

    initial = getattr(analysis, "initial_reference_diagnostics", None)
    if initial is not None and initial.model_dump() != diagnostics.model_dump():
        st.markdown("#### 模型修复前后对比")
        comparison = st.columns(4)
        comparison[0].metric("修复前未解决", initial.invalid_mentions)
        comparison[1].metric(
            "修复后未解决",
            diagnostics.invalid_mentions,
            delta=diagnostics.invalid_mentions - initial.invalid_mentions,
            delta_color="inverse",
        )
        comparison[2].metric("修复前受影响位置", initial.affected_locations)
        comparison[3].metric("修复后受影响位置", diagnostics.affected_locations)
        _render_risk_banner(initial, "模型原始输出")

    corrected_items = [
        item
        for item in diagnostics.items
        if item.status in {"auto_corrected", "partially_invalid"}
        and item.valid_refs
    ]
    st.markdown("#### 已自动修正的格式问题")
    if corrected_items:
        st.dataframe(
            [
                {
                    "位置": _reference_location_label(item.location),
                    "原始写法": item.original,
                    "修正结果": " ".join(item.valid_refs),
                    "类型": _REFERENCE_CATEGORY_LABELS[item.category],
                    "说明": item.reason,
                }
                for item in corrected_items
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.success("没有需要自动修正的引用格式。")

    unresolved_items = [item for item in diagnostics.items if item.invalid_refs]
    st.markdown("#### 仍未解决的引用")
    if unresolved_items:
        occurrence_counts: dict[str, int] = {}
        occurrence_locations: dict[str, list[str]] = {}
        occurrence_categories: dict[str, set[str]] = {}
        occurrence_statements: dict[str, list[str]] = {}
        for item in unresolved_items:
            for reference in item.invalid_refs:
                occurrence_counts[reference] = occurrence_counts.get(reference, 0) + 1
                occurrence_locations.setdefault(reference, []).append(
                    _reference_location_label(item.location)
                )
                occurrence_categories.setdefault(reference, set()).add(item.category)
                if item.statement:
                    statements = occurrence_statements.setdefault(reference, [])
                    if item.statement not in statements:
                        statements.append(item.statement)
        st.dataframe(
            [
                {
                    "原始/规范化引用": reference,
                    "出现次数": count,
                    "类型": "、".join(
                        _REFERENCE_CATEGORY_LABELS[category]
                        for category in sorted(occurrence_categories[reference])
                    ),
                    "影响位置": "；".join(occurrence_locations[reference]),
                    "相关模型陈述": "；".join(occurrence_statements.get(reference, [])),
                }
                for reference, count in occurrence_counts.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.warning(
            f"共有 {diagnostics.affected_locations} 个分析位置受影响，其中 "
            f"{diagnostics.zero_valid_locations} 个位置没有任何可验证出处。"
        )
    else:
        st.success("格式修正后，没有发现仍然不存在或无法解析的引用。")

    st.markdown("#### 语义支持人工核读")
    st.caption(
        "编号存在不代表段落一定支持模型结论。下面把模型陈述和对应原文放在一起，"
        "用于抽查是否存在真实但不相关的引用。"
    )
    paragraph_map = article_paragraph_map(draft)
    semantic_rows = []
    for item in diagnostics.items:
        if not item.valid_refs:
            continue
        semantic_rows.append(
            {
                "位置": _reference_location_label(item.location),
                "模型陈述": item.statement or "请结合所在分析模块核读",
                "引用编号": " ".join(dict.fromkeys(item.valid_refs)),
                "对应原文": " | ".join(
                    f"{reference} {paragraph_map[reference]}"
                    for reference in dict.fromkeys(item.valid_refs)
                ),
            }
        )
    if semantic_rows:
        st.dataframe(semantic_rows, hide_index=True, width="stretch")
    else:
        st.warning("没有可用于语义核读的有效引用。")

    st.info(
        "风险等级衡量的是引用编号能否对应当前正文，不是完整的内容幻觉率。"
        "即使编号存在，仍需抽查原文是否在语义上支持模型结论。"
    )


def render_analysis_view(
    view: str,
    analysis: Analysis,
    draft,
    settings: ModelSettings | None,
    limiter: DemoLimiter | None = None,
) -> None:
    core = analysis.core_analysis
    if view == "深度解析":
        st.markdown("#### 文章试图解决的问题")
        st.text(core.core_question)
        st.markdown("#### 一句话核心结论")
        st.text(core.one_sentence_conclusion)
        st.markdown("#### 作者希望读者接受的观点")
        st.text(core.author_intended_view)

        st.markdown("#### 文章结构")
        for item in sorted(core.structure, key=lambda value: value.order):
            st.markdown(f"**第 {item.order} 部分**")
            st.text(item.title)
            st.text(item.summary)
            _render_reference_summary(item.paragraph_refs, draft)

        _render_cited_items("局限", core.limitations, draft)
        _render_cited_items("反方观点", core.counterarguments, draft)
        _render_cited_items("隐含假设", core.implicit_assumptions, draft)
        _render_cited_items("可能反例", core.possible_counterexamples, draft)
        _render_cited_items("成立边界", core.applicability_boundaries, draft)
        _render_cited_items("待验证内容", core.items_to_verify, draft)

        if analysis.explanations:
            st.markdown("#### AI 补充")
            for explanation in analysis.explanations:
                if explanation.kind == "AI解释":
                    st.info("AI 解释")
                else:
                    st.warning("AI 推断")
                st.text(explanation.text)
                _render_reference_summary(explanation.paragraph_refs, draft)

    if view == "知识点":
        for index, point in enumerate(core.knowledge_points, start=1):
            with st.expander(f"知识点 {index}", expanded=index == 1):
                st.markdown("**名称**")
                st.text(point.title)
                st.markdown("**定义**")
                st.text(point.definition)
                st.markdown("**通俗解释**")
                st.text(point.plain_explanation)
                st.markdown("**例子**")
                st.text(point.example)
                st.markdown("**适用范围**")
                st.text(point.applicability)
                if point.common_misconceptions:
                    st.markdown("**常见误解**")
                    st.text("；".join(point.common_misconceptions))
                _render_reference_summary(point.paragraph_refs, draft)

    if view == "论证与证据":
        for index, claim in enumerate(core.claims, start=1):
            st.markdown(f"#### 观点 {index}")
            st.text(claim.claim)
            labels = st.columns(3)
            labels[0].text(f"类型：{claim.classification}")
            labels[1].text(f"证据类型：{claim.evidence_type}")
            labels[2].text(f"证据强度：{claim.evidence_strength}")
            st.markdown("**原文证据概括**")
            st.text(claim.evidence)
            if claim.logical_leaps:
                st.warning("AI 推断的逻辑跳跃")
                st.text("；".join(claim.logical_leaps))
            _render_reference_summary(claim.paragraph_refs, draft)
            st.divider()

    if view == "推广与水文":
        promotion = analysis.promotion_assessment
        fluff = analysis.fluff_assessment
        left, right = st.columns(2)
        with left:
            st.markdown("#### 推广倾向判断")
            st.metric("推广概率", f"{promotion.probability}%")
            st.markdown("**潜在推广对象**")
            st.text("、".join(promotion.potential_targets) or "未识别")
            st.markdown("**判断理由**")
            st.text(promotion.rationale)
            _render_reference_summary(promotion.paragraph_refs, draft)
            _render_cited_items("即使有推广倾向仍有价值的部分", promotion.valuable_parts, draft)
        with right:
            st.markdown("#### 水文判断")
            st.metric("可压缩程度", f"{fluff.compressibility}%")
            st.text(f"有效观点数量：{fluff.effective_point_count}")
            st.markdown("**重复情况**")
            st.text(fluff.repetition)
            st.markdown("**标题是否大于内容**")
            st.text(fluff.title_overstates_content)
            _render_reference_summary(fluff.paragraph_refs, draft)
            _render_cited_items("可以跳过的部分", fluff.skippable_parts, draft)

    if view == "引用诊断":
        render_reference_diagnostics(analysis, draft)

    if view == "引用原文核对":
        paragraph_map = article_paragraph_map(draft)
        diagnostics = _get_reference_diagnostics(analysis, draft)
        report = validate_references(analysis, draft) if diagnostics is None else None
        valid_references = (
            list(report.valid)
            if report is not None
            else diagnostics.unique_valid_references
        )
        invalid_references = (
            list(report.invalid)
            if report is not None
            else diagnostics.invalid_references
        )
        st.caption(
            "以下仅确认引用编号确实存在于当前正文；系统不宣称该段落在语义上一定支持模型判断，请结合原文核读。"
        )
        if invalid_references:
            st.warning(
                f"格式修正后仍有 {len(invalid_references)} 个未解决引用，"
                "已禁止作为有效出处展示。"
            )
        for reference in valid_references:
            with st.expander(reference):
                st.code(paragraph_map[reference], language=None, wrap_lines=True)

    if view == "主动学习":
        render_learning_tab(analysis, draft, settings, limiter)

    if view == "文章 Chat":
        render_chat_tab(
            analysis,
            draft,
            st.session_state.learning_session,
            settings,
            limiter,
        )

    if view == "可视化":
        visualizations = st.session_state.visualizations
        if visualizations is None or visualizations.article_id != analysis.article_id:
            visualizations = build_visualizations(draft, analysis)
            st.session_state.visualizations = visualizations
        render_visualization_tab(visualizations)


def _render_analysis_status(analysis: Analysis) -> None:
    if analysis.validation_status == "repaired":
        st.info("模型初次输出未通过校验，系统已自动修复并重新校验。")
    elif analysis.validation_status == "valid_with_auto_corrections":
        st.info("模型输出仅存在可安全修正的格式差异，系统已在本地规范化，没有额外调用模型。")
    elif analysis.validation_status == "valid_with_invalid_references":
        st.warning("部分引用在自动修复后仍无效，页面已隐藏这些引用。")
    st.caption(
        f"模型：{analysis.model_name} · Prompt：{analysis.prompt_version} · "
        f"生成时间：{analysis.created_at.astimezone().strftime('%Y-%m-%d %H:%M')}"
    )


def _online_demo_analysis_error(message: str) -> str:
    if message.startswith("无法连接阿里云百炼"):
        return (
            "在线 Demo 后端暂时未能连接阿里云百炼；这不是访客本地 VPN "
            "或浏览器设置导致的。请稍后重试。"
        )
    return message


def _release_demo_analysis_quota(
    message: str,
    limiter: DemoLimiter,
    reservation_id: str,
) -> str:
    if limiter.release_failed("analysis", reservation_id):
        return (
            message
            + " 本次没有生成可用结果，已退还文章分析次数；"
            "失败尝试仍计入每日安全尝试上限和全站预算。"
        )
    return (
        message
        + " 系统暂未确认文章分析次数已退还。请不要继续重复测试，"
        "并把侧栏“遇到问题？”中的体验支持码发给维护者。"
    )


def _analysis_is_busy() -> bool:
    return st.session_state.get("analysis_phase") in {"queued", "running"}


def _recover_interrupted_analysis(
    profile: RuntimeProfile,
    limiter: DemoLimiter | None,
) -> None:
    """Unlock the UI when a rerun interrupted an in-flight model request."""

    if st.session_state.get("analysis_phase") != "running":
        return

    message = (
        "上一次深度分析因页面切换、刷新或连接中断而停止，正文仍然保留；"
        "请在处理结果中重新开始。模型供应商仍可能记录已经发送的请求。"
    )
    reservation_id = st.session_state.get("analysis_reservation_id", "")
    if profile.is_demo and limiter is not None and reservation_id:
        message = _release_demo_analysis_quota(message, limiter, reservation_id)
    st.session_state.analysis_error = message
    st.session_state.analysis_phase = "idle"
    st.session_state.analysis_reservation_id = ""


@st.fragment
def render_analysis_action(
    settings: ModelSettings | None,
    profile: RuntimeProfile,
    limiter: DemoLimiter | None,
) -> None:
    draft = st.session_state.article_draft
    if draft is None:
        return

    analysis: Analysis | None = st.session_state.analysis
    has_current_analysis = analysis is not None
    analysis_busy = _analysis_is_busy()
    button_label = "重新分析（将再次调用模型）" if has_current_analysis else "开始深度分析"
    reanalysis_confirmed = False
    if has_current_analysis and not profile.is_demo:
        confirmation_key = f"confirm_reanalysis_{analysis.created_at.timestamp()}"
        reanalysis_confirmed = st.checkbox(
            "我确认要再次调用模型；这可能产生新的 API 费用。",
            key=confirmation_key,
            disabled=analysis_busy,
        )
    demo_status = limiter.status("analysis") if limiter is not None else None
    if profile.is_demo:
        st.checkbox(
            "我已知晓：点击模型功能后，当前正文和相关图片会发送到 LayerRead 服务器及模型供应商；"
            "我不会提交机密、个人隐私或无权处理的内容。",
            key="demo_privacy_accepted",
            disabled=analysis_busy,
        )
    disabled = (
        settings is None
        or (has_current_analysis and (profile.is_demo or not reanalysis_confirmed))
        or (profile.is_demo and (demo_status is None or not demo_status.allowed))
        or (profile.is_demo and not st.session_state.demo_privacy_accepted)
    )

    if settings is None:
        st.warning("完成侧边栏中的模型配置后，才能开始深度分析。")
    elif profile.is_demo and has_current_analysis:
        st.caption("在线 Demo 不开放重复分析；可以清除当前会话后导入另一篇文章。")
    elif profile.is_demo and demo_status is not None and not demo_status.allowed:
        st.warning(demo_status.message)
    elif profile.is_demo and not st.session_state.demo_privacy_accepted:
        st.caption("请先确认在线数据处理说明，再开始分析。")
    elif draft.images and not settings.supports_vision:
        st.warning(
            f"已导入 {len(draft.images)} 个图片或分片，但当前所选模型不会接收图片。"
            "如需分析图片，请先确认模型支持图片输入并在侧边栏明确授权；"
            "否则可以继续进行纯文本分析。"
        )
    elif draft.images:
        st.info(
            f"本次分析将把 {len(draft.images)} 个高清图片或长图分片发送给当前模型供应商；"
            "可能产生额外费用，并适用该供应商的数据与隐私政策。"
        )
    elif has_current_analysis and not reanalysis_confirmed:
        st.caption("如需重新分析，请先勾选上方确认项。")

    if analysis_busy:
        st.warning(
            "深度分析正在进行，通常需要 1–3 分钟。请保持当前 LayerRead 页面打开，"
            "不要切换应用内功能或标签，也不要刷新或关闭页面；否则本次分析会中断。"
        )
    else:
        st.info(
            "深度分析通常需要 1–3 分钟。开始后请保持当前 LayerRead 页面打开，"
            "不要切换应用内功能或标签，也不要刷新或关闭页面。"
        )

    start_requested = st.session_state.get("analysis_phase") == "queued"
    start_clicked = False
    if not analysis_busy:
        start_clicked = st.button(
            button_label,
            type="primary",
            width="stretch",
            disabled=disabled,
        )
    if start_clicked:
        st.session_state.analysis_error = None
        st.session_state.analysis_phase = "queued"
        st.session_state.pending_main_section = "导入与判断"
        st.session_state.pending_judgment_tab = "处理结果"
        st.rerun(scope="app")

    if start_requested:
        if settings is None:
            st.session_state.analysis_phase = "idle"
            st.session_state.analysis_error = (
                "模型配置在分析开始前失效，请重新检查配置后再试。"
            )
            st.rerun(scope="app")

        assert settings is not None
        st.session_state.analysis_error = None
        st.session_state.analysis_phase = "running"
        reservation_id = ""
        if profile.is_demo:
            if limiter is None:
                st.session_state.analysis_error = "在线体验限制服务暂不可用，请稍后重试。"
                st.session_state.analysis_phase = "idle"
                st.rerun(scope="app")
            decision = limiter.claim("analysis")
            if not decision.allowed:
                st.session_state.analysis_error = decision.message
                st.session_state.analysis_phase = "idle"
                st.rerun(scope="app")
            reservation_id = decision.reservation_id
            st.session_state.analysis_reservation_id = reservation_id
        with st.status(
            "深度分析进行中（预计 1–3 分钟，请保持页面不动）…",
            expanded=True,
        ) as status:
            st.write("正在生成结构化阅读决策和深度解析。")
            st.write("系统将校验 Schema 与原文段落引用；必要时自动修复一次。")
            try:
                st.session_state.analysis = analyze_article(draft, settings)
            except AnalysisError as exc:
                st.session_state.analysis_error = (
                    _online_demo_analysis_error(str(exc))
                    if profile.is_demo
                    else str(exc)
                )
                if profile.is_demo and limiter is not None:
                    st.session_state.analysis_error = _release_demo_analysis_quota(
                        st.session_state.analysis_error,
                        limiter,
                        reservation_id,
                    )
                st.session_state.analysis_phase = "idle"
                st.session_state.analysis_reservation_id = ""
                status.update(label="分析失败，正文已保留", state="error")
            except Exception:
                st.session_state.analysis_error = "分析过程中发生未预期错误，请重新分析。"
                if profile.is_demo and limiter is not None:
                    st.session_state.analysis_error = _release_demo_analysis_quota(
                        st.session_state.analysis_error,
                        limiter,
                        reservation_id,
                    )
                st.session_state.analysis_phase = "idle"
                st.session_state.analysis_reservation_id = ""
                status.update(label="分析失败，正文已保留", state="error")
            else:
                st.session_state.learning_session = None
                st.session_state.learning_error = None
                st.session_state.chat_session = None
                st.session_state.chat_error = None
                st.session_state.chat_phase = "idle"
                st.session_state.chat_pending_question = ""
                st.session_state.visualizations = None
                status.update(label="分析完成", state="complete", expanded=False)
                st.session_state.storage_notice = "分析完成，已进入阅读决策。"
                st.session_state.pending_main_section = "导入与判断"
                st.session_state.pending_judgment_tab = "阅读决策"
                st.session_state.analysis_phase = "idle"
                st.session_state.analysis_reservation_id = ""
        st.rerun(scope="app")

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    analysis = st.session_state.analysis
    if analysis is not None:
        _render_analysis_status(analysis)


def _navigate_to(
    section: str,
    detail_key: str = "",
    detail_value: str = "",
) -> None:
    st.session_state.main_section = section
    if detail_key and detail_value:
        st.session_state[detail_key] = detail_value


def _render_missing_prerequisite(*, needs_analysis: bool) -> None:
    draft = st.session_state.article_draft
    if draft is None:
        st.info("请先导入并处理一篇文章，再使用这个板块。")
        st.button(
            "前往导入文章",
            icon=":material/article:",
            type="primary",
            on_click=_navigate_to,
            args=("导入与判断", "judgment_tab", "导入文章"),
        )
        return
    if needs_analysis and st.session_state.analysis is None:
        st.info("文章已经处理完成。请先生成阅读决策和深度分析。")
        st.button(
            "前往处理结果",
            icon=":material/analytics:",
            type="primary",
            on_click=_navigate_to,
            args=("导入与判断", "judgment_tab", "处理结果"),
        )


def _current_visualizations(draft, analysis: Analysis | None):
    if analysis is None:
        return None
    visualizations = st.session_state.visualizations
    if visualizations is None or visualizations.article_id != analysis.article_id:
        visualizations = build_visualizations(draft, analysis)
        st.session_state.visualizations = visualizations
    return visualizations


def render_import_and_judgment_section(
    settings: ModelSettings | None,
    profile: RuntimeProfile,
    limiter: DemoLimiter | None,
) -> None:
    st.header("导入与判断")
    if _analysis_is_busy():
        st.subheader("处理结果")
        render_result()
        st.divider()
        st.subheader("生成阅读决策和分析")
        render_analysis_action(settings, profile, limiter)
        return

    tabs = st.tabs(
        list(_JUDGMENT_TABS),
        key="judgment_tab",
        on_change="rerun",
    )

    if tabs[0].open:
        with tabs[0]:
            manual_submitted, url_confirmed, connector_confirmed = render_input_form(
                profile
            )
            handle_submission(
                manual_submitted,
                url_confirmed,
                connector_confirmed,
                profile,
                limiter,
            )

    if tabs[1].open:
        with tabs[1]:
            render_result()
            if st.session_state.article_draft is not None:
                st.divider()
                st.subheader("生成阅读决策和分析")
                st.caption("只有点击下方按钮时才会调用当前所选模型。")
                render_analysis_action(settings, profile, limiter)

    if tabs[2].open:
        with tabs[2]:
            draft = st.session_state.article_draft
            analysis = st.session_state.analysis
            if draft is None or analysis is None:
                _render_missing_prerequisite(needs_analysis=True)
            else:
                _render_analysis_status(analysis)
                render_decision_card(analysis, draft)

    if tabs[3].open:
        with tabs[3]:
            draft = st.session_state.article_draft
            analysis = st.session_state.analysis
            if draft is None or analysis is None:
                _render_missing_prerequisite(needs_analysis=True)
            else:
                render_analysis_view("推广与水文", analysis, draft, settings, limiter)


def render_article_analysis_section(
    settings: ModelSettings | None,
    limiter: DemoLimiter | None,
) -> None:
    st.header("文章分析")
    draft = st.session_state.article_draft
    analysis = st.session_state.analysis
    if draft is None or analysis is None:
        _render_missing_prerequisite(needs_analysis=True)
        return

    tabs = st.tabs(
        list(_ARTICLE_ANALYSIS_TABS),
        key="article_analysis_tab",
        on_change="rerun",
    )
    for tab, view in zip(tabs, _ARTICLE_ANALYSIS_TABS, strict=True):
        if tab.open:
            with tab:
                render_analysis_view(view, analysis, draft, settings, limiter)


def render_article_learning_section(
    settings: ModelSettings | None,
    limiter: DemoLimiter | None,
) -> None:
    st.header("文章学习")
    draft = st.session_state.article_draft
    analysis = st.session_state.analysis
    if draft is None or analysis is None:
        _render_missing_prerequisite(needs_analysis=True)
        return

    if learning_is_busy():
        st.session_state.article_learning_tab = "主动学习"
        render_analysis_view("主动学习", analysis, draft, settings, limiter)
        return
    if chat_is_busy():
        st.session_state.article_learning_tab = "文章 Chat"
        render_analysis_view("文章 Chat", analysis, draft, settings, limiter)
        return

    tabs = st.tabs(
        list(_ARTICLE_LEARNING_TABS),
        key="article_learning_tab",
        on_change="rerun",
    )
    for tab, view in zip(tabs, _ARTICLE_LEARNING_TABS, strict=True):
        if tab.open:
            with tab:
                render_analysis_view(view, analysis, draft, settings, limiter)


def render_export_section(profile: RuntimeProfile) -> None:
    st.header("导出")
    draft = st.session_state.article_draft
    if draft is None:
        _render_missing_prerequisite(needs_analysis=False)
        return
    analysis = st.session_state.analysis
    visualizations = _current_visualizations(draft, analysis)
    render_export_tab(
        draft,
        analysis,
        st.session_state.learning_session,
        st.session_state.chat_session,
        visualizations,
        notion_enabled=profile.notion_enabled,
    )


def render_evidence_section(
    settings: ModelSettings | None,
    limiter: DemoLimiter | None,
) -> None:
    st.header("证据与核验")
    draft = st.session_state.article_draft
    analysis = st.session_state.analysis
    if draft is None or analysis is None:
        _render_missing_prerequisite(needs_analysis=True)
        return

    tabs = st.tabs(
        list(_EVIDENCE_TABS),
        key="evidence_tab",
        on_change="rerun",
    )
    for tab, view in zip(tabs, _EVIDENCE_TABS, strict=True):
        if tab.open:
            with tab:
                render_analysis_view(view, analysis, draft, settings, limiter)


def render_main_navigation(
    settings: ModelSettings | None,
    profile: RuntimeProfile,
    limiter: DemoLimiter | None,
) -> None:
    analysis_busy = _analysis_is_busy()
    learning_busy = learning_is_busy()
    chat_busy = chat_is_busy()
    if analysis_busy:
        st.session_state.main_section = "导入与判断"
    elif learning_busy:
        st.session_state.main_section = "文章学习"
        st.session_state.article_learning_tab = "主动学习"
    elif chat_busy:
        st.session_state.main_section = "文章学习"
        st.session_state.article_learning_tab = "文章 Chat"
    selected = st.segmented_control(
        "主要功能",
        options=list(_MAIN_SECTIONS),
        key="main_section",
        required=True,
        label_visibility="collapsed",
        width="stretch",
        disabled=analysis_busy or learning_busy or chat_busy,
    )
    if analysis_busy:
        selected = "导入与判断"
    elif learning_busy:
        selected = "文章学习"
    elif chat_busy:
        selected = "文章学习"
    draft = st.session_state.article_draft
    analysis = st.session_state.analysis
    connector_result = st.session_state.connector_import_result
    connector_token = _connector_import_token_from_query()
    if analysis is not None:
        st.caption(f"当前文章：{draft.source_title or '未命名文章'} · 已完成分析")
    elif draft is not None:
        st.caption(f"当前文章：{draft.source_title or '未命名文章'} · 等待分析")
    elif connector_result is not None:
        st.caption(
            f"已接收文章：{connector_result.title or '未命名文章'} · "
            "请在下方检查并确认"
        )
    elif connector_token:
        st.caption("正在从 Connector 接收文章与高清图片，请稍候…")
    else:
        st.caption("当前尚未确认文章")
    st.divider()

    if selected == "导入与判断":
        render_import_and_judgment_section(settings, profile, limiter)
    elif selected == "文章分析":
        render_article_analysis_section(settings, limiter)
    elif selected == "文章学习":
        render_article_learning_section(settings, limiter)
    elif selected == "导出":
        render_export_section(profile)
    else:
        render_evidence_section(settings, limiter)


def main() -> None:
    profile = read_runtime_profile()
    initialize_state(profile)
    limiter = build_demo_limiter() if profile.is_demo else None
    _recover_interrupted_analysis(profile, limiter)
    recover_interrupted_learning()
    recover_interrupted_chat()
    if profile.long_term_storage_enabled:
        initialize_local_storage()
    if profile.byok_enabled:
        config_result = render_model_configuration(
            (read_portfolio_model_config(),),
            portfolio_env=True,
        )
    else:
        config_results = filter_model_configs(read_model_configs(), profile)
        config_result = render_model_configuration(config_results)
    if profile.long_term_storage_enabled:
        render_local_library()
    else:
        assert limiter is not None
        render_demo_sidebar(limiter)
        render_session_workspace()
    render_header(profile)
    render_storage_messages()
    render_main_navigation(
        config_result.settings if config_result.is_ready else None,
        profile,
        limiter,
    )
    if profile.long_term_storage_enabled:
        autosave_current_snapshot()


if __name__ == "__main__":
    main()
