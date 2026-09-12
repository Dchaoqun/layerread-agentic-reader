from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

from layerread.article import ArticleImage, process_article
from layerread.connector import ConnectorArticleImportResult
from layerread.storage import article_record_id


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_queued_analysis_sets_navigation_before_rendering_the_widget(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()

    app.session_state["main_section"] = "导出"
    app.session_state["analysis_phase"] = "queued"
    app.run()

    assert not app.exception
    assert app.session_state["main_section"] == "导入与判断"
    assert app.segmented_control[0].disabled is True


def test_interrupted_deep_analysis_unlocks_and_keeps_the_draft(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-standard-key")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen3.7-plus")

    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()

    article = process_article(
        raw_text="用于验证深度分析导航锁定与失败恢复。" * 30,
        reading_goal="验证长任务交互",
        familiarity="入门",
        focus_area="技术",
        media_dependency=False,
    )
    app.session_state["article_draft"] = article
    app.session_state["main_section"] = "导入与判断"
    app.session_state["judgment_tab"] = "处理结果"
    app.session_state["analysis_phase"] = "running"
    app.run()

    assert not app.exception
    assert app.session_state["analysis_phase"] == "idle"
    assert app.session_state["analysis_reservation_id"] == ""
    assert app.session_state["article_draft"] == article
    assert any(
        "上一次深度分析因页面切换、刷新或连接中断而停止" in error.value
        for error in app.error
    )
    assert any(
        "深度分析通常需要 1–3 分钟" in info.value
        for info in app.info
    )
    sys.modules.pop("layerread.demo_ui", None)


def test_queued_learning_request_locks_main_navigation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()

    app.session_state["main_section"] = "文章分析"
    app.session_state["article_learning_tab"] = "文章 Chat"
    app.session_state["learning_phase"] = "questions_queued"
    app.run()

    assert not app.exception
    assert app.session_state["main_section"] == "文章学习"
    assert app.session_state["article_learning_tab"] == "主动学习"
    assert app.segmented_control[0].disabled is True


def test_interrupted_learning_request_unlocks_and_reports_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()

    app.session_state["learning_phase"] = "questions_running"
    app.run()

    assert not app.exception
    assert app.session_state["learning_phase"] == "idle"
    assert "上一次主动学习题目生成因页面切换" in app.session_state[
        "learning_error"
    ]


def test_queued_chat_request_locks_main_and_learning_navigation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()

    app.session_state["main_section"] = "文章分析"
    app.session_state["article_learning_tab"] = "主动学习"
    app.session_state["chat_phase"] = "reply_queued"
    app.session_state["chat_pending_question"] = "这个结论如何应用？"
    app.run()

    assert not app.exception
    assert app.session_state["main_section"] == "文章学习"
    assert app.session_state["article_learning_tab"] == "文章 Chat"
    assert app.segmented_control[0].disabled is True


def test_interrupted_chat_request_keeps_question_and_unlocks_navigation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.run()

    app.session_state["chat_phase"] = "reply_running"
    app.session_state["chat_pending_question"] = "请解释文章的核心论点。"
    app.run()

    assert not app.exception
    assert app.session_state["chat_phase"] == "idle"
    assert app.session_state["chat_pending_question"] == "请解释文章的核心论点。"
    assert "上一次 Chat 回答因页面切换" in app.session_state["chat_error"]


def test_demo_mode_keeps_connector_images_but_disables_article_persistence(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "demo")
    article_db = tmp_path / "layerread.sqlite3"
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(article_db))
    monkeypatch.setenv(
        "LAYERREAD_DEMO_QUOTA_DB_PATH",
        str(tmp_path / "demo-usage.sqlite3"),
    )
    monkeypatch.setenv("LAYERREAD_DEMO_QUOTA_DATABASE_URL", "")
    monkeypatch.setenv("LAYERREAD_DEMO_REQUIRE_PERSISTENT_QUOTA", "false")
    sys.modules.pop("layerread.connector_ui", None)
    sys.modules.pop("layerread.demo_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)

    app.run()

    assert not app.exception
    assert app.session_state["import_tab"] == "Chrome 一键导入"
    tab_labels = [tab.label for tab in app.tabs]
    assert "URL 导入" in tab_labels
    assert "粘贴正文" in tab_labels
    assert "Chrome 一键导入" in tab_labels
    assert any(
        checkbox.label == "这篇文章的关键内容依赖图片、图表或视频"
        for checkbox in app.checkbox
    )
    assert any("在线体验模式" in info.value for info in app.info)
    assert app.session_state["local_storage_initialized"] is False
    assert not article_db.exists()
    visible_text = "\n".join(
        str(getattr(element, "value", ""))
        for collection in (app.markdown, app.caption, app.info, app.warning)
        for element in collection
    )
    assert "本地文章" not in visible_text
    assert "不会把文章、图片、分析或对话保存到文章库" in visible_text
    assert "本地源码版 Connector 只连接 localhost" in visible_text
    assert "在线版 0.4.0" in visible_text
    assert "当前在线包版本：0.4.4" in visible_text
    assert "下载并解压压缩包" in visible_text
    assert "导入前的必要步骤" in visible_text
    assert "从顶部滚动到底部" in visible_text
    assert "未滚动到的图片可能无法导入" in visible_text
    assert "体验支持码" in visible_text


def test_portfolio_mode_uses_byok_connector_images_and_hides_notion(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "portfolio")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    monkeypatch.setenv("LLM_API_KEY", "portfolio-test-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://models.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "portfolio-model")
    monkeypatch.setenv("LLM_SUPPORTS_VISION", "true")
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)

    app.run()

    assert not app.exception
    labels = [item.label for item in app.text_input]
    assert "API Key" not in labels
    assert "API Base URL" not in labels
    assert "模型名称" not in labels
    checkbox_labels = [item.label for item in app.checkbox]
    assert "当前模型支持图片输入，并允许在分析时发送文章图片" not in checkbox_labels
    assert "这篇文章的关键内容依赖图片、图表或视频" in checkbox_labels
    assert app.session_state["import_tab"] == "Chrome 一键导入"
    assert "Chrome 一键导入" in [tab.label for tab in app.tabs]
    assert any("Portfolio Edition" in info.value for info in app.info)
    assert any("AGPL-3.0-only" in caption.value for caption in app.caption)
    assert any("项目根目录的本机 .env" in caption.value for caption in app.caption)
    assert any("额外费用" in warning.value for warning in app.warning)

    app.session_state["article_draft"] = process_article(
        raw_text="Portfolio Edition 本地导出验证。" * 30,
        reading_goal="验证公开版功能边界",
        familiarity="入门",
        focus_area="技术",
        media_dependency=False,
    )
    app.session_state["main_section"] = "导出"
    app.run()

    assert not app.exception
    visible_text = "\n".join(
        str(getattr(element, "value", ""))
        for collection in (app.markdown, app.caption, app.info, app.warning)
        for element in collection
    )
    assert "Markdown 导出" in visible_text
    assert "Notion" not in visible_text


def test_connector_import_query_routes_to_import_only_once(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.setenv("LAYERREAD_DB_PATH", str(tmp_path / "layerread.sqlite3"))
    sys.modules.pop("layerread.connector_ui", None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.query_params["layerread_import"] = "11111111-1111-4111-8111-111111111111"

    app.run()

    assert not app.exception
    assert app.session_state["main_section"] == "导入与判断"
    assert app.session_state["judgment_tab"] == "导入文章"
    assert any(
        "正在从 Connector 接收文章与高清图片" in caption.value
        for caption in app.caption
    )
    assert any("通常会在 5–20 秒内完成" in caption.value for caption in app.caption)
    assert any(
        "本次接收的图片可能不完整" in warning.value for warning in app.warning
    )
    assert any("从顶部滚动到底部" in warning.value for warning in app.warning)

    app.session_state["pending_main_section"] = "导入与判断"
    app.session_state["pending_judgment_tab"] = "处理结果"
    app.run()

    assert not app.exception
    assert app.session_state["judgment_tab"] == "处理结果"

    for tab_label in ("阅读决策", "推广与水文"):
        app.session_state["judgment_tab"] = tab_label
        app.run()

        assert not app.exception
        assert app.session_state["judgment_tab"] == tab_label

    result = ConnectorArticleImportResult(
        filename="article.json",
        title="保留的标题",
        account="保留的来源",
        author="保留的作者",
        published_at="2026-08-04",
        source_url="https://mp.weixin.qq.com/s/example",
        body="保留的正文" * 30,
    )
    app.session_state["connector_import_result"] = result
    app.session_state["connector_title_input"] = result.title
    app.session_state["connector_account_input"] = result.account
    app.session_state["connector_author_input"] = result.author
    app.session_state["connector_published_at_input"] = result.published_at
    app.session_state["connector_source_input"] = result.source_url
    app.session_state["connector_body_input"] = result.body
    app.session_state["judgment_tab"] = "导入文章"
    app.run()
    assert not app.exception
    assert any(
        "已接收文章：保留的标题" in caption.value
        and "请在下方检查并确认" in caption.value
        for caption in app.caption
    )
    assert any(
        "请对照微信原文核对关键图片和图片数量" in info.value
        and "若有缺失，先不要确认文章" in info.value
        for info in app.info
    )

    app.session_state["judgment_tab"] = "处理结果"
    app.run()
    app.session_state["judgment_tab"] = "导入文章"
    app.run()

    assert not app.exception
    assert app.session_state["connector_title_input"] == result.title
    assert app.session_state["connector_published_at_input"] == result.published_at
    assert app.session_state["connector_body_input"] == result.body

    article = process_article(
        raw_text="已确认的正文内容。" * 30,
        reading_goal="核对页面状态",
        familiarity="入门",
        focus_area="技术",
        media_dependency=False,
        source_title="已确认的标题",
        source_url="https://mp.weixin.qq.com/s/confirmed",
        account="已确认的来源",
        author="已确认的作者",
        published_at="2026-08-16",
        extraction_method="layerread_chrome_connector",
        extraction_status="confirmed",
    )
    app.session_state["article_draft"] = article
    app.session_state["connector_import_result"] = None
    for key in (
        "connector_title_input",
        "connector_account_input",
        "connector_author_input",
        "connector_published_at_input",
        "connector_source_input",
        "connector_body_input",
    ):
        app.session_state[key] = ""
    app.session_state["main_section"] = "导入与判断"
    app.session_state["judgment_tab"] = "导入文章"
    app.session_state["import_tab"] = "Chrome 一键导入"

    app.run()

    assert not app.exception
    assert app.session_state["connector_title_input"] == article.source_title
    assert app.session_state["connector_published_at_input"] == article.published_at
    assert app.session_state["connector_source_input"] == article.source_url
    assert app.session_state["connector_body_input"] == article.raw_text
    assert any(
        button.label == "确认内容并处理文章" for button in app.button
    )

    image = ArticleImage(
        image_id="IMG01",
        data_url=(
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "/x8AAusB9Y9Z4AAAAABJRU5ErkJggg=="
        ),
        mime_type="image/png",
        width=1,
        height=1,
    )
    article = process_article(
        raw_text="包含图片的已确认正文。" * 30,
        reading_goal="核对图片保留",
        familiarity="入门",
        focus_area="技术",
        media_dependency=True,
        images=(image,),
        image_count_detected=1,
        source_title="图片文章",
        source_url="https://mp.weixin.qq.com/s/with-image",
        extraction_method="layerread_chrome_connector",
        extraction_status="confirmed",
    )
    app.session_state["article_draft"] = article
    app.session_state["connector_import_result"] = None
    app.session_state["connector_body_input"] = ""
    app.session_state["main_section"] = "导入与判断"
    app.session_state["judgment_tab"] = "导入文章"
    app.session_state["import_tab"] = "Chrome 一键导入"
    app.run()

    confirm = next(
        button for button in app.button if button.label == "确认内容并处理文章"
    )
    confirm.click().run()

    assert not app.exception
    assert app.session_state["article_draft"].images == (image,)
    assert app.session_state["article_draft"].image_count_detected == 1

    next(
        button for button in app.button if button.label == "开始新文章"
    ).click().run()
    assert not app.exception
    assert app.session_state["article_draft"] is None

    next(
        selectbox for selectbox in app.selectbox if selectbox.label == "选择本地记录"
    ).select(article_record_id(article)).run()
    next(
        button for button in app.button if button.label == "打开所选文章"
    ).click().run()
    assert not app.exception
    assert app.session_state["article_draft"].images == (image,)
    assert app.session_state["connector_title_input"] == article.source_title
    assert app.session_state["connector_source_input"] == article.source_url
    assert app.session_state["connector_body_input"] == article.raw_text
