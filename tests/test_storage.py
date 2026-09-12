from __future__ import annotations

from dataclasses import replace
import sqlite3

import pytest

from layerread.config import ModelSettings
from layerread.article import ArticleImage
from layerread.storage import (
    DEFAULT_DATABASE_PATH,
    LEGACY_DATABASE_PATH,
    StorageError,
    article_record_id,
    database_path,
    initialize_database,
    list_articles,
    list_notion_exports,
    load_latest_snapshot,
    load_snapshot,
    record_notion_export,
    save_snapshot,
)
from layerread.visualization import build_visualizations
from tests.test_analysis import make_article
from tests.test_learning import make_analysis
from tests.v05_helpers import make_chat_session, make_learning_session


@pytest.fixture
def settings() -> ModelSettings:
    return ModelSettings(
        api_key="test-secret",
        base_url="https://models.example.com/v1",
        model="test-model",
    )


def test_database_initialization_is_idempotent_and_records_schema_version(tmp_path) -> None:
    path = tmp_path / "layerread.sqlite3"

    initialize_database(path)
    initialize_database(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert version == 2
    assert {
        "articles",
        "analyses",
        "learning_sessions",
        "chat_sessions",
        "visualizations",
        "notion_exports",
    } <= tables


def test_default_database_copies_legacy_data_and_preserves_original(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LAYERREAD_DB_PATH", raising=False)
    monkeypatch.delenv("DEEPREAD_DB_PATH", raising=False)
    article = make_article()
    legacy_path = tmp_path / LEGACY_DATABASE_PATH
    save_snapshot(article, None, None, None, None, legacy_path)

    resolved = database_path()

    assert resolved == DEFAULT_DATABASE_PATH
    assert legacy_path.is_file()
    migrated_path = tmp_path / DEFAULT_DATABASE_PATH
    assert migrated_path.is_file()
    assert load_snapshot(article_record_id(article), migrated_path).article == article


def test_legacy_database_environment_variable_remains_supported(
    tmp_path,
    monkeypatch,
) -> None:
    configured = tmp_path / "custom-legacy.sqlite3"
    monkeypatch.delenv("LAYERREAD_DB_PATH", raising=False)
    monkeypatch.setenv("DEEPREAD_DB_PATH", str(configured))

    assert database_path() == configured


def test_complete_snapshot_survives_database_reopen(tmp_path, settings) -> None:
    path = tmp_path / "layerread.sqlite3"
    article = make_article()
    analysis = make_analysis(settings)
    learning = make_learning_session(analysis, settings)
    chat = make_chat_session(analysis, settings)
    visualizations = build_visualizations(article, analysis)

    summary = save_snapshot(
        article,
        analysis,
        learning,
        chat,
        visualizations,
        path,
    )
    restored = load_snapshot(summary.record_id, path)

    assert restored.record_id == article_record_id(article)
    assert restored.article == article
    assert restored.analysis == analysis
    assert restored.learning_session == learning
    assert restored.chat_session == chat
    assert restored.visualizations == visualizations
    assert restored.chat_session.saved_insights[0].user_note == "应用到团队评估清单。"


def test_imported_images_survive_local_snapshot_reopen(tmp_path) -> None:
    path = tmp_path / "layerread.sqlite3"
    image = ArticleImage(
        image_id="IMG01",
        data_url="data:image/png;base64,ZmFrZQ==",
        mime_type="image/png",
        width=2400,
        height=1350,
        caption="产品路线 PPT",
        tile_index=1,
        tile_count=1,
        original_width=2400,
        original_height=1350,
    )
    article = replace(
        make_article(),
        images=(image,),
        image_count_detected=1,
        media_dependency=True,
        media_completeness="images_included",
    )

    summary = save_snapshot(article, None, None, None, None, path)
    restored = load_snapshot(summary.record_id, path)

    assert restored.article.images == (image,)
    assert restored.article.media_completeness == "images_included"


def test_latest_article_and_local_list_are_ordered_by_real_changes(tmp_path, settings) -> None:
    path = tmp_path / "layerread.sqlite3"
    first_article = make_article()
    first_analysis = make_analysis(settings)
    first = save_snapshot(first_article, first_analysis, None, None, None, path)
    unchanged = save_snapshot(first_article, first_analysis, None, None, None, path)
    assert unchanged.updated_at == first.updated_at

    second_article = replace(make_article(), reading_goal="第二个阅读目标")
    second_analysis = make_analysis(settings)
    second = save_snapshot(second_article, second_analysis, None, None, None, path)

    records = list_articles(path)
    latest = load_latest_snapshot(path)
    assert [item.record_id for item in records] == [second.record_id, first.record_id]
    assert latest is not None
    assert latest.record_id == second.record_id


def test_incompatible_learning_data_does_not_overwrite_existing_snapshot(
    tmp_path,
    settings,
) -> None:
    path = tmp_path / "layerread.sqlite3"
    article = make_article()
    analysis = make_analysis(settings)
    original = save_snapshot(article, analysis, None, None, None, path)
    learning = make_learning_session(analysis, settings).model_copy(
        update={"article_id": "another-analysis"}
    )

    with pytest.raises(StorageError, match="不一致"):
        save_snapshot(article, analysis, learning, None, None, path)

    restored = load_snapshot(original.record_id, path)
    assert restored.analysis == analysis
    assert restored.learning_session is None


def test_url_metadata_and_notion_export_record_survive_reopen(tmp_path, settings) -> None:
    path = tmp_path / "layerread.sqlite3"
    article = replace(
        make_article(),
        source_title="URL 导入文章",
        source_url="https://example.com/article",
        account="示例来源",
        author="测试作者",
        published_at="2026-07-19",
        extraction_method="generic_html",
        extraction_status="confirmed",
    )
    summary = save_snapshot(article, None, None, None, None, path)

    record_notion_export(
        summary.record_id,
        "12345678-1234-1234-1234-1234567890ab",
        ["metadata", "original_article"],
        status="success",
        notion_page_id="notion-page",
        notion_page_url="https://www.notion.so/notion-page",
        path=path,
    )
    restored = load_snapshot(summary.record_id, path)
    records = list_notion_exports(summary.record_id, path)

    assert restored.article.source_title == "URL 导入文章"
    assert restored.article.source_url == "https://example.com/article"
    assert restored.title == "URL 导入文章"
    assert records[0].status == "success"
    assert records[0].exported_sections == ("metadata", "original_article")
    assert "token" not in records[0].error_message.lower()
