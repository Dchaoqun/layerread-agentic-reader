from __future__ import annotations

import pytest

from layerread.config import ModelSettings
from layerread.export import (
    ExportError,
    SECTION_LABELS,
    available_export_sections,
    build_export_document,
    markdown_filename,
    render_markdown,
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


def _complete_assets(settings):
    article = make_article()
    analysis = make_analysis(settings)
    learning = make_learning_session(analysis, settings)
    chat = make_chat_session(analysis, settings)
    visualizations = build_visualizations(article, analysis)
    return article, analysis, learning, chat, visualizations


def test_complete_markdown_contains_every_available_asset(settings) -> None:
    article, analysis, learning, chat, visualizations = _complete_assets(settings)
    selected = available_export_sections(
        analysis,
        learning,
        chat,
        visualizations,
    )

    document = build_export_document(
        article,
        analysis,
        learning,
        chat,
        visualizations,
        selected,
    )
    markdown = render_markdown(document)

    assert [section.section_id for section in document.sections] == selected
    for section_id in selected:
        assert f"## {SECTION_LABELS[section_id]}" in markdown
    assert "[P01]" in markdown
    assert "文章认为收益来自流程重构" in markdown
    assert "真正收益来自流程重构" in markdown
    assert "应用到团队评估清单" in markdown
    assert "单一案例能否推广到其他团队" in markdown
    assert "```mermaid" in markdown


def test_selected_subset_excludes_every_unselected_module(settings) -> None:
    article, analysis, learning, chat, visualizations = _complete_assets(settings)
    document = build_export_document(
        article,
        analysis,
        learning,
        chat,
        visualizations,
        ["metadata", "chat_insights"],
    )
    markdown = render_markdown(document)

    assert "## 基本信息" in markdown
    assert "## 对话关键洞察" in markdown
    assert "真正收益来自流程重构" in markdown
    assert "## 原文" not in markdown
    assert "## 用户笔记" not in markdown
    assert "应用到团队评估清单" not in markdown
    assert "## 主动学习" not in markdown


def test_original_source_url_is_preserved_as_clickable_markdown(settings) -> None:
    article, analysis, learning, chat, visualizations = _complete_assets(settings)
    source_url = "https://example.com/article"
    document = build_export_document(
        article,
        analysis,
        learning,
        chat,
        visualizations,
        ["metadata", "original_article"],
        source_url=source_url,
    )

    markdown = render_markdown(document)
    assert f"[{source_url}]({source_url})" in markdown
    assert f"原文链接：{source_url}" in markdown


def test_export_rejects_empty_or_unavailable_selection(settings) -> None:
    article = make_article()

    with pytest.raises(ExportError, match="至少选择"):
        build_export_document(article, None, None, None, None, [])

    with pytest.raises(ExportError, match="没有可导出内容"):
        build_export_document(
            article,
            None,
            None,
            None,
            None,
            ["ai_feedback"],
        )


def test_markdown_filename_removes_windows_unsafe_characters(settings) -> None:
    article, analysis, learning, chat, visualizations = _complete_assets(settings)
    document = build_export_document(
        article,
        analysis,
        learning,
        chat,
        visualizations,
        ["metadata"],
    ).model_copy(update={"title": 'a<b>c:d/e\\f|g?h*"'})

    filename = markdown_filename(document)
    assert filename.endswith(".md")
    assert not any(character in filename for character in '<>:"/\\|?*')


def test_markdown_filename_uses_layerread_brand_fallback(settings) -> None:
    article, analysis, learning, chat, visualizations = _complete_assets(settings)
    document = build_export_document(
        article,
        analysis,
        learning,
        chat,
        visualizations,
        ["metadata"],
    ).model_copy(update={"title": "***"})

    assert markdown_filename(document) == "layerread-export.md"
