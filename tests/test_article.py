from __future__ import annotations

import pytest

from layerread.article import (
    ArticleDraft,
    ArticleValidationError,
    number_paragraphs,
    process_article,
    upgrade_article_draft,
)


def make_long_article() -> str:
    return "\n\n".join(
        [
            "人工智能正在改变软件开发，但效率提升不等于质量自然提升。",
            "第一部分：问题背景",
            "许多团队开始使用生成式工具完成重复编码任务。",
            "第二部分：核心观点",
            "真正的收益来自工作流程改变，而不只是代码生成速度。",
            "团队仍然需要代码评审、测试和清晰的责任边界。",
            "第三部分：实际案例",
            "某团队通过自动生成测试草稿缩短了准备时间。",
            "但工程师仍需检查边界条件并确认测试意图。",
            "结论是工具应当增强判断，而不是替代判断。",
        ]
    )


@pytest.mark.parametrize("raw_text", ["", "   \n\n  ", "内容太短。"])
def test_empty_or_short_article_is_rejected(raw_text: str) -> None:
    with pytest.raises(ArticleValidationError):
        process_article(raw_text)


def test_article_is_numbered_in_original_order() -> None:
    draft = process_article(make_long_article())

    assert draft.paragraph_count == 10
    assert draft.numbered_paragraphs[0].startswith("[P01] 人工智能")
    assert draft.numbered_paragraphs[-1].startswith("[P10] 结论")
    assert [item[:5] for item in draft.numbered_paragraphs] == [
        f"[P{index:02d}]" for index in range(1, 11)
    ]


def test_duplicate_and_operation_paragraphs_are_removed() -> None:
    paragraphs = make_long_article().split("\n\n")
    raw_text = "\n\n".join(
        paragraphs
        + [
            paragraphs[2],
            "欢迎关注公众号并点赞分享",
            "长按二维码关注公众号",
        ]
    )

    draft = process_article(raw_text)

    assert draft.paragraph_count == 10
    assert paragraphs[2] in draft.removed_paragraphs
    assert "欢迎关注公众号并点赞分享" in draft.removed_paragraphs
    assert "长按二维码关注公众号" in draft.removed_paragraphs


def test_headings_and_list_items_are_preserved() -> None:
    raw_text = """文章标题

这是用于验证正文清洗逻辑的第一段，它包含足够多的信息，使测试文章超过最低长度限制。

核心方法包括：
- 保留文章标题
- 保留列表项目
- 只删除明确的运营内容

最后一段说明清洗过程不能为了格式统一而大幅改写作者原本表达的意思。"""

    draft = process_article(raw_text)

    assert "文章标题" in draft.clean_text
    assert "- 保留文章标题" in draft.clean_text
    assert "- 保留列表项目" in draft.clean_text
    assert "核心方法包括：" in draft.clean_text


def test_reprocessing_same_text_produces_stable_numbering() -> None:
    first = process_article(make_long_article())
    second = process_article(make_long_article())

    assert first.numbered_paragraphs == second.numbered_paragraphs


def test_media_dependency_is_recorded_without_importing_media() -> None:
    draft = process_article(make_long_article(), media_dependency=True)

    assert draft.media_dependency is True
    assert draft.media_completeness == "text_only"


def test_numbering_supports_more_than_99_paragraphs() -> None:
    numbered = number_paragraphs([f"段落 {index}" for index in range(1, 101)])

    assert numbered[98].startswith("[P99]")
    assert numbered[99].startswith("[P100]")


def test_pre_v06_in_memory_draft_is_upgraded_with_safe_source_defaults() -> None:
    current = process_article(make_long_article())

    class LegacyDraft:
        pass

    legacy = LegacyDraft()
    legacy_fields = (
        "raw_text",
        "clean_text",
        "numbered_paragraphs",
        "removed_paragraphs",
        "reading_goal",
        "familiarity",
        "focus_area",
        "media_dependency",
        "media_completeness",
        "raw_char_count",
        "clean_char_count",
        "created_at",
    )
    for field_name in legacy_fields:
        setattr(legacy, field_name, getattr(current, field_name))

    upgraded = upgrade_article_draft(legacy)

    assert isinstance(upgraded, ArticleDraft)
    assert upgraded.numbered_text == current.numbered_text
    assert upgraded.source_url == ""
    assert upgraded.extraction_method == "manual_paste"
