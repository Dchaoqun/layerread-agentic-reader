from __future__ import annotations

import pytest

from layerread.config import ModelSettings
from layerread.visualization import (
    build_visualizations,
    mermaid_label,
    repair_mermaid_source,
    validate_mermaid_source,
)
from tests.test_analysis import make_article
from tests.test_learning import make_analysis


@pytest.fixture
def settings() -> ModelSettings:
    return ModelSettings(
        api_key="test-secret",
        base_url="https://models.example.com/v1",
        model="test-model",
    )


def test_both_programmatic_diagrams_pass_local_mermaid_validation(settings) -> None:
    bundle = build_visualizations(make_article(), make_analysis(settings))

    mind_valid, mind_reason = validate_mermaid_source(bundle.mind_map.source)
    argument_valid, argument_reason = validate_mermaid_source(
        bundle.argument_map.source
    )

    assert mind_valid, mind_reason
    assert argument_valid, argument_reason
    assert "背景" in bundle.mind_map.source
    assert "核心问题" in bundle.mind_map.source
    assert "证据或示例" in bundle.mind_map.source
    assert "局限" in bundle.mind_map.source
    assert "应用" in bundle.mind_map.source
    assert "作者假设" in bundle.argument_map.source
    assert "中间推断" in bundle.argument_map.source
    assert "最终结论" in bundle.argument_map.source


def test_argument_map_marks_weak_logic_and_verification_nodes(settings) -> None:
    bundle = build_visualizations(make_article(), make_analysis(settings))
    source = bundle.argument_map.source

    assert "classDef weak" in source
    assert "classDef risk" in source
    assert "classDef verify" in source
    assert "待验证" in source
    assert "-.->" in source


def test_mermaid_labels_escape_syntax_breakers_and_control_characters() -> None:
    escaped = mermaid_label('危险"] --> X\n{test}|value\x00')

    assert '"' not in escaped
    assert "-->" not in escaped
    assert "\x00" not in escaped
    assert "】" in escaped
    assert "｛" in escaped
    assert "｜" in escaped


def test_long_mermaid_labels_wrap_without_truncating_content() -> None:
    original = "这是一段需要完整显示在文章结构图节点中的较长中文文本" * 5

    wrapped = mermaid_label(original, line_width=18)

    assert "<br/>" in wrapped
    assert "…" not in wrapped
    assert wrapped.replace("<br/>", "") == original


def test_safe_repair_adds_header_and_removes_control_characters() -> None:
    repaired = repair_mermaid_source(
        '    A["开始"]\x00\n    B["结束"]\n    A --> B'
    )
    valid, reason = validate_mermaid_source(repaired)

    assert repaired.startswith("flowchart TD")
    assert valid, reason


def test_text_fallback_remains_readable_when_source_is_unavailable(settings) -> None:
    bundle = build_visualizations(make_article(), make_analysis(settings))

    assert "文章主题" in bundle.mind_map.fallback_text
    assert "核心问题" in bundle.mind_map.fallback_text
    assert "论证结构" in bundle.argument_map.fallback_text
    assert "待验证节点" in bundle.argument_map.fallback_text
