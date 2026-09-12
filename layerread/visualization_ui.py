"""Streamlit rendering for deterministic Mermaid visualizations."""

from __future__ import annotations

import streamlit as st

from layerread.visualization import (
    MermaidDiagram,
    VisualizationBundle,
    repair_mermaid_source,
    validate_mermaid_source,
)


def _render_diagram(diagram: MermaidDiagram) -> None:
    source = diagram.source
    valid, reason = validate_mermaid_source(source)
    repaired_now = False
    if not valid:
        source = repair_mermaid_source(source)
        repaired_now = True
        valid, reason = validate_mermaid_source(source)

    if diagram.repaired or repaired_now:
        st.info("检测到可安全修复的 Mermaid 格式问题，已尝试修复一次。")

    if valid:
        try:
            st.mermaid_chart(source, width="stretch")
        except Exception:
            valid = False
            reason = "Streamlit 无法创建 Mermaid 渲染元素。"

    if not valid:
        st.warning(f"图表无法安全渲染：{reason} 已显示源码和等价文本结构。")

    with st.expander("查看 Mermaid 源码与文本降级"):
        st.code(source, language="mermaid", wrap_lines=True)
        st.markdown("**等价文本结构**")
        st.code(diagram.fallback_text, language=None, wrap_lines=True)


def render_visualization_tab(bundle: VisualizationBundle) -> None:
    st.markdown("### 文章可视化")
    st.caption(
        "图表由已通过校验的结构化分析程序化生成，不会额外调用模型。"
        "黄色表示薄弱证据，红色表示逻辑跳跃，蓝色表示待验证节点。"
    )

    with st.container(border=True):
        st.markdown("#### 思维导图")
        _render_diagram(bundle.mind_map)

    with st.container(border=True):
        st.markdown("#### 论证结构图")
        _render_diagram(bundle.argument_map)
