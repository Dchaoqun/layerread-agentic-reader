"""Deterministic Mermaid visualizations with readable text fallbacks."""

from __future__ import annotations

from datetime import UTC, datetime
import re
import textwrap
from typing import Literal

from pydantic import Field

from layerread.analysis_schema import Analysis, StrictModel
from layerread.article import ArticleDraft


DiagramKind = Literal["mind_map", "argument_map"]
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class MermaidDiagram(StrictModel):
    kind: DiagramKind
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    fallback_lines: list[str] = Field(min_length=1)
    repaired: bool = False

    @property
    def fallback_text(self) -> str:
        return "\n".join(self.fallback_lines)


class VisualizationBundle(StrictModel):
    article_id: str = Field(min_length=1)
    mind_map: MermaidDiagram
    argument_map: MermaidDiagram
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def mermaid_label(value: str, line_width: int = 28) -> str:
    """Escape and wrap complete model text for a Mermaid flowchart label."""

    cleaned = _CONTROL_CHARACTERS.sub("", value)
    cleaned = " ".join(cleaned.replace("\n", " ").split())
    translations = str.maketrans(
        {
            '"': "”",
            "[": "【",
            "]": "】",
            "{": "｛",
            "}": "｝",
            "<": "‹",
            ">": "›",
            "|": "｜",
        }
    )
    cleaned = cleaned.translate(translations)
    cleaned = cleaned.replace("-->", "→").replace("---", "—").replace(":::" , "∶")
    if not cleaned:
        cleaned = "未提供"
    wrapped = textwrap.wrap(
        cleaned,
        width=max(8, line_width),
        break_long_words=True,
        break_on_hyphens=False,
    )
    return "<br/>".join(wrapped)


def _node(node_id: str, label: str) -> str:
    return f'    {node_id}["{mermaid_label(label)}"]'


def validate_mermaid_source(source: str) -> tuple[bool, str]:
    lines = [line.rstrip() for line in source.strip().splitlines() if line.strip()]
    if not lines or not re.fullmatch(r"flowchart\s+(?:TD|TB|BT|LR|RL)", lines[0].strip()):
        return False, "缺少合法的 flowchart 方向声明。"
    if _CONTROL_CHARACTERS.search(source):
        return False, "包含 Mermaid 不支持的控制字符。"
    if source.count('"') % 2:
        return False, "节点标签引号未闭合。"
    if source.count("[") != source.count("]"):
        return False, "节点方括号未闭合。"
    if len(lines) < 3:
        return False, "图表节点不足。"
    return True, ""


def repair_mermaid_source(source: str) -> str:
    """Apply one conservative syntax repair without changing graph meaning."""

    repaired = _CONTROL_CHARACTERS.sub("", source).strip()
    lines = [line.rstrip() for line in repaired.splitlines() if line.strip()]
    if not lines or not lines[0].strip().startswith("flowchart "):
        lines.insert(0, "flowchart TD")
    return "\n".join(lines)


def _build_mind_map(article: ArticleDraft, analysis: Analysis) -> MermaidDiagram:
    core = analysis.core_analysis
    topic = core.one_sentence_conclusion
    background = core.structure[0].summary if core.structure else article.numbered_paragraphs[0]
    evidence = [
        f"{claim.claim}；证据：{claim.evidence}"
        for claim in core.claims[:4]
    ] or ["当前分析未列出主要证据"]
    limitations = [item.text for item in core.limitations[:4]] or ["未识别明确局限"]
    applications = [
        point.applicability for point in core.knowledge_points[:4]
    ] or ["未识别明确应用"]

    lines = [
        "flowchart LR",
        _node("topic", f"文章主题：{topic}"),
        _node("background", "背景"),
        _node("question", "核心问题"),
        _node("viewpoint", "核心观点"),
        _node("evidence", "证据或示例"),
        _node("conclusion", "结论"),
        _node("limitations", "局限"),
        _node("applications", "应用"),
        "    topic --> background",
        "    topic --> question",
        "    topic --> viewpoint",
        "    topic --> evidence",
        "    topic --> conclusion",
        "    topic --> limitations",
        "    topic --> applications",
        _node("background_1", background),
        "    background --> background_1",
        _node("question_1", core.core_question),
        "    question --> question_1",
        _node("viewpoint_1", core.author_intended_view),
        "    viewpoint --> viewpoint_1",
        _node("conclusion_1", core.one_sentence_conclusion),
        "    conclusion --> conclusion_1",
    ]
    for index, value in enumerate(evidence, start=1):
        lines.extend([_node(f"evidence_{index}", value), f"    evidence --> evidence_{index}"])
    for index, value in enumerate(limitations, start=1):
        lines.extend(
            [_node(f"limitation_{index}", value), f"    limitations --> limitation_{index}"]
        )
    for index, value in enumerate(applications, start=1):
        lines.extend(
            [_node(f"application_{index}", value), f"    applications --> application_{index}"]
        )

    fallback = [
        f"文章主题：{topic}",
        f"├─ 背景：{background}",
        f"├─ 核心问题：{core.core_question}",
        f"├─ 核心观点：{core.author_intended_view}",
        "├─ 证据或示例",
        *[f"│  └─ {value}" for value in evidence],
        f"├─ 结论：{core.one_sentence_conclusion}",
        "├─ 局限",
        *[f"│  └─ {value}" for value in limitations],
        "└─ 应用",
        *[f"   └─ {value}" for value in applications],
    ]
    source = "\n".join(lines)
    valid, _ = validate_mermaid_source(source)
    if not valid:
        source = repair_mermaid_source(source)
    return MermaidDiagram(
        kind="mind_map",
        title="文章思维导图",
        source=source,
        fallback_lines=fallback,
        repaired=not valid,
    )


def _build_argument_map(analysis: Analysis) -> MermaidDiagram:
    core = analysis.core_analysis
    background_facts = [
        claim.claim
        for claim in core.claims
        if claim.classification in {"事实", "案例"}
    ]
    if not background_facts:
        background_facts = [
            core.structure[0].summary if core.structure else core.core_question
        ]
    assumptions = [item.text for item in core.implicit_assumptions] or ["未识别明确假设"]
    arguments = core.claims
    inferences = [
        leap
        for claim in arguments
        for leap in claim.logical_leaps
    ] or [
        item.text for item in analysis.explanations if item.kind == "AI推断"
    ] or ["主要论据直接指向结论，未单列中间推断"]
    verification = [item.text for item in core.items_to_verify]

    lines = [
        "flowchart LR",
        _node("background", "背景事实"),
        _node("assumptions", "作者假设"),
        _node("arguments", "主要论据"),
        _node("inferences", "中间推断"),
        _node("conclusion", f"最终结论：{core.one_sentence_conclusion}"),
        "    background --> arguments",
        "    assumptions --> arguments",
        "    arguments --> inferences",
        "    inferences --> conclusion",
    ]
    for index, value in enumerate(background_facts[:4], start=1):
        lines.extend([_node(f"fact_{index}", value), f"    fact_{index} --> background"])
    for index, value in enumerate(assumptions[:4], start=1):
        lines.extend(
            [_node(f"assumption_{index}", value), f"    assumption_{index} --> assumptions"]
        )
    weak_nodes: list[str] = []
    for index, claim in enumerate(arguments[:5], start=1):
        node_id = f"argument_{index}"
        label = f"{claim.claim}｜证据：{claim.evidence}｜强度：{claim.evidence_strength}"
        lines.extend([_node(node_id, label), f"    arguments --> {node_id}"])
        if claim.evidence_strength in {"弱", "无直接证据"}:
            weak_nodes.append(node_id)
    risk_nodes: list[str] = []
    for index, value in enumerate(inferences[:5], start=1):
        node_id = f"inference_{index}"
        lines.extend([_node(node_id, value), f"    {node_id} --> inferences"])
        risk_nodes.append(node_id)
    verify_nodes: list[str] = []
    for index, value in enumerate(verification[:4], start=1):
        node_id = f"verify_{index}"
        lines.extend([_node(node_id, f"待验证：{value}"), f"    arguments -.-> {node_id}"])
        verify_nodes.append(node_id)
    lines.extend(
        [
            "    classDef weak fill:#fff3cd,stroke:#b58105,color:#4a3500",
            "    classDef risk fill:#fde2e2,stroke:#b42318,color:#5c0b06",
            "    classDef verify fill:#e8f1ff,stroke:#175cd3,color:#102a56",
        ]
    )
    if weak_nodes:
        lines.append(f"    class {','.join(weak_nodes)} weak")
    if risk_nodes:
        lines.append(f"    class {','.join(risk_nodes)} risk")
    if verify_nodes:
        lines.append(f"    class {','.join(verify_nodes)} verify")

    fallback = [
        "论证结构",
        "├─ 背景事实",
        *[f"│  └─ {value}" for value in background_facts[:4]],
        "├─ 作者假设",
        *[f"│  └─ {value}" for value in assumptions[:4]],
        "├─ 主要论据",
        *[
            f"│  └─ {claim.claim}（证据强度：{claim.evidence_strength}）"
            for claim in arguments[:5]
        ],
        "├─ 中间推断或逻辑跳跃",
        *[f"│  └─ {value}" for value in inferences[:5]],
        f"├─ 最终结论：{core.one_sentence_conclusion}",
        "└─ 待验证节点",
        *([f"   └─ {value}" for value in verification[:4]] or ["   └─ 暂无"]),
    ]
    source = "\n".join(lines)
    valid, _ = validate_mermaid_source(source)
    if not valid:
        source = repair_mermaid_source(source)
    return MermaidDiagram(
        kind="argument_map",
        title="文章论证结构图",
        source=source,
        fallback_lines=fallback,
        repaired=not valid,
    )


def build_visualizations(
    article: ArticleDraft,
    analysis: Analysis,
) -> VisualizationBundle:
    """Build both diagrams from validated structured analysis without an LLM call."""

    return VisualizationBundle(
        article_id=analysis.article_id,
        mind_map=_build_mind_map(article, analysis),
        argument_map=_build_argument_map(analysis),
    )
