"""Unified export document model and Markdown renderer for LayerRead."""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Literal

from pydantic import Field, model_validator

from layerread.analysis_schema import Analysis, StrictModel
from layerread.article import ArticleDraft
from layerread.chat_schema import ChatSession
from layerread.learning_schema import LearningSession
from layerread.storage import article_title
from layerread.visualization import VisualizationBundle


ExportSectionId = Literal[
    "metadata",
    "reading_decision",
    "core_conclusions",
    "knowledge_points",
    "article_structure",
    "plain_explanations",
    "claims_evidence",
    "limitations_counterarguments",
    "visualizations",
    "active_learning",
    "ai_feedback",
    "chat_insights",
    "user_notes",
    "unresolved_questions",
    "original_article",
]
BlockKind = Literal["heading", "paragraph", "list", "quote", "code"]


SECTION_LABELS: dict[ExportSectionId, str] = {
    "metadata": "基本信息",
    "reading_decision": "阅读决策",
    "core_conclusions": "核心结论",
    "knowledge_points": "关键知识点",
    "article_structure": "文章结构",
    "plain_explanations": "通俗讲解",
    "claims_evidence": "观点与证据",
    "limitations_counterarguments": "局限与反方观点",
    "visualizations": "可视化",
    "active_learning": "主动学习",
    "ai_feedback": "AI 反馈",
    "chat_insights": "对话关键洞察",
    "user_notes": "用户笔记",
    "unresolved_questions": "待研究问题",
    "original_article": "原文",
}
SECTION_ORDER = tuple(SECTION_LABELS)


class ExportBlock(StrictModel):
    kind: BlockKind
    text: str = ""
    items: list[str] = Field(default_factory=list)
    level: int = Field(default=3, ge=3, le=6)
    language: str = ""

    @model_validator(mode="after")
    def validate_content(self) -> "ExportBlock":
        if self.kind == "list" and not self.items:
            raise ValueError("列表导出块不能为空。")
        if self.kind != "list" and not self.text:
            raise ValueError("导出块正文不能为空。")
        return self


class ExportSection(StrictModel):
    section_id: ExportSectionId
    title: str = Field(min_length=1)
    blocks: list[ExportBlock] = Field(min_length=1)


class ExportDocument(StrictModel):
    title: str = Field(min_length=1)
    source_url: str = ""
    sections: list[ExportSection] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExportError(RuntimeError):
    """Safe, user-facing export construction failure."""


def _refs(values: list[str]) -> str:
    return " ".join(dict.fromkeys(values)) or "无直接原文引用"


def _section(section_id: ExportSectionId, blocks: list[ExportBlock]) -> ExportSection:
    return ExportSection(
        section_id=section_id,
        title=SECTION_LABELS[section_id],
        blocks=blocks,
    )


def available_export_sections(
    analysis: Analysis | None,
    learning: LearningSession | None,
    chat: ChatSession | None,
    visualizations: VisualizationBundle | None,
) -> list[ExportSectionId]:
    available: list[ExportSectionId] = ["metadata", "original_article"]
    if analysis is not None:
        available.extend(
            [
                "reading_decision",
                "core_conclusions",
                "knowledge_points",
                "article_structure",
                "plain_explanations",
                "claims_evidence",
                "limitations_counterarguments",
            ]
        )
    if visualizations is not None:
        available.append("visualizations")
    if learning is not None:
        available.append("active_learning")
        if learning.feedback is not None:
            available.append("ai_feedback")
    if chat is not None:
        if any(item.selected_for_export for item in chat.saved_insights):
            available.append("chat_insights")
        if any(
            item.selected_for_export and item.user_note.strip()
            for item in chat.saved_insights
        ):
            available.append("user_notes")
        if chat.digest is not None and chat.digest.unresolved_questions:
            available.append("unresolved_questions")
    return [section_id for section_id in SECTION_ORDER if section_id in available]


def _metadata_blocks(
    article: ArticleDraft,
    analysis: Analysis | None,
    source_url: str = "",
) -> list[ExportBlock]:
    effective_source_url = source_url or article.source_url
    items = [
        f"标题：{article_title(article)}",
        f"导入时间：{article.created_at.astimezone().strftime('%Y-%m-%d %H:%M')}",
        f"有效段落：{article.paragraph_count}",
        f"阅读目的：{article.reading_goal or '未填写'}",
        f"主题熟悉度：{article.familiarity}",
        f"关注方向：{article.focus_area}",
        f"导入方式：{'原文链接' if effective_source_url else '手动粘贴'}",
    ]
    optional_metadata = (
        ("公众号/来源", article.account),
        ("作者", article.author),
        ("发布时间", article.published_at),
        ("原文链接", effective_source_url),
        ("抽取方式", article.extraction_method),
    )
    items.extend(f"{label}：{value}" for label, value in optional_metadata if value)
    if analysis is not None:
        items.extend(
            [
                f"分析模型：{analysis.model_name}",
                f"分析 Prompt：{analysis.prompt_version}",
            ]
        )
    return [ExportBlock(kind="list", items=items)]


def _decision_blocks(analysis: Analysis) -> list[ExportBlock]:
    decision = analysis.reading_decision
    criteria = (
        ("整体价值", decision.overall_value),
        ("信息密度", decision.information_density),
        ("证据质量", decision.evidence_quality),
        ("原创分析", decision.original_analysis),
        ("目标相关性", decision.goal_relevance),
        ("推广可能性", decision.promotion_likelihood),
        ("水文可能性", decision.fluff_likelihood),
        ("判断置信度", decision.confidence),
    )
    blocks = [
        ExportBlock(kind="heading", text=decision.recommendation),
        ExportBlock(kind="paragraph", text=decision.recommended_approach),
    ]
    for label, criterion in criteria:
        blocks.append(
            ExportBlock(
                kind="paragraph",
                text=f"**{label}：{criterion.score}/100** — {criterion.explanation}（{_refs(criterion.paragraph_refs)}）",
            )
        )
    return blocks


def _conclusion_blocks(analysis: Analysis) -> list[ExportBlock]:
    core = analysis.core_analysis
    return [
        ExportBlock(kind="heading", text="文章试图解决的问题"),
        ExportBlock(kind="paragraph", text=core.core_question),
        ExportBlock(kind="heading", text="一句话核心结论"),
        ExportBlock(kind="paragraph", text=core.one_sentence_conclusion),
        ExportBlock(kind="heading", text="作者希望读者接受的观点"),
        ExportBlock(kind="paragraph", text=core.author_intended_view),
    ]


def _structure_blocks(analysis: Analysis) -> list[ExportBlock]:
    core = analysis.core_analysis
    return [
        ExportBlock(
            kind="list",
            items=[
                f"{item.order}. {item.title}：{item.summary}（{_refs(item.paragraph_refs)}）"
                for item in sorted(core.structure, key=lambda value: value.order)
            ],
        ),
    ]


def _limitations_blocks(analysis: Analysis) -> list[ExportBlock]:
    core = analysis.core_analysis
    blocks: list[ExportBlock] = []
    groups = (
        ("局限", core.limitations),
        ("反方观点", core.counterarguments),
        ("隐含假设", core.implicit_assumptions),
        ("可能反例", core.possible_counterexamples),
        ("适用边界", core.applicability_boundaries),
        ("待验证内容", core.items_to_verify),
    )
    for title, values in groups:
        if not values:
            continue
        blocks.extend(
            [
                ExportBlock(kind="heading", text=title),
                ExportBlock(
                    kind="list",
                    items=[
                        f"{item.text}（{_refs(item.paragraph_refs)}）"
                        for item in values
                    ],
                ),
            ]
        )
    return blocks or [ExportBlock(kind="paragraph", text="当前分析没有列出局限或反方观点。")]


def _claim_blocks(analysis: Analysis) -> list[ExportBlock]:
    blocks: list[ExportBlock] = []
    for index, claim in enumerate(analysis.core_analysis.claims, start=1):
        blocks.extend(
            [
                ExportBlock(kind="heading", text=f"观点 {index}"),
                ExportBlock(kind="paragraph", text=claim.claim),
                ExportBlock(
                    kind="list",
                    items=[
                        f"类型：{claim.classification}",
                        f"证据类型：{claim.evidence_type}",
                        f"证据强度：{claim.evidence_strength}",
                        f"原文证据概括：{claim.evidence}",
                        f"逻辑跳跃：{'；'.join(claim.logical_leaps) or '未识别'}",
                        f"出处：{_refs(claim.paragraph_refs)}",
                    ],
                ),
            ]
        )
    return blocks or [ExportBlock(kind="paragraph", text="当前分析没有列出观点与证据。")]


def _knowledge_blocks(analysis: Analysis) -> list[ExportBlock]:
    blocks: list[ExportBlock] = []
    for point in analysis.core_analysis.knowledge_points:
        blocks.extend(
            [
                ExportBlock(kind="heading", text=point.title),
                ExportBlock(
                    kind="list",
                    items=[
                        f"定义：{point.definition}",
                        f"通俗解释：{point.plain_explanation}",
                        f"例子：{point.example}",
                        f"适用范围：{point.applicability}",
                        f"常见误解：{'；'.join(point.common_misconceptions) or '未列出'}",
                        f"出处：{_refs(point.paragraph_refs)}",
                    ],
                ),
            ]
        )
    return blocks


def _explanation_blocks(analysis: Analysis) -> list[ExportBlock]:
    explanations = analysis.explanations
    if not explanations:
        return [ExportBlock(kind="paragraph", text="当前分析没有额外解释。")]
    blocks: list[ExportBlock] = []
    for explanation in explanations:
        blocks.extend(
            [
                ExportBlock(kind="heading", text=explanation.kind),
                ExportBlock(kind="paragraph", text=explanation.text),
                ExportBlock(
                    kind="paragraph",
                    text=f"原文出处：{_refs(explanation.paragraph_refs)}",
                ),
            ]
        )
    return blocks


def _visualization_blocks(bundle: VisualizationBundle) -> list[ExportBlock]:
    return [
        ExportBlock(kind="heading", text=bundle.mind_map.title),
        ExportBlock(kind="code", text=bundle.mind_map.source, language="mermaid"),
        ExportBlock(kind="heading", text=bundle.argument_map.title),
        ExportBlock(kind="code", text=bundle.argument_map.source, language="mermaid"),
    ]


def _question_blocks(learning: LearningSession) -> list[ExportBlock]:
    blocks: list[ExportBlock] = []
    for index, question in enumerate(learning.questions.all_questions, start=1):
        blocks.extend(
            [
                ExportBlock(
                    kind="heading",
                    text=f"{index}. {question.question_type} · {question.learning_objective}",
                ),
                ExportBlock(kind="paragraph", text=question.prompt),
                ExportBlock(
                    kind="paragraph",
                    text=f"参考出处：{_refs(question.paragraph_refs)}",
                ),
            ]
        )
    return blocks


def _answer_blocks(learning: LearningSession) -> list[ExportBlock]:
    question_map = {
        question.question_id: question for question in learning.questions.all_questions
    }
    blocks: list[ExportBlock] = []
    for question_id, answer in learning.user_answers.items():
        question = question_map.get(question_id)
        blocks.extend(
            [
                ExportBlock(
                    kind="heading",
                    text=(question.prompt if question else question_id),
                ),
                ExportBlock(kind="quote", text=answer),
            ]
        )
    return blocks or [ExportBlock(kind="paragraph", text="没有已保存的用户回答。")]


def _active_learning_blocks(learning: LearningSession) -> list[ExportBlock]:
    return [
        ExportBlock(kind="heading", text="学习题目"),
        *_question_blocks(learning),
        ExportBlock(kind="heading", text="用户回答"),
        *_answer_blocks(learning),
    ]


def _feedback_blocks(learning: LearningSession) -> list[ExportBlock]:
    feedback = learning.feedback
    if feedback is None:
        return [ExportBlock(kind="paragraph", text="尚未生成 AI 学习反馈。")]
    blocks: list[ExportBlock] = []
    for item in feedback.question_feedback:
        blocks.extend(
            [
                ExportBlock(kind="heading", text=item.question_id),
                ExportBlock(
                    kind="list",
                    items=[
                        f"掌握判断：{item.mastery_status}",
                        f"回答原话：{item.answer_excerpt}",
                        f"正确部分：{'；'.join(item.correct_parts) or '无'}",
                        f"遗漏：{'；'.join(item.omissions) or '无'}",
                        f"不准确之处：{'；'.join(item.inaccuracies) or '无'}",
                        f"改进建议：{'；'.join(item.improvement_suggestions)}",
                        f"出处：{_refs(item.paragraph_refs)}",
                    ],
                ),
            ]
        )
    blocks.extend(
        [
            ExportBlock(kind="heading", text="总体复习建议"),
            ExportBlock(kind="paragraph", text=feedback.overall_review_advice),
            ExportBlock(kind="heading", text="复习卡片"),
            *_review_card_blocks(learning),
        ]
    )
    return blocks


def _review_card_blocks(learning: LearningSession) -> list[ExportBlock]:
    cards = learning.review_cards
    if not cards:
        return [ExportBlock(kind="paragraph", text="尚未生成复习卡片。")]
    blocks: list[ExportBlock] = []
    for index, card in enumerate(cards, start=1):
        blocks.extend(
            [
                ExportBlock(kind="heading", text=f"卡片 {index}：{card.question}"),
                ExportBlock(kind="paragraph", text=f"短答案：{card.short_answer}"),
                ExportBlock(kind="paragraph", text=card.detailed_explanation),
                ExportBlock(kind="paragraph", text=f"出处：{_refs(card.paragraph_refs)}"),
            ]
        )
    return blocks


def _insight_blocks(chat: ChatSession) -> list[ExportBlock]:
    message_map = {message.id: message for message in chat.messages}
    blocks: list[ExportBlock] = []
    for index, insight in enumerate(
        (item for item in chat.saved_insights if item.selected_for_export),
        start=1,
    ):
        source = message_map.get(insight.source_message_id)
        refs = source.source_paragraphs if source is not None else []
        blocks.extend(
            [
                ExportBlock(kind="heading", text=f"洞察 {index}"),
                ExportBlock(kind="paragraph", text=insight.content),
                ExportBlock(kind="paragraph", text=f"原文出处：{_refs(refs)}"),
            ]
        )
    return blocks or [ExportBlock(kind="paragraph", text="没有选择用于导出的洞察。")]


def _note_blocks(chat: ChatSession) -> list[ExportBlock]:
    blocks: list[ExportBlock] = []
    for index, insight in enumerate(
        (
            item
            for item in chat.saved_insights
            if item.selected_for_export and item.user_note.strip()
        ),
        start=1,
    ):
        blocks.extend(
            [
                ExportBlock(kind="heading", text=f"笔记 {index}"),
                ExportBlock(kind="quote", text=insight.user_note),
            ]
        )
    return blocks or [ExportBlock(kind="paragraph", text="没有选择用于导出的用户笔记。")]


def build_export_document(
    article: ArticleDraft,
    analysis: Analysis | None,
    learning: LearningSession | None,
    chat: ChatSession | None,
    visualizations: VisualizationBundle | None,
    selected_sections: list[ExportSectionId],
    *,
    source_url: str = "",
) -> ExportDocument:
    if not selected_sections:
        raise ExportError("请至少选择一个导出模块。")
    available = set(available_export_sections(analysis, learning, chat, visualizations))
    unavailable = set(selected_sections) - available
    if unavailable:
        labels = [SECTION_LABELS[section_id] for section_id in SECTION_ORDER if section_id in unavailable]
        raise ExportError("以下模块当前没有可导出内容：" + "、".join(labels))

    effective_source_url = source_url or article.source_url
    builders = {
        "metadata": lambda: _metadata_blocks(article, analysis, effective_source_url),
        "original_article": lambda: [
            ExportBlock(kind="quote", text=article.numbered_text)
        ],
        "reading_decision": lambda: _decision_blocks(analysis),
        "core_conclusions": lambda: _conclusion_blocks(analysis),
        "knowledge_points": lambda: _knowledge_blocks(analysis),
        "article_structure": lambda: _structure_blocks(analysis),
        "plain_explanations": lambda: _explanation_blocks(analysis),
        "claims_evidence": lambda: _claim_blocks(analysis),
        "limitations_counterarguments": lambda: _limitations_blocks(analysis),
        "visualizations": lambda: _visualization_blocks(visualizations),
        "active_learning": lambda: _active_learning_blocks(learning),
        "ai_feedback": lambda: _feedback_blocks(learning),
        "chat_insights": lambda: _insight_blocks(chat),
        "user_notes": lambda: _note_blocks(chat),
        "unresolved_questions": lambda: [
            ExportBlock(kind="list", items=chat.digest.unresolved_questions)
        ],
    }
    selected = set(selected_sections)
    sections = [
        _section(section_id, builders[section_id]())
        for section_id in SECTION_ORDER
        if section_id in selected
    ]
    return ExportDocument(
        title=article_title(article),
        source_url=effective_source_url,
        sections=sections,
    )


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def render_markdown(document: ExportDocument) -> str:
    lines = [f"# {document.title}", ""]
    if document.source_url:
        lines.extend([f"原文链接：[{document.source_url}]({document.source_url})", ""])
    for section in document.sections:
        lines.extend([f"## {section.title}", ""])
        for block in section.blocks:
            if block.kind == "heading":
                lines.extend([f"{'#' * block.level} {block.text}", ""])
            elif block.kind == "paragraph":
                lines.extend([block.text, ""])
            elif block.kind == "list":
                lines.extend([*(f"- {item}" for item in block.items), ""])
            elif block.kind == "quote":
                lines.extend([_quote(block.text), ""])
            elif block.kind == "code":
                lines.extend([f"```{block.language}", block.text, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def markdown_filename(document: ExportDocument) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", document.title).strip(" ._")
    safe = safe[:60].rstrip() or "layerread-export"
    return f"{safe}.md"
