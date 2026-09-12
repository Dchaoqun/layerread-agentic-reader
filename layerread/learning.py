"""Model workflows and state transitions for LayerRead v0.3 active learning."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import re
import unicodedata
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from layerread.analysis_schema import Analysis
from layerread.analyzer import ChatClient, OpenAICompatibleClient
from layerread.article import ArticleDraft
from layerread.config import ModelSettings
from layerread.learning_schema import (
    LearningFeedback,
    LearningQuestionSet,
    LearningSession,
)
from layerread.prompts import active_prompt_version, prompt_text
from layerread.references import normalize_references


QUESTION_PROMPT_VERSION = "v0.3.1-questions"
FEEDBACK_PROMPT_VERSION = "v0.3.2-feedback-verbatim-map"
SchemaT = TypeVar("SchemaT", bound=BaseModel)
PayloadPreprocessor = Callable[[Any], Any]


class LearningError(RuntimeError):
    """Safe, user-facing active-learning failure."""


def _allowed_references(article: ArticleDraft) -> list[str]:
    return [paragraph.partition(" ")[0] for paragraph in article.numbered_paragraphs]


def _schema_with_allowed_references(
    schema_type: type[SchemaT],
    article: ArticleDraft,
) -> dict[str, Any]:
    schema = schema_type.model_json_schema()
    allowed = _allowed_references(article)

    def constrain(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "paragraph_refs" and isinstance(child, dict):
                    child["items"] = {"type": "string", "enum": allowed}
                else:
                    constrain(child)
        elif isinstance(value, list):
            for child in value:
                constrain(child)

    constrain(schema)
    return schema


def _extract_json(raw_response: str) -> Any:
    text = raw_response.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _validation_issue(exc: ValidationError) -> str:
    issues: list[str] = []
    for error in exc.errors(include_input=False, include_url=False)[:8]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "根对象"
        message = " ".join(str(error.get("msg", "字段无效")).split())
        issues.append(f"{location}: {message}")
    if len(exc.errors()) > len(issues):
        issues.append("还有其他字段未通过校验")
    return "；".join(issues)


def _parse_schema(
    raw_response: str,
    schema_type: type[SchemaT],
    preprocess: PayloadPreprocessor | None = None,
) -> SchemaT:
    try:
        payload = _extract_json(raw_response)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LearningError("JSON 无法解析为完整对象") from exc

    if preprocess is not None:
        payload = preprocess(payload)
    try:
        return schema_type.model_validate(payload)
    except ValidationError as exc:
        raise LearningError(f"字段校验失败：{_validation_issue(exc)}") from exc


def _as_list(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return value


def _append_requirement(question: dict[str, Any], requirement: str) -> None:
    prompt = question.get("prompt")
    if isinstance(prompt, str) and prompt.strip() and requirement not in prompt:
        question["prompt"] = f"{prompt.strip()} {requirement}"


def _normalize_question_fields(question: dict[str, Any]) -> None:
    for key in ("evaluation_points", "paragraph_refs", "critical_dimensions"):
        if key in question:
            question[key] = _as_list(question[key])


def _normalize_question_payload(
    payload: Any,
    article: ArticleDraft,
    analysis: Analysis,
) -> Any:
    """Fix deterministic question metadata without inventing article claims."""

    if not isinstance(payload, dict):
        return payload
    normalized = deepcopy(payload)
    recall_questions = normalized.get("recall_questions")
    recall_requirements = (
        ("文章问题", "请明确文章试图解决的问题。", "recall-problem"),
        ("核心观点", "请用自己的话概括文章的核心观点。", "recall-view"),
        ("证据链", "请重建支持核心观点的主要证据或论证链。", "recall-evidence"),
    )
    if isinstance(recall_questions, list):
        for index, question in enumerate(recall_questions):
            if not isinstance(question, dict):
                continue
            _normalize_question_fields(question)
            objective, requirement, default_id = recall_requirements[
                min(index, len(recall_requirements) - 1)
            ]
            question["question_id"] = default_id if index < 3 else f"recall-{index + 1}"
            question["question_type"] = "核心回忆"
            question["learning_objective"] = objective
            question["critical_dimensions"] = []
            _append_requirement(question, requirement)

    teach_back = normalized.get("teach_back_question")
    if isinstance(teach_back, dict):
        _normalize_question_fields(teach_back)
        teach_back["question_id"] = "teach-back"
        teach_back["question_type"] = "Teach-back"
        teach_back["learning_objective"] = "概念讲授"
        teach_back["critical_dimensions"] = []
        teach_back["scenario"] = "向不熟悉该领域的同事讲解"
        if not str(teach_back.get("focus_concept", "")).strip():
            teach_back["focus_concept"] = analysis.core_analysis.knowledge_points[0].title
        _append_requirement(
            teach_back,
            "请把它讲给一位不熟悉该领域的同事，并说明关键因果关系。",
        )

    application = normalized.get("application_question")
    if isinstance(application, dict):
        _normalize_question_fields(application)
        application["question_id"] = "application"
        application["question_type"] = "应用题"
        application["learning_objective"] = "知识迁移"
        application["critical_dimensions"] = []
        application["scenario"] = f"围绕用户关注方向“{article.focus_area}”进行知识迁移"
        _append_requirement(
            application,
            f"请把文章观点迁移到“{article.focus_area}”相关的真实场景。",
        )

    critical = normalized.get("critical_question")
    if isinstance(critical, dict):
        _normalize_question_fields(critical)
        critical["question_id"] = "critical"
        critical["question_type"] = "批判性问题"
        critical["learning_objective"] = "批判性评估"
        critical["critical_dimensions"] = [
            "隐含假设",
            "薄弱证据",
            "可能反例",
            "事实与预测",
        ]
        _append_requirement(
            critical,
            "请同时讨论隐含假设、薄弱证据、可能反例，以及哪些陈述是事实、哪些是预测。",
        )
    return normalized


def _normalize_feedback_payload(
    payload: Any,
    answers: dict[str, str],
) -> Any:
    """Correct safe list and quotation formatting in feedback JSON."""

    if not isinstance(payload, dict):
        return payload
    normalized = deepcopy(payload)
    question_feedback = normalized.get("question_feedback")
    if isinstance(question_feedback, list):
        for item in question_feedback:
            if not isinstance(item, dict):
                continue
            for key in (
                "correct_parts",
                "omissions",
                "inaccuracies",
                "improvement_suggestions",
                "paragraph_refs",
            ):
                if key in item:
                    item[key] = _as_list(item[key])
            question_id = item.get("question_id")
            excerpt = item.get("answer_excerpt")
            if isinstance(question_id, str) and isinstance(excerpt, str):
                answer = answers.get(question_id, "")
                exact_excerpt = _match_exact_answer_excerpt(answer, excerpt)
                if exact_excerpt is not None:
                    item["answer_excerpt"] = exact_excerpt

    mastery_status = normalized.get("mastery_status")
    if isinstance(mastery_status, list):
        for item in mastery_status:
            if isinstance(item, dict) and "paragraph_refs" in item:
                item["paragraph_refs"] = _as_list(item["paragraph_refs"])

    review_cards = normalized.get("review_cards")
    if isinstance(review_cards, list):
        for item in review_cards:
            if isinstance(item, dict) and "paragraph_refs" in item:
                item["paragraph_refs"] = _as_list(item["paragraph_refs"])
    return normalized


def _normalized_non_whitespace_with_offsets(value: str) -> tuple[str, list[int]]:
    """Return an NFKC/whitespace-insensitive view plus source offsets."""

    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(value):
        for normalized in unicodedata.normalize("NFKC", character):
            if normalized.isspace():
                continue
            characters.append(normalized)
            offsets.append(index)
    return "".join(characters), offsets


def _match_exact_answer_excerpt(answer: str, excerpt: str) -> str | None:
    """Map safe formatting variants back to a real, continuous answer slice."""

    if excerpt in answer:
        return excerpt
    stripped = excerpt.strip().strip("“”‘’\"'").strip()
    if not stripped:
        return None
    if stripped in answer:
        return stripped

    normalized_answer, offsets = _normalized_non_whitespace_with_offsets(answer)
    normalized_excerpt, _ = _normalized_non_whitespace_with_offsets(stripped)
    if not normalized_excerpt:
        return None
    start = normalized_answer.find(normalized_excerpt)
    if start < 0:
        return None
    end = start + len(normalized_excerpt) - 1
    return answer[offsets[start] : offsets[end] + 1]


def _verbatim_answer_excerpt(answer: str, max_chars: int = 72) -> str:
    """Select one deterministic short excerpt that is guaranteed to be verbatim."""

    stripped = answer.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip()


def _feedback_excerpt_map(answers: dict[str, str]) -> dict[str, str]:
    return {
        question_id: _verbatim_answer_excerpt(answer)
        for question_id, answer in answers.items()
        if answer.strip()
    }


def _constrain_feedback_excerpts(
    schema: dict[str, Any],
    excerpt_map: dict[str, str],
) -> None:
    allowed = list(dict.fromkeys(excerpt_map.values()))

    def constrain(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "answer_excerpt" and isinstance(child, dict):
                    child["enum"] = allowed
                else:
                    constrain(child)
        elif isinstance(value, list):
            for child in value:
                constrain(child)

    constrain(schema)


def _repair_messages(
    messages: list[dict[str, str]],
    invalid_response: str,
    issue: str,
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": invalid_response},
        {
            "role": "user",
            "content": prompt_text("learning_repair", issue=issue),
        },
    ]


def build_question_messages(
    article: ArticleDraft,
    analysis: Analysis,
) -> list[dict[str, str]]:
    schema = _schema_with_allowed_references(LearningQuestionSet, article)
    allowed = _allowed_references(article)
    analysis_context = {
        "core_question": analysis.core_analysis.core_question,
        "core_conclusion": analysis.core_analysis.one_sentence_conclusion,
        "author_intended_view": analysis.core_analysis.author_intended_view,
        "knowledge_points": [
            point.model_dump(mode="json")
            for point in analysis.core_analysis.knowledge_points
        ],
        "claims": [
            claim.model_dump(mode="json")
            for claim in analysis.core_analysis.claims
        ],
        "limitations": [
            item.model_dump(mode="json")
            for item in analysis.core_analysis.limitations
        ],
    }
    return [
        {
            "role": "system",
            "content": prompt_text("learning_questions"),
        },
        {
            "role": "user",
            "content": (
                f"阅读目的：{article.reading_goal or '未填写'}\n"
                f"熟悉度：{article.familiarity}\n"
                f"关注方向：{article.focus_area}\n"
                f"合法段落编号：{', '.join(allowed)}\n\n"
                "文章分析：\n"
                f"{json.dumps(analysis_context, ensure_ascii=False)}\n\n"
                "JSON Schema：\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"编号正文：\n{article.numbered_text}"
            ),
        },
    ]


def _validate_feedback_against_answers(
    feedback: LearningFeedback,
    questions: LearningQuestionSet,
    answers: dict[str, str],
    analysis: Analysis,
) -> None:
    known_question_ids = {question.question_id for question in questions.all_questions}
    answered_ids = {question_id for question_id, answer in answers.items() if answer.strip()}
    feedback_ids = {item.question_id for item in feedback.question_feedback}
    if not feedback_ids or feedback_ids != answered_ids:
        raise LearningError("反馈必须逐一覆盖且只覆盖本次已作答题目。")
    if not feedback_ids <= known_question_ids:
        raise LearningError("反馈包含未知题目编号。")

    invalid_excerpt_ids: list[str] = []
    for item in feedback.question_feedback:
        answer = answers[item.question_id]
        if item.answer_excerpt not in answer:
            invalid_excerpt_ids.append(item.question_id)
        question = next(
            question
            for question in questions.all_questions
            if question.question_id == item.question_id
        )
        if question.question_type == "Teach-back" and (
            not item.concept_accuracy or not item.causal_reasoning
        ):
            raise LearningError("Teach-back 反馈必须分别评价概念准确性和因果关系。")

    if invalid_excerpt_ids:
        raise LearningError(
            "以下题目的 answer_excerpt 不是原回答连续原文："
            f"{', '.join(invalid_excerpt_ids)}。"
            "请为每题逐字复制提示中该 question_id 对应的引用候选，"
            "不得改写、拼接或使用省略号。"
        )

    expected_concepts = {
        point.title for point in analysis.core_analysis.knowledge_points
    }
    actual_concepts = {item.concept_title for item in feedback.mastery_status}
    if actual_concepts != expected_concepts:
        raise LearningError("掌握状态必须逐一覆盖分析中的核心知识点并使用原名称。")


def build_feedback_messages(
    article: ArticleDraft,
    analysis: Analysis,
    questions: LearningQuestionSet,
    answers: dict[str, str],
) -> list[dict[str, str]]:
    schema = _schema_with_allowed_references(LearningFeedback, article)
    question_map = {
        question.question_id: question.model_dump(mode="json")
        for question in questions.all_questions
        if question.question_id in answers and answers[question.question_id].strip()
    }
    answer_payload = {
        question_id: answers[question_id]
        for question_id in question_map
    }
    excerpt_map = _feedback_excerpt_map(answer_payload)
    _constrain_feedback_excerpts(schema, excerpt_map)
    concepts = [point.title for point in analysis.core_analysis.knowledge_points]
    return [
        {
            "role": "system",
            "content": prompt_text("learning_feedback"),
        },
        {
            "role": "user",
            "content": (
                f"核心概念名称（必须原样覆盖）：{json.dumps(concepts, ensure_ascii=False)}\n"
                f"合法段落编号：{', '.join(_allowed_references(article))}\n\n"
                "本次题目：\n"
                f"{json.dumps(question_map, ensure_ascii=False)}\n\n"
                "用户回答：\n"
                f"{json.dumps(answer_payload, ensure_ascii=False)}\n\n"
                "逐字引用候选（question_id -> 必须使用的 answer_excerpt）：\n"
                f"{json.dumps(excerpt_map, ensure_ascii=False)}\n\n"
                "JSON Schema：\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"编号正文：\n{article.numbered_text}"
            ),
        },
    ]


def _generate_with_one_repair(
    client: ChatClient,
    messages: list[dict[str, str]],
    schema_type: type[SchemaT],
    article: ArticleDraft,
    cross_validate: Callable[[SchemaT], None] | None = None,
    preprocess: PayloadPreprocessor | None = None,
) -> tuple[SchemaT, str, Any]:
    raw_response = client.complete(messages)
    repaired = False

    def parse_and_check(raw: str) -> SchemaT:
        result = _parse_schema(raw, schema_type, preprocess)
        if cross_validate is not None:
            cross_validate(result)
        return result

    try:
        payload = parse_and_check(raw_response)
    except LearningError as exc:
        raw_response = client.complete(
            _repair_messages(messages, raw_response, str(exc))
        )
        repaired = True
        try:
            payload = parse_and_check(raw_response)
        except LearningError as repair_exc:
            raise LearningError(
                "模型输出在一次自动修复后仍未通过学习校验。"
                f"具体原因：{repair_exc}"
            ) from None

    payload, diagnostics = normalize_references(payload, article)
    if diagnostics.invalid_references and not repaired:
        raw_response = client.complete(
            _repair_messages(
                messages,
                raw_response,
                "格式修正后仍有无效引用："
                f"{', '.join(diagnostics.invalid_references)}",
            )
        )
        try:
            payload = parse_and_check(raw_response)
        except LearningError as repair_exc:
            raise LearningError(
                f"模型修复引用时返回了无效学习结构。具体原因：{repair_exc}"
            ) from None
        payload, diagnostics = normalize_references(payload, article)

    return payload, raw_response, diagnostics


def generate_learning_session(
    article: ArticleDraft,
    analysis: Analysis,
    settings: ModelSettings,
    client_factory: Callable[[ModelSettings], ChatClient] = OpenAICompatibleClient,
) -> LearningSession:
    client = client_factory(settings)
    questions, raw_response, diagnostics = _generate_with_one_repair(
        client,
        build_question_messages(article, analysis),
        LearningQuestionSet,
        article,
        preprocess=lambda payload: _normalize_question_payload(
            payload,
            article,
            analysis,
        ),
    )
    return LearningSession(
        article_id=analysis.article_id,
        model_name=settings.model,
        question_prompt_version=active_prompt_version(QUESTION_PROMPT_VERSION),
        feedback_prompt_version=active_prompt_version(FEEDBACK_PROMPT_VERSION),
        questions=questions,
        question_raw_response=raw_response,
        question_reference_diagnostics=diagnostics,
    )


def answers_fingerprint(answers: dict[str, str]) -> str:
    normalized = {
        question_id: answer.strip()
        for question_id, answer in sorted(answers.items())
        if answer.strip()
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def update_learning_answers(
    session: LearningSession,
    answers: dict[str, str],
) -> LearningSession:
    known_ids = {question.question_id for question in session.questions.all_questions}
    unknown_ids = set(answers) - known_ids
    if unknown_ids:
        raise LearningError("答案包含未知题目，无法保存。")
    normalized = {
        question_id: answer.strip()
        for question_id, answer in answers.items()
        if answer.strip()
    }
    return LearningSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "user_answers": normalized,
            "updated_at": datetime.now(UTC),
        }
    )


def evaluate_learning_session(
    session: LearningSession,
    article: ArticleDraft,
    analysis: Analysis,
    settings: ModelSettings,
    client_factory: Callable[[ModelSettings], ChatClient] = OpenAICompatibleClient,
) -> LearningSession:
    answers = {
        question_id: answer
        for question_id, answer in session.user_answers.items()
        if answer.strip()
    }
    if not answers:
        raise LearningError("请至少回答一道题，再获取学习反馈。")
    if session.article_id != analysis.article_id:
        raise LearningError("学习会话与当前文章分析不一致，请重新生成题目。")

    client = client_factory(settings)
    feedback, raw_response, diagnostics = _generate_with_one_repair(
        client,
        build_feedback_messages(
            article,
            analysis,
            session.questions,
            answers,
        ),
        LearningFeedback,
        article,
        cross_validate=lambda result: _validate_feedback_against_answers(
            result,
            session.questions,
            answers,
            analysis,
        ),
        preprocess=lambda payload: _normalize_feedback_payload(payload, answers),
    )
    return LearningSession.model_validate(
        {
            **session.model_dump(mode="python"),
            "feedback": feedback,
            "mastery_status": feedback.mastery_status,
            "review_cards": feedback.review_cards,
            "feedback_raw_response": raw_response,
            "feedback_reference_diagnostics": diagnostics,
            "feedback_answer_fingerprint": answers_fingerprint(answers),
            "feedback_prompt_version": active_prompt_version(
                FEEDBACK_PROMPT_VERSION
            ),
            "updated_at": datetime.now(UTC),
        }
    )
