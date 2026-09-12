"""Paragraph-reference validation for traceable model output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, TypeVar
import unicodedata

from pydantic import BaseModel

from layerread.analysis_schema import ReferenceDiagnosticItem, ReferenceDiagnostics
from layerread.article import ArticleDraft


_REFERENCE_PATTERN = re.compile(r"^\[P\d+\]$")
_SINGLE_REFERENCE_PATTERN = re.compile(
    r"^(?:\[\s*[Pp]\s*0*(\d+)\s*\]|[Pp]\s*0*(\d+))$"
)
_REFERENCE_TOKEN_PATTERN = re.compile(
    r"(?:\[\s*[Pp]\s*0*(\d+)\s*\]|[Pp]\s*0*(\d+))",
    re.IGNORECASE,
)
_RANGE_PATTERNS = (
    re.compile(
        r"^\[\s*[Pp]\s*0*(\d+)\s*(?:-|–|—|~|至|到)\s*"
        r"[Pp]\s*0*(\d+)\s*\]$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\[\s*[Pp]\s*0*(\d+)\s*\]\s*(?:-|–|—|~|至|到)\s*"
        r"\[\s*[Pp]\s*0*(\d+)\s*\]$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[Pp]\s*0*(\d+)\s*(?:-|–|—|~|至|到)\s*"
        r"[Pp]\s*0*(\d+)$",
        re.IGNORECASE,
    ),
)
_LIST_REMAINDER_PATTERN = re.compile(r"^[\s,，、;；/|]*$")
_MAX_RANGE_SIZE = 20
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ReferenceReport:
    valid: tuple[str, ...]
    invalid: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.invalid


@dataclass(frozen=True, slots=True)
class ReferenceValueResult:
    valid: tuple[str, ...]
    invalid: tuple[str, ...]
    corrections: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class _ParsedReference:
    normalized_refs: tuple[str, ...]
    valid_refs: tuple[str, ...]
    invalid_refs: tuple[str, ...]
    status: str
    category: str
    reason: str


def article_paragraph_map(article: ArticleDraft) -> dict[str, str]:
    """Map stable paragraph IDs to their source text."""

    result: dict[str, str] = {}
    for paragraph in article.numbered_paragraphs:
        ref, _, text = paragraph.partition(" ")
        result[ref] = text
    return result


def _canonical_maps(article: ArticleDraft) -> tuple[set[str], dict[int, str]]:
    available = set(article_paragraph_map(article))
    by_number: dict[int, str] = {}
    for reference in available:
        match = _REFERENCE_PATTERN.fullmatch(reference)
        if match:
            number = int(reference[2:-1])
            by_number[number] = reference
    return available, by_number


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _classify_candidates(
    candidates: list[int],
    by_number: dict[int, str],
    *,
    corrected: bool,
    category: str,
    reason: str,
) -> _ParsedReference:
    normalized: list[str] = []
    valid: list[str] = []
    invalid: list[str] = []
    for number in candidates:
        reference = by_number.get(number, f"[P{number:02d}]")
        normalized.append(reference)
        if number in by_number:
            valid.append(reference)
        else:
            invalid.append(reference)

    normalized_values = _deduplicate(normalized)
    valid_values = tuple(valid)
    invalid_values = tuple(invalid)
    if invalid_values and valid_values:
        status = "partially_invalid"
        final_category = "partially_invalid"
        final_reason = f"{reason}；其中部分编号不存在于当前正文。"
    elif invalid_values:
        status = "invalid"
        final_category = "nonexistent"
        final_reason = "编号不存在于当前正文。"
    elif corrected:
        status = "auto_corrected"
        final_category = category
        final_reason = reason
    else:
        status = "valid"
        final_category = "exact"
        final_reason = "格式正确且编号存在。"
    return _ParsedReference(
        normalized_refs=normalized_values,
        valid_refs=valid_values,
        invalid_refs=invalid_values,
        status=status,
        category=final_category,
        reason=final_reason,
    )


def _parse_reference(
    raw_reference: str,
    available: set[str],
    by_number: dict[int, str],
) -> _ParsedReference:
    stripped = unicodedata.normalize("NFKC", raw_reference).strip()
    if stripped in available:
        return _ParsedReference(
            normalized_refs=(stripped,),
            valid_refs=(stripped,),
            invalid_refs=(),
            status="valid" if stripped == raw_reference else "auto_corrected",
            category="exact" if stripped == raw_reference else "format_corrected",
            reason=(
                "格式正确且编号存在。"
                if stripped == raw_reference
                else "已移除引用前后的多余空白。"
            ),
        )

    range_match = next(
        (pattern.fullmatch(stripped) for pattern in _RANGE_PATTERNS if pattern.fullmatch(stripped)),
        None,
    )
    if range_match:
        start, end = (int(value) for value in range_match.groups())
        if (
            start > end
            or end - start + 1 > _MAX_RANGE_SIZE
            or start not in by_number
            or end not in by_number
        ):
            return _ParsedReference(
                normalized_refs=(stripped,),
                valid_refs=(),
                invalid_refs=(stripped,),
                status="invalid",
                category="invalid_range",
                reason="引用范围倒序、跨度过大或端点不存在，未自动展开。",
            )
        return _classify_candidates(
            list(range(start, end + 1)),
            by_number,
            corrected=True,
            category="expanded_range",
            reason="已将段落范围展开为独立编号。",
        )

    single_match = _SINGLE_REFERENCE_PATTERN.fullmatch(stripped)
    if single_match:
        number = next(int(value) for value in single_match.groups() if value is not None)
        return _classify_candidates(
            [number],
            by_number,
            corrected=True,
            category="format_corrected",
            reason="已补齐括号、大小写或编号前导零。",
        )

    list_source = stripped
    if (
        stripped.startswith("[")
        and stripped.endswith("]")
        and stripped.count("[") == 1
        and stripped.count("]") == 1
    ):
        list_source = stripped[1:-1]
    token_matches = list(_REFERENCE_TOKEN_PATTERN.finditer(list_source))
    if len(token_matches) >= 2:
        remainder = _REFERENCE_TOKEN_PATTERN.sub("", list_source)
        if _LIST_REMAINDER_PATTERN.fullmatch(remainder):
            numbers = [
                next(int(value) for value in match.groups() if value is not None)
                for match in token_matches
            ]
            return _classify_candidates(
                numbers,
                by_number,
                corrected=True,
                category="split_list",
                reason="已将合并引用拆分为独立编号。",
            )

    return _ParsedReference(
        normalized_refs=(stripped or "<空引用>",),
        valid_refs=(),
        invalid_refs=(stripped or "<空引用>",),
        status="invalid",
        category="unparseable",
        reason="无法安全解析为段落编号，未自动修改。",
    )


def _risk_summary(
    total_mentions: int,
    invalid_mentions: int,
    invalid_references: list[str],
) -> tuple[str, str]:
    if invalid_mentions == 0:
        return "none", "格式修正后没有发现不存在或无法解析的引用。"

    invalid_rate = invalid_mentions / max(total_mentions, 1)
    if invalid_rate <= 0.05 and len(invalid_references) <= 1:
        level = "low"
        label = "低"
    elif invalid_rate <= 0.15 and len(invalid_references) <= 3:
        level = "medium"
        label = "中"
    else:
        level = "high"
        label = "高"
    return (
        level,
        f"引用存在性风险为{label}：格式修正后仍有 "
        f"{invalid_mentions}/{total_mentions} 次引用无法对应当前正文。",
    )


def normalize_references(
    model: ModelT,
    article: ArticleDraft,
) -> tuple[ModelT, ReferenceDiagnostics]:
    """Conservatively fix citation formatting and diagnose remaining errors."""

    available, by_number = _canonical_maps(article)
    data = model.model_dump(mode="python")
    items: list[ReferenceDiagnosticItem] = []

    def statement_context(container: dict[str, Any]) -> str:
        for key in (
            "claim",
            "text",
            "explanation",
            "summary",
            "title",
            "rationale",
            "plain_explanation",
            "definition",
            "repetition",
        ):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key == "paragraph_refs" and isinstance(child, list):
                    normalized_slot: list[str] = []
                    for index, reference in enumerate(child):
                        original = str(reference)
                        parsed = _parse_reference(original, available, by_number)
                        normalized_slot.extend(parsed.normalized_refs)
                        items.append(
                            ReferenceDiagnosticItem(
                                location=f"{child_path}[{index}]",
                                original=original,
                                normalized_refs=list(parsed.normalized_refs),
                                valid_refs=list(parsed.valid_refs),
                                invalid_refs=list(parsed.invalid_refs),
                                statement=statement_context(value),
                                status=parsed.status,
                                category=parsed.category,
                                reason=parsed.reason,
                            )
                        )
                    value[key] = list(dict.fromkeys(normalized_slot))
                else:
                    walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(data, "")
    normalized_model = type(model).model_validate(data)

    total_mentions = 0
    exact_valid_mentions = 0
    corrected_valid_mentions = 0
    invalid_mentions = 0
    nonexistent_mentions = 0
    unparseable_mentions = 0
    unique_valid: list[str] = []
    unique_invalid: list[str] = []
    auto_corrections: dict[str, list[str]] = {}
    for item in items:
        mention_count = max(
            1,
            len(item.valid_refs) + len(item.invalid_refs),
        )
        total_mentions += mention_count
        if item.status == "valid":
            exact_valid_mentions += len(item.valid_refs)
        else:
            corrected_valid_mentions += len(item.valid_refs)
        invalid_mentions += len(item.invalid_refs) or (1 if item.status == "invalid" else 0)
        if item.category in {"nonexistent", "partially_invalid"}:
            nonexistent_mentions += len(item.invalid_refs)
        elif item.category in {"invalid_range", "unparseable"}:
            unparseable_mentions += len(item.invalid_refs) or 1
        for reference in item.valid_refs:
            if reference not in unique_valid:
                unique_valid.append(reference)
        for reference in item.invalid_refs:
            if reference not in unique_invalid:
                unique_invalid.append(reference)
        if item.status in {"auto_corrected", "partially_invalid"}:
            auto_corrections[item.original] = item.normalized_refs

    slots: dict[str, dict[str, int]] = {}
    for item in items:
        slot_path = item.location.rsplit("[", 1)[0]
        slot = slots.setdefault(slot_path, {"valid": 0, "invalid": 0})
        slot["valid"] += len(item.valid_refs)
        slot["invalid"] += len(item.invalid_refs)
    affected_locations = sum(1 for slot in slots.values() if slot["invalid"])
    zero_valid_locations = sum(
        1
        for slot in slots.values()
        if slot["invalid"] and not slot["valid"]
    )

    risk_level, risk_summary = _risk_summary(
        total_mentions,
        invalid_mentions,
        unique_invalid,
    )
    diagnostics = ReferenceDiagnostics(
        total_mentions=total_mentions,
        exact_valid_mentions=exact_valid_mentions,
        corrected_valid_mentions=corrected_valid_mentions,
        invalid_mentions=invalid_mentions,
        nonexistent_mentions=nonexistent_mentions,
        unparseable_mentions=unparseable_mentions,
        affected_locations=affected_locations,
        zero_valid_locations=zero_valid_locations,
        unique_valid_references=unique_valid,
        auto_corrections=auto_corrections,
        invalid_references=unique_invalid,
        items=items,
        risk_level=risk_level,
        risk_summary=risk_summary,
    )
    return normalized_model, diagnostics


def validate_references(
    model: BaseModel,
    article: ArticleDraft,
) -> ReferenceReport:
    """Identify real and fabricated paragraph IDs without changing output."""

    _, diagnostics = normalize_references(model, article)
    return ReferenceReport(
        valid=tuple(diagnostics.unique_valid_references),
        invalid=tuple(diagnostics.invalid_references),
    )


def valid_only(references: list[str], article: ArticleDraft) -> list[str]:
    """Return only references that can safely be shown as source citations."""

    return list(normalize_reference_values(references, article).valid)


def normalize_reference_values(
    references: list[str],
    article: ArticleDraft,
) -> ReferenceValueResult:
    """Normalize one reference list for adjacent UI display."""

    available, by_number = _canonical_maps(article)
    verified: list[str] = []
    invalid: list[str] = []
    corrections: list[tuple[str, tuple[str, ...]]] = []
    for reference in references:
        original = str(reference)
        parsed = _parse_reference(original, available, by_number)
        for valid_reference in parsed.valid_refs:
            if valid_reference not in verified:
                verified.append(valid_reference)
        for invalid_reference in parsed.invalid_refs:
            if invalid_reference not in invalid:
                invalid.append(invalid_reference)
        if parsed.status in {"auto_corrected", "partially_invalid"}:
            corrections.append((original, parsed.normalized_refs))
    return ReferenceValueResult(
        valid=tuple(verified),
        invalid=tuple(invalid),
        corrections=tuple(corrections),
    )
