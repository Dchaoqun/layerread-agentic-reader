"""Pure article ingestion and cleaning logic for LayerRead v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import re
from typing import Literal


MIN_ARTICLE_CHARS = 80

_INVISIBLE_CHARACTERS = str.maketrans(
    {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)

_LIST_MARKER = re.compile(r"^\s*(?:[-*•·]|\d+[.、]|[一二三四五六七八九十]+[、.])\s*")

_OPERATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:喜欢|觉得).{0,20}(?:点赞|在看|转发|分享).{0,20}$",
        r"^(?:欢迎|记得|别忘了).{0,30}(?:关注|点赞|转发|分享|在看).{0,20}$",
        r"^(?:长按|扫描|识别).{0,20}(?:二维码|关注公众号).{0,20}$",
        r"^(?:点个|点击).{0,10}(?:在看|关注|点赞).{0,10}$",
        r"^(?:点赞|转发|分享|在看|关注)(?:一下)?[。！!~～]*$",
        r"^推荐阅读[：:]?[。！!]*$",
    )
)


class ArticleValidationError(ValueError):
    """Raised when pasted content is not usable as an article."""


@dataclass(frozen=True, slots=True)
class ArticleImage:
    """A validated article image ready for a multimodal model request."""

    image_id: str
    data_url: str = field(repr=False)
    mime_type: str
    width: int
    height: int
    alt_text: str = ""
    caption: str = ""
    context_before: str = ""
    context_after: str = ""
    source_url: str = ""
    tile_index: int = 1
    tile_count: int = 1
    original_width: int = 0
    original_height: int = 0


@dataclass(frozen=True, slots=True)
class ArticleDraft:
    """Processed article content and the user's reading preferences."""

    raw_text: str
    clean_text: str
    numbered_paragraphs: tuple[str, ...]
    removed_paragraphs: tuple[str, ...]
    reading_goal: str
    familiarity: str
    focus_area: str
    media_dependency: bool
    media_completeness: Literal["text_only", "images_partial", "images_included"]
    raw_char_count: int
    clean_char_count: int
    created_at: datetime
    source_title: str = ""
    source_url: str = ""
    account: str = ""
    author: str = ""
    published_at: str = ""
    extraction_method: str = "manual_paste"
    extraction_status: str = "confirmed"
    images: tuple[ArticleImage, ...] = ()
    image_count_detected: int = 0

    @property
    def paragraph_count(self) -> int:
        return len(self.numbered_paragraphs)

    @property
    def numbered_text(self) -> str:
        return "\n\n".join(self.numbered_paragraphs)


def upgrade_article_draft(article: object) -> ArticleDraft:
    """Upgrade an in-memory pre-v0.6 draft retained by Streamlit hot reload.

    SQLite deserialization already supplies defaults for older records. This
    helper covers the separate case where a browser session still holds an
    instance of the old slotted dataclass after the module has been reloaded.
    """

    v06_fields = (
        "source_title",
        "source_url",
        "account",
        "author",
        "published_at",
        "extraction_method",
        "extraction_status",
        "images",
        "image_count_detected",
    )
    if isinstance(article, ArticleDraft) and all(
        hasattr(article, name) for name in v06_fields
    ):
        return article

    try:
        return ArticleDraft(
            raw_text=getattr(article, "raw_text"),
            clean_text=getattr(article, "clean_text"),
            numbered_paragraphs=tuple(getattr(article, "numbered_paragraphs")),
            removed_paragraphs=tuple(getattr(article, "removed_paragraphs")),
            reading_goal=getattr(article, "reading_goal"),
            familiarity=getattr(article, "familiarity"),
            focus_area=getattr(article, "focus_area"),
            media_dependency=getattr(article, "media_dependency"),
            media_completeness=getattr(article, "media_completeness"),
            raw_char_count=getattr(article, "raw_char_count"),
            clean_char_count=getattr(article, "clean_char_count"),
            created_at=getattr(article, "created_at"),
            source_title=getattr(article, "source_title", ""),
            source_url=getattr(article, "source_url", ""),
            account=getattr(article, "account", ""),
            author=getattr(article, "author", ""),
            published_at=getattr(article, "published_at", ""),
            extraction_method=getattr(article, "extraction_method", "manual_paste"),
            extraction_status=getattr(article, "extraction_status", "confirmed"),
            images=tuple(getattr(article, "images", ())),
            image_count_detected=getattr(article, "image_count_detected", 0),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArticleValidationError("旧版会话中的文章状态无法升级，请开始一篇新文章。") from exc


def count_content_characters(text: str) -> int:
    """Count meaningful characters while ignoring whitespace."""

    return len(re.sub(r"\s+", "", text))


def normalize_text(text: str) -> str:
    """Normalize pasted whitespace without rewriting article wording."""

    normalized = text.translate(_INVISIBLE_CHARACTERS)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", normalized)
    return normalized.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split pasted content while preserving headings and list items."""

    if re.search(r"\n\s*\n", text):
        blocks = re.split(r"\n\s*\n", text)
    else:
        blocks = text.split("\n")

    paragraphs: list[str] = []
    for block in blocks:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            continue

        if len(lines) > 1 and any(_LIST_MARKER.match(line) for line in lines):
            paragraphs.extend(lines)
        else:
            paragraphs.append(" ".join(lines))
    return paragraphs


def is_operation_paragraph(paragraph: str) -> bool:
    """Return whether a short paragraph is clearly a social-operation CTA."""

    if len(paragraph) > 80:
        return False
    return any(pattern.fullmatch(paragraph.strip()) for pattern in _OPERATION_PATTERNS)


def clean_paragraphs(paragraphs: list[str]) -> tuple[list[str], list[str]]:
    """Remove exact duplicates and high-confidence operation paragraphs."""

    cleaned: list[str] = []
    removed: list[str] = []
    seen: set[str] = set()

    for paragraph in paragraphs:
        duplicate_key = re.sub(r"\s+", " ", paragraph).strip()
        if duplicate_key in seen or is_operation_paragraph(paragraph):
            removed.append(paragraph)
            continue

        seen.add(duplicate_key)
        cleaned.append(paragraph)

    return cleaned, removed


def number_paragraphs(paragraphs: list[str]) -> tuple[str, ...]:
    """Assign stable, sequential paragraph identifiers."""

    return tuple(
        f"[P{index:02d}] {paragraph}"
        for index, paragraph in enumerate(paragraphs, start=1)
    )


def process_article(
    raw_text: str,
    reading_goal: str = "",
    familiarity: str = "入门",
    focus_area: str = "通用",
    media_dependency: bool = False,
    source_title: str = "",
    source_url: str = "",
    account: str = "",
    author: str = "",
    published_at: str = "",
    extraction_method: str = "manual_paste",
    extraction_status: str = "confirmed",
    images: tuple[ArticleImage, ...] = (),
    image_count_detected: int = 0,
) -> ArticleDraft:
    """Validate, clean, number, and package a pasted article."""

    normalized = normalize_text(raw_text or "")
    raw_char_count = count_content_characters(normalized)

    if not normalized:
        raise ArticleValidationError("请先粘贴文章正文，再点击“处理文章”。")
    if raw_char_count < MIN_ARTICLE_CHARS:
        raise ArticleValidationError(
            f"正文过短，目前只有 {raw_char_count} 个有效字符；请粘贴至少 "
            f"{MIN_ARTICLE_CHARS} 个有效字符的文章内容。"
        )

    paragraphs = split_paragraphs(normalized)
    cleaned, removed = clean_paragraphs(paragraphs)
    if not cleaned:
        raise ArticleValidationError("清洗后没有可用正文，请检查粘贴的内容。")

    clean_text = "\n\n".join(cleaned)
    imported_source_count = len(
        {re.sub(r"-T\d+$", "", image.image_id) for image in images}
    )
    detected_count = max(image_count_detected, imported_source_count)
    if images and imported_source_count >= detected_count:
        media_completeness = "images_included"
    elif images:
        media_completeness = "images_partial"
    else:
        media_completeness = "text_only"

    return ArticleDraft(
        raw_text=raw_text,
        clean_text=clean_text,
        numbered_paragraphs=number_paragraphs(cleaned),
        removed_paragraphs=tuple(removed),
        reading_goal=reading_goal.strip(),
        familiarity=familiarity,
        focus_area=focus_area,
        media_dependency=media_dependency,
        media_completeness=media_completeness,
        raw_char_count=raw_char_count,
        clean_char_count=count_content_characters(clean_text),
        created_at=datetime.now(UTC),
        source_title=source_title.strip(),
        source_url=source_url.strip(),
        account=account.strip(),
        author=author.strip(),
        published_at=published_at.strip(),
        extraction_method=extraction_method.strip() or "manual_paste",
        extraction_status=extraction_status.strip() or "confirmed",
        images=tuple(images),
        image_count_detected=detected_count,
    )
