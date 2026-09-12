"""Validation for articles delivered by the LayerRead Chrome connector."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from io import BytesIO
import re
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from layerread.article import (
    MIN_ARTICLE_CHARS,
    ArticleImage,
    count_content_characters,
    normalize_text,
)


CONNECTOR_EXTRACTION_METHOD = "layerread_chrome_connector"
LEGACY_CONNECTOR_EXTRACTION_METHOD = "deepread_chrome_connector"
CONNECTOR_EXTRACTION_METHODS = {
    CONNECTOR_EXTRACTION_METHOD,
    LEGACY_CONNECTOR_EXTRACTION_METHOD,
}


def is_connector_extraction_method(value: str) -> bool:
    """Accept new and legacy records during the brand migration."""

    return value in CONNECTOR_EXTRACTION_METHODS


@dataclass(frozen=True, slots=True)
class ConnectorArticleImportResult:
    """Validated article data delivered by the local Chrome connector."""

    filename: str
    title: str = ""
    account: str = ""
    author: str = ""
    published_at: str = ""
    source_url: str = ""
    body: str = ""
    extraction_method: str = CONNECTOR_EXTRACTION_METHOD
    media_detected: bool = False
    image_reference_count: int = 0
    images: tuple[ArticleImage, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_success(self) -> bool:
        return count_content_characters(self.body) >= MIN_ARTICLE_CHARS


CONNECTOR_PROTOCOL_VERSIONS = {1, 2}
MAX_CONNECTOR_BODY_BYTES = 5 * 1024 * 1024
MAX_CONNECTOR_IMAGES = 10_000
MAX_IMPORTED_IMAGE_ITEMS = 24
MAX_IMPORTED_IMAGE_BYTES = 1_500_000
MAX_IMPORTED_IMAGE_TOTAL_BYTES = 7 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
_IMPORT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_IMAGE_ID_PATTERN = re.compile(r"^[A-Z0-9_-]{1,32}$")
_DATA_URL_PATTERN = re.compile(
    r"^data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)$",
    re.IGNORECASE,
)
_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class ConnectorImportError(ValueError):
    """An invalid or unsafe connector payload."""


def _string_field(
    article: dict[str, object],
    field: str,
    *,
    max_length: int,
) -> str:
    value = article.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConnectorImportError(f"扩展返回的 {field} 字段格式不正确。")
    cleaned = value.strip()
    if len(cleaned) > max_length:
        raise ConnectorImportError(f"扩展返回的 {field} 字段过长。")
    return cleaned


def _validate_source_url(value: str) -> str:
    if not value:
        raise ConnectorImportError("扩展没有返回有效的微信公众号原文链接。")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != "mp.weixin.qq.com":
        raise ConnectorImportError("一键导入只接受 mp.weixin.qq.com 的文章页面。")
    if parsed.username or parsed.password:
        raise ConnectorImportError("原文链接不得包含身份信息。")
    return value


def _safe_image_source_url(value: object) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) > 2_048:
        raise ConnectorImportError("扩展返回的图片来源链接格式不正确。")
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or (parsed.hostname or "").lower()
        not in {"mmbiz.qpic.cn", "mmbiz.qlogo.cn", "mp.weixin.qq.com"}
    ):
        raise ConnectorImportError("扩展返回了不受信任的图片来源链接。")
    return value.strip()


def _positive_int(value: object, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ConnectorImportError(f"扩展返回的图片 {field} 格式不正确。")
    return value


def _decode_image_data_url(data_url: object) -> tuple[str, bytes, int, int]:
    if not isinstance(data_url, str):
        raise ConnectorImportError("扩展返回的图片内容格式不正确。")
    match = _DATA_URL_PATTERN.fullmatch(data_url)
    if match is None:
        raise ConnectorImportError("扩展只允许传入 JPEG、PNG 或 WebP 图片。")
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ConnectorImportError("扩展返回的图片 Base64 内容损坏。") from exc
    if not payload or len(payload) > MAX_IMPORTED_IMAGE_BYTES:
        raise ConnectorImportError("单张图片超过 1.5 MB 安全上限。")

    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            detected_mime = _FORMAT_TO_MIME.get((image.format or "").upper())
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ConnectorImportError("扩展返回的图片无法安全解码。") from exc
    if detected_mime is None or detected_mime != match.group(1).lower():
        raise ConnectorImportError("图片声明的类型与实际内容不一致。")
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ConnectorImportError("图片像素尺寸超出安全范围。")
    return detected_mime, payload, width, height


def _parse_images(article: dict[str, object], protocol_version: int) -> tuple[ArticleImage, ...]:
    raw_images = article.get("images", [])
    if protocol_version == 1 and raw_images in (None, []):
        return ()
    if not isinstance(raw_images, list) or len(raw_images) > MAX_IMPORTED_IMAGE_ITEMS:
        raise ConnectorImportError("扩展返回的图片项目数量超出安全范围。")

    images: list[ArticleImage] = []
    seen_ids: set[str] = set()
    total_bytes = 0
    for raw in raw_images:
        if not isinstance(raw, dict):
            raise ConnectorImportError("扩展返回的图片项目格式不正确。")
        image_id = _string_field(raw, "image_id", max_length=32).upper()
        if not _IMAGE_ID_PATTERN.fullmatch(image_id) or image_id in seen_ids:
            raise ConnectorImportError("扩展返回的图片标识无效或重复。")
        seen_ids.add(image_id)

        mime_type, payload, width, height = _decode_image_data_url(raw.get("data_url"))
        declared_mime = _string_field(raw, "mime_type", max_length=50).lower()
        if declared_mime and declared_mime != mime_type:
            raise ConnectorImportError("图片声明的类型与实际内容不一致。")
        total_bytes += len(payload)
        if total_bytes > MAX_IMPORTED_IMAGE_TOTAL_BYTES:
            raise ConnectorImportError("文章图片总量超过 7 MB 安全上限。")

        stated_width = _positive_int(raw.get("width", width), "宽度", maximum=20_000)
        stated_height = _positive_int(raw.get("height", height), "高度", maximum=20_000)
        if (stated_width, stated_height) != (width, height):
            raise ConnectorImportError("图片声明的尺寸与实际内容不一致。")
        tile_index = _positive_int(raw.get("tile_index", 1), "分片序号", maximum=100)
        tile_count = _positive_int(raw.get("tile_count", 1), "分片总数", maximum=100)
        if tile_index > tile_count:
            raise ConnectorImportError("图片分片序号超出分片总数。")

        images.append(
            ArticleImage(
                image_id=image_id,
                data_url=str(raw["data_url"]),
                mime_type=mime_type,
                width=width,
                height=height,
                alt_text=_string_field(raw, "alt_text", max_length=500),
                caption=_string_field(raw, "caption", max_length=500),
                context_before=_string_field(raw, "context_before", max_length=800),
                context_after=_string_field(raw, "context_after", max_length=800),
                source_url=_safe_image_source_url(raw.get("source_url")),
                tile_index=tile_index,
                tile_count=tile_count,
                original_width=_positive_int(
                    raw.get("original_width", width),
                    "原始宽度",
                    maximum=40_000,
                ),
                original_height=_positive_int(
                    raw.get("original_height", height),
                    "原始高度",
                    maximum=40_000,
                ),
            )
        )
    return tuple(images)


def parse_connector_payload(payload: object) -> ConnectorArticleImportResult:
    """Validate a one-time connector envelope and return a local import result."""

    if not isinstance(payload, dict):
        raise ConnectorImportError("扩展返回的数据不是可识别的文章格式。")
    protocol_version = payload.get("protocol_version")
    if protocol_version not in CONNECTOR_PROTOCOL_VERSIONS:
        raise ConnectorImportError("扩展协议版本不兼容，请更新 LayerRead Connector。")

    import_id = payload.get("import_id")
    if not isinstance(import_id, str) or not _IMPORT_ID_PATTERN.fullmatch(import_id):
        raise ConnectorImportError("扩展返回的一次性导入标识无效。")

    article = payload.get("article")
    if not isinstance(article, dict):
        raise ConnectorImportError("扩展返回的数据中缺少文章正文。")

    body_value = article.get("body")
    if not isinstance(body_value, str):
        raise ConnectorImportError("扩展返回的正文格式不正确。")
    if len(body_value.encode("utf-8")) > MAX_CONNECTOR_BODY_BYTES:
        raise ConnectorImportError("扩展返回的文章正文超过 5 MB 安全上限。")
    body = normalize_text(body_value)
    if count_content_characters(body) < MIN_ARTICLE_CHARS:
        raise ConnectorImportError("扩展返回的正文过短，无法安全进入分析。")

    image_count = article.get("image_count", 0)
    if isinstance(image_count, bool) or not isinstance(image_count, int):
        raise ConnectorImportError("扩展返回的图片数量格式不正确。")
    if not 0 <= image_count <= MAX_CONNECTOR_IMAGES:
        raise ConnectorImportError("扩展返回的图片数量超出安全范围。")
    images = _parse_images(article, int(protocol_version))
    source_url = _validate_source_url(_string_field(article, "source_url", max_length=2_048))
    imported_source_count = len(
        {re.sub(r"-T\d+$", "", image.image_id) for image in images}
    )
    if images and imported_source_count >= image_count:
        note = f"已接收 {len(images)} 个高清图片分片，将在支持视觉的模型中结合正文分析。"
    elif images:
        note = f"检测到 {image_count} 张正文图片，已安全接收 {len(images)} 个重要图片或分片。"
    elif image_count:
        note = "检测到正文图片，但没有可安全传入模型的图片；分析时会降低相关结论置信度。"
    else:
        note = ""

    return ConnectorArticleImportResult(
        filename="Chrome 一键导入",
        title=_string_field(article, "title", max_length=500),
        account=_string_field(article, "account", max_length=300),
        author=_string_field(article, "author", max_length=300),
        published_at=_string_field(article, "published_at", max_length=100),
        source_url=source_url,
        body=body,
        extraction_method=CONNECTOR_EXTRACTION_METHOD,
        media_detected=bool(image_count),
        image_reference_count=image_count,
        images=images,
        notes=(note,) if note else (),
    )
