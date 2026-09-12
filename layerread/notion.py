"""Notion REST client and unified-export to Block conversion."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import os
import re
from typing import Any, Literal
from urllib.parse import urlsplit
from urllib.request import getproxies

import httpx

from layerread.article import ArticleImage
from layerread.export import ExportBlock, ExportDocument


NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"
MAX_RICH_TEXT_CHARS = 2000
MAX_BLOCKS_PER_REQUEST = 100
MAX_SINGLE_PART_UPLOAD_BYTES = 20 * 1024 * 1024
_IMAGE_DATA_URL_PATTERN = re.compile(
    r"^data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)$",
    re.IGNORECASE,
)
_IMAGE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


class NotionError(RuntimeError):
    """A secret-safe, user-facing Notion integration failure."""


class NotionPartialExportError(NotionError):
    """Raised when a page exists but one of the append batches failed."""

    def __init__(self, message: str, page_id: str, page_url: str, appended_blocks: int) -> None:
        super().__init__(message)
        self.page_id = page_id
        self.page_url = page_url
        self.appended_blocks = appended_blocks


@dataclass(frozen=True, slots=True)
class NotionTarget:
    target_id: str
    target_type: Literal["page", "data_source"]
    display_name: str
    title_property: str = "title"


@dataclass(frozen=True, slots=True)
class NotionExportResult:
    page_id: str
    page_url: str
    page_title: str
    appended_blocks: int
    uploaded_images: int = 0


@dataclass(frozen=True, slots=True)
class _UploadedArticleImage:
    image_id: str
    file_upload_id: str
    caption: str


@dataclass(frozen=True, slots=True)
class _NotionRoute:
    """One backend-only route to the Notion API."""

    proxy_url: str = ""


def normalize_notion_id(value: str) -> str:
    """Accept a raw UUID or a copied Notion URL and return a hyphenated UUID."""

    candidate = (value or "").strip()
    if not candidate:
        raise NotionError("请填写目标 Page ID 或 Database ID。")
    if candidate.startswith(("http://", "https://")):
        candidate = urlsplit(candidate).path.rstrip("/").split("/")[-1]
    match = re.search(
        r"(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{32})$",
        candidate,
    )
    if not match:
        raise NotionError("目标 ID 格式无效；请粘贴 32 位 ID 或完整 Notion 链接。")
    compact = match.group(0).replace("-", "").lower()
    return f"{compact[:8]}-{compact[8:12]}-{compact[12:16]}-{compact[16:20]}-{compact[20:]}"


def _plain_text(rich_text: list[dict[str, Any]] | None) -> str:
    return "".join(item.get("plain_text", "") for item in rich_text or []).strip()


def _page_title(payload: dict[str, Any]) -> str:
    for value in payload.get("properties", {}).values():
        if value.get("type") == "title":
            return _plain_text(value.get("title")) or "未命名页面"
    return "未命名页面"


def _data_source_title(payload: dict[str, Any]) -> str:
    return _plain_text(payload.get("title")) or "未命名数据库"


def _normalize_proxy_url(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return candidate


def _environment_proxy_url() -> str:
    """Read an HTTP(S) proxy from environment or Windows system settings."""

    try:
        proxies = getproxies()
    except (OSError, ValueError):
        return ""
    for key in ("https", "all", "http"):
        proxy_url = _normalize_proxy_url(str(proxies.get(key, "")))
        if proxy_url:
            return proxy_url
    return ""


def _notion_routes(explicit_proxy_url: str = "") -> tuple[_NotionRoute, ...]:
    """Try an explicit proxy, the system proxy, and finally a direct route."""

    candidates = (
        _normalize_proxy_url(explicit_proxy_url),
        _environment_proxy_url(),
        "",
    )
    routes: list[_NotionRoute] = []
    seen: set[str] = set()
    for proxy_url in candidates:
        if proxy_url in seen:
            continue
        seen.add(proxy_url)
        routes.append(_NotionRoute(proxy_url))
    return tuple(routes)


def _create_http_client(route: _NotionRoute, timeout_seconds: int) -> httpx.Client:
    timeout = httpx.Timeout(timeout_seconds, connect=min(10, timeout_seconds))
    return httpx.Client(
        proxy=route.proxy_url or None,
        trust_env=False,
        timeout=timeout,
        headers={
            "User-Agent": "LayerRead/0.8-dev",
        },
    )


def _safe_notion_detail(value: Any, token: str) -> str:
    detail = str(value or "").strip()
    if not detail:
        return ""
    if token:
        detail = detail.replace(token, "[已隐藏]")
    detail = re.sub(r"\b(?:ntn_|secret_)[A-Za-z0-9_-]+", "[已隐藏]", detail)
    return detail[:300]


def _error_from_response(response: httpx.Response, token: str) -> NotionError:
    try:
        payload = response.json()
        message = _safe_notion_detail(payload.get("message", ""), token)
        code = _safe_notion_detail(payload.get("code", ""), token)
    except (ValueError, TypeError, AttributeError):
        message = ""
        code = ""
    hints = {
        400: "请求内容不符合 Notion 限制，请检查目标类型或页面属性。",
        401: "Integration Token 无效或已失效。",
        403: "Integration 缺少插入内容权限。",
        404: "找不到目标，或该页面/数据库尚未共享给 Integration。",
        409: "Notion 数据发生冲突，请稍后重试。",
        429: "Notion 请求过于频繁，请稍后重试。",
    }
    safe = hints.get(response.status_code, "Notion 服务暂时无法完成请求，请稍后重试。")
    if code and message and len(message) <= 300:
        safe = f"{safe}（{code}：{message}）"
    return NotionError(safe)


class NotionClient:
    """Secret-safe Notion client with proxy/direct connection fallback."""

    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: int = 20,
        proxy_url: str | None = None,
    ) -> None:
        secret = (token or "").strip()
        if not secret:
            raise NotionError("请填写 Notion Integration Token。")
        self._token = secret
        self._timeout_seconds = timeout_seconds
        configured_proxy = (
            os.environ.get("NOTION_PROXY_URL", "")
            if proxy_url is None
            else proxy_url
        )
        self._routes = _notion_routes(configured_proxy)
        self._active_route: _NotionRoute | None = None

    def _ordered_routes(self) -> tuple[_NotionRoute, ...]:
        if self._active_route is None:
            return self._routes
        return (
            self._active_route,
            *(route for route in self._routes if route != self._active_route),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        method = method.upper()
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }
        last_connection_error: httpx.RequestError | None = None
        for index, route in enumerate(self._ordered_routes()):
            try:
                with _create_http_client(route, self._timeout_seconds) as client:
                    response = client.request(
                        method,
                        f"{NOTION_API_BASE}{path}",
                        headers=headers,
                        json=payload,
                    )
                if response.is_error:
                    raise _error_from_response(response, self._token)
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Notion response is not an object")
                self._active_route = route
                return result
            except NotionError:
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError) as exc:
                last_connection_error = exc
                if index + 1 < len(self._ordered_routes()):
                    continue
                break
            except httpx.RequestError as exc:
                if method == "GET" and index + 1 < len(self._ordered_routes()):
                    last_connection_error = exc
                    continue
                if method in {"POST", "PATCH"}:
                    raise NotionError(
                        "Notion 写入请求的结果无法确认。请求可能已经到达 Notion；"
                        "请先检查目标页面，再决定是否重试，以免生成重复内容。"
                    ) from exc
                last_connection_error = exc
                break
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise NotionError("Notion 返回了无法读取的响应，请稍后重试。") from exc

        raise NotionError(
            "无法连接 Notion。系统已尝试可用代理和直连；"
            "请检查 VPN/代理、防火墙或稍后重试。"
        ) from last_connection_error

    def _request_multipart(
        self,
        path: str,
        *,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        """Send one file without retrying an ambiguous in-flight upload."""

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_API_VERSION,
        }
        last_connection_error: httpx.RequestError | None = None
        routes = self._ordered_routes()
        for index, route in enumerate(routes):
            try:
                with _create_http_client(route, self._timeout_seconds) as client:
                    response = client.post(
                        f"{NOTION_API_BASE}{path}",
                        headers=headers,
                        files={"file": (filename, content, content_type)},
                    )
                if response.is_error:
                    raise _error_from_response(response, self._token)
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Notion response is not an object")
                self._active_route = route
                return result
            except NotionError:
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError) as exc:
                last_connection_error = exc
                if index + 1 < len(routes):
                    continue
                break
            except httpx.RequestError as exc:
                raise NotionError(
                    "图片上传结果无法确认。临时文件可能已经到达 Notion，但尚未附加到页面；"
                    "可以稍后重新导出，未附加的临时文件会自动失效。"
                ) from exc
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise NotionError("Notion 返回了无法读取的图片上传响应，请稍后重试。") from exc

        raise NotionError(
            "无法上传图片到 Notion。系统已尝试可用代理和直连；"
            "请检查 VPN/代理、防火墙或稍后重试。"
        ) from last_connection_error

    def _upload_article_image(self, image: ArticleImage) -> _UploadedArticleImage:
        match = _IMAGE_DATA_URL_PATTERN.fullmatch(image.data_url)
        if match is None:
            raise NotionError(f"图片 {image.image_id} 不是受支持的 JPEG、PNG 或 WebP 格式。")
        content_type = match.group(1).lower()
        if content_type != image.mime_type.lower():
            raise NotionError(f"图片 {image.image_id} 的格式信息不一致，已停止导出。")
        try:
            content = base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise NotionError(f"图片 {image.image_id} 的内容损坏，无法导出到 Notion。") from exc
        if not content or len(content) > MAX_SINGLE_PART_UPLOAD_BYTES:
            raise NotionError(
                f"图片 {image.image_id} 超过 Notion 单文件直接上传的 20 MB 上限。"
            )

        extension = _IMAGE_EXTENSIONS[content_type]
        safe_id = re.sub(r"[^a-z0-9_-]", "-", image.image_id.lower()).strip("-")
        filename = f"layerread-{safe_id or 'article-image'}.{extension}"
        created = self._request(
            "POST",
            "/file_uploads",
            {
                "mode": "single_part",
                "filename": filename,
                "content_type": content_type,
            },
        )
        upload_id = str(created.get("id", "")).strip()
        if not upload_id:
            raise NotionError(f"Notion 没有返回图片 {image.image_id} 的上传 ID。")
        sent = self._request_multipart(
            f"/file_uploads/{upload_id}/send",
            filename=filename,
            content=content,
            content_type=content_type,
        )
        if sent.get("status") != "uploaded":
            raise NotionError(f"图片 {image.image_id} 未完成上传，请稍后重试。")

        caption = image.caption.strip() or image.alt_text.strip() or f"原文图片 {image.image_id}"
        if image.tile_count > 1:
            caption = f"{caption}（长图分片 {image.tile_index}/{image.tile_count}）"
        return _UploadedArticleImage(
            image_id=image.image_id,
            file_upload_id=upload_id,
            caption=caption,
        )

    def resolve_target(
        self,
        target_id: str,
        target_kind: Literal["page", "database"],
    ) -> NotionTarget:
        normalized = normalize_notion_id(target_id)
        if target_kind == "page":
            page = self._request("GET", f"/pages/{normalized}")
            return NotionTarget(
                target_id=normalized,
                target_type="page",
                display_name=_page_title(page),
            )

        try:
            database = self._request("GET", f"/databases/{normalized}")
        except NotionError as database_error:
            try:
                source = self._request("GET", f"/data_sources/{normalized}")
            except NotionError:
                raise database_error
        else:
            sources = database.get("data_sources") or []
            if not sources:
                raise NotionError("目标数据库没有可写入的数据源。")
            source = self._request("GET", f"/data_sources/{sources[0]['id']}")

        title_property = next(
            (
                name
                for name, value in source.get("properties", {}).items()
                if value.get("type") == "title"
            ),
            "",
        )
        if not title_property:
            raise NotionError("目标数据库没有标题属性，无法创建页面。")
        return NotionTarget(
            target_id=normalize_notion_id(source["id"]),
            target_type="data_source",
            display_name=_data_source_title(source),
            title_property=title_property,
        )

    def test_connection(
        self,
        target_id: str,
        target_kind: Literal["page", "database"],
    ) -> NotionTarget:
        return self.resolve_target(target_id, target_kind)

    def export_document(
        self,
        document: ExportDocument,
        target_id: str,
        target_kind: Literal["page", "database"],
        *,
        images: tuple[ArticleImage, ...] = (),
    ) -> NotionExportResult:
        target = self.resolve_target(target_id, target_kind)
        uploaded_images = tuple(self._upload_article_image(image) for image in images)
        title_value = _rich_text(document.title[:2000])
        if target.target_type == "page":
            parent = {"type": "page_id", "page_id": target.target_id}
            properties = {"title": {"title": title_value}}
        else:
            parent = {"type": "data_source_id", "data_source_id": target.target_id}
            properties = {target.title_property: {"title": title_value}}

        page = self._request(
            "POST",
            "/pages",
            {"parent": parent, "properties": properties},
        )
        page_id = page.get("id", "")
        if not page_id:
            raise NotionError("Notion 已响应，但没有返回新页面 ID。")

        blocks = document_to_notion_blocks(document, uploaded_images=uploaded_images)
        appended_blocks = 0
        try:
            for offset in range(0, len(blocks), MAX_BLOCKS_PER_REQUEST):
                batch = blocks[offset : offset + MAX_BLOCKS_PER_REQUEST]
                self._request(
                    "PATCH",
                    f"/blocks/{page_id}/children",
                    {"children": batch},
                )
                appended_blocks += len(batch)
        except NotionError as exc:
            raise NotionPartialExportError(
                "Notion 页面已经创建，但追加内容时中断。请检查该页面后再决定是否重试，避免重复页面。"
                f"已写入 {appended_blocks}/{len(blocks)} 个 Block。原错误：{exc}",
                page_id,
                page.get("url", ""),
                appended_blocks,
            ) from exc
        return NotionExportResult(
            page_id=page_id,
            page_url=page.get("url", ""),
            page_title=document.title,
            appended_blocks=appended_blocks,
            uploaded_images=len(uploaded_images),
        )


def split_rich_text(value: str, limit: int = MAX_RICH_TEXT_CHARS) -> list[str]:
    """Split text on sensible boundaries while respecting Notion's text limit."""

    if limit < 1:
        raise ValueError("limit must be positive")
    text = value or ""
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        search_window = remaining[:limit]
        boundary = max(
            search_window.rfind("\n"),
            search_window.rfind("。"),
            search_window.rfind(" "),
        )
        if boundary < limit // 2:
            boundary = limit
        else:
            boundary += 1
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining or not chunks:
        chunks.append(remaining)
    return chunks


def _rich_text(text: str, *, link: str | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for chunk in split_rich_text(text):
        text_payload: dict[str, Any] = {"content": chunk}
        if link:
            text_payload["link"] = {"url": link}
        items.append({"type": "text", "text": text_payload})
    return items


def _text_block(kind: str, text: str, *, language: str = "plain text") -> dict[str, Any]:
    payload: dict[str, Any] = {"rich_text": _rich_text(text)}
    if kind == "code":
        payload["language"] = language
    return {"object": "block", "type": kind, kind: payload}


def _convert_export_block(block: ExportBlock) -> list[dict[str, Any]]:
    if block.kind == "heading":
        heading_type = f"heading_{min(max(block.level, 1), 3)}"
        return [
            _text_block(heading_type, chunk)
            for chunk in split_rich_text(block.text)
        ]
    if block.kind == "list":
        notion_blocks: list[dict[str, Any]] = []
        for item in block.items:
            notion_blocks.extend(
                _text_block("bulleted_list_item", chunk)
                for chunk in split_rich_text(item)
            )
        return notion_blocks
    if block.kind == "quote":
        return [_text_block("quote", chunk) for chunk in split_rich_text(block.text)]
    if block.kind == "code":
        language = block.language if block.language in {"plain text", "markdown", "python", "json", "mermaid"} else "plain text"
        return [
            _text_block("code", chunk, language=language)
            for chunk in split_rich_text(block.text)
        ]
    return [_text_block("paragraph", chunk) for chunk in split_rich_text(block.text)]


def _uploaded_image_blocks(
    images: tuple[_UploadedArticleImage, ...],
) -> list[dict[str, Any]]:
    if not images:
        return []
    blocks: list[dict[str, Any]] = [_text_block("heading_1", "原文重要图片")]
    for image in images:
        blocks.append(
            {
                "object": "block",
                "type": "image",
                "image": {
                    "type": "file_upload",
                    "file_upload": {"id": image.file_upload_id},
                    "caption": _rich_text(image.caption),
                },
            }
        )
    return blocks


def document_to_notion_blocks(
    document: ExportDocument,
    *,
    uploaded_images: tuple[_UploadedArticleImage, ...] = (),
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if document.source_url:
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": _rich_text(
                        f"原文链接：{document.source_url}",
                        link=document.source_url,
                    )
                },
            }
        )
    blocks.extend(_uploaded_image_blocks(uploaded_images))
    for section in document.sections:
        blocks.append(_text_block("heading_1", section.title))
        for block in section.blocks:
            blocks.extend(_convert_export_block(block))
    return blocks
