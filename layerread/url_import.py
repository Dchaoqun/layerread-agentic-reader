"""Safe URL fetching and adapter-based article extraction for LayerRead."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import ipaddress
import re
import socket
import ssl
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from layerread.article import MIN_ARTICLE_CHARS, count_content_characters, normalize_text


MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = "LayerRead/0.8-dev (+local article reader)"
WECHAT_HOSTS = {"mp.weixin.qq.com", "weixin.qq.com"}


class URLImportError(RuntimeError):
    """A user-facing URL validation, download, or extraction failure."""


@dataclass(frozen=True, slots=True)
class ExtractionAttempt:
    adapter: str
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class ArticleExtractionResult:
    source_url: str
    title: str = ""
    account: str = ""
    author: str = ""
    published_at: str = ""
    body: str = ""
    extraction_method: str = ""
    extraction_status: str = "failed"
    attempts: tuple[ExtractionAttempt, ...] = field(default_factory=tuple)

    @property
    def is_success(self) -> bool:
        return self.extraction_status == "extracted" and bool(self.body.strip())


class ArticleExtractionAdapter(Protocol):
    name: str

    def can_handle(self, url: str) -> bool: ...

    def extract(self, url: str, html: str) -> ArticleExtractionResult: ...


def _is_public_address(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def validate_public_url(url: str) -> str:
    """Validate an HTTP(S) URL and reject local/private destinations."""

    candidate = (url or "").strip()
    if not candidate:
        raise URLImportError("请先输入文章链接。")
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise URLImportError("链接格式无效，请输入完整的 http:// 或 https:// 地址。")
    if parsed.username is not None or parsed.password is not None:
        raise URLImportError("链接不得包含账号或密码。")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise URLImportError("链接端口格式无效。") from exc

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise URLImportError("出于安全原因，不能导入本机或内网地址。")

    if _is_public_address(hostname):
        addresses = {hostname}
    else:
        try:
            addresses = {
                entry[4][0]
                for entry in socket.getaddrinfo(
                    hostname,
                    port,
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise URLImportError("无法解析该链接的域名，请检查地址或网络连接。") from exc
    if not addresses or any(not _is_public_address(value) for value in addresses):
        raise URLImportError("出于安全原因，不能导入本机、内网或保留地址。")

    netloc = hostname
    if parsed.port is not None:
        netloc = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        safe_url = validate_public_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, safe_url)


def _network_failure_message(reason: object) -> str:
    """Map low-level connection failures to safe, actionable messages."""

    reason_text = str(reason).lower()
    if isinstance(reason, socket.gaierror) or "name resolution" in reason_text:
        return "无法解析目标站点域名；请检查 DNS、代理或 VPN 设置。"
    if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in reason_text:
        return "连接目标站点超时；该站点可能限制当前网络出口，请稍后重试或改用正文粘贴。"
    if isinstance(reason, ssl.SSLError) or "certificate" in reason_text or "ssl" in reason_text:
        return "与目标站点建立安全连接失败；请检查系统时间、代理或 VPN 的 TLS 设置。"
    if "proxy" in reason_text:
        return "代理连接失败；请检查 VPN/代理配置，或关闭代理后重试。"
    if "reset" in reason_text or "refused" in reason_text:
        return "目标站点拒绝或重置了连接；当前网络出口可能受到站点限制。"
    return "无法访问该网页；目标站点可能限制当前网络出口，请重试或改用正文粘贴。"


def fetch_html(url: str) -> tuple[str, str]:
    """Download bounded HTML while validating the initial and final destination."""

    safe_url = validate_public_url(url)
    request = Request(
        safe_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with build_opener(_SafeRedirectHandler()).open(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            final_url = validate_public_url(response.geturl())
            content_type = response.headers.get_content_type().lower()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise URLImportError("链接返回的不是 HTML 网页，无法提取正文。")
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > MAX_DOWNLOAD_BYTES:
                raise URLImportError("网页内容超过 5 MB 安全上限，未继续下载。")
            payload = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(payload) > MAX_DOWNLOAD_BYTES:
                raise URLImportError("网页内容超过 5 MB 安全上限，未继续下载。")
            charset = response.headers.get_content_charset() or "utf-8"
    except URLImportError:
        raise
    except HTTPError as exc:
        raise URLImportError(f"网页请求失败（HTTP {exc.code}）。") from exc
    except URLError as exc:
        raise URLImportError(_network_failure_message(exc.reason)) from exc
    except (OSError, ValueError) as exc:
        raise URLImportError("读取网页时发生错误，请稍后重试。") from exc

    try:
        html = payload.decode(charset, errors="replace")
    except LookupError:
        html = payload.decode("utf-8", errors="replace")
    return final_url, html


_BLOCK_TAGS = {
    "article",
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "p",
    "section",
}
_SKIP_TAGS = {"aside", "canvas", "footer", "form", "nav", "noscript", "script", "style", "svg"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _ReadableHTMLParser(HTMLParser):
    """Collect metadata and text candidates without executing page content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.buffers = {"wechat": [], "article": [], "body": []}
        self.named_buffers: dict[str, list[str]] = {
            "activity-name": [],
            "js_name": [],
            "publish_time": [],
            "author": [],
        }
        self.stack: list[tuple[bool, bool, bool, bool, tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): (value or "") for key, value in attrs if key}
        if tag == "meta":
            key = (
                values.get("property")
                or values.get("name")
                or values.get("itemprop")
                or ""
            ).strip().lower()
            content = values.get("content", "").strip()
            if key and content:
                self.meta.setdefault(key, content)

        if tag in _VOID_TAGS:
            if tag in _BLOCK_TAGS and self.stack and not self.stack[-1][0]:
                self._append_boundary(self.stack[-1])
            return

        inherited = self.stack[-1] if self.stack else (False, False, False, False, ())
        skip = inherited[0] or tag in _SKIP_TAGS
        body = inherited[1] or tag == "body"
        classes = set(values.get("class", "").split())
        element_id = values.get("id", "")
        article = inherited[2] or tag in {"article", "main"} or values.get("role") == "main" or bool(
            classes & {"article", "article-content", "article_content", "content", "entry-content", "post-content", "rich_media_content"}
        )
        wechat = inherited[3] or element_id == "js_content"
        captures = list(inherited[4])
        if element_id in self.named_buffers:
            captures.append(element_id)
        if "rich_media_meta_text" in classes and "author" not in captures:
            captures.append("author")
        state = (skip, body, article, wechat, tuple(captures))
        self.stack.append(state)
        if tag in _BLOCK_TAGS and not skip:
            self._append_boundary(state)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            return
        state = self.stack.pop()
        if tag in _BLOCK_TAGS and not state[0]:
            self._append_boundary(state)

    def handle_data(self, data: str) -> None:
        if not self.stack:
            return
        skip, body, article, wechat, captures = self.stack[-1]
        if skip:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if body:
            self.buffers["body"].append(text)
        if article:
            self.buffers["article"].append(text)
        if wechat:
            self.buffers["wechat"].append(text)
        for key in captures:
            self.named_buffers[key].append(text)

    def _append_boundary(self, state: tuple[bool, bool, bool, bool, tuple[str, ...]]) -> None:
        _skip, body, article, wechat, captures = state
        if body:
            self.buffers["body"].append("\n\n")
        if article:
            self.buffers["article"].append("\n\n")
        if wechat:
            self.buffers["wechat"].append("\n\n")
        for key in captures:
            self.named_buffers[key].append("\n\n")


def _parse_html(html: str) -> _ReadableHTMLParser:
    if not isinstance(html, str):
        raise URLImportError("网页返回内容不是可解析的 HTML 文本。")
    parser = _ReadableHTMLParser()
    try:
        # Capture <title> with a robust regex; the parser still handles visible text.
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title_text = re.sub(r"<[^>]+>", " ", title_match.group(1))
            parser.title_parts.append(re.sub(r"\s+", " ", title_text).strip())
        parser.feed(html)
        parser.close()
    except (AssertionError, AttributeError, TypeError, ValueError) as exc:
        raise URLImportError("网页 HTML 结构异常，无法安全解析正文。") from exc
    return parser


def _clean_candidate(parts: list[str]) -> str:
    text = " ".join(parts)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return normalize_text(text)


def _first(values: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(values)).strip()


def _metadata(parser: _ReadableHTMLParser) -> tuple[str, str, str, str]:
    meta = parser.meta
    title = (
        meta.get("og:title")
        or meta.get("twitter:title")
        or meta.get("headline")
        or _first(parser.named_buffers["activity-name"])
        or _first(parser.title_parts)
    )
    account = (
        _first(parser.named_buffers["js_name"])
        or meta.get("article:publisher")
        or meta.get("og:site_name")
        or meta.get("publisher")
        or ""
    )
    author = (
        meta.get("author")
        or meta.get("article:author")
        or _first(parser.named_buffers["author"])
    )
    published_at = (
        meta.get("article:published_time")
        or meta.get("datepublished")
        or meta.get("publishdate")
        or _first(parser.named_buffers["publish_time"])
    )
    return title, account, author, published_at


class WeChatArticleAdapter:
    name = "wechat_official_account"

    def can_handle(self, url: str) -> bool:
        return (urlsplit(url).hostname or "").lower() in WECHAT_HOSTS

    def extract(self, url: str, html: str) -> ArticleExtractionResult:
        parser = _parse_html(html)
        body = _clean_candidate(parser.buffers["wechat"])
        if count_content_characters(body) < MIN_ARTICLE_CHARS:
            raise URLImportError("公众号页面未暴露足够正文，可能受到访问或反爬限制。")
        title, account, author, published_at = _metadata(parser)
        return ArticleExtractionResult(
            source_url=url,
            title=title,
            account=account,
            author=author,
            published_at=published_at,
            body=body,
            extraction_method=self.name,
            extraction_status="extracted",
        )


class GenericArticleAdapter:
    name = "generic_html"

    def can_handle(self, url: str) -> bool:
        return True

    def extract(self, url: str, html: str) -> ArticleExtractionResult:
        parser = _parse_html(html)
        article = _clean_candidate(parser.buffers["article"])
        body = article if count_content_characters(article) >= MIN_ARTICLE_CHARS else _clean_candidate(parser.buffers["body"])
        if count_content_characters(body) < MIN_ARTICLE_CHARS:
            raise URLImportError("通用网页抽取后正文过短，无法安全进入分析。")
        title, account, author, published_at = _metadata(parser)
        return ArticleExtractionResult(
            source_url=url,
            title=title,
            account=account,
            author=author,
            published_at=published_at,
            body=body,
            extraction_method=self.name,
            extraction_status="extracted",
        )


DEFAULT_ADAPTERS: tuple[ArticleExtractionAdapter, ...] = (
    WeChatArticleAdapter(),
    GenericArticleAdapter(),
)


def extract_article_from_url(
    url: str,
    *,
    fetcher=fetch_html,
    adapters: tuple[ArticleExtractionAdapter, ...] = DEFAULT_ADAPTERS,
) -> ArticleExtractionResult:
    """Fetch once, then try compatible adapters in priority order."""

    try:
        final_url, html = fetcher(url)
    except URLImportError as exc:
        return ArticleExtractionResult(
            source_url=(url or "").strip(),
            attempts=(ExtractionAttempt("download", False, str(exc)),),
        )

    attempts: list[ExtractionAttempt] = []
    for adapter in adapters:
        if not adapter.can_handle(final_url):
            continue
        try:
            result = adapter.extract(final_url, html)
        except URLImportError as exc:
            attempts.append(ExtractionAttempt(adapter.name, False, str(exc)))
            continue
        except (AssertionError, AttributeError, TypeError, ValueError):
            attempts.append(
                ExtractionAttempt(
                    adapter.name,
                    False,
                    "网页结构异常，当前抽取方式无法安全解析。",
                )
            )
            continue
        attempts.append(ExtractionAttempt(adapter.name, True, "正文提取成功。"))
        return ArticleExtractionResult(
            source_url=result.source_url,
            title=result.title,
            account=result.account,
            author=result.author,
            published_at=result.published_at,
            body=result.body,
            extraction_method=result.extraction_method,
            extraction_status="extracted",
            attempts=tuple(attempts),
        )
    return ArticleExtractionResult(source_url=final_url, attempts=tuple(attempts))
