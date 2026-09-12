"""Provider-neutral model client and v0.2 analysis workflow."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
import re
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.request import getproxies

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from pydantic import ValidationError

from layerread.analysis_schema import Analysis, AnalysisPayload
from layerread.article import ArticleDraft
from layerread.config import ModelSettings
from layerread.prompts import active_prompt_version, prompt_text
from layerread.references import normalize_references


PROMPT_VERSION = "v0.6.5-schema-compatibility"
ANALYSIS_MAX_OUTPUT_TOKENS = 32768
MODEL_REQUEST_TIMEOUT_SECONDS = 300
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ConnectionRoute:
    """One backend-only route to an OpenAI-compatible endpoint."""

    base_url: str
    proxy_url: str = ""


def _environment_proxy_url() -> str:
    """Read an HTTP(S) proxy from environment or Windows system settings."""

    try:
        proxies = getproxies()
    except (OSError, ValueError):
        return ""
    for key in ("https", "all", "http"):
        value = str(proxies.get(key, "")).strip()
        if not value:
            continue
        if "://" not in value:
            value = f"http://{value}"
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return value
    return ""


def _connection_routes(settings: ModelSettings) -> tuple[_ConnectionRoute, ...]:
    """Build explicit-proxy, environment-proxy, and direct fallback routes."""

    proxies = [settings.proxy_url, _environment_proxy_url(), ""]
    base_urls = (settings.base_url, *settings.fallback_base_urls)
    routes: list[_ConnectionRoute] = []
    seen: set[tuple[str, str]] = set()
    for base_url in base_urls:
        for proxy_url in proxies:
            key = (base_url.rstrip("/"), proxy_url)
            if key in seen:
                continue
            seen.add(key)
            routes.append(_ConnectionRoute(*key))
    return tuple(routes)


def _is_official_deepseek(settings: ModelSettings) -> bool:
    """Identify the official DeepSeek endpoint without relying on UI labels."""

    return (urlsplit(settings.base_url).hostname or "").lower() == "api.deepseek.com"


def _create_sdk_client(
    settings: ModelSettings,
    route: _ConnectionRoute,
    timeout_seconds: int,
) -> OpenAI:
    """Create the official OpenAI SDK client used by compatible providers."""

    timeout = httpx.Timeout(timeout_seconds, connect=min(12, timeout_seconds))
    http_client = httpx.Client(
        proxy=route.proxy_url or None,
        trust_env=False,
        timeout=timeout,
    )
    try:
        return OpenAI(
            api_key=settings.api_key,
            base_url=route.base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
    except Exception:
        http_client.close()
        raise


class AnalysisError(RuntimeError):
    """Safe, user-facing analysis failure."""


class AnalysisSchemaError(AnalysisError):
    """Structured output failure with a safe model-facing repair summary."""

    def __init__(self, repair_issue: str) -> None:
        super().__init__("模型输出未通过结构校验。")
        self.repair_issue = repair_issue


class AnalysisOutputTruncated(AnalysisError):
    """The provider stopped before completing the structured response."""

    def __init__(self, partial_response: str = "") -> None:
        super().__init__("模型输出因长度上限被截断。")
        self.partial_response = partial_response


def _public_schema_diagnostic(issue: str) -> str:
    """Expose only bounded schema paths/codes, never raw model or article text."""

    compact = " ".join(issue.split())
    return compact[:500]


def _http_error_message(exc: APIStatusError, settings: ModelSettings) -> str:
    """Build a provider-aware error without exposing credentials or raw bodies."""

    try:
        payload = exc.body if isinstance(exc.body, dict) else exc.response.json()
        error = payload.get("error", payload)
        error_code = str(error.get("code", "")).strip() if isinstance(error, dict) else ""
        request_id = str(
            payload.get("request_id", "") or getattr(exc, "request_id", "")
        ).strip()
    except (ValueError, TypeError, AttributeError):
        error_code = ""
        request_id = str(getattr(exc, "request_id", "") or "").strip()

    status_code = exc.status_code
    if status_code == 401 and settings.provider_id == "aliyun":
        if settings.api_key.startswith("sk-sp-"):
            message = (
                "阿里云专属套餐认证失败（HTTP 401）。"
                "sk-sp- Key 仅限交互式 AI 编程/Agent 工具，不可用于应用后端；"
                "请改用百炼按量付费的标准 API Key 和 API Host。"
            )
        else:
            message = (
                "阿里云认证失败（HTTP 401）。请确认 API Key 完整有效，"
                "并且与创建 Key 时显示的地域和 Base URL 一致。"
            )
    elif (
        status_code == 403
        and settings.provider_id == "aliyun"
        and error_code == "AllocationQuota.FreeTierOnly"
    ):
        message = (
            "阿里云百炼免费额度已用完（HTTP 403）。"
            "供应商侧硬停损已阻止继续计费，当前模型调用已暂停。"
        )
    elif status_code == 403:
        message = "模型服务拒绝访问（HTTP 403）。请检查 API Key 的模型权限和 IP 白名单。"
    elif status_code == 429:
        message = "模型服务当前限流或额度不足（HTTP 429），请稍后重试或检查账户额度。"
    elif status_code >= 500:
        message = f"模型服务暂时异常（HTTP {status_code}），请稍后重试。"
    else:
        message = (
            f"模型服务返回 HTTP {status_code}。"
            "请检查 API 配置、余额和模型名称后重试。"
        )
    if error_code and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", error_code):
        message += f" 错误码：{error_code}。"
    if request_id and re.fullmatch(r"[A-Za-z0-9_-]{1,120}", request_id):
        message += f" Request ID：{request_id}。"
    return message


def _is_connect_stage_error(exc: APIConnectionError) -> bool:
    """Only retry failures that happen before a request can be processed."""

    cause = exc.__cause__
    return isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError))


def _connection_error_message(settings: ModelSettings) -> str:
    if settings.provider_id == "aliyun":
        return (
            "无法连接阿里云百炼。系统已尝试可用代理、直连和同地域兼容端点，"
            "但仍未完成 HTTPS 连接；请检查 VPN/代理模式、网络防火墙或阿里云网络链路。"
        )
    return "无法连接模型服务或请求超时，请检查网络后重试。"


def _read_timeout_message(settings: ModelSettings, timeout_seconds: int) -> str:
    """Explain a slow response without misreporting it as a TLS failure."""

    if settings.provider_id == "aliyun":
        return (
            f"请求已经发送到阿里云百炼，但等待模型生成超过 {timeout_seconds} 秒。"
            "这通常是长文章、图片较多或模型生成较慢，并不等同于 HTTPS 连接失败。"
            "本次请求可能已在供应商侧产生用量，请先检查百炼调用记录，"
            "不要立即重复提交；可先用无图片短文本验证连接。"
        )
    return (
        f"请求已经发送到模型服务，但等待生成超过 {timeout_seconds} 秒。"
        "本次请求可能已产生用量，请先检查供应商调用记录，不要立即重复提交。"
    )


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, Any]]) -> str: ...


class OpenAICompatibleClient:
    """OpenAI SDK client shared by DeepSeek and Alibaba Cloud Model Studio."""

    def __init__(
        self,
        settings: ModelSettings,
        timeout_seconds: int = MODEL_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings
        self._timeout_seconds = timeout_seconds

    def complete(self, messages: list[dict[str, Any]]) -> str:
        routes = _connection_routes(self._settings)
        completion = None
        last_connection_error: APIConnectionError | None = None
        for index, route in enumerate(routes):
            try:
                with _create_sdk_client(
                    self._settings,
                    route,
                    self._timeout_seconds,
                ) as client:
                    request_options: dict[str, Any] = {
                        "max_tokens": ANALYSIS_MAX_OUTPUT_TOKENS,
                    }
                    if self._settings.provider_id == "aliyun":
                        # Current Qwen models prefer max_completion_tokens. The
                        # previous fixed 8K max_tokens truncated image-heavy
                        # structured analyses even though the model had ample
                        # context and output capacity.
                        request_options = {
                            "max_completion_tokens": ANALYSIS_MAX_OUTPUT_TOKENS,
                            "extra_body": {"enable_thinking": False},
                        }
                        has_images = any(
                            isinstance(message.get("content"), list)
                            and any(
                                isinstance(item, dict)
                                and item.get("type") == "image_url"
                                for item in message["content"]
                            )
                            for message in messages
                        )
                        if has_images:
                            request_options["extra_body"][
                                "vl_high_resolution_images"
                            ] = True
                    elif _is_official_deepseek(self._settings):
                        # DeepSeek's official OpenAI-compatible API uses
                        # max_tokens and enables thinking by default on current
                        # V4 models. JSON analysis is more reliable and much
                        # shorter in non-thinking mode.
                        request_options["extra_body"] = {
                            "thinking": {"type": "disabled"}
                        }
                    completion = client.chat.completions.create(
                        model=self._settings.model,
                        messages=messages,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                        **request_options,
                    )
                if index:
                    logger.info(
                        "Model request succeeded through fallback route %s for provider %s",
                        index + 1,
                        self._settings.provider_id,
                    )
                break
            except APIStatusError as exc:
                raise AnalysisError(_http_error_message(exc, self._settings)) from None
            except APITimeoutError as exc:
                last_connection_error = exc
                if not _is_connect_stage_error(exc):
                    raise AnalysisError(
                        _read_timeout_message(
                            self._settings,
                            self._timeout_seconds,
                        )
                    ) from None
            except APIConnectionError as exc:
                last_connection_error = exc
                if not _is_connect_stage_error(exc):
                    raise AnalysisError(_connection_error_message(self._settings)) from None
            except (ValueError, OSError) as exc:
                logger.warning(
                    "Skipping unusable model connection route for provider %s: %s",
                    self._settings.provider_id,
                    type(exc).__name__,
                )

        if completion is None:
            if last_connection_error is not None:
                logger.warning(
                    "All model connection routes failed for provider %s: %s",
                    self._settings.provider_id,
                    type(last_connection_error.__cause__).__name__,
                )
            raise AnalysisError(_connection_error_message(self._settings))

        try:
            choice = completion.choices[0]
            usage = getattr(completion, "usage", None)
            logger.info(
                "Model completion finished for provider %s: finish_reason=%s, output_tokens=%s",
                self._settings.provider_id,
                choice.finish_reason,
                getattr(usage, "completion_tokens", None),
            )
            if choice.finish_reason == "length":
                partial_content = choice.message.content
                raise AnalysisOutputTruncated(
                    partial_content if isinstance(partial_content, str) else ""
                )
            content = choice.message.content
        except (AttributeError, IndexError, TypeError):
            raise AnalysisError("模型服务响应缺少正文内容，请检查接口兼容性。") from None
        if not isinstance(content, str) or not content.strip():
            raise AnalysisError("模型没有返回分析内容，请重新分析。")
        return content


def article_fingerprint(
    article: ArticleDraft,
    model_name: str,
    base_url: str = "",
    supports_vision: bool = False,
) -> str:
    """Build a stable ID for duplicate-call protection and provenance."""

    source = "\n".join(
        [
            active_prompt_version(PROMPT_VERSION),
            base_url.rstrip("/").lower(),
            model_name,
            str(supports_vision),
            article.numbered_text,
            article.reading_goal,
            article.familiarity,
            article.focus_area,
            str(article.media_dependency),
            *(hashlib.sha256(image.data_url.encode("ascii")).hexdigest() for image in article.images),
        ]
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _analysis_schema_with_allowed_references(article: ArticleDraft) -> dict[str, Any]:
    """Constrain every paragraph_refs item to IDs that exist in this article."""

    schema = AnalysisPayload.model_json_schema()
    allowed_references = [
        paragraph.partition(" ")[0]
        for paragraph in article.numbered_paragraphs
    ]

    def constrain(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "paragraph_refs" and isinstance(child, dict):
                    child["items"] = {
                        "type": "string",
                        "enum": allowed_references,
                    }
                else:
                    constrain(child)
        elif isinstance(value, list):
            for child in value:
                constrain(child)

    constrain(schema)
    return schema


def build_analysis_messages(
    article: ArticleDraft,
    *,
    include_images: bool = True,
) -> list[dict[str, Any]]:
    schema = _analysis_schema_with_allowed_references(article)
    allowed_references = [
        paragraph.partition(" ")[0]
        for paragraph in article.numbered_paragraphs
    ]
    preference = (
        f"阅读目的：{article.reading_goal or '未填写'}；"
        f"主题熟悉度：{article.familiarity}；关注方向：{article.focus_area}。"
    )
    if article.images and include_images:
        media_note = (
            f"系统已附上 {len(article.images)} 个正文图片或长图分片。"
            "这些图片可能是 PPT 页面、信息卡片、流程图、思维导图、图表或图文混排内容。"
            "必须结合图片与其 IMG 标识附近的正文分析；先识别图片中的标题、正文、结构和视觉关系，"
            "再形成结论。无法可靠读取的文字或数字必须明确视为不确定，严禁猜测。"
        )
    elif article.media_dependency:
        media_note = "文章依赖未能传入当前模型的图片、图表或视频，相关判断必须降低置信度。"
    else:
        media_note = "用户未标记关键媒体依赖。"

    user_text = (
        f"用户偏好：{preference}\n{media_note}\n\n"
        f"合法段落编号仅限：{', '.join(allowed_references)}\n\n"
        "请严格按以下 JSON Schema 返回，不得添加字段：\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"编号正文：\n{article.numbered_text}"
    )
    user_content: str | list[dict[str, Any]] = user_text
    if article.images and include_images:
        multimodal_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image in article.images:
            tile_note = (
                f"，长图分片 {image.tile_index}/{image.tile_count}"
                if image.tile_count > 1
                else ""
            )
            context = "；".join(
                value
                for value in (
                    f"说明：{image.caption}" if image.caption else "",
                    f"替代文本：{image.alt_text}" if image.alt_text else "",
                    f"上文：{image.context_before}" if image.context_before else "",
                    f"下文：{image.context_after}" if image.context_after else "",
                )
                if value
            )
            multimodal_content.append(
                {
                    "type": "text",
                    "text": f"图片 {image.image_id}{tile_note}" + (f"。{context}" if context else "。"),
                }
            )
            multimodal_content.append(
                {"type": "image_url", "image_url": {"url": image.data_url}}
            )
        user_content = multimodal_content

    return [
        {
            "role": "system",
            "content": prompt_text("analysis"),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _extract_json(raw_response: str) -> Any:
    text = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
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


def _parse_analysis_payload(
    raw_response: str,
    *,
    apply_known_compatibility: bool,
) -> tuple[AnalysisPayload, tuple[str, ...]]:
    try:
        decoded = _extract_json(raw_response)
    except json.JSONDecodeError as exc:
        issue = f"JSON 语法错误（第 {exc.lineno} 行，第 {exc.colno} 列）"
        raise AnalysisSchemaError(issue) from exc
    corrections: tuple[str, ...] = ()
    if apply_known_compatibility:
        decoded, corrections = _normalize_known_schema_variants(decoded)
    try:
        return AnalysisPayload.model_validate(decoded), corrections
    except ValidationError as exc:
        raise AnalysisSchemaError(_safe_schema_issue(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise AnalysisSchemaError("JSON 顶层结构或字段值类型无效") from exc


def parse_analysis(raw_response: str) -> AnalysisPayload:
    """Parse JSON and strictly enforce the documented analysis schema."""

    payload, _ = _parse_analysis_payload(
        raw_response,
        apply_known_compatibility=False,
    )
    return payload


def _normalize_known_schema_variants(value: Any) -> tuple[Any, tuple[str, ...]]:
    """Normalize only lossless, explicitly documented provider variants."""

    if not isinstance(value, dict):
        return value, ()
    normalized = deepcopy(value)
    corrections: list[str] = []
    reading_decision = normalized.get("reading_decision")
    if isinstance(reading_decision, dict) and "paragraph_refs" in reading_decision:
        reading_decision.pop("paragraph_refs")
        corrections.append("removed redundant reading_decision.paragraph_refs")
    fluff_assessment = normalized.get("fluff_assessment")
    if isinstance(fluff_assessment, dict):
        title_overstates = fluff_assessment.get("title_overstates_content")
        if type(title_overstates) is bool:
            fluff_assessment["title_overstates_content"] = (
                "是" if title_overstates else "否"
            )
            corrections.append("converted title_overstates_content boolean to text")
    return normalized, tuple(corrections)


_SCHEMA_ERROR_LABELS = {
    "missing": "缺少必填字段",
    "extra_forbidden": "包含 Schema 未允许的额外字段",
    "int_type": "必须是整数，不能使用字符串或小数",
    "string_type": "必须是字符串",
    "list_type": "必须是数组",
    "dict_type": "必须是对象",
    "literal_error": "必须严格使用 Schema 给定的枚举值",
    "too_short": "数组项目过少",
    "too_long": "数组项目过多",
    "string_too_short": "字符串为空或过短",
    "string_too_long": "字符串超过最大长度",
    "greater_than_equal": "数字低于允许的最小值",
    "less_than_equal": "数字超过允许的最大值",
}


def _safe_schema_issue(exc: ValidationError) -> str:
    """Summarize schema locations without including model output or article data."""

    errors = exc.errors(include_url=False, include_input=False)
    summaries: list[str] = []
    for error in errors[:12]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "根对象"
        error_type = str(error.get("type", "validation_error"))
        label = _SCHEMA_ERROR_LABELS.get(error_type, f"不符合约束（{error_type}）")
        context = error.get("ctx")
        if isinstance(context, dict):
            if error_type == "too_long" and isinstance(
                context.get("max_length"), int
            ):
                label += f"（最多 {context['max_length']} 项）"
            elif error_type == "too_short" and isinstance(
                context.get("min_length"), int
            ):
                label += f"（至少 {context['min_length']} 项）"
        summaries.append(f"{location}: {label}")
    if len(errors) > len(summaries):
        summaries.append(f"另有 {len(errors) - len(summaries)} 项同类错误")
    return "；".join(summaries)[:1600]


def _repair_messages(
    original_messages: list[dict[str, Any]],
    invalid_response: str,
    issue: str,
) -> list[dict[str, Any]]:
    return [
        *original_messages,
        {"role": "assistant", "content": invalid_response},
        {
            "role": "user",
            "content": (
                prompt_text("analysis_repair", issue=issue)
                + "\n修复时必须删除所有 Schema 未允许的字段；严格按字段类型改写值；"
                "数组数量必须符合 minItems/maxItems。必须返回完整对象，不能只返回补丁。"
            ),
        },
    ]


def _compact_retry_messages(
    original_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retry a truncated response once without carrying the huge partial JSON."""

    messages = deepcopy(original_messages)
    directive = prompt_text("analysis_compact_retry")
    user_content = messages[-1].get("content")
    if isinstance(user_content, str):
        messages[-1]["content"] = user_content + directive
    elif isinstance(user_content, list):
        for item in user_content:
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = str(item.get("text", "")) + directive
                break
    return messages


def analyze_article(
    article: ArticleDraft,
    settings: ModelSettings,
    client_factory: Callable[[ModelSettings], ChatClient] = OpenAICompatibleClient,
) -> Analysis:
    """Analyze an article, allowing exactly one repair request."""

    client = client_factory(settings)
    messages = build_analysis_messages(
        article,
        include_images=settings.supports_vision,
    )
    repaired = False
    schema_compatibility_applied = False
    initial_diagnostics = None
    try:
        raw_response = client.complete(messages)
    except AnalysisOutputTruncated:
        try:
            raw_response = client.complete(_compact_retry_messages(messages))
        except AnalysisOutputTruncated:
            raise AnalysisError(
                "模型在一次自动精简重试后仍因输出长度被截断。"
                "系统已保留文章，请稍后重试或减少一次分析中的图片数量。"
            ) from None
        repaired = True

    try:
        payload, compatibility_corrections = _parse_analysis_payload(
            raw_response,
            apply_known_compatibility=True,
        )
        schema_compatibility_applied = bool(compatibility_corrections)
    except AnalysisSchemaError as exc:
        if repaired:
            logger.warning(
                "Compact analysis retry failed schema validation: %s",
                exc.repair_issue,
            )
            raise AnalysisError(
                "模型在一次自动精简重试后仍未返回完整的合法结构，请重新分析。"
                f" 安全诊断：{_public_schema_diagnostic(exc.repair_issue)}"
            ) from None
        raw_response = client.complete(
            _repair_messages(messages, raw_response, exc.repair_issue)
        )
        repaired = True
        try:
            payload, compatibility_corrections = _parse_analysis_payload(
                raw_response,
                apply_known_compatibility=True,
            )
            schema_compatibility_applied = (
                schema_compatibility_applied or bool(compatibility_corrections)
            )
        except AnalysisSchemaError as repair_exc:
            logger.warning(
                "Analysis repair failed schema validation: %s",
                repair_exc.repair_issue,
            )
            raise AnalysisError(
                "模型输出在一次自动修复后仍未通过结构校验，请重新分析。"
                f" 安全诊断：{_public_schema_diagnostic(repair_exc.repair_issue)}"
            ) from None

    payload, diagnostics = normalize_references(payload, article)
    if not repaired:
        initial_diagnostics = diagnostics

    if diagnostics.invalid_references and not repaired:
        raw_response = client.complete(
            _repair_messages(
                messages,
                raw_response,
                "格式修正后仍包含不存在或无法解析的段落引用："
                f"{', '.join(diagnostics.invalid_references)}",
            )
        )
        repaired = True
        try:
            payload, compatibility_corrections = _parse_analysis_payload(
                raw_response,
                apply_known_compatibility=True,
            )
            schema_compatibility_applied = (
                schema_compatibility_applied or bool(compatibility_corrections)
            )
        except AnalysisError:
            raise AnalysisError(
                "模型修复引用时返回了无效结构，请重新分析。"
            ) from None
        payload, diagnostics = normalize_references(payload, article)

    if diagnostics.invalid_references:
        validation_status = "valid_with_invalid_references"
    elif repaired:
        validation_status = "repaired"
    elif schema_compatibility_applied or diagnostics.corrected_valid_mentions:
        validation_status = "valid_with_auto_corrections"
    else:
        validation_status = "valid"

    return Analysis(
        article_id=article_fingerprint(
            article,
            settings.model,
            settings.base_url,
            settings.supports_vision,
        ),
        prompt_version=active_prompt_version(PROMPT_VERSION),
        model_name=settings.model,
        reading_decision=payload.reading_decision,
        core_analysis=payload.core_analysis,
        explanations=payload.explanations,
        promotion_assessment=payload.promotion_assessment,
        fluff_assessment=payload.fluff_assessment,
        raw_response=raw_response,
        validation_status=validation_status,
        invalid_references=diagnostics.invalid_references,
        initial_reference_diagnostics=initial_diagnostics,
        reference_diagnostics=diagnostics,
    )
