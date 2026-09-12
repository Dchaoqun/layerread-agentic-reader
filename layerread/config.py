"""Safe model configuration loading for LayerRead."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


_API_KEY_PLACEHOLDERS = {
    "your_api_key_here",
    "replace_me",
    "changeme",
}

ALIYUN_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_TOKEN_PLAN_BASE_URL = (
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Validated settings for a configurable LLM provider."""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    provider_id: str = "default"
    provider_name: str = "OpenAI-compatible"
    supports_vision: bool = False
    proxy_url: str = field(default="", repr=False)
    fallback_base_urls: tuple[str, ...] = ()

    @property
    def safe_summary(self) -> dict[str, str]:
        """Return display-safe settings that never include the API key."""

        return {"model": self.model}


@dataclass(frozen=True, slots=True)
class ModelConfigResult:
    """Configuration readiness plus user-actionable, secret-safe errors."""

    settings: ModelSettings | None
    errors: tuple[str, ...]
    provider_id: str = "default"
    provider_name: str = "OpenAI-compatible"

    @property
    def is_ready(self) -> bool:
        return self.settings is not None and not self.errors


def _is_valid_base_url(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _is_valid_proxy_url(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme in {"http", "https"} and parsed.hostname)


def _aliyun_fallback_base_urls(base_url: str) -> tuple[str, ...]:
    """Return the official same-region shared endpoint when one exists."""

    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if path != "/compatible-mode/v1":
        return ()
    if host.endswith(".cn-beijing.maas.aliyuncs.com"):
        fallback = ALIYUN_OPENAI_BASE_URL
    elif host.endswith(".ap-southeast-1.maas.aliyuncs.com"):
        fallback = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    else:
        return ()
    return () if fallback.rstrip("/") == base_url.rstrip("/") else (fallback,)


def _read_settings(
    source: Mapping[str, str],
    *,
    api_key_name: str,
    base_url_name: str,
    model_name: str,
    provider_id: str,
    provider_name: str,
    proxy_url_name: str = "",
    default_base_url: str = "",
    default_model: str = "",
    supports_vision: bool = False,
) -> ModelConfigResult:
    api_key = source.get(api_key_name, "").strip()
    base_url = source.get(base_url_name, default_base_url).strip()
    model = source.get(model_name, default_model).strip()
    proxy_url = source.get(proxy_url_name, "").strip() if proxy_url_name else ""

    errors: list[str] = []
    if not api_key or api_key.lower() in _API_KEY_PLACEHOLDERS:
        errors.append(f"缺少 {api_key_name}：请在 .env 中填写{provider_name}的 API Key。")
    if not model:
        errors.append(f"缺少 {model_name}：请在 .env 中填写要使用的模型名称。")
    if not base_url:
        errors.append(f"缺少 {base_url_name}：请在 .env 中填写模型服务的 API 地址。")
    elif not _is_valid_base_url(base_url):
        errors.append(
            f"{base_url_name} 格式无效：请填写不含账号、查询参数或片段的 HTTP(S) 地址。"
        )
    if proxy_url and not _is_valid_proxy_url(proxy_url):
        errors.append(
            f"{proxy_url_name} 格式无效：仅支持 HTTP(S) 代理地址。"
        )

    if provider_id == "aliyun" and api_key and _is_valid_base_url(base_url):
        base_host = (urlsplit(base_url).hostname or "").lower()
        is_plan_key = api_key.startswith("sk-sp-")
        is_plan_endpoint = "token-plan" in base_host or "coding" in base_host
        if is_plan_key or is_plan_endpoint:
            errors.append(
                "Token Plan/Coding Plan 的 sk-sp- Key 和专属地址仅限交互式 AI 编程/Agent 工具，"
                "不可用于 LayerRead 这类应用后端；请改用百炼按量付费的标准 API Key 和 API Host。"
            )
        if model.lower() == "qwen3.8-max-preview":
            errors.append(
                "qwen3.8-max-preview 当前为 Token Plan 专属模型，不可用于 LayerRead 应用后端；"
                "请选择百炼按量付费支持的模型，例如 qwen3.6-flash 或 qwen3.7-plus。"
            )

    if errors:
        return ModelConfigResult(
            settings=None,
            errors=tuple(errors),
            provider_id=provider_id,
            provider_name=provider_name,
        )

    return ModelConfigResult(
        settings=ModelSettings(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            provider_id=provider_id,
            provider_name=provider_name,
            supports_vision=supports_vision,
            proxy_url=proxy_url,
            fallback_base_urls=(
                _aliyun_fallback_base_urls(base_url)
                if provider_id == "aliyun"
                else ()
            ),
        ),
        errors=(),
        provider_id=provider_id,
        provider_name=provider_name,
    )


def _config_source(
    env: Mapping[str, str] | None,
    dotenv_path: str | Path,
) -> Mapping[str, str]:
    if env is not None:
        return env
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return os.environ


def read_model_config(
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path = ".env",
) -> ModelConfigResult:
    """Read and validate model settings without exposing secret values.

    Passing ``env`` makes the function deterministic and skips ``.env`` loading,
    which is useful for tests. In normal application use, existing environment
    variables take precedence over values in ``.env``.
    """

    source = _config_source(env, dotenv_path)
    return _read_settings(
        source,
        api_key_name="LLM_API_KEY",
        base_url_name="LLM_BASE_URL",
        model_name="LLM_MODEL",
        provider_id="default",
        provider_name="默认 OpenAI-compatible",
        proxy_url_name="LLM_PROXY_URL",
    )


def read_model_configs(
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path = ".env",
) -> tuple[ModelConfigResult, ...]:
    """Load the existing provider and optional Alibaba Cloud provider.

    The original ``LLM_*`` variables remain fully backward compatible. Alibaba
    Cloud is added only when at least one ``DASHSCOPE_*`` variable is present,
    so existing installations do not see an unused provider warning.
    """

    source = _config_source(env, dotenv_path)
    has_default = any(source.get(name, "").strip() for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"))
    has_aliyun = any(
        source.get(name, "").strip()
        for name in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL")
    )

    results: list[ModelConfigResult] = []
    if has_default or not has_aliyun:
        results.append(read_model_config(env=source))
    if has_aliyun:
        results.append(
            _read_settings(
                source,
                api_key_name="DASHSCOPE_API_KEY",
                base_url_name="DASHSCOPE_BASE_URL",
                model_name="DASHSCOPE_MODEL",
                provider_id="aliyun",
                provider_name="阿里云百炼",
                proxy_url_name="DASHSCOPE_PROXY_URL",
                supports_vision=True,
            )
        )
    return tuple(results)


def read_portfolio_model_config(
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path = ".env",
) -> ModelConfigResult:
    """Read the user's Portfolio BYOK configuration from the local environment."""

    source = _config_source(env, dotenv_path)
    raw_supports_vision = source.get("LLM_SUPPORTS_VISION", "false").strip().lower()
    valid_boolean_values = {"1", "0", "true", "false", "yes", "no", "on", "off"}
    supports_vision = raw_supports_vision in {"1", "true", "yes", "on"}

    result = _read_settings(
        source,
        api_key_name="LLM_API_KEY",
        base_url_name="LLM_BASE_URL",
        model_name="LLM_MODEL",
        provider_id="byok",
        provider_name="用户自己的 OpenAI-compatible 模型",
        proxy_url_name="LLM_PROXY_URL",
        supports_vision=supports_vision,
    )
    if raw_supports_vision in valid_boolean_values:
        return result
    return ModelConfigResult(
        settings=None,
        errors=(
            *result.errors,
            "LLM_SUPPORTS_VISION 格式无效：请在 .env 中填写 true 或 false。",
        ),
        provider_id=result.provider_id,
        provider_name=result.provider_name,
    )
