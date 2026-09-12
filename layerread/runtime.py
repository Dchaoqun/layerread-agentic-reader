"""Runtime feature profile for local and hosted LayerRead deployments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from layerread.config import ModelConfigResult


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Feature availability for one deployment mode."""

    mode: str
    connector_enabled: bool
    image_features_enabled: bool
    visual_provider_enabled: bool
    notion_enabled: bool
    long_term_storage_enabled: bool
    byok_enabled: bool
    prompt_profile: str

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def is_portfolio(self) -> bool:
        return self.mode == "portfolio"


EDITION_MARKER = Path(".layerread-edition")


def _default_mode() -> str:
    try:
        marker = EDITION_MARKER.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "local"
    return "portfolio" if marker == "portfolio" else "local"


def read_runtime_profile(
    env: Mapping[str, str] | None = None,
) -> RuntimeProfile:
    """Read the deployment profile without granting features by accident.

    ``local`` remains the default for existing installations. The hosted demo
    keeps the Connector, image ingestion, and visual provider so visitors can
    exercise the complete workflow, but disables Notion and long-term article
    storage. ``portfolio`` keeps local persistence and Connector/images while
    forcing BYOK and basic prompts.
    """

    if env is None:
        load_dotenv(dotenv_path=".env", override=False)
        source = os.environ
    else:
        source = env
    mode = source.get("LAYERREAD_APP_MODE", _default_mode()).strip().lower()
    if mode not in {"local", "demo", "portfolio"}:
        mode = _default_mode()
    connector_features = mode in {"local", "demo", "portfolio"}
    local_features = mode == "local"
    configured_prompt_profile = source.get(
        "LAYERREAD_PROMPT_PROFILE", "production"
    ).strip().lower()
    prompt_profile = (
        "basic"
        if mode == "portfolio" or configured_prompt_profile == "basic"
        else "production"
    )
    return RuntimeProfile(
        mode=mode,
        connector_enabled=connector_features,
        image_features_enabled=connector_features,
        visual_provider_enabled=mode in {"local", "demo"},
        notion_enabled=local_features,
        long_term_storage_enabled=mode != "demo",
        byok_enabled=mode == "portfolio",
        prompt_profile=prompt_profile,
    )


def filter_model_configs(
    results: tuple[ModelConfigResult, ...],
    profile: RuntimeProfile,
) -> tuple[ModelConfigResult, ...]:
    """Hide deployment-owned visual providers where the profile forbids them."""

    if profile.visual_provider_enabled:
        return results
    filtered = tuple(result for result in results if result.provider_id != "aliyun")
    if filtered:
        return filtered
    return (
        ModelConfigResult(
            settings=None,
            errors=("在线体验模式尚未配置可用的文本模型。",),
            provider_id="demo",
            provider_name="在线体验",
        ),
    )
