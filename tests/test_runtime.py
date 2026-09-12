from layerread.config import ModelConfigResult
from layerread.runtime import filter_model_configs, read_runtime_profile


def _result(provider_id: str) -> ModelConfigResult:
    return ModelConfigResult(
        settings=None,
        errors=("not configured",),
        provider_id=provider_id,
        provider_name=provider_id,
    )


def test_local_runtime_keeps_full_feature_set() -> None:
    profile = read_runtime_profile({"LAYERREAD_APP_MODE": "local"})

    assert profile.mode == "local"
    assert profile.connector_enabled
    assert profile.image_features_enabled
    assert profile.visual_provider_enabled
    assert profile.notion_enabled
    assert profile.long_term_storage_enabled
    assert not profile.byok_enabled
    assert profile.prompt_profile == "production"


def test_demo_runtime_keeps_connector_images_and_visual_provider_but_no_persistence() -> None:
    profile = read_runtime_profile({"LAYERREAD_APP_MODE": "demo"})
    configs = (_result("default"), _result("aliyun"))

    assert profile.is_demo
    assert profile.connector_enabled
    assert profile.image_features_enabled
    assert profile.visual_provider_enabled
    assert not profile.notion_enabled
    assert not profile.long_term_storage_enabled
    assert not profile.byok_enabled
    assert [result.provider_id for result in filter_model_configs(configs, profile)] == [
        "default",
        "aliyun",
    ]


def test_demo_runtime_can_use_deployment_owned_visual_provider() -> None:
    profile = read_runtime_profile({"LAYERREAD_APP_MODE": "demo"})

    filtered = filter_model_configs((_result("aliyun"),), profile)

    assert len(filtered) == 1
    assert filtered[0].provider_id == "aliyun"


def test_portfolio_runtime_forces_byok_basic_prompts_and_public_boundaries() -> None:
    profile = read_runtime_profile(
        {
            "LAYERREAD_APP_MODE": "portfolio",
            "LAYERREAD_PROMPT_PROFILE": "production",
        }
    )

    assert profile.is_portfolio
    assert profile.byok_enabled
    assert profile.prompt_profile == "basic"
    assert profile.connector_enabled
    assert profile.image_features_enabled
    assert not profile.visual_provider_enabled
    assert not profile.notion_enabled
    assert profile.long_term_storage_enabled
