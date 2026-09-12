from __future__ import annotations

from pathlib import Path

from layerread.config import (
    ALIYUN_OPENAI_BASE_URL,
    ALIYUN_TOKEN_PLAN_BASE_URL,
    read_model_config,
    read_model_configs,
    read_portfolio_model_config,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_valid_generic_provider_config_is_supported() -> None:
    result = read_model_config(
        env={
            "LLM_API_KEY": "test-secret-key",
            "LLM_BASE_URL": "https://api.provider.example/v1",
            "LLM_MODEL": "test-model",
        }
    )

    assert result.is_ready is True
    assert result.settings is not None
    assert result.settings.base_url == "https://api.provider.example/v1"
    assert result.settings.model == "test-model"


def test_custom_compatible_provider_config_is_supported() -> None:
    result = read_model_config(
        env={
            "LLM_API_KEY": "compatible-provider-secret",
            "LLM_BASE_URL": "https://models.example.com/v1/",
            "LLM_MODEL": "provider-model",
        }
    )

    assert result.is_ready is True
    assert result.settings is not None
    assert result.settings.base_url == "https://models.example.com/v1"


def test_default_and_aliyun_openai_compatible_configs_can_coexist() -> None:
    results = read_model_configs(
        env={
            "LLM_API_KEY": "deepseek-secret",
            "LLM_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "deepseek-chat",
            "DASHSCOPE_API_KEY": "dashscope-secret",
            "DASHSCOPE_BASE_URL": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            "DASHSCOPE_MODEL": "qwen3.7-plus",
        }
    )

    assert [result.provider_id for result in results] == ["default", "aliyun"]
    assert all(result.is_ready for result in results)
    aliyun = results[1].settings
    assert aliyun is not None
    assert aliyun.provider_name == "阿里云百炼"
    assert aliyun.base_url.endswith("/compatible-mode/v1")
    assert aliyun.fallback_base_urls == (ALIYUN_OPENAI_BASE_URL,)
    assert aliyun.model == "qwen3.7-plus"
    assert aliyun.supports_vision


def test_aliyun_requires_key_base_url_and_model_together() -> None:
    results = read_model_configs(env={"DASHSCOPE_API_KEY": "dashscope-secret"})

    assert len(results) == 1
    assert results[0].provider_id == "aliyun"
    assert results[0].is_ready is False
    assert any("DASHSCOPE_BASE_URL" in error for error in results[0].errors)
    assert any("DASHSCOPE_MODEL" in error for error in results[0].errors)


def test_aliyun_can_be_the_only_configured_provider() -> None:
    results = read_model_configs(
        env={
            "DASHSCOPE_API_KEY": "dashscope-secret",
            "DASHSCOPE_BASE_URL": "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/",
            "DASHSCOPE_MODEL": "qwen-plus",
        }
    )

    assert len(results) == 1
    assert results[0].provider_id == "aliyun"
    assert results[0].is_ready is True
    assert results[0].settings is not None
    assert results[0].settings.base_url.endswith("/compatible-mode/v1")


def test_aliyun_plan_key_rejects_shared_pay_as_you_go_endpoint() -> None:
    results = read_model_configs(
        env={
            "DASHSCOPE_API_KEY": "sk-sp-plan-secret",
            "DASHSCOPE_BASE_URL": ALIYUN_OPENAI_BASE_URL,
            "DASHSCOPE_MODEL": "qwen3.8-max-preview",
        }
    )

    assert results[0].is_ready is False
    assert any("sk-sp-" in error for error in results[0].errors)
    assert any("应用后端" in error for error in results[0].errors)


def test_aliyun_token_plan_key_is_not_available_to_app_backend() -> None:
    results = read_model_configs(
        env={
            "DASHSCOPE_API_KEY": "sk-sp-plan-secret",
            "DASHSCOPE_BASE_URL": ALIYUN_TOKEN_PLAN_BASE_URL,
            "DASHSCOPE_MODEL": "qwen3.8-max-preview",
        }
    )

    assert results[0].is_ready is False
    assert any("按量付费" in error for error in results[0].errors)


def test_token_plan_only_model_is_not_available_with_standard_key() -> None:
    results = read_model_configs(
        env={
            "DASHSCOPE_API_KEY": "sk-ws-standard-secret",
            "DASHSCOPE_BASE_URL": ALIYUN_OPENAI_BASE_URL,
            "DASHSCOPE_MODEL": "qwen3.8-max-preview",
        }
    )

    assert results[0].is_ready is False
    assert any("Token Plan 专属模型" in error for error in results[0].errors)


def test_missing_required_config_returns_actionable_errors() -> None:
    result = read_model_config(env={})

    assert result.is_ready is False
    assert result.settings is None
    assert any("LLM_API_KEY" in error for error in result.errors)
    assert any("LLM_BASE_URL" in error for error in result.errors)
    assert any("LLM_MODEL" in error for error in result.errors)


def test_example_api_key_placeholder_is_not_considered_ready() -> None:
    result = read_model_config(
        env={
            "LLM_API_KEY": "your_api_key_here",
            "LLM_BASE_URL": "https://api.provider.example/v1",
            "LLM_MODEL": "test-model",
        }
    )

    assert result.is_ready is False
    assert result.settings is None
    assert any("LLM_API_KEY" in error for error in result.errors)


def test_invalid_base_url_is_rejected_without_echoing_value() -> None:
    unsafe_url = "https://user:secret@example.com/v1?token=private"
    result = read_model_config(
        env={
            "LLM_API_KEY": "another-secret",
            "LLM_BASE_URL": unsafe_url,
            "LLM_MODEL": "test-model",
        }
    )

    assert result.is_ready is False
    assert unsafe_url not in " ".join(result.errors)
    assert "another-secret" not in " ".join(result.errors)


def test_api_key_is_redacted_from_repr_and_safe_summary() -> None:
    secret = "must-never-appear"
    result = read_model_config(
        env={
            "LLM_API_KEY": secret,
            "LLM_BASE_URL": "https://api.provider.example/v1",
            "LLM_MODEL": "test-model",
        }
    )

    assert result.settings is not None
    assert secret not in repr(result.settings)
    assert secret not in repr(result.settings.safe_summary)
    assert "api_key" not in result.settings.safe_summary
    assert "base_url" not in result.settings.safe_summary


def test_portfolio_config_is_loaded_from_env_without_echoing_key() -> None:
    key = "sk-user-session-secret"

    result = read_portfolio_model_config(
        env={
            "LLM_API_KEY": key,
            "LLM_BASE_URL": "https://models.example.com/v1",
            "LLM_MODEL": "portfolio-model",
            "LLM_SUPPORTS_VISION": "true",
        }
    )

    assert result.is_ready
    assert result.settings is not None
    assert result.settings.provider_id == "byok"
    assert result.settings.supports_vision
    assert key not in repr(result.settings)
    assert result.settings.safe_summary == {"model": "portfolio-model"}


def test_portfolio_errors_point_to_dotenv_and_never_echo_invalid_values() -> None:
    invalid_url = "https://user:password@example.com/v1?secret=value"

    result = read_portfolio_model_config(
        env={
            "LLM_API_KEY": "",
            "LLM_BASE_URL": invalid_url,
            "LLM_MODEL": "",
        }
    )

    assert not result.is_ready
    assert result.errors
    assert any(".env" in error for error in result.errors)
    assert invalid_url not in " ".join(result.errors)


def test_portfolio_rejects_invalid_vision_flag() -> None:
    result = read_portfolio_model_config(
        env={
            "LLM_API_KEY": "portfolio-secret",
            "LLM_BASE_URL": "https://models.example.com/v1",
            "LLM_MODEL": "portfolio-model",
            "LLM_SUPPORTS_VISION": "sometimes",
        }
    )

    assert not result.is_ready
    assert any("LLM_SUPPORTS_VISION" in error for error in result.errors)


def test_dotenv_file_is_loaded_without_overriding_existing_environment(
    monkeypatch,
) -> None:
    dotenv_file = FIXTURES_DIR / "model-config.env"
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    result = read_model_config(dotenv_path=dotenv_file)

    assert result.is_ready is True
    assert result.settings is not None
    assert result.settings.model == "environment-model"
    assert result.settings.base_url == "https://file.example.com/v1"
