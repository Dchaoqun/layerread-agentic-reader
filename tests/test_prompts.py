from importlib.util import find_spec

from layerread.prompts import active_prompt_profile, active_prompt_version, prompt_text


def test_local_runtime_uses_private_prompts_when_assets_are_installed(monkeypatch) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "local")
    monkeypatch.delenv("LAYERREAD_PROMPT_PROFILE", raising=False)

    try:
        private_assets_installed = find_spec("layerread_private.prompts") is not None
    except ModuleNotFoundError:
        private_assets_installed = False
    expected = "production" if private_assets_installed else "basic"

    assert active_prompt_profile() == expected
    assert active_prompt_version("analysis-v1") == f"analysis-v1-{expected}"
    assert ("知识点至少三个" in prompt_text("analysis")) == private_assets_installed


def test_portfolio_runtime_forces_basic_prompt_assets(monkeypatch) -> None:
    monkeypatch.setenv("LAYERREAD_APP_MODE", "portfolio")
    monkeypatch.setenv("LAYERREAD_PROMPT_PROFILE", "production")

    assert active_prompt_profile() == "basic"
    assert active_prompt_version("analysis-v1") == "analysis-v1-basic"
    prompt = prompt_text("analysis")
    assert "中文文章阅读助手" in prompt
    assert "知识点至少三个" not in prompt
