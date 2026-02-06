"""Tests for AIModel and MODEL_CONFIG."""

import pytest
from skell_e_router.model_config import AIModel, MODEL_CONFIG


# ---------------------------------------------------------------------------
# AIModel provider properties
# ---------------------------------------------------------------------------

class TestAIModelProviderProperties:

    def test_is_gemini(self):
        m = AIModel("gemini/test", "gemini", False, set())
        assert m.is_gemini is True
        assert m.is_openai is False
        assert m.is_anthropic is False

    def test_is_openai(self):
        m = AIModel("openai/gpt-5", "openai", False, set())
        assert m.is_openai is True
        assert m.is_gemini is False

    def test_is_anthropic(self):
        m = AIModel("anthropic/claude", "anthropic", False, set())
        assert m.is_anthropic is True

    def test_is_xai(self):
        m = AIModel("xai/grok", "xai", False, set())
        assert m.is_xai is True

    def test_is_groq(self):
        m = AIModel("groq/compound", "groq", False, set())
        assert m.is_groq is True

    def test_is_openai_o_series_true(self):
        m = AIModel("openai/o3", "openai", True, set())
        assert m.is_openai_o_series is True

    def test_is_openai_o_series_false_for_gpt(self):
        m = AIModel("openai/gpt-5", "openai", False, set())
        assert m.is_openai_o_series is False

    def test_is_openai_o_series_false_for_non_openai(self):
        m = AIModel("anthropic/o3", "anthropic", False, set())
        assert m.is_openai_o_series is False


# ---------------------------------------------------------------------------
# AIModel construction
# ---------------------------------------------------------------------------

class TestAIModelConstruction:

    def test_stores_all_fields(self):
        m = AIModel(
            name="openai/gpt-5",
            provider="openai",
            supports_thinking=True,
            supported_params={"stream", "temperature"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        assert m.name == "openai/gpt-5"
        assert m.provider == "openai"
        assert m.supports_thinking is True
        assert "stream" in m.supported_params
        assert "temperature" in m.supported_params
        assert m.accepted_reasoning_efforts == {"low", "medium", "high"}

    def test_accepted_reasoning_efforts_defaults_none(self):
        m = AIModel("test/m", "test", False, set())
        assert m.accepted_reasoning_efforts is None


# ---------------------------------------------------------------------------
# MODEL_CONFIG registry
# ---------------------------------------------------------------------------

class TestModelConfig:

    def test_registry_is_not_empty(self):
        assert len(MODEL_CONFIG) > 0

    @pytest.mark.parametrize("alias", [
        "gpt-5", "gpt-4o", "o3", "o1",
        "gemini-2.5-pro", "gemini-2.5-flash",
        "nano-banana-3", "gemini-3-pro-image",
        "claude-opus-4-6", "claude-opus-4-5", "claude-haiku-4-5",
        "groq-compound", "groq-compound-mini",
    ])
    def test_known_aliases_exist(self, alias):
        assert alias in MODEL_CONFIG
        assert isinstance(MODEL_CONFIG[alias], AIModel)

    def test_full_name_lookup(self):
        """Models should be accessible by their full LiteLLM name too."""
        assert "openai/gpt-5" in MODEL_CONFIG
        assert MODEL_CONFIG["openai/gpt-5"].name == "openai/gpt-5"

    def test_alias_and_full_name_point_to_same_model(self):
        alias_model = MODEL_CONFIG["gpt-5"]
        full_name_model = MODEL_CONFIG["openai/gpt-5"]
        assert alias_model.name == full_name_model.name
        assert alias_model.provider == full_name_model.provider

    def test_all_models_have_required_fields(self):
        seen = set()
        for key, model in MODEL_CONFIG.items():
            if model.name in seen:
                continue
            seen.add(model.name)
            assert isinstance(model.name, str) and len(model.name) > 0
            assert isinstance(model.provider, str) and len(model.provider) > 0
            assert isinstance(model.supports_thinking, bool)
            assert isinstance(model.supported_params, set)

    def test_all_models_have_stream_param(self):
        """Every model should at least support streaming."""
        seen = set()
        for key, model in MODEL_CONFIG.items():
            if model.name in seen:
                continue
            seen.add(model.name)
            assert "stream" in model.supported_params, f"{model.name} missing 'stream'"

    @pytest.mark.parametrize("alias", [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        "gemini-3-flash-preview", "gemini-3-pro-preview",
        "nano-banana-3",
    ])
    def test_gemini_models_have_safety_settings(self, alias):
        model = MODEL_CONFIG[alias]
        assert "safety_settings" in model.supported_params

    @pytest.mark.parametrize("alias", [
        "claude-opus-4-6", "claude-opus-4-5", "claude-haiku-4-5",
    ])
    def test_anthropic_models_support_thinking(self, alias):
        model = MODEL_CONFIG[alias]
        assert model.supports_thinking is True
        assert "budget_tokens" in model.supported_params
        assert "thinking" in model.supported_params

    def test_opus_4_6_supports_adaptive_thinking(self):
        model = MODEL_CONFIG["claude-opus-4-6"]
        assert "reasoning_effort" in model.supported_params
        assert "thinking" in model.supported_params
        assert "budget_tokens" in model.supported_params
        assert model.accepted_reasoning_efforts == {"low", "medium", "high", "max"}

    def test_groq_compound_models_have_compound_custom(self):
        for alias in ("groq-compound", "groq-compound-mini"):
            model = MODEL_CONFIG[alias]
            assert "compound_custom" in model.supported_params
            assert "extra_headers" in model.supported_params

    def test_nano_banana_has_modalities(self):
        model = MODEL_CONFIG["nano-banana-3"]
        assert model.provider == "gemini"
        assert "modalities" in model.supported_params
        assert model.supports_thinking is False

    def test_nano_banana_aliases_point_to_same_model(self):
        assert MODEL_CONFIG["nano-banana-3"] is MODEL_CONFIG["gemini-3-pro-image"]

    def test_nano_banana_full_name_lookup(self):
        assert "gemini/gemini-3-pro-image-preview" in MODEL_CONFIG
        model = MODEL_CONFIG["gemini/gemini-3-pro-image-preview"]
        assert model.name == "gemini/gemini-3-pro-image-preview"
