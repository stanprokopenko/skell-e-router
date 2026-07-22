"""Tests for AIModel and MODEL_CONFIG."""

import pytest
from skell_e_router.model_config import (
    AIModel, MODEL_CONFIG,
    EmbeddingModel, EMBEDDING_MODEL_CONFIG, resolve_embedding_alias,
)
from skell_e_router.utils import RouterError


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

    def test_is_deepinfra(self):
        m = AIModel("deepinfra/nvidia/test", "deepinfra", False, set())
        assert m.is_deepinfra is True

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
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
        "gpt-5.5", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5.3-codex", "gpt-5", "gpt-4o", "o3", "o1",
        "gemini-3.6-flash", "gemini-3.5-flash-lite",
        "gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash",
        "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview",
        "nano-banana-3", "gemini-3-pro-image",
        "claude-fable-5", "claude-opus-4-8", "claude-sonnet-5",
        "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5",
        "muse-spark-1.1",
        "grok-4.5", "grok-4.20", "grok-4.20-non-reasoning",
        "grok-4-0220", "grok-code-fast-1",
        "groq-compound", "groq-compound-mini",
        "qwen3-32b", "kimi-k2-0905",
        "deepseek-v4-pro", "deepseek-v4-flash", "kimi-k2.6",
        "glm-5.2", "minimax-m3", "qwen3.5-397b",
        "nemotron-3-ultra",
        "nemotron-3-super", "nemotron-super-49b", "nemotron-70b",
        "nemotron-3-nano-30b", "nemotron-nano-12b-vl", "nemotron-nano-9b",
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
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5",
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

    @pytest.mark.parametrize("alias", ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-5"])
    def test_fable_5_and_opus_4_8_config(self, alias):
        """Fable 5, Opus 4.8, Sonnet 5: adaptive thinking only, no sampling params, effort up to max."""
        model = MODEL_CONFIG[alias]
        assert model.provider == "anthropic"
        assert model.use_direct_sdk is True
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert "thinking" in model.supported_params
        assert "temperature" not in model.supported_params
        assert "budget_tokens" not in model.supported_params
        assert model.accepted_reasoning_efforts == {"low", "medium", "high", "xhigh", "max"}

    def test_opus_4_7_config(self):
        """Opus 4.7 drops sampling params and budget_tokens; only adaptive thinking via reasoning_effort."""
        model = MODEL_CONFIG["claude-opus-4-7"]
        assert model.provider == "anthropic"
        assert model.is_anthropic is True
        assert model.use_direct_sdk is True
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert "thinking" in model.supported_params
        # Removed in 4.7 — these return 400 from the API
        assert "temperature" not in model.supported_params
        assert "top_p" not in model.supported_params
        assert "top_k" not in model.supported_params
        assert "budget_tokens" not in model.supported_params
        assert model.accepted_reasoning_efforts == {"low", "medium", "high", "xhigh"}

    def test_sonnet_4_6_supports_adaptive_thinking(self):
        model = MODEL_CONFIG["claude-sonnet-4-6"]
        assert "reasoning_effort" in model.supported_params
        assert "thinking" in model.supported_params
        assert "budget_tokens" in model.supported_params
        assert model.accepted_reasoning_efforts == {"low", "medium", "high"}

    def test_groq_compound_models_have_compound_custom(self):
        for alias in ("groq-compound", "groq-compound-mini"):
            model = MODEL_CONFIG[alias]
            assert "compound_custom" in model.supported_params
            assert "extra_headers" in model.supported_params

    def test_qwen3_32b_config(self):
        model = MODEL_CONFIG["qwen3-32b"]
        assert model.provider == "groq"
        assert model.is_groq is True
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert model.accepted_reasoning_efforts == {"none", "default", "low", "medium", "high"}

    def test_kimi_k2_config(self):
        model = MODEL_CONFIG["kimi-k2-0905"]
        assert model.provider == "groq"
        assert model.is_groq is True
        assert model.supports_thinking is False
        assert "reasoning_effort" not in model.supported_params

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

    def test_gemini_3_5_flash_config(self):
        """gemini-3.5-flash: GA flash model, direct SDK, thinking_level minimal/low/medium/high."""
        model = MODEL_CONFIG["gemini-3.5-flash"]
        assert model.provider == "gemini"
        assert model.supports_thinking is True
        assert model.use_direct_sdk is True
        assert "reasoning_effort" in model.supported_params
        assert model.accepted_reasoning_efforts == {"minimal", "low", "medium", "high"}

    def test_muse_spark_1_1_config(self):
        """Muse Spark 1.1 routes through LiteLLM's generic openai/ provider with a
        custom api_base pointing at Meta Model API. Reasoning always on ("none" rejected),
        stop rejected."""
        model = MODEL_CONFIG["muse-spark-1.1"]
        assert model.provider == "meta"
        assert model.is_meta is True
        assert model.is_openai is False  # routed via openai/ prefix but not an OpenAI model
        assert model.name == "openai/muse-spark-1.1"
        assert model.api_base == "https://api.meta.ai/v1"
        assert model.supports_thinking is True
        assert model.accepted_reasoning_efforts == {"minimal", "low", "medium", "high", "xhigh"}
        assert "stop" not in model.supported_params
        assert "temperature" in model.supported_params
        assert "tools" in model.supported_params
        # LiteLLM can't price custom-api_base models — router-level fallback pricing,
        # per 1M tokens, from https://dev.meta.ai/docs/getting-started/pricing-rate-limits
        assert model.pricing == {"input": 1.25, "cached_input": 0.15, "output": 4.25}
        # api_base defaults to None for models LiteLLM knows natively
        assert MODEL_CONFIG["gpt-5"].api_base is None

    def test_grok_4_5_config(self):
        """grok-4.5 has reasoning always on with configurable effort (low/medium/high, default high)."""
        model = MODEL_CONFIG["grok-4.5"]
        assert model.provider == "xai"
        assert model.is_xai is True
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert model.accepted_reasoning_efforts == {"low", "medium", "high"}
        assert "stop" not in model.supported_params  # rejected by reasoning models per xAI docs
        assert "tools" in model.supported_params

    def test_grok_4_20_reasoning_config(self):
        """grok-4.20 has reasoning always on; rejects reasoning_effort and stop per xAI docs."""
        model = MODEL_CONFIG["grok-4.20"]
        assert model.provider == "xai"
        assert model.is_xai is True
        assert model.supports_thinking is True
        assert "reasoning_effort" not in model.supported_params
        assert "stop" not in model.supported_params  # rejected by reasoning models per xAI docs
        assert "tools" in model.supported_params

    def test_grok_4_20_non_reasoning_config(self):
        """grok-4.20-non-reasoning is the latency-optimized sibling; reasoning off, stop allowed."""
        model = MODEL_CONFIG["grok-4.20-non-reasoning"]
        assert model.provider == "xai"
        assert model.supports_thinking is False
        assert "stop" in model.supported_params
        assert "reasoning_effort" not in model.supported_params

    @pytest.mark.parametrize("alias", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_gpt_5_6_family_config(self, alias):
        model = MODEL_CONFIG[alias]
        assert model.provider == "openai"
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert "stream" in model.supported_params
        assert "tools" in model.supported_params
        # temperature rejected by the API (only default 1), so not in supported params
        assert "temperature" not in model.supported_params
        # same effort vocabulary as gpt-5.5
        assert model.accepted_reasoning_efforts == {"none", "low", "medium", "high", "xhigh"}

    def test_gpt_5_5_config(self):
        model = MODEL_CONFIG["gpt-5.5"]
        assert model.provider == "openai"
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert "stream" in model.supported_params
        assert "tools" in model.supported_params
        # gpt-5.5 uses a different effort vocabulary: drops "minimal", adds "none" + "xhigh"
        assert model.accepted_reasoning_efforts == {"none", "low", "medium", "high", "xhigh"}

    @pytest.mark.parametrize("alias", ["gpt-5.4-mini", "gpt-5.4-nano"])
    def test_gpt_5_4_models_config(self, alias):
        model = MODEL_CONFIG[alias]
        assert model.provider == "openai"
        assert model.supports_thinking is True
        assert "reasoning_effort" in model.supported_params
        assert "stream" in model.supported_params
        assert "tools" in model.supported_params
        assert model.accepted_reasoning_efforts == {"minimal", "low", "medium", "high"}

    @pytest.mark.parametrize("alias", [
        "nemotron-3-super", "nemotron-super-49b", "nemotron-70b",
        "nemotron-3-nano-30b", "nemotron-nano-12b-vl", "nemotron-nano-9b",
    ])
    def test_deepinfra_nemotron_models(self, alias):
        model = MODEL_CONFIG[alias]
        assert model.provider == "deepinfra"
        assert model.is_deepinfra is True
        assert model.supports_thinking is False
        assert "stream" in model.supported_params
        assert "tools" in model.supported_params
        assert "temperature" in model.supported_params
        assert model.name.startswith("deepinfra/nvidia/")

    @pytest.mark.parametrize("alias,name_prefix", [
        ("deepseek-v4-pro", "deepinfra/deepseek-ai/"),
        ("deepseek-v4-flash", "deepinfra/deepseek-ai/"),
        ("kimi-k2.6", "deepinfra/moonshotai/"),
        ("glm-5.2", "deepinfra/zai-org/"),
        ("minimax-m3", "deepinfra/MiniMaxAI/"),
        ("qwen3.5-397b", "deepinfra/Qwen/"),
        ("nemotron-3-ultra", "deepinfra/nvidia/"),
    ])
    def test_deepinfra_open_weight_models(self, alias, name_prefix):
        model = MODEL_CONFIG[alias]
        assert model.provider == "deepinfra"
        assert model.is_deepinfra is True
        assert model.supports_thinking is True  # these reason server-side by default
        assert model.name.startswith(name_prefix)
        assert "stream" in model.supported_params
        assert "tools" in model.supported_params
        assert "temperature" in model.supported_params
        # LiteLLM's cost map lags new DeepInfra models, so each carries fallback pricing.
        assert model.pricing and "input" in model.pricing and "output" in model.pricing

    def test_nemotron_3_super_is_120b_moe(self):
        model = MODEL_CONFIG["nemotron-3-super"]
        assert "120B" in model.name
        assert "A12B" in model.name  # 12B active params


# ---------------------------------------------------------------------------
# EmbeddingModel registry
# ---------------------------------------------------------------------------

class TestEmbeddingModel:

    def test_class_basic_fields(self):
        m = EmbeddingModel(
            name="provider/some-model",
            provider="openai",
            supported_inputs={"text"},
            max_dimensions=1024,
            default_dimensions=1024,
        )
        assert m.name == "provider/some-model"
        assert m.provider == "openai"
        assert m.supported_inputs == {"text"}
        assert m.max_dimensions == 1024
        assert m.default_dimensions == 1024
        assert m.recommended_dimensions == ()
        assert m.max_input_tokens is None
        assert m.supports_aggregation is False

    def test_provider_helpers(self):
        oa = EmbeddingModel(
            name="openai/x", provider="openai",
            supported_inputs={"text"}, max_dimensions=1, default_dimensions=1,
        )
        gm = EmbeddingModel(
            name="gemini/x", provider="gemini",
            supported_inputs={"text"}, max_dimensions=1, default_dimensions=1,
        )
        assert oa.is_openai is True and oa.is_gemini is False
        assert gm.is_openai is False and gm.is_gemini is True


class TestEmbeddingRegistry:

    def test_three_aliases_registered(self):
        assert "openai-embedding-3-large" in EMBEDDING_MODEL_CONFIG
        assert "openai-embedding-3-small" in EMBEDDING_MODEL_CONFIG
        assert "gemini-embedding-2" in EMBEDDING_MODEL_CONFIG

    def test_full_names_also_keyed(self):
        assert "openai/text-embedding-3-large" in EMBEDDING_MODEL_CONFIG
        assert "openai/text-embedding-3-small" in EMBEDDING_MODEL_CONFIG
        assert "gemini/gemini-embedding-2" in EMBEDDING_MODEL_CONFIG

    def test_alias_and_fullname_resolve_to_same_object(self):
        a = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        b = EMBEDDING_MODEL_CONFIG["openai/text-embedding-3-large"]
        assert a is b

    def test_openai_large_specs(self):
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        assert m.provider == "openai"
        assert m.supported_inputs == {"text"}
        assert m.max_dimensions == 3072
        assert m.default_dimensions == 3072
        assert m.recommended_dimensions == (256, 1024, 3072)
        assert m.max_input_tokens == 8192
        assert m.supports_aggregation is False

    def test_openai_small_specs(self):
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-small"]
        assert m.max_dimensions == 1536
        assert m.default_dimensions == 1536
        assert m.recommended_dimensions == (512, 1536)
        assert m.supported_inputs == {"text"}
        assert m.supports_aggregation is False

    def test_gemini_specs(self):
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        assert m.provider == "gemini"
        assert m.supported_inputs == {"text", "image", "audio", "video", "pdf"}
        assert m.max_dimensions == 3072
        assert m.default_dimensions == 3072
        assert m.recommended_dimensions == (768, 1536, 3072)
        assert m.max_input_tokens == 8192
        assert m.supports_aggregation is True


class TestResolveEmbeddingAlias:

    def test_known_alias(self):
        m = resolve_embedding_alias("openai-embedding-3-large")
        assert m.name == "openai/text-embedding-3-large"

    def test_full_name_lookup(self):
        m = resolve_embedding_alias("gemini/gemini-embedding-2")
        assert m.provider == "gemini"

    def test_unknown_alias_raises(self):
        with pytest.raises(RouterError) as exc:
            resolve_embedding_alias("not-a-real-model")
        assert exc.value.code == "INVALID_MODEL"
        assert "not-a-real-model" in exc.value.message
