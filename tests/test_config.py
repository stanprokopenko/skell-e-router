"""Tests for the config parameter across ask_ai and deep research functions.

All tests use mocks — no real API calls are made.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from skell_e_router.model_config import AIModel
from skell_e_router.utils import (
    _resolve_api_key,
    _redact_keys,
    _check_provider_key,
    check_environment_variables,
    ask_ai,
    RouterError,
    PROVIDER_ENV_KEY,
)
from skell_e_router.gemini_deep_research import (
    _check_api_key as dr_check_api_key,
    _get_client as dr_get_client,
    _redact_keys as dr_redact_keys,
    DeepResearchError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(provider: str) -> AIModel:
    """Create a minimal AIModel for testing."""
    return AIModel(
        name=f"{provider}/test-model",
        provider=provider,
        supports_thinking=False,
        supported_params={"stream"},
    )


FAKE_OPENAI_KEY = "sk-test-openai-key-abc123"
FAKE_GEMINI_KEY = "AIzaSy-test-gemini-key-xyz789"
FAKE_ANTHROPIC_KEY = "sk-ant-test-anthropic-key-def456"

FULL_CONFIG = {
    "openai_api_key": FAKE_OPENAI_KEY,
    "gemini_api_key": FAKE_GEMINI_KEY,
    "anthropic_api_key": FAKE_ANTHROPIC_KEY,
    "groq_api_key": "gsk-test-groq-key",
    "xai_api_key": "xai-test-key",
}


# ---------------------------------------------------------------------------
# _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:

    def test_returns_key_for_matching_provider(self):
        model = _make_model("openai")
        assert _resolve_api_key(model, {"openai_api_key": "sk-123"}) == "sk-123"

    def test_returns_none_when_config_is_none(self):
        model = _make_model("openai")
        assert _resolve_api_key(model, None) is None

    def test_returns_none_when_config_is_empty(self):
        model = _make_model("openai")
        assert _resolve_api_key(model, {}) is None

    def test_returns_none_when_provider_key_missing_from_config(self):
        model = _make_model("openai")
        assert _resolve_api_key(model, {"gemini_api_key": "x"}) is None

    @pytest.mark.parametrize("provider,config_key", [
        ("openai", "openai_api_key"),
        ("gemini", "gemini_api_key"),
        ("anthropic", "anthropic_api_key"),
        ("groq", "groq_api_key"),
        ("xai", "xai_api_key"),
    ])
    def test_all_providers_resolve_correctly(self, provider, config_key):
        model = _make_model(provider)
        config = {config_key: "test-key-value"}
        assert _resolve_api_key(model, config) == "test-key-value"

    def test_unknown_provider_returns_none(self):
        model = _make_model("some_new_provider")
        assert _resolve_api_key(model, {"some_new_provider_api_key": "x"}) is None


# ---------------------------------------------------------------------------
# _redact_keys
# ---------------------------------------------------------------------------

class TestRedactKeys:

    def test_redacts_key_value_from_message(self):
        config = {"openai_api_key": "sk-secret123"}
        msg = "Authentication failed for key sk-secret123 on server"
        assert "sk-secret123" not in _redact_keys(msg, config)
        assert "[REDACTED]" in _redact_keys(msg, config)

    def test_redacts_multiple_keys(self):
        config = {
            "openai_api_key": "sk-aaa",
            "gemini_api_key": "AIza-bbb",
        }
        msg = "Keys sk-aaa and AIza-bbb are invalid"
        result = _redact_keys(msg, config)
        assert "sk-aaa" not in result
        assert "AIza-bbb" not in result

    def test_returns_original_when_config_is_none(self):
        msg = "some error"
        assert _redact_keys(msg, None) == msg

    def test_returns_original_when_config_is_empty(self):
        msg = "some error"
        assert _redact_keys(msg, {}) == msg

    def test_ignores_non_string_config_values(self):
        config = {"timeout": 30, "openai_api_key": "sk-abc"}
        msg = "Error with sk-abc and 30"
        result = _redact_keys(msg, config)
        assert "sk-abc" not in result
        # Non-string value "30" should NOT be redacted
        assert "30" in result


# ---------------------------------------------------------------------------
# _check_provider_key
# ---------------------------------------------------------------------------

class TestCheckProviderKey:

    def test_passes_when_key_in_config(self):
        model = _make_model("openai")
        # Should not raise
        _check_provider_key(model, config={"openai_api_key": "sk-x"})

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}, clear=False)
    def test_passes_when_key_in_env(self):
        model = _make_model("openai")
        _check_provider_key(model, config=None)

    def test_raises_when_key_missing_everywhere(self):
        model = _make_model("openai")
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RouterError) as exc_info:
                _check_provider_key(model, config=None)
            assert exc_info.value.code == "MISSING_ENV"
            assert "openai" in exc_info.value.details.get("provider", "")

    def test_config_takes_priority_over_missing_env(self):
        """Key in config is sufficient even if env var is missing."""
        model = _make_model("anthropic")
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise — config supplies the key
            _check_provider_key(model, config={"anthropic_api_key": "sk-ant-x"})

    def test_unknown_provider_passes_silently(self):
        model = _make_model("some_future_provider")
        _check_provider_key(model, config=None)


# ---------------------------------------------------------------------------
# check_environment_variables (public, backward-compat)
# ---------------------------------------------------------------------------

class TestCheckEnvironmentVariables:

    @patch.dict(os.environ, {k: "v" for k in PROVIDER_ENV_KEY.values()}, clear=False)
    def test_passes_when_all_env_vars_set(self):
        assert check_environment_variables() is True

    def test_raises_when_env_vars_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RouterError) as exc_info:
                check_environment_variables()
            assert exc_info.value.code == "MISSING_ENV"

    def test_config_satisfies_some_keys(self):
        """Keys in config offset the requirement for env vars."""
        partial_env = {
            "OPENAI_API_KEY": "x",
            "GROQ_API_KEY": "x",
            "XAI_API_KEY": "x",
        }
        config = {
            "gemini_api_key": "x",
            "anthropic_api_key": "x",
        }
        with patch.dict(os.environ, partial_env, clear=True):
            assert check_environment_variables(config=config) is True

    def test_config_satisfies_all_keys(self):
        """If every key is in config, no env vars are needed at all."""
        with patch.dict(os.environ, {}, clear=True):
            assert check_environment_variables(config=FULL_CONFIG) is True


# ---------------------------------------------------------------------------
# ask_ai with config
# ---------------------------------------------------------------------------

def _make_litellm_response(content="Hello world"):
    """Build a minimal mock that looks like a litellm completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    message.function_call = None
    message.provider_specific_fields = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30
    usage.completion_tokens_details = None

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "openai/gpt-5"
    # Gemini-specific attributes
    response.vertex_ai_grounding_metadata = None
    response.vertex_ai_safety_results = None
    return response


class TestAskAiWithConfig:

    @patch("skell_e_router.utils.litellm")
    def test_passes_api_key_to_litellm(self, mock_litellm):
        mock_litellm.completion.return_value = _make_litellm_response()
        mock_litellm.drop_params = True

        config = {"openai_api_key": FAKE_OPENAI_KEY}
        ask_ai("gpt-5", "Hi", config=config)

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == FAKE_OPENAI_KEY

    @patch("skell_e_router.utils.litellm")
    def test_no_api_key_without_config(self, mock_litellm):
        mock_litellm.completion.return_value = _make_litellm_response()
        mock_litellm.drop_params = True

        # Need env vars when no config
        env = {k: "v" for k in PROVIDER_ENV_KEY.values()}
        with patch.dict(os.environ, env, clear=False):
            ask_ai("gpt-5", "Hi")

        call_kwargs = mock_litellm.completion.call_args
        # api_key should not be in the call at all
        assert "api_key" not in call_kwargs.kwargs

    @patch("skell_e_router.utils.litellm")
    def test_only_provider_key_needed(self, mock_litellm):
        """Calling an OpenAI model only requires the OpenAI key, not all five."""
        mock_litellm.completion.return_value = _make_litellm_response()
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {}, clear=True):
            result = ask_ai(
                "gpt-5", "Hi",
                config={"openai_api_key": FAKE_OPENAI_KEY},
            )
        assert result == "Hello world"

    @patch("skell_e_router.utils.litellm")
    def test_rich_response_with_config(self, mock_litellm):
        mock_litellm.completion.return_value = _make_litellm_response("test content")
        mock_litellm.completion_cost.return_value = 0.001
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {}, clear=True):
            resp = ask_ai(
                "gpt-5", "Hi",
                rich_response=True,
                config={"openai_api_key": FAKE_OPENAI_KEY},
            )
        assert resp.content == "test content"

    @patch("skell_e_router.utils.litellm")
    def test_error_message_redacts_key(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception(
            f"Invalid API key: {FAKE_OPENAI_KEY}"
        )
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RouterError) as exc_info:
                ask_ai(
                    "gpt-5", "Hi",
                    config={"openai_api_key": FAKE_OPENAI_KEY},
                )
        # The key must NOT appear in the error message
        assert FAKE_OPENAI_KEY not in exc_info.value.message
        assert "[REDACTED]" in exc_info.value.message

    def test_missing_provider_key_raises(self):
        """Config with the wrong provider's key still raises for the missing one."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RouterError) as exc_info:
                ask_ai(
                    "gpt-5", "Hi",
                    config={"gemini_api_key": "x"},
                )
            assert exc_info.value.code == "MISSING_ENV"

    @patch("skell_e_router.utils.litellm")
    def test_backward_compat_no_config(self, mock_litellm):
        """Existing callers that don't pass config still work via env vars."""
        mock_litellm.completion.return_value = _make_litellm_response("compat")
        mock_litellm.drop_params = True

        env = {k: "v" for k in PROVIDER_ENV_KEY.values()}
        with patch.dict(os.environ, env, clear=False):
            result = ask_ai("gpt-5", "Hi")
        assert result == "compat"


# ---------------------------------------------------------------------------
# Deep Research: _check_api_key, _get_client
# ---------------------------------------------------------------------------

class TestDeepResearchCheckApiKey:

    def test_passes_when_key_in_config(self):
        dr_check_api_key(config={"gemini_api_key": "x"})

    @patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False)
    def test_passes_when_key_in_env(self):
        dr_check_api_key(config=None)

    def test_raises_when_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(DeepResearchError) as exc_info:
                dr_check_api_key(config=None)
            assert exc_info.value.code == "MISSING_API_KEY"


class TestDeepResearchGetClient:

    @patch("skell_e_router.gemini_deep_research.genai")
    @patch("skell_e_router.gemini_deep_research.GENAI_AVAILABLE", True)
    def test_passes_api_key_from_config(self, mock_genai):
        mock_genai.Client.return_value = MagicMock()
        dr_get_client(config={"gemini_api_key": FAKE_GEMINI_KEY})
        mock_genai.Client.assert_called_once_with(api_key=FAKE_GEMINI_KEY)

    @patch("skell_e_router.gemini_deep_research.genai")
    @patch("skell_e_router.gemini_deep_research.GENAI_AVAILABLE", True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False)
    def test_no_api_key_arg_without_config(self, mock_genai):
        mock_genai.Client.return_value = MagicMock()
        dr_get_client(config=None)
        mock_genai.Client.assert_called_once_with()


# ---------------------------------------------------------------------------
# Deep Research: _redact_keys
# ---------------------------------------------------------------------------

class TestDeepResearchRedactKeys:

    def test_redacts_gemini_key(self):
        config = {"gemini_api_key": FAKE_GEMINI_KEY}
        msg = f"Auth error: {FAKE_GEMINI_KEY}"
        result = dr_redact_keys(msg, config)
        assert FAKE_GEMINI_KEY not in result
        assert "[REDACTED]" in result

    def test_noop_without_config(self):
        assert dr_redact_keys("hello", None) == "hello"


# ---------------------------------------------------------------------------
# Integration-style: config flows end-to-end through ask_deep_research
# ---------------------------------------------------------------------------

class TestAskDeepResearchConfig:

    @patch("skell_e_router.gemini_deep_research.genai")
    @patch("skell_e_router.gemini_deep_research.GENAI_AVAILABLE", True)
    def test_config_reaches_genai_client(self, mock_genai):
        """ask_deep_research(config=...) passes the key through to genai.Client."""
        # Set up mock interaction
        mock_interaction = MagicMock()
        mock_interaction.id = "inter-123"
        mock_interaction.status = "completed"
        mock_interaction.error = None
        mock_interaction.outputs = []
        mock_interaction.usage_metadata = None
        mock_interaction.citations = None

        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction
        mock_client.interactions.get.return_value = mock_interaction
        mock_genai.Client.return_value = mock_client

        from skell_e_router.gemini_deep_research import ask_deep_research

        with patch.dict(os.environ, {}, clear=True):
            ask_deep_research(
                "Test query",
                config={"gemini_api_key": FAKE_GEMINI_KEY},
                resolve_citations=False,
            )

        mock_genai.Client.assert_called_once_with(api_key=FAKE_GEMINI_KEY)

    @patch("skell_e_router.gemini_deep_research.genai")
    @patch("skell_e_router.gemini_deep_research.GENAI_AVAILABLE", True)
    def test_error_redacts_key(self, mock_genai):
        """If ask_deep_research raises, the key is scrubbed from the message."""
        mock_genai.Client.side_effect = Exception(
            f"Bad key: {FAKE_GEMINI_KEY}"
        )

        from skell_e_router.gemini_deep_research import ask_deep_research

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(DeepResearchError) as exc_info:
                ask_deep_research(
                    "Test query",
                    config={"gemini_api_key": FAKE_GEMINI_KEY},
                )
        assert FAKE_GEMINI_KEY not in exc_info.value.message
        assert "[REDACTED]" in exc_info.value.message
