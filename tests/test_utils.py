"""Tests for utils.py — message construction, retry logic, model params, ask_ai."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from skell_e_router.model_config import AIModel, MODEL_CONFIG
from skell_e_router.utils import (
    _construct_messages,
    _encode_image,
    _resolve_api_key,
    _redact_keys,
    _extract_status_and_headers,
    _parse_retry_after_seconds,
    _is_quota_related,
    _is_retryable_exception,
    _retry_after_wait,
    _handle_model_specific_params,
    _build_ai_response,
    resolve_model_alias,
    check_environment_variables,
    _check_provider_key,
    ask_ai,
    RouterError,
    PROVIDER_ENV_KEY,
)

from tests.helpers import (
    make_model,
    make_litellm_response,
    FAKE_OPENAI_KEY,
    FAKE_GEMINI_KEY,
    FULL_CONFIG,
)


# ---------------------------------------------------------------------------
# _construct_messages
# ---------------------------------------------------------------------------

class TestConstructMessages:

    def test_string_input_no_system(self):
        msgs = _construct_messages("hello")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_string_input_with_system(self):
        msgs = _construct_messages("hello", system_message="You are helpful.")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1] == {"role": "user", "content": "hello"}

    def test_list_input(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        msgs = _construct_messages(history)
        assert msgs == history

    def test_list_input_with_system(self):
        history = [{"role": "user", "content": "hi"}]
        msgs = _construct_messages(history, system_message="sys")
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_invalid_list_raises(self):
        with pytest.raises(RouterError) as exc_info:
            _construct_messages([{"bad": "format"}])
        assert exc_info.value.code == "INVALID_INPUT"

    def test_invalid_type_raises(self):
        with pytest.raises(RouterError) as exc_info:
            _construct_messages(12345)
        assert exc_info.value.code == "INVALID_INPUT"

    def test_empty_list_accepted(self):
        """An empty list is valid (no messages to validate)."""
        msgs = _construct_messages([])
        assert msgs == []

    def test_mixed_invalid_list(self):
        """If any item in the list is malformed, it should raise."""
        with pytest.raises(RouterError):
            _construct_messages([
                {"role": "user", "content": "ok"},
                {"only_role": "user"},
            ])

    def test_string_input_with_images(self):
        msgs = _construct_messages("describe this", images=["https://example.com/img.jpg"])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe this"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/img.jpg"

    def test_string_input_with_multiple_images(self):
        imgs = ["https://example.com/a.jpg", "data:image/png;base64,abc123"]
        msgs = _construct_messages("compare these", images=imgs)
        content = msgs[0]["content"]
        assert len(content) == 3  # 1 text + 2 images
        assert content[1]["image_url"]["url"] == "https://example.com/a.jpg"
        assert content[2]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_string_input_with_images_and_system(self):
        msgs = _construct_messages("describe", system_message="Be detailed", images=["https://img.jpg"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert isinstance(msgs[1]["content"], list)

    def test_images_with_list_input_raises(self):
        with pytest.raises(RouterError) as exc_info:
            _construct_messages(
                [{"role": "user", "content": "hi"}],
                images=["https://example.com/img.jpg"]
            )
        assert exc_info.value.code == "INVALID_INPUT"

    def test_images_none_no_effect(self):
        """Passing images=None should behave identically to not passing it."""
        msgs = _construct_messages("hello", images=None)
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_images_empty_list_no_effect(self):
        """An empty images list should not create multimodal content."""
        msgs = _construct_messages("hello", images=[])
        assert msgs == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# _encode_image
# ---------------------------------------------------------------------------

class TestEncodeImage:

    def test_url_http(self):
        result = _encode_image("https://example.com/photo.jpg")
        assert result == {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}

    def test_url_http_plain(self):
        result = _encode_image("http://example.com/photo.jpg")
        assert result == {"type": "image_url", "image_url": {"url": "http://example.com/photo.jpg"}}

    def test_data_uri(self):
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"
        result = _encode_image(data_uri)
        assert result == {"type": "image_url", "image_url": {"url": data_uri}}

    def test_file_path(self):
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
            tmp_path = f.name
        try:
            result = _encode_image(tmp_path)
            assert result["type"] == "image_url"
            assert result["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(tmp_path)

    def test_file_path_jpeg(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
            tmp_path = f.name
        try:
            result = _encode_image(tmp_path)
            assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp_path)

    def test_file_not_found_raises(self):
        with pytest.raises(RouterError) as exc_info:
            _encode_image("/nonexistent/path/image.png")
        assert exc_info.value.code == "INVALID_INPUT"
        assert "not found" in exc_info.value.message


# ---------------------------------------------------------------------------
# _extract_status_and_headers
# ---------------------------------------------------------------------------

class TestExtractStatusAndHeaders:

    def test_status_from_status_code_attr(self):
        exc = Exception("err")
        exc.status_code = 429
        exc.headers = {"Retry-After": "5"}
        status, headers = _extract_status_and_headers(exc)
        assert status == 429
        assert headers["Retry-After"] == "5"

    def test_status_from_response_attr(self):
        exc = Exception("err")
        resp = MagicMock()
        resp.status_code = 503
        resp.headers = {"Retry-After": "10"}
        exc.response = resp
        status, headers = _extract_status_and_headers(exc)
        assert status == 503

    def test_no_status_info(self):
        exc = Exception("plain error")
        status, headers = _extract_status_and_headers(exc)
        assert status is None
        assert headers == {}


# ---------------------------------------------------------------------------
# _parse_retry_after_seconds
# ---------------------------------------------------------------------------

class TestParseRetryAfterSeconds:

    def test_valid_retry_after(self):
        assert _parse_retry_after_seconds({"Retry-After": "30"}) == 30.0

    def test_case_insensitive_header(self):
        assert _parse_retry_after_seconds({"retry-after": "5"}) == 5.0

    def test_float_value(self):
        assert _parse_retry_after_seconds({"Retry-After": "2.5"}) == 2.5

    def test_missing_header(self):
        assert _parse_retry_after_seconds({}) is None

    def test_non_numeric_value(self):
        assert _parse_retry_after_seconds({"Retry-After": "Thu, 01 Dec 2025"}) is None

    def test_negative_value(self):
        assert _parse_retry_after_seconds({"Retry-After": "-1"}) is None

    def test_none_input(self):
        assert _parse_retry_after_seconds(None) is None

    def test_zero_value(self):
        assert _parse_retry_after_seconds({"Retry-After": "0"}) == 0.0


# ---------------------------------------------------------------------------
# _is_quota_related
# ---------------------------------------------------------------------------

class TestIsQuotaRelated:

    def test_quota_code_attribute(self):
        exc = Exception("err")
        exc.code = "insufficient_quota"
        assert _is_quota_related(exc) is True

    def test_quota_in_error_dict(self):
        exc = Exception("err")
        exc.code = None
        exc.error = {"code": "quota_exceeded"}
        assert _is_quota_related(exc) is True

    def test_not_quota(self):
        exc = Exception("err")
        exc.code = "invalid_request"
        assert _is_quota_related(exc) is False

    def test_no_code_no_error(self):
        exc = Exception("plain error")
        assert _is_quota_related(exc) is False

    def test_quota_partial_match(self):
        exc = Exception("err")
        exc.code = "resource_quota_limit"
        assert _is_quota_related(exc) is True


# ---------------------------------------------------------------------------
# _is_retryable_exception
# ---------------------------------------------------------------------------

class TestIsRetryableException:

    def test_timeout_exception(self):
        exc = type("TimeoutError", (Exception,), {})()
        assert _is_retryable_exception(exc) is True

    def test_connection_exception(self):
        exc = type("ConnectionError", (Exception,), {})()
        assert _is_retryable_exception(exc) is True

    def test_500_retryable(self):
        exc = Exception("server error")
        exc.status_code = 500
        exc.headers = {}
        assert _is_retryable_exception(exc) is True

    def test_502_retryable(self):
        exc = Exception("bad gateway")
        exc.status_code = 502
        exc.headers = {}
        assert _is_retryable_exception(exc) is True

    def test_503_retryable(self):
        exc = Exception("unavailable")
        exc.status_code = 503
        exc.headers = {}
        assert _is_retryable_exception(exc) is True

    def test_503_with_huge_retry_after_not_retryable(self):
        exc = Exception("unavailable")
        exc.status_code = 503
        exc.headers = {"Retry-After": "300"}
        assert _is_retryable_exception(exc) is False

    def test_429_retryable(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        exc.headers = {"Retry-After": "5"}
        assert _is_retryable_exception(exc) is True

    def test_429_quota_not_retryable(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        exc.headers = {}
        exc.code = "insufficient_quota"
        assert _is_retryable_exception(exc) is False

    def test_429_with_huge_retry_after_not_retryable(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        exc.headers = {"Retry-After": "200"}
        assert _is_retryable_exception(exc) is False

    def test_400_not_retryable(self):
        exc = Exception("bad request")
        exc.status_code = 400
        exc.headers = {}
        assert _is_retryable_exception(exc) is False

    def test_401_not_retryable(self):
        exc = Exception("unauthorized")
        exc.status_code = 401
        exc.headers = {}
        assert _is_retryable_exception(exc) is False

    def test_plain_exception_not_retryable(self):
        exc = Exception("something went wrong")
        assert _is_retryable_exception(exc) is False


# ---------------------------------------------------------------------------
# _retry_after_wait
# ---------------------------------------------------------------------------

class TestRetryAfterWait:

    def _make_retry_state(self, exc, status_code=None, headers=None):
        state = MagicMock()
        state.attempt_number = 1
        if exc:
            if status_code:
                exc.status_code = status_code
            if headers:
                exc.headers = headers
            outcome = MagicMock()
            outcome.exception.return_value = exc
            state.outcome = outcome
        else:
            state.outcome = None
        return state

    def test_honors_retry_after_on_429(self):
        exc = Exception("rate limited")
        state = self._make_retry_state(exc, 429, {"Retry-After": "10"})
        wait = _retry_after_wait(state)
        assert wait == 10.0

    def test_caps_at_120(self):
        exc = Exception("rate limited")
        state = self._make_retry_state(exc, 429, {"Retry-After": "200"})
        wait = _retry_after_wait(state)
        assert wait == 120.0

    def test_honors_retry_after_on_503(self):
        exc = Exception("unavailable")
        state = self._make_retry_state(exc, 503, {"Retry-After": "15"})
        wait = _retry_after_wait(state)
        assert wait == 15.0

    def test_fallback_for_other_status(self):
        exc = Exception("err")
        state = self._make_retry_state(exc, 500, {})
        wait = _retry_after_wait(state)
        # Fallback returns a float (from random exponential)
        assert isinstance(wait, (int, float))
        assert wait >= 0

    def test_no_exception_uses_fallback(self):
        state = self._make_retry_state(None)
        wait = _retry_after_wait(state)
        assert isinstance(wait, (int, float))
        assert wait >= 0


# ---------------------------------------------------------------------------
# resolve_model_alias
# ---------------------------------------------------------------------------

class TestResolveModelAlias:

    def test_valid_alias(self):
        model = resolve_model_alias("gpt-5")
        assert model.name == "openai/gpt-5"

    def test_full_name(self):
        model = resolve_model_alias("openai/gpt-5")
        assert model.name == "openai/gpt-5"

    def test_invalid_alias_raises(self):
        with pytest.raises(RouterError) as exc_info:
            resolve_model_alias("nonexistent-model-xyz")
        assert exc_info.value.code == "INVALID_MODEL"

    def test_empty_string_raises(self):
        with pytest.raises(RouterError):
            resolve_model_alias("")


# ---------------------------------------------------------------------------
# _handle_model_specific_params
# ---------------------------------------------------------------------------

class TestHandleModelSpecificParams:

    # --- budget_tokens to thinking dict (Anthropic path) ---

    def test_budget_tokens_to_thinking_dict(self):
        model = make_model(
            provider="anthropic",
            supported_params={"budget_tokens", "thinking", "temperature", "stream"},
        )
        kwargs = {"budget_tokens": 2048, "temperature": 0.7}
        result = _handle_model_specific_params(model, kwargs)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        # Anthropic: when thinking enabled, temperature must be 1
        assert result["temperature"] == 1

    def test_budget_tokens_zero_disables_thinking(self):
        model = make_model(
            provider="anthropic",
            supported_params={"budget_tokens", "thinking", "stream"},
        )
        kwargs = {"budget_tokens": 0}
        result = _handle_model_specific_params(model, kwargs)
        assert result["thinking"]["type"] == "disabled"

    # --- budget_tokens to reasoning_effort (OpenAI path) ---

    def test_budget_tokens_to_reasoning_effort_low(self):
        model = make_model(
            provider="openai",
            supported_params={"reasoning_effort", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        kwargs = {"budget_tokens": 512}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "low"

    def test_budget_tokens_to_reasoning_effort_medium(self):
        model = make_model(
            provider="openai",
            supported_params={"reasoning_effort", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        kwargs = {"budget_tokens": 2048}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "medium"

    def test_budget_tokens_to_reasoning_effort_high(self):
        model = make_model(
            provider="openai",
            supported_params={"reasoning_effort", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        kwargs = {"budget_tokens": 4096}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "high"

    def test_budget_tokens_zero_to_minimal(self):
        model = make_model(
            provider="openai",
            supported_params={"reasoning_effort", "stream"},
            accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        )
        kwargs = {"budget_tokens": 0}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "minimal"

    def test_budget_tokens_zero_to_low_without_minimal(self):
        model = make_model(
            provider="openai",
            supported_params={"reasoning_effort", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        kwargs = {"budget_tokens": 0}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "low"

    def test_budget_tokens_invalid_type_raises(self):
        model = make_model(supported_params={"budget_tokens", "stream"})
        with pytest.raises(RouterError) as exc_info:
            _handle_model_specific_params(model, {"budget_tokens": "not_a_number"})
        assert exc_info.value.code == "INVALID_PARAM"

    # --- reasoning_effort to thinking dict (reverse mapping for Anthropic) ---

    def test_reasoning_effort_to_thinking_for_anthropic(self):
        model = make_model(
            provider="anthropic",
            supported_params={"budget_tokens", "thinking", "temperature", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        kwargs = {"reasoning_effort": "medium"}
        result = _handle_model_specific_params(model, kwargs)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "reasoning_effort" not in result

    def test_reasoning_effort_invalid_value_raises(self):
        model = make_model(
            provider="anthropic",
            supported_params={"budget_tokens", "thinking", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        with pytest.raises(RouterError) as exc_info:
            _handle_model_specific_params(model, {"reasoning_effort": "ultra"})
        assert exc_info.value.code == "INVALID_PARAM"

    # --- Gemini safety settings ---

    def test_gemini_safety_settings_added(self):
        model = make_model(
            provider="gemini",
            supported_params={"safety_settings", "temperature", "stream"},
        )
        kwargs = {"temperature": 0.5}
        result = _handle_model_specific_params(model, kwargs)
        assert "safety_settings" in result
        assert len(result["safety_settings"]) == 4

    # --- Anthropic thinking temperature rule ---

    def test_anthropic_thinking_forces_temperature_1(self):
        model = make_model(
            provider="anthropic",
            supported_params={"thinking", "temperature", "stream"},
        )
        kwargs = {"thinking": {"type": "enabled", "budget_tokens": 1024}, "temperature": 0.5}
        result = _handle_model_specific_params(model, kwargs)
        assert result["temperature"] == 1

    def test_anthropic_thinking_top_p_clamped(self):
        model = make_model(
            provider="anthropic",
            supported_params={"thinking", "temperature", "top_p", "stream"},
        )
        kwargs = {"thinking": {"type": "enabled", "budget_tokens": 1024}, "top_p": 0.5}
        result = _handle_model_specific_params(model, kwargs)
        assert result["top_p"] == 1

    def test_anthropic_thinking_top_p_high_kept(self):
        model = make_model(
            provider="anthropic",
            supported_params={"thinking", "temperature", "top_p", "stream"},
        )
        kwargs = {"thinking": {"type": "enabled", "budget_tokens": 1024}, "top_p": 0.99}
        result = _handle_model_specific_params(model, kwargs)
        assert result["top_p"] == 0.99

    # --- Groq compound headers ---

    def test_groq_compound_adds_headers(self):
        model = make_model(
            provider="groq",
            name="groq/groq/compound",
            supported_params={"extra_headers", "compound_custom", "stream"},
        )
        kwargs = {}
        result = _handle_model_specific_params(model, kwargs)
        assert result.get("extra_headers", {}).get("Groq-Model-Version") == "latest"
        assert "compound_custom" in result

    def test_groq_compound_preserves_existing_headers(self):
        model = make_model(
            provider="groq",
            name="groq/groq/compound-mini",
            supported_params={"extra_headers", "compound_custom", "stream"},
        )
        kwargs = {"extra_headers": {"Custom-Header": "val"}}
        result = _handle_model_specific_params(model, kwargs)
        assert result["extra_headers"]["Custom-Header"] == "val"
        assert result["extra_headers"]["Groq-Model-Version"] == "latest"

    # --- Modalities auto-injection for image generation models ---

    def test_auto_injects_modalities_for_image_model(self):
        model = make_model(
            provider="gemini",
            supported_params={"modalities", "safety_settings", "temperature", "stream"},
        )
        kwargs = {"temperature": 0.7}
        result = _handle_model_specific_params(model, kwargs)
        assert result["modalities"] == ["text", "image"]

    def test_user_modalities_not_overridden(self):
        model = make_model(
            provider="gemini",
            supported_params={"modalities", "safety_settings", "temperature", "stream"},
        )
        kwargs = {"temperature": 0.7, "modalities": ["text"]}
        result = _handle_model_specific_params(model, kwargs)
        assert result["modalities"] == ["text"]

    def test_modalities_not_injected_when_not_in_supported_params(self):
        model = make_model(
            provider="gemini",
            supported_params={"safety_settings", "temperature", "stream"},
        )
        kwargs = {"temperature": 0.7}
        result = _handle_model_specific_params(model, kwargs)
        assert "modalities" not in result

    # --- Opus 4.6 adaptive thinking (reasoning_effort passthrough) ---

    def test_opus46_reasoning_effort_passes_through(self):
        """Opus 4.6-style model: reasoning_effort stays in kwargs, not converted to budget_tokens."""
        model = make_model(
            provider="anthropic",
            supported_params={"reasoning_effort", "budget_tokens", "thinking", "temperature", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high", "max"},
        )
        kwargs = {"reasoning_effort": "high"}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "high"
        assert result["thinking"] == {"type": "adaptive"}

    def test_opus46_reasoning_effort_max(self):
        """Opus 4.6 supports the 'max' effort level."""
        model = make_model(
            provider="anthropic",
            supported_params={"reasoning_effort", "budget_tokens", "thinking", "temperature", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high", "max"},
        )
        kwargs = {"reasoning_effort": "max"}
        result = _handle_model_specific_params(model, kwargs)
        assert result["reasoning_effort"] == "max"
        assert result["thinking"] == {"type": "adaptive"}

    def test_opus46_adaptive_thinking_auto_injected(self):
        """When reasoning_effort is used without explicit thinking dict, adaptive thinking is injected."""
        model = make_model(
            provider="anthropic",
            supported_params={"reasoning_effort", "budget_tokens", "thinking", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high", "max"},
        )
        kwargs = {"reasoning_effort": "low"}
        result = _handle_model_specific_params(model, kwargs)
        assert result["thinking"] == {"type": "adaptive"}

    def test_opus46_explicit_thinking_not_overridden(self):
        """User-provided thinking dict is preserved, not replaced with adaptive."""
        model = make_model(
            provider="anthropic",
            supported_params={"reasoning_effort", "budget_tokens", "thinking", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high", "max"},
        )
        kwargs = {"reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": 2048}}
        result = _handle_model_specific_params(model, kwargs)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert result["reasoning_effort"] == "low"

    def test_opus46_invalid_effort_raises(self):
        """Invalid reasoning_effort value raises even on passthrough path."""
        model = make_model(
            provider="anthropic",
            supported_params={"reasoning_effort", "budget_tokens", "thinking", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high", "max"},
        )
        with pytest.raises(RouterError) as exc_info:
            _handle_model_specific_params(model, {"reasoning_effort": "ultra"})
        assert exc_info.value.code == "INVALID_PARAM"

    def test_opus46_no_thinking_without_reasoning_effort(self):
        """Without reasoning_effort, adaptive thinking is NOT auto-injected."""
        model = make_model(
            provider="anthropic",
            supported_params={"reasoning_effort", "budget_tokens", "thinking", "temperature", "stream"},
            accepted_reasoning_efforts={"low", "medium", "high", "max"},
        )
        kwargs = {"temperature": 0.7}
        result = _handle_model_specific_params(model, kwargs)
        assert "thinking" not in result

    # --- Parameter filtering ---

    def test_unsupported_params_filtered(self):
        model = make_model(supported_params={"temperature", "stream"})
        kwargs = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 100}
        result = _handle_model_specific_params(model, kwargs)
        assert "temperature" in result
        assert "stream" not in result  # wasn't in kwargs
        assert "top_p" not in result
        assert "max_tokens" not in result


# ---------------------------------------------------------------------------
# _build_ai_response
# ---------------------------------------------------------------------------

class TestBuildAIResponse:

    def test_builds_complete_response(self):
        mock_resp = make_litellm_response(
            content="test output",
            model="openai/gpt-5",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
        )
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.return_value = 0.01
            result = _build_ai_response(mock_resp, request_duration_s=1.5)

        assert result.content == "test output"
        assert result.model == "openai/gpt-5"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 100
        assert result.total_tokens == 150
        assert result.cost == 0.01
        assert result.duration_seconds == 1.5

    def test_handles_cost_exception(self):
        mock_resp = make_litellm_response()
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.side_effect = Exception("unknown model")
            result = _build_ai_response(mock_resp)

        assert result.cost is None

    def test_extracts_grounding_metadata(self):
        mock_resp = make_litellm_response(grounding_metadata={"web": True})
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.return_value = 0.0
            result = _build_ai_response(mock_resp)

        assert result.grounding_metadata == {"web": True}

    def test_extracts_safety_ratings(self):
        ratings = [{"category": "HARM_CATEGORY_HATE", "probability": "LOW"}]
        mock_resp = make_litellm_response(safety_ratings=ratings)
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.return_value = 0.0
            result = _build_ai_response(mock_resp)

        assert result.safety_ratings == ratings

    def test_extracts_images(self):
        img_data = [{"image_url": {"url": "data:image/png;base64,abc"}, "index": 0, "type": "image_url"}]
        mock_resp = make_litellm_response(images=img_data)
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.return_value = 0.0
            result = _build_ai_response(mock_resp)
        assert result.images == img_data

    def test_images_none_when_not_present(self):
        mock_resp = make_litellm_response()
        mock_resp.choices[0].message.images = None
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.return_value = 0.0
            result = _build_ai_response(mock_resp)
        assert result.images is None

    def test_empty_choices(self):
        mock_resp = MagicMock()
        mock_resp.choices = []
        mock_resp.usage = None
        mock_resp.model = "unknown"
        mock_resp.vertex_ai_grounding_metadata = None
        mock_resp.vertex_ai_safety_results = None
        with patch("skell_e_router.utils.litellm") as mock_litellm:
            mock_litellm.completion_cost.return_value = 0.0
            result = _build_ai_response(mock_resp)

        assert result.content == ""


# ---------------------------------------------------------------------------
# ask_ai
# ---------------------------------------------------------------------------

class TestAskAi:

    @patch("skell_e_router.utils.litellm")
    def test_returns_string_by_default(self, mock_litellm):
        mock_litellm.completion.return_value = make_litellm_response("output text")
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-5", "Hi")

        assert result == "output text"
        assert isinstance(result, str)

    @patch("skell_e_router.utils.litellm")
    def test_returns_ai_response_when_rich(self, mock_litellm):
        mock_litellm.completion.return_value = make_litellm_response("rich output")
        mock_litellm.completion_cost.return_value = 0.002
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-5", "Hi", rich_response=True)

        assert result.content == "rich output"
        assert result.model == "openai/gpt-5"

    @patch("skell_e_router.utils.litellm")
    def test_passes_system_message(self, mock_litellm):
        mock_litellm.completion.return_value = make_litellm_response()
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            ask_ai("gpt-5", "Hi", system_message="Be concise")

        call_kwargs = mock_litellm.completion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"

    @patch("skell_e_router.utils.litellm")
    def test_passes_api_key_from_config(self, mock_litellm):
        mock_litellm.completion.return_value = make_litellm_response()
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {}, clear=True):
            ask_ai("gpt-5", "Hi", config={"openai_api_key": FAKE_OPENAI_KEY})

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == FAKE_OPENAI_KEY

    def test_invalid_model_raises(self):
        with pytest.raises(RouterError) as exc_info:
            ask_ai("totally-fake-model", "Hi")
        assert exc_info.value.code == "INVALID_MODEL"

    def test_missing_provider_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RouterError) as exc_info:
                ask_ai("gpt-5", "Hi")
            assert exc_info.value.code == "MISSING_ENV"

    @patch("skell_e_router.utils.litellm")
    def test_invalid_verbosity_falls_back(self, mock_litellm):
        """Invalid verbosity should fall back to 'response' without crashing."""
        mock_litellm.completion.return_value = make_litellm_response()
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-5", "Hi", verbosity="invalid_level")

        assert isinstance(result, str)

    @patch("skell_e_router.utils.litellm")
    def test_provider_error_wraps_exception(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("API failed")
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            with pytest.raises(RouterError) as exc_info:
                ask_ai("gpt-5", "Hi")
            assert exc_info.value.code == "PROVIDER_ERROR"
            assert "API failed" in exc_info.value.message

    @patch("skell_e_router.utils.litellm")
    def test_error_redacts_api_key(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception(f"Bad key: {FAKE_OPENAI_KEY}")
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RouterError) as exc_info:
                ask_ai("gpt-5", "Hi", config={"openai_api_key": FAKE_OPENAI_KEY})
            assert FAKE_OPENAI_KEY not in exc_info.value.message
            assert "[REDACTED]" in exc_info.value.message

    @patch("skell_e_router.utils.litellm")
    def test_list_input_works(self, mock_litellm):
        mock_litellm.completion.return_value = make_litellm_response("reply")
        mock_litellm.drop_params = True

        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "follow-up"},
        ]
        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-5", history)
        assert result == "reply"

    @patch("skell_e_router.utils.litellm")
    def test_images_passed_to_messages(self, mock_litellm):
        mock_litellm.completion.return_value = make_litellm_response("I see a cat")
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-4o", "What is in this image?", images=["https://example.com/cat.jpg"])

        call_kwargs = mock_litellm.completion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"

    @patch("skell_e_router.utils.litellm")
    def test_rich_response_includes_images(self, mock_litellm):
        img_data = [{"image_url": {"url": "data:image/png;base64,abc"}, "index": 0, "type": "image_url"}]
        mock_litellm.completion.return_value = make_litellm_response(
            content="Here is your image",
            images=img_data,
        )
        mock_litellm.completion_cost.return_value = 0.001
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-5", "Generate an image", rich_response=True)

        assert result.images == img_data

    @patch("skell_e_router.utils.litellm")
    def test_plain_response_returns_text_only(self, mock_litellm):
        img_data = [{"image_url": {"url": "data:image/png;base64,abc"}, "index": 0, "type": "image_url"}]
        mock_litellm.completion.return_value = make_litellm_response(
            content="Here is your image",
            images=img_data,
        )
        mock_litellm.drop_params = True

        with patch.dict(os.environ, {v: "x" for v in PROVIDER_ENV_KEY.values()}):
            result = ask_ai("gpt-5", "Generate an image")

        assert isinstance(result, str)
        assert result == "Here is your image"


# ---------------------------------------------------------------------------
# RouterError
# ---------------------------------------------------------------------------

class TestRouterError:

    def test_has_code_and_message(self):
        err = RouterError(code="TEST_CODE", message="Something broke")
        assert err.code == "TEST_CODE"
        assert err.message == "Something broke"
        assert err.details == {}

    def test_str_includes_code(self):
        err = RouterError(code="ERR", message="msg")
        assert "ERR" in str(err)
        assert "msg" in str(err)

    def test_details_stored(self):
        err = RouterError(code="ERR", message="msg", details={"key": "val"})
        assert err.details == {"key": "val"}

    def test_is_exception(self):
        assert issubclass(RouterError, Exception)
