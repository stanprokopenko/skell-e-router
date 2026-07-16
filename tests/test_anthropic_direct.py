"""Tests for skell_e_router.anthropic_direct — direct Anthropic SDK path."""

import json
import pytest
from unittest.mock import patch, MagicMock

from tests.helpers import (
    make_model,
    make_anthropic_response,
    make_litellm_response,
    FAKE_ANTHROPIC_KEY,
    FAKE_OPENAI_KEY,
)


# ===================================================================
# TestConvertMessagesForAnthropic
# ===================================================================

class TestConvertMessagesForAnthropic:
    """Tests for _convert_messages_for_anthropic."""

    def _call(self, messages):
        from skell_e_router.anthropic_direct import _convert_messages_for_anthropic
        return _convert_messages_for_anthropic(messages)

    def test_system_message_extracted(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys_prompt, converted = self._call(msgs)
        assert sys_prompt == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_user_assistant_roles_pass_through(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        _, converted = self._call(msgs)
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_multi_part_text(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "text", "text": "And this"},
            ]},
        ]
        _, converted = self._call(msgs)
        assert len(converted) == 1
        assert len(converted[0]["content"]) == 2
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][1]["text"] == "And this"

    def test_data_uri_image(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ]},
        ]
        _, converted = self._call(msgs)
        img = converted[0]["content"][0]
        assert img["type"] == "image"
        assert img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/png"
        assert img["source"]["data"] == "aGVsbG8="

    def test_url_image(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]},
        ]
        _, converted = self._call(msgs)
        img = converted[0]["content"][0]
        assert img["type"] == "image"
        assert img["source"]["type"] == "url"
        assert img["source"]["url"] == "https://example.com/img.jpg"

    def test_empty_list(self):
        sys_prompt, converted = self._call([])
        assert sys_prompt is None
        assert converted == []

    def test_input_audio_raises_unsupported_modality(self):
        from skell_e_router.utils import RouterError
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}},
            ]},
        ]
        with pytest.raises(RouterError) as exc_info:
            self._call(msgs)
        assert exc_info.value.code == "UNSUPPORTED_MODALITY"
        assert "audio" in exc_info.value.message.lower()

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        msgs = [
            {"role": "assistant", "content": "Searching now.", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": '{"query": "gesture"}'}},
            ]},
        ]
        _, converted = self._call(msgs)
        assert len(converted) == 1
        blocks = converted[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Searching now."}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "call_1"
        assert blocks[1]["name"] == "search"
        assert blocks[1]["input"] == {"query": "gesture"}

    def test_assistant_tool_calls_without_text(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": ""}},
            ]},
        ]
        _, converted = self._call(msgs)
        blocks = converted[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["input"] == {}

    def test_tool_role_becomes_tool_result_user_message(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "3 results found"},
        ]
        _, converted = self._call(msgs)
        assert len(converted) == 2
        result_msg = converted[1]
        assert result_msg["role"] == "user"
        assert result_msg["content"][0] == {
            "type": "tool_result", "tool_use_id": "call_1", "content": "3 results found",
        }

    def test_gemini_thought_signature_tool_ids_sanitized(self):
        # LiteLLM's Gemini path appends "__thought__<base64>" to tool call ids;
        # the base64 payload carries '/', '+', '=' which Anthropic rejects
        # (ids must match ^[a-zA-Z0-9_-]+$). The tool_use block and its
        # matching tool_result must sanitize to the SAME valid id.
        dirty = "ad0mkyc7__thought__EsgK/CsUK+ARFNMg=="
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": dirty, "type": "function",
                 "function": {"name": "run_command", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": dirty, "content": "ok"},
        ]
        _, converted = self._call(msgs)
        tool_use = converted[0]["content"][0]
        tool_result = converted[1]["content"][0]
        import re as _re
        assert _re.fullmatch(r"[a-zA-Z0-9_-]+", tool_use["id"])
        assert tool_use["id"] == tool_result["tool_use_id"]
        assert tool_use["id"].startswith("ad0mkyc7__thought__")

    def test_distinct_dirty_tool_ids_stay_distinct(self):
        # Two ids differing only in stripped characters must not collide.
        from skell_e_router.anthropic_direct import _sanitize_tool_id
        a = _sanitize_tool_id("x__thought__AAA+BBB/CCC=")
        b = _sanitize_tool_id("x__thought__AAA/BBB+CCC=")
        assert a != b

    def test_valid_tool_ids_pass_through_unchanged(self):
        from skell_e_router.anthropic_direct import _sanitize_tool_id
        assert _sanitize_tool_id("call_abc-123_XYZ") == "call_abc-123_XYZ"
        assert _sanitize_tool_id(None) is None
        assert _sanitize_tool_id("") == ""

    def test_parallel_tool_results_merge_into_one_user_message(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": "{}"}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result A"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result B"},
        ]
        _, converted = self._call(msgs)
        assert len(converted) == 2
        blocks = converted[1]["content"]
        assert [b["tool_use_id"] for b in blocks] == ["call_1", "call_2"]

    def test_user_text_after_tool_result_merges_after_result_blocks(self):
        msgs = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result A"},
            {"role": "user", "content": "Keep going."},
        ]
        _, converted = self._call(msgs)
        assert len(converted) == 1
        blocks = converted[0]["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[1] == {"type": "text", "text": "Keep going."}

    def test_tool_call_dict_arguments_pass_through(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "f", "arguments": {"k": 1}}},
            ]},
        ]
        _, converted = self._call(msgs)
        assert converted[0]["content"][0]["input"] == {"k": 1}

    def test_tool_call_malformed_arguments_fall_back_to_empty(self):
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "f", "arguments": "not json"}},
            ]},
        ]
        _, converted = self._call(msgs)
        assert converted[0]["content"][0]["input"] == {}


# ===================================================================
# TestBuildCreateParams
# ===================================================================

class TestBuildCreateParams:
    """Tests for _build_create_params."""

    def _call(self, ai_model, kwargs):
        from skell_e_router.anthropic_direct import _build_create_params
        return _build_create_params(ai_model, kwargs)

    def _claude_model(self, **overrides):
        defaults = dict(
            provider="anthropic",
            name="anthropic/claude-sonnet-4-6",
            supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens",
                              "thinking", "reasoning_effort", "stream", "tools",
                              "tool_choice", "betas"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        defaults.update(overrides)
        return make_model(**defaults)

    def test_max_tokens_default(self):
        params, _ = self._call(self._claude_model(), {})
        assert params["max_tokens"] == 4096

    def test_max_tokens_custom(self):
        params, _ = self._call(self._claude_model(), {"max_tokens": 1000})
        assert params["max_tokens"] == 1000

    def test_temperature(self):
        params, _ = self._call(self._claude_model(), {"temperature": 0.7})
        assert params["temperature"] == 0.7

    def test_top_p(self):
        params, _ = self._call(self._claude_model(), {"top_p": 0.9})
        assert params["top_p"] == 0.9

    def test_top_k(self):
        params, _ = self._call(self._claude_model(), {"top_k": 40})
        assert params["top_k"] == 40

    def test_stop_string(self):
        params, _ = self._call(self._claude_model(), {"stop": "END"})
        assert params["stop_sequences"] == ["END"]

    def test_stop_list(self):
        params, _ = self._call(self._claude_model(), {"stop": ["END", "STOP"]})
        assert params["stop_sequences"] == ["END", "STOP"]

    def test_budget_tokens_enabled(self):
        params, _ = self._call(self._claude_model(), {"budget_tokens": 2048})
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_budget_tokens_zero_disabled(self):
        params, _ = self._call(self._claude_model(), {"budget_tokens": 0})
        assert params["thinking"] == {"type": "disabled"}

    def test_thinking_dict_enabled(self):
        params, _ = self._call(self._claude_model(), {"thinking": {"type": "enabled", "budget_tokens": 4096}})
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_thinking_dict_disabled(self):
        params, _ = self._call(self._claude_model(), {"thinking": {"type": "disabled"}})
        assert params["thinking"] == {"type": "disabled"}

    def test_thinking_dict_adaptive(self):
        params, _ = self._call(self._claude_model(), {"thinking": {"type": "adaptive"}})
        assert params["thinking"] == {"type": "adaptive"}

    def test_reasoning_effort_validation(self):
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError, match="INVALID_PARAM"):
            self._call(self._claude_model(), {"reasoning_effort": "turbo"})

    def test_reasoning_effort_to_adaptive(self):
        """reasoning_effort on model that supports it natively -> adaptive thinking."""
        params, _ = self._call(self._claude_model(), {"reasoning_effort": "high"})
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}

    def test_reasoning_effort_low_passes_output_config(self):
        params, _ = self._call(self._claude_model(), {"reasoning_effort": "low"})
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "low"}

    def test_reasoning_effort_to_budget_mapping(self):
        """reasoning_effort on model without reasoning_effort support -> budget mapping."""
        model = self._claude_model(
            supported_params={"temperature", "stop", "max_tokens", "budget_tokens",
                              "thinking", "stream", "tools", "tool_choice", "betas"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )
        params, _ = self._call(model, {"reasoning_effort": "low"})
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 1024}

        params, _ = self._call(model, {"reasoning_effort": "medium"})
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 2048}

        params, _ = self._call(model, {"reasoning_effort": "high"})
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_temperature_forced_with_thinking(self):
        """When thinking enabled, temperature must be forced to 1."""
        params, _ = self._call(self._claude_model(), {"temperature": 0.5, "budget_tokens": 2048})
        assert params["temperature"] == 1

    def test_top_p_dropped_with_thinking(self):
        """When thinking enabled, top_p < 0.95 must be dropped."""
        params, _ = self._call(self._claude_model(), {"top_p": 0.8, "budget_tokens": 2048})
        assert "top_p" not in params

    def test_top_p_kept_when_high(self):
        """top_p >= 0.95 should be kept even with thinking."""
        params, _ = self._call(self._claude_model(), {"top_p": 0.95, "budget_tokens": 2048})
        assert params.get("top_p") == 0.95

    def test_opus_4_7_skips_sampling_params_with_thinking(self):
        """Opus 4.7 rejects temperature/top_p entirely — even when thinking is active, don't set them."""
        opus_4_7 = self._claude_model(
            name="anthropic/claude-opus-4-7",
            supported_params={"stop", "max_tokens", "thinking", "reasoning_effort",
                              "stream", "tools", "tool_choice", "betas"},
            accepted_reasoning_efforts={"low", "medium", "high", "xhigh"},
        )
        params, _ = self._call(opus_4_7, {"temperature": 0.5, "top_p": 0.5, "reasoning_effort": "high"})
        assert "temperature" not in params
        assert "top_p" not in params
        assert params["thinking"] == {"type": "adaptive"}

    def test_tools_conversion(self):
        openai_tools = [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }}
        ]
        params, _ = self._call(self._claude_model(), {"tools": openai_tools})
        assert len(params["tools"]) == 1
        tool = params["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather"
        assert "input_schema" in tool
        assert "parameters" not in tool

    def test_tool_choice_auto(self):
        params, _ = self._call(self._claude_model(), {"tool_choice": "auto"})
        assert params["tool_choice"] == {"type": "auto"}

    def test_tool_choice_none(self):
        params, _ = self._call(self._claude_model(), {"tool_choice": "none"})
        assert params["tool_choice"] == {"type": "none"}

    def test_tool_choice_required(self):
        params, _ = self._call(self._claude_model(), {"tool_choice": "required"})
        assert params["tool_choice"] == {"type": "any"}

    def test_tool_choice_specific(self):
        params, _ = self._call(self._claude_model(), {
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        })
        assert params["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_betas_to_extra_headers(self):
        _, extra_headers = self._call(self._claude_model(), {"betas": ["output-128k-2025-02-19"]})
        assert extra_headers == {"anthropic-beta": "output-128k-2025-02-19"}

    def test_betas_list_joined(self):
        _, extra_headers = self._call(self._claude_model(), {"betas": ["beta1", "beta2"]})
        assert extra_headers == {"anthropic-beta": "beta1,beta2"}


# ===================================================================
# TestCallAnthropicDirect
# ===================================================================

class TestCallAnthropicDirect:
    """Tests for _call_anthropic_direct."""

    @pytest.fixture(autouse=True)
    def clear_client_cache(self):
        """Clear cached clients so each test gets its own mock client."""
        from skell_e_router.anthropic_direct import _client_cache
        _client_cache.clear()

    def test_strips_anthropic_prefix(self):
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("skell_e_router.anthropic_direct.ANTHROPIC_AVAILABLE", True), \
             patch("skell_e_router.anthropic_direct.anthropic", mock_anthropic):
            from skell_e_router.anthropic_direct import _call_anthropic_direct
            result = _call_anthropic_direct(
                "anthropic/claude-sonnet-4-6", [{"role": "user", "content": "hi"}],
                None, {"max_tokens": 4096}, None, FAKE_ANTHROPIC_KEY
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        response, request_duration = result
        assert response is mock_response
        assert isinstance(request_duration, float)

    @patch("skell_e_router.anthropic_direct.time.sleep")
    def test_retries_transient_errors(self, mock_sleep):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()

        RateLimitError = type("RateLimitError", (Exception,), {})
        err = RateLimitError("rate limit")
        err.status_code = 429
        mock_client.messages.create.side_effect = [err, err, MagicMock()]
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("skell_e_router.anthropic_direct.ANTHROPIC_AVAILABLE", True), \
             patch("skell_e_router.anthropic_direct.anthropic", mock_anthropic):
            from skell_e_router.anthropic_direct import _call_anthropic_direct
            result = _call_anthropic_direct(
                "claude-sonnet-4-6", [{"role": "user", "content": "hi"}],
                None, {"max_tokens": 4096}, None, FAKE_ANTHROPIC_KEY
            )

        response, request_duration = result
        assert response is not None
        assert isinstance(request_duration, float)
        assert mock_sleep.call_count == 2

    def test_no_retry_on_400(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()

        err = Exception("bad request")
        err.status_code = 400
        mock_client.messages.create.side_effect = err
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("skell_e_router.anthropic_direct.ANTHROPIC_AVAILABLE", True), \
             patch("skell_e_router.anthropic_direct.anthropic", mock_anthropic):
            from skell_e_router.anthropic_direct import _call_anthropic_direct
            with pytest.raises(Exception, match="bad request"):
                _call_anthropic_direct(
                    "claude-sonnet-4-6", [{"role": "user", "content": "hi"}],
                    None, {"max_tokens": 4096}, None, FAKE_ANTHROPIC_KEY
                )

        assert mock_client.messages.create.call_count == 1

    def test_import_error_when_no_sdk(self):
        with patch("skell_e_router.anthropic_direct.ANTHROPIC_AVAILABLE", False):
            from skell_e_router.anthropic_direct import _call_anthropic_direct
            with pytest.raises(ImportError, match="anthropic"):
                _call_anthropic_direct(
                    "claude-sonnet-4-6", [], None, {}, None, FAKE_ANTHROPIC_KEY
                )


# ===================================================================
# TestBuildResponse
# ===================================================================

class TestBuildResponse:
    """Tests for _build_response."""

    def test_text_content_extraction(self):
        resp = make_anthropic_response(text="Hello world")
        from skell_e_router.anthropic_direct import _build_response
        ai_resp = _build_response(resp, "anthropic/claude-sonnet-4-6", 1.5)
        assert ai_resp.content == "Hello world"
        assert ai_resp.model == "claude-sonnet-4-6"
        assert ai_resp.duration_seconds == 1.5

    def test_token_usage(self):
        resp = make_anthropic_response(prompt_tokens=15, completion_tokens=25)
        from skell_e_router.anthropic_direct import _build_response
        ai_resp = _build_response(resp, "anthropic/claude-sonnet-4-6", 1.0)
        assert ai_resp.prompt_tokens == 15
        assert ai_resp.completion_tokens == 25
        assert ai_resp.total_tokens == 40

    def test_cost_calculation(self):
        resp = make_anthropic_response(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        from skell_e_router.anthropic_direct import _build_response
        ai_resp = _build_response(resp, "anthropic/claude-sonnet-4-6", 1.0)
        # 1M * 3.00/1M + 1M * 15.00/1M = 3.00 + 15.00 = 18.00
        assert ai_resp.cost == pytest.approx(18.00)

    def test_no_cost_for_unknown_model(self):
        resp = make_anthropic_response(prompt_tokens=100, completion_tokens=100)
        from skell_e_router.anthropic_direct import _build_response
        ai_resp = _build_response(resp, "anthropic/unknown-model", 1.0)
        assert ai_resp.cost is None

    def test_tool_calls_from_tool_use_block(self):
        resp = make_anthropic_response(
            text=None,
            tool_use_blocks=[
                {"id": "toolu_abc123", "name": "get_weather", "input": {"city": "NYC"}}
            ]
        )
        from skell_e_router.anthropic_direct import _build_response
        ai_resp = _build_response(resp, "anthropic/claude-sonnet-4-6", 1.0)
        assert ai_resp.tool_calls is not None
        assert len(ai_resp.tool_calls) == 1
        tc = ai_resp.tool_calls[0]
        assert tc["type"] == "function"
        assert tc["id"] == "toolu_abc123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}

    def test_thinking_blocks_skipped(self):
        resp = make_anthropic_response(
            text="Final answer",
            thinking_blocks=[{"thinking": "Let me think..."}]
        )
        from skell_e_router.anthropic_direct import _build_response
        ai_resp = _build_response(resp, "anthropic/claude-sonnet-4-6", 1.0)
        assert ai_resp.content == "Final answer"
        assert "think" not in ai_resp.content.lower()


# ===================================================================
# TestStreamAnthropicDirect
# ===================================================================

class TestStreamAnthropicDirect:
    """Tests for _call_anthropic_direct_stream."""

    @pytest.fixture(autouse=True)
    def clear_client_cache(self):
        """Clear cached clients so each test gets its own mock client."""
        from skell_e_router.anthropic_direct import _client_cache
        _client_cache.clear()

    def test_returns_stream_manager(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        stream_sentinel = MagicMock()
        mock_client.messages.stream.return_value = stream_sentinel
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("skell_e_router.anthropic_direct.ANTHROPIC_AVAILABLE", True), \
             patch("skell_e_router.anthropic_direct.anthropic", mock_anthropic):
            from skell_e_router.anthropic_direct import _call_anthropic_direct_stream
            result = _call_anthropic_direct_stream(
                "claude-sonnet-4-6", [{"role": "user", "content": "hi"}],
                None, {"max_tokens": 4096}, None, FAKE_ANTHROPIC_KEY
            )

        assert result is stream_sentinel

    def test_stream_is_usable(self):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        stream_ctx.text_stream = iter(["Hello", " world"])
        mock_client.messages.stream.return_value = stream_ctx
        mock_anthropic.Anthropic.return_value = mock_client

        with patch("skell_e_router.anthropic_direct.ANTHROPIC_AVAILABLE", True), \
             patch("skell_e_router.anthropic_direct.anthropic", mock_anthropic):
            from skell_e_router.anthropic_direct import _call_anthropic_direct_stream
            result = _call_anthropic_direct_stream(
                "claude-sonnet-4-6", [{"role": "user", "content": "hi"}],
                None, {"max_tokens": 4096}, None, FAKE_ANTHROPIC_KEY
            )

        with result as stream:
            chunks = list(stream.text_stream)
            assert chunks == ["Hello", " world"]


# ===================================================================
# TestAskAiDirectAnthropicIntegration
# ===================================================================

class TestAskAiDirectAnthropicIntegration:
    """Integration tests for _ask_ai_direct_anthropic and ask_ai routing."""

    def _make_claude_model(self):
        return make_model(
            provider="anthropic",
            name="anthropic/claude-sonnet-4-6",
            supported_params={"temperature", "stop", "max_tokens", "budget_tokens",
                              "thinking", "reasoning_effort", "stream", "tools",
                              "tool_choice", "betas"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )

    @patch("skell_e_router.utils._call_anthropic_direct")
    @patch("skell_e_router.utils._convert_messages_for_anthropic")
    @patch("skell_e_router.utils._build_create_params")
    def test_direct_sdk_true_routes_correctly(self, mock_params, mock_convert, mock_call):
        """direct_sdk=True should use the direct Anthropic path."""
        mock_convert.return_value = (None, [{"role": "user", "content": "hi"}])
        mock_params.return_value = ({"max_tokens": 4096}, None)
        mock_resp = make_anthropic_response()
        mock_call.return_value = (mock_resp, 0.5)

        from skell_e_router.utils import _ask_ai_direct_anthropic
        model = self._make_claude_model()
        result = _ask_ai_direct_anthropic(
            model, [{"role": "user", "content": "hi"}],
            FAKE_ANTHROPIC_KEY, "none", False, None, {}
        )
        assert result == "Hello from Claude"
        mock_call.assert_called_once()

    @patch("skell_e_router.utils._perform_completion")
    def test_direct_sdk_false_forces_litellm(self, mock_completion):
        """direct_sdk=False should bypass direct path and use LiteLLM."""
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "from litellm"
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock()
        mock_completion.return_value = (mock_resp, 0.5)

        from skell_e_router.utils import ask_ai
        from skell_e_router.model_config import MODEL_CONFIG

        with patch.dict("skell_e_router.model_config.MODEL_CONFIG", {
            "test-claude": make_model(
                provider="anthropic",
                name="anthropic/claude-sonnet-4-6",
                supported_params={"temperature", "stream", "max_tokens", "thinking"},
            )
        }):
            MODEL_CONFIG["test-claude"].use_direct_sdk = True
            result = ask_ai(
                "test-claude", "hi",
                config={"anthropic_api_key": FAKE_ANTHROPIC_KEY},
                direct_sdk=False
            )
        assert result == "from litellm"
        mock_completion.assert_called_once()

    @patch("skell_e_router.utils._call_anthropic_direct_stream")
    @patch("skell_e_router.utils._convert_messages_for_anthropic")
    @patch("skell_e_router.utils._build_create_params")
    def test_stream_true_uses_streaming(self, mock_params, mock_convert, mock_stream):
        """stream=True should call _call_anthropic_direct_stream."""
        mock_convert.return_value = (None, [])
        mock_params.return_value = ({"max_tokens": 4096}, None)
        sentinel = MagicMock()
        mock_stream.return_value = sentinel

        from skell_e_router.utils import _ask_ai_direct_anthropic
        model = self._make_claude_model()
        result = _ask_ai_direct_anthropic(
            model, [{"role": "user", "content": "hi"}],
            FAKE_ANTHROPIC_KEY, "none", False, None, {"stream": True}
        )
        assert result is sentinel
        mock_stream.assert_called_once()

    @patch("skell_e_router.utils._call_anthropic_direct")
    @patch("skell_e_router.utils._convert_messages_for_anthropic")
    @patch("skell_e_router.utils._build_create_params")
    def test_budget_tokens_handled(self, mock_params, mock_convert, mock_call):
        """budget_tokens should be handled by _build_create_params (no crash)."""
        mock_convert.return_value = (None, [])
        mock_params.return_value = ({"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 2048}}, None)
        mock_call.return_value = (make_anthropic_response(), 0.5)

        from skell_e_router.utils import _ask_ai_direct_anthropic
        model = self._make_claude_model()
        result = _ask_ai_direct_anthropic(
            model, [{"role": "user", "content": "hi"}],
            FAKE_ANTHROPIC_KEY, "none", False, None, {"budget_tokens": 2048}
        )
        assert result == "Hello from Claude"

    def test_reasoning_effort_validated(self):
        """Invalid reasoning_effort should raise RouterError."""
        from skell_e_router.utils import _ask_ai_direct_anthropic, RouterError

        model = self._make_claude_model()
        with pytest.raises(RouterError, match="INVALID_PARAM"):
            _ask_ai_direct_anthropic(
                model, [{"role": "user", "content": "hi"}],
                FAKE_ANTHROPIC_KEY, "none", False, None, {"reasoning_effort": "turbo"}
            )

    def test_missing_key_raises(self):
        """Missing API key should raise RouterError."""
        from skell_e_router.utils import _ask_ai_direct_anthropic, RouterError

        model = self._make_claude_model()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RouterError, match="MISSING_ENV"):
                _ask_ai_direct_anthropic(
                    model, [{"role": "user", "content": "hi"}],
                    None, "none", False, None, {}
                )

    @patch("skell_e_router.utils._call_anthropic_direct")
    @patch("skell_e_router.utils._convert_messages_for_anthropic")
    @patch("skell_e_router.utils._build_create_params")
    def test_provider_error_wrapped(self, mock_params, mock_convert, mock_call):
        """SDK exceptions should be wrapped in RouterError(PROVIDER_ERROR)."""
        mock_convert.return_value = (None, [])
        mock_params.return_value = ({"max_tokens": 4096}, None)
        mock_call.side_effect = RuntimeError("SDK boom")

        from skell_e_router.utils import _ask_ai_direct_anthropic, RouterError
        model = self._make_claude_model()
        with pytest.raises(RouterError, match="PROVIDER_ERROR"):
            _ask_ai_direct_anthropic(
                model, [{"role": "user", "content": "hi"}],
                FAKE_ANTHROPIC_KEY, "none", False, None, {}
            )

    def test_audio_to_anthropic_raises_unsupported_modality_end_to_end(self):
        """ask_ai with audio against a Claude model should surface UNSUPPORTED_MODALITY,
        not get silently dropped or wrapped as PROVIDER_ERROR.

        This proves the wiring between _construct_messages -> _ask_ai_direct_anthropic
        -> _convert_messages_for_anthropic -> RouterError actually preserves the
        UNSUPPORTED_MODALITY code through the full call stack.
        """
        from skell_e_router.utils import ask_ai, RouterError
        from skell_e_router.model_config import MODEL_CONFIG

        with patch.dict("skell_e_router.model_config.MODEL_CONFIG", {
            "test-claude-audio": make_model(
                provider="anthropic",
                name="anthropic/claude-sonnet-4-6",
                supported_params={"temperature", "stream", "max_tokens", "thinking"},
            )
        }):
            MODEL_CONFIG["test-claude-audio"].use_direct_sdk = True
            with pytest.raises(RouterError) as exc_info:
                ask_ai(
                    "test-claude-audio",
                    "describe this clip",
                    audio=["data:audio/mpeg;base64,SUQzAA=="],
                    config={"anthropic_api_key": FAKE_ANTHROPIC_KEY},
                )
            assert exc_info.value.code == "UNSUPPORTED_MODALITY"
            assert "audio" in exc_info.value.message.lower()


# ===================================================================
# TestApplyCacheControl
# ===================================================================

class TestApplyCacheControl:
    """Tests for _apply_cache_control (Anthropic prompt caching, direct path)."""

    def _call(self, system_prompt, messages):
        from skell_e_router.anthropic_direct import _apply_cache_control
        return _apply_cache_control(system_prompt, messages)

    def test_system_string_becomes_cached_block(self):
        sys_out, _ = self._call("You are helpful.", [])
        assert isinstance(sys_out, list)
        assert sys_out[0]["type"] == "text"
        assert sys_out[0]["text"] == "You are helpful."
        assert sys_out[0]["cache_control"] == {"type": "ephemeral"}

    def test_none_system_passes_through(self):
        sys_out, _ = self._call(None, [{"role": "user", "content": "hi"}])
        assert sys_out is None

    def test_last_message_string_content_becomes_cached_block(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        _, out = self._call(None, msgs)
        # Earlier messages untouched
        assert out[0]["content"] == "first"
        assert out[1]["content"] == "second"
        # Last message converted to block list with cache_control
        last = out[2]["content"]
        assert isinstance(last, list)
        assert last[0]["text"] == "third"
        assert last[0]["cache_control"] == {"type": "ephemeral"}

    def test_last_message_list_content_marks_last_block(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "text", "text": "here"},
            ]},
        ]
        _, out = self._call(None, msgs)
        blocks = out[0]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_messages_ok(self):
        sys_out, out = self._call("sys", [])
        assert out == []
        assert sys_out[0]["cache_control"] == {"type": "ephemeral"}


class TestBuildResponseCacheTokens:
    """_build_response must account for cache-write and cache-read tokens."""

    def test_cache_tokens_included_in_prompt_tokens_and_cost(self):
        from skell_e_router.anthropic_direct import _build_response
        resp = make_anthropic_response(
            prompt_tokens=100, completion_tokens=50, model="claude-haiku-4-5"
        )
        resp.usage.cache_creation_input_tokens = 1000
        resp.usage.cache_read_input_tokens = 9000

        ai_resp = _build_response(resp, "anthropic/claude-haiku-4-5", 0.5)
        # Full prompt size = uncached + cache-write + cache-read
        assert ai_resp.prompt_tokens == 100 + 1000 + 9000
        assert ai_resp.total_tokens == 10100 + 50
        # haiku-4-5: $1/M input, $5/M output; write 1.25x, read 0.1x
        expected_cost = (100 * 1.0 + 1000 * 1.25 + 9000 * 0.10) / 1_000_000 + 50 * 5.0 / 1_000_000
        assert ai_resp.cost == pytest.approx(expected_cost)

    def test_no_cache_fields_behaves_as_before(self):
        from skell_e_router.anthropic_direct import _build_response
        # make_anthropic_response usage is a MagicMock: cache fields resolve to
        # non-int mocks and must be treated as 0
        resp = make_anthropic_response(prompt_tokens=10, completion_tokens=20, model="claude-haiku-4-5")
        ai_resp = _build_response(resp, "anthropic/claude-haiku-4-5", 0.5)
        assert ai_resp.prompt_tokens == 10
        assert ai_resp.total_tokens == 30
        expected_cost = 10 * 1.0 / 1_000_000 + 20 * 5.0 / 1_000_000
        assert ai_resp.cost == pytest.approx(expected_cost)


class TestEnableCachingIntegration:
    """enable_caching wiring through _ask_ai_direct_anthropic and ask_ai."""

    def _make_claude_model(self):
        return make_model(
            provider="anthropic",
            name="anthropic/claude-sonnet-4-6",
            supported_params={"temperature", "stop", "max_tokens", "budget_tokens",
                              "thinking", "reasoning_effort", "stream", "tools",
                              "tool_choice", "betas"},
            accepted_reasoning_efforts={"low", "medium", "high"},
        )

    @patch("skell_e_router.utils._call_anthropic_direct")
    def test_enable_caching_true_adds_breakpoints(self, mock_call):
        mock_call.return_value = (make_anthropic_response(), 0.5)

        from skell_e_router.utils import _ask_ai_direct_anthropic
        model = self._make_claude_model()
        _ask_ai_direct_anthropic(
            model,
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "hi"}],
            FAKE_ANTHROPIC_KEY, "none", False, None, {},
            enable_caching=True,
        )
        call_kwargs = mock_call.call_args.kwargs
        system = call_kwargs["system_prompt"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        last_content = call_kwargs["messages"][-1]["content"]
        assert last_content[-1]["cache_control"] == {"type": "ephemeral"}

    @patch("skell_e_router.utils._call_anthropic_direct")
    def test_enable_caching_default_off(self, mock_call):
        mock_call.return_value = (make_anthropic_response(), 0.5)

        from skell_e_router.utils import _ask_ai_direct_anthropic
        model = self._make_claude_model()
        _ask_ai_direct_anthropic(
            model,
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "hi"}],
            FAKE_ANTHROPIC_KEY, "none", False, None, {},
        )
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["system_prompt"] == "You are helpful."
        assert call_kwargs["messages"][-1]["content"] == "hi"

    @patch("skell_e_router.utils._perform_completion")
    def test_enable_caching_litellm_anthropic_path(self, mock_completion):
        mock_completion.return_value = (make_litellm_response(content="ok"), 0.5)

        from skell_e_router.utils import ask_ai
        from skell_e_router.model_config import MODEL_CONFIG

        with patch.dict("skell_e_router.model_config.MODEL_CONFIG", {
            "test-claude": make_model(
                provider="anthropic",
                name="anthropic/claude-sonnet-4-6",
                supported_params={"temperature", "stream", "max_tokens", "thinking"},
            )
        }):
            ask_ai(
                "test-claude", "hi", "You are helpful.",
                config={"anthropic_api_key": FAKE_ANTHROPIC_KEY},
                direct_sdk=False,
                enable_caching=True,
            )
        messages = mock_completion.call_args.kwargs["messages"]
        sys_msg = next(m for m in messages if m["role"] == "system")
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    @patch("skell_e_router.utils._perform_completion")
    def test_enable_caching_noop_for_non_anthropic(self, mock_completion):
        mock_completion.return_value = (make_litellm_response(content="ok"), 0.5)

        from skell_e_router.utils import ask_ai
        from skell_e_router.model_config import MODEL_CONFIG

        with patch.dict("skell_e_router.model_config.MODEL_CONFIG", {
            "test-gpt": make_model(
                provider="openai",
                name="openai/gpt-4.1",
                supported_params={"temperature", "stream", "max_tokens"},
            )
        }):
            ask_ai(
                "test-gpt", "hi", "You are helpful.",
                config={"openai_api_key": FAKE_OPENAI_KEY},
                enable_caching=True,
            )
        messages = mock_completion.call_args.kwargs["messages"]
        for m in messages:
            assert isinstance(m["content"], str)
