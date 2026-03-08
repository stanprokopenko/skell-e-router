"""Tests for skell_e_router.gemini_direct — direct Google genai SDK path."""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from tests.helpers import make_model, make_gemini_response, FAKE_GEMINI_KEY


# ---------------------------------------------------------------------------
# We always mock the google-genai types so tests run without the real SDK.
# ---------------------------------------------------------------------------

def _mock_types():
    """Return a MagicMock that mimics google.genai.types for config building."""
    t = MagicMock()

    # ThinkingConfig: store kwargs as attributes
    def _thinking_config(**kw):
        m = MagicMock()
        for k, v in kw.items():
            setattr(m, k, v)
        m._kw = kw
        return m
    t.ThinkingConfig.side_effect = _thinking_config

    # SafetySetting: simple passthrough
    t.SafetySetting.side_effect = lambda **kw: kw

    # GenerateContentConfig: store kwargs and expose as attributes
    def _gen_config(**kw):
        m = MagicMock()
        m._kw = kw
        for k, v in kw.items():
            setattr(m, k, v)
        # Ensure missing attrs return None (for _CONFIG_ATTRS iteration)
        for attr in ("max_output_tokens", "temperature", "top_p", "top_k",
                      "stop_sequences", "thinking_config", "safety_settings",
                      "candidate_count", "tool_config"):
            if attr not in kw:
                setattr(m, attr, None)
        return m
    t.GenerateContentConfig.side_effect = _gen_config

    t.GoogleSearch.return_value = "google_search_sentinel"
    t.Tool.side_effect = lambda **kw: kw

    # FunctionDeclaration
    t.FunctionDeclaration.side_effect = lambda **kw: kw

    # FunctionCallingConfig
    t.FunctionCallingConfig.side_effect = lambda **kw: kw

    # ToolConfig
    t.ToolConfig.side_effect = lambda **kw: kw

    # Content / Part for _convert_messages_to_contents
    t.Content.side_effect = lambda **kw: kw
    t.Part.from_text.side_effect = lambda text: {"text": text}
    t.Part.from_bytes.side_effect = lambda data, mime_type: {"bytes": data, "mime_type": mime_type}
    t.Part.from_uri.side_effect = lambda file_uri, mime_type: {"uri": file_uri, "mime_type": mime_type}

    return t


# ===================================================================
# TestConvertMessagesToContents
# ===================================================================

class TestConvertMessagesToContents:
    """Tests for _convert_messages_to_contents."""

    def _call(self, messages):
        mock_types = _mock_types()
        with patch("skell_e_router.gemini_direct.types", mock_types):
            from skell_e_router.gemini_direct import _convert_messages_to_contents
            return _convert_messages_to_contents(messages)

    def test_system_message_extracted(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        sys_instr, contents = self._call(msgs)
        assert sys_instr == "You are helpful."
        assert len(contents) == 1

    def test_user_assistant_role_mapping(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        _, contents = self._call(msgs)
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_multi_part_content(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "text", "text": "And this"},
            ]},
        ]
        _, contents = self._call(msgs)
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2

    def test_data_uri_image(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ]},
        ]
        _, contents = self._call(msgs)
        parts = contents[0]["parts"]
        assert parts[0]["mime_type"] == "image/png"

    def test_url_image(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]},
        ]
        _, contents = self._call(msgs)
        parts = contents[0]["parts"]
        assert parts[0]["uri"] == "https://example.com/img.jpg"

    def test_empty_list(self):
        sys_instr, contents = self._call([])
        assert sys_instr is None
        assert contents == []


# ===================================================================
# TestBuildGenerateConfig
# ===================================================================

class TestBuildGenerateConfig:
    """Tests for _build_generate_config."""

    def _call(self, ai_model, kwargs):
        mock_types = _mock_types()
        with patch("skell_e_router.gemini_direct.types", mock_types):
            from skell_e_router.gemini_direct import _build_generate_config
            return _build_generate_config(ai_model, kwargs)

    def _gemini_model(self, **overrides):
        defaults = dict(
            provider="gemini",
            name="gemini/gemini-2.5-flash",
            supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens",
                              "reasoning_effort", "stream", "tools", "tool_choice",
                              "candidate_count", "safety_settings", "web_search_options"},
            accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        )
        defaults.update(overrides)
        return make_model(**defaults)

    def test_max_tokens(self):
        config, _ = self._call(self._gemini_model(), {"max_tokens": 1000})
        assert config._kw["max_output_tokens"] == 1000

    def test_temperature_top_p_top_k(self):
        config, _ = self._call(self._gemini_model(), {"temperature": 0.5, "top_p": 0.9, "top_k": 40})
        assert config._kw["temperature"] == 0.5
        assert config._kw["top_p"] == 0.9
        assert config._kw["top_k"] == 40

    def test_stop_string(self):
        config, _ = self._call(self._gemini_model(), {"stop": "END"})
        assert config._kw["stop_sequences"] == ["END"]

    def test_stop_list(self):
        config, _ = self._call(self._gemini_model(), {"stop": ["END", "STOP"]})
        assert config._kw["stop_sequences"] == ["END", "STOP"]

    def test_reasoning_effort_low_uses_thinking_level(self):
        """Model with reasoning_effort but no budget_tokens -> thinking_level."""
        config, _ = self._call(self._gemini_model(), {"reasoning_effort": "low"})
        tc = config._kw["thinking_config"]
        assert tc.thinking_level == "LOW"

    def test_reasoning_effort_medium_uses_thinking_level(self):
        config, _ = self._call(self._gemini_model(), {"reasoning_effort": "medium"})
        tc = config._kw["thinking_config"]
        assert tc.thinking_level == "MEDIUM"

    def test_reasoning_effort_high_uses_thinking_level(self):
        config, _ = self._call(self._gemini_model(), {"reasoning_effort": "high"})
        tc = config._kw["thinking_config"]
        assert tc.thinking_level == "HIGH"

    def test_reasoning_effort_on_budget_model_uses_thinking_budget(self):
        """Model with budget_tokens in supported_params -> thinking_budget."""
        model = self._gemini_model(
            name="gemini/gemini-2.5-flash-lite",
            supported_params={"temperature", "top_p", "stop", "max_tokens", "budget_tokens",
                              "thinking", "reasoning_effort", "stream", "tools", "tool_choice",
                              "candidate_count", "safety_settings", "web_search_options"},
        )
        config, _ = self._call(model, {"reasoning_effort": "high"})
        tc = config._kw["thinking_config"]
        assert tc.thinking_budget == 4096

    def test_reasoning_effort_validation_error(self):
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError, match="INVALID_PARAM"):
            self._call(self._gemini_model(), {"reasoning_effort": "turbo"})

    def test_budget_tokens_to_thinking_config(self):
        """budget_tokens on a model that supports it -> ThinkingConfig(thinking_budget=N)."""
        model = self._gemini_model(
            name="gemini/gemini-2.5-flash-lite",
            supported_params={"temperature", "top_p", "stop", "max_tokens", "budget_tokens",
                              "thinking", "stream", "tools", "tool_choice", "candidate_count",
                              "safety_settings", "web_search_options"},
        )
        config, _ = self._call(model, {"budget_tokens": 2048})
        tc = config._kw["thinking_config"]
        assert tc.thinking_budget == 2048

    def test_budget_tokens_maps_to_thinking_level(self):
        """budget_tokens on model w/o budget_tokens but w/ reasoning_effort -> thinking_level."""
        model = self._gemini_model()  # has reasoning_effort, no budget_tokens
        config, _ = self._call(model, {"budget_tokens": 500})
        tc = config._kw["thinking_config"]
        assert tc.thinking_level == "LOW"

    def test_thinking_dict_enabled(self):
        config, _ = self._call(self._gemini_model(), {"thinking": {"type": "enabled", "budget_tokens": 4096}})
        tc = config._kw["thinking_config"]
        assert tc.thinking_budget == 4096

    def test_thinking_dict_disabled(self):
        config, _ = self._call(self._gemini_model(), {"thinking": {"type": "disabled"}})
        assert "thinking_config" not in config._kw

    def test_candidate_count(self):
        config, _ = self._call(self._gemini_model(), {"candidate_count": 3})
        assert config._kw["candidate_count"] == 3

    def test_web_search_options(self):
        _, tools = self._call(self._gemini_model(), {"web_search_options": {"search_context_size": "high"}})
        assert tools is not None
        assert any("google_search" in str(t) for t in tools)

    def test_tools_conversion(self):
        openai_tools = [
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }}
        ]
        _, tools = self._call(self._gemini_model(), {"tools": openai_tools})
        assert tools is not None
        # Should have a Tool with function_declarations
        fn_tool = [t for t in tools if "function_declarations" in t]
        assert len(fn_tool) == 1
        assert fn_tool[0]["function_declarations"][0]["name"] == "get_weather"

    def test_tool_choice_auto(self):
        config, _ = self._call(self._gemini_model(), {"tool_choice": "auto"})
        tc = config._kw.get("tool_config")
        assert tc is not None
        assert tc["function_calling_config"]["mode"] == "AUTO"

    def test_tool_choice_none(self):
        config, _ = self._call(self._gemini_model(), {"tool_choice": "none"})
        tc = config._kw.get("tool_config")
        assert tc["function_calling_config"]["mode"] == "NONE"

    def test_tool_choice_required(self):
        config, _ = self._call(self._gemini_model(), {"tool_choice": "required"})
        tc = config._kw.get("tool_config")
        assert tc["function_calling_config"]["mode"] == "ANY"

    def test_tool_choice_specific_function(self):
        config, _ = self._call(self._gemini_model(), {
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        })
        tc = config._kw.get("tool_config")
        assert tc["function_calling_config"]["mode"] == "ANY"
        assert tc["function_calling_config"]["allowed_function_names"] == ["get_weather"]

    def test_safety_settings_always_present(self):
        config, _ = self._call(self._gemini_model(), {})
        assert "safety_settings" in config._kw
        assert len(config._kw["safety_settings"]) == 4


# ===================================================================
# TestCallGeminiDirect
# ===================================================================

class TestCallGeminiDirect:
    """Tests for _call_gemini_direct."""

    def test_strips_gemini_prefix(self):
        mock_genai = MagicMock()
        mock_response = MagicMock()
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        mock_types = _mock_types()
        config = MagicMock()
        for attr in ("max_output_tokens", "temperature", "top_p", "top_k",
                      "stop_sequences", "thinking_config", "safety_settings",
                      "candidate_count", "tool_config"):
            setattr(config, attr, None)

        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai), \
             patch("skell_e_router.gemini_direct.types", mock_types):
            from skell_e_router.gemini_direct import _call_gemini_direct
            result = _call_gemini_direct("gemini/gemini-2.5-flash", [], None, config, None, FAKE_GEMINI_KEY)

        call_args = mock_genai.Client.return_value.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-2.5-flash"
        response, request_duration = result
        assert response is mock_response
        assert isinstance(request_duration, float)

    @patch("skell_e_router.gemini_direct.time.sleep")
    def test_retries_transient_errors(self, mock_sleep):
        mock_genai = MagicMock()
        err = Exception("server error")
        err.code = 503
        mock_genai.Client.return_value.models.generate_content.side_effect = [err, err, MagicMock()]

        config = MagicMock()
        for attr in ("max_output_tokens", "temperature", "top_p", "top_k",
                      "stop_sequences", "thinking_config", "safety_settings",
                      "candidate_count", "tool_config"):
            setattr(config, attr, None)

        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai), \
             patch("skell_e_router.gemini_direct.types", _mock_types()):
            from skell_e_router.gemini_direct import _call_gemini_direct
            result = _call_gemini_direct("gemini-2.5-flash", [], None, config, None, FAKE_GEMINI_KEY)

        response, request_duration = result
        assert response is not None
        assert isinstance(request_duration, float)
        assert mock_sleep.call_count == 2

    def test_no_retry_on_400(self):
        mock_genai = MagicMock()
        err = Exception("bad request")
        err.code = 400
        mock_genai.Client.return_value.models.generate_content.side_effect = err

        config = MagicMock()
        for attr in ("max_output_tokens", "temperature", "top_p", "top_k",
                      "stop_sequences", "thinking_config", "safety_settings",
                      "candidate_count", "tool_config"):
            setattr(config, attr, None)

        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai), \
             patch("skell_e_router.gemini_direct.types", _mock_types()):
            from skell_e_router.gemini_direct import _call_gemini_direct
            with pytest.raises(Exception, match="bad request"):
                _call_gemini_direct("gemini-2.5-flash", [], None, config, None, FAKE_GEMINI_KEY)

        assert mock_genai.Client.return_value.models.generate_content.call_count == 1

    def test_import_error_when_no_genai(self):
        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", False):
            from skell_e_router.gemini_direct import _call_gemini_direct
            with pytest.raises(ImportError, match="google-genai"):
                _call_gemini_direct("gemini-2.5-flash", [], None, None, None, FAKE_GEMINI_KEY)


# ===================================================================
# TestBuildResponse
# ===================================================================

class TestBuildResponse:
    """Tests for _build_response."""

    def test_content_tokens_finish_reason_grounding(self):
        resp = make_gemini_response(
            text="Hello",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            finish_reason="STOP",
            grounding_metadata={"key": "value"},
        )
        from skell_e_router.gemini_direct import _build_response
        ai_resp = _build_response(resp, "gemini/gemini-2.5-flash", 1.5)
        assert ai_resp.content == "Hello"
        assert ai_resp.prompt_tokens == 5
        assert ai_resp.completion_tokens == 10
        assert ai_resp.total_tokens == 15
        assert ai_resp.finish_reason == "STOP"
        assert ai_resp.grounding_metadata == {"key": "value"}
        assert ai_resp.model == "gemini-2.5-flash"
        assert ai_resp.duration_seconds == 1.5

    def test_cost_calculation(self):
        resp = make_gemini_response(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        from skell_e_router.gemini_direct import _build_response
        ai_resp = _build_response(resp, "gemini/gemini-2.5-flash", 1.0)
        # 1M * 0.30/1M + 1M * 2.50/1M = 0.30 + 2.50 = 2.80
        assert ai_resp.cost == pytest.approx(2.80)

    def test_no_cost_for_unknown_model(self):
        resp = make_gemini_response(prompt_tokens=100, completion_tokens=100)
        from skell_e_router.gemini_direct import _build_response
        ai_resp = _build_response(resp, "gemini/unknown-model", 1.0)
        assert ai_resp.cost is None

    def test_blocked_response(self):
        resp = make_gemini_response(blocked=True)
        from skell_e_router.gemini_direct import _build_response
        ai_resp = _build_response(resp, "gemini/gemini-2.5-flash", 0.5)
        assert ai_resp.content == ""

    def test_safety_ratings_extracted(self):
        ratings = [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "LOW"}]
        resp = make_gemini_response(safety_ratings=ratings)
        from skell_e_router.gemini_direct import _build_response
        ai_resp = _build_response(resp, "gemini/gemini-2.5-flash", 1.0)
        assert ai_resp.safety_ratings == ratings

    def test_tool_calls_extraction(self):
        resp = make_gemini_response(
            text=None,
            function_call_parts=[
                {"name": "get_weather", "args": {"city": "NYC"}}
            ]
        )
        # Fix: text part shouldn't be added when text is None
        # The mock adds a text part even for None; remove it
        resp.candidates[0].content.parts = [p for p in resp.candidates[0].content.parts if p.function_call is not None]
        resp.text = ""

        from skell_e_router.gemini_direct import _build_response
        ai_resp = _build_response(resp, "gemini/gemini-2.5-flash", 1.0)
        assert ai_resp.tool_calls is not None
        assert len(ai_resp.tool_calls) == 1
        tc = ai_resp.tool_calls[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}
        assert tc["id"].startswith("call_")


# ===================================================================
# TestStreamGeminiDirect
# ===================================================================

class TestStreamGeminiDirect:
    """Tests for _call_gemini_direct_stream."""

    def test_returns_stream_iterator(self):
        mock_genai = MagicMock()
        chunks = [MagicMock(), MagicMock()]
        mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(chunks)

        config = MagicMock()
        for attr in ("max_output_tokens", "temperature", "top_p", "top_k",
                      "stop_sequences", "thinking_config", "safety_settings",
                      "candidate_count", "tool_config"):
            setattr(config, attr, None)

        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai), \
             patch("skell_e_router.gemini_direct.types", _mock_types()):
            from skell_e_router.gemini_direct import _call_gemini_direct_stream
            result = _call_gemini_direct_stream("gemini-2.5-flash", [], None, config, None, FAKE_GEMINI_KEY)

        collected = list(result)
        assert len(collected) == 2

    def test_chunks_have_content(self):
        mock_genai = MagicMock()
        chunk = MagicMock()
        chunk.text = "Hello"
        mock_genai.Client.return_value.models.generate_content_stream.return_value = iter([chunk])

        config = MagicMock()
        for attr in ("max_output_tokens", "temperature", "top_p", "top_k",
                      "stop_sequences", "thinking_config", "safety_settings",
                      "candidate_count", "tool_config"):
            setattr(config, attr, None)

        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai), \
             patch("skell_e_router.gemini_direct.types", _mock_types()):
            from skell_e_router.gemini_direct import _call_gemini_direct_stream
            result = _call_gemini_direct_stream("gemini-2.5-flash", [], None, config, None, FAKE_GEMINI_KEY)

        items = list(result)
        assert items[0].text == "Hello"


# ===================================================================
# TestAskAiDirectGeminiIntegration
# ===================================================================

class TestAskAiDirectGeminiIntegration:
    """Integration tests for _ask_ai_direct_gemini and ask_ai routing."""

    def _make_flash_model(self):
        return make_model(
            provider="gemini",
            name="gemini/gemini-2.5-flash",
            supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens",
                              "reasoning_effort", "stream", "tools", "tool_choice",
                              "candidate_count", "safety_settings", "web_search_options"},
            accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        )

    @patch("skell_e_router.utils._call_gemini_direct")
    @patch("skell_e_router.utils._convert_messages_to_contents")
    @patch("skell_e_router.utils._build_generate_config")
    def test_direct_sdk_true_routes_correctly(self, mock_config, mock_convert, mock_call):
        """direct_sdk=True should use the direct Gemini path."""
        mock_convert.return_value = (None, [])
        mock_config.return_value = (MagicMock(), None)
        mock_resp = make_gemini_response()
        mock_call.return_value = (mock_resp, 0.5)

        from skell_e_router.utils import _ask_ai_direct_gemini
        model = self._make_flash_model()
        result = _ask_ai_direct_gemini(
            model, [{"role": "user", "content": "hi"}],
            FAKE_GEMINI_KEY, "none", False, None, {}
        )
        assert result == "Hello from Gemini"
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
            "test-flash": make_model(
                provider="gemini",
                name="gemini/gemini-2.5-flash",
                supported_params={"temperature", "stream", "safety_settings"},
            )
        }):
            # Force use_direct_sdk to True on model, but pass direct_sdk=False to ask_ai
            MODEL_CONFIG["test-flash"].use_direct_sdk = True
            result = ask_ai(
                "test-flash", "hi",
                config={"gemini_api_key": FAKE_GEMINI_KEY},
                direct_sdk=False
            )
        assert result == "from litellm"
        mock_completion.assert_called_once()

    @patch("skell_e_router.utils._call_gemini_direct_stream")
    @patch("skell_e_router.utils._convert_messages_to_contents")
    @patch("skell_e_router.utils._build_generate_config")
    def test_stream_true_uses_direct_streaming(self, mock_config, mock_convert, mock_stream):
        """stream=True should call _call_gemini_direct_stream."""
        mock_convert.return_value = (None, [])
        mock_config.return_value = (MagicMock(), None)
        sentinel = iter(["chunk1", "chunk2"])
        mock_stream.return_value = sentinel

        from skell_e_router.utils import _ask_ai_direct_gemini
        model = self._make_flash_model()
        result = _ask_ai_direct_gemini(
            model, [{"role": "user", "content": "hi"}],
            FAKE_GEMINI_KEY, "none", False, None, {"stream": True}
        )
        assert result is sentinel
        mock_stream.assert_called_once()

    @patch("skell_e_router.utils._call_gemini_direct")
    @patch("skell_e_router.utils._convert_messages_to_contents")
    @patch("skell_e_router.utils._build_generate_config")
    def test_budget_tokens_transformed(self, mock_config, mock_convert, mock_call):
        """budget_tokens should be handled by _build_generate_config (no crash)."""
        mock_convert.return_value = (None, [])
        mock_config.return_value = (MagicMock(), None)
        mock_call.return_value = (make_gemini_response(), 0.5)

        from skell_e_router.utils import _ask_ai_direct_gemini
        model = self._make_flash_model()
        # Should not raise — budget_tokens is passed to _build_generate_config
        result = _ask_ai_direct_gemini(
            model, [{"role": "user", "content": "hi"}],
            FAKE_GEMINI_KEY, "none", False, None, {"budget_tokens": 2048}
        )
        assert result == "Hello from Gemini"

    def test_reasoning_effort_validated(self):
        """Invalid reasoning_effort should raise RouterError."""
        from skell_e_router.utils import _ask_ai_direct_gemini, RouterError

        mock_types = _mock_types()
        with patch("skell_e_router.gemini_direct.types", mock_types):
            model = self._make_flash_model()
            with pytest.raises(RouterError, match="INVALID_PARAM"):
                _ask_ai_direct_gemini(
                    model, [{"role": "user", "content": "hi"}],
                    FAKE_GEMINI_KEY, "none", False, None, {"reasoning_effort": "turbo"}
                )

    def test_missing_key_raises(self):
        """Missing API key should raise RouterError."""
        from skell_e_router.utils import _ask_ai_direct_gemini, RouterError

        model = self._make_flash_model()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RouterError, match="MISSING_ENV"):
                _ask_ai_direct_gemini(
                    model, [{"role": "user", "content": "hi"}],
                    None, "none", False, None, {}
                )

    @patch("skell_e_router.utils._call_gemini_direct")
    @patch("skell_e_router.utils._convert_messages_to_contents")
    @patch("skell_e_router.utils._build_generate_config")
    def test_provider_error_wrapped(self, mock_config, mock_convert, mock_call):
        """SDK exceptions should be wrapped in RouterError(PROVIDER_ERROR)."""
        mock_convert.return_value = (None, [])
        mock_config.return_value = (MagicMock(), None)
        mock_call.side_effect = RuntimeError("SDK boom")

        from skell_e_router.utils import _ask_ai_direct_gemini, RouterError
        model = self._make_flash_model()
        with pytest.raises(RouterError, match="PROVIDER_ERROR"):
            _ask_ai_direct_gemini(
                model, [{"role": "user", "content": "hi"}],
                FAKE_GEMINI_KEY, "none", False, None, {}
            )
