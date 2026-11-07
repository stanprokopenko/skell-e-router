"""Regression tests for the 2026-11-06 provider-integration audit fixes (v3.44.0).

Each test names its audit finding id. These use the real provider SDK types
(google-genai, anthropic, litellm, httpx) rather than mocks, so they also pin
the SDK attribute names the fixes rely on.
"""

import base64
import json
import logging

import httpx
import pytest

from skell_e_router.model_config import MODEL_CONFIG
from tests.helpers import make_model

genai_types = pytest.importorskip("google.genai.types")
anthropic = pytest.importorskip("anthropic")


# model_dump(mode="json") renders Part.thought_signature bytes as base64.
SKIP_B64 = base64.b64encode(b"skip_thought_signature_validator").decode()


def _dump(contents):
    return [c.model_dump(mode="json", exclude_none=True) for c in contents]


# ---------------------------------------------------------------------------
# G1 / G10: tool history reaches Gemini as functionCall / functionResponse
# ---------------------------------------------------------------------------

class TestGeminiToolHistory:
    def _convert(self, messages):
        from skell_e_router.gemini_direct import _convert_messages_to_contents
        system, contents = _convert_messages_to_contents(messages)
        return system, _dump(contents)

    def test_g1_tool_call_and_result_become_function_parts(self):
        _, contents = self._convert([
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
        ])
        assert contents == [
            {"role": "user", "parts": [{"text": "What time is it?"}]},
            {"role": "model", "parts": [
                {"text": "Checking."},
                {"function_call": {"id": "call_1", "name": "get_time", "args": {"tz": "UTC"}},
                 "thought_signature": SKIP_B64}]},
            {"role": "user", "parts": [
                {"function_response": {"id": "call_1", "name": "get_time",
                                       "response": {"output": "12:00"}}}]},
        ]

    def test_g1_parallel_results_share_one_user_turn_in_order(self):
        calls = [{"id": f"c{i}", "type": "function",
                  "function": {"name": name, "arguments": "{}"}}
                 for i, name in enumerate(["a", "b", "c"])]
        _, contents = self._convert([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c0", "content": "r0"},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "user", "content": "thanks"},
        ])
        assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
        # Tool-calls-only assistant turn is kept (content None, no text part)
        assert [p["function_call"]["id"] for p in contents[1]["parts"]] == ["c0", "c1", "c2"]
        responses = [p["function_response"] for p in contents[2]["parts"]]
        assert [(r["id"], r["name"], r["response"]) for r in responses] == [
            ("c0", "a", {"output": "r0"}), ("c1", "b", {"output": "r1"}),
            ("c2", "c", {"output": "r2"})]
        # A user message after the responses stays its own user turn
        assert contents[3] == {"role": "user", "parts": [{"text": "thanks"}]}

    def test_g1_orphan_tool_result_degrades_to_labeled_text(self):
        """No matching call and no name: a functionResponse would 400, so text."""
        _, contents = self._convert([
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "lost_1", "content": "result"},
        ])
        assert contents[1] == {"role": "user", "parts": [{"text": "[tool result lost_1]\nresult"}]}

    def test_g1_tool_message_name_used_when_call_unknown(self):
        _, contents = self._convert([
            {"role": "tool", "tool_call_id": "x1", "name": "lookup", "content": "ok"},
        ])
        assert contents[0]["parts"][0]["function_response"]["name"] == "lookup"

    def test_g10_empty_text_never_becomes_a_part(self):
        _, contents = self._convert([
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": [{"type": "text", "text": ""},
                                         {"type": "text", "text": "real"}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "e1", "type": "function", "function": {"name": "run", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "e1", "content": ""},
        ])
        assert contents == [
            {"role": "user", "parts": [{"text": "real"}]},
            {"role": "model", "parts": [{"function_call": {"id": "e1", "name": "run", "args": {}},
                                         "thought_signature": SKIP_B64}]},
            {"role": "user", "parts": [{"function_response": {
                "id": "e1", "name": "run", "response": {"output": ""}}}]},
        ]

    def test_g10_all_empty_conversation_raises_invalid_param(self):
        from skell_e_router.utils import RouterError, _ask_ai_direct_gemini
        with pytest.raises(RouterError) as exc:
            _ask_ai_direct_gemini(MODEL_CONFIG["gemini-2.5-flash"],
                                  [{"role": "user", "content": ""}], "fake-key",
                                  "none", False, None, {})
        assert exc.value.code == "INVALID_PARAM"


# ---------------------------------------------------------------------------
# G2 / G4 / E10: Gemini response building
# ---------------------------------------------------------------------------

def _gemini_response(parts, finish="STOP", usage=None):
    return genai_types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}],
        "usage_metadata": usage or {"prompt_token_count": 10, "candidates_token_count": 5,
                                    "total_token_count": 15},
    })


class TestGeminiBuildResponse:
    def _build(self, resp):
        from skell_e_router.gemini_direct import _build_response
        return _build_response(resp, "gemini/gemini-2.5-flash", 0.1)

    def test_g2_gemini_function_call_id_is_kept(self):
        out = self._build(_gemini_response([
            {"function_call": {"name": "get_time", "args": {}, "id": "fc_gem_1"}}]))
        assert out.tool_calls[0]["id"] == "fc_gem_1"

    def test_g2_id_minted_only_when_absent(self):
        out = self._build(_gemini_response([
            {"function_call": {"name": "get_time", "args": {}}}]))
        assert out.tool_calls[0]["id"].startswith("call_")

    @pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION"])
    def test_g4_finish_reason_is_the_enum_name(self, finish):
        out = self._build(_gemini_response([{"text": "partial"}], finish=finish))
        assert out.finish_reason == finish

    def test_e10_cached_content_tokens_reported(self):
        out = self._build(_gemini_response([{"text": "hi"}], usage={
            "prompt_token_count": 1000, "candidates_token_count": 5,
            "total_token_count": 1005, "cached_content_token_count": 800}))
        assert out.cache_read_tokens == 800
        assert out.cache_write_tokens is None

    def test_e10_no_cache_count_is_none(self):
        out = self._build(_gemini_response([{"text": "hi"}]))
        assert out.cache_read_tokens is None


# ---------------------------------------------------------------------------
# G3: Gemini thought_signature round trip on function calls
# ---------------------------------------------------------------------------

SIG = bytes([0, 1, 255]) + b"real-sig"
SIG_B64 = base64.b64encode(SIG).decode()


def _call(call_id, name="get_time", sig=None):
    tc = {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
    if sig is not None:
        tc["thought_signature"] = sig
    return tc


class TestGeminiThoughtSignature:
    def _model_parts(self, messages):
        from skell_e_router.gemini_direct import _convert_messages_to_contents
        _, contents = _convert_messages_to_contents(messages)
        return [c.parts for c in contents if c.role == "model"]

    def test_g3_build_response_returns_signature_as_base64(self):
        from skell_e_router.gemini_direct import _build_response
        out = _build_response(_gemini_response([
            {"function_call": {"name": "get_time", "args": {}, "id": "fc_1"},
             "thought_signature": SIG_B64},
            {"function_call": {"name": "get_time", "args": {}, "id": "fc_2"}}]),
            "gemini/gemini-3.8-flash", 0.1)
        assert out.tool_calls[0]["thought_signature"] == SIG_B64
        assert "thought_signature" not in out.tool_calls[1]

    def test_g3_real_signature_is_replayed_as_bytes_on_its_part(self):
        [parts] = self._model_parts([
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1", sig=SIG_B64)]},
            {"role": "tool", "tool_call_id": "c1", "content": "noon"},
        ])
        assert parts[0].function_call.id == "c1"
        assert parts[0].thought_signature == SIG

    def test_g3_first_call_without_signature_gets_the_skip_value(self):
        from skell_e_router.gemini_direct import SKIP_THOUGHT_SIGNATURE
        [parts] = self._model_parts([
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [_call("c1"), _call("c2")]},
        ])
        assert parts[0].text == "Checking." and parts[0].thought_signature is None
        assert parts[1].thought_signature == SKIP_THOUGHT_SIGNATURE == b"skip_thought_signature_validator"
        # Later calls in the same turn need no signature.
        assert parts[2].thought_signature is None

    def test_g3_later_calls_keep_their_own_signature_and_none_is_added(self):
        [parts] = self._model_parts([
            {"role": "assistant", "content": None, "tool_calls": [
                _call("c1", sig=SIG_B64), _call("c2"), _call("c3", sig=SIG_B64)]},
        ])
        assert [p.thought_signature for p in parts] == [SIG, None, SIG]

    def test_g3_each_model_turn_gets_its_own_first_call_signature(self):
        from skell_e_router.gemini_direct import SKIP_THOUGHT_SIGNATURE
        turns = self._model_parts([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [_call("a1", sig=SIG_B64)]},
            {"role": "tool", "tool_call_id": "a1", "content": "r"},
            {"role": "assistant", "content": None, "tool_calls": [_call("b1")]},
            {"role": "tool", "tool_call_id": "b1", "content": "r"},
            {"role": "assistant", "content": "done"},
        ])
        assert [[p.thought_signature for p in t] for t in turns] == [
            [SIG], [SKIP_THOUGHT_SIGNATURE], [None]]

    def test_g3_url_safe_signature_is_accepted(self):
        [parts] = self._model_parts([
            {"role": "assistant", "content": None, "tool_calls": [
                _call("c1", sig=base64.urlsafe_b64encode(SIG).decode())]},
        ])
        assert parts[0].thought_signature == SIG

    def test_g3_malformed_signature_falls_back_to_skip(self):
        from skell_e_router.gemini_direct import SKIP_THOUGHT_SIGNATURE
        [parts] = self._model_parts([
            {"role": "assistant", "content": None, "tool_calls": [_call("c1", sig="not base64!")]},
        ])
        assert parts[0].thought_signature == SKIP_THOUGHT_SIGNATURE

    def test_g3_anthropic_converter_ignores_the_key(self):
        from skell_e_router.anthropic_direct import _convert_messages_for_anthropic
        _, converted = _convert_messages_for_anthropic([
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1", sig=SIG_B64)]},
            {"role": "tool", "tool_call_id": "c1", "content": "noon"},
        ])
        [block] = converted[1]["content"]
        assert set(block) == {"type", "id", "name", "input"}
        assert "thought_signature" not in json.dumps(converted)

    def test_g3_litellm_path_strips_the_key(self, monkeypatch):
        from tests.helpers import make_litellm_response
        from skell_e_router import utils
        sent = {}

        def fake_completion(**kwargs):
            sent.update(kwargs)
            return make_litellm_response()

        monkeypatch.setattr(utils.litellm, "completion", fake_completion)
        messages = [
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("c1", sig=SIG_B64)]},
            {"role": "tool", "tool_call_id": "c1", "content": "noon"},
        ]
        utils.ask_ai("gpt-5", messages, config={"openai_api_key": "sk-test"})
        assert "thought_signature" not in json.dumps(sent["messages"])
        assert sent["messages"][1]["tool_calls"][0]["id"] == "c1"
        # The caller's transcript is not mutated.
        assert messages[1]["tool_calls"][0]["thought_signature"] == SIG_B64


# ---------------------------------------------------------------------------
# G9 / P5: gemini-2.5-flash reasoning maps to thinking_budget
# ---------------------------------------------------------------------------

class TestGemini25FlashThinking:
    def test_g9_reasoning_effort_becomes_thinking_budget(self):
        from skell_e_router.gemini_direct import _build_generate_config
        from skell_e_router.utils import _handle_model_specific_params
        model = MODEL_CONFIG["gemini-2.5-flash"]
        kwargs = {"reasoning_effort": "low"}
        _handle_model_specific_params(model, kwargs)
        config, _ = _build_generate_config(model, kwargs)
        assert config.thinking_config.thinking_budget == 1024
        assert config.thinking_config.thinking_level is None

    def test_g9_undeclared_effort_never_sends_thinking_level(self):
        """A model declaring neither reasoning_effort nor budget_tokens gets a budget."""
        from skell_e_router.gemini_direct import _build_generate_config
        model = make_model(provider="gemini", name="gemini/gemini-2.5-x",
                           supported_params={"max_tokens", "stream"})
        config, _ = _build_generate_config(model, {"reasoning_effort": "medium"})
        assert config.thinking_config.thinking_budget == 2048
        assert config.thinking_config.thinking_level is None

    def test_gemini_3_still_gets_thinking_level(self):
        from skell_e_router.gemini_direct import _build_generate_config
        config, _ = _build_generate_config(MODEL_CONFIG["gemini-3.5-flash"],
                                           {"reasoning_effort": "high"})
        assert config.thinking_config.thinking_level is not None
        assert config.thinking_config.thinking_budget is None


# ---------------------------------------------------------------------------
# R1 / E10: LiteLLM responses with split choices; cached tokens
# ---------------------------------------------------------------------------

class TestLiteLLMBuildAIResponse:
    def _build(self, response):
        from skell_e_router.utils import _build_ai_response
        response._hidden_params["router_responses_output_parts"] = True
        return _build_ai_response(response, 0.1, 0.1)

    def _two_choice_response(self):
        import litellm
        return litellm.ModelResponse(
            model="gpt-6-luna",
            choices=[
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Checking the clock."}},
                {"index": 1, "finish_reason": "tool_calls",
                 "message": {"role": "assistant", "content": None, "tool_calls": [
                     {"id": "call_a", "type": "function",
                      "function": {"name": "get_time", "arguments": "{}"}},
                     {"id": "call_b", "type": "function",
                      "function": {"name": "get_date", "arguments": "{}"}}]}},
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                   "prompt_tokens_details": {"cached_tokens": 64}},
        )

    def test_r1_text_and_tool_calls_from_separate_choices_are_merged(self):
        out = self._build(self._two_choice_response())
        assert out.content == "Checking the clock."
        assert [tc.id for tc in out.tool_calls] == ["call_a", "call_b"]
        assert out.finish_reason == "tool_calls"

    def test_r1_single_choice_unchanged(self):
        import litellm
        out = self._build(litellm.ModelResponse(
            model="gpt-4.1",
            choices=[{"index": 0, "finish_reason": "length",
                      "message": {"role": "assistant", "content": "cut off"}}]))
        assert out.content == "cut off"
        assert out.finish_reason == "length"
        assert not out.tool_calls

    def test_e10_cached_tokens_reported(self):
        out = self._build(self._two_choice_response())
        assert out.cache_read_tokens == 64
        assert out.cache_write_tokens is None


# ---------------------------------------------------------------------------
# E1: Retry-After from httpx.Headers
# ---------------------------------------------------------------------------

class TestRetryAfterHeaders:
    def test_e1_httpx_headers_are_honored(self):
        from skell_e_router.utils import _parse_retry_after_seconds
        assert _parse_retry_after_seconds(httpx.Headers({"retry-after": "30"})) == 30.0

    def test_e1_long_retry_after_on_httpx_response_stops_retrying(self):
        from skell_e_router.utils import _is_retryable_exception

        class RateLimited(Exception):
            status_code = 429
            response = None

        exc = RateLimited()
        exc.headers = httpx.Headers({"Retry-After": "600"})
        assert _is_retryable_exception(exc) is False


# ---------------------------------------------------------------------------
# E3: Anthropic 529 overloaded is transient
# ---------------------------------------------------------------------------

class TestAnthropicOverloaded:
    def test_e3_overloaded_error_is_transient(self):
        from skell_e_router.anthropic_direct import _is_transient_anthropic_error
        response = httpx.Response(529, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        exc = anthropic.OverloadedError("Overloaded", response=response,
                                        body={"type": "error", "error": {"type": "overloaded_error"}})
        assert exc.status_code == 529
        assert _is_transient_anthropic_error(exc) is True


# ---------------------------------------------------------------------------
# E10 / E11 / E12: Anthropic usage, cache-write TTL pricing, Haiku 5.5 tier
# ---------------------------------------------------------------------------

def _anthropic_message(model, usage):
    return anthropic.types.Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": usage,
    })


class TestAnthropicUsagePricing:
    def _build(self, model, usage):
        from skell_e_router.anthropic_direct import _build_response
        return _build_response(_anthropic_message(model, usage), f"anthropic/{model}", 0.1)

    def test_e11_one_hour_writes_bill_at_2x_and_five_minute_at_1_25x(self):
        out = self._build("claude-opus-5-5", {
            "input_tokens": 100, "output_tokens": 10,
            "cache_creation_input_tokens": 10_000, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 4_000,
                               "ephemeral_1h_input_tokens": 6_000}})
        # opus-5-5: $4 in, $20 out
        expected = (100 * 4 + 4_000 * 4 * 1.25 + 6_000 * 4 * 2.0 + 10 * 20) / 1_000_000
        assert out.cost == pytest.approx(expected)

    def test_e11_no_breakdown_falls_back_to_1_25x(self):
        out = self._build("claude-opus-5-5", {
            "input_tokens": 100, "output_tokens": 10,
            "cache_creation_input_tokens": 10_000, "cache_read_input_tokens": 0})
        expected = (100 * 4 + 10_000 * 4 * 1.25 + 10 * 20) / 1_000_000
        assert out.cost == pytest.approx(expected)

    def test_e10_cache_token_fields(self):
        out = self._build("claude-opus-5-5", {
            "input_tokens": 100, "output_tokens": 10,
            "cache_creation_input_tokens": 300, "cache_read_input_tokens": 700})
        assert out.cache_read_tokens == 700
        assert out.cache_write_tokens == 300
        assert out.prompt_tokens == 1100

    def test_e12_haiku_5_5_at_100k_uses_base_tier(self):
        out = self._build("claude-haiku-5-5", {"input_tokens": 100_000, "output_tokens": 1_000})
        assert out.cost == pytest.approx((100_000 * 0.10 + 1_000 * 0.50) / 1_000_000)

    def test_e12_haiku_5_5_above_100k_bills_whole_request_at_high_tier(self):
        # Prompt size counts cache reads too: 60k uncached + 50k read = 110k
        out = self._build("claude-haiku-5-5", {
            "input_tokens": 60_000, "output_tokens": 1_000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 50_000})
        expected = (60_000 * 0.50 + 50_000 * 0.50 * 0.10 + 1_000 * 2.50) / 1_000_000
        assert out.cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# A3 / P8: forced tool_choice with budget thinking falls back to auto
# ---------------------------------------------------------------------------

class TestAnthropicForcedToolChoiceWithThinking:
    def _params(self, alias, kwargs):
        from skell_e_router.anthropic_direct import _build_create_params
        params, _ = _build_create_params(MODEL_CONFIG[alias], kwargs)
        return params

    @pytest.mark.parametrize("choice", [
        "required", {"type": "function", "function": {"name": "get_time"}}])
    def test_a3_budget_thinking_downgrades_forced_choice(self, choice, caplog):
        with caplog.at_level(logging.WARNING, logger="skell_e_router.anthropic_direct"):
            params = self._params("claude-haiku-4-5", {
                "thinking": {"type": "enabled", "budget_tokens": 1024}, "tool_choice": choice})
        assert params["thinking"]["type"] == "enabled"
        assert params["tool_choice"] == {"type": "auto"}
        assert "incompatible with extended thinking" in caplog.text

    def test_a3_budget_tokens_kwarg_also_downgrades(self):
        params = self._params("claude-opus-4-5", {"budget_tokens": 2048, "tool_choice": "required"})
        assert params["tool_choice"] == {"type": "auto"}

    def test_a3_no_thinking_keeps_forced_choice(self):
        params = self._params("claude-haiku-4-5", {"tool_choice": "required"})
        assert params["tool_choice"] == {"type": "any"}

    def test_a3_adaptive_thinking_keeps_forced_choice(self):
        params = self._params("claude-haiku-5-5", {"reasoning_effort": "low", "tool_choice": "required"})
        assert params["thinking"] == {"type": "adaptive"}
        assert params["tool_choice"] == {"type": "any"}
