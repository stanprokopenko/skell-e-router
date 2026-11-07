"""Tests for the reasoning-state channel (AIResponse.reasoning_state).

Covers capture from Anthropic and from LiteLLM's OpenAI Responses bridge,
replay to Anthropic (thinking blocks) and to the Responses bridge (reasoning
input items), stripping everywhere else, and the request ``include`` flag.
Uses real SDK types (anthropic, litellm, httpx) like test_audit_fixes.
"""

import copy
import dataclasses
import json

import pytest

from tests.test_audit_fixes import _DUMMY_KEYS, _LOOKUP_TOOL, _dump, captured_bodies  # noqa: F401

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("google.genai.types")

ENC_INCLUDE = "reasoning.encrypted_content"

THINKING = {"provider": "anthropic", "kind": "thinking",
            "text": "Let me think: 2+2 → 4.\n", "signature": "EqQBCkYIBxgCKkB+sig/=="}
REDACTED = {"provider": "anthropic", "kind": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4a"}
OPENAI_ITEM = {"provider": "openai", "kind": "reasoning", "id": "rs_1",
               "encrypted_content": "ENC", "summary": [{"type": "summary_text", "text": "s"}]}


def _tool_call(call_id="call_1", name="lookup"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


# ---------------------------------------------------------------------------
# Anthropic converter: replay thinking blocks
# ---------------------------------------------------------------------------

class TestAnthropicReplay:
    def _convert(self, messages):
        from skell_e_router.anthropic_direct import _convert_messages_for_anthropic
        return _convert_messages_for_anthropic(messages)[1]

    def test_tool_call_turn_puts_thinking_first_then_text_then_tool_use(self):
        converted = self._convert([
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": "Looking.", "tool_calls": [_tool_call()],
             "reasoning_state": [THINKING, REDACTED]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ])
        blocks = converted[1]["content"]
        assert [b["type"] for b in blocks] == ["thinking", "redacted_thinking", "text", "tool_use"]
        assert blocks[0] == {"type": "thinking", "thinking": THINKING["text"],
                             "signature": THINKING["signature"]}
        assert blocks[1] == {"type": "redacted_thinking", "data": REDACTED["data"]}
        assert blocks[2] == {"type": "text", "text": "Looking."}
        assert blocks[3]["id"] == "call_1" and blocks[3]["name"] == "lookup"

    def test_string_content_turn_becomes_thinking_then_text(self):
        converted = self._convert([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello.", "reasoning_state": [THINKING]},
            {"role": "user", "content": "again"},
        ])
        assert converted[1]["content"] == [
            {"type": "thinking", "thinking": THINKING["text"], "signature": THINKING["signature"]},
            {"type": "text", "text": "Hello."},
        ]

    def test_foreign_unsigned_and_non_dict_entries_are_dropped(self):
        converted = self._convert([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello.", "reasoning_state": [
                OPENAI_ITEM,
                {"provider": "anthropic", "kind": "thinking", "text": "unsigned"},
                "not a dict",
                THINKING,
            ]},
        ])
        blocks = converted[1]["content"]
        assert [b["type"] for b in blocks] == ["thinking", "text"]
        assert blocks[0]["signature"] == THINKING["signature"]
        dumped = json.dumps(converted)
        assert "ENC" not in dumped and "unsigned" not in dumped and "rs_1" not in dumped

    def test_only_foreign_entries_leave_string_content_unchanged(self):
        converted = self._convert([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello.", "reasoning_state": [OPENAI_ITEM]},
        ])
        assert converted[1] == {"role": "assistant", "content": "Hello."}

    def test_no_reasoning_state_output_is_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": "Looking.", "tool_calls": [_tool_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
        converted = self._convert(messages)
        assert converted[1] == {"role": "assistant", "content": "Hello."}
        assert [b["type"] for b in converted[3]["content"]] == ["text", "tool_use"]
        assert "thinking" not in json.dumps(converted)

    def test_caller_messages_are_not_mutated(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": "Looking.", "tool_calls": [_tool_call()],
             "reasoning_state": [THINKING, REDACTED]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "assistant", "content": "Done.", "reasoning_state": [THINKING]},
        ]
        before = copy.deepcopy(messages)
        self._convert(messages)
        assert messages == before


class TestAnthropicMergeConsecutiveRoles:
    def test_merged_assistant_turn_hoists_thinking_to_front(self):
        from skell_e_router.anthropic_direct import _merge_consecutive_roles
        think = {"type": "thinking", "thinking": "t", "signature": "s"}
        redacted = {"type": "redacted_thinking", "data": "d"}
        merged = _merge_consecutive_roles([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"},
                                              {"type": "tool_use", "id": "t1", "name": "n", "input": {}}]},
            {"role": "assistant", "content": [think, redacted, {"type": "text", "text": "b"}]},
        ])
        assert len(merged) == 2
        assert merged[1]["content"] == [
            think, redacted,
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "t1", "name": "n", "input": {}},
            {"type": "text", "text": "b"},
        ]
        # Hoisted blocks are not altered.
        assert merged[1]["content"][0] is think

    def test_user_user_merge_unchanged(self):
        from skell_e_router.anthropic_direct import _merge_consecutive_roles
        merged = _merge_consecutive_roles([
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
            {"role": "user", "content": "please continue"},
        ])
        assert merged == [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "r"},
            {"type": "text", "text": "please continue"},
        ]}]


# ---------------------------------------------------------------------------
# Anthropic capture: thinking blocks become reasoning_state
# ---------------------------------------------------------------------------

def _anthropic_response(content):
    return anthropic.types.Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


class TestAnthropicCapture:
    def _build(self, content):
        from skell_e_router.anthropic_direct import _build_response
        return _build_response(_anthropic_response(content), "anthropic/claude-opus-5-5", 0.1)

    def test_thinking_text_tool_use(self):
        out = self._build([
            {"type": "thinking", "thinking": THINKING["text"], "signature": THINKING["signature"]},
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": 1}},
        ])
        assert out.reasoning_state == [THINKING]
        assert out.content == "Looking."
        assert len(out.tool_calls) == 1

    def test_redacted_then_thinking_keeps_order(self):
        out = self._build([
            {"type": "redacted_thinking", "data": REDACTED["data"]},
            {"type": "thinking", "thinking": THINKING["text"], "signature": THINKING["signature"]},
            {"type": "text", "text": "Answer."},
        ])
        assert [e["kind"] for e in out.reasoning_state] == ["redacted_thinking", "thinking"]
        assert out.reasoning_state == [REDACTED, THINKING]
        assert out.content == "Answer."

    def test_text_only_has_no_reasoning_state(self):
        out = self._build([{"type": "text", "text": "ok"}])
        assert out.reasoning_state is None

    def test_captured_state_replays_byte_identical(self):
        from skell_e_router.anthropic_direct import _convert_messages_for_anthropic
        out = self._build([
            {"type": "thinking", "thinking": THINKING["text"], "signature": THINKING["signature"]},
            {"type": "text", "text": "Hi."},
        ])
        _, converted = _convert_messages_for_anthropic([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": out.content, "reasoning_state": out.reasoning_state},
        ])
        assert converted[1]["content"][0] == {"type": "thinking", "thinking": THINKING["text"],
                                              "signature": THINKING["signature"]}


# ---------------------------------------------------------------------------
# LiteLLM message preparation
# ---------------------------------------------------------------------------

class TestPrepareMessagesForLiteLLM:
    def _prepare(self, messages, bridge):
        from skell_e_router.utils import _prepare_messages_for_litellm
        return _prepare_messages_for_litellm(messages, responses_bridge=bridge)

    def _assistant(self, state):
        return {"role": "assistant", "content": None, "tool_calls": [_tool_call()],
                "reasoning_state": state}

    def test_bridge_replays_only_openai_items(self):
        [out] = self._prepare([self._assistant([THINKING, OPENAI_ITEM])], bridge=True)
        assert "reasoning_state" not in out
        assert out["reasoning_items"] == [{"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC",
                                           "summary": [{"type": "summary_text", "text": "s"}]}]
        assert "thinking" not in json.dumps(out)
        assert out["tool_calls"] == [_tool_call()]

    def test_no_bridge_strips_everything(self):
        [out] = self._prepare([self._assistant([THINKING, OPENAI_ITEM])], bridge=False)
        assert "reasoning_state" not in out
        assert "reasoning_items" not in out

    def test_openai_entry_without_encrypted_content_not_replayed(self):
        entry = {k: v for k, v in OPENAI_ITEM.items() if k != "encrypted_content"}
        [out] = self._prepare([self._assistant([entry])], bridge=True)
        assert "reasoning_state" not in out
        assert "reasoning_items" not in out

    @pytest.mark.parametrize("msg", [
        {"role": "user", "content": "hi", "reasoning_state": [OPENAI_ITEM]},
        {"role": "tool", "tool_call_id": "call_1", "content": "42", "reasoning_state": [OPENAI_ITEM]},
    ])
    def test_stray_key_on_user_or_tool_is_stripped(self, msg):
        [out] = self._prepare([msg], bridge=True)
        assert "reasoning_state" not in out
        assert "reasoning_items" not in out
        assert out["content"] == msg["content"]

    def test_inputs_not_mutated_and_plain_messages_pass_through(self):
        plain_user = {"role": "user", "content": "hi"}
        plain_tool = {"role": "tool", "tool_call_id": "call_1", "content": "42"}
        with_state = self._assistant([OPENAI_ITEM])
        messages = [plain_user, with_state, plain_tool]
        before = copy.deepcopy(messages)
        out = self._prepare(messages, bridge=True)
        assert messages == before
        assert out[0] is plain_user and out[2] is plain_tool
        assert out[1] is not with_state
        # The replayed summary list is a copy, not the caller's list.
        assert out[1]["reasoning_items"][0]["summary"] is not with_state["reasoning_state"][0]["summary"]

    def test_chat_alias_end_to_end_sends_no_reasoning_keys(self, monkeypatch):
        from tests.helpers import make_litellm_response
        from skell_e_router import utils
        sent = {}

        def fake_completion(**kwargs):
            sent.update(kwargs)
            return make_litellm_response()

        monkeypatch.setattr(utils.litellm, "completion", fake_completion)
        messages = [
            {"role": "user", "content": "look it up"},
            self._assistant([THINKING, OPENAI_ITEM]),
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
        before = copy.deepcopy(messages)
        utils.ask_ai("gpt-5", messages, config={"openai_api_key": "sk-test"})
        dumped = json.dumps(sent["messages"])
        assert "reasoning_state" not in dumped and "reasoning_items" not in dumped
        assert "ENC" not in dumped
        assert sent["messages"][1]["tool_calls"][0]["id"] == "call_1"
        assert messages == before


# ---------------------------------------------------------------------------
# Request bodies: include reasoning.encrypted_content on /responses
# ---------------------------------------------------------------------------

class TestResponsesIncludeEncryptedReasoning:
    def _ask(self, captured, alias, user_input="hi", **kwargs):
        from skell_e_router import ask_ai
        assert ask_ai(alias, user_input, config=_DUMMY_KEYS, **kwargs) == "OK"
        assert len(captured) == 1
        return captured[0]

    def test_gpt6_alias_requests_encrypted_reasoning(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-6.1-sol")
        assert path.endswith("/responses")
        assert body["store"] is False
        assert body["include"] == [ENC_INCLUDE]

    def test_gpt56_tools_plus_effort_bridge_requests_encrypted_reasoning(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-5.6-sol", tools=_LOOKUP_TOOL, reasoning_effort="low")
        assert path.endswith("/responses")
        assert body["store"] is False
        assert body["include"] == [ENC_INCLUDE]

    def test_gpt56_effort_only_stays_on_chat_without_include(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-5.6-sol", reasoning_effort="low")
        assert path.endswith("/chat/completions")
        assert "include" not in body

    def test_gpt41_tools_chat_without_include(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-4.1", tools=_LOOKUP_TOOL)
        assert path.endswith("/chat/completions")
        assert "include" not in body

    def test_caller_extra_body_include_is_kept_and_extended(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-6.1-sol", extra_body={"include": ["something.else"]})
        assert path.endswith("/responses")
        assert body["include"] == ["something.else", ENC_INCLUDE]
        assert body["store"] is False

    def test_caller_top_level_include_is_kept_and_extended(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-6.1-sol", include=["something.else"])
        assert path.endswith("/responses")
        assert body["include"] == ["something.else", ENC_INCLUDE]

    def test_caller_include_already_listing_encrypted_is_not_duplicated(self, captured_bodies):
        _, body = self._ask(captured_bodies, "gpt-6.1-sol", extra_body={"include": [ENC_INCLUDE]})
        assert body["include"] == [ENC_INCLUDE]

    def _replay_history(self, entry):
        return [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_1")],
             "reasoning_state": [entry]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]

    @pytest.mark.parametrize("alias,extra", [
        ("gpt-6.1-sol", {"tools": _LOOKUP_TOOL}),
        ("gpt-5.6-sol", {"tools": _LOOKUP_TOOL, "reasoning_effort": "low"}),
    ])
    def test_openai_reasoning_replayed_right_before_its_function_call(self, captured_bodies, alias, extra):
        messages = self._replay_history(OPENAI_ITEM)
        before = copy.deepcopy(messages)
        path, body = self._ask(captured_bodies, alias, messages, **extra)
        assert path.endswith("/responses")
        assert body["store"] is False
        assert body["include"] == [ENC_INCLUDE]
        items = body["input"]
        reasoning_idx = [i for i, it in enumerate(items) if it.get("type") == "reasoning"]
        call_idx = [i for i, it in enumerate(items)
                    if it.get("type") == "function_call" and it.get("call_id") == "call_1"]
        assert len(reasoning_idx) == 1 and len(call_idx) == 1
        assert reasoning_idx[0] == call_idx[0] - 1
        reasoning = items[reasoning_idx[0]]
        assert reasoning["id"] == "rs_1"
        assert reasoning["encrypted_content"] == "ENC"
        assert reasoning["summary"] == [{"type": "summary_text", "text": "s"}]
        assert "reasoning_state" not in json.dumps(body)
        assert messages == before

    def test_anthropic_state_is_not_replayed_on_the_bridge(self, captured_bodies):
        path, body = self._ask(captured_bodies, "gpt-6.1-sol", self._replay_history(THINKING),
                               tools=_LOOKUP_TOOL)
        assert path.endswith("/responses")
        assert not [it for it in body["input"] if it.get("type") == "reasoning"]
        dumped = json.dumps(body)
        assert THINKING["signature"] not in dumped and "reasoning_state" not in dumped


# ---------------------------------------------------------------------------
# LiteLLM response capture
# ---------------------------------------------------------------------------

class TestLiteLLMCapture:
    def _build(self, response):
        from skell_e_router.utils import _build_ai_response
        response._hidden_params["router_responses_output_parts"] = True
        return _build_ai_response(response, 0.1, 0.1)

    def _two_choice_response(self, reasoning_items):
        import litellm
        return litellm.ModelResponse(
            model="gpt-6.1-sol",
            choices=[
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Checking.",
                             "reasoning_items": reasoning_items}},
                {"index": 1, "finish_reason": "tool_calls",
                 "message": {"role": "assistant", "content": None, "tool_calls": [
                     {"id": "call_a", "type": "function",
                      "function": {"name": "lookup", "arguments": "{}"}}]}},
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def test_encrypted_reasoning_item_becomes_openai_entry(self):
        out = self._build(self._two_choice_response([
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC",
             "summary": [{"type": "summary_text", "text": "s"}]}]))
        assert out.reasoning_state == [OPENAI_ITEM]
        assert out.content == "Checking."
        assert [tc.id for tc in out.tool_calls] == ["call_a"]
        assert out.finish_reason == "tool_calls"

    def test_item_without_encrypted_content_is_skipped(self):
        out = self._build(self._two_choice_response([
            {"type": "reasoning", "id": "rs_1", "summary": []}]))
        assert out.reasoning_state is None
        assert out.content == "Checking."

    def test_plain_chat_response_has_no_reasoning_state(self):
        import litellm
        out = self._build(litellm.ModelResponse(
            model="gpt-4.1",
            choices=[{"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": "hi"}}]))
        assert out.reasoning_state is None


# ---------------------------------------------------------------------------
# Gemini direct ignores reasoning_state
# ---------------------------------------------------------------------------

class TestGeminiIgnoresReasoningState:
    def test_converter_ignores_the_key(self):
        from skell_e_router.gemini_direct import _convert_messages_to_contents
        _, contents = _convert_messages_to_contents([
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": "Looking.", "tool_calls": [_tool_call()],
             "reasoning_state": [THINKING, OPENAI_ITEM]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ])
        dumped = json.dumps(_dump(contents))
        assert "function_call" in dumped
        for marker in ("reasoning_state", "thinking", "ENC", THINKING["signature"]):
            assert marker not in dumped


# ---------------------------------------------------------------------------
# AIResponse field
# ---------------------------------------------------------------------------

class TestAIResponseField:
    def test_reasoning_state_field_defaults_to_none(self):
        from skell_e_router.response import AIResponse
        field = {f.name: f for f in dataclasses.fields(AIResponse)}["reasoning_state"]
        assert field.default is None
        assert AIResponse("x", "m").reasoning_state is None
