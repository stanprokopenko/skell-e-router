"""Offline regression coverage for provider audit review findings."""

import importlib

import litellm
import pytest

from skell_e_router import utils


TOOL = {"type": "function", "function": {"name": "book_ticket", "parameters": {"type": "object"}}}


def assert_complete_response_text(text):
    # Older LiteLLM emits separate choices; newer versions concatenate them
    # before returning. Both must preserve the complete text in output order.
    assert text in (
        "Checking availability.\nI found one flight.",
        "Checking availability.I found one flight.",
    )


@pytest.fixture(autouse=True)
def no_cost_lookup(monkeypatch):
    monkeypatch.setattr(utils, "_compute_response_cost", lambda *args: 0.0)


def responses_reply():
    """Use the installed adapter, including its real lack of an origin marker."""
    from litellm.completion_extras.litellm_responses_transformation.transformation import LiteLLMResponsesTransformationHandler
    from litellm.types.llms.openai import ResponsesAPIResponse

    raw = ResponsesAPIResponse(
        id="resp_test", created_at=0, model="gpt-5.6",
        output=[
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Checking availability.", "annotations": []}]},
            {"id": "msg_2", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "I found one flight.", "annotations": []}]},
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "book_ticket", "arguments": "{}"},
        ],
        usage={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )
    return LiteLLMResponsesTransformationHandler().transform_response(
        model="gpt-5.6", raw_response=raw, model_response=litellm.ModelResponse(),
        logging_obj=None, request_data={}, messages=[], optional_params={}, litellm_params={}, encoding=None,
    )


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol"])
def test_public_responses_routes_merge_actual_output_parts(monkeypatch, model):
    reply = responses_reply()
    # LiteLLM versions differ in whether their chat call id uses call_id or
    # the output item's id. The router must preserve the adapter's call.
    expected_calls = [call.model_dump() for choice in reply.choices
                      for call in choice.message.tool_calls or []]
    monkeypatch.setattr(utils.litellm, "completion", lambda **kwargs: reply)
    out = utils.ask_ai(model, "Book one ticket", rich_response=True,
                       tools=[TOOL], reasoning_effort="low", config={"openai_api_key": "fake"})
    assert_complete_response_text(out.content)
    assert len(expected_calls) == 1
    assert [call.model_dump() for call in out.tool_calls] == expected_calls
    assert out.tool_calls[0].function.name == "book_ticket"
    assert out.finish_reason == "tool_calls"
    assert reply._hidden_params["router_responses_output_parts"] is True


def test_plain_text_responses_return_all_text(monkeypatch):
    monkeypatch.setattr(utils.litellm, "completion", lambda **kwargs: responses_reply())
    out = utils.ask_ai("gpt-6-astra", "Check flights", config={"openai_api_key": "fake"})
    assert_complete_response_text(out)


def test_public_gemini_candidates_select_one_alternative(monkeypatch):
    reply = litellm.ModelResponse(choices=[
        {"index": i, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": f"Option {i}", "tool_calls": [
                {"id": f"call_{i}", "type": "function", "function": {"name": "book_ticket", "arguments": "{}"}}
            ]}}
        for i in range(2)
    ])
    sent = {}

    def completion(**kwargs):
        sent.update(kwargs)
        return reply

    monkeypatch.setattr(utils.litellm, "completion", completion)
    out = utils.ask_ai("gemini-2.5-flash", "Book one ticket", direct_sdk=False,
                       candidate_count=2, tools=[TOOL], rich_response=True,
                       config={"gemini_api_key": "fake"})
    assert sent["candidate_count"] == 2
    assert out.content == "Option 0"
    assert [call.id for call in out.tool_calls] == ["call_0"]


def test_route_detection_without_optional_litellm_helper(monkeypatch):
    main = importlib.import_module("litellm.main")
    monkeypatch.delattr(main, "responses_api_bridge_check")
    assert utils._uses_responses_bridge("openai/responses/gpt-6-astra", {}) is True
    assert utils._uses_responses_bridge("gemini/gemini-2.5-flash", {}) is False


@pytest.mark.parametrize("details", [True, False])
def test_litellm_anthropic_reports_cache_writes(details):
    usage = {"prompt_tokens": 1100, "completion_tokens": 10, "total_tokens": 1110,
             "cache_creation_input_tokens": 300, "cache_read_input_tokens": 700,
             "prompt_tokens_details": {"cached_tokens": 700}}
    if details:
        usage["prompt_tokens_details"]["cache_creation_tokens"] = 300
    out = utils._build_ai_response(litellm.ModelResponse(usage=usage), 0.1)
    assert out.cache_read_tokens == 700
    assert out.cache_write_tokens == 300


@pytest.mark.parametrize("native_refusal,provider_fields,expected", [
    ("I cannot help with that.", None, {"refusal": "I cannot help with that."}),
    ("I cannot help with that.", {"other": "kept"},
     {"other": "kept", "refusal": "I cannot help with that."}),
    (None, {"refusal": "Legacy refusal", "other": "kept"},
     {"refusal": "Legacy refusal", "other": "kept"}),
    ("", {"refusal": "Legacy refusal"}, {"refusal": "Legacy refusal"}),
])
def test_litellm_refusal_metadata_is_preserved_without_mutation(native_refusal, provider_fields, expected):
    reply = litellm.ModelResponse()
    message = reply.choices[0].message
    message.refusal = native_refusal
    message.provider_specific_fields = provider_fields
    original = reply.model_dump()

    out = utils._build_ai_response(reply, 0.1)

    assert out.provider_specific_fields == expected
    assert reply.model_dump() == original
    if native_refusal:
        assert out.provider_specific_fields is not message.provider_specific_fields
