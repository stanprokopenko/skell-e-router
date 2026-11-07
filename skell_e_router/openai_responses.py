"""OpenAI Responses transport retaining reasoning before chat normalization."""

from copy import deepcopy
from types import SimpleNamespace

import litellm
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.responses.utils import ResponseAPILoggingUtils


def complete_openai_responses(model, messages, **kwargs):
    """Return a normal ModelResponse with an explicit, lossless reasoning list.

    LiteLLM's chat bridge has a single pending reasoning slot. Read raw output
    here before using that bridge's conversion for the ordinary response fields.
    """
    handler = LiteLLMResponsesTransformationHandler()
    options = dict(kwargs)
    transport = {key: options.pop(key) for key in (
        "api_key", "api_base", "timeout", "max_retries", "client", "extra_headers",
    ) if key in options}
    options = deepcopy(options)
    summary = None
    for source in (options, options.get("extra_body") or {}):
        for key in ("reasoningSummary", "reasoning_summary"):
            value = source.pop(key, None)
            if summary is None:
                summary = value
    request = handler.transform_request(
        model=model.replace("/responses/", "/", 1), messages=messages,
        optional_params=options, litellm_params={"custom_llm_provider": "openai"},
        headers={}, litellm_logging_obj=SimpleNamespace(),
    )
    # The real Responses call creates its own logger and receives the caller's
    # transport objects without copying them.
    request.pop("litellm_logging_obj", None)
    request.update(transport)
    if summary is not None:
        request["reasoning"] = {**(request.get("reasoning") or {}), "summary": summary}
    # Older LiteLLM lacks the system-only and JSON-mode input handling.
    instructions = request.get("instructions")
    json_mode = (request.get("text") or {}).get("format", {}).get("type") == "json_object"
    if instructions and (not request["input"] or json_mode):
        request["input"] = [{"type": "message", "role": "system", "content": [
            {"type": "input_text", "text": request.pop("instructions")},
        ]}, *request["input"]]
    raw = litellm.responses(**request)
    if kwargs.get("stream"):
        terminal = None
        for event in raw:
            if getattr(event, "type", None) in (
                "response.completed", "response.incomplete", "response.failed",
            ):
                terminal = event.response
        if terminal is None:
            raise ValueError("OpenAI Responses stream ended without a terminal response")
        raw = terminal
    if raw.error is not None:
        raise ValueError(f"Error in response: {raw.error}")
    choices = handler._convert_response_output_to_choices(
        raw.output, handle_raw_dict_callback=handler._handle_raw_dict_response_item,
    )
    reason = getattr(getattr(raw, "incomplete_details", None), "reason", None)
    if not choices:
        choices = [litellm.Choices(message=litellm.Message(role="assistant", content=""))]
    refusals = [
        getattr(part, "refusal", "")
        for item in raw.output for part in (getattr(item, "content", None) or [])
        if getattr(part, "type", None) == "refusal" and getattr(part, "refusal", None)
    ]
    if refusals:
        choices[0].message.refusal = "\n".join(refusals)
    if reason in ("max_output_tokens", "content_filter"):
        for choice in choices:
            choice.finish_reason = "length" if reason == "max_output_tokens" else "content_filter"
    response = litellm.ModelResponse(
        id=raw.id, model=raw.model, choices=choices,
        usage=ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(raw.usage),
    )
    response._hidden_params.update(getattr(raw, "_hidden_params", {}) or {})
    # This explicit field belongs to the adapter, not provider telemetry. Only
    # this OpenAI transport may populate it, including for collected streams.
    response.router_openai_reasoning_items = [
        item.model_dump() if hasattr(item, "model_dump") else dict(item)
        for item in raw.output
        if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "reasoning"
    ]
    return response
