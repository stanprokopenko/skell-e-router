"""Storage defaults on the actual LiteLLM Responses HTTP transport."""

import copy
import inspect
import json

import httpx
import pytest
from litellm.main import responses_api_bridge_check

from skell_e_router import ask_ai
from tests.test_audit_fixes import _DUMMY_KEYS, captured_bodies


summary_routes = pytest.mark.skipif(
    "reasoning_summary" not in inspect.signature(responses_api_bridge_check).parameters,
    reason="This LiteLLM version does not bridge reasoning summary aliases",
)


@summary_routes
@pytest.mark.parametrize("summary_key", ["reasoningSummary", "reasoning_summary"])
@pytest.mark.parametrize(
    "top_level,body_override,expected",
    [(None, None, False), (False, None, False), (True, None, True),
     (False, True, True), (True, False, False)],
)
def test_summary_route_storage(captured_bodies, summary_key, top_level, body_override, expected):
    extra_body = {summary_key: "auto"}
    kwargs = {"reasoning_effort": "low", "extra_body": extra_body}
    if top_level is not None:
        kwargs["store"] = top_level
    if body_override is not None:
        extra_body["store"] = body_override
    before = copy.deepcopy(kwargs)

    assert ask_ai("gpt-5.6-sol", "hi", config=_DUMMY_KEYS, **kwargs) == "OK"

    assert kwargs == before
    assert len(captured_bodies) == 1
    path, body = captured_bodies[0]
    assert path.endswith("/responses")
    assert body["store"] is expected
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert "tools" not in body


@pytest.fixture
def captured_stream_bodies(captured_bodies, monkeypatch):
    """Keep the network guard and capture, replacing the response with valid SSE."""
    capture_send = httpx.Client.send

    def send(client, request, **kwargs):
        response = capture_send(client, request, **kwargs)
        # Older LiteLLM versions synthesize chunks for unregistered models and
        # deliberately send a nonstream request. Answer that request with JSON.
        if not json.loads(request.content).get("stream"):
            return response
        completed = response.json()
        message = completed["output"][0]
        part = message["content"][0]
        initial = {**completed, "status": "in_progress", "output": [], "usage": None}
        events = [
            {"type": "response.created", "response": initial},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**message, "status": "in_progress", "content": []}},
            {"type": "response.content_part.added", "item_id": message["id"],
             "output_index": 0, "content_index": 0, "part": {**part, "text": ""}},
            {"type": "response.output_text.delta", "item_id": message["id"],
             "output_index": 0, "content_index": 0, "delta": "OK"},
            {"type": "response.output_text.done", "item_id": message["id"],
             "output_index": 0, "content_index": 0, "text": "OK"},
            {"type": "response.content_part.done", "item_id": message["id"],
             "output_index": 0, "content_index": 0, "part": part},
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {"type": "response.completed", "response": completed},
        ]
        payload = "".join(
            f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': number})}\n\n"
            for number, event in enumerate(events)
        )
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return captured_bodies


@pytest.mark.parametrize("store", [None, False, True])
@pytest.mark.parametrize("summary_key", [None, "reasoningSummary", "reasoning_summary"])
def test_stream_storage(captured_stream_bodies, summary_key, store):
    if summary_key and "reasoning_summary" not in inspect.signature(responses_api_bridge_check).parameters:
        pytest.skip("This LiteLLM version does not bridge reasoning summary aliases")
    kwargs = {"stream": True}
    alias = "gpt-6.1-sol"
    if summary_key:
        alias = "gpt-5.6-sol"
        kwargs.update(reasoning_effort="low", extra_body={summary_key: "auto"})
    if store is not None:
        kwargs["store"] = store
    before = copy.deepcopy(kwargs)

    assert ask_ai(alias, "hi", config=_DUMMY_KEYS, **kwargs) == "OK"

    assert kwargs == before
    assert len(captured_stream_bodies) == 1
    path, body = captured_stream_bodies[0]
    assert path.endswith("/responses")
    assert body["store"] is (False if store is None else store)
