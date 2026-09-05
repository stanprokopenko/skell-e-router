"""Check output limits after LiteLLM and OpenAI serialize the real request."""

import json
import socket

import httpx
import pytest

from skell_e_router import ask_ai
from skell_e_router.model_config import MODEL_CONFIG
from skell_e_router.utils import RouterError, _handle_model_specific_params


@pytest.fixture
def captured_requests(monkeypatch):
    requests = []
    original_connect = socket.socket.connect

    def deny_network(*args, **kwargs):
        raise AssertionError("Output-limit tests must not use the network")

    def local_connect(sock, address):
        # Windows asyncio creates its internal socket pair over loopback.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return deny_network()

    def send(client, request, **kwargs):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/responses"):
            data = {
                "id": "resp_test", "object": "response", "created_at": 1,
                "status": "completed", "model": body["model"],
                "output": [{"id": "msg_test", "type": "message", "status": "completed",
                            "role": "assistant", "content": [{"type": "output_text", "text": "OK", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        elif body.get("stream"):
            chunks = [
                {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                 "model": body["model"], "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]},
                {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                 "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ]
            content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"}, request=request)
        else:
            data = {
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return httpx.Response(200, json=data, request=request)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(httpx.Client, "send", send)
    return requests


@pytest.mark.parametrize("limit_name", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("stream", [False, True])
def test_luna_limit_reaches_chat_http(captured_requests, limit_name, stream):
    result = ask_ai("gpt-5.6-luna", "hi", config={"openai_api_key": "sk-dummy-no-network"},
                    stream=stream, **{limit_name: 128})
    assert result == "OK"
    assert len(captured_requests) == 1
    path, body = captured_requests[0]
    assert path.endswith("/chat/completions")
    assert body["model"] == "gpt-5.6-luna"
    assert body["max_completion_tokens"] == 128
    assert "max_tokens" not in body
    assert "allowed_openai_params" not in body


@pytest.mark.parametrize("limit_name", ["max_tokens", "max_completion_tokens"])
def test_astra_limit_reaches_responses_http(captured_requests, limit_name):
    assert ask_ai("gpt-6-astra", "hi", config={"openai_api_key": "sk-dummy-no-network"}, **{limit_name: 128}) == "OK"
    assert len(captured_requests) == 1
    path, body = captured_requests[0]
    assert path.endswith("/responses")
    assert body["model"] == "gpt-6-astra"
    assert body["max_output_tokens"] == 128
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_gpt4o_keeps_legacy_limit(captured_requests):
    assert ask_ai("gpt-4o", "hi", config={"openai_api_key": "sk-dummy-no-network"}, max_tokens=128) == "OK"
    assert len(captured_requests) == 1
    path, body = captured_requests[0]
    assert path.endswith("/chat/completions")
    assert body["max_tokens"] == 128
    assert "max_completion_tokens" not in body


REASONING_MODELS = {
    model.name: model for model in MODEL_CONFIG.values()
    if model.provider == "openai" and model.name.startswith(("openai/gpt-5", "openai/gpt-6", "openai/o1", "openai/o3"))
}


@pytest.mark.parametrize("model", REASONING_MODELS.values(), ids=REASONING_MODELS.keys())
@pytest.mark.parametrize("limit_name", ["max_tokens", "max_completion_tokens"])
def test_first_party_reasoning_limits_survive_filtering(model, limit_name):
    result = _handle_model_specific_params(model, {limit_name: 128})
    assert result["max_completion_tokens"] == 128
    assert "max_tokens" not in result
    assert "max_completion_tokens" in result["allowed_openai_params"]


@pytest.mark.parametrize("limit_name", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "128"])
def test_invalid_limits_fail_before_http(captured_requests, limit_name, value):
    with pytest.raises(RouterError, match="INVALID_PARAM"):
        ask_ai("gpt-5.6-luna", "hi", config={"openai_api_key": "sk-dummy-no-network"}, **{limit_name: value})
    assert captured_requests == []


def test_conflicting_limits_fail_before_http(captured_requests):
    with pytest.raises(RouterError, match="INVALID_PARAM"):
        ask_ai("gpt-5.6-luna", "hi", config={"openai_api_key": "sk-dummy-no-network"}, max_tokens=128, max_completion_tokens=256)
    assert captured_requests == []


@pytest.mark.parametrize("params", [
    {"max_tokens": 128, "max_completion_tokens": 128},
    {"max_tokens": None, "max_completion_tokens": 128},
    {"max_tokens": 128, "max_completion_tokens": None},
])
def test_equal_or_omitted_aliases_preserve_limit(params):
    result = _handle_model_specific_params(MODEL_CONFIG["gpt-5.6-luna"], params)
    assert result["max_completion_tokens"] == 128
    assert "max_tokens" not in result


@pytest.mark.parametrize("extra_key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_extra_body_cannot_override_limit(captured_requests, extra_key):
    with pytest.raises(RouterError, match="INVALID_PARAM"):
        ask_ai("gpt-5.6-luna", "hi", config={"openai_api_key": "sk-dummy-no-network"},
               max_tokens=128, extra_body={extra_key: 256})
    assert captured_requests == []


@pytest.mark.parametrize("params", [{}, {"max_tokens": None}, {"max_completion_tokens": None},
                                   {"max_tokens": None, "max_completion_tokens": None}])
def test_omitted_limit_adds_no_http_cap(captured_requests, params):
    assert ask_ai("gpt-5.6-luna", "hi", config={"openai_api_key": "sk-dummy-no-network"}, **params) == "OK"
    assert len(captured_requests) == 1
    body = captured_requests[0][1]
    assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"}.intersection(body)


def test_unrelated_overrides_survive_normalization():
    result = _handle_model_specific_params(MODEL_CONFIG["gpt-5.6-luna"], {
        "max_tokens": 128, "extra_body": {"metadata": {"probe": "test"}},
        "allowed_openai_params": ["metadata"],
    })
    assert result["max_completion_tokens"] == 128
    assert result["extra_body"] == {"metadata": {"probe": "test"}}
    assert {"metadata", "max_completion_tokens"}.issubset(result["allowed_openai_params"])
