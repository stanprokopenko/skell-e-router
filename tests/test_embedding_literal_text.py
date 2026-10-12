"""Exercise literal embedding text after the real LiteLLM HTTP serialization."""

import base64
import builtins
import json
import os
import socket
from types import SimpleNamespace

import httpx
import pytest

from skell_e_router import get_embedding
from skell_e_router import embeddings as embedding_module
from skell_e_router.response import GeminiFileRef
from skell_e_router.utils import RouterError


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)
MODELS = ["gemini-embedding-2", "openai-embedding-3-small"]
CONFIG = {"gemini_api_key": "gemini-config-test", "openai_api_key": "openai-config-test"}


@pytest.fixture
def wire(monkeypatch):
    state = {"requests": [], "reverse": False}
    original_connect = socket.socket.connect

    def deny_network(*args, **kwargs):
        raise AssertionError("Embedding tests must not use the network")

    def connect(sock, address):
        # Windows asyncio creates its internal socket pair on loopback.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return deny_network()

    def send(client, request, **kwargs):
        body = json.loads(request.content)
        state["requests"].append((request, body))
        if request.url.path in {"/v1/embeddings", "/v1beta/openai/embeddings"}:
            assert request.url.host in {"api.openai.com", "generativelanguage.googleapis.com"}
            assert isinstance(body["input"], list)
            data = [{"object": "embedding", "index": i, "embedding": [float(i), 0.5]}
                    for i in range(len(body["input"]))]
            if state.get("first_index") == "omitted":
                del data[0]["index"]
            elif state.get("first_index") == "null":
                data[0]["index"] = None
            if state["reverse"]:
                data.reverse()
            response = {"object": "list", "model": body["model"], "data": data,
                        "usage": {"prompt_tokens": len(data), "total_tokens": len(data)}}
        elif request.url.path == "/v1beta/models/gemini-embedding-2:embedContent":
            assert request.url.host == "generativelanguage.googleapis.com"
            response = {"embedding": {"values": [0.0, 0.5]}}
        else:
            raise AssertionError(f"Unexpected HTTP request: {request.method} {request.url.host}{request.url.path}")
        return httpx.Response(200, json=response, request=request)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-test")
    return state


@pytest.fixture
def literal_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models.json").write_text('{"private": "must never be embedded"}', encoding="utf-8")
    (tmp_path / "local.png").write_bytes(PNG)
    return [
        "models.json", "local.png", str(tmp_path / "models.json"), str(tmp_path / "local.png"),
        str(tmp_path / "missing.png"), "missing.pdf", r"\\server\share\private.png",
        "http://example.invalid/image.png", "https://example.invalid/audio.wav",
        "gs://private-bucket/video.mp4", "files/private-file",
        "data:image/png;base64," + base64.b64encode(PNG).decode(),
        "data:audio/wav;base64,AAAA", "data:video/mp4;base64,AAAA",
        "data:application/pdf;base64,AAAA", "image/png", "audio/wav", "video/mp4",
        "application/pdf", "  white space\n\t雪 café 🐈  ", "", "   ",
    ]


def forbid_input_file_access(monkeypatch, inputs):
    guarded = set(inputs)

    def wrap(original):
        def guarded_call(path, *args, **kwargs):
            value = os.fspath(path) if isinstance(path, (str, bytes, os.PathLike)) else path
            if isinstance(value, str) and value in guarded:
                raise AssertionError(f"Literal input reached filesystem: {value!r}")
            return original(path, *args, **kwargs)
        return guarded_call

    monkeypatch.setattr(os.path, "isfile", wrap(os.path.isfile))
    monkeypatch.setattr(os, "stat", wrap(os.stat))
    monkeypatch.setattr(builtins, "open", wrap(builtins.open))

    def no_conversion(*args, **kwargs):
        raise AssertionError("Literal text reached media conversion")

    monkeypatch.setattr(embedding_module, "_convert_part", no_conversion)


@pytest.mark.parametrize("model", MODELS)
def test_exact_batch_text_survives_real_http(model, literal_inputs, monkeypatch, wire):
    forbid_input_file_access(monkeypatch, literal_inputs)
    wire["reverse"] = True
    result = get_embedding(model, literal_inputs, literal_text=True, dimensions=2, config=CONFIG)
    assert result == [[float(i), 0.5] for i in range(len(literal_inputs))]
    assert len(wire["requests"]) == 1
    request, body = wire["requests"][0]
    assert body["input"] == literal_inputs
    assert body["dimensions"] == 2
    assert "literal_text" not in body
    assert "allowed_openai_params" not in body
    if model.startswith("gemini"):
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"
        assert body["model"] == "gemini-embedding-2"
        assert request.headers["authorization"] == "Bearer gemini-config-test"
        assert "openai-config-test" not in str(request.headers)
        assert "openai-env-test" not in str(request.headers)
    else:
        assert str(request.url) == "https://api.openai.com/v1/embeddings"
        assert body["model"] == "text-embedding-3-small"
        assert request.headers["authorization"] == "Bearer openai-config-test"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("as_list", [False, True])
def test_scalar_and_single_item_batch_shapes(model, as_list, monkeypatch, wire):
    value = "data:image/png;base64,not-an-image"
    forbid_input_file_access(monkeypatch, [value])
    result = get_embedding(model, [value] if as_list else value, literal_text=True, config=CONFIG)
    assert result == ([[0.0, 0.5]] if as_list else [0.0, 0.5])
    assert len(wire["requests"]) == 1
    assert wire["requests"][0][1]["input"] == [value]


@pytest.mark.parametrize("first_index", ["omitted", "null"])
def test_gemini_batch_accepts_missing_first_response_index(first_index, wire):
    # Google's compatibility endpoint can omit zero or return null for index zero.
    wire["first_index"] = first_index
    values = ["first", "second", "third"]
    result = get_embedding("gemini-embedding-2", values, literal_text=True, config=CONFIG)
    assert result == [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]]
    assert len(wire["requests"]) == 1
    assert wire["requests"][0][1]["input"] == values


@pytest.mark.parametrize("first_entry", [
    {"embedding": [0.0, 0.5]},
    {"index": None, "embedding": [0.0, 0.5]},
])
def test_dictionary_response_missing_index_preserves_position(first_entry, wire):
    response = SimpleNamespace(
        model="gemini-embedding-2", usage=None,
        data=[first_entry, {"index": 2, "embedding": [2.0, 0.5]},
              {"index": 1, "embedding": [1.0, 0.5]}],
    )
    result = embedding_module._build_embedding_response(
        response, embedding_module.resolve_embedding_alias("gemini-embedding-2"), None, None,
    )
    assert result.embeddings == [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]]
    assert wire["requests"] == []


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("value", [[], [["text"]], ["text", ["nested"]], 1, None,
                                      ["text", 1], {"text": "value"}, ("text",),
                                      GeminiFileRef(uri="files/private", mime_type="image/png"),
                                      [GeminiFileRef(uri="files/private", mime_type="image/png")]])
def test_invalid_literal_shapes_never_send(model, value, wire):
    with pytest.raises(RouterError, match="INVALID_INPUT"):
        get_embedding(model, value, literal_text=True, config=CONFIG)
    assert wire["requests"] == []


def test_gemini_environment_key_is_explicit_for_openai_transport(wire):
    assert get_embedding("gemini-embedding-2", "text", literal_text=True) == [0.0, 0.5]
    request, _ = wire["requests"][0]
    assert request.headers["authorization"] == "Bearer gemini-env-test"
    assert "openai-env-test" not in str(request.headers)


@pytest.mark.parametrize("gemini_config", [{}, {"gemini_api_key": ""}])
def test_missing_gemini_key_never_uses_openai_key(gemini_config, monkeypatch, wire):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RouterError, match="MISSING_ENV"):
        get_embedding("gemini-embedding-2", "text", literal_text=True,
                      config={"openai_api_key": "openai-config-test", **gemini_config})
    assert wire["requests"] == []


def test_explicit_local_image_still_uses_native_inline_data(tmp_path, wire):
    local_image = tmp_path / "actual.png"
    local_image.write_bytes(PNG)
    assert get_embedding("gemini-embedding-2", [["caption", str(local_image)]], config=CONFIG) == [[0.0, 0.5]]
    assert len(wire["requests"]) == 1
    request, body = wire["requests"][0]
    assert request.url.path == "/v1beta/models/gemini-embedding-2:embedContent"
    parts = body["content"]["parts"]
    assert {"text": "caption"} in parts
    inline = next(part["inline_data"] for part in parts if "inline_data" in part)
    assert inline["mime_type"] == "image/png"
    assert base64.b64decode(inline["data"]) == PNG
