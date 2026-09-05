"""Provider diagnostics must never escape through public errors or router output.

Run with scripts/run_security_tests_offline.py. All credentials are synthetic;
provider transports are mocked, and h11 only validates an in-memory request.
"""

import json
import logging
import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import quote

import h11
import httpx
import pytest

import skell_e_router as router
from skell_e_router import anthropic_direct, gemini_deep_research as research
from skell_e_router import gemini_direct, utils
from skell_e_router import embeddings
from tests.helpers import make_interaction, make_model


SECRET = "synthetic-credential-92/part\x00tail+="
DIAGNOSTIC = "private-provider-diagnostic-486"
CATEGORIES = {
    "authentication", "permission", "rate_limit", "timeout", "connection",
    "invalid_request", "unavailable", "provider_error", "dependency",
}


def diagnostic():
    return " ".join((DIAGNOSTIC, SECRET, repr(SECRET), json.dumps(SECRET), quote(SECRET)))


def provider_failure(error_type=ValueError):
    if error_type in (router.RouterError, router.DeepResearchError):
        error = error_type("UNTRUSTED_PROVIDER_CODE", diagnostic(), {"body": diagnostic()})
    else:
        error = error_type(diagnostic())
    error.status_code = 401
    error.__cause__ = RuntimeError(diagnostic())
    error.__context__ = LookupError(diagnostic())
    error.add_note(diagnostic())
    error.response = SimpleNamespace(text=diagnostic(), headers={"Authorization": SECRET})
    return error


def fail_later(error):
    yield SimpleNamespace(text="safe initial chunk")
    raise error


def assert_no_diagnostics(text):
    for forbidden in (DIAGNOSTIC, SECRET, repr(SECRET)[1:-1], json.dumps(SECRET)[1:-1], quote(SECRET)):
        assert forbidden not in text


def capture_safe_failure(call, capsys, caplog, error_type=router.RouterError,
                         code="PROVIDER_ERROR", status=401):
    with pytest.raises(error_type) as caught:
        try:
            call()
        except Exception:
            logging.getLogger("credential-regression").exception("Public call failed")
            raise
    error = caught.value
    assert error.code == code
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.details["category"] in CATEGORIES
    if status is not None:
        assert error.details["status_code"] == status
        assert type(error.details["status_code"]) is int
    elif "status_code" in error.details:
        assert type(error.details["status_code"]) is int
        assert 100 <= error.details["status_code"] <= 599
    output = capsys.readouterr()
    assert_no_diagnostics("\n".join((
        str(error), repr(error), repr(vars(error)), repr(error.args),
        "".join(traceback.format_exception(error)), output.out, output.err, caplog.text,
    )))
    return error


@pytest.fixture
def credentials(monkeypatch):
    for provider in ("openai", "gemini", "anthropic"):
        monkeypatch.setenv(f"{provider.upper()}_API_KEY", SECRET)
    return {f"{provider}_api_key": SECRET for provider in ("openai", "gemini", "anthropic")}


@pytest.mark.parametrize("verbosity", ["none", "response", "info", "debug"])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_chat_diagnostics_and_exception_graph(monkeypatch, credentials, capsys, caplog,
                                            verbosity, explicit_config):
    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model())
    monkeypatch.setattr(utils.litellm, "completion", MagicMock(side_effect=provider_failure()))
    capture_safe_failure(lambda: router.ask_ai(
        "test", "hello", direct_sdk=False, verbosity=verbosity,
        config=credentials if explicit_config else None,
    ), capsys, caplog)


@pytest.mark.parametrize("operation", ["chat", "embedding", "upload"])
def test_real_h11_malformed_authorization(monkeypatch, credentials, capsys, caplog, operation):
    def malformed_header(**kwargs):
        # h11 formats the invalid bytes with repr, so literal key replacement fails.
        h11.Request(method=b"GET", target=b"/", headers=[
            (b"Host", b"offline.invalid"), (b"Authorization", b"Bearer " + SECRET.encode()),
        ])

    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model())
    monkeypatch.setattr(utils.litellm, "completion", malformed_header)
    monkeypatch.setattr(embeddings.litellm, "embedding", malformed_header)
    monkeypatch.setattr(utils.litellm, "create_file", malformed_header)
    calls = {
        "chat": lambda: router.ask_ai("test", "hello", verbosity="debug", direct_sdk=False),
        "embedding": lambda: router.get_embedding("openai-embedding-3-large", "hello"),
        "upload": lambda: router.upload_file(b"safe data", "text/plain"),
    }
    capture_safe_failure(calls[operation], capsys, caplog,
                         code="UPLOAD_ERROR" if operation == "upload" else "PROVIDER_ERROR",
                         status=None)


def test_litellm_openai_transport_logging(monkeypatch, credentials, capsys, caplog):
    transported = []

    def reject_local_request(self, request):
        transported.append(request)
        # Exercise SDK request construction and error handling without connecting.
        h11.Request(method=request.method, target=b"/", headers=request.headers.raw)
        pytest.fail("Synthetic NUL credential did not reach Authorization")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_local_request)
    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model(name="openai/gpt-4o"))
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setattr(utils, "_retry_after_wait", lambda _: 0)
    capture_safe_failure(lambda: router.ask_ai(
        "test", "hello", direct_sdk=False, config=credentials, verbosity="debug",
    ), capsys, caplog, status=None)
    assert transported, "The test must exercise the real SDK transport boundary"


@pytest.mark.parametrize("operation", ["embedding", "upload"])
def test_other_public_provider_errors(monkeypatch, credentials, capsys, caplog, operation):
    monkeypatch.setattr(embeddings.litellm, "embedding", MagicMock(side_effect=provider_failure()))
    monkeypatch.setattr(utils.litellm, "create_file", MagicMock(side_effect=provider_failure()))
    call = (lambda: router.get_embedding("openai-embedding-3-large", "hello")) if operation == "embedding" else (
        lambda: router.upload_file(b"safe data", "text/plain"))
    capture_safe_failure(call, capsys, caplog,
                         code="UPLOAD_ERROR" if operation == "upload" else "PROVIDER_ERROR")


@pytest.mark.parametrize("provider", ["gemini", "anthropic"])
@pytest.mark.parametrize("stage", ["client", "request", "router_error", "stream_create", "stream_iterate"])
def test_direct_sdk_boundaries(monkeypatch, credentials, capsys, caplog, provider, stage):
    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model(provider))
    module = gemini_direct if provider == "gemini" else anthropic_direct
    client = MagicMock()
    error = provider_failure(router.RouterError if stage == "router_error" else ValueError)
    factory = MagicMock(return_value=client)
    if stage == "client":
        factory.side_effect = error
    monkeypatch.setattr(module, f"_get_{provider}_client", factory)
    if provider == "gemini":
        request, streaming = client.models.generate_content, client.models.generate_content_stream
        streaming.return_value = fail_later(error)
    else:
        request, streaming = client.messages.create, client.messages.stream
        manager = streaming.return_value
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.text_stream = fail_later(error)
    if stage == "stream_create":
        streaming.side_effect = error
    else:
        request.side_effect = error

    def call():
        result = router.ask_ai("test", "hello", direct_sdk=True, verbosity="debug",
                               stream=stage.startswith("stream"))
        if stage == "stream_iterate":
            if provider == "gemini":
                list(result)
            else:
                with result as stream:
                    list(stream.text_stream)

    capture_safe_failure(call, capsys, caplog)


@pytest.mark.parametrize("stage", ["enter", "exit", "final_message"])
def test_anthropic_stream_context_boundaries(monkeypatch, credentials, capsys, caplog, stage):
    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model("anthropic"))
    client = MagicMock()
    monkeypatch.setattr(anthropic_direct, "_get_anthropic_client", lambda _: client)
    manager = client.messages.stream.return_value
    manager.__enter__.return_value = manager
    manager.__exit__.return_value = False
    manager.text_stream = iter(["safe"])
    target = {"enter": manager.__enter__, "exit": manager.__exit__,
              "final_message": manager.get_final_message}[stage]
    target.side_effect = provider_failure()

    def call():
        with router.ask_ai("test", "hello", direct_sdk=True, stream=True) as stream:
            list(stream.text_stream)
            if stage == "final_message":
                stream.get_final_message()

    capture_safe_failure(call, capsys, caplog)


@pytest.mark.parametrize("api", ["ask", "follow_up", "status", "generator"])
@pytest.mark.parametrize("stage", ["client", "request"])
def test_deep_research_public_errors(monkeypatch, credentials, capsys, caplog, api, stage):
    client = MagicMock()
    factory = MagicMock(return_value=client)
    error = provider_failure()
    if stage == "client":
        factory.side_effect = error
    else:
        client.interactions.create.side_effect = error
        client.interactions.get.side_effect = error
    monkeypatch.setattr(research.genai, "Client", factory)
    calls = {
        "ask": lambda: router.ask_deep_research("hello", verbosity="debug"),
        "follow_up": lambda: router.deep_research_follow_up("safe-id", "hello", verbosity="debug"),
        "status": lambda: router.get_research_status("safe-id"),
        "generator": lambda: list(router.stream_deep_research("hello")),
    }
    code = "PROVIDER_ERROR"
    if stage == "request":
        code = {"follow_up": "FOLLOW_UP_ERROR", "generator": "STREAM_FAILED"}.get(api, code)
    capture_safe_failure(calls[api], capsys, caplog, router.DeepResearchError,
                         code=code, status=None if code == "STREAM_FAILED" else 401)


@pytest.mark.parametrize("stream", [False, True])
def test_deep_research_provider_error_payload(monkeypatch, credentials, capsys, caplog, stream):
    client = MagicMock()
    monkeypatch.setattr(research, "_get_client", lambda _: client)
    client.interactions.create.return_value = (
        iter([SimpleNamespace(event_type="error", error=diagnostic())]) if stream else
        SimpleNamespace(id="safe-id"))
    client.interactions.get.return_value = SimpleNamespace(
        id="safe-id", status="failed", error=diagnostic())
    capture_safe_failure(lambda: router.ask_deep_research(
        "hello", stream=stream, verbosity="debug", resolve_citations=False,
    ), capsys, caplog, router.DeepResearchError,
        code="STREAM_ERROR" if stream else "RESEARCH_FAILED", status=None)


def test_deep_research_lazy_failure_logs(monkeypatch, credentials, capsys, caplog):
    client = MagicMock()
    monkeypatch.setattr(research, "_get_client", lambda _: client)
    monkeypatch.setattr(research.time, "sleep", lambda _: None)
    error = provider_failure(router.DeepResearchError)

    def broken_stream():
        if False:
            yield
        raise error

    client.interactions.create.side_effect = lambda **_: broken_stream()
    capture_safe_failure(lambda: router.ask_deep_research(
        "hello", stream=True, verbosity="debug", resolve_citations=False,
    ), capsys, caplog, router.DeepResearchError)


def test_deep_research_reconnect_logs_and_failed_result(monkeypatch, credentials, capsys, caplog):
    client = MagicMock()
    monkeypatch.setattr(research, "_get_client", lambda _: client)
    monkeypatch.setattr(research.time, "sleep", lambda _: None)
    failed = make_interaction(status="failed", error=diagnostic())
    # A provider object can include error details in its debug representation.
    failed.__str__.return_value = diagnostic()
    failed.__repr__ = lambda _: diagnostic()
    client.interactions.create.return_value = iter([
        SimpleNamespace(event_type="interaction.start", interaction=SimpleNamespace(id="safe-id")),
        SimpleNamespace(event_type="error", error="503 service unavailable " + diagnostic()),
    ])
    client.interactions.get.return_value = iter([
        SimpleNamespace(event_type="interaction.complete", interaction=failed),
    ])
    result = router.ask_deep_research("hello", stream=True, verbosity="debug", resolve_citations=False)
    assert result.status == "failed"
    assert result.error
    assert_no_diagnostics(result.error)
    output = capsys.readouterr()
    assert_no_diagnostics(output.out + output.err + caplog.text)
    assert client.interactions.create.call_count == 1
    assert client.interactions.get.call_count == 1


@pytest.mark.parametrize("bad_status", [True, "401", SECRET, -1, 700])
def test_status_metadata_is_a_safe_http_integer(monkeypatch, credentials, capsys, caplog, bad_status):
    error = provider_failure()
    error.status_code = bad_status
    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model())
    monkeypatch.setattr(utils.litellm, "completion", MagicMock(side_effect=error))
    capture_safe_failure(lambda: router.ask_ai("test", "hello", direct_sdk=False),
                         capsys, caplog, status=None)


def test_input_validation_still_reports_local_error(monkeypatch):
    for key in utils.PROVIDER_ENV_KEY.values():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(utils, "resolve_model_alias", lambda _: make_model())
    provider = MagicMock()
    monkeypatch.setattr(utils.litellm, "completion", provider)
    with pytest.raises(router.RouterError) as caught:
        router.ask_ai("test", "hello", direct_sdk=False)
    assert caught.value.code == "MISSING_ENV"
    provider.assert_not_called()
