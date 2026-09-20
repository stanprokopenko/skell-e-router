"""Offline contract checks for TypeSafe classification, with no paid requests."""

from copy import deepcopy
import traceback
from unittest.mock import MagicMock

import pytest
import requests

from skell_e_router import classify, ClassificationResponse, RouterError, resolve_classification_alias
from skell_e_router import classification as implementation
from skell_e_router.model_config import MODEL_CONFIG


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Choose queue", "criteria": {"billing": "Refunds", "tech": None}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "severity": {"type": "score", "instructions": "Rate severity", "criteria": ["Low", "High"]},
}
ANSWER = {
    "model": "jev-1.13.0",
    "answers": {
        "route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.6},
        "urgent": {"type": "noul", "noul": 0.7},
        "severity": {"type": "score", "score": 0.8, "probabilities": {"0": 0.2, "1": 0.8}, "confidence": 0.5,
                     "legend": {"0": "Low", "1": "High"}},
    },
    "usage": {"input_tokens": 1000, "output_tokens": 40},
}


def http_response(data=None, status=200, headers=None):
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.json.return_value = deepcopy(ANSWER if data is None else data)
    response.__enter__.return_value = response
    return response


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    post = MagicMock(return_value=http_response())
    monkeypatch.setattr(implementation.requests, "post", post)
    monkeypatch.setattr(implementation._request.retry, "sleep", lambda _: None)
    return post


def test_contract_and_native_probabilities(transport):
    result = classify("jev", {"message": "Refund me", "amount": 15}, QUESTIONS,
                      config={"typesafe_api_key": "explicit-key"}, timeout=2.5)
    assert isinstance(result, ClassificationResponse)
    assert result.answers == ANSWER["answers"]
    assert result.model == "jev-1.13.0"
    assert (result.input_tokens, result.output_tokens) == (1000, 40)
    assert result.cost == pytest.approx(0.000042)
    assert result.duration_seconds >= 0
    transport.assert_called_once_with(
        "https://api.typesafe.ai/v1/systemone",
        json={"model": "jev-1.13.0", "state": {"message": "Refund me", "amount": 15}, "questions": QUESTIONS},
        headers={"Authorization": "Bearer explicit-key"}, timeout=2.5, allow_redirects=False)


@pytest.mark.parametrize("alias,expected", [("jev", "jev-1.13.0"), ("jev-latest", "jev-latest"), ("jev-1.13.0", "jev-1.13.0")])
def test_aliases_and_chat_rejection(alias, expected):
    from skell_e_router import ask_ai
    model = resolve_classification_alias(alias)
    assert model.name == expected
    assert model.max_input_tokens == 64000 and model.max_state_question_tokens == 32000
    assert alias not in MODEL_CONFIG
    with pytest.raises(RouterError, match="requires classify"):
        ask_ai(alias, "Hello")


@pytest.mark.parametrize("alias", ["unknown", [], None])
def test_invalid_model(alias, transport):
    with pytest.raises(RouterError) as caught:
        classify(alias, "text", QUESTIONS)
    assert caught.value.code == "INVALID_MODEL"
    transport.assert_not_called()


@pytest.mark.parametrize("usage,expected", [(None, (None, None, None)), ({}, (None, None, None)),
    ({"input_tokens": 0}, (0, None, 0.0)), ({"output_tokens": 9}, (None, 9, None))])
def test_unknown_usage_is_not_zero(usage, expected, transport):
    data = deepcopy(ANSWER)
    data["usage"] = usage
    transport.return_value = http_response(data)
    result = classify("jev", "text", QUESTIONS)
    assert (result.input_tokens, result.output_tokens, result.cost) == expected


@pytest.mark.parametrize("usage,expected", [
    ({"input_tokens": 1000, "output_tokens": 40}, 0.000042 + 0.00002),
    ({"input_tokens": 1000}, None)])
def test_output_rate_is_billed_when_a_model_charges_for_it(usage, expected, transport, monkeypatch):
    monkeypatch.setattr(resolve_classification_alias("jev"), "output_cost_per_million", 0.5)
    data = deepcopy(ANSWER)
    data["usage"] = usage
    transport.return_value = http_response(data)
    result = classify("jev", "text", QUESTIONS)
    assert result.cost == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("timeout", [False, 0, -1, "3", float("nan"), float("inf")])
def test_bad_timeout(timeout, transport):
    with pytest.raises(RouterError) as caught:
        classify("jev", "text", QUESTIONS, timeout=timeout)
    assert caught.value.code == "INVALID_PARAM"
    transport.assert_not_called()


@pytest.mark.parametrize("state,questions", [(None, QUESTIONS), (b"bytes", QUESTIONS),
    ({"x": float("nan")}, QUESTIONS), ("text", {}), ("text", {"q": {"type": "chat", "instructions": "Hi"}}),
    ("text", {"q": {"type": "score", "instructions": "Rate", "criteria": ["Only"]}}),
    ("text", {"q": {"type": "noul", "instructions": "Check", "reasoning_effort": "high"}}),
    ("text", {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": 5, "b": None}}}),
    ("text", {"q": {"type": "noul", "instructions": "Check", "criteria": {"maybe": "Perhaps"}}})])
def test_invalid_input(state, questions, transport):
    with pytest.raises(RouterError) as caught:
        classify("jev", state, questions)
    assert caught.value.code == "INVALID_INPUT"
    transport.assert_not_called()


def test_cyclic_state_is_safe(transport):
    state = {}
    state["secret"] = state
    with pytest.raises(RouterError, match="INVALID_INPUT"):
        classify("jev", state, QUESTIONS)
    transport.assert_not_called()


@pytest.mark.parametrize("count", [2, 10, 11])
def test_score_level_boundaries(count, transport):
    criteria = [f"Level {i}" for i in range(count)]
    questions = {"q": {"type": "score", "instructions": "Rate", "criteria": criteria}}
    if count > 10:
        with pytest.raises(RouterError, match="INVALID_INPUT"):
            classify("jev", "text", questions)
        transport.assert_not_called()
    else:
        data = {
            "model": "jev-1.13.0",
            "answers": {"q": {"type": "score", "score": 0, "confidence": 1,
                              "probabilities": {str(i): int(i == 0) for i in range(count)},
                              "legend": {str(i): value for i, value in enumerate(criteria)}}},
        }
        transport.return_value = http_response(data)
        assert classify("jev", "text", questions).answers["q"]["score"] == 0
        transport.assert_called_once()


def test_structured_descriptions_and_environment_key(transport):
    questions = deepcopy(QUESTIONS)
    questions["route"]["instructions"] = {"task": "Choose queue"}
    questions["route"]["criteria"]["billing"] = {"examples": ["Refund me"]}
    classify("jev", ["message"], questions)
    assert transport.call_args.kwargs["headers"]["Authorization"] == "Bearer env-key"


def test_choice_maximum_options(transport):
    questions = {"q": {"type": "choice", "instructions": "Choose", "criteria": {str(i): None for i in range(256)}}}
    with pytest.raises(RouterError, match="INVALID_INPUT"):
        classify("jev", "text", questions)
    transport.assert_not_called()


def test_choice_255_options_is_valid(transport):
    criteria = {str(i): None for i in range(255)}
    questions = {"q": {"type": "choice", "instructions": "Choose", "criteria": criteria}}
    transport.return_value = http_response({"model": "jev-1.13.0", "answers": {"q": {
        "type": "choice", "choice": "0", "confidence": 1,
        "probabilities": {k: 1 if k == "0" else 0 for k in criteria}}}})
    assert classify("jev", "text", questions).answers["q"]["choice"] == "0"


@pytest.mark.parametrize("config", [{}, {"typesafe_api_key": ""}, {"typesafe_api_key": 1}])
def test_explicit_config_never_falls_back(config, transport):
    with pytest.raises(RouterError, match="MISSING_ENV"):
        classify("jev", "text", QUESTIONS, config=config)
    transport.assert_not_called()


@pytest.mark.parametrize("status,attempts", [(401, 1), (403, 1), (422, 1), (302, 1), (429, 3), (503, 3), (529, 3)])
def test_safe_http_errors_and_retry_cap(status, attempts, transport):
    transport.return_value = http_response({"error": {"message": "env-key private-state"}}, status)
    state = "private-state"
    with pytest.raises(RouterError) as caught:
        classify("jev", state, QUESTIONS)
    assert transport.call_count == attempts
    err = caught.value
    assert err.details["status_code"] == status
    assert err.__context__ is None and err.__cause__ is None
    trace = "".join(traceback.format_exception(err))
    assert "env-key" not in trace and "private-state" not in trace


@pytest.mark.parametrize("status,headers,body", [(429, {}, {"error": {"code": "insufficient_quota"}}),
    (529, {"Retry-After": "121"}, {}), (429, {"Retry-After": "121"}, {})])
def test_quota_and_excessive_backoff_are_not_retried(status, headers, body, transport):
    transport.return_value = http_response(body, status, headers)
    with pytest.raises(RouterError):
        classify("jev", "text", QUESTIONS)
    assert transport.call_count == 1


def test_missing_environment_key(monkeypatch, transport):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(RouterError, match="MISSING_ENV"):
        classify("jev", "text", QUESTIONS)
    transport.assert_not_called()


def test_non_dictionary_config(transport):
    with pytest.raises(RouterError, match="INVALID_PARAM"):
        classify("jev", "text", QUESTIONS, config=[])
    transport.assert_not_called()


def test_header_failure_is_sanitized_without_retry(transport):
    transport.side_effect = requests.exceptions.InvalidHeader("env-key\\r\\n")
    with pytest.raises(RouterError) as caught:
        classify("jev", "text", QUESTIONS)
    assert transport.call_count == 1
    assert "env-key" not in str(caught.value)
    assert caught.value.__context__ is None


def test_transient_then_success_honors_retry_after(transport, monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(implementation._request.retry, "sleep", sleep)
    transport.side_effect = [http_response({}, 529, {"Retry-After": "2"}), http_response()]
    assert classify("jev", "text", QUESTIONS).answers == ANSWER["answers"]
    sleep.assert_called_once_with(2)


def test_connection_error_is_safe_and_retried(transport):
    transport.side_effect = requests.ConnectionError("env-key private-state")
    with pytest.raises(RouterError) as caught:
        classify("jev", "private-state", QUESTIONS)
    assert transport.call_count == 3
    assert caught.value.details["category"] == "connection"
    assert caught.value.__context__ is None


@pytest.mark.parametrize("path,value", [(('answers', 'route', 'choice'), 'unknown'),
    (('answers', 'route', 'choice'), 'tech'), (('answers', 'route', 'confidence'), float('nan')),
    (('answers', 'route', 'probabilities'), {'billing': 0.1, 'tech': 0.1}),
    (('answers', 'urgent', 'noul'), True), (('answers', 'urgent', 'type'), 'boolean'),
    (('answers', 'severity', 'score'), 5), (('answers', 'severity', 'legend'), {}),
    (('answers', 'severity', 'score'), 0.2), (('answers', 'severity', 'legend'), {'0': 5, '1': 'High'}),
    (('answers',), {}), (('model',), ''), (('usage', 'input_tokens'), -1), (('usage',), [])])
def test_malformed_success_fails_without_retry(path, value, transport):
    data = deepcopy(ANSWER)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    transport.return_value = http_response(data)
    with pytest.raises(RouterError) as caught:
        classify("jev", "text", QUESTIONS)
    assert caught.value.code == "PROVIDER_ERROR"
    assert caught.value.__context__ is None
    assert transport.call_count == 1


def test_invalid_json_body_is_sanitized(transport):
    transport.return_value.json.side_effect = ValueError("env-key private-state")
    with pytest.raises(RouterError) as caught:
        classify("jev", "text", QUESTIONS)
    assert "env-key" not in str(caught.value)
    assert transport.call_count == 1
