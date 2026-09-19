"""TypeSafe's typed evaluation API. This endpoint does not generate text."""

import math
import os
import time

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt

from .errors import call_provider
from .model_config import resolve_classification_alias
from .response import ClassificationResponse
from .utils import RouterError, _is_retryable_exception, _retry_after_wait


_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _json_value(v) for k, v in value.items())
    return False


def _description(value):
    return type(value) in (str, dict, list) and bool(value) and _json_value(value)


def _validate(state, questions):
    # Do not echo caller data in validation errors, including cyclic objects.
    try:
        valid_state = type(state) in (str, dict, list) and _json_value(state)
        valid_json = _json_value(questions)
    except RecursionError:
        valid_state = valid_json = False
    if not valid_state:
        raise RouterError("INVALID_INPUT", "State must contain JSON text or structured data.")
    if not valid_json or not isinstance(questions, dict) or not questions:
        raise RouterError("INVALID_INPUT", "Questions must be a nonempty map of typed questions.")
    for key, question in questions.items():
        if not key or not isinstance(question, dict):
            raise RouterError("INVALID_INPUT", "Each question needs a nonempty string ID and an object.")
        kind = question.get("type")
        if kind not in ("choice", "score", "noul") or not _description(question.get("instructions")):
            raise RouterError("INVALID_INPUT", "Questions require choice, score, or noul type and instructions.")
        if set(question) - {"type", "instructions", "criteria"}:
            raise RouterError("INVALID_INPUT", "Unsupported question field.")
        criteria = question.get("criteria")
        if kind == "choice":
            valid = (isinstance(criteria, dict) and 2 <= len(criteria) <= 255
                     and all(k and (v is None or _description(v)) for k, v in criteria.items()))
        elif kind == "score":
            valid = isinstance(criteria, list) and 2 <= len(criteria) <= 10 and all(_description(v) for v in criteria)
        else:
            valid = "criteria" not in question or (
                isinstance(criteria, dict) and bool(criteria)
                and not set(criteria) - {"true", "false"}
                and all(_description(v) for v in criteria.values()))
        if not valid:
            raise RouterError("INVALID_INPUT", "Invalid criteria for the question type.")


def _retryable(exc):
    # TypeSafe documents 529 as temporary overload; reuse shared 503 handling.
    if getattr(exc, "status_code", None) == 529:
        proxy = requests.HTTPError()
        proxy.status_code = 503
        proxy.headers = exc.headers
        return _is_retryable_exception(proxy)
    return _is_retryable_exception(exc)


def _wait(retry_state):
    exc = retry_state.outcome.exception()
    if getattr(exc, "status_code", None) == 529:
        # Only the retry helper sees this controlled status; retain 529 in errors.
        original = exc.status_code
        try:
            exc.status_code = 503
            return _retry_after_wait(retry_state)
        finally:
            exc.status_code = original
    return _retry_after_wait(retry_state)


@retry(retry=retry_if_exception(_retryable), wait=_wait,
       stop=stop_after_attempt(3), reraise=True)
def _request(payload, api_key, timeout):
    with requests.post(_ENDPOINT, json=payload,
                       headers={"Authorization": f"Bearer {api_key}"},
                       timeout=timeout, allow_redirects=False) as response:
        if response.status_code != 200:
            exc = requests.HTTPError()
            exc.status_code = response.status_code
            exc.headers = dict(response.headers)
            # Preserve only the machine-readable code for quota-aware retries.
            try:
                error = response.json().get("error", {})
                if isinstance(error, dict):
                    exc.code = error.get("code") or error.get("type")
            except (ValueError, AttributeError):
                pass
            raise exc
        return response.json()


def _number(value, maximum=1):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum


#: Decimal places the provider rounds probabilities and score answers to.
#: Every tolerance below is derived from this, because a value rounded to two
#: places cannot be checked more precisely than the rounding itself allows.
_ANSWER_DECIMALS = 2
_HALF_ULP = 0.5 * 10 ** -_ANSWER_DECIMALS


def _sum_tolerance(options):
    """How far a rounded distribution's sum may honestly sit from 1.

    Each of the ``options`` probabilities is rounded independently, so it
    carries up to half a unit in the last place of error and the reported sum
    can miss 1 by ``options * half_ulp``. A six-level score routinely comes
    back summing to 0.99; demanding 1.000 +/- 0.001 rejected those real answers
    as a malformed provider response. The test that survives is whether SOME
    valid distribution rounds to the reported numbers, which is exactly this
    bound.
    """
    return _HALF_ULP * options


def _score_tolerance(levels):
    """Largest honest gap between a reported score and its probabilities.

    The score is the probability-weighted mean of the level numbers, but the
    caller only ever sees rounded probabilities, so recomputing the mean from
    them cannot reproduce the reported score exactly. The rounding errors sum
    to roughly zero, so the worst case shifts half a unit in the last place off
    the lowest levels onto the highest: ``half_ulp * floor(levels ** 2 / 4)``.
    One more half ulp covers the score's own rounding.

    For a six-level score that is 0.05, eight times what a flat
    ``0.001 * levels`` rule allowed. Real answers drift by up to 0.04, so the
    flat rule failed roughly a third of live requests. A genuinely
    contradictory score is still caught.
    """
    return _HALF_ULP * (levels ** 2 // 4) + _HALF_ULP


def _parse(data, questions, model):
    # A malformed success is a provider failure, never a fabricated decision.
    if not isinstance(data, dict) or not isinstance(data.get("model"), str) or not data["model"]:
        raise ValueError("Malformed evaluation response")
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Mismatched answer IDs")
    for key, question in questions.items():
        answer = answers[key]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("Mismatched answer type")
        if kind == "noul":
            if not _number(answer.get("noul")):
                raise ValueError("Invalid boolean probability")
            continue
        expected = set(question["criteria"]) if kind == "choice" else {
            str(i) for i in range(len(question["criteria"]))}
        probabilities = answer.get("probabilities")
        if (not isinstance(probabilities, dict) or set(probabilities) != expected
                or not all(_number(p) for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1,
                                    abs_tol=_sum_tolerance(len(expected)))
                or not _number(answer.get("confidence"))):
            raise ValueError("Invalid probability distribution")
        if kind == "choice":
            choice = answer.get("choice")
            if (not isinstance(choice, str) or choice not in expected
                    or probabilities[choice] < max(probabilities.values())):
                raise ValueError("Invalid selected option")
        else:
            legend = answer.get("legend")
            if (not _number(answer.get("score"), len(expected) - 1)
                    or not isinstance(legend, dict) or set(legend) != expected
                    or not all(_description(v) for v in legend.values())):
                raise ValueError("Invalid score or legend")
            weighted = sum(int(k) * p for k, p in probabilities.items())
            if not math.isclose(answer["score"], weighted,
                                abs_tol=_score_tolerance(len(expected))):
                raise ValueError("Score contradicts probabilities")
    usage = data.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("Invalid usage")
    counts = [(usage or {}).get(k) for k in ("input_tokens", "output_tokens")]
    if any(v is not None and (type(v) is not int or v < 0) for v in counts):
        raise ValueError("Invalid token count")
    return ClassificationResponse(
        answers=answers, model=data["model"], input_tokens=counts[0], output_tokens=counts[1],
        cost=None if counts[0] is None else counts[0] * model.input_cost_per_million / 1_000_000,
    )


def classify(model: str, state: str | dict | list, questions: dict, *,
             config: dict | None = None, timeout: float = 30) -> ClassificationResponse:
    """Evaluate choice, score, and noul questions through TypeSafe.

    ``jev`` pins 1.13.0; ``jev-latest`` follows provider upgrades. Credentials
    come from ``config['typesafe_api_key']`` or TYPESAFE_API_KEY when config is
    None. Timeout is per HTTP attempt, with at most three transient attempts.
    Context limits are provider-enforced, using the provider's tokenizer.
    """
    selected = resolve_classification_alias(model)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise RouterError("INVALID_PARAM", "Timeout must be a finite positive number.")
    if config is not None and not isinstance(config, dict):
        raise RouterError("INVALID_PARAM", "Config must be a dictionary or None.")
    _validate(state, questions)
    api_key = config.get("typesafe_api_key") if config is not None else os.getenv("TYPESAFE_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RouterError("MISSING_ENV", "TYPESAFE_API_KEY or config typesafe_api_key is required.")
    payload = {"model": selected.name, "state": state, "questions": questions}
    started = time.perf_counter()
    result = call_provider(
        lambda: _parse(_request(payload, api_key, timeout), questions, selected),
        RouterError, details={"provider": selected.provider, "model": selected.name, "operation": "classify"},
    )
    result.duration_seconds = time.perf_counter() - started
    return result
