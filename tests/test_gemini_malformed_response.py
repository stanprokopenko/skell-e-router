"""Gemini finish reason MALFORMED_RESPONSE: retry when safe, otherwise a clear error."""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skell_e_router.errors import MalformedProviderResponse
from skell_e_router.gemini_direct import (
    _call_gemini_direct,
    _call_gemini_direct_stream,
    _chunk_has_output,
    _client_cache,
    _malformed_finish_reason,
    _retry_malformed_stream,
)
from tests.helpers import FAKE_GEMINI_KEY, make_gemini_response, make_model


def _chunk(text=None, finish_reason=None):
    """A stream chunk with real attribute lookups (a MagicMock would invent fields)."""
    part = SimpleNamespace(text=text, function_call=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate])


def _malformed_chunk(text=None):
    return _chunk(text=text, finish_reason="MALFORMED_RESPONSE")


def _bare_config():
    config = MagicMock()
    for attr in ("max_output_tokens", "temperature", "top_p", "top_k", "stop_sequences",
                 "thinking_config", "safety_settings", "candidate_count", "tool_config",
                 "http_options"):
        setattr(config, attr, None)
    return config


@pytest.fixture(autouse=True)
def clear_client_cache():
    _client_cache.clear()


class TestMalformedFinishReason:
    def test_plain_string(self):
        assert _malformed_finish_reason(make_gemini_response(finish_reason="MALFORMED_RESPONSE")) == "MALFORMED_RESPONSE"

    @pytest.mark.parametrize("reason", ["STOP", "MAX_TOKENS", "SAFETY", "MALFORMED_FUNCTION_CALL", None])
    def test_other_reasons_are_not_malformed(self, reason):
        assert _malformed_finish_reason(make_gemini_response(finish_reason=reason)) is None

    def test_no_candidates(self):
        assert _malformed_finish_reason(SimpleNamespace(candidates=None)) is None
        assert _malformed_finish_reason(SimpleNamespace(candidates=[])) is None

    def test_real_sdk_response_with_a_finish_reason_the_sdk_does_not_know(self):
        """google-genai parses an unknown finish reason into a named FinishReason member."""
        types = pytest.importorskip("google.genai.types")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = types.GenerateContentResponse.model_validate({"candidates": [{
                "finish_reason": "MALFORMED_RESPONSE",
                "content": {"role": "model", "parts": [{"text": "partial te"}]},
            }]})
            stop = types.GenerateContentResponse.model_validate({"candidates": [{
                "finish_reason": "STOP",
                "content": {"role": "model", "parts": [{"text": "done"}]},
            }]})
        assert _malformed_finish_reason(response) == "MALFORMED_RESPONSE"
        assert _malformed_finish_reason(stop) is None
        assert _chunk_has_output(response) is True


class TestNonStreaming:
    def _call(self, responses):
        mock_genai = MagicMock()
        generate = mock_genai.Client.return_value.models.generate_content
        generate.side_effect = responses
        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai), \
             patch("skell_e_router.gemini_direct.time.sleep") as sleep:
            try:
                return _call_gemini_direct("gemini/gemini-4-hydrogen", ["hi"], None,
                                           _bare_config(), None, FAKE_GEMINI_KEY), generate, sleep
            except MalformedProviderResponse as exc:
                return exc, generate, sleep

    def test_retries_and_returns_the_good_answer(self):
        bad = make_gemini_response(text="garb", finish_reason="MALFORMED_RESPONSE")
        good = make_gemini_response(text="fine")
        (response, _duration), generate, sleep = self._call([bad, good])
        assert response is good
        assert generate.call_count == 2
        sleep.assert_not_called()

    def test_raises_after_three_malformed_answers(self):
        bad = [make_gemini_response(text="", finish_reason="MALFORMED_RESPONSE") for _ in range(3)]
        exc, generate, _ = self._call(bad)
        assert isinstance(exc, MalformedProviderResponse)
        assert (exc.finish_reason, exc.attempts, exc.partial_output) == ("MALFORMED_RESPONSE", 3, False)
        assert generate.call_count == 3

    def test_transient_error_and_malformed_answer_share_the_attempt_limit(self):
        class TimeoutError_(Exception):
            pass
        bad = make_gemini_response(text="", finish_reason="MALFORMED_RESPONSE")
        exc, generate, sleep = self._call([TimeoutError_(), bad, bad])
        assert isinstance(exc, MalformedProviderResponse)
        assert exc.attempts == 3
        assert generate.call_count == 3
        assert sleep.call_count == 1

    def test_ask_ai_surfaces_a_provider_error_with_its_own_category(self):
        from skell_e_router.utils import RouterError, _ask_ai_direct_gemini
        model = make_model(provider="gemini", name="gemini/gemini-4-hydrogen",
                           supported_params={"max_tokens", "stream"})
        with patch("skell_e_router.utils._call_gemini_direct",
                   side_effect=MalformedProviderResponse("MALFORMED_RESPONSE", attempts=3)), \
             patch("skell_e_router.utils._convert_messages_to_contents", return_value=(None, ["<content>"])), \
             patch("skell_e_router.utils._build_generate_config", return_value=(MagicMock(), None)):
            with pytest.raises(RouterError) as caught:
                _ask_ai_direct_gemini(model, [{"role": "user", "content": "hi"}],
                                      FAKE_GEMINI_KEY, "none", True, None, {})
        error = caught.value
        assert error.code == "PROVIDER_ERROR"
        assert error.message == "Provider returned a malformed response."
        assert error.details == {
            "provider": "gemini", "model": "gemini/gemini-4-hydrogen", "direct_sdk": True,
            "category": "malformed_response", "finish_reason": "MALFORMED_RESPONSE",
            "attempts": 3, "partial_output": False,
        }
        assert error.__cause__ is None and error.__context__ is None


class TestStreaming:
    def test_restarts_when_nothing_was_delivered(self):
        first = iter([_chunk(), _malformed_chunk()])
        good = [_chunk(), _chunk("Hel"), _chunk("lo", finish_reason="STOP")]
        reopen = MagicMock(return_value=iter(good))
        assert list(_retry_malformed_stream(first, reopen)) == good
        reopen.assert_called_once()

    def test_raises_after_output_was_delivered(self):
        delivered = _chunk("Hel")
        reopen = MagicMock()
        stream = _retry_malformed_stream(iter([delivered, _malformed_chunk("lo")]), reopen)
        assert next(stream) is delivered
        with pytest.raises(MalformedProviderResponse) as caught:
            next(stream)
        assert (caught.value.attempts, caught.value.partial_output) == (1, True)
        reopen.assert_not_called()

    def test_gives_up_after_three_empty_malformed_streams(self):
        reopen = MagicMock(side_effect=lambda: iter([_malformed_chunk()]))
        with pytest.raises(MalformedProviderResponse) as caught:
            list(_retry_malformed_stream(iter([_malformed_chunk()]), reopen))
        assert (caught.value.attempts, caught.value.partial_output) == (3, False)
        assert reopen.call_count == 2

    def test_healthy_stream_keeps_every_chunk_in_order(self):
        chunks = [_chunk(), _chunk("a"), _chunk(), _chunk("b", finish_reason="STOP")]
        reopen = MagicMock()
        assert list(_retry_malformed_stream(iter(chunks), reopen)) == chunks
        reopen.assert_not_called()

    def test_stream_with_no_output_at_all_still_yields_its_chunks(self):
        chunks = [_chunk(), _chunk(finish_reason="STOP")]
        assert list(_retry_malformed_stream(iter(chunks), MagicMock())) == chunks

    def test_call_gemini_direct_stream_reopens_through_the_sdk(self):
        mock_genai = MagicMock()
        good = [_chunk("ok", finish_reason="STOP")]
        open_stream = mock_genai.Client.return_value.models.generate_content_stream
        open_stream.side_effect = [iter([_malformed_chunk()]), iter(good)]
        with patch("skell_e_router.gemini_direct.GENAI_AVAILABLE", True), \
             patch("skell_e_router.gemini_direct.genai", mock_genai):
            stream = _call_gemini_direct_stream("gemini/gemini-4-hydrogen", ["hi"], None,
                                                _bare_config(), None, FAKE_GEMINI_KEY)
            assert open_stream.call_count == 1  # first stream opens at call time
            assert list(stream) == good
        assert open_stream.call_count == 2
        assert open_stream.call_args_list[0] == open_stream.call_args_list[1]

    def test_ask_ai_stream_ends_with_a_provider_error(self):
        from skell_e_router.utils import RouterError, _ask_ai_direct_gemini
        model = make_model(provider="gemini", name="gemini/gemini-4-hydrogen",
                           supported_params={"max_tokens", "stream"})
        delivered = _chunk("Hel")
        source = _retry_malformed_stream(iter([delivered, _malformed_chunk()]), MagicMock())
        with patch("skell_e_router.utils._call_gemini_direct_stream", return_value=source), \
             patch("skell_e_router.utils._convert_messages_to_contents", return_value=(None, ["<content>"])), \
             patch("skell_e_router.utils._build_generate_config", return_value=(MagicMock(), None)):
            stream = _ask_ai_direct_gemini(model, [{"role": "user", "content": "hi"}],
                                           FAKE_GEMINI_KEY, "none", False, None, {"stream": True})
        assert next(stream) is delivered
        with pytest.raises(RouterError) as caught:
            next(stream)
        assert caught.value.code == "PROVIDER_ERROR"
        assert caught.value.details["category"] == "malformed_response"
        assert caught.value.details["partial_output"] is True
        assert caught.value.details["attempts"] == 1
