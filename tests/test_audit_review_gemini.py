"""Prompt-level Gemini refusals must remain distinct from empty responses."""
from types import SimpleNamespace

import pytest
from google.genai import types

from skell_e_router.gemini_direct import _build_response


@pytest.mark.parametrize("reason", [
    "SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY",
    "MODEL_ARMOR", "JAILBREAK",
])
def test_prompt_blocks_without_candidates_are_refusals(reason):
    response = types.GenerateContentResponse.model_validate({
        "promptFeedback": {"blockReason": reason},
    })
    result = _build_response(response, "gemini-3.8-flash", 0.1)
    assert result.finish_reason == "content_filter"
    assert result.content == ""


@pytest.mark.parametrize("reason", [None, "BLOCKED_REASON_UNSPECIFIED", "FUTURE_REASON"])
def test_missing_or_unknown_block_reason_is_not_invented(reason):
    response = SimpleNamespace(text="", candidates=[], usage_metadata=None,
                               prompt_feedback=SimpleNamespace(block_reason=reason))
    assert _build_response(response, "gemini-3.8-flash", 0.1).finish_reason is None


@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS", None])
def test_existing_candidate_takes_priority_over_prompt_feedback(finish):
    response = types.GenerateContentResponse.model_validate({
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [{"content": {"role": "model", "parts": [{"text": "partial"}]},
                        "finishReason": finish}],
    })
    result = _build_response(response, "gemini-3.8-flash", 0.1)
    assert result.finish_reason == finish
    assert result.content == "partial"
