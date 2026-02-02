"""Shared test helpers for the skell-e-router test suite."""

from unittest.mock import MagicMock
from skell_e_router.model_config import AIModel


# ---------------------------------------------------------------------------
# Fake API keys
# ---------------------------------------------------------------------------

FAKE_OPENAI_KEY = "sk-test-openai-key-abc123"
FAKE_GEMINI_KEY = "AIzaSy-test-gemini-key-xyz789"
FAKE_ANTHROPIC_KEY = "sk-ant-test-anthropic-key-def456"
FAKE_GROQ_KEY = "gsk-test-groq-key"
FAKE_XAI_KEY = "xai-test-key"

FULL_CONFIG = {
    "openai_api_key": FAKE_OPENAI_KEY,
    "gemini_api_key": FAKE_GEMINI_KEY,
    "anthropic_api_key": FAKE_ANTHROPIC_KEY,
    "groq_api_key": FAKE_GROQ_KEY,
    "xai_api_key": FAKE_XAI_KEY,
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def make_model(
    provider: str = "openai",
    name: str | None = None,
    supports_thinking: bool = False,
    supported_params: set[str] | None = None,
    accepted_reasoning_efforts: set[str] | None = None,
) -> AIModel:
    """Create a minimal AIModel for testing."""
    return AIModel(
        name=name or f"{provider}/test-model",
        provider=provider,
        supports_thinking=supports_thinking,
        supported_params=supported_params or {"stream"},
        accepted_reasoning_efforts=accepted_reasoning_efforts,
    )


# ---------------------------------------------------------------------------
# Mock LiteLLM response builder
# ---------------------------------------------------------------------------

def make_litellm_response(
    content: str = "Hello world",
    model: str = "openai/gpt-5",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
    reasoning_tokens: int | None = None,
    tool_calls=None,
    function_call=None,
    provider_specific_fields=None,
    grounding_metadata=None,
    safety_ratings=None,
):
    """Build a mock that looks like a litellm completion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.function_call = function_call
    message.provider_specific_fields = provider_specific_fields

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    completion_details = MagicMock()
    completion_details.reasoning_tokens = reasoning_tokens

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    usage.completion_tokens_details = completion_details

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.vertex_ai_grounding_metadata = grounding_metadata
    response.vertex_ai_safety_results = safety_ratings
    return response


# ---------------------------------------------------------------------------
# Mock deep-research interaction builder
# ---------------------------------------------------------------------------

def make_interaction(
    interaction_id: str = "inter-123",
    status: str = "completed",
    error=None,
    outputs=None,
    usage_metadata=None,
    citations=None,
):
    """Build a mock that looks like a google-genai interaction object."""
    interaction = MagicMock()
    interaction.id = interaction_id
    interaction.status = status
    interaction.error = error
    interaction.outputs = outputs or []
    interaction.usage_metadata = usage_metadata
    interaction.citations = citations
    return interaction
