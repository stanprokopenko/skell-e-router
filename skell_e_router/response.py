from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeminiFileRef:
    """Reference to a file uploaded via Gemini's Files API."""
    uri: str           # "https://generativelanguage.googleapis.com/v1beta/files/abc123"
    mime_type: str     # "video/mp4"
    display_name: str | None = None


@dataclass
class AIResponse:
    # Core content
    content: str

    # Model info
    model: str
    finish_reason: str | None = None

    # Token usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None

    # Cost and timing
    cost: float | None = None
    duration_seconds: float | None = None
    total_duration_seconds: float | None = None

    # Provider-specific
    grounding_metadata: dict | None = None
    safety_ratings: list[dict] | None = None
    images: list[dict] | None = None
    tool_calls: list | None = None
    function_call: Any = None
    provider_specific_fields: dict | None = None

    # Raw response for advanced use
    raw_response: Any = None

    def __str__(self) -> str:
        """Allow AIResponse to be used as a string (returns content)."""
        return self.content

    def __repr__(self) -> str:
        return f"AIResponse(content={self.content[:50]!r}..., model={self.model!r})"


@dataclass
class EmbeddingResponse:
    """Result of get_embedding(); always carries embeddings as list[list[float]]."""

    # Core data — always nested, even when caller passed a single string
    embeddings: list[list[float]]
    model: str          # provider-reported model name
    dimensions: int     # observed: len(embeddings[0])

    # Token usage (embedding APIs only report prompt tokens)
    prompt_tokens: int | None = None
    total_tokens: int | None = None

    # Cost and timing — same pattern as AIResponse
    cost: float | None = None
    duration_seconds: float | None = None
    total_duration_seconds: float | None = None

    # Raw provider response for advanced inspection
    raw_response: Any = None

    def __repr__(self) -> str:
        return (
            f"EmbeddingResponse(model={self.model!r}, "
            f"n={len(self.embeddings)}, dim={self.dimensions})"
        )
