from dataclasses import dataclass, field
from typing import Any

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
    
    # Provider-specific
    grounding_metadata: dict | None = None
    safety_ratings: list[dict] | None = None
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
