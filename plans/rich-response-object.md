# Rich Response Object for skell-e-router

## Overview

Currently, `ask_ai()` returns only the response content string. Apps need access to additional metadata like token counts, cost, model info, finish reason, grounding metadata, and timing. This plan adds a rich response object while maintaining full backwards compatibility.

## Current Behavior

```python
# Current: returns str
content = ask_ai("gemini-3-pro-preview", "Hello")
print(content)  # "Hello! How can I help you?"
```

## Proposed Behavior

```python
# Option 1: Default behavior unchanged (backwards compatible)
content = ask_ai("gemini-3-pro-preview", "Hello")
print(content)  # "Hello! How can I help you?"

# Option 2: Request rich response via parameter
response = ask_ai("gemini-3-pro-preview", "Hello", return_response=True)
print(response.content)           # "Hello! How can I help you?"
print(response.model)             # "gemini-3-pro-preview"
print(response.cost)              # 0.000123
print(response.prompt_tokens)     # 5
print(response.completion_tokens) # 10
print(response.total_tokens)      # 15
print(response.reasoning_tokens)  # 0 or None
print(response.finish_reason)     # "stop"
print(response.duration_seconds)  # 1.234
print(response.grounding_metadata) # {...} or None
print(response.raw_response)      # Full LiteLLM response object
```

---

## Implementation Plan

### Task 1: Create AIResponse dataclass

Create a new dataclass in `skell_e_router/response.py`:

```python
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
```

**File:** `skell_e_router/response.py`

---

### Task 2: Add response builder function

Add a helper function in `skell_e_router/utils.py` to construct `AIResponse` from LiteLLM response:

```python
def _build_ai_response(
    response,
    request_duration_s: float | None = None
) -> AIResponse:
    """Build AIResponse from LiteLLM completion response."""
    
    content = response.choices[0].message.content if response.choices else ""
    model_name = getattr(response, 'model', 'unknown')
    
    usage = getattr(response, 'usage', None)
    completion_details = getattr(usage, 'completion_tokens_details', None) if usage else None
    first_choice = response.choices[0] if response.choices else None
    message = getattr(first_choice, 'message', None) if first_choice else None
    
    # Compute cost safely
    try:
        computed_cost = litellm.completion_cost(completion_response=response)
    except Exception:
        computed_cost = None
    
    # Extract grounding metadata for Gemini
    grounding_metadata = None
    if hasattr(response, 'vertex_ai_grounding_metadata'):
        grounding_metadata = response.vertex_ai_grounding_metadata
    
    # Extract safety ratings for Gemini
    safety_ratings = None
    if hasattr(response, 'vertex_ai_safety_results'):
        safety_ratings = response.vertex_ai_safety_results
    
    return AIResponse(
        content=content or "",
        model=model_name,
        finish_reason=getattr(first_choice, 'finish_reason', None),
        prompt_tokens=getattr(usage, 'prompt_tokens', None),
        completion_tokens=getattr(usage, 'completion_tokens', None),
        total_tokens=getattr(usage, 'total_tokens', None),
        reasoning_tokens=getattr(completion_details, 'reasoning_tokens', None),
        cost=computed_cost,
        duration_seconds=request_duration_s,
        grounding_metadata=grounding_metadata,
        safety_ratings=safety_ratings,
        tool_calls=getattr(message, 'tool_calls', None),
        function_call=getattr(message, 'function_call', None),
        provider_specific_fields=getattr(message, 'provider_specific_fields', None),
        raw_response=response,
    )
```

---

### Task 3: Modify ask_ai function signature

Update `ask_ai()` in `skell_e_router/utils.py`:

```python
def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    return_response: bool = False,  # NEW PARAMETER
    **kwargs
) -> str | AIResponse:
    """
    Call an AI model with the given input.
    
    Args:
        model_alias: Model identifier (e.g., "gemini-3-pro-preview")
        user_input: Prompt string or conversation history
        system_message: Optional system prompt
        verbosity: Output level ('none', 'response', 'info', 'debug')
        return_response: If True, return AIResponse object instead of string
        **kwargs: Model-specific parameters
    
    Returns:
        str: Response content (default, backwards compatible)
        AIResponse: Rich response object (when return_response=True)
    """
```

---

### Task 4: Update ask_ai return logic

Modify the return section of `ask_ai()`:

```python
# Current code (around line 461-463):
content = response.choices[0].message.content
_print_response_details(response, verbosity, request_duration_s)
return content

# New code:
content = response.choices[0].message.content
_print_response_details(response, verbosity, request_duration_s)

if return_response:
    return _build_ai_response(response, request_duration_s)
return content
```

---

### Task 5: Update __init__.py exports

Add `AIResponse` to the public API in `skell_e_router/__init__.py`:

```python
from .response import AIResponse
from .utils import ask_ai, RouterError

__all__ = [
    "ask_ai",
    "RouterError",
    "AIResponse",
    # ... existing exports
]
```

---

### Task 6: Update type hints

Add return type annotation to `ask_ai()`:

```python
from typing import overload, Literal

@overload
def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    return_response: Literal[False] = False,
    **kwargs
) -> str: ...

@overload
def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    return_response: Literal[True] = ...,
    **kwargs
) -> AIResponse: ...

def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    return_response: bool = False,
    **kwargs
) -> str | AIResponse:
    ...
```

---

### Task 7: Update README.md documentation

Add a new section to README.md:

```markdown
## Rich Response Object

By default, `ask_ai()` returns just the response content string for backwards compatibility. To get full response metadata, use `return_response=True`:

### Basic Usage (Backwards Compatible)
```python
content = ask_ai("gemini-3-pro-preview", "Hello")
print(content)  # Just the string
```

### Rich Response
```python
response = ask_ai("gemini-3-pro-preview", "Hello", return_response=True)

# Access content
print(response.content)

# Token usage
print(f"Tokens: {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens}")
print(f"Reasoning tokens: {response.reasoning_tokens}")

# Cost and timing
print(f"Cost: ${response.cost:.6f}")
print(f"Duration: {response.duration_seconds:.2f}s")

# Model info
print(f"Model: {response.model}")
print(f"Finish reason: {response.finish_reason}")

# Grounding metadata (Gemini with web_search_options)
if response.grounding_metadata:
    print(f"Sources: {len(response.grounding_metadata[0].get('groundingChunks', []))}")

# Can still use as string
print(response)  # Prints content via __str__
```

### AIResponse Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | str | The response text |
| `model` | str | Model name used |
| `finish_reason` | str | Why generation stopped |
| `prompt_tokens` | int | Input token count |
| `completion_tokens` | int | Output token count |
| `total_tokens` | int | Total tokens used |
| `reasoning_tokens` | int | Thinking/reasoning tokens |
| `cost` | float | Estimated cost in USD |
| `duration_seconds` | float | Request duration |
| `grounding_metadata` | dict | Web search citations (Gemini) |
| `safety_ratings` | list | Content safety scores (Gemini) |
| `tool_calls` | list | Function calls made |
| `raw_response` | Any | Full LiteLLM response |
```

---

### Task 8: Create example file

Create `example_rich_response.py`:

```python
from skell_e_router import ask_ai, AIResponse

# Basic usage - backwards compatible
content = ask_ai("gemini-2.5-flash", "What is 2+2?")
print(f"Basic: {content}")

# Rich response
response = ask_ai(
    "gemini-3-pro-preview",
    "What is the latest news?",
    return_response=True,
    web_search_options={"search_context_size": "high"}
)

print(f"\nContent: {response.content[:100]}...")
print(f"Model: {response.model}")
print(f"Cost: ${response.cost:.6f}")
print(f"Duration: {response.duration_seconds:.2f}s")
print(f"Tokens: {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens}")

if response.grounding_metadata:
    chunks = response.grounding_metadata[0].get('groundingChunks', [])
    print(f"Sources: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"  - {chunk.get('web', {}).get('title', 'Unknown')}")
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `skell_e_router/response.py` | Create | New `AIResponse` dataclass |
| `skell_e_router/utils.py` | Modify | Add `_build_ai_response()`, update `ask_ai()` |
| `skell_e_router/__init__.py` | Modify | Export `AIResponse` |
| `README.md` | Modify | Add Rich Response documentation |
| `example_rich_response.py` | Create | Usage examples |

---

## Backwards Compatibility Guarantee

1. **Default behavior unchanged**: `ask_ai()` returns `str` by default
2. **No breaking changes**: Existing code continues to work without modification
3. **Opt-in rich response**: Apps must explicitly pass `return_response=True`
4. **String coercion**: `AIResponse.__str__()` returns content, so `print(response)` works

---

## Assumptions to Review

1. **Parameter name**: Using `return_response=True` as the opt-in flag. Alternatives considered: `full_response`, `include_metadata`, `as_object`.

2. **AIResponse is a dataclass**: Using `@dataclass` for simplicity. Could use Pydantic `BaseModel` for validation, but adds dependency complexity.

3. **`__str__` returns content**: This allows `print(response)` to work naturally, but means `str(response)` loses metadata. Apps needing the string should use `response.content` explicitly.

4. **Grounding metadata structure**: Assuming `vertex_ai_grounding_metadata` is a list with the first element containing `groundingChunks`, `groundingSupports`, etc. This matches observed LiteLLM behavior.

5. **Cost calculation**: Using `litellm.completion_cost()` which may return `None` for unmapped models. Apps should handle `None` gracefully.

6. **No streaming support initially**: This plan covers non-streaming responses. Streaming would require a different approach (generator yielding chunks, then final AIResponse).

7. **raw_response included**: Including the full LiteLLM response object for advanced use cases. This could increase memory usage for apps that store many responses.

8. **Type hints with overloads**: Using `@overload` for precise type inference. This requires Python 3.8+ and modern type checkers.

9. **Single file for response class**: Creating `response.py` rather than putting `AIResponse` in `utils.py` to keep concerns separated.

10. **No serialization methods**: Not adding `to_dict()` or `to_json()` methods initially. Apps can use `dataclasses.asdict()` if needed.
