# Gemini Deep Research Agent Integration Plan

## Overview

The Gemini Deep Research Agent uses a fundamentally different API paradigm than standard LLM completions:

| Aspect | Standard LiteLLM | Deep Research Agent |
|--------|------------------|---------------------|
| API | `litellm.completion()` | `client.interactions.create()` |
| Execution | Synchronous | Asynchronous (background) |
| Response | Immediate | Polling required |
| Latency | Seconds | Minutes (up to 60 min) |
| Client | LiteLLM wrapper | `google.genai.Client()` |

Since LiteLLM does not support the Interactions API, we need to implement a **direct integration** with the Google GenAI SDK.

---

## Proposed Architecture

### Option A: Separate Module (Recommended)

Create a dedicated module for the Deep Research Agent that follows the same patterns as the existing router but uses the native Google SDK.

```
skell_e_router/
├── __init__.py              # Add new exports
├── model_config.py          # Add Deep Research agent config
├── utils.py                 # Existing LiteLLM-based functions
└── deep_research.py         # NEW: Deep Research Agent module
```

**Pros:**
- Clean separation of concerns
- No risk of breaking existing functionality
- Easy to maintain and extend
- Clear API boundary

**Cons:**
- Slightly different API surface (necessary due to async nature)

> **Note:** File will be named `gemini_deep_research.py` to clearly indicate it's Gemini-specific.

### Option B: Unified Interface with Adapter

Extend [`ask_ai()`](skell_e_router/utils.py:434) to detect Deep Research models and route to a different backend.

**Pros:**
- Single entry point for all AI calls

**Cons:**
- Complicates the simple `ask_ai` interface
- Deep Research has fundamentally different semantics (async, polling)
- Would require significant refactoring

### Recommendation: Option A

The Deep Research Agent's async nature and different response model make it unsuitable for the synchronous `ask_ai()` interface. A separate module provides clarity and proper handling of the unique requirements.

---

## Detailed Design

### 1. New File: [`skell_e_router/deep_research.py`](skell_e_router/deep_research.py)

```python
# Core classes and functions:

class DeepResearchConfig:
    """Configuration for Deep Research agent."""
    agent: str = "deep-research-pro-preview-12-2025"
    background: bool = True
    stream: bool = False
    thinking_summaries: str = "auto"
    tools: list = None  # Optional file_search tools

class DeepResearchResult:
    """Result from a Deep Research task."""
    id: str
    status: str  # "in_progress", "completed", "failed"
    text: str | None
    error: str | None
    outputs: list
    citations: list
    # Metadata for cost tracking
    usage: DeepResearchUsage | None  # Token counts and pricing
    duration_seconds: float | None   # Total research time

def ask_deep_research(
    query: str,
    *,
    stream: bool = False,
    tools: list = None,
    poll_interval: float = 10.0,
    timeout: float = 3600.0,  # 60 min max
    verbosity: str = "none",
    on_progress: Callable = None,  # Callback for streaming updates
) -> DeepResearchResult:
    """
    Execute a Deep Research task and wait for completion.
    
    Args:
        query: The research question/prompt
        stream: Enable streaming for real-time updates
        tools: Optional tools like file_search
        poll_interval: Seconds between status checks
        timeout: Maximum wait time in seconds
        verbosity: Output level (none, response, info, debug)
        on_progress: Callback for streaming thought summaries
    
    Returns:
        DeepResearchResult with the final report
    """

async def ask_deep_research_async(
    query: str,
    **kwargs
) -> DeepResearchResult:
    """Async version for use in async contexts."""

def follow_up(
    previous_interaction_id: str,
    query: str,
    model: str = "gemini-3-pro-preview",
    verbosity: str = "none"
) -> str:
    """Ask a follow-up question about a completed research task."""
```

### 2. Update [`skell_e_router/model_config.py`](skell_e_router/model_config.py)

Add a new agent type for Deep Research:

```python
class AIAgent:
    """Configuration for AI agents (non-completion APIs)."""
    def __init__(
        self,
        name: str,
        provider: str,
        api_type: str,  # "interactions"
        supported_params: set[str]
    ):
        self.name = name
        self.provider = provider
        self.api_type = api_type
        self.supported_params = supported_params

AGENT_CONFIG = {
    "deep-research": AIAgent(
        name="deep-research-pro-preview-12-2025",
        provider="gemini",
        api_type="interactions",
        supported_params={
            "background", "stream", "tools", 
            "agent_config", "previous_interaction_id"
        }
    ),
}
```

### 3. Update [`skell_e_router/__init__.py`](skell_e_router/__init__.py)

```python
from .utils import ask_ai, resolve_model_alias, check_environment_variables, RouterError
from .deep_research import (
    ask_deep_research,
    ask_deep_research_async,
    follow_up as deep_research_follow_up,
    DeepResearchResult,
    DeepResearchConfig,
)

__all__ = [
    # Existing
    "ask_ai",
    "resolve_model_alias", 
    "check_environment_variables",
    "RouterError",
    # New Deep Research
    "ask_deep_research",
    "ask_deep_research_async",
    "deep_research_follow_up",
    "DeepResearchResult",
    "DeepResearchConfig",
]
```

### 4. Dependencies

Update [`pyproject.toml`](pyproject.toml):

```toml
dependencies = [
    "litellm>=1.43.0",
    "tenacity>=8.0.0",
    "google-genai>=1.0.0",  # NEW: For Deep Research
]
```

---

## Implementation Details

### Polling Logic

```python
def _poll_for_completion(
    client: genai.Client,
    interaction_id: str,
    poll_interval: float,
    timeout: float,
    verbosity: str
) -> DeepResearchResult:
    """Poll until research completes or times out."""
    start_time = time.time()
    
    while True:
        if time.time() - start_time > timeout:
            raise RouterError(
                code="TIMEOUT",
                message=f"Deep Research timed out after {timeout}s",
                details={"interaction_id": interaction_id}
            )
        
        interaction = client.interactions.get(interaction_id)
        
        if verbosity in ("info", "debug"):
            print(f"Status: {interaction.status}")
        
        if interaction.status == "completed":
            return DeepResearchResult(
                id=interaction_id,
                status="completed",
                text=interaction.outputs[-1].text,
                outputs=interaction.outputs,
                citations=getattr(interaction, "citations", [])
            )
        elif interaction.status == "failed":
            raise RouterError(
                code="RESEARCH_FAILED",
                message=str(interaction.error),
                details={"interaction_id": interaction_id}
            )
        
        time.sleep(poll_interval)
```

### Streaming with Reconnection

```python
def _stream_with_reconnection(
    client: genai.Client,
    query: str,
    config: DeepResearchConfig,
    on_progress: Callable,
    verbosity: str
) -> DeepResearchResult:
    """Stream research with automatic reconnection on failure."""
    interaction_id = None
    last_event_id = None
    final_text = ""
    
    def process_stream(stream):
        nonlocal interaction_id, last_event_id, final_text
        
        for chunk in stream:
            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                
            if chunk.event_id:
                last_event_id = chunk.event_id
                
            if chunk.event_type == "content.delta":
                if chunk.delta.type == "text":
                    final_text += chunk.delta.text
                    if on_progress:
                        on_progress("text", chunk.delta.text)
                elif chunk.delta.type == "thought_summary":
                    if on_progress:
                        on_progress("thought", chunk.delta.content.text)
                        
            if chunk.event_type == "interaction.complete":
                return True
        return False
    
    # Initial stream
    try:
        stream = client.interactions.create(
            input=query,
            agent=config.agent,
            background=True,
            stream=True,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": config.thinking_summaries
            },
            tools=config.tools
        )
        if process_stream(stream):
            return DeepResearchResult(...)
    except Exception as e:
        if verbosity != "none":
            print(f"Stream interrupted: {e}")
    
    # Reconnection loop
    while interaction_id:
        try:
            stream = client.interactions.get(
                id=interaction_id,
                stream=True,
                last_event_id=last_event_id
            )
            if process_stream(stream):
                return DeepResearchResult(...)
        except Exception as e:
            if verbosity != "none":
                print(f"Reconnection failed: {e}")
            time.sleep(2)
```

---

## Usage Examples

### Basic Research

```python
from skell_e_router import ask_deep_research

result = ask_deep_research(
    "Research the competitive landscape of EV batteries",
    verbosity="info"
)
print(result.text)
```

### Streaming with Progress

```python
from skell_e_router import ask_deep_research

def on_update(event_type, content):
    if event_type == "thought":
        print(f"[Thinking] {content}")
    else:
        print(content, end="", flush=True)

result = ask_deep_research(
    "Compare golang SDK test frameworks",
    stream=True,
    on_progress=on_update
)
```

### With File Search

```python
from skell_e_router import ask_deep_research

result = ask_deep_research(
    "Compare our 2025 fiscal year report against current public web news",
    tools=[{
        "type": "file_search",
        "file_search_store_names": ["fileSearchStores/my-store"]
    }]
)
```

### Follow-up Questions

```python
from skell_e_router import ask_deep_research, deep_research_follow_up

# Initial research
result = ask_deep_research("Research the history of Google TPUs")

# Follow-up
clarification = deep_research_follow_up(
    previous_interaction_id=result.id,
    query="Can you elaborate on the second point?"
)
```

---

## Error Handling

Reuse the existing [`RouterError`](skell_e_router/utils.py:20) class with new error codes:

| Code | Description |
|------|-------------|
| `RESEARCH_FAILED` | Deep Research task failed |
| `TIMEOUT` | Research exceeded timeout |
| `STREAM_ERROR` | Streaming connection failed |
| `INVALID_INTERACTION` | Invalid interaction ID for follow-up |

---

## Testing Considerations

1. **Mock the Google GenAI client** for unit tests
2. **Integration tests** with real API (rate-limited)
3. **Timeout handling** tests
4. **Reconnection logic** tests with simulated failures

---

## Migration Path

This is an **additive change** - no breaking changes to existing functionality:

1. Existing `ask_ai()` continues to work unchanged
2. New `ask_deep_research()` provides Deep Research capability
3. Users can adopt incrementally

---

## Open Questions

1. **Should we add a unified interface later?** We could add a `ask_ai_extended()` that handles both sync and async models, but this adds complexity.

2. **Async-first or sync-first?** The current plan provides both, with sync as the default (matches existing `ask_ai` pattern).

3. **File search store management?** Should we add helpers for creating/managing file search stores, or leave that to the user?
