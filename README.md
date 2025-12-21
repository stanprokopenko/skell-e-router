# skell-e-router

Simple AI router using LiteLLM with Gemini Deep Research Agent support.

## Install the package

```bash
pip install git+https://github.com/stanprokopenko/skell-e-router@main
```

## Quick Start

### Standard AI Completion (via LiteLLM)

```python
from skell_e_router import ask_ai

response = ask_ai(
    "gemini-2.5-pro",
    "Explain quantum computing in simple terms",
    verbosity="response"
)
```

### Gemini Deep Research Agent

```python
from skell_e_router import ask_deep_research

result = ask_deep_research(
    "Research the competitive landscape of EV batteries",
    verbosity="info"
)
print(result.text)  # Full research report
print(f"Duration: {result.duration_seconds:.0f}s")
```

## Rich Response Object

By default, `ask_ai()` returns just the response content string for backwards compatibility. To get full response metadata, use `rich_response=True`:

### Basic Usage (Backwards Compatible)
```python
content = ask_ai("gemini-3-pro-preview", "Hello")
print(content)  # Just the string
```

### Rich Response
```python
response = ask_ai("gemini-3-pro-preview", "Hello", rich_response=True)

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

## Gemini Deep Research Agent

The Gemini Deep Research Agent autonomously plans, executes, and synthesizes multi-step research tasks. It navigates complex information landscapes using web search to produce detailed, cited reports.

**Key characteristics:**
- Research tasks take **several minutes** (up to 60 min max)
- Uses Google's Interactions API (not LiteLLM)
- Requires `GEMINI_API_KEY` environment variable

### Basic Usage

```python
from skell_e_router import ask_deep_research, DeepResearchError

try:
    result = ask_deep_research(
        "Research the history of Google TPUs",
        verbosity="info",      # Shows progress updates
        poll_interval=10.0,    # Check status every 10 seconds
        timeout=1800.0,        # 30 minute timeout
    )
    
    print(result.text)         # The research report
    print(result.id)           # Interaction ID for follow-ups
    print(result.citations)    # Sources cited
    
    if result.usage:
        print(f"Tokens: {result.usage.total_tokens}")
    
except DeepResearchError as err:
    print(f"Error: {err.code} - {err.message}")
```

### Streaming with Progress Updates

```python
from skell_e_router import ask_deep_research

def on_progress(event_type: str, content: str):
    if event_type == "thought":
        print(f"[Thinking] {content}")
    elif event_type == "text":
        print(content, end="", flush=True)

result = ask_deep_research(
    "Compare AWS, Azure, and GCP for ML workloads",
    stream=True,
    on_progress=on_progress,
)
```

### Follow-up Questions

```python
from skell_e_router import ask_deep_research, deep_research_follow_up

# Initial research
result = ask_deep_research("Research quantum computing advances in 2024")

# Ask follow-up about the completed research
clarification = deep_research_follow_up(
    previous_interaction_id=result.id,
    query="Summarize the key challenges in 3 bullet points"
)
```

### Formatted Output

You can steer the agent's output by providing formatting instructions:

```python
prompt = """
Research the competitive landscape of EV batteries.

Format the output as a technical report with:
1. Executive Summary
2. Key Players (include a comparison table)
3. Technology Trends
4. Future Outlook
"""

result = ask_deep_research(prompt, verbosity="info")
```

### With File Search (Experimental)

Combine web search with your own data:

```python
result = ask_deep_research(
    "Compare our Q3 report against public market trends",
    tools=[{
        "type": "file_search",
        "file_search_store_names": ["fileSearchStores/my-store"]
    }]
)
```

### DeepResearchResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Interaction ID (use for follow-ups) |
| `status` | str | "completed" or "failed" |
| `text` | str | The final research report |
| `citations` | list | Sources cited in the report |
| `usage` | DeepResearchUsage | Token counts and cost info |
| `duration_seconds` | float | Total research time |
| `outputs` | list | All output objects from the API |
| `raw_interaction` | Any | Raw API response for advanced use |

### Error Codes

| Code | Description |
|------|-------------|
| `TIMEOUT` | Research exceeded timeout limit |
| `RESEARCH_FAILED` | The research task failed |
| `STREAM_ERROR` | Streaming connection error |
| `MISSING_API_KEY` | GEMINI_API_KEY not set |
| `MISSING_DEPENDENCY` | google-genai package not installed |

---

## Google Search Grounding (Gemini 2.0+)

For standard `ask_ai()` calls with Gemini models, you can enable Google Search Grounding to let the model search the web for real-time information. This is different from Deep Research - it's a quick, single-query search capability.

```python
from skell_e_router import ask_ai

response = ask_ai(
    "gemini-3-pro-preview",
    "What is the latest news on the James Webb Telescope?",
    verbosity="response",
    # Enable Google Search Grounding
    web_search_options={"search_context_size": "high"}  # Options: "low", "medium", "high"
)
```

**Key points:**
- Works with Gemini 2.0+ models (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-pro-preview`, etc.)
- Uses `web_search_options` parameter with `search_context_size` to control search depth
- Options for `search_context_size`: `"low"`, `"medium"`, `"high"`
- Returns grounding metadata with citations in the response
- This is a server-side capability, not client-side function calling
- Can be combined with `reasoning_effort` for thinking models

---

## Setting Default Python Encoding to UTF-8 on Windows

This project requires Python to use UTF-8 as its default file encoding. If you encounter `UnicodeDecodeError` errors when running Python scripts (especially related to reading configuration or data files), follow these steps:

1.  **Open PowerShell as Administrator:**
    *   Search for "PowerShell" in the Start menu.
    *   Right-click "Windows PowerShell" and select "Run as administrator".

2.  **Run the following command:**
    This command tells Windows to set the `PYTHONUTF8` environment variable to `1` for your user account, making Python default to UTF-8.

    ```powershell
    [System.Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'User')
    ```

3.  **Restart Your Terminal/Computer:**
    *   Close and reopen any open PowerShell or Command Prompt windows.
    *   For the change to be fully recognized by all applications, it's sometimes necessary to log out and log back in, or even restart your computer.

After completing these steps, Python should correctly interpret files using UTF-8 encoding by default. 

## Retry policy

The router has an internal retry up to 3 times before sending the response.

- Retries: only on network/timeout errors, HTTP 500/502/503/504, and 429 without quota/billing exhaustion.
- No retry: bad params (4xx), auth/permission errors, not found, quota/billing 429, policy blocks.
- Backoff: exponential with jitter; if `Retry-After` is present on 429/503, it is honored up to a maximum of 120 seconds. If `Retry-After` exceeds 120 seconds, no retry is attempted and the error is returned.

## Verbosity Settings

- **none**: No output
- **response**: Response content
- **info**: Response content + response info/stats (Model, Finish Reason, Cost, Speed, Prompt Tokens, Completion Tokens, Reasoning Tokens, Total Tokens, Tool Calls, Function Call, Provider Specific Fields, Safety ratings if applicable)
- **debug**: Request details (kwargs, messages) + response content + response info/stats + raw response

## Groq Compound (temporary header)

When routing to Groq Compound models (`groq/groq/compound`, `groq/groq/compound-mini`), the router injects a request header `Groq-Model-Version: latest`. This selects the Compound profile that exposes built‑in tools like `visit_website`.

- This is a temporary shim for LiteLLM. Once LiteLLM forwards this header by default for Groq Compound, remove the injection in `skell_e_router/utils.py` in the `_handle_model_specific_params` function and the extra header params in `skell_e_router/model_config.py`.
- If you prefer pinning to a specific profile, change the injected header value from `latest` to the desired version.