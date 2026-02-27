# skell-e-router — Technical Reference

Full technical documentation for the `skell_e_router` package. For installation and quick start, see the [root README](../README.md).

## API Keys

By default the router reads API keys from environment variables:

```
OPENAI_API_KEY
GEMINI_API_KEY
ANTHROPIC_API_KEY
GROQ_API_KEY
XAI_API_KEY
```

Only the key for the provider you're actually calling is required — you don't need all five set.

### Passing Keys via `config`

Instead of (or in addition to) environment variables, you can pass keys directly with a `config` dictionary. This keeps the router generic so each consumer (Obsidian plugins, CLI scripts, web apps, etc.) can source its own keys however it wants.

```python
from skell_e_router import ask_ai

response = ask_ai(
    "gpt-5",
    "Explain quantum computing",
    config={"openai_api_key": "sk-..."},
)
```

```python
from skell_e_router import ask_deep_research

result = ask_deep_research(
    "Research EV battery trends",
    config={"gemini_api_key": "AIza..."},
)
```

When `config` is provided, those keys are used directly instead of reading from environment variables. When `config` is `None` (the default), the existing env var behavior applies — nothing changes for existing callers.

**Config key names** are the lowercase form of the environment variable:

| Config Key | Environment Variable |
|---|---|
| `openai_api_key` | `OPENAI_API_KEY` |
| `gemini_api_key` | `GEMINI_API_KEY` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| `groq_api_key` | `GROQ_API_KEY` |
| `xai_api_key` | `XAI_API_KEY` |

Keys passed via `config` are never logged or included in error messages.

---

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
| `images` | list[dict] | Generated images (see Image Output below) |
| `tool_calls` | list | Function calls made |
| `raw_response` | Any | Full LiteLLM response |

---

## Image Input (Vision)

The `images` parameter on `ask_ai()` lets you send images alongside a text prompt. It works with any vision-capable model across all providers.

### Supported Image Sources

Each item in the `images` list is a string. The router detects the type automatically:

| Format | Example | Behavior |
|--------|---------|----------|
| **Local file path** | `"photo.jpg"` | Read from disk, detect MIME type, base64-encode |
| **URL** | `"https://example.com/img.png"` | Passed through to the provider as-is |
| **Base64 data URI** | `"data:image/png;base64,iVBOR..."` | Passed through to the provider as-is |

### Usage

```python
from skell_e_router import ask_ai

# Single image from a local file
response = ask_ai(
    "gemini-3-pro-preview",
    "Describe what you see in this image",
    images=["path/to/photo.jpg"],
)

# Multiple images (file + URL)
response = ask_ai(
    "gpt-4o",
    "Compare these two images",
    images=["local.png", "https://example.com/remote.jpg"],
)

# With rich response for full metadata
response = ask_ai(
    "claude-opus-4-5",
    "What does this diagram show?",
    images=["diagram.png"],
    rich_response=True,
)
print(f"Tokens: {response.prompt_tokens} + {response.completion_tokens}")
```

### How It Works

When `images` is provided with a string `user_input`, the router constructs a multimodal message in OpenAI's content-parts format:

```python
# What the router builds internally:
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
}
```

LiteLLM translates this to the correct provider-specific format for Gemini, OpenAI, Anthropic, etc.

### Constraints

- The `images` parameter only works when `user_input` is a **string**. If you're passing conversation history as a `list[dict]`, embed image content parts directly in your messages.
- If the file path doesn't exist, a `RouterError` with code `INVALID_INPUT` is raised.
- An empty list (`images=[]`) or `images=None` has no effect — the message is constructed as plain text.

---

## Image Output (Generation)

The `nano-banana-3` model (alias: `gemini-3-pro-image`) can generate images. Generated images are returned on the `AIResponse.images` field when using `rich_response=True`.

### Usage

```python
import base64
from skell_e_router import ask_ai

response = ask_ai(
    "nano-banana-3",
    "Generate a charcoal drawing of a skull",
    rich_response=True,
)

# response.images is a list of image dicts (or None if no images)
if response.images:
    for i, img in enumerate(response.images):
        data_url = img["image_url"]["url"]  # "data:image/jpeg;base64,..."
        header, encoded = data_url.split(",", 1)
        with open(f"output_{i}.jpg", "wb") as f:
            f.write(base64.b64decode(encoded))
```

### Combined Input + Output

You can send a reference image and ask the model to generate a new image based on it:

```python
response = ask_ai(
    "nano-banana-3",
    "Read the text in this image and create an illustration of what it says",
    images=["reference.jpg"],
    rich_response=True,
)
```

### Image Response Format

Each item in `response.images` is a dict with this structure:

```python
{
    "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZ...",
        "detail": "auto"
    },
    "index": 0,
    "type": "image_url"
}
```

### How It Works

- The `nano-banana-3` model has `modalities` in its `supported_params`. The router auto-injects `modalities=["text", "image"]` to tell LiteLLM to request image output.
- You can override this by passing `modalities=["text"]` if you only want text from this model.
- When `rich_response=False` (the default), `ask_ai()` returns just the text content string. Images are only accessible via the `AIResponse` object with `rich_response=True`.
- The model aliases `"nano-banana-3"` and `"gemini-3-pro-image"` both point to `gemini/gemini-3-pro-image-preview`.

---

## Gemini Deep Research Agent

The Gemini Deep Research Agent autonomously plans, executes, and synthesizes multi-step research tasks. It navigates complex information landscapes using web search to produce detailed, cited reports.

**Key characteristics:**
- Research tasks take **several minutes** (up to 60 min max)
- Uses Google's Interactions API (not LiteLLM)
- Requires `GEMINI_API_KEY` (via environment variable or `config`)

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

    print(result.text)         # The research report (with resolved URLs)
    print(result.id)           # Interaction ID for follow-ups

    # Parsed citations with titles and permanent URLs
    for cit in result.parsed_citations:
        print(f"[{cit.number}] {cit.title}: {cit.url}")

    if result.usage:
        print(f"Tokens: {result.usage.total_tokens}")

except DeepResearchError as err:
    print(f"Error: {err.code} - {err.message}")
```

### Streaming with Progress Updates

```python
from skell_e_router import ask_deep_research

def on_progress(event_type: str, content: str):
    if event_type == "start":
        print(f"Research started: {content}")  # content is interaction_id
    elif event_type == "thought":
        print(f"[Thinking] {content}")
    elif event_type == "text":
        print(content, end="", flush=True)

result = ask_deep_research(
    "Compare AWS, Azure, and GCP for ML workloads",
    stream=True,
    on_progress=on_progress,
)
```

**Note:** The router automatically handles transient errors like `gateway_timeout` during streaming. If Google's API times out mid-research, the router reconnects and continues from where it left off - no action required from your code.

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
| `text` | str | The final research report (with resolved citation URLs) |
| `text_without_sources` | str | Report text without the Sources section |
| `parsed_citations` | list[ParsedCitation] | Structured citations with resolved URLs and titles |
| `citations` | list | Raw citations from the API |
| `usage` | DeepResearchUsage | Token counts (currently None - see note below) |
| `duration_seconds` | float | Total research time |
| `outputs` | list | All output objects from the API |
| `raw_interaction` | Any | Raw API response for advanced use |

> **Note:** The Deep Research Interactions API does not currently provide token usage data. The `usage` field will be `None`. This differs from the standard `generate_content` API. If Google adds usage tracking in the future, it will be automatically populated.

### Citation Processing

By default, `ask_deep_research()` automatically processes citations:
- Extracts citations from the Sources section
- Resolves temporary `vertexaisearch` redirect URLs to permanent URLs
- Fetches page titles for better context
- Rebuilds the Sources section with resolved links

```python
from skell_e_router import ask_deep_research

result = ask_deep_research("Research quantum computing advances")

# Access parsed citations
for cit in result.parsed_citations:
    print(f"[{cit.number}] {cit.title or cit.domain}")
    print(f"    URL: {cit.url}")

# Get report without sources (useful for custom formatting)
print(result.text_without_sources)
```

**ParsedCitation Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `number` | int | Citation number in the report |
| `domain` | str | Source domain (e.g., "wikipedia.org") |
| `url` | str | Resolved permanent URL |
| `title` | str | Page title (falls back to domain if unavailable) |
| `redirect_url` | str | Original vertexaisearch redirect URL |

**Disable citation resolution** if you want faster results (skips URL resolution):

```python
result = ask_deep_research("My query", resolve_citations=False)
```

### Converting Results to JSON

Use the `.to_dict()` method to serialize results for APIs or storage:

```python
from skell_e_router import ask_deep_research
import json

result = ask_deep_research("Research topic")
data = result.to_dict()
json_output = json.dumps(data, indent=2)
```

### Automatic Reconnection

The router automatically handles transient streaming errors (like `gateway_timeout`, `connection_reset`, `unavailable`). When these occur mid-research, it reconnects and continues from where it left off. This means your streaming code doesn't need special error handling for network interruptions.

### Error Handling

| Code | Description |
|------|-------------|
| `TIMEOUT` | Research exceeded timeout limit |
| `RESEARCH_FAILED` | The research task failed |
| `STREAM_FAILED` | Stream failed after exhausting reconnection attempts |
| `STREAM_ERROR` | Non-retryable streaming error |
| `PROVIDER_ERROR` | API error from Gemini (may be transient) |
| `MISSING_API_KEY` | GEMINI_API_KEY not set (via env or `config`) |
| `MISSING_DEPENDENCY` | google-genai package not installed |

**Note:** `PROVIDER_ERROR` can occur occasionally due to transient API issues. Implement retry logic if needed:

```python
from skell_e_router import ask_deep_research, DeepResearchError
import time

def research_with_retry(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return ask_deep_research(query)
        except DeepResearchError as e:
            if e.code == "PROVIDER_ERROR" and attempt < max_retries - 1:
                print(f"Provider error, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(5)
            else:
                raise
```

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

## Retry Policy

The router has an internal retry up to 3 times before sending the response.

- Retries: only on network/timeout errors, HTTP 500/502/503/504, and 429 without quota/billing exhaustion.
- No retry: bad params (4xx), auth/permission errors, not found, quota/billing 429, policy blocks.
- Backoff: exponential with jitter; if `Retry-After` is present on 429/503, it is honored up to a maximum of 120 seconds. If `Retry-After` exceeds 120 seconds, no retry is attempted and the error is returned.

---

## Verbosity Settings

- **none**: No output
- **response**: Response content
- **info**: Response content + response info/stats (Model, Finish Reason, Cost, Speed, Prompt Tokens, Completion Tokens, Reasoning Tokens, Total Tokens, Tool Calls, Function Call, Provider Specific Fields, Safety ratings if applicable)
- **debug**: Request details (kwargs, messages) + response content + response info/stats + raw response

---

## Groq Models

The router supports several models hosted on Groq's fast inference infrastructure.

### Available Models

| Alias | Model | Thinking | Notes |
|---|---|---|---|
| `groq-compound` | `groq/groq/compound` | Yes | Agentic model with built-in tools (web search, code interpreter, etc.) |
| `groq-compound-mini` | `groq/groq/compound-mini` | Yes | Lighter version of Compound |
| `qwen3-32b` | `groq/qwen/qwen3-32b` | Yes | Qwen 3 32B with toggleable thinking mode |
| `kimi-k2-0905` | `groq/moonshotai/kimi-k2-instruct-0905` | No | Moonshot Kimi K2, fast general-purpose model |

### Disabling Thinking on Qwen3-32B

Qwen3-32B runs in thinking mode by default (responses include `<think>` blocks). You can disable this with `reasoning_effort="none"`:

```python
from skell_e_router import ask_ai

# Default — thinking enabled
response = ask_ai("qwen3-32b", "Explain neural networks")

# Thinking disabled — faster, no <think> block
response = ask_ai("qwen3-32b", "Explain neural networks", reasoning_effort="none")
```

Accepted values for `reasoning_effort` on this model: `"none"`, `"default"`.

**Note:** LiteLLM does not natively recognize `reasoning_effort` for Groq models. The router works around this by injecting `allowed_openai_params` to force the parameter through.

### Groq Compound (temporary header)

When routing to Groq Compound models, the router injects a request header `Groq-Model-Version: latest`. This selects the Compound profile that exposes built-in tools like `visit_website`.

- This is a temporary shim for LiteLLM. Once LiteLLM forwards this header by default for Groq Compound, remove the injection in `skell_e_router/utils.py` in the `_handle_model_specific_params` function and the extra header params in `skell_e_router/model_config.py`.
- If you prefer pinning to a specific profile, change the injected header value from `latest` to the desired version.

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
