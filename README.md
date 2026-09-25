# skell-e-router

Simple AI router using LiteLLM with Gemini Deep Research Agent support.

## Install

```bash
pip install git+https://github.com/stanprokopenko/skell-e-router@main
```

## Quick Start

### AI Completion

```python
from skell_e_router import ask_ai

response = ask_ai(
    "gemini-2.5-pro",
    "Explain quantum computing in simple terms",
    verbosity="response"
)
```

### Output token limits

Pass `max_tokens=600` to cap generated tokens on OpenAI reasoning models, including `gpt-5.6-luna`. The router forwards the provider's output-limit field. The cap includes reasoning tokens, so the visible answer may be shorter or empty. See the [parameter contract](skell_e_router/Skell-E-Router-DOCUMENTATION.md#output-token-limits) for aliases and validation rules.

### Classification with Jev

Direct TypeSafe support with offline contract tests and authenticated verification. Jev returns typed decisions rather than text, so use `classify()` instead of `ask_ai()`. See the comparison report below for measured performance and its limits.

```python
from skell_e_router import classify

result = classify("jev", "I was charged twice.", {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {"billing": "Payments or refunds", "technical": "Bugs or outages"},
    }
})
print(result.answers["team"]["choice"])
print(result.answers["team"]["probabilities"])
```

Set `TYPESAFE_API_KEY`, or pass `config={"typesafe_api_key": key}`. The `jev` alias pins `jev-1.13.0`; `jev-latest` follows TypeSafe upgrades. See the [classification reference](skell_e_router/Skell-E-Router-DOCUMENTATION.md#classification-with-jev) and [research and comparison status](docs/jev-classification.md).

### Image Input (Vision)

Send an image alongside your prompt. You can pass a local file path, a URL, or a base64 data URI — the router handles encoding for you.

```python
from skell_e_router import ask_ai

response = ask_ai(
    "gemini-3-pro-preview",
    "What does this image say?",
    images=["path/to/photo.jpg"],
)
print(response)
```

This works with any vision-capable model across providers (Gemini, GPT-4o, Claude, etc.).

### Audio Input

Send an audio clip alongside your prompt. Same accepted forms as `images` — a local file path or a base64 `data:audio/...` URI.

```python
from skell_e_router import ask_ai

response = ask_ai(
    "gemini-3-pro-preview",
    "Transcribe and summarize this clip.",
    audio=["interview.mp3"],
)
```

Supported on Gemini 2.5+/3.x and OpenAI GPT-4o audio models. Anthropic models raise `RouterError("UNSUPPORTED_MODALITY")`.

**Format support**: `mp3` and `wav` work everywhere. `flac`, `ogg`, `m4a/mp4`, and `webm` are Gemini-only — OpenAI's `input_audio` spec accepts only `mp3` and `wav`, so passing other formats to a GPT-4o audio model returns a 400 from the provider.

For audio files larger than ~20 MB, use `upload_file()` and pass via `files=[ref]` instead.

### Image Generation

Generate images using the `nano-banana-3` model. Use `rich_response=True` to access the generated image data.

```python
import base64
from skell_e_router import ask_ai

response = ask_ai(
    "nano-banana-3",
    "Generate a watercolor painting of a sunset over the ocean",
    rich_response=True,
)

# Save the generated image
if response.images:
    data_url = response.images[0]["image_url"]["url"]
    header, encoded = data_url.split(",", 1)
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(encoded))
```

You can also combine image input and output — send a reference image and ask the model to generate something based on it:

```python
response = ask_ai(
    "nano-banana-3",
    "Create a painting inspired by this photo",
    images=["reference.jpg"],
    rich_response=True,
)
```

For standalone image files, `generate_image()` hands back decoded bytes plus a `save()` helper. One call covers OpenAI GPT-Image, ByteDance Seedream on DeepInfra, and Gemini.

```python
from skell_e_router import generate_image

resp = generate_image("gpt-image", "a simple red circle on white", quality="low")
resp.save("circle.png")
print(resp.cost, resp.input_tokens, resp.output_tokens)

# Same call, cheaper provider — Seedream bills a flat $0.04 per image
resp = generate_image("seedream-4.5", "a simple red circle on white", size="2048x2048")
resp.save("circle.jpg")   # returns JPEG; resp.format says what the bytes really are

# Pass reference images to edit instead of generate
resp = generate_image(
    "gpt-image-2.5-sunburst",
    "put this character on a beach at sunset",
    images=["character.png"],
    size="1536x1024",
    quality="high",
    n=2,
)
resp.save("out/", stem="beach")  # ['out/beach_0.png', 'out/beach_1.png']
```

Available models: `gpt-image` / `gpt-image-2.5` (both the fast `gpt-image-2.5-flare`), `gpt-image-2.5-sunburst`, `gpt-image-2`, `seedream-4.5`, `seedream-4`, `seedream-5-pro`, and `nano-banana-3` / `gemini-3-pro-image` / `nano-banana-pro`.

`quality`, `background`, `output_format` and `output_compression` are GPT-Image only. Setting one on Seedream or Gemini raises `RouterError("INVALID_PARAM")` instead of silently ignoring it; the defaults pass everywhere. See [the technical reference](skell_e_router/Skell-E-Router-DOCUMENTATION.md#image-generation-generate_image) for sizes, quality tiers, pricing, and error codes.

### Embeddings

Route embedding calls through skell-e-router with `get_embedding()`. Supports OpenAI text embeddings and Gemini multimodal embeddings.

```python
from skell_e_router import get_embedding

# Single text → list[float]
v = get_embedding("openai-embedding-3-large", "hello world")

# Batch → list[list[float]]
vs = get_embedding(
    "openai-embedding-3-large",
    ["doc 1", "doc 2", "doc 3"],
)

# Multimodal aggregation (Gemini): text + image → one fused embedding
v = get_embedding(
    "gemini-embedding-2",
    [["a red shoe on wood", "shoe.jpg"]],
)
```

Available models: `openai-embedding-3-large`, `openai-embedding-3-small`, `gemini-embedding-2`. See [the technical reference](skell_e_router/Skell-E-Router-DOCUMENTATION.md#embeddings) for the capability matrix, input shape rules, and error codes.

### Deep Research

```python
from skell_e_router import ask_deep_research

result = ask_deep_research(
    "Research the competitive landscape of EV batteries",
    verbosity="info"
)
print(result.text)
```

## API Keys

Set the environment variable for the provider you're calling:

```
OPENAI_API_KEY
GEMINI_API_KEY
ANTHROPIC_API_KEY
GROQ_API_KEY
XAI_API_KEY
OPENROUTER_API_KEY
```

Only the one you need is required — you don't need them all.

> **OpenRouter models** (`glm-5.3-flash`) use the `OPENROUTER_API_KEY`.

> **Groq-hosted models** (`gpt-oss-120b`, `gpt-oss-20b`) use the `GROQ_API_KEY`.

You can also pass keys directly so your code doesn't depend on environment variables:

```python
response = ask_ai(
    "gpt-5",
    "Explain quantum computing",
    config={"openai_api_key": "sk-..."},
)
```

## Documentation

Version 3.31.1 derives `__version__` from the installed distribution metadata (`importlib.metadata`) instead of a hardcoded literal, which had drifted (3.31.0 reported itself as 3.30.1). Uninstalled source checkouts report `0.0.0+uninstalled`.

Version 3.26.5 defaults direct-SDK Anthropic calls to each model's published output cap (128k for the current Claude family, 64k for haiku 4.5 / opus 4.5 / sonnet 4.5, from the Models API) instead of 4096 when the caller sets no `max_tokens`. Pass `max_tokens` to cap lower.

Version 3.26.4 lets direct-SDK Anthropic calls ask for `max_tokens` above the SDK's ~21k non-streaming guard: the router streams under the hood and returns the assembled response, so long tool-call arguments (30k-token files) no longer need a streaming caller.

Version 3.26.3 protects provider failure diagnostics from credential disclosure, including escaped header values and Python exception chains. Errors keep safe categories and HTTP status numbers. See the [error contract](skell_e_router/Skell-E-Router-DOCUMENTATION.md#provider-error-diagnostics) and [security release record](docs/credential-error-security.md).

### Direct SDK Path

Gemini and Claude models bypass LiteLLM by default, calling the provider SDK directly for lower latency (eliminates 0.3-1.7s overhead). This is controlled per-model via `use_direct_sdk` and can be overridden per-call:

```python
# Force LiteLLM path even for direct-SDK models
response = ask_ai("claude-sonnet-4-6", "Hello", direct_sdk=False)

# Force direct SDK path
response = ask_ai("claude-opus-4-6", "Hello", direct_sdk=True)
```

### Streaming

Gemini and Claude models support streaming via the direct SDK path:

```python
# Gemini streaming
for chunk in ask_ai("gemini-2.5-flash", "Tell me a story", stream=True):
    print(chunk.text, end="", flush=True)

# Claude streaming (returns a context manager)
with ask_ai("claude-sonnet-4-6", "Tell me a story", stream=True) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Function Calling

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
}}]

# Works with both Gemini and Claude models
response = ask_ai(
    "claude-sonnet-4-6", "What's the weather in NYC?",
    tools=tools, tool_choice="auto", rich_response=True
)
print(response.tool_calls)
```

### Reasoning Effort / Thinking

Control thinking depth across providers with `reasoning_effort`:

```python
# Works with Gemini, Claude, and other thinking models
response = ask_ai("gemini-3.1-flash-lite-preview", "Solve this", reasoning_effort="low")
response = ask_ai("claude-opus-4-6", "Analyze this code", reasoning_effort="high")

# Budget tokens (explicit control, Claude & Gemini)
response = ask_ai("claude-sonnet-4-6", "Solve this math problem", budget_tokens=4096)

# Thinking dict (full control, Claude)
response = ask_ai("claude-sonnet-4-6", "Complex task", thinking={"type": "enabled", "budget_tokens": 2048})
```

`gpt-6-astra` automatically uses LiteLLM's Responses API bridge. OpenAI requires the Responses API when Astra combines reasoning with function tools; callers keep using the same `ask_ai` messages, tools, and rich-response interface. The router also uses its official $10/$50 token rates instead of LiteLLM's stale launch-day cost entry.

`gpt-6-sol` and `gpt-6-luna` use the same Responses API bridge, because OpenAI allows function calling on Chat Completions only at `reasoning_effort="none"`. Both accept `none`, `low`, `medium`, `high`, `xhigh`, and `max`; OpenAI defaults to `medium`. Router pricing is authoritative: Sol costs $2 input and $10 output per million tokens, Luna $0.10 and $0.50.

`claude-opus-5-5` uses Anthropic directly, with a 1M token context and a 128,000 token output limit. Set `reasoning_effort` to `low`, `medium`, `high`, `xhigh`, or `max`. Anthropic defaults to `medium` when omitted. Input costs $4 per million tokens, output $20, and cache reads $0.20.

Opus 5.5 always uses adaptive thinking. Do not send `budget_tokens` or a `thinking` dictionary with type `enabled`, type `disabled`, or a token budget. Raw thinking dictionaries pass through, and Anthropic rejects these settings. The router drops `temperature`, `top_p`, and `top_k` and converts forced tool choices to `auto`, so tool execution is never guaranteed. See Anthropic's [Opus 5.5 overview](https://platform.claude.com/docs/en/models/opus-5-5/overview) and [effort reference](https://platform.claude.com/docs/en/build-with-claude/effort).

### Anthropic Betas

Pass beta feature flags to Claude models:

```python
response = ask_ai(
    "claude-sonnet-4-6", "Write a long essay",
    betas=["context-management-2025-06-27"]
)
```
