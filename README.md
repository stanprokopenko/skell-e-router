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
```

Only the one you need is required — you don't need all five.

> **Groq models** (`groq-compound`, `qwen3-32b`, `kimi-k2-0905`, etc.) use the `GROQ_API_KEY`.

You can also pass keys directly so your code doesn't depend on environment variables:

```python
response = ask_ai(
    "gpt-5",
    "Explain quantum computing",
    config={"openai_api_key": "sk-..."},
)
```

## Documentation

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

### Anthropic Betas

Pass beta feature flags to Claude models:

```python
response = ask_ai(
    "claude-3-7-sonnet-20250219", "Write a long essay",
    betas=["output-128k-2025-02-19"]
)
```
