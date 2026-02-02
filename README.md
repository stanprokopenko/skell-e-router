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

You can also pass keys directly so your code doesn't depend on environment variables:

```python
response = ask_ai(
    "gpt-5",
    "Explain quantum computing",
    config={"openai_api_key": "sk-..."},
)
```

## Documentation

For the full technical reference (rich responses, image I/O, streaming, citations, retry policy, verbosity settings, etc.), see [skell_e_router/README.md](skell_e_router/README.md).
