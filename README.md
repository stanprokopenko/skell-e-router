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

For the full technical reference (rich responses, streaming, citations, retry policy, verbosity settings, etc.), see [skell_e_router/README.md](skell_e_router/README.md).
