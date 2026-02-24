# MODEL CONFIGURATION
#--------------------

class AIModel:
    def __init__(self, name: str, provider: str, supports_thinking: bool, supported_params: set[str], accepted_reasoning_efforts: set[str] | None = None):
        self.name = name  # Full model name used by LiteLLM
        self.provider = provider # e.g., "gemini", "openai", "anthropic"
        self.supports_thinking = supports_thinking # True if model supports 'thinking' or 'reasoning_effort'
        self.supported_params = supported_params # Parameters supported by litellm.completion for this model, after our internal transformations
        self.accepted_reasoning_efforts = accepted_reasoning_efforts # Optional per-model allowed values for 'reasoning_effort'

    @property
    def is_gemini(self) -> bool:
        return self.provider == "gemini"

    @property
    def is_anthropic(self) -> bool:
        return self.provider == "anthropic"

    @property
    def is_openai(self) -> bool: # General OpenAI check
        return self.provider == "openai"

    @property
    def is_openai_o_series(self) -> bool: # Specific check for "o" series like o1 and o3
        return self.is_openai and self.name.startswith("openai/o")

    @property
    def is_xai(self) -> bool:
        return self.provider == "xai"
    
    @property
    def is_groq(self) -> bool:
        return self.provider == "groq"


# Models are sorted by provider, then by latest models on top
MODEL_CONFIG = {

    # OPENAI

    "gpt-5.2": AIModel(
        name="openai/gpt-5.2",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5": AIModel(
        name="openai/gpt-5",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    # TODO: add other params for gpt-5 such as verbosity, etc. (implement responses api)
    "gpt-5-mini": AIModel(
        name="openai/gpt-5-mini",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5-nano": AIModel(
        name="openai/gpt-5-nano",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "o3": AIModel(
        name="openai/o3",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "max_tokens", "stream", "tools", "tool_choice"}
    ),
    "o1": AIModel(
        name="openai/o1",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "max_tokens", "stream", "tools", "tool_choice"}
    ),
    "gpt-4.1": AIModel(
        name="openai/gpt-4.1",
        provider="openai",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"}
    ),
    "gpt-4o": AIModel(
        name="openai/gpt-4o",
        provider="openai",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"}
    ),

    "gpt-oss-120b": AIModel(
        name="groq/openai/gpt-oss-120b",
        provider="openai",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"}
    ),
    "gpt-oss-20b": AIModel(
        name="groq/openai/gpt-oss-20b",
        provider="openai",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"}
    ),

    # GEMINI
    # Note: web_search_options enables Google Search Grounding for real-time web search
    # Example: web_search_options={"search_context_size": "high"}  # Options: "low", "medium", "high"

    "gemini-3-flash-preview": AIModel(
        name="gemini/gemini-3-flash-preview",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
    ),
    "gemini-3-pro-preview": AIModel(
        name="gemini/gemini-3-pro-preview",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
    ),
    "gemini-3.1-pro-preview": AIModel(
        name="gemini/gemini-3.1-pro-preview",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
    ),
    "nano-banana-3": AIModel(
        name="gemini/gemini-3-pro-image-preview",
        provider="gemini",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "stream", "candidate_count", "safety_settings", "modalities"},
    ),
    "gemini-2.5-pro": AIModel(
        name="gemini/gemini-2.5-pro",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "max_tokens", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"}
    ),
    "gemini-2.5-flash": AIModel(
        name="gemini/gemini-2.5-flash",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "max_tokens", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"}
    ),
    "gemini-2.5-flash-lite": AIModel(
        name="gemini/gemini-2.5-flash-lite",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"}
    ),

    # ANTHROPIC

    "claude-opus-4-5": AIModel(
        name="anthropic/claude-opus-4-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"}
    ),
    "claude-haiku-4-5": AIModel(
        name="anthropic/claude-haiku-4-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"}
    ),
    "claude-sonnet-4-5-20250929": AIModel(
        name="anthropic/claude-sonnet-4-5-20250929",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"}
    ),
    "claude-opus-4-1-20250805": AIModel(
        name="anthropic/claude-opus-4-1-20250805",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"}
    ),
    "claude-sonnet-4-20250514": AIModel(
        name="anthropic/claude-sonnet-4-20250514",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"}
    ),
    "claude-3-7-sonnet-20250219": AIModel(
        name="anthropic/claude-3-7-sonnet-20250219",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"}
        # betas param such as betas=["output-128k-2025-02-19"] for 128K output tokens (much longer responses)
    ),
    "claude-3-5-sonnet-20241022": AIModel(
        name="anthropic/claude-3-5-sonnet-20241022",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "stream", "tools", "tool_choice"}
    ),

    # XAI

    "grok-4-1-fast-reasoning": AIModel(
        name="xai/grok-4-1-fast-reasoning",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "grok-4-1-fast-non-reasoning": AIModel(
        name="xai/grok-4-1-fast-non-reasoning",
        provider="xai",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "grok-4-0709": AIModel(
        name="xai/grok-4-0709",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "max_tokens", "stream", "tools", "tool_choice"},   # NOTE: It's a reasoning model, but reasoning_effort is NOT SUPPORTED
    ),
    "grok-4-fast-reasoning": AIModel(
        name="xai/grok-4-fast-reasoning",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "grok-4-fast-non-reasoning": AIModel(
        name="xai/grok-4-fast-non-reasoning",
        provider="xai",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "max_tokens", "stream", "tools", "tool_choice"},
    ),

    # GROQ

    "groq-compound": AIModel(
        name="groq/groq/compound",
        provider="groq",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "max_completion_tokens", "stream", "tools", "tool_choice", "compound_custom", "extra_headers", "headers"}
    ),
    "groq-compound-mini": AIModel(
        name="groq/groq/compound-mini",
        provider="groq",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "max_completion_tokens", "stream", "tools", "tool_choice", "compound_custom", "extra_headers", "headers"}
    ),


}

# Additional aliases
MODEL_CONFIG["gemini-3-pro-image"] = MODEL_CONFIG["nano-banana-3"]

# Allow lookup by full name too
for config in list(MODEL_CONFIG.values()): # Iterate over a copy if modifying during iteration (though here it's safe)
    if config.name not in MODEL_CONFIG:
        MODEL_CONFIG[config.name] = config
