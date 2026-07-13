# MODEL CONFIGURATION
#--------------------

class AIModel:
    def __init__(self, name: str, provider: str, supports_thinking: bool, supported_params: set[str], accepted_reasoning_efforts: set[str] | None = None, use_direct_sdk: bool = False):
        self.name = name  # Full model name used by LiteLLM
        self.provider = provider # e.g., "gemini", "openai", "anthropic"
        self.supports_thinking = supports_thinking # True if model supports 'thinking' or 'reasoning_effort'
        self.supported_params = supported_params # Parameters supported by litellm.completion for this model, after our internal transformations
        self.accepted_reasoning_efforts = accepted_reasoning_efforts # Optional per-model allowed values for 'reasoning_effort'
        self.use_direct_sdk = use_direct_sdk # True to bypass LiteLLM and call provider SDK directly

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

    @property
    def is_deepinfra(self) -> bool:
        return self.provider == "deepinfra"


# Models are sorted by provider, then by latest models on top
MODEL_CONFIG = {

    # OPENAI

    # gpt-5.6 family (Sol=flagship, Terra=mid, Luna=cheap/fast), preview launched July 2026.
    # 1M context, 128K max output. Same effort vocabulary as 5.5 (none/low/medium/high/xhigh).
    # temperature is rejected (only default 1 supported), same as 5.5.
    "gpt-5.6-sol": AIModel(
        name="openai/gpt-5.6-sol",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.6-terra": AIModel(
        name="openai/gpt-5.6-terra",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.6-luna": AIModel(
        name="openai/gpt-5.6-luna",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    # gpt-5.5 uses a different effort vocabulary than 5.4/5.3/5: no "minimal", new "xhigh".
    "gpt-5.5": AIModel(
        name="openai/gpt-5.5",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.4-mini": AIModel(
        name="openai/gpt-5.4-mini",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5.4-nano": AIModel(
        name="openai/gpt-5.4-nano",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5.3-codex": AIModel(
        name="openai/gpt-5.3-codex",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.3-chat": AIModel(
        name="openai/gpt-5.3-chat-latest",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "max_tokens", "stream"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
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

    # gemini-3.5-flash: GA May 2026. 1M context, 65K max output. thinking_level default is
    # "medium" (was "high" on Gemini 3). Google recommends leaving temperature/top_p/top_k at defaults.
    "gemini-3.5-flash": AIModel(
        name="gemini/gemini-3.5-flash",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        use_direct_sdk=True,
    ),
    "gemini-3-flash-preview": AIModel(
        name="gemini/gemini-3-flash-preview",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        use_direct_sdk=True,
    ),
    # "gemini-3-pro-preview" is aliased to "gemini-3.1-pro-preview" below (Gemini 3 Pro discontinued March 9, 2026)
    "gemini-3.1-pro-preview": AIModel(
        name="gemini/gemini-3.1-pro-preview",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"low", "medium", "high"},
    ),
    "gemini-3.1-flash-lite": AIModel(
        name="gemini/gemini-3.1-flash-lite",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        use_direct_sdk=True,
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
        supported_params={"temperature", "max_tokens", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        use_direct_sdk=True,
    ),
    "gemini-2.5-flash-lite": AIModel(
        name="gemini/gemini-2.5-flash-lite",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        use_direct_sdk=True,
    ),

    # ANTHROPIC

    # Fable 5: adaptive thinking is always on (thinking "disabled" is rejected).
    # No temperature/top_p/top_k. 1M context, 128k max output. Safety classifiers
    # may decline requests (stop_reason "refusal" on a 200 response).
    "claude-fable-5": AIModel(
        name="anthropic/claude-fable-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        use_direct_sdk=True,
    ),
    # Opus 4.8: same API surface as Opus 4.7 (adaptive thinking only, no temperature).
    "claude-opus-4-8": AIModel(
        name="anthropic/claude-opus-4-8",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        use_direct_sdk=True,
    ),
    # Opus 4.7 removes temperature/top_p/top_k and budget_tokens; only adaptive thinking is supported.
    "claude-opus-4-7": AIModel(
        name="anthropic/claude-opus-4-7",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh"},
        use_direct_sdk=True,
    ),
    "claude-opus-4-6": AIModel(
        name="anthropic/claude-opus-4-6",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "max"},
        use_direct_sdk=True,
    ),
    # Sonnet 5: same API surface as Opus 4.8 — adaptive thinking only (budget_tokens
    # removed; thinking "disabled" is still accepted), no temperature/top_p/top_k.
    # 1M context, 128k max output. Effort defaults to high.
    "claude-sonnet-5": AIModel(
        name="anthropic/claude-sonnet-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        use_direct_sdk=True,
    ),
    "claude-sonnet-4-6": AIModel(
        name="anthropic/claude-sonnet-4-6",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high"},
        use_direct_sdk=True,
    ),
    "claude-opus-4-5": AIModel(
        name="anthropic/claude-opus-4-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
    ),
    "claude-haiku-4-5": AIModel(
        name="anthropic/claude-haiku-4-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
    ),
    "claude-sonnet-4-5-20250929": AIModel(
        name="anthropic/claude-sonnet-4-5-20250929",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
    ),
    "claude-opus-4-1-20250805": AIModel(
        name="anthropic/claude-opus-4-1-20250805",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
    ),
    "claude-sonnet-4-20250514": AIModel(
        name="anthropic/claude-sonnet-4-20250514",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
    ),
    "claude-3-7-sonnet-20250219": AIModel(
        name="anthropic/claude-3-7-sonnet-20250219",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
        # betas param such as betas=["output-128k-2025-02-19"] for 128K output tokens (much longer responses)
    ),
    "claude-3-5-sonnet-20241022": AIModel(
        name="anthropic/claude-3-5-sonnet-20241022",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "stream", "tools", "tool_choice"},
        use_direct_sdk=True,
    ),

    # XAI

    # grok-4.5: reasoning always on with configurable reasoning_effort (low/medium/high, default high).
    # 500K context window. Vision input supported. stop is rejected by xAI reasoning models.
    "grok-4.5": AIModel(
        name="xai/grok-4.5",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"low", "medium", "high"},
    ),
    # grok-4.20: reasoning is always on. reasoning_effort/stop/frequency_penalty/presence_penalty
    # are rejected by the API. 2M context window.
    "grok-4.20": AIModel(
        name="xai/grok-4.20",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    # grok-4.20-non-reasoning: thinking off, supports stop and the usual sampling params.
    "grok-4.20-non-reasoning": AIModel(
        name="xai/grok-4.20-non-reasoning",
        provider="xai",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "grok-4-0220": AIModel(
        name="xai/grok-4-0220",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "max_tokens", "stream", "tools", "tool_choice"},
    ),
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
    "grok-code-fast-1": AIModel(
        name="xai/grok-code-fast-1",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
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
    "qwen3-32b": AIModel(
        name="groq/qwen/qwen3-32b",
        provider="groq",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "max_completion_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"none", "default", "low", "medium", "high"},
    ),
    "kimi-k2-0905": AIModel(
        name="groq/moonshotai/kimi-k2-instruct-0905",
        provider="groq",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "max_completion_tokens", "stream", "tools", "tool_choice"},
    ),

    # DEEPINFRA (NVIDIA NEMOTRON)

    "nemotron-3-super": AIModel(
        name="deepinfra/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
        provider="deepinfra",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "nemotron-super-49b": AIModel(
        name="deepinfra/nvidia/Llama-3.3-Nemotron-Super-49B-v1.5",
        provider="deepinfra",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "nemotron-70b": AIModel(
        name="deepinfra/nvidia/Llama-3.1-Nemotron-70B-Instruct",
        provider="deepinfra",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "nemotron-3-nano-30b": AIModel(
        name="deepinfra/nvidia/Nemotron-3-Nano-30B-A3B",
        provider="deepinfra",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "nemotron-nano-12b-vl": AIModel(
        name="deepinfra/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL",
        provider="deepinfra",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),
    "nemotron-nano-9b": AIModel(
        name="deepinfra/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        provider="deepinfra",
        supports_thinking=False,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
    ),

}

# Additional aliases
MODEL_CONFIG["gemini-3-pro-preview"] = MODEL_CONFIG["gemini-3.1-pro-preview"]  # Gemini 3 Pro discontinued March 9, 2026
MODEL_CONFIG["gemini-3.1-flash-lite-preview"] = MODEL_CONFIG["gemini-3.1-flash-lite"]  # Renamed from preview on GA
MODEL_CONFIG["gemini-3-pro-image"] = MODEL_CONFIG["nano-banana-3"]

# Allow lookup by full name too
for config in list(MODEL_CONFIG.values()): # Iterate over a copy if modifying during iteration (though here it's safe)
    if config.name not in MODEL_CONFIG:
        MODEL_CONFIG[config.name] = config


# ============================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================


class EmbeddingModel:
    """Registry entry for an embedding model. Distinct from chat AIModel."""

    def __init__(
        self,
        name: str,                                  # full LiteLLM identifier
        provider: str,                              # "openai" | "gemini"
        supported_inputs: set[str],                 # subset of {"text","image","audio","video","pdf"}
        max_dimensions: int,
        default_dimensions: int,
        recommended_dimensions: tuple[int, ...] = (),
        max_input_tokens: int | None = None,
        supports_aggregation: bool = False,
    ):
        self.name = name
        self.provider = provider
        self.supported_inputs = supported_inputs
        self.max_dimensions = max_dimensions
        self.default_dimensions = default_dimensions
        self.recommended_dimensions = recommended_dimensions
        self.max_input_tokens = max_input_tokens
        self.supports_aggregation = supports_aggregation

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @property
    def is_gemini(self) -> bool:
        return self.provider == "gemini"


EMBEDDING_MODEL_CONFIG: dict[str, EmbeddingModel] = {
    "openai-embedding-3-large": EmbeddingModel(
        name="openai/text-embedding-3-large",
        provider="openai",
        supported_inputs={"text"},
        max_dimensions=3072,
        default_dimensions=3072,
        recommended_dimensions=(256, 1024, 3072),
        max_input_tokens=8192,
        supports_aggregation=False,
    ),
    "openai-embedding-3-small": EmbeddingModel(
        name="openai/text-embedding-3-small",
        provider="openai",
        supported_inputs={"text"},
        max_dimensions=1536,
        default_dimensions=1536,
        recommended_dimensions=(512, 1536),
        max_input_tokens=8192,
        supports_aggregation=False,
    ),
    "gemini-embedding-2": EmbeddingModel(
        name="gemini/gemini-embedding-2",
        provider="gemini",
        supported_inputs={"text", "image", "audio", "video", "pdf"},
        max_dimensions=3072,
        default_dimensions=3072,
        recommended_dimensions=(768, 1536, 3072),
        max_input_tokens=8192,
        supports_aggregation=True,
    ),
}

# Allow lookup by full LiteLLM name in addition to alias.
for _cfg in list(EMBEDDING_MODEL_CONFIG.values()):
    if _cfg.name not in EMBEDDING_MODEL_CONFIG:
        EMBEDDING_MODEL_CONFIG[_cfg.name] = _cfg


def resolve_embedding_alias(model_alias: str) -> EmbeddingModel:
    """Resolve an embedding model alias (or full LiteLLM name) to its EmbeddingModel."""
    # Local import avoids a circular reference (utils.py imports model_config at module load).
    from .utils import RouterError

    model = EMBEDDING_MODEL_CONFIG.get(model_alias)
    if not model:
        # Show only the short aliases (full-name duplicates would clutter the message).
        available = sorted(k for k in EMBEDDING_MODEL_CONFIG if "/" not in k)
        raise RouterError(
            code="INVALID_MODEL",
            message=f"Invalid embedding model alias '{model_alias}'. Available: {available}",
        )
    return model
