# MODEL CONFIGURATION
#--------------------

class AIModel:
    def __init__(self, name: str, provider: str, supports_thinking: bool, supported_params: set[str], accepted_reasoning_efforts: set[str] | None = None, accepted_tool_choices: set[str] | None = None, use_direct_sdk: bool = False, api_base: str | None = None, pricing: dict | None = None, extra_body: dict | None = None, use_responses_api: bool = False, authoritative_pricing: bool = False, max_output_tokens: int | None = None):
        self.name = name  # Full model name used by LiteLLM
        self.provider = provider # e.g., "gemini", "openai", "anthropic"
        self.supports_thinking = supports_thinking # True if model supports 'thinking' or 'reasoning_effort'
        self.supported_params = supported_params # Parameters supported by litellm.completion for this model, after our internal transformations
        self.accepted_reasoning_efforts = accepted_reasoning_efforts # Optional per-model allowed values for 'reasoning_effort'
        self.accepted_tool_choices = accepted_tool_choices # Optional per-model allowed tool_choice values ("named" covers dict/function choices); unsupported values are coerced to "auto". None = all values pass through.
        self.use_direct_sdk = use_direct_sdk # True to bypass LiteLLM and call provider SDK directly
        self.api_base = api_base # Custom endpoint URL for OpenAI-compatible providers LiteLLM doesn't know natively (routed via the generic "openai/" prefix)
        # Router-level USD pricing per 1M tokens, used ONLY as a fallback when
        # litellm.completion_cost() can't price the model (e.g., custom api_base
        # models routed via the generic "openai/" prefix). Keys: "input", "output",
        # optional "cached_input" (rate for tokens reported in
        # usage.prompt_tokens_details.cached_tokens). Set this on every future
        # model that LiteLLM's cost map doesn't cover.
        self.pricing = pricing
        # Default request-body extras merged into every call (caller's extra_body
        # keys win). Used for OpenRouter provider pinning.
        self.extra_body = extra_body
        # Route LiteLLM chat-shaped calls through its Responses API bridge while
        # preserving ask_ai's existing request and response contract.
        self.use_responses_api = use_responses_api
        # Prefer registry pricing over LiteLLM's cost map when launch-day or
        # provider-specific prices are known to be newer than LiteLLM's data.
        self.authoritative_pricing = authoritative_pricing
        # Provider-published output cap (Anthropic Models API `max_tokens`).
        # The direct Anthropic path uses it as the default `max_tokens` when the
        # caller passes none, so a model is never capped below what it can
        # actually write. None keeps the old 4096 fallback (retired models).
        self.max_output_tokens = max_output_tokens

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
    def is_meta(self) -> bool:
        return self.provider == "meta"

    @property
    def is_moonshot(self) -> bool:
        return self.provider == "moonshot"

    @property
    def is_xai(self) -> bool:
        return self.provider == "xai"
    
    @property
    def is_groq(self) -> bool:
        return self.provider == "groq"

    @property
    def is_deepinfra(self) -> bool:
        return self.provider == "deepinfra"

    @property
    def is_openrouter(self) -> bool:
        return self.provider == "openrouter"


# Models are sorted by provider, then by latest models on top
MODEL_CONFIG = {

    # OPENAI

    # GPT-6 Astra launched September 3, 2026. 1,050,000 context, 128,000 max output.
    # Sampling params are rejected; reasoning effort supports low/medium/high/xhigh/max.
    # OpenAI requires the Responses API for Astra tool calling.
    # https://developers.openai.com/api/docs/guides/latest-model
    "gpt-6-astra": AIModel(
        name="openai/gpt-6-astra",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        pricing={"input": 10.00, "cached_input": 1.00, "output": 50.00},
        use_responses_api=True,
        authoritative_pricing=True,
    ),
    # GPT-6 Sol (high-end) and Luna (cheap/fast) launched September 2026. 1,050,000 context, 128,000 max output.
    # Reasoning effort supports none/low/medium/high/xhigh/max (default medium); sampling params are rejected.
    # Chat Completions allows function calling only at effort none, so both route through the Responses API.
    # Standard pricing below; prompts over 272K input tokens bill at 2x input/cache and 1.5x output.
    # https://developers.openai.com/api/docs/models/gpt-6-sol
    # https://developers.openai.com/api/docs/models/gpt-6-luna
    "gpt-6-sol": AIModel(
        name="openai/gpt-6-sol",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh", "max"},
        pricing={"input": 2.00, "cached_input": 0.20, "output": 10.00},
        use_responses_api=True,
        authoritative_pricing=True,
    ),
    "gpt-6-luna": AIModel(
        name="openai/gpt-6-luna",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh", "max"},
        pricing={"input": 0.10, "cached_input": 0.01, "output": 0.50},
        use_responses_api=True,
        authoritative_pricing=True,
    ),
    # gpt-5.6 family (Sol=flagship, Terra=mid, Luna=cheap/fast), preview launched July 2026.
    # 1M context, 128K max output. Same effort vocabulary as 5.5 (none/low/medium/high/xhigh).
    # temperature is rejected (only default 1 supported), same as 5.5.
    "gpt-5.6-sol": AIModel(
        name="openai/gpt-5.6-sol",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.6-terra": AIModel(
        name="openai/gpt-5.6-terra",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.6-luna": AIModel(
        name="openai/gpt-5.6-luna",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    # gpt-5.5 uses a different effort vocabulary than 5.4/5.3/5: no "minimal", new "xhigh".
    "gpt-5.5": AIModel(
        name="openai/gpt-5.5",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"none", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.4-mini": AIModel(
        name="openai/gpt-5.4-mini",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5.4-nano": AIModel(
        name="openai/gpt-5.4-nano",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5.3-codex": AIModel(
        name="openai/gpt-5.3-codex",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high", "xhigh"}
    ),
    "gpt-5.2": AIModel(
        name="openai/gpt-5.2",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    # gpt-5 / gpt-5-mini / gpt-5-nano / o3 are deprecated by OpenAI, shutdown 2026-12-11
    # (replacements gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna / gpt-5.6-sol). o1 shuts down 2026-10-23.
    # https://developers.openai.com/api/docs/deprecations
    "gpt-5": AIModel(
        name="openai/gpt-5",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    # TODO: add other params for gpt-5 such as verbosity, etc. (implement responses api)
    "gpt-5-mini": AIModel(
        name="openai/gpt-5-mini",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "gpt-5-nano": AIModel(
        name="openai/gpt-5-nano",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "stream", "tools", "tool_choice", "max_tokens", "max_completion_tokens"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"}
    ),
    "o3": AIModel(
        name="openai/o3",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "max_tokens", "stream", "tools", "tool_choice", "max_completion_tokens"}
    ),
    "o1": AIModel(
        name="openai/o1",
        provider="openai",
        supports_thinking=True,
        supported_params={"reasoning_effort", "max_tokens", "stream", "tools", "tool_choice", "max_completion_tokens"}
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

    # OpenAI open-weight models served by Groq: provider is groq so the router checks GROQ_API_KEY.
    "gpt-oss-120b": AIModel(
        name="groq/openai/gpt-oss-120b",
        provider="groq",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"}
    ),
    "gpt-oss-20b": AIModel(
        name="groq/openai/gpt-oss-20b",
        provider="groq",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"}
    ),

    # GEMINI
    # Note: web_search_options enables Google Search Grounding for real-time web search
    # Example: web_search_options={"search_context_size": "high"}  # Options: "low", "medium", "high"

    # gemini-3.8-flash: GA (stable) September 2, 2026. 1M context, 65,536 max output. Same API
    # surface as 3.7-flash: thinking_level low/medium/high ("minimal" returns 400), default medium.
    # Google's migration guide deprecates temperature/top_p/top_k/candidate_count but the API still
    # accepts them (verified 2026-09-02). Tuned for coding/agentic work; thinks heavily by default.
    "gemini-3.8-flash": AIModel(
        name="gemini/gemini-3.8-flash",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"low", "medium", "high"},
        use_direct_sdk=True,
    ),
    # gemini-3.7-flash: GA (stable) August 13, 2026. 1M context, 65,536 max output. thinking_level
    # low/medium/high only — "minimal" returns a 400, unlike 3.5/3.6-flash.
    "gemini-3.7-flash": AIModel(
        name="gemini/gemini-3.7-flash",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"low", "medium", "high"},
        use_direct_sdk=True,
    ),
    # gemini-3.6-flash: GA July 21, 2026. 1M context, 65,536 max output. Same thinking surface
    # as 3.5-flash (thinking_level, default medium); cheaper output than 3.5-flash ($7.50 vs $9).
    "gemini-3.6-flash": AIModel(
        name="gemini/gemini-3.6-flash",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        use_direct_sdk=True,
    ),
    # gemini-3.5-flash-lite: GA July 21, 2026. 1M context, 65,536 max output. Fastest/cheapest
    # of the 3.5 family ($0.30/$2.50 per 1M).
    "gemini-3.5-flash-lite": AIModel(
        name="gemini/gemini-3.5-flash-lite",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        use_direct_sdk=True,
    ),
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
    # Deprecated by Google, shutdown 2027-05-07; replacement gemini-3.5-flash-lite.
    # https://ai.google.dev/gemini-api/docs/deprecations
    "gemini-3.1-flash-lite": AIModel(
        name="gemini/gemini-3.1-flash-lite",
        provider="gemini",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "stop", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice", "candidate_count", "safety_settings", "web_search_options"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high"},
        use_direct_sdk=True,
    ),
    "nano-banana-3": AIModel(
        name="gemini/gemini-3-pro-image",
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

    # Opus 5.5: adaptive thinking always on; default effort medium. No sampling
    # params or thinking budgets. Forced tool choices fall back to auto.
    # 1M context, 128k max output. $4/$20 per 1M; cache reads $0.20 per 1M.
    "claude-opus-5-5": AIModel(
        name="anthropic/claude-opus-5-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        accepted_tool_choices={"auto", "none"},
        use_direct_sdk=True,
        max_output_tokens=128000,
    ),
    # Fable 5.1: released 2026-09-01. Same API surface as Fable 5 (adaptive thinking
    # always on, no temperature/top_p/top_k, effort low..max) with one breaking change:
    # forced tool use returns 400 — tool_choice "any" or a named tool is coerced to
    # "auto" via accepted_tool_choices. 1M context, 128k max output. $10/$50 per 1M;
    # cache reads $0.25 per 1M (0.025x vs the standard 0.1x).
    "claude-fable-5-1": AIModel(
        name="anthropic/claude-fable-5-1",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        accepted_tool_choices={"auto", "none"},
        use_direct_sdk=True,
        max_output_tokens=128000,
    ),
    # Opus 5: released 2026-07-24. Thinking on by default (adaptive); "disabled" is
    # accepted only at effort high or below (400 at xhigh/max). No temperature/top_p/top_k.
    # 1M context, 128k max output. $5/$25 per 1M, same as Opus 4.8.
    "claude-opus-5": AIModel(
        name="anthropic/claude-opus-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        use_direct_sdk=True,
        max_output_tokens=128000,
    ),
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
        max_output_tokens=128000,
    ),
    # Opus 4.8: same API surface as Opus 4.7 (adaptive thinking only, no temperature).
    "claude-opus-4-8": AIModel(
        name="anthropic/claude-opus-4-8",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh", "max"},
        use_direct_sdk=True,
        max_output_tokens=128000,
    ),
    # Opus 4.7 removes temperature/top_p/top_k and budget_tokens; only adaptive thinking is supported.
    "claude-opus-4-7": AIModel(
        name="anthropic/claude-opus-4-7",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"stop", "max_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh"},
        use_direct_sdk=True,
        max_output_tokens=128000,
    ),
    "claude-opus-4-6": AIModel(
        name="anthropic/claude-opus-4-6",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high", "max"},
        use_direct_sdk=True,
        max_output_tokens=128000,
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
        max_output_tokens=128000,
    ),
    "claude-sonnet-4-6": AIModel(
        name="anthropic/claude-sonnet-4-6",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "reasoning_effort", "stream", "tools", "tool_choice", "betas"},
        accepted_reasoning_efforts={"low", "medium", "high"},
        use_direct_sdk=True,
        max_output_tokens=128000,
    ),
    "claude-opus-4-5": AIModel(
        name="anthropic/claude-opus-4-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
        max_output_tokens=64000,
    ),
    "claude-haiku-4-5": AIModel(
        name="anthropic/claude-haiku-4-5",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
        max_output_tokens=64000,
    ),
    "claude-sonnet-4-5-20250929": AIModel(
        name="anthropic/claude-sonnet-4-5-20250929",
        provider="anthropic",
        supports_thinking=True,
        supported_params={"temperature", "stop", "max_tokens", "budget_tokens", "thinking", "stream", "tools", "tool_choice", "betas"},
        use_direct_sdk=True,
        max_output_tokens=64000,
    ),

    # META (MODEL API)

    # Muse Spark 1.2: updated Muse Spark checkpoint, released 2026-08-05. Same family
    # specs as 1.1 (1,048,576-token context, 131,072 max output, multimodal in / text out,
    # reasoning always on) served via Meta Model API (api.meta.ai, OpenAI-compatible) —
    # routed through LiteLLM's generic "openai/" provider with an api_base override;
    # API key comes from META_API_KEY. stop is rejected; reasoning_effort "none" is rejected;
    # tool_choice accepts ONLY "auto" ("none"/"required"/named 400 — verified 2026-08-07).
    "muse-spark-1.2": AIModel(
        name="openai/muse-spark-1.2",
        provider="meta",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high", "xhigh"},
        accepted_tool_choices={"auto"},
        api_base="https://api.meta.ai/v1",
        # Standard tier: prompts/completions NOT used for training.
        # Official pricing (per 1M tokens): https://dev.meta.ai/docs/pricing-rate-limits
        pricing={"input": 1.25, "cached_input": 0.15, "output": 4.25},
    ),

    # Same 1.2 checkpoint on Meta's discounted Contributor tier: ~12-21x cheaper, but
    # prompts AND completions may be used to train future Meta models — never route
    # sensitive or proprietary content here. Rate-limited to 60 RPM / 2.1M TPM (vs
    # 3,000 RPM / 4M TPM standard).
    "muse-spark-1.2-contributor": AIModel(
        name="openai/muse-spark-1.2-contributor",
        provider="meta",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high", "xhigh"},
        accepted_tool_choices={"auto"},
        api_base="https://api.meta.ai/v1",
        # Official pricing (per 1M tokens): https://dev.meta.ai/docs/pricing-rate-limits
        pricing={"input": 0.10, "cached_input": 0.002, "output": 0.20},
    ),

    # Muse Spark 1.1: multimodal reasoning model, released 2026-07-09. 1,048,576-token
    # context, 131,072 max output. Served only via Meta Model API (api.meta.ai), which is
    # OpenAI-compatible — routed through LiteLLM's generic "openai/" provider with an
    # api_base override; API key comes from META_API_KEY. stop is rejected, and
    # reasoning_effort "none" is rejected (model always thinks). Tuned for temperature 1.0.
    "muse-spark-1.1": AIModel(
        name="openai/muse-spark-1.1",
        provider="meta",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"minimal", "low", "medium", "high", "xhigh"},
        api_base="https://api.meta.ai/v1",
        # LiteLLM can't price custom-api_base models, so cost falls back to this.
        # Official Meta Model API pricing (per 1M tokens), reasoning billed as output:
        # https://dev.meta.ai/docs/getting-started/pricing-rate-limits
        pricing={"input": 1.25, "cached_input": 0.15, "output": 4.25},
    ),

    # MOONSHOT (FIRST-PARTY API)

    # Kimi K3: Moonshot's flagship (2.8T MoE, 16-of-896 experts), API launched Jul 16, 2026.
    # API-only until open weights land (~Jul 27, 2026) — served at api.moonshot.ai
    # (OpenAI-compatible), routed via LiteLLM's generic "openai/" provider with an
    # api_base override; API key comes from MOONSHOT_API_KEY. 1M context, 131,072
    # default max output. Reasoning is always on (effort low/high/max, default max);
    # temperature is fixed at 1.0 server-side, so it's not a supported param.
    "kimi-k3": AIModel(
        name="openai/kimi-k3",
        provider="moonshot",
        supports_thinking=True,
        supported_params={"max_tokens", "max_completion_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"low", "high", "max"},
        api_base="https://api.moonshot.ai/v1",
        # LiteLLM can't price custom-api_base models, so cost falls back to this.
        # Official Moonshot pricing: https://platform.kimi.ai/docs/pricing/chat-k3
        pricing={"input": 3.00, "cached_input": 0.30, "output": 15.00},
    ),

    # OPENROUTER (AGGREGATOR)

    # GLM 5.3 Flash: Z.ai's cheap/fast natively-multimodal MoE (320B total / 18B active),
    # released 2026-08-26. Served via OpenRouter (aggregates 12 upstream endpoints) using
    # LiteLLM's native "openrouter/" prefix, which passes any OpenRouter model id through;
    # API key comes from OPENROUTER_API_KEY. 1,048,576-token context, 131,072 max output.
    # Reasoning is mandatory and can't be disabled; effort low/high/max (default max,
    # "medium" rejected). stop is rejected by the first-party Z.AI endpoint.
    # Pinned to Z.ai's first-party endpoint (reference fp8 precision — the official
    # weights ship in FP8; several resellers serve unknown quantizations) with
    # fallback to other providers allowed if Z.AI is down.
    "glm-5.3-flash": AIModel(
        name="openrouter/z-ai/glm-5.3-flash",
        provider="openrouter",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"low", "high", "max"},
        extra_body={"provider": {"order": ["z-ai"], "allow_fallbacks": True}},
        # LiteLLM's cost map doesn't know this model, so cost falls back to this.
        # OpenRouter pricing (per 1M tokens) with the 50% launch discount active as of
        # 2026-08-27; undiscounted list is input 0.15 / cached 0.03 / output 0.50:
        # https://openrouter.ai/z-ai/glm-5.3-flash
        pricing={"input": 0.075, "cached_input": 0.015, "output": 0.25},
    ),

    # XAI

    # grok-4.6: reasoning always on with configurable reasoning_effort (low/medium/high/xhigh, default high).
    # 500K context window. Vision input supported. stop/presencePenalty/frequencyPenalty are rejected
    # by xAI reasoning models. Pricing $2/M in, $6/M out (<200k tokens; doubles above).
    "grok-4.6": AIModel(
        name="xai/grok-4.6",
        provider="xai",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "top_k", "max_tokens", "reasoning_effort", "stream", "tools", "tool_choice"},
        accepted_reasoning_efforts={"low", "medium", "high", "xhigh"},
    ),
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
    # xAI retired the six ids below on 2026-05-15 and now silently serves grok-4.3 (the grok-4-1-fast /
    # grok-4 / grok-4-fast ids) or grok-build-0.1 (grok-code-fast-1), billed at the served model's rate.
    # They stay registered only because skell-e-web still calls them; each is listed in DEPRECATED_MODELS
    # below and warns on use. Delete the entries and their DEPRECATED_MODELS lines once skell-e-web repoints.
    # https://docs.x.ai/developers/migration/may-15-retirement.md
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


    # DEEPINFRA (OPEN-WEIGHT & PARTNER MODELS)
    # DeepInfra serves open-weight and partner models via an OpenAI-compatible API (DEEPINFRA_API_KEY).
    # pricing is set per-model because LiteLLM's cost map lags new DeepInfra additions.
    # DeepInfra's gateway accepts reasoning_effort (none/minimal/low/medium/high/xhigh/max) for
    # every model, but only some honor it. Probed 2026-09-12 (benchmark repo,
    # docs/deepinfra-reasoning-defaults.md): the DeepSeek V4 family does NO reasoning unless an
    # effort is pinned, so those entries declare reasoning_effort and the router whitelists it
    # past LiteLLM's drop_params. GLM/MiniMax/Qwen/Nemotron reason by default and show no clear
    # response to the knob, so they keep no effort support.

    # DeepSeek-V4.1-Flash: new causal encoder-decoder MoE (552B total / 8B-16B active), Sep 10 2026.
    # 1M context, native vision input. DeepSeek reports it beating V4-Pro on code/agent tasks.
    # DeepSeek first-party retired V4-Flash and routes V4-Pro here from Sep 14 2026; DeepInfra
    # ($0.20/$0.60) undercuts first-party peak pricing ($0.30/$1.20), so we stay on DeepInfra.
    # Unlike the older DeepInfra entries below, this one honors reasoning_effort (verified 2026-09-12
    # against the API: bogus values are rejected with the accepted list; omitting it or "none" yields
    # 0 reasoning tokens, so the provider default is NO reasoning). Pin an effort to get thinking.
    "deepseek-v4.1-flash": AIModel(
        name="deepinfra/deepseek-ai/DeepSeek-V4.1-Flash",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice", "reasoning_effort"},
        accepted_reasoning_efforts={"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        pricing={"input": 0.20, "cached_input": 0.006, "output": 0.60},
    ),
    # DeepSeek-V4-Pro: DeepSeek's flagship MoE (1.6T total / 49B active), Apr 2026. 1M context.
    # No reasoning unless reasoning_effort is pinned (verified 2026-09-12: unpinned = 0 reasoning tokens).
    "deepseek-v4-pro": AIModel(
        name="deepinfra/deepseek-ai/DeepSeek-V4-Pro",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice", "reasoning_effort"},
        accepted_reasoning_efforts={"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        pricing={"input": 1.30, "cached_input": 0.10, "output": 2.60},
    ),
    # DeepSeek-V4-Flash: cheap/fast tier of V4, Apr 2026. 1M context. Retired by DeepSeek first-party
    # on 2026-09-10; DeepInfra still serves the original weights. No reasoning unless an effort is pinned.
    "deepseek-v4-flash": AIModel(
        name="deepinfra/deepseek-ai/DeepSeek-V4-Flash",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice", "reasoning_effort"},
        accepted_reasoning_efforts={"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        pricing={"input": 0.09, "cached_input": 0.018, "output": 0.18},
    ),
    # GLM-5.2: Zhipu/Z.ai coding-first 744B MoE, MIT open weights, Jun 2026. 1M context.
    "glm-5.2": AIModel(
        name="deepinfra/zai-org/GLM-5.2",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
        pricing={"input": 0.93, "cached_input": 0.18, "output": 3.00},
    ),
    # MiniMax-M3: open-weight multimodal MoE (~428B/A23B), Jun 2026. 512K context on DeepInfra.
    "minimax-m3": AIModel(
        name="deepinfra/MiniMaxAI/MiniMax-M3",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
        pricing={"input": 0.30, "cached_input": 0.06, "output": 1.20},
    ),
    # Qwen3.8-Max: Alibaba's closed-weight flagship (2.4T MoE), Aug 2026. Served on
    # DeepInfra as a partner model, 256K context, 65,536 max output. DeepInfra's page once
    # tagged it non-reasoning, but the deployment now returns reasoning content unpinned.
    "qwen3.8-max": AIModel(
        name="deepinfra/Qwen/Qwen3.8-Max",
        provider="deepinfra",
        supports_thinking=True,  # reasons by default on DeepInfra (probe 2026-09-12: ~100 reasoning tokens unpinned)
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
        pricing={"input": 1.65, "cached_input": 0.206, "output": 4.95},
    ),
    # Qwen3.5-397B-A17B: Alibaba's open-weight flagship, Feb 2026. 256K context on DeepInfra.
    "qwen3.5-397b": AIModel(
        name="deepinfra/Qwen/Qwen3.5-397B-A17B",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
        pricing={"input": 0.45, "cached_input": 0.22, "output": 3.00},
    ),

    # DEEPINFRA (NVIDIA NEMOTRON)

    # Nemotron 3 Ultra: NVIDIA's open frontier MoE (550B/A55B), Jun 2026. 256K ctx on DeepInfra.
    "nemotron-3-ultra": AIModel(
        name="deepinfra/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
        provider="deepinfra",
        supports_thinking=True,
        supported_params={"temperature", "top_p", "stop", "max_tokens", "stream", "tools", "tool_choice"},
        pricing={"input": 0.50, "cached_input": 0.10, "output": 2.20},
    ),
    "nemotron-3-super": AIModel(
        name="deepinfra/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
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

}

# Additional aliases
MODEL_CONFIG["gemini-3-pro-preview"] = MODEL_CONFIG["gemini-3.1-pro-preview"]  # Gemini 3 Pro discontinued March 9, 2026
MODEL_CONFIG["gemini-3.1-flash-lite-preview"] = MODEL_CONFIG["gemini-3.1-flash-lite"]  # Renamed from preview on GA
MODEL_CONFIG["gemini-3-pro-image"] = MODEL_CONFIG["nano-banana-3"]

# DeepInfra retired these Nemotron ids (Jun-Jul 2026) and forwards requests to the Nemotron 3 successors,
# so the aliases resolve to the model DeepInfra actually serves (confirmed by live calls on 2026-09-25,
# including the 12B vision id). Listed in DEPRECATED_MODELS; delete once
# skell-e-web repoints. https://api.deepinfra.com/models/list
MODEL_CONFIG["nemotron-super-49b"] = MODEL_CONFIG["nemotron-3-ultra"]
MODEL_CONFIG["nemotron-70b"] = MODEL_CONFIG["nemotron-3-ultra"]
MODEL_CONFIG["nemotron-nano-12b-vl"] = MODEL_CONFIG["nemotron-3-ultra"]
MODEL_CONFIG["nemotron-nano-9b"] = MODEL_CONFIG["nemotron-3-nano-30b"]

# Aliases the provider has retired but that still answer (the provider redirects them). They resolve
# normally and resolve_model_alias() logs a warning naming the replacement. Value = message shown to
# the caller. Remove an id from here, its alias/entry above, and any test the same day.
DEPRECATED_MODELS: dict[str, str] = {
    "grok-4-1-fast-reasoning": "xAI retired it on 2026-05-15 and serves grok-4.3 instead; use grok-4.20",
    "grok-4-1-fast-non-reasoning": "xAI retired it on 2026-05-15 and serves grok-4.3 instead; use grok-4.20-non-reasoning",
    "grok-4-0709": "xAI retired it on 2026-05-15 and serves grok-4.3 instead; use grok-4.20",
    "grok-4-fast-reasoning": "xAI retired it on 2026-05-15 and serves grok-4.3 instead; use grok-4.20",
    "grok-4-fast-non-reasoning": "xAI retired it on 2026-05-15 and serves grok-4.3 instead; use grok-4.20-non-reasoning",
    "grok-code-fast-1": "xAI retired it on 2026-05-15 and serves grok-build-0.1 instead; use grok-4.20",
    "nemotron-super-49b": "DeepInfra retired it on 2026-07-17 and serves Nemotron 3 Ultra; use nemotron-3-ultra",
    "nemotron-70b": "DeepInfra retired it on 2026-07-16 and serves Nemotron 3 Ultra; use nemotron-3-ultra",
    "nemotron-nano-12b-vl": "DeepInfra retired it on 2026-07-16 and serves Nemotron 3 Ultra; use nemotron-3-ultra",
    "nemotron-nano-9b": "DeepInfra retired it on 2026-06-11 and serves Nemotron 3 Nano 30B; use nemotron-3-nano-30b",
}

# Allow lookup by full name too
for config in list(MODEL_CONFIG.values()): # Iterate over a copy if modifying during iteration (though here it's safe)
    if config.name not in MODEL_CONFIG:
        MODEL_CONFIG[config.name] = config

# Full-name lookups of a deprecated entry (e.g. "xai/grok-4-0709") warn too. Aliases that point at a
# live successor keep the successor's full name clean.
for _alias in list(DEPRECATED_MODELS):
    _full = MODEL_CONFIG[_alias].name
    if _full.endswith("/" + _alias):
        DEPRECATED_MODELS[_full] = DEPRECATED_MODELS[_alias]


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


# ============================================================
# IMAGE GENERATION MODEL CONFIGURATION
# ============================================================


class ImageModel:
    """Registry entry for an image-generation model. Distinct from chat AIModel."""

    def __init__(
        self,
        name: str,                                  # provider model id (no LiteLLM prefix)
        provider: str,                              # "openai" | "deepinfra" | "gemini"
        supported_sizes: tuple[str, ...],           # named sizes accepted by this model
        supported_qualities: tuple[str, ...] = ("auto",),
        supported_params: tuple[str, ...] = (),     # scalar request params the provider honors
        supports_transparent_background: bool = False,
        supports_edits: bool = False,               # accepts reference images
        allows_custom_sizes: bool = True,           # accepts arbitrary "WIDTHxHEIGHT"
        default_size: str = "auto",                 # what size="auto" resolves to on the wire
        api_base: str | None = None,                # OpenAI-compatible endpoint override
        chat_alias: str | None = None,              # MODEL_CONFIG alias for chat-path providers
        text_input_price: float | None = None,      # USD per 1M text input tokens
        image_input_price: float | None = None,     # USD per 1M image input tokens
        image_output_price: float | None = None,    # USD per 1M image output tokens
        price_per_image: float | None = None,       # flat USD per generated image
        price_per_image_tiers: tuple[tuple[int | None, float], ...] | None = None,
        min_pixels: int | None = None,              # provider-enforced floor on width*height
    ):
        self.name = name
        self.provider = provider
        self.supported_sizes = supported_sizes
        self.supported_qualities = supported_qualities
        # Params outside this set must be left at their generate_image() default;
        # a non-default value raises INVALID_PARAM instead of being silently dropped.
        self.supported_params = frozenset(supported_params)
        self.supports_transparent_background = supports_transparent_background
        self.supports_edits = supports_edits
        self.allows_custom_sizes = allows_custom_sizes
        self.default_size = default_size
        self.api_base = api_base
        self.chat_alias = chat_alias
        self.text_input_price = text_input_price
        self.image_input_price = image_input_price
        self.image_output_price = image_output_price
        # Flat per-image price (DeepInfra bills per image, not per token).
        self.price_per_image = price_per_image
        # Resolution-tiered per-image price as ((max_pixels, usd), ...) ascending.
        # A max_pixels of None is the open-ended top tier.
        self.price_per_image_tiers = price_per_image_tiers
        # Smallest width*height the provider will render. Checked locally so an
        # undersized request fails fast instead of burning three retries on the
        # 500 the gateway returns.
        self.min_pixels = min_pixels

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @property
    def is_deepinfra(self) -> bool:
        return self.provider == "deepinfra"

    @property
    def is_gemini(self) -> bool:
        return self.provider == "gemini"

    @property
    def has_pricing(self) -> bool:
        """True when token-based pricing is registered (OpenAI GPT-Image)."""
        return None not in (
            self.text_input_price,
            self.image_input_price,
            self.image_output_price,
        )

    @property
    def has_per_image_pricing(self) -> bool:
        return self.price_per_image is not None or bool(self.price_per_image_tiers)

    def price_for_size(self, size: str) -> float | None:
        """USD for one image at `size`. None when no per-image price is registered."""
        if self.price_per_image is not None:
            return self.price_per_image
        if not self.price_per_image_tiers:
            return None
        pixels = image_size_pixels(size)
        if pixels is None:
            return None
        for max_pixels, price in self.price_per_image_tiers:
            if max_pixels is None or pixels <= max_pixels:
                return price
        return self.price_per_image_tiers[-1][1]


def image_size_pixels(size: str) -> int | None:
    """Total pixels for a "WIDTHxHEIGHT" size string. None if unparseable."""
    if not isinstance(size, str):
        return None
    normalized = size.strip().lower()
    parts = normalized.split("x")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return int(parts[0]) * int(parts[1])
    return None


# Named sizes OpenAI recommends for every GPT-Image model. Custom "WIDTHxHEIGHT"
# values are also accepted - see IMAGE_CUSTOM_SIZE_RULES in images.py.
_GPT_IMAGE_SIZES = ("auto", "1024x1024", "1536x1024", "1024x1536")

# Every scalar knob generate_image() exposes. GPT-Image honors all of them.
_GPT_IMAGE_PARAMS = ("quality", "background", "output_format", "output_compression", "n")

# DeepInfra fronts its image models with an OpenAI-compatible images endpoint.
DEEPINFRA_IMAGE_API_BASE = "https://api.deepinfra.com/v1/openai"

# DeepInfra's gateway only takes explicit "{width}x{height}" — the shorthand
# "2K" / "4K" the model card mentions is rejected with a 422 (probed 2026-09-14).
# Seedream 4.5 additionally refuses anything under 3,686,400 pixels.
_SEEDREAM_45_SIZES = ("auto", "2560x1440", "2048x2048", "3840x2160", "4096x4096")
_SEEDREAM_4_SIZES = ("auto", "1024x1024", "2048x2048", "2560x1440", "4096x4096")
# Seedream 5.0 Pro renders up to 2K.
_SEEDREAM_5_SIZES = ("auto", "1024x1024", "1536x1536", "2048x2048")

IMAGE_CONFIG: dict[str, ImageModel] = {
    "gpt-image-2.5-flare": ImageModel(
        name="gpt-image-2.5-flare",
        provider="openai",
        supported_sizes=_GPT_IMAGE_SIZES,
        supported_qualities=("auto", "low", "medium", "high", "xhigh", "max"),
        supported_params=_GPT_IMAGE_PARAMS,
        supports_transparent_background=True,
        supports_edits=True,
        text_input_price=5.00,
        image_input_price=8.00,
        image_output_price=30.00,
    ),
    "gpt-image-2.5-sunburst": ImageModel(
        name="gpt-image-2.5-sunburst",
        provider="openai",
        supported_sizes=_GPT_IMAGE_SIZES,
        supported_qualities=("auto", "low", "medium", "high", "xhigh", "max"),
        supported_params=_GPT_IMAGE_PARAMS,
        supports_transparent_background=True,
        supports_edits=True,
        text_input_price=5.00,
        image_input_price=8.00,
        image_output_price=30.00,
    ),
    "gpt-image-2": ImageModel(
        name="gpt-image-2",
        provider="openai",
        supported_sizes=_GPT_IMAGE_SIZES,
        # Models before 2.5 top out at "high" - no "xhigh" / "max" tiers.
        supported_qualities=("auto", "low", "medium", "high"),
        supported_params=_GPT_IMAGE_PARAMS,
        supports_transparent_background=True,
        supports_edits=True,
        text_input_price=5.00,
        image_input_price=8.00,
        image_output_price=30.00,
    ),

    # DEEPINFRA (ByteDance Seedream) - flat per-image billing, JPEG output.
    # No quality / background / output_format knobs: the OpenAI-compatible
    # endpoint takes prompt, model, size, n and response_format only.
    "seedream-4.5": ImageModel(
        name="ByteDance/Seedream-4.5",
        provider="deepinfra",
        supported_sizes=_SEEDREAM_45_SIZES,
        supported_params=("n",),
        # 1024x1024 is below this model's floor, so "auto" opens at 2K square.
        default_size="2048x2048",
        min_pixels=3_686_400,
        api_base=DEEPINFRA_IMAGE_API_BASE,
        price_per_image=0.04,
    ),
    "seedream-4": ImageModel(
        name="ByteDance/Seedream-4",
        provider="deepinfra",
        supported_sizes=_SEEDREAM_4_SIZES,
        supported_params=("n",),
        default_size="1024x1024",
        api_base=DEEPINFRA_IMAGE_API_BASE,
        price_per_image=0.04,
    ),
    # Seedream 5.0 Pro is billed by resolution tier: $0.0495/image up to 1.5K
    # (2,359,296 px) and $0.099/image above that. DeepInfra model page, 2026-09-14.
    # It renders 1024x1024 happily, so it keeps the cheap default.
    "seedream-5-pro": ImageModel(
        name="ByteDance/Seedream-5.0-Pro",
        provider="deepinfra",
        supported_sizes=_SEEDREAM_5_SIZES,
        supported_params=("n",),
        default_size="1024x1024",
        api_base=DEEPINFRA_IMAGE_API_BASE,
        price_per_image_tiers=((1536 * 1536, 0.0495), (None, 0.099)),
    ),

    # GEMINI - image output arrives through the chat path, so generate_image()
    # delegates to ask_ai(chat_alias, ..., rich_response=True) and unpacks the
    # data URLs. Gemini picks the resolution itself; size is never sent.
    "nano-banana-3": ImageModel(
        name="gemini-3-pro-image",
        provider="gemini",
        supported_sizes=("auto", "1024x1024"),
        supported_params=(),
        supports_edits=True,          # reference images ride along as chat input
        allows_custom_sizes=False,
        chat_alias="nano-banana-3",
    ),
}

# Additional aliases - the fast 2.5 variant is the default "just give me an image".
IMAGE_CONFIG["gpt-image-2.5"] = IMAGE_CONFIG["gpt-image-2.5-flare"]
IMAGE_CONFIG["gpt-image"] = IMAGE_CONFIG["gpt-image-2.5-flare"]
IMAGE_CONFIG["gemini-3-pro-image"] = IMAGE_CONFIG["nano-banana-3"]
IMAGE_CONFIG["nano-banana-pro"] = IMAGE_CONFIG["nano-banana-3"]

# Allow lookup by full model name in addition to alias.
for _img_cfg in list(IMAGE_CONFIG.values()):
    if _img_cfg.name not in IMAGE_CONFIG:
        IMAGE_CONFIG[_img_cfg.name] = _img_cfg


def resolve_image_alias(model_alias: str) -> ImageModel:
    """Resolve an image model alias (or full model name) to its ImageModel."""
    # Local import avoids a circular reference (utils.py imports model_config at module load).
    from .utils import RouterError

    model = IMAGE_CONFIG.get(model_alias)
    if not model:
        available = sorted(IMAGE_CONFIG)
        raise RouterError(
            code="INVALID_MODEL",
            message=f"Invalid image model alias '{model_alias}'. Available: {available}",
        )
    return model


class ClassificationModel:
    """Typed evaluation model, separate from generative chat models."""

    def __init__(self, name: str):
        self.name = name
        self.provider = "typesafe"
        self.input_cost_per_million = 0.042
        self.output_cost_per_million = 0.0
        # Provider limits, enforced by the API using its own tokenizer.
        self.max_input_tokens = 64000
        self.max_state_question_tokens = 32000
        self.question_types = frozenset({"choice", "score", "noul"})


# https://docs.typesafe.ai/models and /model-jaggedness/jev-1.13
# The short alias is pinned; callers opt into provider upgrades with jev-latest.
CLASSIFICATION_MODEL_CONFIG = {
    "jev": ClassificationModel("jev-1.13.0"),
    "jev-latest": ClassificationModel("jev-latest"),
}
CLASSIFICATION_MODEL_CONFIG["jev-1.13.0"] = CLASSIFICATION_MODEL_CONFIG["jev"]


def resolve_classification_alias(model_alias: str) -> ClassificationModel:
    from .utils import RouterError

    if not isinstance(model_alias, str) or model_alias not in CLASSIFICATION_MODEL_CONFIG:
        raise RouterError("INVALID_MODEL", "Unknown classification model alias.")
    return CLASSIFICATION_MODEL_CONFIG[model_alias]
