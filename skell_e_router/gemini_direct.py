# GEMINI DIRECT SDK
# -----------------
# Bypass LiteLLM for speed-critical Gemini models by calling the google-genai SDK directly.
# This eliminates 0.3-1.7s of overhead per call that LiteLLM adds.

import time
import json
import uuid

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

from .response import AIResponse


# Reverse of utils._AUDIO_MIME_TO_FORMAT — picks one canonical MIME per format
# value, dropping the "tolerant" duplicates. Used to reconstruct the MIME type
# Gemini's SDK expects from the format suffix carried in OpenAI-canonical
# `input_audio` content parts.
_AUDIO_FORMAT_TO_MIME = {
    "mp3":  "audio/mpeg",
    "wav":  "audio/wav",
    "flac": "audio/flac",
    "ogg":  "audio/ogg",
    "mp4":  "audio/mp4",
    "webm": "audio/webm",
}


# Cache genai.Client instances per API key to avoid ~0.8s construction overhead per call
_client_cache: dict[str, object] = {}


def _get_gemini_client(api_key: str):
    """Return a cached genai.Client for the given API key."""
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(api_key=api_key)
    return _client_cache[api_key]


# Known pricing per 1M tokens (USD) for direct-SDK models
# Per 1M tokens (USD), text/image/video input — source: https://ai.google.dev/gemini-api/docs/pricing
# USD per 1M tokens (text/image/video rates) from
# https://ai.google.dev/gemini-api/docs/pricing. "cached_input" is the
# per-model context-caching read price, which Google also passes on
# automatically for implicit-cache hits (enabled by default on Gemini 2.5+,
# see https://ai.google.dev/gemini-api/docs/caching). Cache hits are reported
# in usage_metadata.cached_content_token_count, a SUBSET of prompt_token_count.
_PRICING = {
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached_input": 0.15},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached_input": 0.05},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cached_input": 0.025},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached_input": 0.01},
}


def _convert_messages_to_contents(messages: list[dict]) -> tuple[str | None, list]:
    """Convert OpenAI-format messages to Google SDK format.

    Returns (system_instruction, contents) where system_instruction is extracted
    from system messages and contents is the list of conversation turns.
    """
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # Google SDK takes system instruction separately via config
            if isinstance(content, str):
                system_instruction = content
            continue

        # Map OpenAI roles to Google SDK roles
        sdk_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append(types.Content(
                role=sdk_role,
                parts=[types.Part.from_text(text=content)]
            ))
        elif isinstance(content, list):
            # Multi-part content (text + images)
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(types.Part.from_text(text=part["text"]))
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # Parse data URI: data:mime;base64,DATA
                            header, b64_data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            import base64
                            parts.append(types.Part.from_bytes(
                                data=base64.b64decode(b64_data),
                                mime_type=mime
                            ))
                        else:
                            parts.append(types.Part.from_uri(
                                file_uri=url,
                                mime_type="image/jpeg"  # Default; SDK may auto-detect
                            ))
                    elif part.get("type") == "input_audio":
                        audio_data = part["input_audio"]["data"]
                        audio_format = part["input_audio"]["format"]
                        mime = _AUDIO_FORMAT_TO_MIME.get(audio_format, f"audio/{audio_format}")
                        import base64
                        parts.append(types.Part.from_bytes(
                            data=base64.b64decode(audio_data),
                            mime_type=mime,
                        ))
            if parts:
                contents.append(types.Content(role=sdk_role, parts=parts))

    return system_instruction, contents


def _thinking_config_for_effort(effort: str):
    """Build a ThinkingConfig using thinking_level for reasoning_effort strings.

    For models that accept reasoning_effort (low/medium/high) but not budget_tokens,
    the Google GenAI SDK expects ThinkingConfig(thinking_level=...) rather than
    ThinkingConfig(thinking_budget=N).
    """
    level_map = {
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }
    level = level_map.get(effort)
    if level is not None:
        return types.ThinkingConfig(thinking_level=level)
    # "minimal" or unknown → disable thinking
    return types.ThinkingConfig(thinking_budget=0)


# JSON-schema keys the Gemini API accepts in function-declaration parameters
# (google.genai types.Schema fields, camelCase as they appear in JSON schema).
# Anything else — additionalProperties, $schema, exclusiveMinimum, const, ... —
# is rejected by the SDK (extra_forbidden) or the API (unknown field), so
# unknown keys are dropped rather than forwarded.
_GEMINI_SCHEMA_KEYS = frozenset({
    "type", "format", "title", "description", "nullable", "default", "enum",
    "example", "items", "properties", "required", "propertyOrdering", "anyOf",
    "minimum", "maximum", "minLength", "maxLength", "pattern",
    "minItems", "maxItems", "minProperties", "maxProperties",
})


def _sanitize_schema_for_gemini(schema):
    """Make an OpenAI-format tool parameter schema safe for the Gemini API.

    Local ``$defs``/``$ref`` pairs (emitted by schema generators for nested
    types) are inlined, then every key the Gemini API does not understand is
    dropped recursively. Cycles are depth-capped rather than expanded forever.
    """
    if not isinstance(schema, dict):
        return schema
    defs = schema.get("$defs") or {}

    def resolve(node, depth: int = 0):
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/") and depth <= 12:
            target = defs.get(ref[len("#/$defs/"):])
            if isinstance(target, dict):
                siblings = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**target, **siblings}, depth + 1)
        out = {}
        for key, value in node.items():
            if key not in _GEMINI_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                # keys here are the parameter NAMES — never filter them
                out[key] = {name: resolve(sub, depth + 1)
                            for name, sub in value.items()}
            elif isinstance(value, (dict, list)):
                out[key] = resolve(value, depth + 1)
            else:
                out[key] = value
        # Gemini requires `items` on every array schema (e.g. a bare `list`
        # type hint produces an items-less array and a 400).
        if str(out.get("type", "")).lower() == "array" and "items" not in out:
            out["items"] = {"type": "string"}
        return out

    return resolve(schema)


def _sanitize_openai_tools_for_gemini(tools):
    """Apply _sanitize_schema_for_gemini across a list of OpenAI-format tool
    declarations. Used by both the direct SDK path and the LiteLLM path —
    LiteLLM forwards parameter schemas to the Gemini API mostly untouched, so
    unsupported keys / items-less arrays 400 there identically."""
    sanitized = []
    for tool_def in tools or []:
        if isinstance(tool_def, dict) and tool_def.get("type") == "function":
            fn = dict(tool_def.get("function") or {})
            if isinstance(fn.get("parameters"), dict):
                fn["parameters"] = _sanitize_schema_for_gemini(fn["parameters"])
            tool_def = {**tool_def, "function": fn}
        sanitized.append(tool_def)
    return sanitized


def _build_generate_config(ai_model, kwargs: dict) -> tuple:
    """Convert router kwargs to Google SDK GenerateContentConfig and tools list.

    Returns (config, tools) where tools is a list to pass separately if needed.
    """
    # Import RouterError here to avoid circular import at module level
    from .utils import RouterError

    config_kwargs = {}
    tools = None
    tool_config = None

    # max_tokens -> max_output_tokens
    if "max_tokens" in kwargs:
        config_kwargs["max_output_tokens"] = kwargs["max_tokens"]

    # Direct pass-through params
    for param in ("temperature", "top_p", "top_k"):
        if param in kwargs:
            config_kwargs[param] = kwargs[param]

    # stop -> stop_sequences (ensure list)
    if "stop" in kwargs:
        stop = kwargs["stop"]
        if isinstance(stop, str):
            stop = [stop]
        config_kwargs["stop_sequences"] = stop

    # candidate_count
    if "candidate_count" in kwargs:
        config_kwargs["candidate_count"] = kwargs["candidate_count"]

    # --- Thinking / reasoning handling ---
    # Priority: budget_tokens > thinking dict > reasoning_effort

    if "budget_tokens" in kwargs:
        budget = kwargs["budget_tokens"]
        if "budget_tokens" in ai_model.supported_params:
            # Direct ThinkingConfig with budget
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget
            )
        elif "reasoning_effort" in ai_model.supported_params:
            # Map budget to effort level, then apply as thinking_level
            accepted = getattr(ai_model, 'accepted_reasoning_efforts', {"low", "medium", "high"})
            if budget == 0:
                effort = "minimal" if "minimal" in accepted else "low"
            elif budget <= 1024:
                effort = "low"
            elif budget <= 2048:
                effort = "medium"
            else:
                effort = "high"
            config_kwargs["thinking_config"] = _thinking_config_for_effort(effort)

    elif "thinking" in kwargs:
        thinking = kwargs["thinking"]
        if isinstance(thinking, dict):
            think_type = thinking.get("type", "disabled")
            if think_type == "enabled":
                budget_val = thinking.get("budget_tokens")
                if budget_val is not None:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=budget_val
                    )
                else:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=2048
                    )
            # disabled -> no thinking_config (skip)

    elif "reasoning_effort" in kwargs:
        effort = kwargs["reasoning_effort"]
        # Validate against model's accepted values
        accepted = getattr(ai_model, 'accepted_reasoning_efforts', None)
        if accepted is None:
            accepted = {"low", "medium", "high"}
        if effort not in accepted:
            raise RouterError(
                code="INVALID_PARAM",
                message=f"'reasoning_effort' must be one of: {sorted(list(accepted))}"
            )
        # Use thinking_level for models that support reasoning_effort,
        # thinking_budget for models that support budget_tokens.
        if "budget_tokens" in ai_model.supported_params:
            budget_map = {
                "minimal": 0,
                "low": 1024,
                "medium": 2048,
                "high": 4096,
            }
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget_map.get(effort, 1024)
            )
        else:
            config_kwargs["thinking_config"] = _thinking_config_for_effort(effort)

    # Safety settings — always set BLOCK_NONE for all categories
    config_kwargs["safety_settings"] = [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
    ]

    # web_search_options -> Google Search tool
    if "web_search_options" in kwargs:
        tools = [types.Tool(google_search=types.GoogleSearch())]

    # tools (function calling) — convert OpenAI format to Google SDK
    if "tools" in kwargs and kwargs["tools"]:
        func_declarations = []
        for tool_def in kwargs["tools"]:
            if isinstance(tool_def, dict) and tool_def.get("type") == "function":
                fn = tool_def["function"]
                fd_kwargs = {"name": fn["name"]}
                if "description" in fn:
                    fd_kwargs["description"] = fn["description"]
                if "parameters" in fn:
                    fd_kwargs["parameters"] = _sanitize_schema_for_gemini(fn["parameters"])
                func_declarations.append(types.FunctionDeclaration(**fd_kwargs))
        if func_declarations:
            fn_tool = types.Tool(function_declarations=func_declarations)
            tools = [fn_tool] if tools is None else tools + [fn_tool]

    # tool_choice -> ToolConfig with FunctionCallingConfig
    if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
        tc = kwargs["tool_choice"]
        if isinstance(tc, str):
            mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            mode = mode_map.get(tc)
            if mode:
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=mode)
                )
        elif isinstance(tc, dict) and tc.get("type") == "function":
            fn_name = tc.get("function", {}).get("name")
            if fn_name:
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[fn_name]
                    )
                )

    if tool_config:
        config_kwargs["tool_config"] = tool_config

    config = types.GenerateContentConfig(**config_kwargs)
    return config, tools


_CONFIG_ATTRS = ("max_output_tokens", "temperature", "top_p", "top_k",
                 "stop_sequences", "thinking_config", "safety_settings",
                 "candidate_count", "tool_config")


def _call_gemini_direct(model_name: str, contents: list, system_instruction: str | None,
                        config, tools, api_key: str) -> object:
    """Call the Google genai SDK with retry on transient errors.

    Retries up to 3 attempts on timeout, 429, and 5xx errors.
    """
    if not GENAI_AVAILABLE:
        raise ImportError(
            "google-genai package is required for direct Gemini SDK calls. "
            "Install it with: pip install google-genai"
        )

    # Strip "gemini/" prefix used by LiteLLM
    if model_name.startswith("gemini/"):
        model_name = model_name[len("gemini/"):]

    client = _get_gemini_client(api_key)

    # Rebuild config to include system_instruction and/or tools if needed
    if system_instruction or tools:
        existing = {}
        for attr in _CONFIG_ATTRS:
            val = getattr(config, attr, None)
            if val is not None:
                existing[attr] = val
        if system_instruction:
            existing["system_instruction"] = system_instruction
        if tools:
            existing["tools"] = tools
        config = types.GenerateContentConfig(**existing)

    call_kwargs = dict(
        model=model_name,
        contents=contents,
        config=config,
    )

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            request_start = time.perf_counter()
            response = client.models.generate_content(**call_kwargs)
            request_duration = time.perf_counter() - request_start
            return response, request_duration
        except Exception as e:
            err_name = type(e).__name__.lower()
            status = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            is_transient = (
                'timeout' in err_name or
                'connection' in err_name or
                status in (429, 500, 502, 503, 504)
            )
            if is_transient and attempt < max_attempts:
                wait_time = 2 ** attempt  # 2s, 4s
                time.sleep(wait_time)
                continue
            raise


def _call_gemini_direct_stream(model_name: str, contents: list, system_instruction: str | None,
                               config, tools, api_key: str):
    """Call the Google genai SDK with streaming. Returns an iterator of chunks.

    Same retry logic as _call_gemini_direct (retry before first chunk).
    """
    if not GENAI_AVAILABLE:
        raise ImportError(
            "google-genai package is required for direct Gemini SDK calls. "
            "Install it with: pip install google-genai"
        )

    # Strip "gemini/" prefix used by LiteLLM
    if model_name.startswith("gemini/"):
        model_name = model_name[len("gemini/"):]

    client = _get_gemini_client(api_key)

    # Rebuild config to include system_instruction and/or tools if needed
    if system_instruction or tools:
        existing = {}
        for attr in _CONFIG_ATTRS:
            val = getattr(config, attr, None)
            if val is not None:
                existing[attr] = val
        if system_instruction:
            existing["system_instruction"] = system_instruction
        if tools:
            existing["tools"] = tools
        config = types.GenerateContentConfig(**existing)

    call_kwargs = dict(
        model=model_name,
        contents=contents,
        config=config,
    )

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            stream = client.models.generate_content_stream(**call_kwargs)
            return stream
        except Exception as e:
            err_name = type(e).__name__.lower()
            status = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            is_transient = (
                'timeout' in err_name or
                'connection' in err_name or
                status in (429, 500, 502, 503, 504)
            )
            if is_transient and attempt < max_attempts:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise


def _build_response(response, model_name: str, duration_s: float, total_duration_s: float | None = None) -> AIResponse:
    """Convert Google SDK response to AIResponse."""
    # Strip prefix for display
    display_model = model_name.replace("gemini/", "")

    # Extract text content
    content = ""
    try:
        content = response.text or ""
    except (AttributeError, ValueError):
        # Some responses may not have .text (e.g., blocked)
        if response.candidates:
            parts = response.candidates[0].content.parts
            content = "".join(p.text for p in parts if hasattr(p, 'text') and p.text)

    # Token usage
    usage = getattr(response, 'usage_metadata', None)
    prompt_tokens = getattr(usage, 'prompt_token_count', None)
    completion_tokens = getattr(usage, 'candidates_token_count', None)
    total_tokens = getattr(usage, 'total_token_count', None)
    reasoning_tokens = getattr(usage, 'thoughts_token_count', None)

    # Finish reason
    finish_reason = None
    if response.candidates:
        fr = getattr(response.candidates[0], 'finish_reason', None)
        if fr is not None:
            finish_reason = str(fr)

    # Grounding metadata
    grounding_metadata = None
    if response.candidates:
        gm = getattr(response.candidates[0], 'grounding_metadata', None)
        if gm:
            grounding_metadata = gm

    # Safety ratings
    safety_ratings = None
    if response.candidates:
        sr = getattr(response.candidates[0], 'safety_ratings', None)
        if sr:
            safety_ratings = sr

    # Tool calls — convert function call parts to OpenAI format
    tool_calls = None
    if response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
            fc_parts = [
                p for p in candidate.content.parts
                if hasattr(p, 'function_call') and p.function_call is not None
            ]
            if fc_parts:
                tool_calls = []
                for p in fc_parts:
                    fc = p.function_call
                    args = getattr(fc, 'args', None)
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = json.dumps(dict(args)) if args else "{}"
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": args_str,
                        }
                    })

    # Compute cost from known pricing. Gemini 2.5+ caches implicitly and bills
    # cache hits at the discounted cached-input rate; cached tokens are
    # reported in usage_metadata.cached_content_token_count and are a subset
    # of prompt_token_count, so subtract them out before applying full rate.
    cached_tokens = getattr(usage, 'cached_content_token_count', None)
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        cached_tokens = 0  # None/absent (no cache hit or older model) -> flat pricing

    cost = None
    pricing = _PRICING.get(display_model)
    if pricing and prompt_tokens is not None and completion_tokens is not None:
        cached_tokens = min(cached_tokens, prompt_tokens)
        cached_rate = pricing.get("cached_input", pricing["input"])
        cost = ((prompt_tokens - cached_tokens) * pricing["input"] / 1_000_000 +
                cached_tokens * cached_rate / 1_000_000 +
                completion_tokens * pricing["output"] / 1_000_000)

    return AIResponse(
        content=content,
        model=display_model,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cost=cost,
        duration_seconds=duration_s,
        total_duration_seconds=total_duration_s,
        grounding_metadata=grounding_metadata,
        safety_ratings=safety_ratings,
        tool_calls=tool_calls,
        raw_response=response,
    )


def _print_response(response: AIResponse, model_name: str, verbosity: str, duration_s: float):
    """Print response details matching the format in utils._print_response_details."""
    if verbosity == 'none':
        return

    if verbosity in ('response', 'info', 'debug'):
        print(f"\nRESPONSE:\n\n{response.content}\n\n\n")

    if verbosity == 'debug':
        try:
            print("\n\n" + "-" * 32 + f"\nRAW RESPONSE:\n\n{response.raw_response}\n\n")
        except Exception:
            pass

    if verbosity in ('info', 'debug'):
        stats = {
            'Model': response.model + " (direct)",
            'Finish Reason': response.finish_reason,
            'Cost': response.cost,
            'Speed': duration_s,
            'Prompt Tokens': response.prompt_tokens,
            'Completion Tokens': response.completion_tokens,
            'Reasoning Tokens': response.reasoning_tokens,
            'Total Tokens': response.total_tokens,
        }

        if response.grounding_metadata:
            stats['Grounding'] = 'Yes'

        if response.tool_calls:
            stats['Tool Calls'] = len(response.tool_calls)

        print("\nRESPONSE INFO:\n")
        max_key_len = max(len(k) for k in stats.keys())
        for key, value in stats.items():
            if key == 'Cost' and value is not None:
                formatted_value = f"${value:.6f}"
            elif key == 'Speed' and value is not None:
                formatted_value = f"{value:.3f}s"
            else:
                formatted_value = str(value)

            note = " (sometimes part of completion)" if key == 'Reasoning Tokens' and value is not None else ""
            print(f"{key:>{max_key_len}} : {formatted_value}{note}")
        print("-" * 32 + "\n")
