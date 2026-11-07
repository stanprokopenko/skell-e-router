# GEMINI DIRECT SDK
# -----------------
# Bypass LiteLLM for speed-critical Gemini models by calling the google-genai SDK directly.
# This eliminates 0.3-1.7s of overhead per call that LiteLLM adds.

import base64
import enum
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
    # gemini-4-argon: launch rates through 2026-11-30; doubles to 4.00/20.00/0.20 on 2026-12-01
    "gemini-4-argon": {"input": 2.00, "output": 10.00, "cached_input": 0.10},
    # gemini-3.8-flash: introductory rates through 2026-12-31; doubles to 1.50/7.50/0.15 on 2027-01-01
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cached_input": 0.075},
    # gemini-3.7-flash: introductory rates through 2026-12-31; doubles to 1.50/7.50/0.15 on 2027-01-01
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75, "cached_input": 0.075},
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50, "cached_input": 0.15},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached_input": 0.15},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached_input": 0.05},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cached_input": 0.025},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached_input": 0.01},
}


def _content_to_parts(content) -> list:
    """Convert OpenAI message content (str or list of parts) to SDK Parts.

    Empty or whitespace-only text never becomes a Part: Gemini rejects an empty
    text part with a 400 ("empty text parameter").
    """
    if isinstance(content, str):
        return [types.Part.from_text(text=content)] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text") or ""
            if text.strip():
                parts.append(types.Part.from_text(text=text))
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                # Parse data URI: data:mime;base64,DATA
                header, b64_data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
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
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(audio_data),
                mime_type=mime,
            ))
    return parts


def _function_call_args(arguments) -> dict:
    """Parse an OpenAI tool_call `arguments` value (JSON string or dict) into
    the dict Gemini expects as functionCall.args."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_result_text(content) -> str:
    """Text of an OpenAI tool message (str, or the text parts of a list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text") or "" for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return "" if content is None else str(content)


# Gemini 3 rejects a model turn whose first functionCall part has no
# thought_signature. History without one (another provider's turn, an older
# transcript) gets this documented placeholder, as pydantic-ai's GoogleModel does:
# https://ai.google.dev/gemini-api/docs/thought-signatures#faqs
SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


def _decode_thought_signature(value) -> bytes | None:
    """Return the raw bytes of a tool_call's base64 ``thought_signature``, or None.

    Accepts standard or URL-safe base64 (the SDK's own JSON uses URL-safe).
    """
    if isinstance(value, bytes):
        return value or None
    if not isinstance(value, str) or not value:
        return None
    try:
        value = value.replace("-", "+").replace("_", "/")
        return base64.b64decode(value, validate=True) or None
    except ValueError:
        return None


def _convert_messages_to_contents(messages: list[dict]) -> tuple[str | None, list]:
    """Convert OpenAI-format messages to Google SDK format.

    Returns (system_instruction, contents) where system_instruction is extracted
    from system messages and contents is the list of conversation turns.

    Tool rounds map to Gemini's native function calling:
    - An assistant message's ``tool_calls`` become ``functionCall`` parts on the
      model turn, after any text, carrying the OpenAI call id as
      ``functionCall.id``. A tool call's base64 ``thought_signature`` rides on
      its part as bytes; the first call of a turn without one gets
      ``SKIP_THOUGHT_SIGNATURE`` (Gemini 3 requires a signature there).
    - A run of consecutive ``role: "tool"`` messages becomes ONE user turn of
      ``functionResponse`` parts in message order, each carrying the
      ``tool_call_id`` as ``id`` and the name of the call it answers. Gemini
      expects all responses to one model turn in the very next user turn.
    - The response payload is ``{"output": <tool text>}``, the key the SDK
      documents for function output (FunctionResponse.response).
    A user message after the function responses stays its own user turn. The
    SDK does not merge or reject adjacent same-role Content objects, and the
    Gemini API accepts them.
    """
    system_instruction = None
    contents = []
    call_names: dict[str, str] = {}  # tool_call_id -> function name
    response_parts: list = []  # functionResponse parts of the current tool run
    extra_parts: list = []  # tool-run parts that cannot be functionResponses

    def flush_tool_run():
        if response_parts:
            contents.append(types.Content(role="user", parts=list(response_parts)))
        if extra_parts:
            contents.append(types.Content(role="user", parts=list(extra_parts)))
        response_parts.clear()
        extra_parts.clear()

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "tool":
            call_id = msg.get("tool_call_id")
            text = _tool_result_text(content)
            name = call_names.get(call_id) or msg.get("name")
            if name:
                response_parts.append(types.Part(function_response=types.FunctionResponse(
                    id=call_id, name=name, response={"output": text},
                )))
            elif text.strip():
                # No preceding call with this id (e.g. trimmed history) and no
                # name on the message. Gemini rejects a functionResponse that
                # does not answer a functionCall in the previous turn, so the
                # result degrades to labeled user text instead of a guessed name.
                label = f"[tool result {call_id}]" if call_id else "[tool result]"
                extra_parts.append(types.Part.from_text(text=f"{label}\n{text}"))
            # Media in a list-form tool result rides in a following user turn.
            if isinstance(content, list):
                extra_parts.extend(
                    p for p in _content_to_parts(content)
                    if getattr(p, "text", None) is None
                )
            continue

        flush_tool_run()

        if role == "system":
            # Google SDK takes system instruction separately via config
            if isinstance(content, str):
                system_instruction = content
            continue

        parts = _content_to_parts(content)
        if role == "assistant":
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                call_id = tc.get("id")
                name = fn.get("name")
                if call_id and name:
                    call_names[call_id] = name
                signature = _decode_thought_signature(tc.get("thought_signature"))
                if signature is None and i == 0:
                    signature = SKIP_THOUGHT_SIGNATURE
                parts.append(types.Part(
                    function_call=types.FunctionCall(
                        id=call_id, name=name, args=_function_call_args(fn.get("arguments")),
                    ),
                    thought_signature=signature,
                ))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif parts:
            contents.append(types.Content(role="user", parts=parts))

    flush_tool_run()
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
        # Gemini requires a non-empty `items` on every array schema. A bare
        # `list` type hint yields items-less OR empty-items arrays depending
        # on the generator (pydantic v2 emits {"items": {}, "type": "array"};
        # google.genai serializes an empty items schema to nothing, so the
        # API 400s "items: missing field" either way).
        if str(out.get("type", "")).lower() == "array" and not out.get("items"):
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
        # thinking_budget for models that declare budget_tokens. thinking_level
        # is a Gemini 3 field (earlier models return 400), so only models that
        # declare reasoning_effort get it; anything else gets a budget too.
        if ("budget_tokens" in ai_model.supported_params
                or "reasoning_effort" not in ai_model.supported_params):
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


def _prompt_block_finish_reason(response) -> str | None:
    """Recognize prompt refusals when Gemini supplies no candidate."""
    feedback = getattr(response, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None)
    value = reason.name if isinstance(reason, enum.Enum) else reason
    if value in {"SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT",
                 "IMAGE_SAFETY", "MODEL_ARMOR", "JAILBREAK"}:
        return "content_filter"
    return None


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
            # FinishReason is a str enum whose str() is "FinishReason.MAX_TOKENS";
            # return the bare API value ("STOP", "MAX_TOKENS", "SAFETY", ...).
            finish_reason = fr.name if isinstance(fr, enum.Enum) else str(fr)
    else:
        finish_reason = _prompt_block_finish_reason(response)

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
                    # Keep Gemini's own id so the next turn's functionCall and
                    # functionResponse can echo it; mint one only when absent.
                    call_id = getattr(fc, 'id', None) or f"call_{uuid.uuid4().hex[:24]}"
                    tool_call = {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": args_str,
                        }
                    }
                    # Gemini 3 requires this signature back on the same
                    # functionCall part next request; base64 keeps it JSON-safe.
                    signature = getattr(p, 'thought_signature', None)
                    if isinstance(signature, bytes) and signature:
                        tool_call["thought_signature"] = base64.b64encode(signature).decode()
                    tool_calls.append(tool_call)

    # Compute cost from known pricing. Gemini 2.5+ caches implicitly and bills
    # cache hits at the discounted cached-input rate; cached tokens are
    # reported in usage_metadata.cached_content_token_count and are a subset
    # of prompt_token_count, so subtract them out before applying full rate.
    cached_tokens = getattr(usage, 'cached_content_token_count', None)
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        cached_tokens = 0  # None/absent (no cache hit or older model) -> flat pricing
    # Reported as-is (None when Gemini sent no count). Gemini has no cache-write
    # charge on this path (implicit caching), so cache_write_tokens stays None.
    reported_cache_read = getattr(usage, 'cached_content_token_count', None)
    if not isinstance(reported_cache_read, int):
        reported_cache_read = None

    # Thinking tokens (usage_metadata.thoughts_token_count) are reported
    # separately from candidates_token_count but billed at the output rate.
    billed_output = completion_tokens
    if isinstance(reasoning_tokens, int) and reasoning_tokens > 0 and completion_tokens is not None:
        billed_output = completion_tokens + reasoning_tokens

    cost = None
    pricing = _PRICING.get(display_model)
    if pricing and prompt_tokens is not None and completion_tokens is not None:
        cached_tokens = min(cached_tokens, prompt_tokens)
        cached_rate = pricing.get("cached_input", pricing["input"])
        cost = ((prompt_tokens - cached_tokens) * pricing["input"] / 1_000_000 +
                cached_tokens * cached_rate / 1_000_000 +
                billed_output * pricing["output"] / 1_000_000)

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
        cache_read_tokens=reported_cache_read,
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
