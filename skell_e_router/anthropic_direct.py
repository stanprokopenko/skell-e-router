# ANTHROPIC DIRECT SDK
# --------------------
# Bypass LiteLLM for speed-critical Claude models by calling the anthropic SDK directly.
# This eliminates 0.3-1.7s of overhead per call that LiteLLM adds.

import time
import json
import base64
import hashlib
import logging
import re

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from .response import AIResponse

logger = logging.getLogger(__name__)


# Cache Anthropic client instances per API key to avoid repeated construction overhead
_client_cache: dict[str, object] = {}


def _get_anthropic_client(api_key: str):
    """Return a cached anthropic.Anthropic for the given API key."""
    if api_key not in _client_cache:
        _client_cache[api_key] = anthropic.Anthropic(api_key=api_key)
    return _client_cache[api_key]


# The Anthropic SDK refuses a non-streaming request whose max_tokens implies
# more than ten minutes of generation (about 21,333 tokens on a 128k-output
# model). Above that line the direct path streams under the hood and returns
# the assembled Message, so callers keep the plain-response contract while
# asking for a large output cap.
NONSTREAM_MAX_TOKENS = 21_333

# Known pricing per 1M tokens (USD) for direct-SDK models.
# Optional "cache_read" key overrides the standard 0.1x-of-input cache-read rate.
# Optional "long_context_threshold" plus "long_context_input_multiplier" /
# "long_context_output_multiplier" (same keys as AIModel.pricing) apply
# Anthropic's long-context rule: when the full prompt (uncached input + cache
# writes + cache reads) exceeds the threshold, the WHOLE request bills at the
# higher tier, cache multipliers included
# (https://platform.claude.com/docs/en/about-claude/pricing#long-context-pricing).
_PRICING = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},  # same prices as Sonnet 5
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25},  # cache reads 0.025x, not the standard 0.1x
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},  # launch intro price made permanent 2026-09-01 (increase to $3/$15 cancelled)
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    # $0.10/$0.50 for prompts up to 100k tokens; $0.50/$2.50 above that.
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "long_context_threshold": 100_000,
                         "long_context_input_multiplier": 5.0, "long_context_output_multiplier": 5.0},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def _tool_call_input(fn: dict) -> dict:
    """Parse an OpenAI function-call `arguments` value (JSON string or dict)
    into the dict Anthropic expects as tool_use `input`."""
    args = fn.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


# Anthropic rejects tool ids outside this pattern with a 400. Other providers
# are looser — most notably LiteLLM's Gemini path round-trips Gemini thought
# signatures by appending "__thought__<base64>" to the tool call id, so a
# conversation that ran a Gemini turn carries ids full of '/', '+' and '='
# that break any later Claude turn.
_VALID_TOOL_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_tool_id(tool_id):
    """Map a foreign tool call id to one Anthropic accepts.

    Deterministic (same input -> same output), so a tool_use block and its
    matching tool_result sanitize to the same id. Valid ids pass through
    untouched; invalid ones keep a readable prefix and gain a short hash of
    the original to preserve uniqueness."""
    if not isinstance(tool_id, str) or not tool_id or _VALID_TOOL_ID.match(tool_id):
        return tool_id
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", tool_id)[:40]
    digest = hashlib.sha1(tool_id.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{digest}" if cleaned else f"tool_{digest}"


_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")


def _merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages into one block-content message
    (tool_result conversion and retry prompts can produce user/user runs).

    Anthropic requires an assistant turn's thinking blocks to precede its text
    and tool_use blocks, so a merged assistant turn hoists every thinking block
    to the front, keeping their relative order; the blocks themselves are not
    modified (a modified or reordered signed block is rejected)."""
    merged: list[dict] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            for m in (prev, msg):
                if isinstance(m["content"], str):
                    m["content"] = [{"type": "text", "text": m["content"]}]
            blocks = prev["content"] + msg["content"]
            if msg["role"] == "assistant":
                thinking = [b for b in blocks if b.get("type") in _THINKING_BLOCK_TYPES]
                if thinking:
                    blocks = thinking + [b for b in blocks if b.get("type") not in _THINKING_BLOCK_TYPES]
            prev["content"] = blocks
        else:
            merged.append(msg)
    return merged


def _reasoning_blocks(msg: dict) -> list[dict]:
    """Anthropic thinking blocks replayed from an assistant message's
    ``reasoning_state`` (see ``AIResponse.reasoning_state``).

    Only entries tagged ``provider == "anthropic"`` are replayed; state from
    another provider means nothing here and is dropped (a transcript that
    switched providers). Blocks are rebuilt exactly as received, in order:
    the API verifies each signature against its text."""
    blocks = []
    for entry in msg.get("reasoning_state") or []:
        if not isinstance(entry, dict) or entry.get("provider") != "anthropic":
            continue
        kind = entry.get("kind")
        if kind == "thinking" and entry.get("signature"):
            blocks.append({"type": "thinking", "thinking": entry.get("text") or "",
                           "signature": entry["signature"]})
        elif kind == "redacted_thinking" and entry.get("data"):
            blocks.append({"type": "redacted_thinking", "data": entry["data"]})
    return blocks


def _convert_content_parts(content: list) -> list[dict]:
    """Convert OpenAI multi-part content (text + images) to Anthropic blocks.

    Caller-placed ``cache_control`` (e.g. skell-e-agent CachePoints) is kept on
    the converted block. Audio parts raise UNSUPPORTED_MODALITY; unknown part
    types are dropped.
    """
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        block = None
        if part.get("type") == "text":
            block = {"type": "text", "text": part["text"]}
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                # Parse data URI: data:mime;base64,DATA
                header, b64_data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64_data,
                    }
                }
            else:
                block = {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url,
                    }
                }
        if block is not None:
            # Preserve caller-placed cache breakpoints (e.g. skell-e-agent CachePoints)
            if "cache_control" in part:
                block["cache_control"] = part["cache_control"]
            parts.append(block)
            continue
        if part.get("type") == "input_audio":
            from .utils import RouterError
            raise RouterError(
                code="UNSUPPORTED_MODALITY",
                message=(
                    "Anthropic models do not support audio inputs. "
                    "Use a Gemini or OpenAI audio-capable model."
                ),
            )
    return parts


def _tool_result_block(msg: dict) -> dict:
    """OpenAI ``role: tool`` message -> Anthropic ``tool_result`` block.

    Content may be a plain string or an OpenAI block list (skell-e-agent sends
    a list when a CachePoint leads the next part); the list is converted
    block-by-block, never stringified. Anthropic documents ``cache_control``
    on the tool_result block itself, not on its inner content, so a marker on
    the message or on any inner block is hoisted onto the tool_result block
    (the last inner marker wins).
    """
    content = msg.get("content")
    block: dict = {
        "type": "tool_result",
        "tool_use_id": _sanitize_tool_id(msg.get("tool_call_id")),
    }
    cache_control = msg.get("cache_control")
    if isinstance(content, str):
        block["content"] = content
    elif isinstance(content, list):
        parts = _convert_content_parts(content)
        for part in parts:
            inner = part.pop("cache_control", None)
            if inner is not None:
                cache_control = inner
        block["content"] = parts if parts else ""
    elif content is None:
        block["content"] = ""
    else:
        block["content"] = str(content)
    if cache_control is not None:
        block["cache_control"] = cache_control
    return block


def _convert_messages_for_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Handles tool-calling turns: assistant `tool_calls` become tool_use
    content blocks, and role "tool" results become tool_result blocks in a
    user message. Returns (system_prompt, messages) where system_prompt is
    extracted from system messages and messages is the list of conversation
    turns.
    """
    system_prompt = None
    converted = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str):
                system_prompt = content
            continue

        # OpenAI tool result -> tool_result block in a user message
        if role == "tool":
            converted.append({
                "role": "user",
                "content": [_tool_result_block(msg)],
            })
            continue

        # Assistant turns replay their own thinking blocks first (Anthropic
        # requires thinking before text and tool_use), then text, then tool_use.
        thinking_blocks = _reasoning_blocks(msg) if role == "assistant" else []

        # Assistant tool calls -> tool_use content blocks (text first, if any)
        if role == "assistant" and msg.get("tool_calls"):
            parts = list(thinking_blocks)
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                parts.append({
                    "type": "tool_use",
                    "id": _sanitize_tool_id(tc.get("id")),
                    "name": fn.get("name"),
                    "input": _tool_call_input(fn),
                })
            converted.append({"role": "assistant", "content": parts})
            continue

        # Roles: "user" and "assistant" pass through directly
        if isinstance(content, str):
            if thinking_blocks:
                parts = list(thinking_blocks)
                if content:
                    parts.append({"type": "text", "text": content})
                converted.append({"role": role, "content": parts})
            else:
                converted.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = thinking_blocks + _convert_content_parts(content)
            if parts:
                converted.append({"role": role, "content": parts})
        elif thinking_blocks:
            converted.append({"role": role, "content": thinking_blocks})

    # Parallel tool results (and retry prompts after them) arrive as separate
    # messages — Anthropic wants them as blocks of a single user turn.
    return system_prompt, _merge_consecutive_roles(converted)


def _raise_ttls_before_1h(marked: list[dict]) -> bool:
    """Enforce Anthropic's non-increasing cache TTL rule over ``marked``.

    ``marked`` is every dict carrying a ``cache_control`` key, in request
    order (system first, then message blocks). Anthropic rejects a request
    (HTTP 400) when a 5m breakpoint precedes a 1h one, so every marker ahead
    of a 1h marker is raised to 1h. Raising keeps the caller's intent (the 1h
    marker already holds that whole prefix for an hour; the shorter prefix
    simply joins it) where lowering would silently shorten their cache. A new
    dict is written so a caller's shared marker object is never mutated.

    Returns True when any marker is 1h (so the router's own system
    breakpoint must be 1h too).
    """
    seen_1h = False
    for holder in reversed(marked):
        cc = holder.get("cache_control")
        if not isinstance(cc, dict):
            continue
        if cc.get("ttl") == "1h":
            seen_1h = True
        elif seen_1h:
            holder["cache_control"] = {**cc, "ttl": "1h"}
    return seen_1h


def _apply_cache_control(system_prompt: str | None, messages: list[dict]) -> tuple[str | list[dict] | None, list[dict]]:
    """Add prompt-caching breakpoints to an Anthropic-format request.

    Places one breakpoint on the system prompt and one on the last content
    block of the last message, so the system prompt and the conversation
    prefix are cached across requests within Anthropic's cache TTL (5 min).
    Caller-placed breakpoints suppress the last-message one. Anthropic
    requires TTLs in non-increasing order (tools -> system -> messages) and
    returns 400 otherwise, so the router raises every marker ahead of a 1h
    caller marker, including its own system breakpoint, to 1h
    (see _raise_ttls_before_1h).

    Anthropic bills the first (cache-write) request at 1.25x input price and
    cache reads at 0.1x, so this is opt-in per call — worth it for multi-turn
    loops and repeated large system prompts, not for one-shot calls.
    """
    # Caller already placed breakpoints (e.g. via skell-e-agent CachePoints):
    # respect their placement, don't add the last-message one on top.
    marked_blocks = [
        block
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict) and "cache_control" in block
    ]
    caller_marked = bool(marked_blocks)

    system_cc: dict = {"type": "ephemeral"}
    if _raise_ttls_before_1h(marked_blocks):
        system_cc["ttl"] = "1h"

    if isinstance(system_prompt, str) and system_prompt:
        system_prompt = [{
            "type": "text",
            "text": system_prompt,
            "cache_control": system_cc,
        }]

    if messages and not caller_marked:
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
        elif isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}

    return system_prompt, messages


def _build_create_params(ai_model, kwargs: dict) -> tuple[dict, dict | None]:
    """Convert router kwargs to Anthropic messages.create params.

    Returns (params, extra_headers).
    """
    from .utils import RouterError

    params = {}
    extra_headers = None

    # max_tokens is required by Anthropic. Default to the model's published
    # output cap (Models API) so nothing is capped below what the model can
    # write; 4096 only for entries without a known cap (retired models).
    params["max_tokens"] = kwargs.get(
        "max_tokens", getattr(ai_model, "max_output_tokens", None) or 4096)

    # Direct pass-through params — skip any that the model doesn't accept
    # (Opus 4.7 rejects temperature/top_p/top_k with a 400 error).
    for param in ("temperature", "top_p", "top_k"):
        if param in kwargs and param in ai_model.supported_params:
            params[param] = kwargs[param]

    # stop -> stop_sequences (ensure list)
    if "stop" in kwargs:
        stop = kwargs["stop"]
        if isinstance(stop, str):
            stop = [stop]
        params["stop_sequences"] = stop

    # --- Thinking / reasoning handling ---
    # Priority: budget_tokens > thinking dict > reasoning_effort

    if "budget_tokens" in kwargs:
        budget = kwargs["budget_tokens"]
        if budget > 0:
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            params["thinking"] = {"type": "disabled"}

    elif "thinking" in kwargs:
        thinking = kwargs["thinking"]
        if isinstance(thinking, dict):
            # Pass through as-is (enabled/disabled/adaptive + budget_tokens)
            params["thinking"] = thinking

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

        # If model supports reasoning_effort natively -> adaptive thinking + effort
        if "reasoning_effort" in ai_model.supported_params:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": effort}
        # If model supports budget_tokens -> map to budget
        elif "budget_tokens" in ai_model.supported_params:
            budget_map = {"low": 1024, "medium": 2048, "high": 4096}
            budget_val = budget_map.get(effort, 2048)
            if budget_val > 0:
                params["thinking"] = {"type": "enabled", "budget_tokens": budget_val}
            else:
                params["thinking"] = {"type": "disabled"}

    # Anthropic constraint: when thinking active, force temperature=1 and drop top_p < 0.95
    # — but only if the model supports those sampling params (Opus 4.7 rejects them outright).
    thinking_cfg = params.get("thinking")
    if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") in ("enabled", "adaptive"):
        if "temperature" in ai_model.supported_params:
            params["temperature"] = 1
        top_p_val = params.get("top_p")
        if isinstance(top_p_val, (int, float)) and top_p_val < 0.95:
            params.pop("top_p")

    # tools -> convert OpenAI format: parameters key -> input_schema key
    if "tools" in kwargs and kwargs["tools"]:
        anthropic_tools = []
        for tool_def in kwargs["tools"]:
            if isinstance(tool_def, dict) and tool_def.get("type") == "function":
                fn = tool_def["function"]
                tool = {"name": fn["name"]}
                if "description" in fn:
                    tool["description"] = fn["description"]
                if "parameters" in fn:
                    tool["input_schema"] = fn["parameters"]
                anthropic_tools.append(tool)
        if anthropic_tools:
            params["tools"] = anthropic_tools

    # tool_choice conversion. Models that restrict tool_choice (Fable 5.1
    # returns 400 on forced tool use) get unsupported values coerced to "auto"
    # here, mirroring the accepted_tool_choices coercion on the LiteLLM path —
    # that one runs in _handle_model_specific_params, which direct-SDK calls
    # never reach. A dict/named function choice matches the sentinel "named".
    if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
        tc = kwargs["tool_choice"]
        accepted = getattr(ai_model, "accepted_tool_choices", None)
        if accepted is not None:
            tc_value = tc if isinstance(tc, str) else "named"
            if tc_value not in accepted and tc != "auto":
                tc = "auto"
        if isinstance(tc, str):
            tc_map = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}
            mapped = tc_map.get(tc)
            if mapped:
                params["tool_choice"] = mapped
        elif isinstance(tc, dict) and tc.get("type") == "function":
            fn_name = tc.get("function", {}).get("name")
            if fn_name:
                params["tool_choice"] = {"type": "tool", "name": fn_name}

    # Manual (budget_tokens) extended thinking supports only tool_choice auto
    # or none; a forced choice returns 400. Adaptive thinking accepts it.
    # https://platform.claude.com/docs/en/build-with-claude/extended-thinking
    # ("Tool choice limitation")
    thinking_cfg = params.get("thinking")
    forced = params.get("tool_choice", {}).get("type") in ("any", "tool")
    if forced and isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled":
        logger.warning(
            "%s: tool_choice %r is incompatible with extended thinking "
            "(budget_tokens); sending tool_choice auto instead.",
            ai_model.name, kwargs.get("tool_choice"))
        params["tool_choice"] = {"type": "auto"}

    # betas -> extra_headers
    if "betas" in kwargs and kwargs["betas"]:
        betas = kwargs["betas"]
        if isinstance(betas, list):
            betas = ",".join(betas)
        extra_headers = {"anthropic-beta": betas}

    return params, extra_headers


def _call_anthropic_direct(model_name: str, messages: list[dict], system_prompt: str | None,
                           params: dict, extra_headers: dict | None, api_key: str) -> object:
    """Call the Anthropic SDK with retry on transient errors.

    Retries up to 3 attempts on timeout, connection, rate limit, and 5xx errors.
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic package is required for direct Anthropic SDK calls. "
            "Install it with: pip install anthropic"
        )

    # Strip "anthropic/" prefix used by LiteLLM
    if model_name.startswith("anthropic/"):
        model_name = model_name[len("anthropic/"):]

    client = _get_anthropic_client(api_key)

    call_kwargs = dict(model=model_name, messages=messages, **params)
    if system_prompt:
        call_kwargs["system"] = system_prompt
    if extra_headers:
        call_kwargs["extra_headers"] = extra_headers

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            request_start = time.perf_counter()
            if (call_kwargs.get("max_tokens") or 0) > NONSTREAM_MAX_TOKENS:
                with client.messages.stream(**call_kwargs) as stream:
                    response = stream.get_final_message()
            else:
                response = client.messages.create(**call_kwargs)
            request_duration = time.perf_counter() - request_start
            return response, request_duration
        except Exception as e:
            if _is_transient_anthropic_error(e) and attempt < max_attempts:
                wait_time = 2 ** attempt  # 2s, 4s
                time.sleep(wait_time)
                continue
            raise


def _call_anthropic_direct_stream(model_name: str, messages: list[dict], system_prompt: str | None,
                                  params: dict, extra_headers: dict | None, api_key: str):
    """Call the Anthropic SDK with streaming. Returns a stream context manager.

    Same retry logic as _call_anthropic_direct (retry before first event).
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic package is required for direct Anthropic SDK calls. "
            "Install it with: pip install anthropic"
        )

    # Strip "anthropic/" prefix used by LiteLLM
    if model_name.startswith("anthropic/"):
        model_name = model_name[len("anthropic/"):]

    client = _get_anthropic_client(api_key)

    call_kwargs = dict(model=model_name, messages=messages, **params)
    if system_prompt:
        call_kwargs["system"] = system_prompt
    if extra_headers:
        call_kwargs["extra_headers"] = extra_headers

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            stream = client.messages.stream(**call_kwargs)
            return stream
        except Exception as e:
            if _is_transient_anthropic_error(e) and attempt < max_attempts:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise


def _is_transient_anthropic_error(e: Exception) -> bool:
    """Check if an Anthropic SDK exception is transient and retryable."""
    err_name = type(e).__name__
    # Retry on RateLimitError, InternalServerError, OverloadedError (529, a
    # direct APIStatusError subclass, not an InternalServerError),
    # APITimeoutError, APIConnectionError
    if err_name in ("RateLimitError", "InternalServerError", "OverloadedError",
                    "APITimeoutError", "APIConnectionError"):
        return True
    # Generic checks
    err_lower = err_name.lower()
    if 'timeout' in err_lower or 'connection' in err_lower:
        return True
    status = getattr(e, 'status_code', None)
    if status in (429, 500, 502, 503, 504, 529):
        return True
    return False


def _build_response(response, model_name: str, duration_s: float, total_duration_s: float | None = None) -> AIResponse:
    """Convert Anthropic SDK response to AIResponse."""
    # Strip prefix for display
    display_model = model_name.replace("anthropic/", "")

    # Extract text content (join all TextBlock.text parts). Thinking blocks are
    # not part of the text; they are kept, verbatim and in order, as
    # reasoning_state so the caller can replay them on the next turn (the API
    # verifies each block's signature against its text).
    content_parts = []
    tool_calls = None
    reasoning_state = []

    for block in getattr(response, 'content', []):
        block_type = getattr(block, 'type', None)
        if block_type == "text":
            content_parts.append(block.text)
        elif block_type == "tool_use":
            if tool_calls is None:
                tool_calls = []
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                }
            })
        elif block_type == "thinking":
            reasoning_state.append({
                "provider": "anthropic", "kind": "thinking",
                "text": getattr(block, 'thinking', None) or "",
                "signature": getattr(block, 'signature', None),
            })
        elif block_type == "redacted_thinking":
            reasoning_state.append({
                "provider": "anthropic", "kind": "redacted_thinking",
                "data": getattr(block, 'data', None),
            })

    content = "".join(content_parts)

    # Token usage. With prompt caching enabled, Anthropic reports cached
    # tokens in separate fields that are NOT included in input_tokens:
    # cache_creation_input_tokens (writes: 1.25x input for the 5-minute TTL,
    # 2.0x for the 1-hour TTL) and cache_read_input_tokens (0.1x input unless
    # the pricing entry sets "cache_read").
    usage = getattr(response, 'usage', None)
    prompt_tokens = getattr(usage, 'input_tokens', None)
    completion_tokens = getattr(usage, 'output_tokens', None)

    def _int_attr(obj, attr: str) -> int | None:
        value = getattr(obj, attr, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    cache_creation_tokens = _int_attr(usage, 'cache_creation_input_tokens') or 0
    cache_read_tokens = _int_attr(usage, 'cache_read_input_tokens') or 0

    # Per-TTL write breakdown (usage.cache_creation). When absent, every write
    # is priced at the 5-minute 1.25x rate.
    breakdown = getattr(usage, 'cache_creation', None)
    write_5m = _int_attr(breakdown, 'ephemeral_5m_input_tokens')
    write_1h = _int_attr(breakdown, 'ephemeral_1h_input_tokens')
    if write_5m is None and write_1h is None:
        write_5m, write_1h = cache_creation_tokens, 0
    else:
        write_5m, write_1h = write_5m or 0, write_1h or 0

    # Report the full prompt size (uncached + cache-write + cache-read)
    if prompt_tokens is not None:
        prompt_tokens += cache_creation_tokens + cache_read_tokens

    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens is not None or completion_tokens is not None else None

    # Finish reason
    finish_reason = getattr(response, 'stop_reason', None)

    # Compute cost from known pricing
    cost = None
    pricing = _PRICING.get(display_model)
    if pricing and prompt_tokens is not None and completion_tokens is not None:
        threshold = pricing.get("long_context_threshold")
        long_context = threshold is not None and prompt_tokens > threshold
        input_mult = pricing.get("long_context_input_multiplier", 1.0) if long_context else 1.0
        output_mult = pricing.get("long_context_output_multiplier", 1.0) if long_context else 1.0
        input_rate = pricing["input"] * input_mult
        output_rate = pricing["output"] * output_mult
        cache_read_rate = pricing.get("cache_read", pricing["input"] * 0.10) * input_mult
        uncached_tokens = prompt_tokens - cache_creation_tokens - cache_read_tokens
        cost = (uncached_tokens * input_rate / 1_000_000 +
                write_5m * input_rate * 1.25 / 1_000_000 +
                write_1h * input_rate * 2.0 / 1_000_000 +
                cache_read_tokens * cache_read_rate / 1_000_000 +
                completion_tokens * output_rate / 1_000_000)

    return AIResponse(
        content=content,
        model=display_model,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        duration_seconds=duration_s,
        total_duration_seconds=total_duration_s,
        tool_calls=tool_calls,
        raw_response=response,
        cache_read_tokens=_int_attr(usage, 'cache_read_input_tokens'),
        cache_write_tokens=_int_attr(usage, 'cache_creation_input_tokens'),
        reasoning_state=reasoning_state or None,
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
            'Total Tokens': response.total_tokens,
        }

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
            print(f"{key:>{max_key_len}} : {formatted_value}")
        print("-" * 32 + "\n")
