# ANTHROPIC DIRECT SDK
# --------------------
# Bypass LiteLLM for speed-critical Claude models by calling the anthropic SDK directly.
# This eliminates 0.3-1.7s of overhead per call that LiteLLM adds.

import time
import json
import base64

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from .response import AIResponse


# Known pricing per 1M tokens (USD) for direct-SDK models
_PRICING = {
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}


def _convert_messages_for_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, messages) where system_prompt is extracted
    from system messages and messages is the list of conversation turns.
    """
    system_prompt = None
    converted = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            if isinstance(content, str):
                system_prompt = content
            continue

        # Roles: "user" and "assistant" pass through directly
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multi-part content (text + images)
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # Parse data URI: data:mime;base64,DATA
                            header, b64_data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64_data,
                                }
                            })
                        else:
                            parts.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                }
                            })
            if parts:
                converted.append({"role": role, "content": parts})

    return system_prompt, converted


def _build_create_params(ai_model, kwargs: dict) -> tuple[dict, dict | None]:
    """Convert router kwargs to Anthropic messages.create params.

    Returns (params, extra_headers).
    """
    from .utils import RouterError

    params = {}
    extra_headers = None

    # max_tokens (required by Anthropic, default 4096)
    params["max_tokens"] = kwargs.get("max_tokens", 4096)

    # Direct pass-through params
    for param in ("temperature", "top_p", "top_k"):
        if param in kwargs:
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

        # If model supports reasoning_effort natively -> adaptive thinking
        if "reasoning_effort" in ai_model.supported_params:
            params["thinking"] = {"type": "adaptive"}
        # If model supports budget_tokens -> map to budget
        elif "budget_tokens" in ai_model.supported_params:
            budget_map = {"low": 1024, "medium": 2048, "high": 4096}
            budget_val = budget_map.get(effort, 2048)
            if budget_val > 0:
                params["thinking"] = {"type": "enabled", "budget_tokens": budget_val}
            else:
                params["thinking"] = {"type": "disabled"}

    # Anthropic constraint: when thinking active, force temperature=1 and drop top_p < 0.95
    thinking_cfg = params.get("thinking")
    if isinstance(thinking_cfg, dict) and thinking_cfg.get("type") in ("enabled", "adaptive"):
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

    # tool_choice conversion
    if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
        tc = kwargs["tool_choice"]
        if isinstance(tc, str):
            tc_map = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}
            mapped = tc_map.get(tc)
            if mapped:
                params["tool_choice"] = mapped
        elif isinstance(tc, dict) and tc.get("type") == "function":
            fn_name = tc.get("function", {}).get("name")
            if fn_name:
                params["tool_choice"] = {"type": "tool", "name": fn_name}

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

    client = anthropic.Anthropic(api_key=api_key)

    call_kwargs = dict(model=model_name, messages=messages, **params)
    if system_prompt:
        call_kwargs["system"] = system_prompt
    if extra_headers:
        call_kwargs["extra_headers"] = extra_headers

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.create(**call_kwargs)
            return response
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

    client = anthropic.Anthropic(api_key=api_key)

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
    # Retry on RateLimitError, InternalServerError, APITimeoutError, APIConnectionError
    if err_name in ("RateLimitError", "InternalServerError", "APITimeoutError", "APIConnectionError"):
        return True
    # Generic checks
    err_lower = err_name.lower()
    if 'timeout' in err_lower or 'connection' in err_lower:
        return True
    status = getattr(e, 'status_code', None)
    if status in (429, 500, 502, 503, 504):
        return True
    return False


def _build_response(response, model_name: str, duration_s: float) -> AIResponse:
    """Convert Anthropic SDK response to AIResponse."""
    # Strip prefix for display
    display_model = model_name.replace("anthropic/", "")

    # Extract text content (join all TextBlock.text parts, skip thinking blocks)
    content_parts = []
    tool_calls = None

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
        # Skip "thinking" blocks — they're internal

    content = "".join(content_parts)

    # Token usage
    usage = getattr(response, 'usage', None)
    prompt_tokens = getattr(usage, 'input_tokens', None)
    completion_tokens = getattr(usage, 'output_tokens', None)
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens is not None or completion_tokens is not None else None

    # Finish reason
    finish_reason = getattr(response, 'stop_reason', None)

    # Compute cost from known pricing
    cost = None
    pricing = _PRICING.get(display_model)
    if pricing and prompt_tokens is not None and completion_tokens is not None:
        cost = (prompt_tokens * pricing["input"] / 1_000_000 +
                completion_tokens * pricing["output"] / 1_000_000)

    return AIResponse(
        content=content,
        model=display_model,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        duration_seconds=duration_s,
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
