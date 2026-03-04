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


# Known pricing per 1M tokens (USD) for direct-SDK models
# Per 1M tokens (USD), text/image/video input — source: https://ai.google.dev/gemini-api/docs/pricing
_PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
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
            if parts:
                contents.append(types.Content(role=sdk_role, parts=parts))

    return system_instruction, contents


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
            # Map budget to effort level, then fall through to reasoning_effort handling
            accepted = getattr(ai_model, 'accepted_reasoning_efforts', {"low", "medium", "high"})
            if budget == 0:
                effort = "minimal" if "minimal" in accepted else "low"
            elif budget <= 1024:
                effort = "low"
            elif budget <= 2048:
                effort = "medium"
            else:
                effort = "high"
            # Apply as reasoning_effort -> ThinkingConfig
            budget_map = {
                "minimal": 0,
                "low": 1024,
                "medium": 2048,
                "high": 4096,
            }
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget_map.get(effort, 1024)
            )

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
        budget_map = {
            "minimal": 0,
            "low": 1024,
            "medium": 2048,
            "high": 4096,
        }
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=budget_map.get(effort, 1024)
        )

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
                    fd_kwargs["parameters"] = fn["parameters"]
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

    client = genai.Client(api_key=api_key)

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
            response = client.models.generate_content(**call_kwargs)
            return response
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

    client = genai.Client(api_key=api_key)

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


def _build_response(response, model_name: str, duration_s: float) -> AIResponse:
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
        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
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
        reasoning_tokens=reasoning_tokens,
        cost=cost,
        duration_seconds=duration_s,
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
