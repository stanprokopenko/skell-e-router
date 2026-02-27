import litellm
import os
import json
import time
import base64
import mimetypes
from typing import overload, Literal
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception
from .model_config import AIModel, MODEL_CONFIG
from .response import AIResponse

# SETUP
#--------

# Enable debug logging for more detailed output
#litellm._turn_on_debug()

# Drop unsupported parameters automatically
litellm.drop_params = True

# ERRORS
#--------

class RouterError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")


# CONSTANTS
#-----------

REQUIRED_ENV_KEYS = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "XAI_API_KEY"]
FALLBACK_WAIT = wait_random_exponential(min=1, max=10)

# Maps model provider names to their corresponding environment variable key names.
# Config dict keys use the lowercase form (e.g., "openai_api_key").
PROVIDER_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
}


# HELPER FUNCTIONS
#-----------------

def _encode_image(source: str) -> dict:
    """Convert an image source (URL, data URI, or file path) to an OpenAI-format content part."""
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    if source.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": source}}
    # File path -- read, detect MIME, base64 encode
    if not os.path.isfile(source):
        raise RouterError(
            code="INVALID_INPUT",
            message=f"Image file not found: {source}"
        )
    mime_type, _ = mimetypes.guess_type(source)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(source, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


# Constructs the messages list for the AI call.
def _construct_messages(user_input: str | list[dict], system_message: str = None, images: list[str] | None = None):
    messages = []

    if images and not isinstance(user_input, str):
        raise RouterError(
            code="INVALID_INPUT",
            message="'images' parameter is only supported when 'user_input' is a string. "
                    "For list input, embed image content parts directly in your messages."
        )

    if system_message:
        messages.append({"role": "system", "content": system_message})

    if isinstance(user_input, str):
        if images:
            content_parts = [{"type": "text", "text": user_input}]
            for img in images:
                content_parts.append(_encode_image(img))
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_input})
    elif isinstance(user_input, list):
        # Ensure all items in the list are dictionaries with 'role' and 'content'
        if all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in user_input):
            messages.extend(user_input)
        else:
            raise RouterError(
                code="INVALID_INPUT",
                message="user_input items must be dicts with 'role' and 'content'"
            )
    else:
        raise RouterError(
            code="INVALID_INPUT",
            message=f"user_input must be a string or list of dicts with 'role' and 'content', got {type(user_input)}"
        )
    return messages


def _resolve_api_key(ai_model: "AIModel", config: dict | None) -> str | None:
    """Get the API key for a model's provider from a config dict, if available."""
    if not config:
        return None
    env_key = PROVIDER_ENV_KEY.get(ai_model.provider)
    if env_key:
        return config.get(env_key.lower())
    return None


def _redact_keys(message: str, config: dict | None) -> str:
    """Remove any config values from a string to prevent accidental key leakage."""
    if not config:
        return message
    for value in config.values():
        if isinstance(value, str) and value:
            message = message.replace(value, "[REDACTED]")
    return message


# Classify retryable errors and honor Retry-After when available
def _extract_status_and_headers(exc: Exception) -> tuple[int | None, dict]:
    status = getattr(exc, 'status_code', None)
    headers = getattr(exc, 'headers', None)
    if headers is None:
        resp = getattr(exc, 'response', None)
        if resp is not None:
            status = getattr(resp, 'status_code', status)
            headers = getattr(resp, 'headers', None)
    if headers is None:
        headers = {}
    return status, headers


def _parse_retry_after_seconds(headers: dict) -> float | None:
    if not isinstance(headers, dict):
        return None
    retry_after = None
    for k, v in headers.items():
        if str(k).lower() == 'retry-after':
            retry_after = v
            break
    if retry_after is None:
        return None
    try:
        seconds = float(str(retry_after).strip())
        if seconds >= 0:
            return seconds
    except Exception:
        return None
    return None


def _is_quota_related(exc: Exception) -> bool:
    code = getattr(exc, 'code', None)
    if code is None:
        error = getattr(exc, 'error', None)
        if isinstance(error, dict):
            code = error.get('code') or error.get('type')
    if isinstance(code, str):
        c = code.lower()
        return ('quota' in c) or ('insufficient_quota' in c) or ('quota_exceeded' in c)
    return False


def _is_retryable_exception(exc: Exception) -> bool:
    status, headers = _extract_status_and_headers(exc)

    name = exc.__class__.__name__.lower()
    if 'timeout' in name or 'connection' in name or 'connect' in name:
        return True

    if status is None:
        return False

    if status in (500, 502, 503, 504):
        # Optional: respect very large Retry-After on 503 if present
        ra = _parse_retry_after_seconds(headers)
        if ra is not None and ra > 120:
            return False
        return True

    if status == 429:
        if _is_quota_related(exc):
            return False
        ra = _parse_retry_after_seconds(headers)
        if ra is not None and ra > 120:
            return False
        return True

    return False


def _retry_after_wait(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return FALLBACK_WAIT(retry_state)

    status, headers = _extract_status_and_headers(exc)
    if status in (429, 503):
        seconds = _parse_retry_after_seconds(headers)
        if seconds is not None and seconds >= 0:
            return min(seconds, 120.0)
        return FALLBACK_WAIT(retry_state)

    return FALLBACK_WAIT(retry_state)


# Checks if required environment variables are set.
# Keys already supplied via config are skipped.
def check_environment_variables(verbosity: str = 'none', config: dict | None = None):
    config = config or {}
    missing = []
    for key in REQUIRED_ENV_KEYS:
        # If the caller provided this key via config, no env var needed
        if key.lower() in config:
            continue
        if key not in os.environ:
            if verbosity != 'none':
                print(f"WARNING: Environment variable '{key}' not set.")
            missing.append(key)
    if missing:
        raise RouterError(
            code="MISSING_ENV",
            message="Required environment variables are not set.",
            details={"required": REQUIRED_ENV_KEYS, "missing": missing}
        )
    return True


def _check_provider_key(ai_model: "AIModel", config: dict | None = None, verbosity: str = 'none'):
    """Check that the API key for the given model's provider is available (via config or env)."""
    env_key = PROVIDER_ENV_KEY.get(ai_model.provider)
    if not env_key:
        return  # Unknown provider, let litellm handle it

    config_key = env_key.lower()

    # If provided in config, we're good
    if config and config_key in config:
        return

    # Otherwise, check env var
    if env_key not in os.environ:
        if verbosity != 'none':
            print(f"WARNING: Environment variable '{env_key}' not set.")
        raise RouterError(
            code="MISSING_ENV",
            message=f"API key for provider '{ai_model.provider}' is not available.",
            details={"required": env_key, "provider": ai_model.provider}
        )


# Resolves a model alias (or full name) to its AIModel object.
def resolve_model_alias(model_alias: str) -> AIModel:
    ai_model = MODEL_CONFIG.get(model_alias)
    if not ai_model:
        raise RouterError(
            code="INVALID_MODEL",
            message=f"Invalid model alias '{model_alias}'."
        )
    return ai_model


def _print_request_details(messages: list[dict], kwargs: dict, model_name: str, verbosity: str = 'none'):
    
    if verbosity == 'debug':
        # Print kwargs
        print(f"\nKWARGS:\n\n{json.dumps(kwargs, indent=4)}\n\n")

        # Print messages
        print(f"\nMESSAGES:\n\n{json.dumps(messages, indent=4)}\n\n")
    
    if verbosity != 'none':
        print(f"\nASKING AI ({model_name})...\n\n")


# Gathers statistics from a LiteLLM response and prints them based on level
# 'none', 'response', 'info', 'debug'
def _print_response_details(response, verbosity: str = 'none', request_duration_s: float | None = None):
    if verbosity == 'none':
        return
    
    # Print Raw Response
    if verbosity == 'debug':
        try:
            # Attempt to pretty-print using Pydantic's model_dump_json if available
            pretty_response = response.model_dump_json(indent=4)
        except AttributeError:
            # Fallback for objects that don't have model_dump_json (e.g., older LiteLLM versions or other types)
            try:
                pretty_response = json.dumps(vars(response), indent=4, default=str) # Fallback to vars()
            except TypeError: # vars() might fail on some objects
                pretty_response = str(response) # Safest fallback
        print("\n\n" + "-" * 32 + f"\nRAW RESPONSE:\n\n{pretty_response}\n\n")

    # Print Response Content
    if verbosity == 'response' or verbosity == 'info' or verbosity == 'debug':

        content = response.choices[0].message.content
        print(f"\nRESPONSE:\n\n{content}\n\n\n")

    # Print Response Info / Stats
    if verbosity == 'info' or verbosity == 'debug':

        model_name = getattr(response, 'model', 'UNKNOWN MODEL')

        usage = getattr(response, 'usage', None)
        completion_details = getattr(usage, 'completion_tokens_details', None)
        
        first_choice = response.choices[0] if response.choices else None
        message = getattr(first_choice, 'message', None) if first_choice else None

        # Compute cost safely – some models might not be mapped in LiteLLM's cost table
        try:
            computed_cost = litellm.completion_cost(completion_response=response)
        except Exception:
            computed_cost = None

        # Initialize stats dictionary
        stats = {
            'Model': model_name,
            'Finish Reason': getattr(first_choice, 'finish_reason', None),
            'Cost': computed_cost,
            'Speed': request_duration_s,
            'Prompt Tokens': getattr(usage, 'prompt_tokens', None),
            'Completion Tokens': getattr(usage, 'completion_tokens', None),
            'Reasoning Tokens': getattr(completion_details, 'reasoning_tokens', None),
            'Total Tokens': getattr(usage, 'total_tokens', None),
            'Tool Calls': getattr(message, 'tool_calls', None),
            'Function Call': getattr(message, 'function_call', None),
            'Provider Specific Fields': getattr(message, 'provider_specific_fields', None),
            # Safety ratings will be added below if applicable
        }

        # Extract and add Safety Ratings directly to stats if available and model is Gemini
        safety_results = getattr(response, 'vertex_ai_safety_results', None)
        if model_name.startswith("gemini") and safety_results and isinstance(safety_results, list) and safety_results:
            # Access the inner list of ratings
            ratings_list = safety_results[0] if isinstance(safety_results[0], list) else safety_results
            for rating in ratings_list:
                category = rating.get('category', 'Unknown Category')
                probability = rating.get('probability', 'Unknown Probability')
                # Add each safety rating as a separate key
                stats[f"Safety: {category}"] = probability

        # Print info for info and debug levels
        print("\nRESPONSE INFO:\n")
        # Calculate max_key_len *after* potentially adding safety keys
        max_key_len = max(len(k) for k in stats.keys()) 
        for key, value in stats.items():
            # Removed special handling for Safety Ratings

            # Format cost specifically
            if key == 'Cost' and value is not None:
                formatted_value = f"${value:.6f}"
            elif key == 'Speed' and value is not None:
                formatted_value = f"{value:.3f}s"
            else:
                formatted_value = str(value) # Convert None to "None"
            
            # Add note for Reasoning Tokens
            note = " (sometimes part of completion)" if key == 'Reasoning Tokens' and value is not None else ""
            
            # Right-align the key within the max_key_len width
            print(f"{key:>{max_key_len}} : {formatted_value}{note}")
        print("-" * 32 + "\n")


# Removes known unsupported parameters from kwargs based on the target model.
def _handle_model_specific_params(ai_model: AIModel, kwargs: dict):

    if "budget_tokens" in kwargs:
        # Validate type to avoid TypeError on comparisons
        if not isinstance(kwargs.get("budget_tokens"), (int, float)):
            raise RouterError(
                code="INVALID_PARAM",
                message="'budget_tokens' must be a number",
                details={"received_type": str(type(kwargs.get("budget_tokens")))}
            )
        budget = kwargs.pop("budget_tokens")

        # Path A: If "budget_tokens" in supported_params, transform to 'thinking' dict.
        if "budget_tokens" in ai_model.supported_params:
            think_type = "enabled" if budget > 0 else "disabled"
            kwargs['thinking'] = {"type": think_type, "budget_tokens": budget}
            kwargs.pop('reasoning_effort', None) # Prioritize 'thinking' dict.
        
        # Path B: Else, if "reasoning_effort" in supported_params, map to 'reasoning_effort' string.
        elif "reasoning_effort" in ai_model.supported_params:
            accepted_efforts = getattr(ai_model, 'accepted_reasoning_efforts', {"low", "medium", "high"})
            if budget == 0:
                kwargs['reasoning_effort'] = "minimal" if "minimal" in accepted_efforts else "low"
            elif budget > 0:
                if budget <= 1024:
                    kwargs['reasoning_effort'] = "low"
                elif budget <= 2048:
                    kwargs['reasoning_effort'] = "medium"
                else:
                    kwargs['reasoning_effort'] = "high"
            kwargs.pop('thinking', None) # Ensure no conflicting 'thinking' dict.\
    elif "reasoning_effort" in kwargs:
        effort_value = kwargs.get("reasoning_effort")
        accepted_efforts = getattr(ai_model, 'accepted_reasoning_efforts', None)
        if accepted_efforts is None:
            accepted_efforts = {"low", "medium", "high"}

        if effort_value not in accepted_efforts:
            raise RouterError(
                code="INVALID_PARAM",
                message=f"'reasoning_effort' must be one of: {sorted(list(accepted_efforts))}"
            )

        # Path A: model supports reasoning_effort natively (e.g., Opus 4.6) — pass through
        # and auto-inject adaptive thinking if no thinking dict was provided.
        if "reasoning_effort" in ai_model.supported_params:
            if ai_model.is_anthropic and "thinking" not in kwargs:
                kwargs["thinking"] = {"type": "adaptive"}

        # Path B: model supports budget_tokens but not reasoning_effort — map to thinking dict.
        elif "budget_tokens" in ai_model.supported_params:
            budget_val = 0
            transformed_to_thinking = False

            if effort_value == "minimal":
                budget_val = 0
                transformed_to_thinking = True
            elif effort_value == "low":
                budget_val = 1024
                transformed_to_thinking = True
            elif effort_value == "medium":
                budget_val = 2048
                transformed_to_thinking = True
            elif effort_value == "high":
                budget_val = 4096
                transformed_to_thinking = True

            if transformed_to_thinking:
                think_type = "enabled" if budget_val > 0 else "disabled"
                kwargs['thinking'] = {"type": think_type, "budget_tokens": budget_val}
                kwargs.pop('reasoning_effort')
            # If not transformed, 'reasoning_effort' stays for final filtering.
    
    # Groq: LiteLLM doesn't natively support reasoning_effort for Groq,
    # so we force it through via allowed_openai_params.
    if ai_model.is_groq and "reasoning_effort" in kwargs:
        existing = kwargs.get("allowed_openai_params", [])
        if "reasoning_effort" not in existing:
            kwargs["allowed_openai_params"] = existing + ["reasoning_effort"]

    # Groq Compound: ensure correct model-version header is sent to enable tools
    if ai_model.is_groq and (ai_model.name.endswith("/compound") or ai_model.name.endswith("/compound-mini")):
        groq_header_key = "Groq-Model-Version"
        header_value = "latest"

        # Prefer LiteLLM's extra_headers; fall back to headers
        extra_headers = kwargs.get("extra_headers")
        headers = kwargs.get("headers")

        if isinstance(extra_headers, dict):
            if groq_header_key not in extra_headers:
                extra_headers[groq_header_key] = header_value
            kwargs["extra_headers"] = extra_headers
        elif isinstance(headers, dict):
            if groq_header_key not in headers:
                headers[groq_header_key] = header_value
            kwargs["headers"] = headers
        else:
            # Create using the more common LiteLLM arg name
            kwargs["extra_headers"] = {groq_header_key: header_value}

        GROQ_COMPOUND_DEFAULT_TOOLS = [
            "browser_automation",
            "web_search",
            "code_interpreter",
            "visit_website",
        ]

        # If caller didn't specify tools, default to all Groq Compound tools
        if "compound_custom" not in kwargs:
            kwargs["compound_custom"] = {
                "tools": {"enabled_tools": GROQ_COMPOUND_DEFAULT_TOOLS}
            }

    # Add safety settings
    if "safety_settings" in ai_model.supported_params:
        kwargs['safety_settings'] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    # Anthropic: when extended thinking is enabled, temperature must be 1
    if ai_model.is_anthropic:
        thinking_cfg = kwargs.get('thinking')
        if isinstance(thinking_cfg, dict) and thinking_cfg.get('type') == 'enabled':
            if 'temperature' in ai_model.supported_params:
                kwargs['temperature'] = 1
            # top_p must be >= 0.95 or unset when thinking is enabled
            top_p_val = kwargs.get('top_p')
            if isinstance(top_p_val, (int, float)):
                if top_p_val < 0.95:
                    kwargs['top_p'] = 1

    # Auto-inject modalities for image generation models
    if "modalities" in ai_model.supported_params and "modalities" not in kwargs:
        kwargs["modalities"] = ["text", "image"]

    # Filter to include only parameters listed in model's supported_params.
    # Also allow LiteLLM meta-params that control param forwarding behavior.
    PASSTHROUGH_KEYS = {"allowed_openai_params"}
    final_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ai_model.supported_params or key in PASSTHROUGH_KEYS
    }
    return final_kwargs


# MAIN CALL FUNCTIONS
#--------------------

@retry(
    retry=retry_if_exception(_is_retryable_exception),
    wait=_retry_after_wait,
    stop=stop_after_attempt(3)
)
def _perform_completion(model_name: str, messages: list[dict], api_key: str | None = None, **kwargs):
    completion_kwargs = dict(model=model_name, messages=messages, **kwargs)
    if api_key:
        completion_kwargs["api_key"] = api_key
    return litellm.completion(**completion_kwargs)


def _build_ai_response(
    response,
    request_duration_s: float | None = None
) -> AIResponse:
    """Build AIResponse from LiteLLM completion response."""
    
    content = response.choices[0].message.content if response.choices else ""
    model_name = getattr(response, 'model', 'unknown')
    
    usage = getattr(response, 'usage', None)
    completion_details = getattr(usage, 'completion_tokens_details', None) if usage else None
    first_choice = response.choices[0] if response.choices else None
    message = getattr(first_choice, 'message', None) if first_choice else None
    
    # Compute cost safely
    try:
        computed_cost = litellm.completion_cost(completion_response=response)
    except Exception:
        computed_cost = None
    
    # Extract grounding metadata for Gemini
    grounding_metadata = None
    if hasattr(response, 'vertex_ai_grounding_metadata'):
        grounding_metadata = response.vertex_ai_grounding_metadata
    
    # Extract safety ratings for Gemini
    safety_ratings = None
    if hasattr(response, 'vertex_ai_safety_results'):
        safety_ratings = response.vertex_ai_safety_results

    # Extract generated images (e.g., from Gemini image generation models)
    response_images = None
    if message is not None:
        raw_images = getattr(message, 'images', None)
        if raw_images:
            response_images = raw_images

    return AIResponse(
        content=content or "",
        model=model_name,
        finish_reason=getattr(first_choice, 'finish_reason', None),
        prompt_tokens=getattr(usage, 'prompt_tokens', None),
        completion_tokens=getattr(usage, 'completion_tokens', None),
        total_tokens=getattr(usage, 'total_tokens', None),
        reasoning_tokens=getattr(completion_details, 'reasoning_tokens', None),
        cost=computed_cost,
        duration_seconds=request_duration_s,
        grounding_metadata=grounding_metadata,
        safety_ratings=safety_ratings,
        images=response_images,
        tool_calls=getattr(message, 'tool_calls', None),
        function_call=getattr(message, 'function_call', None),
        provider_specific_fields=getattr(message, 'provider_specific_fields', None),
        raw_response=response,
    )


@overload
def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    rich_response: Literal[False] = False,
    config: dict | None = None,
    images: list[str] | None = None,
    **kwargs
) -> str: ...

@overload
def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    rich_response: Literal[True] = ...,
    config: dict | None = None,
    images: list[str] | None = None,
    **kwargs
) -> AIResponse: ...

def ask_ai(model_alias: str, user_input: str | list[dict], system_message: str = None, verbosity: str = 'none', rich_response: bool = False, config: dict | None = None, images: list[str] | None = None, **kwargs) -> str | AIResponse:

    verbosity = verbosity.lower()
    if verbosity not in ['none', 'response', 'info', 'debug']:
        print(f"WARNING: Invalid verbosity '{verbosity}'. Must be 'none', 'response', 'info', or 'debug'.\nSetting to 'response'.")
        verbosity = 'response'

    # These helpers will raise RouterError on failure
    ai_model = resolve_model_alias(model_alias)
    _check_provider_key(ai_model, config, verbosity)
    messages = _construct_messages(user_input, system_message, images=images)

    # Resolve API key from config (None falls back to litellm env var lookup)
    api_key = _resolve_api_key(ai_model, config)

    # Swap and filter out parameters for the target model
    kwargs = _handle_model_specific_params(ai_model, kwargs)

    _print_request_details(messages, kwargs, ai_model.name, verbosity)

    try:
        start_time = time.perf_counter()
        response = _perform_completion(
            model_name=ai_model.name,
            messages=messages,
            api_key=api_key,
            **kwargs
        )
        end_time = time.perf_counter()
        request_duration_s = end_time - start_time

        content = response.choices[0].message.content
        _print_response_details(response, verbosity, request_duration_s)

        if rich_response:
            return _build_ai_response(response, request_duration_s)
        return content

    except Exception as e:
        safe_msg = _redact_keys(str(e), config)
        if verbosity != 'none':
            print(f"ERROR calling {ai_model.name}: {safe_msg}")
        raise RouterError(
            code="PROVIDER_ERROR",
            message=safe_msg,
            details={"provider": ai_model.provider, "model": ai_model.name}
        ) from e