import litellm
import os
import json
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception
from .model_config import AIModel, MODEL_CONFIG

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

# HELPER FUNCTIONS
#-----------------

# Constructs the messages list for the AI call.
def _construct_messages(user_input: str | list[dict], system_message: str = None):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    if isinstance(user_input, str):
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
def check_environment_variables(verbosity: str = 'none'):
    missing = []
    for key in REQUIRED_ENV_KEYS:
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
            note = " (part of completion)" if key == 'Reasoning Tokens' and value is not None else ""
            
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
        # If "reasoning_effort" in kwargs but model supports token_budget and not reasoning_effort, map to thinking dict.
        if "budget_tokens" in ai_model.supported_params and "reasoning_effort" not in ai_model.supported_params:
            
            effort_value = kwargs.get("reasoning_effort")
            
            accepted_efforts = getattr(ai_model, 'accepted_reasoning_efforts', None)
            if accepted_efforts is None:
                accepted_efforts = {"low", "medium", "high"}

            if effort_value not in accepted_efforts:
                raise RouterError(
                    code="INVALID_PARAM",
                    message=f"'reasoning_effort' must be one of: {sorted(list(accepted_efforts))}"
                )

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

    # Filter to include only parameters listed in model's supported_params.
    final_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ai_model.supported_params
    }
    return final_kwargs


# MAIN CALL FUNCTIONS
#--------------------

@retry(
    retry=retry_if_exception(_is_retryable_exception),
    wait=_retry_after_wait,
    stop=stop_after_attempt(3)
)
def _perform_completion(model_name: str, messages: list[dict], **kwargs):
    return litellm.completion(
        model=model_name,
        messages=messages,
        **kwargs
    )


def ask_ai(model_alias: str, user_input: str | list[dict], system_message: str = None, verbosity: str = 'none', **kwargs):
    
    verbosity = verbosity.lower()
    if verbosity not in ['none', 'response', 'info', 'debug']:
        print(f"WARNING: Invalid verbosity '{verbosity}'. Must be 'none', 'response', 'info', or 'debug'.\nSetting to 'response'.")
        verbosity = 'response'
 
    # These helpers will raise RouterError on failure
    check_environment_variables(verbosity)
    ai_model = resolve_model_alias(model_alias)
    messages = _construct_messages(user_input, system_message)

    # Swap and filter out parameters for the target model
    kwargs = _handle_model_specific_params(ai_model, kwargs)

    _print_request_details(messages, kwargs, ai_model.name, verbosity) 

    try:
        start_time = time.perf_counter()
        response = _perform_completion(
            model_name=ai_model.name,
            messages=messages,
            **kwargs
        )
        end_time = time.perf_counter()
        request_duration_s = end_time - start_time

        content = response.choices[0].message.content
        _print_response_details(response, verbosity, request_duration_s) 
        return content

    except Exception as e:
        if verbosity != 'none':
            print(f"ERROR calling {ai_model.name}: {e}")
        raise RouterError(
            code="PROVIDER_ERROR",
            message=str(e),
            details={"provider": ai_model.provider, "model": ai_model.name}
        ) from e