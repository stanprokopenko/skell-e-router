import litellm
import os
import json


# SETUP
#--------

# Enable debug logging for more detailed output
#litellm._turn_on_debug()

# Drop unsupported parameters automatically
litellm.drop_params = True

# Models known to support 'thinking' / 'reasoning_effort' parameters
THINKING_MODELS = [
    "gemini/gemini-2.5-flash-preview-04-17",
    "openai/o3"
]


# Checks if required environment variables are set.
def check_environment_variables():

    required_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]
    
    all_set = True
    for key in required_keys:
        if key not in os.environ:
            print(f"WARNING: Environment variable '{key}' not set.")
            all_set = False
    return all_set 


# Resolves a model alias or full name to the actual model name used by LiteLLM.
def resolve_model_alias(model_alias: str):

    MODEL_MAP = {
        "o3": "openai/o3",
        "gpt-4o": "openai/gpt-4o",
        "gemini-2.5-pro": "gemini/gemini-2.5-pro-preview-03-25",
        "gemini-2.5-pro-preview-03-25": "gemini/gemini-2.5-pro-preview-03-25",
        "gemini-2.5-flash": "gemini/gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash-preview-04-17": "gemini/gemini-2.5-flash-preview-04-17",
        "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet-20240620",
    }

    model_name = MODEL_MAP.get(model_alias) # Try resolving alias
    if not model_name:
        if model_alias in MODEL_MAP.values():
            model_name = model_alias # Identifier is a valid full model name
        else:
            # Identifier is neither a valid alias nor a known full model name
            print(f"ERROR: Invalid model identifier '{model_alias}'. Not found in known models.")
            return None
    return model_name


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
            print("ERROR: Invalid format in user_input list. Each item must be a dictionary with 'role' and 'content'.")
            return None
    else:
        print(f"ERROR: Invalid user_input type: {type(user_input)}. Must be string or list of dictionaries.")
        return None
    return messages


# Handles 'thinking' (derived from budget_tokens) and 'reasoning_effort' params
def _handle_thinking_params(model_name: str, kwargs: dict):
    is_gemini = model_name.startswith("gemini")
    is_anthropic = model_name.startswith("anthropic")

    '''For Gemini and Anthropic models, LiteLLM converts reasoning_effort → thinking_config with the mapping low→1024, medium→2048, high→4096 tokens'''
    if is_gemini or is_anthropic:
        # Gemini models: Prioritize budget_tokens if present
        if 'budget_tokens' in kwargs:
            budget = kwargs.pop('budget_tokens')
            think_type = "enabled" if budget > 0 else "disabled"
            kwargs['thinking'] = {"type": think_type, "budget_tokens": budget}
            # If budget_tokens is used, remove reasoning_effort to avoid potential conflicts
            kwargs.pop('reasoning_effort', None)
        # else: If only reasoning_effort is present (and budget_tokens is not), leave it.
    else:
        # Non-Gemini models: Remove Gemini-specific params
        kwargs.pop('budget_tokens', None)
        kwargs.pop('thinking', None)
        # Leave reasoning_effort for _filter_unsupported_params to handle for the specific model

    return kwargs


# Gathers statistics from a LiteLLM response and prints them based on level.
def _print_response_details(response, verbosity: str = 'info'):
    if verbosity == 'none':
        return
    
    model_name = getattr(response, 'model', 'UNKNOWN MODEL') # Get model name early
    
    # Print raw response only if debug level
    if verbosity == 'debug':
        print("\n\n\n\n" + "-" * 32 + f"\nRAW RESPONSE:\n\n{response}\n\n")

    usage = getattr(response, 'usage', None)
    completion_details = getattr(usage, 'completion_tokens_details', None)
    
    first_choice = response.choices[0] if response.choices else None
    message = getattr(first_choice, 'message', None) if first_choice else None

    # Initialize stats dictionary first
    stats = {
        'Model': getattr(response, 'model', 'UNKNOWN MODEL'), # Capitalized for printing
        'Finish Reason': getattr(first_choice, 'finish_reason', None),
        'Cost': litellm.completion_cost(completion_response=response),
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

    # Print response content for info and debug levels
    content = response.choices[0].message.content
    print(f"\nRESPONSE:\n\n{content}\n\n\n")
    
    # Print info for info and debug levels
    print("\nRESPONSE INFO:\n")
    # Calculate max_key_len *after* potentially adding safety keys
    max_key_len = max(len(k) for k in stats.keys()) 
    for key, value in stats.items():
        # Removed special handling for Safety Ratings

        # Format cost specifically
        if key == 'Cost' and value is not None:
            formatted_value = f"${value:.6f}"
        else:
            formatted_value = str(value) # Convert None to "None"
        
        # Add note for Reasoning Tokens
        note = " (part of completion)" if key == 'Reasoning Tokens' and value is not None else ""
        
        # Right-align the key within the max_key_len width
        print(f"{key:>{max_key_len}} : {formatted_value}{note}")
    print("-" * 32 + "\n")


# Adds safety settings to kwargs if the model is a Gemini model.
def _maybe_add_gemini_safety_settings(model_name: str, kwargs: dict):
    if model_name.startswith("gemini"):
        kwargs['safety_settings'] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    return kwargs


# Removes known unsupported parameters from kwargs based on the target model.
def _filter_unsupported_params(model_name: str, kwargs: dict):
    
    # candidate_count
    if not model_name.startswith("gemini"):
        kwargs.pop('candidate_count', None) # Gemini-specific
    
    # thinking / reasoning_effort
    if model_name not in THINKING_MODELS:
        kwargs.pop('reasoning_effort', None)
        kwargs.pop('thinking', None)

    # top_p and stop
    # Remove top_p for specific incompatible models (e.g., o3) or models using thinking params
    if model_name in THINKING_MODELS:
        kwargs.pop('top_p', None)
        kwargs.pop('stop', None)

    # Add more rules here as needed for different providers/models

    # temperature
    # Remove temperature for specific models (e.g., openai/o models)
    if model_name.startswith("openai/o"):
        kwargs.pop('temperature', None)

    return kwargs


# MAIN CALL FUNCTIONS
#--------------------

def ask_ai(model_alias: str, user_input: str | list[dict], system_message: str = None, verbosity: str = 'none', **kwargs):
    
    if not check_environment_variables(): # Verify required API keys are present
        exit(1)

    model_name = resolve_model_alias(model_alias)
    if not model_name:
        return None

    # Construct messages list
    messages = _construct_messages(user_input, system_message)
    if messages is None:
        return None

    # Add Gemini-specific safety settings if applicable
    kwargs = _maybe_add_gemini_safety_settings(model_name, kwargs)

    # Handle thinking/reasoning params (mutually exclusive)
    kwargs = _handle_thinking_params(model_name, kwargs)

    # Filter out known unsupported parameters for the target model
    kwargs = _filter_unsupported_params(model_name, kwargs)

    print(f"\n--- ASKING AI ({model_name}) ---\n")
    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            **kwargs
        )

        # Extract content
        content = response.choices[0].message.content

        # verbosity levels: "none", "info", "debug"
        _print_response_details(response, verbosity=verbosity) 

        return content

    except Exception as e:
        print(f"ERROR calling {model_name}: {e}")
        return None