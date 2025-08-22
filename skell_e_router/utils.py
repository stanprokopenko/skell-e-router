import litellm
import os
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .model_config import AIModel, MODEL_CONFIG

# SETUP
#--------

# Enable debug logging for more detailed output
#litellm._turn_on_debug()

# Drop unsupported parameters automatically
litellm.drop_params = True

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


# Checks if required environment variables are set.
def check_environment_variables():

    required_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]
    
    all_set = True
    for key in required_keys:
        if key not in os.environ:
            print(f"WARNING: Environment variable '{key}' not set.")
            all_set = False
    return all_set 


# Resolves a model alias (or full name) to its AIModel object.
def resolve_model_alias(model_alias: str) -> AIModel | None:
    ai_model = MODEL_CONFIG.get(model_alias)
    if not ai_model:
        print(f"ERROR: Invalid model alias '{model_alias}'. Not found in known models.")
        return None
    return ai_model


def _print_request_details(messages: list[dict], kwargs: dict, verbosity: str = 'none'):
    
    if verbosity == 'debug':
        # Print kwargs
        print(f"\nKWARGS:\n\n{json.dumps(kwargs, indent=4)}\n\n")

        # Print messages
        print(f"\nMESSAGES:\n\n{json.dumps(messages, indent=4)}\n\n")


# Gathers statistics from a LiteLLM response and prints them based on level
# 'none', 'response', 'info', 'debug'
def _print_response_details(response, verbosity: str = 'none'):
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

        # Initialize stats dictionary first
        stats = {
            'Model': model_name,
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


# Removes known unsupported parameters from kwargs based on the target model.
def _handle_model_specific_params(ai_model: AIModel, kwargs: dict):

    if "budget_tokens" in kwargs:
        budget = kwargs.pop("budget_tokens")

        # Path A: If "budget_tokens" in supported_params, transform to 'thinking' dict.
        if "budget_tokens" in ai_model.supported_params:
            think_type = "enabled" if budget > 0 else "disabled"
            kwargs['thinking'] = {"type": think_type, "budget_tokens": budget}
            kwargs.pop('reasoning_effort', None) # Prioritize 'thinking' dict.
        
        # Path B: Else, if "reasoning_effort" in supported_params, map to 'reasoning_effort' string.
        elif "reasoning_effort" in ai_model.supported_params:
            if budget > 0:
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
            
            budget_val = 0
            transformed_to_thinking = False

            if effort_value == "low":
                budget_val = 1024 
                transformed_to_thinking = True
            elif effort_value == "medium":
                budget_val = 2048
                transformed_to_thinking = True
            elif effort_value == "high":
                budget_val = 4096 
                transformed_to_thinking = True
            
            if transformed_to_thinking:
                kwargs['thinking'] = {"type": "enabled", "budget_tokens": budget_val}
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

    # Filter to include only parameters listed in model's supported_params.
    final_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in ai_model.supported_params
    }
    return final_kwargs


# MAIN CALL FUNCTIONS
#--------------------

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def ask_ai(model_alias: str, user_input: str | list[dict], system_message: str = None, verbosity: str = 'none', **kwargs):
    
    if not check_environment_variables(): # Verify required API keys are present
        exit(1)

    ai_model = resolve_model_alias(model_alias)
    if not ai_model:
        return None
    
    # Construct messages list
    messages = _construct_messages(user_input, system_message)
    if messages is None:
        return None

    # Swap and filter out parameters for the target model
    kwargs = _handle_model_specific_params(ai_model, kwargs)

    # Print Request Details
    _print_request_details(messages, kwargs, verbosity) 

    print(f"\nASKING AI ({ai_model.name})...\n\n")
    try:
        response = litellm.completion(
            model=ai_model.name,
            messages=messages,
            **kwargs
        )

        # Extract content
        content = response.choices[0].message.content

        # Print Response Details
        _print_response_details(response, verbosity) 

        return content

    except Exception as e:
        print(f"ERROR calling {ai_model.name}: {e}")
        return None