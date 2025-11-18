from skell_e_router import ask_ai, RouterError


# MAIN EXECUTION
#---------------

if __name__ == "__main__":

    SYSTEM_MESSAGE = "You are funny."
    PROMPT = "write a joke about Proko."
    MODEL = "gemini-3-pro-preview"

    try:
        ask_ai(
            MODEL,
            PROMPT, # accepts str OR list[dict] for conversation history
            SYSTEM_MESSAGE,
            verbosity='debug', # 'none', 'response', 'info', 'debug'
            # KWARGS
            temperature=1, 
            max_tokens=5000,
            max_completion_tokens=2048,
            top_p=1,
            stop=None,
            stream=False,
            response_format={"type": "text"},
            candidate_count=1,
            reasoning_effort="low",
            # compound_custom={"tools": {"enabled_tools": ["visit_website"]}},   # Used only for Groq's compound model.
            #budget_tokens=8000   # Budget conversion: <=1024=low, <=2048=medium, >2048=high
            # Use reasoning_effort because most models suport it and litellm maps it to thinking_config
        )
    except RouterError as err:
        # Pass upstream to your service / translate to your API error shape
        print({"code": err.code, "message": err.message, "details": err.details})

    print("\n--- Tests finished ---")