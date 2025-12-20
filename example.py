from skell_e_router import ask_ai, RouterError


# MAIN EXECUTION
#---------------

if __name__ == "__main__":

    SYSTEM_MESSAGE = "you are a research assistant. always include sources with urls at the end of your response."
    PROMPT = "What are the last 10 videos released by Proko on YouTube?"
    MODEL = "gemini-3-pro-preview"

    try:
        ask_ai(
            MODEL,
            PROMPT, # accepts str OR list[dict] for conversation history
            SYSTEM_MESSAGE,
            verbosity='info', # 'none', 'response', 'info', 'debug'
            # KWARGS
            temperature=0.7,
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
            
            web_search_options={"search_context_size": "high"},
            # GOOGLE SEARCH GROUNDING (Gemini 2.0+ models only)
            # Enables the model to search the web for real-time information.
            # Options for search_context_size: "low", "medium", "high"
            # Returns grounding metadata with citations in the response.

        )
    except RouterError as err:
        # Pass upstream to your service / translate to your API error shape
        print({"code": err.code, "message": err.message, "details": err.details})

    print("\n--- Tests finished ---")