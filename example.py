from skell_e_router import ask_ai


# MAIN EXECUTION
#---------------

if __name__ == "__main__":

    SYSTEM_MESSAGE = "You are funny"
    PROMPT = "write a joke about Proko."
    MODELS = ["gpt-5"]

    for model in MODELS:

        ask_ai(
            model,
            PROMPT, # accepts str OR list[dict] for conversation history
            SYSTEM_MESSAGE,
            verbosity='response', # 'none', 'response', 'info', 'debug'
            # KWARGS
            temperature=0, 
            max_tokens=5000,
            top_p=0.9,
            stop=None,
            stream=False,
            response_format={"type": "text"},
            candidate_count=1,
            #reasoning_effort="high",
            budget_tokens=8000 # Currently supported by Gemini-2.5-flash
            # Use reasoning_effort because most models suport it and litellm maps it to thinking_config
            # Budget conversion: <=1024=low, <=2048=medium, >2048=high
        )

    print("\n--- Tests finished ---")