from skell_e_router import ask_ai


# MAIN EXECUTION
#---------------

if __name__ == "__main__":
    # Test/Execution-Specific Parameters
    SYSTEM_MESSAGE = "You are funny"
    USER_PROMPT = "write a joke about Proko"
    MODELS = ["o3"]

    for model in MODELS:

        # Example passing kwargs for temperature and max_tokens
        ask_ai(
            model,
            USER_PROMPT,
            SYSTEM_MESSAGE,
            verbosity='debug',
            temperature=0.5, 
            max_tokens=5000,
            top_p=0.9,
            stop=None,
            stream=False,
            response_format={"type": "text"},
            candidate_count=1,
            reasoning_effort="high",
            budget_tokens=8000 # Use reasoning_effort because most models suport is and litellm maps it to thinking_config
        )

    print("\n--- Tests finished ---")