from skell_e_router import ask_ai, AIResponse
import traceback

try:
    # Basic usage - backwards compatible
    print("--- Basic Usage ---")
    content = ask_ai("gemini-2.5-flash", "What is 2+2?")
    print(f"Basic: {content}")

    # Rich response
    print("\n--- Rich Response ---")
    response = ask_ai(
        "gemini-2.5-flash",
        "What is the latest news?",
        rich_response=True,
        web_search_options={"search_context_size": "high"}
    )

    print(f"Content: {response.content[:100]}...")
    print(f"Model: {response.model}")
    print(f"Cost: ${response.cost:.6f}" if response.cost is not None else "Cost: None")
    print(f"Duration: {response.duration_seconds:.2f}s" if response.duration_seconds is not None else "Duration: None")
    print(f"Tokens: {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens}")

    if response.grounding_metadata:
        print("\nGrounding Metadata Found:")
        # Handle potential list or dict structure depending on LiteLLM version
        metadata = response.grounding_metadata
        if isinstance(metadata, list) and len(metadata) > 0:
            chunks = metadata[0].get('groundingChunks', [])
        elif isinstance(metadata, dict):
            chunks = metadata.get('groundingChunks', [])
        else:
            chunks = []
            
        print(f"Sources: {len(chunks)}")
        for chunk in chunks[:3]:
            print(f"  - {chunk.get('web', {}).get('title', 'Unknown')}")
    else:
        print("\nNo grounding metadata found.")

except Exception:
    traceback.print_exc()
