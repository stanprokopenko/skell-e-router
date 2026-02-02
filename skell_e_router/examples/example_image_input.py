"""Image input example — send an image to a vision-capable model.

Requirements:
    - GEMINI_API_KEY environment variable set
    - vision-test.jpg in the same directory as this script
"""

import os
import traceback
from skell_e_router import ask_ai, AIResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "vision-test.jpg")

try:
    # Basic usage — pass a local file path via the images parameter
    print("--- Image Input (basic) ---")
    content = ask_ai(
        "gemini-3-pro-preview",
        "What does this image say?",
        images=[IMAGE_PATH],
        verbosity="info",
    )
    print(f"Response: {content}")

    # Rich response — includes token usage, cost, and timing
    print("\n--- Image Input (rich response) ---")
    response = ask_ai(
        "gemini-3-pro-preview",
        "What does this image say?",
        images=[IMAGE_PATH],
        rich_response=True,
        verbosity="info",
    )
    print(f"Content: {response.content}")
    print(f"Model: {response.model}")
    print(f"Cost: ${response.cost:.6f}" if response.cost is not None else "Cost: None")
    print(f"Duration: {response.duration_seconds:.2f}s" if response.duration_seconds is not None else "Duration: None")
    print(f"Tokens: {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens}")

except Exception:
    traceback.print_exc()
