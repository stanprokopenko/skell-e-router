"""Audio input example — send an audio clip to a multimodal model.

Requirements:
    - GEMINI_API_KEY environment variable set
    - "Ba Dum Outro.mp3" in the same directory as this script

Anthropic models will raise UNSUPPORTED_MODALITY.
"""

import os
import traceback
from skell_e_router import ask_ai, AIResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(SCRIPT_DIR, "Ba Dum Outro.mp3")

try:
    print("--- Audio Input (basic) ---")
    content = ask_ai(
        "gemini-3.1-flash-lite-preview",
        "describe this sound effect",
        audio=[AUDIO_PATH],
        verbosity="info",
    )
    print(f"Response: {content}")

    print("\n--- Audio Input (rich response) ---")
    response = ask_ai(
        "gemini-3.1-flash-lite-preview",
        "how long is this sound",
        audio=[AUDIO_PATH],
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
