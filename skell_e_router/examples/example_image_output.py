"""Image output example — generate an image with nano-banana-3.

Requirements:
    - GEMINI_API_KEY environment variable set

The generated image is saved to examples/output/.
"""

import os
import base64
import traceback
from skell_e_router import ask_ai

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

try:
    print("--- Image Generation (nano-banana-3) ---")
    response = ask_ai(
        "nano-banana-3",
        "Generate a drawing of a skull in the style of Nicolai Fechin",
        rich_response=True,
        verbosity="info",
    )

    print(f"Content: {response.content}")
    print(f"Model: {response.model}")
    print(f"Cost: ${response.cost:.6f}" if response.cost is not None else "Cost: None")
    print(f"Duration: {response.duration_seconds:.2f}s" if response.duration_seconds is not None else "Duration: None")
    print(f"Tokens: {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens}")

    if response.images:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for i, img in enumerate(response.images):
            data_url = img.get("image_url", {}).get("url", "")
            if not data_url.startswith("data:"):
                print(f"Image {i}: unexpected format, skipping")
                continue

            # Strip the data URI prefix to get raw base64
            header, encoded = data_url.split(",", 1)
            # Determine extension from MIME type (e.g. "data:image/png;base64")
            mime = header.split(":")[1].split(";")[0]
            ext = mime.split("/")[1] if "/" in mime else "png"

            filename = f"generated_{i}.{ext}" if len(response.images) > 1 else f"generated.{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(base64.b64decode(encoded))
            print(f"Saved: {filepath}")
    else:
        print("No images returned in the response.")

except Exception:
    traceback.print_exc()
