"""Check status of an existing Deep Research interaction."""

import json
import os
from skell_e_router import get_research_status, process_citations

INTERACTION_ID = "v1_ChdpZHRKYWEtVEpzV2N6N0lQMDhhRGlRYxIXaWR0SmFhLVRKc1djejdJUDA4YURpUWM"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "skell_e_router", "examples", "output")


if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY not set")
        exit(1)
    
    print(f"Checking interaction: {INTERACTION_ID[:50]}...")
    
    result = get_research_status(INTERACTION_ID)
    
    print(f"\nStatus: {result.status}")
    print(f"ID: {result.id}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.text:
        print(f"\nReport length: {len(result.text)} characters")
        
        # Process citations if completed
        if result.status == "completed":
            result = process_citations(result)
            print(f"Citations: {len(result.parsed_citations)}")
            
            # Save the report
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Save markdown report
            md_path = os.path.join(OUTPUT_DIR, "proko_research_recovered.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(result.text)
            print(f"\nSaved report to: {md_path}")
            
            # Save JSON with citations
            json_path = os.path.join(OUTPUT_DIR, "proko_research_recovered.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"Saved JSON to: {json_path}")
    else:
        print("\nNo text output yet")
    
    if result.usage and result.usage.total_tokens:
        print(f"Tokens: {result.usage.total_tokens}")

