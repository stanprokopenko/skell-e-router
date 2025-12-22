"""
Detailed streaming example for Gemini Deep Research Agent.

This example streams research results in real-time while saving everything
to files, including:
- A text file with the full report and metadata
- A JSON file with structured citations (resolved URLs)

Requirements:
    - GEMINI_API_KEY environment variable must be set
    - google-genai package must be installed (included in skell-e-router dependencies)
"""

import json
import os
from datetime import datetime

from skell_e_router import (
    ask_deep_research,
    DeepResearchError,
)

# Output directory for saved files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_detailed_streaming_research(query: str):
    """Run research with streaming, printing and saving all output to files."""
    
    # Step 1: Set up output directory and files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_txt = os.path.join(OUTPUT_DIR, f"deep_research_{timestamp}.txt")
    output_json = os.path.join(OUTPUT_DIR, f"deep_research_{timestamp}.json")
    
    print("\n" + "=" * 60)
    print("DETAILED STREAMING RESEARCH")
    print(f"Text output: {output_txt}")
    print(f"JSON output: {output_json}")
    print("=" * 60)
    
    # Open file for writing throughout the process
    with open(output_txt, "w", encoding="utf-8") as f:
        header = "=" * 60 + "\n"
        header += "DEEP RESEARCH STREAMING OUTPUT\n"
        header += f"Query: {query}\n"
        header += f"Started: {datetime.now().isoformat()}\n"
        header += "=" * 60 + "\n\n"
        f.write(header)
        print(header, end="")
        
        def on_progress(event_type: str, content: str):
            """Callback for streaming updates - prints and saves."""
            if event_type == "start":
                start_line = f"Research started: {content}\n"
                print(start_line)
                f.write(start_line + "\n")
                f.flush()
            elif event_type == "thought":
                thought_line = f"\n[THINKING] {content}\n"
                print(thought_line)
                f.write(thought_line + "\n")
                f.flush()
            elif event_type == "text":
                print(content, end="", flush=True)
                f.write(content)
                f.flush()
        
        try:
            # Step 2: Run the research (citations resolved automatically)
            result = ask_deep_research(
                query,
                stream=True,
                on_progress=on_progress,
                verbosity="none",
            )
            
            # Step 3: Write metadata
            meta = "\n\n" + "=" * 60 + "\n"
            meta += "METADATA\n"
            meta += "=" * 60 + "\n"
            meta += f"Interaction ID: {result.id}\n"
            meta += f"Status: {result.status}\n"
            meta += f"Duration: {result.duration_seconds:.1f} seconds\n"
            
            if result.usage:
                meta += "\n--- Token Usage ---\n"
                if result.usage.prompt_tokens:
                    meta += f"Prompt Tokens: {result.usage.prompt_tokens}\n"
                if result.usage.completion_tokens:
                    meta += f"Completion Tokens: {result.usage.completion_tokens}\n"
                if result.usage.total_tokens:
                    meta += f"Total Tokens: {result.usage.total_tokens}\n"
                if result.usage.total_cost:
                    meta += f"Total Cost: ${result.usage.total_cost:.6f}\n"
            
            print(meta)
            f.write(meta)
            
            # Step 4: Write parsed citations with resolved URLs and titles
            if result.parsed_citations:
                cit = "\n" + "=" * 60 + "\n"
                cit += f"CITATIONS ({len(result.parsed_citations)} sources)\n"
                cit += "=" * 60 + "\n"
                for c in result.parsed_citations:
                    cit += f"\n[{c.number}] {c.domain}\n"
                    if c.title:
                        cit += f"    Title: {c.title}\n"
                    cit += f"    URL: {c.url}\n"
                print(cit)
                f.write(cit)
            else:
                no_cit = "\n(No citations found in report)\n"
                print(no_cit)
                f.write(no_cit)
            
            # Step 5: Write full text with resolved URLs
            if result.text:
                full_text = "\n" + "=" * 60 + "\n"
                full_text += "FULL REPORT TEXT (with resolved URLs)\n"
                full_text += "=" * 60 + "\n"
                full_text += result.text + "\n"
                f.write(full_text)
            
            footer = "\n" + "=" * 60 + "\n"
            footer += f"Completed: {datetime.now().isoformat()}\n"
            footer += "=" * 60 + "\n"
            print(footer)
            f.write(footer)
            
            # Step 6: Save structured JSON output
            json_output = result.to_dict()
            json_output["query"] = query
            json_output["timestamp"] = timestamp
            
            with open(output_json, "w", encoding="utf-8") as jf:
                json.dump(json_output, jf, indent=2, ensure_ascii=False)
            
            print(f"\nText output saved to: {output_txt}")
            print(f"JSON output saved to: {output_json}")
            return result
            
        except DeepResearchError as err:
            error_msg = f"\nResearch failed: {err.code} - {err.message}\n"
            if err.details:
                error_msg += f"Details: {err.details}\n"
            print(error_msg)
            f.write(error_msg)
            return None


# MAIN EXECUTION
# --------------

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print("  export GEMINI_API_KEY='your-api-key'  # Linux/Mac")
        print("  set GEMINI_API_KEY=your-api-key      # Windows CMD")
        exit(1)
    
    QUERY = "What are the best practices for building a successful online course business in 2025? Focus on pricing strategies, student retention, and community building."
    
    run_detailed_streaming_research(QUERY)
