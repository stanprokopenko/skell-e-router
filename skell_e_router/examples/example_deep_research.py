"""
Example usage of Gemini Deep Research Agent via skell-e-router.

Deep Research is an autonomous agent that plans, searches, reads, and synthesizes
information to produce detailed, cited reports. Research tasks can take several
minutes to complete.

Requirements:
    - GEMINI_API_KEY environment variable must be set
    - google-genai package must be installed (included in skell-e-router dependencies)
"""

import json
import os

from skell_e_router import (
    ask_deep_research,
    deep_research_follow_up,
    DeepResearchError,
)

# Output directory for saved files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_basic_research():
    """Basic research with polling - citations are resolved automatically."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Research")
    print("=" * 60)
    
    try:
        # Citations are resolved automatically (resolve_citations=True by default)
        result = ask_deep_research(
            "Research the history and evolution of Google TPUs, including key milestones and performance improvements.",
            verbosity="info",
            poll_interval=10.0,
            timeout=1800.0,
        )
        
        print(f"\n--- Research Complete ---")
        print(f"Interaction ID: {result.id}")
        print(f"Duration: {result.duration_seconds:.0f} seconds")
        
        if result.usage:
            print(f"Tokens used: {result.usage.total_tokens}")
        
        print(f"Report length: {len(result.text)} characters")
        print(f"Citations: {len(result.parsed_citations)}")
        
        # Show first few citations with resolved URLs and titles
        for cit in result.parsed_citations[:3]:
            title_display = f'"{cit.title}"' if cit.title else cit.domain
            print(f"  [{cit.number}] {title_display}")
            print(f"       {cit.url}")
        if len(result.parsed_citations) > 3:
            print(f"  ... and {len(result.parsed_citations) - 3} more")
        
        # Save outputs
        ensure_output_dir()
        
        # JSON for app consumption
        json_data = result.to_dict()
        json_path = os.path.join(OUTPUT_DIR, "research_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Markdown report with resolved URLs
        md_path = os.path.join(OUTPUT_DIR, "research_output.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.text)
        
        print(f"\nSaved to output/: research_output.json, research_output.md")
        
        return result
        
    except DeepResearchError as err:
        print(f"Research failed: {err.code} - {err.message}")
        return None


def example_streaming_research():
    """Research with streaming for real-time progress updates."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Streaming Research")
    print("=" * 60)
    
    def on_progress(event_type: str, content: str):
        """Callback for streaming updates."""
        if event_type == "thought":
            print(f"\n[THINKING] {content}\n")
        elif event_type == "text":
            print(content, end="", flush=True)
    
    try:
        # Citations resolved automatically after streaming completes
        result = ask_deep_research(
            "Compare these 3 competitors to Proko (Schoolism, ArtWOD, New Masters Academy).",
            stream=True,
            on_progress=on_progress,
            verbosity="none",
        )
        
        print(f"\n\n--- Streaming Complete ---")
        print(f"Duration: {result.duration_seconds:.0f} seconds")
        print(f"Citations: {len(result.parsed_citations)}")
        
        return result
        
    except DeepResearchError as err:
        print(f"Research failed: {err.code} - {err.message}")
        return None


def example_follow_up():
    """Ask follow-up questions about a completed research task."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Follow-up Questions")
    print("=" * 60)
    
    try:
        result = ask_deep_research(
            "Research the current state of quantum computing in 2024.",
            verbosity="info",
        )
        
        if result and result.id:
            print("\n--- Asking Follow-up Question ---\n")
            
            clarification = deep_research_follow_up(
                previous_interaction_id=result.id,
                query="Can you summarize the key challenges mentioned in the report in 3 bullet points?",
                verbosity="response",
            )
            
            print(f"\nFollow-up response received ({len(clarification)} chars)")
            
    except DeepResearchError as err:
        print(f"Error: {err.code} - {err.message}")


def example_with_formatting():
    """Research with specific output formatting instructions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Formatted Research Output")
    print("=" * 60)
    
    prompt = """
    Research the competitive landscape of electric vehicle batteries.
    
    Format the output as a technical report with the following structure:
    1. Executive Summary (2-3 paragraphs)
    2. Key Players (include a comparison table with columns: Company, Chemistry Type, Energy Density, Market Share)
    3. Technology Trends
    4. Supply Chain Analysis
    5. Future Outlook
    
    Use a professional, technical tone suitable for industry analysts.
    """
    
    try:
        result = ask_deep_research(
            prompt,
            verbosity="info",
        )
        
        print(f"\n--- Formatted Report Complete ---")
        print(f"Report length: {len(result.text)} characters")
        print(f"Citations: {len(result.parsed_citations)}")
        
        ensure_output_dir()
        report_path = os.path.join(OUTPUT_DIR, "ev_battery_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result.text)
        print(f"Report saved to output/ev_battery_report.md")
        
    except DeepResearchError as err:
        print(f"Research failed: {err.code} - {err.message}")


def example_with_file_search():
    """Research combining web search with your own data (experimental)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Research with File Search (Experimental)")
    print("=" * 60)
    
    # Note: You need to create a file search store first via the Google AI API
    # This example shows the structure but won't work without a valid store
    
    try:
        result = ask_deep_research(
            "Compare our internal Q3 2024 sales data against public market trends.",
            tools=[
                {
                    "type": "file_search",
                    "file_search_store_names": ["fileSearchStores/your-store-name"]
                }
            ],
            verbosity="info",
        )
        
        print(f"Research complete: {len(result.text)} characters")
        
    except DeepResearchError as err:
        # Expected to fail without a valid file search store
        print(f"Note: {err.code} - {err.message}")
        print("(This is expected without a valid file search store)")


# MAIN EXECUTION
# --------------

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print("  export GEMINI_API_KEY='your-api-key'  # Linux/Mac")
        print("  set GEMINI_API_KEY=your-api-key      # Windows CMD")
        exit(1)
    
    print("\n" + "=" * 60)
    print("GEMINI DEEP RESEARCH AGENT EXAMPLES")
    print("=" * 60)
    print("\nNote: Deep Research tasks can take several minutes to complete.")
    print("Each example will run a real research task.\n")
    
    # Uncomment the examples you want to run:
    
    # Example 1: Basic research with citation processing
    example_basic_research()
    
    # Example 2: Streaming research with progress callback
    # example_streaming_research()
    
    # Example 3: Follow-up questions
    # example_follow_up()
    
    # Example 4: Formatted output
    # example_with_formatting()
    
    # Example 5: File search (requires setup)
    # example_with_file_search()
    
    print("\n--- Examples finished ---")
