# GEMINI DEEP RESEARCH AGENT
# ---------------------------
# Direct integration with Google's Interactions API for Deep Research
# This bypasses LiteLLM since it doesn't support the Interactions API

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Generator, Any

import requests

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


# ERRORS
# ------

class DeepResearchError(Exception):
    """Base error for Deep Research operations."""
    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")


# DATA CLASSES
# ------------

@dataclass
class DeepResearchUsage:
    """Token usage and cost information for a Deep Research task."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Pricing (if available from API)
    prompt_cost: float | None = None
    completion_cost: float | None = None
    total_cost: float | None = None
    # Raw usage data from API
    raw_usage: dict = field(default_factory=dict)


@dataclass
class ParsedCitation:
    """A citation extracted from the report with resolved URL."""
    number: int
    domain: str  # Original domain name (e.g., "reddit.com")
    url: str | None = None  # Resolved real URL (after following redirect)
    title: str | None = None  # Page title from the resolved URL
    redirect_url: str | None = None  # Original vertexaisearch redirect URL
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "number": self.number,
            "domain": self.domain,
            "url": self.url,
            "title": self.title,
            "redirect_url": self.redirect_url,
        }


@dataclass
class DeepResearchResult:
    """Result from a Deep Research task."""
    id: str
    status: str  # "in_progress", "completed", "failed"
    text: str | None = None
    text_without_sources: str | None = None  # Report text with sources section removed
    error: str | None = None
    outputs: list = field(default_factory=list)
    citations: list = field(default_factory=list)  # Raw citations from API
    parsed_citations: list[ParsedCitation] = field(default_factory=list)  # Structured citations
    # Metadata for cost tracking
    usage: DeepResearchUsage | None = None
    duration_seconds: float | None = None
    # Raw interaction object for advanced use
    raw_interaction: Any = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "status": self.status,
            "text": self.text,
            "text_without_sources": self.text_without_sources,
            "citations": [c.to_dict() for c in self.parsed_citations] if self.parsed_citations else [],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens if self.usage else None,
                "completion_tokens": self.usage.completion_tokens if self.usage else None,
                "total_tokens": self.usage.total_tokens if self.usage else None,
                "total_cost": self.usage.total_cost if self.usage else None,
            } if self.usage else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class DeepResearchConfig:
    """Configuration for Deep Research agent."""
    agent: str = "deep-research-pro-preview-12-2025"
    background: bool = True
    stream: bool = False
    thinking_summaries: str = "auto"
    tools: list | None = None  # Optional file_search tools


# CONSTANTS
# ---------

DEFAULT_AGENT = "deep-research-pro-preview-12-2025"
DEFAULT_POLL_INTERVAL = 10.0  # seconds
DEFAULT_TIMEOUT = 3600.0  # 60 minutes (max research time)
RECONNECT_DELAY = 2.0  # seconds for initial connection retries
STREAM_POLL_INTERVAL = 10.0  # seconds between reconnection attempts (matches poll_interval)

# Transient errors that should trigger automatic reconnection
RETRYABLE_ERRORS = [
    "gateway_timeout",
    "deadline_expired",
    "timeout",
    "connection_reset",
    "connection_closed",
    "unavailable",
    "resource_exhausted",
]


# HELPER FUNCTIONS
# ----------------

def _check_genai_available():
    """Ensure google-genai package is installed."""
    if not GENAI_AVAILABLE:
        raise DeepResearchError(
            code="MISSING_DEPENDENCY",
            message="google-genai package is required for Deep Research. Install with: pip install google-genai"
        )


def _check_api_key():
    """Ensure GEMINI_API_KEY is set."""
    if "GEMINI_API_KEY" not in os.environ:
        raise DeepResearchError(
            code="MISSING_API_KEY",
            message="GEMINI_API_KEY environment variable is required for Deep Research"
        )


def _get_client() -> "genai.Client":
    """Create and return a Google GenAI client."""
    _check_genai_available()
    _check_api_key()
    return genai.Client()


def _extract_usage(interaction) -> DeepResearchUsage | None:
    """Extract usage/cost information from an interaction response."""
    usage_data = getattr(interaction, 'usage_metadata', None)
    if usage_data is None:
        usage_data = getattr(interaction, 'usage', None)
    
    if usage_data is None:
        return None
    
    # Try to extract token counts
    prompt_tokens = getattr(usage_data, 'prompt_token_count', None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage_data, 'prompt_tokens', None)
    
    completion_tokens = getattr(usage_data, 'candidates_token_count', None)
    if completion_tokens is None:
        completion_tokens = getattr(usage_data, 'completion_tokens', None)
    
    total_tokens = getattr(usage_data, 'total_token_count', None)
    if total_tokens is None:
        total_tokens = getattr(usage_data, 'total_tokens', None)
    
    # Try to get raw dict representation
    raw_usage = {}
    if hasattr(usage_data, '__dict__'):
        raw_usage = vars(usage_data)
    elif hasattr(usage_data, 'to_dict'):
        raw_usage = usage_data.to_dict()
    
    return DeepResearchUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage
    )


def _extract_citations(interaction) -> list:
    """Extract citations from an interaction response."""
    citations = getattr(interaction, 'citations', None)
    if citations is None:
        # Try to find citations in outputs
        outputs = getattr(interaction, 'outputs', [])
        for output in outputs:
            if hasattr(output, 'citations'):
                return output.citations
    return citations or []


def _extract_text(interaction) -> str | None:
    """Extract the final text output from an interaction."""
    outputs = getattr(interaction, 'outputs', [])
    if outputs:
        last_output = outputs[-1]
        if hasattr(last_output, 'text'):
            return last_output.text
        if hasattr(last_output, 'content'):
            return last_output.content
    return None


def _build_result(interaction, duration_seconds: float | None = None) -> DeepResearchResult:
    """Build a DeepResearchResult from an interaction object."""
    return DeepResearchResult(
        id=interaction.id,
        status=interaction.status,
        text=_extract_text(interaction),
        error=str(interaction.error) if hasattr(interaction, 'error') and interaction.error else None,
        outputs=list(getattr(interaction, 'outputs', [])),
        citations=_extract_citations(interaction),
        usage=_extract_usage(interaction),
        duration_seconds=duration_seconds,
        raw_interaction=interaction
    )


def _print_request_details(query: str, config: DeepResearchConfig, verbosity: str):
    """Print request details based on verbosity level."""
    if verbosity == 'debug':
        print(f"\nDEEP RESEARCH REQUEST:")
        print(f"  Agent: {config.agent}")
        print(f"  Stream: {config.stream}")
        print(f"  Tools: {config.tools}")
        print(f"  Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        print()
    
    if verbosity != 'none':
        print(f"\nStarting Deep Research ({config.agent})...\n")


def _print_response_details(result: DeepResearchResult, verbosity: str):
    """Print response details based on verbosity level."""
    if verbosity == 'none':
        return
    
    if verbosity == 'debug':
        print(f"\n{'=' * 40}")
        print(f"RAW INTERACTION ID: {result.id}")
        print(f"STATUS: {result.status}")
        if result.raw_interaction:
            print(f"RAW: {result.raw_interaction}")
        print(f"{'=' * 40}\n")
    
    if verbosity in ('response', 'info', 'debug'):
        print(f"\nRESEARCH RESULT:\n")
        print(result.text or "(No text output)")
        print()
    
    if verbosity in ('info', 'debug'):
        print(f"\nRESEARCH INFO:")
        print(f"  Interaction ID: {result.id}")
        print(f"  Status: {result.status}")
        if result.duration_seconds:
            print(f"  Duration: {result.duration_seconds:.1f}s")
        if result.usage:
            if result.usage.prompt_tokens:
                print(f"  Prompt Tokens: {result.usage.prompt_tokens}")
            if result.usage.completion_tokens:
                print(f"  Completion Tokens: {result.usage.completion_tokens}")
            if result.usage.total_tokens:
                print(f"  Total Tokens: {result.usage.total_tokens}")
            if result.usage.total_cost:
                print(f"  Cost: ${result.usage.total_cost:.6f}")
        if result.citations:
            print(f"  Citations: {len(result.citations)}")
        print()


# CITATION PROCESSING
# -------------------

def _resolve_redirect_url(redirect_url: str, timeout: float = 5.0) -> tuple[str | None, str | None]:
    """
    Follow a vertexaisearch redirect URL to get the real destination URL and page title.
    
    Returns:
        tuple of (resolved_url, page_title)
    """
    try:
        # Use GET to fetch the page (needed for title extraction)
        response = requests.get(redirect_url, allow_redirects=True, timeout=timeout, stream=True)
        resolved_url = response.url if response.url != redirect_url else None
        
        # Try to extract title from HTML
        title = None
        if resolved_url:
            try:
                # Read first chunk to get the title (usually in first few KB)
                content = b""
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    if b"</title>" in content.lower() or len(content) > 32768:
                        break
                
                # Parse title from HTML
                html = content.decode("utf-8", errors="ignore")
                title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    # Clean up common HTML entities
                    title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    title = title.replace("&#39;", "'").replace("&quot;", '"').replace("&#x27;", "'")
                    title = title.replace("&nbsp;", " ")
            except Exception:
                pass
        
        response.close()
        return resolved_url, title
        
    except requests.RequestException:
        return None, None


def _extract_citations_from_text(text: str) -> tuple[str, list[ParsedCitation]]:
    """
    Extract citations from report text and return cleaned text + parsed citations.
    
    Returns:
        tuple of (text_without_sources, list of ParsedCitation objects)
    """
    if not text:
        return text, []
    
    # Find the **Sources:** section (case-insensitive, handles markdown bold)
    sources_patterns = [
        r'\n\*\*Sources:\*\*\s*\n',  # **Sources:**
        r'\n## Sources\s*\n',         # ## Sources
        r'\n### Sources\s*\n',        # ### Sources
        r'\nSources:\s*\n',           # Sources:
    ]
    
    sources_start = -1
    for pattern in sources_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sources_start = match.start()
            break
    
    if sources_start == -1:
        return text, []
    
    # Extract the sources section
    sources_section = text[sources_start:]
    text_without_sources = text[:sources_start].rstrip()
    
    # Parse individual citations: [number]. [domain](url)
    # Pattern matches: "1. [reddit.com](https://...)" or "1. [reddit.com](https://...)\n"
    citation_pattern = r'(\d+)\.\s*\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(citation_pattern, sources_section)
    
    parsed_citations = []
    for num_str, domain, url in matches:
        parsed_citations.append(ParsedCitation(
            number=int(num_str),
            domain=domain.strip(),
            redirect_url=url.strip(),
            url=None  # Will be resolved later
        ))
    
    return text_without_sources, parsed_citations


def _is_blocked_page_title(title: str | None) -> bool:
    """Check if a page title indicates a blocked/error page (Cloudflare, etc.)."""
    if not title:
        return True
    blocked_indicators = [
        "attention required",
        "cloudflare",
        "access denied",
        "just a moment",
        "checking your browser",
        "security check",
        "403 forbidden",
        "404 not found",
        "not acceptable",
    ]
    title_lower = title.lower()
    return any(indicator in title_lower for indicator in blocked_indicators)


def _resolve_citation_urls(citations: list[ParsedCitation], timeout: float = 5.0) -> list[ParsedCitation]:
    """Resolve all redirect URLs in citations to get real destination URLs and page titles."""
    for citation in citations:
        if citation.redirect_url and 'vertexaisearch' in citation.redirect_url:
            url, title = _resolve_redirect_url(citation.redirect_url, timeout)
            citation.url = url
            # Use domain as title if page was blocked or title couldn't be fetched
            citation.title = citation.domain if _is_blocked_page_title(title) else title
            # If resolution failed, use the domain as a fallback hint
            if not citation.url:
                citation.url = f"https://{citation.domain}"
    return citations


def _rebuild_sources_section(citations: list[ParsedCitation]) -> str:
    """Rebuild the sources section with resolved URLs."""
    if not citations:
        return ""
    
    lines = ["\n\n**Sources:**"]
    for cit in sorted(citations, key=lambda c: c.number):
        url = cit.url or cit.redirect_url or f"https://{cit.domain}"
        lines.append(f"{cit.number}. [{cit.domain}]({url})")
    
    return "\n".join(lines)


def process_citations(result: DeepResearchResult, resolve_urls: bool = True, timeout: float = 5.0) -> DeepResearchResult:
    """
    Post-process a DeepResearchResult to extract and resolve citations.
    
    This function:
    1. Extracts citations from the **Sources:** section of the report text
    2. Optionally resolves vertexaisearch redirect URLs to real URLs
    3. Stores parsed citations in result.parsed_citations
    4. Stores the report text without sources in result.text_without_sources
    
    Args:
        result: The DeepResearchResult to process
        resolve_urls: Whether to follow redirects to get real URLs (slower but useful)
        timeout: Timeout for each URL resolution request
    
    Returns:
        The same result object with parsed_citations and text_without_sources populated
    """
    if not result.text:
        return result
    
    # Step 1: Extract citations from text
    text_without_sources, parsed_citations = _extract_citations_from_text(result.text)
    result.text_without_sources = text_without_sources
    
    # Step 2: Resolve redirect URLs if requested
    if resolve_urls and parsed_citations:
        parsed_citations = _resolve_citation_urls(parsed_citations, timeout)
    
    result.parsed_citations = parsed_citations
    
    # Step 3: Rebuild text with resolved URLs
    if parsed_citations:
        result.text = text_without_sources + _rebuild_sources_section(parsed_citations)
    
    return result


def citations_to_dict(citations: list[ParsedCitation]) -> list[dict]:
    """Convert ParsedCitation objects to dictionaries for JSON serialization."""
    return [c.to_dict() for c in citations]


def result_to_dict(result: DeepResearchResult) -> dict:
    """Convert a DeepResearchResult to a dictionary for JSON serialization."""
    return result.to_dict()


# POLLING IMPLEMENTATION
# ----------------------

def _poll_for_completion(
    client: "genai.Client",
    interaction_id: str,
    poll_interval: float,
    timeout: float,
    verbosity: str,
    start_time: float
) -> DeepResearchResult:
    """Poll until research completes or times out."""
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise DeepResearchError(
                code="TIMEOUT",
                message=f"Deep Research timed out after {timeout}s",
                details={"interaction_id": interaction_id, "elapsed": elapsed}
            )
        
        interaction = client.interactions.get(interaction_id)
        
        if verbosity in ("info", "debug"):
            print(f"  Status: {interaction.status} (elapsed: {elapsed:.0f}s)")
        
        if interaction.status == "completed":
            duration = time.time() - start_time
            return _build_result(interaction, duration)
        
        elif interaction.status == "failed":
            error_msg = str(interaction.error) if hasattr(interaction, 'error') else "Unknown error"
            raise DeepResearchError(
                code="RESEARCH_FAILED",
                message=error_msg,
                details={"interaction_id": interaction_id}
            )
        
        time.sleep(poll_interval)


# STREAMING IMPLEMENTATION
# ------------------------

def _is_retryable_error(error_msg: str) -> bool:
    """Check if an error message indicates a transient/retryable error."""
    error_lower = error_msg.lower()
    return any(err in error_lower for err in RETRYABLE_ERRORS)


def _stream_research(
    client: "genai.Client",
    query: str,
    config: DeepResearchConfig,
    on_progress: Callable | None,
    verbosity: str,
    start_time: float,
    timeout: float = DEFAULT_TIMEOUT
) -> DeepResearchResult:
    """Stream research with automatic reconnection on failure."""
    
    interaction_id = None
    last_event_id = None
    collected_text = ""
    is_complete = False
    final_interaction = None
    last_error = None  # Track last error for reporting if all retries fail
    
    def process_stream(stream) -> bool:
        """Process events from a stream. Returns True if complete, False to retry."""
        nonlocal interaction_id, last_event_id, collected_text, is_complete, final_interaction, last_error
        
        for chunk in stream:
            # Capture Interaction ID from start event
            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                if on_progress:
                    on_progress("start", interaction_id)
                if verbosity in ("info", "debug"):
                    print(f"  Research started: {interaction_id}")
            
            # Track event ID for reconnection
            if hasattr(chunk, 'event_id') and chunk.event_id:
                last_event_id = chunk.event_id
            
            # Handle content deltas
            if chunk.event_type == "content.delta":
                delta = chunk.delta
                if hasattr(delta, 'type'):
                    if delta.type == "text":
                        text_chunk = delta.text if hasattr(delta, 'text') else ""
                        collected_text += text_chunk
                        if on_progress:
                            on_progress("text", text_chunk)
                        if verbosity == "debug":
                            print(text_chunk, end="", flush=True)
                    
                    elif delta.type == "thought_summary":
                        thought_text = ""
                        if hasattr(delta, 'content') and hasattr(delta.content, 'text'):
                            thought_text = delta.content.text
                        if on_progress:
                            on_progress("thought", thought_text)
                        if verbosity in ("info", "debug"):
                            print(f"  [Thinking] {thought_text}")
            
            # Handle completion
            if chunk.event_type == "interaction.complete":
                is_complete = True
                if hasattr(chunk, 'interaction'):
                    final_interaction = chunk.interaction
                return True
            
            # Handle errors - check if retryable
            if chunk.event_type == "error":
                error_msg = str(chunk.error) if hasattr(chunk, 'error') else "Stream error"
                last_error = error_msg
                
                # If retryable and we have interaction_id, signal for reconnection
                if interaction_id and _is_retryable_error(error_msg):
                    if verbosity != "none":
                        print(f"\n  Transient error (will reconnect): {error_msg}")
                    return False  # Signal to reconnect
                
                # Non-retryable error - raise immediately
                raise DeepResearchError(
                    code="STREAM_ERROR",
                    message=error_msg,
                    details={"interaction_id": interaction_id}
                )
        
        return False
    
    # Initial stream attempt (with retries if stream is None)
    max_initial_retries = 3
    for initial_attempt in range(max_initial_retries):
        try:
            if verbosity != "none":
                if initial_attempt == 0:
                    print("  Starting research stream...")
                else:
                    print(f"  Retrying initial connection (attempt {initial_attempt + 1})...")
            
            stream = client.interactions.create(
                input=query,
                agent=config.agent,
                background=True,
                stream=True,
                agent_config={
                    "type": "deep-research",
                    "thinking_summaries": config.thinking_summaries
                },
                tools=config.tools
            )
            
            # Handle None stream - retry if we haven't exhausted attempts
            if stream is None:
                last_error = "Stream returned None"
                if verbosity != "none":
                    print(f"  Stream returned None, will retry...")
                time.sleep(RECONNECT_DELAY)
                continue
            
            if process_stream(stream):
                duration = time.time() - start_time
                if final_interaction:
                    return _build_result(final_interaction, duration)
                # Fallback: fetch final state
                final = client.interactions.get(interaction_id)
                return _build_result(final, duration)
            
            # If process_stream returned False, we have interaction_id - break to reconnection loop
            break
                
        except DeepResearchError as e:
            # Check if it's a retryable error - if so, fall through to reconnection
            if interaction_id and _is_retryable_error(e.message):
                last_error = e.message
                if verbosity != "none":
                    print(f"\n  Retryable error (will reconnect): {e.message}")
                break  # We have interaction_id, go to reconnection loop
            else:
                raise
        except TypeError as e:
            # Handle "'NoneType' object is not iterable" - stream was None
            if "NoneType" in str(e) and "not iterable" in str(e):
                last_error = "Stream returned None"
                if verbosity != "none":
                    print(f"  Stream returned None, will retry...")
                time.sleep(RECONNECT_DELAY)
                continue
            # Other TypeError - re-raise
            raise
        except Exception as e:
            last_error = str(e)
            if verbosity != "none":
                print(f"\n  Stream interrupted: {e}")
            # If we have interaction_id, break to reconnection loop
            if interaction_id:
                break
            # Otherwise retry initial connection
            time.sleep(RECONNECT_DELAY)
            continue
    
    # Reconnection loop - keep trying until timeout
    reconnect_count = 0
    
    while not is_complete and interaction_id:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise DeepResearchError(
                code="TIMEOUT",
                message=f"Deep Research timed out after {timeout}s",
                details={"interaction_id": interaction_id, "elapsed": elapsed}
            )
        
        reconnect_count += 1
        
        if verbosity != "none":
            print(f"\n  Reconnecting (attempt {reconnect_count}, elapsed: {elapsed:.0f}s)...")
        
        time.sleep(STREAM_POLL_INTERVAL)
        
        try:
            stream = client.interactions.get(
                id=interaction_id,
                stream=True,
                last_event_id=last_event_id
            )
            
            # Handle None stream - just continue to next reconnection attempt
            if stream is None:
                last_error = "Stream returned None"
                if verbosity != "none":
                    print(f"  Stream returned None")
                continue
            
            if process_stream(stream):
                duration = time.time() - start_time
                if final_interaction:
                    return _build_result(final_interaction, duration)
                final = client.interactions.get(interaction_id)
                return _build_result(final, duration)
                
        except DeepResearchError as e:
            # Check if it's a retryable error that slipped through
            if _is_retryable_error(e.message):
                last_error = e.message
                if verbosity != "none":
                    print(f"  Retryable error during reconnection: {e.message}")
            else:
                raise
        except TypeError as e:
            # Handle "'NoneType' object is not iterable" - stream was None
            if "NoneType" in str(e) and "not iterable" in str(e):
                last_error = "Stream returned None"
                if verbosity != "none":
                    print(f"  Stream returned None")
                continue
            raise
        except Exception as e:
            last_error = str(e)
            if verbosity != "none":
                print(f"  Reconnection failed: {e}")
    
    # If we get here without completion, fetch final state
    if interaction_id:
        final = client.interactions.get(interaction_id)
        duration = time.time() - start_time
        return _build_result(final, duration)
    
    raise DeepResearchError(
        code="STREAM_FAILED",
        message=f"Failed to complete research stream after {reconnect_count} reconnection attempts",
        details={"reconnect_attempts": reconnect_count, "last_error": last_error}
    )


# MAIN API FUNCTIONS
# ------------------

def ask_deep_research(
    query: str,
    *,
    agent: str = DEFAULT_AGENT,
    stream: bool = False,
    tools: list | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_TIMEOUT,
    verbosity: str = "none",
    on_progress: Callable[[str, str], None] | None = None,
    resolve_citations: bool = True,
) -> DeepResearchResult:
    """
    Execute a Deep Research task and wait for completion.
    
    This function starts a research task using Google's Deep Research Agent,
    which autonomously plans, searches, reads, and synthesizes information
    to produce detailed, cited reports.
    
    Args:
        query: The research question or prompt. Can include formatting instructions.
        agent: The Deep Research agent to use. Default: "deep-research-pro-preview-12-2025"
        stream: Enable streaming for real-time progress updates. Default: False
        tools: Optional list of tools like file_search for custom data sources.
               Example: [{"type": "file_search", "file_search_store_names": ["fileSearchStores/my-store"]}]
        poll_interval: Seconds between status checks when not streaming. Default: 10.0
        timeout: Maximum wait time in seconds. Default: 3600.0 (60 minutes)
        verbosity: Output level - "none", "response", "info", or "debug". Default: "none"
        on_progress: Callback function for streaming updates. Called with (event_type, content)
                     where event_type is "start" (content=interaction_id), "text", or "thought".
        resolve_citations: Automatically extract citations from the report and resolve
                           redirect URLs to real URLs. Default: True. The original
                           redirect URLs are preserved in parsed_citations[].redirect_url.
    
    Returns:
        DeepResearchResult containing:
            - id: Interaction ID (useful for follow-up questions)
            - status: "completed" or "failed"
            - text: The final research report (with resolved citation URLs if resolve_citations=True)
            - text_without_sources: Report text without the sources section
            - parsed_citations: List of ParsedCitation objects with resolved URLs
            - usage: Token counts and cost information
            - duration_seconds: Total research time
    
    Raises:
        DeepResearchError: On timeout, failure, or API errors
    
    Example:
        >>> result = ask_deep_research(
        ...     "Research the competitive landscape of EV batteries",
        ...     verbosity="info"
        ... )
        >>> print(result.text)
        >>> for cit in result.parsed_citations:
        ...     print(f"[{cit.number}] {cit.url}")
    """
    verbosity = verbosity.lower()
    if verbosity not in ('none', 'response', 'info', 'debug'):
        print(f"WARNING: Invalid verbosity '{verbosity}'. Setting to 'response'.")
        verbosity = 'response'
    
    config = DeepResearchConfig(
        agent=agent,
        stream=stream,
        tools=tools
    )
    
    _print_request_details(query, config, verbosity)
    
    client = _get_client()
    start_time = time.time()
    
    try:
        if stream:
            result = _stream_research(client, query, config, on_progress, verbosity, start_time, timeout)
        else:
            # Start the research task
            interaction = client.interactions.create(
                input=query,
                agent=config.agent,
                background=True,
                tools=config.tools
            )
            
            if verbosity in ("info", "debug"):
                print(f"  Research started: {interaction.id}")
            
            # Poll for completion
            result = _poll_for_completion(
                client, interaction.id, poll_interval, timeout, verbosity, start_time
            )
        
        _print_response_details(result, verbosity)
        
        # Process citations if requested
        if resolve_citations:
            if verbosity in ("info", "debug"):
                print("  Resolving citation URLs...")
            result = process_citations(result, resolve_urls=True)
            if verbosity in ("info", "debug"):
                print(f"  Resolved {len(result.parsed_citations)} citations")
        
        return result
        
    except DeepResearchError:
        raise
    except Exception as e:
        if verbosity != 'none':
            print(f"ERROR in Deep Research: {e}")
        raise DeepResearchError(
            code="PROVIDER_ERROR",
            message=str(e),
            details={"agent": config.agent}
        ) from e


def deep_research_follow_up(
    previous_interaction_id: str,
    query: str,
    *,
    model: str = "gemini-3-pro-preview",
    verbosity: str = "none"
) -> str:
    """
    Ask a follow-up question about a completed Deep Research task.
    
    This allows you to get clarification, summarization, or elaboration
    on specific sections of a research report without restarting the
    entire research task.
    
    Args:
        previous_interaction_id: The interaction ID from a completed research task
                                 (available as result.id from ask_deep_research)
        query: The follow-up question
        model: Model to use for the follow-up. Default: "gemini-3-pro-preview"
        verbosity: Output level - "none", "response", "info", or "debug"
    
    Returns:
        The text response to the follow-up question
    
    Raises:
        DeepResearchError: On invalid interaction ID or API errors
    
    Example:
        >>> result = ask_deep_research("Research the history of Google TPUs")
        >>> clarification = deep_research_follow_up(
        ...     result.id,
        ...     "Can you elaborate on the second point in the report?"
        ... )
    """
    verbosity = verbosity.lower()
    
    client = _get_client()
    
    try:
        if verbosity != "none":
            print(f"\nAsking follow-up question...")
        
        interaction = client.interactions.create(
            input=query,
            model=model,
            previous_interaction_id=previous_interaction_id
        )
        
        text = _extract_text(interaction)
        
        if verbosity in ('response', 'info', 'debug'):
            print(f"\nFOLLOW-UP RESPONSE:\n")
            print(text or "(No response)")
            print()
        
        return text or ""
        
    except Exception as e:
        if verbosity != 'none':
            print(f"ERROR in follow-up: {e}")
        raise DeepResearchError(
            code="FOLLOW_UP_ERROR",
            message=str(e),
            details={"previous_interaction_id": previous_interaction_id}
        ) from e


def get_research_status(interaction_id: str) -> DeepResearchResult:
    """
    Get the current status of a Deep Research task.
    
    Useful for checking on a research task without waiting for completion.
    
    Args:
        interaction_id: The interaction ID from a started research task
    
    Returns:
        DeepResearchResult with current status and any available outputs
    
    Example:
        >>> # Start research without waiting
        >>> client = genai.Client()
        >>> interaction = client.interactions.create(
        ...     input="Research topic",
        ...     agent="deep-research-pro-preview-12-2025",
        ...     background=True
        ... )
        >>> # Check status later
        >>> status = get_research_status(interaction.id)
        >>> print(status.status)  # "in_progress", "completed", or "failed"
    """
    client = _get_client()
    interaction = client.interactions.get(interaction_id)
    return _build_result(interaction)


# GENERATOR FOR STREAMING (Alternative API)
# -----------------------------------------

def stream_deep_research(
    query: str,
    *,
    agent: str = DEFAULT_AGENT,
    tools: list | None = None,
) -> Generator[tuple[str, str], None, DeepResearchResult]:
    """
    Stream Deep Research as a generator yielding (event_type, content) tuples.
    
    This provides an alternative streaming API that can be used in async contexts
    or when you want more control over event processing.
    
    Args:
        query: The research question or prompt
        agent: The Deep Research agent to use
        tools: Optional list of tools like file_search
    
    Yields:
        Tuples of (event_type, content) where event_type is:
            - "start": content is the interaction ID
            - "text": content is a text chunk
            - "thought": content is a thinking summary
            - "status": content is a status update
    
    Returns:
        DeepResearchResult when the generator completes
    
    Example:
        >>> gen = stream_deep_research("Research topic")
        >>> for event_type, content in gen:
        ...     if event_type == "text":
        ...         print(content, end="", flush=True)
        ...     elif event_type == "thought":
        ...         print(f"[Thinking] {content}")
        >>> result = gen.value  # Get final result after iteration
    """
    config = DeepResearchConfig(agent=agent, stream=True, tools=tools)
    client = _get_client()
    start_time = time.time()
    
    interaction_id = None
    last_event_id = None
    final_interaction = None
    
    def process_stream(stream):
        nonlocal interaction_id, last_event_id, final_interaction
        
        for chunk in stream:
            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                yield ("start", interaction_id)
            
            if hasattr(chunk, 'event_id') and chunk.event_id:
                last_event_id = chunk.event_id
            
            if chunk.event_type == "content.delta":
                delta = chunk.delta
                if hasattr(delta, 'type'):
                    if delta.type == "text":
                        yield ("text", delta.text if hasattr(delta, 'text') else "")
                    elif delta.type == "thought_summary":
                        thought = ""
                        if hasattr(delta, 'content') and hasattr(delta.content, 'text'):
                            thought = delta.content.text
                        yield ("thought", thought)
            
            if chunk.event_type == "interaction.complete":
                if hasattr(chunk, 'interaction'):
                    final_interaction = chunk.interaction
                return True
        
        return False
    
    # Initial stream
    try:
        stream = client.interactions.create(
            input=query,
            agent=config.agent,
            background=True,
            stream=True,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": config.thinking_summaries
            },
            tools=config.tools
        )
        
        yield from process_stream(stream)
        
    except Exception:
        pass  # Will attempt reconnection
    
    # Reconnection loop
    while interaction_id and not final_interaction:
        yield ("status", "reconnecting")
        time.sleep(RECONNECT_DELAY)
        
        try:
            stream = client.interactions.get(
                id=interaction_id,
                stream=True,
                last_event_id=last_event_id
            )
            yield from process_stream(stream)
        except Exception:
            continue
    
    # Build and return final result
    duration = time.time() - start_time
    if final_interaction:
        return _build_result(final_interaction, duration)
    elif interaction_id:
        final = client.interactions.get(interaction_id)
        return _build_result(final, duration)
    else:
        raise DeepResearchError(
            code="STREAM_FAILED",
            message="Failed to complete research stream"
        )
