from .response import AIResponse, GeminiFileRef
from .utils import ask_ai, upload_file, resolve_model_alias, check_environment_variables, RouterError
from .gemini_deep_research import (
    ask_deep_research,
    deep_research_follow_up,
    get_research_status,
    stream_deep_research,
    process_citations,
    citations_to_dict,
    result_to_dict,
    DeepResearchResult,
    DeepResearchUsage,
    DeepResearchConfig,
    DeepResearchError,
    ParsedCitation,
)

__all__ = [
    # Core LiteLLM-based functions
    "ask_ai",
    "upload_file",
    "resolve_model_alias",
    "check_environment_variables",
    "RouterError",
    "AIResponse",
    "GeminiFileRef",
    # Gemini Deep Research Agent
    "ask_deep_research",
    "deep_research_follow_up",
    "get_research_status",
    "stream_deep_research",
    "process_citations",
    "citations_to_dict",
    "result_to_dict",
    "DeepResearchResult",
    "DeepResearchUsage",
    "DeepResearchConfig",
    "DeepResearchError",
    "ParsedCitation",
]

__version__ = "3.2.1"
