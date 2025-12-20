from .utils import ask_ai, resolve_model_alias, check_environment_variables, RouterError
from .gemini_deep_research import (
    ask_deep_research,
    deep_research_follow_up,
    get_research_status,
    stream_deep_research,
    DeepResearchResult,
    DeepResearchUsage,
    DeepResearchConfig,
    DeepResearchError,
)

__all__ = [
    # Core LiteLLM-based functions
    "ask_ai",
    "resolve_model_alias",
    "check_environment_variables",
    "RouterError",
    # Gemini Deep Research Agent
    "ask_deep_research",
    "deep_research_follow_up",
    "get_research_status",
    "stream_deep_research",
    "DeepResearchResult",
    "DeepResearchUsage",
    "DeepResearchConfig",
    "DeepResearchError",
]

__version__ = "0.2.0"