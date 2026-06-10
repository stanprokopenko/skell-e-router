from .response import AIResponse, GeminiFileRef, EmbeddingResponse
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
from .embeddings import get_embedding
from .model_config import EmbeddingModel, resolve_embedding_alias

__all__ = [
    # Core LiteLLM-based functions
    "ask_ai",
    "upload_file",
    "resolve_model_alias",
    "check_environment_variables",
    "RouterError",
    "AIResponse",
    "GeminiFileRef",
    # Embeddings
    "get_embedding",
    "EmbeddingResponse",
    "EmbeddingModel",
    "resolve_embedding_alias",
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

__version__ = "3.7.1"
