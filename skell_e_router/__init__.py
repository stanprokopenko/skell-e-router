import importlib.metadata as _metadata

from .response import AIResponse, GeminiFileRef, EmbeddingResponse, ImageResponse, ClassificationResponse
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
from .images import generate_image
from .classification import classify
from .model_config import (
    EmbeddingModel,
    ImageModel,
    resolve_embedding_alias,
    resolve_image_alias,
    ClassificationModel,
    resolve_classification_alias,
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
    # Embeddings
    "get_embedding",
    "EmbeddingResponse",
    "EmbeddingModel",
    "resolve_embedding_alias",
    # Image generation
    "generate_image",
    "ImageResponse",
    "ImageModel",
    "resolve_image_alias",
    "classify",
    "ClassificationResponse",
    "ClassificationModel",
    "resolve_classification_alias",
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

# Single source of truth is the installed distribution metadata (pyproject.toml's
# version); a hardcoded literal here drifted from releases more than once.
try:
    __version__ = _metadata.version("skell-e-router")
except _metadata.PackageNotFoundError:
    # Source checkout without an installed distribution.
    __version__ = "0.0.0+uninstalled"
