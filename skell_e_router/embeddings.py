"""Embedding-model support for skell-e-router. Pure LiteLLM under the hood."""

import os
import time
import mimetypes
from typing import Any, Literal, overload

import litellm
from tenacity import retry, retry_if_exception, stop_after_attempt

from .model_config import EmbeddingModel, resolve_embedding_alias
from .response import EmbeddingResponse, GeminiFileRef
from .utils import (
    RouterError,
    _check_provider_key,
    _resolve_api_key,
    _redact_keys,
    _encode_to_data_uri,
    _is_retryable_exception,
    _retry_after_wait,
)


Modality = Literal["text", "image", "audio", "video", "pdf"]


def _modality_from_mime(mime: str | None) -> Modality:
    """Map a MIME type string to a Modality. Unknown / None → 'text'."""
    if not mime:
        return "text"
    mime = mime.lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime == "application/pdf":
        return "pdf"
    return "text"


def _classify_input_part(part: str) -> Modality:
    """Classify a single string part by its inferred modality.

    Rules:
        - data:image/* | data:audio/* | data:video/* | data:application/pdf -> that modality
        - http(s):// or gs:// with a known media extension -> modality from extension
        - local file that exists on disk -> modality from mimetypes.guess_type
        - everything else (including non-existent paths) -> text
    """
    # 1. data: URIs
    if part.startswith("data:"):
        mime = part.removeprefix("data:").split(";", 1)[0]
        return _modality_from_mime(mime)

    # 2. URLs and gs:// — guess_type only inspects the trailing extension,
    # so it works equally for gs:// despite stdlib not documenting that.
    if part.startswith(("http://", "https://", "gs://")):
        guessed, _ = mimetypes.guess_type(part)
        return _modality_from_mime(guessed)

    # 3. Existing local file
    if os.path.isfile(part):
        guessed, _ = mimetypes.guess_type(part)
        return _modality_from_mime(guessed)

    # 4. Anything else — including path-like strings that don't exist
    return "text"
