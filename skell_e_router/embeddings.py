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
        # Extract the MIME prefix between "data:" and ";"
        mime = part[5:].split(";", 1)[0].lower()
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
        if mime == "application/pdf":
            return "pdf"
        return "text"

    # 2. URLs and gs:// — use file extension
    if part.startswith(("http://", "https://", "gs://")):
        guessed, _ = mimetypes.guess_type(part)
        if guessed:
            if guessed.startswith("image/"):
                return "image"
            if guessed.startswith("audio/"):
                return "audio"
            if guessed.startswith("video/"):
                return "video"
            if guessed == "application/pdf":
                return "pdf"
        return "text"

    # 3. Existing local file
    if os.path.isfile(part):
        guessed, _ = mimetypes.guess_type(part)
        if guessed:
            if guessed.startswith("image/"):
                return "image"
            if guessed.startswith("audio/"):
                return "audio"
            if guessed.startswith("video/"):
                return "video"
            if guessed == "application/pdf":
                return "pdf"
        # Unknown MIME on a real file — treat as text-ish (rare edge case;
        # caller can override by passing GeminiFileRef explicitly).
        return "text"

    # 4. Anything else — including path-like strings that don't exist
    return "text"
