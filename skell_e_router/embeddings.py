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


def _convert_part(part, model: EmbeddingModel) -> tuple[str | dict, Modality]:
    """Convert a single part to its LiteLLM representation and infer its modality.

    Returns:
        (normalized_value, modality) where normalized_value is either a string
        (plain text, data URI, or http/gs URL) or a dict (for GeminiFileRef).
    """
    if isinstance(part, GeminiFileRef):
        modality = _modality_from_mime(part.mime_type)
        return (
            {"file_data": {"file_uri": part.uri, "mime_type": part.mime_type}},
            modality,
        )

    if isinstance(part, str):
        # If it's a real file path, encode to data URI and use the encoded form
        # for both downstream transport AND modality classification.
        if os.path.isfile(part):
            data_uri = _encode_to_data_uri(part)
            return data_uri, _classify_input_part(data_uri)
        # Otherwise pass through as-is; modality classification handles
        # data URIs / URLs / plain text.
        return part, _classify_input_part(part)

    raise RouterError(
        code="INVALID_INPUT",
        message=f"Embedding input part must be str or GeminiFileRef, got {type(part).__name__}",
    )


def _normalize_input(
    input,
    model: EmbeddingModel,
) -> tuple[list, bool]:
    """Normalize the user's `input` argument to the LiteLLM-expected list shape.

    Returns:
        (normalized, was_str) — `normalized` is a list[str | dict | list[str | dict]]
        ready for litellm.embedding(input=...). `was_str` is True iff the caller
        passed a single string (used for unwrapping the return type).

    Raises:
        RouterError("INVALID_INPUT") for: wrong top-level type, invalid part type,
        modality not supported by the model, or aggregation on a non-aggregating model.
    """
    was_str = isinstance(input, str)
    if was_str:
        items = [input]
    elif isinstance(input, list):
        items = input
    else:
        raise RouterError(
            code="INVALID_INPUT",
            message=(
                f"Embedding `input` must be a string or a list, "
                f"got {type(input).__name__}"
            ),
        )

    seen_modalities: set[str] = set()
    has_aggregation = False
    normalized: list = []

    for item in items:
        if isinstance(item, list):
            has_aggregation = has_aggregation or len(item) > 1
            inner: list = []
            for part in item:
                value, modality = _convert_part(part, model)
                seen_modalities.add(modality)
                inner.append(value)
            normalized.append(inner)
        else:
            value, modality = _convert_part(item, model)
            seen_modalities.add(modality)
            normalized.append(value)

    # Capability check: every observed modality must be in supported_inputs.
    unsupported = seen_modalities - model.supported_inputs
    if unsupported:
        bad = sorted(unsupported)
        raise RouterError(
            code="INVALID_INPUT",
            message=(
                f"Model '{model.name}' does not support {bad} inputs "
                f"(supports: {sorted(model.supported_inputs)})"
            ),
        )

    # Aggregation check: nested lists with multiple parts require model support.
    if has_aggregation and not model.supports_aggregation:
        raise RouterError(
            code="INVALID_INPUT",
            message=(
                f"Model '{model.name}' does not support aggregation; "
                f"flatten the list for batch embeddings"
            ),
        )

    return normalized, was_str


@retry(
    retry=retry_if_exception(_is_retryable_exception),
    wait=_retry_after_wait,
    stop=stop_after_attempt(3),
)
def _perform_embedding(
    model_name: str,
    input: list,
    api_key: str | None = None,
    **kwargs,
):
    """Call litellm.embedding() with retry + Retry-After backoff."""
    embedding_kwargs = dict(model=model_name, input=input, **kwargs)
    if api_key:
        embedding_kwargs["api_key"] = api_key
    request_start = time.perf_counter()
    response = litellm.embedding(**embedding_kwargs)
    request_duration = time.perf_counter() - request_start
    return response, request_duration


def _build_embedding_response(
    response,
    embedding_model: EmbeddingModel,
    request_duration_s: float | None,
    total_duration_s: float | None,
) -> EmbeddingResponse:
    """Convert a LiteLLM embedding response into our EmbeddingResponse dataclass."""
    # Sort by `index` defensively — providers should return in order, but
    # protect against any reordering.
    data = sorted(
        response.data,
        key=lambda d: d["index"] if isinstance(d, dict) else d.index,
    )
    embeddings = [
        d["embedding"] if isinstance(d, dict) else d.embedding
        for d in data
    ]

    usage = getattr(response, "usage", None)
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    model_name = getattr(response, "model", None) or embedding_model.name
    dimensions = len(embeddings[0]) if embeddings else 0

    return EmbeddingResponse(
        embeddings=embeddings,
        model=model_name,
        dimensions=dimensions,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cost=cost,
        duration_seconds=request_duration_s,
        total_duration_seconds=total_duration_s,
        raw_response=response,
    )
