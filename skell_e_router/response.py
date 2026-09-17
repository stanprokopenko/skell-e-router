from dataclasses import dataclass, field
from typing import Any


@dataclass
class GeminiFileRef:
    """Reference to a file uploaded via Gemini's Files API."""
    uri: str           # "https://generativelanguage.googleapis.com/v1beta/files/abc123"
    mime_type: str     # "video/mp4"
    display_name: str | None = None


@dataclass
class AIResponse:
    # Core content
    content: str

    # Model info
    model: str
    finish_reason: str | None = None

    # Token usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None

    # Cost and timing
    cost: float | None = None
    duration_seconds: float | None = None
    total_duration_seconds: float | None = None

    # Provider-specific
    grounding_metadata: dict | None = None
    safety_ratings: list[dict] | None = None
    images: list[dict] | None = None
    tool_calls: list | None = None
    function_call: Any = None
    provider_specific_fields: dict | None = None

    # Raw response for advanced use
    raw_response: Any = None

    def __str__(self) -> str:
        """Allow AIResponse to be used as a string (returns content)."""
        return self.content

    def __repr__(self) -> str:
        return f"AIResponse(content={self.content[:50]!r}..., model={self.model!r})"


@dataclass
class EmbeddingResponse:
    """Result of get_embedding(); always carries embeddings as list[list[float]]."""

    # Core data — always nested, even when caller passed a single string
    embeddings: list[list[float]]
    model: str          # provider-reported model name
    dimensions: int     # observed: len(embeddings[0])

    # Token usage (embedding APIs only report prompt tokens)
    prompt_tokens: int | None = None
    total_tokens: int | None = None

    # Cost and timing — same pattern as AIResponse
    cost: float | None = None
    duration_seconds: float | None = None
    total_duration_seconds: float | None = None

    # Raw provider response for advanced inspection
    raw_response: Any = None

    def __repr__(self) -> str:
        return (
            f"EmbeddingResponse(model={self.model!r}, "
            f"n={len(self.embeddings)}, dim={self.dimensions})"
        )


@dataclass
class ClassificationResponse:
    """Typed answers with native probabilities and optional reported usage.

    Cost is USD for the successful response only; duration includes retries.
    """

    answers: dict[str, dict]
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    duration_seconds: float | None = None


# Extension written by ImageResponse.save() for each output format.
_IMAGE_FORMAT_EXTENSION = {"png": "png", "jpeg": "jpg", "webp": "webp"}


@dataclass
class ImageResponse:
    """Result of generate_image(); carries decoded image bytes plus usage/cost."""

    # Core data — one entry per generated image, already base64-decoded
    images: list[bytes]
    format: str         # "png" | "jpeg" | "webp"
    model: str          # provider-reported model name
    size: str           # requested size ("auto" or "WIDTHxHEIGHT")
    quality: str        # requested quality

    # Token usage as reported by the images endpoint
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Cost and timing — same pattern as AIResponse
    cost: float | None = None
    duration_seconds: float | None = None

    # Raw provider response for advanced inspection
    raw_response: Any = None

    @property
    def extension(self) -> str:
        """File extension used by save() for this response's format."""
        return _IMAGE_FORMAT_EXTENSION.get(self.format, self.format)

    def save(self, path_or_dir: str, stem: str = "image") -> list[str]:
        """Write each image to disk and return the paths written.

        Args:
            path_or_dir: A directory (existing, or any path without a file
                extension) or a single file path. A directory gets one file per
                image named ``<stem>_<i>.<ext>``. A file path with exactly one
                image is written verbatim; with several images the index is
                inserted before the extension (``out_0.png``, ``out_1.png``).
            stem: Base filename used when writing into a directory.

        Returns:
            The list of written paths, in image order.
        """
        import os

        ext = self.extension
        target = os.fspath(path_or_dir)

        is_dir = os.path.isdir(target) or not os.path.splitext(target)[1]

        written: list[str] = []
        if is_dir:
            os.makedirs(target, exist_ok=True)
            for i, data in enumerate(self.images):
                out = os.path.join(target, f"{stem}_{i}.{ext}")
                with open(out, "wb") as f:
                    f.write(data)
                written.append(out)
            return written

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if len(self.images) == 1:
            with open(target, "wb") as f:
                f.write(self.images[0])
            return [target]

        base, file_ext = os.path.splitext(target)
        for i, data in enumerate(self.images):
            out = f"{base}_{i}{file_ext}"
            with open(out, "wb") as f:
                f.write(data)
            written.append(out)
        return written

    def __repr__(self) -> str:
        return (
            f"ImageResponse(model={self.model!r}, n={len(self.images)}, "
            f"format={self.format!r}, size={self.size!r}, quality={self.quality!r})"
        )
