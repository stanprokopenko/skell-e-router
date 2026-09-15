"""Image generation for skell-e-router — OpenAI GPT-Image via the openai SDK.

LiteLLM's `image_generation` drops newer GPT-Image parameters (`background`,
`output_format`, `output_compression`), so this module talks to the OpenAI SDK
directly, the same way `anthropic_direct.py` does for Claude. Key resolution,
retry/backoff and error wrapping reuse the shared helpers in `utils.py`.
"""

import base64
import mimetypes
import os
import time

from tenacity import retry, retry_if_exception, stop_after_attempt

from .model_config import ImageModel, resolve_image_alias
from .response import ImageResponse
from .utils import (
    RouterError,
    _check_provider_key,
    _resolve_api_key,
    provider_error,
    _is_retryable_exception,
    _retry_after_wait,
)


# Constraints OpenAI documents for custom "WIDTHxHEIGHT" sizes on GPT-Image
# models. Verified against live 400 responses (2026-09-14).
IMAGE_CUSTOM_SIZE_RULES = {
    "multiple_of": 16,
    "max_edge": 3840,
    "min_pixels": 655_360,
    "max_pixels": 8_294_400,
    "min_aspect_ratio": 1 / 3,
    "max_aspect_ratio": 3.0,
}

SUPPORTED_OUTPUT_FORMATS = ("png", "jpeg", "webp")
SUPPORTED_BACKGROUNDS = ("auto", "transparent", "opaque")
# Transparency needs an alpha-capable container.
TRANSPARENCY_FORMATS = ("png", "webp")
COMPRESSIBLE_FORMATS = ("jpeg", "webp")
MAX_IMAGES_PER_REQUEST = 10


_client_cache: dict[str, object] = {}


def _get_openai_client(api_key: str):
    """Return a cached OpenAI client for this key (client creation is not free)."""
    if api_key not in _client_cache:
        try:
            import openai
        except ImportError as e:  # pragma: no cover - dependency is declared
            raise provider_error(
                e, RouterError, details={"provider": "openai"}
            ) from None
        _client_cache[api_key] = openai.OpenAI(api_key=api_key)
    return _client_cache[api_key]


# VALIDATION
# ----------

def _validate_size(size: str, model: ImageModel) -> None:
    if not isinstance(size, str):
        raise RouterError(
            code="INVALID_PARAM",
            message=f"`size` must be a string, got {type(size).__name__}",
        )
    if size in model.supported_sizes:
        return

    rules = IMAGE_CUSTOM_SIZE_RULES
    allowed = (
        f"Named sizes for '{model.name}': {list(model.supported_sizes)}. "
        f"Custom sizes must be 'WIDTHxHEIGHT' with both edges divisible by "
        f"{rules['multiple_of']}, the longest edge <= {rules['max_edge']}, total "
        f"pixels between {rules['min_pixels']} and {rules['max_pixels']}, and an "
        f"aspect ratio between 1:3 and 3:1."
    )

    parts = size.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise RouterError(
            code="INVALID_PARAM",
            message=f"Invalid size '{size}'. {allowed}",
        )

    width, height = int(parts[0]), int(parts[1])
    problem = None
    if width % rules["multiple_of"] or height % rules["multiple_of"]:
        problem = f"both edges must be divisible by {rules['multiple_of']}"
    elif max(width, height) > rules["max_edge"]:
        problem = f"the longest edge must be <= {rules['max_edge']}"
    elif width * height < rules["min_pixels"]:
        problem = f"total pixels must be >= {rules['min_pixels']}"
    elif width * height > rules["max_pixels"]:
        problem = f"total pixels must be <= {rules['max_pixels']}"
    elif not (
        rules["min_aspect_ratio"] <= width / height <= rules["max_aspect_ratio"]
    ):
        problem = "the aspect ratio must be between 1:3 and 3:1"

    if problem:
        raise RouterError(
            code="INVALID_PARAM",
            message=f"Invalid size '{size}': {problem}. {allowed}",
        )


def _validate_quality(quality: str, model: ImageModel) -> None:
    if quality not in model.supported_qualities:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"Invalid quality '{quality}' for '{model.name}'. "
                f"Allowed: {list(model.supported_qualities)}"
            ),
        )


def _validate_background(
    background: str | None, output_format: str, model: ImageModel
) -> None:
    if background is None:
        return
    if background not in SUPPORTED_BACKGROUNDS:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"Invalid background '{background}'. "
                f"Allowed: {list(SUPPORTED_BACKGROUNDS)}"
            ),
        )
    if background != "transparent":
        return
    if not model.supports_transparent_background:
        raise RouterError(
            code="INVALID_PARAM",
            message=f"Model '{model.name}' does not support transparent backgrounds.",
        )
    if output_format not in TRANSPARENCY_FORMATS:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"background='transparent' requires output_format in "
                f"{list(TRANSPARENCY_FORMATS)}, got '{output_format}'."
            ),
        )


def _validate_output(output_format: str, output_compression: int | None) -> None:
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"Invalid output_format '{output_format}'. "
                f"Allowed: {list(SUPPORTED_OUTPUT_FORMATS)}"
            ),
        )
    if output_compression is None:
        return
    if not isinstance(output_compression, int) or isinstance(output_compression, bool):
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"`output_compression` must be int or None, "
                f"got {type(output_compression).__name__}"
            ),
        )
    if not 0 <= output_compression <= 100:
        raise RouterError(
            code="INVALID_PARAM",
            message=f"`output_compression={output_compression}` must be between 0 and 100.",
        )
    if output_format not in COMPRESSIBLE_FORMATS:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"`output_compression` only applies to "
                f"{list(COMPRESSIBLE_FORMATS)} output, got '{output_format}'."
            ),
        )


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise RouterError(
            code="INVALID_PARAM",
            message=f"`n` must be an int, got {type(n).__name__}",
        )
    if not 1 <= n <= MAX_IMAGES_PER_REQUEST:
        raise RouterError(
            code="INVALID_PARAM",
            message=f"`n={n}` out of range. Allowed: 1..{MAX_IMAGES_PER_REQUEST}",
        )


# REFERENCE IMAGES (edits endpoint)
# ---------------------------------

def _prepare_edit_image(source) -> tuple[str, bytes, str]:
    """Turn one reference-image source into an SDK upload tuple.

    Accepts a local file path, an http(s) URL, a `data:` URI, or raw bytes.
    Returns `(filename, content, mime_type)` as the OpenAI SDK expects.
    """
    if isinstance(source, (bytes, bytearray)):
        return ("image.png", bytes(source), "image/png")

    if not isinstance(source, str):
        raise RouterError(
            code="INVALID_INPUT",
            message=(
                f"Reference image must be a path, URL, data URI or bytes, "
                f"got {type(source).__name__}"
            ),
        )

    if source.startswith("data:"):
        header, _, encoded = source.partition(",")
        if not encoded:
            raise RouterError(
                code="INVALID_INPUT",
                message="Malformed data URI passed as a reference image.",
            )
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/png"
        try:
            content = base64.b64decode(encoded)
        except Exception:
            raise RouterError(
                code="INVALID_INPUT",
                message="Reference image data URI is not valid base64.",
            ) from None
        ext = mimetypes.guess_extension(mime) or ".png"
        return (f"image{ext}", content, mime)

    if source.startswith(("http://", "https://")):
        import requests

        try:
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            raise provider_error(
                e, RouterError, code="INVALID_INPUT", details={"stage": "reference_image_fetch"}
            ) from None
        mime = resp.headers.get("content-type", "image/png").split(";", 1)[0]
        name = os.path.basename(source.split("?", 1)[0]) or "image.png"
        return (name, resp.content, mime)

    if not os.path.isfile(source):
        raise RouterError(
            code="INVALID_INPUT",
            message=f"Reference image not found: {source}",
        )
    mime, _ = mimetypes.guess_type(source)
    with open(source, "rb") as f:
        content = f.read()
    return (os.path.basename(source), content, mime or "image/png")


# TRANSPORT
# ---------

@retry(
    retry=retry_if_exception(_is_retryable_exception),
    wait=_retry_after_wait,
    stop=stop_after_attempt(3),
    reraise=True,
)
def _perform_image_request(client, prompt: str, edit_images: list | None, params: dict):
    """Call the OpenAI images endpoint with retry + Retry-After backoff."""
    request_start = time.perf_counter()
    if edit_images:
        response = client.images.edit(prompt=prompt, image=edit_images, **params)
    else:
        response = client.images.generate(prompt=prompt, **params)
    return response, time.perf_counter() - request_start


# RESPONSE
# --------

def _usage_value(usage, key: str):
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _compute_cost(usage, model: ImageModel) -> float | None:
    """Price the call from usage tokens. None when the registry has no prices."""
    if not model.has_pricing or usage is None:
        return None

    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None

    details = _usage_value(usage, "input_tokens_details")
    text_tokens = _usage_value(details, "text_tokens")
    image_tokens = _usage_value(details, "image_tokens")
    if text_tokens is None and image_tokens is None:
        # No breakdown reported — all input is prompt text.
        text_tokens, image_tokens = (input_tokens or 0), 0
    else:
        text_tokens, image_tokens = (text_tokens or 0), (image_tokens or 0)

    return (
        text_tokens * model.text_input_price
        + image_tokens * model.image_input_price
        + (output_tokens or 0) * model.image_output_price
    ) / 1_000_000


def _build_image_response(
    response,
    image_model: ImageModel,
    size: str,
    quality: str,
    output_format: str,
    duration_s: float | None,
) -> ImageResponse:
    """Convert an OpenAI images response into our ImageResponse dataclass."""
    data = getattr(response, "data", None) or []
    decoded: list[bytes] = []
    for item in data:
        b64 = item.get("b64_json") if isinstance(item, dict) else getattr(item, "b64_json", None)
        if not b64:
            continue
        decoded.append(base64.b64decode(b64))

    if not decoded:
        raise RouterError(
            code="PROVIDER_ERROR",
            message="Image response contained no base64 image data.",
            details={"provider": image_model.provider, "model": image_model.name},
        )

    usage = getattr(response, "usage", None)
    return ImageResponse(
        images=decoded,
        format=output_format,
        model=getattr(response, "model", None) or image_model.name,
        size=size,
        quality=quality,
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
        cost=_compute_cost(usage, image_model),
        duration_seconds=duration_s,
        raw_response=response,
    )


# PUBLIC API
# ----------

def generate_image(
    model_alias: str,
    prompt: str,
    *,
    size: str = "auto",
    quality: str = "auto",
    n: int = 1,
    background: str | None = None,
    output_format: str = "png",
    output_compression: int | None = None,
    images: list | None = None,
    verbosity: str = "none",
    config: dict | None = None,
    **kwargs,
) -> ImageResponse:
    """Generate (or edit) images with OpenAI's GPT-Image models.

    Args:
        model_alias: Alias from `IMAGE_CONFIG` — `"gpt-image"` / `"gpt-image-2.5"`
            (both point at the fast `gpt-image-2.5-flare`),
            `"gpt-image-2.5-flare"`, `"gpt-image-2.5-sunburst"`, `"gpt-image-2"`.
        prompt: What to draw, or how to change the supplied reference images.
        size: `"auto"`, one of the model's named sizes, or a custom
            `"WIDTHxHEIGHT"` (edges divisible by 16, longest edge <= 3840,
            655,360–8,294,400 total pixels, aspect ratio between 1:3 and 3:1).
        quality: `"auto" | "low" | "medium" | "high"`, plus `"xhigh"` and
            `"max"` on the 2.5 models.
        n: How many images to return, 1..10.
        background: `"auto" | "transparent" | "opaque"`. `"transparent"`
            requires `output_format` of `"png"` or `"webp"`.
        output_format: `"png" | "jpeg" | "webp"`.
        output_compression: 0..100, JPEG and WebP only.
        images: Optional reference images. Passing any routes the call to the
            image **edits** endpoint instead of generations. Each entry may be a
            local file path, an http(s) URL, a `data:` URI, or raw bytes.
        verbosity: `"none" | "response" | "info" | "debug"`.
        config: Optional dict of API keys (overrides env vars), e.g.
            `{"openai_api_key": "sk-..."}`.
        **kwargs: Forwarded verbatim to the OpenAI images endpoint.

    Returns:
        `ImageResponse` — decoded image bytes, usage, cost, timing, and a
        `save()` helper.

    Raises:
        RouterError: codes `INVALID_MODEL`, `MISSING_ENV`, `INVALID_INPUT`,
        `INVALID_PARAM`, or `PROVIDER_ERROR`.
    """
    verbosity = (verbosity or "none").lower()
    if verbosity not in ("none", "response", "info", "debug"):
        print(f"WARNING: Invalid verbosity '{verbosity}'. Must be 'none', 'response', 'info', or 'debug'.\nSetting to 'response'.")
        verbosity = "response"

    image_model = resolve_image_alias(model_alias)
    _check_provider_key(image_model, config, verbosity)

    if not isinstance(prompt, str) or not prompt.strip():
        raise RouterError(
            code="INVALID_INPUT",
            message="`prompt` must be a non-empty string.",
        )

    _validate_output(output_format, output_compression)
    _validate_size(size, image_model)
    _validate_quality(quality, image_model)
    _validate_background(background, output_format, image_model)
    _validate_n(n)

    edit_images = None
    if images:
        if not isinstance(images, (list, tuple)):
            raise RouterError(
                code="INVALID_INPUT",
                message=f"`images` must be a list, got {type(images).__name__}",
            )
        if not image_model.supports_edits:
            raise RouterError(
                code="INVALID_INPUT",
                message=f"Model '{image_model.name}' does not support image edits.",
            )
        edit_images = [_prepare_edit_image(src) for src in images]

    api_key = _resolve_api_key(image_model, config) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RouterError(
            code="MISSING_ENV",
            message="OPENAI_API_KEY is required for image generation.",
            details={"required": "OPENAI_API_KEY", "provider": image_model.provider},
        )

    params: dict = {
        "model": image_model.name,
        "size": size,
        "quality": quality,
        "n": n,
        "output_format": output_format,
    }
    if background is not None:
        params["background"] = background
    if output_compression is not None:
        params["output_compression"] = output_compression
    params.update(kwargs)

    endpoint = "edits" if edit_images else "generations"
    if verbosity != "none":
        print(f"\nIMAGE ({image_model.name}) — {n} image(s) via {endpoint}...\n")
    if verbosity == "debug":
        print(f"PROMPT: {prompt}")
        print(f"PARAMS: {params}")
        if edit_images:
            print(f"REFERENCE IMAGES: {[name for name, _, _ in edit_images]}")

    client = _get_openai_client(api_key)

    try:
        response, duration_s = _perform_image_request(
            client=client,
            prompt=prompt,
            edit_images=edit_images,
            params=params,
        )
    except Exception as e:
        error = provider_error(e, RouterError, details={
            "provider": image_model.provider, "model": image_model.name,
            "endpoint": endpoint})
    else:
        error = None
    if error is not None:
        if verbosity != "none":
            print(f"ERROR calling {image_model.name}: {error.message}")
        raise error from None

    image_response = _build_image_response(
        response=response,
        image_model=image_model,
        size=size,
        quality=quality,
        output_format=output_format,
        duration_s=duration_s,
    )

    if verbosity in ("info", "debug"):
        print(
            f"  model={image_response.model} "
            f"n={len(image_response.images)} "
            f"format={image_response.format} "
            f"input_tokens={image_response.input_tokens} "
            f"output_tokens={image_response.output_tokens} "
            f"cost={image_response.cost} "
            f"duration={duration_s:.3f}s"
        )

    return image_response
