"""Image generation for skell-e-router — one `generate_image()` over three providers.

OpenAI GPT-Image and DeepInfra's Seedream models both speak the OpenAI images
API, so both go through the `openai` SDK (DeepInfra with `base_url` pointed at
its OpenAI-compatible gateway). LiteLLM's `image_generation` drops newer
GPT-Image parameters (`background`, `output_format`, `output_compression`), so
this module talks to the SDK directly, the same way `anthropic_direct.py` does
for Claude.

Gemini has no images endpoint — image output arrives inside a chat turn — so
that provider delegates to `ask_ai(..., rich_response=True)` and unpacks the
returned data URLs into the same `ImageResponse`.

Key resolution, retry/backoff and error wrapping reuse the shared helpers in
`utils.py`.
"""

import base64
import mimetypes
import os
import time

from tenacity import retry, retry_if_exception, stop_after_attempt

from .model_config import ImageModel, image_size_pixels, resolve_image_alias
from .response import ImageResponse
from .utils import (
    PROVIDER_ENV_KEY,
    RouterError,
    ask_ai,
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

# The value each optional parameter must keep on a provider that doesn't support
# it. Anything else raises instead of being silently dropped on the floor.
PARAM_DEFAULTS = {
    "quality": "auto",
    "background": None,
    "output_format": "png",
    "output_compression": None,
    "n": 1,
}

# Leading bytes that identify the container a provider actually returned.
_FORMAT_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
)

_MIME_FORMAT = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg", "image/webp": "webp"}


_client_cache: dict[tuple[str, str | None], object] = {}


def _get_openai_client(api_key: str, base_url: str | None = None):
    """Return a cached OpenAI-SDK client for this key + endpoint.

    DeepInfra is served by the same SDK with `base_url` pointed at its
    OpenAI-compatible gateway, so the cache key carries the endpoint too.
    """
    cache_key = (api_key, base_url)
    if cache_key not in _client_cache:
        try:
            import openai
        except ImportError as e:  # pragma: no cover - dependency is declared
            raise provider_error(
                e, RouterError, details={"provider": "openai"}
            ) from None
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _client_cache[cache_key] = openai.OpenAI(**kwargs)
    return _client_cache[cache_key]


# VALIDATION
# ----------

def _validate_min_pixels(size: str, model: ImageModel) -> None:
    """Enforce a provider's pixel floor before spending a request on it.

    Seedream 4.5 answers an undersized `size` with a 500 from DeepInfra's
    gateway, which reads as a transient outage and burns all three retries.
    """
    if not model.min_pixels:
        return
    pixels = image_size_pixels(size)
    if pixels is None or pixels >= model.min_pixels:
        return
    raise RouterError(
        code="INVALID_PARAM",
        message=(
            f"Invalid size '{size}' for '{model.name}': the model needs at least "
            f"{model.min_pixels:,} pixels. Try {list(model.supported_sizes)}."
        ),
    )


def _validate_size(size: str, model: ImageModel) -> None:
    if not isinstance(size, str):
        raise RouterError(
            code="INVALID_PARAM",
            message=f"`size` must be a string, got {type(size).__name__}",
        )
    _validate_min_pixels(size, model)
    if size in model.supported_sizes:
        return

    if not model.allows_custom_sizes:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"Invalid size '{size}' for '{model.name}'. "
                f"Allowed: {list(model.supported_sizes)} — the model picks the "
                f"output resolution itself, so no custom size is sent."
            ),
        )

    if not model.is_openai:
        # Non-OpenAI providers publish no documented custom-size grid; check the
        # shape here and let the provider reject anything it can't render.
        parts = size.lower().split("x")
        if len(parts) == 2 and all(p.isdigit() and int(p) > 0 for p in parts):
            return
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"Invalid size '{size}' for '{model.name}'. Use one of "
                f"{list(model.supported_sizes)} or a custom 'WIDTHxHEIGHT'."
            ),
        )

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


def _reject_unsupported_params(model: ImageModel, requested: dict) -> None:
    """Raise when the caller set a parameter this provider cannot honor.

    Defaults pass through untouched so `generate_image(alias, prompt)` works
    everywhere; anything else fails loudly rather than being dropped, which
    would hand back an image that quietly ignored what was asked for.
    """
    for param, value in requested.items():
        if param in model.supported_params:
            continue
        if value == PARAM_DEFAULTS[param]:
            continue
        supported = sorted(model.supported_params) or "none"
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"`{param}` is not supported by '{model.name}' (provider "
                f"'{model.provider}'); only the default "
                f"{param}={PARAM_DEFAULTS[param]!r} is accepted. "
                f"Supported parameters: {supported}."
            ),
        )


def _sniff_format(data: bytes, fallback: str) -> str:
    """Identify an image container from its magic bytes.

    Providers that pick the output format themselves (DeepInfra returns JPEG,
    Gemini varies) get their real format reported instead of a guess.
    """
    for magic, fmt in _FORMAT_MAGIC:
        if data.startswith(magic):
            return fmt
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return fallback


def _resolve_image_key(model: ImageModel, config: dict | None) -> str:
    """Resolve the provider API key from `config` then the environment."""
    env_key = PROVIDER_ENV_KEY.get(model.provider)
    api_key = _resolve_api_key(model, config)
    if not api_key and env_key:
        api_key = os.environ.get(env_key)
    if not api_key:
        raise RouterError(
            code="MISSING_ENV",
            message=f"{env_key or 'An API key'} is required for image generation.",
            details={"required": env_key, "provider": model.provider},
        )
    return api_key


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


def _decode_b64_images(response, image_model: ImageModel) -> list[bytes]:
    """Pull the base64 payloads out of an OpenAI-shaped images response."""
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
    return decoded


def _build_image_response(
    response,
    image_model: ImageModel,
    size: str,
    quality: str,
    output_format: str,
    duration_s: float | None,
) -> ImageResponse:
    """Convert an OpenAI images response into our ImageResponse dataclass."""
    decoded = _decode_b64_images(response, image_model)
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


# PROVIDER PATHS
# --------------

def _generate_image_openai_compatible(
    image_model: ImageModel,
    prompt: str,
    size: str,
    quality: str,
    n: int,
    background: str | None,
    output_format: str,
    output_compression: int | None,
    images: list | None,
    verbosity: str,
    config: dict | None,
    kwargs: dict,
) -> ImageResponse:
    """OpenAI GPT-Image and DeepInfra Seedream — both speak the images API."""
    edit_images = None
    if images:
        edit_images = [_prepare_edit_image(src) for src in images]

    api_key = _resolve_image_key(image_model, config)

    # DeepInfra picks the container itself and only accepts prompt/model/size/n,
    # so the GPT-Image knobs are left off the wire entirely.
    if image_model.is_deepinfra:
        request_size = image_model.default_size if size == "auto" else size
        params: dict = {
            "model": image_model.name,
            "size": request_size,
            "n": n,
            "response_format": "b64_json",
        }
    else:
        request_size = size
        params = {
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

    client = _get_openai_client(api_key, image_model.api_base)

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

    if image_model.is_deepinfra:
        return _build_per_image_priced_response(
            response=response,
            image_model=image_model,
            size=request_size,
            quality=quality,
            fallback_format=output_format,
            duration_s=duration_s,
        )

    return _build_image_response(
        response=response,
        image_model=image_model,
        size=size,
        quality=quality,
        output_format=output_format,
        duration_s=duration_s,
    )


def _build_per_image_priced_response(
    response,
    image_model: ImageModel,
    size: str,
    quality: str,
    fallback_format: str,
    duration_s: float | None,
) -> ImageResponse:
    """ImageResponse for providers that bill a flat rate per generated image."""
    decoded = _decode_b64_images(response, image_model)
    per_image = image_model.price_for_size(size)
    usage = getattr(response, "usage", None)

    return ImageResponse(
        images=decoded,
        format=_sniff_format(decoded[0], fallback_format),
        model=getattr(response, "model", None) or image_model.name,
        size=size,
        quality=quality,
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
        cost=None if per_image is None else per_image * len(decoded),
        duration_seconds=duration_s,
        raw_response=response,
    )


def _extract_data_url(item) -> str:
    """Read the data URL out of one entry of AIResponse.images."""
    if isinstance(item, str):
        return item
    image_url = item.get("image_url") if isinstance(item, dict) else getattr(item, "image_url", None)
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return image_url.get("url") or ""
    return getattr(image_url, "url", "") or ""


def _decode_data_url(data_url: str) -> tuple[bytes, str | None]:
    """Split a `data:<mime>;base64,<payload>` string into bytes and its mime."""
    header, _, encoded = data_url.partition(",")
    if not encoded:
        raise RouterError(
            code="PROVIDER_ERROR",
            message="Image response contained a malformed data URL.",
        )
    mime = header.removeprefix("data:").split(";", 1)[0] or None
    try:
        return base64.b64decode(encoded), mime
    except Exception:
        raise RouterError(
            code="PROVIDER_ERROR",
            message="Image response data URL is not valid base64.",
        ) from None


def _generate_image_gemini(
    image_model: ImageModel,
    prompt: str,
    size: str,
    quality: str,
    images: list | None,
    verbosity: str,
    config: dict | None,
    kwargs: dict,
) -> ImageResponse:
    """Gemini returns images inside a chat turn, so reuse the chat path.

    `size` is validated but never sent — Gemini chooses the resolution.
    """
    if verbosity != "none":
        print(f"\nIMAGE ({image_model.name}) — via chat alias '{image_model.chat_alias}'...\n")
    if verbosity == "debug":
        print(f"PROMPT: {prompt}")
        print(f"REFERENCE IMAGES: {images}")

    started = time.perf_counter()
    ai_response = ask_ai(
        image_model.chat_alias,
        prompt,
        rich_response=True,
        config=config,
        images=list(images) if images else None,
        verbosity=verbosity,
        **kwargs,
    )
    elapsed = time.perf_counter() - started

    decoded: list[bytes] = []
    mimes: list[str | None] = []
    for item in ai_response.images or []:
        data_url = _extract_data_url(item)
        if not data_url:
            continue
        payload, mime = _decode_data_url(data_url)
        decoded.append(payload)
        mimes.append(mime)

    if not decoded:
        raise RouterError(
            code="PROVIDER_ERROR",
            message=(
                f"'{image_model.name}' returned no image data. The model answers "
                f"in text when it declines a prompt; check the chat path with "
                f"ask_ai(rich_response=True) to see what it said."
            ),
            details={"provider": image_model.provider, "model": image_model.name},
        )

    return ImageResponse(
        images=decoded,
        format=_sniff_format(decoded[0], _MIME_FORMAT.get(mimes[0] or "", "png")),
        model=ai_response.model or image_model.name,
        size=size,
        quality=quality,
        input_tokens=ai_response.prompt_tokens,
        output_tokens=ai_response.completion_tokens,
        total_tokens=ai_response.total_tokens,
        cost=ai_response.cost,
        duration_seconds=ai_response.duration_seconds or elapsed,
        raw_response=ai_response,
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
    """Generate (or edit) images with OpenAI, DeepInfra or Gemini image models.

    Args:
        model_alias: Alias from `IMAGE_CONFIG`. OpenAI: `"gpt-image"` /
            `"gpt-image-2.5"` (both the fast `gpt-image-2.5-flare`),
            `"gpt-image-2.5-flare"`, `"gpt-image-2.5-sunburst"`, `"gpt-image-2"`.
            DeepInfra: `"seedream-4.5"`, `"seedream-4"`, `"seedream-5-pro"`.
            Gemini: `"nano-banana-3"` / `"gemini-3-pro-image"` / `"nano-banana-pro"`.
        prompt: What to draw, or how to change the supplied reference images.
        size: `"auto"`, one of the model's named sizes, or a custom
            `"WIDTHxHEIGHT"`. GPT-Image custom sizes need edges divisible by 16,
            longest edge <= 3840, 655,360–8,294,400 total pixels, aspect ratio
            between 1:3 and 3:1. Seedream takes any `"WIDTHxHEIGHT"` and
            resolves `"auto"` to the model's default (`2048x2048` on
            `seedream-4.5`, which rejects anything under 3,686,400 pixels;
            `1024x1024` on the others). Gemini accepts only `"auto"` or
            `"1024x1024"` and picks the resolution itself.
        quality: GPT-Image only — `"auto" | "low" | "medium" | "high"`, plus
            `"xhigh"` and `"max"` on the 2.5 models.
        n: How many images to return, 1..10. GPT-Image and Seedream only.
        background: GPT-Image only — `"auto" | "transparent" | "opaque"`.
            `"transparent"` requires `output_format` of `"png"` or `"webp"`.
        output_format: GPT-Image only — `"png" | "jpeg" | "webp"`. Seedream and
            Gemini choose the container; `ImageResponse.format` reports what
            actually came back.
        output_compression: GPT-Image only — 0..100, JPEG and WebP output.
        images: Optional reference images. On GPT-Image this routes the call to
            the image **edits** endpoint; on Gemini they ride along as chat
            input. Each entry may be a local file path, an http(s) URL, a
            `data:` URI, or raw bytes (GPT-Image only for bytes).
        verbosity: `"none" | "response" | "info" | "debug"`.
        config: Optional dict of API keys (overrides env vars), e.g.
            `{"openai_api_key": "sk-..."}`, `{"deepinfra_api_key": "..."}`,
            `{"gemini_api_key": "..."}`.
        **kwargs: Forwarded verbatim to the provider call.

    Returns:
        `ImageResponse` — decoded image bytes, usage, cost, timing, and a
        `save()` helper.

    Raises:
        RouterError: codes `INVALID_MODEL`, `MISSING_ENV`, `INVALID_INPUT`,
        `INVALID_PARAM`, or `PROVIDER_ERROR`. Setting a parameter the chosen
        provider does not support raises `INVALID_PARAM` rather than silently
        ignoring it.
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

    _validate_n(n)
    _reject_unsupported_params(image_model, {
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "output_compression": output_compression,
        "n": n,
    })
    _validate_output(output_format, output_compression)
    _validate_size(size, image_model)
    _validate_quality(quality, image_model)
    _validate_background(background, output_format, image_model)

    if images:
        if not isinstance(images, (list, tuple)):
            raise RouterError(
                code="INVALID_INPUT",
                message=f"`images` must be a list, got {type(images).__name__}",
            )
        if not image_model.supports_edits:
            raise RouterError(
                code="INVALID_INPUT",
                message=f"Model '{image_model.name}' does not accept reference images.",
            )

    if image_model.is_gemini:
        image_response = _generate_image_gemini(
            image_model=image_model,
            prompt=prompt,
            size=size,
            quality=quality,
            images=images,
            verbosity=verbosity,
            config=config,
            kwargs=kwargs,
        )
    else:
        image_response = _generate_image_openai_compatible(
            image_model=image_model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            images=images,
            verbosity=verbosity,
            config=config,
            kwargs=kwargs,
        )

    if verbosity in ("info", "debug"):
        print(
            f"  model={image_response.model} "
            f"n={len(image_response.images)} "
            f"format={image_response.format} "
            f"input_tokens={image_response.input_tokens} "
            f"output_tokens={image_response.output_tokens} "
            f"cost={image_response.cost} "
            f"duration={image_response.duration_seconds}"
        )

    return image_response
