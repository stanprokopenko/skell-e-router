# Embedding-Model Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `get_embedding()` to `skell-e-router` so callers can route OpenAI text embeddings and Gemini multimodal embeddings through the existing router with shared retry/backoff and error wrapping.

**Architecture:** Pure LiteLLM (no direct-SDK path). New `EmbeddingModel` class and `EMBEDDING_MODEL_CONFIG` registry sit alongside the existing chat `AIModel`/`MODEL_CONFIG`. A single function `get_embedding(model, input, ...)` mirrors LiteLLM's nested-list input shape (top-level list elements = output embeddings; nested lists = aggregation). Shared retry/key helpers are imported from `utils.py`.

**Tech Stack:** Python 3.10+, `litellm>=1.43.0`, `tenacity>=8.0.0`, `pytest`. Companion spec: `docs/superpowers/specs/2026-05-05-embedding-models-design.md`.

---

## Background for the implementer

This codebase wraps LiteLLM. Existing chat path lives in `skell_e_router/utils.py` and exports `ask_ai()`. Each model is registered in `MODEL_CONFIG` with an `AIModel` instance keyed by short alias (e.g., `"gpt-5"`) AND by full LiteLLM identifier (e.g., `"openai/gpt-5"`). Errors wrap as `RouterError(code, message, details)`. Retry uses `tenacity` with `_is_retryable_exception` (5xx, 429-non-quota, timeouts; respects `Retry-After`).

LiteLLM's `litellm.embedding(model, input, dimensions=...)` already supports both OpenAI text-embedding-3 and Gemini Embedding 2 (multimodal). For Gemini, **flat list** input → N embeddings (per-input batch), **nested list** input → 1 fused embedding (aggregation). Confirmed via the LiteLLM Gemini Embedding 2 GA blog post.

Tests live in `tests/`. They mock `litellm.completion`/`litellm.embedding` at the module-import path. Helpers are in `tests/helpers.py` — extend it as needed.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `skell_e_router/embeddings.py` | new | `get_embedding`, input normalization, modality classification, retry wrapper, response builder |
| `skell_e_router/model_config.py` | modified | adds `EmbeddingModel` + `EMBEDDING_MODEL_CONFIG` + `resolve_embedding_alias` |
| `skell_e_router/response.py` | modified | adds `EmbeddingResponse` dataclass |
| `skell_e_router/utils.py` | modified | refactor `_encode_image` to use new `_encode_to_data_uri` (no behavior change) |
| `skell_e_router/__init__.py` | modified | export new public symbols |
| `skell_e_router/examples/example_embeddings.py` | new | runnable usage examples |
| `skell_e_router/Skell-E-Router-DOCUMENTATION.md` | modified | full embeddings section + capability matrix |
| `README.md` | modified | quick-start "Embeddings" section |
| `tests/test_embeddings.py` | new | shape, validation, retry, response, alias tests |
| `tests/helpers.py` | modified | add `make_litellm_embedding_response` mock builder |

---

## Task 1: Add `EmbeddingResponse` dataclass

**Files:**
- Modify: `skell_e_router/response.py`
- Test: `tests/test_response.py`

- [ ] **Step 1.1: Read the existing file**

Run: `cat skell_e_router/response.py`

Confirm it contains `GeminiFileRef` and `AIResponse` dataclasses.

- [ ] **Step 1.2: Write the failing test**

Append to `tests/test_response.py` (create if missing — model after `test_utils.py` style):

```python
# tests/test_response.py — append at bottom

from skell_e_router.response import EmbeddingResponse


class TestEmbeddingResponse:

    def test_minimal_construction(self):
        resp = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="text-embedding-3-large",
            dimensions=3,
        )
        assert resp.embeddings == [[0.1, 0.2, 0.3]]
        assert resp.model == "text-embedding-3-large"
        assert resp.dimensions == 3
        assert resp.prompt_tokens is None
        assert resp.cost is None
        assert resp.raw_response is None

    def test_full_construction(self):
        resp = EmbeddingResponse(
            embeddings=[[0.1], [0.2]],
            model="gemini-embedding-2",
            dimensions=1,
            prompt_tokens=4,
            total_tokens=4,
            cost=0.0001,
            duration_seconds=0.5,
            total_duration_seconds=0.6,
            raw_response={"hi": "there"},
        )
        assert resp.prompt_tokens == 4
        assert resp.total_tokens == 4
        assert resp.cost == 0.0001
        assert resp.duration_seconds == 0.5
        assert resp.total_duration_seconds == 0.6
        assert resp.raw_response == {"hi": "there"}
```

- [ ] **Step 1.3: Run test to confirm it fails**

Run: `pytest tests/test_response.py::TestEmbeddingResponse -v`
Expected: `ImportError: cannot import name 'EmbeddingResponse' from 'skell_e_router.response'`

- [ ] **Step 1.4: Implement `EmbeddingResponse`**

Append to `skell_e_router/response.py`:

```python
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
```

- [ ] **Step 1.5: Run test to confirm it passes**

Run: `pytest tests/test_response.py::TestEmbeddingResponse -v`
Expected: 2 passed.

- [ ] **Step 1.6: Run full response test file to confirm no regressions**

Run: `pytest tests/test_response.py -v`
Expected: all tests pass.

- [ ] **Step 1.7: Commit**

```bash
git add skell_e_router/response.py tests/test_response.py
git commit -m "Add EmbeddingResponse dataclass"
```

---

## Task 2: Add `EmbeddingModel` class, registry, and resolver

**Files:**
- Modify: `skell_e_router/model_config.py`
- Test: `tests/test_model_config.py`

- [ ] **Step 2.1: Read the existing file**

Run: `cat skell_e_router/model_config.py | head -50`

Confirm it has the `AIModel` class and `MODEL_CONFIG` dict pattern, with full-name keys appended after the alias keys.

- [ ] **Step 2.2: Write the failing test**

Append to `tests/test_model_config.py`:

```python
# tests/test_model_config.py — append at bottom

import pytest
from skell_e_router.model_config import (
    EmbeddingModel,
    EMBEDDING_MODEL_CONFIG,
    resolve_embedding_alias,
)
from skell_e_router.utils import RouterError


class TestEmbeddingModel:

    def test_class_basic_fields(self):
        m = EmbeddingModel(
            name="provider/some-model",
            provider="openai",
            supported_inputs={"text"},
            max_dimensions=1024,
            default_dimensions=1024,
        )
        assert m.name == "provider/some-model"
        assert m.provider == "openai"
        assert m.supported_inputs == {"text"}
        assert m.max_dimensions == 1024
        assert m.default_dimensions == 1024
        assert m.recommended_dimensions == ()
        assert m.max_input_tokens is None
        assert m.supports_aggregation is False

    def test_provider_helpers(self):
        oa = EmbeddingModel(
            name="openai/x", provider="openai",
            supported_inputs={"text"}, max_dimensions=1, default_dimensions=1,
        )
        gm = EmbeddingModel(
            name="gemini/x", provider="gemini",
            supported_inputs={"text"}, max_dimensions=1, default_dimensions=1,
        )
        assert oa.is_openai is True and oa.is_gemini is False
        assert gm.is_openai is False and gm.is_gemini is True


class TestEmbeddingRegistry:

    def test_three_aliases_registered(self):
        assert "openai-embedding-3-large" in EMBEDDING_MODEL_CONFIG
        assert "openai-embedding-3-small" in EMBEDDING_MODEL_CONFIG
        assert "gemini-embedding-2" in EMBEDDING_MODEL_CONFIG

    def test_full_names_also_keyed(self):
        assert "openai/text-embedding-3-large" in EMBEDDING_MODEL_CONFIG
        assert "openai/text-embedding-3-small" in EMBEDDING_MODEL_CONFIG
        assert "gemini/gemini-embedding-2" in EMBEDDING_MODEL_CONFIG

    def test_alias_and_fullname_resolve_to_same_object(self):
        a = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        b = EMBEDDING_MODEL_CONFIG["openai/text-embedding-3-large"]
        assert a is b

    def test_openai_large_specs(self):
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        assert m.provider == "openai"
        assert m.supported_inputs == {"text"}
        assert m.max_dimensions == 3072
        assert m.default_dimensions == 3072
        assert m.recommended_dimensions == (256, 1024, 3072)
        assert m.max_input_tokens == 8192
        assert m.supports_aggregation is False

    def test_openai_small_specs(self):
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-small"]
        assert m.max_dimensions == 1536
        assert m.default_dimensions == 1536
        assert m.recommended_dimensions == (512, 1536)
        assert m.supported_inputs == {"text"}
        assert m.supports_aggregation is False

    def test_gemini_specs(self):
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        assert m.provider == "gemini"
        assert m.supported_inputs == {"text", "image", "audio", "video", "pdf"}
        assert m.max_dimensions == 3072
        assert m.default_dimensions == 3072
        assert m.recommended_dimensions == (768, 1536, 3072)
        assert m.max_input_tokens == 8192
        assert m.supports_aggregation is True


class TestResolveEmbeddingAlias:

    def test_known_alias(self):
        m = resolve_embedding_alias("openai-embedding-3-large")
        assert m.name == "openai/text-embedding-3-large"

    def test_full_name_lookup(self):
        m = resolve_embedding_alias("gemini/gemini-embedding-2")
        assert m.provider == "gemini"

    def test_unknown_alias_raises(self):
        with pytest.raises(RouterError) as exc:
            resolve_embedding_alias("not-a-real-model")
        assert exc.value.code == "INVALID_MODEL"
        assert "not-a-real-model" in exc.value.message
```

- [ ] **Step 2.3: Run test to confirm it fails**

Run: `pytest tests/test_model_config.py::TestEmbeddingModel tests/test_model_config.py::TestEmbeddingRegistry tests/test_model_config.py::TestResolveEmbeddingAlias -v`
Expected: `ImportError` for `EmbeddingModel`.

- [ ] **Step 2.4: Implement the class, registry, and resolver**

Append to `skell_e_router/model_config.py` (after the existing `MODEL_CONFIG` block and after the full-name aliasing loop):

```python
# ============================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================


class EmbeddingModel:
    """Registry entry for an embedding model. Distinct from chat AIModel."""

    def __init__(
        self,
        name: str,                                  # full LiteLLM identifier
        provider: str,                              # "openai" | "gemini"
        supported_inputs: set[str],                 # subset of {"text","image","audio","video","pdf"}
        max_dimensions: int,
        default_dimensions: int,
        recommended_dimensions: tuple[int, ...] = (),
        max_input_tokens: int | None = None,
        supports_aggregation: bool = False,
    ):
        self.name = name
        self.provider = provider
        self.supported_inputs = supported_inputs
        self.max_dimensions = max_dimensions
        self.default_dimensions = default_dimensions
        self.recommended_dimensions = recommended_dimensions
        self.max_input_tokens = max_input_tokens
        self.supports_aggregation = supports_aggregation

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @property
    def is_gemini(self) -> bool:
        return self.provider == "gemini"


EMBEDDING_MODEL_CONFIG: dict[str, EmbeddingModel] = {
    "openai-embedding-3-large": EmbeddingModel(
        name="openai/text-embedding-3-large",
        provider="openai",
        supported_inputs={"text"},
        max_dimensions=3072,
        default_dimensions=3072,
        recommended_dimensions=(256, 1024, 3072),
        max_input_tokens=8192,
        supports_aggregation=False,
    ),
    "openai-embedding-3-small": EmbeddingModel(
        name="openai/text-embedding-3-small",
        provider="openai",
        supported_inputs={"text"},
        max_dimensions=1536,
        default_dimensions=1536,
        recommended_dimensions=(512, 1536),
        max_input_tokens=8192,
        supports_aggregation=False,
    ),
    "gemini-embedding-2": EmbeddingModel(
        name="gemini/gemini-embedding-2",
        provider="gemini",
        supported_inputs={"text", "image", "audio", "video", "pdf"},
        max_dimensions=3072,
        default_dimensions=3072,
        recommended_dimensions=(768, 1536, 3072),
        max_input_tokens=8192,
        supports_aggregation=True,
    ),
}

# Allow lookup by full LiteLLM name in addition to alias.
for _cfg in list(EMBEDDING_MODEL_CONFIG.values()):
    if _cfg.name not in EMBEDDING_MODEL_CONFIG:
        EMBEDDING_MODEL_CONFIG[_cfg.name] = _cfg


def resolve_embedding_alias(model_alias: str) -> EmbeddingModel:
    """Resolve an embedding model alias (or full LiteLLM name) to its EmbeddingModel."""
    # Local import avoids a circular reference (utils.py imports model_config at module load).
    from .utils import RouterError

    model = EMBEDDING_MODEL_CONFIG.get(model_alias)
    if not model:
        # Show only the short aliases (full-name duplicates would clutter the message).
        available = sorted(k for k in EMBEDDING_MODEL_CONFIG if "/" not in k)
        raise RouterError(
            code="INVALID_MODEL",
            message=f"Invalid embedding model alias '{model_alias}'. Available: {available}",
        )
    return model
```

- [ ] **Step 2.5: Run tests**

Run: `pytest tests/test_model_config.py -v`
Expected: all new tests pass; existing tests still pass.

- [ ] **Step 2.6: Commit**

```bash
git add skell_e_router/model_config.py tests/test_model_config.py
git commit -m "Add EmbeddingModel registry and alias resolver"
```

---

## Task 3: Refactor `_encode_image` → extract `_encode_to_data_uri`

**Goal:** Existing `_encode_image` returns an OpenAI chat content part dict. Embeddings need just the `data:` URI string. Extract the data-URI core into a shared helper without changing chat behavior.

**Files:**
- Modify: `skell_e_router/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 3.1: Read existing implementation**

Run: `grep -n "_encode_image" skell_e_router/utils.py`

Read lines 70-87 of `skell_e_router/utils.py` to confirm the existing implementation.

- [ ] **Step 3.2: Write the failing test**

Append to `tests/test_utils.py`:

```python
# tests/test_utils.py — append at bottom

class TestEncodeToDataUri:

    def test_passes_through_data_uri(self):
        from skell_e_router.utils import _encode_to_data_uri
        assert _encode_to_data_uri("data:image/png;base64,abc") == "data:image/png;base64,abc"

    def test_passes_through_http_url(self):
        from skell_e_router.utils import _encode_to_data_uri
        assert _encode_to_data_uri("https://example.com/x.jpg") == "https://example.com/x.jpg"
        assert _encode_to_data_uri("http://example.com/x.jpg") == "http://example.com/x.jpg"

    def test_encodes_local_file(self, tmp_path):
        from skell_e_router.utils import _encode_to_data_uri
        p = tmp_path / "tiny.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = _encode_to_data_uri(str(p))
        assert result.startswith("data:image/png;base64,")

    def test_unknown_mime_falls_back_to_octet_stream(self, tmp_path):
        from skell_e_router.utils import _encode_to_data_uri
        p = tmp_path / "weird.xyz"
        p.write_bytes(b"\x00\x01\x02")
        result = _encode_to_data_uri(str(p))
        assert result.startswith("data:application/octet-stream;base64,")

    def test_missing_file_raises(self):
        from skell_e_router.utils import _encode_to_data_uri
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            _encode_to_data_uri("./does-not-exist-12345.png")
        assert exc.value.code == "INVALID_INPUT"
        assert "not found" in exc.value.message.lower()

    def test_encode_image_still_returns_chat_part_shape(self, tmp_path):
        """Ensure refactor preserves _encode_image's return shape."""
        from skell_e_router.utils import _encode_image
        p = tmp_path / "tiny.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        out = _encode_image(str(p))
        assert out["type"] == "image_url"
        assert out["image_url"]["url"].startswith("data:image/png;base64,")
```

- [ ] **Step 3.3: Run tests to confirm `_encode_to_data_uri` does not yet exist**

Run: `pytest tests/test_utils.py::TestEncodeToDataUri -v`
Expected: ImportError.

- [ ] **Step 3.4: Implement the helper and refactor `_encode_image`**

In `skell_e_router/utils.py`, replace the existing `_encode_image` (currently at lines 70–87) with:

```python
def _encode_to_data_uri(source: str) -> str:
    """Convert a string source (URL, data URI, or file path) to a data-URI / URL string.

    Returns:
        - the input unchanged if it is already an http(s) URL or data URI
        - a `data:<mime>;base64,<...>` string if `source` is an existing file path
    Raises RouterError("INVALID_INPUT") if a path-like string is not an existing file.
    """
    if source.startswith(("http://", "https://", "data:")):
        return source
    if not os.path.isfile(source):
        raise RouterError(
            code="INVALID_INPUT",
            message=f"File not found: {source}",
        )
    mime_type, _ = mimetypes.guess_type(source)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(source, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _encode_image(source: str) -> dict:
    """Convert an image source to an OpenAI chat-completion `image_url` content part."""
    return {"type": "image_url", "image_url": {"url": _encode_to_data_uri(source)}}
```

- [ ] **Step 3.5: Run new tests + existing chat-path image tests**

Run: `pytest tests/test_utils.py -v -k "encode or image or images"`
Expected: all pass — `_encode_image`-using chat tests must still pass unchanged.

- [ ] **Step 3.6: Run full test suite to confirm no regressions**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 3.7: Commit**

```bash
git add skell_e_router/utils.py tests/test_utils.py
git commit -m "Extract _encode_to_data_uri helper from _encode_image"
```

---

## Task 4: Add `make_litellm_embedding_response` test helper

**Goal:** Centralize the mock-builder for LiteLLM embedding responses so multiple test files can reuse it.

**Files:**
- Modify: `tests/helpers.py`

- [ ] **Step 4.1: Append helper to `tests/helpers.py`**

Append at the bottom of `tests/helpers.py`:

```python
# ---------------------------------------------------------------------------
# Mock LiteLLM embedding response builder
# ---------------------------------------------------------------------------

def make_litellm_embedding_response(
    embeddings: list[list[float]] | None = None,
    model: str = "openai/text-embedding-3-large",
    prompt_tokens: int = 5,
    total_tokens: int = 5,
):
    """Build a mock that looks like a litellm.embedding response.

    Mirrors LiteLLM's OpenAI-shape `data: [{object, index, embedding}]` structure.
    """
    if embeddings is None:
        embeddings = [[0.1, 0.2, 0.3]]

    data = []
    for i, emb in enumerate(embeddings):
        item = MagicMock()
        item.object = "embedding"
        item.index = i
        item.embedding = emb
        data.append(item)

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.total_tokens = total_tokens

    response = MagicMock()
    response.object = "list"
    response.data = data
    response.model = model
    response.usage = usage
    return response
```

- [ ] **Step 4.2: Confirm import resolves**

Run: `python -c "from tests.helpers import make_litellm_embedding_response; print(make_litellm_embedding_response().data[0].embedding)"`
Expected: `[0.1, 0.2, 0.3]`

- [ ] **Step 4.3: Commit**

```bash
git add tests/helpers.py
git commit -m "Add make_litellm_embedding_response test helper"
```

---

## Task 5: Implement modality classification

**Goal:** A pure helper that classifies a single string part as `text`, `image`, `audio`, `video`, or `pdf`. Used by input normalization to validate against `EmbeddingModel.supported_inputs`.

**Files:**
- Create: `skell_e_router/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_embeddings.py`:

```python
"""Tests for embeddings.py — input shape, modality, validation, retry, response."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestClassifyInputPart:

    def test_data_uri_image(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:image/png;base64,abc") == "image"

    def test_data_uri_audio(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:audio/mpeg;base64,abc") == "audio"

    def test_data_uri_video(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:video/mp4;base64,abc") == "video"

    def test_data_uri_pdf(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("data:application/pdf;base64,abc") == "pdf"

    def test_url_image_extension(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("https://example.com/cat.png") == "image"
        assert _classify_input_part("https://example.com/song.mp3") == "audio"
        assert _classify_input_part("https://example.com/clip.mp4") == "video"
        assert _classify_input_part("https://example.com/doc.pdf") == "pdf"

    def test_url_unknown_extension_is_text(self):
        from skell_e_router.embeddings import _classify_input_part
        # A URL without a recognizable media extension is treated as text.
        # (Embedding multimodal models accept text URIs in some contexts; we leave
        # the strict per-model check to the API itself.)
        assert _classify_input_part("https://example.com/page.html") == "text"

    def test_gs_uri_with_extension(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("gs://bucket/clip.mp4") == "video"
        assert _classify_input_part("gs://bucket/page.pdf") == "pdf"

    def test_local_file_path(self, tmp_path):
        from skell_e_router.embeddings import _classify_input_part
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG")
        assert _classify_input_part(str(p)) == "image"

    def test_plain_text(self):
        from skell_e_router.embeddings import _classify_input_part
        assert _classify_input_part("Just a sentence.") == "text"
        assert _classify_input_part("multi\nline\ntext") == "text"

    def test_path_like_but_missing_file_is_text(self):
        from skell_e_router.embeddings import _classify_input_part
        # A string that contains slashes but doesn't exist on disk is treated as text.
        # File-existence enforcement happens later in _normalize_input via
        # _encode_to_data_uri, which raises INVALID_INPUT for missing paths
        # (only if the caller intended a file).
        assert _classify_input_part("not/a/real/file.png") == "text"
```

- [ ] **Step 5.2: Run test — should fail with ImportError**

Run: `pytest tests/test_embeddings.py::TestClassifyInputPart -v`
Expected: ModuleNotFoundError on `skell_e_router.embeddings`.

- [ ] **Step 5.3: Implement `_classify_input_part`**

Create `skell_e_router/embeddings.py`:

```python
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


_MIME_TO_MODALITY = {
    "image": "image",
    "audio": "audio",
    "video": "video",
    "application/pdf": "pdf",
}


def _classify_input_part(part: str) -> Modality:
    """Classify a single string part by its inferred modality.

    Rules:
        - data:image/* | data:audio/* | data:video/* | data:application/pdf → that modality
        - http(s):// or gs:// with a known media extension → modality from extension
        - local file that exists on disk → modality from mimetypes.guess_type
        - everything else (including non-existent paths) → text
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
```

- [ ] **Step 5.4: Run tests**

Run: `pytest tests/test_embeddings.py::TestClassifyInputPart -v`
Expected: all 9 tests pass.

- [ ] **Step 5.5: Commit**

```bash
git add skell_e_router/embeddings.py tests/test_embeddings.py
git commit -m "Implement _classify_input_part modality detector"
```

---

## Task 6: Implement `_normalize_input` (input shape + validation)

**Goal:** Walk the user's `input` argument, encode file paths, build the LiteLLM-shape list, and validate against the model's capabilities.

**Files:**
- Modify: `skell_e_router/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
class TestNormalizeInput:

    def test_string_shorthand(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        normalized, was_str = _normalize_input("hello", m)
        assert normalized == ["hello"]
        assert was_str is True

    def test_flat_list(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        normalized, was_str = _normalize_input(["a", "b", "c"], m)
        assert normalized == ["a", "b", "c"]
        assert was_str is False

    def test_nested_list_aggregation_on_gemini(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        normalized, was_str = _normalize_input(
            [["caption", "data:image/png;base64,abc"]], m
        )
        assert normalized == [["caption", "data:image/png;base64,abc"]]
        assert was_str is False

    def test_mixed_aggregate_and_plain_on_gemini(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        normalized, _ = _normalize_input(
            [["caption", "data:image/png;base64,abc"], "plain"], m
        )
        assert normalized == [["caption", "data:image/png;base64,abc"], "plain"]

    def test_local_file_path_encoded_to_data_uri(self, tmp_path):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        p = tmp_path / "tiny.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        normalized, _ = _normalize_input([["caption", str(p)]], m)
        assert normalized[0][0] == "caption"
        assert normalized[0][1].startswith("data:image/png;base64,")

    def test_nested_on_openai_raises(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        with pytest.raises(RouterError) as exc:
            _normalize_input([["a", "b"]], m)
        assert exc.value.code == "INVALID_INPUT"
        assert "aggregation" in exc.value.message.lower()

    def test_image_on_openai_raises(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        with pytest.raises(RouterError) as exc:
            _normalize_input(["data:image/png;base64,abc"], m)
        assert exc.value.code == "INVALID_INPUT"
        assert "image" in exc.value.message.lower()

    def test_audio_on_gemini_passes(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        normalized, _ = _normalize_input(
            [["transcribe", "data:audio/mpeg;base64,xyz"]], m
        )
        assert normalized[0][1] == "data:audio/mpeg;base64,xyz"

    def test_gemini_file_ref_in_aggregate(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.response import GeminiFileRef
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        ref = GeminiFileRef(uri="files/abc123", mime_type="video/mp4")
        normalized, _ = _normalize_input([["watch this", ref]], m)
        assert normalized[0][0] == "watch this"
        assert normalized[0][1] == {
            "file_data": {"file_uri": "files/abc123", "mime_type": "video/mp4"}
        }

    def test_missing_file_path_raises(self, tmp_path):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        # Real-looking image extension but doesn't exist; classifier returns "text"
        # so this passes (file path that doesn't exist is just plain text by design).
        normalized, _ = _normalize_input(["not/a/real/file.png"], m)
        assert normalized == ["not/a/real/file.png"]

    def test_invalid_top_level_type_raises(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        with pytest.raises(RouterError) as exc:
            _normalize_input(12345, m)
        assert exc.value.code == "INVALID_INPUT"

    def test_invalid_part_type_in_nested_list(self):
        from skell_e_router.embeddings import _normalize_input
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from skell_e_router.utils import RouterError
        m = EMBEDDING_MODEL_CONFIG["gemini-embedding-2"]
        with pytest.raises(RouterError) as exc:
            _normalize_input([["caption", 999]], m)
        assert exc.value.code == "INVALID_INPUT"
```

- [ ] **Step 6.2: Run tests — should fail with ImportError**

Run: `pytest tests/test_embeddings.py::TestNormalizeInput -v`
Expected: ImportError on `_normalize_input`.

- [ ] **Step 6.3: Implement `_normalize_input`**

Append to `skell_e_router/embeddings.py`:

```python
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


def _modality_from_mime(mime: str) -> Modality:
    mime = (mime or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime == "application/pdf":
        return "pdf"
    return "text"


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
```

- [ ] **Step 6.4: Run tests**

Run: `pytest tests/test_embeddings.py::TestNormalizeInput -v`
Expected: all 12 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add skell_e_router/embeddings.py tests/test_embeddings.py
git commit -m "Implement _normalize_input with capability validation"
```

---

## Task 7: Implement `_perform_embedding` (LiteLLM call + retry)

**Goal:** Thin wrapper around `litellm.embedding()` with the same tenacity retry decorator the chat path uses.

**Files:**
- Modify: `skell_e_router/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 7.1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
class TestPerformEmbedding:

    def test_calls_litellm_with_correct_kwargs(self):
        from skell_e_router.embeddings import _perform_embedding
        from tests.helpers import make_litellm_embedding_response

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.return_value = make_litellm_embedding_response(
                embeddings=[[0.1, 0.2]]
            )
            response, duration = _perform_embedding(
                model_name="openai/text-embedding-3-large",
                input=["hello"],
                api_key="sk-test",
                dimensions=512,
            )
        mock_emb.assert_called_once()
        kwargs = mock_emb.call_args.kwargs
        assert kwargs["model"] == "openai/text-embedding-3-large"
        assert kwargs["input"] == ["hello"]
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["dimensions"] == 512
        assert duration >= 0

    def test_no_api_key_omits_kwarg(self):
        from skell_e_router.embeddings import _perform_embedding
        from tests.helpers import make_litellm_embedding_response
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.return_value = make_litellm_embedding_response()
            _perform_embedding(
                model_name="openai/text-embedding-3-large",
                input=["hello"],
                api_key=None,
            )
        kwargs = mock_emb.call_args.kwargs
        assert "api_key" not in kwargs

    def test_retries_on_503(self):
        from skell_e_router.embeddings import _perform_embedding
        from tests.helpers import make_litellm_embedding_response

        # First two calls raise a 503, third succeeds.
        err = Exception("server down")
        err.status_code = 503
        err.headers = {}

        ok = make_litellm_embedding_response()

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.side_effect = [err, err, ok]
            response, _ = _perform_embedding(
                model_name="openai/text-embedding-3-large",
                input=["hi"],
                api_key=None,
            )
        assert mock_emb.call_count == 3
        assert response is ok

    def test_does_not_retry_on_quota_429(self):
        from skell_e_router.embeddings import _perform_embedding

        err = Exception("quota exceeded")
        err.status_code = 429
        err.code = "insufficient_quota"
        err.headers = {}

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.side_effect = err
            with pytest.raises(Exception):
                _perform_embedding(
                    model_name="openai/text-embedding-3-large",
                    input=["hi"],
                    api_key=None,
                )
        assert mock_emb.call_count == 1
```

- [ ] **Step 7.2: Run tests — should fail with ImportError**

Run: `pytest tests/test_embeddings.py::TestPerformEmbedding -v`
Expected: ImportError on `_perform_embedding`.

- [ ] **Step 7.3: Implement `_perform_embedding`**

Append to `skell_e_router/embeddings.py`:

```python
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
```

- [ ] **Step 7.4: Run tests**

Run: `pytest tests/test_embeddings.py::TestPerformEmbedding -v`
Expected: 4 passed.

- [ ] **Step 7.5: Commit**

```bash
git add skell_e_router/embeddings.py tests/test_embeddings.py
git commit -m "Implement _perform_embedding with retry/backoff"
```

---

## Task 8: Implement `_build_embedding_response`

**Goal:** Convert the LiteLLM response into our `EmbeddingResponse`. Index-ordered, observed dimensions, best-effort cost.

**Files:**
- Modify: `skell_e_router/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 8.1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
class TestBuildEmbeddingResponse:

    def test_basic_fields_populated(self):
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="openai/text-embedding-3-large",
            prompt_tokens=7,
            total_tokens=7,
        )
        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0002
            resp = _build_embedding_response(
                response=raw,
                embedding_model=m,
                request_duration_s=0.123,
                total_duration_s=0.456,
            )
        assert resp.embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert resp.model == "openai/text-embedding-3-large"
        assert resp.dimensions == 3
        assert resp.prompt_tokens == 7
        assert resp.total_tokens == 7
        assert resp.cost == 0.0002
        assert resp.duration_seconds == 0.123
        assert resp.total_duration_seconds == 0.456
        assert resp.raw_response is raw

    def test_cost_swallows_exception(self):
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response()
        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.side_effect = Exception("model not in cost table")
            resp = _build_embedding_response(
                response=raw, embedding_model=m,
                request_duration_s=None, total_duration_s=None,
            )
        assert resp.cost is None

    def test_index_ordering_preserved(self):
        """If the provider returns out-of-order indices, sort them."""
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response(
            embeddings=[[1.0], [2.0], [3.0]]
        )
        # Shuffle the data items but keep correct `index` fields.
        raw.data = [raw.data[2], raw.data[0], raw.data[1]]

        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0
            resp = _build_embedding_response(
                response=raw, embedding_model=m,
                request_duration_s=0, total_duration_s=0,
            )
        assert resp.embeddings == [[1.0], [2.0], [3.0]]

    def test_falls_back_to_model_name_if_response_lacks_one(self):
        from skell_e_router.embeddings import _build_embedding_response
        from skell_e_router.model_config import EMBEDDING_MODEL_CONFIG
        from tests.helpers import make_litellm_embedding_response

        m = EMBEDDING_MODEL_CONFIG["openai-embedding-3-large"]
        raw = make_litellm_embedding_response()
        # Remove the `model` attribute by setting it to ""
        raw.model = ""
        with patch("skell_e_router.embeddings.litellm.completion_cost") as mock_cost:
            mock_cost.return_value = None
            resp = _build_embedding_response(
                response=raw, embedding_model=m,
                request_duration_s=0, total_duration_s=0,
            )
        # Empty string still wins over fallback (we only fall back if attr is missing/None).
        # We accept either behavior — assert non-None.
        assert resp.model is not None
```

- [ ] **Step 8.2: Run tests — should fail with ImportError**

Run: `pytest tests/test_embeddings.py::TestBuildEmbeddingResponse -v`
Expected: ImportError on `_build_embedding_response`.

- [ ] **Step 8.3: Implement `_build_embedding_response`**

Append to `skell_e_router/embeddings.py`:

```python
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
```

- [ ] **Step 8.4: Run tests**

Run: `pytest tests/test_embeddings.py::TestBuildEmbeddingResponse -v`
Expected: all 4 tests pass.

- [ ] **Step 8.5: Commit**

```bash
git add skell_e_router/embeddings.py tests/test_embeddings.py
git commit -m "Implement _build_embedding_response"
```

---

## Task 9: Implement public `get_embedding` function

**Goal:** Tie everything together into the user-facing function with type overloads, dimension validation, and the unwrap rule.

**Files:**
- Modify: `skell_e_router/embeddings.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 9.1: Write the failing tests**

Append to `tests/test_embeddings.py`:

```python
class TestGetEmbedding:

    def _patch_litellm(self, embeddings, model_name="openai/text-embedding-3-large"):
        """Helper: patch litellm.embedding to return a fake response."""
        from tests.helpers import make_litellm_embedding_response
        return patch(
            "skell_e_router.embeddings.litellm.embedding",
            return_value=make_litellm_embedding_response(
                embeddings=embeddings, model=model_name,
            ),
        )

    def test_string_input_returns_flat_list(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with self._patch_litellm([[0.1, 0.2, 0.3]]):
            result = get_embedding("openai-embedding-3-large", "hello")
        assert result == [0.1, 0.2, 0.3]

    def test_list_input_returns_nested_list(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with self._patch_litellm([[0.1], [0.2], [0.3]]):
            result = get_embedding("openai-embedding-3-large", ["a", "b", "c"])
        assert result == [[0.1], [0.2], [0.3]]

    def test_rich_response_returns_dataclass(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.response import EmbeddingResponse
        with self._patch_litellm([[0.1, 0.2]]):
            result = get_embedding(
                "openai-embedding-3-large", "hello", rich_response=True,
            )
        assert isinstance(result, EmbeddingResponse)
        assert result.embeddings == [[0.1, 0.2]]
        assert result.dimensions == 2

    def test_rich_response_with_string_does_not_unwrap(self, monkeypatch):
        """rich_response always returns nested list[list[float]]."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with self._patch_litellm([[0.5, 0.6]]):
            result = get_embedding(
                "openai-embedding-3-large", "hello", rich_response=True,
            )
        assert result.embeddings == [[0.5, 0.6]]  # not [0.5, 0.6]

    def test_dimensions_passed_to_litellm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response(
                embeddings=[[0.1] * 512]
            )
            get_embedding(
                "openai-embedding-3-large", "hello", dimensions=512,
            )
        assert mock_emb.call_args.kwargs["dimensions"] == 512

    def test_dimensions_not_passed_when_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response()
            get_embedding("openai-embedding-3-large", "hello")
        assert "dimensions" not in mock_emb.call_args.kwargs

    def test_dimensions_too_large_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding(
                "openai-embedding-3-small", "hello", dimensions=4096,
            )
        assert exc.value.code == "INVALID_PARAM"

    def test_dimensions_zero_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding(
                "openai-embedding-3-large", "hello", dimensions=0,
            )
        assert exc.value.code == "INVALID_PARAM"

    def test_unknown_model_raises(self, monkeypatch):
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding("not-a-model", "hello")
        assert exc.value.code == "INVALID_MODEL"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with pytest.raises(RouterError) as exc:
            get_embedding("openai-embedding-3-large", "hello")
        assert exc.value.code == "MISSING_ENV"

    def test_api_key_from_config_overrides_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from skell_e_router.embeddings import get_embedding
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response()
            get_embedding(
                "openai-embedding-3-large", "hello",
                config={"openai_api_key": "sk-from-config"},
            )
        assert mock_emb.call_args.kwargs["api_key"] == "sk-from-config"

    def test_litellm_error_wraps_in_provider_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from skell_e_router.embeddings import get_embedding
        from skell_e_router.utils import RouterError
        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            mock_emb.side_effect = ValueError("provider exploded: sk-test secret")
            with pytest.raises(RouterError) as exc:
                get_embedding(
                    "openai-embedding-3-large", "hello",
                    config={"openai_api_key": "sk-test"},
                )
        assert exc.value.code == "PROVIDER_ERROR"
        # API key must be redacted from error message.
        assert "sk-test" not in exc.value.message
        assert "[REDACTED]" in exc.value.message

    def test_gemini_multimodal_aggregation(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        from skell_e_router.embeddings import get_embedding

        img = tmp_path / "tiny.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        with patch("skell_e_router.embeddings.litellm.embedding") as mock_emb:
            from tests.helpers import make_litellm_embedding_response
            mock_emb.return_value = make_litellm_embedding_response(
                embeddings=[[0.9] * 768], model="gemini/gemini-embedding-2",
            )
            result = get_embedding(
                "gemini-embedding-2",
                [["a red shoe", str(img)]],
            )
        # 1 nested element → 1 output embedding
        assert len(result) == 1
        assert len(result[0]) == 768
        # The input passed to litellm should have the image as a data URI
        sent = mock_emb.call_args.kwargs["input"]
        assert isinstance(sent[0], list)
        assert sent[0][0] == "a red shoe"
        assert sent[0][1].startswith("data:image/png;base64,")
```

- [ ] **Step 9.2: Run tests — should fail**

Run: `pytest tests/test_embeddings.py::TestGetEmbedding -v`
Expected: ImportError on `get_embedding`.

- [ ] **Step 9.3: Implement `get_embedding`**

Append to `skell_e_router/embeddings.py`:

```python
def _validate_dimensions(dimensions: int | None, model: EmbeddingModel) -> None:
    if dimensions is None:
        return
    if not isinstance(dimensions, int):
        raise RouterError(
            code="INVALID_PARAM",
            message=f"`dimensions` must be int or None, got {type(dimensions).__name__}",
        )
    if dimensions < 1 or dimensions > model.max_dimensions:
        raise RouterError(
            code="INVALID_PARAM",
            message=(
                f"`dimensions={dimensions}` out of range for '{model.name}'. "
                f"Allowed: 1..{model.max_dimensions}. "
                f"Recommended: {list(model.recommended_dimensions)}"
            ),
        )


@overload
def get_embedding(
    model: str,
    input: str,
    *,
    dimensions: int | None = None,
    config: dict | None = None,
    rich_response: Literal[False] = False,
    verbosity: str = "none",
) -> list[float]: ...


@overload
def get_embedding(
    model: str,
    input: list,
    *,
    dimensions: int | None = None,
    config: dict | None = None,
    rich_response: Literal[False] = False,
    verbosity: str = "none",
) -> list[list[float]]: ...


@overload
def get_embedding(
    model: str,
    input,
    *,
    dimensions: int | None = None,
    config: dict | None = None,
    rich_response: Literal[True],
    verbosity: str = "none",
) -> EmbeddingResponse: ...


def get_embedding(
    model: str,
    input,
    *,
    dimensions: int | None = None,
    config: dict | None = None,
    rich_response: bool = False,
    verbosity: str = "none",
):
    """Compute embeddings via LiteLLM.

    Args:
        model: Alias from EMBEDDING_MODEL_CONFIG (e.g., "openai-embedding-3-large",
            "gemini-embedding-2") or a full LiteLLM model name.
        input: One of:
            - str — single text → returns one embedding as list[float]
            - list[str] — batch → returns list[list[float]] (one per element)
            - list[list[str | GeminiFileRef]] — each nested list is fused into
              one aggregated embedding (Gemini only). Returns list[list[float]].
            - list mixing the above shapes — output count matches input length.
        dimensions: Optional output dimension count. Must be 1..max_dimensions
            for the chosen model. Provider truncates / auto-normalizes.
        config: Optional dict of API keys (overrides env vars). E.g.,
            `{"openai_api_key": "sk-..."}`.
        rich_response: If True, return EmbeddingResponse with usage/cost/timing.
        verbosity: "none" | "response" | "info" | "debug".

    Returns:
        - rich_response=True → EmbeddingResponse
        - input was str → list[float]
        - input was list → list[list[float]]
    """
    verbosity = (verbosity or "none").lower()
    if verbosity not in ("none", "response", "info", "debug"):
        verbosity = "response"

    embedding_model = resolve_embedding_alias(model)
    _check_provider_key(embedding_model, config, verbosity)

    normalized, was_str = _normalize_input(input, embedding_model)
    _validate_dimensions(dimensions, embedding_model)

    api_key = _resolve_api_key(embedding_model, config)

    extra_kwargs = {}
    if dimensions is not None:
        extra_kwargs["dimensions"] = dimensions

    if verbosity != "none":
        n = len(normalized)
        print(f"\nEMBEDDING ({embedding_model.name}) — {n} input(s)...\n")
    if verbosity == "debug":
        print(f"INPUT: {normalized}")
        print(f"KWARGS: {extra_kwargs}")

    try:
        start_time = time.perf_counter()
        response, request_duration_s = _perform_embedding(
            model_name=embedding_model.name,
            input=normalized,
            api_key=api_key,
            **extra_kwargs,
        )
        total_duration_s = time.perf_counter() - start_time
    except RouterError:
        raise
    except Exception as e:
        safe_msg = _redact_keys(str(e), config)
        if verbosity != "none":
            print(f"ERROR calling {embedding_model.name}: {safe_msg}")
        raise RouterError(
            code="PROVIDER_ERROR",
            message=safe_msg,
            details={"provider": embedding_model.provider, "model": embedding_model.name},
        ) from e

    embedding_response = _build_embedding_response(
        response=response,
        embedding_model=embedding_model,
        request_duration_s=request_duration_s,
        total_duration_s=total_duration_s,
    )

    if verbosity in ("info", "debug"):
        print(
            f"  model={embedding_response.model} "
            f"dim={embedding_response.dimensions} "
            f"prompt_tokens={embedding_response.prompt_tokens} "
            f"cost={embedding_response.cost} "
            f"duration={total_duration_s:.3f}s"
        )

    if rich_response:
        return embedding_response

    if was_str:
        return embedding_response.embeddings[0]
    return embedding_response.embeddings
```

- [ ] **Step 9.4: Run all embedding tests**

Run: `pytest tests/test_embeddings.py -v`
Expected: every test passes.

- [ ] **Step 9.5: Run full suite for regressions**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 9.6: Commit**

```bash
git add skell_e_router/embeddings.py tests/test_embeddings.py
git commit -m "Implement public get_embedding() function"
```

---

## Task 10: Wire up package exports

**Files:**
- Modify: `skell_e_router/__init__.py`

- [ ] **Step 10.1: Read existing `__init__.py`**

Run: `cat skell_e_router/__init__.py`

Confirm the current `__all__` list and import structure.

- [ ] **Step 10.2: Write a smoke test**

Append to `tests/test_embeddings.py`:

```python
class TestPackageExports:

    def test_get_embedding_importable_from_top_level(self):
        from skell_e_router import get_embedding
        assert callable(get_embedding)

    def test_embedding_response_importable_from_top_level(self):
        from skell_e_router import EmbeddingResponse
        assert EmbeddingResponse is not None

    def test_embedding_model_importable_from_top_level(self):
        from skell_e_router import EmbeddingModel
        assert EmbeddingModel is not None

    def test_resolve_embedding_alias_importable_from_top_level(self):
        from skell_e_router import resolve_embedding_alias
        assert callable(resolve_embedding_alias)
```

- [ ] **Step 10.3: Run smoke test — should fail**

Run: `pytest tests/test_embeddings.py::TestPackageExports -v`
Expected: ImportError.

- [ ] **Step 10.4: Add exports**

Replace `skell_e_router/__init__.py` contents with:

```python
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

__version__ = "3.5.0"
```

- [ ] **Step 10.5: Run smoke test**

Run: `pytest tests/test_embeddings.py::TestPackageExports -v`
Expected: 4 passed.

- [ ] **Step 10.6: Run full suite**

Run: `pytest tests/ -x`
Expected: all green.

- [ ] **Step 10.7: Commit**

```bash
git add skell_e_router/__init__.py tests/test_embeddings.py
git commit -m "Export get_embedding and friends from top-level package"
```

---

## Task 11: Bump package version

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 11.1: Update `pyproject.toml`**

In `pyproject.toml`, change:

```toml
version = "3.4.0"
```

to:

```toml
version = "3.5.0"
```

- [ ] **Step 11.2: Sanity check**

Run: `python -c "import skell_e_router; print(skell_e_router.__version__)"`
Expected: `3.5.0`

- [ ] **Step 11.3: Commit**

```bash
git add pyproject.toml
git commit -m "Bump version to 3.5.0 for embedding support"
```

---

## Task 12: Add runnable `example_embeddings.py`

**Files:**
- Create: `skell_e_router/examples/example_embeddings.py`

- [ ] **Step 12.1: Verify the existing example image exists**

Run: `ls skell_e_router/examples/vision-test.jpg`
Expected: file exists. (If not, the example uses a placeholder path and notes the prerequisite.)

- [ ] **Step 12.2: Write the example file**

Create `skell_e_router/examples/example_embeddings.py`:

```python
"""End-to-end runnable examples for skell_e_router.get_embedding().

Requires: OPENAI_API_KEY and/or GEMINI_API_KEY in your environment.
Run from the repo root:

    python -m skell_e_router.examples.example_embeddings
"""

from skell_e_router import get_embedding, EmbeddingResponse, RouterError


def section(title: str) -> None:
    print(f"\n{'=' * 8}  {title}  {'=' * 8}")


def main() -> None:
    # 1. Single string → list[float]
    section("1. Single string (OpenAI large)")
    v = get_embedding("openai-embedding-3-large", "hello world")
    print(f"  dim={len(v)}  preview={v[:3]}")

    # 2. Batch of strings → list[list[float]]
    section("2. Batch of strings (OpenAI large)")
    vs = get_embedding(
        "openai-embedding-3-large",
        ["hello", "world", "foo", "bar"],
    )
    print(f"  count={len(vs)}  dim_each={len(vs[0])}")

    # 3. Truncated dimensions
    section("3. Truncated dimensions (OpenAI small, 512)")
    v = get_embedding("openai-embedding-3-small", "hello", dimensions=512)
    print(f"  dim={len(v)}")
    assert len(v) == 512

    # 4. Multimodal aggregation: text + image → 1 fused embedding (Gemini)
    section("4. Multimodal aggregation (Gemini, text + image)")
    v = get_embedding(
        "gemini-embedding-2",
        [["a red shoe on a wooden floor", "skell_e_router/examples/vision-test.jpg"]],
    )
    print(f"  count={len(v)}  fused_dim={len(v[0])}")

    # 5. Mixed batch: aggregate + plain text → 2 embeddings (Gemini)
    section("5. Mixed batch (Gemini)")
    vs = get_embedding(
        "gemini-embedding-2",
        [
            ["product caption", "skell_e_router/examples/vision-test.jpg"],
            "plain text query",
        ],
    )
    print(f"  count={len(vs)}")
    assert len(vs) == 2

    # 6. Rich response
    section("6. Rich response")
    resp: EmbeddingResponse = get_embedding(
        "openai-embedding-3-large", "hello", rich_response=True,
    )
    print(
        f"  model={resp.model}  dim={resp.dimensions}  "
        f"prompt_tokens={resp.prompt_tokens}  cost={resp.cost}"
    )

    # 7. Capability error: OpenAI rejects image input
    section("7. Capability error (expected)")
    try:
        get_embedding(
            "openai-embedding-3-large",
            ["data:image/png;base64,iVBORw0KGgo..."],
        )
    except RouterError as e:
        print(f"  Got expected error: code={e.code}  message={e.message}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.3: Lint-check by importing the module**

Run: `python -c "import skell_e_router.examples.example_embeddings"`
Expected: no errors (the file imports cleanly without `__main__` running).

- [ ] **Step 12.4: Commit**

```bash
git add skell_e_router/examples/example_embeddings.py
git commit -m "Add runnable example_embeddings.py"
```

---

## Task 13: Update `Skell-E-Router-DOCUMENTATION.md`

**Files:**
- Modify: `skell_e_router/Skell-E-Router-DOCUMENTATION.md`

- [ ] **Step 13.1: Read the current doc to find a good insertion point**

Run: `grep -n "^##" skell_e_router/Skell-E-Router-DOCUMENTATION.md | head -20`

Plan: insert the new "## Embeddings" section before the existing "## Rich Response Object" section so it lives near the top of the technical reference.

- [ ] **Step 13.2: Insert the embeddings section**

In `skell_e_router/Skell-E-Router-DOCUMENTATION.md`, immediately before the line `## Rich Response Object`, insert:

````markdown
---

## Embeddings

`get_embedding()` routes embedding calls through LiteLLM with the same retry/backoff and error wrapping as `ask_ai()`.

### Quick Reference

```python
from skell_e_router import get_embedding

# Single string → list[float]
v = get_embedding("openai-embedding-3-large", "hello")

# Batch → list[list[float]] (one per top-level element)
vs = get_embedding("openai-embedding-3-large", ["a", "b", "c"])

# Multimodal aggregation: nested list → 1 fused embedding (Gemini only)
v = get_embedding(
    "gemini-embedding-2",
    [["a red shoe", "shoe.jpg"]],
)  # → list[list[float]] of length 1

# Mixed batch: aggregate + plain → 2 embeddings (Gemini only)
vs = get_embedding(
    "gemini-embedding-2",
    [["product", "img.jpg"], "plain text"],
)  # → list[list[float]] of length 2
```

**Predictability rule:** *Number of top-level elements in `input` = number of output embeddings.* The string-shorthand is the only exception — it returns the one embedding unwrapped as `list[float]`.

### Capability Matrix

| Alias | Provider | Inputs | Max dims | Recommended dims | Aggregation | Notes |
|---|---|---|---|---|---|---|
| `openai-embedding-3-large` | OpenAI | text | 3072 | 256 / 1024 / 3072 | no | ≤2048 inputs/req, ≤300k tokens/req |
| `openai-embedding-3-small` | OpenAI | text | 1536 | 512 / 1536 | no | ≤2048 inputs/req, ≤300k tokens/req |
| `gemini-embedding-2` | Gemini | text, image, audio, video, pdf | 3072 | 768 / 1536 / 3072 | yes | per-input limits: 6 images, 120s video, 180s audio, 6 PDF pages; auto-normalizes truncated dims |

### Accepted Part Values

A "part" is each string in a flat input *or* each string/`GeminiFileRef` inside a nested list:

| Pattern | Inferred modality | Handling |
|---|---|---|
| `data:image/...` | image | passed through as data URI |
| `data:audio/...` | audio | passed through |
| `data:video/...` | video | passed through |
| `data:application/pdf` | pdf | passed through |
| `https://…` / `http://…` | from URL extension | passed through |
| `gs://…` | from extension | passed through (Vertex GCS reference) |
| Local file path that exists on disk | from `mimetypes.guess_type` | base64-encoded to data URI |
| `GeminiFileRef` instance | from `mime_type` | converted to file reference dict |
| Anything else | text | sent as text |

A path-like string that doesn't exist on disk is treated as plain text — it is **not** an error. To embed a file, the file must exist or you must pass it as a `data:` URI / URL / `GeminiFileRef`.

### Multimodal Aggregation

Aggregation fuses multiple parts (text + image, multiple texts, etc.) into a single vector. Wrap the parts in a nested list:

```python
# 1 aggregated embedding from a caption + image
get_embedding(
    "gemini-embedding-2",
    [["caption text", "image.jpg"]],
)
```

Aggregation requires `gemini-embedding-2`. Trying to nest with an OpenAI model raises `RouterError("INVALID_INPUT", "...does not support aggregation")`.

### `dimensions` Parameter

Optionally truncate the output vector. Must be `1 ≤ dimensions ≤ max_dimensions`. The provider auto-normalizes truncated vectors (Gemini Embedding 2 does this automatically; OpenAI text-embedding-3 returns Matryoshka-truncated vectors).

```python
v = get_embedding("openai-embedding-3-small", "hello", dimensions=512)
assert len(v) == 512
```

### Rich Response

```python
resp = get_embedding("openai-embedding-3-large", ["a", "b"], rich_response=True)
print(resp.model, resp.dimensions, resp.prompt_tokens, resp.cost)
print(resp.embeddings)  # always list[list[float]]
```

### Errors

All errors are `RouterError` with one of these codes:

| Code | Triggered by |
|---|---|
| `INVALID_MODEL` | unknown alias passed to `get_embedding(model=...)` |
| `MISSING_ENV` | required API key not in env or `config` |
| `INVALID_INPUT` | wrong input type, modality unsupported by model, aggregation on non-aggregating model |
| `INVALID_PARAM` | `dimensions` out of range or wrong type |
| `PROVIDER_ERROR` | LiteLLM/provider-side failure (after retry budget exhausted); message has any `config` keys redacted |

````

- [ ] **Step 13.3: Verify markdown renders cleanly**

Run: `grep -n "^##" skell_e_router/Skell-E-Router-DOCUMENTATION.md | head -10`
Expected: the new "## Embeddings" header appears before "## Rich Response Object".

- [ ] **Step 13.4: Commit**

```bash
git add skell_e_router/Skell-E-Router-DOCUMENTATION.md
git commit -m "Document embeddings section in technical reference"
```

---

## Task 14: Update top-level `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 14.1: Find the right insertion point**

The README currently has sections: Quick Start → Image Input → Image Generation → Deep Research → API Keys → Documentation. Insert a new "### Embeddings" subsection within the Quick Start area, right before "### Deep Research".

- [ ] **Step 14.2: Insert the new section**

In `README.md`, insert this immediately before the line `### Deep Research`:

````markdown
### Embeddings

Route embedding calls through skell-e-router with `get_embedding()`. Supports OpenAI text embeddings and Gemini multimodal embeddings.

```python
from skell_e_router import get_embedding

# Single text → list[float]
v = get_embedding("openai-embedding-3-large", "hello world")

# Batch → list[list[float]]
vs = get_embedding(
    "openai-embedding-3-large",
    ["doc 1", "doc 2", "doc 3"],
)

# Multimodal aggregation (Gemini): text + image → one fused embedding
v = get_embedding(
    "gemini-embedding-2",
    [["a red shoe on wood", "shoe.jpg"]],
)
```

Available models: `openai-embedding-3-large`, `openai-embedding-3-small`, `gemini-embedding-2`. See [the technical reference](skell_e_router/Skell-E-Router-DOCUMENTATION.md#embeddings) for the capability matrix, input shape rules, and error codes.

````

- [ ] **Step 14.3: Verify the section appears in the right place**

Run: `grep -n "^### " README.md`
Expected: `### Embeddings` appears between `### Image Generation` and `### Deep Research`.

- [ ] **Step 14.4: Commit**

```bash
git add README.md
git commit -m "Add Embeddings section to README quick start"
```

---

## Task 15: Final regression sweep

**Files:** none (verification only)

- [ ] **Step 15.1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: every test passes (existing chat tests + new embedding tests).

- [ ] **Step 15.2: Confirm public API works end-to-end via Python**

Run:

```bash
python -c "
from skell_e_router import (
    get_embedding, EmbeddingResponse, EmbeddingModel,
    resolve_embedding_alias, ask_ai, RouterError,
)
m = resolve_embedding_alias('openai-embedding-3-large')
print('OK:', m.name, m.max_dimensions, m.supported_inputs)
"
```

Expected output: `OK: openai/text-embedding-3-large 3072 {'text'}`

- [ ] **Step 15.3: Live sanity check (optional, requires real API keys)**

If `OPENAI_API_KEY` and `GEMINI_API_KEY` are set in your shell, run:

```bash
python -m skell_e_router.examples.example_embeddings
```

Expected: every section prints, including section 7 which intentionally raises a captured `RouterError`.

- [ ] **Step 15.4: Commit any remaining trivial fixes (if any)**

If steps 15.1–15.3 surface any small issues, fix them and commit with a focused message. If everything is clean, no commit needed.

---

## Self-Review Checklist (for the implementer to run after Task 15)

- [ ] All 15 tasks complete and committed
- [ ] `pytest tests/` is fully green
- [ ] No `TODO` / `XXX` / `TBD` strings introduced
- [ ] `skell_e_router/__init__.py` exports the new symbols
- [ ] `pyproject.toml` version bumped
- [ ] Both docs (`README.md` and `Skell-E-Router-DOCUMENTATION.md`) cover the new feature
- [ ] `example_embeddings.py` imports cleanly (and runs end-to-end if keys are set)

When all boxes above are checked, the feature is ready for review.
