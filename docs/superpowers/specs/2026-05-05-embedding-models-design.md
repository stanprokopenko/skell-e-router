# Embedding-model support for skell-e-router

**Date:** 2026-05-05
**Status:** Design approved, pre-implementation
**Author:** Stan Prokopenko (with Claude)

## Goal

Add embedding-model support to `skell-e-router` so callers can route OpenAI and Gemini embedding requests through the same router that already handles chat. Initial scope: text and multimodal embeddings, single and batched, with the same retry/backoff and error wrapping the chat path uses.

## Models in scope

Three models, all routed via `litellm.embedding()`:

| Alias | LiteLLM identifier | Provider | Modalities | Max dims | Recommended dims | Default dims | Max input tokens | Aggregation | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `openai-embedding-3-large` | `openai/text-embedding-3-large` | OpenAI | text | 3072 | 256, 1024, 3072 | 3072 | 8192 | no | ≤2048 inputs/req, ≤300k tokens/req |
| `openai-embedding-3-small` | `openai/text-embedding-3-small` | OpenAI | text | 1536 | 512, 1536 | 1536 | 8192 | no | ≤2048 inputs/req, ≤300k tokens/req |
| `gemini-embedding-2` | `gemini/gemini-embedding-2` | Gemini | text, image, audio, video, pdf | 3072 | 768, 1536, 3072 | 3072 | 8192 | yes | per-input limits: 6 images, 120s video, 180s audio, 6 PDF pages; auto-normalizes truncated dims; no `task_type` param |

Older `gemini-embedding-001` (text-only, predates multimodal) is intentionally excluded; the embedding spaces between v1 and v2 are incompatible per Google's docs, and we want a single recommended path.

## Public API

One function. Singular noun, polymorphic input. Cardinality of input = cardinality of output.

```python
from typing import overload, Literal
from skell_e_router.response import EmbeddingResponse, GeminiFileRef

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
    input: list[str | list[str | GeminiFileRef]],
    *,
    dimensions: int | None = None,
    config: dict | None = None,
    rich_response: Literal[False] = False,
    verbosity: str = "none",
) -> list[list[float]]: ...

@overload
def get_embedding(
    model: str,
    input: str | list[str | list[str | GeminiFileRef]],
    *,
    dimensions: int | None = None,
    config: dict | None = None,
    rich_response: Literal[True],
    verbosity: str = "none",
) -> EmbeddingResponse: ...
```

### Input shape rules

The shape mirrors LiteLLM's nested-list semantics:

- **`input="hello"`** — string shorthand. Returns a single embedding (`list[float]`).
- **`input=["a", "b", "c"]`** — flat list. Each top-level element produces one output embedding. Returns `list[list[float]]` of length 3.
- **`input=[["caption", "img.jpg"]]`** — single-element list containing a nested list. The nested list's parts are fused into one aggregated embedding. Returns `list[list[float]]` of length 1. Requires `supports_aggregation=True` on the model.
- **`input=[["caption", "img.jpg"], "plain text"]`** — mixed. Two top-level elements → two output embeddings; the first is aggregated, the second is plain. Returns `list[list[float]]` of length 2.

**Predictability rule:** *number of top-level elements in `input` = number of output embeddings*. The string shorthand is the one exception, and it unwraps the result for ergonomic `list[float]` returns.

### Accepted part values

Each "part" (string in a flat input or string/`GeminiFileRef` inside a nested list) is one of:

| Pattern | Inferred modality | Handling |
|---|---|---|
| `data:image/...` | image | passed through as data URI |
| `data:audio/...` | audio | passed through |
| `data:video/...` | video | passed through |
| `data:application/pdf` | pdf | passed through |
| `https://...` / `http://...` (URL) | inferred from URL extension | passed through |
| `gs://...` | inferred from extension | passed through (Vertex GCS reference) |
| Local file path that exists on disk | inferred from `mimetypes.guess_type` | base64-encoded to data URI |
| `GeminiFileRef` instance | by `mime_type` | converted to LiteLLM file reference dict |
| Anything else | text | sent as text |

A string that *looks like* a path (starts with `./`, `/`, or `<drive>:\`) but doesn't exist on disk → `RouterError("INVALID_INPUT", "File not found: <path>")`. Plain text never collides because path-like strings only match the file-path branch *after* an existence check.

### Validation

Enforced before any network call:

- **Unknown alias** → `RouterError("INVALID_MODEL")`.
- **Missing API key** → `RouterError("MISSING_ENV")`.
- **Modality not supported by model** (e.g., OpenAI + image) → `RouterError("INVALID_INPUT", "<model> does not support <modality> inputs")`.
- **Aggregation on a non-aggregating model** (nested list to OpenAI) → `RouterError("INVALID_INPUT", "<model> does not support aggregation; flatten the list for batch")`.
- **`dimensions` out of range** (`< 1` or `> max_dimensions`) → `RouterError("INVALID_PARAM")`.
- **File path that doesn't exist** → `RouterError("INVALID_INPUT")`.

LiteLLM/provider exceptions surface as `RouterError("PROVIDER_ERROR", safe_msg, details={provider, model})` after running through `_redact_keys` to scrub any config-supplied secrets.

## Architecture

Pure LiteLLM. No direct-SDK path for v1. Embeddings don't carry the same per-call latency pressure as chat (no streaming, no thinking budgets), and LiteLLM already handles `gemini-embedding-2` multimodal correctly via the `batchEmbedContents` endpoint. If batch-of-250 latency proves unacceptable, we can add a direct path later — the design keeps `litellm.embedding()` behind one internal function so the swap is localized.

### File layout

| File | Status | Purpose |
|---|---|---|
| `skell_e_router/embeddings.py` | new | `get_embedding`, input normalization, retry decorator, response builder |
| `skell_e_router/model_config.py` | modified | adds `EmbeddingModel` class + `EMBEDDING_MODEL_CONFIG` + `resolve_embedding_alias` |
| `skell_e_router/response.py` | modified | adds `EmbeddingResponse` dataclass |
| `skell_e_router/utils.py` | modified | exports shared retry/key helpers (no behavior change) |
| `skell_e_router/__init__.py` | modified | exports `get_embedding`, `EmbeddingResponse`, `EmbeddingModel`, `resolve_embedding_alias` |
| `skell_e_router/examples/example_embeddings.py` | new | runnable usage examples |
| `skell_e_router/Skell-E-Router-DOCUMENTATION.md` | modified | full embeddings section + capability matrix |
| `README.md` | modified | quick-start "Embeddings" section |
| `tests/test_embeddings.py` | new | shape, validation, retry, response, alias tests |

### Reuse strategy

The chat path's helpers in `utils.py` are imported, not duplicated:

- `_is_retryable_exception`, `_retry_after_wait` — retry policy
- `_resolve_api_key`, `_check_provider_key`, `_redact_keys` — auth + redaction
- `_encode_image` — extended/generalized into `_encode_to_data_uri` if needed for non-image MIME types

The two paths share retry policy, error wrapping, key resolution, and key redaction. They diverge only at the call site (`litellm.embedding` vs `litellm.completion`) and response shape.

## Data structures

### `EmbeddingModel`

```python
class EmbeddingModel:
    def __init__(
        self,
        name: str,                              # full LiteLLM model string
        provider: str,                          # "openai" | "gemini"
        supported_inputs: set[str],             # subset of {"text","image","audio","video","pdf"}
        max_dimensions: int,
        default_dimensions: int,
        recommended_dimensions: tuple[int, ...] = (),
        max_input_tokens: int | None = None,
        supports_aggregation: bool = False,
        native_batch: bool = True,
    ): ...

    @property
    def is_openai(self) -> bool: return self.provider == "openai"
    @property
    def is_gemini(self) -> bool: return self.provider == "gemini"
```

`EMBEDDING_MODEL_CONFIG` registers the three models above. After registration, full LiteLLM names are also added as keys for direct lookup, mirroring how `MODEL_CONFIG` does it.

### `EmbeddingResponse`

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]      # always nested, even for single-input
    model: str                          # provider's reported model name
    dimensions: int                     # observed: len(embeddings[0])

    prompt_tokens: int | None = None
    total_tokens: int | None = None

    cost: float | None = None
    duration_seconds: float | None = None
    total_duration_seconds: float | None = None

    raw_response: Any = None
```

## Execution flow

1. **Resolve model** — `resolve_embedding_alias(model)` → `EmbeddingModel`.
2. **Validate verbosity** (same lower-case check as `ask_ai`).
3. **Check provider key** — `_check_provider_key(...)`; raises `MISSING_ENV`.
4. **Normalize input**:
   - Track `was_str = isinstance(input, str)`; if so, wrap as `[input]`.
   - Walk the (now always-list) input. For each top-level element:
     - If it's a string or `GeminiFileRef`, classify modality and convert to a single LiteLLM part.
     - If it's a `list`, mark aggregation; classify each part; produce a nested LiteLLM list.
   - Validate inferred modality set ⊆ `embedding_model.supported_inputs`.
   - If any aggregation occurred, require `embedding_model.supports_aggregation`.
5. **Validate `dimensions`** — `1 ≤ dimensions ≤ max_dimensions` if provided.
6. **Resolve API key** — `_resolve_api_key(embedding_model, config)`; falls through to LiteLLM's env-var lookup if `None`.
7. **Call LiteLLM with retry**:
   ```python
   @retry(
       retry=retry_if_exception(_is_retryable_exception),
       wait=_retry_after_wait,
       stop=stop_after_attempt(3),
   )
   def _perform_embedding(model_name, input, api_key=None, **kwargs):
       request_start = time.perf_counter()
       response = litellm.embedding(
           model=model_name, input=input,
           **({"api_key": api_key} if api_key else {}),
           **kwargs,
       )
       return response, time.perf_counter() - request_start
   ```
   `kwargs` is `{"dimensions": dimensions}` if set, else empty.
8. **Wrap LiteLLM exceptions** in `RouterError("PROVIDER_ERROR", _redact_keys(str(e), config), ...)`.
9. **Build response** — extract `data[*].embedding` sorted by index; compute `len(embeddings[0])`; compute cost via `litellm.completion_cost(...)` (best-effort, swallow exceptions like the chat path does).
10. **Return**:
    - `rich_response=True` → `EmbeddingResponse`.
    - `rich_response=False` and `was_str=True` → `embeddings[0]` as `list[float]`.
    - `rich_response=False` and `was_str=False` → `embeddings` as `list[list[float]]`.

## Verbosity behavior

Mirrors `ask_ai`'s policy but skips the "RESPONSE" content block (vectors are noise to print):

- `none` — silent.
- `response` — prints `ASKING AI (<model>)…` only.
- `info` — also prints model, dimension count, prompt tokens, total tokens, cost, duration.
- `debug` — also pretty-prints kwargs and the raw response.

## Testing

`tests/test_embeddings.py` mocks `litellm.embedding` via the same patching pattern used in `tests/test_utils.py`. Test groups:

| Group | Coverage |
|---|---|
| Alias resolution | unknown alias raises `INVALID_MODEL`; full-name lookup works; both short and full names resolve to the same `EmbeddingModel` |
| Single-input normalization | `str` input becomes `[str]` going in; return is unwrapped `list[float]` |
| Batch-input normalization | flat `list[str]` passes through unchanged; return is `list[list[float]]` |
| Aggregation | nested list flagged as aggregate; rejected on OpenAI with `INVALID_INPUT`; allowed on Gemini |
| File-path encoding | local image path → data URI; missing file → `INVALID_INPUT` |
| Modality classification | data URI / file path / URL / plain text classifications correct |
| Capability validation | OpenAI + image → `INVALID_INPUT`; Gemini + audio file → ok; Gemini + nested → ok |
| Dimension validation | `dimensions=4096` on small → `INVALID_PARAM`; in-range value passes through to LiteLLM kwargs |
| Retry behavior | 503 retries; 429-with-quota does not retry; 200 does not retry; respects `Retry-After` |
| Response building | `EmbeddingResponse` fields populated; `dimensions = len(embeddings[0])`; `cost=None` when `litellm.completion_cost` raises; index ordering preserved |
| API-key plumbing | `config={"openai_api_key": "..."}` overrides env; missing key raises `MISSING_ENV` |

No live network calls in unit tests. A small set of integration smoke tests can be gated behind an env flag (mirroring however existing live tests are handled in this repo, if any) — not required for first merge.

## Documentation deliverables

The user explicitly requested clear documentation covering input shapes, expected outputs, and per-model capability differences. Two locations:

1. **`Skell-E-Router-DOCUMENTATION.md`** gets a full "Embeddings" section containing:
   - Quick reference: three input shapes with example calls and described outputs.
   - The capability matrix (table at top of this spec, copied verbatim).
   - Per-model notes — Gemini auto-normalization, OpenAI batch limits, no `task_type`.
   - Multimodal aggregation explanation — when nesting fuses, when it batches.
   - File-path resolution rules — the modality classification table.
   - Errors — every `RouterError` code with example trigger.

2. **`README.md`** gets a short "Embeddings" subsection with two minimal examples (single text, multimodal aggregation) and a pointer to the full doc.

3. **`example_embeddings.py`** is runnable end-to-end and demonstrates: single, batch, dimensions, multimodal aggregation, mixed batch, rich response, and a capability-error case.

## Out of scope (explicit YAGNI)

- Direct-SDK path (Gemini/Anthropic-style bypass). Re-evaluate if batch latency is unacceptable.
- `task_type` parameter — not exposed by `gemini-embedding-2` and not implemented for `gemini-embedding-001` (which we don't support).
- Streaming — embedding APIs don't support it.
- Async API — skell-e-router is synchronous today; embedding follows suit.
- Vertex AI route for Gemini — only the Gemini Developer API path is supported. Vertex would need a separate alias and different aggregation semantics (`embedContent` instead of `batchEmbedContents`).
- Cohere, Voyage, Mistral, or other embedding providers — not requested.
- Cosine-similarity / search helpers — out of router scope.

## Open questions

None at design time. Spec-level decisions are settled.
