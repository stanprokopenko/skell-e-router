# Audio input support for `ask_ai`

**Date:** 2026-05-08
**Status:** Design approved, pre-implementation
**Author:** Stan Prokopenko (with Claude)

## Goal

Add audio inputs to `ask_ai` as a peer to `images`, so callers can attach audio segments to chat completions on multimodal models that accept them (Gemini 2.5+/3.x natively, OpenAI GPT-4o audio family). Anthropic models do not accept audio and will raise a clear error. No new model-registry entries are added — this enables capability for models already in `MODEL_CONFIG` that already support audio at the provider level.

## Why this is a code change, not just docs

A user *could* hand-craft OpenAI-canonical `input_audio` content parts and pass them via `user_input=list[dict]` today. However, the direct-SDK paths drop unknown content-part types silently:

- `gemini_direct._convert_messages_to_contents` (`gemini_direct.py:74-92`) only handles `text` and `image_url`. Any `input_audio` part is silently discarded before reaching Gemini.
- `anthropic_direct._convert_messages_for_anthropic` (`anthropic_direct.py:71-96`) has the same gap — silently drops `input_audio`.

So even users who know OpenAI's spec cannot exercise audio on Gemini through the router today. That is the bug this spec fixes. The new `audio=[...]` parameter is a thin convenience layer on top of that fix; users no longer need to know OpenAI's content-part shape.

## Public API

```python
from skell_e_router import ask_ai

# Local file path
response = ask_ai(
    "gemini-3-pro-preview",
    "Transcribe and summarize this clip.",
    audio=["interview.mp3"],
    rich_response=True,
)

# Pre-encoded data URI
response = ask_ai(
    "gemini-3-pro-preview",
    "What instrument is playing?",
    audio=["data:audio/wav;base64,UklGRi..."],
)

# Combined with images and text
response = ask_ai(
    "gemini-3-pro-preview",
    "Compare what's said in the clip with what's shown in the photo.",
    audio=["clip.mp3"],
    images=["scene.jpg"],
)
```

The new `audio` parameter is added to both `@overload` declarations and the implementation:

```python
def ask_ai(
    model_alias: str,
    user_input: str | list[dict],
    system_message: str = None,
    verbosity: str = 'none',
    rich_response: bool = False,
    config: dict | None = None,
    images: list[str] | None = None,
    audio: list[str] | None = None,        # NEW
    files: list[GeminiFileRef] | None = None,
    direct_sdk: bool | None = None,
    **kwargs,
) -> str | AIResponse:
```

### Accepted source forms

| Form | Behavior |
|---|---|
| Local file path that exists on disk | Encoded to base64 via `_encode_to_data_uri()`; MIME inferred via `mimetypes.guess_type` |
| `data:audio/<mime>;base64,...` | Passed through; MIME read from prefix |
| `http://` / `https://` URL | **Rejected** with `RouterError("INVALID_INPUT")`. URL inputs require Files API — caller should use `upload_file()` + `files=[ref]` |
| Anything else | `RouterError("INVALID_INPUT")` (e.g., non-existent path, malformed data URI) |

URLs are rejected because OpenAI's `input_audio` content-part spec requires inline base64; supporting URLs would require hidden network I/O during `_construct_messages`, which the rest of the router avoids.

### Supported MIME types

| MIME | Format suffix | Common extension |
|---|---|---|
| `audio/mpeg` | `mp3` | `.mp3` |
| `audio/mp3` (tolerant) | `mp3` | `.mp3` |
| `audio/wav` | `wav` | `.wav` |
| `audio/x-wav` (tolerant) | `wav` | `.wav` |
| `audio/flac` | `flac` | `.flac` |
| `audio/ogg` | `ogg` | `.ogg` |
| `audio/mp4` | `mp4` | `.m4a` / `.mp4` |
| `audio/webm` | `webm` | `.webm` |

Unknown MIME → `RouterError("INVALID_INPUT", "Unsupported audio MIME type: <mime>")`.

### Validation rules

These mirror the existing `images` rules:

- `audio` is only valid when `user_input` is a `str`. Combining `audio` with `list[dict]` input raises `RouterError("INVALID_INPUT")` — caller should embed parts directly in messages.
- Empty list / `None` → no-op (no content-part added).
- May be combined freely with `images` and `files`.
- No per-call cap on the number of audio attachments (provider applies its own limits; we do not preempt).

## Internal representation

We use OpenAI's canonical `input_audio` content-part shape internally, mirroring the way `images` use OpenAI's canonical `image_url` shape:

```python
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "mp3"}}
```

This is what LiteLLM forwards verbatim to the OpenAI API. Direct-SDK adapters parse this shape and convert to provider-native form.

### Why not mirror `images` with `audio_url` + data URI?

Considered. Trade-off: prettier symmetry with `image_url`, but adds a shape-conversion step at the LiteLLM boundary that does not exist for images (since OpenAI's audio spec uses raw base64 + format suffix, not data URIs). The "asymmetry" between `image_url` and `input_audio` is OpenAI's, not ours — we follow whatever the upstream canonical shape is for each modality.

## Internal flow

### `_construct_messages` (`skell_e_router/utils.py`)

Extend the signature with `audio: list[str] | None = None`. In the `isinstance(user_input, str)` branch, when *any* of `images`, `audio`, or `files` is non-empty, build the multipart content list. Iterate `audio` after `images` and append one part per audio source via a new `_encode_audio()` helper.

The list-input guard mirrors the existing image guard:

```python
if audio and not isinstance(user_input, str):
    raise RouterError(
        code="INVALID_INPUT",
        message="'audio' parameter is only supported when 'user_input' is a string. "
                "For list input, embed audio content parts directly in your messages.",
    )
```

### New helper `_encode_audio(source: str) -> dict` (`skell_e_router/utils.py`)

Resolves a string source to an OpenAI-canonical `input_audio` content part.

Steps:

1. Reject `http://` / `https://` with `RouterError("INVALID_INPUT", ...)` pointing the caller at `upload_file()` + `files=[]`.
2. If `source` starts with `data:`, parse the MIME and base64 payload from the prefix.
3. Otherwise, treat as a local file path. Reuse `_encode_to_data_uri()` to base64-encode and infer MIME via `mimetypes.guess_type`. Then strip the `data:<mime>;base64,` prefix to extract raw base64 + MIME (since OpenAI's spec wants them separately).
4. Map MIME → format suffix via `_AUDIO_MIME_TO_FORMAT`. Unknown MIME → `RouterError("INVALID_INPUT")`.
5. Return `{"type": "input_audio", "input_audio": {"data": <base64>, "format": <format>}}`.

A new module-level constant in `utils.py`:

```python
_AUDIO_MIME_TO_FORMAT = {
    "audio/mpeg": "mp3",
    "audio/mp3":  "mp3",   # tolerant
    "audio/wav":  "wav",
    "audio/x-wav":"wav",   # tolerant
    "audio/flac": "flac",
    "audio/ogg":  "ogg",
    "audio/mp4":  "mp4",   # m4a containers
    "audio/webm": "webm",
}
```

### `_convert_messages_to_contents` (`skell_e_router/gemini_direct.py`)

Extend the existing per-part dispatch (`gemini_direct.py:74-92`) with a new branch that handles `input_audio`. After base64-decoding the data and mapping format → MIME, append `types.Part.from_bytes(data=<bytes>, mime_type=<mime>)`. The reverse format → MIME map lives in `gemini_direct.py` (or is imported from `utils.py` — implementation choice during build).

```python
elif part.get("type") == "input_audio":
    audio_data = part["input_audio"]["data"]
    audio_format = part["input_audio"]["format"]
    mime = _AUDIO_FORMAT_TO_MIME.get(audio_format, f"audio/{audio_format}")
    parts.append(types.Part.from_bytes(
        data=base64.b64decode(audio_data),
        mime_type=mime,
    ))
```

### `_convert_messages_for_anthropic` (`skell_e_router/anthropic_direct.py`)

Extend the existing per-part dispatch (`anthropic_direct.py:71-96`) with an explicit error branch — silently dropping audio is rejected because it would produce a confusing answer (Claude responding as if no audio existed):

```python
elif part.get("type") == "input_audio":
    raise RouterError(
        code="UNSUPPORTED_MODALITY",
        message="Anthropic models do not support audio inputs. "
                "Use a Gemini or OpenAI audio-capable model.",
    )
```

`RouterError` is imported lazily inside the function body to avoid a circular import (matching the pattern in `_build_create_params`).

### LiteLLM path (`utils.py` `_perform_completion`)

No changes. `input_audio` content parts are forwarded by LiteLLM verbatim. Behavior per provider:

- OpenAI audio-capable models (e.g., `gpt-4o-audio-preview`, if registered later): native acceptance.
- OpenAI text-only models: provider rejects → surfaces as `RouterError("PROVIDER_ERROR")`. Same shape as today's images-on-text-only-models error.
- xAI / Groq / DeepInfra: provider rejects → `PROVIDER_ERROR`.

## Error handling summary

| Condition | Code | Where raised |
|---|---|---|
| Source is `http://` or `https://` | `INVALID_INPUT` | `_encode_audio` |
| Source path doesn't exist | `INVALID_INPUT` | `_encode_to_data_uri` (existing) |
| Unknown audio MIME type | `INVALID_INPUT` | `_encode_audio` |
| `audio` combined with `list[dict]` `user_input` | `INVALID_INPUT` | `_construct_messages` |
| Anthropic model + audio part | `UNSUPPORTED_MODALITY` | `_convert_messages_for_anthropic` |
| Other unsupported model + audio | `PROVIDER_ERROR` | bubbled up from provider |

`UNSUPPORTED_MODALITY` is a new error code. It's not used elsewhere today, so introducing it now sets the pattern for future modality gaps.

## Testing

Match the existing test layout (no live API calls):

**`tests/test_utils.py`** — parallel to existing image tests:
- `_encode_audio`: file path → `input_audio` dict, data URI passthrough, http URL rejected, unknown MIME rejected, missing file rejected.
- `_construct_messages`: string input + audio, audio + images combined, audio + files combined, audio + system_message, audio with empty list = no-op, `audio=None` = no-op, audio with `list[dict]` input rejected.

**`tests/test_gemini_direct.py`**:
- `_convert_messages_to_contents` produces a `Part.from_bytes` call with the right MIME for each supported format, when given an `input_audio` content part.
- Combined `image_url` + `input_audio` parts both end up in the `Content.parts` list.

**`tests/test_anthropic_direct.py`**:
- `_convert_messages_for_anthropic` raises `RouterError("UNSUPPORTED_MODALITY")` on `input_audio` parts.
- Existing image tests still pass (no regression).

**`skell_e_router/examples/example_audio_input.py`** — runnable example modeled on `example_image_input.py`. Uses a small bundled clip (e.g., `audio-test.mp3` next to the example, similar to how `vision-test.jpg` ships with the image example) and demonstrates both bare-string and `rich_response=True` returns.

## Documentation

**`README.md`** — new "### Audio Input" section after "### Image Input":

```markdown
### Audio Input

Send an audio clip alongside your prompt. Same accepted forms as `images` — a local file path or a base64 `data:audio/...` URI.

\`\`\`python
from skell_e_router import ask_ai

response = ask_ai(
    "gemini-3-pro-preview",
    "Transcribe and summarize this clip.",
    audio=["interview.mp3"],
)
\`\`\`

Supported on Gemini 2.5+/3.x and OpenAI GPT-4o audio models. Anthropic models raise `RouterError("UNSUPPORTED_MODALITY")`. Supported formats: mp3, wav, flac, ogg, m4a/mp4, webm.

For audio files larger than ~20 MB, use `upload_file()` and pass via `files=[ref]` instead.
```

**`skell_e_router/Skell-E-Router-DOCUMENTATION.md`** — add an "Audio Input" subsection alongside the existing image / files documentation:
- Accepted source forms table
- Supported MIME / format mapping table
- URL rejection rationale + `upload_file()` pointer
- Combined-with-images example
- Per-provider behavior matrix (Gemini = native, OpenAI = audio-capable models only, Anthropic = error)

## Files touched

| File | Change |
|---|---|
| `skell_e_router/utils.py` | New `audio` param on `ask_ai` + overloads; `_encode_audio()` helper; `_AUDIO_MIME_TO_FORMAT` constant; `_construct_messages` accepts `audio` |
| `skell_e_router/gemini_direct.py` | `_convert_messages_to_contents` handles `input_audio` parts |
| `skell_e_router/anthropic_direct.py` | `_convert_messages_for_anthropic` raises on `input_audio` parts |
| `tests/test_utils.py` | New audio tests parallel to image tests |
| `tests/test_gemini_direct.py` | `input_audio` conversion tests |
| `tests/test_anthropic_direct.py` | `input_audio` rejection test |
| `skell_e_router/examples/example_audio_input.py` | New runnable example + small audio fixture |
| `README.md` | New Audio Input section |
| `skell_e_router/Skell-E-Router-DOCUMENTATION.md` | Audio Input subsection in technical reference |

## Out of scope

- Adding new model registry entries (e.g., `gpt-4o-audio-preview`). Handled separately via the `add-llm-model` skill.
- Audio output / TTS.
- Per-model `supports_audio` capability flag. Current image flow has no equivalent flag and relies on provider error fall-through; audio matches that pattern.
- Auto-routing large audio through Gemini Files API. Caller already has `upload_file()` + `files=[]` for that.
- Cost-table updates for audio token pricing. Existing `_PRICING` is text-only-rate; audio inputs may bill slightly off until pricing tables are extended. Not blocking.
