# Migrating direct TypeSafe SDK code to skell-e-router

For developers. This is the target API for moving existing `typesafe_sdk` (Jev) call sites onto the router. Policy per Stan (2026-09-20): every Jev call goes through `skell_e_router.classify()`; `typesafe_sdk` and raw `api.typesafe.ai` calls are banned in all projects. The typesafe-ai and skell-e-router skills already tell future agents this; this doc is for migrating what exists.

## Target API

`classify(model, state, questions, *, config=None, timeout=30) -> ClassificationResponse`, exported from `skell_e_router`. Full reference: `skell_e_router/Skell-E-Router-DOCUMENTATION.md`, section "Classification with Jev". Install in consumer repos with `pip install --upgrade git+https://github.com/stanprokopenko/skell-e-router@main`.

## Mapping from typesafe_sdk

| typesafe_sdk | skell-e-router |
| --- | --- |
| `client = TypeSafeClient()` | none; `classify()` is a plain function |
| `client.system_one(state, questions)` | `classify("jev", state, questions)` |
| `Score(instructions=..., criteria=[...])` | `{"type": "score", "instructions": ..., "criteria": [...]}` |
| `Noul(instructions=...)` | `{"type": "noul", "instructions": ...}` |
| `Choice(instructions=..., criteria={...})` | `{"type": "choice", "instructions": ..., "criteria": {...}}` |
| `resp.answers[k].score` / `.confidence` | `result.answers[k]["score"]` / `["confidence"]` |
| `resp.answers[k].probabilities` (int-like keys) | `result.answers[k]["probabilities"]` with **string keys** (`"0"`, `"1"`, …; choice questions use the criteria labels) |
| `resp.answers[k].noul` | `result.answers[k]["noul"]` |
| `resp.usage.input_tokens` / `.output_tokens` | `result.input_tokens` / `result.output_tokens`; also `result.cost` (USD) and `result.duration_seconds` |
| `except TypeSafeRateLimitError: backoff loop` | drop it; the router retries 429/5xx/529 up to 3 attempts honoring Retry-After. Catch `skell_e_router.RouterError` for what still fails and re-queue the chunk |
| SDK default model | `"jev"` (pins jev-1.13.0); `"jev-latest"` opts into provider upgrades |

Questions are the native TypeSafe JSON objects, so anything the SDK classes serialized to (structured instructions dicts, criteria lists/dicts) carries over verbatim. State is unchanged (string, dict, or list). Batching many questions per request and threading across requests stay caller-side, same as with the SDK.

Credentials: `TYPESAFE_API_KEY` env var or `config={"typesafe_api_key": key}`. On Stan's PC the key lives at Machine scope and is not in a fresh shell's process environment; hydrate it first (see `docs/jev-real/routing-notes.md`).

## Verified

2026-09-20: the first chunk of solar-sailer's `benchmarks/jev-chapter-split-probe/score_sentences.py` (5 target sentences, identical state and score/noul questions) was re-run live through `classify("jev", ...)`. Answers matched the probe's recorded `results.jsonl` values within normal model jitter (scores within ±0.08, nouls within ±0.03), with score, probabilities, confidence, legend, and noul all present. One transient `RouterError: PROVIDER_ERROR` occurred on the very first call and did not reproduce; batch callers should treat a failed request as retryable at their level.

## Known direct-SDK call sites to migrate

- `solar-sailer/benchmarks/jev-chapter-split-probe/score_sentences.py` (and that probe's README usage notes)
- Any other hits for `typesafe_sdk` or `api.typesafe.ai` outside skell-e-router; re-run the search at migration time before starting.

Experiment code that already went through the router (`docs/jev-real/` benchmarks in this repo) needs no migration.
