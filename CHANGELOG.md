# Changelog

Earlier releases are described in the version notes in [README.md](README.md).

## 3.45.0 - 2026-11-06

Reasoning-state channel (audit findings R3 and A2). `AIResponse.reasoning_state` carries the provider-opaque reasoning a model expects back on its next turn, in the order it was returned: Claude's signed `thinking` and `redacted_thinking` blocks from the direct Anthropic path, and OpenAI's `reasoning` items (id, `encrypted_content`, summary) from LiteLLM's Responses bridge. Put the list back on the assistant message as `"reasoning_state"` and each converter replays only its own provider's entries: the Anthropic converter emits the blocks first in the assistant content, before text and `tool_use`; the LiteLLM path turns OpenAI entries into LiteLLM `reasoning_items`, which its bridge places right before the turn's `function_call` items, and strips the key on plain chat/completions and for other providers. Every OpenAI request that reaches `/v1/responses` now also asks for `include: ["reasoning.encrypted_content"]` next to `store: false`, since that blob is the only way to resume reasoning without stored state. Gemini's per-call `thought_signature` stays on `tool_calls` as before. Live: Anthropic and OpenAI both accept the replay (200); Opus 4.6 stops re-thinking after each tool result, Haiku 4.5 is unchanged because that model strips prior thinking itself. Tests: `tests/test_reasoning_state.py`. Transcripts that keep the state grow by a few KB per step.

## 3.44.4 - 2026-11-06

Apply the `store: false` default when current LiteLLM routes a reasoning-summary request to OpenAI Responses without tools. Route detection now forwards both summary aliases to LiteLLM's own check when supported, while retaining compatibility with older versions. Explicit caller storage settings still win.

## 3.44.3 - 2026-11-06

Send `store: false` on every OpenAI request that reaches the Responses API: the GPT-6 aliases, and gpt-5.4+ calls with tools plus a reasoning effort, which LiteLLM bridges there on its own. OpenAI's Responses API defaults to `store: true` and keeps each request and response for 30 days, and the router never reuses them through `previous_response_id`, so nothing was gained by storing them (audit finding R3). LiteLLM 1.83 drops a top-level `store` before building the request, so the router sends it through `extra_body`. A caller's own `store=True`, top-level or in `extra_body`, still wins. Chat Completions requests and other providers are unchanged.

## 3.44.2 - 2026-11-06

Install LiteLLM's MCP extra. LiteLLM 1.108.2 imports its MCP catalog during ordinary completion calls, so the base package alone fails before reaching the provider. Preserve native refusal text from newer LiteLLM and keep caller-requested Gemini image/caption embeddings grouped when its newer adapter expects nested input. Fresh-install tests cover both SDK generations, including valid field aliases and Responses text joining.

## 3.44.1 - 2026-11-06

Review fixes for 3.44.0:

- Keep alternative completion candidates separate, so tool calls from different answers cannot execute together. Merge all text and tool calls only for LiteLLM's Responses API output.
- Forward Anthropic cache-write token counts on the LiteLLM route as well as the direct route.
- Preserve Gemini prompt-level safety blocks when there are no candidates, so the agent stops instead of retrying the blocked request.

## 3.44.0

Fixes from the 2026-11-06 provider-integration audit.

- G1, G10: direct Gemini sends assistant `tool_calls` as `functionCall` parts with their ids, and consecutive tool messages as one user turn of `functionResponse` parts (`{"output": text}`, matching id and name). Empty text is never sent as a part, and an all-empty conversation raises `INVALID_PARAM`.
- G2: direct Gemini keeps Gemini's own `function_call.id` and mints `call_<hex>` only when it is missing.
- G3: direct Gemini returns each function call's `thoughtSignature` as base64 `tool_calls[i]["thought_signature"]` and replays it as bytes on the same `functionCall` part. A model turn whose first call has none gets `skip_thought_signature_validator`, so Gemini 3 tool loops no longer 400 on request 2. The Anthropic and LiteLLM paths ignore the key.
- G4 (router half): direct Gemini `finish_reason` is the bare value (`"MAX_TOKENS"`, `"SAFETY"`), not `"FinishReason.MAX_TOKENS"`.
- G9 / P5: `gemini-2.5-flash` maps `reasoning_effort` to `thinking_budget`. A Gemini model that declares neither `reasoning_effort` nor `budget_tokens` never gets `thinking_level`.
- R1: on the LiteLLM path, tool calls in a later `choices` entry (LiteLLM's Responses API bridge splits text and calls) are merged into the response instead of dropped.
- E1: Retry-After is read from `httpx.Headers`, so the LiteLLM path honors it and skips retries when it exceeds 120 seconds.
- E3: Anthropic 529 `OverloadedError` retries like a 500.
- E10: `AIResponse.cache_read_tokens` and `AIResponse.cache_write_tokens` are filled on the Anthropic, Gemini and LiteLLM paths (write is Anthropic only).
- E11: Anthropic 1-hour cache writes cost 2x input and 5-minute writes 1.25x, from `usage.cache_creation`. Without that breakdown all writes cost 1.25x.
- E12: `claude-haiku-5-5` costs $0.50/$2.50 per million tokens for the whole request when the prompt exceeds 100k tokens.
- A3 / P8: Anthropic budget-thinking requests with a forced `tool_choice` fall back to `auto` with a logged warning instead of returning HTTP 400.
