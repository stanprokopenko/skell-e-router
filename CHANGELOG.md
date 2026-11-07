# Changelog

Earlier releases are described in the version notes in [README.md](README.md).

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
