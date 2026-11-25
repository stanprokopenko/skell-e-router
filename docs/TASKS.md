# Active Tasks & Deferred Follow-ups

## Video input and dropped file parts, 2026-11-25

- [x] Released 3.54.0: `files=[GeminiFileRef]` reaches Gemini on the direct SDK path (it was dropped with no error), `ask_ai` takes `video=`, unconvertible content parts raise on the direct Gemini and Anthropic paths, list-form system messages are no longer ignored there, and `upload_file()` waits for Gemini to finish processing. Found by Solar Sailer's take-picking prototype (`solar-sailer/docs/take-picking-audio-video-prototype-2026-11-25.md`, "Video path"). Real-call evidence: `.agent-scratch/2026-11-25-video-file-parts/`.
- [ ] Gemini omits `prompt_token_count` when a prompt references a file that is still processing, and the router then reports `prompt_tokens` and `cost` as `None`. Only reachable with `upload_file(wait_until_ready=False)`. Not fixed: deriving the count from the total would be a guess.
- [ ] skell-e-agent's bridge (`skell_e_agent/bridge.py:117`) and skell-e-web's copy (`backend/rag/chat_v2.py:134`) put any pydantic-ai `BinaryContent` into an `image_url` part, video and PDF included. It works on Gemini because the data URI keeps its MIME type. Moving non-image content to `file` parts is the cleaner shape; owned by those repos.

## Jev 1.14 versioned id, 2026-11-21

- [x] Released 3.53.0: `classify()` accepts `jev-1.14.0`. `jev` stays pinned to `jev-1.13.0` and `jev-latest` is unchanged. Unblocks Solar Sailer's Jev pin in Face tracking and Motion Action.
- [ ] Solar Sailer: point `sources.json` at the 3.53.0 commit, regenerate both Python locks (`editor/server/locked-wheels/README.md`), then pin `jev-1.14.0`. Its installer build refuses to start until the locks catch up with this repo's `main`. Owned by a solar-sailer lead.
- [ ] 19 tests fail on `main` under the shared default Python on Stan's PC (LiteLLM 1.100.1, openai 2.54.0), the same set at `1fabf39` and at 3.53.0, so not caused by either release. 18 are OpenAI Responses route tests in `test_responses_storage.py`, `test_reasoning_state.py`, `test_output_limits.py`, `test_audit_fixes.py` and `test_audit_review.py` (15 are `KeyError: 'store'`, 3 reach the network and fail on auth or availability). The other is the root `test_grounding.py`, a live script pytest collects. Cause not investigated; the suite needs its own locked test environment or the tests need to follow current LiteLLM.

## Gemini 4 Neon, 2026-11-18

- [x] Released 3.52.0 with `gemini-4-neon` (GA November 18, paid tier only, $3/$15/$0.15 per 1M flat with no introductory price, 1M context, 65,536 max output, thinking level low/medium/high/xhigh, default high, sampling params ignored by Google). Benchmarked at pinned low, high and xhigh in the benchmark repo: low scores 93.0% (tied for rank 7 of 132), the best Gemini 4 score so far, but costs 3.6x Carbon low and 10x Gemini 3.8 Flash low for a lead inside the noise ([results](../../benchmark/docs/gemini-4-neon-benchmark-results.md)).
- [ ] The shared default Python on Stan's PC still has Solar Sailer's locked router build (3.51.0+solar1), so `gemini-4-neon` is not available there until Solar Sailer moves its lock. The benchmark has its own runtime on 3.52.0.
- [ ] Google's `models.get` now reports a 32,768 output limit for `gemini-4-argon` and `gemini-4-carbon`, where Google's docs and our comments say 65,536. Both still accept `maxOutputTokens` 65536 (checked live 2026-11-18). Recheck if long answers start truncating.

## Gemini 4 Hydrogen, 2026-11-17

- [x] Released 3.50.0 with `gemini-4-hydrogen` (stable, released November 12, $1.50/$7.50/$0.075 per 1M flat, 1M context, 65,536 max output, thinking level low/medium/high/xhigh, default high). The same release makes the Gemini path send `xhigh` instead of turning thinking off. Benchmarked at pinned low, high and xhigh in the benchmark repo: high scores 92.5% (rank 15 of 129), level with Argon high at 71% of the cost, but Gemini 3.8 Flash low still scores higher at an eighth of the cost ([results](../../benchmark/docs/gemini-4-hydrogen-benchmark-results.md)).
- [x] Released 3.50.1: the direct Gemini path now handles finish reason `MALFORMED_RESPONSE` (seen by skell-e-web on a Hydrogen chat turn). It resends up to 3 times, restarts a stream that has delivered nothing, and otherwise raises `PROVIDER_ERROR` with category `malformed_response`. Verified with real google-genai response objects on SDK 1.62 and 2.20; the fault could not be triggered live on demand.
- [x] Released 3.51.0 with the three follow-ups: `MALFORMED_FUNCTION_CALL` gets the same resend-then-error handling as `MALFORMED_RESPONSE`, `AIResponse.cost` includes attempts discarded for a malformed finish reason (listed in `provider_specific_fields["discarded_attempts"]`), and Gemini `xhigh` with `direct_sdk=False` raises a clear `INVALID_PARAM`. skell-e-agent's bridge maps `MALFORMED_FUNCTION_CALL` to `error`; that entry is now unreachable and harmless.

## Gemini 4 Carbon, 2026-11-10

- [x] Released 3.49.0 with `gemini-4-carbon` (stable, paid tier only, same API surface and $2/$10/$0.10 launch pricing as `gemini-4-argon`, doubling December 1). Benchmarked at pinned low and high in the benchmark repo: low scores 92.5% (rank 15 of 126), just under Argon low, and high costs 16x more for a lower score ([results](../../benchmark/docs/gemini-4-carbon-benchmark-results.md)).

## Claude Fable 5.2, 2026-11-09

- [x] Released 3.48.0 with `claude-fable-5-2` ($8/$40 per 1M, $0.20 cache reads, same API surface as Fable 5.1). Benchmarked at pinned low and high in the benchmark repo: low matches Fable 5.1's best score at about half the cost and time ([results](../../benchmark/docs/claude-fable-5-2-benchmark-results.md)). Consumers moved off Fable 5.1 the same day: houston crons, skell-e-web, solar-sailer and the orchestrator's notes ([scan and resolution](fable-5-1-references-2026-11-09.md)).

## Provider integration audit, 2026-11-06

- [x] Reviewed the reasoning-state channel and released 3.45.1. Capture every raw OpenAI reasoning item before chat normalization, including collected streams; retain cutoff/refusal metadata and own-provider filtering. Review and live evidence: skell-e-agent `docs/reasoning-state-review-2026-11-06.md`.

- [x] Released 3.45.0 with the reasoning-state channel: `AIResponse.reasoning_state` returns Claude signed thinking blocks and OpenAI encrypted reasoning items and replays them to their own provider only; Responses requests ask for `reasoning.encrypted_content`. Closes R3 (replay half) and A2. Tests: `tests/test_reasoning_state.py`. Live evidence: skell-e-agent `docs/reasoning-state-live-verification-2026-11-06.md`.

- [x] Reviewed 3.44.3 and fixed the missing storage default on current LiteLLM's reasoning-summary route in 3.44.4. Agent contracts now check storage by imported router version. Both router dependency sets and all three agent compatibility runs pass. [Review and evidence](responses-storage-review-2026-11-06.md).

- [x] Reviewed 3.44.0 and corrected alternative-candidate merging, Responses text loss, LiteLLM cache-write reporting, and Gemini prompt-block retries in 3.44.1. Version 3.44.2 also declares the MCP extra required by current LiteLLM, preserves its native refusal field, keeps Gemini embedding groups intact, and verifies fresh dependency installs. Regression tests: `tests/test_audit_review.py` and `tests/test_audit_review_gemini.py`. Cross-repo review and verification: skell-e-agent `docs/provider-audit-review-2026-11-06.md`.

- [x] Released 3.44.0 with the audit's router fixes: Gemini direct tool history (function calls, grouped function responses, thought signatures), multi-choice merge on the OpenAI Responses bridge, Retry-After on the LiteLLM path, 529 retries, 1h cache-write and Haiku 5.5 tier pricing, cache token fields on `AIResponse`, gemini-2.5-flash thinking budget, forced tool choice with budget thinking. Tests in `tests/test_audit_fixes.py`. Findings, evidence and the new-provider checklist: skell-e-agent `docs/provider-integration-audit-2026-11.md`.
- [ ] Open router findings from the audit, by cost: E2 stacked SDK and router retries (up to 9 attempts); E5 out-of-credit needs a never-retried `quota` category; E7 context-length category; P1 effort `max` dropped by LiteLLM for gpt-6/6.1; P2 xhigh workaround keeps gpt-5.5/5.6 tools on chat/completions; R5 reasoning-exhausted reply retried as a fake 500; R10 gpt-6 long-context surcharge; E13 Gemini intro prices expire 2026-12-01 and 2027-01-01; R2 assistant text dropped beside tool calls on replay; A5 cap cache breakpoints at four; A6 empty user content; C6 carry the provider's error type in `RouterError.details`; E4, E6, E14, P6, P7, P11, G8 one-liners. Sizes in the audit's tables.

## Tool-message cache markers

- [x] Reviewed v3.43.1 and fixed caller-history mutation on the LiteLLM route in v3.43.2. Two blind review rounds completed, the final review was clean, 1,164 tests and 17 subtests passed, and live bridge and tool-result checks passed. [Review and evidence](cache-marker-review-2026-11-06.md).

## Literal text embeddings

- [x] Added `get_embedding(..., literal_text=True)` in 3.38.1 for untrusted search and transcript text. Gemini and OpenAI preserve exact strings and batch size without input filesystem probes or URL/media interpretation. Existing image uploads keep their default path. Consumer migration requires adding this keyword; the Solar Sailer moment-search review lead owns that change. [Supported API](../skell_e_router/Skell-E-Router-DOCUMENTATION.md#literal-text-for-search-queries-and-transcripts).
- [x] Verified 34 HTTP-boundary and response regressions, plus the full suite of 1,075 tests and 17 subtests. Live Gemini and OpenAI checks each returned eight 768-dimensional vectors for eight inputs; ordinary text matched the existing transport within 4.36e-9 for Gemini and exactly for OpenAI. Live Gemini image aggregation passed. Independent security review found no blocking issues. Google may omit usage, and literal Gemini cost estimates can be unavailable; total probe spend is estimated below $0.01.

## Jev classification comparison

- [x] Added base64 image assets to `classify()` with preview-header handling and offline contract tests. Image cost remains unknown until verified with TypeSafe. [First-run checkpoint](jev-images/2026-11-04-first-run.md).
- [x] Completed the Jev image and Solar Sailer import evaluations for about $0.14, reviewed item by item by Stan. With the reworded hand question Jev 1.13 scores 51/52 against Gemini 3.1 Flash Lite's 48/52, and 7/7 against 6/7 on the import benchmark, at about a tenth of the latency. [Scored results](jev-images/2026-11-04-results.md).
- [x] Tested Jev 1.14's box question on pencil-tip position against Stan's 23 labeled tip points (router 3.43.0 added box support). Median miss 121 source px against a 452 px baseline, mark inside the box 43% of the time, confidence honest; a region finder, not a contact-point detector. [Scored results](jev-images/2026-11-04-results.md#pencil-tip-box-added-later-on-2026-11-04).
- [x] Stan declined a fresh-frame confirmation of the reworded hand question on 2026-11-04; its 98% stands as measured on the 52 frames the wording was tuned against, and he confirmed the S-12 "first marks" label.
- [ ] Handed off: Jev tool-tip tracker prototype (hand-holding-tool gate, tip box, crop-and-refine, smoothing, overlay video, speed and cost ledger) on one minute of the stroke-detection test take. Brief in [handoffs/2026-11-04-jev-tool-tip-tracker.md](handoffs/2026-11-04-jev-tool-tip-tracker.md); lives in solar-sailer `benchmarks/jev-tool-tip-tracker/`.
- [ ] Confirm TypeSafe's image billing rate with TypeSafe; all Jev image costs are the text rate applied to reported tokens. Jev rejects images under 512 px wide, so a Solar Sailer adoption would raise its 384 px import frames.

- [x] Added and authenticated Jev support, including all three aliases and Choice/Noul/Score. The 333-call fixture benchmark verified integration and measured speed/cost, but used examples covered by deterministic rules. It does not establish whether Jev can replace an existing AI step. Total task API spend was about $0.0461. No production routing changed. [Evidence and corrected interpretation](jev-classification.md).
- [x] Compared Jev with gpt-5.6-luna on three real AI steps on 2026-09-19: chat routing in skell-e-web (565 labeled messages, Jev 92.4% vs Luna 87.4% vs production 86.9%), support spam triage (120 tickets labeled by the lead, roughly a tie, Jev cheapest and fastest), and rough cut sentence rating in solar-sailer (five episodes, Luna ahead). Runners in scripts/jev_real/, results in docs/jev-real/, Stan-facing write-up in [jev-classification.md](jev-classification.md). Task spend $0.78. No production change.
- [ ] Follow-up: limited live trial of Jev on chat routing. Implemented in skell-e-web behind the `SKELLE_ROUTING_JEV` switch (off, shadow, on) with gpt-5.6-luna deciding below 0.2 confidence; offline replay 515/562 (91.6%). Live since 2026-09-21 (shadow skipped on Stan's call); the secret is bound and live probes routed correctly. Pending: the seven-day audit, tracked in skell-e-web docs/TASKS.md; record the numbers here when it lands.
- [ ] Follow-up: add the contact-form bot pattern (fake name, unrelated email, one Latin word) as a spam example in the Jev spam criteria and rerun the 120-ticket set; optionally have a human pass over docs/jev-real/spam-labels.json.
- [ ] Follow-up (handed off): design a Jev-based rough cut with Stan. Brief in docs/handoffs/2026-09-19-jev-roughcut-brainstorm.md; orchestrator asked to dispatch a Fable 5.1 lead.
- [ ] Follow-up (skell-e-web, reported to orchestrator): backend/benchmarks/routing/routing-labels.jsonl still holds one customer email address in a text_head; redact it there.
- [x] Made the router the only sanctioned Jev path on 2026-09-20: verified `classify()` live against the solar-sailer probe's scoring (answers matched recorded results within model jitter), rewrote the typesafe-ai and skell-e-router skills to ban direct `typesafe_sdk` use, and synced skills to all agent accounts. Migration mapping in [jev-router-migration-target.md](jev-router-migration-target.md).
- [ ] Follow-up (for a migration lead): move remaining direct `typesafe_sdk` call sites (solar-sailer `benchmarks/jev-chapter-split-probe/`) onto `classify()` per [jev-router-migration-target.md](jev-router-migration-target.md).

## Provider error credential disclosure

- [x] Fixed v3.26.2 provider error and traceback disclosure in v3.26.3 source `63b5fd22bacef9100b09cdee355bd8839439be78`. All 781 tests pass on source and wheel, the actual Houston helper passes offline, and fresh independent Astra review is clean in round 2 of the three-round cap. [Reproduction, validation and release record](credential-error-security.md).
- Router batch and shared-Python rollout permanently halted by Stan on 2026-09-05. The source fix is complete; no installation, phase A or activation will occur under this assignment. [Rollout preparation](credential-error-rollout.md), [SDK baselines](credential-error-sdk-baseline.json), exact recovery artifacts and [isolated-copy reconciliation](isolated-router-credential-exposure.md) are preserved as historical evidence. These are closed preparation records, not active upgrade tasks.

## Houston TLDR output limit

- [x] Released and installed v3.26.2 on 2026-09-04. The actual helper sends `max_completion_tokens=600` to OpenAI through the installed router. All 727 tests pass and the blind review was clean. [Contract and integration evidence](houston-output-limit.md). The Houston lead owns helper response validation and app shipping.

## Model registry follow-ups (from 2026-07-21 model-gap work)

- [x] **Swap `kimi-k2.6` → Kimi K3** — done 2026-07-22 (v3.16.0): Moonshot's K3 API launched early (Jul 16), so `kimi-k3` now routes first-party via `api.moonshot.ai` (MOONSHOT_API_KEY, $3/$15 per 1M) and the `kimi-k2.6` DeepInfra stand-in was removed.
- [x] **Add Kimi K3.1 Pro** — done 2026-10-26 (v3.41.0): `kimi-k3.1-pro` ($2/$10) and `kimi-k3.1-pro-highspeed` ($4/$20) route first-party like `kimi-k3`. Benchmarked in benchmark's `docs/kimi-k3-1-pro-benchmark-results.md`.
- [ ] **Re-evaluate Kimi hosting** once K3.1 Pro open weights land on third-party hosts (Moonshot promised by Nov 4, 2026). A host may undercut Moonshot's first-party pricing, but check latency first (DeepInfra ran Kimi-K2.6 at ~3.5 min/answer).
- [x] **Remove or replace the dead Groq entries** — done 2026-09-25 (v3.33.0) as part of the retired-model sweep; `groq-compound`, `groq-compound-mini`, `qwen3-32b` and `kimi-k2-0905` are gone and the Groq-only request code with them. Audit in [model-deprecation-audit-2026-09-25.md](model-deprecation-audit-2026-09-25.md).
- [ ] **Add Gemini 3.5 Pro when it goes GA** — still in limited Vertex preview as of Jul 21, 2026; Google shipped 3.6 Flash instead and teased Gemini 4.
- [x] **Consider DeepSeek first-party API** — closed 2026-09-12 (v3.27.0). DeepSeek retired V4-Flash and repriced; for the new `deepseek-v4.1-flash`, DeepInfra ($0.20/$0.60 per 1M) is cheaper than first-party peak ($0.30/$1.20). No DEEPSEEK_API_KEY needed. Revisit only if DeepInfra lags a future DeepSeek release.
- [ ] **Decide the fate of `deepseek-v4-flash`** — DeepSeek first-party retired the model on 2026-09-10 (requests now route to V4.1-Flash). DeepInfra still serves the original weights, so the alias keeps working; removal needs Stan's sign-off. Re-checked 2026-09-25: DeepInfra still lists it as active, so it stayed in the retired-model sweep.

## Retired-model sweep follow-ups (from 2026-09-25 audit)

- [x] **Delete the ten deprecated aliases** — done 2026-09-25 (v3.34.0) on Stan's call; the six xAI grok ids and four Nemotron ids are removed outright, and skell-e-web is being repointed in parallel (super-fast default becomes gpt-6-luna).
- [ ] **Before 2026-10-23: drop `o1`** — OpenAI shuts it down that day; replacement gpt-5.6-sol. skell-e-web's picker and skell-e-scripter's model list still offer it.
- [ ] **Before 2026-12-11: drop `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `o3`** — OpenAI shutdown date; replacements gpt-5.6-sol / terra / luna / sol. `gpt-5` is the default model in most router tests (tests/helpers.py and test_utils.py), so swap the test default first. skell-e-web, benchmark config.yaml and skell-e-web's filter_tickets_llm.py use them.
- [ ] **Before 2027-05-07: drop `gemini-3.1-flash-lite`** — Google shutdown date; replacement gemini-3.5-flash-lite. skell-e-web's RAG subagent and routing-classifier fallback, solar-sailer's editor config and beverly-bica's OCR tool all pin it.
- [ ] **Watch `gemini-3-flash-preview`** — no date yet, but Google already names gemini-3.6-flash as its replacement.
- [x] **Dropped `claude-sonnet-4-5-20250929` on 2026-10-01 (v3.37.0)** after Anthropic announced its retirement for 2026-11-24 (availability may degrade from 2026-10-30). Replacement `claude-sonnet-5`. Benchmark `config.yaml`, skell-e-web (`backend/constants.py`, `proko-app/src/app/model-options.constants.ts`, `backend/benchmarks/looping_bench.py`; the old id now maps to `claude-sonnet-5` in `RETIRED_MODEL_FALLBACKS`) and skell-e-scripter (`backend/services/ai_service.py`) were repointed the same day. [Plan](plans/2026-10-01-sonnet-4-5-retirement.md).
- [ ] **Watch `claude-haiku-4-5` and `claude-opus-4-5`** — earliest possible retirement dates are claude-haiku-4-5 (2026-10-15) and claude-opus-4-5 (2026-11-24). Nothing announced; Anthropic gives 60 days' notice.

## OpenRouter / GLM 5.3 Flash follow-ups (from 2026-08-27 work)

- [ ] **Update glm-5.3-flash pricing when the 50% launch discount ends** — router `pricing` dict (model_config.py) and benchmark config.yaml both carry the discounted $0.075/$0.25 per 1M; undiscounted list is $0.15/$0.50 ($0.03 cached). Check https://openrouter.ai/z-ai/glm-5.3-flash.
- [ ] **Re-check the Z.AI endpoint if GLM answers ever look wrong** — one transient cross-prompt contamination incident during launch-day load, details in benchmark's docs/glm-5.3-flash-benchmark-results.md. Model is pinned to z-ai with fallbacks allowed (extra_body on the model entry).

## Gemini Flash intro pricing

- [ ] **Update gemini-3.7-flash and gemini-3.8-flash pricing on 2027-01-01** — intro $0.75/$3.75 (cache $0.075) doubles to $1.50/$7.50 (cache $0.15). Change `_PRICING` in gemini_direct.py and both entries in benchmark's config.yaml. Source: https://ai.google.dev/gemini-api/docs/pricing.

## Benchmark notes

- MiniMax-M3 (4×) and Kimi-K2.6 (1×) time out on the largest ~30k-token clipping prompts via DeepInfra (see benchmark run 20260722_030307_4e84ca). Re-run those pairs if DeepInfra latency improves.
- Kimi-K3 (Moonshot first-party) also failed 2 of those clipping prompts (`clipper_v30`/`v31`, run 20260723_013149_c03f00) after ~90 min of retries at reasoning high; the ones that succeeded took 20–90 min each. Re-run those two if Moonshot capacity improves, or try reasoning low for that prompt family.

## Dev environment

- [x] Derive `skell_e_router.__version__` from package metadata instead of a separate literal. Done in 3.31.1: `__version__` now comes from `importlib.metadata` (fallback `0.0.0+uninstalled` for uninstalled checkouts), guarded by tests/test_version.py; probe/verify scripts accept the missing literal. See [consumer verification](router-consumers-2026-09-22.md).
- [ ] `python -m pytest` on Stan's PC dies at startup because the global `logfire` pydantic plugin imports a missing `opentelemetry.sdk`; run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` until the global env is repaired or the plugin is pinned out in pytest config.
- [ ] Jev rough cut round two, handoff at [docs/handoffs/2026-09-20-jev-roughcut-round-two.md](handoffs/2026-09-20-jev-roughcut-round-two.md); retakes panel wiring and the benchmark page arm handed to a solar-sailer lead via [docs/handoffs/2026-09-20-retakes-panel-jev-pick.md](handoffs/2026-09-20-retakes-panel-jev-pick.md). Round one (2026-09-20) built a Jev-only rough cut in scripts/jev_real/roughcut_jev.py: 80.47 sentence points with modules on the 18 ladder episodes, 17th of 24, 3.2 under the best Luna arm and 6.0 under the shipped Opus agentic, about 6 s and $0.05 per episode. Design and status: [spec](superpowers/specs/2026-09-20-jev-roughcut-design.md); results: docs/jev-real/roughcut-jev-heldout.md. Next levers, pending Stan's call: a paragraph or topic-block pass for the critiques (reuse the production Retakes module's paragraph flags), the over-keeping on greco-2.2-thumbnailing (kept ratio 178%), and the retake pass as a standalone retakes-panel feature in solar-sailer. Round-one spend about $2.40.
