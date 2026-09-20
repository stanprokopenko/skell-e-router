# Active Tasks & Deferred Follow-ups

## Jev classification comparison

- [x] Added and authenticated Jev support, including all three aliases and Choice/Noul/Score. The 333-call fixture benchmark verified integration and measured speed/cost, but used examples covered by deterministic rules. It does not establish whether Jev can replace an existing AI step. Total task API spend was about $0.0461. No production routing changed. [Evidence and corrected interpretation](jev-classification.md).
- [x] Compared Jev with gpt-5.6-luna on three real AI steps on 2026-09-19: chat routing in skell-e-web (565 labeled messages, Jev 92.4% vs Luna 87.4% vs production 86.9%), support spam triage (120 tickets labeled by the lead, roughly a tie, Jev cheapest and fastest), and rough cut sentence rating in solar-sailer (five episodes, Luna ahead). Runners in scripts/jev_real/, results in docs/jev-real/, Stan-facing write-up in [jev-classification.md](jev-classification.md). Task spend $0.78. No production change.
- [ ] Follow-up: limited live trial of Jev on chat routing. Implemented in skell-e-web behind the `SKELLE_ROUTING_JEV` switch (off, shadow, on) with gpt-5.6-luna deciding below 0.2 confidence; offline replay 515/562 (91.6%). Pending: Stan confirms the Cloud Run secret, then shadow for three days, then on; record the audit numbers here and in skell-e-web docs/TASKS.md when the switch goes on.
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
- [ ] **Re-evaluate kimi-k3 hosting** once the K3 open weights land on DeepInfra (promised ~Jul 27, 2026) — DeepInfra may undercut Moonshot's $3/$15 first-party pricing, but check latency first (DeepInfra ran Kimi-K2.6 at ~3.5 min/answer).
- [ ] **Remove or replace the dead Groq entries** `qwen3-32b` and `kimi-k2-0905` — Groq deprecated both (Jun 17 / Mar 23, 2026); calls fail on free/developer tiers. Removal needs Stan's sign-off. `qwen3.5-397b` (DeepInfra) and `kimi-k3` (Moonshot first-party) are the successors.
- [ ] **Add Gemini 3.5 Pro when it goes GA** — still in limited Vertex preview as of Jul 21, 2026; Google shipped 3.6 Flash instead and teased Gemini 4.
- [x] **Consider DeepSeek first-party API** — closed 2026-09-12 (v3.27.0). DeepSeek retired V4-Flash and repriced; for the new `deepseek-v4.1-flash`, DeepInfra ($0.20/$0.60 per 1M) is cheaper than first-party peak ($0.30/$1.20). No DEEPSEEK_API_KEY needed. Revisit only if DeepInfra lags a future DeepSeek release.
- [ ] **Decide the fate of `deepseek-v4-flash`** — DeepSeek first-party retired the model on 2026-09-10 (requests now route to V4.1-Flash). DeepInfra still serves the original weights, so the alias keeps working; removal needs Stan's sign-off.

## OpenRouter / GLM 5.3 Flash follow-ups (from 2026-08-27 work)

- [ ] **Update glm-5.3-flash pricing when the 50% launch discount ends** — router `pricing` dict (model_config.py) and benchmark config.yaml both carry the discounted $0.075/$0.25 per 1M; undiscounted list is $0.15/$0.50 ($0.03 cached). Check https://openrouter.ai/z-ai/glm-5.3-flash.
- [ ] **Re-check the Z.AI endpoint if GLM answers ever look wrong** — one transient cross-prompt contamination incident during launch-day load, details in benchmark's docs/glm-5.3-flash-benchmark-results.md. Model is pinned to z-ai with fallbacks allowed (extra_body on the model entry).

## Gemini Flash intro pricing

- [ ] **Update gemini-3.7-flash and gemini-3.8-flash pricing on 2027-01-01** — intro $0.75/$3.75 (cache $0.075) doubles to $1.50/$7.50 (cache $0.15). Change `_PRICING` in gemini_direct.py and both entries in benchmark's config.yaml. Source: https://ai.google.dev/gemini-api/docs/pricing.

## Benchmark notes

- MiniMax-M3 (4×) and Kimi-K2.6 (1×) time out on the largest ~30k-token clipping prompts via DeepInfra (see benchmark run 20260722_030307_4e84ca). Re-run those pairs if DeepInfra latency improves.
- Kimi-K3 (Moonshot first-party) also failed 2 of those clipping prompts (`clipper_v30`/`v31`, run 20260723_013149_c03f00) after ~90 min of retries at reasoning high; the ones that succeeded took 20–90 min each. Re-run those two if Moonshot capacity improves, or try reasoning low for that prompt family.

## Dev environment

- [ ] `python -m pytest` on Stan's PC dies at startup because the global `logfire` pydantic plugin imports a missing `opentelemetry.sdk`; run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` until the global env is repaired or the plugin is pinned out in pytest config.
- [ ] Jev rough cut round two. Round one (2026-09-20) built a Jev-only rough cut in scripts/jev_real/roughcut_jev.py: 80.47 sentence points with modules on the 18 ladder episodes, 17th of 24, 3.2 under the best Luna arm and 6.0 under the shipped Opus agentic, about 6 s and $0.05 per episode. Design and status: [spec](superpowers/specs/2026-09-20-jev-roughcut-design.md); results: docs/jev-real/roughcut-jev-heldout.md. Next levers, pending Stan's call: a paragraph or topic-block pass for the critiques (reuse the production Retakes module's paragraph flags), the over-keeping on greco-2.2-thumbnailing (kept ratio 178%), and the retake pass as a standalone retakes-panel feature in solar-sailer. Round-one spend about $2.40.
