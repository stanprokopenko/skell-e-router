# Retired-model sweep, September 25, 2026

Developer record of the audit Stan asked for: "we need to make sure we remove all the deprecated models from our Skell-E router." Shipped as skell-e-router 3.33.0, then 3.34.0 the same day (see the update at the end).

## Method

Two passes, then a live check. First an inventory of every model id in the registry (`MODEL_CONFIG`, `EMBEDDING_MODEL_CONFIG`, `IMAGE_CONFIG`, `CLASSIFICATION_MODEL_CONFIG`), the Anthropic and Gemini price tables, tests, README and the built-in documentation file. Second, each id checked against the provider's own deprecation page on 2026-09-25: Anthropic, OpenAI, Google Gemini API, xAI, Groq, DeepInfra's public model list, DeepSeek's news page, Meta, Moonshot and OpenRouter. Then one tiny live call per suspect id through the router (script and raw results in `.agent-scratch/2026-09-25-deprecation/`, gitignored). Total probe spend was under $0.40, most of it two test image generations.

The live check matched the research in every case. It also settled the one id no provider page mentioned, `grok-4-0220`, which xAI rejects.

## Removed (provider returns an error)

| Alias | Provider | Retired | Provider's replacement |
|---|---|---|---|
| claude-opus-4-1-20250805 | Anthropic | 2026-08-05 | claude-opus-4-8 |
| claude-sonnet-4-20250514 | Anthropic | 2026-06-15 | claude-sonnet-4-6 |
| claude-3-7-sonnet-20250219 | Anthropic | 2026-02-19 | claude-sonnet-4-6 |
| claude-3-5-sonnet-20241022 | Anthropic | 2025-10-28 | claude-sonnet-4-6 |
| gpt-5.3-chat (`gpt-5.3-chat-latest`) | OpenAI | 2026-08-10 | gpt-5.6-sol |
| grok-4-0220 | xAI | unlisted, API rejects it | grok-4.20 |
| groq-compound | Groq | 2026-09-21 | none named |
| groq-compound-mini | Groq | 2026-09-21 | none named |
| qwen3-32b | Groq | 2026-07-17 | gpt-oss-120b |
| kimi-k2-0905 | Groq | 2026-04-15 | gpt-oss-120b |

The four Claude rows also left `_PRICING` in `anthropic_direct.py`. The Groq-only request code in `utils.py` (the Qwen3 `reasoning_effort` remap and the Compound header and tool injection) had no remaining caller and went with them, along with its tests. `gpt-oss-120b` and `gpt-oss-20b` still route through Groq. Review found they were registered with `provider="openai"`, which made the router demand `OPENAI_API_KEY` for a Groq call and ignore a `groq_api_key` passed in config; both now carry `provider="groq"`.

## Kept as deprecated aliases (provider silently serves a different model)

These ten ids no longer reach the model they name, but the provider answers anyway with a successor and bills at the successor's rate. Removing them would break skell-e-web on its next router update (see below), so they stay registered, listed in `DEPRECATED_MODELS` in `model_config.py`, and `resolve_model_alias()` logs one warning per process naming the replacement. Delete the alias, its `DEPRECATED_MODELS` line and the tests together once skell-e-web repoints.

| Alias | What actually answers | Since | Use instead |
|---|---|---|---|
| grok-4-1-fast-reasoning | grok-4.3 | 2026-05-15 | grok-4.20 |
| grok-4-1-fast-non-reasoning | grok-4.3 | 2026-05-15 | grok-4.20-non-reasoning |
| grok-4-0709 | grok-4.3 | 2026-05-15 | grok-4.20 |
| grok-4-fast-reasoning | grok-4.3 | 2026-05-15 | grok-4.20 |
| grok-4-fast-non-reasoning | grok-4.3 | 2026-05-15 | grok-4.20-non-reasoning |
| grok-code-fast-1 | grok-build-0.1 | 2026-05-15 | grok-4.20 |
| nemotron-super-49b | Nemotron 3 Ultra | 2026-07-17 | nemotron-3-ultra |
| nemotron-70b | Nemotron 3 Ultra | 2026-07-16 | nemotron-3-ultra |
| nemotron-nano-12b-vl | Nemotron 3 Ultra | 2026-07-16 | nemotron-3-ultra |
| nemotron-nano-9b | Nemotron 3 Nano 30B | 2026-06-11 | nemotron-3-nano-30b |

The four Nemotron aliases now point at the successor's registry entry, so the router's config and pricing match what DeepInfra runs. The six xAI entries keep their own config because grok-4.3 is not a registered model; adding one is a separate job.

## Repointed

`nano-banana-3` (and its aliases `gemini-3-pro-image`, `nano-banana-pro`) now sends `gemini-3-pro-image` instead of `gemini-3-pro-image-preview`. Google's deprecation page lists the preview id as retired on 2026-06-25 and names the GA id as its replacement. Both ids still generated an image on 2026-09-25, so this is future-proofing, not a fix.

`gemini-3-pro-preview` was already an alias of `gemini-3.1-pro-preview` and stays that way.

## Kept, with a shutdown date

| Alias | Provider | Shutdown | Replacement |
|---|---|---|---|
| o1 | OpenAI | 2026-10-23 | gpt-5.6-sol |
| gpt-5, gpt-5-mini, gpt-5-nano, o3 | OpenAI | 2026-12-11 | gpt-5.6-sol / terra / luna / sol |
| gemini-3.1-flash-lite | Google | 2027-05-07 | gemini-3.5-flash-lite |

Each has a dated entry in `docs/TASKS.md`. `gpt-5` is the default model in most router tests, so that swap has to land before December. No date yet for `gemini-3-flash-preview`, but Google already names `gemini-3.6-flash` as its replacement. Anthropic's earliest possible retirement dates for the 4.5 family are Sonnet 4.5 on 2026-09-29, Haiku 4.5 on 2026-10-15 and Opus 4.5 on 2026-11-24; nothing is announced and Anthropic gives 60 days' notice.

`deepseek-v4-flash` stays. DeepSeek's own API retired it, but DeepInfra, which the router uses, still serves the original weights.

## Downstream pins of removed or deprecated ids

Searched every repo under `Documents\GitHub` and `Documents\GitLab` except this one, skipping vendored and generated folders. Only live code and config are listed; docs, logs and benchmark result files are counted at the end. None of these were edited.

**skell-e-web, the one live product that will notice.**

| File | Ids | Effect after upgrading the router |
|---|---|---|
| backend/constants.py:74 `SUPER_FAST_MODEL` | grok-4-1-fast-non-reasoning | Still works, logs a deprecation warning. Repoint to grok-4.20-non-reasoning. |
| proko-app/src/app/recommended-settings.constants.ts:32 | groq-compound (default for download_course_page) | Breaks. Already broken upstream since 2026-09-21. Needs a new default. |
| proko-app/src/app/model-options.constants.ts | gpt-5.3-chat, grok-4-0220, groq-compound, groq-compound-mini, the six grok ids, the four nemotron ids | The first four options error on pick (three of them already did). The rest work with a warning. |
| backend/constants.py:94-111 `VISION_MODELS` | gpt-5.3-chat, grok-4-0220, the grok ids | A list of names; harmless on its own. |
| backend/benchmarks/research_agent_bench.py:105-112 | kimi-k2-0905, grok-4-1-fast-reasoning | Benchmark script, not a user path. |
| backend/tests/test_chat_attachments.py:150-152 | qwen3-32b, kimi-k2-0905, grok-code-fast-1, grok-4-0709 | Test data. |

**benchmark.** `config.yaml` lists gpt-5.3-chat (170-172), grok-code-fast-1 (649-651), grok-4-1-fast-* (658-669), grok-4-0709 (676-678), grok-4-fast-* (685-696) and the four nemotrons (888-926). The next full run fails on gpt-5.3-chat and quietly benchmarks the wrong model for the rest. The retired Claude ids are already gone from it.

**solar-sailer-sync-boundary-20260916 and -followup-20260916.** Three roughcut experiment configs under `benchmarks/roughcut/experiments/` pin grok-4-0709. Worktree copies on the `qualify/sync-followup` branch.

**skell-e-scripter-drafts and skell-e-scripter `.router-candidates`.** A fixed model list in `backend/routers/search.py:155` offers claude-sonnet-4-20250514. Draft and snapshot copies, not the live scripter.

**Not affected by this package, still naming retired models.** knowledge-base has its own copy of a router with all the retired ids registered and groq-compound as the default for its course-page endpoint. image-spam-checker has them in database migrations. skell-e, e-mitry, solar-sailer and misc use `claude-3-5-sonnet-20240620` against the Anthropic SDK directly.

**Pins of the future-dated ids** (o1, o3, gpt-5 family, gemini-3.1-flash-lite) are broader: skell-e-web's picker, RAG subagent and routing-classifier fallback, skell-e-scripter's model list, solar-sailer's editor config, beverly-bica's OCR tool, user-spam-algorithm's classifier, benchmark config.yaml and knowledge-base settings. They are in `docs/TASKS.md` with their dates.

Docs, logs and result-file mentions, not listed above: benchmark about 780, claude-orchestrator about 235, skell-e 155 (all the 2024 Sonnet in logs), knowledge-base 46, visual-language-zettelkasten 37, solar-sailer and its sync copies about 70, skell-e-scripter 12.

## Verification

- `python -m pytest tests`: 1033 passed, 17 subtests passed. Root `test_grounding.py` is a live-network test and was not part of this run.
- Live probe of all 21 suspect ids plus two image generations, results in the scratch folder.
- One blind review round on the diff before commit (Fable). It caught the gpt-oss provider tag, a placeholder beta header in the docs examples, and that full-name lookups like `xai/grok-4-0709` bypassed the warning. All three fixed.

## Update, later on 2026-09-25

Stan asked to remove the ten deprecated aliases outright from both the router and skell-e-web, and to make skell-e-web's super-fast default `gpt-6-luna`. Router 3.34.0 deletes the six xAI entries, the four Nemotron aliases and the whole `DEPRECATED_MODELS` mechanism (the dict, the full-name mirror and the warning in `resolve_model_alias()`), so all twenty ids in the two tables above now raise `INVALID_MODEL`. The skell-e-web side is being done by a separate lead in that repo, which already carries its own retired-id fallback map for saved settings; the `gpt-6-luna` instruction was passed to that lead through the orchestrator.
