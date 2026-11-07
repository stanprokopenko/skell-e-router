# Responses storage review

For developers. The next paragraph is the summary for Stan.

I found one request type that still used OpenAI's default response storage. Router 3.44.4 fixes it. I also updated the agent's tests to check the new storage behavior while continuing to support older routers. No application setting changes are required.

## Scope

Reviewed router commit `86fb33e`, version 3.44.3, on top of the previous provider-integration review. The agent baseline was `b99dec8`, version 0.8.12. Work took place in separate worktrees in both repositories. The reasoning-state channel remains with its existing lead; this review covers storage defaults and their contract tests.

## Confirmed findings and fixes

| Severity | Finding | Fix |
|---|---|---|
| High | LiteLLM 1.108.2 routes a GPT-5 reasoning-summary request without tools to Responses. The router's route check omitted the summary, so it sent no storage setting. Even top-level `store=False` was lost on this path. | Pass summary aliases through LiteLLM's own summary extractor and route check when that version supports the argument. Older LiteLLM keeps its existing call signature. |
| Medium | The agent's exact Responses body assertions still expected the pre-3.44.3 body. This broke 30 tests with older dependencies and 34 with current dependencies. | Require `store: false` for router 3.44.3 and newer. Read the imported checkout's version when testing source, or installed package metadata otherwise. Keep exact body assertions and test explicit true/false overrides. |

The runtime fix changes no public signature. Caller storage overrides and caller-owned dictionaries remain intact. No new setting, rejection, or user step was added. Router version is 3.44.4; agent runtime code and version are unchanged.

The reproduction used `ask_ai("gpt-5.6-sol", "hi", reasoning_effort="low", extra_body={"reasoningSummary": "auto"})` with dummy credentials and a mocked HTTP transport. Both `reasoningSummary` and `reasoning_summary` reached `/v1/responses` without `store` before the fix. The regression tests require the corrected body.

## Verification

Two blind review rounds used gpt-6-astra. The first round had two independent reviewers and identified the missing summary route. A fresh reviewer checked the completed router and agent changes in round two and found no actionable issues. Its focused rerun passed 27 tests with current LiteLLM and 11 with older LiteLLM, with 16 unsupported-route skips. The loop stopped after that clean round.

All tests ran offline with synthetic provider responses and existing network guards. No provider account or shared Python installation was changed.

| Suite | Passed | Skipped | Expected failures |
|---|---:|---:|---:|
| Router, installed dependencies | 1242 plus 17 subtests | 16 | 0 |
| Router, current dependencies | 1258 plus 17 subtests | 0 | 0 |
| Agent, installed router 3.43.2+solar1 | 994 | 1 | 59 |
| Agent, fixed router source and installed dependencies | 1025 | 1 | 28 |
| Agent, fixed router source and current dependencies | 1031 | 1 | 22 |

The 16 router skips cover summary routing absent from older LiteLLM. Agent expected failures retain known provider and old-router gaps. The two added old-router expected failures demonstrate that its top-level storage setting is dropped. The skipped agent module requires its existing dedicated offline launcher and is unrelated to storage.

The new router regression file covers 19 cases on current LiteLLM: both summary aliases, default storage, explicit top-level and extra-body overrides, caller dictionary preservation, and streaming requests. Current LiteLLM consumes Responses SSE events; older LiteLLM uses its JSON-to-stream fallback. Existing tests retain chat and non-OpenAI controls.

| Dependency | Installed environment | Current environment |
|---|---|---|
| LiteLLM | 1.83.14 | 1.108.2 |
| OpenAI SDK | 2.24.0 | 3.46.0 |
| Anthropic SDK | 0.116.0 | 2.6.0 |
| google-genai | 2.20.0 | 3.0.0 |

Commands used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `LITELLM_LOCAL_MODEL_COST_MAP=True`, existing tokenizer caches, and child-only `PYTHONPATH` for the router worktree. Router command: `python -m pytest tests -q --tb=short -p no:cacheprovider`. Agent command: `python -m pytest -p pytest_asyncio.plugin -q --tb=short -p no:cacheprovider`.

Evidence remains in the router's `.agent-scratch/2026-11-06-store-review/`: `summary-route-probe.py`, `summary-route-probe.log`, `storage-regression-*.log`, `router-final-*.log`, `agent-final-system-old-router.log`, `agent-fixed-system-new-router.log`, `agent-fixed-fresh-new-router.log`, and `matrix-receipt.txt`. The receipt records exact interpreter and import paths. No live provider calls were made. External API spend was $0.
