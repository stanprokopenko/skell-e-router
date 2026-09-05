This document is for developers assigned to upgrade retained router installations outside shared Python.

# Isolated router ownership and upgrade handoffs

Read-only follow-up on 2026-09-05, with observations from approximately 12:38 to 12:49 PDT. Four groups contain eleven retained router copies. No copy was executed, imported, updated, restarted or removed. External API spend was $0. Only this handoff document was written.

The [static exposure assessment](isolated-router-credential-exposure.md), [per-copy evidence JSON](isolated-router-copy-evidence.json) and [consumer inventory](shared-python-router-consumers-2026-09-05.md) remain the source for the eleven unsafe error handlers and their router identities. The nine conflicting version labels already match historical source. This follow-up did not repeat that reconciliation or reproduce failures in older runtimes.

The installed Solar app has recent use evidence. Solar development and packaging, and the agent library's local tests, have configured routes. The benchmark project has recent use evidence, but its old venv is not attributed to those runs. Scripter's current use remains unknown; one draft has a historical startup record. No current isolated-copy upgrade owner was established in the bounded roster and project-record checks.

Shared-Python rollout remains owned by `lead-1032891d-b8b5-4bbc-a05e-37284eaef9e3`. These handoffs do not modify or reopen its source, review, rollout packet or evidence. The orchestrator recorded this follow-up in its deferred-request ledger at `22e61fce`.

## How to read the use classifications

- **Observed use** means a dated launch or execution record. An app launch does not prove a particular router call occurred.
- **Configured current** means a maintained launcher, build recipe or test instruction selects the route. It does not establish a running process today.
- **Historical** means the evidence establishes an earlier run or generated artifact. It does not authorize retirement.
- **Unknown** means the bounded checks did not attribute current use. File dates, package presence and a recent project commit are not runtime evidence.

At the process snapshot, Windows reported no `python.exe`, `pythonw.exe`, `uvicorn.exe`, `Solar Sailer.exe` or `electron.exe`. No matching Windows service name or executable-path association was found for Solar, benchmark, agent or Scripter. No non-Microsoft scheduled-task name, executable or working directory matched these projects. These are momentary, bounded negative observations. Generic wrappers whose arguments identify a project, renamed executables and remote jobs remain outside that attribution. No process command lines, memory or environment blocks were read. Service path strings were filtered inside the query; no service arguments were printed.

Houston's matching `crons/jobs.json:21-33` entry is `solar-sailer-docs`, a website-documentation job. It is not evidence of a Solar editor Python launch. No job was invoked or changed.

The current roster contained qualification work for Solar and a closed September benchmark-monitoring lead, `1788521204.003959`. It did not establish an active Solar release, agent-library, Scripter or benchmark environment owner. Qualification lead `lead-30949f07-8c13-4173-9d6f-7ac77b16f685` and its successors are not assigned these release copies by this handoff. No parked lead or remote machine was contacted.

## Retained installation and dependency identities

In the tables, `G` expands to `C:/Users/Stan/Documents/GitHub` and `P` to `C:/Users/Stan/AppData/Local/Programs`. Each listed directory is retained as found. This is preservation pending a controlled upgrade or a separate retention decision, not a decision that every directory must remain a runnable product.

| Copy | Exact retained environment directory | Router distribution | Recorded router source revision |
| --- | --- | --- | --- |
| Solar development | `G/solar-sailer/editor/server/.venv` | 3.23.1 | `3b6244ab6020b919a05b6e8bc1e2a461783f47a5` |
| Solar installed | `P/Solar Sailer/resources/python` | 3.23.1 | `3b6244ab6020b919a05b6e8bc1e2a461783f47a5` |
| Solar staging | `G/solar-sailer/editor/.staging/python` | 3.24.1 | `2e3a72f3d37d326fb454f9a22644f25195767972` |
| Solar unpacked | `G/solar-sailer/editor/dist-installer/win-unpacked/resources/python` | 3.24.1 | `2e3a72f3d37d326fb454f9a22644f25195767972` |
| Benchmark | `G/benchmark/env` | 2.4.0 | `2691886542d305b6f40bfd9bde4cff21c2cfdc4c` |
| Agent library | `G/skell-e-agent/.venv` | 3.22.1 | `7860ef71f54eb75eebf42d9ca7c410bbb87e6ac5` |
| Scripter backend | `G/skell-e-scripter/backend/venv` | 2.1.4 | `3be82c9c2ccb1a0295a92c72df03f251c8cd6b55` |
| Scripter Opus | `G/skell-e-scripter-drafts/claude-opus-4.5/backend/venv` | 1.2.1 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Gemini | `G/skell-e-scripter-drafts/gemini-3-pro/backend/venv` | 1.2.1 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Cursor | `G/skell-e-scripter-drafts/skell-e-scripter-cursor-claude/backend/venv` | 1.2.1 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Kilo | `G/skell-e-scripter-drafts/skell-e-scripter-kilo-claude/backend/venv` | 2.0.2 | `9dfecee4e4e08012df46534352168a85fe49ebd8` |

Venv interpreters are `<environment>/Scripts/python.exe`; embedded interpreters are `<environment>/python.exe`. Router files are `<environment>/Lib/site-packages/skell_e_router`. Metadata paths and inspected file hashes are already recorded in the evidence JSON.

The following installed dependency identities were read statically from distribution metadata using the shared interpreter with `-S -B`. No older interpreter or provider SDK was imported. An absent entry means no matching distribution was found in that explicit site-packages directory; it is not a claim about an arbitrary inherited runtime path.

| Copy | LiteLLM | OpenAI | Anthropic | google-genai | Pydantic | FastAPI |
| --- | --- | --- | --- | --- | --- | --- |
| Solar development | 1.90.1 | 2.44.0 | 0.115.0 | 2.10.0 | 2.13.4 | 0.141.1 |
| Solar installed | 1.98.0 | 2.54.0 | 1.2.0 | 2.20.0 | 2.13.5 | 0.141.1 |
| Solar staging and unpacked | 1.99.0 | 2.54.0 | 1.3.0 | 2.21.0 | 2.13.5 | 0.141.1 |
| Benchmark | 1.81.8 | 2.17.0 | absent | 1.62.0 | 2.12.5 | 0.128.2 |
| Agent library | 1.96.2 | 2.54.0 | 0.122.0 | 2.18.1 | 2.13.4 | absent |
| Scripter backend and Kilo | 1.80.10 | 2.14.0 | absent | 1.56.0 | 2.12.5 | 0.115.6 |
| Scripter Opus | 1.80.10 | 2.12.0 | absent | absent | 2.10.4 | 0.115.6 |
| Scripter Gemini | 1.80.10 | 2.12.0 | absent | absent | 2.12.5 | 0.124.4 |
| Scripter Cursor | 1.80.10 | 2.12.0 | absent | absent | 2.12.5 | 0.115.6 |

All eleven have httpx 0.28.1 and h11 0.16.0. Solar and agent have tenacity 9.1.4 and requests 2.34.2. Benchmark has tenacity 9.1.3 and requests 2.32.5. All Scripter copies have tenacity 9.1.2 and requests 2.32.5. Solar's four copies contain skell-e-agent 0.7.0 and pydantic-ai-slim 2.29.0. The agent development environment contains pydantic-ai-slim 2.29.0 and pydantic-ai-harness 0.18.1.

These different SDK sets are a validation dependency. Do not copy the shared-Python installation command into every environment. In particular, the new router requires dependencies missing from some older copies. Prepare a reproducible candidate environment and document necessary dependency additions before replacing a retained runtime. Do not opportunistically upgrade unrelated packages.

## Solar release and development handoff

Assign a Solar release/development owner. This group has four retained copies, not four independent products. The owning repository is `G/solar-sailer`, observed HEAD `9457e6dbaae9ebc63fc425605f1c9b9bdc6b9e3d`. That current source identity does not identify the installed app's source. `docs/release-runbook.md:3` owns the build/release workflow; follow it before a later release change.

| Route | Use classification | Evidence |
| --- | --- | --- |
| Installed `P/Solar Sailer/Solar Sailer.exe` and its embedded Python | App observed in recent use; configured current installation. Exact Python invocation and router calls remain unobserved. | `solar-sailer/docs/TASKS.md:186` records a September 2 approved install of unsigned 0.4.2 test build `c5ff9e65`, left over production until the next release. A cold protocol link launched the installed executable and its recorded live gate passed 31/31 checks. |
| Development `editor/server/.venv` | Configured current development route; recent exact execution unknown. | `docs/TASKS.md:872` documents the colocated environment. `editor/electron/pythonManager.ts:227` tries explicit `PROKO_PYTHON`, packaged runtime, colocated venv, then PATH. |
| `.staging/python` | Configured current reusable build input; recent execution unknown. | `editor/scripts/build-installer.mjs:327-338` selects and caches this assembly. `editor/electron-builder.yml:135-136` copies it into packaged `resources/python`. |
| Root `dist-installer/win-unpacked/resources/python` | Generated historical artifact; execution and current use unknown. | Packaging defines the output purpose. No launch record was found for this exact unpacked artifact. Do not confuse it with a release worktree's same-named directory. |

A safe-field read of `C:/Users/Stan/AppData/Roaming/solar-sailer/logs/2026-09-02T16-34-18_8cd1/session.json` found `startTime=2026-09-02T16:34:18.296Z` and `appVersion=0.4.2`. Those fields corroborate a session, not its executable or interpreter. The task record says packaged builds lack a durable main-process log. Do not infer Python selection from a missing log or inspect provider failures to fill that gap.

The staging cache is a concrete release hazard. Requirements line 22 tracks router `@main`, while the assembly stamp hashes requirements text and Python/platform identity. Lines 333-335 explicitly state that upstream router movement does not invalidate the stamp. The existing `.staging/python/.assembly-stamp` contains the current requirements file's SHA-256; this was checked without running the builder. A later owner must deliberately rebuild the bundle with `--force-python` or an equivalent validated assembly change. Merely fetching the repaired router repository or rerunning an unchanged packaging command can reuse the unsafe copy.

Bounded work for this owner:

1. Retain the four paths above while preparing replacement environments. Record the installed executable, package artifact and effective selected Python through a controlled launch preflight before any later authorized activation. Keep the unsigned installed test build distinct from published 0.4.2 bytes and from root unpacked output.
2. Prepare the repaired router against each distinct SDK baseline in the dependency table. Preserve the model configuration and agent 0.7.0 contract. Validate development and embedded candidates independently; a venv pass does not establish installer compatibility.
3. Reuse `editor/server/tests/test_skell_e_router_smoke.py:6-38` for primary/backup model catalog expectations and `tests/test_embeddings_router.py:7-68` for synthetic rich embeddings and usage/cost mapping. Both shown embedding cases replace the router call; the cost case also replaces the ledger writer. Ensure the whole selected suite isolates or stubs ledger writes. In addition, run the repaired router's own synthetic failure suite against the actual candidate package, because a stubbed consumer test cannot prove provider-error containment. Include chat-agent bridge compatibility because Solar imports skell-e-agent.
4. Build replacement staging and unpacked artifacts deliberately, then verify their package files, dependencies and source identities. Follow release-owner coordination for installed-app activation. Do not reuse or ship the old staging directory unnoticed.

Stan decision needed before a release schedule change: `docs/TASKS.md:36` records his September 1 direction to defer the next team installer until more features land. The assigned owner must present whether a security-only installed-app replacement should precede that release, or be included in it. No decision is needed to preserve these files and prepare tested replacements. Retirement of the old unpacked artifact remains a separate decision; no cleanup is authorized here.

Done for this owner means each retained current route points to a verified repaired package, replacement packaging cannot silently reuse the old cache, the applicable consumer fixtures pass, and installed activation is recorded separately from artifact preparation. Historical artifacts remain identified until their retention decision is made.

## Benchmark environment handoff

Assign a benchmark owner to resolve the maintained launch route and upgrade it. Retained environment is `G/benchmark/env`; owning repo is `G/benchmark`, observed HEAD `ee958d4f4b9ef14825d96cf48ffaf512dfa473af`. No current owner was established. The most recent matching roster lead, `1788521204.003959`, is closed and was not contacted.

The benchmark project has recent observed run evidence in `docs/gpt-6-astra-benchmark-results.md:1-5`, dated September 4, and `docs/gemini-3-8-flash-benchmark-results.md:3`, dated September 2. These records report completed runs but do not identify `env/Scripts/python.exe`. Therefore project use is recent, while current use of the retained 2.4.0 environment remains unknown. Its old directory timestamp is not proof of retirement.

README lines 184 and 284 document bare `python main.py` and `python -m uvicorn web_api.main:app`; neither selects `env`. The bounded tracked-launcher/test search found no explicit venv launcher or established automated Python test suite. `TASK_LIST.md:128-131` describes tests as planned. Requirements line 9 tracks router main. Do not infer that September's model runs used this old venv merely because both belong to the same repository.

Bounded work for this owner:

1. Establish one explicit maintained interpreter for CLI and API child launches. Preserve the old `env` until its reproduction value is decided. If it will remain runnable, prepare a repaired candidate against its exact baseline. If the maintained route is elsewhere, identify that route and keep this copy labeled historical/unknown; do not claim this copy was upgraded.
2. Check missing dependencies before selecting the router-only replacement method. The retained environment has no Anthropic distribution in its site-packages. A dependency-preserving shared-Python recipe is not automatically sufficient here.
3. Add or select small offline fixtures around `models/skelle_router_model.py:49-81`: response text, `rich_response=True`, absent usage fields, reasoning-token accounting, `RouterError.code/message`, unexpected failures and safe returned/logged error text. The wrapper currently returns and logs error text. Use a fixture evaluator and temporary output destination to verify CLI/API result serialization without provider calls. Preserve existing evaluation data under `prompt_archives/eval type tests/` as inputs; do not launch a real benchmark to validate.
4. Run the candidate router's synthetic containment tests against the selected environment and record package/dependency identity. An exact launcher and passing isolated compatibility checks are required before future paid runs resume through that route.

Concrete retention decision if the old env will not be maintained: does Stan need `G/benchmark/env` to remain runnable for historical benchmark reproduction, or may it remain a reference artifact while future runs use a repaired, explicit project environment? Recent project use is already established, so this is not a proposal to retire the benchmark project. Preserve files until that choice is recorded.

## Agent-library development handoff

Assign a skell-e-agent development owner. Retained environment is `G/skell-e-agent/.venv`; owning repository is `G/skell-e-agent`, observed HEAD `687eea9a0333cf9fe45808932638674302765676`. No current named maintenance lead was established.

This is a configured local development/test route. `tests/recorded_feed.py:9` explicitly uses `.venv/Scripts/python` for fixture generation, and `docs/superpowers/plans/2026-08-13-02-bridge-model.md:536` specifies it for pytest. Current execution remains unknown. CI uses its own Python 3.11 setup and installation in `.github/workflows/ci.yml:13-17`; CI success is not a runtime observation of this local venv.

The local agent distribution metadata says 0.1.0 and its direct URL identifies an editable install of `G/skell-e-agent`. Current `pyproject.toml:3` says 0.7.0. This is a distinct package-identity fact, not a reopened router-label reconciliation. A new candidate must record both editable source HEAD and installed metadata before testing.

Solar Sailer is a configured product consumer of the library, through `editor/server/requirements.txt:25` at tag `v0.7.0` and `services/chat_agent/model.py:42,321`. Solar development's non-editable agent metadata resolves that tag to `687eea9a0333cf9fe45808932638674302765676`. Solar's ordinary launchers select its own venv or bundled Python, not the agent repository's `.venv`. Upgrading this local development environment does not update those product copies.

The agent README's skell-e-web consumer language is not enough to prove current integration. The package design describes a planned migration; the inspected `skell-e-web/backend/requirements.txt` has no agent dependency and `backend/rag/chat_v2.py:292` still defines its own `RouterModel`. Do not add it as an observed local-agent-venv consumer.

Bounded work for this owner:

1. Preserve the local venv while preparing a reproducible development environment with the accepted router. Preserve pydantic-ai-slim 2.29.0 and pydantic-ai-harness 0.18.1 unless a demonstrated compatibility failure requires a separate change.
2. Run the offline bridge, capabilities and event-contract fixtures, including `tests/test_bridge.py`, `tests/test_capabilities.py` and `tests/test_events.py:118`. Verify the existing `tests/fixtures/recorded_event_feed.jsonl`; do not regenerate it as a way to silence a regression. A changed event fixture is a contract change.
3. Add a targeted synthetic integration case through the real repaired router boundary. Existing bridge tests at lines 263 and 278 stub `ask_ai` and expect particular exception behavior, including a retained cause for an actionable tools/reasoning hint. Check that the new safe router errors preserve usable caller behavior without restoring raw provider text or chains. Passing those stub-only tests is not containment evidence.
4. Record local environment identity and fixture results. Coordinate any actual agent source/API change with Solar's separate owner. An environment-only router update does not require publishing a new agent tag or upgrading unrelated product copies.

No product-retention decision is required to preserve this documented development route and prepare an upgrade. If the assigned owner proposes discontinuing local development, Stan must choose whether `.venv` needs to remain runnable. That choice is separate from retaining the agent package used by Solar. No current-use proof or retirement decision is inferred from its directory date.

## Scripter backend and draft handoff

Assign one Scripter triage/upgrade owner for the five environments. Do not wake the old draft leads. There is no current Scripter owner in the inspected roster, and no current launch or scheduled use was established.

| Retained source directory under `G` | Use classification | Bounded evidence and source identity |
| --- | --- | --- |
| `skell-e-scripter` | Current use unknown; historical launch instructions retained. | README lines 22-25 activate backend venv and run `python main.py`; TASKS marks backend/router implementation complete. Repo HEAD `25ee5e4fef22575f2bef65675287170a80131e64`. Completed tasks do not prove current use. |
| `skell-e-scripter-drafts/claude-opus-4.5` | Unknown. | README lines 37-49 documents venv/uvicorn setup, while TASKS still leaves backend/router milestones unchecked. No Git root resolved in the bounded query. Preserve this directory as its own source copy. |
| `skell-e-scripter-drafts/gemini-3-pro` | Unknown, including concrete launch route. | Service imports router and requirements declares it, but no concrete maintained launcher was found; TASKS backend/router milestones remain unchecked. No Git root resolved. |
| `skell-e-scripter-drafts/skell-e-scripter-cursor-claude` | Historical app startup observed; current use and historical interpreter attribution unknown. | `logs/app.log` contains the exact fixed event `Skell-E Scripter API started` at local timestamp `2025-12-15 23:41:39`. Source startup emits it at `backend/main.py:48`. README lines 40-56 documents venv launch. Repo HEAD `04d8cd0c22e3217be6cce1aeafb5ea613b6908d5`. |
| `skell-e-scripter-drafts/skell-e-scripter-kilo-claude` | Current use unknown; historical launch instructions retained. | README lines 22-25 activates backend venv; TASKS marks implementation complete. Independent Git root, HEAD `04d8cd0c22e3217be6cce1aeafb5ea613b6908d5`. Equal HEAD does not prove equal working content or runtime. |

Only the fixed startup marker and timestamp were extracted from the Cursor app log. Its app and AI log file metadata stop in December 2025; no AI log contents or provider errors were read. The orchestrator's earlier raw-history Scripter lookup found today's inventory mentions only. That lack of evidence does not settle current use or retirement, and the lookup was not repeated.

The first decision for Stan is which implementations should remain runnable: the main Scripter backend, any specifically named drafts, or none pending a future restart. The recommendation is to retain all files now and upgrade only the runnable implementations he selects. Do not silently delete, archive, retire or consolidate drafts. Source-only/reference retention is different from maintaining an executable environment.

Bounded work for the assigned owner:

1. Obtain and record that per-implementation retention choice. Until then, prepare compatibility fixtures and candidate dependency plans without altering retained environments. Preserve the two source copies without a resolved Git root; they need a named provenance record before any replacement work.
2. For each selected implementation, identify an explicit backend interpreter and prepare a candidate environment. Backend/Kilo and the three 1.2.1 drafts have different router API/dependency starting points. All lack an Anthropic distribution, and the three 1.2.1 drafts also lack google-genai. Do not blindly install the new wheel with dependencies disabled and assume it is usable.
3. No established Python test suite was found in the bounded backend/draft source search. Create targeted offline service fixtures. Main `backend/services/ai_service.py` imports chat and deep-research APIs at line 23, calls chat around lines 284 and 349, and deep research around lines 498 and 571. Validate ordinary string responses, model/options forwarding, token limits, error handling, cancellation and streaming progress where the selected implementation has those routes. Keep prompts, models and content files unchanged. Mock providers, HTTP and file side effects before imports; no FastAPI server startup or real example run is a health check.
4. Add synthetic credential-bearing provider failures through the real repaired router and each selected service's error/output path. Assert safe logs, HTTP/streamed error text and tracebacks. Check each retained draft's actual service code rather than treating the main backend test as coverage for all copies. The older handler findings are already in the exposure assessment.
5. Replace only explicitly retained runnable environments after candidate checks pass, then record installed identity and an owner-approved launch route. Leave every unselected/unknown draft labeled as such, with files intact and its outstanding retention choice visible.

## Common candidate and acceptance requirements

The accepted security artifact at inspection time is router 3.26.3 from source `63b5fd22bacef9100b09cdee355bd8839439be78`, wheel `G/skell-e-router-security/dist/skell_e_router-3.26.3-py3-none-any.whl`, SHA-256 `f7f21d1e30dcad7a4d46bc57ff87eeed7cb1255617fce876546fee7f23e01707`. Confirm the current accepted artifact with the [shared rollout packet](credential-error-rollout.md) and security owner before using it. The source/wheel review is already complete; this work is consumer compatibility and independent environment replacement.

The packet now provides installed-package assertions and a guarded `--require-installed` test route. Reuse the current owner-maintained validation implementation, not a stale source-only command copied from the earlier inventory. Do not edit the shared owner's packet, runner or evidence as part of these upgrades. Each group writes its own evidence in its owning repository and selects its explicit candidate interpreter.

For every retained runnable environment, the new owner must capture the old dependency baseline, prepare and inspect the candidate, run guarded synthetic failures plus relevant consumer fixtures, and assert the actual imported candidate/installed path. Required failures include raw and escaped synthetic credentials, malformed local h11 headers, public messages/details, standard tracebacks, stdout/stderr/logs, and original exception cause/context. Successful response shapes, output limits and usage fields must also survive. Run tests in an isolated profile with synthetic credentials and network denial in place before importing the provider stack. No paid calls are necessary.

Completion requires a per-copy result, not a group-wide assumption: retained path, chosen interpreter, router/dependency identities, source/artifact hash, fixture outcomes, current-use classification, actual owner, and whether activation occurred. Do not claim a source checkout, a cached build or a passing stub test proves a different installed runtime is repaired. Report preserved historical/unknown copies separately from upgraded current copies.

This follow-up's work is complete when the handoffs are delivered. It does not complete the eleven environment upgrades or decide Stan's retention and release-timing choices. Those choices and the separate upgrade assignments remain explicit above.
