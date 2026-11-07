# Gemini 2.5 redirect verification

Developer evidence for the four-repo migration. The router now resolves the two retired names to their replacements. The web picker and saved-chat changes require Stan's approval before production shipping.

## Router 3.47.0

`gemini-2.5-pro` resolves to `gemini-3.8-flash`; `gemini-2.5-flash` resolves to `gemini-3.5-flash-lite`. The provider-prefixed old names resolve identically. `gemini-2.5-flash-lite` retains its original model. Reusing the successor registry objects preserves their parameter rules, SDK dispatch and pricing without a second configuration to maintain.

Google's [deprecation notice](https://ai.google.dev/gemini-api/docs/deprecations) and [pricing](https://ai.google.dev/gemini-api/docs/pricing) were checked on November 7, 2026. They confirm the December 1 mappings, 3.8 Flash's introductory $0.75/$3.75 input/output rates through December 31 and $1.50/$7.50 rates from January 1, and Flash-Lite's $0.30/$2.50 rates. The existing router January pricing follow-up remains in `docs/TASKS.md`.

Validation on November 7:

- Focused tests on the installed dependencies: 428 passed.
- Full `python -m pytest tests -q --tb=short -p no:cacheprovider` on the current dependency environment: 1,325 passed plus 17 subtests, no failures and 23 dependency warnings.
- Live `ask_ai` calls using each old name, `reasoning_effort="low"`, `temperature=1`, `candidate_count=1`, a required function tool and `rich_response=True` both returned the required tool call. Responses named the expected successors and reported nonzero cost.
- Combined live-call cost: $0.00018075.
- `git diff --check` passed.

The full suite used `D:/agent-builds/skell-e-agent/audit-review-20261106/venv/Scripts/python.exe`, with `PYTHONPATH` pointing at this checkout. Installed distribution metadata differs from source, so the source version in `pyproject.toml` identifies the code tested. Tests disabled pytest plugin autoload and used local LiteLLM pricing and existing tokenizer caches. No shared interpreter was modified.

Raw logs and the live probe are retained in `.agent-scratch/2026-11-07-gemini-25-redirect/`: `router-focused.log`, `router-full-current.log`, `live_smoke.py`, `live-smoke.json` and `live-smoke.log`. Google's SDK printed a warning when a reply contained only a function call; the assertions passed and the probe ended with `PASS`.

## Consumer verification

- Router release commit: `5c34ea4c72e5fe7731c24eaedd2040e7746a9bea`, pushed.
- Benchmark commit: `e8215947d7956708afa99aacef2747a20261f8c9`, pushed. It pins that router commit and removes both runnable 2.5 entries. Maintained-runtime validation passed 129 router contract tests and 149 consumer tests; the final launcher regression suite passed 11 tests. Historical viewing still displays 64 saved rows per retired model. Both successor smoke calls scored 1.0, with no errors and $0.00058510 combined cost.
- Knowledge-base: Stan clarified that Shakuro maintains the Proko repos and agents must not modify them. All seven tracked-file changes and the new verification document were reverted before any commit or publication. The earlier local tests and live probes cost $0.00014315. A short request for Shakuro was drafted instead. Automatic approval review blocked temporary-folder removal, so unused task logs and the isolated runtime remain locally; they are not application changes.
- Web: full backend suite passed 2,219 tests; the Angular development build and five saved-chat restoration tests passed. These cover the release script's tests and build. The core chat benchmark passed 11/11 with no errors or skips, costing $0.278575 in reported model usage. Both old saved IDs displayed Gemini 3.8 Flash in the real running app. A saved Pro chat used it for two successful lesson searches and a cited answer with seven retrieved chunks, costing $0.029217. Screenshots, persisted-response evidence and backend logs are retained in that repo.

Total recorded model cost across the task is $0.308701. Retrieval embedding costs are not separately reported by the chat benchmark.

## Remaining release gates

- Knowledge-base is outside agent implementation scope. Shakuro must make and release that change; no Jira number or Linux test work is requested from Stan for this agent task.
- Stan approved the web release. Frontend and backend were committed and pushed together as `1e09e45f6bc96caaf6c62083794eff4dc7b52b19`; no unrelated commits were included. The working tree is clean. Deployment is still running: [backend workflow](https://github.com/stanprokopenko/skell-e-web/actions/runs/42422661340) and [frontend workflow](https://github.com/stanprokopenko/skell-e-web/actions/runs/42422661326). Verify both complete successfully and the live frontend/backend identify the new release. No decision remains with Stan.
