# Gemini 2.5 redirect verification

Developer evidence for the four-repo migration. The router now resolves the two retired names to their replacements. The web picker and saved-chat changes require Stan's approval before production shipping.

## Router 3.47.0

`gemini-2.5-pro` resolves to `gemini-3.8-flash`; `gemini-2.5-flash` resolves to `gemini-3.5-flash-lite`. The provider-prefixed old names resolve identically. `gemini-2.5-flash-lite` retains its original model. Reusing the successor registry objects preserves their parameter rules, SDK dispatch and pricing without a second configuration to maintain.

Google's [deprecation notice](https://ai.google.dev/gemini-api/docs/deprecations) and [pricing](https://ai.google.dev/gemini-api/docs/pricing) were checked on November 7, 2026. They confirm the December 1 mappings, 3.8 Flash's introductory $0.75/$3.75 input/output rates through December 31 and $1.50/$7.50 rates from January 1, and Flash-Lite's $0.30/$2.50 rates. The existing router January pricing follow-up remains in `docs/TASKS.md`.

Validation on November 7:

- Focused tests on the installed dependencies: 428 passed.
- Full `python -m pytest tests -q --tb=short -p no:cacheprovider` on the current dependency environment: 1,325 passed plus 17 subtests, no failures. The 23 warnings are dependency serialization warnings.
- Live `ask_ai` calls using each old name, `reasoning_effort="low"`, `temperature=1`, `candidate_count=1`, a required function tool and `rich_response=True` both returned the required tool call. Responses named the expected successors and reported nonzero cost.
- Combined live-call cost: $0.00018075.
- `git diff --check` passed.

The full suite used `D:/agent-builds/skell-e-agent/audit-review-20261106/venv/Scripts/python.exe`, with `PYTHONPATH` pointing at this checkout. Installed distribution metadata differs from source, so the source version in `pyproject.toml` identifies the code tested. Tests disabled pytest plugin autoload and used local LiteLLM pricing and existing tokenizer caches. No shared interpreter was modified.

Raw logs and the live probe are retained in `.agent-scratch/2026-11-07-gemini-25-redirect/`: `router-focused.log`, `router-full-current.log`, `live_smoke.py`, `live-smoke.json` and `live-smoke.log`. Google's SDK printed a warning when a reply contained only a function call; the assertions passed and the probe ended with `PASS`.

## Remaining release gates

- Benchmark must pin the released router commit, pass its maintained runtime checks, open historical results and complete a successor smoke run.
- Knowledge-base needs the mandatory Jira task number before branching or committing, its full tests and repo review.
- Web needs full backend checks, the chat core benchmark, real-app saved-chat tool verification and Stan's picker/price approval before any push.
