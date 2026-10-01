# Retire Claude Sonnet 4.5 from the router and the benchmark

Status: proposed, not started. Written 2026-10-01.

## Context

On 2026-09-30 Anthropic emailed that Claude Sonnet 4.5 (`claude-sonnet-4-5`, dated id `claude-sonnet-4-5-20250929`) retires on the Claude API on 2026-11-24 at 9:00 AM PT. Availability may degrade from 2026-10-30. The recommended replacement is Claude Sonnet 5 (`claude-sonnet-5`), which both repos already have as its own entry. The benchmark is the one caller that will fire on its own: `main.py` runs every model in `config.yaml` when `--models` is omitted. The 2026-09-25 work set the convention. Retired entries get deleted outright, never repointed to the successor, because a key like `claude-sonnet-4.5` serving Sonnet 5 would put Sonnet 5 scores under the wrong name (see `benchmark/docs/plans/2026-09-25-retired-claude-models-swap.md`, and router commit 475341b, v3.33.0). Do the same here. Do not register or repoint anything for Sonnet 5.

## Benchmark (`C:\Users\Stan\Documents\GitHub\benchmark`, its own checkout, commit by path there)

1. `config.yaml`: delete the `claude-sonnet-4.5:` block, lines 506-514 (header through the blank line before `claude-haiku-4.5:`).
2. `CLAUDE.md` line 17: in the example command, change `claude-sonnet-4.5` to `claude-sonnet-5`.
3. `docs/TASKS.md`: add a ticked line under line 23, matching its style: removed `claude-sonnet-4.5` from `config.yaml` on 2026-10-01 ahead of Anthropic's 2026-11-24 retirement, successor `claude-sonnet-5` already has entries, historical results unchanged, link to this plan.
4. Leave `results/`, `prompt_archives/`, `docs/*-results.md`, `docs/unslop-scorer.md` and `sub_projects/` alone. They are history. Every stored `claude-sonnet-4.5` row already carries `provider`, so `web_api/main.py` (around line 327) never needs the config entry to label it.
5. Do not touch `runtime-manifest.json` or `.venv-router-3.36.0`. The benchmark does not need a new router release for this.

## Router (this repo)

1. `skell_e_router/model_config.py`: delete the `"claude-sonnet-4-5-20250929": AIModel(...)` block, lines 506-513.
2. `skell_e_router/anthropic_direct.py`: delete the price entry on line 55.
3. `skell_e_router/Skell-E-Router-DOCUMENTATION.md`: delete the pricing table row on line 1154.
4. `tests/test_model_config.py`: add `"claude-sonnet-4-5-20250929"` to the `test_retired_aliases_are_gone` list (lines 223-231).
5. `pyproject.toml` line 7: bump `3.36.0` to `3.37.0` (a removal is a minor bump, as in v3.33.0 and v3.34.0).
6. No routing tiers, aliases, defaults or other tests name the id. README does not mention it.
7. `docs/TASKS.md` line 39: replace the watch item with two lines.
   - `[x]` Sonnet 4.5 removed 2026-10-01 (v3.37.0) after Anthropic announced retirement for 2026-11-24, replacement `claude-sonnet-5`.
   - `[ ]` Watch `claude-haiku-4-5` (earliest possible retirement 2026-10-15) and `claude-opus-4-5` (earliest 2026-11-24). Nothing announced; Anthropic gives 60 days' notice.
8. Add a third TASKS line, open: repoint `claude-sonnet-4-5-20250929` in skell-e-web (`backend/constants.py` lines 74 and 101, `proko-app/src/app/model-options.constants.ts` lines 42 and 96, `backend/benchmarks/looping_bench.py` lines 90 and 97) and skell-e-scripter (`backend/services/ai_service.py` line 253) before 2026-10-30.

## Push order for the router

skell-e-web and skell-e-scripter install the router from `@main`, and skell-e-web's RAG planner runs on this id today. A pushed router without the entry breaks their next install. Commit the router change locally, then stop and tell Stan. Push only after those two repos stop naming the id, or when Stan says to push anyway. The benchmark commit has no such dependency.

## Tests

- Router: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; python -m pytest tests -q`. The env var works around the broken global `logfire` plugin (router `docs/TASKS.md` line 58). Expect all green with one more retired-alias case.
- Benchmark config parses: `.\.venv-router-3.36.0\Scripts\python.exe -E -s -c "import yaml; d=yaml.safe_load(open('config.yaml', encoding='utf-8')); print(len(d['models']), 'claude-sonnet-4.5' in d['models'])"`. Expect the count down by 1 and `False`.
- Benchmark offline suite: `.\.venv-router-3.36.0\Scripts\python.exe -E -s scripts\run_benchmark_tests_offline.py`. No paid calls.
- `.\benchmark.ps1 view-scores` still lists `claude-sonnet-4.5` with its old score (76.18 on the 61-prompt set). Do not start a paid run.

## Commit messages

- Benchmark: `Remove claude-sonnet-4.5 ahead of Anthropic's 2026-11-24 retirement`
- Router: `feat: remove claude-sonnet-4-5-20250929 ahead of 2026-11-24 retirement (v3.37.0)`, body naming `claude-sonnet-5` as the replacement and the skell-e-web and skell-e-scripter pins as the push blocker.

## Done when

- `git grep claude-sonnet-4.5` in benchmark finds only history (results docs, `sub_projects/`, older plans).
- `git grep claude-sonnet-4-5-20250929` in the router finds only the retired-alias test, TASKS, the 09-25 audit and old conversation exports.
- Router tests and the benchmark offline suite pass.
- Router TASKS has the ticked Sonnet line, the Haiku and Opus watch with their dates, and the downstream repoint item.
- Stan has been told the router commit is waiting on skell-e-web and skell-e-scripter before push.
