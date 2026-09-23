# Router consumers, September 22, 2026

Developer record of Stan's request to make sure every local consumer has the latest skell-e-router.

## Release identity

The supported distribution channel is GitHub main, as documented in the [installation instructions](https://github.com/stanprokopenko/skell-e-router#install). PyPI's project JSON endpoint returned HTTP 404. GitHub has no release entries or tags. Fetched main and the local checkout both pointed to `9f93abbaf2ce141f208ea13b83d2bb7cf4fed57d`, whose project metadata is 3.31.0 and whose changes add Claude Opus 5.5. There were no newer source commits at audit time.

The default interpreter is `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe`. Its installed distribution is already 3.31.0+solar1, from Solar Sailer's locked wheel with SHA256 `5478ce1592931bd4642a714b226f00476abe1a8ecb4bbf64c9978ebc32ad0bce`. The wheel's source manifest identifies the same upstream commit. Preserved the Solar patch.

Upstream `skell_e_router/__init__.py` contains an outdated `__version__ = "3.30.1"` literal. This also appears in the installed package. The distribution metadata, source identity, and Opus 5.5 model configuration establish the actual release. A follow-up in TASKS tracks removing this duplicate version definition.

## Current consumers

| Repo or runtime | Before | After | Commit | Pushed or deployed |
|---|---|---|---|---|
| skell-e-router source | 3.31.0 | Unchanged | Existing `9f93abb` | Already pushed |
| skell-e-web | Floating main, local environment 3.31.0 | Unchanged | Existing `1d9236e` | Already pushed and deployed |
| benchmark | Exact upstream `9f93abb`, locked and installed 3.31.0 | Unchanged | Existing `9b7f97a` | Already pushed |
| solar-sailer | Locked and installed 3.31.0+solar1 | Unchanged | Existing `68082ac4` | Already pushed |
| skell-e-agent `.venv` | 3.22.1 | 3.31.0 | Evidence `0752dc5` | Pushed; local environment updated |
| skell-e-scripter `backend/venv` | 2.1.4 | 3.31.0 | Evidence `93baf76` | Pushed; local environment updated |
| Scripter draft `claude-opus-4.5` | 1.2.1 | 3.31.0 | Central evidence `93baf76` | Evidence pushed; local environment updated |
| Scripter draft `gemini-3-pro` | 1.2.1 | 3.31.0 | Central evidence `93baf76` | Evidence pushed; local environment updated |
| Scripter draft `skell-e-scripter-cursor-claude` | 1.2.1 | 3.31.0 | Central evidence `93baf76` | Evidence pushed; local environment updated |
| Scripter draft `skell-e-scripter-kilo-claude` | 2.0.2 | 3.31.0 | Central evidence `93baf76` | Evidence pushed; local environment updated |
| houston | Shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| claude-orchestrator | Shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| proko-second-brain | Complex-version crawler uses shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| beverly-bica | OCR tools use shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| kids-playground | Image generation scripts use shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| user-spam-algorithm | AI review scripts use shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| misc/test router | Standalone script uses shared Python 3.31.0+solar1 | Unchanged | None | No change needed |
| skell-e-scripter-main | Brainstorm template uses shared Python 3.31.0+solar1 | Unchanged | None | No application environment |

Six development environments were stale and were updated using the verified release wheel. Their dependency manifests already track GitHub main, which resolves to 3.31.0. No stale source pin or selected lock remained, so no manifest or lock change was necessary. Existing exact benchmark and patched Solar locks were preserved. This avoids changing consumer versioning policy as part of an installed-package update.

## Coverage and preserved artifacts

Searched GitHub and GitLab dependency manifests, requirement inputs and locks, Dockerfiles, tracked and untracked Python source imports, wheel files, and installed distribution metadata. No router consumer was found under GitLab. Aingel has planned router usage but no current dependency or installed environment. `skell-e-scripter-main` contains brainstorm documents and a router template, with no separate application environment. `solar-sailer-auto-research` has no router dependency. `benchmark - Copy` predates router integration.

Historical copies intentionally remain unchanged:

- Solar sibling worktrees `solar-sailer-p4b-20260916`, `solar-sailer-p4c-20260916`, and `solar-sailer-portrait-repair-20260916`, plus reserved `.worktrees/docs-cron` and `.worktrees/release`, contain 3.27.2+solar1. These are branches of the same repo, not independent current dependency sources. Other sibling Solar worktrees track main without a locked wheel.
- Solar generated `editor/.staging/python` and `.free-tier-live-verify` contain 3.24.1. `.dependency-lock-work/runtime-verify` contains 3.27.2+solar1. The current dependency gate checks identity before build reuse. An older 3.30.1+solar1 wheel remains beside the current wheel but is not selected by the lock.
- Benchmark retains `.venv-router-3.27.2` and a 3.26.3 reproduction environment. Its maintained launcher selects `.venv-router-3.31.0`.
- Scripter retains five `.router-candidates` environments at 3.26.3, and skell-e-agent retains two candidate/reproduction environments at 3.26.3.
- `_archive/skell-e-web-v2` retains 0.1.0. The orchestrator's September 5 incident evidence retains 3.26.2.
- Router `dist` contains older release wheels alongside 3.31.0. Historical package files and rollback evidence were not rewritten.

## Verification

- Router model configuration and Anthropic tests: 255 passed against source, and 255 passed against the actual shared installed package. Used `scripts/run_security_tests_offline.py` to block network access and avoid global pytest plugins.
- Solar: locked wheel provenance and hashes verified, 17 dependency lock tests passed, and two router smoke tests passed.
- Web: release check passed with 68 backend tests and Angular build; another 185 routing tests passed. Live Cloud Run revision `skell-e-backend-00351-4p8` serves 100% of traffic. Existing [backend deployment](https://github.com/stanprokopenko/skell-e-web/actions/runs/35806531599) and [hosting deployment](https://github.com/stanprokopenko/skell-e-web/actions/runs/35806531560) succeeded. Docker force-installs router from main with a fresh cache key on each build.
- Benchmark: current launcher check passed and verified installed package, lock, and runtime manifest identity. Its earlier upgrade evidence records 115 router and 45 consumer tests passed; those counts were not rerun in this audit.
- Skell-e-agent: 264 guarded tests passed. See its `docs/router-3.31.0-installed.json` and upgrade documentation in commit `0752dc5` for the installed identity receipt.
- Scripter: main and Kilo each passed 127 checks; Opus and Gemini each passed 125 checks. Cursor passed 125 checks with two previously documented application failures, an unavailable Gemini default and zero temperature changing to 0.7. All five passed the 115 router security and output-limit cases. No new router regression was found. Exact receipts and missing runtime dependency additions are in Scripter `docs/router-upgrade/router-3.31.0-installed.json` at `93baf76`. Existing dependency versions were unchanged.

Cursor's two application defects remain outside this dependency update. Archived environments and historical branch snapshots listed above remain old intentionally. Current default Python, selected locks, and all six original development environments are current.

No paid model calls were made. Task API spend is $0.

## Resources

Release wheel and verification scratch: `.agent-scratch/2026-09-22-router-consumers/`. Built the wheel directly from immutable upstream commit `9f93abbaf2ce141f208ea13b83d2bb7cf4fed57d`; SHA256 is `6b4dc989c645ab2e0d4502961813ae190900c49c07e46f5ed8af9b2be4f0eef7`.

Development environment backups and install receipts: `C:/Users/Stan/Documents/GitHub/skell-e-agent/.agent-scratch/2026-09-22-router-consumers/`. Scratch resources remain available for inspection.
