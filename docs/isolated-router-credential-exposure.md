This document is for developers assigning security follow-up for router copies outside shared Python.

# Older isolated router copies

Read-only reconciliation on 2026-09-05 found 11 older installations. Nine have conflicting package and code version labels. Every inspected version declaration matches the recorded Git source commit, so these are stale code labels already present in those releases. They are not evidence of unexplained changes to the inspected installed files.

All 11 still contain unsafe provider error handling. This is a static source finding, not a fresh runtime reproduction or evidence that a real credential was exposed. None will change when shared Python receives 3.26.3. No older environment was imported, executed, updated or restarted during this investigation.

## Exact versions and source identity

| Installations | Count | Package metadata | Code `__version__` | Recorded source commit |
| --- | --- | --- | --- | --- |
| Solar development venv and installed desktop bundle | 2 | 3.23.1 | 3.22.1 | `3b6244ab6020b919a05b6e8bc1e2a461783f47a5` |
| Solar staging bundle and unpacked build | 2 | 3.24.1 | 3.22.1 | `2e3a72f3d37d326fb454f9a22644f25195767972` |
| Benchmark venv | 1 | 2.4.0 | 2.4.0 | `2691886542d305b6f40bfd9bde4cff21c2cfdc4c` |
| Agent-library venv | 1 | 3.22.1 | 3.22.1 | `7860ef71f54eb75eebf42d9ca7c410bbb87e6ac5` |
| Scripter backend venv | 1 | 2.1.4 | 0.2.0 | `3be82c9c2ccb1a0295a92c72df03f251c8cd6b55` |
| Scripter Opus, Gemini and Cursor draft venvs | 3 | 1.2.1 | 0.1.0 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Kilo draft venv | 1 | 2.0.2 | 0.2.0 | `9dfecee4e4e08012df46534352168a85fe49ebd8` |

[Evidence JSON](isolated-router-copy-evidence.json) records exact installation and metadata paths, Git identities, file hashes, line anchors and exception snippets. The [consumer inventory](shared-python-router-consumers-2026-09-05.md) maps those locations to launchers. Pinned `pyproject.toml:7` matches package metadata in each group; pinned `__init__.py` matches the installed code label. All inspected files match pinned Git source after CRLF/LF normalization. This comparison covers the inspected version/error files, not whole-wheel integrity.

Use metadata, source revision and artifact contents together when assigning later updates. A bare `__version__` check would misidentify nine of these installations.

## Static exposure by group

- The four Solar copies and the agent-library copy have the same embedding error handler at `embeddings.py:394-404`. It replaces exact config strings in the final message but attaches the original provider exception. Escaped provider text can evade that replacement. Router-typed provider exceptions bypass the wrapper. These are the mechanisms independently reproduced and fixed in current router source, but they were not executed in these older environments.
- Solar, agent and benchmark chat wrappers also exact-redact provider text and attach the original exception. Representative handler starts are Solar `utils.py:1149`, agent `utils.py:1130` and benchmark `utils.py:652`. Related direct-chat and upload paths retain the same mechanism where present.
- Solar, agent and benchmark have identical inspected deep-research source. It prints raw stream/reconnection exceptions at `gemini_deep_research.py:693` and `:760`, retains raw `last_error` in details at `:771`, and preserves original causes in its outer wrappers.
- Scripter backend and Kilo directly print and publish provider exception text in the identical chat handler at `utils.py:539`. The three 1.2.1 drafts do the same at `utils.py:465`. Backend and Kilo deep-research wrappers also print and publish raw provider text.

None of the 11 copies has the 3.26.3 shared error helper, `errors.py`. Six older copies predate embeddings; the three 1.2.1 drafts also lack deep research. Their absent APIs should not be confused with safe handling in the chat APIs they do contain.

If credential-bearing provider text reaches these handlers, the source permits disclosure through messages, logs, details or original exception chains. Actual invocation, provider/dependency behavior, application-level containment and historical disclosure were not established here. No real credential was read or used.

## Ownership and separate follow-up

The filesystem and launcher records establish which repositories own the installations. They do not identify a currently assigned lead responsible for upgrading each one. Narrow checks of the owning task/readme files found no named active installation owner. Keep that assignment gap explicit rather than assigning unrelated active leads.

| Installation group | Owning project and needed assignment | Current-use limit |
| --- | --- | --- |
| Solar development, installed, staging and unpacked copies | Solar editor/release owner must own an independent venv update and any packaged release. Orchestrator must identify that owner. | Installed and build artifacts exist. Active interpreter selection and any override remain unverified. The shared-Python qualification successors do not automatically own these separate copies. |
| Benchmark venv | Benchmark owner must confirm the actual launcher, then own its environment update. | Bare-Python and venv routes both exist. The currently used route remains unresolved. |
| Agent-library venv | Agent-library owner must identify product consumers and own separate compatibility checks. | The library's local venv does not establish which package environment a product uses. |
| Scripter backend and four drafts | Scripter project/draft owner must confirm continued use and own each retained environment's update. | Environment files and some activation instructions exist. Active use and named owners remain unresolved. No parked draft was awakened. |

The orchestrator should assign these follow-ups after establishing active use. Each owner needs an isolated synthetic reproduction or verification against its exact dependency set, compatibility checks appropriate to its consumer and a separately coordinated update. Do not batch them into the shared-Python security installation or infer compatibility from their old version labels. Preserve model pins and do not run the projects' real model examples as health checks.

Shared rollout continues under its [existing packet and gates](credential-error-rollout.md). The original qualification owner's quiet boundary is confirmed in its handoff at `9a62d075`; that does not release the fresh successors' windows or the shared installation. External API spend for this reconciliation is $0.
