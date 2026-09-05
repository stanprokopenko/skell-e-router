This document is for developers coordinating the router security release and validating its local consumers.

# Shared Python router consumers and rollout checks

Inventory date: 2026-09-05, late morning America/Los_Angeles. This is a read-only inventory and validation plan, not an installed-package rollout approval or compatibility result.

The shared Python 3.11 installation contains router 3.26.2. Session search, Houston TLDR and the relay's vector-index rebuild default to PATH `python`, which resolves to that installation in the inspected shell. Solar Sailer input qualification explicitly documents the same interpreter. Several other projects have independent virtual environments or embedded Python installations. A shared-package update will not update those copies.

The inventory is complete within the bounded local search described below. Rollout still needs runtime confirmation for existing parent processes, coordination with qualification work, an installed-package security probe, and attribution of one persistent Python process. Unknown activity or interpreter choice is a gap, not evidence of compatibility.

## Scope and evidence limits

- Read launchers, dependency declarations and selected operating instructions within `C:/Users/Stan/Documents/GitHub`. Searched source and dependency files for `skell_e_router` and `skell-e-router`, including Python, JavaScript, TypeScript, shell launchers and project manifests. Followed relevant Scripter draft directories separately because the broad search did not expose their ignored contents.
- Skipped repository histories, transcripts, credentials, environment dumps, process command lines, generated dependency trees and the `_archive` repository collection. Solar Sailer's main `benchmarks` path is a junction to `D:/solar-sailer/benchmarks`; the qualification worktree has a real tracked benchmarks directory. No broad search of D: was performed.
- No GitLab access, remote machine access, parked-lead wakeups, application launches, live search, model calls, package installs, pin changes or consumer-environment changes occurred. Only this document was written. External service spend was $0.
- Metadata and package-location probes used the standard library without importing router or application code. Some probes used normal Python startup to inspect current-shell resolution. Explicit-site probes used `-S -B` to bypass site startup and bytecode writes. Explicit-site results establish what is installed at that path; they do not establish the import path of a running process.
- Launcher defaults are source evidence. They do not prove the installed Houston bundle matches this checkout or that an existing Electron/Node process inherited this shell's PATH. Source entry points may also accept an injected interpreter. Those limits remain explicit below.

## Verified runtime identities

Path abbreviations in the tables are literal prefixes: `G = C:/Users/Stan/Documents/GitHub`, `P = C:/Users/Stan/AppData/Local/Programs`, and `S = P/Python/Python311/python.exe`. Expand them before running a command.

Every router distribution below is a non-editable installation at the runtime's `Lib/site-packages/skell_e_router/__init__.py`. Source revisions come from distribution `direct_url.json` metadata. The main router checkout can shadow this installation when used as the import directory, even though distribution metadata still reports the installed package. Elsewhere in this document, an unqualified installed version means distribution metadata; the table records the code's literal `__version__` separately.

| Runtime | Exact interpreter relative to the prefixes above | Distribution version | Code `__version__` | Installed source revision |
| --- | --- | --- | --- | --- |
| Shared system Python | `S`, Python 3.11.2 | 3.26.2 | 3.26.2 | `d8ae9876fd2f095d5e6e03e11710c6d7a8ddcefe` |
| Solar development environment | `G/solar-sailer/editor/server/.venv/Scripts/python.exe`, Python 3.11.2 | 3.23.1 | 3.22.1 | `3b6244ab6020b919a05b6e8bc1e2a461783f47a5` |
| Solar installed desktop bundle | `P/Solar Sailer/resources/python/python.exe` | 3.23.1 | 3.22.1 | `3b6244ab6020b919a05b6e8bc1e2a461783f47a5` |
| Solar staging bundle | `G/solar-sailer/editor/.staging/python/python.exe`, Python 3.11.15 | 3.24.1 | 3.22.1 | `2e3a72f3d37d326fb454f9a22644f25195767972` |
| Solar unpacked build | `G/solar-sailer/editor/dist-installer/win-unpacked/resources/python/python.exe` | 3.24.1 | 3.22.1 | `2e3a72f3d37d326fb454f9a22644f25195767972` |
| Benchmark virtual environment | `G/benchmark/env/Scripts/python.exe`, Python 3.11.2 | 2.4.0 | 2.4.0 | `2691886542d305b6f40bfd9bde4cff21c2cfdc4c` |
| Agent library environment | `G/skell-e-agent/.venv/Scripts/python.exe`, Python 3.11.2 | 3.22.1 | 3.22.1 | `7860ef71f54eb75eebf42d9ca7c410bbb87e6ac5` |
| Scripter backend | `G/skell-e-scripter/backend/venv/Scripts/python.exe`, Python 3.11.2 | 2.1.4 | 0.2.0 | `3be82c9c2ccb1a0295a92c72df03f251c8cd6b55` |
| Scripter Opus draft | `G/skell-e-scripter-drafts/claude-opus-4.5/backend/venv/Scripts/python.exe`, Python 3.11.2 | 1.2.1 | 0.1.0 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Gemini draft | `G/skell-e-scripter-drafts/gemini-3-pro/backend/venv/Scripts/python.exe`, Python 3.11.2 | 1.2.1 | 0.1.0 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Cursor draft | `G/skell-e-scripter-drafts/skell-e-scripter-cursor-claude/backend/venv/Scripts/python.exe`, Python 3.11.2 | 1.2.1 | 0.1.0 | `ba96b8129cbe98f52377823a2cee9fa7eda721db` |
| Scripter Kilo draft | `G/skell-e-scripter-drafts/skell-e-scripter-kilo-claude/backend/venv/Scripts/python.exe`, Python 3.11.2 | 2.0.2 | 0.2.0 | `9dfecee4e4e08012df46534352168a85fe49ebd8` |

Nine older copies have mismatched distribution and code version labels. This inventory did not establish the cause or alter them. Their later upgrade owners must verify artifact contents and both version labels rather than accepting a version string alone. The shared 3.26.2 installation is consistent.

The shared package's source `__version__` assignment also reads 3.26.2. Shared dependency versions are LiteLLM 1.83.14, OpenAI 2.24.0, Anthropic 0.116.0, google-genai 2.20.0, tenacity 9.1.4 and requests 2.34.2. Preserve these versions during the security-only rollout so an SDK upgrade does not confound the validation.

The security owner's `G/skell-e-router-security/.security-venv` is different. Its site-packages contains no router distribution. `scripts/run_security_tests_offline.py` inserts that worktree's source directory into `sys.path`. This is an isolated dependency environment testing task-local router source, not a repaired shared installation. The main checkout was at `7ffefbd55d361d765ff73baa9a58cbc51c810639` when inspected. The security worktree was at `3e0f8dd35925cb18cd369420a7eca85f854b627c`, with its runner being edited concurrently. These are observations, not release pins.

## Shared and unresolved launch routes

`M` in the validation column means the read-only `Test-RouterResolution` command defined below, using the specified interpreter and entry directory. `H`, `Q` and `D` are future offline behavior checks listed after it. None of the behavior suites was executed for this inventory.

| Consumer entry point | Interpreter and router resolution | Active or scheduled status | Owner | Safe offline validation | Coordination before shared rollout |
| --- | --- | --- | --- | --- | --- |
| Houston deep search: `G/claude-orchestrator/scripts/session-digest/search.py query` | `houston/app/src/main/vectorSearch.ts:122,125,145` defaults to PATH `python`; current-shell resolution is `S`, installed 3.26.2. Helper import is `search.py:35`. | On demand. Installed Houston processes were present. No live query or child runtime capture was performed. | Search integration `lead-3e0ce7b9-aad7-48ab-bacb-9d8c8966dcb9`. | `M S G/claude-orchestrator/scripts/session-digest`; then `Q`. | Confirm the installed app's launch route and inherited resolution. Preserve the independent local containment repair; it does not wait for router rollout. |
| Agent search shim: `G/claude-orchestrator/scripts/search-sessions.mjs` | Lines 24,27,51 select Houston's `searchCli.ts` through tsx. Deep search reaches the Python route above. Raw search alone is not proof of router use. | On demand; actual use not exercised. `HOUSTON_ROOT` can select another checkout. | Same search integration owner. | Same metadata command as deep search. | Confirm the selected Houston checkout without dumping environment contents. No live `--deep` validation. |
| Houston TLDR: `G/claude-orchestrator/scripts/houston-lead-tldr.py` | `houston/app/src/main/tldrService.ts:161,293,296` defaults to PATH `python`; current-shell resolution is `S`, installed 3.26.2. Helper lines 25,50,58,61 require router at least 3.26.2, clamp to 600 and call `ask_ai`. | On demand in installed Houston; app is present, helper execution not observed. | Houston integration lead not identified; route through orchestrator. Router contract owner is the security lead below. | `M S G/claude-orchestrator/scripts`; then `H`. | Confirm installed app provenance and parent resolution. Recheck the actual helper after any helper change. Keep model selection and token cap unchanged. |
| Automatic digest vector rebuild: `G/claude-orchestrator/scripts/session-digest/search.py build` | `relay/src/digest-queue.js:102,219` defaults to `python`; `relay/src/relay.js:801` does not override it. Current-shell resolution is `S`, installed 3.26.2. | Source schedules `0 5 * * *` at `scheduler.js:184`, plus lead-close and orchestrator-rotation triggers. Task Scheduler's `claude-orchestrator-relay` was Running. Actual queue state and last successful execution were not read. | Orchestrator/relay owner. | Same `M` as search; `D` for queue behavior; `Q` for helper boundary. | Choose a quiet window that covers event-triggered jobs as well as the daily sweep. Confirm the relay parent's resolution. Do not run a live build to validate. |
| Solar input qualification: module fixtures and rerun scripts in `G/solar-sailer-input-qualification-20260905` | `docs/module-input-qualification.md:104,110` and `benchmarks/module-input-qualification/p1-filler-reruns.md:44` explicitly use `S`, installed 3.26.2. No colocated server `.venv` was found. | Active qualification work is recorded in `docs/TASKS.md:14-18`; specific running fixture not inferred. | `lead-30949f07-8c13-4173-9d6f-7ac77b16f685`. | `M S G/solar-sailer-input-qualification-20260905/editor/server`; owner selects existing fully stubbed module tests. | Coordinate a quiet test window and record before/after dependency identity. The owning document explicitly requires preserving shared Python. Avoid npm postinstall, which can upgrade Python dependencies. |
| Solar Rough Cut experiments: `benchmarks/roughcut/roughcut_bench/experiments.py` | `benchmarks/roughcut/README.md:269` documents bare `python -m roughcut_bench.experiments`; `experiments.py:100` uses router model lookup. Bare python maps to `S` here, but the actual experiment interpreter is unresolved. | Manual benchmark route; no active or scheduled run established. | Solar benchmark owner not identified. | `M` with the owner's actual interpreter and Rough Cut directory. AST-only source check is available below. | Confirm whether any active run uses shared Python. Do not launch an experiment or replay script as an assumed offline check. |
| Benchmark CLI and API: `G/benchmark/main.py`, `web_api/main.py` | `benchmark/CLAUDE.md:14,29` documents bare `python`; current shell gives `S`/3.26.2. `env/Scripts/python.exe` separately has 2.4.0. `requirements.txt:9` tracks router main; `models/skelle_router_model.py:3` imports it. | Manual CLI/API routes documented; running or scheduled use unresolved. | Benchmark owner not identified. | `M` for both `S` and the benchmark env, with `G/benchmark` as entry directory. | Establish which interpreter the real launcher activates. API child jobs may inherit its interpreter. No CLI benchmark or uvicorn startup for validation. |
| Crawler comparison: `G/proko-second-brain/complex-version/scripts/model-comparison-2/run_crawlers.py` | README line 11 documents bare `python`; script line 22 imports router. No local `.venv` found at repository or complex-version root. Actual interpreter/version unresolved; `S` is only a candidate. | Manual experiment, activity unknown. | Project owner not identified. | `M` only after establishing the intended interpreter; AST-only check meanwhile. | Script line 24 still names the old `proko-knowledge-base` data/output root. Resolve launcher and path ownership before any later execution. |
| Manual router sample: `G/misc/test router/test.py` | Line 1 imports router; line 7 guards a main routine that makes model calls. No interpreter launcher or local `.venv` established. | Activity unknown. | Owner not identified. | AST-only check; then `M` after interpreter identification. | Count as an unresolved manual consumer. Never execute the sample as a health check. |

Digest summarization and markdown index generation are not additional Python router installations. The queue launches a summarizer through `runClaudeOneShot` and `write-index.mjs` through Node, then launches the Python vector rebuild. Houston's `crons/jobs.json` contains documentation jobs; no session-digest job was identified there. The relay's internal scheduler is the evidenced digest route. No task or cron was enabled, disabled or changed.

## Independent runtimes and task copies

| Consumer entry point | Interpreter and router resolution | Status and owner | Safe offline validation | Coordination |
| --- | --- | --- | --- | --- |
| Solar editor modules under `editor/server/modules`, including Rough Cut, retakes and transcript review | Default development candidate is the Solar development environment above, router 3.23.1. `editor/electron/pythonManager.ts:227` tries `PROKO_PYTHON`, bundled runtime, colocated `.venv`, then PATH; spawn is at line 392. Router dependency is `editor/server/requirements.txt:22`. | Project active per briefing; particular editor instance and override unknown. Solar owner to be routed. | `M` with its explicit venv and `G/solar-sailer/editor/server`. | Shared update does not repair this venv. Only a confirmed override to `S` creates a shared-rollout dependency. Server module imports can start work, so use owner-selected stubbed tests later. |
| Installed Solar desktop server under `P/Solar Sailer/resources/server` | Embedded runtime above contains router 3.23.1. Default packaged launcher selects `resources/python`; active override not inspected. | Installed artifact exists. Running app/runtime not established. | `M` with the installed embedded interpreter and resources/server directory. | Separate packaged upgrade and validation. Do not mistake staging 3.24.1 for the installed 3.23.1 package. |
| Solar staging and unpacked build servers | Both embedded artifact runtimes contain router 3.24.1. `editor/scripts/build-installer.mjs:11` documents staging. | Build artifacts, not proof of active installs. | `M` for each artifact with its corresponding server directory. | Rebuild and validate separately if included in a later release. No shared installation effect. |
| `G/skell-e-agent/skell_e_agent/bridge.py` | Its `.venv` has router 3.22.1. Import at line 19; `pyproject.toml:9` tracks router main. Actual product caller may choose a different environment. | Library and test environment; current product use unknown. | `M` with agent `.venv` and repository root. | Identify consuming products before a separate library-environment upgrade. README minimum 3.22.1 is a compatibility requirement, not proof of security mitigation. |
| Scripter backend `G/skell-e-scripter/backend/main.py` | README lines 22-25 activate backend venv before `python main.py`; that venv has router 2.1.4. Requirements line 8 tracks main. | Documented server launcher, activity and owner unknown. | `M` with backend venv and backend directory. | Separate upgrade. Do not start the server to test metadata. |
| Scripter drafts: `claude-opus-4.5`, `gemini-3-pro`, `skell-e-scripter-cursor-claude`, `skell-e-scripter-kilo-claude` backend entry points | Separate venv versions are 1.2.1, 1.2.1, 1.2.1 and 2.0.2 respectively. Opus README lines 37-49, Cursor lines 40-56 and Kilo lines 22-25 document activation. Gemini service imports router, but its concrete launcher was not established. | Existing environments; activity and owners unknown. | `M` once per explicit draft interpreter with that draft's backend directory. | Treat each as independent. Establish continued use before scheduling further upgrades; do not wake parked work for this inventory. |
| Router security tests in `G/skell-e-router-security` | `.security-venv` supplies copied dependencies; runner inserts local source. No installed router distribution in that venv. | Active repair, owned by `lead-1032891d-b8b5-4bbc-a05e-37284eaef9e3`. | Inspect source/metadata only here; source security commands below are for the owner. | Preserve owner's files and environment. Source tests are not installed mitigation evidence. |
| Router main checkout and parked model work | Using `S` from the router source directory locates `G/skell-e-router/skell_e_router/__init__.py`; package metadata still says shared 3.26.2. | Shared checkout; parked model work remains untouched. | `M S G/skell-e-router` demonstrates shadowing. | Never use a test from this directory alone to certify the installed package. Do not edit source, version files or release notes during inventory. |

Solar's legacy `main.py`/`modules` pipeline and `benchmark - Copy` were inspected as candidates but not established as router consumers. The former has direct SDK imports and an `E-MITRY.bat` launcher targeting an old `ProkoKAT` Conda environment with a bare-python fallback. The latter declares direct provider SDKs and had no router reference in inspected requirements, models or main code. `skell-e-scripter-main` and the `gpt-5.2-high`/`skell-e-scripter-custom-css` drafts had documentation or templates without a backend runtime in the inspected locations. These observations do not certify all historical or remote copies.

## Activity gaps that matter to rollout

The installed Houston executable was running at `P/houston/Houston.exe`. Task Scheduler reported `claude-orchestrator-relay` as Running, executable `powershell.exe`, working directory `G/claude-orchestrator/relay`. These establish active parent applications, not their child Python identity or current job state.

A persistent Python process, PID 23572, reported executable `S`. Its recorded parent PID 31460 no longer existed when queried. Without reading command arguments or importing into that process, its project, effective virtual environment and already-imported router version remain unresolved. Re-identify it at rollout time through the owning lead or an approved existing status source; do not stop it by process name. During a later snapshot, the security venv also had a child process whose executable was `S`. That demonstrates why a base executable path alone does not prove use of shared site-packages.

The inventory does not require waiting for answers from other leads. The following are preconditions for the later rollout owner: identify or explicitly exclude active ambiguous consumers, verify parent launch resolution, coordinate shared qualification work, and record the final reviewed release artifact. Independent installations can have separate upgrade tasks without being called fixed by the shared update.

## Read-only reproduction commands

### M: Resolve a selected interpreter without importing the consumer

Run this PowerShell definition once. The probe emulates the entry directory's import precedence, reads distribution metadata and parses the package's literal version assignment. It prints no credentials, environment values or provider URLs. A result from the shell is still not a capture from an already-running parent.

```powershell
$routerMetadataProbe = @'
import ast, json, pathlib, sys
import importlib.metadata as metadata
from importlib.machinery import PathFinder

sys.path[0] = str(pathlib.Path(sys.argv[1]).resolve())
spec = PathFinder.find_spec('skell_e_router', sys.path)
result = {'python': sys.executable, 'python_version': sys.version.split()[0],
          'entry_directory': sys.path[0], 'router_origin': spec.origin if spec else None}
if spec and spec.origin:
    tree = ast.parse(pathlib.Path(spec.origin).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            if isinstance(node.value, ast.Constant):
                result['source_version'] = node.value.value
try:
    dist = metadata.distribution('skell-e-router')
    source = json.loads(dist.read_text('direct_url.json') or '{}')
    result.update(distribution_version=dist.version,
                  distribution_root=str(dist.locate_file('')),
                  vcs_commit=source.get('vcs_info', {}).get('commit_id'),
                  editable=source.get('dir_info', {}).get('editable', False))
except metadata.PackageNotFoundError:
    result['distribution_version'] = None
print(json.dumps(result, indent=2))
'@
function Test-RouterResolution {
    param([string]$Python, [string]$EntryDirectory)
    $routerMetadataProbe | & $Python -B - $EntryDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Router metadata probe failed' }
}
$sharedPython = 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe'
$githubRoot = 'C:/Users/Stan/Documents/GitHub'
Test-RouterResolution $sharedPython "$githubRoot/claude-orchestrator/scripts/session-digest"
Test-RouterResolution $sharedPython "$githubRoot/claude-orchestrator/scripts"
Test-RouterResolution "$githubRoot/solar-sailer/editor/server/.venv/Scripts/python.exe" "$githubRoot/solar-sailer/editor/server"
Test-RouterResolution "$githubRoot/benchmark/env/Scripts/python.exe" "$githubRoot/benchmark"
```

For an untrusted or unknown startup configuration, use a static inspection instead: run Python with `-S -B`, enumerate `importlib.metadata.distributions(path=[explicit_site_packages])`, and use `PathFinder.find_spec('skell_e_router', [explicit_site_packages])`. This bypasses `.pth` and site hooks. Label the result as installed files at that location, not effective application resolution.

For a source entry point whose interpreter is unresolved, this syntax check is offline and does not import it. Passing means only that Python can parse the file.

```powershell
& $sharedPython -S -B -c "import ast,pathlib; p=pathlib.Path('C:/Users/Stan/Documents/GitHub/misc/test router/test.py'); ast.parse(p.read_text(encoding='utf-8-sig')); print('syntax OK')"
```

## Offline behavior checks for the rollout owner

These are planned checks. They may create isolated test/cache files and execute guarded library code, so they were not run during this read-only inventory. Do not substitute live requests or whole-application startup.

### Security checks and the installed-package requirement

The repair owner's source instructions are in `G/skell-e-router-security/docs/credential-error-security.md`. At inspection time they specified:

```powershell
Set-Location -LiteralPath 'C:/Users/Stan/Documents/GitHub/skell-e-router-security'
& ./.security-venv/Scripts/python.exe -I -B scripts/run_security_tests_offline.py --probe
& ./.security-venv/Scripts/python.exe -I -B scripts/run_security_tests_offline.py tests -q -p no:cacheprovider
```

The runner clears its environment, uses synthetic credentials and an isolated profile, and denies network connections except the standard library's internal Windows socket pair. It inserts `ROOT` into `sys.path` and changes to `ROOT`. Therefore these commands validate source even with `-I`. Its `--probe` prints containment booleans; the inspected implementation did not assert that every boolean was false. A process exit of zero is insufficient by itself.

Before installed validation, the security owner must supply or approve a variant that selects the installed wheel/package without adding a router checkout to the import path. This is a concrete validation dependency, not an instruction to edit the owner's runner during inventory. The variant must assert the exact imported path, source version and matching distribution version before testing. Test the reviewed wheel in an isolated environment first, then repeat against `S` after the coordinated installation.

Required synthetic cases and pass criteria:

1. Stub embedding transport to raise an exception containing a synthetic key. Repeat with a NUL-containing synthetic key passed to real local h11 Authorization-header validation. Both cases must reach the intended transport and raise the expected router operation error.
2. Assert that the marker and its raw, repr, JSON and URL-escaped variants are absent from the public message, error details, formatted traceback, stdout, stderr and captured router logs. Assert the original provider exception is absent from both `__cause__` and `__context__`; include nested causes, notes and hostile router-typed provider errors.
3. Verify allowed category/status fields and unchanged validation/error codes. Include successful embedding shape and the existing output-limit regression so privacy repair does not silently change successful caller contracts.
4. Include deferred streaming failures and directly affected chat, upload and deep-research boundaries from `tests/test_credential_errors.py`. Stub transports before SDK imports, use only synthetic inputs, and keep networking denied. Standard traceback checks must not rely solely on suppression of displayed exception chains.
5. Record candidate commit or wheel digest, Python/dependency identity, imported package path, test totals and containment assertions. Source and installed evidence must have separate labels. Do not inspect real credential values or log real provider failures to reproduce this issue.

### H: Houston's existing output-limit check

The existing router script executes the real TLDR helper with synthetic input, intercepts HTTP serialization, and blocks external networking. It checks the configured model, exactly one request, `max_completion_tokens=600`, helper success, matching router metadata/code versions, and an import path outside this router checkout. Omit `--output` to avoid replacing the owner's existing JSON evidence.

```powershell
& $sharedPython -I -B 'C:/Users/Stan/Documents/GitHub/skell-e-router/scripts/verify_houston_output_limit.py' 'C:/Users/Stan/Documents/GitHub/claude-orchestrator'
```

Existing contract and prior release evidence are in [Houston output limit](houston-output-limit.md). The prior 3.26.2 success is historical evidence, not a fresh check of the security release. Record helper/model-file hashes again. The script permits loopback connections for asyncio, so the security owner's stricter socket-pair guard is the stronger pattern for any new probe.

Houston also has a local output-length test at `app/src/main/tldrCore.test.ts:338` and service tests using a fake spawner. Run the selected tests from `G/houston/app` after checking the current files still use fixtures:

```powershell
node node_modules/vitest/vitest.mjs run src/main/tldrCore.test.ts src/main/tldrService.test.ts
```

### Q: Search boundary checks

The integration owner selects the integrated version of `scripts/session-digest/test/search-credential.test.mjs`. The search-security worktree contains an expanded synthetic suite with temporary profile fixtures and provider-failure cases; main's file at inspection time was older. From the selected orchestrator checkout, the existing entry command is:

```powershell
node --test scripts/session-digest/test/search-credential.test.mjs
```

The integration owner must confirm the selected version uses synthetic keys, isolated profile paths, stubbed router/provider calls and blocked networking before execution. Validate missing and malformed credentials, provider failures on both `build` and `query`, fixed safe stderr, no traceback or secret-bearing diagnostic, and the application's bounded-output behavior. Do not load the real index, rebuild vectors or issue a live query. This suite validates local containment independently of the router release; it does not replace the installed-router synthetic test.

### D: Digest queue checks

`relay/test/digest-queue.test.js` injects fake command execution and summarization and uses temporary repositories. It verifies routing and queue behavior without running the real rebuild:

```powershell
Set-Location -LiteralPath 'C:/Users/Stan/Documents/GitHub/claude-orchestrator'
node --test relay/test/digest-queue.test.js
```

This does not establish the running relay's interpreter or validate an embedding. Pair it with `M` from the real launch context and the installed security probe. Do not enqueue a sweep as an offline smoke test.

## Rollout sequence and completion criteria

1. The orchestrator routes this inventory to security owner `lead-1032891d-b8b5-4bbc-a05e-37284eaef9e3`, search integration owner `lead-3e0ce7b9-aad7-48ab-bacb-9d8c8966dcb9`, qualification owner `lead-30949f07-8c13-4173-9d6f-7ac77b16f685`, and the active Houston owner. It assigns the ambiguous benchmark and persistent-process checks without waking parked leads or remote machines.
2. Before any shared-environment mutation, establish which active launches use `S`. Record executable, package origin/version and source revision through a safe metadata-only child in the owner's actual launch context, or record why that consumer is excluded. Confirm PID 23572's current identity or that it has ended naturally. An unattributed running process cannot be declared compatible.
3. Coordinate a quiet interval for shared consumers and qualification tests. Cover interactive helper launches and digest triggers, including lead closure and rotation. Existing Python processes retain imported modules in memory; identify the particular processes needing a later restart. The Node/Electron parents generally launch fresh Python children and do not need a blanket restart merely because router files change.
4. Obtain the reviewed immutable repair commit or matching wheel, full source-test result and independent review result from the security owner. Confirm the wheel contains the intended source and versions. Validate that artifact in an isolated environment with the same dependency versions. Run synthetic security checks and H, then relevant Q/D and owner-selected Solar/benchmark fixture tests. Do not use an unqualified main-branch tip as the release identity.
5. Only the coordinated rollout owner changes the shared package. Install the exact accepted artifact with dependency upgrades disabled, preserving model pins and consumer configs. Keep the prior exact package artifact and dependency identity available for recovery. This document does not authorize or execute that installation.
6. Repeat M outside all router source directories and assert the expected installed path, version and revision. Run the installed variant of the synthetic security checks, H, and the agreed local consumer checks. Confirm dependency versions did not change. Restart only the identified persistent Python consumers through their owning project and repeat their offline checks. Never kill all Python, Electron, Houston or relay processes.
7. Before normal shared jobs resume, require passing installed containment and output-limit checks, consumer runtime attribution, and recorded per-consumer results. If validation fails, keep affected launches held through their owner. Any reversion to 3.26.2 must be recorded as restoring the known affected package, not as security mitigation.
8. Record source completion and installed mitigation as separate milestones in the security owner's existing task/release record. Route independent venv and embedded-copy upgrades as separate work after establishing active use. This inventory makes no claim that those older versions are fixed or compatible, and the shared rollout does not silently upgrade them.

## Inventory verification record

Verified against local files and metadata: shared interpreter/version/revision, helper and queue launcher defaults, qualification's explicit Python path, independent environment package identities, Solar's installed-versus-staged difference, router checkout shadowing, the source-only security runner, scheduled relay presence and the sanitized process observations above. The parent investigator corroborated the delegated version results through direct metadata reads.

Not executed: security suites, Houston output-limit capture, search tests, digest tests, Solar tests, benchmark jobs, application startups or installed-environment changes. The listed commands are a rollout-validation sequence, not passing results. The final document's metadata recipe was exercised separately without imports or filesystem writes; path references and Markdown formatting were checked before committing only this document.
