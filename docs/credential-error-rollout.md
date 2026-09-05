This packet is for developers executing and coordinating the installed router security update.

# Installed credential-error repair

## Release identity and current hold

Execution owner is router lead `lead-1032891d-b8b5-4bbc-a05e-37284eaef9e3`. The orchestrator coordinates and releases the shared update. Installation remains held. Passing a clock deadline alone does not release it.

Use package 3.26.3 from reviewed source `63b5fd22bacef9100b09cdee355bd8839439be78`. The accepted wheel is `C:/Users/Stan/Documents/GitHub/skell-e-router-security/dist/skell_e_router-3.26.3-py3-none-any.whl`, SHA-256 `f7f21d1e30dcad7a4d46bc57ff87eeed7cb1255617fce876546fee7f23e01707`. Reuse the [source, wheel and independent-review evidence](credential-error-security.md). Both source and wheel passed 781 tests; the second fresh Astra review was clean. The actual Houston helper already passed against the isolated wheel. No new source repair or review round is part of rollout.

The target is `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe`, with package files in its `Lib/site-packages`. At 11:58 PDT on 2026-09-05, installed router was still 3.26.2 at `d8ae9876fd2f095d5e6e03e11710c6d7a8ddcefe`. [Before snapshot](credential-error-installed-before.json) records all 407 distribution versions and both wheel hashes without importing router or reading credentials.

The [SDK baseline](credential-error-sdk-baseline.json) additionally records 89 user-site distributions. System packages contain Anthropic 0.84.0 and requests 2.32.5, unchanged from the before snapshot. Normal startup selects user-site Anthropic 0.116.0 and requests 2.34.2. Both reports were correct for different paths. OpenAI 2.24.0 exists in both locations and normal startup selects the user copy. Preserve both locations. The runtime probe now reports selected SDK versions and roots without importing them. The candidate wheel's 115 security/output-limit cases also pass under normal startup with user-site priority; this supplements, rather than replaces, the completed isolated source/wheel evidence.

## Concrete gates

| Gate | Current evidence | Release requirement |
| --- | --- | --- |
| Accepted artifact | Reviewed source, wheel hash and completed tests above | Recheck wheel hash immediately before installation. |
| Solar qualification window | Cleared by the orchestrator. Original owner, security fixer and P2 explicitly confirm no remaining shared-Python tests/workers/imports. P2 continues only in its self-contained environment. | Keep shared-Python launches paused through installed checks. PID 21532 was absent when checked. |
| Actual Houston and relay parent resolution | Houston owner observed Python PID 22124 under Houston PID 1004 using the shared Python executable. Package metadata came from a separate probe, not the original child. The controlled relaunch boundary is accepted by orchestrator. Relay startup can override inherited settings. | Houston owner performs its controlled shutdown and accepted relaunch sequence. Relay runtime owner supplies its post-configuration metadata capture and enforces the launch pause/drain. No consumer is excluded merely by a note. |
| Persistent old imports | The 12:00 PDT process snapshot contains only Python PID 23572. Orchestrator identifies it as the connectivity sampler, not a router consumer. No persistent qualification Python worker was observed. | Recheck Python processes at the released boundary. Attribute any new consumer and have its owner finish or hold it. Never terminate by process name. |
| Digest triggers and interactive launches | Daily 05:00 rebuild plus lead-close/rotation triggers exist. No queue hold or drain was performed by this owner. | Relay and Houston owners confirm no new Python consumer launches during the brief package replacement and installed checks. |
| Houston activation | Network/time gate cleared by orchestrator at 12:06 PDT. This owner subsequently confirmed `claude-orchestrator/docs/stanstudio-connectivity-2026-09-05/summary.json` exists. Consolidated activation belongs to TLDR release owner `lead-7c24d887-559d-4e3c-b257-a86800ebc1f3`. | Router/qualification coordination remains. TLDR owner activates its integrated release after installed router checks, using the deterministic relaunch boundary below. |

PID 23572 was left untouched. The 12:12 PDT process snapshot showed no `python.exe` or `pythonw.exe` process; the orchestrator has cleared the sampler/network gate. The [inventory](shared-python-router-consumers-2026-09-05.md) remains the wider consumer record. Its earlier unknown-PID description is superseded by the orchestrator's positive sampler attribution.

The [older-copy reconciliation](isolated-router-credential-exposure.md) separately verifies 11 isolated installations and their nine stale version-label conflicts. It reports unsafe static error handling and unresolved upgrade ownership. Those copies remain outside this shared installation.

## Capture from an actual parent

The [runtime probe](../scripts/probe_router_runtime.py) prints only process IDs, interpreter identity, selected package path and version/source metadata. It parses the router's version without importing router or application code. It clears inherited environment variables before its metadata work. Normal Python startup retains the parent's actual import-path selection; no credential values are printed.

The consumer owner must launch it through the parent's normal Python spawning mechanism, with the same working directory and environment policy as its helper. Do not run an application request, embedding, TLDR generation or index rebuild as a diagnostic. The following arguments are a packet for the owner to pass to that existing spawn mechanism, not a claim that a shell can impersonate the running parent:

```text
python -B C:/Users/Stan/Documents/GitHub/skell-e-router/scripts/probe_router_runtime.py --entry-directory C:/Users/Stan/Documents/GitHub/claude-orchestrator/scripts --expected-site C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages --expected-version 3.26.2 --expected-parent 1004
```

Use `scripts/session-digest` as the entry directory for search and vector rebuilds. Use the actual relay parent PID for relay capture, and the new Houston parent PID if capture happens after consolidated activation. Change expected version to 3.26.3 after the package update. Record the returned JSON and recheck PID creation times so a reused PID cannot pass as the earlier parent. A new parent's result supersedes the old parent's unresolved state; it does not prove the old environment.

A shell smoke check of this probe passed against 3.26.2. Its parent was the test shell, so it is not accepted as Houston or relay attribution. No process environment block, process memory, secret-bearing command line or application transcript was read. No supported Python diagnostic IPC was present in the inspected Houston routes; relay run-host controls report agent liveness, not this runtime identity. Do not add an improvised process-injection mechanism to bypass that gap.

## Deterministic Houston relaunch boundary

The selected alternative is a verified new launch context. Old PID 1004's Start Menu/tray ancestry and current registry PATH remain inference, not an inside-process capture. Its owner must gracefully close that instance during the coordinated activation. Do not let Electron forward the new launch to the old single instance.

After installed checks pass, the TLDR activation owner uses its existing approved launch shell, prefixes only that shell's PATH with shared Python, and runs the following Node preflight. Preserve all other inherited settings without printing them. Use the expected version 3.26.3 for activation. The [preflight capture](credential-error-controlled-launch-preflight.json) already proves this mechanism selects shared Python and its current 3.26.2 files in a fresh test launcher; no application was started during that smoke check.

```powershell
$env:Path = 'C:/Users/Stan/AppData/Local/Programs/Python/Python311;C:/Users/Stan/AppData/Local/Programs/Python/Python311/Scripts;' + $env:Path
@'
const { spawnSync } = require('node:child_process');
const root = 'C:/Users/Stan/Documents/GitHub/claude-orchestrator';
for (const entry of ['scripts', 'scripts/session-digest']) {
  const result = spawnSync('python', ['-B', 'C:/Users/Stan/Documents/GitHub/skell-e-router/scripts/probe_router_runtime.py', '--entry-directory', root + '/' + entry, '--expected-site', 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages', '--expected-version', '3.26.3'], { cwd: root, windowsHide: true, encoding: 'utf8' });
  if (result.status !== 0) throw new Error('Router launch preflight failed');
  process.stdout.write(result.stdout);
}
'@ | & 'C:/nvm4w/nodejs/node.exe' -
if ($LASTEXITCODE -ne 0) { throw 'Do not launch Houston from this context' }
```

The owner then launches the newly installed Houston executable directly from that same shell and environment, with `Start-Process -WindowStyle Hidden -WorkingDirectory 'C:/Users/Stan/Documents/GitHub/claude-orchestrator' -FilePath 'C:/Users/Stan/AppData/Local/Programs/houston/Houston.exe' -PassThru`. Record the new main PID, start time and installed artifact identity. Do not use Start Menu, an installer auto-launch or another launcher between the preflight and this start, because those may use a different parent environment. Confirm old PID 1004 has exited and the replacement PID is new.

The Houston owner has prepared this sequence in `houston/scripts/router-capture/controlled-relaunch.ps1`, documented at Houston source `f2bbd68` and main `40773b8`. Its separate [capture note](../../houston/docs/router-consumer-capture-houston-2026-09-05.md) records a live deep-search child at 12:17 PDT: Python PID 22124 under Houston PID 1004 used the shared Python311 executable. The subsequent metadata probe reported router 3.26.2 at `d8ae9876`, but did not run inside that original child. TLDR executable selection remains an inference from its matching source launcher. This was owner-performed live evidence, not this lead's offline test. The orchestrator stopped further diagnostic live queries, so do not rerun `capture.ps1` as an offline check. Post-install live query acceptance remains the orchestrator's single coordinated check.

Acceptance also requires the activation owner to confirm the shipped main-process code still spawns bare `python`, passes the orchestrator working directory, supplies no Python override, and does not change PATH before these helpers run. The inspected source satisfies those conditions: `index.ts` supplies no Python override, and `tldrService.ts`/`vectorSearch.ts` launch the documented command. No `process.env.PATH` or `process.env.Path` assignment was found under `app/src/main`. No `python.exe` exists in the inspected orchestrator, Houston installation or Node executable directories to shadow the chosen PATH entry. The owner repeats the relevant identity checks if packaging changes those assumptions.

This boundary establishes controlled inheritance for the replacement parent. It does not claim a diagnostic was executed inside PID 1004 or the replacement Houston process. If the owner changes the launch path or startup code, require an actual-parent capture instead.

Relay is a separate boundary. `relay/src/config.js:65` loads its configuration with `override: true`; this owner did not inspect credential configuration to infer whether it changes PATH. A verified shell plus Task Scheduler start is therefore insufficient. The relay owner must run the metadata-only child through the normal Python spawn configuration after that startup configuration is loaded, or explicitly keep digest launches excluded until a coordinated restart can provide that capture. The existing wrapper is `scripts/relay-loop.ps1` with an absolute `-NodeExe`; do not bypass its restart supervision or start a duplicate relay to obtain evidence. No relay restart is required merely because the router distribution changes. An explicit digest hold can isolate this unresolved runtime while the shared package and Houston are validated.

## Replacement barrier and release order

The package command is not a lock on consumer launches. A written exclusion alone provides no protection. Before the orchestrator releases installation, the Houston owner must have actually stopped interactive helper production, and the relay activation owner must have actually stopped digest submissions and drained or ended existing work through its approved maintenance controls. A queue described as idle without disabling its lead-close, rotation and scheduled triggers is insufficient. If an owner uses a controlled parent stop, that owner must also prevent its supervisor from immediately respawning it. This packet does not authorize stopping any parent or scheduled task.

Relay runtime/activation owner is `lead-9d13b0f7-68cc-4eb5-8bdf-825ff0a98395`. It is preparing the enforced pause and post-configuration metadata capture with `scripts/probe_router_runtime.py`. The qualification and network gates are clear; the remaining install-release conditions are that enforced pause/drain and Houston's controlled shutdown boundary. The 12:38 PDT static check confirms the saved 407 system and 89 user-site distribution versions are unchanged, and no persistent Python process appeared in that snapshot.

Before the relay restart, the orchestrator records conditional authorization for the whole local sequence, its operation ID, named surviving router/relay operators, deadline and failure route. This owner must read that authorization before beginning. Once the relay is held, no new Slack reply is required: the execution owner proceeds only after matching local hold/drain acknowledgements and process checks satisfy the preauthorized conditions. It remains active through installation and local evidence handoff rather than parking on a Slack wake. Immediately before pip, it rechecks Python processes and both dependency locations. Any new unattributed worker or package drift stops the sequence before replacement.

Run the exact pip command synchronously and keep all producer holds active until the installed security/output-limit/helper/search gates finish. Do not terminate an unexpected consumer or interrupt pip mid-replacement. If a hold is breached during installation, keep the remaining holds, report the breach, and let the relevant owner quiesce its consumer before validation and release. Do not resume work based only on pip's exit status.

On passing gates, notify the orchestrator for its one real Codex deep-query acceptance. That authorized live acceptance belongs to the orchestrator; this owner performs only offline checks. Houston's release owner then completes the agreed controlled activation and Agents-page acceptance. The relay activation owner completes its post-configuration runtime capture before releasing digest production. Qualification owners resume only after the orchestrator ends their hold. No blanket process termination or restart occurs.

### Local evidence contract while Slack is offline

This owner accepts the preauthorized local sequencing in `claude-orchestrator/docs/relay-router-activation-2026-09-05.md`; it is prepared, not yet authorized. The relay operator owns `claude-orchestrator/state/router-window-ack.json`. Before restart, both operators must agree that exact path and the following fields in the authorization record: `operation_id`, `relay_pid`, `boot_id`, `operator_lead`, `houston_closed`, `old_digest_work_drained`, `python_launchers_held`, and `qualification_quiet`. The four confirmation fields must be true and supported by fresh process/owner evidence. The acknowledgement must match the current operation, PID and boot UUID in `state/router-maintenance-status.json` at phase `held`. This owner also checks those processes independently. A stale file or a marker written before the old relay restarts is not a hold.

The accepted verification dependencies are frozen in [the verification lock](credential-error-verification-lock.json). The probe SHA-256 is `24aa4560871b205bac657745f4cc6e6abb0d431e8c950fc939e2ced18ba687d9`. No probe, runner, search verifier, helper or baseline edit is planned before activation. Recheck the hashes and accepted wheel before the outage.

After all installed checks pass, this owner writes `docs/credential-error-installed-after.json` with `status="passed"`, the matching operation/PID/boot identity, test exit codes and totals, SDK preservation, package identity and evidence paths. The relay operator checks these fields and the helper/search results, then requests its own bound metadata verification and release under the already-recorded authorization. Receipt existence alone is insufficient. The real Codex query follows relay release; it cannot be a prerequisite for restoring the offline relay. Houston remains under its owner's controlled activation sequence.

The tested exact-file recovery below preserves the original 3.26.2 VCS identity and can use the relay's existing 3.26.2 abort/release branch without changing its hook. The cached 3.26.2 wheel still cannot do that because pip replaces its origin metadata. Before the outage, the orchestrator must explicitly authorize whether failure recovery may restore the exact backup and release known affected 3.26.2. Without that authorization, retain holds and restore/verify the accepted 3.26.3 artifact. A timeout never chooses recovery or release automatically.

## Completed consumer checks

[Captured outputs and source hashes](rollout-consumer-checks.json) record search 10/10 and digest queue 15/15, with no failures or skips. These ran against orchestrator `6c28b48b1a8aa2cec2ce18a361cd48fea1e8587e`. Search fix `540dbb4a65e70ac63c02c4990e8f120f1b40d4ae` and integration `7a4f73dee33b5387d8930dfb6df910c10afee3d1` are ancestors.

Both checks used sanitized subprocess environments, synthetic profiles and fixtures. Search replaced the router with a stub and blocked provider networking; digest injected fake execution and summarization. No consumer source hashes changed. These establish integrated local boundaries, not compatibility of the still-old installed router. Reuse them unless the relevant source hashes change before rollout; pair them with installed-package checks below.

## Install only after the orchestrator releases the gates

Start from the router repository. Confirm the before snapshot still describes installed dependencies. If another package changed, stop and reconcile the baseline rather than overwrite it. Preserve the exact SDK versions and all consumer model pins.

```powershell
$routerPython = 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe'
$routerWheel = 'C:/Users/Stan/Documents/GitHub/skell-e-router-security/dist/skell_e_router-3.26.3-py3-none-any.whl'
$routerWheelHash = 'f7f21d1e30dcad7a4d46bc57ff87eeed7cb1255617fce876546fee7f23e01707'
if ((Get-FileHash -LiteralPath $routerWheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $routerWheelHash) { throw 'Router wheel hash mismatch' }
@'
from importlib import metadata
from pathlib import Path
import json
site = Path(r'C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages')
baseline = json.loads(Path('docs/credential-error-sdk-baseline.json').read_text(encoding='utf-8'))
for label, record in baseline['sites'].items():
    current = {d.metadata['Name']: d.version for d in metadata.distributions(path=[record['path']])}
    assert current == record['packages'], 'Installed packages drifted in ' + label
print('Before snapshot still matches installed packages')
'@ | & $routerPython -I -S -B -
if ($LASTEXITCODE -ne 0) { throw 'Router preinstall inventory check failed' }
& $routerPython -I -m pip --isolated --disable-pip-version-check install --no-index --no-deps --force-reinstall $routerWheel
if ($LASTEXITCODE -ne 0) { throw 'Router installation failed' }
```

This changes only the router distribution and reads no provider configuration. `--no-index` forbids package-index access and `--no-deps` preserves SDK installations. Do not replace the exact wheel with a moving Git branch installation. Do not run npm postinstall, rebuild applications or restart processes as part of this command.

## Installed checks before consumer release

Use the confirmed shared interpreter with normal startup so user-site SDK priority matches the actual helpers. The following block first compares both dependency locations and every installed router file with the accepted wheel, then runs the 54 security cases plus 61 output-limit cases. `--require-installed` removes checkout shadowing without prepending the system SDK directory. It reuses the offline runner's cleared environment and socket guard for the actual Houston helper check. A failed gate stops the sequence.

```powershell
@'
from pathlib import Path
from importlib import metadata
from datetime import datetime, timezone
import hashlib, json, os, runpy, subprocess, sys, zipfile
repo = Path(r'C:/Users/Stan/Documents/GitHub/skell-e-router')
site = Path(r'C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages')
state = Path(r'C:/Users/Stan/Documents/GitHub/claude-orchestrator/state')
ack = json.loads((state / 'router-window-ack.json').read_text(encoding='utf-8-sig'))
held = json.loads((state / 'router-maintenance-status.json').read_text(encoding='utf-8-sig'))
assert held['phase'] == 'held'
assert all(ack[key] == held[key] for key in ('operation_id', 'relay_pid', 'boot_id'))
assert all(ack[key] is True for key in ('houston_closed', 'old_digest_work_drained', 'python_launchers_held', 'qualification_quiet'))
wheel = Path(r'C:/Users/Stan/Documents/GitHub/skell-e-router-security/dist/skell_e_router-3.26.3-py3-none-any.whl')
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == 'f7f21d1e30dcad7a4d46bc57ff87eeed7cb1255617fce876546fee7f23e01707'
baseline = json.loads((repo / 'docs/credential-error-sdk-baseline.json').read_text(encoding='utf-8'))
current = {d.metadata['Name']: d.version for d in metadata.distributions(path=[site])}
assert current['skell-e-router'] == '3.26.3'
for label, record in baseline['sites'].items():
    observed = {d.metadata['Name']: d.version for d in metadata.distributions(path=[record['path']])}
    expected = record['packages']
    if label == 'system':
        observed = {k:v for k,v in observed.items() if k != 'skell-e-router'}
        expected = {k:v for k,v in expected.items() if k != 'skell-e-router'}
    assert observed == expected, 'Dependency versions changed in ' + label
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.startswith('skell_e_router/'):
            assert (site / name).read_bytes() == archive.read(name), name
sys.argv = ['run_security_tests_offline.py', '--require-installed', str(site), 'tests/test_credential_errors.py', 'tests/test_output_limits.py', '-q']
try:
    runpy.run_path(str(repo / 'scripts/run_security_tests_offline.py'), run_name='__main__')
except SystemExit as result:
    if result.code != 0:
        raise
import skell_e_router
assert Path(skell_e_router.__file__).resolve() == site / 'skell_e_router/__init__.py'
assert skell_e_router.__version__ == current['skell-e-router']
sys.argv = ['verify_houston_output_limit.py', r'C:/Users/Stan/Documents/GitHub/claude-orchestrator', '--output', str(repo / 'docs/credential-error-houston-installed.json')]
runpy.run_path(str(repo / 'scripts/verify_houston_output_limit.py'), run_name='__main__')
system_command = [sys.executable, '-I', '-B', str(repo / 'scripts/run_security_tests_offline.py'), '--require-installed', str(site), 'tests/test_credential_errors.py', 'tests/test_output_limits.py', '-q']
system = subprocess.run(system_command, cwd=repo, capture_output=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
(repo / 'docs/credential-error-system-tests.log').write_bytes(system.stdout + system.stderr)
assert system.returncode == 0 and b'115 passed' in system.stdout, 'System-only tests failed'
search_env = dict(os.environ)  # Already cleared by the offline runner.
search_env['PYTHONUSERBASE'] = str(Path(baseline['sites']['user']['path']).parents[1])
search = subprocess.run([sys.executable, '-B', str(repo / 'scripts/verify_installed_search.py'), '--output', str(repo / 'docs/credential-error-search-after.json')], cwd=repo, env=search_env, capture_output=True, timeout=190, creationflags=subprocess.CREATE_NO_WINDOW)
assert search.returncode == 0, 'Installed search checks failed'
search_evidence = json.loads((repo / 'docs/credential-error-search-after.json').read_text(encoding='utf-8'))
assert search_evidence['passed'] and search_evidence['expected_version'] == '3.26.3'
assert all(case['passed'] for case in search_evidence['cases'])
receipt = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(), **{key: ack[key] for key in ('operation_id', 'relay_pid', 'boot_id')}, 'normal_security_output_tests': {'exit_code': 0, 'passed': 115}, 'system_security_output_tests': {'exit_code': system.returncode, 'passed': 115}, 'search_exit_code': search.returncode, 'sdk_versions_unchanged': True, 'packages': current, 'router_origin': skell_e_router.__file__, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'reviewed_source': '63b5fd22bacef9100b09cdee355bd8839439be78', 'helper_evidence': 'docs/credential-error-houston-installed.json', 'search_evidence': 'docs/credential-error-search-after.json'}
(repo / 'docs/credential-error-installed-after.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
'@ | & 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe' -B -
if ($LASTEXITCODE -ne 0) { throw 'Installed router verification failed' }
```

The selected installed router must resolve outside both source checkouts. The runner asserts it before testing, and the helper checks code/metadata version agreement. Do not prepend the entire system site-packages directory during normal-startup validation, because that would hide the user-site SDKs. The block also executes the 115 system-only cases and the actual installed-search negative pair before producing its bound passing receipt. Do not repeat them merely because the individual commands are documented below. Do not claim installed mitigation from the source-only command or `--probe` process status alone.

The [installed search verifier](../scripts/verify_installed_search.py) adds the real-router negative checks that the 10 stub-router tests cannot establish. Run `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe -B scripts/verify_installed_search.py --output docs/credential-error-search-after.json` after installation. Default expected version is 3.26.3. It executes the actual search.py CLI function with `query -n 3 --json -- anything`, the real installed router and real SDKs, a fresh child-only synthetic credential profile, disabled networking and a 90-second limit per child. Only VECTORS/CHUNKS point to a miniature synthetic index, so it never loads the real index. The source file remains unchanged.

The credential case must exit 1 with empty stdout and exactly one stderr line beginning `session-search: embedding provider failed`. The verifier removes only its own temporary synthetic key file, then requires the missing case's single `session-search: missing embedding credential` line. Neither case may emit a traceback or synthetic key. Metadata records exact interpreter, imported router path, code/metadata agreement, `get_embedding` signature and actual SDK origins. The [3.26.2 compatibility capture](credential-error-search-before.json) passes both cases under normal user-site priority. It is compatibility evidence, not proof of the router security fix; the post-install 3.26.3 capture and security tests are still required.

Repeat the parent metadata captures for released consumers and record the package archive hash from wheel installation. Recheck search/digest test-file hashes against the completed capture; rerun only changed consumer suites under the same synthetic setup. Qualification successors choose and run their agreed stubbed checks at their quiet boundaries. No live provider call, real index read or rebuild is needed.

## Failure and completion

If installation or validation fails, keep affected launches held and report the exact failing gate. Do not resume them merely because the package version changed. Recovery to 3.26.2 restores a known affected router and is an aborted mitigation, not security completion.

The exact backup is `C:/Users/Stan/Documents/GitHub/skell-e-router-security/dist/router-3.26.2-exact-backup-20260905`. Its `manifest.json` SHA-256 is `85410ee9080a10c3c858ef6c48294e678f9b1bd452a1a709954c4ef972cbe703`. It contains 35 package files, including existing bytecode, and six original distribution-metadata files. The original `direct_url.json` hash is `0cc1b95c939eed92d1ff83b394d6bb11fa7e11d87fee4247e3286a68660d4a0e`, retaining VCS commit `d8ae9876fd2f095d5e6e03e11710c6d7a8ddcefe`. The restore helper and backup identity are included in the verification lock.

[Recovery evidence](credential-error-exact-recovery.json) records a real offline pip upgrade to the accepted 3.26.3 wheel inside an isolated venv, followed by exact restoration of all 41 files. All 887 nonrouter files retained their before hash, `f1a23e1ba9dc7845cbad04bc4e570ac9af4a4f46e50c0cb42e9b0f9398f2b931`. The shared package's files still matched the backup afterward. Twenty-four synthetic regression tests and 17 subtests passed, including interrupted-install states, tampering, unsafe paths, unexpected metadata, symlinks and Windows junctions; none skipped. No shared installation changed.

A separate temporary-state test exercised the actual relay maintenance controller with its real metadata-only probe and the original shared VCS identity, which matches the isolated restored metadata byte for byte. It reached `released` for 3.26.2 without altering any metadata fields or live relay state. This proves acceptance of that identity; it is not a live-parent capture or shared restore.

Only if the orchestrator preauthorizes exact recovery and all consumer holds remain active, run:

```powershell
& 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe' -I -S -B scripts/router_package_recovery.py restore --site 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages' --backup-dir 'C:/Users/Stan/Documents/GitHub/skell-e-router-security/dist/router-3.26.2-exact-backup-20260905' --expected-sha256 85410ee9080a10c3c858ef6c48294e678f9b1bd452a1a709954c4ef972cbe703 --replace-version 3.26.3
if ($LASTEXITCODE -ne 0) { throw 'Exact router recovery failed; retain consumer holds' }
```

The helper verifies every backup file and all destination roots before mutation, rejects links and unexpected distributions, removes only the router package and the explicitly named old/new metadata directories, and restores original bytes and modification times. It never removes the site-packages directory or changes another distribution. `--replace-version 3.26.3` also permits interrupted-install states with missing metadata or both known metadata directories present. Any third router version or unsafe path still stops before mutation. Keep holds if validation fails; never bypass those guards.

After recovery, verify all 41 hashes, original metadata/code version and VCS identity, and both untouched SDK baselines. Record the operation-bound recovery result, then have the relay operator send `hold`, `verify` for 3.26.2, and a matching bound `release` only under the preauthorized abort branch. The 3.26.3 security tests are expected to reject a recovered 3.26.2 package. The cached old wheel in the earlier snapshot is retained for provenance; it is not this releasable recovery path.

The requested PID 21896 was absent in the fresh [process check](credential-error-process-recheck.json) at 13:05 PDT. It is not retained as a blocker. Re-enumerate processes at the actual installation boundary instead of relying on that expired PID.

Installed completion requires matching wheel/code/metadata identity, unchanged SDK versions, passing installed security/output-limit/helper checks, agreed consumer checks and owner-confirmed runtime/quiet-window gates. Record actual installed results in the existing security release document and mark only the installed milestone in `docs/TASKS.md` complete. Independent virtual environments and embedded copies remain separate work. External API spend to date is $0.
