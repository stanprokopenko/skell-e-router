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
| Solar qualification window | Original owner is quiet. Security successor explicitly confirmed all tests/helpers exited and no more shared-Python tests during the hold at 11:59:55 PDT. P2 has paused shared-Python launches and is restricted to a self-contained environment without shared-site-packages. | Orchestrator confirms P2 has no remaining shared worker and keeps these holds active through installed checks. PID 21532 was absent when checked. |
| Actual Houston and relay parent resolution | Old Houston PID 1004 remains unproven. The controlled fresh-Node preflight below passes for both helper entry directories. Relay startup can override inherited settings. | Use the deterministic Houston relaunch boundary below. Relay requires a metadata child after its startup configuration loads, or explicit exclusion with digest launches held until that capture. Do not block on reconstructing old Houston's environment. |
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

Acceptance also requires the activation owner to confirm the shipped main-process code still spawns bare `python`, passes the orchestrator working directory, supplies no Python override, and does not change PATH before these helpers run. The inspected source satisfies those conditions: `index.ts` supplies no Python override, and `tldrService.ts`/`vectorSearch.ts` launch the documented command. No `process.env.PATH` or `process.env.Path` assignment was found under `app/src/main`. No `python.exe` exists in the inspected orchestrator, Houston installation or Node executable directories to shadow the chosen PATH entry. The owner repeats the relevant identity checks if packaging changes those assumptions.

This boundary establishes controlled inheritance for the replacement parent. It does not claim a diagnostic was executed inside PID 1004 or the replacement Houston process. If the owner changes the launch path or startup code, require an actual-parent capture instead.

Relay is a separate boundary. `relay/src/config.js:65` loads its configuration with `override: true`; this owner did not inspect credential configuration to infer whether it changes PATH. A verified shell plus Task Scheduler start is therefore insufficient. The relay owner must run the metadata-only child through the normal Python spawn configuration after that startup configuration is loaded, or explicitly keep digest launches excluded until a coordinated restart can provide that capture. The existing wrapper is `scripts/relay-loop.ps1` with an absolute `-NodeExe`; do not bypass its restart supervision or start a duplicate relay to obtain evidence. No relay restart is required merely because the router distribution changes. An explicit digest hold can isolate this unresolved runtime while the shared package and Houston are validated.

## Replacement barrier and release order

The package command is not a lock on consumer launches. A written exclusion alone provides no protection. Before the orchestrator releases installation, the Houston owner must have actually stopped interactive helper production, and the relay activation owner must have actually stopped digest submissions and drained or ended existing work through its approved maintenance controls. A queue described as idle without disabling its lead-close, rotation and scheduled triggers is insufficient. If an owner uses a controlled parent stop, that owner must also prevent its supervisor from immediately respawning it. This packet does not authorize stopping any parent or scheduled task.

The orchestrator records the active hold acknowledgements and their start time, confirms qualification workers have exited and stay paused, then sends the execution owner one explicit install release. Immediately before pip, the execution owner rechecks Python processes and the two-location dependency baseline. Any new unattributed worker or package drift stops the sequence before replacement. These checks detect a broken barrier; they do not substitute for producer holds.

Run the exact pip command synchronously and keep all producer holds active until the installed security/output-limit/helper/search gates finish. Do not terminate an unexpected consumer or interrupt pip mid-replacement. If a hold is breached during installation, keep the remaining holds, report the breach, and let the relevant owner quiesce its consumer before validation and release. Do not resume work based only on pip's exit status.

On passing gates, notify the orchestrator for its one real Codex deep-query acceptance. That authorized live acceptance belongs to the orchestrator; this owner performs only offline checks. Houston's release owner then completes the agreed controlled activation and Agents-page acceptance. The relay activation owner completes its post-configuration runtime capture before releasing digest production. Qualification owners resume only after the orchestrator ends their hold. No blanket process termination or restart occurs.

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
import hashlib, json, runpy, sys, zipfile
repo = Path(r'C:/Users/Stan/Documents/GitHub/skell-e-router')
site = Path(r'C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages')
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
(repo / 'docs/credential-error-installed-after.json').write_text(json.dumps({'packages': current, 'router_origin': skell_e_router.__file__, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'reviewed_source': '63b5fd22bacef9100b09cdee355bd8839439be78'}, indent=2) + '\n', encoding='utf-8')
'@ | & 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe' -B -
if ($LASTEXITCODE -ne 0) { throw 'Installed router verification failed' }
```

The selected installed router must resolve outside both source checkouts. The runner asserts it before testing, and the helper checks code/metadata version agreement. Do not prepend the entire system site-packages directory during normal-startup validation, because that would hide the user-site SDKs. Repeat the 115 selected tests with `python -I -B scripts/run_security_tests_offline.py --require-installed C:/Users/Stan/AppData/Local/Programs/Python/Python311/Lib/site-packages tests/test_credential_errors.py tests/test_output_limits.py -q` for the system-only SDK path. Do not claim installed mitigation from the source-only command or `--probe` process status alone.

The [installed search verifier](../scripts/verify_installed_search.py) adds the real-router negative checks that the 10 stub-router tests cannot establish. Run `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe -B scripts/verify_installed_search.py --output docs/credential-error-search-after.json` after installation. Default expected version is 3.26.3. It executes the actual search.py CLI function with `query -n 3 --json -- anything`, the real installed router and real SDKs, a fresh child-only synthetic credential profile, disabled networking and a 90-second limit per child. Only VECTORS/CHUNKS point to a miniature synthetic index, so it never loads the real index. The source file remains unchanged.

The credential case must exit 1 with empty stdout and exactly one stderr line beginning `session-search: embedding provider failed`. The verifier removes only its own temporary synthetic key file, then requires the missing case's single `session-search: missing embedding credential` line. Neither case may emit a traceback or synthetic key. Metadata records exact interpreter, imported router path, code/metadata agreement, `get_embedding` signature and actual SDK origins. The [3.26.2 compatibility capture](credential-error-search-before.json) passes both cases under normal user-site priority. It is compatibility evidence, not proof of the router security fix; the post-install 3.26.3 capture and security tests are still required.

Repeat the parent metadata captures for released consumers and record the package archive hash from wheel installation. Recheck search/digest test-file hashes against the completed capture; rerun only changed consumer suites under the same synthetic setup. Qualification successors choose and run their agreed stubbed checks at their quiet boundaries. No live provider call, real index read or rebuild is needed.

## Failure and completion

If installation or validation fails, keep affected launches held and report the exact failing gate. Do not resume them merely because the package version changed. The cached 3.26.2 recovery wheel and its SHA-256 are in the before snapshot. Its 19 package files match the current installation after normalizing line endings; six files differ in CRLF versus LF bytes, with no content differences. Recovery to that wheel restores a known affected version and requires recording that fact. Do not roll back or restart anything silently.

If the orchestrator directs recovery while holds remain active, use the exact cached artifact with the same dependency-preserving options:

```powershell
$routerRecovery = Get-Content -LiteralPath 'docs/credential-error-installed-before.json' -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $routerRecovery.rollback_wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $routerRecovery.rollback_wheel_sha256) { throw 'Recovery wheel hash mismatch' }
& 'C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe' -I -m pip --isolated --disable-pip-version-check install --no-index --no-deps --force-reinstall $routerRecovery.rollback_wheel
if ($LASTEXITCODE -ne 0) { throw 'Router recovery failed; retain consumer holds' }
```

Verify both SDK-location baselines remain unchanged and router files/version match the recovery artifact. Record recovery as 3.26.2 with the known router disclosure risk. The 3.26.3 security acceptance tests are expected to reject that recovered package. Only the orchestrator can release consumers based on their separately verified local containment; recovery is not installed security completion.

Installed completion requires matching wheel/code/metadata identity, unchanged SDK versions, passing installed security/output-limit/helper checks, agreed consumer checks and owner-confirmed runtime/quiet-window gates. Record actual installed results in the existing security release document and mark only the installed milestone in `docs/TASKS.md` complete. Independent virtual environments and embedded copies remain separate work. External API spend to date is $0.
