This document is for developers maintaining the router and coordinating its installed consumers.

# Credential-safe provider errors

## Reproduction and ownership

On 2026-09-05, both reported disclosure paths reproduced against router source `7ffefbd55d361d765ff73baa9a58cbc51c810639`, package version 3.26.2. The installed shared Python 3.11 package separately reports version 3.26.2 and source revision `d8ae9876fd2f095d5e6e03e11710c6d7a8ddcefe` in package metadata. The source and installed revisions are different; the source reproduction is independent evidence.

The originating finding is H1 and "Actual checks and limits" in `C:/Users/Stan/Documents/GitHub/claude-orchestrator/docs/session-search-credential-security-2026-09-05.md`. This router issue has its own owner and task entries. Search repair lead `lead-3f2b41f7-00b5-4342-ad67-d95b14916b2e` owns the local search boundary. No search or Houston files belong to this repair, and search integration does not depend on router rollout.

The reproduction supplies a synthetic key directly to `get_embedding`. One stub raises an error containing that key. A second stub passes a synthetic key with a NUL to real h11 Authorization-header validation. No HTTP request leaves the process.

| Probe against original source | Final message | Formatted traceback | Router failure output | Original cause/context |
| --- | --- | --- | --- | --- |
| Raw synthetic key | Key absent | Key present | Key absent | Both attached |
| NUL escaped by h11 | Key present | Key present | Key present | Both attached |
| Both probes after repair | Key absent | Key absent | Key absent | Neither attached |

No real credential exposure was demonstrated. Development read no real key values, made no provider calls, changed no model pins, rotated no keys and upgraded no shared installed packages. External API spend is $0.

## Repair and compatibility

Version 3.26.3 uses one dependency-light provider diagnostic helper. It maps failures to fixed categories and accepts only integer HTTP status numbers. It never formats the provider exception or copies its body, headers, notes or original chain. Raising the new error after leaving the catch block prevents Python from retaining the provider exception as either `__cause__` or `__context__`. Suppressing the displayed chain alone would leave the raw object attached.

The change covers embeddings, LiteLLM chat, direct Gemini and Anthropic calls, file uploads, deep-research client creation, polling, follow-up and generator calls. Stream wrappers cover deferred failures. Deep-research retry logs, provider error events, final failure details and raw-interaction debug output no longer print provider diagnostics. Retry classification continues to use the original exception internally before the public conversion.

Existing error classes, operation codes, validation behavior and successful results remain. Callers that parsed arbitrary provider message text must use `details.category` and `details.status_code` instead. Stream wrappers preserve iteration and context-manager use; SDK object identity is no longer part of the returned behavior. The legacy private `_redact_keys` import remains for compatibility but no provider error boundary relies on it.

The guarantee covers public provider errors, standard formatted tracebacks and router failure logs. It does not cover debugger capture of frame locals, caller-enabled third-party SDK debug logging, secrets placed in prompt content, successful model output or explicitly requested raw response objects. It is not a general-purpose text sanitizer.

## Offline validation

Dependencies were copied into an isolated virtual environment from installed distributions without updating them. The test entrypoint clears the process environment before importing SDKs, uses isolated profile directories and blocks networking. The only socket exception permits the standard library to create Windows asyncio's internal local wakeup pair. Provider transports are stubs. Python is 3.11.2, LiteLLM 1.83.14, OpenAI 2.24.0, h11 0.16.0 and httpx 0.28.1.

```powershell
.\.security-venv\Scripts\python.exe -I scripts/run_security_tests_offline.py --probe
.\.security-venv\Scripts\python.exe -I scripts/run_security_tests_offline.py tests -q
.\.security-venv\Scripts\python.exe -I scripts/run_security_tests_offline.py --package dist/skell_e_router-3.26.3-py3-none-any.whl tests -q
```

The new regression suite covers synthetic config and environment keys; raw, repr, JSON and URL escaping; nested exception chains and notes; error fields; traceback and logging output; h11 header validation; hostile provider exceptions using the router's own exception types; safe HTTP status metadata; deferred stream failures; deep-research error events and reconnection; and unchanged missing-key validation. One integration case runs real LiteLLM and OpenAI request construction into a mocked httpx transport that invokes h11 locally.

All 781 tests pass against both the source and the built wheel, without warnings. This includes 54 new security and compatibility cases. The wheel's 20 packaged files match source byte for byte. Both original disclosure probes pass against the wheel with all disclosure and original-chain flags false.

The actual Houston TLDR helper also passes against an isolated extracted wheel with stubbed HTTP. It still sends `max_completion_tokens=600`, uses the unchanged `gpt-5.6-luna` model pin, and returns the fixture response. [Captured integration evidence](credential-error-houston-wheel.json) records helper and model-file hashes, package path and wheel hash. This is isolated compatibility evidence, not an update of Houston's shared installation.

## Release and independent review

The source fix is commit `3e0f8dd35925cb18cd369420a7eca85f854b627c`, followed by stream and retry compatibility corrections in `63b5fd22bacef9100b09cdee355bd8839439be78`. The latter is the exact reviewed and tested source revision for package 3.26.3. Both reached `origin/main`, with release-record commit `3b1553eb881b5163f5c49d2c1547dda4687deeac`; the remote hash was verified after push. Subsequent release-record changes contain documentation and captured verification data only.

The built wheel is `dist/skell_e_router-3.26.3-py3-none-any.whl` in the isolated security worktree. Its SHA-256 is `f7f21d1e30dcad7a4d46bc57ff87eeed7cb1255617fce876546fee7f23e01707`. The offline build used cached Hatchling 1.27.0, pathspec 0.12.1 and trove-classifiers 2026.6.1.19. No package-registry publication or shared-package installation occurred.

Two fresh independent `gpt-6-astra` reviewers reviewed bounded implementation packets without earlier findings. The loop stopped at the clean second round, within the three-round cap.

| Round | Candidate | Result and action |
| --- | --- | --- |
| 1 | `3e0f8dd35925cb18cd369420a7eca85f854b627c` | No HIGH. Three MEDIUM compatibility issues covered direct `next(stream)`, resumed partial text iteration and diagnostic categories after retry exhaustion. Fixed in the second source commit with nine new regression cases. |
| 2 | `63b5fd22bacef9100b09cdee355bd8839439be78` | CLEAN, no actionable HIGH, MEDIUM or LOW. The reviewer independently ran all 781 offline tests. Review stopped. |

The review covers changed error boundaries and related stream/retry behavior. It does not certify live provider behavior, caller-enabled SDK diagnostics, debugger frame-local capture or arbitrary external consumers. The lead, not the independent reviewers, performed wheel parity and Houston helper verification.

## Installed rollout

The shared consumer interpreter is `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe`. Its `Lib/site-packages/skell_e_router` remains on affected 3.26.2 at the installed revision above. The router checkout and isolated security environment do not mitigate that installed package.

The parallel consumer inventory reports four shared-Python routes still on this installed version: session search, Houston's TLDR helper, the relay's automatic digest vector rebuild and Solar Sailer's input-qualification work. The qualification instructions name the shared interpreter explicitly; helper and queue launchers default to PATH Python. Existing parent-process resolution still needs confirmation before rollout. Benchmark and other manual routes have unresolved interpreter choices. The parked Gemini work remains separate and untouched.

The inventory also reports independent Solar, benchmark, agent-library and Scripter installations at older versions. This repair did not reproduce against those revisions or assess their compatibility. Updating shared Python will not update them. The orchestrator owns the inventory and coordinated validation, including attribution of persistent Python processes.

1. Record each consumer's interpreter, package path, metadata version and source revision without reading credentials. Identify long-running processes that have already imported the old package.
2. Run the synthetic regression suite against the built release in an isolated environment. Run the existing `scripts/verify_houston_output_limit.py` check with the orchestrator path to verify the helper's output limit and model pin remain unchanged.
3. Have the orchestrator schedule the shared-package update to the exact reviewed source commit, or to its matching wheel. Use dependency-preserving installation so this security release does not upgrade provider SDKs.
4. Repeat the synthetic disclosure probes against the installed package using Python isolated mode and an explicit package-path assertion. Repeat the Houston output-limit check and the search owner's offline boundary tests. Use synthetic inputs and stubbed transports only.
5. Restart only identified persistent consumers that loaded the old version. Record actual installed revision and per-consumer validation before claiming installed mitigation.

The committed runner supports `--package PATH` before `--probe` or pytest arguments. It asserts that the imported router resides under the selected wheel or installation path. This prevents the checkout from making a still-affected consumer appear fixed. Use the exact source revision above for installation rather than a moving branch tip. `--probe` prints evidence; require every disclosure and chain flag to be false. Use `--package PATH tests/test_credential_errors.py tests/test_output_limits.py -q` as the automated pass/fail gate, together with the Houston helper's metadata/version check.

Source completion and installed mitigation are separate milestones in `docs/TASKS.md`. No shared environment changes are authorized by the source release itself.
