# MCP dependency security review

Developer reference. For Stan: the MCP extra fixes the reproduced crash, and I found no new security blocker in the reviewed package set. HTTPX2 is a legitimate Pydantic project. I recommend the extra on the reviewed LiteLLM 1.108.2 version, with controlled dependency updates, rather than keeping the proposed open-ended minimum of 1.43.0. No application code or dependency declarations changed during this review.

Reviewed November 6, 2026, Pacific time. Advisory queries ran November 7 UTC. Requested change: `litellm>=1.43.0` to `litellm[mcp]>=1.43.0`.

## Decision

Publish verdict: CLEAR for the proposed MCP extra. After the initial report, the requesting lead supplied `.worktrees/audit-review-20261106/pyproject.toml` and `D:\agent-builds\skell-e-agent\audit-review-20261106\venv`. Read-only inspection confirmed the proposed dependency line and an exact match for all 73 reviewed package versions in that environment. Its other 22 packages are outside this dependency review. The tighter-pin recommendation below addresses preexisting range risk and is not a blocker for this publish.

Accept the extra as a supported compatibility fix for the reviewed versions. Prefer `litellm[mcp]==1.108.2` for the immediate fix, subject to Stan's approval. For deployed applications, preserve a tested dependency lock or constraints with hashes as well: pinning LiteLLM alone does not pin its transitive packages. Revisit the extra when upstream fixes the unnecessary import. Exact pinning costs deliberate upgrades and may require coordination with consumers that demand another LiteLLM version.

A local monkeypatch creates a second implementation to maintain. A rollback to 1.105.7 avoids this particular import regression, but the router's recent provider changes were not tested against that version. Neither alternative has a demonstrated security advantage over the reviewed current package set.

No applicable HIGH or CRITICAL finding emerged from this focused review. This is not a claim that every permitted version, every native binary, or every future release is safe.

## Findings

| Severity | Finding, reachability, and consequence | Fix cost |
| --- | --- | --- |
| MEDIUM, correctness | LiteLLM 1.108.2 imports MCP unconditionally on the Responses path. An ordinary caller using the router's configured GPT-6 route can fail before reaching OpenAI. This is a packaging crash, not an attacker exploit. Reproduced without MCP; the same local fixture request passed after adding the extra. | The extra adds 12 packages in the reviewed Windows resolution and more import/native-code dependencies. An upstream import guard would avoid that cost. |
| MEDIUM, preexisting supply-chain constraint | Both the existing and proposed `>=1.43.0` requirements admit the confirmed malicious LiteLLM releases 1.82.7, 1.82.8, and 1.107.0. A compromised publisher can deliver code that runs with the application's credentials without prior access to Stan's computer. These releases have been withdrawn; normal fresh public-PyPI resolution did not select them. Cached or mirrored artifacts and older constraints remain relevant. | A reviewed pin or newer floor excludes these releases. Exact exclusions lose no legitimate functionality, but alone do not address unknown future compromises. Pins require intentional upgrades. |
| LOW, conditional exposure in older allowed dependencies | The proposed ranges permit HTTPX2 and MCP transitive versions with published vulnerabilities. HTTPX2 flaws require actual affected HTTP behavior, such as a hostile compressed response; MCP server and authentication flaws require those features to run. Merely importing the SDK does not expose those flows. No match affected the 12 newly resolved packages. | Keep tested deployment constraints. If older OpenAI SDKs remain supported, an explicit HTTPX2 floor of 2.12 prevents the five known HTTPX2 ranges discussed below. Broadly tightening MCP server dependencies may conflict with other applications and needs compatibility testing. |
| Informational, preexisting | Filelock 3.32.7 matched 13 advisory records in the full resolution. Their security attack paths require another local account with access to lock locations, outside this review's internal-tool threat model. Some also describe lock race correctness risks. LiteLLM's `<4` restriction prevents adopting the commonly listed 4.2.1 fixes. | No MCP-specific remediation. Changing this requires upstream compatibility work. Do not treat the extra as the source of this issue. |

The malicious release list is confirmed in [GHSA-92x9-889m-jgmw](https://github.com/advisories/GHSA-92x9-889m-jgmw). LiteLLM documents the October compromise in its [incident account](https://docs.litellm.ai/blog/security-incident-oct-21-2026-attack-chain). No compromised package was downloaded or run for this review.

## What actually resolves

On Windows x64 with Python 3.11, LiteLLM 1.108.2 resolves to 61 packages without the extra and 73 with it. All artifacts were wheels. MCP resolves to 2.3.0, HTTPX2 and HTTPCore2 to 2.16.0.

HTTPX2 is already present without the extra: OpenAI 3.46.0 requires `httpx2>=2.12,<3`. MCP 2.3.0 independently requires `httpx2>=2.10`. The two packages named in the original briefing therefore do not describe the actual incremental dependency set.

The extra also raises LiteLLM's Pydantic requirement to `>=2.12,<3`; today's base and extra resolutions already select the same newer Pydantic. See [LiteLLM 1.108.2 metadata](https://pypi.org/pypi/litellm/1.108.2/json), [MCP 2.3.0 metadata](https://pypi.org/pypi/mcp/2.3.0/json), and [OpenAI 3.46.0 metadata](https://pypi.org/pypi/openai/3.46.0/json).

Python 3.10 metadata resolution also succeeds with 73 packages. It selects HTTPX2 and HTTPCore2 2.14.0 because newer versions require Python 3.11. Eleven package versions differ; all were included in advisory checks. Python 3.10 execution, Linux, macOS, and the router's full application dependency set were not tested.

The [committed evidence](evidence/mcp-dependency-review-2026-11-06.json) contains exact versions, artifact URLs, SHA256 hashes, dependency declarations, advisory results, and the successful and failing probe results. These are a dated review snapshot, not a replacement for a deployment lock.

## Publishers and package age

HTTPX2 is Pydantic's continuation of HTTPX, confirmed by [Pydantic's documentation](https://pydantic.dev/docs/httpx2/get-started/) and the [official repository](https://github.com/pydantic/httpx2). No typosquat evidence was found. HTTPX2 and its HTTPCore2 transport first appeared on PyPI on May 11, 2026. Current 2.16.0 was uploaded October 21. They are newer projects built on an established predecessor.

The following are the 12 added packages. Dates are the earliest uploaded files still listed in PyPI JSON, not necessarily project creation dates. Publisher names are PyPI organizations or maintainer accounts, cross-checked against official repositories.

| Package reviewed | Publisher or maintainers | First listed upload | Official source |
| --- | --- | --- | --- |
| MCP 2.3.0 | Model Context Protocol, a Series of LF Projects; dsp, jspahrsummers, Kludex, maxisbey | 2024-11-20 | [python-sdk](https://github.com/modelcontextprotocol/python-sdk) |
| MCP-types 2.3.0 | MCP; dsp, Kludex, maxisbey | 2025-06-21 | [python-sdk](https://github.com/modelcontextprotocol/python-sdk) |
| OpenTelemetry API 1.45.2 | OpenTelemetry; LeightonC | 2019-10-01 | [opentelemetry-python](https://github.com/open-telemetry/opentelemetry-python) |
| PyJWT 2.15.1 | Jose Padilla, jpadilla | 2011-02-24 | [pyjwt](https://github.com/jpadilla/pyjwt) |
| cryptography 50.0.2 | Python Cryptographic Authority; reaperhulk | 2014-01-08 | [cryptography](https://github.com/pyca/cryptography) |
| cffi 2.1.1 | Armin Rigo and the CFFI maintainer group | 2012-06-18 | [cffi](https://github.com/python-cffi/cffi) |
| python-multipart 0.0.33 | Marcelo Trylesinski, Kludex | 2013-03-26 | [python-multipart](https://github.com/Kludex/python-multipart) |
| pywin32 312 | mhammond, glyph, pf_moore | 2018-01-21 | [pywin32](https://github.com/mhammond/pywin32) |
| sse-starlette 3.6.0 | sysid | 2020-02-21 | [sse-starlette](https://github.com/sysid/sse-starlette) |
| Starlette 1.10.0 | Marcelo Trylesinski, Kludex | 2018-06-25 | [starlette](https://github.com/Kludex/starlette) |
| Uvicorn 0.54.0 | Marcelo Trylesinski, Kludex | 2017-06-05 | [uvicorn](https://github.com/Kludex/uvicorn) |
| pycparser 3.12 | Eli Bendersky, eliben | 2011-02-18 | [pycparser](https://github.com/eliben/pycparser) |

MCP and MCP-types 2.3.0 have PyPI provenance attestations tied to the official repository's publishing workflow and commit `2118f14f8a19bc158d8a1cf90af58d85d187f849`. HTTPX2 and HTTPCore2 2.16.0 report Trusted Publishing but do not provide artifact provenance attestations; those are distinct assurances. LiteLLM 1.108.2 also lacks published provenance on the inspected wheel. Absence of provenance limits verification but does not prove malicious content. The review inspected published provenance, not an independent Sigstore verification. See [MCP release files](https://pypi.org/project/mcp/2.3.0/), [HTTPX2 release files](https://pypi.org/project/httpx2/2.16.0/), and [LiteLLM release files](https://pypi.org/project/litellm/1.108.2/).

## Installation and import behavior

All 73 selected Windows wheels were downloaded and matched their PyPI SHA256 values. Every wheel was scanned for `.pth`, `sitecustomize`, `usercustomize`, and packaged script entries. Wheel installation does not invoke package source-build hooks; command-line entry points are installed but not launched by pip.

One new Python startup hook exists. `pywin32.pth` adds pywin32 directories and imports `pywin32_bootstrap`. Its inspected bootstrap locates the installed `pywin32_system32` directory and calls `os.add_dll_directory`. It does not download or run a separate program. This code runs at Python startup, before the later probe's audit hook. Native pywin32, CFFI, cryptography, and existing native dependencies were not binary-audited.

Source-build fallback received a separate static check. Eleven of the 12 new packages publish source archives; pywin32 312 publishes no source archive. All 11 were downloaded and hash-verified. Pure-Python packages use Hatchling or setuptools; MCP and MCP-types use a versioning metadata hook. CFFI invokes compiler and pkg-config probes, including a Homebrew lookup on macOS. Cryptography uses expected Maturin, Rust, and CFFI compilation commands. No suspicious downloader or credential collector appeared in inspected build entry points. Build backends, Rust dependencies, and source builds were not exhaustively audited or executed. Ordinary source-build tooling may fetch build dependencies; the wheel-only verification avoided those paths.

The relevant runtime chain is:

1. The router calls `litellm.completion` with an `openai/responses/` route for configured GPT-6 models.
2. LiteLLM `responses/main.py` calls `_responses_try_dispatch_mcp_gateway`. At line 1115 it imports the MCP handler before checking the skip flag or tool list at line 1119.
3. The handler imports the MCP catalog at line 15; the catalog imports `mcp.types.CacheableResult` at line 20.
4. `mcp/__init__.py` imports client and server modules. These define models, type adapters, logger handles, locks, regular expressions, and an OpenTelemetry tracer handle. Windows also loads native pywin32 modules.

No MCP network session, server listener, or child process launches in the reviewed Python module initialization. The stdio client launches a process only when its context manager is entered. HTTPX2 imports its APIs and defaults; it does not make a request. Its optional `alias_httpx()` operation is not called automatically. An application's separately configured tracing integrations may have their own behavior.

Direct Chat Completions imports the MCP handler when `tools` is nonempty, even for ordinary function tools. The no-tools direct Chat Completions path differs from the Responses path. `_skip_mcp_handler=True` does not prevent the unconditional Responses import. See [Responses dispatch](https://github.com/BerriAI/litellm/blob/v1.108.2/litellm/responses/main.py#L1078-L1120), [catalog import](https://github.com/BerriAI/litellm/blob/v1.108.2/litellm/proxy/_experimental/mcp_server/catalog.py#L20), and [Chat Completions](https://github.com/BerriAI/litellm/blob/v1.108.2/litellm/main.py#L5368-L5374).

LiteLLM itself already has import and first-use side effects unrelated to this extra, including default `.env` loading and model-price/tokenizer data handling. The probe disabled `.env` loading, used local model pricing, and prefetched two tokenizer files with their published expected hashes. It also cached Python's Windows version probe before blocking subprocesses. These controls are documented so the result is not mistaken for an unmodified default-runtime trace.

## Verification and advisory coverage

The isolated environment lived under `D:\agent-builds\skell-e-router\mcp-security-review-2026-11-06`. The probe child inherited only selected operating-system variables and explicit test settings, with a placeholder API key and redirected home directory. Python audit hooks rejected external connections and subprocess launches. A virtual environment and audit hooks are test controls, not a security boundary against arbitrary malicious native code.

The base install reproduced `ModuleNotFoundError: mcp` through the exact catalog import chain, with zero HTTP requests reaching the fixture. After installing the extra, the same `litellm.completion` call sent one request to `/v1/responses` on localhost and returned `fixture ok`. No real provider request or paid API call ran. The completed probe observed only local socket activity. `pip check` returned `No broken requirements found`.

This verifies the failure and fix at LiteLLM's real completion-to-Responses boundary. It is not a full router regression suite or live OpenAI verification. Python 3.10 was resolved and advisory-checked only.

OSV and PyPI version-specific advisory records were queried for all 73 Python 3.11 packages. Only the preexisting Filelock version matched. OSV also checked all 73 Python 3.10 packages, and PyPI cross-checked all 11 changed versions. All 12 newly added versions had zero matches in both sources. Database absence does not establish absence of vulnerabilities.

The HTTPX2 requirement permits older affected versions. The inspected advisories are [compressed-response memory exhaustion, fixed in 2.12](https://github.com/pydantic/httpx2/security/advisories/GHSA-8xx6-hgc6-gc2m), [SSE CPU exhaustion, fixed in 2.10](https://github.com/pydantic/httpx2/security/advisories/GHSA-f2fp-rgf2-35cp), [WebSocket-over-SOCKS plaintext traffic, fixed in 2.10](https://github.com/pydantic/httpx2/security/advisories/GHSA-7mj9-2mp8-4m2p), [request framing, fixed in 2.11](https://github.com/pydantic/httpx2/security/advisories/GHSA-pf96-p4fj-6566), and [multipart header injection, fixed in 2.11](https://github.com/pydantic/httpx2/security/advisories/GHSA-h4x7-gw46-3wm6). Today's OpenAI SDK already enforces the 2.12 floor.

MCP's 2.2 minimum excludes the published SDK advisory ranges inspected, including [OAuth credential theft fixed in 2.2](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-qx49-fqc8-xw99). MCP's looser transitive requirements still permit older Python-Multipart, Starlette, PyJWT, and OpenTelemetry releases with advisories. Those server, authentication, and parser flaws were not shown reachable by the ordinary completion probe.

## Alternatives

| Option | Assessment |
| --- | --- |
| Extra on reviewed LiteLLM 1.108.2 | Recommended immediate fix. Reproduced failure disappears without a private patch. Added packages have expected publishers and no advisory matches at reviewed versions. Use controlled upgrades. |
| Exact LiteLLM 1.105.7 pin | Static tagged-source inspection identifies 1.106.0 as the first catalog-import regression. Version 1.105.7 has no OSV or PyPI advisory matches and was published October 28, after the October compromise. It remains a fallback candidate, not a verified compatible router release. A broad `<1.106` rollback can select much older releases and is less precise. |
| Upstream lazy-import correction | Best long-term dependency behavior. Check the skip flag and whether MCP tools are present before importing the handler, preserving tool iterators. Requires an upstream fixed release or a maintained patch. Moving only `CacheableResult` under `TYPE_CHECKING` is insufficient because runtime MCP types are used elsewhere. |
| Local monkeypatch, fake MCP module, or broad ImportError suppression | Not recommended. These depend on upstream internals or hide real failures, creating maintenance and debugging costs. |

The version boundary comes from [1.105.7 handler source](https://github.com/BerriAI/litellm/blob/v1.105.7/litellm/responses/mcp/litellm_proxy_mcp_handler.py) and [1.106.0 handler source](https://github.com/BerriAI/litellm/blob/v1.106.0/litellm/responses/mcp/litellm_proxy_mcp_handler.py). No dependency changes, source patches, or fixes were applied. Those remain subject to Stan's approval through the requesting lead. Review cost: $0 in paid service calls.
