This note is for developers integrating Houston's TLDR helper with skell-e-router.

# Houston output limit

## Correction

Version 3.26.2 preserves the helper's `max_tokens=600` through the router, LiteLLM and OpenAI request serialization. `TLDR_MODEL` remains `gpt-5.6-luna`. No helper, Houston app or model pin changes belong to this router release.

The pre-fix router returned `{}` when filtering Luna's `{"max_tokens": 600}`. Its model registry omitted output-limit parameters. The correction declares support on all 15 registered first-party OpenAI reasoning models, normalizes the legacy keyword to `max_completion_tokens`, and adds that field to LiteLLM's forwarding allowlist. This keeps the provider contract in the router and prevents LiteLLM's model registry from silently dropping the cap.

## Caller contract

Use `ask_ai(model, prompt, max_tokens=600, rich_response=True)` or replace `max_tokens` with `max_completion_tokens`. For Luna, the serialized Chat Completions body has `"max_completion_tokens": 600` and no `max_tokens` field. Astra's existing Responses bridge converts the same keyword to `max_output_tokens`.

Values must be positive integers. Boolean, string, float, zero and negative values raise `RouterError` with `code="INVALID_PARAM"` before a request. `None` means omitted. Supplying both aliases requires matching values. Combining a finite top-level cap with any output-limit field in `extra_body` raises `INVALID_PARAM`, because SDK body overrides could otherwise replace the cap. Unrelated request-body extras and allowlist entries survive.

The cap includes visible output and reasoning tokens. It does not promise 600 visible tokens, cap input tokens or limit total dollars across retries. The helper must reject unsuitable empty or truncated results. The Houston implementation lead owns that behavior and app shipping. The [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) defines `max_completion_tokens`; the router's [technical reference](../skell_e_router/Skell-E-Router-DOCUMENTATION.md#output-token-limits) documents its aliases.

## Verification

- `python -m pytest tests/ -q` passed all 727 tests, including 61 new output-limit cases.
- The new tests invoke the real `ask_ai`, LiteLLM and OpenAI serialization code. They intercept `httpx.Client.send`, return fixture JSON or streaming events, and deny external socket connections. Both aliases reach Luna's HTTP body, including streaming. Both aliases reach Astra's Responses body. GPT-4o retains its existing legacy field.
- Parameter tests cover all 15 first-party reasoning models, invalid values, conflicting aliases, omitted limits and body override protection.
- `python -m pip wheel --no-deps --wheel-dir dist .` built `skell_e_router-3.26.2-py3-none-any.whl`. Its model configuration, utility code, package version and bundled documentation match the working source byte for byte.
- A fresh blind code and security reviewer returned clean in round 1. The review loop stopped there.
- All provider responses in this verification are fixtures. Live API spend is $0. These checks prove request construction and installed integration; they do not measure a live model response.

## Release and installation

Commit `d8ae9876fd2f095d5e6e03e11710c6d7a8ddcefe` contains the correction, tests, documentation and v3.26.2 version bump. It was pushed to `origin/main`. The installation used `python -m pip install --upgrade git+https://github.com/stanprokopenko/skell-e-router@main`, which resolved that exact revision and replaced the previous v3.26.0 editable installation.

Houston's `python` resolves to `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe`. The helper imports v3.26.2 from that interpreter's `Lib/site-packages/skell_e_router`, with matching package metadata and module version. LiteLLM is 1.83.14 and OpenAI is 2.24.0. Both normal Python startup and isolated startup passed the actual-helper capture after installation. The relay was not restarted.

The [installed verification JSON](houston-output-limit-installed.json) records one mocked request to `https://api.openai.com/v1/chat/completions`, model `gpt-5.6-luna`, with `{"max_completion_tokens": 600}`. The helper returned `ok: true` and the fixture response text. This capture used the helper file hash recorded in the JSON; the Houston lead's subsequent helper changes require a new integration capture before app shipping.

## Installed helper reproduction

The verification script reads `TLDR_MODEL` from the orchestrator's real `relay/src/models.js`, supplies JSON on stdin to the actual `scripts/houston-lead-tldr.py` using `runpy`, and captures the provider HTTP body. It does not replace `ask_ai` or LiteLLM. It uses dummy credentials and blocks external networking. Python's isolated mode and a package-path assertion prevent the router checkout from masquerading as the installed package.

Run from the same Python environment Houston uses:

```powershell
python -I C:/Users/Stan/Documents/GitHub/skell-e-router/scripts/verify_houston_output_limit.py C:/Users/Stan/Documents/GitHub/claude-orchestrator --output C:/Users/Stan/Documents/GitHub/skell-e-router/docs/houston-output-limit-installed.json
```

The JSON records the interpreter, imported package path, installed version, Git source revision, helper and model-file hashes, endpoint, model and actual output-limit field. Re-run after any helper change before Houston ships.
