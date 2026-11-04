# Jev image evaluations handoff

For the Fable 5.1 lead requested by Stan. The router implementation and live smoke test are complete. Run the evaluations next. Do not ask Stan to unlock 1Password again or repeat the smoke-test approval checkpoint.

## Start here

1. Read the governing plan at `C:/Users/Stan/Documents/GitHub/aingel/docs/plans/2026-11-02-jev-image-classification-test.md`, including "Stan's amendments, 2026-11-04". Its amendments override direct-HTTP instructions in the older plan.
2. Read [the checkpoint and scouting findings](../jev-images/2026-11-04-first-run.md). It contains the exact Solar Sailer classifier, historical fixture, label and result locations. Avoid repeating the initial broad search.
3. Inspect [the smoke runner](../jev-images/run_smoke.py) and [saved responses](../jev-images/2026-11-04-smoke.json). Then resolve image billing and prepare the labeled samples before making further paid calls.

## Authorization and destination

Stan's latest instruction was "got it ok. Please handoff the test running to fable 5.1 lead via orchestrator" after receiving the successful smoke results. This authorizes the successor to run the tests; the earlier instruction to wait after the smoke test is satisfied. The orchestrator must dispatch Fable 5.1, not an Astra replacement.

Work in `C:/Users/Stan/Documents/GitHub/skell-e-router`. Solar Sailer and the sibling benchmark repository are read-only except docs. Save the runner, frame-ID manifest with labels, raw responses without image bytes, and scored report in `docs/jev-images/`. Update the existing short summary at `C:/Users/Stan/Documents/Obsidian/Aingel/reports/2026-11-04 Jev image test.md`. Do not commit course images. Commit and push only your files by path in the shared checkout.

## Working environment

- PowerShell on Stan's PC. `python` resolves to `C:/Users/Stan/AppData/Local/Programs/Python/Python311/python.exe`. Pillow, requests, pytest and router dependencies are available there. No new environment is needed.
- Use this router checkout, not a potentially older installed distribution. `docs/jev-images/run_smoke.py` prepends the repository root to `sys.path`. Mirror that pattern in the evaluation runner or launch with the root on `PYTHONPATH`.
- Read the key with `op read "op://Stan Dev Secrets/TYPESAFE_API_KEY/credential"` into the requesting child process. The agent service account has verified read access. No desktop unlock, key creation, console visit or relay `.env` read is needed.
- No live jobs, temporary environments or worktrees remain from this lead. Existing untracked `.playwright-mcp/`, `.security-home/`, `grep.exe.stackdump` and `report-top.jpeg` belong to other work. Leave them alone.

## What is already verified

The router's `classify(..., assets=...)` accepts PNG/JPEG base64 envelopes and sends TypeSafe's preview header. Text behavior, retries and sanitized errors remain covered by tests. All 1,116 tests and 17 subtests passed, and an independent Astra review found no blocking issues. Source change is `fff8d13`; smoke evidence is `c38cf66`. Both are on origin/main. No shared installed package was upgraded.

One real frame resized to 1024 by 576 pixels went through each of `jev` and `jev-latest`. The returned versions were 1.13.0 and 1.14.0. Both accepted it and correctly detected a hand in 0.149 and 0.163 seconds. This is an access check, not a quality benchmark or negative control. The model-list endpoint listed Latest and Preview without vision or pricing metadata; the current docs say both aliases point to 1.14. Preview was not separately tested or added to the router registry.

The whole task has made two inference requests, totaling 3,738 input tokens and 40 output tokens. At the published text rate that is $0.000156996, but actual image billing is unconfirmed. The hard cap is $5 across this entire task, not $5 per model or per successor. Keep those two calls in the ledger. Router image `cost` deliberately returns `None`; do not treat it as zero or silently apply text rates as verified image prices. Establish a defensible cost bound before the full evaluations.

## Remaining work and traps

- Label the drawing tasks before seeing predictions. No 20-frame evaluation manifest, stage sample, best-still groups or subjective ground truth has been created yet. The first smoke frame was only labeled for obvious hand presence, not drawing obstruction.
- The 50 labeled pencil frames exist, but they all come from one source clip. Use other instructor folders for the planned diversity. All input paths and sample rules are in the governing plan and checkpoint.
- Test hand obstruction, drawing stage and best still, plus the original plan's pencil-contact bonus. Record accuracy, confusion, confidence bands, latency, tokens and honest cost estimates for each. If both versions remain eligible under budget, the plan calls for testing both.
- Solar Sailer's current import model is `gemini-3.1-flash-lite`. Its original selection used a seven-video informal trial, not a completed scored benchmark. All recorded source paths are missing, some category labels were guesses, and raw historical runs were not found. Search for relocated files before concluding the exact replay is unavailable. Do not turn guessed labels into claimed ground truth.
- The five accessible audio fixtures are a different task. Their reported 5/5 baseline cannot serve as the image comparison. Do not introduce transcription and call it like for like.
- If the original video comparison cannot be recovered, report that clearly and use the drawing-camera fallback authorized by amendment 2. Any fresh Gemini comparison needs the same inputs and labels and must stay inside the combined cap; distinguish it from the historical selection benchmark.
- Current TypeSafe docs list text input, yet image calls worked. Observed API behavior supports proceeding; it does not settle image pricing or quality. Keep all scored model calls through the router.

The existing entry at `C:/Users/Stan/Documents/GitHub/claude-orchestrator/docs/deferred-requests.md:76` tracks both evaluations. Its older waiting status predates Stan's handoff instruction above. Ask the orchestrator to mark continuation approved, assign it to the Fable 5.1 lead, and close it only after scored results or the plan's documented stop condition are delivered.
