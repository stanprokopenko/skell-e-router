# Handoff: put Jev in front of the chat routing classifier

For the next lead, a Fable 5.1 session working in `C:\Users\Stan\Documents\GitHub\skell-e-web`. Stan's ask, in his words: "I like the idea of using Jev and if it's confidence is < 20% then we ask again to production model or luna (whichever would get us better overall score on our benchmark)." He also said, explicitly: "don't want security review on the implementation. Only code review."

## Read first

1. The plan, written to be executed cold, task by task with tests and code: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\plans\2026-09-20-jev-routing-classifier.md`. Its "Context a cold engineer needs" section lists the seven files to read before Task 1 and the facts that are easy to get wrong.
2. The result this acts on, for Stan: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-classification.md`, sections "Bottom line" and "Task 1". Developer numbers: `docs/jev-real/routing-notes.md`. Stan reviewed the interactive report at `docs/jev-real/routing-report/routing-report.html` and chose Jev with a 0.2 confidence threshold.

## Decisions already made

- Fallback model is gpt-5.6-luna with low reasoning and the production prompt. On the 565-message benchmark, Jev then Luna at 0.2 scores 527 correct; Jev then today's gemini-3.5-flash-lite scores 524. Recomputed 2026-09-20 from `docs/jev-real/routing-results.jsonl`.
- Threshold 0.2 on Jev's `tier` choice confidence, inclusive: 0.2 and above is Jev's call.
- Rollout is behind an environment switch `SKELLE_ROUTING_JEV` with values off (default), shadow, on. Deploying the code changes nothing until the switch is flipped. Flipping it is a production configuration change and needs Stan's magic-word confirmation each time; the plan's Task 11 has the words.
- The Jev request is the exact one that produced the benchmark (`scripts/jev_real/routing_bench.py` lines 142 to 318 in skell-e-router). Do not redesign it in this task; if you think it should change, that is a new benchmark run first.

## Working rules

- All code changes are in skell-e-web. Nothing in skell-e-router needs to change; its `classify()` on main is what production installs.
- Commit only your own files by path; the checkout may be shared.
- A push to main that touches `backend/**` deploys the backend automatically.
- API keys are Machine-scope on Stan's PC and missing from a fresh shell; the plan has the hydration snippet.
- TYPESAFE_API_KEY must be added to Cloud Run as a Secret Manager reference before the switch can leave "off". Task 11 step 1.
- Code review only. No security review, by Stan's instruction.
- Budget: the standing $20 external-API allowance. The offline score in Task 8 costs under $0.10.

## When you are done

Report to Stan in your own thread with the offline score from Task 8, the deploy status, and the two confirmations you need from him (secret, then shadow). Update `docs/TASKS.md` in both repos: skell-e-router's has an open "Follow-up: limited live trial of Jev on chat routing" line.
