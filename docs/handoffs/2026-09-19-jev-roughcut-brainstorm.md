# Handoff: design a Jev-based rough cut with Stan

Written for the next lead, a Fable 5.1 session that will brainstorm with Stan in its own thread. Stan's ask, in his words: "I think if we spend some time brainstorming, prototyping and running a bunch of tests, we could make Jev beat Luna and be incredibly fast. Retiring the Agentic rough cut in place of a jev based rough cut that takes 10 seconds would be the dream." This is a brainstorm with him first, not a build. Use the superpowers brainstorming skill and the typesafe-ai skill.

## What exists already

- Results and every number: `docs/jev-real/roughcut-notes.md` (developer notes), `docs/jev-real/roughcut-summary.json`, per-sentence rows in `docs/jev-real/roughcut-results.jsonl`, per-request rows in `docs/jev-real/roughcut-requests.jsonl`. Stan-facing summary in `docs/jev-classification.md`, section "Task 2".
- The runner: `scripts/jev_real/roughcut_bench.py`. It scores any arm with the solar-sailer harness's own metrics (SENTENCE POINTS, WORD SCORE, MCC) by importing the harness read-only from `D:\solar-sailer\benchmarks\roughcut\roughcut_bench`. Add arms there rather than starting over. Without `--run` it prints a plan and cost estimate. Result files refuse overwrite; use a new output name per experiment.
- The corpus: 19 episodes with human answer keys at `D:\solar-sailer\benchmarks\roughcut\` (read-only; never write there). Harness README explains the metrics and the retired ones. The five episodes used so far: colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo. The other 14 are untouched and can serve as a held-out set.
- Production today: `solar-sailer/editor/server/modules/roughcut.py` (one Luna/sol full-transcript call, prompt `editor/prompts/roughcut_system_v1.md` or the partial-keep `roughcut_system_partial_v5.md`) and the agentic variant `roughcut_router_agent.py`. Archived model results live in `D:\solar-sailer\benchmarks\roughcut\results\`, including partial-capable Luna and sol runs that set the bar.
- Router: `skell_e_router.classify()` in `skell_e_router/classification.py`. Score-answer rounding tolerance was fixed on 2026-09-19 (commit 5d56160). About 2.5% of Jev requests returned a bare PROVIDER_ERROR that succeeded on retry; the runner re-issues once.
- API keys are Machine-scope on this PC and absent from a fresh shell; hydration recipe in `docs/jev-real/routing-notes.md`.

## What the first attempt showed (baseline to beat)

Pooled over 1,597 sentences: Luna WORD SCORE 0.770 and keep/cut accuracy 85.1%; Jev whole-transcript 0.759 and 79.5%; Jev 30-sentence window 0.713 and 76.5%. Jev keeps with 96% precision, same as Luna, but cuts 297 sentences the editor kept versus Luna's 200. Less context made Jev worse, and accuracy was flat by position in the episode, so long context is not the problem. Luna's own weak spots are the same as Jev's: sentences it labels `rambling` (editor kept 23 of 30) and `false_start` (editor kept 83 of 197). The retake yes/no question correlates 0.57 with the editor's cuts at 83% precision. Jev's confidence tracks accuracy only loosely on this task (top quarter 83%, bottom 72%).

Jev is text only. State must be a string, JSON object, or array of text; images, audio and video are not supported, per the TypeSafe state docs. So the design works from transcripts, word timings and whatever code can derive (pauses, repeats, sentence length, position in take).

The first attempt did the obvious translation: one score question per sentence with the production rubric as the levels. Nothing was architected around Jev's strengths. That is the gap Stan wants explored.

## Directions worth bringing to the brainstorm

These are starting points, not conclusions.

- Decompose the judgment. Instead of one "does it earn its place" score, ask several narrow questions per sentence (is it a false start, is it a repeated take of a nearby sentence and which take is better, is it off the lesson's topic, is it a joke or personality beat, does the next sentence depend on it) and let code combine them. The composed-noul approach lost in routing and spam because of hand-set cutoffs, so fit the combination on some episodes and hold others out.
- Pairwise take selection. Repeated takes are where a comparison question fits: give Jev two candidate takes and ask which one an editor keeps. Code finds candidates by text similarity first.
- Two-pass with code in the middle. Pass one asks cheap structural questions; code builds a cleaner transcript (retakes collapsed, false starts marked); pass two judges content on the cleaned version.
- Hierarchical context. Score paragraphs or topic blocks first, then sentences inside blocks that survive, so the per-sentence question sees the block's role in the lesson.
- Calibrate against the editor. The harness calibrates the keep threshold per arm already. Look at where Jev's over-cuts sit on its score distribution before changing the questions.
- Fit the rubric levels to what the editor actually does. The rubric text was written for a generative model. Score levels for Jev should describe concrete situations and include examples that look like real transcript lines, per the TypeSafe score docs.
- Speed target. Whole-transcript Jev ran all five episodes in 11 seconds wall-clock at concurrency 4. Batches re-send the transcript, so cost scales with sentences squared over batch size; measure it, it was $0.05 for five episodes.

## Constraints to keep

- Never write under `C:\Users\Stan\Documents\GitHub\solar-sailer` or `D:\solar-sailer`. All code and results stay in skell-e-router until Stan decides to move a design into solar-sailer.
- Hold out episodes. Do not tune on all 19.
- Lesson transcripts are fine to commit; they are course content, not customer data.
- Standing $20 external-API budget per lead. Jev is cheap; Luna reruns for comparison cost about 1 cent per episode.
