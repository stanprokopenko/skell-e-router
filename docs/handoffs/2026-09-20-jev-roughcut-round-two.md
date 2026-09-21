# Handoff: Jev rough cut, round two

Written for the next lead, a Fable 5.1 session that will keep prototyping the Jev rough cut with Stan in its own thread, working in `C:\Users\Stan\Documents\GitHub\skell-e-router`. Stan's ask, in his words: "continue this experimental rough cut prototyping with me. I'm sure there are more routes we can explore such as a combination of jev and then Luna or opus on the remaining things Jev gets wrong like partial sentences. Maybe Jev can accurately identify which sentences need a partial cut and then a smarter model can make those cuts? Or we let the smarter model decide and cut." Brainstorm with him first, in the thread, one decision at a time; build after he is aligned. `C:\Users\Stan\Documents\GitHub\solar-sailer` and `D:\solar-sailer` are read-only reference; everything stays in this repo. Standing $20 budget; round one spent about $2.40 and a full Jev run over all 19 corpus episodes costs about $0.75. The corpus has 19 episodes; the published ladder covers 18 of them (greco-2.2-thumbnailing has no ladder entry), so ladder comparisons use 18. `python` on PATH is Python 3.11.

## Where round one ended

Design and status: `docs/superpowers/specs/2026-09-20-jev-roughcut-design.md`, including the "Status after round one" section. Results, all developer-facing:

- Fit-set iterations v1, v2, v3: `docs/jev-real/roughcut-jev-v1-v2-v3.md`. v3 is frozen.
- Held-out 13 episodes and the 18-episode ladder placement: `docs/jev-real/roughcut-jev-heldout.md`. Headline: jev_a v3 with modules 80.47 SENTENCE POINTS, 17th of 24, level with gpt-5.6-luna-medium agentic, 3.2 under the best Luna chapters arm (83.71), 6.0 under the shipped Opus agentic (86.47). Mean 5.6 s per episode over the 18, worst 19.3 s (flanders-03-thematic-crit), and the slow episodes are retries rather than model latency: about a fifth of requests fail first time with a bare provider error, mostly the large sentence-pass requests. The token-fitted window plus block size 25 brought flanders to 12.7 s in a probe (`roughcut-jev-longepisodes.md`), and those are now the defaults, but no full 18-episode run has been done with them yet.
- Miss patterns with examples: `docs/jev-real/roughcut-jev-v3-misses.md` (and `docs/jev-real/roughcut-jev-misses.md`, the v1 version, for the patterns v2 fixed and the confidence-quartile table).
- Long episodes, context window, block size: `docs/jev-real/roughcut-jev-longepisodes.md`.
- How much of the editor's inside-sentence trimming lands on code-detectable boundaries: `docs/jev-real/partial-coverage.md`.
- The v3 export in the harness's result format: `docs/jev-real/export/`.

Tooling, all under `scripts/jev_real/`: `roughcut_jev.py` (pipeline; `--prompt-version`, `--block`, `--context-tokens`, `--trim-pick`, `--repair`, `--budget`, plan mode without `--run`), `roughcut_jev_prompts.py` (versioned prompt bundle), `roughcut_partial_scoring.py` (harness metrics from decisions, validated against the leaderboard; `--validate`, `--removals`), `roughcut_jev_report.py` (rescoring any arm offline, ladder comparison), `roughcut_jev_misses.py` (confusion, miss lists, trim quality, veto sweep), `roughcut_jev_export.py` (harness-format result JSON). Every request and answer of every run is in `docs/jev-real/*-decisions.jsonl` and `*-requests.jsonl`, so most questions below can be answered with no new calls. Keys are Machine-scope; hydration recipe in `docs/jev-real/routing-notes.md`.

Discipline to keep: prompts and thresholds are tuned only on the six fit episodes named in the spec; held-out runs happen after freezing and are reported with the prompt version. Every result table carries seconds per episode. Compare on the "with modules" column, because every published arm gets the same um removal and delete silence layered on.

## What the data says about the remaining gap

- Jev's whole-sentence judgment is where the points are. On critiques and the scripted intro it keeps about twice as many sentences the editor removed as Luna does (255 versus 122 wrong keeps on the fit set, similar on held-out). Those are stretches the editor dropped, not single lines. Jev's confidence tracks this: agreement with the editor rises from 74% in its lowest confidence quartile to 95% in the highest (`roughcut-jev-misses.md`, section "Do the six score levels separate?").
- Jev's own inside-sentence trims were poor: of 32 surviving trims, 21 landed on sentences the editor kept whole and 5 matched the editor's trim. Whether Jev can tell WHICH sentences need a trim, as opposed to where, has not been measured. The `first_p_whole` and `last_p_whole` fields on every decision row make that a $0 question: compute how well they predict the human-partial sentences (the scoring module's human sentence states give the labels).
- Three prompt rounds moved the fit-set score by about a point either way, which is inside the noise between runs. Further prompt wording is not the lever.
- greco-2.2-thumbnailing over-keeps badly (kept ratio 178%) and is not on the ladder; worth a look at what that transcript does differently.
- The um and silence modules supply nearly all the partial-sentence credit today. The trim pick pass (variant B) was dropped for that reason; the code is still there behind `--trim-pick`.

## Directions to bring to the brainstorm

Stan's, first:

- Jev flags, a smarter model cuts. Jev identifies sentences that need a partial cut; Luna or Opus receives only those sentences with context and returns word ranges. Start with the $0 measurement above; if Jev's flags have good recall on human-partial sentences, the smart model sees maybe 15 to 25% of sentences. Cost and latency scale with that share, so report both.
- The smarter model decides and cuts. Confidence routing: Jev scores everything, the lowest-confidence quartile goes to Luna in windowed calls, and Luna's keep/cut (and trims) replace Jev's there. The archived Luna chapters runs have per-sentence ratings in `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-*-chapters-r5-gpt-5.6-luna.json` (`run_ratings`), so a first estimate of the ceiling of this hybrid needs no Luna calls at all: substitute Luna's saved decisions on Jev's low-confidence sentences and rescore.

Mine:

- A paragraph or topic-block pass for the critiques. The production Retakes module already flags retake paragraphs (`editor/server/modules/retakes_paragraph_v2.py`), and silence between sentences is in the transcript timings; Jev can score blocks before sentences so a dropped stretch is judged once. Stan agreed to hold this for round two.
- The tie between decisions and the calibrated threshold. Calibration maximises the retired GRADE pooled across episodes, and demos and critiques want different operating points. Any change here must apply to every arm the same way to stay comparable; discuss with Stan before touching it.

## Constraints to keep

- Never write under either solar-sailer checkout. If a design should move to production, write a handoff like `docs/handoffs/2026-09-20-retakes-panel-jev-pick.md`.
- Hold out episodes. Do not tune on all 19.
- Report actual spend. Jev is cheap; Luna reruns cost about a cent an episode, Opus agentic runs cost dollars and take an hour per episode, so prefer the archived Luna and Opus ratings for hybrid estimates.
- Result files refuse overwrite; use a new output name per experiment.
