# Jev rough cut: design

Developer-facing design, written to be executed cold. Approved in shape by Stan on 2026-09-20 in the lead thread. Background and the numbers it builds on: `docs/handoffs/2026-09-19-jev-roughcut-brainstorm.md`, `docs/jev-real/roughcut-notes.md`, `docs/jev-real/partial-coverage.md`.

## Goal

Find the ceiling of a Jev-only rough cut on the solar-sailer benchmark, measured on the harness's own ladder (SENTENCE POINTS headline, WORD SCORE, GRADE, frame match), on held-out episodes, with wall-clock seconds per episode reported next to every score. The runtime target is about 10 seconds per episode. There is no fixed quality bar: report where the design lands on the ladder in `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-sentence-points-rescore.md` (best Luna 79.96, shipped Opus agentic 83.45, rule-only baseline 63.72).

## Principles

- Timing decisions in code, meaning decisions in Jev. Anything that depends on how long a pause is (which um to cut, which of two doubled words to drop) is decided by code from word timings and audio. Jev decides what a sentence is worth, where a wind-up or fade-out ends, and which take of a repeated line is best.
- Jev never sees a decision the production modules already made. Ums the Um Removal module cuts are stripped from the transcript Jev reads. Ums the module kept stay, and Jev is told to leave them alone.
- Every Jev request and answer is written to disk so any variant can be rescored without new calls.
- Nothing is written under `D:\solar-sailer` or `C:\Users\Stan\Documents\GitHub\solar-sailer`. Both are imported read-only.

## Data

Per episode, under `D:\solar-sailer\benchmarks\roughcut\corpus\<episode>\`:

- `sentences.json`: `id`, `text`, `timecode`, `is_retake` (stamped by the production Retakes module).
- `transcript.json`: `word_segments` with `id`, `text`, `start`, `end` (seconds), `sentence_id`, `start_custom`, `end_custom`.
- `retakes.json`: `groups` keyed by group id, each `{members: [sentence ids in spoken order], take_index, winner}`.
- `cache/*_conform.wav`: the audio the Um Removal and Delete Silence detectors need.

Answer keys: `episodes/<episode>.answerkey.json`, read through the cache-only loader in `scripts/jev_real/roughcut_bench.py` (`_cached_answer_key`).

Episode split. Fit set (prompts and thresholds may be tuned here): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique. Held-out set: the other 13. Held-out episodes are run only after prompts and thresholds are frozen, and each held-out run is recorded with the prompt version it used.

## Pipeline, per episode

### Step 0: deterministic layer (code, no Jev)

Run the shipping Um Removal and Delete Silence detectors through `detect_removals(corpus_path, transcript, frame_rate)` in `D:\solar-sailer\benchmarks\roughcut\scripts\deterministic_baseline.py` (imported read-only; it puts solar-sailer's `server` package on the path itself). Keep the returned `umm` and `silence` frame ranges. Derive the set of word ids the um cuts remove (a word is removed when more than half its frames fall inside an um range). Those words are stripped from every transcript rendering Jev sees.

### Step 1: retake pass (Jev)

Input: every group in `retakes.json`. For each group, state is the takes in spoken order plus the five sentences before the first member and the five after the last member, rendered as plain `id = text` lines with removed ums stripped. Up to 6 groups per request, as `groups[k]`.

Questions per group:

- `take_k`, choice. Options are the takes, keyed `take1`, `take2`, ... in spoken order, each described by its sentence text. Instructions: "`groups[k].takes` are sentences the retake detector flagged as repeated attempts at the same line, in the order spoken, with the surrounding sentences in `groups[k].before` and `groups[k].after`. Which take does the video editor keep in the final cut? Prefer the take that is complete, fluent and best worded. When two takes are equally good, choose the later one."
- `real_k`, noul. Instructions: "The takes in `groups[k].takes` are repeated attempts at the same line, where the speaker said it again to get it right, rather than different sentences that happen to share words, a deliberate repetition for emphasis, or a callback to an earlier point."

Code rule: if `real_k` < 0.5 the group is not a retake and no member is cut. Otherwise the winner is the top choice, except when the top probability minus the last take's probability is under 0.15, in which case the last take wins (Stan's rule: identical takes, keep the later one). Every other member is cut. The corpus `is_retake` flag is replaced by this decision for the Jev arm, so the harness's unconditional loser cut reflects Jev's choice, not the module's.

Also score a variant that keeps the module's flags untouched, so the retake pass's contribution is measured on its own.

### Step 2: sentence pass (Jev)

State: `{"rules": <editing rules text>, "transcript": [...], "targets": [...]}`.

- `transcript` is the whole episode as plain `id = text` strings, retake losers from step 1 removed, ums from step 0 stripped, with a `<pause>` marker between two words whose gap is 0.5 s or more and `<long pause>` at 1.5 s or more. If the rendered transcript exceeds 24,000 tokens by the provider's count, use a window of 200 sentences either side of the target block instead, and record that the window was used.
- `targets` is the block of up to 25 sentences being asked about, each as `{"id", "words": [{"w": <word id>, "t": <word text>}, ...]}` with the same um stripping and pause markers. Questions reference `targets[k]`.

Questions per target sentence:

- `score_k`, score, six levels, in this order (level index 0 to 5 is the harness's 0 to 5 rating):
  - 0: "A false start the speaker abandons, the losing take of a line said again, dead air, or the speaker operating the screen or software or talking to the producer about what to show. Examples: 'So the, uh..', 'Let me just scroll down here.', 'Can you put that on screen?'"
  - 1: "Filler with no lesson content: throat-clearing, 'okay so', 'um yeah', or a sentence that only announces what is about to be said. Example: 'Okay, um, so yeah, let's see.'"
  - 2: "Rambling or a tangent that is not funny: the point was already made, or it wanders away from the lesson. Cut unless the next sentence depends on it."
  - 3: "Ordinary connective teaching talk: fine, keeps the flow, nothing memorable. Example: 'So that's the first thing to look at.'"
  - 4: "Clear teaching content or a genuine personality moment: it explains a point, gives a reason, or is funny. The edit is weaker without it."
  - 5: "An essential teaching point or a great moment: the core idea of the lesson, the punchline of a joke, or the key correction on a student's work."
  Instructions: "How much does sentence `targets[k]` earn its place in the final edit of this art lesson, given the whole transcript in `transcript` and the editing rules in `rules`? Later sentences change what earlier ones are worth: a line said again means the earlier attempt loses, and a sentence a later one refers back to must stay."
- `first_k`, choice. Options are the sentence's word ids in order, keyed by word id, each described by the word text, plus `whole` described as "nothing is trimmed from the start". Instructions: "Assume the editor keeps sentence `targets[k]`. Editors often trim the start of a kept sentence: a wind-up like 'So, um, yeah, okay', a stranded false start before the real sentence begins, or a first attempt that the rest of the sentence replaces. Which word is the first word the editor keeps? Choose `whole` when nothing should be trimmed. Never trim only to remove an 'um', 'uh' or another single filler word inside otherwise good speech: those are handled elsewhere and stay. What survives must read as a complete, grammatical line."
- `last_k`, choice. Same options plus `whole` described as "nothing is trimmed from the end". Instructions: "Assume the editor keeps sentence `targets[k]`. Editors often trim the end of a kept sentence: a fade-out like 'you know, kind of, right?', a trail-off, or a dangling 'and' or 'so' the next sentence does not need. Which word is the last word the editor keeps? Choose `whole` when nothing should be trimmed. Never trim only to remove an 'um', 'uh' or another single filler word inside otherwise good speech: those are handled elsewhere and stay. What survives must read as a complete, grammatical line."

Sentences with one word get only `score_k`.

### Step 3: trim pick pass (Jev, variant B only)

For every sentence where `first_k` or `last_k` puts less than `T_trim` probability on `whole` (T_trim swept on the fit set, start at 0.5), build candidate versions: the top three starts by probability (including `whole`) crossed with the top three ends, keeping only start before end, plus the whole sentence and `cut`. Render each version as its surviving words. State per item: five sentences before, the sentence, five sentences after, `versions` keyed `v1`, `v2`, ... and `whole` and `cut`. Up to 10 items per request as `items[k]`.

Question `pick_k`, choice over the version keys. Instructions: "`items[k].versions` are candidate ways to keep sentence `items[k].sentence`, each made of the sentence's own words in order. `whole` keeps every word; `cut` removes the sentence. Given the surrounding sentences, which version does the video editor keep? Prefer `whole` unless a version removes something that clearly does not belong: a wind-up, a fade-out, or an abandoned first attempt. Do not prefer a version because it drops an 'um' or a single filler word. Choose `cut` only when nothing in the sentence is worth keeping."

### Decision assembly

Per sentence: `score` (Jev's probability-weighted score, 0 to 5), `keep_words` (a list of `[first_word_id, last_word_id]` or null for whole), `cut_retake` (from step 1).

- Variant A, one shot: if `whole` is the top option on both `first_k` and `last_k`, no trim. Otherwise `keep_words` runs from the top `first_k` word (or the first word) to the top `last_k` word (or the last word). If the trim would leave nothing, treat as no trim.
- Variant B, two pass: `keep_words` from `pick_k`; `cut` sets the score to 0.
- The harness keep threshold on `score` is calibrated per arm by the existing pooled sweep. Retake losers are cut regardless of score.

Both variants come from the same step 2 answers, so one run produces both.

## Scoring

A scoring module in this repo takes `{sentence_id: {score, keep_words, cut_retake}}` per episode and returns the harness metrics by calling the harness's own code (`sentence_scoring`, `word_scoring`, `partial.build_word_tiles`, `partial.attach_keep_ranges` or their callers, `calibrate`), read-only. It reports:

- plain: the model's cut alone;
- with modules: the same cut with the step 0 `umm` and `silence` ranges subtracted, the way `scripts/model_plus_deterministic.py` layers them onto every ladder arm.

Validation gate before any Jev numbers are trusted: rescore the archived `results/2026-09-06-all18-partial-single-gpt-5.6-luna.json` ratings through this module and reproduce its published per-episode SENTENCE POINTS and WORD SCORE to two decimals on at least three episodes.

Latency is recorded per pass and per episode as wall-clock seconds at the concurrency used, plus request count, input tokens and cost. Every result table carries a seconds-per-episode column.

## Arms to report

| Arm | What it is |
|---|---|
| jev_a | Variant A (one-shot trim), Jev retake pass, plain |
| jev_b | Variant B (two-pass trim), Jev retake pass, plain |
| jev_a_mod, jev_b_mod | The same with um removal and delete silence layered on |
| jev_b_moduleretakes | Variant B with the production retake flags instead of Jev's |
| jev_b_notrim | Variant B scores only, no partial trims, to isolate what trimming adds |

Compared against the ladder's published arms with the same modules layered on (`results/2026-09-11-model-plus-deterministic.md`).

## Files

- `scripts/jev_real/roughcut_jev.py`: the pipeline. Without `--run` it prints the plan and a cost estimate. Writes `docs/jev-real/roughcut-jev-requests.jsonl` (every request and answer, with pass, episode, block, elapsed seconds, tokens, cost, prompt version) and `docs/jev-real/roughcut-jev-decisions.jsonl` (per sentence per arm). Refuses to overwrite existing output names.
- `scripts/jev_real/roughcut_partial_scoring.py`: the scoring module above, with the validation gate as a runnable check.
- `scripts/jev_real/roughcut_jev_report.py`: rescoring and the notes file `docs/jev-real/roughcut-jev-notes.md`.
- Prompts live in `scripts/jev_real/roughcut_jev_prompts.py` with a `PROMPT_VERSION` string recorded in every request row.

## Cost and speed expectations

Jev input is $0.042 per million tokens, output free. The earlier whole-transcript arm cost $0.05 and ran 11 seconds of wall-clock for five episodes at concurrency 4. Step 2 re-sends the transcript once per 25 sentences, so the long critiques dominate: about 70 requests of roughly 25k tokens for perspective-13d, around $0.07. A full 19-episode run should stay under $0.50. Concurrency 8 is the starting point; the provider allows 1,200 requests a minute.

## Out of scope for round one

- Paragraph or topic blocks (the production Retakes module already flags retake paragraphs; reuse that when this is picked up).
- Middle-of-sentence cuts (a self-correction inside a sentence).
- Finding retakes the production module missed.
- A production integration in solar-sailer. If the retake pass works, Stan wants it considered for the retakes panel separately from rough cut.

## Status after round one (2026-09-20)

Built and run as specified, prompt v3 frozen on the six fit episodes, then run on the 13 held-out episodes. Results: fit set in `docs/jev-real/roughcut-jev-v1-v2-v3.md`, held-out and the 18-episode ladder placement in `docs/jev-real/roughcut-jev-heldout.md`, miss patterns in `docs/jev-real/roughcut-jev-v3-misses.md`, long-episode context handling in `docs/jev-real/roughcut-jev-longepisodes.md`. Headline: jev_a v3 with modules layered scores 80.47 SENTENCE POINTS on the 18 ladder episodes, 17th of 24 arms, level with gpt-5.6-luna-medium agentic, 3.2 under the best Luna chapters arm (83.71) and 6.0 under the shipped Opus agentic (86.47), at a mean of 5.6 seconds per episode (worst 19 s, retries) and about $0.05 per episode. The trim pick pass (variant B) was dropped: Jev's trims land on sentences the editor kept whole far more often than on the editor's actual trims, and the um and silence modules already supply the partials. The retake veto at 0.5 held up. The remaining gap is whole-sentence judgment on critiques, where Jev keeps about twice as many sentences the editor removed as Luna does; the paragraph or topic-block pass deferred from this round is the next lever.
