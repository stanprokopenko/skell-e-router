# Jev rough cut, round two: design

Developer-facing design, written to be executed cold. Approved in shape by Stan on 2026-09-26 in the lead thread ("do option B, both"). Builds on the round-one spec `docs/superpowers/specs/2026-09-20-jev-roughcut-design.md` (pipeline, data, episode split, scoring discipline all still apply) and the round-two handoff `docs/handoffs/2026-09-20-jev-roughcut-round-two.md`.

## What round two established before building

Both at $0, from stored decisions, in `docs/jev-real/roughcut-route1-flags.md` and `docs/jev-real/roughcut-route2-routing.md`, scripts `scripts/jev_real/roughcut_route1_flags.py` and `roughcut_route2_routing.py`.

- Route 1 (Jev flags sentences needing an inside-sentence trim, a smart model trims) is dead. Jev's head/tail signals have AUC 58 against the editor's partial sentences, sentence length alone does better (65), and substituting Luna's real archived trims on the flagged sentences moves SENTENCE POINTS by -0.15 to +0.48. Trimming stays with the um and silence modules. Not built.
- Route 2 (Jev decides, a smart model overrides on Jev's least confident slice) has a high ceiling. Ranking by `abs(score - 2.5)` (the "margin") with one global cutoff, and substituting the archived Luna chapters decision on the routed slice: 84.06 SP at 25% routed (global cutoff 0.46), 84.27 at 50% (cutoff 0.80), against 80.47 pure Jev and 83.71 pure Luna. Opus as donor: 85.03 at 25%, 86.21 at 50%, 86.47 pure. The archived donor decisions came from whole-episode agentic runs, so these are upper bounds until a real windowed run confirms them. That run is build A.

Build B is Stan's prompt-breakup idea: replace the one six-level "how good is this sentence" question with many one-look yes/no questions per sentence and fit a combiner in code. First version Jev-only.

## Build A: real Luna routing on Jev's unsure slice

Script: `scripts/jev_real/roughcut_hybrid_luna.py`. Reuses `roughcut_route2_routing.py` for selection, substitution and rescoring (refactor the shared pieces into importable functions rather than copying them).

Inputs: `docs/jev-real/roughcut-jev-all18-v3-decisions.jsonl`, arm `jev_a`, the 18 ladder episodes. Selection: margin `abs(score - 2.5)` below a global cutoff, 0.46 (about 25% of sentences pooled) and 0.80 (about 50%). Both cutoffs are run; 25% is the headline because it is the knee of the offline sweep. Say in the write-up that the cutoffs were chosen from the offline sweep over all 18, so the share is not held out; the model's decisions on the slice are.

The Luna call, through `skell_e_router` (never a provider SDK; see the `skell-e-router` skill), model `gpt-5.6-luna`, reasoning effort medium first. Keys: Machine-scope environment, hydration recipe in `docs/jev-real/routing-notes.md`.

- Context: the whole episode transcript as Jev sees it (same builder as `roughcut_jev.py`: module-cut ums stripped, retake losers removed, `<pause>` markers), one sentence per line with its id. Luna's 1,050,000 context takes the longest episode whole, so no window fallback is needed; the ceiling came from whole-episode reading, so give it the whole episode.
- Rules: the rules5 system prompt verbatim, read from `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md` at run time (read-only; never copy it into this repo), plus a short task preamble saying that only the listed target sentences need a verdict and the rest are context.
- Targets: the routed sentence ids, in transcript order, in groups of up to 40 per request (a group closes early when it spans more than 120 sentences of transcript). Concurrency 8, retries with backoff, every request and answer written to disk.
- Answer per target: `score` 0-5 on the rules5 rubric, `decision` keep or cut, one short `reason`. No `keep_words`: route 1 showed trims from Luna land on sentences the editor kept whole more often than they match, and the modules already supply the partials. Ask for JSON, validate, re-ask a group once on malformed output, and record unanswered targets (they fall back to Jev's own decision, counted in the write-up).
- Substitution: on routed sentences Luna's `decision` replaces Jev's keep/cut; `keep_words` stays null; Jev's `cut_retake` veto stays as it was; um removal and delete silence layer on as for every arm. Also store Luna's score so an offline threshold sweep (keep at score >= 2, 3, 4) can be reported next to the decision-based number.
- Outputs: `docs/jev-real/roughcut-hybrid-luna-m046-decisions.jsonl` and `-requests.jsonl` and `-timing.json`, same for `m080`; result files refuse overwrite. Write-up `docs/jev-real/roughcut-hybrid-luna.md`: pooled 18, fit six, held-out 12, SP/WORD/GRADE with modules, ladder placement (the report script's `ladder()`), agreement of the live Luna decision with the archived Luna decision on the same sentences (how much of the ceiling survives the windowed call, and why), flip counts split by agreeing with the editor, seconds per episode (Luna wall clock at concurrency 8, plus Jev's 5.6 s), Luna cost per episode from the router's usage accounting, retries and unanswered targets.
- Budget: estimate first from the token counts (about 25k input tokens per request, 6 to 13 requests per episode); cap $4 across both cutoffs. If medium effort lands more than 1.5 SP under the 84.06 ceiling at 25%, run the 25% cutoff once more at high effort before writing up, inside the same cap.

Done when: the write-up exists with every number above, the three baselines reproduce (80.47, 83.71, 86.47) in the same script run, and the spend is stated.

## Build B: prompt breakup, Jev-only

Two scripts. `scripts/jev_real/roughcut_jev_features.py` asks the questions and stores probabilities; `scripts/jev_real/roughcut_jev_combine.py` fits and scores the combiner offline.

### Questions

Prompt bundle version `f1`, added to `scripts/jev_real/roughcut_jev_prompts.py` next to v1 to v3. Same state as v3 (`{rules, transcript, targets}`, built by the existing block builder in `roughcut_jev.py`: 25 target sentences per block, whole transcript when it fits, else the token-fitted window). Every question is a yes/no Choice per target sentence, phrased as one look at that sentence with the transcript in view; each returns the probability of yes. Questions run independently and cannot see each other's answers, which is the point. The eighteen, with the short key stored per sentence:

1. `false_start`: the speaker abandons this sentence and restarts the same thought right after.
2. `retake_loser`: this is one attempt at a line that is said again nearby, and it is not the best attempt.
3. `crew_talk`: this is addressed to the editor, producer or crew, not the students.
4. `screen_ops`: this is about operating the screen, software or recording.
5. `pre_lesson`: this is chatter before the lesson has started.
6. `off_topic`: this is off the lesson's topic.
7. `repeats_point`: this repeats a point already made in the last few sentences with nothing new.
8. `pure_filler`: this is filler with no lesson content (ok, alright, yeah, so).
9. `pep_talk`: this is encouragement, praise or wrap-up with nothing new.
10. `funny`: this is funny or shows the instructor's personality.
11. `teaching_point`: this states a teaching point, a reason or a correction.
12. `essential`: the lesson would lose something if this were cut.
13. `referenced_later`: a later sentence depends on this one being heard.
14. `transition`: this is a spoken transition between students, sections or steps.
15. `describes_screen`: this only describes what is visible on screen rather than explaining it.
16. `rambling`: this is rambling or thinking aloud.
17. `split_fragment`: this is half of one spoken sentence that continues in the next row or from the previous one.
18. `tangent`: this is a tangent from the lesson.

Each question's instructions are two or three sentences and quote the relevant rules5 line where one exists. Stan may edit the list; the bundle is versioned so an edited list becomes `f2`.

### Request shape

The v3 sentence request already sits near the 40k-estimated-token cap, so the feature questions go in their own requests: per block, as many requests as needed to fit the cap, each carrying the same state and a subset of the 18 questions for all 25 sentences (one yes/no question is about 4.5k real tokens per block, the state 12 to 28k, so expect two or three requests per block). Concurrency 8, same retry, repair and disk-logging discipline as `roughcut_jev.py`. Do not re-ask v3's score, cut, first, last or retake questions; their answers are already in `roughcut-jev-all18-v3-decisions.jsonl` and `roughcut-jev-heldout-v3-decisions.jsonl` (greco-2.2-thumbnailing is in the held-out file) and are joined by episode and sentence id.

Output: `docs/jev-real/roughcut-jev-f1-fit-features.jsonl` (fit six) and `roughcut-jev-f1-heldout-features.jsonl` (13 held-out), one row per sentence: episode, id, the 18 probabilities, and the code features below. Requests and timing files alongside. Expect about double the v3 cost and time per episode (about $0.08 and 10 s); estimate from the first block before running an episode, cap $3 for both sets.

### Code features, free

Computed in `roughcut_jev_features.py` from the corpus and the removals cache, per sentence: word count after um stripping, pause before and after (seconds, from raw word bounds), duration and words per second, position in the episode (fraction), `is_retake`, retake group winner flag, trail-off marker, lowercase continuation, split-chain piece index, um count detected and removed, mean ASR confidence, word overlap with the previous and next sentence. Plus, joined from the v3 decisions: `score`, `cut_p`, `first_p_whole`, `last_p_whole`, `cut_retake`, `retake_real`.

### Combiner

`roughcut_jev_combine.py`, no model calls. Target per sentence: editor kept (full or partial) versus removed, from the scoring module's human states. Model: `sklearn.linear_model.LogisticRegression` on standardised features, L2 with C chosen by leave-one-episode-out over the six fit episodes; sklearn 1.9 and numpy 2.4 are installed. Feature sets, each fitted and reported:

- `q`: the 18 question probabilities only.
- `q+code`: plus the code features.
- `q+code+v3`: plus the v3 score, cut_p and the trim fields.
- `v3`: the v3 score alone through the same fitting path, the control that says whether the breakup adds anything.

Turning the combiner into an arm: synthetic `score = 5 * p_keep`, `keep_words` null, `cut_retake` from the v3 row (the retake pass is unchanged this round, so the experiment isolates sentence judgment), then the standard pooled threshold calibration and the with-modules rescoring the report script does for any arm. Leave-one-episode-out SP on the fit six decides the feature set and C. Freeze, then run the questions on the 13 held-out episodes, score with the frozen weights, and place the arm on the 18-episode ladder next to jev_a v3 (80.47).

Write-up `docs/jev-real/roughcut-jev-f1.md`: per-episode and pooled SP/WORD/GRADE for every feature set on fit (leave-one-out) and held-out, the ladder placement, the standardised weights sorted by size with one line on what each says, the confusion table against the editor and against jev_a v3, each question's AUC alone against the editor's keep/cut, seconds and dollars per episode, and the calibrated-probability agreement by quartile (the input route 2 needs). Also report what a `q+code+v3` combiner would hand route 2: the margin-ranked bottom 25% by the combiner's probability, substituted with archived Luna decisions, rescored, next to the 84.06 the v3 margin gives.

Done when: both feature files and the write-up exist, the fit-set numbers are leave-one-episode-out, held-out numbers use frozen weights and say so, and the spend is stated.

## Shared discipline

- Nothing under either solar-sailer checkout is written. Prompts read from there are read at run time.
- Every request and answer to disk; result files refuse overwrite; every table carries seconds per episode; compare on the with-modules column.
- Spend is reported per build and summed in the thread. Round-two cap for both builds together: $8 of the $20 standing budget.

## Round two, second pass (approved by Stan 2026-09-26: "yes all 3")

Results of the first pass, all with modules on the 18 ladder episodes: build A live Luna on the v3 margin bottom 25% 83.63 SP (ceiling 84.06), 50% 83.69; build B f1 combiner `q+code+v3` 82.91 (fit six leave-one-out 85.63 against the v3 control 82.86, held-out 13 81.17 against 76.74). Write-ups `docs/jev-real/roughcut-hybrid-luna.md` and `roughcut-jev-f1.md`. Three follow-ups.

### Step 1: Luna keep rule, held out properly ($0)

Build A stored Luna's 0-5 score next to its keep/cut decision. Keeping at score >= 2 instead of the decision field scored 84.05 at 25% and 84.61 at 50%, but that threshold was read off all 18. Redo it with the discipline: choose the rule (decision field, or score >= 1, 2, 3, 4) on the six fit episodes of the m046 and m080 runs, freeze it, report the 12 held-out and the 18 pooled with that rule, next to the decision-field numbers. Add the result to `roughcut-hybrid-luna.md` as a new section, and use the chosen rule as the default for step 2. Script: extend `roughcut_hybrid_luna.py` with an offline `--keep-rule` sweep that makes no calls.

### Step 2: the stack, real run

Selection from the f1 combiner instead of the v3 score: margin `abs(5 * p_keep - 3.00)` over the 18 ladder episodes (fit six from their leave-one-out predictions, held-out 12 from the frozen weights, exactly the `p_keep` the f1 write-up's route-2 section used), one global cutoff giving the bottom 25% pooled. The Jev side of the arm is the f1 combiner's keep/cut, not v3's. Luna call identical to build A (whole transcript, rules5, medium effort, groups of up to 40). Output files `roughcut-hybrid-f1luna-m25-*`. Report with the decision field and with the rule step 1 froze, next to the offline ceiling 84.81 and build A's 83.63, on the ladder, with seconds and dollars per episode. Cap $1.

### Step 3: question list f2

Bundle `f2` in `roughcut_jev_prompts.py`, run and fitted exactly as f1 (fit six, leave-one-episode-out, freeze, then held-out 13), with f1 kept as the comparison arm at its frozen weights. Drop the four questions with no signal alone (`funny`, `referenced_later`, `describes_screen`, `split_fragment`; `split_fragment` is covered by the code features). Keep the other fourteen unchanged. Add eight aimed at the largest remaining misses (teaching lines the editor cut because the drawing already shows it, encouragement and wrap-up the editor tightens, scripted lines Jev reads as filler):

1. `play_by_play`: this narrates the instructor's own hand action as it happens ("I'm going to put a line here", "let me just darken this") without giving a reason a viewer could not see.
2. `said_earlier`: the same point was already made earlier in this episode, anywhere, not only in the last few sentences.
3. `wrap_up`: this closes a section or the episode ("so that's the arm", "alright, moving on") and adds nothing new.
4. `praise_only`: this praises a student's work with no correction, reason or next step.
5. `verbal_check`: this is a check on the listener ("right?", "you know?", "does that make sense?") or a hedge with no content of its own.
6. `scripted`: this reads like a prepared or scripted lesson line rather than spontaneous talk.
7. `sets_up_next`: this short line exists only to set up the sentence that follows, and the next sentence would land oddly without it.
8. `student_address`: this names or addresses a specific student or their drawing in a critique.

Feature sets to fit and report: `q2` (22 questions), `q2+code`, `q2+code+v3`, plus f1's chosen set through the same path as the control. Choose on leave-one-out, freeze, then held-out; ladder placement next to f1 (82.91) and v3 (80.47). Report each new question's AUC alone and whether the four dropped questions cost anything (refit f1's set without them on the fit six). Also hand route 2 the same offline number f1 did (combiner-margin bottom 25% with archived Luna). Cap $1.50.
