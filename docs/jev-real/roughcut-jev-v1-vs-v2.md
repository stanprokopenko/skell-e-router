# Jev rough cut: prompt v1 against prompt v2

Developer-facing. Both runs are the six fit episodes, 2,743 sentences, arm `jev_a` at trim trigger 0.3 with um removal and delete silence layered on, each calibrated by the same pooled sweep. Both landed on keep threshold 2.40, so every number below is read off the same cut point.

Sources: `docs/jev-real/roughcut-jev-v1-summary.json` and `docs/jev-real/roughcut-jev-misses.json` for v1, `docs/jev-real/roughcut-jev-v2-summary.json` and `docs/jev-real/roughcut-jev-v2-misses.json` for v2. Prompt text: `scripts/jev_real/roughcut_jev_prompts.py`, `PROMPTS["v1"]` and `PROMPTS["v2"]`.

## What v2 changed

Only the score question. The retake pass, the head and tail trim questions and the trim-pick prompt are byte-identical.

1. Six rewritten levels. Level 0 now names a false start as something the speaker *restarts*, and separates crew talk from teaching. Level 1 is restricted to rows with no lesson content at all, explicitly not to a short line that sets up the next one. Level 3 was widened to cover transitions between students, scripted set-up lines, list items read one per row, and long loosely worded teaching talk.
2. A new clause in the score instructions pointing at `targets[k].spoken_sentence`, plus the note that a row ending in `..` is a transcriber's mark, not proof of a false start.
3. New `spoken_sentence` and `piece` fields on any target the transcript split across rows. The pipeline detects these in code: a row whose corpus text ends in `..` joins the next row when that row opens with a lowercase letter, and chains follow. Across the six episodes that flags 504 rows in 236 chains, 18% of all sentences. The transcript rendering is untouched; the flags ride on the target only.

The trim-pick pass (`--trim-pick`) was off for the v2 run, so v2 has no `jev_b` arm. `jev_a` is unaffected by that pass in either version, so the headline comparison is unchanged by it.

## Per episode, jev_a at t_trim 0.3, layered

| episode | SP v1 | SP v2 | SP Δ | WORD v1 | WORD v2 | WORD Δ | GRADE v1 | GRADE v2 | GRADE Δ | Luna SP | Luna WORD | Opus SP | Opus WORD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 83.66 | 85.36 | +1.70 | 73.69 | 73.88 | +0.19 | 105.79 | 105.60 | -0.19 | 82.47 | 76.08 | 86.49 | 76.10 |
| hampton-5.4-assignment-demo | 86.33 | 85.63 | -0.70 | 74.99 | 76.15 | +1.16 | 94.39 | 94.94 | +0.55 | 90.57 | 85.60 | 94.60 | 89.47 |
| colman-03.03-muscles-crit | 75.41 | 74.95 | -0.46 | 62.22 | 58.74 | -3.48 | 74.96 | 72.77 | -2.19 | 69.01 | 68.34 | 73.63 | 66.40 |
| edges-7.01-intro | 77.56 | 78.56 | +1.00 | 74.27 | 78.98 | +4.71 | 75.18 | 79.35 | +4.17 | 89.59 | 89.34 | 90.39 | 89.71 |
| hampton-5.2-shape-demo | 92.04 | 91.31 | -0.73 | 84.30 | 83.75 | -0.55 | 95.69 | 95.16 | -0.53 | 90.78 | 81.60 | 95.06 | 87.60 |
| perspective-14e-boxes-critique | 84.62 | 81.40 | -3.22 | 80.29 | 79.60 | -0.69 | 83.29 | 82.74 | -0.55 | 89.01 | 85.03 | 90.54 | 86.36 |
| **pooled** | **83.84** | **82.52** | **-1.32** | **76.46** | **76.31** | **-0.15** | **86.41** | **86.40** | **-0.01** | | | | |

Luna is `luna-chapters-rules5` and Opus is `opus5-cc-agentic`, both quoted from the misses doc's own per-episode reference columns. They are fixed references and do not move between the two runs.

The pooled kept ratio moves from 89.63 to 96.11. v2 keeps almost everything, which is the whole story below.

## Sentence-level confusion, pooled over 2,743 sentences

| | both keep | Jev kept, editor removed | Jev removed, editor kept | of which editor full | of which editor partial | both removed | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 1,234 | 157 | 271 | 216 | 55 | 1,081 | 84.40 |
| v2 | 1,306 | 255 | 199 | 157 | 42 | 983 | 83.45 |
| Luna chapters | 1,286 | 122 | 219 | 167 | 52 | 1,116 | 87.57 |

v2 traded 72 wrong drops for 98 wrong keeps. The over-cutting v1 was criticised for is genuinely reduced, and v2 lands closer to Luna on the drop side (199 against Luna's 219) while being twice as loose on the keep side (255 against 122).

Per episode, the same two error columns:

| episode | v1 wrong drops | v2 wrong drops | v1 wrong keeps | v2 wrong keeps |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 19 | 8 | 7 | 11 |
| hampton-5.4-assignment-demo | 19 | 11 | 19 | 30 |
| colman-03.03-muscles-crit | 35 | 30 | 17 | 18 |
| edges-7.01-intro | 71 | 52 | 17 | 33 |
| hampton-5.2-shape-demo | 23 | 17 | 23 | 32 |
| perspective-14e-boxes-critique | 104 | 81 | 74 | 131 |

## Did the two targeted patterns move

**Split spoken sentences.** Counted in code, over every row the chain detector flags, not over the hand-labelled worst-40 list:

| episode | chain rows | v1 wrongly dropped | v2 wrongly dropped |
|---|---:|---:|---:|
| colman-02.04-skeleton-demo | 44 | 5 | 1 |
| hampton-5.4-assignment-demo | 61 | 8 | 5 |
| colman-03.03-muscles-crit | 101 | 16 | 13 |
| edges-7.01-intro | 73 | 9 | 7 |
| hampton-5.2-shape-demo | 72 | 6 | 4 |
| perspective-14e-boxes-critique | 153 | 8 | 2 |
| **total** | **504** | **52** | **32** |

A 38% cut, and every episode improved. This is the one fix that clearly worked.

**Scripted set-up lines.** The misses script labels patterns by sentence id, so the v1 list can be re-checked against v2 directly. Of v1's six hand-labelled scripted set-up drops, five are still dropped in v2. Of v1's eleven hand-labelled split fragments, seven are still dropped. The hand-labelled list is the hardest tail of each pattern, so it moves less than the code-detected population; the level-3 rewrite did not rescue "Sharp, firm, soft, and lost." or "When do you use a firm edge?".

The other v1 patterns, re-checked the same way: ordinary connective talk 9 of 9 still dropped in v2, retake pair wrong side 6 of 6, producer talk and student names 5 of 5, thinking aloud 2 of 2. The one joke v1 dropped, v2 keeps.

## Where the level mass went

Mean probability per level, split by what the editor actually did:

| level | v1 kept | v2 kept | v1 removed | v2 removed |
|---|---:|---:|---:|---:|
| 0 | 0.144 | 0.132 | 0.470 | 0.397 |
| 1 | 0.057 | 0.030 | 0.120 | 0.080 |
| 2 | 0.035 | 0.027 | 0.062 | 0.057 |
| 3 | 0.249 | 0.337 | 0.139 | 0.260 |
| 4 | 0.420 | 0.399 | 0.166 | 0.172 |
| 5 | 0.095 | 0.074 | 0.042 | 0.035 |

Level 3 nearly doubled on material the editor removed, 0.139 to 0.260, and grew by a third on material the editor kept. The widened level 3 pulled from level 0 and level 1 on both sides of the split, so it bought back real drops and paid for them with almost as many keeps. That is why SP fell while WORD and GRADE held.

## Speed and cost

Wall clock per episode, at concurrency 8. v1 ran three passes, v2 ran two (no trim pick), so the sentence-pass column is the fair comparison.

| episode | v1 retake | v1 sentence | v1 trim pick | v1 total | v2 retake | v2 sentence | v2 total |
|---|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0.35 | 1.35 | 0.67 | 2.37 | 0.30 | 0.72 | 1.01 |
| hampton-5.4-assignment-demo | 0.23 | 1.56 | 0.64 | 2.42 | 0.17 | 4.41 | 4.58 |
| colman-03.03-muscles-crit | 0.22 | 1.79 | 0.90 | 2.90 | 0.19 | 2.65 | 2.84 |
| edges-7.01-intro | 0.52 | 1.64 | 0.69 | 2.85 | 0.46 | 4.40 | 4.87 |
| hampton-5.2-shape-demo | 0.25 | 2.16 | 0.95 | 3.35 | 0.25 | 2.95 | 3.19 |
| perspective-14e-boxes-critique | 0.76 | 5.88 | 2.19 | 8.83 | 0.49 | 9.32 | 9.81 |
| **mean per episode** | | | | **3.79** | | | **4.38** |

v2 is slower per episode despite dropping a whole pass, on half the requests (167 against 323). The extra time is retries: v2 took 22 retries over 167 requests where v1 took 22 over 323, and a retried block waits out `RETRY_BACKOFF_S` inside the pass's wall clock. Both runs are well inside the 10 second per episode target except perspective-14e, which is 1,146 sentences.

Cost: v1 $0.2325, v2 $0.2202 including the one repaired block. Input tokens 5.53M against 5.24M. The `spoken_sentence` fields add about 20k estimated input tokens across all six episodes, under $0.001.

## Run health

v2: 167 requests, 23 errors, 22 retries, 1 block that failed all three attempts (perspective-14e sentence block 12) and was recovered by `--repair roughcut-jev-v2` in one request. Zero sentences ended without an answer.

## Verdict

v2 fixes what it was aimed at and loses on the trade. Split fragments dropped 52 to 32 and total wrong drops fell 271 to 199, but wrong keeps rose 157 to 255, so pooled SP fell 1.32 to 82.52. WORD and GRADE are flat, which says the words v2 newly keeps are cheap in frames but expensive in sentence points.

The fault is level 3, not the split-sentence flags. Those are clean, cheap and per-episode monotone. A v3 should keep the `spoken_sentence` work and the level 0 and level 1 rewrites, and narrow level 3 back towards v1's wording, keeping only the transition and scripted set-up clauses that the misses list actually asked for. Dropping the "even when plain, long or loosely worded" phrase is the first thing to try: the critique episodes, where that phrasing has the most material to grab, are exactly where v2 lost the most (perspective-14e, -3.22 SP, wrong keeps 74 to 131).
