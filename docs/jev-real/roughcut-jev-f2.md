Developer-facing notes on build B of the Jev rough-cut round two: the prompt breakup, 22 yes/no questions per sentence (bundle `f2`) and a logistic combiner fitted in code, Jev only.

Generated 2026-09-26T15:27:11+00:00 by `scripts/jev_real/roughcut_jev_combine.py` from `roughcut-jev-f2-fit-features.jsonl` and `roughcut-jev-f2-heldout-features.jsonl`, control rows from `roughcut-jev-f1-fit-features.jsonl` and `roughcut-jev-f1-heldout-features.jsonl`, weights in `roughcut-jev-f2-weights.json`.

# Jev rough cut, build B: prompt breakup (f2)

The v3 sentence pass asks one six-level score per sentence. This build asks 22 one-look yes/no questions instead (bundle `f2` in `roughcut_jev_prompts.py`), over the same state v3 sent, and fits an L2 logistic regression on the probabilities of yes. Sentence judgment is the only thing that changes: `keep_words` is null, the retake cut is jev_a v3's. Every number below is with um removal and delete silence layered on. Fit-set numbers are leave-one-episode-out over the 6 fit episodes with the keep threshold calibrated on the pooled out-of-fold predictions. C and the feature set were chosen on those numbers alone, then frozen. Bundle `f2` is `f1` with 4 questions dropped (`funny`, `referenced_later`, `describes_screen`, `split_fragment`), the other 14 unchanged, and 8 added (`play_by_play`, `said_earlier`, `wrap_up`, `praise_only`, `verbal_check`, `scripted`, `sets_up_next`, `student_address`). `f1`'s chosen set `q+code+v3` runs through the same fitting path as the control, from its own feature run, and its held-out and ladder rows use its own frozen weights.

Bottom line: the chosen set is `q2+code+v3` (C 0.01, threshold 3.20). Leave-one-out on the fit six it scores 84.15 SP against 82.86 for the v3 score through the same fitting path, 85.63 for `f1`'s `q+code+v3` through the same path (84.66 with the dropped questions removed) and 83.00 for jev_a v3 as published on the same six. On the 13 held-out episodes with the frozen weights and threshold it scores 80.67 SP against 81.17 for `f1` at its frozen weights and 76.74 for jev_a v3. On the 18-episode ladder it lands at 82.03 next to `f1`'s 82.91 and jev_a v3's 80.47 (below Jev f1 `q+code+v3` combiner (frozen), above Jev jev_a v3 (pure Jev)). Spend $1.30 in Jev calls, 8.2 s per ladder episode with the v3 pass included. Routing the bottom 25% by combiner margin to archived Luna gives 84.51 against 84.81 for `f1` and 84.06 for the v3 margin.

## Fit set, leave-one-episode-out

Pooled over the 6 fit episodes, every feature set at its best C. `v3` is the control: jev_a v3's 0-5 score alone through the same fitting path. `code+v3` is a diagnostic set outside the spec's four, there to show what the questions add over the free features and v3 together. `f1 q+code+v3` is `f1`'s chosen set refitted through the same path from its own feature run, the comparison arm; `f1 q+code+v3 minus dropped` is that set without the 4 questions `f2` dropped, the ablation. Neither was eligible for selection. Seconds per episode are the f2 pass plus the v3 pass it joins (both at concurrency 8).

| feature set | features | C | threshold | SENTENCE POINTS | WORD SCORE | GRADE | LOO AUC | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q2` | 22 | 0.01 | 2.50 | 82.81 | 78.26 | 87.60 | 0.882 | 7.2 |
| `q2+code` | 41 | 0.01 | 3.10 | 84.04 | 78.84 | 88.56 | 0.902 | 7.2 |
| `q2+code+v3` (chosen) | 45 | 0.01 | 3.20 | 84.15 | 78.84 | 88.44 | 0.906 | 7.2 |
| `v3` | 4 | 0.3 | 3.20 | 82.86 | 76.16 | 86.32 | 0.881 | 7.2 |
| `code+v3` (diagnostic) | 23 | 3 | 3.60 | 81.71 | 75.84 | 86.22 | 0.883 | 7.2 |
| `f1 q+code+v3` (control) | 41 | 0.003 | 3.00 | 85.63 | 79.79 | 89.34 | 0.922 | 7.2 |
| `f1 q+code+v3 minus dropped` (ablation) | 37 | 0.003 | 3.00 | 84.66 | 78.93 | 88.53 | 0.915 | 7.2 |
| jev_a v3 as published (trims at 0.3, calibrated) |  |  | 2.50 | 83.00 | 76.78 | 86.72 |  | 5.1 |
| jev_a v3, keep_words null (calibrated) |  |  | 2.50 | 82.97 | 76.61 | 86.56 |  | 5.1 |

C sweep, leave-one-out SP per feature set (the chosen C is the best; ties go to the smaller C):

| feature set | C 0.001 | C 0.003 | C 0.01 | C 0.03 | C 0.1 | C 0.3 | C 1 | C 3 | C 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q2` | 81.87 | 82.42 | 82.81 | 82.28 | 82.79 | 82.54 | 82.54 | 82.57 | 82.53 |
| `q2+code` | 82.95 | 83.72 | 84.04 | 83.69 | 82.81 | 82.30 | 82.75 | 82.41 | 82.49 |
| `q2+code+v3` | 83.90 | 83.50 | 84.15 | 82.91 | 82.91 | 82.25 | 81.88 | 81.84 | 81.95 |
| `v3` | 78.72 | 82.10 | 82.68 | 82.75 | 82.58 | 82.86 | 82.75 | 82.75 | 82.75 |
| `code+v3` | 75.98 | 79.67 | 81.37 | 81.67 | 81.62 | 81.62 | 81.60 | 81.71 | 81.71 |
| `f1 q+code+v3` | 85.05 | 85.63 | 85.63 | 84.97 | 84.75 | 84.15 | 84.34 | 83.90 | 84.20 |
| `f1 q+code+v3 minus dropped` | 84.29 | 84.66 | 84.41 | 83.45 | 82.23 | 80.58 | 80.52 | 80.38 | 80.45 |

Per episode, leave-one-out, at each set's pooled threshold:

| episode | sentences | `q2` | `q2+code` | `q2+code+v3` | `v3` | `code+v3` | `f1 q+code+v3` | `f1 q+code+v3 minus dropped` | jev_a v3 | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 84.23 | 85.98 | 85.26 | 83.09 | 81.13 | 84.12 | 85.26 | 85.88 | 5.3 |
| hampton-5.4-assignment-demo | 300 | 91.77 | 89.90 | 90.57 | 86.90 | 81.77 | 91.53 | 89.57 | 84.80 | 4.1 |
| colman-03.03-muscles-crit | 303 | 79.93 | 77.43 | 76.47 | 73.93 | 67.72 | 77.66 | 77.03 | 75.05 | 4.5 |
| edges-7.01-intro | 389 | 80.28 | 83.98 | 83.37 | 77.30 | 77.71 | 84.91 | 83.62 | 79.69 | 6.5 |
| hampton-5.2-shape-demo | 411 | 96.06 | 96.06 | 95.57 | 93.28 | 91.19 | 96.25 | 96.25 | 92.70 | 5.0 |
| perspective-14e-boxes-critique | 1146 | 77.09 | 79.62 | 80.47 | 82.27 | 83.46 | 82.88 | 81.48 | 81.78 | 17.8 |

Ablation: `f1`'s `q+code+v3` refitted on the fit six without `funny`, `referenced_later`, `describes_screen`, `split_fragment` scores 84.66 SP leave-one-out against 85.63 with them, a change of -0.97 SP (LOO AUC 0.915 against 0.922). Dropping them cost something on the fit six; see the notes.

## Held-out, frozen weights

The 13 held-out episodes scored with the weights and the keep threshold frozen after stage 1. Nothing here was fitted, chosen or calibrated on these episodes. The last column recalibrates the threshold on the held-out set itself and is not held out; it is there to show how much the frozen threshold costs. `f1 q+code+v3` uses `f1`'s own frozen weights and threshold (stage 1 reproduced them exactly) on `f1`'s held-out feature rows.

| feature set | threshold | SENTENCE POINTS | WORD SCORE | GRADE | AUC | SP recalibrated (not held out) | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q2` | 2.50 | 80.53 | 74.09 | 87.95 | 0.882 | 80.47 at t=2.20 | 9.5 |
| `q2+code` | 3.10 | 80.67 | 74.63 | 88.47 | 0.899 | 80.97 at t=2.70 | 9.5 |
| `q2+code+v3` (chosen) | 3.20 | 80.67 | 74.60 | 88.47 | 0.902 | 81.65 at t=2.60 | 9.5 |
| `v3` | 3.20 | 77.81 | 71.76 | 85.56 | 0.858 | 77.79 at t=3.00 | 9.5 |
| `code+v3` (diagnostic) | 3.60 | 78.51 | 73.66 | 87.20 | 0.872 | 78.98 at t=3.20 | 9.5 |
| `f1 q+code+v3` (control) | 3.00 | 81.17 | 74.73 | 88.87 | 0.903 | 81.54 at t=2.70 | 9.5 |
| `f1 q+code+v3 minus dropped` (ablation) | 3.00 | 80.62 | 74.31 | 88.39 | 0.895 | 80.85 at t=2.80 | 9.5 |
| jev_a v3 as published (trims at 0.3, t 2.50) | 2.50 | 76.74 | 72.14 | 86.08 |  |  | 6.4 |

Per episode, frozen weights:

| episode | sentences | `q2` | `q2+code` | `q2+code+v3` | `v3` | `code+v3` | `f1 q+code+v3` | `f1 q+code+v3 minus dropped` | jev_a v3 | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 1752 | 82.93 | 83.41 | 83.24 | 80.55 | 83.38 | 84.61 | 84.30 | 78.09 | 21.5 |
| hampton-5.5-crit1 | 181 | 90.50 | 89.94 | 90.50 | 85.52 | 82.76 | 89.94 | 89.94 | 87.35 | 3.2 |
| hampton-5.5-crit2 | 127 | 91.42 | 90.63 | 89.84 | 79.92 | 68.11 | 89.84 | 89.84 | 79.45 | 1.6 |
| hampton-5.5-crit3 | 137 | 83.21 | 82.48 | 83.21 | 80.29 | 78.83 | 82.48 | 83.94 | 81.17 | 1.7 |
| hampton-5.5-crit4 | 156 | 91.47 | 92.12 | 92.63 | 88.27 | 81.86 | 93.40 | 93.40 | 88.46 | 1.6 |
| hampton-5.5-crit5 | 295 | 91.49 | 91.49 | 91.49 | 89.46 | 87.42 | 91.49 | 91.49 | 90.88 | 6.2 |
| flanders-03-thematic-crit | 1309 | 76.95 | 73.64 | 75.28 | 75.57 | 75.05 | 76.95 | 75.10 | 77.14 | 27.3 |
| anatomy-30b-hamstring-crit | 951 | 84.10 | 84.71 | 84.68 | 83.71 | 82.44 | 85.16 | 85.66 | 84.90 | 19.2 |
| colman-04.03-life-crit | 495 | 78.55 | 78.34 | 77.33 | 74.71 | 73.21 | 76.93 | 76.93 | 76.18 | 8.2 |
| colman-05.02-master-studies-crit | 381 | 66.43 | 71.94 | 70.89 | 65.91 | 68.35 | 69.06 | 69.84 | 66.33 | 6.8 |
| colman-06.06-species-crit | 373 | 77.80 | 77.27 | 76.09 | 76.41 | 74.02 | 75.76 | 75.23 | 79.57 | 4.6 |
| hampton-7-conclusion | 43 | 83.95 | 83.95 | 83.95 | 79.30 | 81.63 | 79.30 | 83.95 | 72.33 | 2.3 |
| greco-2.2-thumbnailing | 1388 | 77.49 | 79.23 | 78.78 | 72.02 | 76.80 | 78.79 | 77.20 | 65.09 | 19.2 |

## Ladder, 18 episodes

The fit six enter with their leave-one-out predictions and the 12 ladder held-out episodes with the frozen weights, all at the frozen threshold 3.20; greco-2.2-thumbnailing is not a ladder episode and is left out here. jev_a v3 is rebuilt through the same scoring path and reproduces its published 80.47 at 80.47. `f1`'s row is built the same way from its own rows at its frozen threshold 3.00. Seconds per episode: 8.2 (f2 plus v3).

| arm | SENTENCE POINTS | s/episode |
|---|---:|---:|
| shipped Opus agentic | 86.47 |  |
| best Luna chapters | 83.71 |  |
| Jev f1 `q+code+v3` combiner (frozen) | 82.91 | 7.9 |
| Jev f2 `q2+code+v3` combiner (this build) | 82.03 | 8.2 |
| Jev jev_a v3 (pure Jev) | 80.47 |  |
| Luna single call | 65.36 |  |
| deterministic baseline (um removal + retakes + delete silence) | 63.72 |  |

## Standardised weights, chosen set

`q2+code+v3` refitted on all 6 fit episodes with C 0.01, sorted by size. A positive weight pushes toward keep. Weights are per standard deviation of the feature, so they compare across features. Intercept 0.227.

| feature | weight | what it says |
|---|---:|---:|
| `off_topic` | -0.465 | yes: off the lesson's topic; pushes toward cut |
| `v3_score` | +0.431 | v3 0-5 score; pushes toward keep |
| `pre_lesson` | -0.403 | yes: chatter before the lesson starts; pushes toward cut |
| `screen_ops` | -0.309 | yes: operating the screen or software; pushes toward cut |
| `crew_talk` | -0.292 | yes: addressed to the crew, not students; pushes toward cut |
| `is_retake` | -0.257 | corpus retake flag (module loser); pushes toward cut |
| `asr_confidence` | +0.249 | mean ASR word confidence; pushes toward keep |
| `duration_s` | +0.249 | spoken duration in seconds; pushes toward keep |
| `essential` | +0.226 | yes: the lesson loses something without it; pushes toward keep |
| `pure_filler` | -0.206 | yes: filler with no content; pushes toward cut |
| `play_by_play` (new) | +0.183 | yes: narrates the hand action with no reason; pushes toward keep |
| `transition` | +0.177 | yes: a spoken transition between students or steps; pushes toward keep |
| `teaching_point` | +0.172 | yes: states a point, reason or correction; pushes toward keep |
| `overlap_next` | -0.169 | word overlap with the next sentence; pushes toward cut |
| `trail_off` | -0.167 | row ends in the transcriber's '..' mark; pushes toward cut |
| `repeats_point` | -0.157 | yes: repeats a point just made; pushes toward cut |
| `position` | -0.152 | position in the episode, 0 to 1; pushes toward cut |
| `v3_cut_p` | -0.148 | v3 P(editor removes it); pushes toward cut |
| `verbal_check` (new) | +0.145 | yes: 'right?', a hedge, no content of its own; pushes toward keep |
| `v3_last_p_whole` | -0.138 | v3 P(nothing trimmed from the end); pushes toward cut |
| `retake_loser` | -0.134 | yes: a losing take of a repeated line; pushes toward cut |
| `retake_member` | -0.132 | in a retake group; pushes toward cut |
| `retake_winner` | +0.124 | the module's winning take; pushes toward keep |
| `student_address` (new) | -0.123 | yes: names or addresses a student or their drawing; pushes toward cut |
| `chain_piece` | -0.104 | piece index in a split-sentence chain; pushes toward cut |
| `rambling` | -0.090 | yes: rambling or thinking aloud; pushes toward cut |
| `n_words` | +0.087 | word count after um stripping; pushes toward keep |
| `n_words_raw` | +0.085 | word count as spoken; pushes toward keep |
| `wrap_up` (new) | -0.084 | yes: closes a section with nothing new; pushes toward cut |
| `words_per_s` | -0.078 | speaking rate; pushes toward cut |
| `pep_talk` | +0.058 | yes: praise or wrap-up with nothing new; pushes toward keep |
| `um_removed` | -0.044 | words the um module removed; pushes toward cut |
| `tangent` | -0.036 | yes: an aside the lesson resumes after; pushes toward cut |
| `chain_len` | +0.035 | length of the split-sentence chain; pushes toward keep |
| `false_start` | -0.030 | yes: the row is an abandoned attempt; pushes toward cut |
| `sets_up_next` (new) | -0.028 | yes: exists to set up the next sentence; pushes toward cut |
| `um_detected` | -0.024 | um-like words as spoken; pushes toward cut |
| `praise_only` (new) | +0.023 | yes: praise with no correction or reason; pushes toward keep |
| `pause_before` | -0.020 | seconds of silence before the sentence; pushes toward cut |
| `overlap_prev` | +0.017 | word overlap with the previous sentence; pushes toward keep |
| `lower_start` | +0.015 | row starts lowercase (continuation); pushes toward keep |
| `said_earlier` (new) | -0.005 | yes: the point was made earlier in the episode; pushes toward cut |
| `v3_first_p_whole` | -0.004 | v3 P(nothing trimmed from the start); pushes toward cut |
| `pause_after` | +0.004 | seconds of silence after the sentence; pushes toward keep |
| `scripted` (new) | -0.003 | yes: reads like a prepared lesson line; pushes toward cut |

## Each question alone

AUC of each probability of yes against the editor's keep (full or partial) versus removed, pooled over the fit six and, when present, the held-out episodes. 0.50 is no signal; a cut question reads below 0.50 and a keep question above. The v3 score and cut_p are listed on the same footing. Questions marked new were added in `f2`; the dropped `f1` questions are listed below the table from `f1`'s own rows.

| question | AUC fit | AUC held-out | abs(AUC - 0.5) fit | signal |
|---|---:|---:|---:|---:|
| `v3_score` | 0.888 | 0.845 | 0.388 | yes |
| `v3_cut_p` | 0.134 | 0.178 | 0.366 | yes |
| `off_topic` | 0.173 | 0.176 | 0.327 | yes |
| `crew_talk` | 0.174 | 0.160 | 0.326 | yes |
| `pre_lesson` | 0.185 | 0.167 | 0.315 | yes |
| `pure_filler` | 0.209 | 0.184 | 0.291 | yes |
| `essential` | 0.781 | 0.766 | 0.281 | yes |
| `screen_ops` | 0.239 | 0.279 | 0.261 | yes |
| `teaching_point` | 0.729 | 0.753 | 0.229 | yes |
| `repeats_point` | 0.284 | 0.340 | 0.216 | yes |
| `tangent` | 0.285 | 0.320 | 0.215 | yes |
| `retake_loser` | 0.291 | 0.358 | 0.209 | yes |
| `rambling` | 0.322 | 0.404 | 0.178 | yes |
| `false_start` | 0.323 | 0.396 | 0.177 | yes |
| `said_earlier` (new) | 0.339 | 0.373 | 0.161 | yes |
| `wrap_up` (new) | 0.379 | 0.329 | 0.121 | yes |
| `pep_talk` | 0.396 | 0.352 | 0.104 | yes |
| `scripted` (new) | 0.592 | 0.513 | 0.092 | weak |
| `sets_up_next` (new) | 0.414 | 0.397 | 0.086 | weak |
| `praise_only` (new) | 0.428 | 0.535 | 0.072 | weak |
| `verbal_check` (new) | 0.433 | 0.406 | 0.067 | weak |
| `student_address` (new) | 0.445 | 0.545 | 0.055 | weak |
| `transition` | 0.542 | 0.450 | 0.042 | weak |
| `play_by_play` (new) | 0.537 | 0.479 | 0.037 | weak |

Questions with no signal on their own (abs(AUC - 0.5) under 0.03): none.

Dropped from `f1`, AUC alone on `f1`'s rows: `funny` 0.491 fit, 0.478 held-out; `referenced_later` 0.503 fit, 0.589 held-out; `describes_screen` 0.519 fit, 0.540 held-out; `split_fragment` 0.508 fit, 0.543 held-out.

## Confusion

Keep or cut of the chosen arm against the editor and against jev_a v3, with modules, per split. Fit rows are leave-one-out; held-out rows use the frozen weights.

| split | reference | n | both keep | ref keep, f2 cut | ref cut, f2 keep | both cut | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit | editor | 2743 | 1296 | 209 | 197 | 1041 | 85.20 | 7.2 |
| fit | jev_a v3 | 2743 | 1363 | 155 | 130 | 1095 | 89.61 | 7.2 |
| heldout | editor | 7588 | 3697 | 707 | 575 | 2609 | 83.10 | 9.5 |
| heldout | jev_a v3 | 7588 | 4075 | 941 | 197 | 2375 | 85.00 | 9.5 |
| ladder18 | editor | 8943 | 4674 | 869 | 546 | 2854 | 84.18 | 8.2 |
| ladder18 | jev_a v3 | 8943 | 4907 | 850 | 313 | 2873 | 87.00 | 8.2 |

## Calibration by quartile

The chosen combiner's `p_keep` over the 18 ladder episodes (fit leave-one-out plus held-out frozen). The margin table is the input route 2 needs: the bottom margin quartile is the slice a bigger model would take.

| p_keep quartile | n | p range | mean p_keep | editor kept | arm kept | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2236 | 0.00 to 0.31 | 0.116 | 10.29 | 0.04 | 89.67 | 8.2 |
| 2 | 2235 | 0.31 to 0.77 | 0.558 | 54.05 | 34.50 | 63.18 | 8.2 |
| 3 | 2236 | 0.77 to 0.92 | 0.861 | 87.66 | 98.97 | 87.88 | 8.2 |
| 4 | 2236 | 0.92 to 1.00 | 0.957 | 95.93 | 99.96 | 95.97 | 8.2 |

| margin quartile | n | margin range | mean p_keep | editor kept | arm kept | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2236 | 0.00 to 0.96 | 0.677 | 68.25 | 61.81 | 66.19 | 8.2 |
| 2 | 2235 | 0.96 to 1.44 | 0.793 | 79.06 | 80.13 | 86.40 | 8.2 |
| 3 | 2236 | 1.44 to 1.75 | 0.885 | 88.15 | 88.42 | 93.83 | 8.2 |
| 4 | 2236 | 1.75 to 3.20 | 0.137 | 12.48 | 3.13 | 90.30 | 8.2 |

## What this hands route 2

Bottom share of the 18 ladder episodes by combiner margin `abs(5 * p_keep - threshold)`, one global cutoff, the archived Luna chapters decision substituted on the routed slice and rescored with modules, exactly as `roughcut_route2_routing.py` does for the v3 margin. The v3 margin gives 84.06 at 25%. Seconds per episode are Jev only; the Luna call is build A's number.

| combiner | share | routed | margin cutoff | SENTENCE POINTS | WORD SCORE | s/episode |
|---|---:|---:|---:|---:|---:|---:|
| `q2+code+v3` | 25% | 2236 | 0.964 | 84.51 | 78.78 | 8.2 |
| `q2+code+v3` | 50% | 4472 | 1.441 | 84.11 | 79.43 | 8.2 |
| `f1 q+code+v3` | 25% | 2236 | 0.879 | 84.81 | 78.72 | 7.9 |
| `f1 q+code+v3` | 50% | 4472 | 1.391 | 84.46 | 79.54 | 7.9 |
| v3 margin, archived Luna (route 2 write-up) | 25% | 2236 | 0.46 | 84.06 |  | 5.6 |

## Seconds and dollars per episode

The f2 pass at concurrency 8 (1 to 2 requests per 25-sentence block, every part carrying the whole state), plus the v3 run it joins. Cost is the router's usage accounting at $0.042 per million input tokens.

| episode | split | sentences | f2 requests | f2 errors | unanswered cells | f2 s | v3 s | total s | f2 $ | v3 $ | total $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | fit | 194 | 16 | 0 | 0 | 0.89 | 4.45 | 5.33 | 0.0202 | 0.0149 | 0.0351 |
| hampton-5.4-assignment-demo | fit | 300 | 24 | 0 | 0 | 1.29 | 2.79 | 4.09 | 0.0306 | 0.0213 | 0.0519 |
| colman-03.03-muscles-crit | fit | 303 | 25 | 0 | 0 | 1.39 | 3.10 | 4.48 | 0.0345 | 0.0252 | 0.0597 |
| edges-7.01-intro | fit | 389 | 31 | 0 | 0 | 1.56 | 4.90 | 6.46 | 0.0376 | 0.0275 | 0.0651 |
| hampton-5.2-shape-demo | fit | 411 | 33 | 0 | 0 | 1.82 | 3.20 | 5.02 | 0.0479 | 0.0338 | 0.0816 |
| perspective-14e-boxes-critique | fit | 1146 | 92 | 0 | 0 | 5.32 | 12.44 | 17.77 | 0.1553 | 0.1053 | 0.2606 |
| perspective-13d-critique | held-out | 1752 | 141 | 0 | 0 | 7.85 | 13.61 | 21.46 | 0.2382 | 0.1252 | 0.3635 |
| hampton-5.5-crit1 | held-out | 181 | 15 | 0 | 0 | 1.02 | 2.22 | 3.24 | 0.0196 | 0.0152 | 0.0348 |
| hampton-5.5-crit2 | held-out | 127 | 11 | 0 | 0 | 0.68 | 0.93 | 1.61 | 0.0131 | 0.0102 | 0.0233 |
| hampton-5.5-crit3 | held-out | 137 | 11 | 0 | 0 | 0.74 | 0.99 | 1.73 | 0.0143 | 0.0112 | 0.0255 |
| hampton-5.5-crit4 | held-out | 156 | 13 | 0 | 0 | 0.82 | 0.79 | 1.61 | 0.0170 | 0.0133 | 0.0304 |
| hampton-5.5-crit5 | held-out | 295 | 24 | 0 | 0 | 1.40 | 4.81 | 6.21 | 0.0336 | 0.0245 | 0.0581 |
| flanders-03-thematic-crit | held-out | 1309 | 105 | 0 | 0 | 8.08 | 19.26 | 27.33 | 0.1707 | 0.1246 | 0.2953 |
| anatomy-30b-hamstring-crit | held-out | 951 | 77 | 0 | 0 | 7.45 | 11.76 | 19.21 | 0.1268 | 0.0913 | 0.2181 |
| colman-04.03-life-crit | held-out | 495 | 40 | 0 | 0 | 2.37 | 5.88 | 8.25 | 0.0621 | 0.0420 | 0.1041 |
| colman-05.02-master-studies-crit | held-out | 381 | 31 | 0 | 0 | 1.83 | 5.01 | 6.84 | 0.0462 | 0.0329 | 0.0790 |
| colman-06.06-species-crit | held-out | 373 | 30 | 0 | 0 | 1.72 | 2.83 | 4.55 | 0.0433 | 0.0307 | 0.0741 |
| hampton-7-conclusion | held-out | 43 | 3 | 0 | 0 | 0.45 | 1.81 | 2.26 | 0.0038 | 0.0030 | 0.0068 |
| greco-2.2-thumbnailing | held-out | 1388 | 112 | 0 | 0 | 6.16 | 13.02 | 19.18 | 0.1807 | 0.1156 | 0.2963 |

Spend on this build: $0.3261 on the fit six, $0.9694 on the held-out episodes, $0.0025 on the smoke block, $1.2979 in all against the $1.50 cap. Requests: 836, errors 0, unanswered cells 0. The `f1` control rows cost nothing new; they are `f1`'s own run.

## Notes

- Target per sentence: editor kept (full or partial) versus removed, from the harness's human sentence states, the same states SENTENCE POINTS reads. Fit six: 1505 kept of 2743.
- Missing values: an unanswered noul is filled with 0.5, a missing v3 score with 2.5, a missing v3 cut_p with 0.5, a missing trim answer (one-word rows) with 1.0 (whole). Counts: {'fit': {'q_cells': 0, 'v3_score': 0, 'v3_cut_p': 0, 'sentences': 2743}, 'heldout': {'q_cells': 0, 'v3_score': 0, 'v3_cut_p': 0, 'sentences': 7588}}.
- Fit-set threshold: calibrated by the harness's pooled Neutral sweep on the out-of-fold predictions of all six episodes, so it is chosen on the fit set only and then frozen with the weights. The pooled SP in the tables is the sentence-count weighted mean of per-episode SP; the harness's own pooled figure for the chosen set is 84.15.
- The v3 control lands at 82.86 against 83.00 for jev_a v3 as published on the same six and 82.97 with keep_words null. The control has no trims and a logistic squashing of the score; the keep threshold moves accordingly.
- The questions are TypeSafe nouls (probability of yes), one per target sentence per question, the same primitive as v3's cut_k. Each block's 22 questions go out in 1 to 2 requests that all carry the full v3 state; the answers never see each other.
- Selection was among the spec's four sets only; `code+v3` is reported as a diagnostic and was not eligible, nor were the `f1` control and ablation sets.
- Control: `f1`'s chosen set `q+code+v3` refitted through this script's path from `roughcut-jev-f1-fit` reproduces the weights, C and threshold frozen in `roughcut-jev-f1-weights.json` (largest coefficient difference 0.0e+00). Its held-out and ladder rows use that frozen file, so they are the same arm `f1`'s write-up reports.
- Ablation: removing `funny`, `referenced_later`, `describes_screen`, `split_fragment` from `q+code+v3` and refitting on the fit six moves leave-one-out SP by -0.97 (85.63 to 84.66), C 0.003 to 0.003.
- Held-out numbers use the frozen weights and the frozen threshold. Nothing was refitted, recalibrated or chosen after the held-out features were read; stage 2 refuses to run if the stage 1 recomputation drifts from the frozen file. The control and ablation sets were frozen at stage 1 in the same file.
- Against `f1` at its frozen weights: held-out 80.67 versus 81.17 (-0.50), ladder 82.03 versus 82.91 (-0.87).
- Request shape: the 22 questions go out as 1 to 2 requests per block, all carrying the whole v3 state; a block of 25 sentences measured 58,863 real input tokens across its 2 requests on the smoke block. Real tokens ran about 1/0.90 of the 4-chars-per-token estimate over the held-out run.
- Request packing differs from `f1`: the predicted real-token cap per request was 60,000 here against 50,000 for `f1` (the provider limit is 64,000 and predictions ran 10 to 20 percent high), so windowed blocks pack their 22 questions into two requests rather than three. The state and the questions are unchanged; only how many questions share a request.
- The smoke block (`roughcut-jev-f2-smoke-*`, block 0 of colman-02.04-skeleton-demo) is kept on disk and counted in the spend; its answers were not used for fitting.

## Inputs

- fit features: `docs/jev-real/roughcut-jev-f2-fit-features.jsonl` md5 ddddfe249a7c558b93886f42a85afc81, 3149499 bytes, modified 2026-09-26T14:48:51+00:00
- fit timing: `docs/jev-real/roughcut-jev-f2-fit-timing.json` md5 b847b516e88026b25f4d520e9dbdccb6, 11575 bytes, modified 2026-09-26T14:48:51+00:00
- frozen weights: `docs/jev-real/roughcut-jev-f2-weights.json` md5 b74619ad24458f21924a2e51fce40220, 51233 bytes, modified 2026-09-26T15:17:05+00:00
- held-out features: `docs/jev-real/roughcut-jev-f2-heldout-features.jsonl` md5 41554f59253aaa363f95f07950e2ec16, 8746735 bytes, modified 2026-09-26T14:57:41+00:00
- held-out timing: `docs/jev-real/roughcut-jev-f2-heldout-timing.json` md5 96e4f42f579688136199ad33cdc4e13c, 23815 bytes, modified 2026-09-26T14:57:41+00:00
- control f1 fit features: `docs/jev-real/roughcut-jev-f1-fit-features.jsonl` md5 bca3b191584ab0d8096ae731eb11ce32, 2936519 bytes, modified 2026-09-26T07:35:32+00:00
- control f1 held-out features: `docs/jev-real/roughcut-jev-f1-heldout-features.jsonl` md5 37c568d73d2c775d5d13c52ad8cb4c5d, 8157947 bytes, modified 2026-09-26T07:49:47+00:00
- control f1 weights: `docs/jev-real/roughcut-jev-f1-weights.json` md5 242943da549e74f78f45d096ecb0d638, 16656 bytes, modified 2026-09-26T07:48:20+00:00
- v3 decisions roughcut-jev-all18-v3: `docs/jev-real/roughcut-jev-all18-v3-decisions.jsonl` md5 0063cbe0dd7f4b2a6314986c8b1295c4, 16041883 bytes, modified 2026-09-26T06:39:11+00:00
- v3 decisions roughcut-jev-heldout-v3: `docs/jev-real/roughcut-jev-heldout-v3-decisions.jsonl` md5 400c49b431ec04e46d9428fc6c399b68, 13570354 bytes, modified 2026-09-26T06:39:11+00:00
- prompt bundle: `scripts/jev_real/roughcut_jev_prompts.py` md5 cae01f3dbb1af5396c7c57eb38f05a20, 27422 bytes, modified 2026-09-26T14:32:08+00:00
