Developer-facing notes on build B of the Jev rough-cut round two: the prompt breakup, 20 yes/no questions per sentence (bundle `f3`) and a logistic combiner fitted in code, Jev only.

Generated 2026-09-27T04:54:31+00:00 by `scripts/jev_real/roughcut_jev_combine.py` from `roughcut-jev-f3-fit-features.jsonl` and `roughcut-jev-f3-heldout-features.jsonl`, control rows from `roughcut-jev-f1-fit-features.jsonl` and `roughcut-jev-f1-heldout-features.jsonl`, weights in `roughcut-jev-f3-weights.json`.

# Jev rough cut, build B: prompt breakup (f3)

The v3 sentence pass asks one six-level score per sentence. This build asks 20 one-look yes/no questions instead (bundle `f3` in `roughcut_jev_prompts.py`), over the same state v3 sent, and fits an L2 logistic regression on the probabilities of yes. Sentence judgment is the only thing that changes: `keep_words` is null, the retake cut is jev_a v3's. Every number below is with um removal and delete silence layered on. Fit-set numbers are leave-one-episode-out over the 6 fit episodes with the keep threshold calibrated on the pooled out-of-fold predictions. C and the feature set were chosen on those numbers alone, then frozen. Bundle `f3` is `f1`'s 18 questions unchanged plus `said_earlier`, `wrap_up` from `f2`. It asked nothing itself: its rows are the `f1` feature rows with the 2 added columns joined from the `f2` feature rows by episode and sentence id (`roughcut_jev_join.py`), which works because both runs asked over the same state, blocks and sentences and each question is answered on its own. No new Jev requests were made. `f1`'s chosen set `q+code+v3` runs through the same fitting path as the control, from its own feature run, and its held-out and ladder rows use its own frozen weights.

Bottom line: the chosen set is `q3+code+v3` (C 0.01, threshold 2.80). Leave-one-out on the fit six it scores 85.52 SP against 82.86 for the v3 score through the same fitting path, 85.63 for `f1`'s `q+code+v3` through the same path and 83.00 for jev_a v3 as published on the same six. On the 13 held-out episodes with the frozen weights and threshold it scores 81.72 SP against 81.17 for `f1` at its frozen weights and 76.74 for jev_a v3. On the 18-episode ladder it lands at 83.37 next to `f1`'s 82.91 and jev_a v3's 80.47 (below best Luna chapters, above Jev f1 `q+code+v3` combiner (frozen)). Spend $0.00 in new Jev calls (the rows are a join of runs already paid for), 10.5 s per ladder episode with the v3 pass included (the source passes summed). Routing the bottom 25% by combiner margin to archived Luna gives 85.22 against 84.81 for `f1` and 84.06 for the v3 margin.

## Win test, decided before held-out

The test: `f3` wins if its chosen set scores above `f1`'s frozen chosen set `q+code+v3` (85.63 SP) leave-one-out on the fit six. Stage 1 decided it and froze it in `roughcut-jev-f3-weights.json` at 2026-09-27T04:36:30+00:00 (weights file generated 2026-09-27T04:36:30+00:00), before the held-out rows were read; this write-up (generated 2026-09-27T04:54:31+00:00) only reports it. Leave-one-out per eligible set: `q3` 83.94, `q3+code` 85.49, `q3+code+v3` 85.52, `v3` 82.86. Chosen `q3+code+v3` at 85.52, -0.11 against the bar: `f3` does not win, so the Luna stack rerun is skipped; the held-out and ladder numbers below are reported anyway. Seconds per episode on the fit six: 9.1.

## Fit set, leave-one-episode-out

Pooled over the 6 fit episodes, every feature set at its best C. `v3` is the control: jev_a v3's 0-5 score alone through the same fitting path. `code+v3` is a diagnostic set outside the spec's four, there to show what the questions add over the free features and v3 together. `f1 q+code+v3` is `f1`'s chosen set refitted through the same path from its own feature run, the comparison arm, not eligible for selection. Seconds per episode are the `f1` and `f2` feature passes whose answers the `f3` rows read, plus the v3 pass (all at concurrency 8).

| feature set | features | C | threshold | SENTENCE POINTS | WORD SCORE | GRADE | LOO AUC | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q3` | 20 | 0.3 | 2.00 | 83.94 | 77.92 | 87.07 | 0.894 | 9.1 |
| `q3+code` | 39 | 0.003 | 3.00 | 85.49 | 79.63 | 89.15 | 0.919 | 9.1 |
| `q3+code+v3` (chosen) | 43 | 0.01 | 2.80 | 85.52 | 79.73 | 89.29 | 0.922 | 9.1 |
| `v3` | 4 | 0.3 | 3.20 | 82.86 | 76.16 | 86.32 | 0.881 | 9.1 |
| `code+v3` (diagnostic) | 23 | 3 | 3.60 | 81.71 | 75.84 | 86.22 | 0.883 | 9.1 |
| `f1 q+code+v3` (control) | 41 | 0.003 | 3.00 | 85.63 | 79.79 | 89.34 | 0.922 | 9.1 |
| jev_a v3 as published (trims at 0.3, calibrated) |  |  | 2.50 | 83.00 | 76.78 | 86.72 |  | 5.1 |
| jev_a v3, keep_words null (calibrated) |  |  | 2.50 | 82.97 | 76.61 | 86.56 |  | 5.1 |

C sweep, leave-one-out SP per feature set (the chosen C is the best; ties go to the smaller C):

| feature set | C 0.001 | C 0.003 | C 0.01 | C 0.03 | C 0.1 | C 0.3 | C 1 | C 3 | C 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q3` | 83.33 | 83.85 | 83.92 | 83.74 | 83.75 | 83.94 | 83.16 | 83.13 | 83.13 |
| `q3+code` | 84.60 | 85.49 | 85.40 | 85.34 | 84.72 | 84.30 | 84.55 | 84.47 | 84.29 |
| `q3+code+v3` | 84.48 | 85.50 | 85.52 | 84.90 | 84.68 | 84.31 | 83.94 | 84.29 | 84.19 |
| `v3` | 78.72 | 82.10 | 82.68 | 82.75 | 82.58 | 82.86 | 82.75 | 82.75 | 82.75 |
| `code+v3` | 75.98 | 79.67 | 81.37 | 81.67 | 81.62 | 81.62 | 81.60 | 81.71 | 81.71 |
| `f1 q+code+v3` | 85.05 | 85.63 | 85.63 | 84.97 | 84.75 | 84.15 | 84.34 | 83.90 | 84.20 |

Per episode, leave-one-out, at each set's pooled threshold:

| episode | sentences | `q3` | `q3+code` | `q3+code+v3` | `v3` | `code+v3` | `f1 q+code+v3` | jev_a v3 | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 84.23 | 84.43 | 84.95 | 83.09 | 81.13 | 84.12 | 85.88 | 6.1 |
| hampton-5.4-assignment-demo | 300 | 91.77 | 89.67 | 91.43 | 86.90 | 81.77 | 91.53 | 84.80 | 5.2 |
| colman-03.03-muscles-crit | 303 | 81.16 | 78.88 | 79.17 | 73.93 | 67.72 | 77.66 | 75.05 | 5.7 |
| edges-7.01-intro | 389 | 78.84 | 83.47 | 84.40 | 77.30 | 77.71 | 84.91 | 79.69 | 7.9 |
| hampton-5.2-shape-demo | 411 | 96.06 | 96.25 | 95.52 | 93.28 | 91.19 | 96.25 | 92.70 | 6.7 |
| perspective-14e-boxes-critique | 1146 | 79.97 | 83.14 | 82.53 | 82.27 | 83.46 | 82.88 | 81.78 | 22.7 |

## Held-out, frozen weights

The 13 held-out episodes scored with the weights and the keep threshold frozen after stage 1. Nothing here was fitted, chosen or calibrated on these episodes. The last column recalibrates the threshold on the held-out set itself and is not held out; it is there to show how much the frozen threshold costs. `f1 q+code+v3` uses `f1`'s own frozen weights and threshold (stage 1 reproduced them exactly) on `f1`'s held-out feature rows.

| feature set | threshold | SENTENCE POINTS | WORD SCORE | GRADE | AUC | SP recalibrated (not held out) | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q3` | 2.00 | 80.37 | 74.44 | 88.46 | 0.884 | 80.37 at t=2.00 | 12.5 |
| `q3+code` | 3.00 | 81.01 | 74.70 | 88.71 | 0.901 | 81.30 at t=2.70 | 12.5 |
| `q3+code+v3` (chosen) | 2.80 | 81.72 | 75.02 | 89.23 | 0.906 | 81.64 at t=3.00 | 12.5 |
| `v3` | 3.20 | 77.81 | 71.76 | 85.56 | 0.858 | 77.79 at t=3.00 | 12.5 |
| `code+v3` (diagnostic) | 3.60 | 78.51 | 73.66 | 87.20 | 0.872 | 78.98 at t=3.20 | 12.5 |
| `f1 q+code+v3` (control) | 3.00 | 81.17 | 74.73 | 88.87 | 0.903 | 81.54 at t=2.70 | 12.5 |
| jev_a v3 as published (trims at 0.3, t 2.50) | 2.50 | 76.74 | 72.14 | 86.08 |  |  | 6.4 |

Per episode, frozen weights:

| episode | sentences | `q3` | `q3+code` | `q3+code+v3` | `v3` | `code+v3` | `f1 q+code+v3` | jev_a v3 | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 1752 | 82.38 | 83.58 | 84.63 | 80.55 | 83.38 | 84.61 | 78.09 | 28.8 |
| hampton-5.5-crit1 | 181 | 90.50 | 89.94 | 90.50 | 85.52 | 82.76 | 89.94 | 87.35 | 4.0 |
| hampton-5.5-crit2 | 127 | 91.42 | 90.63 | 90.63 | 79.92 | 68.11 | 89.84 | 79.45 | 2.2 |
| hampton-5.5-crit3 | 137 | 82.48 | 82.48 | 81.75 | 80.29 | 78.83 | 82.48 | 81.17 | 2.4 |
| hampton-5.5-crit4 | 156 | 92.76 | 92.76 | 93.40 | 88.27 | 81.86 | 93.40 | 88.46 | 2.3 |
| hampton-5.5-crit5 | 295 | 91.49 | 91.49 | 91.49 | 89.46 | 87.42 | 91.49 | 90.88 | 7.3 |
| flanders-03-thematic-crit | 1309 | 78.48 | 74.93 | 79.02 | 75.57 | 75.05 | 76.95 | 77.14 | 32.8 |
| anatomy-30b-hamstring-crit | 951 | 84.93 | 84.85 | 85.81 | 83.71 | 82.44 | 85.16 | 84.90 | 25.4 |
| colman-04.03-life-crit | 495 | 77.94 | 77.94 | 77.33 | 74.71 | 73.21 | 76.93 | 76.18 | 10.2 |
| colman-05.02-master-studies-crit | 381 | 65.91 | 70.21 | 69.32 | 65.91 | 68.35 | 69.06 | 66.33 | 10.5 |
| colman-06.06-species-crit | 373 | 77.96 | 76.14 | 77.37 | 76.41 | 74.02 | 75.76 | 79.57 | 6.1 |
| hampton-7-conclusion | 43 | 79.30 | 81.63 | 81.63 | 79.30 | 81.63 | 79.30 | 72.33 | 2.7 |
| greco-2.2-thumbnailing | 1388 | 75.69 | 80.45 | 78.57 | 72.02 | 76.80 | 78.79 | 65.09 | 27.3 |

## Ladder, 18 episodes

The fit six enter with their leave-one-out predictions and the 12 ladder held-out episodes with the frozen weights, all at the frozen threshold 2.80; greco-2.2-thumbnailing is not a ladder episode and is left out here. jev_a v3 is rebuilt through the same scoring path and reproduces its published 80.47 at 80.47. `f1`'s row is built the same way from its own rows at its frozen threshold 3.00. Seconds per episode: 10.5 (f1 and f2 passes plus v3).

| arm | SENTENCE POINTS | s/episode |
|---|---:|---:|
| shipped Opus agentic | 86.47 |  |
| best Luna chapters | 83.71 |  |
| Jev f3 `q3+code+v3` combiner (this build) | 83.37 | 10.5 |
| Jev f1 `q+code+v3` combiner (frozen) | 82.91 | 7.9 |
| Jev jev_a v3 (pure Jev) | 80.47 |  |
| Luna single call | 65.36 |  |
| deterministic baseline (um removal + retakes + delete silence) | 63.72 |  |

## Standardised weights, chosen set

`q3+code+v3` refitted on all 6 fit episodes with C 0.01, sorted by size. A positive weight pushes toward keep. Weights are per standard deviation of the feature, so they compare across features. Intercept 0.212.

| feature | weight | what it says |
|---|---:|---:|
| `off_topic` | -0.427 | yes: off the lesson's topic; pushes toward cut |
| `v3_score` | +0.414 | v3 0-5 score; pushes toward keep |
| `pre_lesson` | -0.408 | yes: chatter before the lesson starts; pushes toward cut |
| `screen_ops` | -0.311 | yes: operating the screen or software; pushes toward cut |
| `asr_confidence` | +0.287 | mean ASR word confidence; pushes toward keep |
| `is_retake` | -0.264 | corpus retake flag (module loser); pushes toward cut |
| `describes_screen` | +0.261 | yes: only describes what is on screen; pushes toward keep |
| `crew_talk` | -0.259 | yes: addressed to the crew, not students; pushes toward cut |
| `referenced_later` | -0.259 | yes: a later sentence depends on it; pushes toward cut |
| `essential` | +0.255 | yes: the lesson loses something without it; pushes toward keep |
| `duration_s` | +0.242 | spoken duration in seconds; pushes toward keep |
| `pure_filler` | -0.214 | yes: filler with no content; pushes toward cut |
| `funny` | +0.186 | yes: funny or shows personality; pushes toward keep |
| `position` | -0.181 | position in the episode, 0 to 1; pushes toward cut |
| `v3_cut_p` | -0.163 | v3 P(editor removes it); pushes toward cut |
| `overlap_next` | -0.157 | word overlap with the next sentence; pushes toward cut |
| `trail_off` | -0.157 | row ends in the transcriber's '..' mark; pushes toward cut |
| `teaching_point` | +0.148 | yes: states a point, reason or correction; pushes toward keep |
| `retake_loser` | -0.147 | yes: a losing take of a repeated line; pushes toward cut |
| `repeats_point` | -0.146 | yes: repeats a point just made; pushes toward cut |
| `retake_member` | -0.138 | in a retake group; pushes toward cut |
| `v3_last_p_whole` | -0.132 | v3 P(nothing trimmed from the end); pushes toward cut |
| `transition` | +0.129 | yes: a spoken transition between students or steps; pushes toward keep |
| `retake_winner` | +0.123 | the module's winning take; pushes toward keep |
| `tangent` | -0.117 | yes: an aside the lesson resumes after; pushes toward cut |
| `chain_piece` | -0.100 | piece index in a split-sentence chain; pushes toward cut |
| `n_words` | +0.093 | word count after um stripping; pushes toward keep |
| `n_words_raw` | +0.091 | word count as spoken; pushes toward keep |
| `split_fragment` | +0.087 | yes: half of a transcriber-split sentence; pushes toward keep |
| `rambling` | -0.082 | yes: rambling or thinking aloud; pushes toward cut |
| `words_per_s` | -0.076 | speaking rate; pushes toward cut |
| `wrap_up` (new) | -0.074 | yes: closes a section with nothing new; pushes toward cut |
| `false_start` | -0.050 | yes: the row is an abandoned attempt; pushes toward cut |
| `pause_before` | -0.046 | seconds of silence before the sentence; pushes toward cut |
| `chain_len` | +0.033 | length of the split-sentence chain; pushes toward keep |
| `um_removed` | -0.029 | words the um module removed; pushes toward cut |
| `said_earlier` (new) | -0.026 | yes: the point was made earlier in the episode; pushes toward cut |
| `pause_after` | -0.021 | seconds of silence after the sentence; pushes toward cut |
| `overlap_prev` | +0.018 | word overlap with the previous sentence; pushes toward keep |
| `um_detected` | -0.015 | um-like words as spoken; pushes toward cut |
| `lower_start` | +0.015 | row starts lowercase (continuation); pushes toward keep |
| `v3_first_p_whole` | +0.014 | v3 P(nothing trimmed from the start); pushes toward keep |
| `pep_talk` | +0.002 | yes: praise or wrap-up with nothing new; pushes toward keep |

## Each question alone

AUC of each probability of yes against the editor's keep (full or partial) versus removed, pooled over the fit six and, when present, the held-out episodes. 0.50 is no signal; a cut question reads below 0.50 and a keep question above. The v3 score and cut_p are listed on the same footing. Questions marked new were added in `f3`.

| question | AUC fit | AUC held-out | abs(AUC - 0.5) fit | signal |
|---|---:|---:|---:|---:|
| `v3_score` | 0.888 | 0.845 | 0.388 | yes |
| `v3_cut_p` | 0.134 | 0.178 | 0.366 | yes |
| `crew_talk` | 0.175 | 0.161 | 0.325 | yes |
| `off_topic` | 0.176 | 0.178 | 0.324 | yes |
| `pre_lesson` | 0.185 | 0.166 | 0.315 | yes |
| `pure_filler` | 0.207 | 0.185 | 0.293 | yes |
| `essential` | 0.779 | 0.766 | 0.279 | yes |
| `screen_ops` | 0.241 | 0.281 | 0.259 | yes |
| `teaching_point` | 0.727 | 0.753 | 0.227 | yes |
| `tangent` | 0.285 | 0.319 | 0.215 | yes |
| `repeats_point` | 0.286 | 0.340 | 0.214 | yes |
| `retake_loser` | 0.298 | 0.355 | 0.202 | yes |
| `rambling` | 0.323 | 0.405 | 0.177 | yes |
| `false_start` | 0.327 | 0.401 | 0.173 | yes |
| `said_earlier` (new) | 0.339 | 0.373 | 0.161 | yes |
| `wrap_up` (new) | 0.379 | 0.329 | 0.121 | yes |
| `pep_talk` | 0.399 | 0.354 | 0.101 | yes |
| `transition` | 0.543 | 0.449 | 0.043 | weak |
| `describes_screen` | 0.519 | 0.540 | 0.019 | none |
| `funny` | 0.491 | 0.478 | 0.009 | none |
| `split_fragment` | 0.508 | 0.543 | 0.008 | none |
| `referenced_later` | 0.503 | 0.589 | 0.003 | none |

Questions with no signal on their own (abs(AUC - 0.5) under 0.03): `funny`, `referenced_later`, `describes_screen`, `split_fragment`.

## Confusion

Keep or cut of the chosen arm against the editor and against jev_a v3, with modules, per split. Fit rows are leave-one-out; held-out rows use the frozen weights.

| split | reference | n | both keep | ref keep, f3 cut | ref cut, f3 keep | both cut | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit | editor | 2743 | 1321 | 184 | 183 | 1055 | 86.62 | 9.1 |
| fit | jev_a v3 | 2743 | 1370 | 148 | 134 | 1091 | 89.72 | 9.1 |
| heldout | editor | 7588 | 3919 | 485 | 691 | 2493 | 84.50 | 12.5 |
| heldout | jev_a v3 | 7588 | 4340 | 676 | 270 | 2302 | 87.53 | 12.5 |
| ladder18 | editor | 8943 | 4915 | 628 | 641 | 2759 | 85.81 | 10.5 |
| ladder18 | jev_a v3 | 8943 | 5169 | 588 | 387 | 2799 | 89.10 | 10.5 |

## Calibration by quartile

The chosen combiner's `p_keep` over the 18 ladder episodes (fit leave-one-out plus held-out frozen). The margin table is the input route 2 needs: the bottom margin quartile is the slice a bigger model would take.

| p_keep quartile | n | p range | mean p_keep | editor kept | arm kept | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2236 | 0.00 to 0.30 | 0.108 | 9.35 | 0.04 | 90.61 | 10.5 |
| 2 | 2235 | 0.30 to 0.78 | 0.558 | 53.51 | 49.80 | 67.20 | 10.5 |
| 3 | 2236 | 0.78 to 0.92 | 0.861 | 88.10 | 98.70 | 88.51 | 10.5 |
| 4 | 2236 | 0.92 to 1.00 | 0.955 | 96.96 | 99.96 | 96.91 | 10.5 |

| margin quartile | n | margin range | mean p_keep | editor kept | arm kept | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2236 | 0.00 to 1.16 | 0.588 | 57.20 | 55.90 | 67.67 | 10.5 |
| 2 | 2235 | 1.16 to 1.72 | 0.722 | 72.21 | 76.06 | 86.49 | 10.5 |
| 3 | 2236 | 1.72 to 2.04 | 0.827 | 83.77 | 85.20 | 93.29 | 10.5 |
| 4 | 2236 | 2.04 to 2.80 | 0.346 | 34.75 | 31.35 | 95.80 | 10.5 |

## What this hands route 2

Bottom share of the 18 ladder episodes by combiner margin `abs(5 * p_keep - threshold)`, one global cutoff, the archived Luna chapters decision substituted on the routed slice and rescored with modules, exactly as `roughcut_route2_routing.py` does for the v3 margin. The v3 margin gives 84.06 at 25%. Seconds per episode are Jev only; the Luna call is build A's number.

| combiner | share | routed | margin cutoff | SENTENCE POINTS | WORD SCORE | s/episode |
|---|---:|---:|---:|---:|---:|---:|
| `q3+code+v3` | 25% | 2236 | 1.164 | 85.22 | 78.94 | 10.5 |
| `q3+code+v3` | 50% | 4472 | 1.718 | 84.54 | 79.46 | 10.5 |
| `f1 q+code+v3` | 25% | 2236 | 0.879 | 84.81 | 78.72 | 7.9 |
| `f1 q+code+v3` | 50% | 4472 | 1.391 | 84.46 | 79.54 | 7.9 |
| v3 margin, archived Luna (route 2 write-up) | 25% | 2236 | 0.46 | 84.06 |  | 5.6 |

## Seconds and dollars per episode

`f3` made no Jev requests: the join reads the `f1` and `f2` feature runs, so its new spend is $0 on every episode. The source columns are those runs' own measured seconds and router cost (concurrency 8, $0.042 per million input tokens), and the v3 columns the v3 run the rows join; total seconds is their sum.

| episode | split | sentences | f1 s | f2 s | v3 s | total s | f3 new $ | f1 $ | f2 $ | v3 $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | fit | 194 | 0.79 | 0.89 | 4.45 | 6.12 | 0.0000 | 0.0178 | 0.0202 | 0.0149 |
| hampton-5.4-assignment-demo | fit | 300 | 1.13 | 1.29 | 2.79 | 5.21 | 0.0000 | 0.0268 | 0.0306 | 0.0213 |
| colman-03.03-muscles-crit | fit | 303 | 1.18 | 1.39 | 3.10 | 5.66 | 0.0000 | 0.0307 | 0.0345 | 0.0252 |
| edges-7.01-intro | fit | 389 | 1.41 | 1.56 | 4.90 | 7.87 | 0.0000 | 0.0326 | 0.0376 | 0.0275 |
| hampton-5.2-shape-demo | fit | 411 | 1.72 | 1.82 | 3.20 | 6.74 | 0.0000 | 0.0427 | 0.0479 | 0.0338 |
| perspective-14e-boxes-critique | fit | 1146 | 4.95 | 5.32 | 12.44 | 22.71 | 0.0000 | 0.1408 | 0.1553 | 0.1053 |
| perspective-13d-critique | held-out | 1752 | 7.31 | 7.85 | 13.61 | 28.77 | 0.0000 | 0.2160 | 0.2382 | 0.1252 |
| hampton-5.5-crit1 | held-out | 181 | 0.79 | 1.02 | 2.22 | 4.02 | 0.0000 | 0.0173 | 0.0196 | 0.0152 |
| hampton-5.5-crit2 | held-out | 127 | 0.60 | 0.68 | 0.93 | 2.21 | 0.0000 | 0.0115 | 0.0131 | 0.0102 |
| hampton-5.5-crit3 | held-out | 137 | 0.70 | 0.74 | 0.99 | 2.42 | 0.0000 | 0.0125 | 0.0143 | 0.0112 |
| hampton-5.5-crit4 | held-out | 156 | 0.69 | 0.82 | 0.79 | 2.30 | 0.0000 | 0.0151 | 0.0170 | 0.0133 |
| hampton-5.5-crit5 | held-out | 295 | 1.13 | 1.40 | 4.81 | 7.34 | 0.0000 | 0.0299 | 0.0336 | 0.0245 |
| flanders-03-thematic-crit | held-out | 1309 | 5.48 | 8.08 | 19.26 | 32.81 | 0.0000 | 0.1541 | 0.1707 | 0.1246 |
| anatomy-30b-hamstring-crit | held-out | 951 | 6.20 | 7.45 | 11.76 | 25.41 | 0.0000 | 0.1148 | 0.1268 | 0.0913 |
| colman-04.03-life-crit | held-out | 495 | 1.99 | 2.37 | 5.88 | 10.23 | 0.0000 | 0.0558 | 0.0621 | 0.0420 |
| colman-05.02-master-studies-crit | held-out | 381 | 3.71 | 1.83 | 5.01 | 10.54 | 0.0000 | 0.0413 | 0.0462 | 0.0329 |
| colman-06.06-species-crit | held-out | 373 | 1.51 | 1.72 | 2.83 | 6.06 | 0.0000 | 0.0386 | 0.0433 | 0.0307 |
| hampton-7-conclusion | held-out | 43 | 0.41 | 0.45 | 1.81 | 2.68 | 0.0000 | 0.0032 | 0.0038 | 0.0030 |
| greco-2.2-thumbnailing | held-out | 1388 | 8.08 | 6.16 | 13.02 | 27.25 | 0.0000 | 0.1631 | 0.1807 | 0.1156 |

Spend on this build: $0.00, no new Jev calls (0 requests, against the $0.00 cap). The answers were paid for in the source runs: `f1` $1.1645 on these 19 episodes (`roughcut-jev-f1.md`, section "Seconds and dollars per episode", $1.1667 in all with its smoke block); `f2` $1.2954 on these 19 episodes (`roughcut-jev-f2.md`, section "Seconds and dollars per episode", $1.2979 in all with its smoke block).

## Notes

- Target per sentence: editor kept (full or partial) versus removed, from the harness's human sentence states, the same states SENTENCE POINTS reads. Fit six: 1505 kept of 2743.
- Missing values: an unanswered noul is filled with 0.5, a missing v3 score with 2.5, a missing v3 cut_p with 0.5, a missing trim answer (one-word rows) with 1.0 (whole). Counts: {'fit': {'q_cells': 0, 'v3_score': 0, 'v3_cut_p': 0, 'sentences': 2743}, 'heldout': {'q_cells': 0, 'v3_score': 0, 'v3_cut_p': 0, 'sentences': 7588}}.
- Fit-set threshold: calibrated by the harness's pooled Neutral sweep on the out-of-fold predictions of all six episodes, so it is chosen on the fit set only and then frozen with the weights. The pooled SP in the tables is the sentence-count weighted mean of per-episode SP; the harness's own pooled figure for the chosen set is 85.52.
- The v3 control lands at 82.86 against 83.00 for jev_a v3 as published on the same six and 82.97 with keep_words null. The control has no trims and a logistic squashing of the score; the keep threshold moves accordingly.
- The questions are TypeSafe nouls (probability of yes), one per target sentence per question, the same primitive as v3's cut_k. `f3` asked nothing itself: its rows are a join of the `f1` and `f2` feature runs by episode and sentence id (`roughcut_jev_join.py`), `f1` supplies 18 (all of its own 18); `f2` supplies 2 (`said_earlier`, `wrap_up`). Both runs asked over the same state, blocks and sentences and every question is answered on its own, so the joined columns are what a single 20-question run would have asked. The join asserts every sentence matches (ids, text, code features, v3 join) and no cell is unanswered.
- Selection was among the spec's four sets only; `code+v3` is reported as a diagnostic and was not eligible, nor was the `f1` control set.
- Control: `f1`'s chosen set `q+code+v3` refitted through this script's path from `roughcut-jev-f1-fit` reproduces the weights, C and threshold frozen in `roughcut-jev-f1-weights.json` (largest coefficient difference 0.0e+00). Its held-out and ladder rows use that frozen file, so they are the same arm `f1`'s write-up reports.
- Held-out numbers use the frozen weights and the frozen threshold. Nothing was refitted, recalibrated or chosen after the held-out features were read; stage 2 refuses to run if the stage 1 recomputation drifts from the frozen file. The control set was frozen at stage 1 in the same file.
- Against `f1` at its frozen weights: held-out 81.72 versus 81.17 (+0.55), ladder 83.37 versus 82.91 (+0.46).
- Held-out and ladder go the other way from the win test: `f3` is ahead of `f1` on both, after losing leave-one-out on the fit six by 0.11. The test stands as it was frozen before these numbers existed, so the follow-up it gates was not run.
- Spend: no new Jev calls were made for `f3`, so its spend is $0.00. The answers it reads were paid for in the `f1` run, $1.1645 on these episodes and $1.1667 in all with its smoke block, reported in `roughcut-jev-f1.md` under "Seconds and dollars per episode"; the `f2` run, $1.2954 on these episodes and $1.2979 in all with its smoke block, reported in `roughcut-jev-f2.md` under "Seconds and dollars per episode". Seconds per episode in the tables are the measured wall clock of those source passes plus the v3 pass, the time it took to produce the answers on disk; a single live `f3` pass would ask 20 questions in one run, so it would sit near one source pass, not their sum.

## Inputs

- fit features: `docs/jev-real/roughcut-jev-f3-fit-features.jsonl` md5 7617714da6335195caa7d11171fee17a, 3295318 bytes, modified 2026-09-27T04:07:19+00:00
- fit timing: `docs/jev-real/roughcut-jev-f3-fit-timing.json` md5 8078b13767ade4586902e31f26a49aeb, 7255 bytes, modified 2026-09-27T04:07:19+00:00
- frozen weights: `docs/jev-real/roughcut-jev-f3-weights.json` md5 b39151db65d7b961f8bb8a370967da33, 43417 bytes, modified 2026-09-27T04:36:30+00:00
- held-out features: `docs/jev-real/roughcut-jev-f3-heldout-features.jsonl` md5 73ca07072f3b3dc5159e3b7fcd3a5952, 9150356 bytes, modified 2026-09-27T04:07:19+00:00
- held-out timing: `docs/jev-real/roughcut-jev-f3-heldout-timing.json` md5 8241d110010dcf636b6e89efdd25dccf, 13225 bytes, modified 2026-09-27T04:07:19+00:00
- control f1 fit features: `docs/jev-real/roughcut-jev-f1-fit-features.jsonl` md5 bca3b191584ab0d8096ae731eb11ce32, 2936519 bytes, modified 2026-09-26T07:35:32+00:00
- control f1 held-out features: `docs/jev-real/roughcut-jev-f1-heldout-features.jsonl` md5 37c568d73d2c775d5d13c52ad8cb4c5d, 8157947 bytes, modified 2026-09-26T07:49:47+00:00
- control f1 weights: `docs/jev-real/roughcut-jev-f1-weights.json` md5 242943da549e74f78f45d096ecb0d638, 16656 bytes, modified 2026-09-26T07:48:20+00:00
- v3 decisions roughcut-jev-all18-v3: `docs/jev-real/roughcut-jev-all18-v3-decisions.jsonl` md5 0063cbe0dd7f4b2a6314986c8b1295c4, 16041883 bytes, modified 2026-09-26T06:39:11+00:00
- v3 decisions roughcut-jev-heldout-v3: `docs/jev-real/roughcut-jev-heldout-v3-decisions.jsonl` md5 400c49b431ec04e46d9428fc6c399b68, 13570354 bytes, modified 2026-09-26T06:39:11+00:00
- join source f1 fit features: `docs/jev-real/roughcut-jev-f1-fit-features.jsonl` md5 bca3b191584ab0d8096ae731eb11ce32, 2936519 bytes, modified 2026-09-26T07:35:32+00:00
- join source f1 fit timing: `docs/jev-real/roughcut-jev-f1-fit-timing.json` md5 0d0506b78962b9bf55d5156df279c575, 11503 bytes, modified 2026-09-26T07:35:32+00:00
- join source f2 fit features: `docs/jev-real/roughcut-jev-f2-fit-features.jsonl` md5 ddddfe249a7c558b93886f42a85afc81, 3149499 bytes, modified 2026-09-26T14:48:51+00:00
- join source f2 fit timing: `docs/jev-real/roughcut-jev-f2-fit-timing.json` md5 b847b516e88026b25f4d520e9dbdccb6, 11575 bytes, modified 2026-09-26T14:48:51+00:00
- join source f1 held-out features: `docs/jev-real/roughcut-jev-f1-heldout-features.jsonl` md5 37c568d73d2c775d5d13c52ad8cb4c5d, 8157947 bytes, modified 2026-09-26T07:49:47+00:00
- join source f1 held-out timing: `docs/jev-real/roughcut-jev-f1-heldout-timing.json` md5 f5e20384e73eaa5e160d8ce13769098a, 23588 bytes, modified 2026-09-26T07:49:47+00:00
- join source f2 held-out features: `docs/jev-real/roughcut-jev-f2-heldout-features.jsonl` md5 41554f59253aaa363f95f07950e2ec16, 8746735 bytes, modified 2026-09-26T14:57:41+00:00
- join source f2 held-out timing: `docs/jev-real/roughcut-jev-f2-heldout-timing.json` md5 96e4f42f579688136199ad33cdc4e13c, 23815 bytes, modified 2026-09-26T14:57:41+00:00
- prompt bundle: `scripts/jev_real/roughcut_jev_prompts.py` md5 939404eeca6b6b43db23b966f22686ec, 28688 bytes, modified 2026-09-27T04:07:16+00:00
