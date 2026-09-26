Developer-facing notes on build B of the Jev rough-cut round two: the prompt breakup, eighteen yes/no questions per sentence and a logistic combiner fitted in code, Jev only.

Generated 2026-09-26T08:02:53+00:00 by `scripts/jev_real/roughcut_jev_combine.py` from `roughcut-jev-f1-fit-features.jsonl` and `roughcut-jev-f1-heldout-features.jsonl`, weights in `roughcut-jev-f1-weights.json`.

# Jev rough cut, build B: prompt breakup (f1)

The v3 sentence pass asks one six-level score per sentence. This build asks 18 one-look yes/no questions instead (bundle `f1` in `roughcut_jev_prompts.py`), over the same state v3 sent, and fits an L2 logistic regression on the probabilities of yes. Sentence judgment is the only thing that changes: `keep_words` is null, the retake cut is jev_a v3's. Every number below is with um removal and delete silence layered on. Fit-set numbers are leave-one-episode-out over the 6 fit episodes with the keep threshold calibrated on the pooled out-of-fold predictions. C and the feature set were chosen on those numbers alone, then frozen.

Bottom line: the chosen set is `q+code+v3` (C 0.003, threshold 3.00). Leave-one-out on the fit six it scores 85.63 SP against 82.86 for the v3 score through the same fitting path and 83.00 for jev_a v3 as published on the same six. On the 13 held-out episodes with the frozen weights and threshold it scores 81.17 SP against 76.74 for jev_a v3. On the 18-episode ladder it lands at 82.91 next to jev_a v3's 80.47 (below best Luna chapters, above Jev jev_a v3 (pure Jev)). Spend $1.17 in Jev calls, 7.9 s per ladder episode with the v3 pass included. Routing the bottom 25% by combiner margin to archived Luna gives 84.81 against 84.06 for the v3 margin.

## Fit set, leave-one-episode-out

Pooled over the 6 fit episodes, every feature set at its best C. `v3` is the control: jev_a v3's 0-5 score alone through the same fitting path. `code+v3` is a diagnostic set outside the spec's four, there to show what the questions add over the free features and v3 together. Seconds per episode are the f1 pass plus the v3 pass it joins (both at concurrency 8).

| feature set | features | C | threshold | SENTENCE POINTS | WORD SCORE | GRADE | LOO AUC | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q` | 18 | 0.003 | 2.60 | 84.76 | 78.64 | 87.87 | 0.897 | 7.0 |
| `q+code` | 37 | 0.01 | 2.90 | 85.52 | 79.88 | 89.33 | 0.922 | 7.0 |
| `q+code+v3` (chosen) | 41 | 0.003 | 3.00 | 85.63 | 79.79 | 89.34 | 0.922 | 7.0 |
| `v3` | 4 | 0.3 | 3.20 | 82.86 | 76.16 | 86.32 | 0.881 | 7.0 |
| `code+v3` (diagnostic) | 23 | 3 | 3.60 | 81.71 | 75.84 | 86.22 | 0.883 | 7.0 |
| jev_a v3 as published (trims at 0.3, calibrated) |  |  | 2.50 | 83.00 | 76.78 | 86.72 |  | 5.1 |
| jev_a v3, keep_words null (calibrated) |  |  | 2.50 | 82.97 | 76.61 | 86.56 |  | 5.1 |

C sweep, leave-one-out SP per feature set (the chosen C is the best; ties go to the smaller C):

| feature set | C 0.001 | C 0.003 | C 0.01 | C 0.03 | C 0.1 | C 0.3 | C 1 | C 3 | C 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q` | 83.31 | 84.76 | 84.46 | 83.90 | 84.59 | 83.57 | 84.44 | 84.05 | 84.36 |
| `q+code` | 84.50 | 85.50 | 85.52 | 85.46 | 84.80 | 84.74 | 84.59 | 84.52 | 84.52 |
| `q+code+v3` | 85.05 | 85.63 | 85.63 | 84.97 | 84.75 | 84.15 | 84.34 | 83.90 | 84.20 |
| `v3` | 78.72 | 82.10 | 82.68 | 82.75 | 82.58 | 82.86 | 82.75 | 82.75 | 82.75 |
| `code+v3` | 75.98 | 79.67 | 81.37 | 81.67 | 81.62 | 81.62 | 81.60 | 81.71 | 81.71 |

Per episode, leave-one-out, at each set's pooled threshold:

| episode | sentences | `q` | `q+code` | `q+code+v3` | `v3` | `code+v3` | jev_a v3 | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 83.71 | 83.92 | 84.12 | 83.09 | 81.13 | 85.88 | 5.2 |
| hampton-5.4-assignment-demo | 300 | 92.10 | 91.87 | 91.53 | 86.90 | 81.77 | 84.80 | 3.9 |
| colman-03.03-muscles-crit | 303 | 80.83 | 79.64 | 77.66 | 73.93 | 67.72 | 75.05 | 4.3 |
| edges-7.01-intro | 389 | 80.90 | 83.73 | 84.91 | 77.30 | 77.71 | 79.69 | 6.3 |
| hampton-5.2-shape-demo | 411 | 96.25 | 96.25 | 96.25 | 93.28 | 91.19 | 92.70 | 4.9 |
| perspective-14e-boxes-critique | 1146 | 81.26 | 82.44 | 82.88 | 82.27 | 83.46 | 81.78 | 17.4 |

## Held-out, frozen weights

The 13 held-out episodes scored with the weights and the keep threshold frozen after stage 1. Nothing here was fitted, chosen or calibrated on these episodes. The last column recalibrates the threshold on the held-out set itself and is not held out; it is there to show how much the frozen threshold costs.

| feature set | threshold | SENTENCE POINTS | WORD SCORE | GRADE | AUC | SP recalibrated (not held out) | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q` | 2.60 | 80.26 | 73.45 | 87.38 | 0.881 | 79.89 at t=2.10 | 9.3 |
| `q+code` | 2.90 | 81.37 | 74.82 | 88.85 | 0.903 | 81.53 at t=2.60 | 9.3 |
| `q+code+v3` (chosen) | 3.00 | 81.17 | 74.73 | 88.87 | 0.903 | 81.54 at t=2.70 | 9.3 |
| `v3` | 3.20 | 77.81 | 71.76 | 85.56 | 0.858 | 77.79 at t=3.00 | 9.3 |
| `code+v3` (diagnostic) | 3.60 | 78.51 | 73.66 | 87.20 | 0.872 | 78.98 at t=3.20 | 9.3 |
| jev_a v3 as published (trims at 0.3, t 2.50) | 2.50 | 76.74 | 72.14 | 86.08 |  |  | 6.4 |

Per episode, frozen weights:

| episode | sentences | `q` | `q+code` | `q+code+v3` | `v3` | `code+v3` | jev_a v3 | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 1752 | 82.41 | 83.45 | 84.61 | 80.55 | 83.38 | 78.09 | 20.9 |
| hampton-5.5-crit1 | 181 | 89.94 | 89.94 | 89.94 | 85.52 | 82.76 | 87.35 | 3.0 |
| hampton-5.5-crit2 | 127 | 91.42 | 90.63 | 89.84 | 79.92 | 68.11 | 79.45 | 1.5 |
| hampton-5.5-crit3 | 137 | 83.21 | 81.75 | 82.48 | 80.29 | 78.83 | 81.17 | 1.7 |
| hampton-5.5-crit4 | 156 | 92.76 | 93.40 | 93.40 | 88.27 | 81.86 | 88.46 | 1.5 |
| hampton-5.5-crit5 | 295 | 91.49 | 91.49 | 91.49 | 89.46 | 87.42 | 90.88 | 5.9 |
| flanders-03-thematic-crit | 1309 | 76.10 | 76.95 | 76.95 | 75.57 | 75.05 | 77.14 | 24.7 |
| anatomy-30b-hamstring-crit | 951 | 84.19 | 85.70 | 85.16 | 83.71 | 82.44 | 84.90 | 18.0 |
| colman-04.03-life-crit | 495 | 78.14 | 77.74 | 76.93 | 74.71 | 73.21 | 76.18 | 7.9 |
| colman-05.02-master-studies-crit | 381 | 65.64 | 68.53 | 69.06 | 65.91 | 68.35 | 66.33 | 8.7 |
| colman-06.06-species-crit | 373 | 76.09 | 77.16 | 75.76 | 76.41 | 74.02 | 79.57 | 4.3 |
| hampton-7-conclusion | 43 | 83.95 | 81.63 | 79.30 | 79.30 | 81.63 | 72.33 | 2.2 |
| greco-2.2-thumbnailing | 1388 | 78.16 | 80.38 | 78.79 | 72.02 | 76.80 | 65.09 | 21.1 |

## Ladder, 18 episodes

The fit six enter with their leave-one-out predictions and the 12 ladder held-out episodes with the frozen weights, all at the frozen threshold 3.00; greco-2.2-thumbnailing is not a ladder episode and is left out here. jev_a v3 is rebuilt through the same scoring path and reproduces its published 80.47 at 80.47. Seconds per episode: 7.9 (f1 plus v3).

| arm | SENTENCE POINTS | s/episode |
|---|---:|---:|
| shipped Opus agentic | 86.47 |  |
| best Luna chapters | 83.71 |  |
| Jev f1 `q+code+v3` combiner (this build) | 82.91 | 7.9 |
| Jev jev_a v3 (pure Jev) | 80.47 |  |
| Luna single call | 65.36 |  |
| deterministic baseline (um removal + retakes + delete silence) | 63.72 |  |

## Standardised weights, chosen set

`q+code+v3` refitted on all 6 fit episodes with C 0.003, sorted by size. A positive weight pushes toward keep. Weights are per standard deviation of the feature, so they compare across features. Intercept 0.234.

| feature | weight | what it says |
|---|---:|---:|
| `v3_score` | +0.317 | v3 0-5 score; pushes toward keep |
| `off_topic` | -0.303 | yes: off the lesson's topic; pushes toward cut |
| `pre_lesson` | -0.251 | yes: chatter before the lesson starts; pushes toward cut |
| `screen_ops` | -0.225 | yes: operating the screen or software; pushes toward cut |
| `crew_talk` | -0.223 | yes: addressed to the crew, not students; pushes toward cut |
| `v3_cut_p` | -0.198 | v3 P(editor removes it); pushes toward cut |
| `essential` | +0.193 | yes: the lesson loses something without it; pushes toward keep |
| `is_retake` | -0.192 | corpus retake flag (module loser); pushes toward cut |
| `asr_confidence` | +0.188 | mean ASR word confidence; pushes toward keep |
| `pure_filler` | -0.185 | yes: filler with no content; pushes toward cut |
| `describes_screen` | +0.158 | yes: only describes what is on screen; pushes toward keep |
| `teaching_point` | +0.154 | yes: states a point, reason or correction; pushes toward keep |
| `duration_s` | +0.153 | spoken duration in seconds; pushes toward keep |
| `referenced_later` | -0.141 | yes: a later sentence depends on it; pushes toward cut |
| `repeats_point` | -0.137 | yes: repeats a point just made; pushes toward cut |
| `tangent` | -0.127 | yes: an aside the lesson resumes after; pushes toward cut |
| `retake_loser` | -0.122 | yes: a losing take of a repeated line; pushes toward cut |
| `overlap_next` | -0.115 | word overlap with the next sentence; pushes toward cut |
| `retake_member` | -0.111 | in a retake group; pushes toward cut |
| `funny` | +0.102 | yes: funny or shows personality; pushes toward keep |
| `n_words` | +0.101 | word count after um stripping; pushes toward keep |
| `n_words_raw` | +0.100 | word count as spoken; pushes toward keep |
| `trail_off` | -0.100 | row ends in the transcriber's '..' mark; pushes toward cut |
| `position` | -0.096 | position in the episode, 0 to 1; pushes toward cut |
| `v3_last_p_whole` | -0.091 | v3 P(nothing trimmed from the end); pushes toward cut |
| `rambling` | -0.082 | yes: rambling or thinking aloud; pushes toward cut |
| `transition` | +0.081 | yes: a spoken transition between students or steps; pushes toward keep |
| `retake_winner` | +0.073 | the module's winning take; pushes toward keep |
| `words_per_s` | -0.068 | speaking rate; pushes toward cut |
| `false_start` | -0.058 | yes: the row is an abandoned attempt; pushes toward cut |
| `split_fragment` | +0.056 | yes: half of a transcriber-split sentence; pushes toward keep |
| `chain_piece` | -0.041 | piece index in a split-sentence chain; pushes toward cut |
| `pep_talk` | -0.027 | yes: praise or wrap-up with nothing new; pushes toward cut |
| `um_detected` | -0.019 | um-like words as spoken; pushes toward cut |
| `um_removed` | -0.018 | words the um module removed; pushes toward cut |
| `pause_before` | -0.018 | seconds of silence before the sentence; pushes toward cut |
| `lower_start` | +0.013 | row starts lowercase (continuation); pushes toward keep |
| `overlap_prev` | +0.010 | word overlap with the previous sentence; pushes toward keep |
| `pause_after` | -0.003 | seconds of silence after the sentence; pushes toward cut |
| `v3_first_p_whole` | -0.003 | v3 P(nothing trimmed from the start); pushes toward cut |
| `chain_len` | -0.003 | length of the split-sentence chain; pushes toward cut |

## Each question alone

AUC of each probability of yes against the editor's keep (full or partial) versus removed, pooled over the fit six and, when present, the held-out episodes. 0.50 is no signal; a cut question reads below 0.50 and a keep question above. The v3 score and cut_p are listed on the same footing.

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
| `pep_talk` | 0.399 | 0.354 | 0.101 | yes |
| `transition` | 0.543 | 0.449 | 0.043 | weak |
| `describes_screen` | 0.519 | 0.540 | 0.019 | none |
| `funny` | 0.491 | 0.478 | 0.009 | none |
| `split_fragment` | 0.508 | 0.543 | 0.008 | none |
| `referenced_later` | 0.503 | 0.589 | 0.003 | none |

Questions with no signal on their own (abs(AUC - 0.5) under 0.03): `funny`, `referenced_later`, `describes_screen`, `split_fragment`.

## Confusion

Keep or cut of the chosen arm against the editor and against jev_a v3, with modules, per split. Fit rows are leave-one-out; held-out rows use the frozen weights.

| split | reference | n | both keep | ref keep, f1 cut | ref cut, f1 keep | both cut | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit | editor | 2743 | 1299 | 206 | 160 | 1078 | 86.66 | 7.0 |
| fit | jev_a v3 | 2743 | 1353 | 165 | 106 | 1119 | 90.12 | 7.0 |
| heldout | editor | 7588 | 3809 | 595 | 641 | 2543 | 83.71 | 9.3 |
| heldout | jev_a v3 | 7588 | 4232 | 784 | 218 | 2354 | 86.79 | 9.3 |
| ladder18 | editor | 8943 | 4783 | 760 | 571 | 2829 | 85.12 | 7.9 |
| ladder18 | jev_a v3 | 8943 | 5045 | 712 | 309 | 2877 | 88.58 | 7.9 |

## Calibration by quartile

The chosen combiner's `p_keep` over the 18 ladder episodes (fit leave-one-out plus held-out frozen). The margin table is the input route 2 needs: the bottom margin quartile is the slice a bigger model would take.

| p_keep quartile | n | p range | mean p_keep | editor kept | arm kept | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2236 | 0.00 to 0.34 | 0.153 | 9.30 | 0.00 | 90.70 | 7.9 |
| 2 | 2235 | 0.34 to 0.74 | 0.554 | 53.78 | 40.89 | 64.38 | 7.9 |
| 3 | 2236 | 0.74 to 0.88 | 0.819 | 87.70 | 98.70 | 88.19 | 7.9 |
| 4 | 2236 | 0.88 to 1.00 | 0.929 | 97.14 | 99.87 | 97.18 | 7.9 |

| margin quartile | n | margin range | mean p_keep | editor kept | arm kept | agreement | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2236 | 0.00 to 0.88 | 0.626 | 63.73 | 59.44 | 67.53 | 7.9 |
| 2 | 2235 | 0.88 to 1.39 | 0.719 | 73.83 | 74.90 | 84.52 | 7.9 |
| 3 | 2236 | 1.39 to 1.77 | 0.812 | 83.68 | 83.32 | 93.92 | 7.9 |
| 4 | 2236 | 1.77 to 3.00 | 0.300 | 26.70 | 21.82 | 94.50 | 7.9 |

## What this hands route 2

Bottom share of the 18 ladder episodes by combiner margin `abs(5 * p_keep - threshold)`, one global cutoff, the archived Luna chapters decision substituted on the routed slice and rescored with modules, exactly as `roughcut_route2_routing.py` does for the v3 margin. The v3 margin gives 84.06 at 25%. Seconds per episode are Jev only; the Luna call is build A's number.

| combiner | share | routed | margin cutoff | SENTENCE POINTS | WORD SCORE | s/episode |
|---|---:|---:|---:|---:|---:|---:|
| `q+code+v3` | 25% | 2236 | 0.879 | 84.81 | 78.72 | 7.9 |
| `q+code+v3` | 50% | 4472 | 1.391 | 84.46 | 79.54 | 7.9 |
| v3 margin, archived Luna (route 2 write-up) | 25% | 2236 | 0.46 | 84.06 |  | 5.6 |

## Seconds and dollars per episode

The f1 pass at concurrency 8 (two requests per 25-sentence block, every part carrying the whole state), plus the v3 run it joins. Cost is the router's usage accounting at $0.042 per million input tokens.

| episode | split | sentences | f1 requests | f1 errors | unanswered cells | f1 s | v3 s | total s | f1 $ | v3 $ | total $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | fit | 194 | 16 | 0 | 0 | 0.79 | 4.45 | 5.23 | 0.0178 | 0.0149 | 0.0327 |
| hampton-5.4-assignment-demo | fit | 300 | 24 | 0 | 0 | 1.13 | 2.79 | 3.92 | 0.0268 | 0.0213 | 0.0481 |
| colman-03.03-muscles-crit | fit | 303 | 25 | 0 | 0 | 1.18 | 3.10 | 4.28 | 0.0307 | 0.0252 | 0.0559 |
| edges-7.01-intro | fit | 389 | 31 | 0 | 0 | 1.41 | 4.90 | 6.31 | 0.0326 | 0.0275 | 0.0601 |
| hampton-5.2-shape-demo | fit | 411 | 33 | 0 | 0 | 1.72 | 3.20 | 4.92 | 0.0427 | 0.0338 | 0.0764 |
| perspective-14e-boxes-critique | fit | 1146 | 92 | 0 | 0 | 4.95 | 12.44 | 17.39 | 0.1408 | 0.1053 | 0.2460 |
| perspective-13d-critique | held-out | 1752 | 141 | 0 | 0 | 7.31 | 13.61 | 20.92 | 0.2160 | 0.1252 | 0.3413 |
| hampton-5.5-crit1 | held-out | 181 | 15 | 0 | 0 | 0.79 | 2.22 | 3.00 | 0.0173 | 0.0152 | 0.0325 |
| hampton-5.5-crit2 | held-out | 127 | 11 | 0 | 0 | 0.60 | 0.93 | 1.53 | 0.0115 | 0.0102 | 0.0217 |
| hampton-5.5-crit3 | held-out | 137 | 11 | 0 | 0 | 0.70 | 0.99 | 1.69 | 0.0125 | 0.0112 | 0.0238 |
| hampton-5.5-crit4 | held-out | 156 | 13 | 0 | 0 | 0.69 | 0.79 | 1.49 | 0.0151 | 0.0133 | 0.0284 |
| hampton-5.5-crit5 | held-out | 295 | 24 | 0 | 0 | 1.13 | 4.81 | 5.94 | 0.0299 | 0.0245 | 0.0544 |
| flanders-03-thematic-crit | held-out | 1309 | 105 | 0 | 0 | 5.48 | 19.26 | 24.74 | 0.1541 | 0.1246 | 0.2787 |
| anatomy-30b-hamstring-crit | held-out | 951 | 77 | 0 | 0 | 6.20 | 11.76 | 17.97 | 0.1148 | 0.0913 | 0.2061 |
| colman-04.03-life-crit | held-out | 495 | 40 | 0 | 0 | 1.99 | 5.88 | 7.87 | 0.0558 | 0.0420 | 0.0978 |
| colman-05.02-master-studies-crit | held-out | 381 | 31 | 0 | 0 | 3.71 | 5.01 | 8.72 | 0.0413 | 0.0329 | 0.0742 |
| colman-06.06-species-crit | held-out | 373 | 30 | 0 | 0 | 1.51 | 2.83 | 4.34 | 0.0386 | 0.0307 | 0.0693 |
| hampton-7-conclusion | held-out | 43 | 3 | 0 | 0 | 0.41 | 1.81 | 2.23 | 0.0032 | 0.0030 | 0.0062 |
| greco-2.2-thumbnailing | held-out | 1388 | 112 | 0 | 0 | 8.08 | 13.02 | 21.09 | 0.1631 | 0.1156 | 0.2787 |

Spend on this build: $0.2913 on the fit six, $0.8732 on the held-out episodes, $0.0022 on the smoke block, $1.1667 in all against the $3 cap. Requests: 836, errors 0, unanswered cells 0.

## Notes

- Target per sentence: editor kept (full or partial) versus removed, from the harness's human sentence states, the same states SENTENCE POINTS reads. Fit six: 1505 kept of 2743.
- Missing values: an unanswered noul is filled with 0.5, a missing v3 score with 2.5, a missing v3 cut_p with 0.5, a missing trim answer (one-word rows) with 1.0 (whole). Counts: {'fit': {'q_cells': 0, 'v3_score': 0, 'v3_cut_p': 0, 'sentences': 2743}, 'heldout': {'q_cells': 0, 'v3_score': 0, 'v3_cut_p': 0, 'sentences': 7588}}.
- Fit-set threshold: calibrated by the harness's pooled Neutral sweep on the out-of-fold predictions of all six episodes, so it is chosen on the fit set only and then frozen with the weights. The pooled SP in the tables is the sentence-count weighted mean of per-episode SP; the harness's own pooled figure for the chosen set is 85.63.
- The v3 control lands at 82.86 against 83.00 for jev_a v3 as published on the same six and 82.97 with keep_words null. The control has no trims and a logistic squashing of the score; the keep threshold moves accordingly.
- The questions are TypeSafe nouls (probability of yes), one per target sentence per question, the same primitive as v3's cut_k. Each block's 18 questions go out in two requests that both carry the full v3 state; the answers never see each other.
- Selection was among the spec's four sets only; `code+v3` is reported as a diagnostic and was not eligible.
- Held-out numbers use the frozen weights and the frozen threshold. Nothing was refitted, recalibrated or chosen after the held-out features were read; stage 2 refuses to run if the stage 1 recomputation drifts from the frozen file.
- On held-out, `q+code` (81.37) edges the chosen `q+code+v3` (81.17). The choice stands: it was made on the fit six before any held-out number was read, and the gap is inside the per-episode spread.
- Request shape: the 18 questions go out as two requests per block, both carrying the whole v3 state; a block of 25 sentences measured 51,318 real input tokens across its two requests on the smoke block. Real tokens ran about 1/0.87 of the 4-chars-per-token estimate over the held-out run. Cost per sentence came out about the same as v3's sentence pass, not double as the spec expected, because a noul question is short next to v3's word-level trim choices.
- The smoke block (`roughcut-jev-f1-smoke-*`, block 0 of colman-02.04-skeleton-demo) is kept on disk and counted in the spend; its answers were not used for fitting.

## Inputs

- fit features: `docs/jev-real/roughcut-jev-f1-fit-features.jsonl` md5 bca3b191584ab0d8096ae731eb11ce32, 2936519 bytes, modified 2026-09-26T07:35:32+00:00
- fit timing: `docs/jev-real/roughcut-jev-f1-fit-timing.json` md5 0d0506b78962b9bf55d5156df279c575, 11503 bytes, modified 2026-09-26T07:35:32+00:00
- frozen weights: `docs/jev-real/roughcut-jev-f1-weights.json` md5 242943da549e74f78f45d096ecb0d638, 16656 bytes, modified 2026-09-26T07:48:20+00:00
- held-out features: `docs/jev-real/roughcut-jev-f1-heldout-features.jsonl` md5 37c568d73d2c775d5d13c52ad8cb4c5d, 8157947 bytes, modified 2026-09-26T07:49:47+00:00
- held-out timing: `docs/jev-real/roughcut-jev-f1-heldout-timing.json` md5 f5e20384e73eaa5e160d8ce13769098a, 23588 bytes, modified 2026-09-26T07:49:47+00:00
- v3 decisions roughcut-jev-all18-v3: `docs/jev-real/roughcut-jev-all18-v3-decisions.jsonl` md5 0063cbe0dd7f4b2a6314986c8b1295c4, 16041883 bytes, modified 2026-09-26T06:39:11+00:00
- v3 decisions roughcut-jev-heldout-v3: `docs/jev-real/roughcut-jev-heldout-v3-decisions.jsonl` md5 400c49b431ec04e46d9428fc6c399b68, 13570354 bytes, modified 2026-09-26T06:39:11+00:00
- prompt bundle: `scripts/jev_real/roughcut_jev_prompts.py` md5 17e08d9e30f9fc42928a5e1e7a2b080f, 22435 bytes, modified 2026-09-26T07:31:18+00:00
