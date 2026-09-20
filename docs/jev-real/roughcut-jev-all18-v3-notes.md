# Jev rough cut: run report

Generated 2026-09-20T22:45:13+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-all18-v3-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v3, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 18 episode(s): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique, perspective-13d-critique, hampton-5.5-crit1, hampton-5.5-crit2, hampton-5.5-crit3, hampton-5.5-crit4, hampton-5.5-crit5, flanders-03-thematic-crit, anatomy-30b-hamstring-crit, colman-04.03-life-crit, colman-05.02-master-studies-crit, colman-06.06-species-crit, hampton-7-conclusion.
575 requests, 134 errored, 128 retried, 18,149,991 input tokens, $0.7623, 100.777 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 8943.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 8943 sentences with 0 mismatches.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 72.19 | 74.61 | 72.30 | 80.69 | 69.63 | 107.14 | 5.60 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 80.47 | 80.47 | 74.12 | 88.44 | 75.79 | 95.16 | 5.60 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 62.27 | 62.27 | 66.31 | 85.88 | 61.31 | 121.22 | no | 4.45 |
| hampton-5.4-assignment-demo | 71.17 | 71.17 | 73.30 | 82.42 | 67.00 | 118.69 | no | 2.79 |
| colman-03.03-muscles-crit | 72.41 | 72.41 | 58.53 | 69.90 | 58.81 | 92.76 | no | 3.10 |
| edges-7.01-intro | 27.89 | 77.89 | 79.33 | 79.86 | 78.16 | 82.06 | yes | 4.90 |
| hampton-5.2-shape-demo | 78.91 | 78.91 | 84.86 | 89.41 | 77.54 | 115.56 | no | 3.20 |
| perspective-14e-boxes-critique | 80.03 | 80.03 | 78.89 | 80.80 | 76.21 | 102.75 | no | 12.44 |
| perspective-13d-critique | 76.03 | 76.03 | 78.61 | 80.54 | 75.82 | 110.63 | no | 13.61 |
| hampton-5.5-crit1 | 82.60 | 82.60 | 63.44 | 87.08 | 60.77 | 105.02 | no | 2.22 |
| hampton-5.5-crit2 | 79.21 | 79.21 | 57.72 | 81.69 | 57.28 | 99.06 | no | 0.93 |
| hampton-5.5-crit3 | 81.02 | 81.02 | 67.92 | 91.28 | 65.96 | 105.06 | no | 0.99 |
| hampton-5.5-crit4 | 84.42 | 84.42 | 61.08 | 84.99 | 59.46 | 100.77 | no | 0.79 |
| hampton-5.5-crit5 | 84.34 | 84.34 | 63.25 | 82.18 | 61.08 | 104.82 | no | 4.81 |
| flanders-03-thematic-crit | 73.64 | 73.64 | 76.80 | 79.68 | 72.54 | 96.03 | no | 19.26 |
| anatomy-30b-hamstring-crit | 74.48 | 74.48 | 75.22 | 81.26 | 70.50 | 104.43 | no | 11.76 |
| colman-04.03-life-crit | 66.85 | 66.85 | 63.50 | 70.04 | 61.39 | 122.55 | no | 5.88 |
| colman-05.02-master-studies-crit | 58.16 | 58.16 | 63.09 | 77.64 | 60.36 | 125.09 | no | 5.01 |
| colman-06.06-species-crit | 65.76 | 65.76 | 66.06 | 81.55 | 61.26 | 128.92 | no | 2.83 |
| hampton-7-conclusion | 20.70 | 70.70 | 71.66 | 80.68 | 70.02 | 104.79 | yes | 1.81 |
| **pooled** | 72.19 | 74.61 | 72.30 | 80.69 | 69.63 | 107.14 | — | 5.60 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 85.88 | 85.88 | 74.48 | 106.07 | 75.72 | 105.97 | no | 4.45 |
| hampton-5.4-assignment-demo | 84.80 | 84.80 | 75.53 | 94.34 | 76.69 | 99.62 | no | 2.79 |
| colman-03.03-muscles-crit | 75.05 | 75.05 | 59.92 | 74.19 | 62.42 | 85.28 | no | 3.10 |
| edges-7.01-intro | 79.69 | 79.69 | 79.74 | 79.65 | 77.95 | 75.82 | no | 4.90 |
| hampton-5.2-shape-demo | 92.70 | 92.70 | 85.72 | 96.97 | 84.10 | 100.13 | no | 3.20 |
| perspective-14e-boxes-critique | 81.78 | 81.78 | 79.21 | 82.33 | 77.65 | 96.14 | no | 12.44 |
| perspective-13d-critique | 78.09 | 78.09 | 79.05 | 83.28 | 78.40 | 101.97 | no | 13.61 |
| hampton-5.5-crit1 | 87.35 | 87.35 | 67.36 | 111.46 | 77.79 | 96.65 | no | 2.22 |
| hampton-5.5-crit2 | 79.45 | 79.45 | 59.31 | 102.20 | 71.65 | 90.95 | no | 0.93 |
| hampton-5.5-crit3 | 81.17 | 81.17 | 68.98 | 105.91 | 76.53 | 96.00 | no | 0.99 |
| hampton-5.5-crit4 | 88.46 | 88.46 | 64.13 | 104.93 | 73.41 | 92.47 | no | 0.79 |
| hampton-5.5-crit5 | 90.88 | 90.88 | 67.20 | 103.00 | 76.55 | 95.24 | no | 4.81 |
| flanders-03-thematic-crit | 77.14 | 77.14 | 77.61 | 83.24 | 75.79 | 82.18 | no | 19.26 |
| anatomy-30b-hamstring-crit | 84.90 | 84.90 | 78.18 | 89.85 | 77.95 | 82.46 | no | 11.76 |
| colman-04.03-life-crit | 76.18 | 76.18 | 65.37 | 77.14 | 67.62 | 110.91 | no | 5.88 |
| colman-05.02-master-studies-crit | 66.33 | 66.33 | 65.98 | 88.21 | 68.57 | 114.18 | no | 5.01 |
| colman-06.06-species-crit | 79.57 | 79.57 | 70.43 | 97.21 | 73.02 | 115.66 | no | 2.83 |
| hampton-7-conclusion | 72.33 | 72.33 | 72.07 | 85.16 | 73.91 | 98.32 | no | 1.81 |
| **pooled** | 80.47 | 80.47 | 74.12 | 88.44 | 75.79 | 95.16 | — | 5.60 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0.29 | 4.16 | n/a | 4.45 | 12 | 354,246 | 0.0149 | 3 | 3 |
| hampton-5.4-assignment-demo | 0.23 | 2.56 | n/a | 2.79 | 15 | 507,202 | 0.0213 | 2 | 2 |
| colman-03.03-muscles-crit | 0.32 | 2.78 | n/a | 3.10 | 17 | 599,799 | 0.0252 | 2 | 2 |
| edges-7.01-intro | 0.41 | 4.49 | n/a | 4.90 | 33 | 654,776 | 0.0275 | 6 | 6 |
| hampton-5.2-shape-demo | 0.25 | 2.95 | n/a | 3.20 | 23 | 803,557 | 0.0337 | 3 | 3 |
| perspective-14e-boxes-critique | 0.47 | 11.97 | n/a | 12.44 | 89 | 2,618,694 | 0.1100 | 26 | 29 |
| perspective-13d-critique | 0.61 | 13.01 | n/a | 13.61 | 114 | 3,058,721 | 0.1285 | 24 | 26 |
| hampton-5.5-crit1 | 0.22 | 2.00 | n/a | 2.22 | 10 | 360,841 | 0.0152 | 1 | 1 |
| hampton-5.5-crit2 | 0.20 | 0.73 | n/a | 0.93 | 7 | 243,203 | 0.0102 | 0 | 0 |
| hampton-5.5-crit3 | 0.22 | 0.77 | n/a | 0.99 | 7 | 267,744 | 0.0112 | 0 | 0 |
| hampton-5.5-crit4 | 0.00 | 0.79 | n/a | 0.79 | 7 | 317,488 | 0.0133 | 0 | 0 |
| hampton-5.5-crit5 | 0.21 | 4.60 | n/a | 4.81 | 15 | 582,998 | 0.0245 | 2 | 2 |
| flanders-03-thematic-crit | 0.27 | 18.99 | n/a | 19.26 | 97 | 2,966,909 | 0.1246 | 38 | 38 |
| anatomy-30b-hamstring-crit | 0.37 | 11.40 | n/a | 11.76 | 58 | 2,174,169 | 0.0913 | 10 | 10 |
| colman-04.03-life-crit | 0.42 | 5.46 | n/a | 5.88 | 28 | 1,053,621 | 0.0443 | 4 | 5 |
| colman-05.02-master-studies-crit | 0.26 | 4.75 | n/a | 5.01 | 21 | 782,659 | 0.0329 | 4 | 4 |
| colman-06.06-species-crit | 0.21 | 2.62 | n/a | 2.83 | 18 | 731,524 | 0.0307 | 2 | 2 |
| hampton-7-conclusion | 0.18 | 1.63 | n/a | 1.81 | 4 | 71,840 | 0.0030 | 1 | 1 |
| **total** | 5.11 | 95.67 | 0.00 | 100.78 | 575 | 18,149,991 | 0.7623 | 128 | 134 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 9 | 5 | 91 | no |
| hampton-5.4-assignment-demo | 5 | 2 | 77 | no |
| colman-03.03-muscles-crit | 16 | 11 | 70 | no |
| edges-7.01-intro | 8 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 22 | 11 | 77 | no |
| perspective-14e-boxes-critique | 29 | 6 | 58 | no |
| perspective-13d-critique | 38 | 9 | 107 | no |
| hampton-5.5-crit1 | 9 | 6 | 28 | no |
| hampton-5.5-crit2 | 5 | 2 | 18 | no |
| hampton-5.5-crit3 | 11 | 8 | 27 | no |
| hampton-5.5-crit4 | 8 | 7 | 27 | no |
| hampton-5.5-crit5 | 7 | 6 | 44 | no |
| flanders-03-thematic-crit | 50 | 29 | 145 | no |
| anatomy-30b-hamstring-crit | 25 | 14 | 150 | no |
| colman-04.03-life-crit | 18 | 11 | 110 | no |
| colman-05.02-master-studies-crit | 18 | 14 | 149 | no |
| colman-06.06-species-crit | 13 | 8 | 159 | no |
| hampton-7-conclusion | 2 | 0 | 11 | yes |
| **total** | 293 | 149 | 1363 | — |

## The cut_k question (cut_p)

`cut_p` is P(the editor removes this sentence) from the noul asked next to the 0-5 score on every target. `jev_noul` scores a sentence 5 x (1 - cut_p), `jev_mix` averages that with the score; both carry jev_a's trims and retake cut, so the keep decision is the only thing that differs.

| episode | sentences | unusable | r(cut_p, score) | mean cut_p | mean score |
|---|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 0 | -0.846 | 0.320 | 3.21 |
| hampton-5.4-assignment-demo | 300 | 0 | -0.880 | 0.306 | 3.00 |
| colman-03.03-muscles-crit | 303 | 0 | -0.818 | 0.304 | 3.12 |
| edges-7.01-intro | 389 | 0 | -0.877 | 0.478 | 2.26 |
| hampton-5.2-shape-demo | 411 | 0 | -0.893 | 0.335 | 2.82 |
| perspective-14e-boxes-critique | 1146 | 0 | -0.775 | 0.414 | 2.15 |
| perspective-13d-critique | 1752 | 0 | -0.788 | 0.407 | 2.27 |
| hampton-5.5-crit1 | 181 | 0 | -0.745 | 0.273 | 3.27 |
| hampton-5.5-crit2 | 127 | 0 | -0.805 | 0.246 | 3.22 |
| hampton-5.5-crit3 | 137 | 0 | -0.829 | 0.285 | 3.22 |
| hampton-5.5-crit4 | 156 | 0 | -0.744 | 0.283 | 3.34 |
| hampton-5.5-crit5 | 295 | 0 | -0.628 | 0.242 | 3.38 |
| flanders-03-thematic-crit | 1309 | 0 | -0.812 | 0.371 | 2.79 |
| anatomy-30b-hamstring-crit | 951 | 0 | -0.856 | 0.360 | 2.85 |
| colman-04.03-life-crit | 495 | 0 | -0.823 | 0.300 | 3.24 |
| colman-05.02-master-studies-crit | 381 | 0 | -0.775 | 0.298 | 3.20 |
| colman-06.06-species-crit | 373 | 0 | -0.785 | 0.326 | 3.27 |
| hampton-7-conclusion | 43 | 0 | -0.907 | 0.356 | 2.57 |
| **pooled** | 8943 | — | -0.824 | 0.360 | 2.73 |

Keep/cut agreement with the editor at each arm's own calibrated threshold, layered scoring over 18 episode(s). Counts are sentences: `wrong drops` is the editor kept it and the arm removed it.

| arm | threshold | sentences | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio | SENTENCE POINTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 8943 | 4872 | 2515 | 671 | 885 | 82.60 | 95.16 | 80.47 |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 6 | 2 | 0 | 4 | 6 | 0 (0 kept / 0 cut) | 2 (1 kept / 1 cut) |
| hampton-5.4-assignment-demo | 6 | 5 | 0 | 1 | 7 | 0 (0 kept / 0 cut) | 6 (1 kept / 5 cut) |
| colman-03.03-muscles-crit | 8 | 6 | 0 | 2 | 8 | 1 (0 kept / 1 cut) | 7 (5 kept / 2 cut) |
| edges-7.01-intro | 63 | 19 | 0 | 108 | 169 | 5 (2 kept / 3 cut) | 66 (21 kept / 45 cut) |
| hampton-5.2-shape-demo | 14 | 2 | 0 | 14 | 16 | 0 (0 kept / 0 cut) | 2 (0 kept / 2 cut) |
| perspective-14e-boxes-critique | 80 | 24 | 0 | 60 | 98 | 4 (0 kept / 4 cut) | 42 (9 kept / 33 cut) |
| perspective-13d-critique | 101 | 47 | 0 | 70 | 157 | 4 (1 kept / 3 cut) | 91 (10 kept / 81 cut) |
| hampton-5.5-crit1 | 2 | 0 | 0 | 2 | 2 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit2 | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit3 | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit4 | 0 | 0 | 0 | 0 | 0 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit5 | 1 | 1 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (1 kept / 0 cut) |
| flanders-03-thematic-crit | 31 | 9 | 0 | 80 | 96 | 3 (3 kept / 0 cut) | 19 (11 kept / 8 cut) |
| anatomy-30b-hamstring-crit | 51 | 21 | 0 | 32 | 57 | 0 (0 kept / 0 cut) | 25 (10 kept / 15 cut) |
| colman-04.03-life-crit | 14 | 5 | 0 | 10 | 16 | 0 (0 kept / 0 cut) | 6 (4 kept / 2 cut) |
| colman-05.02-master-studies-crit | 4 | 3 | 0 | 1 | 6 | 0 (0 kept / 0 cut) | 5 (3 kept / 2 cut) |
| colman-06.06-species-crit | 6 | 3 | 0 | 3 | 6 | 0 (0 kept / 0 cut) | 3 (1 kept / 2 cut) |
| hampton-7-conclusion | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |

## Where this lands on the published ladder

Reference arms re-pooled over the same 18 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 18 | 79.96 | 79.44 | 88.34 | 83.71 | 80.35 | 93.55 |
| shipped Opus agentic | 18 | 83.45 | 80.84 | 90.37 | 86.47 | 81.44 | 95.02 |
| Luna single call | 18 | 18.61 | 61.23 | 69.43 | 65.36 | 64.63 | 80.67 |
| deterministic baseline (um removal + retakes + delete silence) | 18 | 63.72 | 64.07 | 79.92 | 63.72 | 64.07 | 79.92 |

Best Jev arm here is jev_a (t_trim 0.3) at 72.19 SENTENCE POINTS (72.30 WORD SCORE, 5.60 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 80.47 SENTENCE POINTS (74.12 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v3-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v3-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v3-summary.json`
