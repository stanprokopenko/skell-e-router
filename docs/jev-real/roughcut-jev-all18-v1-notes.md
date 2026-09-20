# Jev rough cut: run report

Generated 2026-09-20T22:45:10+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-all18-v1-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v1, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 18 episode(s): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique, perspective-13d-critique, hampton-5.5-crit1, hampton-5.5-crit2, hampton-5.5-crit3, hampton-5.5-crit4, hampton-5.5-crit5, flanders-03-thematic-crit, anatomy-30b-hamstring-crit, colman-04.03-life-crit, colman-05.02-master-studies-crit, colman-06.06-species-crit, hampton-7-conclusion.
707 requests, 113 errored, 106 retried, 16,113,810 input tokens, $0.6768, 82.942 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 8943.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 8943 sentences with 0 mismatches.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.10 | 72.01 | 74.42 | 72.59 | 80.96 | 69.90 | 107.80 | 4.61 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 1.90 | 79.41 | 79.41 | 74.18 | 88.52 | 75.88 | 100.23 | 4.61 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 63.66 | 63.66 | 67.54 | 87.97 | 62.80 | 118.21 | no | 2.37 |
| hampton-5.4-assignment-demo | 69.80 | 69.80 | 66.16 | 76.15 | 61.90 | 124.58 | no | 2.42 |
| colman-03.03-muscles-crit | 74.72 | 74.72 | 60.90 | 70.96 | 59.70 | 92.31 | no | 2.90 |
| edges-7.01-intro | 26.09 | 76.09 | 75.96 | 77.67 | 76.02 | 78.70 | yes | 2.85 |
| hampton-5.2-shape-demo | 76.64 | 76.64 | 80.00 | 85.13 | 73.83 | 119.88 | no | 3.35 |
| perspective-14e-boxes-critique | 81.65 | 81.65 | 81.67 | 83.35 | 78.62 | 112.33 | no | 8.82 |
| perspective-13d-critique | 77.15 | 77.15 | 80.60 | 82.39 | 77.56 | 119.73 | no | 10.57 |
| hampton-5.5-crit1 | 82.60 | 82.60 | 69.04 | 89.28 | 62.30 | 99.88 | no | 2.12 |
| hampton-5.5-crit2 | 79.45 | 79.45 | 57.49 | 81.58 | 57.20 | 99.11 | no | 0.97 |
| hampton-5.5-crit3 | 77.08 | 77.08 | 70.35 | 91.28 | 65.96 | 101.78 | no | 1.85 |
| hampton-5.5-crit4 | 81.03 | 81.03 | 56.98 | 80.72 | 56.47 | 97.34 | no | 0.75 |
| hampton-5.5-crit5 | 84.81 | 84.81 | 70.26 | 87.08 | 64.72 | 105.18 | no | 2.43 |
| flanders-03-thematic-crit | 73.52 | 73.52 | 76.11 | 79.10 | 72.01 | 94.74 | no | 17.25 |
| anatomy-30b-hamstring-crit | 70.77 | 70.77 | 75.03 | 81.27 | 70.51 | 97.37 | no | 8.39 |
| colman-04.03-life-crit | 67.37 | 67.37 | 62.07 | 69.80 | 61.19 | 121.35 | no | 3.26 |
| colman-05.02-master-studies-crit | 57.72 | 57.72 | 62.02 | 76.69 | 59.62 | 121.69 | no | 5.96 |
| colman-06.06-species-crit | 65.44 | 65.44 | 64.86 | 80.59 | 60.54 | 127.81 | no | 4.74 |
| hampton-7-conclusion | 25.35 | 75.35 | 76.87 | 85.96 | 74.60 | 109.12 | yes | 1.94 |
| **pooled** | 72.01 | 74.42 | 72.59 | 80.96 | 69.90 | 107.80 | — | 4.61 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 1.90.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 87.53 | 87.53 | 76.19 | 107.87 | 77.01 | 103.67 | no | 2.37 |
| hampton-5.4-assignment-demo | 80.40 | 80.40 | 66.93 | 87.25 | 70.92 | 107.65 | no | 2.42 |
| colman-03.03-muscles-crit | 79.44 | 79.44 | 62.63 | 74.96 | 63.07 | 85.93 | no | 2.90 |
| edges-7.01-intro | 77.53 | 77.53 | 76.59 | 77.45 | 75.81 | 84.26 | no | 2.85 |
| hampton-5.2-shape-demo | 88.78 | 88.78 | 78.35 | 90.95 | 78.88 | 107.42 | no | 3.35 |
| perspective-14e-boxes-critique | 81.40 | 81.40 | 81.87 | 84.95 | 80.12 | 114.61 | no | 8.82 |
| perspective-13d-critique | 75.75 | 75.75 | 79.67 | 84.48 | 79.53 | 118.63 | no | 10.57 |
| hampton-5.5-crit1 | 87.35 | 87.35 | 72.49 | 113.78 | 79.41 | 94.43 | no | 2.12 |
| hampton-5.5-crit2 | 80.47 | 80.47 | 59.38 | 102.60 | 71.94 | 91.37 | no | 0.97 |
| hampton-5.5-crit3 | 77.23 | 77.23 | 67.92 | 104.60 | 75.59 | 93.55 | no | 1.85 |
| hampton-5.5-crit4 | 88.97 | 88.97 | 63.04 | 104.34 | 72.99 | 93.87 | no | 0.75 |
| hampton-5.5-crit5 | 92.64 | 92.64 | 74.48 | 106.79 | 79.37 | 96.04 | no | 2.43 |
| flanders-03-thematic-crit | 77.27 | 77.27 | 77.71 | 83.27 | 75.81 | 84.83 | no | 17.25 |
| anatomy-30b-hamstring-crit | 82.22 | 82.22 | 78.86 | 89.90 | 77.99 | 82.02 | no | 8.39 |
| colman-04.03-life-crit | 76.38 | 76.38 | 64.09 | 76.49 | 67.05 | 111.30 | no | 3.26 |
| colman-05.02-master-studies-crit | 66.51 | 66.51 | 65.45 | 87.80 | 68.25 | 114.58 | no | 5.96 |
| colman-06.06-species-crit | 77.69 | 77.69 | 68.25 | 95.55 | 71.77 | 115.39 | no | 4.74 |
| hampton-7-conclusion | 72.33 | 72.33 | 69.49 | 83.79 | 72.72 | 107.07 | no | 1.94 |
| **pooled** | 79.41 | 79.41 | 74.18 | 88.52 | 75.88 | 100.23 | — | 4.61 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0.35 | 1.34 | 0.67 | 2.37 | 24 | 363,424 | 0.0153 | 3 | 3 |
| hampton-5.4-assignment-demo | 0.23 | 1.56 | 0.64 | 2.42 | 29 | 490,494 | 0.0206 | 1 | 1 |
| colman-03.03-muscles-crit | 0.21 | 1.79 | 0.90 | 2.90 | 38 | 639,506 | 0.0269 | 1 | 1 |
| edges-7.01-intro | 0.52 | 1.64 | 0.69 | 2.85 | 45 | 609,990 | 0.0256 | 2 | 2 |
| hampton-5.2-shape-demo | 0.25 | 2.16 | 0.95 | 3.35 | 46 | 809,180 | 0.0340 | 2 | 3 |
| perspective-14e-boxes-critique | 0.76 | 5.88 | 2.19 | 8.82 | 141 | 2,622,403 | 0.1101 | 13 | 16 |
| perspective-13d-critique | 0.61 | 9.96 | n/a | 10.57 | 111 | 2,484,109 | 0.1043 | 21 | 23 |
| hampton-5.5-crit1 | 0.32 | 1.81 | n/a | 2.12 | 12 | 301,520 | 0.0127 | 3 | 3 |
| hampton-5.5-crit2 | 0.19 | 0.79 | n/a | 0.97 | 7 | 201,576 | 0.0085 | 0 | 0 |
| hampton-5.5-crit3 | 0.22 | 1.63 | n/a | 1.85 | 8 | 222,851 | 0.0094 | 1 | 1 |
| hampton-5.5-crit4 | 0.00 | 0.75 | n/a | 0.75 | 7 | 266,362 | 0.0112 | 0 | 0 |
| hampton-5.5-crit5 | 0.17 | 2.25 | n/a | 2.43 | 17 | 486,303 | 0.0204 | 4 | 4 |
| flanders-03-thematic-crit | 0.30 | 16.96 | n/a | 17.25 | 93 | 2,538,125 | 0.1066 | 34 | 34 |
| anatomy-30b-hamstring-crit | 0.45 | 7.93 | n/a | 8.39 | 56 | 1,861,224 | 0.0782 | 8 | 8 |
| colman-04.03-life-crit | 0.23 | 3.03 | n/a | 3.26 | 27 | 892,346 | 0.0375 | 4 | 4 |
| colman-05.02-master-studies-crit | 0.28 | 5.68 | n/a | 5.96 | 24 | 657,522 | 0.0276 | 6 | 7 |
| colman-06.06-species-crit | 0.26 | 4.48 | n/a | 4.74 | 18 | 609,122 | 0.0256 | 2 | 2 |
| hampton-7-conclusion | 0.23 | 1.71 | n/a | 1.94 | 4 | 57,753 | 0.0024 | 1 | 1 |
| **total** | 5.57 | 71.35 | 6.02 | 82.94 | 707 | 16,113,810 | 0.6768 | 106 | 113 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 14 | 7 | 91 | no |
| hampton-5.4-assignment-demo | 5 | 2 | 77 | no |
| colman-03.03-muscles-crit | 20 | 18 | 70 | no |
| edges-7.01-intro | 7 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 20 | 9 | 77 | no |
| perspective-14e-boxes-critique | 24 | 4 | 58 | no |
| perspective-13d-critique | 44 | 9 | 107 | no |
| hampton-5.5-crit1 | 8 | 4 | 28 | no |
| hampton-5.5-crit2 | 4 | 1 | 18 | no |
| hampton-5.5-crit3 | 9 | 6 | 27 | no |
| hampton-5.5-crit4 | 8 | 3 | 27 | no |
| hampton-5.5-crit5 | 5 | 5 | 44 | no |
| flanders-03-thematic-crit | 48 | 28 | 145 | no |
| anatomy-30b-hamstring-crit | 28 | 15 | 150 | no |
| colman-04.03-life-crit | 19 | 10 | 110 | no |
| colman-05.02-master-studies-crit | 14 | 9 | 149 | no |
| colman-06.06-species-crit | 14 | 9 | 159 | no |
| hampton-7-conclusion | 2 | 0 | 11 | yes |
| **total** | 293 | 139 | 1363 | — |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 6 | 2 | 0 | 4 | 6 | 0 (0 kept / 0 cut) | 2 (1 kept / 1 cut) |
| hampton-5.4-assignment-demo | 6 | 5 | 0 | 1 | 7 | 0 (0 kept / 0 cut) | 6 (1 kept / 5 cut) |
| colman-03.03-muscles-crit | 8 | 6 | 0 | 2 | 8 | 1 (0 kept / 1 cut) | 7 (5 kept / 2 cut) |
| edges-7.01-intro | 63 | 19 | 0 | 109 | 169 | 4 (1 kept / 3 cut) | 64 (21 kept / 43 cut) |
| hampton-5.2-shape-demo | 14 | 3 | 0 | 13 | 16 | 0 (0 kept / 0 cut) | 3 (1 kept / 2 cut) |
| perspective-14e-boxes-critique | 80 | 27 | 0 | 57 | 98 | 4 (0 kept / 4 cut) | 45 (9 kept / 36 cut) |
| perspective-13d-critique | 0 | 0 | 0 | 0 | 157 | 0 (0 kept / 0 cut) | 157 (17 kept / 140 cut) |
| hampton-5.5-crit1 | 0 | 0 | 0 | 0 | 2 | 0 (0 kept / 0 cut) | 2 (1 kept / 1 cut) |
| hampton-5.5-crit2 | 0 | 0 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (1 kept / 0 cut) |
| hampton-5.5-crit3 | 0 | 0 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (0 kept / 1 cut) |
| hampton-5.5-crit4 | 0 | 0 | 0 | 0 | 0 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit5 | 0 | 0 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (1 kept / 0 cut) |
| flanders-03-thematic-crit | 0 | 0 | 0 | 0 | 96 | 0 (0 kept / 0 cut) | 96 (24 kept / 72 cut) |
| anatomy-30b-hamstring-crit | 0 | 0 | 0 | 0 | 57 | 0 (0 kept / 0 cut) | 57 (15 kept / 42 cut) |
| colman-04.03-life-crit | 0 | 0 | 0 | 0 | 16 | 0 (0 kept / 0 cut) | 16 (9 kept / 7 cut) |
| colman-05.02-master-studies-crit | 0 | 0 | 0 | 0 | 6 | 0 (0 kept / 0 cut) | 6 (3 kept / 3 cut) |
| colman-06.06-species-crit | 0 | 0 | 0 | 0 | 6 | 0 (0 kept / 0 cut) | 6 (2 kept / 4 cut) |
| hampton-7-conclusion | 0 | 0 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (0 kept / 1 cut) |

## Where this lands on the published ladder

Reference arms re-pooled over the same 18 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 18 | 79.96 | 79.44 | 88.34 | 83.71 | 80.35 | 93.55 |
| shipped Opus agentic | 18 | 83.45 | 80.84 | 90.37 | 86.47 | 81.44 | 95.02 |
| Luna single call | 18 | 18.61 | 61.23 | 69.43 | 65.36 | 64.63 | 80.67 |
| deterministic baseline (um removal + retakes + delete silence) | 18 | 63.72 | 64.07 | 79.92 | 63.72 | 64.07 | 79.92 |

Best Jev arm here is jev_a (t_trim 0.3) at 72.01 SENTENCE POINTS (72.59 WORD SCORE, 4.61 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 79.41 SENTENCE POINTS (74.18 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v1-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v1-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v1-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v1-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-all18-v1-summary.json`
