# Jev rough cut: run report

Generated 2026-09-20T22:43:44+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-heldout-v1-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v1, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 13 episode(s): perspective-13d-critique, hampton-5.5-crit1, hampton-5.5-crit2, hampton-5.5-crit3, hampton-5.5-crit4, hampton-5.5-crit5, flanders-03-thematic-crit, anatomy-30b-hamstring-crit, colman-04.03-life-crit, colman-05.02-master-studies-crit, colman-06.06-species-crit, hampton-7-conclusion, greco-2.2-thumbnailing.
463 requests, 100 errored, 97 retried, 12,876,720 input tokens, $0.5408, 69.445 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 7588.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 7588 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-heldout-v1-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.
- roughcut-jev-heldout-v1-decisions.jsonl carries no `cut_p`, so the run's prompt version asked no `cut_k` question. The jev_noul, jev_mix arms and the cut_p diagnostic are left out.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.30 | 72.02 | 72.30 | 71.16 | 78.95 | 68.65 | 110.27 | 5.34 |
| jev_b_moduleretakes | 2.30 | 22.55 | 72.55 | 71.32 | 79.07 | 68.77 | 109.72 | 5.34 |
| jev_b_notrim | 2.30 | 22.28 | 72.28 | 71.04 | 78.83 | 68.56 | 110.78 | 5.34 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.30 | 76.75 | 76.75 | 72.58 | 85.81 | 74.10 | 98.37 | 5.34 |
| jev_b_moduleretakes | 2.30 | 77.01 | 77.01 | 72.76 | 85.99 | 74.26 | 97.81 | 5.34 |
| jev_b_notrim | 2.30 | 76.86 | 76.86 | 72.49 | 85.78 | 74.08 | 98.77 | 5.34 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.30.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 79.43 | 79.43 | 80.97 | 82.60 | 77.75 | 108.33 | no | 10.57 |
| hampton-5.5-crit1 | 82.04 | 82.04 | 69.28 | 89.31 | 62.33 | 98.83 | no | 2.12 |
| hampton-5.5-crit2 | 78.66 | 78.66 | 57.28 | 81.31 | 57.01 | 98.79 | no | 0.97 |
| hampton-5.5-crit3 | 74.89 | 74.89 | 70.30 | 90.03 | 65.05 | 97.73 | no | 1.85 |
| hampton-5.5-crit4 | 78.46 | 78.46 | 55.79 | 79.53 | 55.64 | 95.16 | no | 0.75 |
| hampton-5.5-crit5 | 84.47 | 84.47 | 69.83 | 86.32 | 64.16 | 104.20 | no | 2.43 |
| flanders-03-thematic-crit | 72.83 | 72.83 | 76.44 | 79.55 | 72.42 | 87.50 | no | 17.25 |
| anatomy-30b-hamstring-crit | 70.69 | 70.69 | 74.23 | 80.17 | 69.55 | 91.86 | no | 8.39 |
| colman-04.03-life-crit | 66.73 | 66.73 | 62.44 | 70.10 | 61.45 | 119.09 | no | 3.26 |
| colman-05.02-master-studies-crit | 57.45 | 57.45 | 62.32 | 76.84 | 59.74 | 119.19 | no | 5.96 |
| colman-06.06-species-crit | 64.99 | 64.99 | 65.88 | 80.85 | 60.73 | 123.31 | no | 4.74 |
| hampton-7-conclusion | 30.00 | 80.00 | 80.33 | 88.16 | 76.51 | 107.47 | yes | 1.94 |
| greco-2.2-thumbnailing | 66.28 | 66.28 | 68.48 | 70.46 | 67.57 | 187.32 | no | 9.21 |
| **pooled** | 72.02 | 72.30 | 71.16 | 78.95 | 68.65 | 110.27 | — | 5.34 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.30.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 30.30 | 80.30 | 81.25 | 82.84 | 77.98 | 106.24 | yes | 10.57 |
| hampton-5.5-crit1 | 32.43 | 82.43 | 69.50 | 89.57 | 62.51 | 99.10 | yes | 2.12 |
| hampton-5.5-crit2 | 28.90 | 78.90 | 57.34 | 81.36 | 57.04 | 98.85 | yes | 0.97 |
| hampton-5.5-crit3 | 25.77 | 75.77 | 70.07 | 89.95 | 65.00 | 98.29 | yes | 1.85 |
| hampton-5.5-crit4 | 28.72 | 78.72 | 55.86 | 79.23 | 55.43 | 95.30 | yes | 0.75 |
| hampton-5.5-crit5 | 33.25 | 83.25 | 68.03 | 84.85 | 63.07 | 104.38 | yes | 2.43 |
| flanders-03-thematic-crit | 23.11 | 73.11 | 77.29 | 80.34 | 73.14 | 87.68 | yes | 17.25 |
| anatomy-30b-hamstring-crit | 20.45 | 70.45 | 73.84 | 79.63 | 69.09 | 91.10 | yes | 8.39 |
| colman-04.03-life-crit | 15.66 | 65.66 | 61.69 | 69.23 | 60.68 | 118.25 | yes | 3.26 |
| colman-05.02-master-studies-crit | 7.38 | 57.38 | 64.65 | 79.41 | 61.73 | 117.39 | yes | 5.96 |
| colman-06.06-species-crit | 14.24 | 64.24 | 65.55 | 80.28 | 60.30 | 123.81 | yes | 4.74 |
| hampton-7-conclusion | 30.00 | 80.00 | 80.33 | 88.16 | 76.51 | 107.47 | yes | 1.94 |
| greco-2.2-thumbnailing | 17.13 | 67.13 | 68.69 | 70.70 | 67.80 | 186.99 | yes | 9.21 |
| **pooled** | 22.55 | 72.55 | 71.32 | 79.07 | 68.77 | 109.72 | — | 5.34 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.30.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 29.38 | 79.38 | 80.98 | 82.63 | 77.78 | 108.55 | yes | 10.57 |
| hampton-5.5-crit1 | 32.43 | 82.43 | 69.50 | 89.57 | 62.51 | 99.10 | yes | 2.12 |
| hampton-5.5-crit2 | 28.90 | 78.90 | 57.34 | 81.36 | 57.04 | 98.85 | yes | 0.97 |
| hampton-5.5-crit3 | 25.77 | 75.77 | 70.07 | 89.95 | 65.00 | 98.29 | yes | 1.85 |
| hampton-5.5-crit4 | 28.72 | 78.72 | 55.86 | 79.23 | 55.43 | 95.30 | yes | 0.75 |
| hampton-5.5-crit5 | 33.59 | 83.59 | 68.24 | 85.06 | 63.22 | 104.55 | yes | 2.43 |
| flanders-03-thematic-crit | 23.19 | 73.19 | 76.49 | 79.67 | 72.53 | 88.46 | yes | 17.25 |
| anatomy-30b-hamstring-crit | 20.77 | 70.77 | 74.27 | 80.14 | 69.52 | 92.27 | yes | 8.39 |
| colman-04.03-life-crit | 16.67 | 66.67 | 62.41 | 69.91 | 61.29 | 119.49 | yes | 3.26 |
| colman-05.02-master-studies-crit | 7.11 | 57.11 | 62.10 | 76.62 | 59.57 | 119.78 | yes | 5.96 |
| colman-06.06-species-crit | 13.97 | 63.97 | 65.39 | 80.23 | 60.27 | 124.26 | yes | 4.74 |
| hampton-7-conclusion | 30.00 | 80.00 | 80.33 | 88.16 | 76.51 | 107.47 | yes | 1.94 |
| greco-2.2-thumbnailing | 16.27 | 66.27 | 68.40 | 70.34 | 67.46 | 187.97 | yes | 9.21 |
| **pooled** | 22.28 | 72.28 | 71.04 | 78.83 | 68.56 | 110.78 | — | 5.34 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.30.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 81.43 | 81.43 | 81.41 | 85.27 | 80.26 | 99.80 | no | 10.57 |
| hampton-5.5-crit1 | 86.46 | 86.46 | 72.31 | 111.35 | 77.71 | 90.78 | no | 2.12 |
| hampton-5.5-crit2 | 78.90 | 78.90 | 58.85 | 101.93 | 71.47 | 90.68 | no | 0.97 |
| hampton-5.5-crit3 | 75.04 | 75.04 | 71.30 | 104.35 | 75.41 | 89.01 | no | 1.85 |
| hampton-5.5-crit4 | 82.76 | 82.76 | 58.79 | 98.48 | 68.90 | 87.48 | no | 0.75 |
| hampton-5.5-crit5 | 90.95 | 90.95 | 73.29 | 105.89 | 78.70 | 94.71 | no | 2.43 |
| flanders-03-thematic-crit | 75.38 | 75.38 | 77.05 | 82.38 | 75.00 | 75.27 | no | 17.25 |
| anatomy-30b-hamstring-crit | 79.86 | 79.86 | 76.88 | 87.19 | 75.64 | 72.84 | no | 8.39 |
| colman-04.03-life-crit | 74.83 | 74.83 | 64.08 | 76.27 | 66.86 | 108.28 | no | 3.26 |
| colman-05.02-master-studies-crit | 65.96 | 65.96 | 65.05 | 86.84 | 67.51 | 108.83 | no | 5.96 |
| colman-06.06-species-crit | 77.08 | 77.08 | 69.97 | 96.03 | 72.14 | 110.34 | no | 4.74 |
| hampton-7-conclusion | 81.63 | 81.63 | 80.70 | 92.63 | 80.39 | 100.40 | no | 1.94 |
| greco-2.2-thumbnailing | 68.40 | 68.40 | 68.64 | 71.92 | 68.97 | 171.71 | no | 9.21 |
| **pooled** | 76.75 | 76.75 | 72.58 | 85.81 | 74.10 | 98.37 | — | 5.34 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.30.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 82.13 | 82.13 | 81.69 | 85.48 | 80.47 | 97.94 | no | 10.57 |
| hampton-5.5-crit1 | 86.85 | 86.85 | 72.54 | 111.61 | 77.89 | 91.04 | no | 2.12 |
| hampton-5.5-crit2 | 79.13 | 79.13 | 58.91 | 101.99 | 71.51 | 90.74 | no | 0.97 |
| hampton-5.5-crit3 | 75.91 | 75.91 | 71.07 | 104.36 | 75.41 | 89.52 | no | 1.85 |
| hampton-5.5-crit4 | 83.01 | 83.01 | 58.86 | 98.50 | 68.91 | 87.56 | no | 0.75 |
| hampton-5.5-crit5 | 89.73 | 89.73 | 71.53 | 104.80 | 77.89 | 94.83 | no | 2.43 |
| flanders-03-thematic-crit | 75.65 | 75.65 | 77.91 | 83.23 | 75.78 | 75.22 | no | 17.25 |
| anatomy-30b-hamstring-crit | 79.46 | 79.46 | 76.49 | 86.74 | 75.25 | 72.19 | no | 8.39 |
| colman-04.03-life-crit | 74.00 | 74.00 | 63.37 | 75.46 | 66.15 | 107.42 | no | 3.26 |
| colman-05.02-master-studies-crit | 65.85 | 65.85 | 67.38 | 89.14 | 69.29 | 107.10 | no | 5.96 |
| colman-06.06-species-crit | 77.13 | 77.13 | 69.76 | 95.79 | 71.95 | 110.58 | no | 4.74 |
| hampton-7-conclusion | 81.63 | 81.63 | 80.70 | 92.63 | 80.39 | 100.40 | no | 1.94 |
| greco-2.2-thumbnailing | 69.37 | 69.37 | 68.85 | 72.16 | 69.20 | 171.34 | no | 9.21 |
| **pooled** | 77.01 | 77.01 | 72.76 | 85.99 | 74.26 | 97.81 | — | 5.34 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.30.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 81.38 | 81.38 | 81.42 | 85.30 | 80.29 | 100.01 | no | 10.57 |
| hampton-5.5-crit1 | 86.85 | 86.85 | 72.54 | 111.61 | 77.89 | 91.04 | no | 2.12 |
| hampton-5.5-crit2 | 79.13 | 79.13 | 58.91 | 101.99 | 71.51 | 90.74 | no | 0.97 |
| hampton-5.5-crit3 | 75.91 | 75.91 | 71.07 | 104.36 | 75.41 | 89.52 | no | 1.85 |
| hampton-5.5-crit4 | 83.01 | 83.01 | 58.86 | 98.50 | 68.91 | 87.56 | no | 0.75 |
| hampton-5.5-crit5 | 90.07 | 90.07 | 71.75 | 104.98 | 78.03 | 95.00 | no | 2.43 |
| flanders-03-thematic-crit | 75.85 | 75.85 | 77.12 | 82.59 | 75.19 | 75.96 | no | 17.25 |
| anatomy-30b-hamstring-crit | 79.99 | 79.99 | 76.94 | 87.22 | 75.67 | 73.13 | no | 8.39 |
| colman-04.03-life-crit | 75.33 | 75.33 | 64.11 | 76.20 | 66.80 | 108.58 | no | 3.26 |
| colman-05.02-master-studies-crit | 65.59 | 65.59 | 64.89 | 86.68 | 67.39 | 109.39 | no | 5.96 |
| colman-06.06-species-crit | 76.92 | 76.92 | 69.67 | 95.83 | 71.98 | 110.99 | no | 4.74 |
| hampton-7-conclusion | 81.63 | 81.63 | 80.70 | 92.63 | 80.39 | 100.40 | no | 1.94 |
| greco-2.2-thumbnailing | 68.50 | 68.50 | 68.56 | 71.82 | 68.88 | 172.27 | no | 9.21 |
| **pooled** | 76.86 | 76.86 | 72.49 | 85.78 | 74.08 | 98.77 | — | 5.34 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
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
| greco-2.2-thumbnailing | 0.44 | 8.78 | n/a | 9.21 | 79 | 2,297,907 | 0.0965 | 13 | 13 |
| **total** | 3.69 | 65.75 | 0.00 | 69.44 | 463 | 12,876,720 | 0.5408 | 97 | 100 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 44 | 8 | 107 | no |
| hampton-5.5-crit1 | 8 | 4 | 28 | no |
| hampton-5.5-crit2 | 4 | 1 | 18 | no |
| hampton-5.5-crit3 | 9 | 6 | 27 | no |
| hampton-5.5-crit4 | 8 | 3 | 27 | no |
| hampton-5.5-crit5 | 5 | 4 | 44 | no |
| flanders-03-thematic-crit | 48 | 27 | 145 | no |
| anatomy-30b-hamstring-crit | 28 | 13 | 150 | no |
| colman-04.03-life-crit | 19 | 9 | 110 | no |
| colman-05.02-master-studies-crit | 14 | 9 | 149 | no |
| colman-06.06-species-crit | 14 | 8 | 159 | no |
| hampton-7-conclusion | 2 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 67 | 15 | 97 | no |
| **total** | 270 | 107 | 1072 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 0 | 0 | 107 | yes |
| hampton-5.5-crit1 | 0 | 0 | 28 | yes |
| hampton-5.5-crit2 | 0 | 0 | 18 | yes |
| hampton-5.5-crit3 | 0 | 0 | 27 | yes |
| hampton-5.5-crit4 | 0 | 0 | 27 | yes |
| hampton-5.5-crit5 | 0 | 0 | 44 | yes |
| flanders-03-thematic-crit | 0 | 0 | 145 | yes |
| anatomy-30b-hamstring-crit | 0 | 0 | 150 | yes |
| colman-04.03-life-crit | 0 | 0 | 110 | yes |
| colman-05.02-master-studies-crit | 0 | 0 | 149 | yes |
| colman-06.06-species-crit | 0 | 0 | 159 | yes |
| hampton-7-conclusion | 0 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 0 | 0 | 97 | yes |
| **total** | 0 | 0 | 1072 | — |

### jev_b_notrim

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 0 | 0 | 107 | yes |
| hampton-5.5-crit1 | 0 | 0 | 28 | yes |
| hampton-5.5-crit2 | 0 | 0 | 18 | yes |
| hampton-5.5-crit3 | 0 | 0 | 27 | yes |
| hampton-5.5-crit4 | 0 | 0 | 27 | yes |
| hampton-5.5-crit5 | 0 | 0 | 44 | yes |
| flanders-03-thematic-crit | 0 | 0 | 145 | yes |
| anatomy-30b-hamstring-crit | 0 | 0 | 150 | yes |
| colman-04.03-life-crit | 0 | 0 | 110 | yes |
| colman-05.02-master-studies-crit | 0 | 0 | 149 | yes |
| colman-06.06-species-crit | 0 | 0 | 159 | yes |
| hampton-7-conclusion | 0 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 0 | 0 | 97 | yes |
| **total** | 0 | 0 | 1072 | — |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 101 | 46 | 0 | 72 | 157 | 5 (1 kept / 4 cut) | 90 (10 kept / 80 cut) |
| hampton-5.5-crit1 | 2 | 0 | 0 | 2 | 2 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit2 | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit3 | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit4 | 0 | 0 | 0 | 0 | 0 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit5 | 1 | 1 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (1 kept / 0 cut) |
| flanders-03-thematic-crit | 31 | 9 | 0 | 80 | 96 | 3 (3 kept / 0 cut) | 19 (10 kept / 9 cut) |
| anatomy-30b-hamstring-crit | 51 | 19 | 0 | 35 | 57 | 0 (0 kept / 0 cut) | 22 (9 kept / 13 cut) |
| colman-04.03-life-crit | 14 | 9 | 0 | 6 | 16 | 0 (0 kept / 0 cut) | 10 (8 kept / 2 cut) |
| colman-05.02-master-studies-crit | 4 | 2 | 0 | 2 | 6 | 0 (0 kept / 0 cut) | 4 (2 kept / 2 cut) |
| colman-06.06-species-crit | 6 | 2 | 0 | 4 | 6 | 0 (0 kept / 0 cut) | 2 (1 kept / 1 cut) |
| hampton-7-conclusion | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| greco-2.2-thumbnailing | 59 | 25 | 0 | 52 | 86 | 2 (1 kept / 1 cut) | 36 (2 kept / 34 cut) |

## Where this lands on the published ladder

Reference arms are the 18-episode pooled numbers from `2026-09-11-model-plus-deterministic.json`. NOT THE SAME EPISODE SET as the tables above (missing per-episode numbers for: greco-2.2-thumbnailing), so the comparison is indicative only.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 18 | 79.96 | 79.44 | 88.34 | 83.71 | 80.35 | 93.55 |
| shipped Opus agentic | 18 | 83.45 | 80.84 | 90.37 | 86.47 | 81.44 | 95.02 |
| Luna single call | 18 | 18.61 | 61.23 | 69.43 | 65.36 | 64.63 | 80.67 |
| deterministic baseline (um removal + retakes + delete silence) | 18 | 63.72 | 64.07 | 79.92 | 63.72 | 64.07 | 79.92 |

Best Jev arm here is jev_a (t_trim 0.3) at 72.02 SENTENCE POINTS (71.16 WORD SCORE, 5.34 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 76.75 SENTENCE POINTS (72.58 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v1-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v1-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v1-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v1-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v1-summary.json`
