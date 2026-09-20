# Jev rough cut: run report

Generated 2026-09-20T22:27:06+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-v3-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v3, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 6 episode(s): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique.
189 requests, 45 errored, 42 retried, 5,538,274 input tokens, $0.2326, 30.876 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 2743.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 2743 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-v3-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 69.40 | 76.49 | 75.27 | 81.36 | 72.27 | 104.99 | 5.15 |
| jev_b_moduleretakes | 2.50 | 34.09 | 76.60 | 74.84 | 80.86 | 71.83 | 103.70 | 5.15 |
| jev_b_notrim | 2.50 | 34.02 | 76.53 | 75.09 | 81.14 | 72.09 | 105.39 | 5.15 |
| jev_noul (t_trim 0.3) | 3.20 | 68.22 | 75.32 | 74.39 | 80.74 | 71.60 | 96.06 | 5.15 |
| jev_mix (t_trim 0.3) | 2.80 | 69.93 | 77.03 | 75.11 | 81.52 | 72.41 | 103.18 | 5.15 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 83.00 | 83.00 | 76.78 | 86.72 | 76.60 | 94.39 | 5.15 |
| jev_b_moduleretakes | 2.50 | 83.05 | 83.05 | 76.36 | 86.29 | 76.22 | 93.10 | 5.15 |
| jev_b_notrim | 2.50 | 82.97 | 82.97 | 76.61 | 86.56 | 76.47 | 94.72 | 5.15 |
| jev_noul (t_trim 0.3) | 3.10 | 81.08 | 81.08 | 75.31 | 85.41 | 75.36 | 90.33 | 5.15 |
| jev_mix (t_trim 0.3) | 2.70 | 82.65 | 82.65 | 76.67 | 86.93 | 76.82 | 96.47 | 5.15 |

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
| **pooled** | 69.40 | 76.49 | 75.27 | 81.36 | 72.27 | 104.99 | — | 5.15 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 65.82 | 85.17 | 60.80 | 121.23 | yes | 4.45 |
| hampton-5.4-assignment-demo | 21.60 | 71.60 | 73.74 | 82.63 | 67.17 | 118.33 | yes | 2.79 |
| colman-03.03-muscles-crit | 21.62 | 71.62 | 57.98 | 68.99 | 58.04 | 93.05 | yes | 3.10 |
| edges-7.01-intro | 28.92 | 78.92 | 79.73 | 80.16 | 78.45 | 73.97 | yes | 4.90 |
| hampton-5.2-shape-demo | 79.51 | 79.51 | 84.72 | 89.13 | 77.30 | 116.21 | no | 3.20 |
| perspective-14e-boxes-critique | 29.95 | 79.95 | 77.85 | 79.96 | 75.42 | 100.74 | yes | 12.44 |
| **pooled** | 34.09 | 76.60 | 74.84 | 80.86 | 71.83 | 103.70 | — | 5.15 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 65.72 | 85.33 | 60.92 | 121.63 | yes | 4.45 |
| hampton-5.4-assignment-demo | 21.27 | 71.27 | 73.26 | 82.40 | 66.98 | 118.76 | yes | 2.79 |
| colman-03.03-muscles-crit | 22.61 | 72.61 | 58.18 | 69.14 | 58.17 | 93.43 | yes | 3.10 |
| edges-7.01-intro | 27.89 | 77.89 | 79.33 | 79.86 | 78.16 | 82.06 | yes | 4.90 |
| hampton-5.2-shape-demo | 79.27 | 79.27 | 84.65 | 89.10 | 77.28 | 116.27 | no | 3.20 |
| perspective-14e-boxes-critique | 30.03 | 80.03 | 78.79 | 80.72 | 76.14 | 103.04 | yes | 12.44 |
| **pooled** | 34.02 | 76.53 | 75.09 | 81.14 | 72.09 | 105.39 | — | 5.15 |

### jev_noul (t_trim 0.3) (plain)

Calibrated pooled keep threshold 3.20.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 60.62 | 60.62 | 62.37 | 85.53 | 61.06 | 100.17 | no | 4.45 |
| hampton-5.4-assignment-demo | 72.50 | 72.50 | 74.64 | 84.66 | 68.81 | 111.62 | no | 2.79 |
| colman-03.03-muscles-crit | 70.00 | 70.00 | 60.77 | 72.22 | 60.76 | 88.92 | no | 3.10 |
| edges-7.01-intro | 26.86 | 76.86 | 72.94 | 74.34 | 72.76 | 59.15 | yes | 4.90 |
| hampton-5.2-shape-demo | 79.22 | 79.22 | 86.30 | 90.22 | 78.25 | 112.19 | no | 3.20 |
| perspective-14e-boxes-critique | 78.02 | 78.02 | 77.62 | 79.47 | 74.96 | 95.41 | no | 12.44 |
| **pooled** | 68.22 | 75.32 | 74.39 | 80.74 | 71.60 | 96.06 | — | 5.15 |

### jev_mix (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.80.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 62.78 | 62.78 | 65.62 | 85.99 | 61.39 | 115.50 | no | 4.45 |
| hampton-5.4-assignment-demo | 70.83 | 70.83 | 72.70 | 82.03 | 66.68 | 116.75 | no | 2.79 |
| colman-03.03-muscles-crit | 72.64 | 72.64 | 57.96 | 69.90 | 58.81 | 92.20 | no | 3.10 |
| edges-7.01-intro | 26.61 | 76.61 | 73.84 | 75.46 | 73.86 | 63.59 | yes | 4.90 |
| hampton-5.2-shape-demo | 79.39 | 79.39 | 85.32 | 89.67 | 77.77 | 115.57 | no | 3.20 |
| perspective-14e-boxes-critique | 81.51 | 81.51 | 80.62 | 82.52 | 77.84 | 105.34 | no | 12.44 |
| **pooled** | 69.93 | 77.03 | 75.11 | 81.52 | 72.41 | 103.18 | — | 5.15 |

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
| **pooled** | 83.00 | 83.00 | 76.78 | 86.72 | 76.60 | 94.39 | — | 5.15 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 84.33 | 84.33 | 74.01 | 105.56 | 75.36 | 105.91 | no | 4.45 |
| hampton-5.4-assignment-demo | 85.23 | 85.23 | 75.96 | 94.54 | 76.85 | 99.26 | no | 2.79 |
| colman-03.03-muscles-crit | 74.26 | 74.26 | 59.39 | 73.47 | 61.81 | 85.50 | no | 3.10 |
| edges-7.01-intro | 80.72 | 80.72 | 80.16 | 79.94 | 78.24 | 68.13 | no | 4.90 |
| hampton-5.2-shape-demo | 93.31 | 93.31 | 85.62 | 96.83 | 83.98 | 100.64 | no | 3.20 |
| perspective-14e-boxes-critique | 81.69 | 81.69 | 78.19 | 81.52 | 76.89 | 94.15 | no | 12.44 |
| **pooled** | 83.05 | 83.05 | 76.36 | 86.29 | 76.22 | 93.10 | — | 5.15 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 84.33 | 84.33 | 73.95 | 105.71 | 75.46 | 106.31 | no | 4.45 |
| hampton-5.4-assignment-demo | 84.90 | 84.90 | 75.49 | 94.31 | 76.66 | 99.69 | no | 2.79 |
| colman-03.03-muscles-crit | 75.25 | 75.25 | 59.59 | 73.62 | 61.94 | 85.88 | no | 3.10 |
| edges-7.01-intro | 79.69 | 79.69 | 79.74 | 79.65 | 77.95 | 75.82 | no | 4.90 |
| hampton-5.2-shape-demo | 93.07 | 93.07 | 85.55 | 96.80 | 83.96 | 100.69 | no | 3.20 |
| perspective-14e-boxes-critique | 81.78 | 81.78 | 79.12 | 82.28 | 77.60 | 96.38 | no | 12.44 |
| **pooled** | 82.97 | 82.97 | 76.61 | 86.56 | 76.47 | 94.72 | — | 5.15 |

### jev_noul (t_trim 0.3) (layered)

Calibrated pooled keep threshold 3.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 80.82 | 80.82 | 70.96 | 103.61 | 73.97 | 94.24 | no | 4.45 |
| hampton-5.4-assignment-demo | 83.90 | 83.90 | 75.00 | 94.77 | 77.03 | 96.68 | no | 2.79 |
| colman-03.03-muscles-crit | 76.11 | 76.11 | 61.72 | 76.58 | 64.43 | 83.67 | no | 3.10 |
| edges-7.01-intro | 76.68 | 76.68 | 73.02 | 73.65 | 72.09 | 57.21 | no | 4.90 |
| hampton-5.2-shape-demo | 92.55 | 92.55 | 86.57 | 97.17 | 84.27 | 99.44 | no | 3.20 |
| perspective-14e-boxes-critique | 79.08 | 79.08 | 77.33 | 80.62 | 76.04 | 95.24 | no | 12.44 |
| **pooled** | 81.08 | 81.08 | 75.31 | 85.41 | 75.36 | 90.33 | — | 5.15 |

### jev_mix (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 85.88 | 85.88 | 73.89 | 105.76 | 75.50 | 103.27 | no | 4.45 |
| hampton-5.4-assignment-demo | 81.87 | 81.87 | 71.50 | 91.77 | 74.59 | 100.96 | no | 2.79 |
| colman-03.03-muscles-crit | 76.47 | 76.47 | 60.92 | 75.94 | 63.89 | 86.89 | no | 3.10 |
| edges-7.01-intro | 77.97 | 77.97 | 76.86 | 77.09 | 75.45 | 69.87 | no | 4.90 |
| hampton-5.2-shape-demo | 92.36 | 92.36 | 85.20 | 96.26 | 83.49 | 102.12 | no | 3.20 |
| perspective-14e-boxes-critique | 82.04 | 82.04 | 81.03 | 84.12 | 79.34 | 102.63 | no | 12.44 |
| **pooled** | 82.65 | 82.65 | 76.67 | 86.93 | 76.82 | 96.47 | — | 5.15 |

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
| **total** | 1.96 | 28.92 | 0.00 | 30.88 | 189 | 5,538,274 | 0.2326 | 42 | 45 |

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
| **total** | 89 | 35 | 388 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 0 | 0 | 70 | yes |
| edges-7.01-intro | 0 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 0 | 0 | 58 | yes |
| **total** | 0 | 1 | 388 | — |

### jev_b_notrim

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 0 | 0 | 70 | yes |
| edges-7.01-intro | 0 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 0 | 0 | 58 | yes |
| **total** | 0 | 1 | 388 | — |

### jev_noul (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 9 | 4 | 91 | no |
| hampton-5.4-assignment-demo | 5 | 1 | 77 | no |
| colman-03.03-muscles-crit | 16 | 12 | 70 | no |
| edges-7.01-intro | 8 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 22 | 6 | 77 | no |
| perspective-14e-boxes-critique | 29 | 8 | 58 | no |
| **total** | 89 | 31 | 388 | — |

### jev_mix (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 9 | 5 | 91 | no |
| hampton-5.4-assignment-demo | 5 | 1 | 77 | no |
| colman-03.03-muscles-crit | 16 | 12 | 70 | no |
| edges-7.01-intro | 8 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 22 | 8 | 77 | no |
| perspective-14e-boxes-critique | 29 | 6 | 58 | no |
| **total** | 89 | 32 | 388 | — |

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
| **pooled** | 2743 | — | -0.824 | 0.381 | 2.54 |

Keep/cut agreement with the editor at each arm's own calibrated threshold, layered scoring over 6 episode(s). Counts are sentences: `wrong drops` is the editor kept it and the arm removed it.

| arm | threshold | sentences | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio | SENTENCE POINTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 2743 | 1290 | 1010 | 215 | 228 | 83.85 | 94.39 | 83.00 |
| jev_noul (t_trim 0.3) | 3.10 | 2743 | 1234 | 1006 | 271 | 232 | 81.66 | 90.33 | 81.08 |
| jev_mix (t_trim 0.3) | 2.70 | 2743 | 1305 | 988 | 200 | 250 | 83.59 | 96.47 | 82.65 |

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

## Where this lands on the published ladder

Reference arms re-pooled over the same 6 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 6 | 81.78 | 80.81 | 87.08 | 86.86 | 81.75 | 91.29 |
| shipped Opus agentic | 6 | 84.54 | 82.49 | 88.89 | 89.49 | 83.44 | 93.06 |
| Luna single call | 6 | 17.34 | 60.02 | 66.48 | 62.08 | 62.74 | 75.22 |
| deterministic baseline (um removal + retakes + delete silence) | 6 | 60.27 | 62.20 | 74.52 | 60.27 | 62.20 | 74.52 |

Best Jev arm here is jev_mix (t_trim 0.3) at 69.93 SENTENCE POINTS (75.11 WORD SCORE, 5.15 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 82.65 SENTENCE POINTS (76.67 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v3-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v3-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v3-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v3-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v3-summary.json`
