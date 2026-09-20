# Jev rough cut: run report

Generated 2026-09-20T22:05:13+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v1, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 6 episode(s): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique.
323 requests, 26 errored, 22 retried, 5,534,997 input tokens, $0.2325, 22.712 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 2743.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 2743 sentences with 0 mismatches.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.40 | 70.58 | 77.67 | 75.10 | 81.62 | 72.45 | 99.35 | 3.79 |
| jev_b | 2.40 | 40.58 | 77.56 | 75.00 | 81.51 | 72.37 | 99.75 | 3.79 |
| jev_b_moduleretakes | 2.40 | 40.62 | 77.60 | 74.88 | 81.41 | 72.27 | 98.23 | 3.79 |
| jev_b_notrim | 2.40 | 35.08 | 77.59 | 75.01 | 81.52 | 72.38 | 99.77 | 3.79 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.40 | 83.84 | 83.84 | 76.46 | 86.41 | 76.31 | 89.63 | 3.79 |
| jev_b | 2.40 | 83.76 | 83.76 | 76.36 | 86.33 | 76.24 | 89.98 | 3.79 |
| jev_b_moduleretakes | 2.40 | 83.79 | 83.79 | 76.25 | 86.21 | 76.13 | 88.51 | 3.79 |
| jev_b_notrim | 2.40 | 83.77 | 83.77 | 76.37 | 86.34 | 76.25 | 89.99 | 3.79 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 63.66 | 63.66 | 66.76 | 88.19 | 62.95 | 113.03 | no | 2.37 |
| hampton-5.4-assignment-demo | 73.27 | 73.27 | 72.97 | 82.55 | 67.10 | 114.13 | no | 2.42 |
| colman-03.03-muscles-crit | 71.35 | 71.35 | 60.75 | 71.27 | 59.96 | 89.96 | no | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 78.18 | 78.18 | 83.40 | 88.03 | 76.35 | 114.79 | no | 3.35 |
| perspective-14e-boxes-critique | 83.05 | 83.05 | 80.03 | 82.32 | 77.64 | 96.84 | no | 8.82 |
| **pooled** | 70.58 | 77.67 | 75.10 | 81.62 | 72.45 | 99.35 | — | 3.79 |

### jev_b (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.10 | 87.25 | 62.28 | 113.94 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.27 | 73.27 | 72.82 | 82.44 | 67.01 | 114.22 | yes | 2.42 |
| colman-03.03-muscles-crit | 71.75 | 71.75 | 60.52 | 71.10 | 59.81 | 90.83 | no | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 78.54 | 78.54 | 83.56 | 88.18 | 76.48 | 115.05 | no | 3.35 |
| perspective-14e-boxes-critique | 32.91 | 82.91 | 79.95 | 82.27 | 77.60 | 97.11 | yes | 8.82 |
| **pooled** | 40.58 | 77.56 | 75.00 | 81.51 | 72.37 | 99.75 | — | 3.79 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.21 | 87.11 | 62.19 | 113.54 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.60 | 73.60 | 73.30 | 82.68 | 67.21 | 113.79 | yes | 2.42 |
| colman-03.03-muscles-crit | 71.09 | 71.09 | 60.64 | 71.67 | 60.29 | 90.20 | no | 2.90 |
| edges-7.01-intro | 27.63 | 77.63 | 74.63 | 76.24 | 74.62 | 58.68 | yes | 2.85 |
| hampton-5.2-shape-demo | 78.54 | 78.54 | 83.61 | 88.14 | 76.44 | 114.91 | no | 3.35 |
| perspective-14e-boxes-critique | 32.74 | 82.74 | 79.15 | 81.67 | 77.03 | 94.86 | yes | 8.82 |
| **pooled** | 40.62 | 77.60 | 74.88 | 81.41 | 72.27 | 98.23 | — | 3.79 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.10 | 87.25 | 62.28 | 113.94 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.27 | 73.27 | 72.82 | 82.44 | 67.01 | 114.22 | yes | 2.42 |
| colman-03.03-muscles-crit | 21.95 | 71.95 | 60.59 | 71.20 | 59.91 | 90.95 | yes | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 78.54 | 78.54 | 83.56 | 88.18 | 76.48 | 115.05 | no | 3.35 |
| perspective-14e-boxes-critique | 32.91 | 82.91 | 79.95 | 82.27 | 77.60 | 97.11 | yes | 8.82 |
| **pooled** | 35.08 | 77.59 | 75.01 | 81.52 | 72.38 | 99.77 | — | 3.79 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 83.66 | 83.66 | 73.69 | 105.79 | 75.52 | 99.18 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.99 | 94.39 | 76.73 | 95.26 | no | 2.42 |
| colman-03.03-muscles-crit | 75.41 | 75.41 | 62.22 | 74.96 | 63.07 | 82.94 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 92.04 | 92.04 | 84.30 | 95.69 | 82.99 | 99.64 | no | 3.35 |
| perspective-14e-boxes-critique | 84.62 | 84.62 | 80.29 | 83.29 | 78.56 | 91.40 | no | 8.82 |
| **pooled** | 83.84 | 83.84 | 76.46 | 86.41 | 76.31 | 89.63 | — | 3.79 |

### jev_b (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 81.96 | 81.96 | 73.07 | 105.15 | 75.06 | 99.93 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.85 | 94.29 | 76.64 | 95.35 | no | 2.42 |
| colman-03.03-muscles-crit | 75.81 | 75.81 | 61.99 | 74.76 | 62.89 | 83.69 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 92.41 | 92.41 | 84.45 | 95.83 | 83.11 | 99.90 | no | 3.35 |
| perspective-14e-boxes-critique | 84.49 | 84.49 | 80.22 | 83.26 | 78.53 | 91.61 | no | 8.82 |
| **pooled** | 83.76 | 83.76 | 76.36 | 86.33 | 76.24 | 89.98 | — | 3.79 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 81.96 | 81.96 | 73.16 | 105.02 | 74.97 | 99.53 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.67 | 86.67 | 75.32 | 94.52 | 76.83 | 94.92 | no | 2.42 |
| colman-03.03-muscles-crit | 75.15 | 75.15 | 62.11 | 75.21 | 63.28 | 83.08 | no | 2.90 |
| edges-7.01-intro | 78.59 | 78.59 | 75.03 | 75.75 | 74.14 | 54.79 | no | 2.85 |
| hampton-5.2-shape-demo | 92.41 | 92.41 | 84.51 | 95.80 | 83.08 | 99.76 | no | 3.35 |
| perspective-14e-boxes-critique | 84.31 | 84.31 | 79.42 | 82.65 | 77.95 | 89.42 | no | 8.82 |
| **pooled** | 83.79 | 83.79 | 76.25 | 86.21 | 76.13 | 88.51 | — | 3.79 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 81.96 | 81.96 | 73.07 | 105.15 | 75.06 | 99.93 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.85 | 94.29 | 76.64 | 95.35 | no | 2.42 |
| colman-03.03-muscles-crit | 75.91 | 75.91 | 62.05 | 74.82 | 62.95 | 83.77 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 92.41 | 92.41 | 84.45 | 95.83 | 83.11 | 99.90 | no | 3.35 |
| perspective-14e-boxes-critique | 84.49 | 84.49 | 80.22 | 83.26 | 78.53 | 91.61 | no | 8.82 |
| **pooled** | 83.77 | 83.77 | 76.37 | 86.34 | 76.25 | 89.99 | — | 3.79 |

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
| **total** | 2.32 | 14.37 | 6.02 | 22.71 | 323 | 5,534,997 | 0.2325 | 22 | 26 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 14 | 7 | 91 | no |
| hampton-5.4-assignment-demo | 5 | 1 | 77 | no |
| colman-03.03-muscles-crit | 20 | 15 | 70 | no |
| edges-7.01-intro | 7 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 20 | 6 | 77 | no |
| perspective-14e-boxes-critique | 24 | 4 | 58 | no |
| **total** | 90 | 33 | 388 | — |

### jev_b

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 1 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 2 | 2 | 70 | no |
| edges-7.01-intro | 4 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 1 | 0 | 58 | yes |
| **total** | 8 | 3 | 388 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 1 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 2 | 2 | 70 | no |
| edges-7.01-intro | 4 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 1 | 0 | 58 | yes |
| **total** | 8 | 3 | 388 | — |

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

## Where this lands on the published ladder

Reference arms re-pooled over the same 6 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 6 | 81.78 | 80.81 | 87.08 | 86.86 | 81.75 | 91.29 |
| shipped Opus agentic | 6 | 84.54 | 82.49 | 88.89 | 89.49 | 83.44 | 93.06 |
| Luna single call | 6 | 17.34 | 60.02 | 66.48 | 62.08 | 62.74 | 75.22 |
| deterministic baseline (um removal + retakes + delete silence) | 6 | 60.27 | 62.20 | 74.52 | 60.27 | 62.20 | 74.52 |

Best Jev arm here is jev_a (t_trim 0.3) at 70.58 SENTENCE POINTS (75.10 WORD SCORE, 3.79 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 83.84 SENTENCE POINTS (76.46 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v1-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v1-summary.json`
