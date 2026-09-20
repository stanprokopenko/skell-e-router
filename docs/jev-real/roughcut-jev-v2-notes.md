# Jev rough cut: run report

Generated 2026-09-20T22:15:47+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-v2-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v2, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 6 episode(s): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique.
167 requests, 23 errored, 22 retried, 5,244,029 input tokens, $0.2202, 26.3 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 2743.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 2743 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-v2-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.40 | 68.94 | 76.03 | 74.81 | 81.00 | 71.98 | 106.96 | 4.38 |
| jev_b_moduleretakes | 2.40 | 33.69 | 76.20 | 74.61 | 80.67 | 71.70 | 105.34 | 4.38 |
| jev_b_notrim | 2.40 | 33.47 | 75.98 | 74.73 | 80.85 | 71.87 | 107.26 | 4.38 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.40 | 82.52 | 82.52 | 76.31 | 86.40 | 76.35 | 96.11 | 4.38 |
| jev_b_moduleretakes | 2.40 | 82.68 | 82.68 | 76.12 | 86.14 | 76.13 | 94.51 | 4.38 |
| jev_b_notrim | 2.40 | 82.46 | 82.46 | 76.24 | 86.33 | 76.29 | 96.35 | 4.38 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 62.58 | 62.58 | 65.84 | 85.49 | 61.03 | 121.11 | no | 1.01 |
| hampton-5.4-assignment-demo | 72.13 | 72.13 | 74.06 | 82.93 | 67.41 | 120.38 | no | 4.58 |
| colman-03.03-muscles-crit | 72.31 | 72.31 | 57.24 | 68.21 | 57.38 | 93.04 | no | 2.84 |
| edges-7.01-intro | 26.35 | 76.35 | 78.54 | 79.59 | 77.90 | 85.24 | yes | 4.87 |
| hampton-5.2-shape-demo | 77.59 | 77.59 | 82.84 | 87.31 | 75.73 | 117.89 | no | 3.19 |
| perspective-14e-boxes-critique | 79.65 | 79.65 | 79.28 | 81.27 | 76.65 | 105.90 | no | 9.81 |
| **pooled** | 68.94 | 76.03 | 74.81 | 81.00 | 71.98 | 106.96 | — | 4.38 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.03 | 61.03 | 65.12 | 84.20 | 60.11 | 121.48 | yes | 1.01 |
| hampton-5.4-assignment-demo | 21.93 | 71.93 | 74.39 | 82.86 | 67.36 | 120.18 | yes | 4.58 |
| colman-03.03-muscles-crit | 21.95 | 71.95 | 57.11 | 68.05 | 57.25 | 93.05 | yes | 2.84 |
| edges-7.01-intro | 28.66 | 78.66 | 80.08 | 80.70 | 78.98 | 74.93 | yes | 4.87 |
| hampton-5.2-shape-demo | 77.81 | 77.81 | 82.90 | 87.25 | 75.67 | 118.01 | no | 3.19 |
| perspective-14e-boxes-critique | 29.60 | 79.60 | 78.31 | 80.49 | 75.92 | 103.79 | yes | 9.81 |
| **pooled** | 33.69 | 76.20 | 74.61 | 80.67 | 71.70 | 105.34 | — | 4.38 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.03 | 61.03 | 65.01 | 84.36 | 60.23 | 121.88 | yes | 1.01 |
| hampton-5.4-assignment-demo | 21.60 | 71.60 | 73.92 | 82.63 | 67.17 | 120.61 | yes | 4.58 |
| colman-03.03-muscles-crit | 22.94 | 72.94 | 57.30 | 68.20 | 57.38 | 93.43 | yes | 2.84 |
| edges-7.01-intro | 26.35 | 76.35 | 78.54 | 79.59 | 77.90 | 85.24 | yes | 4.87 |
| hampton-5.2-shape-demo | 77.81 | 77.81 | 82.85 | 87.29 | 75.71 | 118.16 | no | 3.19 |
| perspective-14e-boxes-critique | 29.69 | 79.69 | 79.27 | 81.25 | 76.63 | 106.11 | yes | 9.81 |
| **pooled** | 33.47 | 75.98 | 74.73 | 80.85 | 71.87 | 107.26 | — | 4.38 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 85.36 | 85.36 | 73.88 | 105.60 | 75.38 | 105.80 | no | 1.01 |
| hampton-5.4-assignment-demo | 85.63 | 85.63 | 76.15 | 94.94 | 77.17 | 100.77 | no | 4.58 |
| colman-03.03-muscles-crit | 74.95 | 74.95 | 58.74 | 72.77 | 61.22 | 85.43 | no | 2.84 |
| edges-7.01-intro | 78.56 | 78.56 | 78.98 | 79.35 | 77.67 | 78.61 | no | 4.87 |
| hampton-5.2-shape-demo | 91.31 | 91.31 | 83.75 | 95.16 | 82.54 | 102.15 | no | 3.19 |
| perspective-14e-boxes-critique | 81.40 | 81.40 | 79.60 | 82.74 | 78.04 | 99.09 | no | 9.81 |
| **pooled** | 82.52 | 82.52 | 76.31 | 86.40 | 76.35 | 96.11 | — | 4.38 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 83.81 | 83.81 | 73.19 | 104.70 | 74.75 | 106.01 | no | 1.01 |
| hampton-5.4-assignment-demo | 85.43 | 85.43 | 76.47 | 95.01 | 77.23 | 100.47 | no | 4.58 |
| colman-03.03-muscles-crit | 74.59 | 74.59 | 58.61 | 72.68 | 61.15 | 85.38 | no | 2.84 |
| edges-7.01-intro | 80.87 | 80.87 | 80.54 | 80.45 | 78.74 | 68.82 | no | 4.87 |
| hampton-5.2-shape-demo | 91.53 | 91.53 | 83.81 | 95.11 | 82.49 | 102.27 | no | 3.19 |
| perspective-14e-boxes-critique | 81.34 | 81.34 | 78.64 | 81.99 | 77.33 | 97.01 | no | 9.81 |
| **pooled** | 82.68 | 82.68 | 76.12 | 86.14 | 76.13 | 94.51 | — | 4.38 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 83.81 | 83.81 | 73.13 | 104.85 | 74.85 | 106.41 | no | 1.01 |
| hampton-5.4-assignment-demo | 85.10 | 85.10 | 76.01 | 94.79 | 77.05 | 100.90 | no | 4.58 |
| colman-03.03-muscles-crit | 75.58 | 75.58 | 58.80 | 72.83 | 61.28 | 85.77 | no | 2.84 |
| edges-7.01-intro | 78.56 | 78.56 | 78.98 | 79.35 | 77.67 | 78.61 | no | 4.87 |
| hampton-5.2-shape-demo | 91.53 | 91.53 | 83.76 | 95.14 | 82.52 | 102.41 | no | 3.19 |
| perspective-14e-boxes-critique | 81.43 | 81.43 | 79.60 | 82.75 | 78.04 | 99.25 | no | 9.81 |
| **pooled** | 82.46 | 82.46 | 76.24 | 86.33 | 76.29 | 96.35 | — | 4.38 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0.30 | 0.71 | n/a | 1.01 | 9 | 332,290 | 0.0140 | 0 | 0 |
| hampton-5.4-assignment-demo | 0.17 | 4.41 | n/a | 4.58 | 15 | 473,242 | 0.0199 | 2 | 2 |
| colman-03.03-muscles-crit | 0.19 | 2.65 | n/a | 2.84 | 17 | 565,503 | 0.0238 | 2 | 2 |
| edges-7.01-intro | 0.46 | 4.40 | n/a | 4.87 | 33 | 611,022 | 0.0257 | 6 | 6 |
| hampton-5.2-shape-demo | 0.25 | 2.95 | n/a | 3.19 | 23 | 757,213 | 0.0318 | 3 | 3 |
| perspective-14e-boxes-critique | 0.49 | 9.32 | n/a | 9.81 | 70 | 2,504,759 | 0.1052 | 9 | 10 |
| **total** | 1.86 | 24.44 | 0.00 | 26.30 | 167 | 5,244,029 | 0.2202 | 22 | 23 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 10 | 6 | 91 | no |
| hampton-5.4-assignment-demo | 3 | 2 | 77 | no |
| colman-03.03-muscles-crit | 13 | 8 | 70 | no |
| edges-7.01-intro | 7 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 18 | 7 | 77 | no |
| perspective-14e-boxes-critique | 26 | 5 | 58 | no |
| **total** | 77 | 28 | 388 | — |

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

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 6 | 2 | 0 | 4 | 6 | 0 (0 kept / 0 cut) | 2 (1 kept / 1 cut) |
| hampton-5.4-assignment-demo | 6 | 5 | 0 | 1 | 7 | 0 (0 kept / 0 cut) | 6 (1 kept / 5 cut) |
| colman-03.03-muscles-crit | 8 | 6 | 0 | 2 | 8 | 1 (0 kept / 1 cut) | 7 (5 kept / 2 cut) |
| edges-7.01-intro | 63 | 21 | 0 | 107 | 169 | 6 (3 kept / 3 cut) | 68 (21 kept / 47 cut) |
| hampton-5.2-shape-demo | 14 | 3 | 0 | 13 | 16 | 0 (0 kept / 0 cut) | 3 (1 kept / 2 cut) |
| perspective-14e-boxes-critique | 80 | 28 | 0 | 56 | 98 | 5 (0 kept / 5 cut) | 47 (9 kept / 38 cut) |

## Where this lands on the published ladder

Reference arms re-pooled over the same 6 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 6 | 81.78 | 80.81 | 87.08 | 86.86 | 81.75 | 91.29 |
| shipped Opus agentic | 6 | 84.54 | 82.49 | 88.89 | 89.49 | 83.44 | 93.06 |
| Luna single call | 6 | 17.34 | 60.02 | 66.48 | 62.08 | 62.74 | 75.22 |
| deterministic baseline (um removal + retakes + delete silence) | 6 | 60.27 | 62.20 | 74.52 | 60.27 | 62.20 | 74.52 |

Best Jev arm here is jev_a (t_trim 0.3) at 68.94 SENTENCE POINTS (74.81 WORD SCORE, 4.38 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 82.52 SENTENCE POINTS (76.31 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-summary.json`
