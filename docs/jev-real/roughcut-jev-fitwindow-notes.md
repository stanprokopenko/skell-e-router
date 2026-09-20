# Jev rough cut: run report

Generated 2026-09-20T23:03:31+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-fitwindow-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v3, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 2 episode(s): perspective-13d-critique, greco-2.2-thumbnailing.
207 requests, 53 errored, 48 retried, 7,595,626 input tokens, $0.3190, 35.906 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 3140.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 3140 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-fitwindow-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.70 | 74.54 | 74.54 | 74.35 | 76.70 | 72.78 | 127.01 | 17.95 |
| jev_b_moduleretakes | 2.70 | 25.27 | 75.27 | 74.51 | 76.87 | 72.94 | 125.47 | 17.95 |
| jev_b_notrim | 2.70 | 24.57 | 74.57 | 74.32 | 76.65 | 72.73 | 127.38 | 17.95 |
| jev_noul (t_trim 0.3) | 3.20 | 74.09 | 74.09 | 72.95 | 75.51 | 71.65 | 126.24 | 17.95 |
| jev_mix (t_trim 0.3) | 3.10 | 77.53 | 77.53 | 73.87 | 76.60 | 72.69 | 108.27 | 17.95 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.70 | 76.26 | 76.26 | 74.57 | 78.53 | 74.52 | 116.87 | 17.95 |
| jev_b_moduleretakes | 2.60 | 75.84 | 75.84 | 74.61 | 78.74 | 74.71 | 122.03 | 17.95 |
| jev_b_notrim | 2.70 | 76.35 | 76.35 | 74.55 | 78.51 | 74.49 | 117.16 | 17.95 |
| jev_noul (t_trim 0.3) | 3.20 | 76.04 | 76.04 | 73.21 | 77.41 | 73.45 | 116.18 | 17.95 |
| jev_mix (t_trim 0.3) | 2.80 | 74.80 | 74.80 | 74.77 | 78.81 | 74.77 | 128.57 | 17.95 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 78.64 | 78.64 | 79.05 | 80.60 | 75.87 | 97.99 | no | 18.74 |
| greco-2.2-thumbnailing | 69.36 | 69.36 | 70.17 | 72.02 | 69.07 | 177.22 | no | 17.16 |
| **pooled** | 74.54 | 74.54 | 74.35 | 76.70 | 72.78 | 127.01 | — | 17.95 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 29.44 | 79.44 | 79.21 | 80.78 | 76.04 | 95.58 | yes | 18.74 |
| greco-2.2-thumbnailing | 20.01 | 70.01 | 70.33 | 72.17 | 69.22 | 177.20 | yes | 17.16 |
| **pooled** | 25.27 | 75.27 | 74.51 | 76.87 | 72.94 | 125.47 | — | 17.95 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 28.70 | 78.70 | 79.03 | 80.57 | 75.85 | 98.24 | yes | 18.74 |
| greco-2.2-thumbnailing | 19.37 | 69.37 | 70.13 | 71.93 | 68.99 | 177.80 | yes | 17.16 |
| **pooled** | 24.57 | 74.57 | 74.32 | 76.65 | 72.73 | 127.38 | — | 17.95 |

### jev_noul (t_trim 0.3) (plain)

Calibrated pooled keep threshold 3.20.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 77.04 | 77.04 | 76.76 | 79.07 | 74.43 | 100.11 | no | 18.74 |
| greco-2.2-thumbnailing | 70.37 | 70.37 | 69.56 | 71.22 | 68.31 | 171.46 | no | 17.16 |
| **pooled** | 74.09 | 74.09 | 72.95 | 75.51 | 71.65 | 126.24 | — | 17.95 |

### jev_mix (t_trim 0.3) (plain)

Calibrated pooled keep threshold 3.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 79.54 | 79.54 | 76.98 | 79.37 | 74.71 | 81.58 | no | 18.74 |
| greco-2.2-thumbnailing | 74.98 | 74.98 | 71.11 | 73.27 | 70.27 | 154.44 | no | 17.16 |
| **pooled** | 77.53 | 77.53 | 73.87 | 76.60 | 72.69 | 108.27 | — | 17.95 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 80.47 | 80.47 | 79.37 | 82.88 | 78.02 | 90.55 | no | 18.74 |
| greco-2.2-thumbnailing | 70.94 | 70.94 | 70.31 | 73.30 | 70.30 | 162.42 | no | 17.16 |
| **pooled** | 76.26 | 76.26 | 74.57 | 78.53 | 74.52 | 116.87 | — | 17.95 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.60.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 80.95 | 80.95 | 80.22 | 83.75 | 78.84 | 94.74 | no | 18.74 |
| greco-2.2-thumbnailing | 69.38 | 69.38 | 69.62 | 72.73 | 69.75 | 169.27 | no | 17.16 |
| **pooled** | 75.84 | 75.84 | 74.61 | 78.74 | 74.71 | 122.03 | — | 17.95 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 80.53 | 80.53 | 79.36 | 82.89 | 78.03 | 90.73 | no | 18.74 |
| greco-2.2-thumbnailing | 71.07 | 71.07 | 70.28 | 73.24 | 70.24 | 162.90 | no | 17.16 |
| **pooled** | 76.35 | 76.35 | 74.55 | 78.51 | 74.49 | 117.16 | — | 17.95 |

### jev_noul (t_trim 0.3) (layered)

Calibrated pooled keep threshold 3.20.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 79.08 | 79.08 | 77.13 | 81.43 | 76.65 | 92.74 | no | 18.74 |
| greco-2.2-thumbnailing | 72.19 | 72.19 | 69.73 | 72.58 | 69.61 | 156.75 | no | 17.16 |
| **pooled** | 76.04 | 76.04 | 73.21 | 77.41 | 73.45 | 116.18 | — | 17.95 |

### jev_mix (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.80.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 79.86 | 79.86 | 80.15 | 83.85 | 78.93 | 102.67 | no | 18.74 |
| greco-2.2-thumbnailing | 68.41 | 68.41 | 69.98 | 72.76 | 69.78 | 173.38 | no | 17.16 |
| **pooled** | 74.80 | 74.80 | 74.77 | 78.81 | 74.77 | 128.57 | — | 17.95 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 0.69 | 18.05 | n/a | 18.74 | 115 | 4,239,027 | 0.1780 | 24 | 27 |
| greco-2.2-thumbnailing | 0.42 | 16.74 | n/a | 17.16 | 92 | 3,356,599 | 0.1410 | 24 | 26 |
| **total** | 1.11 | 34.79 | 0.00 | 35.91 | 207 | 7,595,626 | 0.3190 | 48 | 53 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 40 | 9 | 107 | no |
| greco-2.2-thumbnailing | 58 | 14 | 97 | no |
| **total** | 98 | 23 | 204 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 0 | 0 | 107 | yes |
| greco-2.2-thumbnailing | 0 | 0 | 97 | yes |
| **total** | 0 | 0 | 204 | — |

### jev_b_notrim

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 0 | 0 | 107 | yes |
| greco-2.2-thumbnailing | 0 | 0 | 97 | yes |
| **total** | 0 | 0 | 204 | — |

### jev_noul (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 40 | 11 | 107 | no |
| greco-2.2-thumbnailing | 58 | 11 | 97 | no |
| **total** | 98 | 22 | 204 | — |

### jev_mix (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 40 | 7 | 107 | no |
| greco-2.2-thumbnailing | 58 | 6 | 97 | no |
| **total** | 98 | 13 | 204 | — |

## The cut_k question (cut_p)

`cut_p` is P(the editor removes this sentence) from the noul asked next to the 0-5 score on every target. `jev_noul` scores a sentence 5 x (1 - cut_p), `jev_mix` averages that with the score; both carry jev_a's trims and retake cut, so the keep decision is the only thing that differs.

| episode | sentences | unusable | r(cut_p, score) | mean cut_p | mean score |
|---|---:|---:|---:|---:|---:|
| perspective-13d-critique | 1752 | 0 | -0.785 | 0.410 | 2.23 |
| greco-2.2-thumbnailing | 1388 | 0 | -0.839 | 0.383 | 2.62 |
| **pooled** | 3140 | — | -0.803 | 0.398 | 2.40 |

Keep/cut agreement with the editor at each arm's own calibrated threshold, layered scoring over 2 episode(s). Counts are sentences: `wrong drops` is the editor kept it and the arm removed it.

| arm | threshold | sentences | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio | SENTENCE POINTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.70 | 3140 | 757 | 1654 | 220 | 509 | 76.78 | 116.87 | 76.26 |
| jev_noul (t_trim 0.3) | 3.20 | 3140 | 746 | 1657 | 231 | 506 | 76.53 | 116.18 | 76.04 |
| jev_mix (t_trim 0.3) | 2.80 | 3140 | 823 | 1554 | 154 | 609 | 75.70 | 128.57 | 74.80 |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 101 | 47 | 0 | 71 | 157 | 5 (1 kept / 4 cut) | 91 (10 kept / 81 cut) |
| greco-2.2-thumbnailing | 59 | 30 | 0 | 44 | 86 | 2 (1 kept / 1 cut) | 44 (2 kept / 42 cut) |

## Where this lands on the published ladder

Reference arms are the 18-episode pooled numbers from `2026-09-11-model-plus-deterministic.json`. NOT THE SAME EPISODE SET as the tables above (missing per-episode numbers for: greco-2.2-thumbnailing), so the comparison is indicative only.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 18 | 79.96 | 79.44 | 88.34 | 83.71 | 80.35 | 93.55 |
| shipped Opus agentic | 18 | 83.45 | 80.84 | 90.37 | 86.47 | 81.44 | 95.02 |
| Luna single call | 18 | 18.61 | 61.23 | 69.43 | 65.36 | 64.63 | 80.67 |
| deterministic baseline (um removal + retakes + delete silence) | 18 | 63.72 | 64.07 | 79.92 | 63.72 | 64.07 | 79.92 |

Best Jev arm here is jev_mix (t_trim 0.3) at 77.53 SENTENCE POINTS (73.87 WORD SCORE, 17.95 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 74.80 SENTENCE POINTS (74.77 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-fitwindow-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-fitwindow-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-fitwindow-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-fitwindow-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-fitwindow-summary.json`
