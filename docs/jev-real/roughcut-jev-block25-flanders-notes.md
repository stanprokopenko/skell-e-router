# Jev rough cut: run report

Generated 2026-09-20T23:04:16+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-block25-flanders-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v3, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 1 episode(s): flanders-03-thematic-crit.
70 requests, 11 errored, 11 retried, 3,123,723 input tokens, $0.1312, 12.982 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 1309.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 1309 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-block25-flanders-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.10 | 75.56 | 75.56 | 77.58 | 80.11 | 72.93 | 105.46 | 12.98 |
| jev_b_moduleretakes | 2.10 | 25.78 | 75.78 | 79.20 | 81.33 | 74.05 | 105.99 | 12.98 |
| jev_b_notrim | 2.10 | 26.01 | 76.01 | 77.98 | 80.41 | 73.21 | 106.62 | 12.98 |
| jev_noul (t_trim 0.3) | 2.90 | 73.68 | 73.68 | 75.10 | 78.93 | 71.86 | 101.00 | 12.98 |
| jev_mix (t_trim 0.3) | 2.60 | 75.24 | 75.24 | 77.20 | 80.32 | 73.12 | 102.78 | 12.98 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.10 | 78.97 | 78.97 | 78.38 | 84.01 | 76.48 | 90.50 | 12.98 |
| jev_b_moduleretakes | 2.10 | 79.14 | 79.14 | 79.97 | 85.34 | 77.70 | 90.61 | 12.98 |
| jev_b_notrim | 2.10 | 79.50 | 79.50 | 78.78 | 84.40 | 76.84 | 91.24 | 12.98 |
| jev_noul (t_trim 0.3) | 2.90 | 76.75 | 76.75 | 75.87 | 82.41 | 75.03 | 87.03 | 12.98 |
| jev_mix (t_trim 0.3) | 2.60 | 78.72 | 78.72 | 78.06 | 84.01 | 76.48 | 88.32 | 12.98 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 75.56 | 75.56 | 77.58 | 80.11 | 72.93 | 105.46 | no | 12.98 |
| **pooled** | 75.56 | 75.56 | 77.58 | 80.11 | 72.93 | 105.46 | — | 12.98 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 25.78 | 75.78 | 79.20 | 81.33 | 74.05 | 105.99 | yes | 12.98 |
| **pooled** | 25.78 | 75.78 | 79.20 | 81.33 | 74.05 | 105.99 | — | 12.98 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 26.01 | 76.01 | 77.98 | 80.41 | 73.21 | 106.62 | yes | 12.98 |
| **pooled** | 26.01 | 76.01 | 77.98 | 80.41 | 73.21 | 106.62 | — | 12.98 |

### jev_noul (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.90.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 73.68 | 73.68 | 75.10 | 78.93 | 71.86 | 101.00 | no | 12.98 |
| **pooled** | 73.68 | 73.68 | 75.10 | 78.93 | 71.86 | 101.00 | — | 12.98 |

### jev_mix (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.60.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 75.24 | 75.24 | 77.20 | 80.32 | 73.12 | 102.78 | no | 12.98 |
| **pooled** | 75.24 | 75.24 | 77.20 | 80.32 | 73.12 | 102.78 | — | 12.98 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 78.97 | 78.97 | 78.38 | 84.01 | 76.48 | 90.50 | no | 12.98 |
| **pooled** | 78.97 | 78.97 | 78.38 | 84.01 | 76.48 | 90.50 | — | 12.98 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 79.14 | 79.14 | 79.97 | 85.34 | 77.70 | 90.61 | no | 12.98 |
| **pooled** | 79.14 | 79.14 | 79.97 | 85.34 | 77.70 | 90.61 | — | 12.98 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 79.50 | 79.50 | 78.78 | 84.40 | 76.84 | 91.24 | no | 12.98 |
| **pooled** | 79.50 | 79.50 | 78.78 | 84.40 | 76.84 | 91.24 | — | 12.98 |

### jev_noul (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.90.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 76.75 | 76.75 | 75.87 | 82.41 | 75.03 | 87.03 | no | 12.98 |
| **pooled** | 76.75 | 76.75 | 75.87 | 82.41 | 75.03 | 87.03 | — | 12.98 |

### jev_mix (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.60.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 78.72 | 78.72 | 78.06 | 84.01 | 76.48 | 88.32 | no | 12.98 |
| **pooled** | 78.72 | 78.72 | 78.06 | 84.01 | 76.48 | 88.32 | — | 12.98 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 0.28 | 12.70 | n/a | 12.98 | 70 | 3,123,723 | 0.1312 | 11 | 11 |
| **total** | 0.28 | 12.70 | 0.00 | 12.98 | 70 | 3,123,723 | 0.1312 | 11 | 11 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 50 | 34 | 145 | no |
| **total** | 50 | 34 | 145 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 0 | 0 | 145 | yes |
| **total** | 0 | 0 | 145 | — |

### jev_b_notrim

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 0 | 0 | 145 | yes |
| **total** | 0 | 0 | 145 | — |

### jev_noul (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 50 | 34 | 145 | no |
| **total** | 50 | 34 | 145 | — |

### jev_mix (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 50 | 33 | 145 | no |
| **total** | 50 | 33 | 145 | — |

## The cut_k question (cut_p)

`cut_p` is P(the editor removes this sentence) from the noul asked next to the 0-5 score on every target. `jev_noul` scores a sentence 5 x (1 - cut_p), `jev_mix` averages that with the score; both carry jev_a's trims and retake cut, so the keep decision is the only thing that differs.

| episode | sentences | unusable | r(cut_p, score) | mean cut_p | mean score |
|---|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 1309 | 0 | -0.838 | 0.363 | 2.83 |
| **pooled** | 1309 | — | -0.838 | 0.363 | 2.83 |

Keep/cut agreement with the editor at each arm's own calibrated threshold, layered scoring over 1 episode(s). Counts are sentences: `wrong drops` is the editor kept it and the arm removed it.

| arm | threshold | sentences | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio | SENTENCE POINTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.10 | 1309 | 823 | 254 | 54 | 178 | 82.28 | 90.50 | 78.97 |
| jev_noul (t_trim 0.3) | 2.90 | 1309 | 770 | 266 | 107 | 166 | 79.14 | 87.03 | 76.75 |
| jev_mix (t_trim 0.3) | 2.60 | 1309 | 800 | 268 | 77 | 164 | 81.59 | 88.32 | 78.72 |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 31 | 7 | 0 | 84 | 96 | 4 (4 kept / 0 cut) | 16 (9 kept / 7 cut) |

## Where this lands on the published ladder

Reference arms re-pooled over the same 1 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 1 | 75.74 | 80.63 | 85.07 | 77.01 | 80.72 | 86.38 |
| shipped Opus agentic | 1 | 76.36 | 81.28 | 86.16 | 78.05 | 81.36 | 87.30 |
| Luna single call | 1 | 14.25 | 66.32 | 69.02 | 68.72 | 67.98 | 75.81 |
| deterministic baseline (um removal + retakes + delete silence) | 1 | 67.96 | 67.65 | 75.29 | 67.96 | 67.65 | 75.29 |

Best Jev arm here is jev_a (t_trim 0.3) at 75.56 SENTENCE POINTS (77.58 WORD SCORE, 12.98 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 78.97 SENTENCE POINTS (78.38 WORD SCORE), above best Luna chapters, shipped Opus agentic, Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block25-flanders-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block25-flanders-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block25-flanders-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block25-flanders-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block25-flanders-summary.json`
