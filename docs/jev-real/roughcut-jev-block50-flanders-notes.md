# Jev rough cut: run report

Generated 2026-09-20T23:04:28+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-block50-flanders-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v3, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 1 episode(s): flanders-03-thematic-crit.
80 requests, 70 errored, 47 retried, 246,059 input tokens, $0.0103, 19.354 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 1150 per arm out of 1309.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 1309 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-block50-flanders-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 0.00 | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | 19.35 |
| jev_b_moduleretakes | 0.00 | 14.25 | 64.25 | 66.32 | 69.02 | 62.84 | 119.34 | 19.35 |
| jev_b_notrim | 0.00 | 14.17 | 64.17 | 64.43 | 67.59 | 61.53 | 119.96 | 19.35 |
| jev_noul (t_trim 0.3) | 0.00 | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | 19.35 |
| jev_mix (t_trim 0.3) | 0.00 | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | 19.35 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 0.00 | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | 19.35 |
| jev_b_moduleretakes | 0.00 | 68.72 | 68.72 | 67.98 | 75.81 | 69.02 | 101.61 | 19.35 |
| jev_b_notrim | 0.00 | 68.77 | 68.77 | 66.25 | 74.56 | 67.88 | 102.22 | 19.35 |
| jev_noul (t_trim 0.3) | 0.00 | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | 19.35 |
| jev_mix (t_trim 0.3) | 0.00 | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | 19.35 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | no | 19.35 |
| **pooled** | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | — | 19.35 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 14.25 | 64.25 | 66.32 | 69.02 | 62.84 | 119.34 | yes | 19.35 |
| **pooled** | 14.25 | 64.25 | 66.32 | 69.02 | 62.84 | 119.34 | — | 19.35 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 14.17 | 64.17 | 64.43 | 67.59 | 61.53 | 119.96 | yes | 19.35 |
| **pooled** | 14.17 | 64.17 | 64.43 | 67.59 | 61.53 | 119.96 | — | 19.35 |

### jev_noul (t_trim 0.3) (plain)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | no | 19.35 |
| **pooled** | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | — | 19.35 |

### jev_mix (t_trim 0.3) (plain)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | no | 19.35 |
| **pooled** | 64.12 | 64.12 | 64.46 | 67.62 | 61.56 | 119.75 | — | 19.35 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | no | 19.35 |
| **pooled** | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | — | 19.35 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 68.72 | 68.72 | 67.98 | 75.81 | 69.02 | 101.61 | no | 19.35 |
| **pooled** | 68.72 | 68.72 | 67.98 | 75.81 | 69.02 | 101.61 | — | 19.35 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 68.77 | 68.77 | 66.25 | 74.56 | 67.88 | 102.22 | no | 19.35 |
| **pooled** | 68.77 | 68.77 | 66.25 | 74.56 | 67.88 | 102.22 | — | 19.35 |

### jev_noul (t_trim 0.3) (layered)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | no | 19.35 |
| **pooled** | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | — | 19.35 |

### jev_mix (t_trim 0.3) (layered)

Calibrated pooled keep threshold 0.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | no | 19.35 |
| **pooled** | 68.72 | 68.72 | 66.29 | 74.62 | 67.93 | 102.08 | — | 19.35 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 0.25 | 19.11 | n/a | 19.35 | 80 | 246,059 | 0.0103 | 47 | 70 |
| **total** | 0.25 | 19.11 | 0.00 | 19.35 | 80 | 246,059 | 0.0103 | 47 | 70 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 9 | 9 | 145 | no |
| **total** | 9 | 9 | 145 | — |

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
| flanders-03-thematic-crit | 9 | 9 | 145 | no |
| **total** | 9 | 9 | 145 | — |

### jev_mix (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| flanders-03-thematic-crit | 9 | 9 | 145 | no |
| **total** | 9 | 9 | 145 | — |

## The cut_k question (cut_p)

`cut_p` is P(the editor removes this sentence) from the noul asked next to the 0-5 score on every target. `jev_noul` scores a sentence 5 x (1 - cut_p), `jev_mix` averages that with the score; both carry jev_a's trims and retake cut, so the keep decision is the only thing that differs.

| episode | sentences | unusable | r(cut_p, score) | mean cut_p | mean score |
|---|---:|---:|---:|---:|---:|
| flanders-03-thematic-crit | 159 | 1150 | -0.778 | 0.505 | 1.84 |
| **pooled** | 159 | — | -0.778 | 0.505 | 1.84 |

Keep/cut agreement with the editor at each arm's own calibrated threshold, layered scoring over 1 episode(s). Counts are sentences: `wrong drops` is the editor kept it and the arm removed it.

| arm | threshold | sentences | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio | SENTENCE POINTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 0.00 | 1309 | 852 | 87 | 25 | 345 | 71.73 | 102.08 | 68.72 |
| jev_noul (t_trim 0.3) | 0.00 | 1309 | 852 | 87 | 25 | 345 | 71.73 | 102.08 | 68.72 |
| jev_mix (t_trim 0.3) | 0.00 | 1309 | 852 | 87 | 25 | 345 | 71.73 | 102.08 | 68.72 |

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

Best Jev arm here is jev_a (t_trim 0.3) at 64.12 SENTENCE POINTS (64.46 WORD SCORE, 19.35 s per episode), above Luna single call.
With um removal and delete silence layered on, the same arm is 68.72 SENTENCE POINTS (66.29 WORD SCORE), above deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block50-flanders-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block50-flanders-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block50-flanders-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block50-flanders-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-block50-flanders-summary.json`
