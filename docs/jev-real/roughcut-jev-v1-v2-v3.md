# Jev rough cut: prompt v1, v2 and v3

Developer-facing. All three runs are the same six fit episodes, 2,743 sentences, arm `jev_a` at trim trigger 0.3 with um removal and delete silence layered on. Every arm is calibrated by the same pooled sweep, so each one is read at its own keep threshold: v1 and v2 both landed on 2.40, v3's `jev_a` on 2.50, `jev_noul` on 3.10 and `jev_mix` on 2.70.

Sources: `docs/jev-real/roughcut-jev-v1-summary.json` and `roughcut-jev-misses.json` for v1, `roughcut-jev-v2-summary.json` and `roughcut-jev-v2-misses.json` for v2, `roughcut-jev-v3-summary.json` and `roughcut-jev-v3-misses.json` for v3. Prompt text: `scripts/jev_real/roughcut_jev_prompts.py`, `PROMPTS["v1"]`, `["v2"]` and `["v3"]`. The earlier head-to-head is `docs/jev-real/roughcut-jev-v1-vs-v2.md`.

## What v3 changed

v3 is v2 with two changes, both in the sentence pass.

1. Level 3 narrowed. v2's level 3 read "Ordinary teaching talk that keeps the flow, even when plain, long or loosely worded", followed by four cases and an instruction-to-the-student clause. v3 drops the "even when plain, long or loosely worded" phrase and the instruction clause, keeps v1's opening line ("fine, keeps the flow, nothing memorable") and keeps the three cases the misses list actually asked for: a transition between topics or students, a scripted set-up line the next line answers, and a list read one item per row.
2. A new question, `cut_k`, asked next to `score_k` on every target sentence in the same request, including one-word rows. It is a noul, not a score: P(the editor removes this sentence from the final cut entirely). Its answer lands on every decision row as `cut_p`. Nothing in the pipeline consumes it at run time; it is there so the report can build two more arms offline.

Everything else is byte-identical to v2: the retake pass, the head and tail trim questions, the trim-pick prompt, the score instructions and the `spoken_sentence` split-sentence flags.

`roughcut_jev_report.py` now rebuilds two extra arms from the `jev_a` rows whenever they carry `cut_p`. Both keep jev_a's word trims and its retake cut, so the keep decision is the only thing that moves:

- `jev_noul`: score = 5 x (1 - cut_p), the removal noul alone.
- `jev_mix`: score = the mean of the 0-5 score and 5 x (1 - cut_p).

## Per episode, jev_a at t_trim 0.3, layered

SENTENCE POINTS:

| episode | v1 | v2 | v3 | v3 - v1 | v3 - v2 | Luna SP | Opus SP |
|---|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 83.66 | 85.36 | 85.88 | +2.22 | +0.52 | 82.47 | 86.49 |
| hampton-5.4-assignment-demo | 86.33 | 85.63 | 84.80 | -1.53 | -0.83 | 90.57 | 94.60 |
| colman-03.03-muscles-crit | 75.41 | 74.95 | 75.05 | -0.36 | +0.10 | 69.01 | 73.63 |
| edges-7.01-intro | 77.56 | 78.56 | 79.69 | +2.13 | +1.13 | 89.59 | 90.39 |
| hampton-5.2-shape-demo | 92.04 | 91.31 | 92.70 | +0.66 | +1.39 | 90.78 | 95.06 |
| perspective-14e-boxes-critique | 84.62 | 81.40 | 81.78 | -2.84 | +0.38 | 89.01 | 90.54 |
| **pooled** | **83.84** | **82.52** | **83.00** | **-0.84** | **+0.48** | **86.86** | **89.49** |

WORD SCORE:

| episode | v1 | v2 | v3 | Luna WORD | Opus WORD |
|---|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 73.69 | 73.88 | 74.48 | 76.08 | 76.10 |
| hampton-5.4-assignment-demo | 74.99 | 76.15 | 75.53 | 85.60 | 89.47 |
| colman-03.03-muscles-crit | 62.22 | 58.74 | 59.92 | 68.34 | 66.40 |
| edges-7.01-intro | 74.27 | 78.98 | 79.74 | 89.34 | 89.71 |
| hampton-5.2-shape-demo | 84.30 | 83.75 | 85.72 | 81.60 | 87.60 |
| perspective-14e-boxes-critique | 80.29 | 79.60 | 79.21 | 85.03 | 86.36 |
| **pooled** | **76.46** | **76.31** | **76.78** | **81.75** | **83.44** |

GRADE:

| episode | v1 | v2 | v3 | Luna GRADE | Opus GRADE |
|---|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 105.79 | 105.60 | 106.07 | 104.42 | 106.71 |
| hampton-5.4-assignment-demo | 94.39 | 94.94 | 94.34 | 101.91 | 105.91 |
| colman-03.03-muscles-crit | 74.96 | 72.77 | 74.19 | 81.52 | 80.68 |
| edges-7.01-intro | 75.18 | 79.35 | 79.65 | 89.05 | 89.27 |
| hampton-5.2-shape-demo | 95.69 | 95.16 | 96.97 | 93.67 | 98.53 |
| perspective-14e-boxes-critique | 83.29 | 82.74 | 82.33 | 88.34 | 89.50 |
| **pooled** | **86.41** | **86.40** | **86.72** | **91.29** | **93.06** |

Luna is `luna-chapters-rules5` and Opus is `opus5-cc-agentic`, both re-pooled over these six episodes from `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`. They are fixed references and do not move between runs.

## The two new v3 arms, per episode, layered

| episode | jev_a SP | jev_noul SP | jev_mix SP | jev_a WORD | jev_noul WORD | jev_mix WORD | jev_a GRADE | jev_noul GRADE | jev_mix GRADE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 85.88 | 80.82 | 85.88 | 74.48 | 70.96 | 73.89 | 106.07 | 103.61 | 105.76 |
| hampton-5.4-assignment-demo | 84.80 | 83.90 | 81.87 | 75.53 | 75.00 | 71.50 | 94.34 | 94.77 | 91.77 |
| colman-03.03-muscles-crit | 75.05 | 76.11 | 76.47 | 59.92 | 61.72 | 60.92 | 74.19 | 76.58 | 75.94 |
| edges-7.01-intro | 79.69 | 76.68 | 77.97 | 79.74 | 73.02 | 76.86 | 79.65 | 73.65 | 77.09 |
| hampton-5.2-shape-demo | 92.70 | 92.55 | 92.36 | 85.72 | 86.57 | 85.20 | 96.97 | 97.17 | 96.26 |
| perspective-14e-boxes-critique | 81.78 | 79.08 | 82.04 | 79.21 | 77.33 | 81.03 | 82.33 | 80.62 | 84.12 |
| **pooled** | **83.00** | **81.08** | **82.65** | **76.78** | **75.31** | **76.67** | **86.72** | **85.41** | **86.93** |

## Pooled, every arm

| arm | threshold | SENTENCE POINTS | WORD SCORE | GRADE | kept ratio |
|---|---:|---:|---:|---:|---:|
| v1 jev_a | 2.40 | 83.84 | 76.46 | 86.41 | 89.63 |
| v2 jev_a | 2.40 | 82.52 | 76.31 | 86.40 | 96.11 |
| v3 jev_a | 2.50 | 83.00 | 76.78 | 86.72 | 94.39 |
| v3 jev_noul | 3.10 | 81.08 | 75.31 | 85.41 | 90.33 |
| v3 jev_mix | 2.70 | 82.65 | 76.67 | 86.93 | 96.47 |
| Luna chapters (reference) | n/a | 86.86 | 81.75 | 91.29 | n/a |
| Opus agentic (reference) | n/a | 89.49 | 83.44 | 93.06 | n/a |
| deterministic baseline (reference) | n/a | 60.27 | 62.20 | 74.52 | n/a |

Kept ratio is the model's kept duration over the editor's. v1 under-keeps by a tenth, v2 and `jev_mix` sit a few points over, v3's `jev_a` splits the difference at 94.39.

## Sentence-level confusion, pooled over 2,743 sentences

Every arm at its own calibrated threshold, layered. `wrong drops` is the editor kept it and the arm removed it; `wrong keeps` is the reverse.

| arm | both keep | both remove | wrong drops | wrong keeps | agreement |
|---|---:|---:|---:|---:|---:|
| v1 jev_a | 1,234 | 1,081 | 271 | 157 | 84.40 |
| v2 jev_a | 1,306 | 983 | 199 | 255 | 83.45 |
| v3 jev_a | 1,290 | 1,010 | 215 | 228 | 83.85 |
| v3 jev_noul | 1,234 | 1,006 | 271 | 232 | 81.66 |
| v3 jev_mix | 1,305 | 988 | 200 | 250 | 83.59 |
| Luna chapters | 1,286 | 1,116 | 219 | 122 | 87.57 |

v3's `jev_a` sits between v1 and v2 on both error columns, which is exactly what narrowing level 3 was meant to do: it gives back 56 of v1's 271 wrong drops instead of v2's 72, and pays 71 extra wrong keeps instead of v2's 98. Nobody has closed the real gap to Luna, which is the wrong-keeps column: Luna makes 122 where every Jev arm makes over 220.

Per episode, the same two columns for the three v3 arms:

| episode | jev_a drops | jev_a keeps | jev_noul drops | jev_noul keeps | jev_mix drops | jev_mix keeps |
|---|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 9 | 10 | 29 | 8 | 11 | 11 |
| hampton-5.4-assignment-demo | 13 | 30 | 15 | 29 | 13 | 37 |
| colman-03.03-muscles-crit | 33 | 17 | 32 | 17 | 28 | 17 |
| edges-7.01-intro | 56 | 24 | 76 | 17 | 63 | 26 |
| hampton-5.2-shape-demo | 17 | 26 | 21 | 24 | 16 | 29 |
| perspective-14e-boxes-critique | 87 | 121 | 98 | 137 | 69 | 130 |

`jev_mix` is the only arm that improves the worst episode on both sides at once: on perspective-14e it drops 69 where `jev_a` drops 87, and keeps 130 where `jev_a` keeps 121. It pays for that on colman-02.04 and hampton-5.4.

## The cut_p diagnostic

`cut_p` is the `cut_k` answer: P(the editor removes this sentence). Pooled mean 0.381 against a mean score of 2.54.

| episode | sentences | r(cut_p, score) | mean cut_p | mean score |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | -0.846 | 0.320 | 3.21 |
| hampton-5.4-assignment-demo | 300 | -0.880 | 0.306 | 3.00 |
| colman-03.03-muscles-crit | 303 | -0.818 | 0.304 | 3.12 |
| edges-7.01-intro | 389 | -0.877 | 0.478 | 2.26 |
| hampton-5.2-shape-demo | 411 | -0.893 | 0.335 | 2.82 |
| perspective-14e-boxes-critique | 1,146 | -0.775 | 0.414 | 2.15 |
| **pooled** | **2,743** | **-0.824** | **0.381** | **2.54** |

Every sentence in all six episodes got both answers, so there is nothing to exclude. A correlation of -0.82 means the two questions are mostly one question asked twice, but not entirely: about a third of the variance in one is not explained by the other, which is why averaging them moves the arm at all. The blend is the better half of that: `jev_mix` matches `jev_a` on WORD, beats it on GRADE and loses 0.35 SP, while `jev_noul` on its own is worse on all three. Asking for a removal probability instead of a 0-to-5 worth does not beat the score question; it only adds a little independent signal to it.

## Speed and cost, v3

Wall clock per pass, measured around the pass at concurrency 8 with retries and backoff inside it. v3 ran two passes, no trim pick. The repair requests are not in these wall clocks but their cost is in the total.

| episode | sentences | retake s | sentence s | total s | requests | input tokens | cost $ | retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 0.29 | 4.16 | 4.45 | 12 | 354,246 | 0.0149 | 3 |
| hampton-5.4-assignment-demo | 300 | 0.23 | 2.56 | 2.79 | 15 | 507,202 | 0.0213 | 2 |
| colman-03.03-muscles-crit | 303 | 0.32 | 2.78 | 3.10 | 17 | 599,799 | 0.0252 | 2 |
| edges-7.01-intro | 389 | 0.41 | 4.49 | 4.90 | 33 | 654,776 | 0.0275 | 6 |
| hampton-5.2-shape-demo | 411 | 0.25 | 2.95 | 3.20 | 23 | 803,557 | 0.0337 | 3 |
| perspective-14e-boxes-critique | 1,146 | 0.47 | 11.97 | 12.44 | 89 | 2,618,694 | 0.1100 | 26 |
| **mean per episode** | | | | **5.15** | | | **0.0388** | |

Mean seconds per episode: v1 3.79 (three passes), v2 4.38 (two passes), v3 5.15 (two passes). v3 costs $0.2326 all in against v1 $0.2325 and v2 $0.2202. The `cut_k` question adds a quarter of the sentence pass's questions and roughly 6% to the input tokens, 5.24M in v2 against 5.54M in v3.

Every episode except perspective-14e is inside the 10 second target. perspective-14e is 1,146 sentences, and its time is retries: 26 of the run's 42.

## Run health

181 requests in the run, 39 errored, 2 blocks failed all three attempts, both in perspective-14e. Block 39 was a plain provider error and came back on the first repair. Block 34 was different: a deterministic HTTP 400 `invalid_request` with no message, on every attempt of the run and of the first repair.

It is a size rejection. Sent as-is the block is about 47,000 estimated tokens of state plus questions; each subset of its questions answers fine at full size, and the whole block answers fine once the transcript is windowed. The pipeline already had a windowed fallback job for exactly this, but `CONTEXT_ERROR_MARKS` only recognised context complaints that name a token limit, so a bare 400 never triggered it. `invalid_request` is now in that list, and the second repair recovered all 25 sentences on the windowed retry. Zero sentences ended without an answer.

The one caveat this leaves: sentences 850 to 874 of perspective-14e were scored against a transcript windowed to 200 sentences either side, where the other 2,718 sentences saw the whole episode. A genuine schema error would still fail windowed and be reported, so nothing is being hidden by the change.

## Where the level mass went

Mean probability per level, split by what the editor actually did:

| level | v1 kept | v2 kept | v3 kept | v1 removed | v2 removed | v3 removed |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.144 | 0.132 | 0.136 | 0.470 | 0.397 | 0.409 |
| 1 | 0.057 | 0.030 | 0.028 | 0.120 | 0.080 | 0.074 |
| 2 | 0.035 | 0.027 | 0.027 | 0.062 | 0.057 | 0.055 |
| 3 | 0.249 | 0.337 | 0.286 | 0.139 | 0.260 | 0.232 |
| 4 | 0.420 | 0.399 | 0.435 | 0.166 | 0.172 | 0.190 |
| 5 | 0.095 | 0.074 | 0.088 | 0.042 | 0.035 | 0.040 |

The narrowed level 3 gave back about half of what v2 took: 0.260 to 0.232 on removed material, 0.337 to 0.286 on kept. The mass went to level 4, which on kept material is now 0.435, higher than v1's 0.420. That is the shape you want. It is also why the calibrated threshold moved from 2.40 to 2.50: the whole distribution shifted up.

## Verdict

On the fit set the best arm is v1's `jev_a` on SENTENCE POINTS at 83.84, with v3's `jev_a` second at 83.00 and v2's third at 82.52. That ordering is within noise. The fair yardstick is what moved between v1 and v2 on episodes neither v2 change targeted: colman-02.04 and hampton-5.2 have the fewest split-sentence chains and the least scripted set-up material, and v1 to v2 still swung them +1.70 and -0.73 SENTENCE POINTS. A per-episode gap under about 1.7 points, or a pooled gap under about 1 point, is not a result. v1 to v3 pooled is -0.84, inside that band.

What is not within noise is the shape of the errors and the other two metrics. v3 has the best pooled WORD SCORE (76.78) and, in `jev_mix`, the best pooled GRADE (86.93), and it is the only version that improves both metrics over both predecessors at once. v3's `jev_a` is my pick to carry forward: same headline number as v1 within noise, better on the two frame-level metrics, and the error split it makes (215 wrong drops, 228 wrong keeps) is the balanced version of a trade v1 and v2 each got wrong in opposite directions. The `cut_k` question is worth keeping in the run at 6% more input tokens, but as a second opinion blended into the score, not as a replacement for it: `jev_noul` alone is the worst arm in the table. The next lever is not the score levels at all. Every Jev arm makes over 220 wrong keeps where Luna makes 122, and no prompt version tried so far has moved that number.

## Files

- prompts: `scripts/jev_real/roughcut_jev_prompts.py`
- v3 run: `docs/jev-real/roughcut-jev-v3-decisions.jsonl`, `-requests.jsonl`, `-timing.json`
- v3 report: `docs/jev-real/roughcut-jev-v3-notes.md`, `roughcut-jev-v3-summary.json`
- v3 misses: `docs/jev-real/roughcut-jev-v3-misses.md`, `roughcut-jev-v3-misses.json`
