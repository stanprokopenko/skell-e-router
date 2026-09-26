# Route 1 flags (developer-facing): can Jev point the smart trimmer at the right sentences?

Developer-facing record. Generated 2026-09-26T06:53:51+00:00 by `scripts/jev_real/roughcut_route1_flags.py` from the stored `roughcut-jev-all18-v3` decisions and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals, unless it is a count, a threshold or a dollar figure; the JSON next to this file keeps the raw values.

Route 1 is a hybrid: Jev decides keep or cut for every sentence as now, and flags the sentences it thinks need an inside-sentence trim; only those go to a smarter model that picks the kept words. This file asks whether the stored Jev fields can do the flagging, what the route could add at best, and roughly what the smart step would cost.

The arm is jev_a (t_trim 0.3) at threshold 2.50, um removal + delete silence layered on. Reproduced baseline: 80.47 SENTENCE POINTS, 74.12 WORD SCORE, 88.44 GRADE over the 18 ladder episodes, matching the published 80.47 / 74.12 / 88.44. `partial`, `full` and `removed` are the editor's per-sentence states from the harness's `sentence_states` (majority rule), the same states SENTENCE POINTS reads.

| split | sentences | editor full | editor partial | editor removed | partial share of kept | Jev keeps at 2.50 |
|---|---:|---:|---:|---:|---:|---:|
| fit six | 2743 | 1117 | 388 | 1238 | 25.78 | 1541 |
| held-out 12 | 6200 | 3063 | 975 | 2162 | 24.15 | 4299 |
| pooled 18 | 8943 | 4180 | 1363 | 3400 | 24.59 | 5840 |

SENTENCE POINTS per sentence, from `sentence_scoring.sentence_points`, is what makes trims matter: an editor-partial sentence the model keeps whole or removes earns 0.4; the model's partial earns 2.0 when its runs match the editor's exactly, 1.2 when they are a subset, 1.0 when they overlap and 0.6 when they are disjoint. Trimming a sentence the editor kept whole drops it from 1.0 to 0.7. One exact trim is worth 1.6 points on its sentence, and a wrong trim on a whole sentence costs 0.3.

## Summary

- `max_head_tail` on editor-kept sentences: AUC 58.14; at 70.36 recall precision 27.63 and 59.88 of all sentences flagged; at 90.02 recall precision 25.83, 80.59 flagged. Random flagging has precision 24.59.
- `word_count` on editor-kept sentences: AUC 64.60; at 70.87 recall precision 30.86 and 42.87 of all sentences flagged; at 91.78 recall precision 27.11, 70.00 flagged. Random flagging has precision 24.59.
- The modules alone already get 345 of 1363 editor partials exactly right (25.31); in the v3 arm the layers make 308 exact and Jev's own trims 16.
- Oracle SP with perfect trims, modules on top, against 80.47: `max_head_tail` 70.00: 89.70 (+9.23); `max_head_tail` 90.00: 92.34 (+11.87); `word_count` 70.00: 90.22 (+9.76); `perfect flag` 100.00: 93.75 (+13.29).
- With Luna's stored trims in place of the oracle's, SP gain: `max_head_tail` 70.00: -0.15; `max_head_tail` 90.00: -0.29; `word_count` 70.00: -0.30; `word_count` 90.00: -0.39; `perfect flag` 100.00: +0.48; `every sentence Jev keeps` n/a: -0.29.
- Cost at `max_head_tail` 70.00: 199 sentences sent per episode (max 636), about $0.025 per episode on gpt-5.6-luna (max $0.078), 12 s added per episode (max 32 s).
- Cost at `max_head_tail` 90.00: 272 sentences sent per episode (max 805), about $0.034 per episode on gpt-5.6-luna (max $0.099), 16 s added per episode (max 44 s).

## Signals

- `head`: 1 - first_p_whole (Jev thinks the start should go)
- `tail`: 1 - last_p_whole (Jev thinks the end should go)
- `max_head_tail`: max of head and tail
- `either_side`: 1 - first_p_whole x last_p_whole (either side, as if independent)
- `mid_mass`: keep-score probability on levels 2 and 3
- `score_entropy`: entropy of the 0-5 keep-score distribution, bits
- `cut_p`: cut_p, Jev's P(editor removes this sentence)
- `word_count`: NOT Jev: transcript words in the sentence, ums included
- `jev_trim_t03`: NOT a score: Jev's own trim at t_trim 0.3 (1 or 0)

Null trim fields: 812 of 8943 rows have `first_p_whole` and `last_p_whole` both null, and no row has only one of them. The pipeline asks the first and last questions only when a sentence has at least two words after the um-removal words are stripped (`sentence_jobs` in `roughcut_jev.py`, `if len(words) >= 2`), and every one of these rows is under that bar: 106 have no words left, 706 have one. No row is null because a request failed. Head and tail signals score these rows 0. Only 1 of the editor's 1363 partial sentences sits in these rows (editor states in null rows: full 112, partial 1, removed 699), so the nulls cost the head and tail signals almost nothing.

## What the modules already make partial

The um removal and delete silence modules cut inside sentences on their own, so some of the editor's partials need no smart model. `modules alone` keeps every sentence whole and layers the two modules on; `v3 arm` is the actual baseline. Branch names are the metric's: `exact` pays full partial credit, the others pay less.

| split | editor partials | modules alone exact | modules alone partial, any branch | v3 arm exact via layers | v3 arm partial via layers | v3 arm exact via Jev trim | v3 arm partial via Jev trim | v3 arm keeps whole | v3 arm removes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fit six | 388 | 115 (29.64) | 189 | 104 (26.80) | 168 | 4 | 6 | 168 | 46 |
| held-out 12 | 975 | 230 (23.59) | 462 | 204 (20.92) | 407 | 12 | 34 | 422 | 112 |
| pooled 18 | 1363 | 345 (25.31) | 651 | 308 (22.60) | 575 | 16 | 40 | 590 | 158 |

Pooled detail, v3 arm: Jev trim: disjoint 19, Jev trim: exact 16, Jev trim: overlap 3, Jev trim: subset 2, arm full 590, arm removed 158, layers only: disjoint 160, layers only: exact 308, layers only: overlap 107. Modules alone: disjoint 186, exact 345, model full 709, model removed 3, overlap 120.

## Flag quality

Positive = the editor made the sentence partial. `editor kept` restricts to sentences the editor kept (full against partial), the population the brief asks about; `all sentences` adds the removed ones as negatives; `Jev keeps` restricts further to sentences Jev keeps at 2.50, which is the only place a trim can land. AUC 50 is a coin flip.

| signal | AUC fit, editor kept | AUC held-out, editor kept | AUC pooled, editor kept | AUC pooled, all sentences | AUC pooled, Jev keeps and editor kept |
|---|---:|---:|---:|---:|---:|
| `head` | 52.81 | 57.63 | 56.18 | 56.82 | 55.55 |
| `tail` | 57.27 | 58.94 | 58.20 | 60.36 | 57.78 |
| `max_head_tail` | 55.43 | 59.46 | 58.14 | 59.13 | 57.88 |
| `either_side` | 55.64 | 59.45 | 58.19 | 59.09 | 57.68 |
| `mid_mass` | 49.41 | 46.36 | 47.35 | 46.31 | 47.84 |
| `score_entropy` | 47.64 | 51.67 | 50.48 | 47.63 | 50.93 |
| `cut_p` | 48.19 | 49.51 | 49.09 | 33.79 | 49.62 |
| `word_count` | 67.06 | 63.70 | 64.60 | 73.15 | 64.11 |
| `jev_trim_t03` | 50.30 | 51.17 | 50.92 | 50.58 | 50.64 |

Best Jev signal by pooled AUC on editor-kept sentences: `tail`. The top three (`tail` 58.20, `either_side` 58.19, `max_head_tail` 58.14) are within a point of each other. The oracle, Luna and cost sections use `max_head_tail` as the Jev flag and `word_count` as the no-Jev comparison.

### Operating points, pooled 18, editor kept

Threshold picked inside this split to reach each recall target (the flag is `signal >= threshold`; ties can overshoot the target). Base rate, the precision of flagging at random: 24.59. `share of all` is the share of every sentence in the split, removed ones included, the flag fires on.

| signal | recall target | threshold | recall | precision | share of kept flagged | share of all |
|---|---:|---:|---:|---:|---:|---:|
| `head` | 30.00 | 0.580 | 31.62 | 30.57 | 25.44 | 25.79 |
| `head` | 50.00 | 0.490 | 52.46 | 28.35 | 45.50 | 45.59 |
| `head` | 70.00 | 0.400 | 71.68 | 26.51 | 66.48 | 64.68 |
| `head` | 90.00 | 0.260 | 90.68 | 25.26 | 88.29 | 83.14 |
| `tail` | 30.00 | 0.610 | 31.69 | 31.67 | 24.61 | 23.72 |
| `tail` | 50.00 | 0.540 | 52.53 | 30.33 | 42.59 | 40.18 |
| `tail` | 70.00 | 0.450 | 71.75 | 27.65 | 63.81 | 59.55 |
| `tail` | 90.00 | 0.310 | 90.90 | 25.70 | 86.97 | 81.04 |
| `max_head_tail` | 30.00 | 0.650 | 31.55 | 31.66 | 24.50 | 24.87 |
| `max_head_tail` | 50.00 | 0.590 | 50.26 | 30.10 | 41.06 | 40.10 |
| `max_head_tail` | 70.00 | 0.510 | 70.36 | 27.63 | 62.62 | 59.88 |
| `max_head_tail` | 90.00 | 0.380 | 90.02 | 25.83 | 85.71 | 80.59 |
| `either_side` | 30.00 | 0.836 | 30.01 | 32.96 | 22.39 | 22.86 |
| `either_side` | 50.00 | 0.777 | 50.11 | 30.13 | 40.90 | 40.37 |
| `either_side` | 70.00 | 0.700 | 70.21 | 27.88 | 61.93 | 59.53 |
| `either_side` | 90.00 | 0.541 | 90.10 | 25.66 | 86.34 | 81.26 |
| `mid_mass` | 30.00 | 0.340 | 30.23 | 22.56 | 32.94 | 33.29 |
| `mid_mass` | 50.00 | 0.230 | 51.14 | 23.21 | 54.18 | 56.25 |
| `mid_mass` | 70.00 | 0.160 | 71.53 | 24.05 | 73.14 | 75.66 |
| `mid_mass` | 90.00 | 0.080 | 91.93 | 24.43 | 92.51 | 93.63 |
| `score_entropy` | 30.00 | 1.933 | 30.01 | 26.39 | 27.96 | 31.22 |
| `score_entropy` | 50.00 | 1.781 | 50.04 | 25.50 | 48.26 | 51.40 |
| `score_entropy` | 70.00 | 1.553 | 70.07 | 24.41 | 70.59 | 73.30 |
| `score_entropy` | 90.00 | 1.227 | 90.02 | 24.57 | 90.08 | 90.71 |
| `cut_p` | 30.00 | 0.330 | 30.45 | 22.74 | 32.92 | 52.53 |
| `cut_p` | 50.00 | 0.280 | 53.26 | 24.14 | 54.25 | 69.04 |
| `cut_p` | 70.00 | 0.240 | 71.46 | 24.56 | 71.53 | 81.05 |
| `cut_p` | 90.00 | 0.190 | 90.54 | 25.30 | 87.98 | 92.20 |
| `word_count` | 30.00 | 20.000 | 32.36 | 41.25 | 19.29 | 13.23 |
| `word_count` | 50.00 | 13.000 | 54.22 | 34.36 | 38.81 | 28.02 |
| `word_count` | 70.00 | 9.000 | 70.87 | 30.86 | 56.47 | 42.87 |
| `word_count` | 90.00 | 5.000 | 91.78 | 27.11 | 83.26 | 70.00 |
| `jev_trim_t03` | 30.00 | 0.000 | 100.00 | 24.59 | 100.00 | 100.00 |
| `jev_trim_t03` | 50.00 | 0.000 | 100.00 | 24.59 | 100.00 | 100.00 |
| `jev_trim_t03` | 70.00 | 0.000 | 100.00 | 24.59 | 100.00 | 100.00 |
| `jev_trim_t03` | 90.00 | 0.000 | 100.00 | 24.59 | 100.00 | 100.00 |

### Operating points, fit six, editor kept

Threshold picked inside this split to reach each recall target (the flag is `signal >= threshold`; ties can overshoot the target). Base rate, the precision of flagging at random: 25.78. `share of all` is the share of every sentence in the split, removed ones included, the flag fires on.

| signal | recall target | threshold | recall | precision | share of kept flagged | share of all |
|---|---:|---:|---:|---:|---:|---:|
| `head` | 30.00 | 0.530 | 31.70 | 28.21 | 28.97 | 34.52 |
| `head` | 50.00 | 0.450 | 51.80 | 26.69 | 50.03 | 53.37 |
| `head` | 70.00 | 0.370 | 72.42 | 26.92 | 69.37 | 68.76 |
| `head` | 90.00 | 0.220 | 90.21 | 26.00 | 89.44 | 83.89 |
| `tail` | 30.00 | 0.570 | 30.15 | 31.20 | 24.92 | 26.21 |
| `tail` | 50.00 | 0.500 | 50.26 | 30.23 | 42.86 | 42.03 |
| `tail` | 70.00 | 0.400 | 71.91 | 28.76 | 64.45 | 61.90 |
| `tail` | 90.00 | 0.280 | 90.21 | 26.90 | 86.45 | 80.17 |
| `max_head_tail` | 30.00 | 0.610 | 31.19 | 30.10 | 26.71 | 30.70 |
| `max_head_tail` | 50.00 | 0.540 | 51.80 | 29.17 | 45.78 | 47.83 |
| `max_head_tail` | 70.00 | 0.450 | 73.45 | 27.46 | 68.97 | 67.70 |
| `max_head_tail` | 90.00 | 0.340 | 90.21 | 26.74 | 86.98 | 81.81 |
| `either_side` | 30.00 | 0.798 | 30.15 | 31.03 | 25.05 | 29.49 |
| `either_side` | 50.00 | 0.737 | 50.26 | 29.59 | 43.79 | 46.08 |
| `either_side` | 70.00 | 0.652 | 70.36 | 27.91 | 64.98 | 64.53 |
| `either_side` | 90.00 | 0.492 | 90.21 | 26.72 | 87.04 | 81.81 |
| `mid_mass` | 30.00 | 0.390 | 30.41 | 24.63 | 31.83 | 27.27 |
| `mid_mass` | 50.00 | 0.280 | 50.00 | 26.01 | 49.57 | 47.50 |
| `mid_mass` | 70.00 | 0.170 | 71.65 | 25.53 | 72.36 | 73.82 |
| `mid_mass` | 90.00 | 0.080 | 92.78 | 25.66 | 93.22 | 94.53 |
| `score_entropy` | 30.00 | 1.872 | 30.15 | 23.73 | 32.76 | 38.61 |
| `score_entropy` | 50.00 | 1.701 | 50.00 | 23.29 | 55.35 | 59.57 |
| `score_entropy` | 70.00 | 1.501 | 70.10 | 25.23 | 71.63 | 74.95 |
| `score_entropy` | 90.00 | 1.214 | 90.21 | 25.93 | 89.70 | 90.30 |
| `cut_p` | 30.00 | 0.320 | 32.22 | 23.54 | 35.28 | 60.26 |
| `cut_p` | 50.00 | 0.270 | 55.15 | 24.65 | 57.67 | 74.95 |
| `cut_p` | 70.00 | 0.230 | 74.48 | 25.69 | 74.75 | 85.60 |
| `cut_p` | 90.00 | 0.190 | 90.98 | 26.70 | 87.84 | 93.15 |
| `word_count` | 30.00 | 20.000 | 33.51 | 47.97 | 18.01 | 11.67 |
| `word_count` | 50.00 | 13.000 | 52.06 | 38.70 | 34.68 | 24.61 |
| `word_count` | 70.00 | 8.000 | 72.68 | 32.68 | 57.34 | 43.86 |
| `word_count` | 90.00 | 5.000 | 92.01 | 29.53 | 80.33 | 66.64 |
| `jev_trim_t03` | 30.00 | 0.000 | 100.00 | 25.78 | 100.00 | 100.00 |
| `jev_trim_t03` | 50.00 | 0.000 | 100.00 | 25.78 | 100.00 | 100.00 |
| `jev_trim_t03` | 70.00 | 0.000 | 100.00 | 25.78 | 100.00 | 100.00 |
| `jev_trim_t03` | 90.00 | 0.000 | 100.00 | 25.78 | 100.00 | 100.00 |

### Operating points, held-out 12, editor kept

Threshold picked inside this split to reach each recall target (the flag is `signal >= threshold`; ties can overshoot the target). Base rate, the precision of flagging at random: 24.15. `share of all` is the share of every sentence in the split, removed ones included, the flag fires on.

| signal | recall target | threshold | recall | precision | share of kept flagged | share of all |
|---|---:|---:|---:|---:|---:|---:|
| `head` | 30.00 | 0.600 | 30.26 | 30.99 | 23.58 | 22.48 |
| `head` | 50.00 | 0.510 | 50.97 | 28.30 | 43.49 | 41.66 |
| `head` | 70.00 | 0.420 | 70.77 | 26.60 | 64.24 | 61.53 |
| `head` | 90.00 | 0.280 | 90.46 | 24.99 | 87.39 | 82.42 |
| `tail` | 30.00 | 0.630 | 30.36 | 32.00 | 22.91 | 21.65 |
| `tail` | 50.00 | 0.560 | 51.18 | 30.71 | 40.24 | 37.66 |
| `tail` | 70.00 | 0.470 | 71.49 | 27.45 | 62.88 | 58.50 |
| `tail` | 90.00 | 0.340 | 90.26 | 25.37 | 85.88 | 79.79 |
| `max_head_tail` | 30.00 | 0.660 | 32.62 | 31.93 | 24.67 | 23.97 |
| `max_head_tail` | 50.00 | 0.600 | 52.10 | 30.38 | 41.41 | 39.34 |
| `max_head_tail` | 70.00 | 0.530 | 71.28 | 28.05 | 61.37 | 57.76 |
| `max_head_tail` | 90.00 | 0.400 | 90.36 | 25.76 | 84.70 | 79.56 |
| `either_side` | 30.00 | 0.846 | 30.26 | 33.52 | 21.79 | 21.03 |
| `either_side` | 50.00 | 0.792 | 50.05 | 30.75 | 39.30 | 37.81 |
| `either_side` | 70.00 | 0.719 | 70.15 | 27.65 | 61.27 | 57.92 |
| `either_side` | 90.00 | 0.574 | 90.05 | 25.70 | 84.60 | 79.56 |
| `mid_mass` | 30.00 | 0.310 | 30.67 | 20.71 | 35.76 | 37.34 |
| `mid_mass` | 50.00 | 0.220 | 51.90 | 22.50 | 55.70 | 59.10 |
| `mid_mass` | 70.00 | 0.160 | 70.15 | 23.42 | 72.34 | 75.10 |
| `mid_mass` | 90.00 | 0.080 | 91.59 | 23.97 | 92.25 | 93.23 |
| `score_entropy` | 30.00 | 1.939 | 30.05 | 25.88 | 28.03 | 30.63 |
| `score_entropy` | 50.00 | 1.799 | 50.05 | 25.99 | 46.51 | 49.29 |
| `score_entropy` | 70.00 | 1.572 | 70.05 | 24.09 | 70.21 | 72.85 |
| `score_entropy` | 90.00 | 1.238 | 90.05 | 24.11 | 90.19 | 90.90 |
| `cut_p` | 30.00 | 0.330 | 30.67 | 22.15 | 33.43 | 50.45 |
| `cut_p` | 50.00 | 0.280 | 54.67 | 24.07 | 54.83 | 67.76 |
| `cut_p` | 70.00 | 0.240 | 72.62 | 24.33 | 72.07 | 80.32 |
| `cut_p` | 90.00 | 0.190 | 90.36 | 24.78 | 88.04 | 91.77 |
| `word_count` | 30.00 | 21.000 | 30.26 | 40.75 | 17.93 | 12.56 |
| `word_count` | 50.00 | 13.000 | 55.08 | 32.97 | 40.34 | 29.53 |
| `word_count` | 70.00 | 9.000 | 71.90 | 29.78 | 58.30 | 44.56 |
| `word_count` | 90.00 | 5.000 | 91.69 | 26.25 | 84.35 | 71.48 |
| `jev_trim_t03` | 30.00 | 0.000 | 100.00 | 24.15 | 100.00 | 100.00 |
| `jev_trim_t03` | 50.00 | 0.000 | 100.00 | 24.15 | 100.00 | 100.00 |
| `jev_trim_t03` | 70.00 | 0.000 | 100.00 | 24.15 | 100.00 | 100.00 |
| `jev_trim_t03` | 90.00 | 0.000 | 100.00 | 24.15 | 100.00 | 100.00 |

### Operating points, pooled 18, all sentences

Same, with the editor's removed sentences counted as negatives.

| signal | recall target | threshold | recall | precision | share of all flagged |
|---|---:|---:|---:|---:|---:|
| `head` | 30.00 | 0.580 | 31.62 | 18.69 | 25.79 |
| `head` | 50.00 | 0.490 | 52.46 | 17.54 | 45.59 |
| `head` | 70.00 | 0.400 | 71.68 | 16.89 | 64.68 |
| `head` | 90.00 | 0.260 | 90.68 | 16.62 | 83.14 |
| `tail` | 30.00 | 0.610 | 31.69 | 20.37 | 23.72 |
| `tail` | 50.00 | 0.540 | 52.53 | 19.93 | 40.18 |
| `tail` | 70.00 | 0.450 | 71.75 | 18.36 | 59.55 |
| `tail` | 90.00 | 0.310 | 90.90 | 17.10 | 81.04 |
| `max_head_tail` | 30.00 | 0.650 | 31.55 | 19.33 | 24.87 |
| `max_head_tail` | 50.00 | 0.590 | 50.26 | 19.10 | 40.10 |
| `max_head_tail` | 70.00 | 0.510 | 70.36 | 17.91 | 59.88 |
| `max_head_tail` | 90.00 | 0.380 | 90.02 | 17.03 | 80.59 |
| `either_side` | 30.00 | 0.836 | 30.01 | 20.01 | 22.86 |
| `either_side` | 50.00 | 0.777 | 50.11 | 18.92 | 40.37 |
| `either_side` | 70.00 | 0.700 | 70.21 | 17.98 | 59.53 |
| `either_side` | 90.00 | 0.541 | 90.10 | 16.90 | 81.26 |
| `mid_mass` | 30.00 | 0.340 | 30.23 | 13.84 | 33.29 |
| `mid_mass` | 50.00 | 0.230 | 51.14 | 13.86 | 56.25 |
| `mid_mass` | 70.00 | 0.160 | 71.53 | 14.41 | 75.66 |
| `mid_mass` | 90.00 | 0.080 | 91.93 | 14.96 | 93.63 |
| `score_entropy` | 30.00 | 1.933 | 30.01 | 14.65 | 31.22 |
| `score_entropy` | 50.00 | 1.781 | 50.04 | 14.84 | 51.40 |
| `score_entropy` | 70.00 | 1.553 | 70.07 | 14.57 | 73.30 |
| `score_entropy` | 90.00 | 1.227 | 90.02 | 15.13 | 90.71 |
| `cut_p` | 30.00 | 0.330 | 30.45 | 8.83 | 52.53 |
| `cut_p` | 50.00 | 0.280 | 53.26 | 11.76 | 69.04 |
| `cut_p` | 70.00 | 0.240 | 71.46 | 13.44 | 81.05 |
| `cut_p` | 90.00 | 0.190 | 90.54 | 14.97 | 92.20 |
| `word_count` | 30.00 | 20.000 | 32.36 | 37.28 | 13.23 |
| `word_count` | 50.00 | 13.000 | 54.22 | 29.49 | 28.02 |
| `word_count` | 70.00 | 9.000 | 70.87 | 25.20 | 42.87 |
| `word_count` | 90.00 | 5.000 | 91.78 | 19.98 | 70.00 |
| `jev_trim_t03` | 30.00 | 0.000 | 100.00 | 15.24 | 100.00 |
| `jev_trim_t03` | 50.00 | 0.000 | 100.00 | 15.24 | 100.00 |
| `jev_trim_t03` | 70.00 | 0.000 | 100.00 | 15.24 | 100.00 |
| `jev_trim_t03` | 90.00 | 0.000 | 100.00 | 15.24 | 100.00 |

### Held-out transfer

Thresholds picked on the fit six (editor kept), then applied unchanged to the held-out 12. This is what a frozen flag would have done on unseen episodes.

| signal | fit recall target | threshold | held-out recall | held-out precision | held-out share of all |
|---|---:|---:|---:|---:|---:|
| `head` | 30.00 | 0.530 | 46.87 | 28.87 | 37.40 |
| `head` | 50.00 | 0.450 | 65.74 | 27.53 | 55.37 |
| `head` | 70.00 | 0.370 | 80.62 | 26.31 | 70.55 |
| `head` | 90.00 | 0.220 | 95.28 | 24.91 | 86.68 |
| `tail` | 30.00 | 0.570 | 48.21 | 30.86 | 35.23 |
| `tail` | 50.00 | 0.500 | 65.64 | 28.52 | 51.73 |
| `tail` | 70.00 | 0.400 | 83.49 | 26.22 | 71.39 |
| `tail` | 90.00 | 0.280 | 94.87 | 24.97 | 85.61 |
| `max_head_tail` | 30.00 | 0.610 | 48.82 | 30.47 | 36.94 |
| `max_head_tail` | 50.00 | 0.540 | 69.64 | 28.65 | 55.39 |
| `max_head_tail` | 70.00 | 0.450 | 85.23 | 26.52 | 72.90 |
| `max_head_tail` | 90.00 | 0.340 | 94.46 | 24.93 | 85.60 |
| `either_side` | 30.00 | 0.798 | 47.38 | 30.72 | 36.02 |
| `either_side` | 50.00 | 0.737 | 65.95 | 28.18 | 53.37 |
| `either_side` | 70.00 | 0.652 | 82.26 | 26.73 | 70.08 |
| `either_side` | 90.00 | 0.492 | 94.77 | 25.03 | 85.61 |
| `mid_mass` | 30.00 | 0.390 | 20.10 | 21.01 | 24.66 |
| `mid_mass` | 50.00 | 0.280 | 36.82 | 20.90 | 44.47 |
| `mid_mass` | 70.00 | 0.170 | 65.23 | 23.06 | 71.27 |
| `mid_mass` | 90.00 | 0.080 | 91.59 | 23.97 | 93.23 |
| `score_entropy` | 30.00 | 1.872 | 39.90 | 25.99 | 39.69 |
| `score_entropy` | 50.00 | 1.701 | 59.28 | 24.61 | 60.53 |
| `score_entropy` | 70.00 | 1.501 | 75.28 | 24.08 | 77.85 |
| `score_entropy` | 90.00 | 1.214 | 90.87 | 24.12 | 91.50 |
| `cut_p` | 30.00 | 0.320 | 33.74 | 22.04 | 53.55 |
| `cut_p` | 50.00 | 0.270 | 59.38 | 24.20 | 71.10 |
| `cut_p` | 70.00 | 0.230 | 77.03 | 24.32 | 83.48 |
| `cut_p` | 90.00 | 0.190 | 90.36 | 24.78 | 91.77 |
| `word_count` | 30.00 | 20.000 | 31.90 | 38.97 | 13.92 |
| `word_count` | 50.00 | 13.000 | 55.08 | 32.97 | 29.53 |
| `word_count` | 70.00 | 8.000 | 76.62 | 28.84 | 50.08 |
| `word_count` | 90.00 | 5.000 | 91.69 | 26.25 | 71.48 |
| `jev_trim_t03` | 30.00 | 0.000 | 100.00 | 24.15 | 100.00 |
| `jev_trim_t03` | 50.00 | 0.000 | 100.00 | 24.15 | 100.00 |
| `jev_trim_t03` | 70.00 | 0.000 | 100.00 | 24.15 | 100.00 |
| `jev_trim_t03` | 90.00 | 0.000 | 100.00 | 24.15 | 100.00 |

## Ceiling of the route

Oracle arm: start from the baseline decisions above. For every sentence the flag fires on, if the editor made it partial, replace Jev's `keep_words` with the editor's exact kept runs (a perfect smart model); if the editor kept it whole or removed it, leave Jev's decision alone (a perfect smart model declines to trim). Flags come from the pooled editor-kept thresholds in the table above. Scored at 2.50 with the modules layered on, against the reproduced baseline 80.47 SP.

Three variants. `trim kept only, modules on top` is the realistic one: the trimmer only touches sentences Jev keeps at 2.50, and the um and silence modules still run over its output, so an editor run that keeps an um or a pause loses it again. `oracle trims exempt from modules` scores the editor's runs untouched, the pure value of perfect trims. `trim and keep` also keeps a flagged partial sentence Jev had cut, which is outside route 1 (it is a keep decision, not a trim) and is shown only for scale. `perfect flag` fires on every editor-partial sentence and nothing else, which separates flag quality from trim value.

| flag | recall | threshold | variant | flagged | flagged and Jev keeps | replaced | SP | SP gain | WORD | WORD gain | GRADE | GRADE gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `head` | 70.00 | 0.400 | trim kept only, modules on top | 5784 | 3803 | 857 | 89.84 | 9.37 | 79.37 | 5.25 | 92.29 | 3.85 |
| `head` | 90.00 | 0.260 | trim kept only, modules on top | 7435 | 5030 | 1097 | 92.47 | 12.00 | 80.85 | 6.72 | 93.39 | 4.95 |
| `tail` | 70.00 | 0.450 | trim kept only, modules on top | 5326 | 3622 | 861 | 89.68 | 9.22 | 79.51 | 5.39 | 92.44 | 4.00 |
| `tail` | 90.00 | 0.310 | trim kept only, modules on top | 7247 | 4955 | 1096 | 92.39 | 11.93 | 81.12 | 7.00 | 93.55 | 5.11 |
| `max_head_tail` | 70.00 | 0.510 | trim kept only, modules on top | 5355 | 3574 | 845 | 89.70 | 9.23 | 79.67 | 5.54 | 92.54 | 4.10 |
| `max_head_tail` | 70.00 | 0.510 | trim kept only, oracle trims exempt from modules | 5355 | 3574 | 845 | 90.65 | 10.19 | 79.74 | 5.62 | 91.70 | 3.26 |
| `max_head_tail` | 70.00 | 0.510 | trim and keep, modules on top | 5355 | 3574 | 959 | 91.62 | 11.15 | 80.98 | 6.86 | 93.82 | 5.38 |
| `max_head_tail` | 90.00 | 0.380 | trim kept only, modules on top | 7207 | 4892 | 1086 | 92.34 | 11.87 | 80.85 | 6.73 | 93.42 | 4.98 |
| `max_head_tail` | 90.00 | 0.380 | trim kept only, oracle trims exempt from modules | 7207 | 4892 | 1086 | 93.50 | 13.04 | 80.94 | 6.82 | 92.41 | 3.97 |
| `max_head_tail` | 90.00 | 0.380 | trim and keep, modules on top | 7207 | 4892 | 1227 | 94.71 | 14.24 | 82.56 | 8.44 | 95.03 | 6.59 |
| `either_side` | 70.00 | 0.700 | trim kept only, modules on top | 5324 | 3531 | 843 | 89.63 | 9.17 | 79.38 | 5.26 | 92.29 | 3.85 |
| `either_side` | 70.00 | 0.700 | trim kept only, oracle trims exempt from modules | 5324 | 3531 | 843 | 90.58 | 10.11 | 79.46 | 5.33 | 91.46 | 3.02 |
| `either_side` | 70.00 | 0.700 | trim and keep, modules on top | 5324 | 3531 | 957 | 91.57 | 11.10 | 80.68 | 6.56 | 93.52 | 5.08 |
| `either_side` | 90.00 | 0.541 | trim kept only, modules on top | 7267 | 4918 | 1087 | 92.43 | 11.96 | 80.82 | 6.70 | 93.41 | 4.97 |
| `either_side` | 90.00 | 0.541 | trim kept only, oracle trims exempt from modules | 7267 | 4918 | 1087 | 93.54 | 13.08 | 80.91 | 6.78 | 92.41 | 3.97 |
| `either_side` | 90.00 | 0.541 | trim and keep, modules on top | 7267 | 4918 | 1228 | 94.80 | 14.34 | 82.50 | 8.38 | 95.01 | 6.57 |
| `mid_mass` | 70.00 | 0.160 | trim kept only, modules on top | 6766 | 4344 | 869 | 90.13 | 9.66 | 79.54 | 5.42 | 92.44 | 4.00 |
| `mid_mass` | 90.00 | 0.080 | trim kept only, modules on top | 8373 | 5426 | 1112 | 92.71 | 12.25 | 80.92 | 6.80 | 93.44 | 5.00 |
| `score_entropy` | 70.00 | 1.553 | trim kept only, modules on top | 6555 | 4176 | 834 | 89.75 | 9.29 | 79.32 | 5.20 | 92.34 | 3.90 |
| `score_entropy` | 90.00 | 1.227 | trim kept only, modules on top | 8112 | 5282 | 1083 | 92.43 | 11.97 | 80.58 | 6.46 | 93.24 | 4.80 |
| `cut_p` | 70.00 | 0.240 | trim kept only, modules on top | 7248 | 4166 | 825 | 89.56 | 9.09 | 79.27 | 5.15 | 92.18 | 3.74 |
| `cut_p` | 90.00 | 0.190 | trim kept only, modules on top | 8245 | 5148 | 1080 | 92.40 | 11.93 | 80.71 | 6.59 | 93.34 | 4.90 |
| `word_count` | 70.00 | 9.000 | trim kept only, modules on top | 3834 | 3043 | 865 | 90.22 | 9.76 | 80.87 | 6.74 | 93.38 | 4.94 |
| `word_count` | 70.00 | 9.000 | trim kept only, oracle trims exempt from modules | 3834 | 3043 | 865 | 91.33 | 10.86 | 80.95 | 6.83 | 92.33 | 3.89 |
| `word_count` | 70.00 | 9.000 | trim and keep, modules on top | 3834 | 3043 | 966 | 91.90 | 11.44 | 82.48 | 8.36 | 94.93 | 6.49 |
| `word_count` | 90.00 | 5.000 | trim kept only, modules on top | 6260 | 4600 | 1115 | 92.81 | 12.35 | 81.53 | 7.41 | 93.88 | 5.44 |
| `word_count` | 90.00 | 5.000 | trim kept only, oracle trims exempt from modules | 6260 | 4600 | 1115 | 94.07 | 13.61 | 81.64 | 7.51 | 92.76 | 4.32 |
| `word_count` | 90.00 | 5.000 | trim and keep, modules on top | 6260 | 4600 | 1251 | 95.08 | 14.62 | 83.33 | 9.20 | 95.57 | 7.13 |
| `perfect flag` | 100.00 | n/a | trim kept only, modules on top | 1363 | 1207 | 1207 | 93.75 | 13.29 | 81.74 | 7.62 | 94.01 | 5.57 |
| `perfect flag` | 100.00 | n/a | trim kept only, oracle trims exempt from modules | 1363 | 1207 | 1207 | 95.10 | 14.63 | 81.85 | 7.72 | 92.89 | 4.45 |
| `perfect flag` | 100.00 | n/a | trim and keep, modules on top | 1363 | 1207 | 1363 | 96.38 | 15.91 | 83.57 | 9.44 | 95.74 | 7.30 |

Self-check, `perfect flag`, trim kept only, modules on top: how the metric reads the 1207 replaced sentences: disjoint 52, exact 1110, model removed 6, overlap 37, subset 2.

Self-check, `perfect flag`, trim kept only, oracle trims exempt from modules: how the metric reads the 1207 replaced sentences: exact 1206, subset 1.

Self-check, `perfect flag`, trim and keep, modules on top: how the metric reads the 1363 replaced sentences: disjoint 60, exact 1255, model removed 7, overlap 39, subset 2.

## A real smart model instead of the oracle

The oracle assumes a trimmer that matches the editor word for word. For a grounded number, the stored Luna chapters arm (gpt-5.6-luna, agentic, xhigh effort, the best Luna arm on the ladder at 83.71 SP) already picked kept words for every sentence of the 18 episodes. Here each flagged sentence Jev keeps takes Luna's `keep_words` (whole when Luna kept it whole), Jev's keep decisions are untouched, and the modules run on top. `every sentence Jev keeps` sends everything, so it is the no-flag version of the route. Luna chose those trims with the whole episode in view and a far bigger budget than the batched call priced below, so treat this as optimistic for a cheap Luna step.

| flag | recall | sent (Jev keeps) | Luna trimmed | SP | SP gain | WORD | WORD gain | GRADE | GRADE gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `max_head_tail` | 70.00 | 3574 | 581 | 80.32 | -0.15 | 74.16 | 0.03 | 88.31 | -0.13 |
| `max_head_tail` | 90.00 | 4892 | 694 | 80.17 | -0.29 | 74.21 | 0.09 | 88.28 | -0.16 |
| `word_count` | 70.00 | 3043 | 622 | 80.17 | -0.30 | 74.29 | 0.16 | 88.35 | -0.09 |
| `word_count` | 90.00 | 4600 | 716 | 80.08 | -0.39 | 74.28 | 0.15 | 88.33 | -0.11 |
| `perfect flag` | 100.00 | 1207 | 325 | 80.94 | 0.48 | 75.10 | 0.98 | 89.12 | 0.68 |
| `every sentence Jev keeps` | n/a | 5840 | 745 | 80.17 | -0.29 | 74.25 | 0.13 | 88.29 | -0.15 |

Where Luna's trims landed, over the sentences sent. `exact` and `other partial branch` are editor-partial sentences Luna trimmed; `on editor-whole` and `on editor-removed` are trims the editor did not make; `missed` is an editor-partial sentence Luna kept whole. The full outcome counts are in the JSON.

| flag | recall | Luna trim exact | Luna trim, other partial branch | Luna trim on editor-whole | Luna trim on editor-removed | missed: Luna kept an editor partial whole |
|---|---:|---:|---:|---:|---:|---:|
| `max_head_tail` | 70.00 | 75 | 180 | 291 | 32 | 309 |
| `max_head_tail` | 90.00 | 83 | 219 | 351 | 38 | 411 |
| `word_count` | 70.00 | 67 | 219 | 302 | 34 | 304 |
| `word_count` | 90.00 | 80 | 236 | 360 | 38 | 423 |
| `perfect flag` | 100.00 | 86 | 236 | 0 | 0 | 471 |
| `every sentence Jev keeps` | n/a | 86 | 236 | 378 | 42 | 471 |

## Cost and latency of the smart step

Assumptions, all rough. Only flagged sentences Jev keeps at 2.50 are sent (a cut sentence needs no trim). Each goes with 3 corpus sentences of context either side, not deduplicated across neighbours in a batch, at 1.3 tokens per transcript word, plus 1000 instruction tokens per request. 10 flagged sentences per request, 8 requests in flight, 4 s per request, so latency is `ceil(requests / 8) x 4 s` per episode. Output is 40 tokens per sentence plus 300 reasoning tokens per request (low effort). Prices per million tokens: gpt-5.6-luna 0.20 in / 1.20 out from docs/jev-real/routing-notes.md (router rates 0.20 in / 1.20 out); gpt-6-luna 0.10 in / 0.50 out from skell_e_router/model_config.py pricing (0.10 in / 0.50 out). No caching assumed. This step comes on top of Jev's own run, about $0.05 and 5.6 s per episode.

| flag | recall | sent per episode, mean | max | share of sentences sent | requests, mean | input tokens, mean | input tokens, max | gpt-5.6-luna $, mean | gpt-5.6-luna $, max | gpt-6-luna $, mean | latency s, mean | latency s, max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `max_head_tail` | 70.00 | 198.6 | 636 | 39.96 | 20.2 | 41,259 | 123,019 | $0.0251 | $0.0782 | $0.0111 | 11.8 | 32.0 |
| `max_head_tail` | 90.00 | 271.8 | 805 | 54.70 | 27.6 | 56,292 | 156,423 | $0.0342 | $0.0991 | $0.0152 | 15.8 | 44.0 |
| `either_side` | 70.00 | 196.2 | 632 | 39.48 | 20.0 | 40,702 | 122,668 | $0.0248 | $0.0779 | $0.0110 | 12.0 | 32.0 |
| `either_side` | 90.00 | 273.2 | 815 | 54.99 | 27.7 | 56,523 | 158,281 | $0.0344 | $0.1003 | $0.0153 | 15.8 | 44.0 |
| `word_count` | 70.00 | 169.1 | 411 | 34.03 | 17.2 | 37,440 | 86,812 | $0.0218 | $0.0522 | $0.0097 | 10.9 | 24.0 |
| `word_count` | 90.00 | 255.6 | 662 | 51.44 | 25.9 | 54,256 | 133,283 | $0.0325 | $0.0826 | $0.0144 | 14.7 | 36.0 |
| `perfect flag` | 100.00 | 67.1 | 152 | 13.50 | 7.2 | 15,017 | 32,661 | $0.0088 | $0.0196 | $0.0039 | 5.3 | 8.0 |

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input removals colman-02.04-skeleton-demo: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\colman-02.04-skeleton-demo.json`, md5 641b458cfac9, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.4-assignment-demo: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.4-assignment-demo.json`, md5 7b7063eaa8ea, modified 2026-09-26T06:39:11+00:00
- input removals colman-03.03-muscles-crit: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\colman-03.03-muscles-crit.json`, md5 0a501b8ab857, modified 2026-09-26T06:39:11+00:00
- input removals edges-7.01-intro: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\edges-7.01-intro.json`, md5 33c18f061339, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.2-shape-demo: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.2-shape-demo.json`, md5 76718b6b963c, modified 2026-09-26T06:39:11+00:00
- input removals perspective-14e-boxes-critique: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\perspective-14e-boxes-critique.json`, md5 440b216fa1a0, modified 2026-09-26T06:39:11+00:00
- input removals perspective-13d-critique: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\perspective-13d-critique.json`, md5 86f3cb9d37cf, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.5-crit1: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.5-crit1.json`, md5 5355b43bb6b4, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.5-crit2: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.5-crit2.json`, md5 6882c353a54b, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.5-crit3: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.5-crit3.json`, md5 e6b6f2639de8, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.5-crit4: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.5-crit4.json`, md5 829303961793, modified 2026-09-26T06:39:11+00:00
- input removals hampton-5.5-crit5: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-5.5-crit5.json`, md5 65e8fc7dfb11, modified 2026-09-26T06:39:11+00:00
- input removals flanders-03-thematic-crit: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\flanders-03-thematic-crit.json`, md5 e7c222246453, modified 2026-09-26T06:39:11+00:00
- input removals anatomy-30b-hamstring-crit: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\anatomy-30b-hamstring-crit.json`, md5 8507ead646c9, modified 2026-09-26T06:39:11+00:00
- input removals colman-04.03-life-crit: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\colman-04.03-life-crit.json`, md5 6b741a5c0530, modified 2026-09-26T06:39:11+00:00
- input removals colman-05.02-master-studies-crit: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\colman-05.02-master-studies-crit.json`, md5 2935f536eb21, modified 2026-09-26T06:39:11+00:00
- input removals colman-06.06-species-crit: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\colman-06.06-species-crit.json`, md5 719c0c13fbc2, modified 2026-09-26T06:39:11+00:00
- input removals hampton-7-conclusion: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\removals\hampton-7-conclusion.json`, md5 77c4acbfb7ef, modified 2026-09-26T06:39:11+00:00
- input luna chapters colman-02.04-skeleton-demo: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0204-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 f1edfe25dd96, modified 2026-09-11T21:45:37+00:00
- input luna chapters hampton-5.4-assignment-demo: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton54-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 7ff2280aa54e, modified 2026-09-11T21:45:40+00:00
- input luna chapters colman-03.03-muscles-crit: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0303-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 1e5b54282916, modified 2026-09-11T21:45:38+00:00
- input luna chapters edges-7.01-intro: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-edges701-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 ccf54db821a3, modified 2026-09-11T21:45:39+00:00
- input luna chapters hampton-5.2-shape-demo: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton52-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 f6a402f077d0, modified 2026-09-11T21:45:40+00:00
- input luna chapters perspective-14e-boxes-critique: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-14e-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 691647bac740, modified 2026-09-11T21:45:36+00:00
- input luna chapters perspective-13d-critique: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-13d-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 3a5d76e1fc4e, modified 2026-09-11T21:45:36+00:00
- input luna chapters hampton-5.5-crit1: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit1-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 54c7ae96eeec, modified 2026-09-11T21:45:40+00:00
- input luna chapters hampton-5.5-crit2: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit2-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 2e0c9a78c836, modified 2026-09-11T21:45:40+00:00
- input luna chapters hampton-5.5-crit3: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit3-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 09f1c710b42e, modified 2026-09-11T21:45:40+00:00
- input luna chapters hampton-5.5-crit4: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit4-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 def50ff50e65, modified 2026-09-11T21:45:41+00:00
- input luna chapters hampton-5.5-crit5: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit5-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 ffde8bfecc7f, modified 2026-09-11T21:45:41+00:00
- input luna chapters flanders-03-thematic-crit: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-flanders03-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 a87662839c73, modified 2026-09-11T21:45:40+00:00
- input luna chapters anatomy-30b-hamstring-crit: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-anatomy30b-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 0ca8d9348753, modified 2026-09-11T21:45:37+00:00
- input luna chapters colman-04.03-life-crit: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0403-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 fd9b959669fd, modified 2026-09-11T21:45:38+00:00
- input luna chapters colman-05.02-master-studies-crit: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0502-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 46b2f9304b06, modified 2026-09-11T21:45:38+00:00
- input luna chapters colman-06.06-species-crit: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0606-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 706b124d09e1, modified 2026-09-11T21:45:38+00:00
- input luna chapters hampton-7-conclusion: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton7-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 b632239066e8, modified 2026-09-11T21:45:41+00:00
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-route1-flags.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-route1-flags.json`
