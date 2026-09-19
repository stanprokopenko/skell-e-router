# Jev vs Luna on the rough-cut sentence rating

Generated 2026-09-19T19:20:38.566642+00:00 by `scripts/jev_real/roughcut_bench.py`.

## What was run

Episodes: colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo.

`luna` is one full-context call per episode against gpt-5.6-luna (reasoning_effort=medium, max_tokens=32000), with `prompts/roughcut_system_v1.md` as the system prompt and the production `id = text` user message. Missing ids get one targeted retry; anything still missing is filled with score 3.

`jev_full` sends state `{rules, transcript}` with the whole episode and asks 25 sentences' worth of questions per request (a score and a retake noul each), so the transcript is re-sent about n/25 times. `jev_window` asks 10 sentences per request over a +/-15 sentence excerpt. Jev requests run at concurrency 4. The `_retake` variants reuse the same answers and force the score to 0 when the retake noul is >= 0.6.

Scoring calls the benchmark harness itself (`roughcut_bench.calibrate` + `roughcut_bench.replay.score_sentences`), pooled calibration across the scored episodes, read at the harness's Neutral level 4.

## Metrics

### luna

| arm | episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match | keep/cut acc |
|---|---|---|---|---|---|---|---|
| luna | colman-02.04-skeleton-demo | 0.121 | 0.621 | 0.681 | 0.239 | 0.630 | 0.835 |
| luna | hampton-5.4-assignment-demo | 0.233 | 0.733 | 0.809 | 0.452 | 0.724 | 0.867 |
| luna | colman-03.03-muscles-crit | 0.180 | 0.680 | 0.650 | 0.213 | 0.647 | 0.782 |
| luna | edges-7.01-intro | 0.348 | 0.848 | 0.846 | 0.689 | 0.839 | 0.856 |
| luna | hampton-5.2-shape-demo | 0.300 | 0.800 | 0.842 | 0.558 | 0.769 | 0.893 |
| **luna** | **weighted** | **0.255** | -- | **0.770** | **0.449** | **0.730** | **0.851** |

Calibrated Neutral threshold: 2.1, kept ratio vs the editor: 1.025.

### jev_full

| arm | episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match | keep/cut acc |
|---|---|---|---|---|---|---|---|
| jev_full | colman-02.04-skeleton-demo | 0.115 | 0.615 | 0.691 | 0.269 | 0.648 | 0.799 |
| jev_full | hampton-5.4-assignment-demo | 0.209 | 0.709 | 0.800 | 0.439 | 0.725 | 0.810 |
| jev_full | colman-03.03-muscles-crit | 0.144 | 0.644 | 0.685 | 0.240 | 0.660 | 0.723 |
| jev_full | edges-7.01-intro | 0.328 | 0.828 | 0.824 | 0.648 | 0.817 | 0.833 |
| jev_full | hampton-5.2-shape-demo | 0.220 | 0.720 | 0.788 | 0.441 | 0.726 | 0.800 |
| **jev_full** | **weighted** | **0.217** | -- | **0.759** | **0.417** | **0.720** | **0.795** |

Calibrated Neutral threshold: 2.4, kept ratio vs the editor: 0.948.

### jev_full_retake

| arm | episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match | keep/cut acc |
|---|---|---|---|---|---|---|---|
| jev_full_retake | colman-02.04-skeleton-demo | 0.115 | 0.615 | 0.692 | 0.263 | 0.646 | 0.789 |
| jev_full_retake | hampton-5.4-assignment-demo | 0.213 | 0.713 | 0.819 | 0.471 | 0.741 | 0.813 |
| jev_full_retake | colman-03.03-muscles-crit | 0.144 | 0.644 | 0.685 | 0.240 | 0.660 | 0.723 |
| jev_full_retake | edges-7.01-intro | 0.323 | 0.823 | 0.811 | 0.635 | 0.806 | 0.823 |
| jev_full_retake | hampton-5.2-shape-demo | 0.222 | 0.722 | 0.808 | 0.469 | 0.740 | 0.803 |
| **jev_full_retake** | **weighted** | **0.217** | -- | **0.766** | **0.426** | **0.723** | **0.793** |

Calibrated Neutral threshold: 2.4, kept ratio vs the editor: 0.930.

### jev_window

| arm | episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match | keep/cut acc |
|---|---|---|---|---|---|---|---|
| jev_window | colman-02.04-skeleton-demo | 0.115 | 0.615 | 0.679 | 0.246 | 0.639 | 0.804 |
| jev_window | hampton-5.4-assignment-demo | 0.149 | 0.649 | 0.708 | 0.305 | 0.663 | 0.753 |
| jev_window | colman-03.03-muscles-crit | 0.150 | 0.650 | 0.705 | 0.273 | 0.675 | 0.723 |
| jev_window | edges-7.01-intro | 0.315 | 0.815 | 0.804 | 0.613 | 0.800 | 0.823 |
| jev_window | hampton-5.2-shape-demo | 0.164 | 0.664 | 0.676 | 0.258 | 0.641 | 0.730 |
| **jev_window** | **weighted** | **0.189** | -- | **0.713** | **0.342** | **0.685** | **0.765** |

Calibrated Neutral threshold: 2.2, kept ratio vs the editor: 0.954.

### jev_window_retake

| arm | episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match | keep/cut acc |
|---|---|---|---|---|---|---|---|
| jev_window_retake | colman-02.04-skeleton-demo | 0.115 | 0.615 | 0.677 | 0.239 | 0.637 | 0.799 |
| jev_window_retake | hampton-5.4-assignment-demo | 0.146 | 0.646 | 0.704 | 0.297 | 0.659 | 0.750 |
| jev_window_retake | colman-03.03-muscles-crit | 0.147 | 0.647 | 0.705 | 0.273 | 0.675 | 0.719 |
| jev_window_retake | edges-7.01-intro | 0.315 | 0.815 | 0.802 | 0.624 | 0.801 | 0.817 |
| jev_window_retake | hampton-5.2-shape-demo | 0.166 | 0.666 | 0.697 | 0.289 | 0.655 | 0.732 |
| **jev_window_retake** | **weighted** | **0.189** | -- | **0.717** | **0.350** | **0.688** | **0.762** |

Calibrated Neutral threshold: 2.2, kept ratio vs the editor: 0.940.

## Archived reference

`2026-09-06-all18-partial-single-gpt-5.6-luna.json` (single_full_context_partial). partial-capable arm: it emits sub-sentence keeps, so it escapes the flat 0.5 no-partial penalty that every whole-sentence arm here pays.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match |
|---|---|---|---|---|---|
| colman-02.04-skeleton-demo | 0.584 | 0.584 | 0.637 | 0.158 | 0.590 |
| hampton-5.4-assignment-demo | 0.610 | 0.610 | 0.584 | 0.015 | 0.547 |
| colman-03.03-muscles-crit | 0.259 | 0.759 | 0.600 | 0.067 | 0.593 |
| edges-7.01-intro | 0.163 | 0.663 | 0.803 | 0.590 | 0.793 |
| hampton-5.2-shape-demo | 0.091 | 0.591 | 0.584 | 0.073 | 0.560 |

## Cost and latency

| arm | requests | input tokens | cost USD | median s | p95 s | wall-clock s |
|---|---|---|---|---|---|---|
| luna | 5 | 26404 | 0.0308 | 20.8 | 25.6 | 116.0 |
| jev_full | 68 | 1117080 | 0.0469 | 0.4 | 0.5 | 11.0 |
| jev_window | 166 | 701982 | 0.0295 | 0.3 | 0.4 | 11.4 |

Total recorded spend: $0.1072.

| arm | episode | wall-clock s | requests | cost USD |
|---|---|---|---|---|
| luna | colman-02.04-skeleton-demo | 24.3 | 1 | 0.0048 |
| jev_full | colman-02.04-skeleton-demo | 4.4 | 8 | 0.0044 |
| jev_window | colman-02.04-skeleton-demo | 1.6 | 20 | 0.0036 |
| luna | hampton-5.4-assignment-demo | 23.0 | 1 | 0.0055 |
| luna | colman-03.03-muscles-crit | 19.1 | 1 | 0.0058 |
| luna | edges-7.01-intro | 24.1 | 1 | 0.0071 |
| luna | hampton-5.2-shape-demo | 25.6 | 1 | 0.0076 |
| jev_full | hampton-5.4-assignment-demo | 1.3 | 12 | 0.0078 |
| jev_full | colman-03.03-muscles-crit | 1.5 | 14 | 0.0090 |
| jev_full | edges-7.01-intro | 1.8 | 17 | 0.0117 |
| jev_full | hampton-5.2-shape-demo | 2.0 | 17 | 0.0139 |
| jev_window | hampton-5.4-assignment-demo | 2.0 | 31 | 0.0054 |
| jev_window | colman-03.03-muscles-crit | 2.2 | 32 | 0.0057 |
| jev_window | edges-7.01-intro | 2.7 | 40 | 0.0070 |
| jev_window | hampton-5.2-shape-demo | 2.9 | 43 | 0.0077 |

## Jev accuracy by confidence quartile

| arm | quartile | n | confidence range | keep/cut accuracy |
|---|---|---|---|---|
| jev_full | Q1 | 399 | 0.000-0.430 | 0.722 |
| jev_full | Q2 | 399 | 0.430-0.550 | 0.799 |
| jev_full | Q3 | 399 | 0.550-0.640 | 0.827 |
| jev_full | Q4 | 400 | 0.640-0.990 | 0.833 |
| jev_window | Q1 | 399 | 0.000-0.430 | 0.692 |
| jev_window | Q2 | 399 | 0.430-0.580 | 0.784 |
| jev_window | Q3 | 399 | 0.580-0.710 | 0.794 |
| jev_window | Q4 | 400 | 0.710-1.000 | 0.787 |

## Retake noul vs the editor's cuts

| arm | n | mean noul (human cut) | mean noul (human kept) | r vs human cut | r vs corpus is_retake | fired >=0.6 | precision | recall |
|---|---|---|---|---|---|---|---|---|
| jev_full | 1597 | 0.542 | 0.271 | 0.570 | 0.551 | 277 | 0.827 | 0.459 |
| jev_full_retake | 1597 | 0.542 | 0.271 | 0.570 | 0.551 | 277 | 0.827 | 0.459 |
| jev_window | 1597 | 0.547 | 0.279 | 0.530 | 0.577 | 301 | 0.764 | 0.461 |
| jev_window_retake | 1597 | 0.547 | 0.279 | 0.530 | 0.577 | 301 | 0.764 | 0.461 |

## Where each arm disagrees with the editor

| arm | both keep | kept, editor cut | cut, editor kept | both cut | keep precision | keep recall |
|---|---|---|---|---|---|---|
| luna | 898 | 38 | 200 | 461 | 0.959 | 0.818 |
| jev_full | 801 | 30 | 297 | 469 | 0.964 | 0.730 |
| jev_full_retake | 793 | 26 | 305 | 473 | 0.968 | 0.722 |
| jev_window | 783 | 61 | 315 | 438 | 0.928 | 0.713 |
| jev_window_retake | 776 | 58 | 322 | 441 | 0.930 | 0.707 |

## Keep/cut accuracy by position in the episode

| arm | first fifth | 2nd | 3rd | 4th | last fifth |
|---|---|---|---|---|---|
| luna | 0.847 | 0.834 | 0.859 | 0.831 | 0.883 |
| jev_full | 0.794 | 0.750 | 0.796 | 0.784 | 0.852 |
| jev_full_retake | 0.791 | 0.738 | 0.793 | 0.791 | 0.852 |
| jev_window | 0.785 | 0.738 | 0.793 | 0.728 | 0.779 |
| jev_window_retake | 0.779 | 0.734 | 0.784 | 0.734 | 0.779 |

## Keep/cut accuracy on the corpus's retake sentences

| arm | retake n | retake accuracy | other n | other accuracy |
|---|---|---|---|---|
| luna | 206 | 0.796 | 1391 | 0.859 |
| jev_full | 206 | 0.796 | 1391 | 0.795 |
| jev_full_retake | 206 | 0.796 | 1391 | 0.792 |
| jev_window | 206 | 0.796 | 1391 | 0.760 |
| jev_window_retake | 206 | 0.796 | 1391 | 0.757 |

## Luna categories vs the editor

| category | n | model kept | human kept | keep/cut accuracy |
|---|---|---|---|---|
| keep | 1009 | 933 | 927 | 0.937 |
| false_start | 197 | 0 | 83 | 0.579 |
| filler | 142 | 0 | 28 | 0.803 |
| off_topic | 110 | 0 | 3 | 0.973 |
| repeated_take | 86 | 2 | 31 | 0.616 |
| rambling | 30 | 1 | 23 | 0.200 |
| tangent | 23 | 0 | 3 | 0.870 |

## Caveats

- SENTENCE POINTS punishes every whole-sentence arm here with the harness's flat 0.5 no-partial penalty on any episode where the editor trimmed inside sentences, because none of these arms can emit sub-sentence keeps. The penalty is identical across arms, so the ranking holds; the `SP raw` column is the same score before the penalty.
- The corpus already carries `is_retake` from solar-sailer's own retakes pass, and `ranges.kept_segments_from_score` cuts those sentences for EVERY arm regardless of model score. That deterministic layer sits underneath all five arms and dampens the measurable effect of Jev's retake noul, especially on edges-7.01-intro (169 of 389 sentences are flagged).
- The keep threshold is calibrated per arm by the harness (the threshold that maximises the retired GRADE, pooled across the scored episodes), so the arms are compared at each one's own best operating point rather than at a fixed cut-off.
- The archived Luna run 2026-07-15-13d-v1-single-gpt-5.6-luna.json records model/prompt/max_tokens but not reasoning effort, so the Luna arm here uses medium.
- Costs are token-based estimates from the recorded usage, not an invoice. Failed and retried requests may add unreported provider charges.
- Read-only against solar-sailer: the answer-key loader is monkeypatched to read the committed cache rather than re-extract and rewrite it.
- This benchmark could not run until a router bug was fixed. `skell_e_router.classification._parse` required a score answer's probabilities to sum to 1.000 +/- 0.001 and to reproduce the reported score within 0.001 per level. Jev rounds both to two decimals, so a six-level score legitimately sums to 0.99 and drifts up to 0.04 from its score. About a third of live requests were rejected as malformed. Both tolerances now derive from that rounding.
- 6 of 239 requests came back as a bare PROVIDER_ERROR with no HTTP status and did not reproduce on a re-request. Each was re-issued once; all succeeded, and both attempts are in roughcut-requests.jsonl (the repaired rows carry `repair: true`). Cause unknown, roughly a 2.5% sporadic failure rate.
