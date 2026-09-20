# Jev rough cut: the 13 held-out episodes

Developer-facing. The first run of the Jev rough cut on episodes that no prompt or threshold was ever tuned on.

## What was frozen before these runs

Everything the pipeline decides with was fixed on the six fit episodes named in `docs/superpowers/specs/2026-09-20-jev-roughcut-design.md` (colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique) and not touched afterwards:

- the prompt text in `scripts/jev_real/roughcut_jev_prompts.py`, versions `v1` and `v3`, byte-identical to what produced `docs/jev-real/roughcut-jev-v1-v2-v3.md`;
- the trim trigger `t_trim` 0.3, the retake veto at `real_k` 0.5, and concurrency 8;
- the keep threshold, which is not a tuned constant at all: every arm is read at the threshold the pooled calibration sweep picks for it, the same sweep every published ladder arm gets.

The 13 held-out episodes were run once each at `v3` and once at `v1`, in that order, on 2026-09-20. No prompt, threshold or code change happened between the fit runs and these.

Twelve of the 13 are on the published 18-episode ladder. `greco-2.2-thumbnailing` is not: it has no entry in `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`, so it has no Luna or Opus reference and it is excluded from every ladder comparison. It is still in the 13-episode pool, because the pool is the held-out set, not the ladder.

Every number below is the `jev_a` arm at `t_trim` 0.3 with um removal and delete silence layered on, the column the ladder compares on. Luna is `luna-chapters-rules5` and Opus is `opus5-cc-agentic`, both read from the reference JSON's `umm_silence` layer.

## Per episode

SENTENCE POINTS:

| episode | sentences | v3 | v1 | Luna | Opus |
|---|---:|---:|---:|---:|---:|
| perspective-13d-critique | 1,752 | 78.09 | 81.43 | 88.61 | 91.48 |
| hampton-5.5-crit1 | 181 | 87.35 | 86.46 | 84.36 | 82.32 |
| hampton-5.5-crit2 | 127 | 79.45 | 78.90 | 82.52 | 83.86 |
| hampton-5.5-crit3 | 137 | 81.17 | 75.04 | 76.50 | 81.17 |
| hampton-5.5-crit4 | 156 | 88.46 | 82.76 | 81.28 | 81.41 |
| hampton-5.5-crit5 | 295 | 90.88 | 90.95 | 86.88 | 85.90 |
| flanders-03-thematic-crit | 1,309 | 77.14 | 75.38 | 77.01 | 78.05 |
| anatomy-30b-hamstring-crit | 951 | 84.90 | 79.86 | 85.87 | 91.42 |
| colman-04.03-life-crit | 495 | 76.18 | 74.83 | 73.72 | 77.37 |
| colman-05.02-master-studies-crit | 381 | 66.33 | 65.96 | 73.41 | 77.45 |
| colman-06.06-species-crit | 373 | 79.57 | 77.08 | 79.89 | 86.38 |
| hampton-7-conclusion | 43 | 72.33 | 81.63 | 89.07 | 87.21 |
| greco-2.2-thumbnailing (not on the ladder) | 1,388 | 65.09 | 68.40 | n/a | n/a |
| **pooled, all 13** | **7,588** | **76.74** | **76.75** | n/a | n/a |
| **pooled, the 12 on the ladder** | **6,200** | **79.35** | **78.61** | **82.31** | **85.14** |

WORD SCORE:

| episode | v3 | v1 | Luna | Opus |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 79.05 | 81.41 | 88.33 | 91.02 |
| hampton-5.5-crit1 | 67.36 | 72.31 | 74.80 | 72.42 |
| hampton-5.5-crit2 | 59.31 | 58.85 | 79.28 | 71.58 |
| hampton-5.5-crit3 | 68.98 | 71.30 | 78.83 | 75.05 |
| hampton-5.5-crit4 | 64.13 | 58.79 | 74.97 | 78.56 |
| hampton-5.5-crit5 | 67.20 | 73.29 | 73.92 | 71.70 |
| flanders-03-thematic-crit | 77.61 | 77.05 | 80.72 | 81.36 |
| anatomy-30b-hamstring-crit | 78.18 | 76.88 | 85.99 | 89.07 |
| colman-04.03-life-crit | 65.37 | 64.08 | 67.67 | 67.10 |
| colman-05.02-master-studies-crit | 65.98 | 65.05 | 73.90 | 73.73 |
| colman-06.06-species-crit | 70.43 | 69.97 | 70.87 | 74.00 |
| hampton-7-conclusion | 72.07 | 80.70 | 84.79 | 84.70 |
| greco-2.2-thumbnailing (not on the ladder) | 68.55 | 68.64 | n/a | n/a |
| **pooled, all 13** | **72.14** | **72.58** | n/a | n/a |
| **pooled, the 12 on the ladder** | **73.05** | **73.58** | **79.78** | **80.63** |

GRADE:

| episode | v3 | v1 | Luna | Opus |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 83.28 | 85.27 | 91.97 | 94.53 |
| hampton-5.5-crit1 | 111.46 | 111.35 | 113.48 | 112.46 |
| hampton-5.5-crit2 | 102.20 | 101.93 | 114.91 | 109.65 |
| hampton-5.5-crit3 | 105.91 | 104.35 | 110.75 | 108.23 |
| hampton-5.5-crit4 | 104.93 | 98.48 | 111.13 | 114.49 |
| hampton-5.5-crit5 | 103.00 | 105.89 | 105.77 | 105.83 |
| flanders-03-thematic-crit | 83.24 | 82.38 | 86.38 | 87.30 |
| anatomy-30b-hamstring-crit | 89.85 | 87.19 | 95.50 | 98.26 |
| colman-04.03-life-crit | 77.14 | 76.27 | 79.55 | 79.25 |
| colman-05.02-master-studies-crit | 88.21 | 86.84 | 94.87 | 95.11 |
| colman-06.06-species-crit | 97.21 | 96.03 | 96.93 | 100.57 |
| hampton-7-conclusion | 85.16 | 92.63 | 96.80 | 96.04 |
| greco-2.2-thumbnailing (not on the ladder) | 71.80 | 71.92 | n/a | n/a |
| **pooled, all 13** | **86.08** | **85.81** | n/a | n/a |
| **pooled, the 12 on the ladder** | **89.18** | **88.83** | **94.52** | **95.86** |

Human kept share is the editor's kept frames over the episode's dialogue frames: how much of the raw talk survived the real edit. Seconds are wall clock per episode at concurrency 8, both passes, repair requests excluded.

| episode | human kept share | s, v3 | s, v1 |
|---|---:|---:|---:|
| perspective-13d-critique | 52.36 | 13.61 | 10.57 |
| hampton-5.5-crit1 | 92.98 | 2.22 | 2.12 |
| hampton-5.5-crit2 | 93.50 | 0.93 | 0.97 |
| hampton-5.5-crit3 | 89.99 | 0.99 | 1.85 |
| hampton-5.5-crit4 | 92.47 | 0.79 | 0.75 |
| hampton-5.5-crit5 | 92.99 | 4.81 | 2.43 |
| flanders-03-thematic-crit | 79.39 | 19.26 | 17.25 |
| anatomy-30b-hamstring-crit | 76.21 | 11.76 | 8.39 |
| colman-04.03-life-crit | 73.20 | 5.88 | 3.26 |
| colman-05.02-master-studies-crit | 73.09 | 5.01 | 5.96 |
| colman-06.06-species-crit | 73.31 | 2.83 | 4.74 |
| hampton-7-conclusion | 73.15 | 1.81 | 1.94 |
| greco-2.2-thumbnailing | 36.38 | 13.02 | 9.21 |
| **pooled / mean** | **66.47** | **6.38** | **5.34** |

Held-out is not a softer set than fit by this measure: the editor kept 66.47% of the held-out frames against 64.14% on the fit six. It is a more spread-out set. The two extremes are here, and they are both held-out: `hampton-5.5-crit2` at 93.50 (the editor kept almost everything) and `greco-2.2-thumbnailing` at 36.38 (the editor threw away nearly two thirds).

## Pooled over the 13 held-out episodes

Weighted the way the report pools: SENTENCE POINTS by corpus sentence count, WORD SCORE by word count, GRADE, frame match and kept ratio by dialogue frames. Each arm at its own calibrated threshold, layered.

| arm | threshold | SENTENCE POINTS | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 jev_a | 2.50 | 76.74 | 72.14 | 86.08 | 74.27 | 103.53 | 6.38 |
| v1 jev_a | 2.30 | 76.75 | 72.58 | 85.81 | 74.10 | 98.37 | 5.34 |
| v3 jev_mix | 2.70 | 76.40 | 71.65 | 85.74 | 73.95 | 105.68 | 6.38 |
| v3 jev_noul | 3.10 | 75.43 | 70.50 | 84.31 | 72.73 | 99.78 | 6.38 |

Luna and Opus cannot be pooled over 13; `greco-2.2-thumbnailing` is not in the reference file. Over the 12 that are, at the same 12-episode pooling, Luna is 82.31 SENTENCE POINTS / 79.78 WORD / 94.52 GRADE and Opus is 85.14 / 80.63 / 95.86, against v3 jev_a's 79.35 / 73.05 / 89.18.

`jev_mix` and `jev_noul` are the two v3-only arms the report rebuilds offline from the `cut_k` answer; they cost nothing extra to report. On the fit set `jev_mix` had the best pooled GRADE of any arm. On held-out it is behind plain `jev_a` on all three metrics. That reverses the fit-set finding, so the blend is not worth carrying forward.

## Where the Jev arms land on the published 18-episode ladder

To place a Jev arm on the ladder it has to be scored on the ladder's own 18 episodes, calibrated once over those 18, the way every published arm is. So the six fit episodes and the 12 on-ladder held-out episodes were merged into one run and re-reported. The merged inputs are `roughcut-jev-all18-v3-*` and `roughcut-jev-all18-v1-*` in `docs/jev-real/`; they are concatenations of the two runs' own decision, request and timing files with `greco-2.2-thumbnailing` dropped, not new model calls. Two checks say the merge is sound: the report's Luna and Opus rows over these 18 come back at 83.71 and 86.47 SENTENCE POINTS, matching the published ladder to the cent, and pooling the v3 per-episode values by hand at each run's own threshold reproduces the merged run's 80.47 / 74.12 / 88.44 exactly (both halves calibrated to 2.50, so there is nothing to reconcile).

| arm, 18 episodes | threshold | SENTENCE POINTS | WORD SCORE | GRADE | kept ratio |
|---|---:|---:|---:|---:|---:|
| v3 jev_a | 2.50 | 80.47 | 74.12 | 88.44 | 95.16 |
| v1 jev_a | 1.90 | 79.41 | 74.18 | 88.52 | 100.23 |

One caveat on v1: calibrating over all 18 at once moves its threshold to 1.90, well below the 2.40 the fit six picked and the 2.30 the held-out 13 picked. Read instead at each half's own threshold, v1 pools to 80.22 / 74.41 / 88.10. Either reading puts v1 within about a point of v3, which is inside the noise band the v1-v2-v3 write-up established (about 1 pooled point).

The published ladder's SENTENCE POINTS column with um removal and delete silence layered on, top ten, with the Jev rows inserted where they fall. The list is extended past ten because neither Jev arm reaches the top ten.

| rank | arm | SENTENCE POINTS + modules |
|---:|---|---:|
| 1 | claude-opus-5-high, agentic, Claude Code, rules1 (shipped) | 86.47 |
| 2 | claude-fable-5-1-high, chapters, Claude Code, rules5 | 86.45 |
| 3 | claude-opus-5-high, agentic, API, rules1 | 86.30 |
| 4 | gpt-6-astra-high, chapters, Codex CLI, rules5 | 85.94 |
| 5 | gpt-5.6-sol-high, chapters, Codex CLI, rules5 | 84.61 |
| 6 | gpt-5.6-sol-high, agentic, API, rules1 | 84.00 |
| 7 | gpt-5.6-luna-xhigh, agentic, API, rules5 | 83.84 |
| 8 | gpt-5.6-luna-xhigh, chapters, API, rules5 | 83.71 |
| 9 | gpt-5.6-luna-xhigh, chapters, API, rules6 | 83.48 |
| 10 | gpt-5.6-luna-xhigh, chapters, API, rules5 + kept-parts tool | 83.18 |
| 11 | gpt-5.6-luna-xhigh, chapters, API, rules5, topic-neutral | 82.82 |
| 12 | gpt-5.6-luna-xhigh, agentic, API, rules5, no review pass | 82.36 |
| 13 | gpt-5.6-luna-xhigh, agentic, API, rules1 | 81.96 |
| 14 | gpt-5.6-sol-low, agentic, API, rules1 | 81.88 |
| 15 | gpt-5.6-luna-high, agentic, API, rules1, default harness prompt | 81.32 |
| 16 | gpt-5.6-luna-medium, agentic, API, rules1 | 80.57 |
| **17** | **v3 jev_a, t_trim 0.3** | **80.47** |
| 18 | gpt-5.6-terra-high, agentic, API, rules1 | 80.20 |
| 19 | gpt-5.6-terra-low, agentic, API, rules1 | 80.15 |
| **20** | **v1 jev_a, t_trim 0.3** | **79.41** |
| 21 | gpt-5.6-luna-medium, agentic, API, rules1, default harness prompt | 79.08 |
| 22 | gpt-5.6-luna-medium, agentic, Codex CLI, rules1 | 74.01 |
| 23 | gpt-5.6-luna-medium, single call, API, rules1 | 65.36 |
| 24 | deterministic baseline (um removal + retakes + delete silence) | 63.72 |

v3 `jev_a` lands 17th of 24, a tenth of a point under `gpt-5.6-luna-medium` running the full agentic workflow, 16.75 points above the deterministic baseline and 6.00 under the shipped Opus arm. Every arm above it is a multi-turn reasoning model; every arm below it is either a weaker model on the same agentic workflow or not a real workflow at all. That is the honest placement of a two-request-per-25-sentences classifier that runs in six seconds an episode.

## Latency, requests and cost per episode

Wall clock is measured around each pass at concurrency 8, retries and backoff inside it. Repair requests are outside these wall clocks; their cost and request count are in the totals. Window fallbacks are requests that sent a 200-sentence window either side of the target block instead of the whole transcript.

v3:

| episode | s | retake s | sentence s | requests | retries | errors | failed blocks | window fallbacks | cost $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 13.61 | 0.61 | 13.01 | 114 | 24 | 26 | 2 | 97 | 0.1285 |
| hampton-5.5-crit1 | 2.22 | 0.22 | 2.00 | 10 | 1 | 1 | 0 | 0 | 0.0152 |
| hampton-5.5-crit2 | 0.93 | 0.20 | 0.73 | 7 | 0 | 0 | 0 | 0 | 0.0102 |
| hampton-5.5-crit3 | 0.99 | 0.22 | 0.77 | 7 | 0 | 0 | 0 | 0 | 0.0112 |
| hampton-5.5-crit4 | 0.79 | 0.00 | 0.79 | 7 | 0 | 0 | 0 | 0 | 0.0133 |
| hampton-5.5-crit5 | 4.81 | 0.21 | 4.60 | 15 | 2 | 2 | 0 | 0 | 0.0245 |
| flanders-03-thematic-crit | 19.26 | 0.27 | 18.99 | 97 | 38 | 38 | 0 | 25 | 0.1246 |
| anatomy-30b-hamstring-crit | 11.76 | 0.37 | 11.40 | 58 | 10 | 10 | 0 | 0 | 0.0913 |
| colman-04.03-life-crit | 5.88 | 0.42 | 5.46 | 28 | 4 | 5 | 1 | 0 | 0.0443 |
| colman-05.02-master-studies-crit | 5.01 | 0.26 | 4.75 | 21 | 4 | 4 | 0 | 0 | 0.0329 |
| colman-06.06-species-crit | 2.83 | 0.21 | 2.62 | 18 | 2 | 2 | 0 | 0 | 0.0307 |
| hampton-7-conclusion | 1.81 | 0.18 | 1.63 | 4 | 1 | 1 | 0 | 0 | 0.0030 |
| greco-2.2-thumbnailing | 13.02 | 0.43 | 12.59 | 81 | 15 | 15 | 0 | 71 | 0.1156 |
| **mean** | **6.38** | | | | | | | | **0.0496** |
| **max** | **19.26 (flanders-03-thematic-crit)** | | | | | | | | |
| **total** | **82.92** | | | **467** | **101** | **104** | **3** | **193** | **0.6453** |

v1:

| episode | s | retake s | sentence s | requests | retries | errors | failed blocks | window fallbacks | cost $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 10.57 | 0.61 | 9.96 | 111 | 21 | 23 | 2 | 94 | 0.1043 |
| hampton-5.5-crit1 | 2.12 | 0.32 | 1.81 | 12 | 3 | 3 | 0 | 0 | 0.0127 |
| hampton-5.5-crit2 | 0.97 | 0.19 | 0.79 | 7 | 0 | 0 | 0 | 0 | 0.0085 |
| hampton-5.5-crit3 | 1.85 | 0.22 | 1.63 | 8 | 1 | 1 | 0 | 0 | 0.0094 |
| hampton-5.5-crit4 | 0.75 | 0.00 | 0.75 | 7 | 0 | 0 | 0 | 0 | 0.0112 |
| hampton-5.5-crit5 | 2.43 | 0.17 | 2.25 | 17 | 4 | 4 | 0 | 0 | 0.0204 |
| flanders-03-thematic-crit | 17.25 | 0.30 | 16.96 | 93 | 34 | 34 | 0 | 24 | 0.1066 |
| anatomy-30b-hamstring-crit | 8.39 | 0.45 | 7.93 | 56 | 8 | 8 | 0 | 0 | 0.0782 |
| colman-04.03-life-crit | 3.26 | 0.23 | 3.03 | 27 | 4 | 4 | 0 | 0 | 0.0375 |
| colman-05.02-master-studies-crit | 5.96 | 0.28 | 5.68 | 24 | 6 | 7 | 1 | 0 | 0.0276 |
| colman-06.06-species-crit | 4.74 | 0.26 | 4.48 | 18 | 2 | 2 | 0 | 0 | 0.0256 |
| hampton-7-conclusion | 1.94 | 0.23 | 1.71 | 4 | 1 | 1 | 0 | 0 | 0.0024 |
| greco-2.2-thumbnailing | 9.21 | 0.44 | 8.78 | 79 | 13 | 13 | 0 | 69 | 0.0965 |
| **mean** | **5.34** | | | | | | | | **0.0416** |
| **max** | **17.25 (flanders-03-thematic-crit)** | | | | | | | | |
| **total** | **69.45** | | | **463** | **97** | **100** | **3** | **187** | **0.5408** |

Both runs cost $1.1861 all in, against a $0.5624 plan estimate for v3 alone and a $2.50 budget cap per run. The estimate is low because it counts a planned request once; a fifth of the requests in each run are retries.

Three points worth carrying:

- The 10 second target holds for nine of 13 episodes on v3 and ten of 13 on v1. The four that miss are the four longest, and their time is retries, not model latency: `flanders-03-thematic-crit` spent 19.26 seconds on 97 requests of which 38 were retries.
- Three blocks failed all three attempts in each run: two in `perspective-13d-critique` plus one in `colman-04.03-life-crit` on v3, two in `perspective-13d-critique` plus one in `colman-05.02-master-studies-crit` on v1. `--repair` recovered every one of them, 75 sentences in each run, with zero sentences left unanswered in either run. Repair cost $0.0055 on v3 and $0.0044 on v1.
- The window fallback fired hard, and for a reason the fit set never showed. `perspective-13d-critique` and `greco-2.2-thumbnailing` ran their entire sentence pass windowed: their transcripts exceed the 24,000-token cap once Jev's retake pass cuts fewer sentences than the production module's flags do, so the up-front `over_cap` check trips for the whole episode. Every sentence in those two episodes, 3,140 of the 7,588, was scored against a 200-sentence window instead of the whole transcript. `flanders-03-thematic-crit` fell back on 25 requests after context errors. That is 193 of 467 requests on v3. The fit set had one windowed block in one episode. The design's claim that sentences are scored against the whole episode does not hold for the two longest held-out episodes, and those two are the ones Jev scores worst.

## Sentence-level confusion, pooled over the 13 held-out episodes

Each arm at its own calibrated threshold, layered, 7,588 sentences. `wrong drops` is the editor kept it and the arm removed it; `wrong keeps` is the reverse.

| arm | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio |
|---|---:|---:|---:|---:|---:|---:|
| v3 jev_a | 3,924 | 2,092 | 480 | 1,092 | 79.28 | 103.53 |
| v1 jev_a | 3,690 | 2,283 | 714 | 901 | 78.72 | 98.37 |

The fit set's error shape repeats exactly: v1 drops more and keeps less, v3 keeps more and drops less, and v3 wins agreement by about half a point. What is new is the ratio. On the fit six the two error columns were roughly balanced (v3 215 drops against 228 keeps). Here v3 makes 1,092 wrong keeps against 480 wrong drops, better than two to one, and its kept ratio is 103.53 against the editor's own duration. The threshold calibration is doing its job (it is keeping about as much as the editor did) but the sentences it keeps are the wrong ones. Wrong keeps was already named as the open problem in the v1-v2-v3 write-up, where Luna made 122 to every Jev arm's 220-plus. Held-out does not change that diagnosis; it enlarges it.

## Does held-out match the fit-set picture?

Mostly yes on the ordering, and yes on the gap, once you compare on the same episodes.

The absolute numbers drop hard from fit to held-out, v3 `jev_a` from 83.00 to 76.74 SENTENCE POINTS, but so do the references. On the fit six, Luna is 86.86 and Opus 89.49. On the 12 held-out ladder episodes they are 82.31 and 85.14. The gap between Jev and Luna is 3.86 points on the fit set and 2.96 points on held-out: it narrowed. On WORD SCORE and GRADE the gap widened a little, 4.97 to 6.73 and 4.57 to 5.34. So the headline metric held up and the frame-level metrics slipped. Nothing here says the prompts were overfit to the six.

Two things did not carry over. `jev_mix`, the blend of the score with the removal probability, was the best GRADE arm on the fit set and is now behind plain `jev_a` on all three metrics; it should be dropped. And the error split that made v3 look balanced on the fit set is not balanced here: over two wrong keeps for every wrong drop.

Between the versions, v3 held up better, but not by much and not everywhere. Pooled over the 13 the two are a dead heat on SENTENCE POINTS (76.74 against 76.75), v1 is ahead by 0.44 on WORD and v3 by 0.27 on GRADE. Pooled over all 18 ladder episodes v3 leads by 1.06 SENTENCE POINTS (80.47 against 79.41) and ties on the other two. Per episode, v3 wins nine of 13 on SENTENCE POINTS, and its two worst losses are informative: `hampton-7-conclusion` is 43 sentences, where one decision moves the number nine points, and `perspective-13d-critique` is the biggest fully windowed episode. My read is that the v1-v2-v3 conclusion stands and needs no revision: the two versions are within noise on the headline, v3 has the better error shape and the better GRADE, and it is the one to carry forward. But held-out did not produce the separation that would make that a confident call rather than a preference.

The real finding is not about the prompt version. It is that both versions sit around 80 on the 18-episode ladder, below every reasoning-model arm and just under `gpt-5.6-luna-medium`, and that the next 3 points are not in the score levels. They are in the 1,092 wrong keeps and in the two longest episodes that never saw their own transcript.

## Files

- runs: `docs/jev-real/roughcut-jev-heldout-v3-{decisions,requests}.jsonl`, `-timing.json`, and the same for `-heldout-v1`
- reports: `docs/jev-real/roughcut-jev-heldout-v3-notes.md`, `-summary.json`, and the same for `-heldout-v1`
- 18-episode merges: `docs/jev-real/roughcut-jev-all18-v3-*` and `roughcut-jev-all18-v1-*`
- fit-set reference: `docs/jev-real/roughcut-jev-v1-v2-v3.md`, `roughcut-jev-v3-summary.json`, `roughcut-jev-v1-summary.json`
- published ladder: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.md` and its `.json`
- design and episode split: `docs/superpowers/specs/2026-09-20-jev-roughcut-design.md`
