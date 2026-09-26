# The f1-Luna stack: combiner decides, Luna overrides its unsure slice (developer-facing notes)

Generated 2026-09-26T14:50:58+00:00 by `scripts/jev_real/roughcut_hybrid_f1luna.py` from the run files on disk, the f1 feature files and frozen weights, the stored v3 decisions, the archived donor ratings and the cached removal ranges. The report step makes no model calls; the run it reads cost $0.3635 by the router's accounting ($0.3636 at list rates 0.20 in, 0.02 cached, 1.20 out per million), plus $0.0031 for the one-group smoke request in `smoke-hybrid-f1luna-*`. Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.

Question: second pass, step 2 of the round-two design. Build B's f1 combiner (`q+code+v3`, C 0.003, keep threshold 3.00 on `5 * p_keep`) decides every one of the 8,943 sentences of the 18 ladder episodes, `keep_words` null, `cut_retake` from the v3 row. The sentences whose `5 * p_keep` sits closest to the threshold (margin `abs(5 * p_keep - 3.00)` under one global cutoff, 0.879, the bottom 25%, 2,236 sentences) go to `gpt-5.6-luna` exactly as build A sent Jev's v3 slice: whole episode transcript as Jev saw it, rules5 system prompt, groups of up to 40, medium effort. On routed sentences Luna's verdict replaces the combiner's keep/cut, `keep_words` stays null, the retake veto stays. Two substitutions are reported: Luna's `decision` field, and the keep rule step 1 froze on build A's fit six (`score>=2`, `roughcut-hybrid-luna.md`, chosen before this run was made). The `p_keep` values are the ones the f1 write-up's route-2 section used: fit six out of fold by leave-one-episode-out at the frozen C, held-out 12 from the frozen weights, so the selection on the fit six is not held out in the strict sense (the C and threshold were chosen there), and neither is the cutoff (chosen from the pooled 18); Luna's decisions on the slice are.

## Reproduction check

| arm | SENTENCE POINTS here | published |
|---|---:|---:|
| pure Jev (jev_a v3) | 80.47 | 80.47 |
| f1 `q+code+v3` combiner alone (build B, ladder 18) | 82.91 | 82.91 |
| offline ceiling: combiner margin bottom 25%, archived Luna (f1 write-up) | 84.81 | 84.81 |
| build A m046: v3 margin bottom 25%, live Luna decision | 83.63 | 83.63 |
| pure archived Luna chapters (routed 100%) | 83.71 | 83.71 |

The combiner's `p_keep` was recomputed from the feature files and the frozen weights (`roughcut-jev-f1-weights.json`); refitting on the fit six lands 0.0e+00 from the frozen coefficients, the slice has the same 2,236 sentences and the same cutoff as the f1 write-up's route-2 row, and the archived substitution reproduces its number through the combiner's own `route2_handoff` before any call was made.

## Pooled results

Pooled over the fit six, the held-out 12 and all 18, with modules. `decision` substitutes Luna's decision field on the routed slice; `score>=2` substitutes the frozen keep rule; `ceiling` substitutes the archived Luna chapters decision (the offline number). Build A is the same live call on Jev's v3 slice. The fit-six column is where the combiner's C, threshold and (for build A) the keep rule were chosen; the held-out 12 column is not. Seconds per episode are Luna's wall clock at concurrency 8 plus the combiner's own per-episode time from the f1 run (v3 sentence pass plus the feature questions); dollars are the router's accounting, Luna only, with the stack total (Luna plus f1) alongside.

| arm | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean | Luna $/ep mean | arm total $/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f1 combiner alone (build B) | 85.63 | 81.70 | 82.91 | 76.27 | 90.78 | 7.9 | n/a | $0.0974 |
| stack, Luna `decision` | 87.35 | 82.55 | 84.02 | 78.43 | 92.59 | 29.7 | $0.0202 | $0.1176 |
| stack, frozen rule `score>=2` | 87.58 | 83.30 | 84.61 | 78.66 | 92.84 | 29.7 | $0.0202 | $0.1176 |
| stack ceiling (archived Luna on the same slice) | 87.34 | 83.70 | 84.81 | 78.72 | 92.85 | n/a | n/a | n/a |
| build A m046, Luna `decision` | 86.91 | 82.17 | 83.63 | 77.35 | 91.28 | n/a | n/a | n/a |
| build A m046, frozen rule `score>=2` | 86.92 | 82.77 | 84.05 | 77.47 | 91.44 | n/a | n/a | n/a |

Ladder, with modules, same 18 episodes (build A and the f1 combiner added as rows): shipped Opus agentic 86.47, best Luna chapters 83.71, build A m046 (v3 margin 25%, live Luna decision) 83.63, Jev f1 `q+code+v3` combiner (build B) 82.91, Jev jev_a v3 (pure Jev) 80.47, Luna single call 65.36, deterministic baseline (um removal + retakes + delete silence) 63.72. Placement: stack with `decision` below shipped Opus agentic, above best Luna chapters (rank 2 of 8); stack with `score>=2` below shipped Opus agentic, above best Luna chapters (rank 2 of 8); the ceiling on this slice below shipped Opus agentic, above best Luna chapters.

Against the offline ceiling 84.81 and build A's 83.63: the stack lands at 84.02 with the decision field (+1.12 on the combiner alone, +0.40 on build A) and 84.61 with `score>=2` (+1.71 on the combiner alone, +0.57 on build A under the same rule). Of the 1.91 SP the archived substitution adds over the combiner, the live call keeps 59% with the decision field and 89% with `score>=2`. Held out: 82.55 and 83.30 against build A's 82.17 and 82.77, the combiner alone 81.70, pure Jev v3 79.35.

## Every keep rule on this slice

The same stored answers under each keep rule, for the shape of the curve. `score>=2` is the frozen one; nothing here was chosen on these numbers.

| keep rule | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | all 18 minus decision | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| decision | 87.35 | 82.55 | 84.02 | 78.43 | 92.59 | +0.00 | 29.7 |
| score>=1 | 86.73 | 83.80 | 84.69 | 78.28 | 92.62 | +0.67 | 29.7 |
| * score>=2 | 87.58 | 83.30 | 84.61 | 78.66 | 92.84 | +0.59 | 29.7 |
| score>=3 | 87.19 | 82.51 | 83.94 | 78.45 | 92.65 | -0.08 | 29.7 |
| score>=4 | 84.06 | 78.35 | 80.10 | 76.26 | 90.46 | -3.92 | 29.7 |

## The slice

The combiner's bottom 25% shares 1,073 sentences with build A's v3-margin bottom 25% (48% of 2,236); on the 1,041 shared sentences both runs got an answer for, Luna gave the same decision 88% of the time (two separate medium-effort calls on the same transcript). Agreement with the editor on the routed slice (kept means full or partial), read off each arm's own states with modules: combiner 67.53, Jev v3 on the same sentences 65.43, live Luna decision 71.29, live Luna `score>=2` 74.06, archived Luna 74.91. Seconds per episode as above, 29.7.

## Live Luna against archived Luna on the same sentences

| run | answered | live agrees with archived | live right | archived right | combiner right | disagreements live right / archived right | live keep & archived cut / live cut & archived keep | live keep rate | archived keep rate | editor keep rate | SP live decision | SP archived (ceiling) | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m25 (25% routed, medium effort) | 2164 | 80.27 | 71.03 | 75.32 | 67.53 | 167 / 260 of 427 | 116 / 311 | 49.91 | 58.92 | 64.93 | 84.02 | 84.81 | 29.7 |

Same pattern as build A: the live call is cut-heavier than the archive on the unsure slice (311 live cut where the archive keeps against 116 the other way), the editor keeps more of the slice than either, so the frozen `score>=2` rule, which keeps anything Luna scores 2 or more, recovers part of the gap. Live scores land within one point of the archived score on 84.47% of the answered sentences.

## Where the gains come from

A flip is a routed sentence whose keep/cut changed when Luna's verdict replaced the combiner's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor. Trim changed is always zero here because neither side carries trims on the slice.

| substitution | routed | flips | right | wrong | cut to kept right / wrong | kept to cut right / wrong | agreement on slice, combiner | agreement on slice, after routing | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stack, `decision` | 2236 | 840 | 462 | 378 | 233 / 56 | 229 / 322 | 67.53 | 71.29 | 29.7 |
| stack, `score>=2` | 2236 | 798 | 472 | 326 | 259 / 70 | 213 / 256 | 67.53 | 74.06 | 29.7 |
| ceiling (archived Luna) | 2236 | 755 | 460 | 295 | 255 / 73 | 205 / 222 | 67.53 | 74.91 | 29.7 |

## Per episode, m25 (25% routed, cutoff 0.879, medium effort)

| episode | sentences | routed | asked | unanswered | SP Jev v3 | SP f1 | SP build A | SP stack decision | SP stack score>=2 | SP ceiling | WORD stack score>=2 | GRADE stack score>=2 | requests | retries | Luna s | f1 s | s/ep | Luna $ | stack $ | input tokens/request | cached share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo (fit) | 194 | 47 | 46 | 0 | 85.88 | 84.12 | 85.82 | 82.06 | 82.06 | 81.91 | 74.67 | 105.94 | 2 | 0 | 27.9 | 5.2 | 33.1 | $0.0060 | $0.0386 | 5.5k | 63% |
| hampton-5.4-assignment-demo (fit) | 300 | 58 | 57 | 0 | 84.80 | 91.53 | 90.20 | 90.53 | 90.87 | 91.00 | 84.87 | 102.08 | 3 | 0 | 21.3 | 3.9 | 25.2 | $0.0072 | $0.0553 | 6.1k | 23% |
| colman-03.03-muscles-crit (fit) | 303 | 81 | 81 | 0 | 75.05 | 77.66 | 75.61 | 73.76 | 75.51 | 74.55 | 63.64 | 77.50 | 3 | 0 | 31.6 | 4.3 | 35.9 | $0.0114 | $0.0672 | 7.4k | 19% |
| edges-7.01-intro (fit) | 389 | 123 | 91 | 0 | 79.69 | 84.91 | 86.89 | 87.15 | 86.89 | 86.45 | 87.93 | 87.87 | 3 | 0 | 33.0 | 6.3 | 39.3 | $0.0109 | $0.0710 | 5.2k | 27% |
| hampton-5.2-shape-demo (fit) | 411 | 61 | 58 | 0 | 92.70 | 96.25 | 91.27 | 95.26 | 95.26 | 96.23 | 87.25 | 98.24 | 4 | 0 | 26.6 | 4.9 | 31.5 | $0.0121 | $0.0885 | 8.8k | 16% |
| perspective-14e-boxes-critique (fit) | 1146 | 317 | 313 | 0 | 81.78 | 82.88 | 87.67 | 88.25 | 88.33 | 87.80 | 85.54 | 88.21 | 10 | 0 | 35.7 | 17.4 | 53.1 | $0.0497 | $0.2957 | 17.3k | 8% |
| perspective-13d-critique | 1752 | 451 | 443 | 0 | 78.09 | 84.61 | 85.74 | 87.74 | 88.80 | 88.74 | 88.33 | 91.59 | 14 | 0 | 36.6 | 20.9 | 57.5 | $0.0938 | $0.4351 | 25.9k | 6% |
| hampton-5.5-crit1 | 181 | 18 | 17 | 0 | 87.35 | 89.94 | 86.24 | 88.29 | 88.29 | 89.39 | 68.18 | 111.76 | 2 | 0 | 20.0 | 3.0 | 23.0 | $0.0045 | $0.0370 | 5.6k | 25% |
| hampton-5.5-crit2 | 127 | 8 | 8 | 0 | 79.45 | 89.84 | 77.32 | 88.27 | 88.27 | 88.27 | 63.65 | 108.56 | 1 | 0 | 4.9 | 1.5 | 6.5 | $0.0012 | $0.0229 | 4.4k | 32% |
| hampton-5.5-crit3 | 137 | 12 | 12 | 0 | 81.17 | 82.48 | 81.17 | 82.48 | 82.48 | 83.50 | 76.59 | 110.67 | 2 | 0 | 9.0 | 1.7 | 10.7 | $0.0024 | $0.0262 | 4.8k | 30% |
| hampton-5.5-crit4 | 156 | 16 | 16 | 0 | 88.46 | 93.40 | 88.85 | 88.14 | 88.14 | 87.88 | 67.38 | 109.10 | 2 | 0 | 7.0 | 1.5 | 8.5 | $0.0029 | $0.0313 | 5.4k | 27% |
| hampton-5.5-crit5 | 295 | 12 | 12 | 0 | 90.88 | 91.49 | 90.47 | 90.81 | 91.83 | 92.07 | 69.57 | 105.30 | 3 | 0 | 5.3 | 5.9 | 11.2 | $0.0048 | $0.0592 | 7.4k | 19% |
| flanders-03-thematic-crit | 1309 | 472 | 465 | 0 | 77.14 | 76.95 | 78.17 | 75.41 | 77.01 | 78.37 | 80.02 | 86.11 | 13 | 0 | 38.6 | 24.7 | 63.3 | $0.0784 | $0.3571 | 21.2k | 7% |
| anatomy-30b-hamstring-crit | 951 | 279 | 269 | 0 | 84.90 | 85.16 | 87.48 | 86.78 | 86.64 | 87.69 | 83.81 | 93.90 | 8 | 0 | 21.1 | 18.0 | 39.0 | $0.0391 | $0.2451 | 16.4k | 9% |
| colman-04.03-life-crit | 495 | 90 | 87 | 0 | 76.18 | 76.93 | 75.80 | 76.28 | 77.49 | 76.51 | 65.80 | 78.41 | 4 | 0 | 23.7 | 7.9 | 31.6 | $0.0143 | $0.1121 | 10.7k | 13% |
| colman-05.02-master-studies-crit | 381 | 92 | 92 | 0 | 66.33 | 69.06 | 68.66 | 71.63 | 70.89 | 70.73 | 67.63 | 90.03 | 4 | 0 | 27.3 | 8.7 | 36.0 | $0.0137 | $0.0879 | 9.1k | 16% |
| colman-06.06-species-crit | 373 | 95 | 93 | 0 | 79.57 | 75.76 | 78.95 | 78.47 | 78.74 | 78.61 | 71.17 | 97.28 | 3 | 0 | 18.8 | 4.3 | 23.2 | $0.0106 | $0.0799 | 8.4k | 17% |
| hampton-7-conclusion | 43 | 4 | 4 | 0 | 72.33 | 79.30 | 81.63 | 81.63 | 83.95 | 84.42 | 86.26 | 98.01 | 1 | 0 | 4.3 | 2.2 | 6.6 | $0.0007 | $0.0069 | 2.6k | 56% |

Routed sentences the v3 retake pass had already cut were not sent: 72 of 2,236. 82 requests, 0 retries, 0 errored attempts, 0 malformed or incomplete answers, 0 targets left unanswered after retries (those keep the combiner's own decision). Tokens: 1,210,726 prompt of which 121,200 cached, 119,352 completion of which 68,236 reasoning. Router cost $0.3635 ($0.0202 per episode, max $0.0938); the plan estimated $0.5014. Stack cost per episode, Luna plus the f1 combiner's own calls: $0.1176. Seconds per episode 29.7 mean, 63.3 max (Luna 21.8, f1 7.9). Run wall clock 393 s, generated 2026-09-26T14:47:26+00:00.

## How the call was made

Identical to build A (`roughcut_hybrid_luna.py`, whose `execute` this script calls): system message the rules5 prompt read from `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md` at run time (md5 afba04cbbfc9), never copied into this repo; user message the `hybrid-preamble-v1` preamble, the whole transcript as Jev's sentence pass rendered it, then the target ids in groups of up to 40 closing early past 120 sentences; 8 in flight, 3 attempts on an error, 1 re-ask on malformed JSON, 600 s timeout, `reasoning_effort` medium. Only the selection and the Jev side differ: the slice is the combiner's margin, and on non-routed sentences the combiner's keep/cut (as 5 or 0 at its 3.00 threshold) stands, with no trims anywhere. Every attempt is a row in the requests file with the raw answer, the parsed verdicts, token counts and cost.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input rules_prompt: `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md`, md5 afba04cbbfc9, modified 2026-09-08T05:43:31+00:00
- input ladder_reference: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`, md5 d7a26ffcf5b5, modified 2026-09-23T20:57:03+00:00
- input f1_weights: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-weights.json`, md5 242943da549e, modified 2026-09-26T07:48:20+00:00
- input f1_fit_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-fit-features.jsonl`, md5 bca3b191584a, modified 2026-09-26T07:35:32+00:00
- input f1_fit_timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-fit-timing.json`, md5 0d0506b78962, modified 2026-09-26T07:35:32+00:00
- input f1_heldout_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-heldout-features.jsonl`, md5 37c568d73d2c, modified 2026-09-26T07:49:47+00:00
- input f1_heldout_timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-heldout-timing.json`, md5 f5e20384e73e, modified 2026-09-26T07:49:47+00:00
- input f1_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1.json`, md5 20e878d65110, modified 2026-09-26T08:05:24+00:00
- input build_a_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna.json`, md5 eeff751876a9, modified 2026-09-26T14:48:38+00:00
- input luna:2026-09-08-13d-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-13d-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 3a5d76e1fc4e, modified 2026-09-11T21:45:36+00:00
- input luna:2026-09-08-14e-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-14e-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 691647bac740, modified 2026-09-11T21:45:36+00:00
- input luna:2026-09-08-anatomy30b-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-anatomy30b-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 0ca8d9348753, modified 2026-09-11T21:45:37+00:00
- input luna:2026-09-08-colman0204-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0204-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 f1edfe25dd96, modified 2026-09-11T21:45:37+00:00
- input luna:2026-09-08-colman0303-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0303-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 1e5b54282916, modified 2026-09-11T21:45:38+00:00
- input luna:2026-09-08-colman0403-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0403-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 fd9b959669fd, modified 2026-09-11T21:45:38+00:00
- input luna:2026-09-08-colman0502-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0502-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 46b2f9304b06, modified 2026-09-11T21:45:38+00:00
- input luna:2026-09-08-colman0606-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-colman0606-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 706b124d09e1, modified 2026-09-11T21:45:38+00:00
- input luna:2026-09-08-edges701-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-edges701-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 ccf54db821a3, modified 2026-09-11T21:45:39+00:00
- input luna:2026-09-08-flanders03-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-flanders03-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 a87662839c73, modified 2026-09-11T21:45:40+00:00
- input luna:2026-09-08-hampton52-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton52-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 f6a402f077d0, modified 2026-09-11T21:45:40+00:00
- input luna:2026-09-08-hampton54-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton54-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 7ff2280aa54e, modified 2026-09-11T21:45:40+00:00
- input luna:2026-09-08-hampton55crit1-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit1-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 54c7ae96eeec, modified 2026-09-11T21:45:40+00:00
- input luna:2026-09-08-hampton55crit2-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit2-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 2e0c9a78c836, modified 2026-09-11T21:45:40+00:00
- input luna:2026-09-08-hampton55crit3-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit3-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 09f1c710b42e, modified 2026-09-11T21:45:40+00:00
- input luna:2026-09-08-hampton55crit4-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit4-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 def50ff50e65, modified 2026-09-11T21:45:41+00:00
- input luna:2026-09-08-hampton55crit5-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton55crit5-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 ffde8bfecc7f, modified 2026-09-11T21:45:41+00:00
- input luna:2026-09-08-hampton7-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-08-hampton7-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json`, md5 b632239066e8, modified 2026-09-11T21:45:41+00:00
- input opus:2026-07-25-13d-partial-agentic-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-25-13d-partial-agentic-claude-opus-5.json`, md5 0d0736c8c5df, modified 2026-09-11T21:44:02+00:00
- input opus:2026-07-25-14e-partial-agentic-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-25-14e-partial-agentic-claude-opus-5.json`, md5 3559c4faf8e8, modified 2026-09-11T21:44:03+00:00
- input opus:2026-07-25-edges701-partial-agentic-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-25-edges701-partial-agentic-claude-opus-5.json`, md5 2501cf86f765, modified 2026-09-11T21:44:04+00:00
- input opus:2026-07-25-hampton55-partial-agentic-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-25-hampton55-partial-agentic-claude-opus-5.json`, md5 6edd01a43e62, modified 2026-09-11T21:44:05+00:00
- input opus:2026-07-26-flanders03-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-26-flanders03-partial-agentic-v1-claude-opus-5.json`, md5 5c560057811b, modified 2026-09-11T21:43:57+00:00
- input opus:2026-07-26-greco-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-26-greco-partial-agentic-v1-claude-opus-5.json`, md5 fa5fe0affe30, modified 2026-07-27T06:59:42+00:00
- input opus:2026-07-27-anatomy30b-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-anatomy30b-partial-agentic-v1-claude-opus-5.json`, md5 a5cc0a241a2b, modified 2026-09-11T21:43:58+00:00
- input opus:2026-07-27-colman0204-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-colman0204-partial-agentic-v1-claude-opus-5.json`, md5 036d2513db5f, modified 2026-09-11T21:43:58+00:00
- input opus:2026-07-27-colman0303-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-colman0303-partial-agentic-v1-claude-opus-5.json`, md5 80dd988c0f4c, modified 2026-09-11T21:43:59+00:00
- input opus:2026-07-27-colman0403-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-colman0403-partial-agentic-v1-claude-opus-5.json`, md5 cfee48584b5a, modified 2026-09-11T21:43:59+00:00
- input opus:2026-07-27-colman0502-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-colman0502-partial-agentic-v1-claude-opus-5.json`, md5 8c9b4580f169, modified 2026-09-11T21:44:00+00:00
- input opus:2026-07-27-colman0606-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-colman0606-partial-agentic-v1-claude-opus-5.json`, md5 3d9ec92599b3, modified 2026-09-11T21:44:00+00:00
- input opus:2026-07-27-hampton52-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-hampton52-partial-agentic-v1-claude-opus-5.json`, md5 692baf888bef, modified 2026-09-11T21:44:00+00:00
- input opus:2026-07-27-hampton54-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-hampton54-partial-agentic-v1-claude-opus-5.json`, md5 de9829fadf41, modified 2026-09-11T21:44:01+00:00
- input opus:2026-07-27-hampton7-partial-agentic-v1-claude-opus-5.json: `D:\solar-sailer\benchmarks\roughcut\results\2026-07-27-hampton7-partial-agentic-v1-claude-opus-5.json`, md5 80c4ef42877c, modified 2026-09-11T21:44:01+00:00
- input m25:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna-m25-decisions.jsonl`, md5 69aa9f944845, modified 2026-09-26T14:47:26+00:00
- input m25:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna-m25-requests.jsonl`, md5 de899087b739, modified 2026-09-26T14:47:26+00:00
- input m25:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna-m25-timing.json`, md5 53b4ae7bc293, modified 2026-09-26T14:47:26+00:00
- input build_a_m046:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m046-decisions.jsonl`, md5 e30c0a008c86, modified 2026-09-26T07:42:19+00:00
- input build_a_m046:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m046-requests.jsonl`, md5 7e44b01875c2, modified 2026-09-26T07:42:19+00:00
- input build_a_m046:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m046-timing.json`, md5 a6396a5c1d54, modified 2026-09-26T07:42:19+00:00
- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 ee6934d58943
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna.json`
