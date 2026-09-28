Developer-facing notes on the f3-Opus 5.5 stack: the f3 combiner decides, claude-opus-5-5 overrides its unsure 25%, with the Opus 5 runs alongside.

Generated 2026-09-28T06:03:35+00:00 by `scripts/jev_real/roughcut_hybrid_f3opus55.py` from the run files on disk, the frozen keep-rule files, the f3 feature files and frozen weights, the Opus 5 f3 run, the archived donor ratings and the cached removal ranges. The report step makes no model calls. Every metric is x100, two decimals, with um removal and delete silence layered on (the ladder column). The JSON next to this file keeps the raw values.

# The f3-Opus 5.5 stack

Step 6 of the round-two design. f3 (`q3+code+v3`, keep threshold 2.80 on `5 * p_keep`) decides all 8,943 sentences of the 18 ladder episodes. Its unsure sentences, margin `abs(5 * p_keep - threshold)` at or under a global cutoff, go to `claude-opus-5-5` with the Opus 5 f3 run's call unchanged: rules5 system prompt, preamble plus whole transcript in the cached prefix, targets uncached, groups of up to 80, first group alone, medium effort. The router sends Opus 5.5 adaptive thinking with that effort, as it did for Opus 5. No Opus 5 answer is reused.

The 25% slice (2,236 sentences, cutoff 1.164) was asked fresh: 2,150 sentences sent, the rest already cut by the retake pass. The 50% top-up was not run (see the cost section). Each slice's keep rule was chosen on the fit six alone and frozen to disk before that slice's held-out calls: 25% `decision` at 2026-09-28T05:53:17+00:00.

Bottom line: at 25% with `decision` f3 plus Opus 5.5 scores 86.19 SP pooled 18, 89.29 on the fit six and 84.82 held-out 12, against Opus 5's 86.21 / 88.91 / 85.02. f3 alone is 83.37. On the full published ladder with modules, f3 + Opus 5.5 25% rank 8 of 30, below claude-opus-5-high · agentic · API · rules1 86.30, above gpt-6-astra-high · chapters · Codex CLI · rules5 85.94. The top published arm is claude-opus-5-5-low · chapters · API · rules5 at 87.57.

## Results

Seconds per episode: Opus wall clock (first group alone, then up to 8 in flight) plus f3's own per-episode time. Dollars per episode are model spend: the Opus 5.5 25% row is what that run paid. The Opus 5 rows are the fresh-run figures from its write-up: its 50% seconds are measured, its 25% seconds an estimate from step 5's wall clock, and its dollars what a run asking every sentence would have paid at Opus 5 prices ($5 in, $25 out), since its runs reused step 5's answers.

| arm | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean | model $/ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| f3 + Opus 5.5 25%, fit-chosen `decision` | 89.29 | 84.82 | 86.19 | 79.79 | 94.03 | 35.9 | $0.1749 |
| f3 + Opus 5 25%, fit-chosen `decision` (reference) | 88.91 | 85.02 | 86.21 | 79.85 | 94.11 | 52.5 | $0.2447 |
| f3 + Opus 5 50%, fit-chosen `decision` (reference) | 89.56 | 84.72 | 86.20 | 80.82 | 94.84 | 51.0 | $0.4928 |
| f3 alone | 85.52 | 82.42 | 83.37 | 76.55 | 91.11 | 10.5 | $0.0000 |
| claude-opus-5-5-low · chapters · API · rules5 (top published arm) |  |  | 87.57 |  |  |  |  |

f3 alone and every stack score the fit six with f3's leave-one-out `p_keep` and the held-out 12 with its frozen weights. The top published arm has only its ladder number here.

Ladder, with modules, same 18 episodes (published arms quoted as the top arm, the bench page's headline best, each model family's best and the Opus 5 agentic arm): claude-opus-5-5-low · chapters · API · rules5 (top arm with modules; best claude-opus-5-5; shipped in v0.4.15) 87.57, claude-opus-5-5-high · agentic · API · rules1 (bench page headline best, 85.05 without modules) 87.44, claude-opus-5-high · agentic · Claude Code · rules1 (best claude-opus-5; the arm earlier Jev write-ups compare to) 86.47, claude-fable-5-1-high · chapters · Claude Code · rules5 (best claude-fable-5-1) 86.45, f3 + Opus 5 25%, `decision` 86.21, f3 + Opus 5 50%, `decision` 86.20, f3 + Opus 5.5 25%, `decision` 86.19, gpt-6-astra-high · chapters · Codex CLI · rules5 (best gpt-6-astra; shipped in v0.4.15) 85.94, gpt-5.6-sol-high · chapters · Codex CLI · rules5 (best gpt-5.6-sol) 84.61, gpt-6-luna-xhigh · chapters · API · rules5 (best gpt-6-luna) 84.25, gpt-5.6-luna-xhigh · agentic · API · rules5 (best gpt-5.6-luna) 83.84, Jev f3 `q3+code+v3` combiner alone 83.37, gpt-5.6-terra-high · agentic · API · rules1 (best gpt-5.6-terra) 80.20, deterministic baseline (um removal + retakes + delete silence) 63.72. Ranks count all 29 rows of the published ladder with modules plus the placed row (so out of 30); the top arm is claude-opus-5-5-low · chapters · API · rules5 at 87.57. Placement: f3 + Opus 5 25%, `decision` rank 8 of 30, below claude-opus-5-high · agentic · API · rules1 86.30, above gpt-6-astra-high · chapters · Codex CLI · rules5 85.94; f3 + Opus 5 50%, `decision` rank 8 of 30, below claude-opus-5-high · agentic · API · rules1 86.30, above gpt-6-astra-high · chapters · Codex CLI · rules5 85.94; f3 + Opus 5.5 25%, `decision` rank 8 of 30, below claude-opus-5-high · agentic · API · rules1 86.30, above gpt-6-astra-high · chapters · Codex CLI · rules5 85.94; Jev f3 `q3+code+v3` combiner alone rank 16 of 30, below gpt-5.6-luna-xhigh · chapters · API · rules6 83.48, above gpt-5.6-luna-xhigh · chapters · API · rules5 + kept-parts tool 83.18.

## Every keep rule

Chosen on the fit six alone (best fit-six SP, ties to the decision field, then the lower threshold); the starred row is the frozen one. The Opus 5 columns are the same rules on the Opus 5 answers.

| share | keep rule | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | Opus 5 SP fit 6 | Opus 5 SP held-out 12 | Opus 5 SP all 18 | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | * decision | 89.29 | 84.82 | 86.19 | 79.79 | 94.03 | 88.91 | 85.02 | 86.21 | 35.9 |
| 25% | score>=1 | 87.64 | 84.55 | 85.50 | 78.45 | 92.87 | 87.21 | 84.36 | 85.24 | 35.9 |
| 25% | score>=2 | 89.15 | 85.21 | 86.42 | 79.72 | 94.06 | 88.81 | 84.96 | 86.14 | 35.9 |
| 25% | score>=3 | 89.07 | 84.60 | 85.97 | 79.70 | 93.99 | 88.74 | 84.81 | 86.01 | 35.9 |
| 25% | score>=4 | 86.47 | 80.55 | 82.37 | 77.88 | 92.01 | 86.89 | 81.10 | 82.87 | 35.9 |

## Opus 5.5 against Opus 5 on the same sentences

Keep/cut decision fields on the sentences both live runs answered in each slice, same system prompt and same cached prefix. Group composition differs: 1,828 of the 2,150 Opus 5 answers on the 25% slice came from step 5's groups, built on the f1 slice. Right means matching the editor, where kept means full or partial. Score within 1 compares the two 0-5 scores. The last row per share compares Opus 5.5 with the archived agentic Opus 5 ratings instead, a different prompt and session.

| share | compared with | n | agree | Opus 5.5 right | other right | disagreements 5.5 / other right | 5.5 keep rate | other keep rate | editor keep rate | score within 1 | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | all 18, Opus 5 live | 2150 | 90.60 | 77.81 | 78.84 | 90 / 112 of 202 | 47.40 | 50.93 | 58.42 | 95.12 | 35.9 |
| 25% | fit six, Opus 5 live | 634 | 93.53 | 85.17 | 83.75 | 25 / 16 of 41 | 46.69 | 46.53 | 50.47 | 96.53 | 35.9 |
| 25% | held-out 12, Opus 5 live | 1516 | 89.38 | 74.74 | 76.78 | 65 / 96 of 161 | 47.69 | 52.77 | 61.74 | 94.53 | 35.9 |
| 25% | all 18, archived agentic Opus 5 | 2150 | 87.35 | 77.81 | 80.14 | 111 / 161 of 272 | 47.40 | 55.49 | 58.42 | 90.05 | 35.9 |

## Where the gains come from

A flip is a routed sentence whose keep/cut changed when the Opus verdict replaced f3's, read off the scoring module's sentence states with the modules layered. Right means the new state matches the editor.

| substitution | routed | flips | right | wrong | cut to kept right / wrong | kept to cut right / wrong | agreement on slice, f3 | agreement on slice, after routing | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Opus 5.5 25% `decision` | 2236 | 769 | 497 | 272 | 221 / 43 | 276 / 229 | 67.67 | 77.73 | 35.9 |
| Opus 5 25% `decision` | 2236 | 726 | 487 | 239 | 235 / 44 | 252 / 195 | 67.67 | 78.76 | 52.5 |
| Opus 5 50% `decision` | 4472 | 1097 | 661 | 436 | 297 / 68 | 364 / 368 | 77.06 | 82.09 | 51.0 |

## Per episode (25% cutoff 1.164, 50% cutoff 1.718)

| episode | sentences | routed 25% | SP f3 | SP Opus 5 25% | SP Opus 5.5 25% decision | requests | retries | fallback targets | f3 s | s/ep 25% | Opus 5.5 $ 25% | cache read share 25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo (fit) | 194 | 36 | 84.95 | 85.26 | 83.20 | 1 | 0 | 0 | 6.1 | 17.1 | $0.0554 | 27% |
| hampton-5.4-assignment-demo (fit) | 300 | 43 | 91.43 | 92.53 | 92.87 | 2 | 0 | 0 | 5.2 | 20.8 | $0.0683 | 62% |
| colman-03.03-muscles-crit (fit) | 303 | 65 | 79.17 | 75.51 | 78.81 | 2 | 0 | 0 | 5.7 | 27.5 | $0.1062 | 50% |
| edges-7.01-intro (fit) | 389 | 136 | 84.40 | 90.23 | 90.49 | 2 | 0 | 0 | 7.9 | 35.0 | $0.1021 | 62% |
| hampton-5.2-shape-demo (fit) | 411 | 46 | 95.52 | 96.55 | 97.10 | 2 | 0 | 0 | 6.7 | 26.2 | $0.0969 | 58% |
| perspective-14e-boxes-critique (fit) | 1146 | 358 | 82.53 | 88.94 | 88.94 | 5 | 0 | 0 | 22.7 | 90.1 | $0.4876 | 81% |
| perspective-13d-critique | 1752 | 515 | 84.63 | 90.58 | 90.75 | 8 | 0 | 0 | 28.8 | 86.6 | $0.7487 | 88% |
| hampton-5.5-crit1 | 181 | 19 | 90.50 | 89.39 | 90.50 | 1 | 0 | 0 | 4.0 | 11.0 | $0.0440 | 27% |
| hampton-5.5-crit2 | 127 | 9 | 90.63 | 87.48 | 88.27 | 1 | 0 | 0 | 2.2 | 6.7 | $0.0292 | 34% |
| hampton-5.5-crit3 | 137 | 12 | 81.75 | 82.48 | 83.94 | 1 | 0 | 0 | 2.4 | 7.6 | $0.0333 | 31% |
| hampton-5.5-crit4 | 156 | 14 | 93.40 | 88.91 | 90.06 | 1 | 0 | 0 | 2.3 | 8.2 | $0.0390 | 28% |
| hampton-5.5-crit5 | 295 | 10 | 91.49 | 91.49 | 91.49 | 2 | 0 | 0 | 7.3 | 14.2 | $0.0536 | 60% |
| flanders-03-thematic-crit | 1309 | 429 | 79.02 | 80.38 | 79.14 | 6 | 0 | 0 | 32.8 | 98.4 | $0.5346 | 84% |
| anatomy-30b-hamstring-crit | 951 | 290 | 85.81 | 89.75 | 89.81 | 4 | 0 | 0 | 25.4 | 88.2 | $0.3722 | 77% |
| colman-04.03-life-crit | 495 | 80 | 77.33 | 77.33 | 76.32 | 2 | 0 | 0 | 10.2 | 35.2 | $0.1303 | 57% |
| colman-05.02-master-studies-crit | 381 | 84 | 69.32 | 71.21 | 70.60 | 2 | 0 | 0 | 10.5 | 35.7 | $0.1191 | 58% |
| colman-06.06-species-crit | 373 | 86 | 77.37 | 78.74 | 79.06 | 2 | 0 | 0 | 6.1 | 32.5 | $0.1153 | 58% |
| hampton-7-conclusion | 43 | 4 | 81.63 | 83.95 | 83.95 | 1 | 0 | 0 | 2.7 | 5.6 | $0.0119 | 57% |

## Cost, caching and failures

The 25% run (`roughcut-hybrid-f3opus55-m25`): 45 requests over 45 groups, 0 retries, 0 errored attempts, 3 malformed or incomplete answers (colman-03.03-muscles-crit group 0: id 163 not a target, 0 targets missing; flanders-03-thematic-crit group 0: id 228 not a target, 0 targets missing; flanders-03-thematic-crit group 1: id 341 not a target, 0 targets missing), 0 refusals, 0 targets left on f3's decision after the re-ask. It paid $3.1476 by the router's accounting ($3.1476 at list rates) for 2,150 sentences, $0.00146 each; the plan-mode estimate was $5.3403. Input tokens 938,897: 212,581 cache writes, 719,058 cache reads (76.6% of input), 7,258 uncached; 27 of the 27 successful requests after an episode's first read the cache (1 of 1 on the first episode run, colman-03.03-muscles-crit). Output 95,594 tokens, thinking included. Provider model claude-opus-5-5. Run wall clock 458 s, files last written 2026-09-28T05:58:32+00:00.

Prices used by the router and in the listed column: $4.00 input, $5.00 cache write, $0.20 cache read, $20.00 output per million tokens. Opus 5.5 spend in this step $3.1476. Per episode, 25%: $0.1749 mean, $0.7487 max, 35.9 s mean, 98.4 s max. Opus 5's cache read share was 74.7%.

Budget gates. Before the 25% run, the plan-mode estimate was $5.34 (4 characters per token, 2,000 output tokens per request plus 60 per target); the probe on colman-03.03-muscles-crit cost $0.1062 and projected $3.15 for the 18 against the $4.50 stop line, so the run went ahead and came in at $3.15.

The 50% top-up, priced at the 25% run's real token rates (2,180 new sentences, 2,150 reused), projected $3.55; with the 25% actual $3.15 that is $6.70 against the $6.50 line in the step-6 brief, so the top-up was skipped and no 50% Opus 5.5 row exists.

## Reproduction check

| number | SP here | published | s/ep mean |
|---|---:|---:|---:|
| f3_combiner | 83.37 | 83.37 | n/a |
| opus5_m25_decision | 86.21 | 86.21 | n/a |
| opus5_m50_decision | 86.20 | 86.20 | n/a |

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input rules_prompt: `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md`, md5 afba04cbbfc9, modified 2026-09-08T05:43:31+00:00
- input ladder_reference: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`, md5 d7a26ffcf5b5, modified 2026-09-23T20:57:03+00:00
- input f3_weights: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3-weights.json`, md5 b39151db65d7, modified 2026-09-27T04:36:30+00:00
- input f3_fit_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3-fit-features.jsonl`, md5 7617714da633, modified 2026-09-27T04:07:19+00:00
- input f3_heldout_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3-heldout-features.jsonl`, md5 73ca07072f3b, modified 2026-09-27T04:07:19+00:00
- input f3_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3.json`, md5 3e8812b13cc9, modified 2026-09-27T06:46:21+00:00
- input opus5_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus.json`, md5 1886deba4541, modified 2026-09-27T06:47:09+00:00
- input opus5_run:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-decisions.jsonl`, md5 537aca60dbe2, modified 2026-09-27T06:06:34+00:00
- input opus5_run:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-requests.jsonl`, md5 5868eadab8a0, modified 2026-09-27T06:06:33+00:00
- input opus5_run:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-timing.json`, md5 f3a45eeb0772, modified 2026-09-27T06:06:34+00:00
- input opus5_keep_rule: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-keeprule.json`, md5 998f90a41f1d, modified 2026-09-27T05:58:08+00:00
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
- input opus55_m25_projection: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55-m25-projection.json`, md5 ccba20799045, modified 2026-09-28T06:00:16+00:00
- input opus55_m50_projection: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55-m50-projection.json`, md5 ed06a9d65f3b, modified 2026-09-28T05:59:52+00:00
- input opus55_m25_run:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55-m25-decisions.jsonl`, md5 cbcbf40150dd, modified 2026-09-28T05:58:32+00:00
- input opus55_m25_run:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55-m25-requests.jsonl`, md5 e221522c3b58, modified 2026-09-28T05:58:32+00:00
- input opus55_m25_run:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55-m25-timing.json`, md5 4e4fc9718eaf, modified 2026-09-28T05:58:32+00:00
- input opus55_m25_keep_rule: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55-m25-keeprule.json`, md5 4364e3ec8b28, modified 2026-09-28T05:53:17+00:00
- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 ee6934d58943
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus55.json`
