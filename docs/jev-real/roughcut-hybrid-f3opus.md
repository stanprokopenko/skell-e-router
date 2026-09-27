Developer-facing notes on the f3-Opus stack: the f3 combiner decides, claude-opus-5 overrides its unsure 50% and 25%.

Generated 2026-09-27T06:10:39+00:00 by `scripts/jev_real/roughcut_hybrid_f3opus.py` from the run files on disk, the f3 feature files and frozen weights, step 5's Opus run (whose answers are reused), the archived donor ratings and the cached removal ranges. The report step makes no model calls. Every metric is x100, two decimals, with um removal and delete silence layered on (the ladder column). The JSON next to this file keeps the raw values.

# The f3-Opus stack

Stan picked f3 (`q3+code+v3`, keep threshold 2.80 on `5 * p_keep`) as the winning Jev-only combiner on its held-out result, so this run puts Opus on top of it. f3 decides all 8,943 sentences of the 18 ladder episodes; the 4,472 whose margin `abs(5 * p_keep - threshold)` sits at or under 1.718 (the pooled bottom 50%, one global cutoff) go to `claude-opus-5` with step 5's call unchanged: rules5 system prompt, preamble plus whole transcript in the cached prefix, targets uncached, groups of up to 80, medium effort. f3's own bottom 25% (2,236 sentences, cutoff 1.164) sits inside that slice, so the same answers give the 25% point too. The fit six carry f3's leave-one-out `p_keep`, the held-out 12 its frozen weights, as in the f1 stack.

Cost saver: 2,164 of the 4,330 sentences asked for the 50% slice already had a step 5 answer (same model, same system prompt and same cached prefix, both checked by md5), so only 2,166 were sent. The reused answers came from step 5's groups, which held a different mix of targets than a fresh 50% run would have, so a reused answer is what Opus said in a different group. For the 25% point, 1,828 of its 2,150 asked sentences are reused and the rest come from this run. Keep rules for both shares were chosen on the fit six alone and frozen to `roughcut-hybrid-f3opus-m50-keeprule.json` at 2026-09-27T05:58:08+00:00, before any held-out call of this run: `decision` at 50%, `decision` at 25%.

Bottom line: f3 plus Opus at 50% with `decision` scores 86.20 SP pooled 18 (84.72 held-out 12), and at 25% with `decision` 86.21 (85.02 held out), against 85.81 for the f1-Opus 25% stack, 83.37 for f3 alone and 86.47 for shipped Opus agentic.

## Offline ceiling on the f3 slices

Before any call: the archived Opus agentic ratings substituted on the same slices ($0). The archived Opus carries partial keeps the live call does not ask for, so the keep/cut-only row is the like-for-like bound. No live calls behind these rows, so their seconds column is n/a.

| substitution on the slice | share | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| f3 combiner alone (winning Jev-only run) |  | 85.52 | 82.42 | 83.37 | 76.55 | 91.11 | 10.5 |
| archived Opus, keep/cut only | 25% | 88.34 | 85.06 | 86.06 | 79.37 | 93.59 | n/a |
| archived Opus with trims | 25% | 88.34 | 85.14 | 86.12 | 79.49 | 93.73 | n/a |
| archived Luna chapters | 25% | 87.54 | 84.19 | 85.22 | 78.94 | 93.10 | n/a |
| archived Opus, keep/cut only | 50% | 89.38 | 84.97 | 86.32 | 80.05 | 94.22 | n/a |
| archived Opus with trims | 50% | 89.69 | 85.35 | 86.68 | 80.38 | 94.51 | n/a |
| archived Luna chapters | 50% | 87.24 | 83.34 | 84.54 | 79.46 | 93.27 | n/a |

## Results

Seconds per episode: the 50% rows are this run's Opus wall clock (new asks only, first group alone then up to 8 in flight) plus f3's own per-episode time; the 25% rows use step 5's Opus wall clock on a slice of the same size plus f3's time, an estimate. Model $/ep is what this run paid for its new asks; fresh $/ep is what a run asking every sentence of the slice would have cost at this run's per-sentence cost ($0.00205). f1 rows are for reference.

| arm | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean | model $/ep paid | model $/ep fresh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f3 + Opus 50%, fit-chosen `decision` (the decision field) | 89.56 | 84.72 | 86.20 | 80.82 | 94.84 | 51.0 | $0.2465 | $0.4928 |
| f3 + Opus 25%, fit-chosen `decision` (the decision field) | 88.91 | 85.02 | 86.21 | 79.85 | 94.11 | 52.5 | n/a | $0.2447 |
| f1 + Opus 25%, `score>=2` (step 5, reference) | 88.55 | 84.60 | 85.81 | 79.27 | 93.72 | 49.9 | $0.2732 | $0.2732 |
| f1 + Opus 25%, Opus `decision` (step 5, reference) | 88.43 | 84.61 | 85.78 | 79.53 | 93.88 | 49.9 | $0.2732 | $0.2732 |
| f3 alone (winning Jev-only run) | 85.52 | 82.42 | 83.37 | 76.55 | 91.11 | 10.5 | $0.0000 | $0.0000 |
| f1 alone (reference) | 85.63 | 81.70 | 82.91 | 76.27 | 90.78 | 7.9 | $0.0000 | $0.0000 |
| shipped Opus agentic (ladder) |  |  | 86.47 |  |  |  |  |  |

f3 and f1 alone score the fit six with their leave-one-out predictions and the held-out 12 with their frozen weights, the same split as the stacks. Shipped Opus agentic has only its ladder number here; `docs/TASKS.md` puts it at about $5 and an hour per episode. The f3 combiner's $0 is new spend only; its rows are a join of Jev runs already paid for.

Ladder, with modules, same 18 episodes: shipped Opus agentic 86.47, f1-Opus stack 25%, `score>=2` (step 5) 85.81, f1-Opus stack 25%, Opus `decision` (step 5) 85.78, f1-Luna stack 25%, `score>=2` (step 2) 84.61, best Luna chapters 83.71, Jev f3 `q3+code+v3` combiner (winning Jev-only run) 83.37, Jev f1 `q+code+v3` combiner (reference) 82.91, Jev jev_a v3 (pure Jev) 80.47, Luna single call 65.36, deterministic baseline (um removal + retakes + delete silence) 63.72. Placement: 50% `decision` below shipped Opus agentic, above f1-Opus stack 25%, `score>=2` (step 5) (rank 2 of 11); 25% `decision` below shipped Opus agentic, above f1-Opus stack 25%, `score>=2` (step 5) (rank 2 of 11).

## Every keep rule

Chosen on the fit six alone (best fit-six SP, ties to the decision field, then the lower threshold). The other rows show the shape of the curve.

| share | keep rule | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50% | * decision | 89.56 | 84.72 | 86.20 | 80.82 | 94.84 | 51.0 |
| 50% | score>=1 | 87.26 | 84.08 | 85.06 | 78.54 | 92.95 | 51.0 |
| 50% | score>=2 | 89.51 | 84.87 | 86.29 | 80.37 | 94.55 | 51.0 |
| 50% | score>=3 | 88.78 | 84.13 | 85.56 | 80.56 | 94.61 | 51.0 |
| 50% | score>=4 | 81.20 | 74.95 | 76.87 | 76.74 | 90.36 | 51.0 |
| 25% | * decision | 88.91 | 85.02 | 86.21 | 79.85 | 94.11 | 52.5 |
| 25% | score>=1 | 87.21 | 84.36 | 85.24 | 78.43 | 92.86 | 52.5 |
| 25% | score>=2 | 88.81 | 84.96 | 86.14 | 79.49 | 93.90 | 52.5 |
| 25% | score>=3 | 88.74 | 84.81 | 86.01 | 79.76 | 94.05 | 52.5 |
| 25% | score>=4 | 86.89 | 81.10 | 82.87 | 78.34 | 92.48 | 52.5 |

## Live Opus against archived Opus

Keep/cut on the 50% slice's answered sentences, split by where the answer came from. The archived Opus ran agentically on rules1 at high effort over whole episodes, so this is a different prompt as well as a different session.

| answers | n | agree | live right | archived right | disagreements live / archived right | live keep rate | archived keep rate | editor keep rate | score within 1 | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all answered | 4330 | 90.85 | 82.31 | 84.11 | 159 / 237 of 396 | 60.02 | 64.92 | 65.94 | 93.26 | 51.0 |
| reused from step 5 | 2164 | 90.16 | 79.34 | 80.87 | 90 / 123 of 213 | 57.76 | 62.06 | 64.93 | 92.65 | 51.0 |
| asked in this run | 2166 | 91.55 | 85.27 | 87.35 | 69 / 114 of 183 | 62.28 | 67.77 | 66.94 | 93.86 | 51.0 |

## Where the gains come from

A flip is a routed sentence whose keep/cut changed when Opus's verdict replaced f3's, read off the scoring module's sentence states with the modules layered. Right means the new state matches the editor (kept means full or partial).

| substitution | routed | flips | right | wrong | cut to kept right / wrong | kept to cut right / wrong | agreement on slice, f3 | agreement on slice, after routing | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% `decision` | 4472 | 1097 | 661 | 436 | 297 / 68 | 364 / 368 | 77.06 | 82.09 | 51.0 |
| 25% `decision` | 2236 | 726 | 487 | 239 | 235 / 44 | 252 / 195 | 67.67 | 78.76 | 52.5 |
| 50% archived Opus keep/cut only (ceiling) | 4472 | 1004 | 638 | 366 | 301 / 98 | 337 / 268 | 77.06 | 83.14 | n/a |

## Per episode (50% cutoff 1.718, 25% cutoff 1.164)

| episode | sentences | routed 50% | reused | asked now | SP f3 | SP 25% decision | SP 50% decision | requests | retries | fallback targets | Opus s | f3 s | s/ep | Opus $ paid | Opus $ fresh 50% | cache read share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo (fit) | 194 | 87 | 46 | 38 | 84.95 | 85.26 | 84.95 | 1 | 0 | 0 | 24.1 | 6.1 | 30.3 | $0.0940 | $0.1721 | 27% |
| hampton-5.4-assignment-demo (fit) | 300 | 145 | 57 | 87 | 91.43 | 92.53 | 90.87 | 2 | 0 | 0 | 38.6 | 5.2 | 43.8 | $0.1470 | $0.2950 | 61% |
| colman-03.03-muscles-crit (fit) | 303 | 174 | 81 | 92 | 79.17 | 75.51 | 70.59 | 2 | 0 | 0 | 32.0 | 5.7 | 37.7 | $0.1602 | $0.3544 | 49% |
| edges-7.01-intro (fit) | 389 | 212 | 91 | 58 | 84.40 | 90.23 | 92.29 | 2 | 0 | 0 | 35.6 | 7.9 | 43.5 | $0.1208 | $0.3052 | 63% |
| hampton-5.2-shape-demo (fit) | 411 | 156 | 58 | 92 | 95.52 | 96.55 | 95.82 | 2 | 0 | 0 | 45.9 | 6.7 | 52.6 | $0.1889 | $0.3073 | 58% |
| perspective-14e-boxes-critique (fit) | 1146 | 662 | 313 | 339 | 82.53 | 88.94 | 91.83 | 5 | 0 | 0 | 82.6 | 22.7 | 105.3 | $0.6444 | $1.3357 | 81% |
| perspective-13d-critique | 1752 | 860 | 443 | 403 | 84.63 | 90.58 | 91.41 | 7 | 0 | 0 | 75.3 | 28.8 | 104.0 | $0.9121 | $1.7331 | 86% |
| hampton-5.5-crit1 | 181 | 54 | 17 | 35 | 90.50 | 89.39 | 84.42 | 1 | 0 | 0 | 19.3 | 4.0 | 23.3 | $0.0861 | $0.1065 | 27% |
| hampton-5.5-crit2 | 127 | 27 | 8 | 18 | 90.63 | 87.48 | 85.91 | 1 | 0 | 0 | 11.4 | 2.2 | 13.6 | $0.0550 | $0.0533 | 34% |
| hampton-5.5-crit3 | 137 | 38 | 12 | 26 | 81.75 | 82.48 | 79.56 | 1 | 0 | 0 | 10.3 | 2.4 | 12.7 | $0.0552 | $0.0778 | 31% |
| hampton-5.5-crit4 | 156 | 34 | 16 | 18 | 93.40 | 88.91 | 84.42 | 1 | 0 | 0 | 8.2 | 2.3 | 10.5 | $0.0511 | $0.0697 | 28% |
| hampton-5.5-crit5 | 295 | 39 | 12 | 27 | 91.49 | 91.49 | 90.14 | 2 | 0 | 0 | 11.8 | 7.3 | 19.1 | $0.0838 | $0.0799 | 60% |
| flanders-03-thematic-crit | 1309 | 755 | 465 | 278 | 79.02 | 80.38 | 80.51 | 5 | 0 | 0 | 66.2 | 32.8 | 99.0 | $0.5951 | $1.5221 | 81% |
| anatomy-30b-hamstring-crit | 951 | 531 | 269 | 243 | 85.81 | 89.75 | 90.05 | 4 | 0 | 0 | 74.5 | 25.4 | 99.9 | $0.5128 | $1.0489 | 77% |
| colman-04.03-life-crit | 495 | 235 | 87 | 140 | 77.33 | 77.33 | 75.17 | 2 | 0 | 0 | 53.4 | 10.2 | 63.7 | $0.2349 | $0.4650 | 56% |
| colman-05.02-master-studies-crit | 381 | 210 | 92 | 118 | 69.32 | 71.21 | 71.05 | 2 | 0 | 0 | 77.5 | 10.5 | 88.1 | $0.2582 | $0.4302 | 57% |
| colman-06.06-species-crit | 373 | 237 | 93 | 142 | 77.37 | 78.74 | 78.63 | 2 | 0 | 0 | 57.3 | 6.1 | 63.4 | $0.2155 | $0.4814 | 58% |
| hampton-7-conclusion | 43 | 16 | 4 | 12 | 81.63 | 83.95 | 83.95 | 1 | 0 | 0 | 4.5 | 2.7 | 7.2 | $0.0220 | $0.0328 | 57% |

## Cost, caching and failures

43 requests over 43 groups, 0 retries, 0 errored attempts, 3 malformed or incomplete answers, 0 refusals, 0 targets left on f3's decision after the re-ask. Routed sentences the v3 retake pass had already cut were not sent: 142 of 4,472. A fresh 50% run would have sent 4,330 targets in 64 groups.

This run paid $4.4372 by the router's accounting ($4.4372 at list rates) for 2,166 new sentences, $0.00205 per sentence against step 5's $0.00227. Per episode $0.2465 mean, $0.9121 max. A fresh 50% run at this run's per-sentence cost: $8.87 in all, $0.4928 per episode; a fresh 25% run $0.2447 per episode. The f3 combiner's own Jev spend was $0 new (a join of paid runs; its source passes cost $0.1594 per episode). The plan-mode estimate for the new asks over all 18, before the probe, was $6.6903 (4 characters per token, 2,000 output tokens per request plus 60 per target).

Input tokens 869,893: 212,581 cache writes, 650,224 cache reads (74.7% of input), 7,088 uncached; 25 of the 25 successful requests after an episode's first read the cache. Output 109,920 tokens, thinking included. Seconds per episode 51.0 mean, 105.3 max (Opus 40.5, f3 10.5); adding step 5's Opus wall clock for the reused part gives 93.0, an upper bound for a fresh 50% run since a fresh run would overlap the two. Run wall clock 729 s, files last written 2026-09-27T06:06:33+00:00.

## Reproduction check

| number | SP here | published | s/ep mean |
|---|---:|---:|---:|
| f3_combiner | 83.37 | 83.37 | n/a |
| luna_ceiling_25 | 85.22 | 85.22 | n/a |
| luna_ceiling_50 | 84.54 | 84.54 | n/a |

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input rules_prompt: `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md`, md5 afba04cbbfc9, modified 2026-09-08T05:43:31+00:00
- input ladder_reference: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`, md5 d7a26ffcf5b5, modified 2026-09-23T20:57:03+00:00
- input f3_weights: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3-weights.json`, md5 b39151db65d7, modified 2026-09-27T04:36:30+00:00
- input f3_fit_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3-fit-features.jsonl`, md5 7617714da633, modified 2026-09-27T04:07:19+00:00
- input f3_heldout_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3-heldout-features.jsonl`, md5 73ca07072f3b, modified 2026-09-27T04:07:19+00:00
- input f3_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f3.json`, md5 3041556629d3, modified 2026-09-27T05:46:34+00:00
- input f1_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1.json`, md5 20e878d65110, modified 2026-09-26T08:05:24+00:00
- input f1luna_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna.json`, md5 5a33bbc60669, modified 2026-09-26T14:50:58+00:00
- input step5_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus.json`, md5 c9798bbbf24c, modified 2026-09-27T04:31:47+00:00
- input step5_run:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-decisions.jsonl`, md5 d184b0a6a847, modified 2026-09-27T04:30:46+00:00
- input step5_run:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-requests.jsonl`, md5 2ebb03c3953c, modified 2026-09-27T04:30:45+00:00
- input step5_run:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-timing.json`, md5 3229d819f369, modified 2026-09-27T04:30:46+00:00
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
- input opus_run:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-decisions.jsonl`, md5 537aca60dbe2, modified 2026-09-27T06:06:34+00:00
- input opus_run:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-requests.jsonl`, md5 5868eadab8a0, modified 2026-09-27T06:06:33+00:00
- input opus_run:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-timing.json`, md5 f3a45eeb0772, modified 2026-09-27T06:06:34+00:00
- input opus_keep_rule: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus-m50-keeprule.json`, md5 998f90a41f1d, modified 2026-09-27T05:58:08+00:00
- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 ee6934d58943
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f3opus.json`
