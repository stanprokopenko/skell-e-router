# The f1-Opus stack: combiner decides, Opus overrides its unsure slice (developer-facing notes)

Generated 2026-09-27T04:31:47+00:00 by `scripts/jev_real/roughcut_hybrid_f1opus.py` from the run files on disk, the f1 feature files and frozen weights, the f1-Luna stack run, the archived donor ratings and the cached removal ranges. The report step makes no model calls; the run it reads cost $4.9172 by the router's accounting ($4.9172 at list rates 5.00 in, 6.25 cache write, 0.50 cache read, 25.00 out per million). Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.

Question: third pass, step 5 of the round-two design. The slice is step 2's: the f1 combiner (`q+code+v3`, keep threshold 3.00 on `5 * p_keep`) decides all 8,943 sentences of the 18 ladder episodes, and the 2,236 whose margin sits under 0.879 (the pooled bottom 25%) go to `claude-opus-5` with the same rules5 system prompt, preamble, whole transcript and answer format the Luna call used, medium effort. Only the model and the request mechanics differ: the prompt is cached, groups hold up to 80 targets instead of 40, and an episode's first group goes alone to write the cache. Two substitutions are reported: Opus's `decision` field, and the keep rule chosen on the fit six (`score>=2`), frozen to `roughcut-hybrid-f1opus-m25-keeprule.json` before any held-out call was made.

## Offline ceiling on this slice

Before any call: the archived Opus agentic ratings (`opus5-cc-agentic`, read the way `roughcut_route2_routing.py` reads its Opus donor, each file's own Neutral threshold) substituted on the same slice. The archived Opus carries partial keeps (`keep_words`), which the live call does not ask for, so the row without them is the like-for-like bound. No live calls behind these rows, so their seconds column is n/a.

| substitution on the slice | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|
| f1 combiner alone | 85.63 | 81.70 | 82.91 | 76.27 | 90.78 | n/a |
| archived Opus with its trims | 88.52 | 84.68 | 85.86 | 79.28 | 93.60 | n/a |
| archived Opus, keep/cut only (trims dropped) | 88.27 | 84.60 | 85.73 | 79.14 | 93.44 | n/a |
| archived Luna chapters (step 2's ceiling) | 87.34 | 83.70 | 84.81 | 78.72 | 92.85 | n/a |

## Pooled results

Seconds per episode are the model's wall clock (first group alone, then up to 8 in flight) plus the combiner's own per-episode time from the f1 run; dollars are the router's accounting for the model calls, with the stack total (model plus the f1 combiner's Jev calls) alongside.

| arm | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | s/ep mean | model $/ep mean | stack $/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f1 combiner alone (build B) | 85.63 | 81.70 | 82.91 | 76.27 | 90.78 | 7.9 | n/a | $0.0974 |
| f1-Luna stack, Luna `decision` (step 2) | 87.35 | 82.55 | 84.02 | 78.43 | 92.59 | 29.7 | $0.0202 | $0.1176 |
| f1-Luna stack, `score>=2` (step 2) | 87.58 | 83.30 | 84.61 | 78.66 | 92.84 | 29.7 | $0.0202 | $0.1176 |
| f1-Opus stack, Opus `decision` | 88.43 | 84.61 | 85.78 | 79.53 | 93.88 | 49.9 | $0.2732 | $0.3706 |
| f1-Opus stack, fit-chosen `score>=2` | 88.55 | 84.60 | 85.81 | 79.27 | 93.72 | 49.9 | $0.2732 | $0.3706 |
| ceiling: archived Opus, keep/cut only | 88.27 | 84.60 | 85.73 | 79.14 | 93.44 | n/a | n/a | n/a |
| ceiling: archived Opus with trims | 88.52 | 84.68 | 85.86 | 79.28 | 93.60 | n/a | n/a | n/a |

Ladder, with modules, same 18 episodes (the f1 combiner and both f1-Luna stack rows added): shipped Opus agentic 86.47, f1-Luna stack, `score>=2` 84.61, f1-Luna stack, Luna decision 84.02, best Luna chapters 83.71, Jev f1 `q+code+v3` combiner (build B) 82.91, Jev jev_a v3 (pure Jev) 80.47, Luna single call 65.36, deterministic baseline (um removal + retakes + delete silence) 63.72. Placement: Opus stack with `decision` below shipped Opus agentic, above f1-Luna stack, `score>=2` (rank 2 of 9); with `score>=2` below shipped Opus agentic, above f1-Luna stack, `score>=2` (rank 2 of 9); the keep/cut-only ceiling below shipped Opus agentic, above f1-Luna stack, `score>=2`.

Against the Luna stack: Opus with its decision field lands +1.76 SP on Luna's decision field pooled and +2.05 held out; with each stack's own frozen rule, +1.20 pooled and +1.30 held out. Of the 2.82 SP the archived keep/cut-only Opus substitution adds over the combiner, the live decision field keeps 102%.

## Every keep rule on this slice

The rule was chosen on the fit six alone (best fit-six SP, ties to the decision field, then the lower threshold) and written to disk at 2026-09-27T04:19:27+00:00, when the run on disk held the fit six and 0 held-out episodes. The other rows are for the shape of the curve.

| keep rule | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | all 18 minus decision | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| decision | 88.43 | 84.61 | 85.78 | 79.53 | 93.88 | +0.00 | 49.9 |
| score>=1 | 87.26 | 84.27 | 85.19 | 78.29 | 92.78 | -0.59 | 49.9 |
| * score>=2 | 88.55 | 84.60 | 85.81 | 79.27 | 93.72 | +0.03 | 49.9 |
| score>=3 | 88.16 | 84.35 | 85.52 | 79.41 | 93.78 | -0.26 | 49.9 |
| score>=4 | 85.30 | 79.78 | 81.48 | 77.56 | 91.77 | -4.30 | 49.9 |

## Live Opus against archived Opus and live Luna

Keep/cut on the routed sentences Opus answered. The archived Opus decision is its score against its own file's Neutral threshold; the archived Opus ran agentically on rules1 at high effort over whole episodes, so this is a different prompt as well as a different session.

| pair | answered by both | agree | live Opus right | other right | disagreements Opus right / other right | Opus keep rate | other keep rate | editor keep rate | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| live Opus vs archived Opus | 2164 | 90.16 | 79.34 | 80.87 | 90 / 123 of 213 | 57.76 | 62.06 | 64.93 | 49.9 |
| live Opus vs archived Luna chapters | 2164 | 85.72 | 79.34 | 75.32 | 198 / 111 of 309 | 57.76 | 58.92 | 64.93 | 49.9 |
| live Opus vs live Luna (step 2) | 2164 | 83.83 | 79.34 | 71.03 | 265 / 85 of 350 | 57.76 | 49.91 | 64.93 | 49.9 |

Live Opus scores land within one point of the archived Opus score on 92.65% of answered sentences.

## Where the gains come from

A flip is a routed sentence whose keep/cut changed when the model's verdict replaced the combiner's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor (kept means full or partial).

| substitution | routed | flips | right | wrong | cut to kept right / wrong | kept to cut right / wrong | agreement on slice, combiner | agreement on slice, after routing | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Opus `decision` | 2236 | 730 | 496 | 234 | 275 / 44 | 221 / 190 | 67.53 | 79.25 | 49.9 |
| Opus `score>=2` | 2236 | 716 | 494 | 222 | 291 / 68 | 203 / 154 | 67.53 | 79.70 | 49.9 |
| Luna `decision` (step 2) | 2236 | 840 | 462 | 378 | 233 / 56 | 229 / 322 | 67.53 | 71.29 | 49.9 |
| archived Opus with trims (ceiling) | 2236 | 712 | 493 | 219 | 282 / 62 | 211 / 157 | 67.53 | 79.79 | 49.9 |

## Per episode (25% routed, cutoff 0.879, medium effort)

| episode | sentences | routed | asked | SP f1 | SP Luna stack decision | SP Opus stack decision | SP Opus stack score>=2 | SP archived Opus | requests | retries | fallback targets | Opus s | first group s | f1 s | s/ep | Opus $ | Luna $ (step 2) | cache read share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo (fit) | 194 | 47 | 46 | 84.12 | 82.06 | 85.26 | 85.26 | 84.85 | 1 | 0 | 0 | 26.0 | 26.0 | 5.2 | 31.3 | $0.1021 | $0.0060 | 27% |
| hampton-5.4-assignment-demo (fit) | 300 | 58 | 57 | 91.53 | 90.53 | 91.53 | 91.53 | 92.83 | 2 | 0 | 0 | 33.9 | 26.1 | 3.9 | 37.8 | $0.1269 | $0.0072 | 62% |
| colman-03.03-muscles-crit (fit) | 303 | 81 | 81 | 77.66 | 73.76 | 74.09 | 75.08 | 76.73 | 2 | 0 | 0 | 46.9 | 37.0 | 4.3 | 51.2 | $0.1896 | $0.0114 | 49% |
| edges-7.01-intro (fit) | 389 | 123 | 91 | 84.91 | 87.15 | 88.69 | 88.69 | 87.35 | 2 | 0 | 0 | 48.4 | 30.4 | 6.3 | 54.7 | $0.1655 | $0.0109 | 63% |
| hampton-5.2-shape-demo (fit) | 411 | 61 | 58 | 96.25 | 95.26 | 96.06 | 96.93 | 97.03 | 2 | 0 | 0 | 26.8 | 10.5 | 4.9 | 31.7 | $0.1401 | $0.0121 | 58% |
| perspective-14e-boxes-critique (fit) | 1146 | 317 | 313 | 82.88 | 88.25 | 89.13 | 88.83 | 88.46 | 5 | 0 | 0 | 84.3 | 39.8 | 17.4 | 101.7 | $0.6487 | $0.0497 | 81% |
| perspective-13d-critique | 1752 | 451 | 443 | 84.61 | 87.74 | 89.91 | 89.86 | 90.29 | 8 | 0 | 0 | 80.3 | 33.2 | 20.9 | 101.2 | $1.0989 | $0.0938 | 88% |
| hampton-5.5-crit1 | 181 | 18 | 17 | 89.94 | 88.29 | 88.29 | 88.29 | 89.01 | 1 | 0 | 0 | 10.9 | 10.9 | 3.0 | 13.9 | $0.0641 | $0.0045 | 27% |
| hampton-5.5-crit2 | 127 | 8 | 8 | 89.84 | 88.27 | 88.27 | 89.06 | 89.37 | 1 | 0 | 0 | 7.4 | 7.4 | 1.5 | 8.9 | $0.0440 | $0.0012 | 34% |
| hampton-5.5-crit3 | 137 | 12 | 12 | 82.48 | 82.48 | 82.48 | 81.75 | 82.99 | 1 | 0 | 0 | 9.2 | 9.2 | 1.7 | 10.9 | $0.0491 | $0.0024 | 31% |
| hampton-5.5-crit4 | 156 | 16 | 16 | 93.40 | 88.14 | 88.27 | 88.91 | 88.59 | 1 | 0 | 0 | 6.6 | 6.6 | 1.5 | 8.1 | $0.0489 | $0.0029 | 28% |
| hampton-5.5-crit5 | 295 | 12 | 12 | 91.49 | 90.81 | 91.49 | 91.49 | 91.63 | 2 | 0 | 0 | 10.2 | 8.1 | 5.9 | 16.1 | $0.0788 | $0.0048 | 60% |
| flanders-03-thematic-crit | 1309 | 472 | 465 | 76.95 | 75.41 | 79.74 | 80.11 | 78.98 | 6 | 0 | 0 | 92.8 | 42.5 | 24.7 | 117.6 | $0.9051 | $0.0784 | 84% |
| anatomy-30b-hamstring-crit | 951 | 279 | 269 | 85.16 | 86.78 | 89.03 | 88.46 | 89.13 | 4 | 0 | 0 | 103.7 | 52.1 | 18.0 | 121.7 | $0.5970 | $0.0391 | 77% |
| colman-04.03-life-crit | 495 | 90 | 87 | 76.93 | 76.28 | 77.62 | 77.62 | 77.43 | 2 | 0 | 0 | 59.0 | 27.0 | 7.9 | 66.9 | $0.2328 | $0.0143 | 56% |
| colman-05.02-master-studies-crit | 381 | 92 | 92 | 69.06 | 71.63 | 71.73 | 71.47 | 70.42 | 2 | 0 | 0 | 51.5 | 33.7 | 8.7 | 60.2 | $0.2078 | $0.0137 | 58% |
| colman-06.06-species-crit | 373 | 95 | 93 | 75.76 | 78.47 | 78.74 | 79.06 | 80.64 | 2 | 0 | 0 | 55.0 | 36.2 | 4.3 | 59.3 | $0.2022 | $0.0106 | 58% |
| hampton-7-conclusion | 43 | 4 | 4 | 79.30 | 81.63 | 83.95 | 83.95 | 87.67 | 1 | 0 | 0 | 2.8 | 2.8 | 2.2 | 5.0 | $0.0155 | $0.0007 | 57% |

## Cost, caching and failures

45 requests over 45 groups, 0 retries, 0 errored attempts, 3 malformed or incomplete answers, 0 refusals, 0 targets left on the combiner's decision after the re-ask. Routed sentences the v3 retake pass had already cut were not sent: 72 of 2,236.

Input tokens 938,847: 212,581 cache writes, 719,058 cache reads (76.6% of input), 7,208 uncached. 27 of the 27 successful requests after an episode's first read the cache. Input cost $1.7242 against $4.6942 had nothing been cached. Output 127,718 tokens (thinking included), $3.1930 of the $4.9172. Per episode $0.2732 mean, $1.0989 max, against Luna's $0.0202; stack per episode, Opus plus the f1 combiner, $0.3706. The plan estimate was $0.2740. Seconds per episode 49.9 mean, 121.7 max (Opus 42.0, f1 7.9; Luna was 21.8). Run wall clock 757 s, files last written 2026-09-27T04:30:45+00:00.

## How the call was made

`skell_e_router.ask_ai('claude-opus-5', ...)` on the direct Anthropic path with `enable_caching=True` and `reasoning_effort='medium'` (adaptive thinking), `max_tokens` 32,000. System message: the rules5 prompt read from `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md` at run time (md5 afba04cbbfc9), which the router marks with a cache breakpoint. User message: two text blocks, the `hybrid-preamble-v1` preamble plus the whole transcript as Jev's sentence pass rendered it (caller breakpoint, so the router adds no last-message one), then the target ids. The concatenated text is byte for byte what Luna received. Groups of up to 80 targets, closing early past 240 sentences (Luna's 40 / 120 ratio). An episode's first group goes alone; the rest go 8 in flight once it returns. 3 attempts on a transport error, 1 re-ask on a refused, malformed or incomplete answer, then the combiner's decision stands. Every attempt is a row in the requests file with the raw answer, the parsed verdicts, token counts split by cache write and read, and cost.

## Reproduction check

| number | SP here | published | s/ep mean |
|---|---:|---:|---:|
| f1_combiner | 82.91 | 82.91 | n/a |
| luna_ceiling | 84.81 | 84.81 | n/a |
| opus_ceiling | 85.86 | 85.86 | n/a |
| luna_stack_decision | 84.02 | 84.02 | n/a |
| luna_stack_score>=2 | 84.61 | 84.61 | n/a |

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input rules_prompt: `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md`, md5 afba04cbbfc9, modified 2026-09-08T05:43:31+00:00
- input ladder_reference: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`, md5 d7a26ffcf5b5, modified 2026-09-23T20:57:03+00:00
- input f1_weights: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-weights.json`, md5 242943da549e, modified 2026-09-26T07:48:20+00:00
- input f1_fit_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-fit-features.jsonl`, md5 bca3b191584a, modified 2026-09-26T07:35:32+00:00
- input f1_heldout_features: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1-heldout-features.jsonl`, md5 37c568d73d2c, modified 2026-09-26T07:49:47+00:00
- input f1_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-f1.json`, md5 20e878d65110, modified 2026-09-26T08:05:24+00:00
- input f1luna_writeup_json: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna.json`, md5 5a33bbc60669, modified 2026-09-26T14:50:58+00:00
- input f1luna_m25:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna-m25-decisions.jsonl`, md5 69aa9f944845, modified 2026-09-26T14:47:26+00:00
- input f1luna_m25:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna-m25-requests.jsonl`, md5 de899087b739, modified 2026-09-26T14:47:26+00:00
- input f1luna_m25:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1luna-m25-timing.json`, md5 53b4ae7bc293, modified 2026-09-26T14:47:26+00:00
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
- input opus_run:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-decisions.jsonl`, md5 d184b0a6a847, modified 2026-09-27T04:30:46+00:00
- input opus_run:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-requests.jsonl`, md5 2ebb03c3953c, modified 2026-09-27T04:30:45+00:00
- input opus_run:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-timing.json`, md5 3229d819f369, modified 2026-09-27T04:30:46+00:00
- input opus_keep_rule: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus-m25-keeprule.json`, md5 c65bfac2fcde, modified 2026-09-27T04:19:27+00:00
- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 ee6934d58943
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-f1opus.json`
