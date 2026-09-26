# Route 2 routing ceiling (developer-facing notes)

Generated 2026-09-26T07:03:23+00:00 by `scripts/jev_real/roughcut_route2_routing.py` from stored decisions, the archived donor ratings and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.

Question: Jev scores all 8,943 sentences of the 18 ladder episodes, the least confident slice goes to a bigger model, and that model's keep/cut and trims replace Jev's on the slice. The ceiling is estimated by swapping in the donor's archived per-sentence decision on the routed slice and rescoring. Jev sentences are the `jev_a` rows of `roughcut-jev-all18-v3` rebuilt at trim trigger 0.3 and scored at keep threshold 2.50, the setting behind the published 80.47. A donor sentence is kept when its archived score clears its own file's Neutral threshold, carries its own `keep_words`, and takes the retake flags the donor's published cut used (corpus flags plus any `retake_overrides`). The threshold is not recalibrated after mixing.

## Reproduction check

| arm | SENTENCE POINTS here | published |
|---|---:|---:|
| pure Jev (0%) | 80.47 | 80.47 |
| pure donor (100%), Luna chapters (gpt-5.6-luna xhigh, rules5) | 83.71 | 83.71 |
| pure donor (100%), Opus agentic (claude-opus-5-high, Claude Code, rules1) | 86.47 | 86.47 |

Each donor's published number uses the Neutral keep threshold of its own result file, and those files hold one episode each (the Opus hampton-5.5 file holds five), so the threshold was picked per episode against the editor. Jev gets one pooled 2.50 on all 18. For a like-for-like ceiling every donor also appears below as `<donor>@pooled`: its raw scores recalibrated to one pooled threshold across the 18 with the harness's own sweep, everything else unchanged.

| donor | pooled threshold | pure donor SP at it |
|---|---:|---:|
| Luna chapters (gpt-5.6-luna xhigh, rules5) | 0.1 | 83.55 |
| Opus agentic (claude-opus-5-high, Claude Code, rules1) | 2.1 | 86.15 |

## Sweep, pooled over all 18

Pooled SENTENCE POINTS for every confidence definition and selection mode. Confidence definitions: `top_mass` is top-level probability mass (max of the six score_probabilities); `margin` is distance of the score from the 2.50 threshold, abs(score - 2.5). Selection: `per_episode` routes the bottom X% of each episode; `pooled` routes everything under one global confidence cutoff that routes X% of all 8,943 sentences.

### Donor: Luna chapters (gpt-5.6-luna xhigh, rules5)

| routed share | top_mass/per_episode | top_mass/pooled | margin/per_episode | margin/pooled |
|---|---:|---:|---:|---:|
| 0% | 80.47 | 80.47 | 80.47 | 80.47 |
| 10% | 81.68 | 81.78 | 82.09 | 82.26 |
| 20% | 82.32 | 82.35 | 83.08 | 83.50 |
| 25% | 82.38 | 82.54 | 83.55 | 84.06 |
| 30% | 82.65 | 82.79 | 83.64 | 83.96 |
| 40% | 82.95 | 83.27 | 83.84 | 84.06 |
| 50% | 83.25 | 83.31 | 84.02 | 84.27 |
| 75% | 83.58 | 83.76 | 83.91 | 83.84 |
| 100% | 83.71 | 83.71 | 83.71 | 83.71 |

Best variant by mean pooled SP over the 10% to 75% shares: `margin/pooled`.

### Donor: Opus agentic (claude-opus-5-high, Claude Code, rules1)

| routed share | top_mass/per_episode | top_mass/pooled | margin/per_episode | margin/pooled |
|---|---:|---:|---:|---:|
| 0% | 80.47 | 80.47 | 80.47 | 80.47 |
| 10% | 81.97 | 81.99 | 82.37 | 82.70 |
| 20% | 83.08 | 83.13 | 83.99 | 84.37 |
| 25% | 83.48 | 83.57 | 84.60 | 85.03 |
| 30% | 83.85 | 83.99 | 84.85 | 85.31 |
| 40% | 84.30 | 84.67 | 85.59 | 85.74 |
| 50% | 84.78 | 84.84 | 85.97 | 86.21 |
| 75% | 85.88 | 86.03 | 86.59 | 86.51 |
| 100% | 86.47 | 86.47 | 86.47 | 86.47 |

Best variant by mean pooled SP over the 10% to 75% shares: `margin/pooled`.

### Donor: Luna chapters (gpt-5.6-luna xhigh, rules5), one pooled threshold 0.1

| routed share | top_mass/per_episode | top_mass/pooled | margin/per_episode | margin/pooled |
|---|---:|---:|---:|---:|
| 0% | 80.47 | 80.47 | 80.47 | 80.47 |
| 10% | 81.65 | 81.73 | 81.96 | 82.16 |
| 20% | 82.28 | 82.29 | 82.88 | 83.30 |
| 25% | 82.30 | 82.45 | 83.32 | 83.74 |
| 30% | 82.55 | 82.68 | 83.43 | 83.70 |
| 40% | 82.89 | 83.16 | 83.61 | 83.84 |
| 50% | 83.19 | 83.20 | 83.86 | 84.08 |
| 75% | 83.44 | 83.58 | 83.79 | 83.73 |
| 100% | 83.55 | 83.55 | 83.55 | 83.55 |

Best variant by mean pooled SP over the 10% to 75% shares: `margin/pooled`.

### Donor: Opus agentic (claude-opus-5-high, Claude Code, rules1), one pooled threshold 2.1

| routed share | top_mass/per_episode | top_mass/pooled | margin/per_episode | margin/pooled |
|---|---:|---:|---:|---:|
| 0% | 80.47 | 80.47 | 80.47 | 80.47 |
| 10% | 81.92 | 81.95 | 82.36 | 82.70 |
| 20% | 83.01 | 83.06 | 83.95 | 84.34 |
| 25% | 83.40 | 83.51 | 84.51 | 84.97 |
| 30% | 83.77 | 83.92 | 84.75 | 85.22 |
| 40% | 84.18 | 84.60 | 85.43 | 85.60 |
| 50% | 84.56 | 84.72 | 85.76 | 86.02 |
| 75% | 85.61 | 85.80 | 86.34 | 86.27 |
| 100% | 86.15 | 86.15 | 86.15 | 86.15 |

Best variant by mean pooled SP over the 10% to 75% shares: `margin/pooled`.

## Best variant in full, with the ladder

Ladder, with modules, same 18 episodes: shipped Opus agentic 86.47, best Luna chapters 83.71, Jev jev_a v3 (pure Jev) 80.47, Luna single call 65.36, deterministic baseline (um removal + retakes + delete silence) 63.72.

### Luna chapters (gpt-5.6-luna xhigh, rules5), `margin/pooled`

| share | routed | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | ladder rank | placement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 83.00 | 79.35 | 80.47 | 74.12 | 88.44 | 3 of 5 | level with Jev jev_a v3 (pure Jev) |
| 10% | 894 | 84.76 | 81.15 | 82.26 | 75.63 | 89.83 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 20% | 1789 | 86.30 | 82.27 | 83.50 | 77.30 | 91.21 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 25% | 2236 | 86.84 | 82.83 | 84.06 | 77.82 | 91.61 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 30% | 2683 | 86.77 | 82.72 | 83.96 | 78.16 | 91.94 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 40% | 3577 | 86.61 | 82.93 | 84.06 | 78.48 | 92.29 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 50% | 4472 | 86.57 | 83.25 | 84.27 | 78.86 | 92.62 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 75% | 6707 | 86.32 | 82.74 | 83.84 | 79.63 | 93.12 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 100% | 8943 | 86.86 | 82.31 | 83.71 | 80.35 | 93.55 | 2 of 5 | level with best Luna chapters |

### Opus agentic (claude-opus-5-high, Claude Code, rules1), `margin/pooled`

| share | routed | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | ladder rank | placement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 83.00 | 79.35 | 80.47 | 74.12 | 88.44 | 3 of 5 | level with Jev jev_a v3 (pure Jev) |
| 10% | 894 | 85.11 | 81.64 | 82.70 | 76.02 | 90.23 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 20% | 1789 | 87.01 | 83.20 | 84.37 | 77.36 | 91.57 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 25% | 2236 | 87.71 | 83.85 | 85.03 | 77.93 | 92.04 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 30% | 2683 | 88.03 | 84.11 | 85.31 | 78.40 | 92.54 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 40% | 3577 | 88.53 | 84.50 | 85.74 | 78.97 | 93.12 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 50% | 4472 | 88.84 | 85.04 | 86.21 | 79.70 | 93.69 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 75% | 6707 | 89.18 | 85.34 | 86.51 | 80.80 | 94.56 | 1 of 6 | top, above shipped Opus agentic |
| 100% | 8943 | 89.49 | 85.14 | 86.47 | 81.44 | 95.02 | 1 of 5 | level with shipped Opus agentic |

### Luna chapters (gpt-5.6-luna xhigh, rules5), one pooled threshold 0.1, `margin/pooled`

| share | routed | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | ladder rank | placement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 83.00 | 79.35 | 80.47 | 74.12 | 88.44 | 3 of 5 | level with Jev jev_a v3 (pure Jev) |
| 10% | 894 | 84.62 | 81.07 | 82.16 | 75.56 | 89.79 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 20% | 1789 | 86.06 | 82.08 | 83.30 | 77.06 | 91.07 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 25% | 2236 | 86.53 | 82.51 | 83.74 | 77.53 | 91.43 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 30% | 2683 | 86.65 | 82.39 | 83.70 | 77.88 | 91.78 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 40% | 3577 | 86.59 | 82.62 | 83.84 | 78.20 | 92.11 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 50% | 4472 | 86.61 | 82.97 | 84.08 | 78.58 | 92.45 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 75% | 6707 | 86.39 | 82.56 | 83.73 | 79.24 | 92.90 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 100% | 8943 | 86.81 | 82.12 | 83.55 | 79.91 | 93.30 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |

### Opus agentic (claude-opus-5-high, Claude Code, rules1), one pooled threshold 2.1, `margin/pooled`

| share | routed | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | ladder rank | placement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 83.00 | 79.35 | 80.47 | 74.12 | 88.44 | 3 of 5 | level with Jev jev_a v3 (pure Jev) |
| 10% | 894 | 85.14 | 81.62 | 82.70 | 76.02 | 90.23 | 3 of 6 | below best Luna chapters, above Jev jev_a v3 (pure Jev) |
| 20% | 1789 | 86.99 | 83.16 | 84.34 | 77.37 | 91.56 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 25% | 2236 | 87.62 | 83.80 | 84.97 | 77.92 | 92.02 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 30% | 2683 | 87.84 | 84.06 | 85.22 | 78.38 | 92.50 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 40% | 3577 | 88.21 | 84.44 | 85.60 | 78.94 | 93.06 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 50% | 4472 | 88.49 | 84.93 | 86.02 | 79.63 | 93.58 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 75% | 6707 | 88.82 | 85.14 | 86.27 | 80.77 | 94.45 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |
| 100% | 8943 | 89.07 | 84.87 | 86.15 | 81.33 | 94.86 | 2 of 6 | below shipped Opus agentic, above best Luna chapters |

## Per-episode or pooled selection

| donor | confidence | share | SP per-episode | SP pooled | pooled minus per-episode | global cutoff |
|---|---:|---:|---:|---:|---:|---:|
| luna | margin | 10% | 82.09 | 82.26 | +0.17 | 0.200 |
| luna | margin | 20% | 83.08 | 83.50 | +0.43 | 0.380 |
| luna | margin | 25% | 83.55 | 84.06 | +0.51 | 0.460 |
| luna | margin | 30% | 83.64 | 83.96 | +0.32 | 0.540 |
| luna | margin | 40% | 83.84 | 84.06 | +0.23 | 0.680 |
| luna | margin | 50% | 84.02 | 84.27 | +0.25 | 0.800 |
| luna | margin | 75% | 83.91 | 83.84 | -0.07 | 1.130 |
| opus | margin | 10% | 82.37 | 82.70 | +0.33 | 0.200 |
| opus | margin | 20% | 83.99 | 84.37 | +0.38 | 0.380 |
| opus | margin | 25% | 84.60 | 85.03 | +0.43 | 0.460 |
| opus | margin | 30% | 84.85 | 85.31 | +0.46 | 0.540 |
| opus | margin | 40% | 85.59 | 85.74 | +0.15 | 0.680 |
| opus | margin | 50% | 85.97 | 86.21 | +0.24 | 0.800 |
| opus | margin | 75% | 86.59 | 86.51 | -0.08 | 1.130 |

Pooled selection routes more of the episodes where Jev is least sure overall and fewer of the ones it finds easy, so the per-episode share varies. The per-episode SP for every row is in the JSON under `sweeps`.

## Where the gains come from, 25% routed

A flip is a routed sentence whose keep/cut changed when the donor's decision replaced Jev's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor (kept means full or partial). Trim changed counts routed sentences both sides kept but trimmed differently.

| donor/confidence/selection | routed | flips | right | wrong | cut to kept right / wrong | kept to cut right / wrong | trim changed | agreement on slice, Jev | agreement on slice, after routing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| luna/margin/per_episode | 2237 | 759 | 516 | 243 | 212 / 63 | 304 / 180 | 116 | 67.19 | 79.39 |
| luna/margin/pooled | 2236 | 777 | 543 | 234 | 217 / 69 | 326 / 165 | 96 | 65.83 | 79.65 |
| opus/margin/per_episode | 2237 | 724 | 545 | 179 | 239 / 68 | 306 / 111 | 192 | 67.19 | 83.55 |
| opus/margin/pooled | 2236 | 760 | 579 | 181 | 246 / 75 | 333 / 106 | 151 | 65.83 | 83.63 |
| luna@pooled/margin/per_episode | 2237 | 743 | 498 | 245 | 213 / 76 | 285 / 169 | 116 | 67.19 | 78.50 |
| luna@pooled/margin/pooled | 2236 | 764 | 523 | 241 | 218 / 86 | 305 / 155 | 96 | 65.83 | 78.44 |
| opus@pooled/margin/per_episode | 2237 | 734 | 545 | 189 | 238 / 66 | 307 / 123 | 192 | 67.19 | 83.10 |
| opus@pooled/margin/pooled | 2236 | 767 | 579 | 188 | 245 / 73 | 334 / 115 | 151 | 65.83 | 83.32 |

Agreement with the editor on the routed slice, Jev against the donor, for the best variant at every share (read from each arm's full-run states):

| donor | share | routed | Jev agrees | donor agrees |
|---|---:|---:|---:|---:|
| luna | 10% | 894 | 59.84 | 77.63 |
| luna | 20% | 1789 | 63.83 | 78.87 |
| luna | 25% | 2236 | 65.83 | 79.65 |
| luna | 30% | 2683 | 68.06 | 79.50 |
| luna | 40% | 3577 | 71.74 | 80.37 |
| luna | 50% | 4472 | 74.17 | 81.40 |
| luna | 75% | 6707 | 78.83 | 83.14 |
| luna | 100% | 8943 | 82.60 | 85.79 |
| opus | 10% | 894 | 59.84 | 83.11 |
| opus | 20% | 1789 | 63.83 | 83.40 |
| opus | 25% | 2236 | 65.83 | 83.63 |
| opus | 30% | 2683 | 68.06 | 83.64 |
| opus | 40% | 3577 | 71.74 | 84.46 |
| opus | 50% | 4472 | 74.17 | 85.06 |
| opus | 75% | 6707 | 78.83 | 86.58 |
| opus | 100% | 8943 | 82.60 | 88.54 |

## Cost and latency

Assumptions, for a windowed Luna call on the routed sentences: routed ids are taken in transcript order and grouped greedily into windows of up to 40 routed sentences, a window closing early once it spans 80 sentences; each request sends the whole span plus 5 sentences of context either side, at 1.3 tokens per word, plus 1,500 tokens of prompt (the rules v5 system prompt is 1,047 words). Output is 60 tokens per routed sentence (a score and optional keep_words, low reasoning effort); the heavy column instead uses the archived Luna chapters run's 416 completion tokens per sentence, which is xhigh reasoning. Price $0.20 in and $1.20 out per million tokens, the gpt-5.6-luna rates `routing-notes.md` used (the router's model table carries no price for gpt-5.6-luna today), no cache discount. Latency is Jev's ~6 s plus 6 s per wave of 8 concurrent Luna requests. Selection uses the best Luna confidence definition, `margin`.

| share | selection | sentences/ep mean | max | requests/ep mean | max | input tokens/ep mean | Luna $/ep mean | max | heavy-output $/ep mean | s/ep mean | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | per_episode | 0 | 0 | 0.0 | 0 | 0.0k | $0.0000 | $0.0000 | $0.0000 | 6 | 6 |
| 0% | pooled | 0 | 0 | 0.0 | 0 | 0.0k | $0.0000 | $0.0000 | $0.0000 | 6 | 6 |
| 10% | per_episode | 50 | 175 | 5.5 | 18 | 13.2k | $0.0062 | $0.0210 | $0.0275 | 14 | 24 |
| 10% | pooled | 50 | 236 | 5.4 | 18 | 12.9k | $0.0061 | $0.0256 | $0.0274 | 14 | 24 |
| 20% | per_episode | 99 | 350 | 6.1 | 19 | 15.2k | $0.0102 | $0.0344 | $0.0526 | 14 | 24 |
| 20% | pooled | 99 | 480 | 5.9 | 20 | 14.7k | $0.0101 | $0.0442 | $0.0526 | 14 | 24 |
| 25% | per_episode | 124 | 438 | 6.1 | 19 | 15.4k | $0.0120 | $0.0408 | $0.0651 | 14 | 24 |
| 25% | pooled | 124 | 600 | 6.1 | 21 | 15.4k | $0.0120 | $0.0533 | $0.0651 | 14 | 24 |
| 30% | per_episode | 149 | 526 | 6.2 | 20 | 15.8k | $0.0139 | $0.0476 | $0.0776 | 14 | 24 |
| 30% | pooled | 149 | 699 | 6.3 | 23 | 15.8k | $0.0139 | $0.0611 | $0.0776 | 14 | 24 |
| 40% | per_episode | 199 | 701 | 6.6 | 23 | 16.7k | $0.0176 | $0.0612 | $0.1026 | 14 | 24 |
| 40% | pooled | 199 | 859 | 6.7 | 25 | 16.8k | $0.0177 | $0.0734 | $0.1026 | 14 | 30 |
| 50% | per_episode | 249 | 876 | 7.3 | 25 | 18.0k | $0.0215 | $0.0747 | $0.1278 | 14 | 30 |
| 50% | pooled | 248 | 984 | 7.3 | 27 | 18.1k | $0.0215 | $0.0831 | $0.1277 | 14 | 30 |
| 75% | per_episode | 373 | 1314 | 9.8 | 34 | 22.2k | $0.0313 | $0.1092 | $0.1906 | 16 | 36 |
| 75% | pooled | 373 | 1311 | 9.8 | 33 | 22.3k | $0.0313 | $0.1087 | $0.1905 | 16 | 36 |
| 100% | per_episode | 497 | 1752 | 12.9 | 44 | 27.4k | $0.0413 | $0.1440 | $0.2536 | 18 | 42 |
| 100% | pooled | 497 | 1752 | 12.9 | 44 | 27.4k | $0.0413 | $0.1440 | $0.2536 | 18 | 42 |

Jev alone on this run: $0.0423 and 5.6 s per episode on average; add it to every row for the hybrid total.

Sanity bound from the archived donor runs themselves (whole-episode agentic sessions, not windowed calls):

| donor | $/ep mean | $/ep max | s/ep mean | s/ep max | completion tokens/sentence |
|---|---:|---:|---:|---:|---:|
| Luna chapters (gpt-5.6-luna xhigh, rules5) | $0.308 | $0.949 | 895 | 2009 | 416 |
| Opus agentic (claude-opus-5-high, Claude Code, rules1) | $5.151 | $15.621 | n/a | n/a | n/a |

Seconds are the session wall clock; where one archived file covers several episodes (the Opus hampton-5.5 crits) its wall clock is split evenly across them. Opus cost is the registry's estimated basis, not a bill.

## Limits of this estimate

The donor decisions come from whole-episode agentic runs that read the full transcript, reviewed their own ratings, and were scored at a Neutral threshold calibrated per result file (luna: 0.1 on 10, 1.1 on 3, 2.1 on 5; opus: 0.1 on 2, 1.1 on 8, 2.1 on 8 episodes). A windowed call on 40 routed sentences with 5 lines of context sees far less, and nothing here says it would reproduce those decisions. Treat every routed number as an upper bound on what the same model gives on a window.

The per-file thresholds also flatter the donors against Jev, which uses one pooled 2.50 on every episode.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input ladder_reference: `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.json`, md5 d7a26ffcf5b5, modified 2026-09-23T20:57:03+00:00
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
- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 ee6934d58943
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-route2-routing.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-route2-routing.json`
