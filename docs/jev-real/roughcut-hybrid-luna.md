# Real Luna routing on Jev's unsure slice (developer-facing notes)

Generated 2026-09-26T14:48:38+00:00 by `scripts/jev_real/roughcut_hybrid_luna.py` from the hybrid run files on disk, the stored Jev decisions, the archived donor ratings and the cached removal ranges. The report step itself makes no model calls; the runs it reads cost $0.9277 by the router's accounting ($0.9277 at list rates 0.20 in, 0.02 cached, 1.20 out per million), plus $0.0014 for the one-group smoke request in `smoke-hybrid-luna-*`. Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.

Question: route 2 for real. Jev's `jev_a` v3 rows score all 8,943 sentences of the 18 ladder episodes; the sentences whose score sits closest to the 2.50 keep threshold (margin `abs(score - 2.5)` under a global cutoff) go to `gpt-5.6-luna`, which reads the whole episode transcript as Jev saw it under the rules5 system prompt and returns a score, a keep or cut decision and a reason for each routed sentence. On routed sentences Luna's decision replaces Jev's keep/cut (score 5 or 0), `keep_words` is dropped, Jev's retake veto stays; everything else is Jev's decision rebuilt at trim trigger 0.3 and keep threshold 2.50, the setting behind the published 80.47. The cutoffs were chosen from the offline sweep over all 18 episodes (`roughcut-route2-routing.md`), so the routed share is not held out; the model's decisions on the slice are. Ties at the cutoff are broken by episode order and sentence id the way the sweep broke them, so the slice is the one the ceiling was computed on.

## Reproduction check

| arm | SENTENCE POINTS here | published |
|---|---:|---:|
| pure Jev (jev_a v3) | 80.47 | 80.47 |
| pure archived Luna chapters (routed 100%) | 83.71 | 83.71 |
| pure archived Opus agentic (routed 100%) | 86.47 | 86.47 |

## Pooled results

Pooled over the fit six, the held-out 12 and all 18, with modules. `ceiling` is the same routed slice substituted with the archived Luna chapters decision (whole-episode agentic run, per-file Neutral threshold), the number `roughcut-route2-routing.md` reported as the upper bound. Seconds per episode are Luna's wall clock at concurrency 8 plus Jev's own per-episode wall clock from the v3 run; dollars are the router's usage accounting summed per episode, Luna only.

| run | routed | SP fit 6 | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | ceiling SP all 18 | ladder rank | s/ep mean | s/ep max | Luna $/ep mean | $/ep max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m046 (25% routed, cutoff 0.46, medium effort) | 2236 | 86.91 | 82.17 | 83.63 | 77.35 | 91.28 | 84.06 | 3 of 6 | 26.6 | 64.5 | $0.0201 | $0.1081 |
| m080 (50% routed, cutoff 0.80, medium effort) | 4472 | 87.31 | 82.09 | 83.69 | 78.50 | 92.30 | 84.27 | 3 of 6 | 32.2 | 67.4 | $0.0315 | $0.1681 |

Ladder, with modules, same 18 episodes: shipped Opus agentic 86.47, best Luna chapters 83.71, Jev jev_a v3 (pure Jev) 80.47, Luna single call 65.36, deterministic baseline (um removal + retakes + delete silence) 63.72. Placement: m046 below best Luna chapters, above Jev jev_a v3 (pure Jev) (ceiling on the same slice: below shipped Opus agentic, above best Luna chapters); m080 below best Luna chapters, above Jev jev_a v3 (pure Jev) (ceiling on the same slice: below shipped Opus agentic, above best Luna chapters).

High-effort trigger: medium effort at 25% routed sits 0.43 SP under the 84.06 ceiling; the spec reruns at high effort when the gap is over 1.5 SP, so the rerun was not required and did not run.

How much of the ceiling survives the live call, as a share of the gain the archived substitution makes over pure Jev: m046 88% (83.63 of 84.06 against 80.47); m080 85% (83.69 of 84.27 against 80.47).

## Score sweep on the routed slice

Luna's 0-5 score is stored next to its decision, so the routed slice can also be cut at a score threshold instead of the decision field. `decision` is the number above; `score >= t` keeps a routed sentence when Luna's score clears t. Pooled 18, with modules.

| run | SP decision | SP score >= 1 | SP score >= 2 | SP score >= 3 | SP score >= 4 | SP ceiling | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| m046 (25% routed, cutoff 0.46, medium effort) | 83.63 | 83.45 | 84.05 | 83.46 | 80.87 | 84.06 | 26.6 |
| m080 (50% routed, cutoff 0.80, medium effort) | 83.69 | 84.05 | 84.61 | 83.17 | 75.94 | 84.27 | 32.2 |

Best score threshold per run: m046 keeps at score >= 2 for 84.05, +0.42 on the decision field and -0.01 on the archived ceiling; m080 keeps at score >= 2 for 84.61, +0.92 on the decision field and +0.34 on the archived ceiling. Luna's own keep/cut decision is cut-heavier than the editor on this slice (keep rates in the next table), so keeping anything it scores 2 or more recovers part of that. The threshold is picked on the same 18 episodes it is reported on, so read it as the shape of the curve, not a held-out number; the decision-field SP above is the number this build set out to measure. The next section redoes the choice with the fit/held-out split.

## Keep rule, held out (second pass, step 1)

Same runs, same stored answers, $0. Each keep rule (Luna's `decision` field, or keep when Luna's score clears 1, 2, 3 or 4) is scored on the six fit episodes first; the best fit-six SENTENCE POINTS is frozen (ties go to the decision field, then to the lower threshold), and only then are the 12 held-out episodes and the pooled 18 read under that rule. The fit-six column is where the choice was made and is not held out; the held-out 12 column is. Pooled 18 mixes the two. The chosen row of each run is marked with `*`. Seconds per episode are the run's Luna wall clock plus Jev's, as above; the rule changes nothing about the call.

### m046 (25% routed, cutoff 0.46, medium effort)

| keep rule | SP fit 6 (chosen on) | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | held-out minus decision | all 18 minus decision | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decision | 86.91 | 82.17 | 83.63 | 77.35 | 91.28 | +0.00 | +0.00 | 26.6 |
| score>=1 | 85.91 | 82.36 | 83.45 | 76.63 | 90.81 | +0.19 | -0.18 | 26.6 |
| * score>=2 | 86.92 | 82.77 | 84.05 | 77.47 | 91.44 | +0.60 | +0.42 | 26.6 |
| score>=3 | 86.80 | 81.99 | 83.46 | 77.24 | 91.17 | -0.18 | -0.16 | 26.6 |
| score>=4 | 84.59 | 79.22 | 80.87 | 75.55 | 89.45 | -2.95 | -2.76 | 26.6 |

### m080 (50% routed, cutoff 0.80, medium effort)

| keep rule | SP fit 6 (chosen on) | SP held-out 12 | SP all 18 | WORD all 18 | GRADE all 18 | held-out minus decision | all 18 minus decision | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decision | 87.31 | 82.09 | 83.69 | 78.50 | 92.30 | +0.00 | +0.00 | 32.2 |
| score>=1 | 87.36 | 82.58 | 84.05 | 77.77 | 91.94 | +0.49 | +0.36 | 32.2 |
| * score>=2 | 88.04 | 83.09 | 84.61 | 78.67 | 92.64 | +1.00 | +0.92 | 32.2 |
| score>=3 | 86.38 | 81.75 | 83.17 | 78.38 | 92.16 | -0.34 | -0.52 | 32.2 |
| score>=4 | 79.01 | 74.59 | 75.94 | 74.33 | 87.61 | -7.50 | -7.75 | 32.2 |

Result: m046 chooses `score>=2` on the fit six (86.92 against 86.91 for the decision field, a +0.01 margin, close to a tie on its own); held out it gives 82.77 against 82.17, pooled 18 84.05 against 83.63, ceiling on the slice 84.06; m080 chooses `score>=2` on the fit six (88.04 against 87.31 for the decision field, a +0.73 margin); held out it gives 83.09 against 82.09, pooled 18 84.61 against 83.69, ceiling on the slice 84.27. Frozen rule for the second pass: `score>=2`, chosen on the fit six of m046 (both runs choose the same rule). The f1-Luna stack (`roughcut-hybrid-f1luna.md`) reports its slice under the decision field and under this rule.

## Live Luna against archived Luna on the same sentences

The ceiling substituted the archived Luna chapters decision (score at the file's own Neutral threshold, 0.1 to 2.1 per episode) on the routed slice. The live run substitutes the decision the windowed call returned. Agreement is measured on the routed sentences the live run got an answer for; right means the decision matches the editor (kept means full or partial). SP columns are the pooled 18 with modules for each substitution.

| run | answered | live agrees with archived | live right | archived right | Jev right | disagreements live right / archived right | live keep & archived cut / live cut & archived keep | live keep rate | archived keep rate | editor keep rate | SP live | SP archived (ceiling) | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m046 (25% routed, cutoff 0.46, medium effort) | 2108 | 83.63 | 77.61 | 79.36 | 65.83 | 154 / 191 of 345 | 99 / 246 | 43.74 | 50.71 | 54.46 | 83.63 | 84.06 | 26.6 |
| m080 (50% routed, cutoff 0.80, medium effort) | 4254 | 84.72 | 79.78 | 81.66 | 74.17 | 285 / 365 of 650 | 169 / 481 | 51.32 | 58.65 | 62.04 | 83.69 | 84.27 | 32.2 |

Why the two differ where they do: m046: of the 345 disagreements the live call cuts where the archive keeps 246 times and keeps where the archive cuts 99 times, and the archive is right on 191 of them against the live call's 154; m080: of the 650 disagreements the live call cuts where the archive keeps 481 times and keeps where the archive cuts 169 times, and the archive is right on 365 of them against the live call's 285. The archived run rated every sentence in one agentic session with a review loop and was then thresholded per episode against the editor, with a 0.1 threshold on 10 of the 18 files (anything not scored 0 is kept), so it leans keep; the live call answers only the routed ids, has no review loop, and its keep/cut is the model's own call, which leans cut on exactly the sentences Jev was unsure about. The editor keeps more of this slice than either, so the cut-heavy side loses more. Live scores land within one point of the archived score on 86.20% (m046), 87.73% (m080) of the answered sentences.

## Where the gains come from

A flip is a routed sentence whose keep/cut changed when Luna's decision replaced Jev's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor. Trim changed counts routed sentences kept on both sides whose word runs differ (the live run drops Jev's trims on routed sentences, so this is mostly Jev trims that vanished). The ceiling row is the archived substitution on the same slice.

| run | routed | flips | right | wrong | cut to kept right / wrong | kept to cut right / wrong | trim changed | agreement on slice, Jev | agreement on slice, after routing | s/ep mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m046 (25% routed, cutoff 0.46, medium effort) | 2236 | 822 | 548 | 274 | 200 / 45 | 348 / 229 | 8 | 65.83 | 78.09 | 26.6 |
| m046 ceiling (archived Luna) | 2236 | 777 | 543 | 234 | 217 / 69 | 326 / 165 | 96 | 65.83 | 79.65 | 26.6 |
| m080 (50% routed, cutoff 0.80, medium effort) | 4472 | 1396 | 826 | 570 | 295 / 62 | 531 / 508 | 35 | 74.17 | 79.90 | 32.2 |
| m080 ceiling (archived Luna) | 4472 | 1213 | 768 | 445 | 305 / 94 | 463 / 351 | 276 | 74.17 | 81.40 | 32.2 |

## Per episode, m046 (25% routed, cutoff 0.46, medium effort)

| episode | sentences | routed | asked | unanswered | SP Jev | SP hybrid | SP ceiling | WORD hybrid | GRADE hybrid | requests | retries | Luna s | Jev s | s/ep | Luna $ | input tokens/request | cached share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo (fit) | 194 | 42 | 40 | 0 | 85.88 | 85.82 | 86.75 | 75.10 | 106.69 | 2 | 0 | 21.3 | 4.4 | 25.7 | $0.0053 | 5.5k | 26% |
| hampton-5.4-assignment-demo (fit) | 300 | 67 | 67 | 0 | 84.80 | 90.20 | 91.13 | 83.22 | 100.87 | 3 | 0 | 20.3 | 2.8 | 23.1 | $0.0075 | 6.1k | 23% |
| colman-03.03-muscles-crit (fit) | 303 | 58 | 57 | 0 | 75.05 | 75.61 | 75.35 | 61.77 | 75.97 | 3 | 0 | 15.3 | 3.1 | 18.4 | $0.0087 | 7.4k | 19% |
| edges-7.01-intro (fit) | 389 | 151 | 111 | 0 | 79.69 | 86.89 | 84.83 | 85.08 | 84.99 | 4 | 0 | 21.5 | 4.9 | 26.4 | $0.0102 | 5.2k | 27% |
| hampton-5.2-shape-demo (fit) | 411 | 100 | 92 | 0 | 92.70 | 91.27 | 92.43 | 84.25 | 95.37 | 4 | 0 | 16.0 | 3.2 | 19.2 | $0.0116 | 8.8k | 16% |
| perspective-14e-boxes-critique (fit) | 1146 | 286 | 272 | 0 | 81.78 | 87.67 | 87.43 | 84.78 | 87.69 | 10 | 0 | 19.9 | 12.4 | 32.4 | $0.0476 | 17.3k | 8% |
| perspective-13d-critique | 1752 | 600 | 576 | 0 | 78.09 | 85.74 | 85.30 | 84.70 | 87.68 | 16 | 0 | 37.3 | 13.6 | 50.9 | $0.1081 | 25.9k | 6% |
| hampton-5.5-crit1 | 181 | 32 | 31 | 0 | 87.35 | 86.24 | 87.90 | 68.72 | 111.89 | 2 | 0 | 14.8 | 2.2 | 17.0 | $0.0041 | 5.6k | 25% |
| hampton-5.5-crit2 | 127 | 30 | 30 | 0 | 79.45 | 77.32 | 80.71 | 59.56 | 102.86 | 2 | 0 | 21.9 | 0.9 | 22.8 | $0.0044 | 4.5k | 32% |
| hampton-5.5-crit3 | 137 | 17 | 17 | 0 | 81.17 | 81.17 | 81.53 | 73.78 | 107.91 | 2 | 0 | 16.7 | 1.0 | 17.7 | $0.0035 | 4.8k | 30% |
| hampton-5.5-crit4 | 156 | 21 | 21 | 0 | 88.46 | 88.85 | 89.23 | 66.54 | 107.23 | 2 | 0 | 15.6 | 0.8 | 16.4 | $0.0037 | 5.4k | 27% |
| hampton-5.5-crit5 | 295 | 31 | 31 | 0 | 90.88 | 90.47 | 91.80 | 73.14 | 105.89 | 3 | 0 | 8.9 | 4.8 | 13.7 | $0.0059 | 7.5k | 19% |
| flanders-03-thematic-crit | 1309 | 321 | 300 | 0 | 77.14 | 78.17 | 80.31 | 79.76 | 85.74 | 11 | 0 | 20.4 | 19.3 | 39.6 | $0.0606 | 21.2k | 7% |
| anatomy-30b-hamstring-crit | 951 | 252 | 240 | 0 | 84.90 | 87.48 | 88.08 | 84.19 | 94.09 | 8 | 0 | 19.5 | 11.8 | 31.3 | $0.0385 | 16.4k | 9% |
| colman-04.03-life-crit | 495 | 82 | 78 | 0 | 76.18 | 75.80 | 75.94 | 64.20 | 77.01 | 4 | 0 | 24.9 | 5.9 | 30.7 | $0.0158 | 10.7k | 13% |
| colman-05.02-master-studies-crit | 381 | 75 | 75 | 0 | 66.33 | 68.66 | 68.37 | 65.94 | 87.69 | 5 | 1 | 59.5 | 5.0 | 64.5 | $0.0161 | 9.1k | 33% |
| colman-06.06-species-crit | 373 | 61 | 60 | 0 | 79.57 | 78.95 | 79.52 | 69.95 | 96.37 | 3 | 0 | 16.1 | 2.8 | 18.9 | $0.0091 | 8.4k | 17% |
| hampton-7-conclusion | 43 | 10 | 10 | 0 | 72.33 | 81.63 | 82.56 | 80.70 | 92.63 | 1 | 0 | 7.9 | 1.8 | 9.7 | $0.0010 | 2.6k | 100% |

Routed sentences that Jev's retake pass had already cut were not sent (they are cut whichever way Luna would vote): 128 of 2236. 85 requests in total, 1 retries, 0 errored attempts, 1 malformed or incomplete answers, 0 targets left unanswered after retries (those keep Jev's own decision). Tokens: 1,238,585 prompt of which 130,270 cached, 114,352 completion of which 64,927 reasoning. Router cost $0.3615, list-rate cost $0.3615; the plan estimated $0.5057. Run wall clock 378 s, generated 2026-09-26T07:42:19+00:00.

## Per episode, m080 (50% routed, cutoff 0.80, medium effort)

| episode | sentences | routed | asked | unanswered | SP Jev | SP hybrid | SP ceiling | WORD hybrid | GRADE hybrid | requests | retries | Luna s | Jev s | s/ep | Luna $ | input tokens/request | cached share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo (fit) | 194 | 86 | 84 | 0 | 85.88 | 82.99 | 83.45 | 74.35 | 105.11 | 3 | 0 | 22.0 | 4.4 | 26.5 | $0.0073 | 5.5k | 26% |
| hampton-5.4-assignment-demo (fit) | 300 | 171 | 170 | 0 | 84.80 | 86.37 | 90.77 | 81.73 | 99.12 | 5 | 0 | 21.4 | 2.8 | 24.2 | $0.0129 | 6.2k | 23% |
| colman-03.03-muscles-crit (fit) | 303 | 163 | 162 | 0 | 75.05 | 71.32 | 71.49 | 63.85 | 78.12 | 5 | 0 | 29.7 | 3.1 | 32.8 | $0.0161 | 7.4k | 19% |
| edges-7.01-intro (fit) | 389 | 226 | 163 | 0 | 79.69 | 89.36 | 85.78 | 87.32 | 86.51 | 5 | 0 | 24.0 | 4.9 | 28.9 | $0.0135 | 5.2k | 27% |
| hampton-5.2-shape-demo (fit) | 411 | 214 | 203 | 0 | 92.70 | 94.50 | 92.26 | 88.82 | 99.49 | 6 | 0 | 20.2 | 3.2 | 23.4 | $0.0202 | 8.8k | 16% |
| perspective-14e-boxes-critique (fit) | 1146 | 530 | 498 | 0 | 81.78 | 89.24 | 88.21 | 85.70 | 88.37 | 13 | 0 | 32.4 | 12.4 | 44.9 | $0.0674 | 17.3k | 8% |
| perspective-13d-critique | 1752 | 984 | 945 | 0 | 78.09 | 88.70 | 88.11 | 87.90 | 91.06 | 25 | 0 | 53.8 | 13.6 | 67.4 | $0.1681 | 25.9k | 6% |
| hampton-5.5-crit1 | 181 | 90 | 88 | 0 | 87.35 | 84.48 | 87.29 | 72.51 | 113.67 | 3 | 0 | 26.6 | 2.2 | 28.9 | $0.0091 | 5.7k | 25% |
| hampton-5.5-crit2 | 127 | 57 | 56 | 0 | 79.45 | 77.01 | 81.89 | 60.25 | 103.45 | 2 | 0 | 25.1 | 0.9 | 26.0 | $0.0057 | 4.5k | 32% |
| hampton-5.5-crit3 | 137 | 62 | 62 | 0 | 81.17 | 78.69 | 80.07 | 72.72 | 105.77 | 2 | 0 | 16.6 | 1.0 | 17.6 | $0.0046 | 4.9k | 29% |
| hampton-5.5-crit4 | 156 | 60 | 60 | 0 | 88.46 | 85.26 | 84.87 | 66.62 | 107.77 | 2 | 0 | 24.3 | 0.8 | 25.1 | $0.0063 | 5.4k | 26% |
| hampton-5.5-crit5 | 295 | 118 | 118 | 0 | 90.88 | 85.97 | 89.66 | 72.16 | 105.09 | 4 | 0 | 20.9 | 4.8 | 25.7 | $0.0122 | 7.5k | 19% |
| flanders-03-thematic-crit | 1309 | 664 | 628 | 0 | 77.14 | 77.27 | 79.55 | 79.68 | 85.97 | 17 | 0 | 46.8 | 19.3 | 66.1 | $0.1009 | 21.2k | 7% |
| anatomy-30b-hamstring-crit | 951 | 506 | 485 | 0 | 84.90 | 87.47 | 87.36 | 85.45 | 95.05 | 13 | 0 | 30.7 | 11.8 | 42.4 | $0.0637 | 16.4k | 9% |
| colman-04.03-life-crit | 495 | 194 | 187 | 0 | 76.18 | 74.91 | 75.03 | 65.51 | 77.82 | 5 | 0 | 31.0 | 5.9 | 36.9 | $0.0227 | 10.7k | 13% |
| colman-05.02-master-studies-crit | 381 | 183 | 183 | 0 | 66.33 | 66.48 | 70.39 | 64.79 | 86.07 | 5 | 0 | 22.9 | 5.0 | 27.9 | $0.0204 | 9.2k | 16% |
| colman-06.06-species-crit | 373 | 146 | 145 | 0 | 79.57 | 77.75 | 80.99 | 70.02 | 95.57 | 4 | 0 | 20.2 | 2.8 | 23.1 | $0.0140 | 8.5k | 17% |
| hampton-7-conclusion | 43 | 18 | 17 | 0 | 72.33 | 76.98 | 83.02 | 82.59 | 95.19 | 1 | 0 | 10.3 | 1.8 | 12.1 | $0.0014 | 2.6k | 55% |

Routed sentences that Jev's retake pass had already cut were not sent (they are cut whichever way Luna would vote): 218 of 4472. 120 requests in total, 0 retries, 0 errored attempts, 1 malformed or incomplete answers, 0 targets left unanswered after retries (those keep Jev's own decision). Tokens: 1,823,975 prompt of which 171,360 cached, 193,579 completion of which 95,601 reasoning. Router cost $0.5662, list-rate cost $0.5662; the plan estimated $0.8021. Run wall clock 479 s, generated 2026-09-26T07:50:56+00:00.

## How the call was made

System message: the rules5 prompt read from `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md` at run time (md5 afba04cbbfc9), never copied into this repo. User message: a short preamble (`hybrid-preamble-v1`, in the script) saying the transcript is `id = text` lines without word ids, that only the listed targets need a verdict, and the JSON shape to answer with; then the whole episode transcript as Jev's sentence pass rendered it (module-cut ums stripped, Jev's retake losers removed, pause markers); then the target ids. Targets are the routed ids in transcript order in groups of up to 40, a group closing early once it would span more than 120 sentences; 8 requests in flight per episode, 3 attempts on an error with backoff, 1 re-ask on malformed or incomplete JSON, 600 s timeout, `reasoning_effort` as labelled. Every attempt is a row in the requests file with the raw answer text, the parsed verdicts, token counts (prompt, cached, completion, reasoning), the router's cost and a list-rate cost. Jev alone on the v3 run: $0.0423 and 5.6 s per episode on average.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-decisions.jsonl`, md5 0063cbe0dd7f, modified 2026-09-26T06:39:11+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-requests.jsonl`, md5 284831fbe9a0, modified 2026-09-26T06:39:11+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-jev-all18-v3-timing.json`, md5 c77e5c1644e7, modified 2026-09-26T06:39:11+00:00
- input rules_prompt: `D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md`, md5 afba04cbbfc9, modified 2026-09-08T05:43:31+00:00
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
- input m046:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m046-decisions.jsonl`, md5 e30c0a008c86, modified 2026-09-26T07:42:19+00:00
- input m046:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m046-requests.jsonl`, md5 7e44b01875c2, modified 2026-09-26T07:42:19+00:00
- input m046:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m046-timing.json`, md5 a6396a5c1d54, modified 2026-09-26T07:42:19+00:00
- input m080:decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m080-decisions.jsonl`, md5 676b31b4ea92, modified 2026-09-26T07:50:56+00:00
- input m080:requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m080-requests.jsonl`, md5 426027cc2376, modified 2026-09-26T07:50:56+00:00
- input m080:timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna-m080-timing.json`, md5 731b4a7b3766, modified 2026-09-26T07:50:56+00:00
- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 ee6934d58943
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\.worktrees\jev-roughcut-r2-2026-09-25\docs\jev-real\roughcut-hybrid-luna.json`
