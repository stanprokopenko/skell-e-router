# Jev rough cut: run report

Generated 2026-09-20T22:41:42+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-heldout-v3-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v3, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 13 episode(s): perspective-13d-critique, hampton-5.5-crit1, hampton-5.5-crit2, hampton-5.5-crit3, hampton-5.5-crit4, hampton-5.5-crit5, flanders-03-thematic-crit, anatomy-30b-hamstring-crit, colman-04.03-life-crit, colman-05.02-master-studies-crit, colman-06.06-species-crit, hampton-7-conclusion, greco-2.2-thumbnailing.
467 requests, 104 errored, 101 retried, 15,364,723 input tokens, $0.6453, 82.918 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 0 per arm out of 7588.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 7588 sentences with 0 mismatches.

Warnings from this report:
- roughcut-jev-heldout-v3-decisions.jsonl has no rows for jev_b; the run wrote jev_a, jev_b_moduleretakes, jev_b_notrim only (a run without `--trim-pick` writes no jev_b). Those arms are left out of every table below.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 71.51 | 71.79 | 70.55 | 78.56 | 68.27 | 116.43 | 6.38 |
| jev_b_moduleretakes | 2.50 | 22.12 | 72.12 | 70.71 | 78.70 | 68.43 | 115.93 | 6.38 |
| jev_b_notrim | 2.50 | 21.72 | 71.72 | 70.33 | 78.36 | 68.13 | 117.09 | 6.38 |
| jev_noul (t_trim 0.3) | 3.10 | 70.57 | 70.85 | 68.97 | 77.11 | 66.99 | 111.85 | 6.38 |
| jev_mix (t_trim 0.3) | 3.00 | 73.02 | 73.30 | 70.53 | 78.74 | 68.49 | 104.37 | 6.38 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 76.74 | 76.74 | 72.14 | 86.08 | 74.27 | 103.53 | 6.38 |
| jev_b_moduleretakes | 2.50 | 77.05 | 77.05 | 72.33 | 86.31 | 74.49 | 102.99 | 6.38 |
| jev_b_notrim | 2.50 | 76.74 | 76.74 | 71.96 | 86.00 | 74.22 | 104.05 | 6.38 |
| jev_noul (t_trim 0.3) | 3.10 | 75.43 | 75.43 | 70.50 | 84.31 | 72.73 | 99.78 | 6.38 |
| jev_mix (t_trim 0.3) | 2.70 | 76.40 | 76.40 | 71.65 | 85.74 | 73.95 | 105.68 | 6.38 |

## Per episode, plain

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 76.03 | 76.03 | 78.61 | 80.54 | 75.82 | 110.63 | no | 13.61 |
| hampton-5.5-crit1 | 82.60 | 82.60 | 63.44 | 87.08 | 60.77 | 105.02 | no | 2.22 |
| hampton-5.5-crit2 | 79.21 | 79.21 | 57.72 | 81.69 | 57.28 | 99.06 | no | 0.93 |
| hampton-5.5-crit3 | 81.02 | 81.02 | 67.92 | 91.28 | 65.96 | 105.06 | no | 0.99 |
| hampton-5.5-crit4 | 84.42 | 84.42 | 61.08 | 84.99 | 59.46 | 100.77 | no | 0.79 |
| hampton-5.5-crit5 | 84.34 | 84.34 | 63.25 | 82.18 | 61.08 | 104.82 | no | 4.81 |
| flanders-03-thematic-crit | 73.64 | 73.64 | 76.80 | 79.68 | 72.54 | 96.03 | no | 19.26 |
| anatomy-30b-hamstring-crit | 74.48 | 74.48 | 75.22 | 81.26 | 70.50 | 104.43 | no | 11.76 |
| colman-04.03-life-crit | 66.85 | 66.85 | 63.50 | 70.04 | 61.39 | 122.55 | no | 5.88 |
| colman-05.02-master-studies-crit | 58.16 | 58.16 | 63.09 | 77.64 | 60.36 | 125.09 | no | 5.01 |
| colman-06.06-species-crit | 65.76 | 65.76 | 66.06 | 81.55 | 61.26 | 128.92 | no | 2.83 |
| hampton-7-conclusion | 20.70 | 70.70 | 71.66 | 80.68 | 70.02 | 104.79 | yes | 1.81 |
| greco-2.2-thumbnailing | 62.94 | 62.94 | 68.34 | 70.08 | 67.21 | 194.85 | no | 13.02 |
| **pooled** | 71.51 | 71.79 | 70.55 | 78.56 | 68.27 | 116.43 | — | 6.38 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 27.21 | 77.21 | 79.16 | 81.08 | 76.32 | 108.14 | yes | 13.61 |
| hampton-5.5-crit1 | 32.98 | 82.98 | 62.75 | 86.33 | 60.25 | 105.42 | yes | 2.22 |
| hampton-5.5-crit2 | 29.69 | 79.69 | 57.81 | 81.78 | 57.34 | 99.16 | yes | 0.93 |
| hampton-5.5-crit3 | 32.34 | 82.34 | 67.84 | 91.19 | 65.89 | 105.80 | yes | 0.99 |
| hampton-5.5-crit4 | 35.13 | 85.13 | 60.29 | 84.53 | 59.13 | 101.29 | yes | 0.79 |
| hampton-5.5-crit5 | 32.58 | 82.58 | 60.81 | 80.01 | 59.47 | 105.03 | yes | 4.81 |
| flanders-03-thematic-crit | 23.80 | 73.80 | 77.79 | 80.53 | 73.32 | 95.97 | yes | 19.26 |
| anatomy-30b-hamstring-crit | 23.82 | 73.82 | 74.92 | 80.82 | 70.12 | 104.09 | yes | 11.76 |
| colman-04.03-life-crit | 16.67 | 66.67 | 62.96 | 69.40 | 60.84 | 122.08 | yes | 5.88 |
| colman-05.02-master-studies-crit | 7.38 | 57.38 | 64.97 | 79.84 | 62.07 | 123.41 | yes | 5.01 |
| colman-06.06-species-crit | 15.04 | 65.04 | 65.49 | 80.71 | 60.63 | 129.44 | yes | 2.83 |
| hampton-7-conclusion | 20.70 | 70.70 | 71.66 | 80.68 | 70.02 | 104.79 | yes | 1.81 |
| greco-2.2-thumbnailing | 14.11 | 64.11 | 68.72 | 70.42 | 67.53 | 194.87 | yes | 13.02 |
| **pooled** | 22.12 | 72.12 | 70.71 | 78.70 | 68.43 | 115.93 | — | 6.38 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 25.96 | 75.96 | 78.58 | 80.51 | 75.79 | 110.88 | yes | 13.61 |
| hampton-5.5-crit1 | 32.98 | 82.98 | 62.75 | 86.33 | 60.25 | 105.42 | yes | 2.22 |
| hampton-5.5-crit2 | 29.69 | 79.69 | 57.81 | 81.78 | 57.34 | 99.16 | yes | 0.93 |
| hampton-5.5-crit3 | 32.34 | 82.34 | 67.84 | 91.19 | 65.89 | 105.80 | yes | 0.99 |
| hampton-5.5-crit4 | 35.13 | 85.13 | 60.29 | 84.53 | 59.13 | 101.29 | yes | 0.79 |
| hampton-5.5-crit5 | 32.92 | 82.92 | 60.95 | 80.17 | 59.59 | 105.19 | yes | 4.81 |
| flanders-03-thematic-crit | 24.03 | 74.03 | 76.90 | 79.87 | 72.71 | 97.02 | yes | 19.26 |
| anatomy-30b-hamstring-crit | 24.24 | 74.24 | 75.43 | 81.41 | 70.63 | 105.28 | yes | 11.76 |
| colman-04.03-life-crit | 16.87 | 66.87 | 63.46 | 69.87 | 61.25 | 123.10 | yes | 5.88 |
| colman-05.02-master-studies-crit | 7.11 | 57.11 | 62.39 | 77.00 | 59.86 | 125.95 | yes | 5.01 |
| colman-06.06-species-crit | 14.77 | 64.77 | 65.36 | 80.69 | 60.62 | 129.89 | yes | 2.83 |
| hampton-7-conclusion | 20.70 | 70.70 | 71.66 | 80.68 | 70.02 | 104.79 | yes | 1.81 |
| greco-2.2-thumbnailing | 12.95 | 62.95 | 68.29 | 69.97 | 67.11 | 195.65 | yes | 13.02 |
| **pooled** | 21.72 | 71.72 | 70.33 | 78.36 | 68.13 | 117.09 | — | 6.38 |

### jev_noul (t_trim 0.3) (plain)

Calibrated pooled keep threshold 3.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 73.61 | 73.61 | 75.52 | 78.27 | 73.68 | 114.40 | no | 13.61 |
| hampton-5.5-crit1 | 79.28 | 79.28 | 62.09 | 85.18 | 59.45 | 102.05 | no | 2.22 |
| hampton-5.5-crit2 | 85.04 | 85.04 | 60.31 | 84.28 | 59.09 | 104.68 | no | 0.93 |
| hampton-5.5-crit3 | 78.10 | 78.10 | 70.48 | 92.29 | 66.69 | 103.62 | no | 0.99 |
| hampton-5.5-crit4 | 80.19 | 80.19 | 59.06 | 82.79 | 57.92 | 96.56 | no | 0.79 |
| hampton-5.5-crit5 | 86.47 | 86.47 | 65.90 | 86.13 | 64.02 | 105.87 | no | 4.81 |
| flanders-03-thematic-crit | 71.26 | 71.26 | 74.31 | 77.52 | 70.57 | 87.23 | no | 19.26 |
| anatomy-30b-hamstring-crit | 69.99 | 69.99 | 69.47 | 75.96 | 65.90 | 96.28 | no | 11.76 |
| colman-04.03-life-crit | 66.24 | 66.24 | 63.58 | 70.76 | 62.03 | 120.58 | no | 5.88 |
| colman-05.02-master-studies-crit | 59.21 | 59.21 | 62.46 | 77.00 | 59.86 | 123.86 | no | 5.01 |
| colman-06.06-species-crit | 60.97 | 60.97 | 61.69 | 77.01 | 57.85 | 114.87 | no | 2.83 |
| hampton-7-conclusion | 30.00 | 80.00 | 80.33 | 88.16 | 76.51 | 107.47 | yes | 1.81 |
| greco-2.2-thumbnailing | 67.29 | 67.29 | 68.57 | 70.37 | 67.49 | 180.93 | no | 13.02 |
| **pooled** | 70.57 | 70.85 | 68.97 | 77.11 | 66.99 | 111.85 | — | 6.38 |

### jev_mix (t_trim 0.3) (plain)

Calibrated pooled keep threshold 3.00.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 79.39 | 79.39 | 78.50 | 80.67 | 75.94 | 93.53 | no | 13.61 |
| hampton-5.5-crit1 | 80.39 | 80.39 | 63.70 | 85.81 | 59.89 | 101.34 | no | 2.22 |
| hampton-5.5-crit2 | 79.21 | 79.21 | 57.81 | 81.87 | 57.40 | 99.39 | no | 0.93 |
| hampton-5.5-crit3 | 78.10 | 78.10 | 68.43 | 91.23 | 65.92 | 104.17 | no | 0.99 |
| hampton-5.5-crit4 | 80.83 | 80.83 | 59.35 | 82.56 | 57.76 | 97.23 | no | 0.79 |
| hampton-5.5-crit5 | 85.69 | 85.69 | 65.10 | 85.53 | 63.57 | 104.85 | no | 4.81 |
| flanders-03-thematic-crit | 70.22 | 70.22 | 72.80 | 76.49 | 69.64 | 81.75 | no | 19.26 |
| anatomy-30b-hamstring-crit | 72.72 | 72.72 | 75.62 | 81.56 | 70.76 | 90.27 | no | 11.76 |
| colman-04.03-life-crit | 65.13 | 65.13 | 61.72 | 69.24 | 60.69 | 117.02 | no | 5.88 |
| colman-05.02-master-studies-crit | 58.24 | 58.24 | 63.08 | 77.84 | 60.51 | 120.24 | no | 5.01 |
| colman-06.06-species-crit | 62.63 | 62.63 | 63.15 | 78.53 | 58.99 | 119.52 | no | 2.83 |
| hampton-7-conclusion | 30.00 | 80.00 | 80.33 | 88.16 | 76.51 | 107.47 | yes | 1.81 |
| greco-2.2-thumbnailing | 73.20 | 73.20 | 72.05 | 73.97 | 70.94 | 163.60 | no | 13.02 |
| **pooled** | 73.02 | 73.30 | 70.53 | 78.74 | 68.49 | 104.37 | — | 6.38 |

## Per episode, with modules

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 78.09 | 78.09 | 79.05 | 83.28 | 78.40 | 101.97 | no | 13.61 |
| hampton-5.5-crit1 | 87.35 | 87.35 | 67.36 | 111.46 | 77.79 | 96.65 | no | 2.22 |
| hampton-5.5-crit2 | 79.45 | 79.45 | 59.31 | 102.20 | 71.65 | 90.95 | no | 0.93 |
| hampton-5.5-crit3 | 81.17 | 81.17 | 68.98 | 105.91 | 76.53 | 96.00 | no | 0.99 |
| hampton-5.5-crit4 | 88.46 | 88.46 | 64.13 | 104.93 | 73.41 | 92.47 | no | 0.79 |
| hampton-5.5-crit5 | 90.88 | 90.88 | 67.20 | 103.00 | 76.55 | 95.24 | no | 4.81 |
| flanders-03-thematic-crit | 77.14 | 77.14 | 77.61 | 83.24 | 75.79 | 82.18 | no | 19.26 |
| anatomy-30b-hamstring-crit | 84.90 | 84.90 | 78.18 | 89.85 | 77.95 | 82.46 | no | 11.76 |
| colman-04.03-life-crit | 76.18 | 76.18 | 65.37 | 77.14 | 67.62 | 110.91 | no | 5.88 |
| colman-05.02-master-studies-crit | 66.33 | 66.33 | 65.98 | 88.21 | 68.57 | 114.18 | no | 5.01 |
| colman-06.06-species-crit | 79.57 | 79.57 | 70.43 | 97.21 | 73.02 | 115.66 | no | 2.83 |
| hampton-7-conclusion | 72.33 | 72.33 | 72.07 | 85.16 | 73.91 | 98.32 | no | 1.81 |
| greco-2.2-thumbnailing | 65.09 | 65.09 | 68.55 | 71.80 | 68.86 | 178.17 | no | 13.02 |
| **pooled** | 76.74 | 76.74 | 72.14 | 86.08 | 74.27 | 103.53 | — | 6.38 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 79.12 | 79.12 | 79.60 | 83.79 | 78.87 | 99.69 | no | 13.61 |
| hampton-5.5-crit1 | 87.73 | 87.73 | 66.77 | 111.02 | 77.48 | 97.05 | no | 2.22 |
| hampton-5.5-crit2 | 79.92 | 79.92 | 59.41 | 102.30 | 71.72 | 91.05 | no | 0.93 |
| hampton-5.5-crit3 | 82.48 | 82.48 | 68.90 | 106.06 | 76.64 | 96.61 | no | 0.99 |
| hampton-5.5-crit4 | 89.94 | 89.94 | 63.37 | 104.60 | 73.18 | 92.98 | no | 0.79 |
| hampton-5.5-crit5 | 89.12 | 89.12 | 64.88 | 101.61 | 75.52 | 95.38 | no | 4.81 |
| flanders-03-thematic-crit | 77.16 | 77.16 | 78.57 | 84.14 | 76.60 | 81.86 | no | 19.26 |
| anatomy-30b-hamstring-crit | 84.25 | 84.25 | 77.91 | 89.55 | 77.69 | 82.18 | no | 11.76 |
| colman-04.03-life-crit | 75.68 | 75.68 | 64.80 | 76.55 | 67.10 | 110.43 | no | 5.88 |
| colman-05.02-master-studies-crit | 65.64 | 65.64 | 67.83 | 90.12 | 70.06 | 112.56 | no | 5.01 |
| colman-06.06-species-crit | 79.25 | 79.25 | 70.05 | 96.85 | 72.75 | 115.86 | no | 2.83 |
| hampton-7-conclusion | 72.33 | 72.33 | 72.07 | 85.16 | 73.91 | 98.32 | no | 1.81 |
| greco-2.2-thumbnailing | 66.37 | 66.37 | 68.92 | 72.15 | 69.20 | 178.10 | no | 13.02 |
| **pooled** | 77.05 | 77.05 | 72.33 | 86.31 | 74.49 | 102.99 | — | 6.38 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.50.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 78.03 | 78.03 | 79.02 | 83.28 | 78.39 | 102.17 | no | 13.61 |
| hampton-5.5-crit1 | 87.73 | 87.73 | 66.77 | 111.02 | 77.48 | 97.05 | no | 2.22 |
| hampton-5.5-crit2 | 79.92 | 79.92 | 59.41 | 102.30 | 71.72 | 91.05 | no | 0.93 |
| hampton-5.5-crit3 | 82.48 | 82.48 | 68.90 | 106.06 | 76.64 | 96.61 | no | 0.99 |
| hampton-5.5-crit4 | 89.94 | 89.94 | 63.37 | 104.60 | 73.18 | 92.98 | no | 0.79 |
| hampton-5.5-crit5 | 89.46 | 89.46 | 65.05 | 101.79 | 75.65 | 95.54 | no | 4.81 |
| flanders-03-thematic-crit | 77.51 | 77.51 | 77.69 | 83.45 | 75.97 | 82.90 | no | 19.26 |
| anatomy-30b-hamstring-crit | 84.85 | 84.85 | 78.43 | 90.07 | 78.14 | 83.14 | no | 11.76 |
| colman-04.03-life-crit | 76.20 | 76.20 | 65.32 | 77.07 | 67.56 | 111.37 | no | 5.88 |
| colman-05.02-master-studies-crit | 65.38 | 65.38 | 65.36 | 87.71 | 68.18 | 115.00 | no | 5.01 |
| colman-06.06-species-crit | 79.03 | 79.03 | 70.00 | 96.91 | 72.80 | 116.27 | no | 2.83 |
| hampton-7-conclusion | 72.33 | 72.33 | 72.07 | 85.16 | 73.91 | 98.32 | no | 1.81 |
| greco-2.2-thumbnailing | 65.22 | 65.22 | 68.51 | 71.72 | 68.78 | 178.85 | no | 13.02 |
| **pooled** | 76.74 | 76.74 | 71.96 | 86.00 | 74.22 | 104.05 | — | 6.38 |

### jev_noul (t_trim 0.3) (layered)

Calibrated pooled keep threshold 3.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 75.91 | 75.91 | 76.09 | 81.15 | 76.39 | 105.63 | no | 13.61 |
| hampton-5.5-crit1 | 84.03 | 84.03 | 65.71 | 109.16 | 76.18 | 93.78 | no | 2.22 |
| hampton-5.5-crit2 | 86.54 | 86.54 | 62.87 | 107.45 | 75.34 | 96.11 | no | 0.93 |
| hampton-5.5-crit3 | 78.25 | 78.25 | 71.50 | 106.70 | 77.10 | 94.60 | no | 0.99 |
| hampton-5.5-crit4 | 83.97 | 83.97 | 61.68 | 101.35 | 70.91 | 88.57 | no | 0.79 |
| hampton-5.5-crit5 | 92.61 | 92.61 | 69.48 | 105.18 | 78.18 | 96.36 | no | 4.81 |
| flanders-03-thematic-crit | 74.09 | 74.09 | 75.00 | 80.89 | 73.64 | 75.34 | no | 19.26 |
| anatomy-30b-hamstring-crit | 79.56 | 79.56 | 72.74 | 84.69 | 73.47 | 75.66 | no | 11.76 |
| colman-04.03-life-crit | 75.25 | 75.25 | 65.15 | 76.80 | 67.33 | 109.91 | no | 5.88 |
| colman-05.02-master-studies-crit | 67.17 | 67.17 | 65.17 | 86.76 | 67.45 | 113.52 | no | 5.01 |
| colman-06.06-species-crit | 71.18 | 71.18 | 65.14 | 90.69 | 68.13 | 103.21 | no | 2.83 |
| hampton-7-conclusion | 81.63 | 81.63 | 80.70 | 92.63 | 80.39 | 100.40 | no | 1.81 |
| greco-2.2-thumbnailing | 69.51 | 69.51 | 68.77 | 71.88 | 68.94 | 165.29 | no | 13.02 |
| **pooled** | 75.43 | 75.43 | 70.50 | 84.31 | 72.73 | 99.78 | — | 6.38 |

### jev_mix (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.70.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 76.93 | 76.93 | 78.25 | 82.81 | 77.95 | 108.33 | no | 13.61 |
| hampton-5.5-crit1 | 87.35 | 87.35 | 67.94 | 111.75 | 77.98 | 96.21 | no | 2.22 |
| hampton-5.5-crit2 | 84.49 | 84.49 | 62.26 | 106.60 | 74.74 | 95.35 | no | 0.93 |
| hampton-5.5-crit3 | 78.98 | 78.98 | 68.52 | 105.94 | 76.55 | 96.00 | no | 0.99 |
| hampton-5.5-crit4 | 87.63 | 87.63 | 63.53 | 103.89 | 72.68 | 91.42 | no | 0.79 |
| hampton-5.5-crit5 | 92.92 | 92.92 | 69.72 | 105.21 | 78.20 | 96.88 | no | 4.81 |
| flanders-03-thematic-crit | 77.96 | 77.96 | 78.08 | 83.79 | 76.29 | 84.72 | no | 19.26 |
| anatomy-30b-hamstring-crit | 84.38 | 84.38 | 77.70 | 89.65 | 77.78 | 82.64 | no | 11.76 |
| colman-04.03-life-crit | 75.58 | 75.58 | 64.04 | 76.00 | 66.62 | 112.32 | no | 5.88 |
| colman-05.02-master-studies-crit | 65.91 | 65.91 | 64.00 | 86.13 | 66.96 | 116.75 | no | 5.01 |
| colman-06.06-species-crit | 77.61 | 77.61 | 68.56 | 95.09 | 71.43 | 114.66 | no | 2.83 |
| hampton-7-conclusion | 81.63 | 81.63 | 80.70 | 92.63 | 80.39 | 100.40 | no | 1.81 |
| greco-2.2-thumbnailing | 64.27 | 64.27 | 67.12 | 70.51 | 67.62 | 180.30 | no | 13.02 |
| **pooled** | 76.40 | 76.40 | 71.65 | 85.74 | 73.95 | 105.68 | — | 6.38 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 0.61 | 13.01 | n/a | 13.61 | 114 | 3,058,721 | 0.1285 | 24 | 26 |
| hampton-5.5-crit1 | 0.22 | 2.00 | n/a | 2.22 | 10 | 360,841 | 0.0152 | 1 | 1 |
| hampton-5.5-crit2 | 0.20 | 0.73 | n/a | 0.93 | 7 | 243,203 | 0.0102 | 0 | 0 |
| hampton-5.5-crit3 | 0.22 | 0.77 | n/a | 0.99 | 7 | 267,744 | 0.0112 | 0 | 0 |
| hampton-5.5-crit4 | 0.00 | 0.79 | n/a | 0.79 | 7 | 317,488 | 0.0133 | 0 | 0 |
| hampton-5.5-crit5 | 0.21 | 4.60 | n/a | 4.81 | 15 | 582,998 | 0.0245 | 2 | 2 |
| flanders-03-thematic-crit | 0.27 | 18.99 | n/a | 19.26 | 97 | 2,966,909 | 0.1246 | 38 | 38 |
| anatomy-30b-hamstring-crit | 0.37 | 11.40 | n/a | 11.76 | 58 | 2,174,169 | 0.0913 | 10 | 10 |
| colman-04.03-life-crit | 0.42 | 5.46 | n/a | 5.88 | 28 | 1,053,621 | 0.0443 | 4 | 5 |
| colman-05.02-master-studies-crit | 0.26 | 4.75 | n/a | 5.01 | 21 | 782,659 | 0.0329 | 4 | 4 |
| colman-06.06-species-crit | 0.21 | 2.62 | n/a | 2.83 | 18 | 731,524 | 0.0307 | 2 | 2 |
| hampton-7-conclusion | 0.18 | 1.63 | n/a | 1.81 | 4 | 71,840 | 0.0030 | 1 | 1 |
| greco-2.2-thumbnailing | 0.43 | 12.59 | n/a | 13.02 | 81 | 2,753,006 | 0.1156 | 15 | 15 |
| **total** | 3.58 | 79.34 | 0.00 | 82.92 | 467 | 15,364,723 | 0.6453 | 101 | 104 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 38 | 9 | 107 | no |
| hampton-5.5-crit1 | 9 | 6 | 28 | no |
| hampton-5.5-crit2 | 5 | 2 | 18 | no |
| hampton-5.5-crit3 | 11 | 8 | 27 | no |
| hampton-5.5-crit4 | 8 | 7 | 27 | no |
| hampton-5.5-crit5 | 7 | 6 | 44 | no |
| flanders-03-thematic-crit | 50 | 29 | 145 | no |
| anatomy-30b-hamstring-crit | 25 | 14 | 150 | no |
| colman-04.03-life-crit | 18 | 11 | 110 | no |
| colman-05.02-master-studies-crit | 18 | 14 | 149 | no |
| colman-06.06-species-crit | 13 | 8 | 159 | no |
| hampton-7-conclusion | 2 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 70 | 20 | 97 | no |
| **total** | 274 | 134 | 1072 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 0 | 0 | 107 | yes |
| hampton-5.5-crit1 | 0 | 0 | 28 | yes |
| hampton-5.5-crit2 | 0 | 0 | 18 | yes |
| hampton-5.5-crit3 | 0 | 0 | 27 | yes |
| hampton-5.5-crit4 | 0 | 0 | 27 | yes |
| hampton-5.5-crit5 | 0 | 0 | 44 | yes |
| flanders-03-thematic-crit | 0 | 0 | 145 | yes |
| anatomy-30b-hamstring-crit | 0 | 0 | 150 | yes |
| colman-04.03-life-crit | 0 | 0 | 110 | yes |
| colman-05.02-master-studies-crit | 0 | 0 | 149 | yes |
| colman-06.06-species-crit | 0 | 0 | 159 | yes |
| hampton-7-conclusion | 0 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 0 | 0 | 97 | yes |
| **total** | 0 | 0 | 1072 | — |

### jev_b_notrim

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 0 | 0 | 107 | yes |
| hampton-5.5-crit1 | 0 | 0 | 28 | yes |
| hampton-5.5-crit2 | 0 | 0 | 18 | yes |
| hampton-5.5-crit3 | 0 | 0 | 27 | yes |
| hampton-5.5-crit4 | 0 | 0 | 27 | yes |
| hampton-5.5-crit5 | 0 | 0 | 44 | yes |
| flanders-03-thematic-crit | 0 | 0 | 145 | yes |
| anatomy-30b-hamstring-crit | 0 | 0 | 150 | yes |
| colman-04.03-life-crit | 0 | 0 | 110 | yes |
| colman-05.02-master-studies-crit | 0 | 0 | 149 | yes |
| colman-06.06-species-crit | 0 | 0 | 159 | yes |
| hampton-7-conclusion | 0 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 0 | 0 | 97 | yes |
| **total** | 0 | 0 | 1072 | — |

### jev_noul (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 38 | 14 | 107 | no |
| hampton-5.5-crit1 | 9 | 7 | 28 | no |
| hampton-5.5-crit2 | 5 | 4 | 18 | no |
| hampton-5.5-crit3 | 11 | 8 | 27 | no |
| hampton-5.5-crit4 | 8 | 4 | 27 | no |
| hampton-5.5-crit5 | 7 | 4 | 44 | no |
| flanders-03-thematic-crit | 50 | 27 | 145 | no |
| anatomy-30b-hamstring-crit | 25 | 13 | 150 | no |
| colman-04.03-life-crit | 18 | 12 | 110 | no |
| colman-05.02-master-studies-crit | 18 | 15 | 149 | no |
| colman-06.06-species-crit | 13 | 6 | 159 | no |
| hampton-7-conclusion | 2 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 70 | 14 | 97 | no |
| **total** | 274 | 128 | 1072 | — |

### jev_mix (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| perspective-13d-critique | 38 | 8 | 107 | no |
| hampton-5.5-crit1 | 9 | 5 | 28 | no |
| hampton-5.5-crit2 | 5 | 2 | 18 | no |
| hampton-5.5-crit3 | 11 | 8 | 27 | no |
| hampton-5.5-crit4 | 8 | 4 | 27 | no |
| hampton-5.5-crit5 | 7 | 5 | 44 | no |
| flanders-03-thematic-crit | 50 | 27 | 145 | no |
| anatomy-30b-hamstring-crit | 25 | 13 | 150 | no |
| colman-04.03-life-crit | 18 | 9 | 110 | no |
| colman-05.02-master-studies-crit | 18 | 14 | 149 | no |
| colman-06.06-species-crit | 13 | 7 | 159 | no |
| hampton-7-conclusion | 2 | 0 | 11 | yes |
| greco-2.2-thumbnailing | 70 | 7 | 97 | no |
| **total** | 274 | 109 | 1072 | — |

## The cut_k question (cut_p)

`cut_p` is P(the editor removes this sentence) from the noul asked next to the 0-5 score on every target. `jev_noul` scores a sentence 5 x (1 - cut_p), `jev_mix` averages that with the score; both carry jev_a's trims and retake cut, so the keep decision is the only thing that differs.

| episode | sentences | unusable | r(cut_p, score) | mean cut_p | mean score |
|---|---:|---:|---:|---:|---:|
| perspective-13d-critique | 1752 | 0 | -0.788 | 0.407 | 2.27 |
| hampton-5.5-crit1 | 181 | 0 | -0.745 | 0.273 | 3.27 |
| hampton-5.5-crit2 | 127 | 0 | -0.805 | 0.246 | 3.22 |
| hampton-5.5-crit3 | 137 | 0 | -0.829 | 0.285 | 3.22 |
| hampton-5.5-crit4 | 156 | 0 | -0.744 | 0.283 | 3.34 |
| hampton-5.5-crit5 | 295 | 0 | -0.628 | 0.242 | 3.38 |
| flanders-03-thematic-crit | 1309 | 0 | -0.812 | 0.371 | 2.79 |
| anatomy-30b-hamstring-crit | 951 | 0 | -0.856 | 0.360 | 2.85 |
| colman-04.03-life-crit | 495 | 0 | -0.823 | 0.300 | 3.24 |
| colman-05.02-master-studies-crit | 381 | 0 | -0.775 | 0.298 | 3.20 |
| colman-06.06-species-crit | 373 | 0 | -0.785 | 0.326 | 3.27 |
| hampton-7-conclusion | 43 | 0 | -0.907 | 0.356 | 2.57 |
| greco-2.2-thumbnailing | 1388 | 0 | -0.853 | 0.390 | 2.60 |
| **pooled** | 7588 | — | -0.827 | 0.358 | 2.77 |

Keep/cut agreement with the editor at each arm's own calibrated threshold, layered scoring over 13 episode(s). Counts are sentences: `wrong drops` is the editor kept it and the arm removed it.

| arm | threshold | sentences | both keep | both remove | wrong drops | wrong keeps | agreement | kept ratio | SENTENCE POINTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.3) | 2.50 | 7588 | 3924 | 2092 | 480 | 1092 | 79.28 | 103.53 | 76.74 |
| jev_noul (t_trim 0.3) | 3.10 | 7588 | 3688 | 2188 | 716 | 996 | 77.44 | 99.78 | 75.43 |
| jev_mix (t_trim 0.3) | 2.70 | 7588 | 3964 | 2037 | 440 | 1147 | 79.09 | 105.68 | 76.40 |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | 101 | 47 | 0 | 70 | 157 | 4 (1 kept / 3 cut) | 91 (10 kept / 81 cut) |
| hampton-5.5-crit1 | 2 | 0 | 0 | 2 | 2 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit2 | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit3 | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit4 | 0 | 0 | 0 | 0 | 0 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| hampton-5.5-crit5 | 1 | 1 | 0 | 0 | 1 | 0 (0 kept / 0 cut) | 1 (1 kept / 0 cut) |
| flanders-03-thematic-crit | 31 | 9 | 0 | 80 | 96 | 3 (3 kept / 0 cut) | 19 (11 kept / 8 cut) |
| anatomy-30b-hamstring-crit | 51 | 21 | 0 | 32 | 57 | 0 (0 kept / 0 cut) | 25 (10 kept / 15 cut) |
| colman-04.03-life-crit | 14 | 5 | 0 | 10 | 16 | 0 (0 kept / 0 cut) | 6 (4 kept / 2 cut) |
| colman-05.02-master-studies-crit | 4 | 3 | 0 | 1 | 6 | 0 (0 kept / 0 cut) | 5 (3 kept / 2 cut) |
| colman-06.06-species-crit | 6 | 3 | 0 | 3 | 6 | 0 (0 kept / 0 cut) | 3 (1 kept / 2 cut) |
| hampton-7-conclusion | 1 | 0 | 0 | 1 | 1 | 0 (0 kept / 0 cut) | 0 (0 kept / 0 cut) |
| greco-2.2-thumbnailing | 59 | 26 | 0 | 46 | 86 | 2 (1 kept / 1 cut) | 42 (1 kept / 41 cut) |

## Where this lands on the published ladder

Reference arms are the 18-episode pooled numbers from `2026-09-11-model-plus-deterministic.json`. NOT THE SAME EPISODE SET as the tables above (missing per-episode numbers for: greco-2.2-thumbnailing), so the comparison is indicative only.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 18 | 79.96 | 79.44 | 88.34 | 83.71 | 80.35 | 93.55 |
| shipped Opus agentic | 18 | 83.45 | 80.84 | 90.37 | 86.47 | 81.44 | 95.02 |
| Luna single call | 18 | 18.61 | 61.23 | 69.43 | 65.36 | 64.63 | 80.67 |
| deterministic baseline (um removal + retakes + delete silence) | 18 | 63.72 | 64.07 | 79.92 | 63.72 | 64.07 | 79.92 |

Best Jev arm here is jev_mix (t_trim 0.3) at 73.02 SENTENCE POINTS (70.53 WORD SCORE, 6.38 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 76.40 SENTENCE POINTS (71.65 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v3-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v3-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v3-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v3-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-heldout-v3-summary.json`
