# Jev rough cut: run report

Generated 2026-09-20T21:45:54+00:00 by `scripts/jev_real/roughcut_jev_report.py` from `roughcut-jev-decisions.jsonl` and its request and timing files. No model calls, no detector runs, $0. Every metric is x100, two decimals; the summary JSON next to this file keeps the raw 0-to-1 values.

## What was run

Model jev-1.13.0, prompt version v1, concurrency 8, pick-pass trigger t_trim 0.5 at run time. 6 episode(s): colman-02.04-skeleton-demo, hampton-5.4-assignment-demo, colman-03.03-muscles-crit, edges-7.01-intro, hampton-5.2-shape-demo, perspective-14e-boxes-critique.
309 requests, 23 errored, 19 retried, 5,325,258 input tokens, $0.2237, 22.712 s of wall clock in total.
Sentences the run never got an answer for, scored at 0.0: 100 per arm out of 2743.

Variant A is rebuilt per trim-trigger threshold from `first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. A side is trimmed only when its top option is a word and P(whole) is below the threshold. The decisions file does not store the full choice distribution, so no threshold can add a trim to a sentence where `whole` was the top option; the sweep only withholds trims the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, and it does here on 2743 sentences with 0 mismatches.

## Pooled, plain (the model's cut alone)

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.1) | 2.40 | 34.86 | 77.37 | 75.48 | 81.69 | 72.49 | 98.04 | 3.79 |
| jev_a (t_trim 0.2) | 2.40 | 49.38 | 77.36 | 75.50 | 81.71 | 72.51 | 98.00 | 3.79 |
| jev_a (t_trim 0.3) | 2.40 | 70.36 | 77.45 | 75.58 | 81.79 | 72.57 | 97.62 | 3.79 |
| jev_a (t_trim 0.5) | 2.40 | 70.30 | 77.39 | 75.56 | 81.79 | 72.56 | 97.46 | 3.79 |
| jev_b | 2.40 | 40.36 | 77.35 | 75.47 | 81.68 | 72.48 | 98.02 | 3.79 |
| jev_b_moduleretakes | 2.40 | 40.40 | 77.38 | 75.35 | 81.58 | 72.39 | 96.50 | 3.79 |
| jev_b_notrim | 2.40 | 34.86 | 77.37 | 75.48 | 81.69 | 72.49 | 98.04 | 3.79 |

## Pooled, with um removal and delete silence layered on

| arm | threshold | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jev_a (t_trim 0.1) | 2.40 | 83.51 | 83.51 | 76.84 | 86.52 | 76.38 | 88.31 | 3.79 |
| jev_a (t_trim 0.2) | 2.40 | 83.53 | 83.53 | 76.86 | 86.53 | 76.39 | 88.27 | 3.79 |
| jev_a (t_trim 0.3) | 2.10 | 83.10 | 83.10 | 76.76 | 86.61 | 76.52 | 96.34 | 3.79 |
| jev_a (t_trim 0.5) | 2.10 | 83.03 | 83.03 | 76.74 | 86.59 | 76.51 | 96.20 | 3.79 |
| jev_b | 2.40 | 83.50 | 83.50 | 76.83 | 86.51 | 76.37 | 88.30 | 3.79 |
| jev_b_moduleretakes | 2.10 | 83.12 | 83.12 | 76.51 | 86.44 | 76.38 | 94.86 | 3.79 |
| jev_b_notrim | 2.40 | 83.51 | 83.51 | 76.84 | 86.52 | 76.38 | 88.31 | 3.79 |

## Per episode, plain

### jev_a (t_trim 0.1) (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.10 | 87.25 | 62.28 | 113.94 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.27 | 73.27 | 72.82 | 82.44 | 67.01 | 114.22 | yes | 2.42 |
| colman-03.03-muscles-crit | 21.95 | 71.95 | 60.59 | 71.20 | 59.91 | 90.95 | yes | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.76 | 79.76 | 87.81 | 91.75 | 79.58 | 111.74 | no | 3.35 |
| perspective-14e-boxes-critique | 31.95 | 81.95 | 79.21 | 81.28 | 76.67 | 93.57 | yes | 8.82 |
| **pooled** | 34.86 | 77.37 | 75.48 | 81.69 | 72.49 | 98.04 | — | 3.79 |

### jev_a (t_trim 0.2) (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 61.65 | 61.65 | 66.23 | 87.35 | 62.36 | 113.86 | no | 2.37 |
| hampton-5.4-assignment-demo | 73.27 | 73.27 | 72.97 | 82.55 | 67.10 | 114.13 | no | 2.42 |
| colman-03.03-muscles-crit | 71.91 | 71.91 | 60.64 | 71.26 | 59.95 | 90.86 | no | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.68 | 79.68 | 87.75 | 91.71 | 79.54 | 111.69 | no | 3.35 |
| perspective-14e-boxes-critique | 31.95 | 81.95 | 79.21 | 81.28 | 76.67 | 93.57 | yes | 8.82 |
| **pooled** | 49.38 | 77.36 | 75.50 | 81.71 | 72.51 | 98.00 | — | 3.79 |

### jev_a (t_trim 0.3) (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 63.66 | 63.66 | 66.76 | 88.19 | 62.95 | 113.03 | no | 2.37 |
| hampton-5.4-assignment-demo | 73.27 | 73.27 | 72.97 | 82.55 | 67.10 | 114.13 | no | 2.42 |
| colman-03.03-muscles-crit | 71.35 | 71.35 | 60.75 | 71.27 | 59.96 | 89.96 | no | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.39 | 79.39 | 87.65 | 91.60 | 79.45 | 111.48 | no | 3.35 |
| perspective-14e-boxes-critique | 82.09 | 82.09 | 79.29 | 81.33 | 76.71 | 93.31 | no | 8.82 |
| **pooled** | 70.36 | 77.45 | 75.58 | 81.79 | 72.57 | 97.62 | — | 3.79 |

### jev_a (t_trim 0.5) (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 63.61 | 63.61 | 66.66 | 88.13 | 62.91 | 112.83 | no | 2.37 |
| hampton-5.4-assignment-demo | 73.27 | 73.27 | 72.97 | 82.55 | 67.10 | 114.13 | no | 2.42 |
| colman-03.03-muscles-crit | 71.09 | 71.09 | 60.71 | 71.32 | 60.00 | 89.34 | no | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.25 | 79.25 | 87.65 | 91.59 | 79.43 | 111.39 | no | 3.35 |
| perspective-14e-boxes-critique | 82.06 | 82.06 | 79.28 | 81.33 | 76.70 | 93.29 | no | 8.82 |
| **pooled** | 70.30 | 77.39 | 75.56 | 81.79 | 72.56 | 97.46 | — | 3.79 |

### jev_b (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.10 | 87.25 | 62.28 | 113.94 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.27 | 73.27 | 72.82 | 82.44 | 67.01 | 114.22 | yes | 2.42 |
| colman-03.03-muscles-crit | 71.75 | 71.75 | 60.52 | 71.10 | 59.81 | 90.83 | no | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.76 | 79.76 | 87.81 | 91.75 | 79.58 | 111.74 | no | 3.35 |
| perspective-14e-boxes-critique | 31.95 | 81.95 | 79.21 | 81.28 | 76.67 | 93.57 | yes | 8.82 |
| **pooled** | 40.36 | 77.35 | 75.47 | 81.68 | 72.48 | 98.02 | — | 3.79 |

### jev_b_moduleretakes (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.21 | 87.11 | 62.19 | 113.54 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.60 | 73.60 | 73.30 | 82.68 | 67.21 | 113.79 | yes | 2.42 |
| colman-03.03-muscles-crit | 71.09 | 71.09 | 60.64 | 71.67 | 60.29 | 90.20 | no | 2.90 |
| edges-7.01-intro | 27.63 | 77.63 | 74.63 | 76.24 | 74.62 | 58.68 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.76 | 79.76 | 87.86 | 91.71 | 79.54 | 111.60 | no | 3.35 |
| perspective-14e-boxes-critique | 31.78 | 81.78 | 78.41 | 80.69 | 76.10 | 91.32 | yes | 8.82 |
| **pooled** | 40.40 | 77.38 | 75.35 | 81.58 | 72.39 | 96.50 | — | 3.79 |

### jev_b_notrim (plain)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 11.55 | 61.55 | 66.10 | 87.25 | 62.28 | 113.94 | yes | 2.37 |
| hampton-5.4-assignment-demo | 23.27 | 73.27 | 72.82 | 82.44 | 67.01 | 114.22 | yes | 2.42 |
| colman-03.03-muscles-crit | 21.95 | 71.95 | 60.59 | 71.20 | 59.91 | 90.95 | yes | 2.90 |
| edges-7.01-intro | 26.61 | 76.61 | 73.88 | 75.67 | 74.06 | 64.44 | yes | 2.85 |
| hampton-5.2-shape-demo | 79.76 | 79.76 | 87.81 | 91.75 | 79.58 | 111.74 | no | 3.35 |
| perspective-14e-boxes-critique | 31.95 | 81.95 | 79.21 | 81.28 | 76.67 | 93.57 | yes | 8.82 |
| **pooled** | 34.86 | 77.37 | 75.48 | 81.69 | 72.49 | 98.04 | — | 3.79 |

## Per episode, with modules

### jev_a (t_trim 0.1) (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 81.96 | 81.96 | 73.07 | 105.15 | 75.06 | 99.93 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.85 | 94.29 | 76.64 | 95.35 | no | 2.42 |
| colman-03.03-muscles-crit | 75.91 | 75.91 | 62.05 | 74.82 | 62.95 | 83.77 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 93.63 | 93.63 | 88.70 | 99.38 | 86.19 | 96.61 | no | 3.35 |
| perspective-14e-boxes-critique | 83.44 | 83.44 | 79.47 | 82.31 | 77.63 | 88.22 | no | 8.82 |
| **pooled** | 83.51 | 83.51 | 76.84 | 86.52 | 76.38 | 88.31 | — | 3.79 |

### jev_a (t_trim 0.2) (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 82.47 | 82.47 | 73.19 | 105.22 | 75.12 | 99.88 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.99 | 94.39 | 76.73 | 95.26 | no | 2.42 |
| colman-03.03-muscles-crit | 75.87 | 75.87 | 62.10 | 74.87 | 62.99 | 83.67 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 93.55 | 93.55 | 88.64 | 99.34 | 86.16 | 96.56 | no | 3.35 |
| perspective-14e-boxes-critique | 83.44 | 83.44 | 79.47 | 82.31 | 77.63 | 88.22 | no | 8.82 |
| **pooled** | 83.53 | 83.53 | 76.86 | 86.53 | 76.39 | 88.27 | — | 3.79 |

### jev_a (t_trim 0.3) (layered)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 86.65 | 86.65 | 75.57 | 107.76 | 76.93 | 102.81 | no | 2.37 |
| hampton-5.4-assignment-demo | 82.87 | 82.87 | 68.42 | 88.80 | 72.18 | 105.14 | no | 2.42 |
| colman-03.03-muscles-crit | 78.78 | 78.78 | 62.38 | 74.84 | 62.97 | 85.12 | no | 2.90 |
| edges-7.01-intro | 77.46 | 77.46 | 76.44 | 77.15 | 75.51 | 72.99 | no | 2.85 |
| hampton-5.2-shape-demo | 92.43 | 92.43 | 85.53 | 96.88 | 84.02 | 100.72 | no | 3.35 |
| perspective-14e-boxes-critique | 82.27 | 82.27 | 81.11 | 83.67 | 78.91 | 101.78 | no | 8.82 |
| **pooled** | 83.10 | 83.10 | 76.76 | 86.61 | 76.52 | 96.34 | — | 3.79 |

### jev_a (t_trim 0.5) (layered)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 87.42 | 87.42 | 75.58 | 107.75 | 76.92 | 102.57 | no | 2.37 |
| hampton-5.4-assignment-demo | 82.87 | 82.87 | 68.42 | 88.80 | 72.18 | 105.14 | no | 2.42 |
| colman-03.03-muscles-crit | 77.99 | 77.99 | 62.25 | 74.74 | 62.88 | 84.64 | no | 2.90 |
| edges-7.01-intro | 77.46 | 77.46 | 76.44 | 77.15 | 75.51 | 72.99 | no | 2.85 |
| hampton-5.2-shape-demo | 92.29 | 92.29 | 85.53 | 96.85 | 84.00 | 100.64 | no | 3.35 |
| perspective-14e-boxes-critique | 82.24 | 82.24 | 81.12 | 83.67 | 78.92 | 101.74 | no | 8.82 |
| **pooled** | 83.03 | 83.03 | 76.74 | 86.59 | 76.51 | 96.20 | — | 3.79 |

### jev_b (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 81.96 | 81.96 | 73.07 | 105.15 | 75.06 | 99.93 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.85 | 94.29 | 76.64 | 95.35 | no | 2.42 |
| colman-03.03-muscles-crit | 75.81 | 75.81 | 61.99 | 74.76 | 62.89 | 83.69 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 93.63 | 93.63 | 88.70 | 99.38 | 86.19 | 96.61 | no | 3.35 |
| perspective-14e-boxes-critique | 83.44 | 83.44 | 79.47 | 82.31 | 77.63 | 88.22 | no | 8.82 |
| **pooled** | 83.50 | 83.50 | 76.83 | 86.51 | 76.37 | 88.30 | — | 3.79 |

### jev_b_moduleretakes (layered)

Calibrated pooled keep threshold 2.10.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 84.95 | 84.95 | 75.01 | 106.98 | 76.37 | 103.16 | no | 2.37 |
| hampton-5.4-assignment-demo | 83.00 | 83.00 | 68.76 | 88.91 | 72.27 | 104.81 | no | 2.42 |
| colman-03.03-muscles-crit | 77.46 | 77.46 | 61.62 | 74.72 | 62.86 | 85.41 | no | 2.90 |
| edges-7.01-intro | 79.00 | 79.00 | 77.54 | 77.94 | 76.29 | 65.44 | no | 2.85 |
| hampton-5.2-shape-demo | 92.97 | 92.97 | 85.64 | 96.89 | 84.03 | 101.03 | no | 3.35 |
| perspective-14e-boxes-critique | 82.22 | 82.22 | 80.37 | 83.18 | 78.45 | 99.11 | no | 8.82 |
| **pooled** | 83.12 | 83.12 | 76.51 | 86.44 | 76.38 | 94.86 | — | 3.79 |

### jev_b_notrim (layered)

Calibrated pooled keep threshold 2.40.

| episode | SENTENCE POINTS | SP raw | WORD SCORE | GRADE | frame match | kept ratio | penalty | s/episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 81.96 | 81.96 | 73.07 | 105.15 | 75.06 | 99.93 | no | 2.37 |
| hampton-5.4-assignment-demo | 86.33 | 86.33 | 74.85 | 94.29 | 76.64 | 95.35 | no | 2.42 |
| colman-03.03-muscles-crit | 75.91 | 75.91 | 62.05 | 74.82 | 62.95 | 83.77 | no | 2.90 |
| edges-7.01-intro | 77.56 | 77.56 | 74.27 | 75.18 | 73.58 | 60.29 | no | 2.85 |
| hampton-5.2-shape-demo | 93.63 | 93.63 | 88.70 | 99.38 | 86.19 | 96.61 | no | 3.35 |
| perspective-14e-boxes-critique | 83.44 | 83.44 | 79.47 | 82.31 | 77.63 | 88.22 | no | 8.82 |
| **pooled** | 83.51 | 83.51 | 76.84 | 86.52 | 76.38 | 88.31 | — | 3.79 |

## Latency, tokens and cost per episode

Wall clock is measured around each pass at the run's concurrency, retries included, so the three pass columns add up to the total.

| episode | retake s | sentence s | trim pick s | total s | requests | input tokens | cost $ | retries | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0.35 | 1.34 | 0.67 | 2.37 | 24 | 363,424 | 0.0153 | 3 | 3 |
| hampton-5.4-assignment-demo | 0.23 | 1.56 | 0.64 | 2.42 | 29 | 490,494 | 0.0206 | 1 | 1 |
| colman-03.03-muscles-crit | 0.21 | 1.79 | 0.90 | 2.90 | 38 | 639,506 | 0.0269 | 1 | 1 |
| edges-7.01-intro | 0.52 | 1.64 | 0.69 | 2.85 | 45 | 609,990 | 0.0256 | 2 | 2 |
| hampton-5.2-shape-demo | 0.25 | 2.16 | 0.95 | 3.35 | 42 | 764,001 | 0.0321 | 1 | 2 |
| perspective-14e-boxes-critique | 0.76 | 5.88 | 2.19 | 8.82 | 131 | 2,457,843 | 0.1032 | 11 | 14 |
| **total** | 2.32 | 14.37 | 6.02 | 22.71 | 309 | 5,325,258 | 0.2237 | 19 | 23 |

## Trims

`trims emitted` counts sentences the arm gave a word range to, before the keep threshold cuts any of them. `model partials` and `human partials` are the SENTENCE POINTS partial counts at the calibrated threshold: when the editor trimmed sentences and the model trimmed none, the metric takes the 0.5 penalty.

The full/partial/removed cross-tab is not in this table: `score_episode` returns `n_partial_human` and `n_partial_model` but drops the `pair_counts` block that `sentence_scoring` computes, so there is no public way to read it without reaching into the scoring module's internals.

### jev_a (t_trim 0.1)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 0 | 0 | 70 | yes |
| edges-7.01-intro | 0 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 1 | 1 | 77 | no |
| perspective-14e-boxes-critique | 0 | 0 | 58 | yes |
| **total** | 1 | 1 | 388 | — |

### jev_a (t_trim 0.2)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 3 | 1 | 91 | no |
| hampton-5.4-assignment-demo | 2 | 1 | 77 | no |
| colman-03.03-muscles-crit | 2 | 2 | 70 | no |
| edges-7.01-intro | 3 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 6 | 2 | 77 | no |
| perspective-14e-boxes-critique | 4 | 0 | 58 | yes |
| **total** | 20 | 6 | 388 | — |

### jev_a (t_trim 0.3)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 14 | 7 | 91 | no |
| hampton-5.4-assignment-demo | 5 | 1 | 77 | no |
| colman-03.03-muscles-crit | 20 | 15 | 70 | no |
| edges-7.01-intro | 7 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 18 | 6 | 77 | no |
| perspective-14e-boxes-critique | 24 | 4 | 58 | no |
| **total** | 88 | 33 | 388 | — |

### jev_a (t_trim 0.5)

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 17 | 9 | 91 | no |
| hampton-5.4-assignment-demo | 4 | 1 | 77 | no |
| colman-03.03-muscles-crit | 33 | 25 | 70 | no |
| edges-7.01-intro | 11 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 24 | 9 | 77 | no |
| perspective-14e-boxes-critique | 40 | 5 | 58 | no |
| **total** | 129 | 49 | 388 | — |

### jev_b

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 1 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 2 | 2 | 70 | no |
| edges-7.01-intro | 4 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 0 | 0 | 58 | yes |
| **total** | 7 | 3 | 388 | — |

### jev_b_moduleretakes

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 1 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 2 | 2 | 70 | no |
| edges-7.01-intro | 4 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 0 | 0 | 58 | yes |
| **total** | 7 | 3 | 388 | — |

### jev_b_notrim

| episode | trims emitted | model partials | human partials | penalty |
|---|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 0 | 0 | 91 | yes |
| hampton-5.4-assignment-demo | 0 | 0 | 77 | yes |
| colman-03.03-muscles-crit | 0 | 0 | 70 | yes |
| edges-7.01-intro | 0 | 0 | 15 | yes |
| hampton-5.2-shape-demo | 0 | 1 | 77 | no |
| perspective-14e-boxes-critique | 0 | 0 | 58 | yes |
| **total** | 0 | 1 | 388 | — |

## Retake pass

`not real` are groups Jev scored under 0.5 on `real_k`, where nothing is cut. The last four columns take the sentences where Jev's cut and the production module's flags disagree and ask what the editor did with them, using the harness's own human sentence state: `kept` is full or partial in the real edit, `cut` is removed.

| episode | groups | not real | module fallback | losers cut by Jev | losers cut by the module | Jev cuts only | module cuts only |
|---|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 6 | 2 | 0 | 4 | 6 | 0 (0 kept / 0 cut) | 2 (1 kept / 1 cut) |
| hampton-5.4-assignment-demo | 6 | 5 | 0 | 1 | 7 | 0 (0 kept / 0 cut) | 6 (1 kept / 5 cut) |
| colman-03.03-muscles-crit | 8 | 6 | 0 | 2 | 8 | 1 (0 kept / 1 cut) | 7 (5 kept / 2 cut) |
| edges-7.01-intro | 63 | 19 | 0 | 109 | 169 | 4 (1 kept / 3 cut) | 64 (21 kept / 43 cut) |
| hampton-5.2-shape-demo | 14 | 3 | 0 | 13 | 16 | 0 (0 kept / 0 cut) | 3 (1 kept / 2 cut) |
| perspective-14e-boxes-critique | 80 | 27 | 0 | 57 | 98 | 4 (0 kept / 4 cut) | 45 (9 kept / 36 cut) |

## Where this lands on the published ladder

Reference arms re-pooled over the same 6 episode(s) from the per-episode numbers in `2026-09-11-model-plus-deterministic.json`, so they are directly comparable to the tables above.

| reference arm | episodes | SP plain | WORD plain | GRADE plain | SP + modules | WORD + modules | GRADE + modules |
|---|---:|---:|---:|---:|---:|---:|---:|
| best Luna chapters | 6 | 81.78 | 80.81 | 87.08 | 86.86 | 81.75 | 91.29 |
| shipped Opus agentic | 6 | 84.54 | 82.49 | 88.89 | 89.49 | 83.44 | 93.06 |
| Luna single call | 6 | 17.34 | 60.02 | 66.48 | 62.08 | 62.74 | 75.22 |
| deterministic baseline (um removal + retakes + delete silence) | 6 | 60.27 | 62.20 | 74.52 | 60.27 | 62.20 | 74.52 |

Best Jev arm here is jev_a (t_trim 0.3) at 70.36 SENTENCE POINTS (75.58 WORD SCORE, 3.79 s per episode), above Luna single call, deterministic baseline (um removal + retakes + delete silence).
With um removal and delete silence layered on, the same arm is 83.10 SENTENCE POINTS (76.76 WORD SCORE), above Luna single call, deterministic baseline (um removal + retakes + delete silence). That is the column the ladder actually compares on, because every published arm gets the same modules.

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-decisions.jsonl`
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-requests.jsonl`
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-timing.json`
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-notes.md`
- summary JSON: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-summary.json`
