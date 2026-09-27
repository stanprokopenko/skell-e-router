Developer-facing note, written 2026-09-26 by a read-only scout. No model calls, $0. Sources: `D:\solar-sailer\benchmarks\roughcut\scripts\export_bench_page.py`, its output `C:\Users\Stan\Documents\GitHub\solar-sailer\website-docs\static\data\rough-cut-bench.json` (updated 2026-09-23, shipped_release v0.4.15; `D:\solar-sailer` has no `website-docs`), and `D:\solar-sailer\benchmarks\roughcut\results\2026-09-11-model-plus-deterministic.{md,json}` (the same reference file `roughcut_jev_report.py` reads, regenerated in place on 2026-09-23 13:57 despite the 09-11 name).

# Bench page 85.05 vs the Jev ladder's 86.47

## Why the two numbers differ

They are two different arms on two different layers of the same metric. The page's 85.05 is `claude-opus-5-5-high · agentic · API · rules1` (exp26, run 2026-09-22) scored on the model's own cut with no modules layered. The Jev docs' 86.47 is `claude-opus-5-high · agentic · Claude Code · rules1` (the 2026-07-27 arm) with um removal and delete silence layered on. Everything else is identical: SENTENCE POINTS at each run's Neutral threshold, the same 18 ladder episodes, pooled by sentence count (8943 sentences), the same scorer and answer keys. Both numbers come from the same `model-plus-deterministic.json` the exporter and `ladder()` both read.

The page headline and its `best` flag and record timeline use the plain `weighted_grade` (no modules). The with-modules numbers are a toggle on the page, the `umm_silence` view ("Deterministic modules applied"). On the page the Opus 5 Claude Code agentic arm shows 83.45 plain and 86.47 with modules, so the page and the Jev docs agree on that arm to the cent.

## Same arms, both bases

| arm | page plain SP | page SP with modules (`umm_silence` view) | Jev ladder SP with modules | basis differences |
|---|---:|---:|---:|---|
| claude-opus-5-5-high, agentic, API, rules1 (page `best`) | 85.05 | 87.44 | not quoted | page headline is the plain layer; Jev ladder never quotes this arm |
| claude-opus-5-5-low, chapters, API, rules5 (shipped at v0.4.15) | 84.43 | 87.57 | not quoted | highest with-modules arm on the page; not quoted |
| claude-opus-5-5-high, chapters, API, rules5 | 84.21 | 86.59 | not quoted | not quoted |
| claude-opus-5-5-high, chapters, Claude Code, rules5 (shipped at v0.4.15) | 84.02 | 86.53 | not quoted | not quoted |
| claude-opus-5-high, agentic, Claude Code, rules1 (Jev "shipped Opus agentic") | 83.45 | 86.47 | 86.47 | identical basis; page flags it not shipped |
| claude-fable-5-1-high, chapters, Claude Code, rules5 | 84.01 | 86.45 | 86.45 (heldout ranking only) | identical basis |
| claude-opus-5-high, agentic, API, rules1 | 83.36 | 86.30 | 86.30 (heldout ranking only) | identical basis |
| claude-opus-5-5-low, agentic, API, rules1 | 82.78 | 85.72 | not quoted | not quoted |
| gpt-6-luna-xhigh, chapters, API, rules5 | 79.82 | 84.25 | not quoted | beats the "best Luna chapters" 83.71 the Jev docs quote |
| gpt-5.6-luna-xhigh, chapters, API, rules5 (Jev "best Luna chapters") | 79.96 | 83.71 | 83.71 | identical basis |
| jev-1.13.0 sentence pass, jev prompts v3 (Jev "jev_a v3") | 72.19 | 80.47 | 80.47 | identical basis |

All rows: SENTENCE POINTS, 18 episodes, pooled by sentence count. Plain is the model's own cut, which already includes Retakes. "With modules" adds Um Removal and Delete Silence at shipped defaults on top.

## Is Opus 5.5 newer than Opus 5

Yes. `claude-opus-5-5` was first benchmarked on 2026-09-22 (`results/2026-09-22-opus-5-5-rough-cut-benchmark.md`, which calls it the new best score and notes Claude Code CLI 2.1.280 or newer is needed to run it). The Opus 5 arms date from 2026-07-27 and 2026-09-10. On the Jev docs' basis (SP with modules, 18 episodes), Opus 5.5 high agentic API is 87.44 and the top with-modules arm on the page is Opus 5.5 low chapters API at 87.57.

## What is wrong or stale in the Jev docs

The 86.47 figure itself is correct for the arm it names. Two things around it are out of date.

The label "shipped Opus agentic" is stale. `REFERENCE_ARMS` in `scripts/jev_real/roughcut_jev_report.py` hardcodes `opus5-cc-agentic` as "shipped". The page's shipped flags follow the newest installer release (v0.4.15): Opus 5.5 low chapters API, Opus 5.5 high chapters Claude Code, and gpt-6-astra-high chapters Codex CLI. The Opus 5 agentic arm is not shipped. Even the 09-11 version of the reference file marked Fable 5.1 chapters as shipped, not Opus 5 agentic.

Treating 86.47 as the top of the ladder is wrong for every doc written after 2026-09-22. `ladder()` quotes only three reference arms, so docs generated after the reference file gained the Opus 5.5 rows still present 86.47 as the ceiling to beat. Four arms sit above it with modules (87.57, 87.44, 86.59, 86.53). Affected: `roughcut-hybrid-f1opus.md` and `roughcut-hybrid-f1luna.md` line 32 placement sentences ("below shipped Opus agentic ... rank 2 of 9"), `roughcut-route2-routing.md`, `roughcut-hybrid-luna.md`, and the `roughcut-jev-f1/f2/f3.md` ladder tables. Their relative placements are correct; the framing that the Opus row is the best published arm is not.

`roughcut-jev-heldout.md` section "Where the Jev arms land on the published 18-episode ladder" (committed 2026-09-20) was correct when written but is now stale. Its ranked table puts Opus 5 agentic Claude Code at rank 1 with "(shipped)" and omits the five Opus 5.5 arms and gpt-6-luna. Re-ranked today, v3 jev_a at 80.47 falls from 17th to 23rd (22 page arms now score above it with modules), and gpt-6-luna chapters (84.25) displaces gpt-5.6-luna chapters rules5 (83.71) as the best Luna chapters arm.

The "best Luna chapters 83.71" label has the same problem: it names gpt-5.6-luna-xhigh chapters rules5, while gpt-6-luna-xhigh chapters rules5 scores 84.25 on the same basis.

No number in the Jev docs is miscomputed. The two sanity checks in `roughcut-jev-heldout.md` (Luna 83.71, Opus 86.47 matching the ladder to the cent) still hold against the current file.
