# Jev vs Luna vs production on the real chat routing decision

Developer notes for `scripts/jev_real/routing_bench.py`. Run date 2026-09-19, total spend $0.32.

## What ran

565 labeled user messages from `skell-e-web/backend/benchmarks/routing/routing-labels.jsonl`, full text joined from the gitignored export `backend/scripts/temp/chat-history-export.jsonl`. Three arms, one call per case per arm, 1695 calls, concurrency 4, zero failures and zero unparseable replies.

- `gemini_prod` — production path: `rag.routing.build_classifier_prompt(message, history, None, prior_pct=23)`, gemini-3.5-flash-lite, temperature 0, max_tokens 200, parsed by `rag.routing.parse_classifier_output`.
- `luna_low` — the same prompt string, gpt-5.6-luna, reasoning_effort low, max_tokens 4096, same parser.
- `jev` — one `classify("jev-1.13.0", state, questions)` request per case: a `tier` choice plus four nouls (`wants_quick`, `stakes`, `customer_facing`, `simple_pull`). State is a JSON object with `setting`, `earlier_turns`, `attachments`, `sender_history`, `new_user_message`.

Both sides see the same context. `structured_turns()` reuses `rag.routing`'s own turn selection (last two user plus last two assistant, 400-character trim) and the message goes through `_clip_message`, so the Jev state is the production prompt's content in structured form rather than rendered text.

Reproduce:

```
python scripts/jev_real/routing_bench.py                 # plan and estimate, no calls
python scripts/jev_real/routing_bench.py --run --limit 8 # smoke test
python scripts/jev_real/routing_bench.py --run           # all 565, ~5 minutes
```

`TYPESAFE_API_KEY`, `GEMINI_API_KEY` and `OPENAI_API_KEY` live at Machine scope on this box and are not in a fresh shell's process environment. Hydrate them first:

```powershell
foreach ($k in @('TYPESAFE_API_KEY','GEMINI_API_KEY','OPENAI_API_KEY')) {
  if (-not [Environment]::GetEnvironmentVariable($k,'Process')) {
    [Environment]::SetEnvironmentVariable($k, [Environment]::GetEnvironmentVariable($k,'Machine'), 'Process') } }
```

## Numbers

Model-only: the arm decides all 565, rules ignored. Production-shaped: `decide_rules` runs first (270 settled, 3 explicit-model rows dropped as unscorable, 292 unsettled) and the arm only answers what the rules left open.

| arm | model-only acc | missed-big | false-big | prod-shaped acc | missed-big | false-big | median | p95 | $/1k calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini_prod | 86.9% | 21 | 53 | 88.6% | 31 | 33 | 0.61s | 0.77s | $0.301 |
| luna_low | 87.4% | 13 | 58 | 87.9% | 23 | 45 | 1.09s | 1.81s | $0.204 |
| jev | 92.4% | 23 | 20 | 91.3% | 33 | 16 | 0.21s | 0.28s | $0.062 |

Labels: 129 big, 436 fast. Every arm parsed cleanly, so the strict scoring (error counts as wrong) and the production fallback scoring (error routes to fast) are identical here.

Jev policy variants, model-only:

| policy | acc | missed-big | false-big |
| --- | --- | --- | --- |
| (a) raw `tier` choice | 92.4% | 23 | 20 |
| (b) composed from nouls | 76.1% | 44 | 91 |
| (c) choice, big when confidence < 0.3 unless wants_quick | 92.0% | 13 | 32 |
| (d) P(big) >= 0.40, fitted on odd cases | 93.3% held out | 3 | 16 |

The composed rule (b) is the clear loser. `stakes >= 0.5` fires on far more messages than the label calls big, so hand-set 0.5 cutoffs on the nouls are not a usable policy without fitting. (c) is the interesting one: it buys Luna's missed-big count (13) at Jev's price and keeps false-big at 32, well under either text model.

(d) is a held-out result. The sweep covered 19 P(big) thresholds plus 243 composed-threshold combinations, fitted on the 283 odd-numbered cases and reported on the 282 even-numbered ones. Best on fit was P(big) >= 0.40, which scored 93.3% on the held-out half (3 missed-big, 16 false-big). Thresholds 0.40 to 0.60 all tied at 91.9% on fit, so treat the exact number as a range, not a tuned constant.

Hybrid, Jev first and escalate to Luna below a confidence cutoff:

| cutoff | escalated | acc | missed-big | false-big | blended $/1k |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 5.7% | 93.3% | 14 | 24 | $0.073 |
| 0.4 | 11.5% | 92.6% | 14 | 28 | $0.085 |
| 0.6 | 20.7% | 90.1% | 15 | 41 | $0.104 |

Escalating more makes it worse past 0.2, because the cases Jev is unsure about are exactly the ones Luna over-escalates on.

Jev accuracy by confidence quartile (cuts at 0.69 / 0.95 / 0.99):

| quartile | n | acc | missed-big | false-big |
| --- | --- | --- | --- | --- |
| q1 lowest | 146 | 78.8% | 17 | 14 |
| q2 | 154 | 92.9% | 6 | 5 |
| q3 | 139 | 99.3% | 0 | 1 |
| q4 highest | 126 | 100% | 0 | 0 |

Calibration is real: 40 of Jev's 43 errors sit in the bottom two quartiles, and it is perfect on the top quarter.

Agreement on raw predictions: gemini vs luna 88.5%, gemini vs jev 88.5%, luna vs jev 88.3%. The three disagree with each other about as much as each disagrees with the labels, so they are not making the same mistakes. Jev is right where Luna is wrong 47 times; Luna is right where Jev is wrong 19 times.

## Failure patterns

Jev's misses cluster on short marketing-copy asks that read like a data pull: "generate titles for that episode", "what are some punchy YouTube titles that play off this sentence", "I want to create a h3 title heading for 'Who This Course is For'". The prompt's own rule says thumbnail ideas are fast, and Jev generalizes that to titles and headings, which the labels call big. Its false-bigs are mostly customer-reply framing on a routine task: "Tell this guy his VAT was refunded with his order", "give me a polished answer i can add the the FAQ".

Luna's misses are sparse but confident: "Give me a non concise answer" (confidence 0.99, labeled big) and "Now do the same task for these two transcripts". Its 58 false-bigs are the real problem, and they are overwhelmingly data pulls it talks itself into escalating: "give me the average 90 days sales for all courses not just 3rd party courses, exclude major outliers", plus repeated "same thing" / "same thing here" follow-ups where it re-reads the thread and escalates. Luna errors by category are led by data-pull (17) and continuation (14). Jev's are led by continuation (10) and marketing-copy (6).

## Caveats

- Sticky state comes from the labels, not from each arm's own earlier decision, the same choice the skell-e-web harness makes. Real production compounds its own mistakes; these numbers do not.
- Luna cost uses the router's own per-million rates with the cached-input split (0.20 / 0.02 / 1.20). Prompts differ per case, so cache hits are near zero and the number is effectively uncached pricing. Gemini and Jev costs come straight off the router response.
- Jev bills a flat 102 output tokens per request regardless of how many questions you ask, so the four extra nouls are nearly free. Dropping them would not save money.
- Attachments were always "none": the export does not carry them. Production sometimes does.
- The results file stores only the committed 120-character `text_head` and the model's own one-sentence reason (trimmed to 200 characters). No full customer or staff message text is written.
- One deviation from the brief: the runner records the parsed tier, confidence and reason rather than the complete raw completion, for the same data-exposure reason.

## Files

- `scripts/jev_real/common.py` — timed call wrapper, append-only jsonl writer, latency summary, `--run` gate.
- `scripts/jev_real/routing_bench.py` — the runner.
- `docs/jev-real/routing-results.jsonl` — 1695 rows plus a metadata header carrying the exact question set.
- `docs/jev-real/routing-summary.json` — every number above, machine-readable.
