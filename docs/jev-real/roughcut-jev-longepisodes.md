# Long episodes in the Jev rough cut: fitted context window and block size

Mechanism work on `scripts/jev_real/roughcut_jev.py`, prompt version v3 throughout. No prompt text changed. Every number below is the `jev_a` arm at `--t-trim 0.3`, layered (um removal plus delete silence), scored by `scripts/jev_real/roughcut_jev_report.py`.

Read this first: `perspective-13d-critique` and `greco-2.2-thumbnailing` are held-out episodes, and `flanders-03-thematic-crit` is too. This was a mechanism fix rather than a prompt change, so nothing here was tuned against the labels. The recommendation at the bottom is still a decision taken after seeing held-out numbers, and it should be read that way.

## What changed

The sentence pass used to send a fixed window of 200 corpus sentences either side of the target block whenever the whole rendered transcript went over 24,000 estimated tokens. Three episodes hit that rule in the held-out run, and the two longest of them, 13d (1,752 sentences) and greco (1,388), ran their entire sentence pass on that window. It is now a fitted window: the largest symmetric window around the block whose rendered transcript fits the budget, found by binary search, recomputed per block.

Two budgets bind it.

`--context-tokens`, default 24,000, covers one request's state: rules, targets and transcript together. Targets now count against it, which they did not before. The provider caps the state plus the longest question at 32,000 tokens, and the state is what that cap is about.

`REQUEST_TOKEN_CAP`, a module constant at 46,000 estimated tokens, covers the whole request, state plus questions. The provider caps a request at 64,000 tokens. In the held-out log every sentence-pass request estimated over 49,176 tokens came back a deterministic 400 with `category: invalid_request`, nothing under that estimate did, and the largest accepted request measured 64,127 real input tokens. This second budget is the one that usually binds, because the trim questions list every word of every target sentence and take roughly 1,070 estimated tokens per target.

Also new:

- `--block N`, default 25, sets target sentences per sentence-pass request. It lands on every request row as `block_size`.
- Every request row carries `window_sentences` (transcript lines sent), `window_radius` (corpus sentences either side, null when the whole transcript went), and `est_over_actual` (the 4-chars-per-token estimate divided by the provider's own `input_tokens`).
- Every job now has a half-size fallback window, not only the jobs that started whole. A block that comes back a context error is re-sent at half the rendered transcript.
- Per-pass timing gains `window_sentences_min/median/max`, `whole_transcript_requests`, `est_input_tokens_max`, `over_request_cap` and `est_over_actual_median`.

An episode whose whole transcript fits both budgets still sends the whole transcript. On the fit six nothing about the requests changes.

What the window actually became, at the default block size: 13d went from 401 sentences to a median of 1,250 of its 1,643 kept sentences, greco from 401 to a median of 848 of 1,219.

## The two long held-out episodes, before and after

Run: `--episodes perspective-13d-critique greco-2.2-thumbnailing --out roughcut-jev-fitwindow`, 197 requests, $0.310, plus $0.009 of repairs for four blocks that failed three times each on flaky provider errors. Every one of the 3,140 sentences has an answer in the final decisions file. The held-out baseline left two blocks (50 sentences) unanswered in 13d and none in greco.

At each run's own calibrated keep threshold, which is how the report script writes it:

| episode | run | threshold | SP | WORD | GRADE | wrong drops | wrong keeps | kept ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | held-out, 200-sentence window | 2.5 | 78.09 | 79.05 | 83.28 | 129 | 257 | 101.97 |
| perspective-13d-critique | fitted window | 2.7 | 80.47 | 79.37 | 82.88 | 181 | 166 | 90.55 |
| greco-2.2-thumbnailing | held-out, 200-sentence window | 2.5 | 65.09 | 68.55 | 71.80 | 24 | 435 | 178.17 |
| greco-2.2-thumbnailing | fitted window | 2.7 | 70.94 | 70.31 | 73.30 | 39 | 343 | 162.42 |

Those two rows are not comparable, and the difference is not small. The threshold is calibrated per run over the pooled episodes in it: the baseline pooled 13 episodes and landed on 2.5, this run pooled 2 and landed on 2.7. A higher threshold cuts more, which is most of what moved. Scoring both runs' decisions at the same 2.5 gives the real comparison:

| episode | run | SP | WORD | GRADE | wrong drops | wrong keeps | kept ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| perspective-13d-critique | held-out, 200-sentence window | 78.09 | 79.05 | 83.28 | 129 | 257 | 101.97 |
| perspective-13d-critique | fitted window | 79.27 | 79.58 | 83.38 | 132 | 227 | 101.46 |
| greco-2.2-thumbnailing | held-out, 200-sentence window | 65.09 | 68.55 | 71.80 | 24 | 435 | 178.17 |
| greco-2.2-thumbnailing | fitted window | 65.24 | 68.02 | 71.20 | 25 | 434 | 178.06 |

So: roughly three times the context buys 13d about 1.2 SENTENCE POINTS and 30 fewer wrong keeps, and buys greco nothing at all. Greco's problem is not that it cannot see its own transcript. It keeps 178% of the editor's duration and makes 434 wrong keeps against 25 wrong drops, and a wider window does not touch either number.

The two episodes also got slower and dearer, because each request now carries about three times the transcript: 13d went 13.6 s to 18.7 s and $0.125 to $0.173, greco 13.0 s to 17.2 s and $0.116 to $0.136.

## Block size, on flanders

`flanders-03-thematic-crit`, 1,309 sentences, an episode whose whole transcript fits the state budget. In the held-out run it sent the whole transcript on 66 of its 91 sentence requests, which put those requests at 48,000 to 52,500 estimated tokens, and it is the episode where retries dominated wall clock.

| run | block | sentence wall | requests (first attempt + retries) | errors | failed blocks | cost | median window | SP | WORD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out, whole transcript | 25 | 18.99 s | 91 (53 + 38) | 38 | 0 | $0.125 | whole on 66 of 91 | 77.14 | 77.61 |
| fitted window | 25 | 12.70 s | 64 (53 + 11) | 11 | 0 | $0.131 | 1,026 | 77.30 | 77.02 |
| fitted window | 50 | 19.11 s | 74 (27 + 47) | 70 | 23 | $0.010 | 49 | collapsed | collapsed |

SP and WORD in that table are at the fixed 2.5 threshold, for the same reason as above. At its own calibrated threshold (2.1) the fitted block-25 run reads SP 78.97, WORD 78.38, GRADE 84.01.

Block 25 with a fitted window is the result worth having: same quality inside noise, a third off the wall clock, and errors down from 38 to 11, because the requests no longer sit on top of the provider's 400 boundary. Cost went up 5%.

Block 50 does not work and cannot be made to work at this prompt. The trim questions for 50 targets are about 55,000 estimated tokens on their own, so 26 of the 27 blocks were over the 46,000 ceiling before a single transcript line went in, the fit squeezed the window down to a median of 49 sentences and it still did not help. 64 of 74 requests came back the deterministic 400, 23 of 27 blocks never answered, and 1,150 of 1,309 sentences have no score. The run cost a cent and told us one thing: the question payload, not the transcript, is what a bigger block spends. Anything above roughly 35 targets per request starves the transcript window and then breaks.

## Estimate against reality

The 4-chars-per-token estimate runs low. Median `est_over_actual` was 0.774 on the two-episode run and 0.783 on flanders block 25, with a per-request range of 0.68 to 0.84. Real input tokens are about 1.29 times the estimate, and the tail of that ratio is what matters: a request estimated at 46,000 is usually 59,000 real tokens but can be 67,600.

That tail is visible in the run. One 13d block failed six times in a row at 45,996 estimated tokens with a bare `PROVIDER_ERROR: Provider request failed` (not the 400 shape), then answered on the first try when re-sent with `--context-tokens 18000`. The generic provider error is therefore not purely flaky: at the top of the size band some of it is the size. Across the fitted-window run, 25% of first attempts failed at 44,000 to 46,000 estimated tokens, against about 20% at every size in the held-out run.

One request was also rejected at only 26,230 estimated tokens, the last block of 13d, where two targets meant tiny questions and the whole transcript went in. Its state alone was near 24,000 estimated, or about 31,000 real, which is the 32,000 state cap. The 24,000 default is too close to that cap once the estimate's low tail is taken into account.

## Recommended defaults

Keep `--block 25`. Larger blocks spend the request on questions rather than transcript, and 50 is off the cliff.

Lower `--context-tokens` from 24,000 to 22,000. At 24,000 a state can reach about 35,000 real tokens on the low tail of the ratio, over the provider's 32,000 state cap, which is what rejected 13d's last block. 22,000 stays under it at the worst ratio observed (0.68).

Lower `REQUEST_TOKEN_CAP` from 46,000 to 40,000. 40,000 estimated is at most about 59,000 real tokens at the worst observed ratio, comfortably under the 64,000 request cap, where 46,000 can reach 67,600 and does sometimes get refused. The cost of the smaller ceiling is a smaller window, and the fixed-threshold table says that window is worth about 1 SENTENCE POINT on the one episode where it helped at all and nothing on the other. Fewer three-attempt failures is worth more than that.

I have not changed either default in code. The measurements above were taken at 24,000 and 46,000, and changing the constants now would leave the numbers describing something the script no longer does. Both are one-line changes once the call is made.

The thing these numbers do not fix is the one the held-out write-up already named. Greco keeps 178% of the editor's duration with 434 wrong keeps against 25 wrong drops, and neither a wider window nor a different block size moved it. That is a prompt or a threshold problem, not a context problem.

## Files

- Fitted window, two episodes: `roughcut-jev-fitwindow-{requests.jsonl,decisions.jsonl,timing.json,summary.json,notes.md}`
- Flanders block 25: `roughcut-jev-block25-flanders-*`
- Flanders block 50: `roughcut-jev-block50-flanders-*`
- Baseline: `roughcut-jev-heldout-v3-*`, written up in `roughcut-jev-heldout.md`
