# Jev for classification

This note is for Stan, with reproduction details for developers at the end. Research began September 17; authenticated testing completed September 19, 2026.

Jev works through our router and this account. The benchmark selected examples already covered by deterministic application rules, then asked models to reproduce those rules. That was the wrong task selection for evaluating whether Jev could replace an existing AI step. The recorded results are valid for those fixtures: 109 of 111 correct decisions, versus Luna's 111 of 111, about 3.9 times faster and 4.0 to 4.7 times cheaper per call. The intended adoption comparison remains unfinished. No production recommendation should rely on this accuracy score.

## What launched

TypeSafe AI, founded by former OpenAI researcher Diogo Almeida, announced Jev on September 15. It takes text or structured data and returns a choice, a score, or a probability that a statement is true. It cannot write a response or explanation. Its output restrictions prevent invented labels, but they do not make every decision correct. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [introduction](https://docs.typesafe.ai/introduction).

TypeSafe advertises early access and is admitting developers from a waitlist. It has not published a date when everyone will receive access. Vercel announced its own Jev integration on September 16. Its model listing is evidence of a supported API route, not evidence that our account can call it. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [Vercel announcement](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway).

Stan created a TypeSafe account and configured TYPESAFE_API_KEY in the Windows machine environment. The router successfully called Jev on September 19. Keys remain outside the repository and result files. The official TypeSafe skill and license are installed in C:/Users/Stan/.claude/skills/typesafe-ai for relay synchronization.

## Published prices and speed

| Model and route | Input per million tokens | Output per million tokens | Measured speed in this comparison |
| --- | ---: | ---: | --- |
| Jev direct | $0.042 | $0 | 0.213 seconds median |
| Jev through Vercel | $0.04 listed | Not separately shown in the listing | Not tested |
| gpt-5.6-luna direct | $0.20 | $1.20 | 0.835 seconds low, 0.827 seconds high, median |

Luna cached input is $0.02 per million tokens. Prices above are the standard rates for these short requests. Jev direct input is about 4.76 times cheaper than uncached Luna input. Actual savings also depend on prompt size, caching and Luna's reasoning tokens. As an illustration, 1,000 input tokens and 3 output tokens cost $0.000042 on Jev and $0.0002036 on Luna before reasoning or caching. Those are calculated examples, not measured call costs. [TypeSafe models and pricing](https://docs.typesafe.ai/models), [Vercel listing](https://vercel.com/ai-gateway/models/jev), [OpenAI Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

TypeSafe claims 70 to 500 milliseconds per call. Its headline speed and cost multiples use selected workflows and stronger models, so they do not establish a speed advantage over Luna on our tasks. The matched measurements below establish the difference on our selected cases. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Our comparison

The sample file contains 37 developer-authored test fixtures from skell-e-web. These exercise real application tasks, but they are not a held-out sample of customer traffic.

| Task | Samples | Expected labels |
| --- | ---: | --- |
| Choose the chat model tier | 25 | 15 big, 10 fast |
| Check support message completeness | 12 | 4 sufficient, 8 insufficient |

The expected labels come from existing deterministic test assertions. Each case records its original file, test function and commit. No customer exports or labels invented by another model are included. Both task families already have deterministic rules for these cases. Matching those rules tests instruction following; it does not justify replacing free local rules with an API call.

The runner uses the same state, instructions and label definitions for each model. Jev gets its native choice question. Luna gets a label-only prompt at explicitly pinned low and high reasoning efforts. Jev has no documented reasoning-effort setting. A call that fails or returns an invalid label counts as incorrect. Calls run sequentially in a seeded shuffled order, and wall-clock latency includes router retries. Token-based cost estimates are recorded separately from the conservative spend reservation.

Jev does not fit the benchmark repository's ordinary writing and reasoning prompts, which use ask_ai and require low/high variants. This targeted classification runner is kept in the router repository. It does not add fake Jev reasoning variants to benchmark/config.yaml or mix these fixture scores into the benchmark leaderboard.

The matched run made 333 sequential calls on September 19: 37 cases times three repetitions for each model setting. All calls succeeded, returned valid labels and supplied usage. Luna used a 4,096-token output cap including reasoning; no response hit that cap and no cached input tokens were reported. The returned model identifiers were jev-1.13.0 and gpt-5.6-luna.

| Variant | Routing correct | Completeness correct | Total decisions correct | Median seconds | 95th-percentile seconds | Estimated cost per call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.13.0 | 75/75 | 34/36 | 109/111, 98.2% | 0.213 | 0.287 | $0.00003259 |
| Luna low | 75/75 | 36/36 | 111/111, 100% | 0.835 | 1.282 | $0.00013181 |
| Luna high | 75/75 | 36/36 | 111/111, 100% | 0.827 | 1.507 | $0.00015388 |

Jev was 3.92 times faster than Luna low and 3.88 times faster than Luna high by median latency. Its average call cost was 4.04 times lower than Luna low and 4.72 times lower than Luna high. At these measured prompt sizes, a million calls would cost approximately $32.59 for Jev, $131.81 for Luna low and $153.88 for Luna high. That extrapolation excludes changes in traffic, caching, retries and pricing. The very small median difference between Luna low and high is not evidence that high reasoning is generally faster.

Both Jev mistakes were the same case, "refund please," on two of its three attempts. The existing application rule accepts this as sufficient text. Jev returned insufficient with confidence 0.00 and 0.06. On its correct attempt, confidence was only 0.02. This suggests a useful fallback signal for that case; it does not establish a generally reliable confidence threshold or calibration. Luna passed this case on every matched-run attempt. Its earlier low-effort baseline had missed it once, so neither model should be assumed deterministic from this small sample. See the [complete request and all three outcomes](jev-refund-classification.md).

There are 37 distinct fixture inputs, not 111 independent cases per model. Repetitions check consistency and timing. The labels and prompts were fixed before Jev testing, and no prompts were tuned after seeing mistakes. Both task families already have deterministic rules for these examples, so keep those free local rules. The next suitable adoption step is a limited test on harder routing cases with independent labels and a fallback to Luna or human review. No production routing changed in this task.

The [matched raw results](jev-classification-matched.jsonl) contain every expected and returned label, confidence, model identifier, token count, elapsed time and estimated cost. The [summary](jev-classification-summary.json) records the aggregates and the raw file hash. The September 17 [Luna baseline](jev-luna-baseline.jsonl) and [Luna pilot](jev-luna-pilot.jsonl) remain historical evidence and are excluded from the matched table. The [Jev access pilot](jev-access-pilot.jsonl) is also excluded.

## Later threshold replay

Re-scoring the saved ticket-completeness probabilities with a rule that flags insufficient only at probability 0.60 or above fixes both misses and changes no other answers. The fixture score becomes 111/111, with no new model calls. This is an adjustment made after inspecting the failures, not fresh evidence of model quality or a production change. The actual support pipeline uses deterministic code for this advisory check and has no probability threshold. Its warning helps the drafting agent identify incomplete content; it does not automatically close or spam tickets.

## Spend

| Calls | Estimated USD |
| --- | ---: |
| Earlier Luna baseline and pilot | $0.010665600 |
| Jev access pilot | $0.000036540 |
| Matched Jev calls | $0.003617586 |
| Matched Luna low calls | $0.014630400 |
| Matched Luna high calls | $0.017080800 |
| Three mixed-question integration checks | $0.000049266 |
| Total task spend | $0.046080192 |

Total spend was about 4.6 cents against the $20 cap. These estimates use provider-reported token counts and official prices. They are not an invoice reconciliation. Successful-response usage does not establish whether a provider billed any unseen retry work.

## Integration and limits

The direct API uses POST https://api.typesafe.ai/v1/systemone with a bearer key. The current pinned model is jev-1.13.0. The router integration uses a separate classify call and retains the native answers, probabilities, confidence and usage. Keeping classification separate from chat follows the router's existing separate methods for embeddings and image generation. [API reference](https://docs.typesafe.ai/api), [model names](https://docs.typesafe.ai/models).

TypeSafe documents two input limits for Jev 1.13: 64,000 tokens across the state and all questions, and 32,000 across the state and the longest question. It warns about counting, arithmetic, date handling and adversarial instructions in the state. We do not treat confidence as proof of correctness. [Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

Vercel exposes Jev through its experimental AI SDK evaluation method. It does not support evaluation through its OpenAI-compatible chat endpoint. The initial Python integration therefore targets the documented TypeSafe HTTP API. [Vercel evaluation API](https://vercel.com/docs/ai-gateway/modalities/evaluation).

## Reproduction and verification

Developer details follow. The fixture file is [jev-classification-samples.json](jev-classification-samples.json), and the runner is [benchmark_jev_classification.py](../scripts/benchmark_jev_classification.py). Running without --run prints the planned call count and spend reservation without calling a model.

```powershell
python scripts/benchmark_jev_classification.py
python scripts/benchmark_jev_classification.py --run --repeats 3 --budget 6 --output docs/jev-classification-matched.jsonl
```

The process must have OPENAI_API_KEY and TYPESAFE_API_KEY. Keep keys out of commits and messages. The runner imports this router checkout and does not require an upgrade to the shared Python installation or benchmark's pinned environment. It refuses to overwrite an existing result file.

The saved output filename already exists, so use a new filename for another run. The completed runner used source commit 5d5afe9. All calls went through skell-e-router. The task did not install the TypeSafe SDK or change the shared Python or benchmark environment. Test customer traffic separately with independent labels before changing production routing.

Verification so far: all 37 fixture inputs and labels match the source tests, and an independent check ran all 12 completeness fixtures through the original deterministic logic. The full offline suite passed 992 tests plus 17 subtests. Independent code review found no high-severity issues and two smaller defects, both fixed. After adding the score-limit fix and three boundary cases, all 66 classification tests passed. The built 3.30.0 wheel then passed 289 focused classification, model, response and credential-error tests. Offline checks block network access and use synthetic credentials. Authenticated checks now establish that this account can use all three router aliases, jev, jev-1.13.0 and jev-latest. Each returned jev-1.13.0. A single request containing Choice, Noul and Score succeeded through every alias and preserved native probabilities, usage and scoring. See [live contract evidence](jev-live-contract.json).

Independent results review confirmed all 333 sample/model/repetition combinations, both evidence hashes, every label and cost calculation, the latency summaries and the complete spend total. It found no material errors.

The September 17 wheel is dist/skell_e_router-3.30.0-py3-none-any.whl, SHA-256 d9ae154b1378bcb653f1e09b91d400da36ba9647065b8246dd61bbf39a88a87a. Its runtime code matches the authenticated run; this report and the current documentation also contain the September 19 verification. No global package installation or benchmark environment upgrade has occurred. Ordinary pytest startup on this PC encounters an unrelated installed Logfire plugin with a missing dependency; the existing offline test launcher disables plugin auto-loading, so no shared environment repair was needed.

## Next session

Developer continuation notes. Stan ended the session after identifying the task-selection mistake. The original request to evaluate an actual AI classification step remains open; do not repeat the deterministic completeness test as a substitute. Inspect a real model call in a consuming project, preserve its purpose and policy, select representative inputs with justified labels, then compare Jev and Luna. Retain the existing fixture results as historical evidence rather than overwriting them. No production migration is requested.

Use classify() for Jev and ask_ai() for Luna through this router checkout. Jev does not support the generative benchmark suite or low/high reasoning controls. TYPESAFE_API_KEY is saved in the Windows machine environment; OPENAI_API_KEY is available in the user environment. A process launched before those variables were saved may need to load them into its child environment. Never print or commit their values. The task has spent an estimated $0.046080192 of its standing $20 API budget, leaving about $19.9539.

The canonical TypeSafe skill is C:/Users/Stan/.claude/skills/typesafe-ai/SKILL.md. Its shortened description was verified in both that file and the relay-synced Codex copy. Existing integration tests and source files are listed above. The support check we mistakenly converted lives in skell-e-web/backend/rag/ticket_quality.py, called by support_pipeline.py while preparing agent context. Production deployment itself was not inspected during this clarification.
