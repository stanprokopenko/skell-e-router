# Jev for classification

This note is for Stan, with reproduction details for the next developer at the end. Researched September 17, 2026.

Jev is available through an API, but we have not established access for this account. It looks worth testing for repeated classification work. We should keep the current production models until the comparison runs successfully.

## What launched

TypeSafe AI, founded by former OpenAI researcher Diogo Almeida, announced Jev on September 15. It takes text or structured data and returns a choice, a score, or a probability that a statement is true. It cannot write a response or explanation. Its output restrictions prevent invented labels, but they do not make every decision correct. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [introduction](https://docs.typesafe.ai/introduction).

TypeSafe advertises early access and is admitting developers from a waitlist. It has not published a date when everyone will receive access. Vercel announced its own Jev integration on September 16. Its model listing is evidence of a supported API route, not evidence that our account can call it. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [Vercel announcement](https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway).

No TypeSafe or Vercel AI Gateway credential was found in this process, Windows user or machine environment settings, or the benchmark and orchestrator environment files. Only credential names and presence were inspected for this check. We asked Stan whether he already has access. No account was created or waitlist submission sent.

## Published prices and speed

| Model and route | Input per million tokens | Output per million tokens | Measured speed in this comparison |
| --- | ---: | ---: | --- |
| Jev direct | $0.042 | $0 | Pending account access |
| Jev through Vercel | $0.04 listed | Not separately shown in the listing | Not tested |
| gpt-5.6-luna direct | $0.20 | $1.20 | 0.78 seconds low, 0.94 seconds high, median |

Luna cached input is $0.02 per million tokens. Prices above are the standard rates for these short requests. Jev direct input is about 4.76 times cheaper than uncached Luna input. Actual savings also depend on prompt size, caching and Luna's reasoning tokens. As an illustration, 1,000 input tokens and 3 output tokens cost $0.000042 on Jev and $0.0002036 on Luna before reasoning or caching. Those are calculated examples, not measured call costs. [TypeSafe models and pricing](https://docs.typesafe.ai/models), [Vercel listing](https://vercel.com/ai-gateway/models/jev), [OpenAI Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

TypeSafe claims 70 to 500 milliseconds per call. Its headline speed and cost multiples use selected workflows and stronger models, so they do not establish a speed advantage over Luna on our tasks. We need matched measurements. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Our comparison

The sample file contains 37 developer-authored test fixtures from skell-e-web. These exercise real application tasks, but they are not a held-out sample of customer traffic.

| Task | Samples | Expected labels |
| --- | ---: | --- |
| Choose the chat model tier | 25 | 15 big, 10 fast |
| Check support message completeness | 12 | 4 sufficient, 8 insufficient |

The expected labels come from existing deterministic test assertions. Each case records its original file, test function and commit. No customer exports or labels invented by another model are included. Both task families already have deterministic rules for these cases. Matching those rules tests instruction following; it does not justify replacing free local rules with an API call.

The runner uses the same state, instructions and label definitions for each model. Jev gets its native choice question. Luna gets a label-only prompt at explicitly pinned low and high reasoning efforts. Jev has no documented reasoning-effort setting. A call that fails or returns an invalid label counts as incorrect. Calls run sequentially in a seeded shuffled order, and wall-clock latency includes router retries. Token-based cost estimates are recorded separately from the conservative spend reservation.

Jev does not fit the benchmark repository's ordinary writing and reasoning prompts, which use ask_ai and require low/high variants. This targeted classification runner is kept in the router repository. It does not add fake Jev reasoning variants to benchmark/config.yaml or mix these fixture scores into the benchmark leaderboard.

The completed Luna baseline used one call per sample per effort, with a 4,096-token output cap including reasoning. All 74 calls completed without errors or truncation. No cached input tokens were reported. These are single-run measurements from this PC, not a matched Jev speed comparison.

| Variant | Routing correct | Completeness correct | Total correct | Median seconds | 95th-percentile seconds | Estimated cost per call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna low | 25/25 | 11/12 | 36/37, 97.3% | 0.776 | 1.972 | $0.000132 |
| Luna high | 25/25 | 12/12 | 37/37, 100% | 0.937 | 1.348 | $0.000153 |
| Jev | Pending | Pending | Pending | Pending | Pending | Pending |

Luna low rejected "refund please" as insufficient, although the existing application rule accepts it. High reasoning passed that case. With only 37 examples and one run, this does not establish a reliable quality difference between reasoning settings.

The [raw baseline](jev-luna-baseline.jsonl) records expected and returned labels, model identifiers, token usage, timing, cost and the fixture file hash. A separate [one-call pilot](jev-luna-pilot.jsonl) used a 1,024-token cap and is excluded from the accuracy and speed table. The baseline cost $0.0105296; the pilot cost $0.000136. Total estimated API spend so far is $0.0106656, calculated from reported tokens at current official prices. This is not an invoice reconciliation. Jev spend is $0.

## Integration and limits

The direct API uses POST https://api.typesafe.ai/v1/systemone with a bearer key. The current pinned model is jev-1.13.0. The router integration uses a separate classify call and retains the native answers, probabilities, confidence and usage. Keeping classification separate from chat follows the router's existing separate methods for embeddings and image generation. [API reference](https://docs.typesafe.ai/api), [model names](https://docs.typesafe.ai/models).

TypeSafe documents two input limits for Jev 1.13: 64,000 tokens across the state and all questions, and 32,000 across the state and the longest question. It warns about counting, arithmetic, date handling and adversarial instructions in the state. We do not treat confidence as proof of correctness. [Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

Vercel exposes Jev through its experimental AI SDK evaluation method. It does not support evaluation through its OpenAI-compatible chat endpoint. The initial Python integration therefore targets the documented TypeSafe HTTP API. [Vercel evaluation API](https://vercel.com/docs/ai-gateway/modalities/evaluation).

## Reproduction and outstanding work

Developer details follow. The fixture file is [jev-classification-samples.json](jev-classification-samples.json), and the runner is [benchmark_jev_classification.py](../scripts/benchmark_jev_classification.py). Running without --run prints the planned call count and spend reservation without calling a model.

```powershell
python scripts/benchmark_jev_classification.py
python scripts/benchmark_jev_classification.py --run --repeats 3 --budget 6 --output docs/jev-classification-matched.jsonl
```

The process must have OPENAI_API_KEY and TYPESAFE_API_KEY. Keep keys out of commits and messages. The runner imports this router checkout and does not require an upgrade to the shared Python installation or benchmark's pinned environment. It refuses to overwrite an existing result file.

Before adoption, complete an authenticated Jev smoke test and run both models in the same shuffled batch. Check disagreements, per-task accuracy, median and tail latency, reported model identifiers, token usage and costs. Test customer traffic separately with appropriate labels before changing any production routing.

Verification so far: all 37 fixture inputs and labels match the source tests, and an independent check ran all 12 completeness fixtures through the original deterministic logic. The full offline suite passed 992 tests plus 17 subtests. Independent code review found no high-severity issues and two smaller defects, both fixed. After adding the score-limit fix and three boundary cases, all 66 classification tests passed. The built 3.30.0 wheel then passed 289 focused classification, model, response and credential-error tests. Offline checks block network access and use synthetic credentials. They do not establish that TypeSafe accepts our credential or serves the pinned model to this account.

The wheel is dist/skell_e_router-3.30.0-py3-none-any.whl, SHA-256 d9ae154b1378bcb653f1e09b91d400da36ba9647065b8246dd61bbf39a88a87a. No global package installation or benchmark environment upgrade has occurred. Ordinary pytest startup on this PC encounters an unrelated installed Logfire plugin with a missing dependency; the existing offline test launcher disables plugin auto-loading, so no shared environment repair was needed.
