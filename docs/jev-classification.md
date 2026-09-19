# Jev for classification

This note is for Stan. Developer notes with every number live in docs/jev-real/. Research began September 17; the real-task comparison ran September 19, 2026.

## Bottom line

Jev is a real option for two of our three AI steps and not for the third. On the chat routing decision in skell-e-web (which model answers a staff message), Jev beat both gpt-5.6-luna and the model we run today, at a fifth of the cost and a third of the latency. On support ticket spam triage it is roughly as good as today's model once you gate it on its own confidence, and it is the fastest and cheapest of the three. On the rough cut sentence rating in solar-sailer (which transcript sentences survive the edit), Luna is still better and Jev cuts too much.

The pattern across all three: Jev's confidence number is honest. When it says it is sure, it is right. That is what makes it usable as a first pass with a fallback, and it is the thing our generative prompts never gave us.

Total spend for this session was $0.74, and $0.78 for the whole task against the $20 allowance.

## Task 1: chat routing (skell-e-web)

What it is. Every message to the internal Proko assistant gets sent to a cheap model or an expensive one. Simple rules catch the obvious cases and an AI classifier decides the rest. Getting it wrong costs money one way (expensive model on a data pull) and quality the other (cheap model on a pricing decision). The test set is 565 real staff messages with hand-written labels, already in the skell-e-web repo, with the surrounding conversation joined in.

How Jev was set up. Instead of the production prompt, Jev got the conversation as structured fields (setting, earlier turns, the new message, the sender's history) and one "which model" choice question whose two options carry the prompt's own definitions and examples. Four extra yes/no questions rode along in the same request (does the user want it quick, is there money at stake, is it customer-facing writing, is it a simple pull) so code could combine them. Luna and today's production model got the production prompt unchanged.

| Model | Correct of 565 | Expensive model missed | Cheap model wrongly skipped | Median latency | Cost per 1,000 calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Today's production model (gemini-3.5-flash-lite) | 491 (86.9%) | 21 | 53 | 0.61 s | $0.30 |
| gpt-5.6-luna, low reasoning | 494 (87.4%) | 13 | 58 | 1.09 s | $0.20 |
| Jev 1.13.0 | 522 (92.4%) | 23 | 20 | 0.21 s | $0.06 |

Jev's win comes almost entirely from not over-escalating. Luna and the production model send about 55 cheap messages to the expensive model; Jev sends 20. Jev misses a few more big ones, mostly short marketing asks that look like data pulls ("generate titles for that episode").

Two things make this more than a one-off number. First, Jev's confidence means something: it was perfect on its most confident quarter of messages and 40 of its 43 mistakes sat in the least confident half. Second, the cheap tricks work. A rule that says "when Jev's confidence is under 0.3, choose big unless the user asked for something quick" drops the missed-big count to 13, matching Luna, while keeping false-bigs at 32. Fitting a probability threshold on half the data and testing on the other half gave 93.3% with only 3 missed-big. Combining the yes/no side questions with hand-picked cutoffs did not work (76%), so the choice question is the part that carries the result.

What I did not do: change production. The numbers use the labels' own history for context, not the model's earlier decisions, so a live rollout will compound its own mistakes in a way this test does not. A limited live trial with the current classifier as a fallback would settle it.

## Task 2: rough cut sentence rating (solar-sailer)

What it is. The video editor rates every transcript sentence 0 to 5 for whether it earns a place in the final edit, and the app cuts from those scores. Ground truth is the human editor's actual cut for 19 real lessons. I used five mid-sized episodes, 1,597 sentences in total.

How Jev was set up. The production prompt asks for one big pass over the whole transcript. Jev cannot write, so each sentence became its own "score" question with the production rubric's six levels as the answer set, plus a yes/no question "is this a repeated take that should go". Questions were batched 25 per request with the whole transcript as shared context. A second variant gave Jev only a window of 30 surrounding sentences, in case the long transcript was hurting it. Luna ran the production prompt as it does today.

| Model | Editor agreement (WORD SCORE) | Keep/cut accuracy | Sentences cut that the editor kept | Time per episode | Cost for five episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-5.6-luna, medium reasoning | 0.770 | 85.1% | 200 | 21 s | $0.03 |
| Jev, whole transcript | 0.759 | 79.5% | 297 | 2 s | $0.05 |
| Jev, 30-sentence window | 0.713 | 76.5% | 315 | 2 s | $0.03 |

Luna wins, and not by a little. Jev is right about what to keep when it keeps (96% precision, same as Luna) but it cuts about half again as many sentences the editor wanted. Giving it less context made it worse, so this is not a long-context problem. The judgment itself is the issue: deciding whether a rambling sentence stays because it is funny, or a false start stays for continuity, is exactly where both models struggle and where Luna's full read-through helps. Jev did beat Luna on the two smallest episodes, so on short clips it is competitive.

Jev is ten times faster per episode and its confidence again tracked its accuracy, so a hybrid (Jev first, Luna on the sentences it is unsure about) is conceivable. I would not build it: the editor already runs one Luna call per episode, and the cost is 3 cents. Jev is slightly more expensive here because every batch re-sends the transcript.

One bug fell out of this. The router rejected about a third of Jev's score answers as malformed because it checked that the probabilities add up to exactly 1, and Jev rounds them to two decimals. The check now allows for that rounding, tests pass, and the fix is committed in this repo.

## Task 3: support ticket spam triage (skell-e-web)

What it is. Every incoming support ticket gets sorted into spam (delete), close (archive a notification), not spam (a real customer, draft a reply) or unsure (a human looks). A miss in the wrong direction loses a paying customer. There were no labels for this one, so I sampled 120 real tickets (40 that Teamwork itself filed as spam, 40 that the knowledge base filter had excluded, 40 ordinary kept tickets) and labeled them myself before running any model. Those labels are mine, not human-verified, and the five I marked unsure are judgment calls. The results file says so.

How Jev was set up. Same shape as routing: the ticket as structured fields, one four-way choice question with the production prompt's category lists as examples, and four yes/no side questions (real customer, solicitation, automated mail, money issue). Today's production model (claude-haiku-4-5) and Luna got the production prompt unchanged.

| Model | Correct of 120 | Real customers lost | Spam let through | Sent to a human | Median latency | Cost per 1,000 tickets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Today's production model (claude-haiku-4-5) | 101 (84%) | 0 | 2 | 8 | 0.53 s | $2.10 |
| gpt-5.6-luna, low reasoning | 107 (89%) | 0 | 0 | 4 | 0.59 s | $0.06 |
| Jev 1.13.0, its own choice | 97 (81%) | 0 | 0 | 0 | 0.21 s | $0.06 |

Nobody lost a customer. Jev's raw score looks worst, but 15 of its 23 misses are one pattern: bot form submissions (a fake name, an unrelated email, one random Latin word) that it filed as "close" instead of "spam". Both labels remove the ticket without a reply, so operationally that is not an error. Counting spam and close as the same outcome, Jev gets 112 of 120, Luna 107 and Haiku 101.

The bigger difference is that Jev never says "unsure". It always picks. That would be a problem, except its confidence does the job instead: on the 90 tickets where it was at least 91% sure it was right on 87, and on the 30 where it was less sure it was right on 10. Routing anything under 0.9 confidence to a human sends 29 tickets to a person and leaves 3 mistakes in the other 91, two of which are tickets I had marked unsure myself. That threshold was picked after seeing the results, so treat it as the shape of a policy, not a measured one.

Combining the yes/no side questions with hand-set cutoffs did worse again (83 of 120), same as in routing. The choice question is the part that works. If we ever adopt Jev here, the fix for the bot-form pattern is to add it as an example under spam, which is one line.

One caveat that applies to all three models: the Teamwork export carries no sender address, so nobody saw the "From" line production sees. That is a strong spam signal, so all three would do better live.

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

## Earlier fixture comparison (September 19, historical)

This round compared the models on fixtures that already had deterministic rules. It is kept as evidence of the integration working, not as a task comparison. The results above supersede it.

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
| September 17 to 19 fixture round (all calls) | $0.046 |
| Chat routing, 1,695 calls across three models | $0.320 |
| Rough cut, 239 requests across three arms | $0.107 |
| Spam triage, including one aborted run that only reached Luna | $0.309 |
| Total task spend | $0.783 |

About 78 cents against the $20 cap. These estimates use provider-reported token counts and official prices. They are not an invoice reconciliation.

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

Developer continuation notes. The real-task comparison is done and documented above; the earlier fixture round is historical. Runners live in scripts/jev_real/ (routing_bench.py, roughcut_bench.py, spam_bench.py, common.py) and every result, summary and developer note lives in docs/jev-real/. Each runner prints a plan and cost estimate without --run and refuses to overwrite existing result files. The API keys are Machine-scope on this PC and missing from a fresh shell; the hydration recipe is in docs/jev-real/routing-notes.md. Never print or commit key values.

Customer text stays out of git: the chat export and the ticket sample live under scripts/temp/ (gitignored) and can be rebuilt with scripts/temp/build_spam_sample.py and the skell-e-web export. Result files carry ids, subjects and the committed 120-character text heads only.

Open follow-ups, none of them started: a limited live trial of Jev on chat routing with the current classifier as fallback; adding the bot-form pattern to the spam criteria and re-running the 120; and deciding whether the spam labels should get a human pass. No production routing changed in this task.
