# Spam triage benchmark — developer notes

Run at 2026-09-19T19:32:17.016508+00:00. Spend: $0.2650.
Tickets: 120. Raw rows: `spam-results.jsonl`. Misses: `spam-disagreements.md`.

## Results

| arm | accuracy | real customers lost | spam let through | unsure | agrees with Teamwork spam | median s | p95 s | $/1k tickets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haiku_prod | 0.842 | 0 | 2 | 8 | 0.525 (21/40) | 0.53 | 1.55 | $2.097 |
| luna_low | 0.892 | 0 | 0 | 4 | 0.675 (27/40) | 0.59 | 1.14 | $0.056 |
| jev_raw | 0.808 | 0 | 0 | 0 | 0.400 (16/40) | 0.21 | 0.28 | $0.055 |
| jev_composed | 0.692 | 0 | 3 | 5 | 0.175 (7/40) | 0.21 | 0.28 | n/a |

Real customers lost = labeled NOT_SPAM, predicted SPAM or CLOSE. Spam let through = labeled SPAM, predicted NOT_SPAM. UNSURE routes to a human and never counts as a loss.

The Teamwork column is a second, independent ground truth: the share of the 40 tickets Teamwork itself filed as Spam that the arm also called SPAM. It is noisy on purpose. A few of those tickets are real refund requests a Teamwork rule misfiled, so a score of 1.00 there is not the goal; the human labels remain the scoring truth.

## Agreement between arms

| pair | agreement |
| --- | --- |
| haiku_prod vs jev_composed | 0.742 (89/120) |
| haiku_prod vs jev_raw | 0.858 (103/120) |
| haiku_prod vs luna_low | 0.933 (112/120) |
| jev_composed vs jev_raw | 0.833 (100/120) |
| jev_composed vs luna_low | 0.725 (87/120) |
| jev_raw vs luna_low | 0.858 (103/120) |

## How to read this

`haiku_prod` is production today: the exact prompt, system message, and model from skell-e-web `rag/spam_classifier.py`. `luna_low` swaps only the model. `jev_raw` and `jev_composed` come from one typed classify() call each: the first takes the model's own triage choice, the second applies fixed thresholds over the four nouls (money_issue or real_customer >= 0.5 wins NOT_SPAM, then automated >= 0.6 gives CLOSE, then solicitation >= 0.8 gives SPAM, else UNSURE).

Jev bills input tokens only, so its per-1k figure is not comparable line-for-line with the generative arms; the wall-clock and error columns are.

## Aborted first run

A first `--run` in a shell without the Machine-scope TYPESAFE_API_KEY and ANTHROPIC_API_KEY made 120 successful Luna calls ($0.0443) while the Haiku and Jev arms failed with MISSING_ENV. Its files were deleted and the run repeated with the keys hydrated. Task spend for this benchmark is therefore $0.2650 + $0.0443 = $0.3093.

## Caveats

- The sample is three pools of 40: tickets Teamwork filed as Spam, tickets the knowledge base excluded, and ordinary support tickets it kept.
- The Teamwork dumps carry no sender name or address, only a numeric customerID, so `From:` is empty for every ticket in all four arms. Production sees a real sender, which is a strong spam signal, so absolute accuracy here understates all three models.
- Production runs Firestore pattern rules before the LLM. This benchmark skips them and measures the LLM stage alone.
- Message bodies come from the dumps' HTML, flattened the way production flattens a message with no textBody. Line breaks inside marketing HTML are lost, which is what production would also see on that path.