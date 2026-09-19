# Jev and "refund please"

Both failed calls used the same ticket and the same instructions. This was one distinct test case repeated three times. Jev chose the wrong label twice and the correct label once.

The task was to apply the app's minimum-content rules to a customer message. The expected label was sufficient: "refund please" has two words and 13 characters including the space, so it passes the specified minimum. It also has 12 letters without the space. This check does not establish that staff have everything needed to process a refund.

## What Jev received

The complete request body follows. The state field is a JSON-encoded string, as sent by the runner. Jane Doe and jane@example.com are existing test placeholders.

```json
{
  "model": "jev-1.13.0",
  "state": "{\"id\": \"1\", \"subject\": \"New Message\", \"customer\": \"Jane Doe\", \"customer_email\": \"jane@example.com\", \"messages\": [{\"body\": \"refund please\", \"type\": \"customer\"}]}",
  "questions": {
    "label": {
      "type": "choice",
      "instructions": "Classify whether a Proko support ticket has sufficient readable customer text according to its existing advisory content-quality check. State is a JSON ticket. Consider only messages whose type is customer, combining all customer messages in chronological order. Ignore the ticket subject, sender identity, agent replies and internal notes for this check. Strip HTML markup, quoted text and boilerplate; decode HTML entities. Remove empty contact-form field-label lines, URLs, email addresses, image placeholders and tracking-code tokens. Tracking codes are alphanumeric, underscore or hyphen tokens of at least 12 characters containing a digit. Collapse whitespace to single spaces and trim. Classify as insufficient if there are no customer messages, no remaining substantive text, exactly one remaining word, or fewer than 12 remaining substantive characters. Otherwise classify as sufficient. A short final reply can be sufficient when earlier customer messages provide substance. This is an advisory content check, not a spam verdict. Return the selected label only.",
      "criteria": {
        "sufficient": "Combined customer text passes the specified existing content-quality rules.",
        "insufficient": "Combined customer text triggers at least one specified existing content-quality rule."
      }
    }
  }
}
```

For readability, the ticket inside the state string is:

```json
{
  "id": "1",
  "subject": "New Message",
  "customer": "Jane Doe",
  "customer_email": "jane@example.com",
  "messages": [
    {
      "body": "refund please",
      "type": "customer"
    }
  ]
}
```

## What Jev chose

The following rows are in the order the calls occurred within the shuffled run. Repeat IDs in the raw results are zero-based and do not indicate call order.

| Call on this ticket | Choice | Sufficient probability | Insufficient probability | Reported confidence | Correct? |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | insufficient | 0.50 | 0.50 | 0.00 | No |
| 2 | sufficient | 0.51 | 0.49 | 0.02 | Yes |
| 3 | insufficient | 0.47 | 0.53 | 0.06 | No |

Both failures were uncertain decisions: a 50/50 tie and a 53/47 split. Confidence is a separate measure of how concentrated that distribution is, not another probability of correctness.

Jev supplies no written reasoning, so the evidence does not reveal why it missed the rule. These results show an error applying this exact policy, not an inability to understand refunds. The original app already checks this rule in code, which remains the right approach for exact word and character thresholds.

Both Luna settings chose sufficient on all three matched-run attempts. Luna low had missed the same case once in the earlier baseline, which is excluded from the matched comparison.

## Evidence

- [Input fixtures and source-test provenance](jev-classification-samples.json), sample content-02.
- [Matched raw results](jev-classification-matched.jsonl), variant jev and sample content-02. The failed repeat IDs are 1 and 2; repeat 0 is correct.
- [Benchmark runner](../scripts/benchmark_jev_classification.py), jev_call and question construct the request shown above.
- [Complete comparison](jev-classification.md).
