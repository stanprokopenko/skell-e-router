"""Matched spam-triage benchmark on real Proko support tickets.

Three arms over the same 120 sampled tickets:
  haiku_prod  — the exact production prompt and system message on claude-haiku-4-5
  luna_low    — the same prompt on gpt-5.6-luna, reasoning_effort="low"
  jev         — one typed classify() per ticket, scored under two fixed policies

Nothing is called without --run. Existing result files are never overwritten.

    python scripts/jev_real/spam_bench.py                 # plan and cost estimate
    python scripts/jev_real/spam_bench.py --run           # the real thing

The sample comes from scripts/temp/spam-sample.json (built by
scripts/temp/build_spam_sample.py) and the labels from
docs/jev-real/spam-labels.json, which the lead writes before any run.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, JsonlWriter, add_run_gate, gate, latency_summary, timed

from skell_e_router import ask_ai, classify

LABELS = ("SPAM", "CLOSE", "UNSURE", "NOT_SPAM")
ARMS = ("haiku_prod", "luna_low", "jev_raw", "jev_composed")

HAIKU_MODEL = "claude-haiku-4-5"
LUNA_MODEL = "gpt-5.6-luna"
JEV_MODEL = "jev-1.13.0"

# USD per million tokens. Haiku from the router documentation table; luna from
# the existing jev classification benchmark; jev bills input only.
PRICES = {
    HAIKU_MODEL: {"input": 1.00, "output": 5.00, "cached_input": 0.10},
    LUNA_MODEL: {"input": 0.20, "output": 1.20, "cached_input": 0.02},
    JEV_MODEL: {"input": 0.042, "output": 0.0},
}

# ---------------------------------------------------------------------------
# Production prompt — copied verbatim from skell-e-web
# backend/rag/spam_classifier.py (SPAM_PROMPT and the _call_llm system message).
# Any drift here invalidates the comparison, so keep it byte-for-byte.
# ---------------------------------------------------------------------------

SPAM_SYSTEM = (
    "You are a spam classifier. Respond with exactly one word: "
    "SPAM, CLOSE, UNSURE, or NOT_SPAM"
)

SPAM_PROMPT = """Classify this customer support ticket for Proko (an online art education company) as SPAM, CLOSE, LIKELY_SPAM, or NOT_SPAM.

SPAM — junk that should be deleted. Mark as SPAM if the ticket matches ANY of these:

1. Unsolicited sales and marketing pitches: Cold outreach offering marketing tools, software,
   SEO services, business partnerships, or SaaS products. Examples: HubSpot sales reps, PlanHub
   project invitations, Tapcart mobile commerce, TikTok Shop seller recruitment ("Join TikTok
   Shop Today"), recruitment platforms.

2. B2B solicitation and freelancer marketplace outreach: Emails from platforms like Upwork, Fiverr,
   Freelancer.com, or Toptal pitching freelancer hiring. Subject lines like "Ready to hire top
   Development & IT freelancers?" or "Find your next developer." Sender addresses from
   donotreply@upwork.com, noreply@fiverr.com, etc.

3. SEO / digital marketing service pitches: Cold emails offering SEO audits, link building,
   content marketing proposals, Google Ads management, social media management, or website redesign.
   Often from agencies or individual consultants.

4. Partnership and guest post requests: Unsolicited requests to publish guest posts on Proko's blog,
   exchange backlinks, or "collaborate on content." Often from unknown senders with no connection
   to art education.

5. Brand collaboration and sponsorship requests: Companies offering product exchanges, sponsorship,
   or collaboration (e.g., art supply brands like "Languo", tool companies wanting "creator
   partnerships", agencies like "Onspace.AI").

6. Creator outreach and channel scams: Messages claiming to have "selected" Proko's channel for
   a partnership program (e.g., "Beast Industries: We selected the @proko3d channel", "1000
   Creators" rosters). These are typically scam or mass-outreach messages.

7. Press kits, PR pitches, and product announcements: Unsolicited press releases, game/software
   launch announcements, trailer links, or journalist-style pitches. Common senders include PR
   agencies (ascotpr.com, otterpr.com) and game studios.

8. Contact form spam: Subject "New Message" where the actual content is promotional, contains
   external blog/site links being promoted, is gibberish, or is foreign-language advertising.

9. Foreign-language marketing and government notices: Automated government correspondence (Spanish
   tax authority "AVISO IMPORTANTE", IVA OSS notices), or foreign-language commercial emails.

10. Affiliate account scams: Elaborately worded "URGENT" requests claiming an affiliate account
    was compromised, asking to change payout details or recover affiliate accounts.

11. Miscellaneous non-customer junk: Political newsletters, event marketing (marathons, retreats),
    random mass emails not from a Proko customer asking a question.

CLOSE — legitimate notifications that should be archived but don't need a response. Mark as CLOSE if the ticket matches ANY of these:

1. DMARC/SPF/email authentication reports: Technical aggregate reports about email delivery.
   Look for "Report Domain: proko.com", DMARC aggregate reports from Microsoft, Google, Comcast,
   Yahoo, etc. These are legitimate email infrastructure reports.

2. Payment processor notifications: Stripe payout notifications ("Your $X payout for Proko is
   on the way"), dispute alerts/timeouts, PayPal billing agreement changes/cancellations,
   Shopify payout summaries. These are financial records — archive them.

3. SaaS/vendor operational notifications: Automated emails from services the company uses:
   - Mailchimp (daily list status, campaign results, subscriber reports)
   - Pinterest (recommendation emails "these belong in your world")
   - Splashtop (login alerts)
   - IPinfo (weekly request summaries)
   - AWS/Amazon (RDS notices, health events)
   - Vimeo (weekly review stats, W-9 confirmations)
   - ClickUp, Adobe Creative Cloud notifications
   - iDevAffiliate (publishing review notifications)

4. Email service reports: Mandrill account activity reports (weekly email stats), Mailchimp
   campaign performance summaries. Operational reports, not customer questions.

5. Financial report notifications: "Financial Reports are now available" and similar automated
   reports from accounting/payment systems.

6. Auto-replies to newsletters: Customers replying with "out of office" or auto-responses
   to Proko marketing emails. The auto-reply itself doesn't need a response.

7. Personal notifications forwarded to support: 23andMe, Marriott, U.S. Bank, YouTube Creator
   program notifications, Google Play developer notices, Steam platform announcements that
   ended up in the support inbox.

UNSURE — mark as UNSURE if you cannot confidently classify the ticket:

- The message feels like solicitation but uses ambiguous language
- The sender could be a real customer but the content looks promotional
- Foreign-language messages where you cannot determine intent with certainty
- You're torn between two categories

When unsure, it's better to let it through for the full pipeline to handle.

NOT_SPAM — mark as NOT_SPAM if the ticket matches ANY of these patterns:

1. Course access questions: Students asking about accessing purchased courses, downloading content,
   finding materials, or viewing lessons.

2. Account issues: Login problems, password resets, email changes, 2FA resets, account recovery
   from REAL customers who mention specific Proko courses or orders.

3. Billing and refunds: Refund requests (mentioning order numbers, specific products like pose packs,
   eBooks, or courses), payment issues, subscription questions, unauthorized charge inquiries.

4. Course content questions: Questions about course materials, lessons, reference photos, critique
   submissions, video buffering issues, or how to use Proko's educational content.

5. Contact form messages with real questions (CRITICAL): Subject line "New Message" with body
   containing actual questions about Proko courses, access, art education, refunds, or account
   issues. ~70% of "New Message" tickets are real customer questions.

6. Student feedback: Compliments, complaints, progress updates, feature requests, or portfolio
   sharing from students. Includes replies to Proko emails/newsletters from customers.

7. Forum/community moderation requests: Users disputing content moderation decisions on the Proko
   community forum or asking about hidden posts.

8. Artist inquiries: Professional artists reaching out about teaching opportunities, content
   contributions, or genuine collaboration proposals that mention specific Proko courses or content.

IMPORTANT EDGE CASES — READ CAREFULLY:
- "New Message" subject does NOT mean spam. Check the actual body content.
- A message mentioning Proko courses, orders, the classroom, specific course names (Anatomy,
  Drawing Basics, Perspective, Figure Drawing, Marvel), or proko.com is almost certainly NOT spam.
- Refund requests are NEVER spam, even if brief.
- Customer replies to Proko emails (newsletters like "Closing Thoughts") are NOT spam.
- When in doubt between CLOSE and NOT_SPAM, choose NOT_SPAM. Better to draft an unnecessary
  response than to miss a real customer.
- When in doubt between SPAM and UNSURE, choose UNSURE.

Respond with exactly one word: SPAM, CLOSE, UNSURE, or NOT_SPAM

Subject: {subject}
From: {customer}
Body:
{body}
"""

# ---------------------------------------------------------------------------
# Jev question set — the same categories and edge cases, expressed as types
# ---------------------------------------------------------------------------

COMPANY = (
    "Proko is an online art school. It sells video courses, a premium membership, and physical "
    "products (books, skeleton/anatomy models) to art students and hobbyists. This inbox is its "
    "customer support desk."
)

TRIAGE_QUESTION = {
    "type": "choice",
    "instructions": {
        "question": "Which triage action fits `ticket`?",
        "rules": [
            "A 'New Message' subject means nothing on its own; judge the body.",
            "Refund requests are never spam.",
            "A real customer mentioning a Proko product, course, or order is not spam.",
            "If torn between CLOSE and NOT_SPAM, pick NOT_SPAM.",
            "If torn between SPAM and UNSURE, pick UNSURE.",
        ],
    },
    "criteria": {
        "SPAM": {
            "what": "Junk to delete: solicitation or mass outreach, not a customer.",
            "examples": [
                "Unsolicited sales or marketing pitch for software, SaaS, or marketing tools",
                "B2B or freelancer marketplace outreach (Upwork, Fiverr, Toptal)",
                "SEO, link building, ads, or website redesign service pitch",
                "Guest post or backlink exchange request",
                "Brand collaboration or sponsorship offer",
                "Creator outreach or channel partnership scam",
                "Press kit, PR pitch, or product launch announcement",
                "Contact form spam: promotional, gibberish, or link-pushing body",
                "Foreign-language commercial email or automated government notice",
                "Affiliate account takeover or payout-change scam",
                "Other non-customer junk: political newsletters, event marketing, mass mail",
            ],
        },
        "CLOSE": {
            "what": "A legitimate automated notification to archive without a reply.",
            "examples": [
                "DMARC, SPF, or email authentication aggregate report",
                "Payment processor notification (Stripe payout, PayPal, Shopify)",
                "SaaS or vendor operational notice (Mailchimp, AWS, Vimeo, ClickUp, Adobe)",
                "Email service activity or campaign performance report",
                "Automated financial report availability notice",
                "Out-of-office or auto-reply to a Proko newsletter",
                "Personal platform notification forwarded into the support inbox",
            ],
        },
        "NOT_SPAM": {
            "what": "A real customer or prospect who needs the support pipeline.",
            "examples": [
                "Course access question from a student",
                "Account issue: login, password, email change, 2FA, recovery",
                "Billing or refund request, payment problem, unauthorized charge",
                "Question about course content, lessons, critiques, or video playback",
                "Contact form message containing a real question",
                "Student feedback: compliment, complaint, progress update, feature request",
                "Forum or community moderation dispute",
                "Artist inquiry about teaching or contributing that references Proko content",
            ],
        },
        "UNSURE": {
            "what": "Could be either a solicitation or a real customer; a human should look",
        },
    },
}

NOUL_QUESTIONS = {
    "real_customer": {
        "type": "noul",
        "instructions": (
            "The sender of `ticket` is a current or prospective Proko customer or student writing "
            "about their own purchase, account, course access, order, or art learning."
        ),
    },
    "solicitation": {
        "type": "noul",
        "instructions": (
            "The sender of `ticket` is offering, pitching, or requesting something for their own "
            "business: services, partnerships, collaborations, backlinks, guest posts, "
            "sponsorships, press, or products."
        ),
    },
    "automated": {
        "type": "noul",
        "instructions": (
            "`ticket` is an automated message: an auto-reply, out-of-office, system notification, "
            "receipt, newsletter, or forwarded platform notification rather than a person writing "
            "to support."
        ),
    },
    "money_issue": {
        "type": "noul",
        "instructions": (
            "`ticket` concerns a refund, charge, cancellation, billing problem, or a purchase that "
            "went wrong."
        ),
    },
}

JEV_QUESTIONS = {"triage": TRIAGE_QUESTION, **NOUL_QUESTIONS}


def compose_policy(nouls: dict) -> str:
    """Policy (b): thresholds over the nouls, ordered so customers win ties."""
    if nouls["money_issue"] >= 0.5 or nouls["real_customer"] >= 0.5:
        return "NOT_SPAM"
    if nouls["automated"] >= 0.6:
        return "CLOSE"
    if nouls["solicitation"] >= 0.8:
        return "SPAM"
    return "UNSURE"


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_prompt(ticket: dict) -> str:
    return SPAM_PROMPT.format(
        subject=ticket.get("subject", ""),
        customer=ticket.get("from", ""),
        body=ticket.get("body", "")[:2000],
    )


def parse_word(raw: str) -> str | None:
    """Pull one of the four labels out of a one-word answer.

    Tolerates case, surrounding punctuation, and a stray sentence. NOT_SPAM is
    checked first because 'SPAM' is a substring of it.
    """
    if not raw:
        return None
    text = raw.strip().upper()
    for label in ("NOT_SPAM", "NOT SPAM", "CLOSE", "UNSURE", "SPAM"):
        if label in text:
            return "NOT_SPAM" if label == "NOT SPAM" else label
    return None


def _usd(model: str, prompt_tokens, completion_tokens, cached: int = 0) -> float | None:
    price = PRICES[model]
    if prompt_tokens is None or completion_tokens is None:
        return None
    fresh = max(prompt_tokens - cached, 0)
    return (fresh * price["input"] + cached * price.get("cached_input", price["input"])
            + completion_tokens * price["output"]) / 1_000_000


def call_haiku(ticket: dict) -> dict:
    """Production settings: model, prompt, system message, nothing else set.

    rich_response only changes what the router hands back, not the request, so
    the call the provider sees is identical to _call_llm().
    """
    response = ask_ai(HAIKU_MODEL, build_prompt(ticket), SPAM_SYSTEM, rich_response=True)
    raw = (response.content or "").strip()
    return {
        "prediction": parse_word(raw),
        "raw": raw[:120],
        "provider_model": response.model,
        "input_tokens": response.prompt_tokens,
        "output_tokens": response.completion_tokens,
        "cost_usd": response.cost if response.cost is not None
        else _usd(HAIKU_MODEL, response.prompt_tokens, response.completion_tokens),
        "finish_reason": response.finish_reason,
    }


def call_luna(ticket: dict) -> dict:
    response = ask_ai(LUNA_MODEL, build_prompt(ticket), SPAM_SYSTEM,
                      reasoning_effort="low", max_tokens=4096, rich_response=True)
    raw = (response.content or "").strip()
    usage = getattr(response.raw_response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "prediction": parse_word(raw),
        "raw": raw[:120],
        "provider_model": response.model,
        "input_tokens": response.prompt_tokens,
        "output_tokens": response.completion_tokens,
        "cached_input_tokens": cached,
        "reasoning_tokens": response.reasoning_tokens,
        "cost_usd": _usd(LUNA_MODEL, response.prompt_tokens, response.completion_tokens, cached),
        "finish_reason": response.finish_reason,
    }


def call_jev(ticket: dict) -> dict:
    state = {
        "company": COMPANY,
        "ticket": {
            "subject": ticket.get("subject", ""),
            "from": ticket.get("from", ""),
            "body": ticket.get("body", "")[:2000],
        },
    }
    response = classify(JEV_MODEL, state, JEV_QUESTIONS, timeout=45)
    triage = response.answers["triage"]
    nouls = {key: response.answers[key]["noul"] for key in NOUL_QUESTIONS}
    return {
        "prediction": triage["choice"],
        "raw": triage["choice"],
        "provider_model": response.model,
        "probabilities": triage["probabilities"],
        "confidence": triage["confidence"],
        "nouls": nouls,
        "composed": compose_policy(nouls),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_arm(rows: list[dict], labels: dict) -> dict:
    """Accuracy plus the two errors that actually cost money.

    UNSURE predictions are never counted as a loss: they route to a human,
    which is the behaviour the production prompt asks for when in doubt.
    """
    scored = [r for r in rows if r["id"] in labels]
    graded = [r for r in scored if r.get("prediction")]
    correct = sum(1 for r in graded if r["prediction"] == labels[r["id"]]["label"])
    lost = [r["id"] for r in graded
            if labels[r["id"]]["label"] == "NOT_SPAM" and r["prediction"] in ("SPAM", "CLOSE")]
    through = [r["id"] for r in graded
               if labels[r["id"]]["label"] == "SPAM" and r["prediction"] == "NOT_SPAM"]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    matrix = defaultdict(Counter)
    for row in graded:
        matrix[labels[row["id"]]["label"]][row["prediction"]] += 1

    # Second, independent ground truth: what Teamwork itself filed as Spam.
    # Noisier than the human labels (a few of those tickets are real refund
    # requests a rule misfiled), so it is reported, never used for accuracy.
    tw_spam = [r for r in rows if r.get("teamwork_spam_status") and r.get("prediction")]
    tw_called_spam = sum(1 for r in tw_spam if r["prediction"] == "SPAM")
    return {
        "attempted": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "unparsed": sum(1 for r in rows if not r.get("error") and not r.get("prediction")),
        "labeled": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "real_customers_lost": len(lost),
        "real_customers_lost_ids": lost,
        "spam_let_through": len(through),
        "spam_let_through_ids": through,
        "unsure_predictions": sum(1 for r in graded if r["prediction"] == "UNSURE"),
        "confusion": {label: dict(counts) for label, counts in matrix.items()},
        "teamwork_spam_status_tickets": len(tw_spam),
        "teamwork_spam_status_called_spam": tw_called_spam,
        "teamwork_spam_status_agreement": (tw_called_spam / len(tw_spam)) if tw_spam else None,
        "teamwork_spam_status_called_not_spam": sum(
            1 for r in tw_spam if r["prediction"] == "NOT_SPAM"),
        "latency": latency_summary([r.get("elapsed_s") for r in rows]),
        "cost_usd_total": sum(costs),
        "cost_usd_per_1000_tickets": (sum(costs) / len(rows) * 1000) if rows and costs else None,
    }


def agreement(by_arm: dict[str, list[dict]]) -> dict:
    """Pairwise share of tickets where two arms returned the same label."""
    out = {}
    for left, right in itertools.combinations(sorted(by_arm), 2):
        lmap = {r["id"]: r.get("prediction") for r in by_arm[left]}
        rmap = {r["id"]: r.get("prediction") for r in by_arm[right]}
        shared = [i for i in lmap if i in rmap and lmap[i] and rmap[i]]
        same = sum(1 for i in shared if lmap[i] == rmap[i])
        # " vs " rather than a pipe: these keys become markdown table cells.
        out[f"{left} vs {right}"] = {
            "compared": len(shared),
            "same": same,
            "agreement": same / len(shared) if shared else None,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def estimate(tickets: list[dict]) -> dict:
    """Rough pre-run cost, from characters / 4 as a token proxy."""
    template_tokens = len(SPAM_PROMPT) / 4
    jev_overhead = len(json.dumps(JEV_QUESTIONS, ensure_ascii=False)) / 4 + len(COMPANY) / 4
    prompt_tokens = sum(template_tokens + len(t["subject"]) / 4 + len(t["body"]) / 4
                        for t in tickets)
    state_tokens = sum(jev_overhead + len(t["subject"]) / 4 + len(t["body"]) / 4 for t in tickets)
    haiku = (prompt_tokens * PRICES[HAIKU_MODEL]["input"] + 10 * len(tickets)
             * PRICES[HAIKU_MODEL]["output"]) / 1_000_000
    luna = (prompt_tokens * PRICES[LUNA_MODEL]["input"] + 600 * len(tickets)
            * PRICES[LUNA_MODEL]["output"]) / 1_000_000
    jev = state_tokens * PRICES[JEV_MODEL]["input"] / 1_000_000
    return {"haiku_prod_usd": round(haiku, 4), "luna_low_usd": round(luna, 4),
            "jev_usd": round(jev, 4), "total_usd": haiku + luna + jev}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "scripts/temp/spam-sample.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "docs/jev-real/spam-labels.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs/jev-real")
    add_run_gate(parser, default_budget=1.00)
    args = parser.parse_args()

    tickets = json.loads(args.sample.read_text(encoding="utf-8"))["tickets"]
    if args.limit:
        tickets = tickets[:args.limit]
    if not tickets:
        parser.error("No tickets selected.")

    results_path = args.out_dir / "spam-results.jsonl"
    summary_path = args.out_dir / "spam-summary.json"
    notes_path = args.out_dir / "spam-notes.md"
    disagree_path = args.out_dir / "spam-disagreements.md"
    existing = [p for p in (results_path, summary_path, notes_path, disagree_path) if p.exists()]
    if existing and args.run:
        parser.error("Refusing to overwrite: " + ", ".join(p.name for p in existing))

    costs = estimate(tickets)
    labels_ready = args.labels.exists()
    proceed = gate(args, planned_calls=len(tickets) * 3, estimated_usd=costs["total_usd"],
                   detail={"tickets": len(tickets), "arms": list(ARMS),
                           "per_arm_estimate_usd": costs, "labels_file": str(args.labels),
                           "labels_present": labels_ready,
                           "outputs": [str(p) for p in (results_path, summary_path,
                                                        notes_path, disagree_path)]})
    if not proceed:
        return
    if not labels_ready:
        parser.error(f"Label the sample first: {args.labels} is missing.")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    bad = {k: v for k, v in labels.items() if v.get("label") not in LABELS}
    if bad:
        parser.error(f"Labels must be one of {LABELS}; bad ids: {sorted(bad)[:5]}")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    spent = 0.0
    with JsonlWriter(results_path) as writer:
        for ticket in tickets:
            # No body text and no sender address ever reach the results file.
            base = {"id": ticket["id"], "source_file": ticket["source_file"],
                    "subject": ticket["subject"],
                    "teamwork_status": ticket.get("teamwork_status"),
                    "teamwork_spam_status": ticket.get("teamwork_spam_status", False),
                    "label": labels.get(ticket["id"], {}).get("label")}

            for arm, fn in (("haiku_prod", call_haiku), ("luna_low", call_luna)):
                row = {**base, "arm": arm, **timed(fn, ticket)}
                writer.write(row)
                by_arm[arm].append(row)
                spent += row.get("cost_usd") or 0.0

            jev_row = timed(call_jev, ticket)
            raw_row = {**base, "arm": "jev_raw", **jev_row}
            composed = {**base, "arm": "jev_composed", **jev_row,
                        "prediction": jev_row.get("composed"),
                        "raw": jev_row.get("composed"),
                        "cost_usd": None, "cost_shared_with": "jev_raw"}
            for row in (raw_row, composed):
                writer.write(row)
                by_arm[row["arm"]].append(row)
            spent += jev_row.get("cost_usd") or 0.0

            if spent > args.budget:
                print(f"Budget cap hit after {len(by_arm['haiku_prod'])} tickets "
                      f"(${spent:.4f}). Stopping.", flush=True)
                break

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": str(args.sample),
        "labels": str(args.labels),
        "tickets_attempted": len(by_arm["haiku_prod"]),
        "spend_usd": spent,
        "arms": {arm: score_arm(rows, labels) for arm, rows in sorted(by_arm.items())},
        "agreement": agreement(by_arm),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_disagreements(disagree_path, by_arm, labels)
    write_notes(notes_path, summary, results_path, disagree_path)
    print(json.dumps({"spend_usd": round(spent, 4),
                      "accuracy": {a: s["accuracy"] for a, s in summary["arms"].items()}},
                     indent=2))


def write_disagreements(path: Path, by_arm: dict[str, list[dict]], labels: dict) -> None:
    """Subject lines only. No body text, no sender address."""
    per_ticket: dict[str, dict] = {}
    for arm, rows in by_arm.items():
        for row in rows:
            entry = per_ticket.setdefault(row["id"], {"subject": row["subject"], "arms": {}})
            entry["arms"][arm] = row.get("prediction") or f"(error: {row.get('error')})"

    lines = ["# Spam triage disagreements", "",
             "Every ticket where at least one arm missed the human label. Subject lines only.", ""]
    misses = 0
    for ticket_id, entry in per_ticket.items():
        label = labels.get(ticket_id, {}).get("label")
        if not label or all(v == label for v in entry["arms"].values()):
            continue
        misses += 1
        lines.append(f"## {ticket_id} — {entry['subject']}")
        lines.append(f"Human label: {label}")
        for arm in ARMS:
            if arm in entry["arms"]:
                mark = "ok" if entry["arms"][arm] == label else "MISS"
                lines.append(f"- {arm}: {entry['arms'][arm]} ({mark})")
        note = labels.get(ticket_id, {}).get("note")
        if note:
            lines.append(f"- labeler note: {note}")
        lines.append("")
    lines.insert(3, f"{misses} of {len(per_ticket)} tickets had at least one miss.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_notes(path: Path, summary: dict, results_path: Path, disagree_path: Path) -> None:
    arms = summary["arms"]
    lines = [
        "# Spam triage benchmark — developer notes", "",
        f"Run at {summary['generated_at']}. Spend: ${summary['spend_usd']:.4f}.",
        f"Tickets: {summary['tickets_attempted']}. Raw rows: `{results_path.name}`. "
        f"Misses: `{disagree_path.name}`.", "",
        "## Results", "",
        "| arm | accuracy | real customers lost | spam let through | unsure | agrees with Teamwork spam | median s | p95 s | $/1k tickets |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def fmt(value, spec="{:.3f}"):
        return "n/a" if value is None else spec.format(value)

    for arm in ARMS:
        s = arms.get(arm)
        if not s:
            continue
        lines.append(
            f"| {arm} | {fmt(s['accuracy'])} | {s['real_customers_lost']} | "
            f"{s['spam_let_through']} | {s['unsure_predictions']} | "
            f"{fmt(s['teamwork_spam_status_agreement'])} "
            f"({s['teamwork_spam_status_called_spam']}/{s['teamwork_spam_status_tickets']}) | "
            f"{fmt(s['latency']['median_s'], '{:.2f}')} | {fmt(s['latency']['p95_s'], '{:.2f}')} | "
            f"{fmt(s['cost_usd_per_1000_tickets'], '${:.3f}')} |"
        )
    lines += [
        "", "Real customers lost = labeled NOT_SPAM, predicted SPAM or CLOSE. Spam let through = "
        "labeled SPAM, predicted NOT_SPAM. UNSURE routes to a human and never counts as a loss.",
        "", "The Teamwork column is a second, independent ground truth: the share of the 40 "
        "tickets Teamwork itself filed as Spam that the arm also called SPAM. It is noisy on "
        "purpose. A few of those tickets are real refund requests a Teamwork rule misfiled, so a "
        "score of 1.00 there is not the goal; the human labels remain the scoring truth.",
        "", "## Agreement between arms", "",
        "| pair | agreement |", "| --- | --- |",
    ]
    for pair, stats in summary["agreement"].items():
        lines.append(f"| {pair} | {fmt(stats['agreement'])} ({stats['same']}/{stats['compared']}) |")
    lines += [
        "", "## How to read this", "",
        "`haiku_prod` is production today: the exact prompt, system message, and model from "
        "skell-e-web `rag/spam_classifier.py`. `luna_low` swaps only the model. `jev_raw` and "
        "`jev_composed` come from one typed classify() call each: the first takes the model's own "
        "triage choice, the second applies fixed thresholds over the four nouls "
        "(money_issue or real_customer >= 0.5 wins NOT_SPAM, then automated >= 0.6 gives CLOSE, "
        "then solicitation >= 0.8 gives SPAM, else UNSURE).", "",
        "Jev bills input tokens only, so its per-1k figure is not comparable line-for-line with "
        "the generative arms; the wall-clock and error columns are.", "",
        "## Caveats", "",
        "- The sample is three pools of 40: tickets Teamwork filed as Spam, tickets the knowledge "
        "base excluded, and ordinary support tickets it kept.",
        "- The Teamwork dumps carry no sender name or address, only a numeric customerID, so "
        "`From:` is empty for every ticket in all four arms. Production sees a real sender, which "
        "is a strong spam signal, so absolute accuracy here understates all three models.",
        "- Production runs Firestore pattern rules before the LLM. This benchmark skips them and "
        "measures the LLM stage alone.",
        "- Message bodies come from the dumps' HTML, flattened the way production flattens a "
        "message with no textBody. Line breaks inside marketing HTML are lost, which is what "
        "production would also see on that path.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
