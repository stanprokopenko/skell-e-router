"""Read the routing benchmark results and write flat CSVs for the report builder.

Inputs (single source of truth):
  ../routing-results.jsonl   1 metadata header line + 1695 result rows
  ../routing-summary.json    precomputed aggregates

Everything the report shows is computed here from the jsonl where the jsonl can
answer it, and read from the summary only where the jsonl cannot (the rules
engine's production-shaped scoring, the fitted threshold sweep, the hybrid).
Numbers that exist in both places are recomputed and asserted to match.

Usage:  python extract.py
"""

import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
RESULTS = os.path.join(SRC, "routing-results.jsonl")
SUMMARY = os.path.join(SRC, "routing-summary.json")
DATA = os.path.join(HERE, "data")

ARMS = ["gemini_prod", "luna_low", "jev"]
# Message types with fewer than this many cases are folded into "other".
# generate_report.py reads the same number back out of data/meta.csv.
MIN_CATEGORY_N = 10
ARM_LABEL = {
    "gemini_prod": "Today's classifier",
    "luna_low": "gpt-5.6-luna",
    "jev": "Jev 1.13.0",
}
ARM_MODEL = {
    "gemini_prod": "gemini-3.5-flash-lite",
    "luna_low": "gpt-5.6-luna",
    "jev": "jev-1.13.0",
}
ARM_NOTE = {
    "gemini_prod": "gemini-3.5-flash-lite, running in production today",
    "luna_low": "same prompt, low reasoning effort",
    "jev": "jev-1.13.0, built only for picking between fixed choices",
}


def load():
    with open(RESULTS, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    meta = json.loads(lines[0])
    assert meta.get("metadata") is True, "first line is not the metadata header"
    rows = [json.loads(ln) for ln in lines[1:]]
    with open(SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    return meta, rows, summary


def percentile(values, q):
    """The index method scripts/jev_real/common.py:latency_summary uses."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def write_csv(name, fieldnames, records):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print("wrote %-28s %4d rows" % (name, len(records)))


def main():
    os.makedirs(DATA, exist_ok=True)
    meta, rows, summary = load()

    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    for arm in ARMS:
        assert len(by_arm[arm]) == meta["cases"], arm
    # case key -> per-arm row
    def key(r):
        return (r["conv"], r["idx"])

    cases = {}
    for arm in ARMS:
        for r in by_arm[arm]:
            cases.setdefault(key(r), {})[arm] = r
    assert len(cases) == meta["cases"]

    checks = []

    def check(name, got, want, tol=1e-6):
        ok = abs(got - want) <= tol if isinstance(got, float) else got == want
        checks.append({"check": name, "recomputed": got, "summary": want, "match": "yes" if ok else "NO"})
        assert ok, "%s: recomputed %r vs summary %r" % (name, got, want)

    # ---- headline, recomputed from the jsonl ------------------------------
    headline = []
    for arm in ARMS:
        rs = by_arm[arm]
        correct = sum(1 for r in rs if r["prediction"] == r["label"])
        missed = sum(1 for r in rs if r["label"] == "big" and r["prediction"] == "fast")
        false = sum(1 for r in rs if r["label"] == "fast" and r["prediction"] == "big")
        lat = [r["elapsed_s"] for r in rs]
        med = statistics.median(lat)
        p95 = percentile(lat, 0.95)
        cost_1k = sum(r["cost_usd"] for r in rs) / len(rs) * 1000.0

        s = summary["arms"][arm]["model_only"]["strict_errors_wrong"]
        check("%s correct" % arm, correct, s["correct"])
        check("%s missed_big" % arm, missed, s["missed_big"])
        check("%s false_big" % arm, false, s["false_big"])
        check("%s median_s" % arm, med, summary["arms"][arm]["latency"]["median_s"])
        check("%s p95_s" % arm, p95, summary["arms"][arm]["latency"]["p95_s"])
        check("%s cost_per_1000" % arm, cost_1k, summary["arms"][arm]["cost"]["cost_per_1000_usd"])

        headline.append({
            "arm": arm,
            "model_label": ARM_LABEL[arm],
            "model_id": ARM_MODEL[arm],
            "description": ARM_NOTE[arm],
            "n": len(rs),
            "correct": correct,
            "accuracy_pct": round(100.0 * correct / len(rs), 1),
            "missed_big": missed,
            "false_big": false,
            "median_s": round(med, 3),
            "p95_s": round(p95, 3),
            "mean_s": round(statistics.fmean(lat), 3),
            "cost_per_1000_usd": round(cost_1k, 4),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rs), 4),
            "errors": sum(1 for r in rs if r["error"]),
        })
    write_csv("headline.csv", list(headline[0].keys()), headline)

    # ---- production-shaped (rules first) ---------------------------------
    prod = []
    for arm in ARMS:
        p = summary["arms"][arm]["production_shaped"]
        prod.append({
            "arm": arm,
            "model_label": ARM_LABEL[arm],
            "n": p["n"],
            "correct": p["correct"],
            "accuracy_pct": round(100.0 * p["accuracy"], 1),
            "missed_big": p["missed_big"],
            "false_big": p["false_big"],
        })
    write_csv("prod_shaped.csv", list(prod[0].keys()), prod)

    # ---- Jev accuracy by confidence quartile -----------------------------
    q = summary["jev_by_confidence_quartile"]
    qlabel = {
        "q1_lowest": "Least sure quarter",
        "q2": "Second quarter",
        "q3": "Third quarter",
        "q4_highest": "Most sure quarter",
    }
    quart = []
    for k in ["q1_lowest", "q2", "q3", "q4_highest"]:
        v = q[k]
        quart.append({
            "quartile": k,
            "quartile_label": qlabel[k],
            "n": v["n"],
            "correct": v["correct"],
            "accuracy_pct": round(100.0 * v["accuracy"], 1),
            "missed_big": v["missed_big"],
            "false_big": v["false_big"],
        })
    # recompute the quartile split from the jsonl using the same cuts
    cuts = q["cuts"]
    jev_rows = sorted(by_arm["jev"], key=lambda r: r["confidence"])
    buckets = {k: [] for k in qlabel}
    for r in jev_rows:
        c = r["confidence"]
        if c <= cuts[0]:
            buckets["q1_lowest"].append(r)
        elif c <= cuts[1]:
            buckets["q2"].append(r)
        elif c <= cuts[2]:
            buckets["q3"].append(r)
        else:
            buckets["q4_highest"].append(r)
    for k in qlabel:
        check("jev quartile %s n" % k, len(buckets[k]), q[k]["n"])
        check("jev quartile %s correct" % k,
              sum(1 for r in buckets[k] if r["prediction"] == r["label"]), q[k]["correct"])
    write_csv("jev_quartiles.csv", list(quart[0].keys()), quart)
    write_csv("jev_quartile_cuts.csv", ["cut", "cut_pct"],
              [{"cut": c, "cut_pct": round(100.0 * c)} for c in cuts])

    # ---- accuracy by message category ------------------------------------
    cat_n = {}
    for r in by_arm["jev"]:
        cat_n[r["category"]] = cat_n.get(r["category"], 0) + 1
    rare = {c for c, n in cat_n.items() if n < MIN_CATEGORY_N}

    def catname(c):
        return "other" if c in rare else c

    grouped = {}
    for k, per_arm in cases.items():
        c = catname(per_arm["jev"]["category"])
        g = grouped.setdefault(c, {"n": 0, **{a: 0 for a in ARMS}})
        g["n"] += 1
        for arm in ARMS:
            if per_arm[arm]["prediction"] == per_arm[arm]["label"]:
                g[arm] += 1
    assert sum(g["n"] for g in grouped.values()) == meta["cases"]
    cat_rows = []
    for c in sorted(grouped, key=lambda c: (c == "other", -grouped[c]["n"])):
        g = grouped[c]
        rec = {"category": c.replace("-", " "), "n": g["n"]}
        for arm in ARMS:
            rec["%s_correct" % arm] = g[arm]
            rec["%s_accuracy_pct" % arm] = round(100.0 * g[arm] / g["n"], 1)
        cat_rows.append(rec)
    write_csv("category_accuracy.csv", list(cat_rows[0].keys()), cat_rows)
    write_csv("category_grouping.csv", ["grouped_into_other", "n"],
              [{"grouped_into_other": c, "n": cat_n[c]} for c in sorted(rare)])

    # ---- Jev policy variants and the hybrid ------------------------------
    pol_label = {
        "a_raw_choice": ("Jev's own answer", "Use the tier Jev picks. Nothing else."),
        "b_composed": ("Built from the yes/no answers", "Ignore Jev's tier and rebuild the decision from its four yes/no scores at a 0.5 cutoff."),
        "c_choice_with_tiebreak": ("Jev's answer, big when unsure", "Use Jev's tier, but send it to the expensive model whenever Jev is under 30% sure, unless the writer asked for something quick."),
    }
    pol = []
    for k in ["a_raw_choice", "b_composed", "c_choice_with_tiebreak"]:
        v = summary["jev_policies"][k]
        pol.append({
            "policy": k,
            "policy_label": pol_label[k][0],
            "explanation": pol_label[k][1],
            "n": v["n"],
            "correct": v["correct"],
            "accuracy_pct": round(100.0 * v["accuracy"], 1),
            "missed_big": v["missed_big"],
            "false_big": v["false_big"],
            "kind": "policy",
        })
    sw = summary["jev_threshold_sweep"]
    pol.append({
        "policy": "d_prob_threshold",
        "policy_label": "Tuned probability cutoff",
        "explanation": "Ignore the tier and send to the expensive model when Jev puts the chance of big at %d%% or higher. Tuned on half the cases and scored on the other half, so this row is the held-out result." % round(sw["best_spec"]["threshold"] * 100),
        "n": sw["held_out_even"]["n"],
        "correct": sw["held_out_even"]["correct"],
        "accuracy_pct": round(100.0 * sw["held_out_even"]["accuracy"], 1),
        "missed_big": sw["held_out_even"]["missed_big"],
        "false_big": sw["held_out_even"]["false_big"],
        "kind": "policy",
    })
    write_csv("jev_policies.csv", list(pol[0].keys()), pol)

    hyb = []
    for k in sorted(summary["hybrid_jev_then_luna"]):
        v = summary["hybrid_jev_then_luna"][k]
        cutoff = k.split("=")[1]
        # Every case costs a Jev call; the escalated ones cost a Luna call on top.
        escalated_cases = [c for c in cases.values() if c["jev"]["confidence"] < float(cutoff)]
        check("hybrid t=%s escalated" % cutoff, len(escalated_cases), v["escalated"])
        blended = (sum(c["jev"]["cost_usd"] for c in cases.values())
                   + sum(c["luna_low"]["cost_usd"] for c in escalated_cases)) / len(cases) * 1000.0
        hyb.append({
            "cutoff": cutoff,
            "policy_label": "Jev first, ask Luna when under %d%% sure" % round(float(cutoff) * 100),
            "explanation": "Jev answers. When it is less than %d%% sure, the message is sent to gpt-5.6-luna instead. That happened on %.1f%% of messages." % (round(float(cutoff) * 100), 100.0 * v["escalated_share"]),
            "escalated": v["escalated"],
            "escalated_pct": round(100.0 * v["escalated_share"], 1),
            "n": v["n"],
            "correct": v["correct"],
            "accuracy_pct": round(100.0 * v["accuracy"], 1),
            "missed_big": v["missed_big"],
            "false_big": v["false_big"],
            "blended_cost_per_1000_usd": round(blended, 4),
            "summary_estimate_per_1000_usd": round(v["blended_cost_per_1000_usd"], 4),
        })
    write_csv("hybrid.csv", list(hyb[0].keys()), hyb)

    # ---- agreement -------------------------------------------------------
    agree = []
    pairs = [("gemini_prod", "luna_low"), ("gemini_prod", "jev"), ("luna_low", "jev")]
    for a, b in pairs:
        k = "%s_vs_%s" % (a, b)
        v = summary["agreement"][k]
        same = sum(1 for c in cases.values() if c[a]["prediction"] == c[b]["prediction"])
        check("agreement %s" % k, same, v["agree"])
        agree.append({
            "pair": "%s vs %s" % (ARM_LABEL[a], ARM_LABEL[b]),
            "n": v["n"],
            "agree": v["agree"],
            "agree_pct": round(100.0 * v["agree_rate"], 1),
            "both_big": v["matrix"]["big|big"],
            "both_fast": v["matrix"]["fast|fast"],
            "disagree": v["n"] - v["agree"],
        })
    write_csv("agreement.csv", list(agree[0].keys()), agree)

    # ---- evidence case tables -------------------------------------------
    def rec(per_arm):
        j, l, g = per_arm["jev"], per_arm["luna_low"], per_arm["gemini_prod"]
        return {
            "text_head": j["text_head"],
            "label": j["label"],
            "jev_prediction": j["prediction"],
            "jev_confidence": round(j["confidence"], 2),
            "jev_prob_big": round(j["probabilities"]["big"], 2),
            "luna_prediction": l["prediction"],
            "gemini_prediction": g["prediction"],
            "category": j["category"].replace("-", " "),
            "sender": j["sender"],
        }

    def ok(per_arm, arm):
        return per_arm[arm]["prediction"] == per_arm[arm]["label"]

    ordered_cases = sorted(cases.values(), key=lambda c: (c["jev"]["confidence"], c["jev"]["text_head"]))
    sets = {
        "cases_jev_wrong.csv": [c for c in ordered_cases if not ok(c, "jev")],
        "cases_jev_right_luna_wrong.csv": [c for c in ordered_cases if ok(c, "jev") and not ok(c, "luna_low")],
        "cases_jev_right_gemini_wrong.csv": [c for c in ordered_cases if ok(c, "jev") and not ok(c, "gemini_prod")],
        "cases_all_wrong.csv": [c for c in ordered_cases if not ok(c, "jev") and not ok(c, "luna_low") and not ok(c, "gemini_prod")],
    }
    for name, sel in sets.items():
        recs = [rec(c) for c in sel]
        write_csv(name, list(rec(ordered_cases[0]).keys()), recs)
    check("jev errors", len(sets["cases_jev_wrong.csv"]), 565 - summary["arms"]["jev"]["model_only"]["strict_errors_wrong"]["correct"])
    check("jev right, luna wrong", len(sets["cases_jev_right_luna_wrong.csv"]), 47)
    check("luna right, jev wrong",
          sum(1 for c in cases.values() if ok(c, "luna_low") and not ok(c, "jev")), 19)

    # ---- meta ------------------------------------------------------------
    metarec = [{
        "cases": summary["cases"],
        "category_min_n": MIN_CATEGORY_N,
        "label_big": summary["label_big"],
        "label_fast": summary["label_fast"],
        "rules_settled": summary["rules_settled"],
        "rules_explicit_model": summary["rules_explicit_model"],
        "rules_unsettled": summary["rules_unsettled"],
        "prod_shaped_n": summary["arms"]["jev"]["production_shaped"]["n"],
        "run_date": meta["started_utc"][:10],
        "generated_utc": summary["generated_utc"],
        "spend_usd": summary["spend_usd"],
        "total_calls": len(rows),
        "jev_right_luna_wrong": len(sets["cases_jev_right_luna_wrong.csv"]),
        "luna_right_jev_wrong": sum(1 for c in cases.values() if ok(c, "luna_low") and not ok(c, "jev")),
        "jev_errors_bottom_two_quartiles": sum(
            1 for r in buckets["q1_lowest"] + buckets["q2"] if r["prediction"] != r["label"]),
    }]
    write_csv("meta.csv", list(metarec[0].keys()), metarec)

    write_csv("verification.csv", ["check", "recomputed", "summary", "match"], checks)
    print("\n%d cross-checks against routing-summary.json, all match." % len(checks))


if __name__ == "__main__":
    main()
