"""Score the Jev image runs against the blind labels and write results JSON.

Inputs: manifest, labels, the Jev drawing JSONL, the Gemini drawing JSONL and
the import JSONL files (384-wide production replay, 512-wide replay). Output:
one JSON file with every number the write-up quotes, plus a printed summary.
Keeps cant-tell frames out of pencil accuracy and reports them separately.
Image pricing is unconfirmed; the "text_rate" cost numbers apply TypeSafe's
published $0.042 per million input tokens and are a bound, not a verified price.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TEXT_RATE_PER_TOKEN = 0.042 / 1_000_000
STAGE_NAMES = ["blank", "first_marks", "in_progress", "finished"]
LENIENT = {"sketchbook_C006": {"WRK", "SBJ"}, "ddr_world_record": {"WRK", "SCR"}}


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def band(p: float) -> str:
    """Confidence band on a probability of yes: how far from 0.5 it sits."""
    strength = max(p, 1 - p)
    return "high (>=0.8)" if strength >= 0.8 else "mid (0.6-0.8)" if strength >= 0.6 else "low (<0.6)"


def choice_band(confidence: float) -> str:
    return "high (>=0.8)" if confidence >= 0.8 else "mid (0.5-0.8)" if confidence >= 0.5 else "low (<0.5)"


def binary_scores(pairs):
    """pairs: list of (truth bool, probability of yes). Accuracy, balanced accuracy, confusion, bands."""
    confusion = Counter()
    bands = defaultdict(lambda: [0, 0])
    for truth, p in pairs:
        pred = p >= 0.5
        confusion[("true" if truth else "false", "yes" if pred else "no")] += 1
        bands[band(p)][0] += int(pred == truth)
        bands[band(p)][1] += 1
    tp, fn = confusion[("true", "yes")], confusion[("true", "no")]
    tn, fp = confusion[("false", "no")], confusion[("false", "yes")]
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"n": len(pairs), "accuracy": round((tp + tn) / len(pairs), 3),
            "recall_true": round(recall, 3) if recall is not None else None,
            "recall_false": round(specificity, 3) if specificity is not None else None,
            "balanced_accuracy": round((recall + specificity) / 2, 3) if None not in (recall, specificity) else None,
            "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
            "accuracy_by_band": {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 3)}
                                 for k, v in sorted(bands.items())}}


def usage(rows):
    ok = [r for r in rows if "error" not in r and "error_code" not in r]
    tokens = [r["input_tokens"] for r in ok if r.get("input_tokens")]
    latency = [r["duration_seconds"] for r in ok]
    costs = [r["cost"] for r in ok if r.get("cost") is not None]
    out = {"requests": len(ok), "errors": len(rows) - len(ok),
           "median_latency_s": round(statistics.median(latency), 3) if latency else None,
           "mean_input_tokens": round(statistics.mean(tokens)) if tokens else None,
           "total_input_tokens": sum(tokens)}
    if costs:
        out["reported_cost_usd"] = round(sum(costs), 5)
        out["cost_per_1000_requests_usd"] = round(1000 * statistics.mean(costs), 4)
    elif tokens:
        out["text_rate_cost_usd"] = round(sum(tokens) * TEXT_RATE_PER_TOKEN, 5)
        out["text_rate_cost_per_1000_requests_usd"] = round(1000 * statistics.mean(tokens) * TEXT_RATE_PER_TOKEN, 4)
    return out


def score_drawing(manifest, labels, normalised, raw_by_model, results):
    def p_yes(model, task, item, key):
        answer = normalised[model][task][item][key]
        return float(answer["noul"]) if isinstance(answer, dict) else float(answer)

    for model, tasks in normalised.items():
        out = {"usage": usage(raw_by_model[model]), "usage_by_task": {}}
        for task in ("hand_obstruction", "drawing_stage", "best_still", "pencil_contact"):
            out["usage_by_task"][task] = usage([r for r in raw_by_model[model] if r["task"] == task])

        hand = labels["hand_obstruction"]
        covered = [(hand[i]["covered"], p_yes(model, "hand_obstruction", i, "covered")) for i in tasks["hand_obstruction"]]
        visible = [(hand[i]["hand_visible"], p_yes(model, "hand_obstruction", i, "hand_visible")) for i in tasks["hand_obstruction"]]
        out["hand_obstruction"] = {"covered": binary_scores(covered), "hand_visible": binary_scores(visible),
                                   "always_yes_baseline_covered": round(sum(t for t, _ in covered) / len(covered), 3),
                                   "mistakes_covered": sorted(i for i in tasks["hand_obstruction"]
                                                              if (p_yes(model, "hand_obstruction", i, "covered") >= 0.5) != hand[i]["covered"])}

        stage_rows = []
        for item, answer in tasks["drawing_stage"].items():
            truth = labels["drawing_stage"][item]
            if isinstance(answer["stage"], dict):
                score = float(answer["stage"]["score"])
                probs = answer["stage"]["probabilities"]
                pred = int(max(probs, key=probs.get))
                confidence = float(answer["stage"]["confidence"])
            else:
                score = float(answer["stage"])
                pred = int(round(score))
                confidence = float(answer.get("confidence", 0))
            stage_rows.append({"item": item, "truth": truth["stage"], "acceptable": truth["acceptable"],
                               "pred": pred, "score": round(score, 2), "confidence": round(confidence, 2),
                               "exact": pred == truth["stage"], "acceptable_hit": pred in truth["acceptable"],
                               "abs_error": abs(pred - truth["stage"])})
        out["drawing_stage"] = {"n": len(stage_rows),
                                "exact_accuracy": round(sum(r["exact"] for r in stage_rows) / len(stage_rows), 3),
                                "acceptable_accuracy": round(sum(r["acceptable_hit"] for r in stage_rows) / len(stage_rows), 3),
                                "mean_abs_error_levels": round(statistics.mean(r["abs_error"] for r in stage_rows), 2),
                                "confusion": dict(Counter(f"{STAGE_NAMES[r['truth']]}->{STAGE_NAMES[r['pred']]}" for r in stage_rows)),
                                "rows": stage_rows}

        best_rows = []
        for group in labels["best_still"]:
            answer = tasks["best_still"][group["clip"]]["best"]
            frames = next(g["frames"] for g in manifest["best_still_groups"] if g["clip"] == group["clip"])
            key = answer["choice"] if isinstance(answer, dict) else answer
            pick = frames[int(key.split("_")[1]) - 1]
            confidence = float(answer["confidence"]) if isinstance(answer, dict) else float(
                tasks["best_still"][group["clip"]].get("confidence", 0))
            best_rows.append({"clip": group["clip"], "pick": pick, "best": group["best"], "acceptable": group["acceptable"],
                              "exact": pick == group["best"], "acceptable_hit": pick in group["acceptable"],
                              "confidence": round(confidence, 2),
                              "probabilities": answer.get("probabilities") if isinstance(answer, dict) else None})
        out["best_still"] = {"n": len(best_rows), "exact_hits": sum(r["exact"] for r in best_rows),
                             "acceptable_hits": sum(r["acceptable_hit"] for r in best_rows), "rows": best_rows}

        cant_tell = []
        confusion = Counter()
        bands = defaultdict(lambda: [0, 0])
        mistakes = []
        reverse = {"Pencil tip touching the paper": "touching",
                   "Pencil tip visible but not touching the paper": "not-touching",
                   "No pencil tip visible": "no-tip"}
        for item, answer in tasks["pencil_contact"].items():
            truth = manifest["frames"][item]["pencil_label"]
            tip = answer["tip"]
            if isinstance(tip, dict):
                pred, confidence = tip["choice"], float(tip["confidence"])
            else:
                pred, confidence = reverse.get(tip, tip), float(answer.get("confidence", 0))
            if truth == "cant-tell":
                cant_tell.append({"item": item, "pred": pred, "confidence": round(confidence, 2)})
                continue
            confusion[f"{truth}->{pred}"] += 1
            b = choice_band(confidence)
            bands[b][0] += int(pred == truth)
            bands[b][1] += 1
            if pred != truth:
                mistakes.append({"item": item.split("__")[-1], "truth": truth, "pred": pred, "confidence": round(confidence, 2)})
        n = sum(confusion.values())
        correct = sum(c for k, c in confusion.items() if k.split("->")[0] == k.split("->")[1])
        per_class = {}
        for cls in ("touching", "not-touching", "no-tip"):
            total = sum(c for k, c in confusion.items() if k.startswith(cls + "->"))
            per_class[cls] = {"n": total, "recall": round(confusion[f"{cls}->{cls}"] / total, 3) if total else None}
        subset = manifest["hand_obstruction_labeled_subset"]
        subset_correct = sum(1 for i in subset if manifest["frames"][i]["pencil_label"] != "cant-tell"
                             and not any(m["item"] == i.split("__")[-1] for m in mistakes))
        subset_n = sum(1 for i in subset if manifest["frames"][i]["pencil_label"] != "cant-tell")
        out["pencil_contact"] = {"n_scored": n, "accuracy": round(correct / n, 3), "per_class": per_class,
                                 "majority_class_baseline": round(max(v["n"] for v in per_class.values()) / n, 3),
                                 "confusion": dict(sorted(confusion.items())),
                                 "accuracy_by_band": {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 3)}
                                                      for k, v in sorted(bands.items())},
                                 "planned_12_frame_subset": {"n": subset_n, "accuracy": round(subset_correct / subset_n, 3)},
                                 "mistakes": mistakes}
        results["pencil_contact_cant_tell"][model] = cant_tell
        results["models"][model] = out


def score_import(import_rows, target):
    per_model = defaultdict(list)
    for row in import_rows:
        per_model[row["requested_model"]].append(row)
    for model, rows in per_model.items():
        table = []
        for row in rows:
            pred = row.get("predicted_code")
            answer = row.get("answers") or {}
            if isinstance(answer.get("type_code"), dict):
                confidence = answer["type_code"]["confidence"]
                probs = answer["type_code"]["probabilities"]
                runner_up = sorted(probs.items(), key=lambda kv: -kv[1])[1]
                thumb = answer["best_thumbnail"]["choice"]
                framing = answer["framing"]["choice"]
            else:
                confidence = answer.get("type_confidence")
                alternates = answer.get("alternates") or []
                runner_up = (alternates[0]["code"], alternates[0]["confidence"]) if alternates else None
                thumb = answer.get("best_thumbnail_index")
                framing = answer.get("framing")
            table.append({"fixture": row["fixture_id"], "true": row["true_code"], "pred": pred,
                          "strict": pred == row["true_code"],
                          "lenient": pred in (LENIENT.get(row["fixture_id"], set()) | {row["true_code"]}),
                          "confidence": confidence, "runner_up": runner_up, "framing": framing, "thumbnail": thumb,
                          "input_tokens": row.get("input_tokens"), "cost": row.get("cost"),
                          "latency_s": row.get("duration_seconds"), "error": row.get("error")})
        target[model] = {"model": next((r.get("model") for r in rows if r.get("model")), model),
                         "max_width": rows[0].get("max_width"),
                         "n": len(table), "strict_correct": sum(r["strict"] for r in table),
                         "lenient_correct": sum(r["lenient"] for r in table),
                         "usage": usage(rows), "rows": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--jev", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--imports", type=Path, required=True, help="production-width (384) replay")
    parser.add_argument("--imports-512", type=Path, default=None, help="512-wide replay, the smallest Jev accepts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    jev_rows = [r for r in load(args.jev) if "error_code" not in r]
    gem_rows = [r for r in load(args.gemini) if "error" not in r and "parse_error" not in r]
    results = {"models": {}, "pencil_contact_cant_tell": {}}

    normalised = defaultdict(lambda: defaultdict(dict))  # model -> task -> item -> answer
    raw_by_model = defaultdict(list)
    for row in jev_rows:
        normalised[row["model"]][row["task"]][row["item_id"]] = row["answers"]
        raw_by_model[row["model"]].append(row)
    for row in gem_rows:
        normalised["gemini-3.1-flash-lite"][row["task"]][row["item_id"]] = row["answers"]
        raw_by_model["gemini-3.1-flash-lite"].append(row)
    score_drawing(manifest, labels, normalised, raw_by_model, results)

    for bucket, path in (("import_384", args.imports), ("import_512", args.imports_512)):
        if path is not None:
            results[bucket] = {}
            score_import(load(path), results[bucket])

    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for model, out in results["models"].items():
        print(model, "hand", out["hand_obstruction"]["covered"]["accuracy"], "stage", out["drawing_stage"]["exact_accuracy"],
              out["drawing_stage"]["acceptable_accuracy"], "best", out["best_still"]["exact_hits"],
              out["best_still"]["acceptable_hits"], "pencil", out["pencil_contact"]["accuracy"], out["usage"])
    for bucket in ("import_384", "import_512"):
        for model, out in results.get(bucket, {}).items():
            print(bucket, model, "strict", out["strict_correct"], "lenient", out["lenient_correct"], out["usage"])


if __name__ == "__main__":
    main()
