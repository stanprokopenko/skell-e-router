"""Score Jev's pencil-tip boxes against Stan's labeled tip points.

Inputs: the manifest, the box JSONL from run_box.py and Stan's StanStudio
label files. Output: one JSON file with every number the write-up quotes,
plus a printed summary. Only frames labeled touching or not-touching carry a
tip point, so those 23 are scored; no-tip and cant-tell frames are reported
separately. Distances are in source pixels (3840 by 2160) unless noted.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

LABEL_DIR = Path("D:/solar-sailer/labeling/dataset/labels/tp-20260831-001")
SOURCE = (3840, 2160)
THRESHOLDS = (25, 50, 100)
BANDS = [("high (>=0.6)", 0.6), ("mid (0.3-0.6)", 0.3), ("low (<0.3)", 0.0)]


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(frame_id: str):
    data = json.loads((LABEL_DIR / f"{frame_id}.StanStudio.json").read_text(encoding="utf-8"))
    point = data["point"]
    frac = None if point is None else (point["x"] / data["sourceWidth"], point["y"] / data["sourceHeight"])
    return data["state"], frac


def px(dx: float, dy: float) -> float:
    return math.hypot(dx * SOURCE[0], dy * SOURCE[1])


def band(conf) -> str:
    if conf is None:
        return "none"
    return next(name for name, floor in BANDS if conf >= floor)


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(rows):
    """Aggregate per-frame rows (each has distance_px, optionally inside)."""
    if not rows:
        return None
    d = [r["distance_px"] for r in rows]
    return {
        "n": len(rows),
        "median_px": statistics.median(d),
        "mean_px": statistics.fmean(d),
        "p90_px": sorted(d)[max(0, math.ceil(0.9 * len(d)) - 1)],
        "max_px": max(d),
        **{f"within_{t}px": sum(x <= t for x in d) / len(d) for t in THRESHOLDS},
        "inside_box": sum(r["inside"] for r in rows) / len(rows) if "inside" in rows[0] else None,
    }


def score_variant(rows_by_frame, frames_scored, truth):
    per_frame, by_state, by_band = [], defaultdict(list), defaultdict(list)
    for fid in frames_scored:
        row = rows_by_frame[fid]
        state, (px_, py_) = truth[fid]
        box = row["box"]
        cx, cy = (box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2
        inside = box["left"] <= px_ <= box["right"] and box["top"] <= py_ <= box["bottom"]
        # Distance from the point to the nearest box edge, 0 when inside.
        ex = max(box["left"] - px_, 0, px_ - box["right"])
        ey = max(box["top"] - py_, 0, py_ - box["bottom"])
        entry = {
            "review_id": row["review_id"], "frame_id": fid, "state": state,
            "stan_point": [round(px_, 4), round(py_, 4)], "box": box, "confidence": row["confidence"],
            "center": [round(cx, 4), round(cy, 4)],
            "distance_px": round(px(cx - px_, cy - py_), 1),
            "distance_1024px": round(px(cx - px_, cy - py_) / (SOURCE[0] / 1024), 1),
            "edge_distance_px": round(px(ex, ey), 1),
            "inside": inside,
            "box_width_px": round((box["right"] - box["left"]) * SOURCE[0]),
            "box_height_px": round((box["bottom"] - box["top"]) * SOURCE[1]),
            "box_area_share": round((box["right"] - box["left"]) * (box["bottom"] - box["top"]), 5),
        }
        per_frame.append(entry)
        by_state[state].append(entry)
        by_band[band(row["confidence"])].append(entry)
    return {
        "overall": summarize(per_frame),
        "by_state": {k: summarize(v) for k, v in by_state.items()},
        "by_confidence": {name: summarize(by_band[name]) for name in [*(n for n, _ in BANDS), "none"] if by_band[name]},
        "median_confidence": median_or_none(e["confidence"] for e in per_frame),
        "median_box_px": [statistics.median(e["box_width_px"] for e in per_frame),
                          statistics.median(e["box_height_px"] for e in per_frame)],
        "tall_boxes_over_30pct_height": sum(e["box_height_px"] > 0.3 * SOURCE[1] for e in per_frame),
        "per_frame": per_frame,
    }


def baselines(frames_scored, truth):
    pts = [truth[f][1] for f in frames_scored]
    mean = (statistics.fmean(p[0] for p in pts), statistics.fmean(p[1] for p in pts))

    def loo(i):
        others = [p for j, p in enumerate(pts) if j != i]
        return statistics.fmean(p[0] for p in others), statistics.fmean(p[1] for p in others)

    out = {}
    for name, guess in (("image_center", lambda i: (0.5, 0.5)), ("mean_labeled_point", lambda i: mean),
                        ("mean_labeled_point_leave_one_out", loo)):
        rows = [{"distance_px": px(guess(i)[0] - p[0], guess(i)[1] - p[1])} for i, p in enumerate(pts)]
        out[name] = summarize(rows)
    out["mean_labeled_point_fraction"] = [round(mean[0], 4), round(mean[1], 4)]
    out["label_spread_px"] = {
        "x_std": round(statistics.pstdev(p[0] for p in pts) * SOURCE[0], 1),
        "y_std": round(statistics.pstdev(p[1] for p in pts) * SOURCE[1], 1),
    }
    return out


def unscored(rows_by_frame, frame_ids, truth):
    out = []
    for fid in frame_ids:
        row = rows_by_frame[fid]
        state, pt = truth[fid]
        box = row["box"]
        entry = {"review_id": row["review_id"], "state": state, "box": box, "confidence": row["confidence"],
                 "box_area_share": round((box["right"] - box["left"]) * (box["bottom"] - box["top"]), 5)}
        if pt is not None:
            cx, cy = (box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2
            entry["distance_px"] = round(px(cx - pt[0], cy - pt[1]), 1)
            entry["inside"] = box["left"] <= pt[0] <= box["right"] and box["top"] <= pt[1] <= box["bottom"]
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("docs/jev-images/2026-11-04-manifest.json"))
    parser.add_argument("--runs", type=Path, default=Path("docs/jev-images/2026-11-04-box-runs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/jev-images/2026-11-04-box-results.json"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frame_ids = manifest["pencil_contact_frames"]
    truth = {fid: label(fid) for fid in frame_ids}
    all_rows = load_jsonl(args.runs)
    rows = [r for r in all_rows if r["error"] is None and r["box"] is not None]
    errors = [r for r in all_rows if r["error"] is not None]
    by_variant = defaultdict(dict)
    for r in rows:
        by_variant[r["variant"]][r["frame_id"]] = r

    scored = [f for f in frame_ids if truth[f][0] in ("touching", "not-touching")]
    no_tip = [f for f in frame_ids if truth[f][0] == "no-tip"]
    cant_tell = [f for f in frame_ids if truth[f][0] == "cant-tell"]
    assert all(truth[f][1] is not None for f in scored)
    assert all(truth[f][1] is None for f in no_tip)

    results = {
        "model": sorted({r["model"] for r in rows}),
        "requests": len(rows), "errors": len(errors),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "cost_text_rate": round(sum(r["cost_text_rate"] or 0 for r in rows), 5),
        "median_duration_seconds": round(statistics.median(r["duration_seconds"] for r in rows), 3),
        "frames": {"scored": len(scored), "no_tip": len(no_tip), "cant_tell": len(cant_tell)},
        "baselines": baselines(scored, truth),
        "variants": {},
    }
    for variant, rows_by_frame in by_variant.items():
        missing = [f for f in frame_ids if f not in rows_by_frame]
        entry = {
            "instructions": next(iter(rows_by_frame.values()))["instructions"],
            "missing_frames": missing,
            "scored": score_variant(rows_by_frame, [f for f in scored if f in rows_by_frame], truth),
            "no_tip": unscored(rows_by_frame, [f for f in no_tip if f in rows_by_frame], truth),
            "cant_tell": unscored(rows_by_frame, [f for f in cant_tell if f in rows_by_frame], truth),
        }
        # Jev always returns a box; does its confidence tell tip frames from no-tip frames?
        tip_conf = [e["confidence"] for e in entry["scored"]["per_frame"]]
        none_conf = [e["confidence"] for e in entry["no_tip"]]
        entry["tip_presence_by_confidence"] = {
            "median_confidence_tip_frames": median_or_none(tip_conf),
            "median_confidence_no_tip_frames": median_or_none(none_conf),
            "thresholds": {str(t): {
                "tip_frames_at_or_above": sum(c is not None and c >= t for c in tip_conf),
                "no_tip_frames_at_or_above": sum(c is not None and c >= t for c in none_conf),
                "accuracy_as_presence_detector": round(
                    (sum(c is not None and c >= t for c in tip_conf) + sum(c is None or c < t for c in none_conf))
                    / (len(tip_conf) + len(none_conf)), 3),
            } for t in (0.3, 0.4, 0.5, 0.6)},
        }
        entry["scored"]["edge_distance_within_35px"] = sum(
            e["edge_distance_px"] <= 35 for e in entry["scored"]["per_frame"])
        results["variants"][variant] = entry
    # Agreement between the two wordings on scored frames: center-to-center distance.
    if {"detailed", "short"} <= set(by_variant):
        gaps = []
        for f in [f for f in scored if f in by_variant["detailed"] and f in by_variant["short"]]:
            a, b = by_variant["detailed"][f]["box"], by_variant["short"][f]["box"]
            gaps.append(px((a["left"] + a["right"] - b["left"] - b["right"]) / 2,
                           (a["top"] + a["bottom"] - b["top"] - b["bottom"]) / 2))
        results["wording_agreement"] = {"median_center_gap_px": round(statistics.median(gaps), 1),
                                        "within_100px": sum(x <= 100 for x in gaps) / len(gaps)}
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"{results['requests']} requests, {results['errors']} errors, model {results['model']}, "
          f"{results['input_tokens']} input tokens, ${results['cost_text_rate']} at text rate")
    print(f"scored {len(scored)} frames (touching + not-touching), {len(no_tip)} no-tip, {len(cant_tell)} cant-tell")
    for name, s in results["baselines"].items():
        if isinstance(s, dict) and "median_px" in s:
            print(f"baseline {name:36s} median {s['median_px']:6.1f} px  p90 {s['p90_px']:6.1f}  "
                  f"<=100px {s['within_100px']:.0%}")
    for variant, v in results["variants"].items():
        s = v["scored"]["overall"]
        if s is None:
            print(f"{variant}: no scored frames")
            continue
        print(f"{variant:9s} median {s['median_px']:6.1f} px  p90 {s['p90_px']:6.1f}  <=25 {s['within_25px']:.0%}  "
              f"<=50 {s['within_50px']:.0%}  <=100 {s['within_100px']:.0%}  inside {s['inside_box']:.0%}  "
              f"tall boxes {v['scored']['tall_boxes_over_30pct_height']}")
        for state, s in v["scored"]["by_state"].items():
            print(f"  {state:13s} n={s['n']:2d} median {s['median_px']:6.1f}  <=100 {s['within_100px']:.0%}  "
                  f"inside {s['inside_box']:.0%}")
        for b, s in v["scored"]["by_confidence"].items():
            print(f"  {b:13s} n={s['n']:2d} median {s['median_px']:6.1f}  <=100 {s['within_100px']:.0%}  "
                  f"inside {s['inside_box']:.0%}")
    if "wording_agreement" in results:
        print("wording agreement", results["wording_agreement"])


if __name__ == "__main__":
    main()
