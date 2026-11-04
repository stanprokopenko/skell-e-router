"""Ask Jev to box the pencil tip in each pencil-contact frame.

Sends one box question per frame per wording through the router's classify()
and appends one JSON line per request to the output file. Never writes image
bytes or the API key. Re-running skips (variant, model, frame) rows already in
the output without an error, so a partial run can be resumed.
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skell_e_router import classify, RouterError  # noqa: E402

WORDINGS = {
    "detailed": "The tip of the pencil in `frame`, where the graphite point meets or nearly meets the paper.",
    "short": "The pencil tip in `frame`.",
}
#: Text input rate in dollars per million tokens; image pricing is unpublished.
TEXT_RATE_PER_MILLION = 0.042


def asset(path: Path):
    data = path.read_bytes()
    return str(uuid4()), {"type": "image", "format": "jpg", "data_base64": base64.b64encode(data).decode("ascii")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL, appended to")
    parser.add_argument("--review-ids", type=Path, default=Path(__file__).with_name("2026-11-04-review-ids.json"))
    parser.add_argument("--frames-root", type=Path, default=Path("C:/Users/Stan/Documents/GitHub/skell-e-router"),
                        help="directory the manifest's image paths are relative to")
    parser.add_argument("--models", default="jev-latest")
    parser.add_argument("--variants", default=",".join(WORDINGS))
    parser.add_argument("--limit", type=int, default=0, help="stop after N new requests (cost probe)")
    args = parser.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("Read TYPESAFE_API_KEY from the agent vault into this process first.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review_ids = {v["frame_id"]: rid for rid, v in json.loads(args.review_ids.read_text(encoding="utf-8")).items()
                  if v["task"] == "pencil_contact"}
    frame_ids = manifest["pencil_contact_frames"]
    paths = {fid: args.frames_root / manifest["frames"][fid]["image"].replace("\\", "/") for fid in frame_ids}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} frame images missing, first: {missing[0]}")
    variants = args.variants.split(",")
    models = args.models.split(",")

    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("error") is None:
                done.add((row["variant"], row["requested_model"], row["frame_id"]))
    sent = 0
    totals = {"requests": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}
    with args.output.open("a", encoding="utf-8") as out:
        for frame_id in frame_ids:
            for variant in variants:
                for model in models:
                    if (variant, model, frame_id) in done:
                        continue
                    if args.limit and sent >= args.limit:
                        print(json.dumps(totals))
                        return
                    placeholder, blob = asset(paths[frame_id])
                    instructions = WORDINGS[variant]
                    row = {"task": "pencil_tip_box", "variant": variant, "review_id": review_ids.get(frame_id),
                           "frame_id": frame_id, "requested_model": model, "model": None,
                           "instructions": instructions, "box": None, "confidence": None,
                           "input_tokens": None, "output_tokens": None}
                    started = time.perf_counter()
                    try:
                        result = classify(model, {"frame": placeholder},
                                          {"tip": {"type": "box", "instructions": instructions}},
                                          assets={placeholder: blob}, config={"typesafe_api_key": key}, timeout=60)
                        answer = result.answers["tip"]
                        row.update({"model": result.model, "box": answer["box"],
                                    "confidence": answer.get("confidence"),
                                    "input_tokens": result.input_tokens, "output_tokens": result.output_tokens})
                        error = None
                    except RouterError as exc:
                        error = f"{exc.code}: {exc}"
                        totals["errors"] += 1
                    duration = round(time.perf_counter() - started, 4)
                    row.update({"duration_seconds": duration,
                                "cost_text_rate": (row["input_tokens"] * TEXT_RATE_PER_MILLION / 1e6
                                                   if row["input_tokens"] is not None else None),
                                "error": error,
                                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    encoded = json.dumps(row)
                    if key in encoded or blob["data_base64"][:64] in encoded:
                        raise SystemExit("Sensitive output detected; row not written.")
                    out.write(encoded + "\n")
                    out.flush()
                    sent += 1
                    totals["requests"] += 1
                    totals["input_tokens"] += row["input_tokens"] or 0
                    totals["output_tokens"] += row["output_tokens"] or 0
                    print(f"{row['review_id']} {variant} {model} box={row['box']} conf={row['confidence']} "
                          f"tokens={row['input_tokens']}/{row['output_tokens']} {duration}s {error or ''}")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
