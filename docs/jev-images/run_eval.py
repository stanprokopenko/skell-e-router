"""Run the drawing-camera tasks through the router's classify() with images.

Reads the manifest, sends one request per frame (or per best-still group),
and appends one JSON line per request to the output file. Never writes image
bytes or the API key. Re-running skips (task, model, item) rows already in
the output, so a partial run can be resumed.

Labels are deliberately not read here; scoring happens in score.py.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skell_e_router import classify, RouterError  # noqa: E402

STAGE_CRITERIA = [
    "Blank page: no marks on the paper",
    "First marks: only a few lines, construction or gesture lines",
    "Drawing in progress: clearly underway, substantial but unfinished",
    "Finished drawing: complete, little left to add",
]
PENCIL_CRITERIA = {
    "touching": "Pencil tip touching the paper",
    "not-touching": "Pencil tip visible but not touching the paper",
    "no-tip": "No pencil tip visible",
}
BEST_STILL_KEYS = ["frame_1", "frame_2", "frame_3", "frame_4"]
HAND_QUESTIONS = {
    "v1": "Does a hand, arm or tool cover part of the drawing on the page in `frame`?",
    # v2 after Stan's review: the first wording let a blank page not count as "the drawing".
    "v2": ("Does a hand, arm or tool cover any part of the paper in `frame`? Treat the whole sheet of paper as the "
           "drawing, even when it is blank. A hand or tool resting anywhere on the paper counts as covering it; "
           "a hand beside the paper or off the page does not."),
}


def asset(path: str):
    data = Path(path).read_bytes()
    return str(uuid4()), {"type": "image", "format": "jpg", "data_base64": base64.b64encode(data).decode("ascii")}


def requests_for(manifest, tasks, hand_wording="v1"):
    """Yield (task, item_id, frame_ids, state, questions, assets)."""
    frames = manifest["frames"]
    if "hand_obstruction" in tasks:
        # All 52 frames carry blind hand-obstruction labels: the 20 selected
        # for the task plus the drawing-stage and best-still frames.
        hand_ids = list(manifest["hand_obstruction_frames"])
        for group in manifest["drawing_stage_groups"] + manifest["best_still_groups"]:
            hand_ids += [f for f in group["frames"] if f not in hand_ids]
        for frame_id in hand_ids:
            placeholder, blob = asset(frames[frame_id]["image"])
            yield ("hand_obstruction", frame_id, [frame_id], {"frame": placeholder}, {
                "covered": {"type": "noul", "instructions": HAND_QUESTIONS[hand_wording]},
                "hand_visible": {"type": "noul",
                                 "instructions": "Is a human hand or arm visible anywhere in `frame`?"},
            }, {placeholder: blob})
    if "drawing_stage" in tasks:
        for group in manifest["drawing_stage_groups"]:
            for frame_id in group["frames"]:
                placeholder, blob = asset(frames[frame_id]["image"])
                yield ("drawing_stage", frame_id, [frame_id], {"frame": placeholder}, {
                    "stage": {"type": "score",
                              "instructions": "What stage of the drawing process does the page in `frame` show?",
                              "criteria": STAGE_CRITERIA},
                }, {placeholder: blob})
    if "best_still" in tasks:
        for group in manifest["best_still_groups"]:
            state, assets = {}, {}
            for key, frame_id in zip(BEST_STILL_KEYS, group["frames"]):
                placeholder, blob = asset(frames[frame_id]["image"])
                state[key] = placeholder
                assets[placeholder] = blob
            yield ("best_still", group["clip"], group["frames"], state, {
                "best": {"type": "choice",
                         "instructions": ("`frame_1`, `frame_2`, `frame_3` and `frame_4` are stills from one video of "
                                          "a drawing being made. Which one is the best still to show this drawing? "
                                          "The best still shows the drawing most fully, with no hand or tool in the "
                                          "way, sharp, well lit, and showing the most advanced state of the drawing."),
                         "criteria": {key: None for key in BEST_STILL_KEYS}},
            }, assets)
    if "pencil_contact" in tasks:
        for frame_id in manifest["pencil_contact_frames"]:
            placeholder, blob = asset(frames[frame_id]["image"])
            yield ("pencil_contact", frame_id, [frame_id], {"frame": placeholder}, {
                "tip": {"type": "choice",
                        "instructions": ("`frame` is an overhead view of someone drawing with a pencil. "
                                         "What is the state of the pencil tip?"),
                        "criteria": PENCIL_CRITERIA},
            }, {placeholder: blob})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL, appended to")
    parser.add_argument("--models", default="jev,jev-latest")
    parser.add_argument("--tasks", default="hand_obstruction,drawing_stage,best_still,pencil_contact")
    parser.add_argument("--limit", type=int, default=0, help="stop after N new requests (cost probe)")
    parser.add_argument("--hand-wording", default="v1", choices=sorted(HAND_QUESTIONS))
    args = parser.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("Read TYPESAFE_API_KEY from the agent vault into this process first.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "error_code" not in row:
                done.add((row["task"], row["requested_model"], row["item_id"]))
    sent = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "errors": 0}
    with args.output.open("a", encoding="utf-8") as out:
        for task, item_id, frame_ids, state, questions, assets in requests_for(manifest, args.tasks.split(","), args.hand_wording):
            if args.limit and sent >= args.limit:
                break
            for model in args.models.split(","):
                if (task, model, item_id) in done:
                    continue
                if args.limit and sent >= args.limit:
                    break
                row = {"task": task, "item_id": item_id, "frame_ids": frame_ids, "requested_model": model,
                       "state_keys": list(state), "questions": questions, "hand_wording": args.hand_wording}
                started = time.perf_counter()
                try:
                    answer = classify(model, state, questions, assets=assets,
                                      config={"typesafe_api_key": key}, timeout=60)
                    row.update({"model": answer.model, "answers": answer.answers,
                                "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
                                "cost": answer.cost})
                    totals["input_tokens"] += answer.input_tokens or 0
                    totals["output_tokens"] += answer.output_tokens or 0
                except RouterError as error:
                    row.update({"error_code": error.code, "error": str(error), "details": error.details})
                    totals["errors"] += 1
                row["duration_seconds"] = round(time.perf_counter() - started, 4)
                encoded = json.dumps(row)
                if key in encoded or any(blob["data_base64"][:64] in encoded for blob in assets.values()):
                    raise SystemExit("Sensitive output detected; row not written.")
                out.write(encoded + "\n")
                out.flush()
                sent += 1
                totals["requests"] += 1
                print(f"{task} {model} {item_id[:48]} tokens={row.get('input_tokens')} "
                      f"{row['duration_seconds']}s {row.get('error_code', '')}")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
