"""Gemini baseline for the drawing-camera tasks, through the router's ask_ai().

Same frames and the same four questions as run_eval.py, asked as strict JSON.
One JSON line per request; image bytes never written. Reruns skip finished rows.
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skell_e_router import ask_ai  # noqa: E402
from run_eval import STAGE_CRITERIA, PENCIL_CRITERIA, BEST_STILL_KEYS  # noqa: E402

JSON_RULE = "Return ONLY one JSON object, no prose, no code fences."


def data_uri(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def requests_for(manifest, tasks):
    frames = manifest["frames"]
    if "hand_obstruction" in tasks:
        hand_ids = list(manifest["hand_obstruction_frames"])
        for group in manifest["drawing_stage_groups"] + manifest["best_still_groups"]:
            hand_ids += [f for f in group["frames"] if f not in hand_ids]
        for frame_id in hand_ids:
            yield ("hand_obstruction", frame_id, [frames[frame_id]["image"]],
                   "This is an overhead drawing-camera frame. Answer two yes/no questions with probabilities. "
                   "covered: does a hand, arm or tool cover part of the drawing on the page? "
                   "hand_visible: is a human hand or arm visible anywhere in the frame? "
                   f'{JSON_RULE} Schema: {{"covered": <0..1>, "hand_visible": <0..1>}}')
    if "drawing_stage" in tasks:
        for group in manifest["drawing_stage_groups"]:
            for frame_id in group["frames"]:
                levels = "; ".join(f"{i}: {c}" for i, c in enumerate(STAGE_CRITERIA))
                yield ("drawing_stage", frame_id, [frames[frame_id]["image"]],
                       f"What stage of the drawing process does the page in this frame show? Levels: {levels}. "
                       f'{JSON_RULE} Schema: {{"stage": <0|1|2|3>, "confidence": <0..1>}}')
    if "best_still" in tasks:
        for group in manifest["best_still_groups"]:
            yield ("best_still", group["clip"], [frames[f]["image"] for f in group["frames"]],
                   "The four attached images are stills frame_1..frame_4, in order, from one video of a drawing "
                   "being made. Which one is the best still to show this drawing? The best still shows the drawing "
                   "most fully, with no hand or tool in the way, sharp, well lit, and showing the most advanced state "
                   f'of the drawing. {JSON_RULE} Schema: {{"best": "frame_1"|"frame_2"|"frame_3"|"frame_4", "confidence": <0..1>}}')
    if "pencil_contact" in tasks:
        options = " | ".join(f'"{v}"' for v in PENCIL_CRITERIA.values())
        for frame_id in manifest["pencil_contact_frames"]:
            yield ("pencil_contact", frame_id, [frames[frame_id]["image"]],
                   "This is an overhead view of someone drawing with a pencil. What is the state of the pencil tip? "
                   f'{JSON_RULE} Schema: {{"tip": {options}, "confidence": <0..1>}}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--tasks", default="hand_obstruction,drawing_stage,best_still,pencil_contact")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["task"], row["item_id"]))
    sent = 0
    total_cost = 0.0
    with args.output.open("a", encoding="utf-8") as out:
        for task, item_id, paths, prompt in requests_for(manifest, args.tasks.split(",")):
            if (task, item_id) in done or (args.limit and sent >= args.limit):
                continue
            row = {"task": task, "item_id": item_id, "requested_model": args.model, "prompt": prompt}
            started = time.perf_counter()
            try:
                response = ask_ai(args.model, prompt, images=[data_uri(p) for p in paths], rich_response=True)
                text = response.content
                row["raw_text"] = text
                match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text.strip(), re.S)
                try:
                    row["answers"] = json.loads(match.group(1) if match else text.strip())
                except json.JSONDecodeError as error:
                    row["parse_error"] = str(error)
                row.update({"model": getattr(response, "model", args.model),
                            "input_tokens": getattr(response, "prompt_tokens", None),
                            "output_tokens": getattr(response, "completion_tokens", None),
                            "cost": getattr(response, "cost", None)})
                total_cost += row["cost"] or 0.0
            except Exception as error:  # noqa: BLE001 - record and continue
                row["error"] = f"{type(error).__name__}: {error}"
            row["duration_seconds"] = round(time.perf_counter() - started, 3)
            out.write(json.dumps(row, default=str) + "\n")
            out.flush()
            sent += 1
            print(f"{task} {item_id[:48]} tokens={row.get('input_tokens')} cost={row.get('cost')} "
                  f"{row['duration_seconds']}s {row.get('error', '')[:100]}")
    print(json.dumps({"requests": sent, "total_cost_usd": total_cost}))


if __name__ == "__main__":
    main()
