"""Replay Solar Sailer's import-classification benchmark on Jev and on its current model.

Mirrors the production path in solar-sailer's media_classification.py without
calling it: 10 frames per clip at timestamps duration*(i+0.5)/10, width capped at
384 px, JPEG. Jev gets the frames as assets with a Choice question over the 12
shot-type codes. Gemini 3.1 Flash Lite (the production model) gets the frames
plus the production prompt through the router's ask_ai(), exactly as Solar
Sailer's llm_router does. One JSON line per request; image bytes never written.

--extract only samples frames; --models runs the listed models (jev, jev-latest,
gemini-3.1-flash-lite). Reruns skip finished (clip, model) rows. --width caps the
frame width: 384 is production; Jev rejects anything under 512 wide with
"Invalid image dimensions", so the Jev replay uses --width 512.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skell_e_router import ask_ai, classify, RouterError  # noqa: E402

MANIFEST = Path("C:/Users/Stan/Documents/GitHub/solar-sailer/editor/tests/fixtures/media_classification/diversity_manifest.json")
PROMPT = Path("C:/Users/Stan/Documents/GitHub/solar-sailer/editor/prompts/media_classification_video.md")
VIDEO_ROOT = Path("R:/Google Drive/Proko/Episodes/@Stand-Alone Episodes/@LightBox Expo 2024/Demos/303 - Feefal - Sketchbook Tour/Video")
RELOCATED = {
    "sketchbook_C004": VIDEO_ROOT / "Face Cam/1124_10161602_C004_v2.mp4",
    "sketchbook_C006": VIDEO_ROOT / "Demo Cam/1091_02020620_C006_V1-0011.mp4",
    "ddr_world_record": VIDEO_ROOT / "(DDR World Record) Blew My Mind -DOUBLE CHALLENGE- TAKASKE-.mp4",
    "kaiba_hippo": VIDEO_ROOT / "Hippo man gets an extreme makeover ｜ Kaiba (2008).mp4",
    "kaiba_opening": VIDEO_ROOT / "Kaiba - Opening ｜ Seira Kagami - Never.mp4",
    "kaiba_ep1": VIDEO_ROOT / "Kaiba Episode 1.mp4",
    "twilight_baseball": VIDEO_ROOT / "Twilight  (2008) - Vampire Baseball Scene ｜ Movieclips.mp4",
}
N_FRAMES = 10
MAX_WIDTH = 384
FRAME_KEYS = [f"frame_{i}" for i in range(N_FRAMES)]

# Condensed from the production prompt's code table. Jev compares the clip
# against these descriptions; the code is the text before the colon.
CODE_CRITERIA = [
    "FAC: talking head straight to camera, a single presenter, work surface not visible",
    "SCN: scene-level master, multiple subjects or a full environment, location or atmosphere is the content",
    "WRK: the activity or work product itself dominates the frame (drawing, cooking, building, sketchbook pages)",
    "SBJ: the thing being presented or examined, a product, model or object under study",
    "HND: tight macro shot of a hand and tool contact point",
    "SCR: on-screen workflow recorded from a computer, tablet or phone",
    "BRD: presenter drawing or writing on a whiteboard or chalkboard",
    "POV: first-person perspective, body-mounted or handheld as the viewer's eye",
    "MIX: one continuous take where the camera itself moves between several of the shot types above",
    "GFX: titles, lower thirds, diagrams, slides or motion graphics as the content",
    "ANI: drawn or CG animation as the primary content, anime, cartoons",
    "FLM: finished live-action film, TV, music video or commercial footage, letterboxing, watermarks, cinematic grading",
]
CODE_CRITERIA = {line.split(":")[0]: line.split(": ", 1)[1] for line in CODE_CRITERIA}
FRAMING_CRITERIA = {"wide": "full body and surroundings, or several subjects", "medium": "waist-up framing",
                    "close": "head and shoulders dominant", "macro": "extreme close-up detail"}


def probe_duration(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True, timeout=120)
    return float(json.loads(out.stdout)["format"]["duration"])


def sample_timestamps(duration: float, n: int):
    import math
    effective = n if duration >= 2.0 * n else max(1, math.ceil(duration / 0.5))
    return [duration * (i + 0.5) / effective for i in range(effective)]


def extract(video: Path, seconds: float, dst: Path, width: int = MAX_WIDTH):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "1", "-ss", f"{seconds:.3f}", "-i", str(video),
                    "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-q:v", "2", str(dst)],
                   check=True, timeout=300)


def strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return match.group(1) if match else text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--width", type=int, default=MAX_WIDTH)
    args = parser.parse_args()
    fixtures = json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"]
    args.scratch.mkdir(parents=True, exist_ok=True)
    index_path = args.scratch / "frames.json"

    if args.extract:
        index = {}
        for fixture in fixtures:
            video = RELOCATED[fixture["id"]]
            duration = probe_duration(video)
            stamps = sample_timestamps(duration, N_FRAMES)
            entry = {"video": str(video), "size_bytes": video.stat().st_size, "duration_seconds": duration,
                     "timestamps": stamps, "max_width": args.width, "frames": []}
            for i, seconds in enumerate(stamps):
                dst = args.scratch / f"{fixture['id']}__{i:02d}.jpg"
                extract(video, seconds, dst, args.width)
                entry["frames"].append({"path": str(dst), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
            index[fixture["id"]] = entry
            print(f"{fixture['id']}: {len(stamps)} frames from {duration:.1f}s")
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return

    index = json.loads(index_path.read_text(encoding="utf-8"))
    prompt_text = PROMPT.read_text(encoding="utf-8")
    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["fixture_id"], row["requested_model"]))
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")
    with args.output.open("a", encoding="utf-8") as out:
        for fixture in fixtures:
            entry = index[fixture["id"]]
            blobs = [Path(f["path"]).read_bytes() for f in entry["frames"]]
            for model in [m for m in args.models.split(",") if m]:
                if (fixture["id"], model) in done:
                    continue
                row = {"fixture_id": fixture["id"], "true_code": fixture["true_code"], "notes": fixture["notes"],
                       "requested_model": model, "n_frames": len(blobs), "timestamps": entry["timestamps"],
                       "max_width": entry.get("max_width", MAX_WIDTH)}
                started = time.perf_counter()
                try:
                    if model.startswith("jev"):
                        if not typesafe_key:
                            raise SystemExit("TYPESAFE_API_KEY missing")
                        state, assets = {}, {}
                        for key, blob in zip(FRAME_KEYS, blobs):
                            placeholder = str(uuid4())
                            state[key] = placeholder
                            assets[placeholder] = {"type": "image", "format": "jpg",
                                                   "data_base64": base64.b64encode(blob).decode("ascii")}
                        keys = ", ".join(f"`{k}`" for k in state)
                        questions = {
                            "type_code": {"type": "choice",
                                          "instructions": (f"{keys} are frames sampled evenly through one video clip, "
                                                           "in time order. Which shot-type code best describes the clip?"),
                                          "criteria": CODE_CRITERIA},
                            "framing": {"type": "choice",
                                        "instructions": f"How tight is the camera framing across {keys}?",
                                        "criteria": FRAMING_CRITERIA},
                            "best_thumbnail": {"type": "choice",
                                               "instructions": (f"Which of {keys} best represents this clip as a thumbnail: "
                                                                "in focus, well composed, clearly shows what the clip is about?"),
                                               "criteria": {key: None for key in state}},
                        }
                        row["questions"] = questions
                        answer = classify(model, state, questions, assets=assets,
                                          config={"typesafe_api_key": typesafe_key}, timeout=120)
                        predicted = answer.answers["type_code"]["choice"]
                        row.update({"model": answer.model, "answers": answer.answers, "predicted_code": predicted,
                                    "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
                                    "cost": answer.cost})
                    else:
                        full_prompt = (f"{prompt_text.rstrip()}\n\nYou are receiving {len(blobs)} frames sampled evenly "
                                       f"through the clip, indexed 0..{len(blobs) - 1} in the order attached.\n")
                        images = ["data:image/jpeg;base64," + base64.b64encode(b).decode("ascii") for b in blobs]
                        response = ask_ai(model, full_prompt, images=images, rich_response=True)
                        text = response.content if hasattr(response, "content") else str(response)
                        row["raw_text"] = text
                        row["prompt_sha256"] = prompt_sha
                        try:
                            doc = json.loads(strip_fences(text))
                            row["answers"] = doc
                            row["predicted_code"] = doc.get("type_code")
                        except json.JSONDecodeError as error:
                            row["parse_error"] = str(error)
                        row.update({"model": getattr(response, "model", model),
                                    "input_tokens": getattr(response, "prompt_tokens", None),
                                    "output_tokens": getattr(response, "completion_tokens", None),
                                    "cost": getattr(response, "cost", None)})
                except RouterError as error:
                    row.update({"error": str(error), "error_code": error.code})
                except Exception as error:  # noqa: BLE001 - record and continue
                    row.update({"error": f"{type(error).__name__}: {error}"})
                row["duration_seconds"] = round(time.perf_counter() - started, 3)
                encoded = json.dumps(row, default=str)
                if (typesafe_key and typesafe_key in encoded) or any(
                        base64.b64encode(b).decode("ascii")[:64] in encoded for b in blobs):
                    raise SystemExit("Sensitive output detected; row not written.")
                out.write(encoded + "\n")
                out.flush()
                print(f"{fixture['id']} {model} true={fixture['true_code']} pred={row.get('predicted_code')} "
                      f"tokens={row.get('input_tokens')} cost={row.get('cost')} {row['duration_seconds']}s "
                      f"{row.get('error', '')[:120]}")


if __name__ == "__main__":
    main()
