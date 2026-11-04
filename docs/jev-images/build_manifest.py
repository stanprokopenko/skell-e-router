"""Select the drawing-camera evaluation frames and resize them into scratch.

Writes a manifest of frame IDs, task membership and Stan's existing pencil
labels. Subjective labels (hand obstruction, drawing stage, best still) are
added later by the labeler, before any model sees the frames. Images go to a
gitignored scratch folder and are never committed.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image

EXTRACT = Path("D:/solar-sailer/labeling/extract")
INDEX = Path("D:/solar-sailer/labeling/dataset/index.csv")
LABELS = Path("D:/solar-sailer/labeling/dataset/labels/tp-20260831-001")
LONG_SIDE = 1024

# Other-instructor frames for the hand-obstruction task: (folder, count).
HAND_EXTRA_SOURCES = [("bauman", 2), ("finch-pencil", 2), ("court-jones", 2), ("sanford", 2)]

# Drawing-stage clips taken from extract folders: first, median and last frame
# number available. Position in the clip is supporting evidence only.
STAGE_CLIPS = [
    "finch-pencil__interior-page-start-to-finish-part-4-drawing-6ca708",
    "sanford__3030-01030129-c002-553730",
    "eliza__1133-12250805-c003-00699711-762ab5",
]
# The one clip we have on disk: real start / middle / end via ffmpeg.
STAGE_VIDEO = Path("D:/solar-sailer/e2e/media/stroke-detection/1154_12130305_C001_V1-0001.mov")

# Best-still groups: four frames from one clip, at least 300 source frames apart.
BEST_STILL_CLIPS = [
    "court-jones__mvi-0012-ff263c",
    "solovev__camera-2-nest-f9a5cc",
    "elitha__1037-12032254-c001-5a9d28",
    "ryan-benjamin__1017-06061628-c001-406ec4",
    "blaise__blaise-gh5-007-bd5030",
]
MIN_GAP = 300


def clip_frames():
    """Map clip id -> sorted list of (frame number, frame id, jpg path)."""
    clips = defaultdict(list)
    for folder in EXTRACT.iterdir():
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            match = re.match(r"(.+)__f(\d+)$", entry.name)
            jpg = entry / "frame.jpg"
            if match and jpg.exists():
                clips[match.group(1)].append((int(match.group(2)), entry.name, jpg))
    for value in clips.values():
        value.sort()
    return clips


def resize(src: Path, dst: Path):
    with Image.open(src) as image:
        image = image.convert("RGB")
        image.thumbnail((LONG_SIDE, LONG_SIDE))
        size = image.size
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    data = buffer.getvalue()
    dst.write_bytes(data)
    return {"dimensions": list(size), "sha256": hashlib.sha256(data).hexdigest()}


def spaced_pick(frames, count, gap):
    """Pick `count` frames evenly across the clip, each at least `gap` apart."""
    first, last = frames[0][0], frames[-1][0]
    targets = [first + (last - first) * i / (count - 1) for i in range(count)]
    picked = []
    for target in targets:
        candidates = [f for f in frames if all(abs(f[0] - p[0]) >= gap for p in picked)]
        picked.append(min(candidates, key=lambda f: abs(f[0] - target)))
    picked.sort()
    assert all(b[0] - a[0] >= gap for a, b in zip(picked, picked[1:]))
    return picked


def probe_duration(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output path; the manifest is never overwritten.")
    args.scratch.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261104)
    frames = {}  # frame id -> record

    def add(frame_id, src, source, task, extra=None):
        record = frames.setdefault(frame_id, {"frame_id": frame_id, "source": source,
                                              "tasks": [], "image": str(args.scratch / f"{frame_id}.jpg")})
        if task not in record["tasks"]:
            record["tasks"].append(task)
        if extra:
            record.update(extra)
        if "sha256" not in record:
            record.update(resize(src, Path(record["image"])))

    # Task 4: every labeled pencil frame, with Stan's label.
    with INDEX.open(encoding="utf-8-sig", newline="") as handle:
        labeled = list(csv.DictReader(handle))
    by_state = defaultdict(list)
    for row in labeled:
        label = json.loads((LABELS / f"{row['frameId']}.StanStudio.json").read_text(encoding="utf-8"))
        add(row["frameId"], Path(row["localFrameJpg"]), "stroke-samples", "pencil_contact",
            {"pencil_label": label["state"], "source_frame": int(row["sourceFrame"])})
        by_state[label["state"]].append(row["frameId"])

    # Task 1: 12 labeled frames covering all four states, plus 8 from other folders.
    hand_ids = []
    for state, count in (("touching", 5), ("not-touching", 3), ("no-tip", 3), ("cant-tell", 1)):
        hand_ids += rng.sample(sorted(by_state[state]), count)
    for frame_id in hand_ids:
        frames[frame_id]["tasks"].append("hand_obstruction")
    clips = clip_frames()
    for folder, count in HAND_EXTRA_SOURCES:
        pool = sorted(e.name for e in (EXTRACT / folder).iterdir() if e.is_dir() and "proxy" not in e.name)
        for frame_id in rng.sample(pool, count):
            add(frame_id, EXTRACT / folder / frame_id / "frame.jpg", folder, "hand_obstruction")

    # Task 2: drawing stage.
    stage_groups = []
    for clip in STAGE_CLIPS:
        entries = clips[clip]
        picks = [entries[0], entries[len(entries) // 2], entries[-1]]
        group = {"clip": clip, "positions": ["first_extracted", "median_extracted", "last_extracted"], "frames": []}
        for position, (number, frame_id, jpg) in zip(group["positions"], picks):
            add(frame_id, jpg, clip.split("__")[0], "drawing_stage",
                {"source_frame": number, "clip": clip, "clip_position": position})
            group["frames"].append(frame_id)
        stage_groups.append(group)
    duration = probe_duration(STAGE_VIDEO)
    group = {"clip": STAGE_VIDEO.name, "positions": ["start_1s", "middle", "end_minus_1s"], "frames": [],
             "duration_seconds": duration}
    for position, seconds in zip(group["positions"], (1.0, duration / 2, duration - 1.0)):
        frame_id = f"stroke-samples__video__{position}"
        raw = args.scratch / f"{frame_id}.raw.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds:.3f}", "-i", str(STAGE_VIDEO),
                        "-frames:v", "1", "-q:v", "2", str(raw)], check=True)
        add(frame_id, raw, "stroke-samples", "drawing_stage",
            {"clip": STAGE_VIDEO.name, "clip_position": position, "seconds": round(seconds, 3)})
        raw.unlink()
        group["frames"].append(frame_id)
    stage_groups.append(group)

    # Task 3: best still.
    best_groups = []
    for clip in BEST_STILL_CLIPS:
        picks = spaced_pick(clips[clip], 4, MIN_GAP)
        group = {"clip": clip, "frames": []}
        for number, frame_id, jpg in picks:
            add(frame_id, jpg, clip.split("__")[0], "best_still", {"source_frame": number, "clip": clip})
            group["frames"].append(frame_id)
        best_groups.append(group)

    manifest = {
        "built": "2026-11-04",
        "long_side": LONG_SIDE,
        "hand_obstruction_frames": hand_ids + [f for f in frames if frames[f]["source"] != "stroke-samples"
                                               and "hand_obstruction" in frames[f]["tasks"]],
        "hand_obstruction_labeled_subset": hand_ids,
        "drawing_stage_groups": stage_groups,
        "best_still_groups": best_groups,
        "pencil_contact_frames": [row["frameId"] for row in labeled],
        "frames": frames,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(frames)} frames; hand {len(manifest['hand_obstruction_frames'])}, "
          f"stage {sum(len(g['frames']) for g in stage_groups)}, "
          f"best-still {sum(len(g['frames']) for g in best_groups)}, pencil {len(labeled)}")


if __name__ == "__main__":
    main()
