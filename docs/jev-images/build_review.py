"""Build the Jev image test review page for Stan.

Reads the 2026-11-04 manifest, blind labels, model runs and scored results,
and writes one self-contained HTML gallery (images embedded as base64) plus a
JSON map from review IDs (H-01, S-01, B-1a, P-01, I-1, ...) to frame ids.

Every number and answer on the page comes from the data files.

Run: python docs/jev-images/build_review.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = ROOT / ".agent-scratch" / "2026-11-04-jev-images"
OUT_DIR = SCRATCH / "review"
OUT_HTML = OUT_DIR / "2026-11-04-jev-review.html"
OUT_IDS = HERE / "2026-11-04-review-ids.json"
PENCIL_LABELS = Path("D:/solar-sailer/labeling/dataset/labels/tp-20260831-001")
IMPORT_FRAMES = SCRATCH / "import-frames-512"

LONG_SIDE = 800
QUALITY = 80

# (results.json models key, import requested_model, display name, chip name)
MODELS = [
    ("jev-1.13.0", "jev", "Jev 1.13", "Jev 1.13"),
    ("jev-1.14.0", "jev-latest", "Jev 1.14", "Jev 1.14"),
    ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite", "Gemini"),
]
GEMINI_TIP = {
    "Pencil tip touching the paper": "touching",
    "Pencil tip visible but not touching the paper": "not-touching",
    "No pencil tip visible": "no-tip",
}
STAGE_NAMES = ["blank", "first marks", "in progress", "finished"]
TIP_NAMES = {"touching": "touching", "not-touching": "not touching", "no-tip": "no tip", "cant-tell": "can't tell"}
LETTERS = "abcd"
POSITION_NAMES = {"first_extracted": "start", "median_extracted": "middle", "last_extracted": "end"}


def load_json(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def load_jsonl(name: str):
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(text) -> str:
    """Escape for HTML and strip dashes Stan's copy rules forbid."""
    text = str(text).replace(" \u2014 ", ", ").replace("\u2014", ", ").replace("\u2013", "-")
    return html.escape(text, quote=True)


def num(x: float, places: int = 2) -> str:
    return f"{x:.{places}f}"


def pct(x: float) -> str:
    return f"{round(x * 100)}%"


def source_name(clip: str) -> str:
    return clip.split("__")[0]


# ---------------------------------------------------------------- images

class Images:
    """Collects embedded images once each and hands back short keys."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.size: dict[str, tuple[int, int]] = {}
        self.by_path: dict[str, str] = {}

    def _add(self, path: Path, raw: bytes, size: tuple[int, int]) -> str:
        key = f"i{len(self.data)}"
        self.data[key] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        self.size[key] = size
        self.by_path[str(path)] = key
        return key

    def drawing(self, path: Path) -> str:
        if str(path) in self.by_path:
            return self.by_path[str(path)]
        with Image.open(path) as im:
            im = im.convert("RGB")
            scale = LONG_SIDE / max(im.size)
            if scale < 1:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=QUALITY, optimize=True)
            return self._add(path, buf.getvalue(), im.size)

    def as_is(self, path: Path) -> str:
        if str(path) in self.by_path:
            return self.by_path[str(path)]
        with Image.open(path) as im:
            size = im.size
        return self._add(path, path.read_bytes(), size)

    def tag(self, key: str, alt: str = "") -> str:
        w, h = self.size[key]
        return f'<img data-k="{key}" width="{w}" height="{h}" alt="{esc(alt)}">'


# ---------------------------------------------------------------- data

def load_all():
    manifest = load_json("2026-11-04-manifest.json")
    labels = load_json("2026-11-04-labels.json")
    results = load_json("2026-11-04-results.json")

    runs: dict[tuple[str, str, str], dict] = {}
    questions: dict[str, dict] = {}
    for row in load_jsonl("2026-11-04-drawing-runs.jsonl"):
        if row.get("error_code"):
            continue
        key = (row["task"], row["item_id"], row["model"])
        assert key not in runs, key
        runs[key] = row["answers"]
        questions.setdefault(row["task"], row["questions"])
    for row in load_jsonl("2026-11-04-drawing-gemini-runs.jsonl"):
        if row.get("error_code"):
            continue
        key = (row["task"], row["item_id"], row["model"])
        assert key not in runs, key
        runs[key] = row["answers"]

    imports = [r for r in load_jsonl("2026-11-04-import-runs-512.jsonl") if not r.get("error_code")]
    imports_384 = {r["fixture_id"]: r for r in load_jsonl("2026-11-04-import-runs.jsonl")
                   if not r.get("error_code") and r["requested_model"] == "gemini-3.1-flash-lite"}
    frames_index = json.loads((IMPORT_FRAMES / "frames.json").read_text(encoding="utf-8"))
    return manifest, labels, results, runs, questions, imports, imports_384, frames_index


def frame_path(manifest, frame_id: str) -> Path:
    p = Path(manifest["frames"][frame_id]["image"])
    return p if p.is_absolute() else ROOT / p


def stan_point(frame_id: str):
    path = PENCIL_LABELS / f"{frame_id}.StanStudio.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if not d.get("point"):
        return None
    return d["point"]["x"] / d["sourceWidth"], d["point"]["y"] / d["sourceHeight"]


# ---------------------------------------------------------------- model answers

def hand_answer(runs, frame_id, model):
    a = runs[("hand_obstruction", frame_id, model)]
    p = a["covered"]["noul"] if isinstance(a["covered"], dict) else float(a["covered"])
    yes = p >= 0.5
    return ("yes" if yes else "no"), (p if yes else 1 - p)


def stage_answer(runs, frame_id, model):
    a = runs[("drawing_stage", frame_id, model)]["stage"]
    if isinstance(a, dict):
        probs = a["probabilities"]
        level = int(max(probs, key=probs.get))
        return level, f"({num(float(a['score']), 1)})"
    gem = runs[("drawing_stage", frame_id, model)]
    return int(round(float(a))), num(float(gem["confidence"]))


def best_answer(runs, clip, model):
    a = runs[("best_still", clip, model)]
    if isinstance(a["best"], dict):
        return a["best"]["choice"], float(a["best"]["confidence"])
    return a["best"], float(a["confidence"])


def tip_answer(runs, frame_id, model):
    a = runs[("pencil_contact", frame_id, model)]
    if isinstance(a["tip"], dict):
        return a["tip"]["choice"], float(a["tip"]["confidence"])
    return GEMINI_TIP[a["tip"]], float(a["confidence"])


def import_answer(row):
    a = row["answers"]
    if isinstance(a.get("type_code"), dict):
        tc = a["type_code"]
        ranked = sorted(tc["probabilities"].items(), key=lambda kv: -kv[1])
        runner = next((kv for kv in ranked if kv[0] != row["predicted_code"]), None)
        thumb = int(a["best_thumbnail"]["choice"].split("_")[1])
        return {"code": row["predicted_code"], "conf": float(tc["confidence"]), "runner": runner,
                "framing": a["framing"]["choice"], "thumb": thumb}
    alts = a.get("alternates") or []
    runner = (alts[0]["code"], float(alts[0]["confidence"])) if alts else None
    return {"code": row["predicted_code"], "conf": float(a["type_confidence"]), "runner": runner,
            "framing": a.get("framing"), "thumb": a.get("best_thumbnail_index")}


# ---------------------------------------------------------------- html pieces

def chip(name: str, answer: str, extra: str, disagree: bool) -> str:
    cls = "chip dis" if disagree else "chip"
    return (f'<div class="{cls}"><span class="cm">{esc(name)}</span>'
            f'<span class="ca">{esc(answer)} <span class="cc">{esc(extra)}</span></span></div>')


def chips(items) -> str:
    return '<div class="chips">' + "".join(chip(*i) for i in items) + "</div>"


def feedback(review_id: str, kind: str) -> str:
    return f'<div class="fb" data-id="{review_id}" data-kind="{kind}"></div>'


def label_row(prefix: str, value: str, note: str = "", extra: str = "") -> str:
    out = f'<div class="lab"><span class="lk">{esc(prefix)}</span> <b>{esc(value)}</b>'
    if extra:
        out += f' <span class="lx">{esc(extra)}</span>'
    out += "</div>"
    if note:
        out += f'<div class="note" title="Tap to expand">{esc(note)}</div>'
    return out


def section(sec_id, title, intro_html, n_dis, body, extra_class=""):
    return f"""
<section id="{sec_id}" class="sec {extra_class}">
  <div class="sech">
    <h2>{esc(title)}</h2>
    {intro_html}
    <label class="tog"><input type="checkbox" data-sec="{sec_id}"><span class="sw"></span>Show disagreements only ({n_dis})</label>
  </div>
  {body}
</section>"""


def strip_ticks(text: str, replacements: dict[str, str]) -> str:
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.replace("`", "")


# ---------------------------------------------------------------- build

def build():
    manifest, labels, results, runs, questions, imports, imports_384, frames_index = load_all()
    images = Images()
    ids: dict[str, dict] = {}
    model_keys = [m[0] for m in MODELS]
    chip_names = {m[0]: m[3] for m in MODELS}
    correct = {task: {m: 0 for m in model_keys} for task in ("hand", "stage", "best", "tip", "import")}

    # ---- hand obstruction
    hand_order = list(manifest["hand_obstruction_frames"])
    for g in manifest["drawing_stage_groups"]:
        hand_order += g["frames"]
    for g in manifest["best_still_groups"]:
        hand_order += g["frames"]
    assert len(hand_order) == len(set(hand_order)) == len(labels["hand_obstruction"]), len(hand_order)

    hand_cards, hand_dis = [], 0
    for i, fid in enumerate(hand_order, 1):
        rid = f"H-{i:02d}"
        ids[rid] = {"task": "hand_obstruction", "frame_id": fid}
        lab = labels["hand_obstruction"][fid]
        truth = "yes" if lab["covered"] else "no"
        items, dis = [], False
        for m in model_keys:
            ans, conf = hand_answer(runs, fid, m)
            bad = ans != truth
            dis |= bad
            correct["hand"][m] += not bad
            items.append((chip_names[m], ans, num(conf), bad))
        hand_dis += dis
        key = images.drawing(frame_path(manifest, fid))
        hand_cards.append(
            f'<article class="card" data-dis="{int(dis)}"><div class="cid">{rid}</div>'
            f'<div class="imgw">{images.tag(key)}</div>'
            f'{label_row("Label", truth, lab["note"])}{chips(items)}{feedback(rid, "yn")}</article>')

    # ---- drawing stage
    stage_rows, stage_dis, s_i = [], 0, 0
    for g in manifest["drawing_stage_groups"]:
        cards, row_dis = [], False
        first_id = f"S-{s_i + 1:02d}"
        for fid, pos in zip(g["frames"], g["positions"]):
            s_i += 1
            rid = f"S-{s_i:02d}"
            ids[rid] = {"task": "drawing_stage", "frame_id": fid, "clip": g["clip"]}
            lab = labels["drawing_stage"][fid]
            also = [STAGE_NAMES[x] for x in lab["acceptable"] if x != lab["stage"]]
            items, dis = [], False
            for m in model_keys:
                level, extra = stage_answer(runs, fid, m)
                bad = level != lab["stage"]
                dis |= bad
                correct["stage"][m] += not bad
                items.append((chip_names[m], STAGE_NAMES[level], extra, bad))
            stage_dis += dis
            row_dis |= dis
            key = images.drawing(frame_path(manifest, fid))
            cards.append(
                f'<article class="card" data-dis="{int(dis)}"><div class="cid">{rid}'
                f' <span class="pos">{esc(POSITION_NAMES.get(pos, pos))}</span></div>'
                f'<div class="imgw">{images.tag(key)}</div>'
                f'{label_row("Label", STAGE_NAMES[lab["stage"]], lab["note"], ("also ok: " + ", ".join(also)) if also else "")}'
                f'{chips(items)}{feedback(rid, "stage")}</article>')
        stage_rows.append(
            f'<div class="grp" data-dis="{int(row_dis)}"><h3>{first_id} to S-{s_i:02d}'
            f' <span class="src">{esc(source_name(g["clip"]))}, in time order</span></h3>'
            f'<div class="row3">{"".join(cards)}</div></div>')

    # ---- best still
    best_by_clip = {b["clip"]: b for b in labels["best_still"]}
    best_cards, best_dis = [], 0
    for gi, g in enumerate(manifest["best_still_groups"], 1):
        rid = f"B-{gi}"
        letter_of = {fid: LETTERS[k] for k, fid in enumerate(g["frames"])}
        ids[rid] = {"task": "best_still", "clip": g["clip"], "frame_ids": g["frames"]}
        for fid in g["frames"]:
            ids[f"{rid}{letter_of[fid]}"] = {"task": "best_still", "clip": g["clip"], "frame_id": fid}
        lab = best_by_clip[g["clip"]]
        truth = letter_of[lab["best"]]
        also = [letter_of[f] for f in lab["acceptable"] if f != lab["best"]]
        # the rater's note names frames by number; swap in review IDs
        frame_num = {fid.split("__")[-1]: f"{rid}{letter_of[fid]}" for fid in g["frames"]}
        note = re.sub(r"f\d{7}", lambda mt: frame_num.get(mt.group(0), mt.group(0)), lab["note"])
        items, dis = [], False
        for m in model_keys:
            choice, conf = best_answer(runs, g["clip"], m)
            letter = LETTERS[int(choice.split("_")[1]) - 1]
            bad = letter != truth
            dis |= bad
            correct["best"][m] += not bad
            extra = num(conf) + (" (also ok)" if bad and letter in also else "")
            items.append((chip_names[m], letter, extra, bad))
        best_dis += dis
        figs = []
        for fid in g["frames"]:
            key = images.drawing(frame_path(manifest, fid))
            badge = '<span class="pick">Label</span>' if fid == lab["best"] else ""
            figs.append(f'<figure class="imgw">{images.tag(key)}<span class="ov">{rid}{letter_of[fid]}</span>{badge}</figure>')
        best_cards.append(
            f'<article class="card wide" data-dis="{int(dis)}"><div class="cid">{rid}'
            f' <span class="src">{esc(source_name(g["clip"]))}</span></div>'
            f'<div class="row4">{"".join(figs)}</div>'
            f'{label_row("Label", truth, note, ("also ok: " + ", ".join(also)) if also else "")}'
            f'{chips(items)}{feedback(rid, "abcd")}</article>')

    # ---- pencil contact
    pencil_cards, pencil_dis = [], 0
    for i, fid in enumerate(manifest["pencil_contact_frames"], 1):
        rid = f"P-{i:02d}"
        ids[rid] = {"task": "pencil_contact", "frame_id": fid}
        truth = manifest["frames"][fid]["pencil_label"]
        scored = truth != "cant-tell"
        items, dis = [], False
        for m in model_keys:
            choice, conf = tip_answer(runs, fid, m)
            bad = scored and choice != truth
            dis |= bad
            correct["tip"][m] += scored and not bad
            items.append((chip_names[m], TIP_NAMES[choice], num(conf), bad))
        pencil_dis += dis
        key = images.drawing(frame_path(manifest, fid))
        pt = stan_point(fid)
        ring = (f'<span class="ring" style="left:{pt[0] * 100:.2f}%;top:{pt[1] * 100:.2f}%"></span>'
                if pt else "")
        lab_html = label_row("Stan", TIP_NAMES[truth], extra="" if scored else "(not scored)")
        pencil_cards.append(
            f'<article class="card" data-dis="{int(dis)}"><div class="cid">{rid}</div>'
            f'<div class="imgw">{images.tag(key)}{ring}</div>'
            f'{lab_html}{chips(items)}{feedback(rid, "tip")}</article>')

    # ---- import
    imp_by = {}
    fixtures = []
    for r in imports:
        if r["fixture_id"] not in imp_by:
            imp_by[r["fixture_id"]] = {}
            fixtures.append(r["fixture_id"])
        imp_by[r["fixture_id"]][r["requested_model"]] = r
    imp_model_key = {m[1]: m[0] for m in MODELS}
    import_cards, import_dis = [], 0
    for i, fx in enumerate(fixtures, 1):
        rid = f"I-{i}"
        ids[rid] = {"task": "import", "clip": fx}
        rows = imp_by[fx]
        any_row = next(iter(rows.values()))
        true_code = any_row["true_code"]
        cols = [(name, import_answer(rows[req])) for _, req, name, _ in MODELS]
        for req, row in rows.items():
            correct["import"][imp_model_key[req]] += row["predicted_code"] == true_code
        cols.append(("Gemini at 384px", import_answer(imports_384[fx])))
        dis = any(c[1]["code"] != true_code for c in cols)
        import_dis += dis
        thumbs = []
        stamps = frames_index[fx]["timestamps"]
        for k in range(len(stamps)):
            key = images.as_is(IMPORT_FRAMES / f"{fx}__{k:02d}.jpg")
            t = int(stamps[k])
            thumbs.append(f'<figure class="imgw">{images.tag(key)}<span class="ov">{k + 1}</span>'
                          f'<span class="ts">{t // 60}:{t % 60:02d}</span></figure>')
        col_html = []
        for name, a in cols:
            bad = a["code"] != true_code
            runner = f'next {a["runner"][0]} {num(a["runner"][1])}' if a["runner"] else "next: none"
            thumb = f'thumbnail {a["thumb"] + 1}' if a["thumb"] is not None else ""
            col_html.append(
                f'<div class="icol{" dis" if bad else ""}"><span class="cm">{esc(name)}</span>'
                f'<span class="ic">{esc(a["code"])} <span class="cc">{num(a["conf"])}</span></span>'
                f'<span class="cs">{esc(runner)}</span><span class="cs">{esc(thumb)}</span></div>')
        import_cards.append(
            f'<article class="card wide" data-dis="{int(dis)}"><div class="cid">{rid}'
            f' <span class="src">{esc(fx)}</span></div>'
            f'<div class="strip">{"".join(thumbs)}</div>'
            f'{label_row("Benchmark", true_code, any_row["notes"])}'
            f'<div class="icols">{"".join(col_html)}</div>{feedback(rid, "code")}</article>')

    # ---- check our counts against the scored results file
    for m in model_keys:
        r = results["models"][m]
        hn = r["hand_obstruction"]["covered"]["n"]
        assert round(correct["hand"][m] / hn, 3) == r["hand_obstruction"]["covered"]["accuracy"], ("hand", m)
        assert round(correct["stage"][m] / r["drawing_stage"]["n"], 3) == r["drawing_stage"]["exact_accuracy"], ("stage", m)
        assert correct["best"][m] == r["best_still"]["exact_hits"], ("best", m)
        assert round(correct["tip"][m] / r["pencil_contact"]["n_scored"], 3) == r["pencil_contact"]["accuracy"], ("tip", m)
    for _, req, _, _ in MODELS:
        assert correct["import"][imp_model_key[req]] == results["import_512"][req]["strict_correct"], ("import", req)

    # ---- summary table
    def cell_vals(fn):
        return "".join(f"<td>{esc(fn(m))}</td>" for m in MODELS)

    M = results["models"]
    I512 = results["import_512"]
    first = model_keys[0]
    n_pencil = len(manifest["pencil_contact_frames"])
    rows_html = [
        ("Hand over the drawing", "#hand",
         cell_vals(lambda m: f'{pct(M[m[0]]["hand_obstruction"]["covered"]["accuracy"])} ({pct(M[m[0]]["hand_obstruction"]["covered"]["balanced_accuracy"])})'),
         f'{M[first]["hand_obstruction"]["covered"]["n"]}'),
        ("Drawing stage", "#stage",
         cell_vals(lambda m: pct(M[m[0]]["drawing_stage"]["exact_accuracy"])),
         f'{M[first]["drawing_stage"]["n"]}'),
        ("Best still", "#best",
         cell_vals(lambda m: f'{M[m[0]]["best_still"]["exact_hits"]} of {M[m[0]]["best_still"]["n"]}'
                             f' ({M[m[0]]["best_still"]["acceptable_hits"]} ok)'),
         f'{M[first]["best_still"]["n"]} sets of 4'),
        ("Pencil tip", "#pencil",
         cell_vals(lambda m: pct(M[m[0]]["pencil_contact"]["accuracy"])),
         f'{M[first]["pencil_contact"]["n_scored"]} of {n_pencil} scored'),
        ("Import shot type", "#import",
         cell_vals(lambda m: f'{I512[m[1]]["strict_correct"]} of {I512[m[1]]["n"]}'),
         f'{I512["jev"]["n"]} clips'),
    ]
    table = ('<table class="sum"><thead><tr><th>Test</th>' + "".join(f"<th>{esc(m[2])}</th>" for m in MODELS)
             + "<th>Frames</th></tr></thead><tbody>"
             + "".join(f'<tr><th><a href="{href}">{esc(name)}</a></th>{cells}<td class="mut">{esc(n)}</td></tr>'
                       for name, href, cells, n in rows_html)
             + '<tr class="speed"><th>Median answer time</th>'
             + cell_vals(lambda m: f'{num(M[m[0]]["usage"]["median_latency_s"])} s') + '<td class="mut"></td></tr>'
             + "</tbody></table>")
    g384 = results["import_384"]["gemini-3.1-flash-lite"]
    table_notes = (
        f'<p class="cap">Hand: share right, and in brackets the score with "yes" and "no" frames weighted equally. '
        f'Saying "yes" every time would score {pct(M[first]["hand_obstruction"]["always_yes_baseline_covered"])}. '
        f'Best still: exact hits, then how many landed in the acceptable set. '
        f'Imports: Gemini at the production 384px width got {g384["strict_correct"]} of {g384["n"]}.</p>')

    # ---- section intros (questions quoted from the run files)
    q = questions
    hand_q = strip_ticks(q["hand_obstruction"]["covered"]["instructions"], {"`frame`": "this frame"})
    stage_q = strip_ticks(q["drawing_stage"]["stage"]["instructions"], {"`frame`": "this frame"})
    best_q = strip_ticks(q["best_still"]["best"]["instructions"],
                         {"`frame_1`, `frame_2`, `frame_3` and `frame_4` are": "These four images are"})
    tip_q = strip_ticks(q["pencil_contact"]["tip"]["instructions"], {"`frame`": "This frame"})
    imp_q_raw = any_row["questions"]["type_code"]
    imp_q = re.sub(r"`frame_0`.*?`frame_\d+` are", "Ten frames are", imp_q_raw["instructions"])
    code_list = "".join(f"<li><b>{esc(c)}</b> {esc(d)}</li>" for c, d in imp_q_raw["criteria"].items())
    level_list = "; ".join(f"{STAGE_NAMES[k]} = {c.split(':', 1)[1].strip()}"
                           for k, c in enumerate(q["drawing_stage"]["stage"]["criteria"]))
    rater_line = ("A separate Claude Opus agent labeled these frames before any model ran. "
                  "Its call is what \"right\" means here, and you can overrule it.")

    intro_hand = (f'<p>Question: "{esc(hand_q)}" {rater_line} '
                  f'Numbers show how sure each model was of its answer, 0 to 1.</p>')
    intro_stage = (f'<p>Question: "{esc(stage_q)}" Levels: {esc(level_list)}. {rater_line} '
                   f'Jev\'s number is its score on the 0 to 3 scale; Gemini\'s is how sure it was.</p>')
    intro_best = (f'<p>Question: "{esc(best_q)}" {rater_line} '
                  f'The frame marked "Label" is the rater\'s pick.</p>')
    intro_tip = (f'<p>Question: "{esc(tip_q)}" The answer key is your own labels from the stroke-detection set. '
                 f'The ring marks where you put the tip. Frames you marked "can\'t tell" are not scored.</p>')
    intro_import = (f'<p>Question: "{esc(imp_q)}" The answer key and its notes come from the Solar Sailer '
                    f'import benchmark file. Each model saw the ten frames at 512px wide; the last column is Gemini '
                    f'at the 384px width production uses.</p>'
                    f'<details><summary>What the 12 codes mean</summary><ul class="codes">{code_list}</ul></details>')

    body = "".join([
        section("hand", "Hand over the drawing (H-01 to H-52)", intro_hand, hand_dis,
                f'<div class="grid">{"".join(hand_cards)}</div>'),
        section("stage", "Drawing stage (S-01 to S-12)", intro_stage, stage_dis, "".join(stage_rows)),
        section("best", "Best still (B-1 to B-5)", intro_best, best_dis,
                f'<div class="stack">{"".join(best_cards)}</div>'),
        section("pencil", "Pencil tip (P-01 to P-50)", intro_tip, pencil_dis,
                f'<div class="grid">{"".join(pencil_cards)}</div>'),
        section("import", "Import shot type (I-1 to I-7)", intro_import, import_dis,
                f'<div class="stack">{"".join(import_cards)}</div>'),
    ])

    options = {
        "yn": [["yes", "yes"], ["no", "no"]],
        "stage": [[str(k), f"{n} ({k})"] for k, n in enumerate(STAGE_NAMES)],
        "abcd": [[x, x] for x in LETTERS],
        "tip": [[k, v] for k, v in TIP_NAMES.items()],
        "code": [[c, c] for c in imp_q_raw["criteria"]],
    }

    page = TEMPLATE.replace("/*CSS*/", CSS).replace("<!--TABLE-->", table + table_notes) \
        .replace("<!--BODY-->", body).replace("/*OPTS*/", json.dumps(options)) \
        .replace("/*IMG*/", json.dumps(images.data))
    assert "\u2014" not in page

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    OUT_IDS.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.2f} MB, {len(images.data)} images)")
    print(f"wrote {OUT_IDS} ({len(ids)} ids)")
    print("disagreements:", {"hand": hand_dis, "stage": stage_dis, "best": best_dis,
                             "pencil": pencil_dis, "import": import_dis})


CSS = r"""
:root{--bg-app:#1c1c1c;--bg-panel:#242424;--bg-hover:rgba(247,247,247,.06);--bg-input:rgba(247,247,247,.06);
--bg-tag-accent:#3a2422;--bg-scrim-heavy:rgba(18,18,18,.6);--accent-primary:#e5473c;--text-primary:#e3e3e3;
--text-bright:#f7f7f7;--text-muted:#adadad;--text-disabled:#474747;--border-subtle:rgba(247,247,247,.06);
--shadow-card:rgba(0,0,0,.16) 0 4px 16px 0;--shadow-header:rgba(0,0,0,.08) 0 0 16px 0;
--radius-xs:4px;--radius-button:8px;--radius-chip:10px;--radius-card:12px;--radius-panel:16px;
--t-fast:.15s cubic-bezier(.4,0,.2,1);
--f-body:"Open Sans","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
--f-ui:Rubik,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
html{scroll-padding-top:64px}
body{margin:0;background:var(--bg-app);color:var(--text-primary);font:14px/20px var(--f-body);-webkit-text-size-adjust:100%}
a{color:var(--text-primary)}
.bar{position:sticky;top:0;z-index:9999;background:var(--bg-panel);box-shadow:var(--shadow-header);border-bottom:1px solid var(--border-subtle)}
.bar-in{max-width:1280px;margin:0 auto;height:52px;display:flex;align-items:center;gap:8px;padding:0 12px}
.bar b{font:500 14px/16px var(--f-ui);text-transform:uppercase;white-space:nowrap}
.bar nav{display:flex;gap:2px;margin-left:4px;overflow-x:auto}
.bar nav a{font:500 14px/16px var(--f-ui);color:var(--text-muted);text-decoration:none;padding:8px 8px;border-radius:var(--radius-button)}
.bar nav a:hover{background:var(--bg-hover);color:var(--text-primary)}
.bar .cnt{margin-left:auto;font-size:12px;color:var(--text-muted);white-space:nowrap}
@media(max-width:480px){.bar b{display:none}.bar nav{margin-left:-4px}}
.btn{font:500 14px/16px var(--f-ui);text-transform:uppercase;border:0;border-radius:var(--radius-button);height:36px;padding:0 12px;cursor:pointer;background:var(--text-bright);color:#1c1c1c;white-space:nowrap}
main{max-width:1280px;margin:0 auto;padding:16px 12px 48px}
h1{font:400 24px/32px var(--f-ui);margin:0 0 8px}
h2{font:500 16px/24px var(--f-ui);text-transform:uppercase;margin:0 0 4px;letter-spacing:.02em}
h3{font:500 14px/20px var(--f-ui);margin:0 0 8px;text-transform:uppercase}
p{margin:0 0 8px}
.top{background:var(--bg-panel);border-radius:var(--radius-panel);padding:16px;box-shadow:var(--shadow-card)}
.top textarea{width:100%;height:64px;background:var(--bg-input);color:var(--text-primary);border:0;border-radius:var(--radius-button);padding:8px;font:12px/16px ui-monospace,Consolas,monospace;resize:vertical;margin-bottom:12px}
.tw{overflow-x:auto}
table.sum{border-collapse:collapse;width:100%}
.sum th,.sum td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--border-subtle);font-weight:400;vertical-align:top}
@media(max-width:600px){.sum th,.sum td{padding:6px 3px}.sum thead th{font-size:12px}.sum th:first-child,.sum td:first-child{padding-left:0}}
.sum thead th{font:500 12px/16px var(--f-ui);text-transform:uppercase;color:var(--text-muted)}
.sum tbody th a{text-decoration:none;font-weight:600}
.sum .speed th,.sum .speed td{color:var(--text-muted)}
.mut,.cap{color:var(--text-muted)}
.cap{font-size:12px;line-height:16px;margin:8px 0 0}
.sec{margin-top:32px}
.sech{margin-bottom:12px}
.sech p{color:var(--text-muted);max-width:920px}
details{color:var(--text-muted);margin-bottom:8px}
summary{cursor:pointer;color:var(--text-primary)}
.codes{margin:8px 0 0;padding-left:18px;font-size:12px;line-height:18px}
.codes b{color:var(--text-primary)}
.tog{display:inline-flex;align-items:center;gap:8px;cursor:pointer;user-select:none;font-size:14px}
.tog input{position:absolute;opacity:0}
.tog .sw{width:32px;height:16px;border-radius:9999px;background:var(--text-disabled);position:relative;transition:all var(--t-fast)}
.tog .sw:after{content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;background:var(--text-bright);transition:all var(--t-fast)}
.tog input:checked+.sw{background:var(--text-muted)}
.tog input:checked+.sw:after{left:18px}
.only-dis [data-dis="0"]{display:none!important}
.grid{display:grid;gap:12px;grid-template-columns:1fr}
.stack{display:grid;gap:12px}
.grp{margin-bottom:16px}
.row3{display:grid;gap:12px;grid-template-columns:1fr}
.row4{display:grid;gap:4px;grid-template-columns:1fr 1fr}
.strip{display:grid;gap:4px;grid-template-columns:repeat(5,1fr)}
@media(min-width:700px){.grid{grid-template-columns:repeat(2,1fr)}.row3{grid-template-columns:repeat(3,1fr)}.row4{grid-template-columns:repeat(4,1fr)}}
@media(min-width:960px){.grid{grid-template-columns:repeat(3,1fr)}}
@media(min-width:1180px){.grid{grid-template-columns:repeat(4,1fr)}.strip{grid-template-columns:repeat(10,1fr)}}
.card{background:var(--bg-panel);border-radius:var(--radius-card);padding:8px;box-shadow:var(--shadow-card);display:flex;flex-direction:column;gap:8px;min-width:0}
.cid{font:600 20px/24px var(--f-ui);color:var(--text-bright);display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.cid .src,.cid .pos,h3 .src{font:400 12px/16px var(--f-body);color:var(--text-muted);text-transform:none}
.imgw{position:relative;margin:0;border-radius:var(--radius-button);overflow:hidden;line-height:0;background:#111}
.imgw img{width:100%;height:auto;display:block}
.ov,.ts,.pick{position:absolute;font:600 12px/16px var(--f-body);background:var(--bg-scrim-heavy);color:var(--text-bright);border-radius:var(--radius-xs);padding:2px 6px}
.ov{left:4px;top:4px}.ts{right:4px;bottom:4px}.pick{right:4px;top:4px;background:var(--text-bright);color:#1c1c1c}
.strip .ov,.strip .ts{padding:0 4px}
.ring{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 0 1.5px rgba(0,0,0,.85),inset 0 0 0 1.5px rgba(0,0,0,.85);pointer-events:none}
.lab{font-size:14px;line-height:20px}
.lk{color:var(--text-muted);font-size:12px}
.lab b{color:var(--text-bright)}
.lx{color:var(--text-muted);font-size:12px}
.note{font-size:12px;line-height:16px;color:var(--text-muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer;min-height:32px;margin-top:-6px}
.note.open{display:block}
.chips{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.chip,.icol{background:var(--bg-hover);border-radius:var(--radius-chip);padding:4px 8px;min-height:44px;display:flex;flex-direction:column;justify-content:center;min-width:0}
.cm{font-size:12px;line-height:16px;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ca{font-size:14px;line-height:18px;color:var(--text-primary)}
.cc{font-size:12px;color:var(--text-muted)}
.chip.dis,.icol.dis{background:var(--bg-tag-accent)}
.chip.dis .ca,.chip.dis .cm,.icol.dis .ic,.icol.dis .cm{color:var(--accent-primary)}
.icols{display:grid;grid-template-columns:1fr 1fr;gap:4px}
@media(min-width:700px){.icols{grid-template-columns:repeat(4,1fr)}}
.ic{font:600 16px/22px var(--f-ui);color:var(--text-primary)}
.cs{font-size:12px;line-height:16px;color:var(--text-muted)}
.fb{display:flex;flex-direction:column;gap:6px;margin-top:auto}
.fbb{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.fbb button{height:36px;border:0;border-radius:var(--radius-chip);background:var(--bg-hover);color:var(--text-muted);font:400 14px/16px var(--f-body);cursor:pointer;transition:all var(--t-fast)}
.fbb button.on{background:var(--accent-primary);color:var(--text-bright)}
.fbw{display:none;grid-template-columns:auto 1fr;gap:4px}
.fb.wrong .fbw{display:grid}
.fbw select,.fbw input{height:36px;background:var(--bg-input);color:var(--text-primary);border:0;border-radius:var(--radius-button);padding:0 8px;font:14px var(--f-body);min-width:0}
.fbw select{color-scheme:dark}
.fbw select.set{outline:2px solid var(--accent-primary)}
.card:has(.fb.wrong){box-shadow:var(--shadow-card),inset 0 0 0 1px var(--accent-primary)}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev image test: review</title>
<style>/*CSS*/</style></head>
<body>
<header class="bar"><div class="bar-in"><b>Jev review</b>
<nav><a href="#hand">H</a><a href="#stage">S</a><a href="#best">B</a><a href="#pencil">P</a><a href="#import">I</a></nav>
<span class="cnt" id="cnt">0 notes</span><button class="btn" id="copy" type="button">Copy feedback</button></div></header>
<main>
<div class="top">
<h1>Jev image test: review</h1>
<p>Tap "Label wrong" on any item you disagree with, then tap "Copy feedback" and paste it into the thread.</p>
<textarea id="out" readonly placeholder="Your feedback shows up here."></textarea>
<div class="tw"><!--TABLE--></div>
</div>
<!--BODY-->
</main>
<script>
const OPTS=/*OPTS*/;
const IMG=/*IMG*/;
const KEY="jev-review-2026-11-04:";
const store={get(id){try{return JSON.parse(localStorage.getItem(KEY+id))||{}}catch(e){return {}}},
 set(id,v){try{localStorage.setItem(KEY+id,JSON.stringify(v))}catch(e){}}};
document.querySelectorAll("img[data-k]").forEach(i=>{i.src=IMG[i.dataset.k]});
document.querySelectorAll(".note").forEach(n=>n.addEventListener("click",()=>n.classList.toggle("open")));
document.querySelectorAll(".tog input").forEach(t=>t.addEventListener("change",()=>{
 document.getElementById(t.dataset.sec).classList.toggle("only-dis",t.checked)}));
const fbs=[...document.querySelectorAll(".fb")];
function optName(kind,v){const o=OPTS[kind].find(x=>x[0]===v);return o?o[1]:v}
function render(fb){const s=store.get(fb.dataset.id);
 fb.classList.toggle("wrong",s.v==="wrong");
 fb.querySelector('[data-v="right"]').classList.toggle("on",s.v==="right");
 fb.querySelector('[data-v="wrong"]').classList.toggle("on",s.v==="wrong");
 const sel=fb.querySelector("select");sel.value=s.c||"";sel.classList.toggle("set",!!s.c);
 fb.querySelector("input").value=s.n||"";}
function text(){const lines=[];
 for(const fb of fbs){const id=fb.dataset.id,s=store.get(id);if(s.v!=="wrong")continue;
  const note=(s.n||"").trim();
  let l=s.c?`${id}: label should be ${optName(fb.dataset.kind,s.c)}.`:`${id}: label is wrong.`;
  if(note)l+=" "+note;lines.push(l)}
 return lines}
function refresh(){const lines=text();document.getElementById("out").value=lines.join("\\n");
 document.getElementById("cnt").textContent=lines.length+(lines.length===1?" note":" notes")}
for(const fb of fbs){const id=fb.dataset.id,kind=fb.dataset.kind;
 fb.innerHTML='<div class="fbb"><button type="button" data-v="right">Label right</button><button type="button" data-v="wrong">Label wrong</button></div>'
  +'<div class="fbw"><select aria-label="Correct answer"><option value="">Should be...</option>'
  +OPTS[kind].map(o=>`<option value="${o[0]}">${o[1]}</option>`).join("")
  +'</select><input type="text" placeholder="Note (optional)" aria-label="Note"></div>';
 fb.querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{const s=store.get(id);
  s.v=s.v===b.dataset.v?"":b.dataset.v;store.set(id,s);render(fb);refresh()}));
 fb.querySelector("select").addEventListener("change",e=>{const s=store.get(id);s.c=e.target.value;store.set(id,s);render(fb);refresh()});
 fb.querySelector("input").addEventListener("input",e=>{const s=store.get(id);s.n=e.target.value;store.set(id,s);refresh()});
 render(fb)}
refresh();
document.getElementById("copy").addEventListener("click",async()=>{const b=document.getElementById("copy"),
 out=document.getElementById("out"),t=out.value;
 if(!t){b.textContent="Nothing yet";setTimeout(()=>b.textContent="Copy feedback",1500);return}
 let ok=false;try{await navigator.clipboard.writeText(t);ok=true}catch(e){}
 if(!ok){out.focus();out.select();try{ok=document.execCommand("copy")}catch(e){}}
 b.textContent=ok?"Copied":"Select text below";setTimeout(()=>b.textContent="Copy feedback",1500)});
</script>
</body></html>
"""

if __name__ == "__main__":
    build()
