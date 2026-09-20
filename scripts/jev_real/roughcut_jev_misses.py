"""Where the Jev rough cut loses points against the human editor.

One arm is put under the microscope: ``jev_a`` at trim trigger 0.3 with the um
removal and delete silence layers on, which is the column the published ladder
compares every arm on. ``jev_b_notrim`` (same scores, no partial trims) rides
along so the cost of trimming is visible next to it.

No model calls, no detector runs, $0. The decisions file
``scripts/jev_real/roughcut_jev.py`` wrote and the cached removal ranges are
the only inputs; the arms are rebuilt with ``roughcut_jev_report``'s own
helpers and scored through ``roughcut_partial_scoring``.

Writes ``docs/jev-real/<out>-misses.md`` and ``docs/jev-real/<out>-misses.json``
and refuses to overwrite either.

Usage::

  python scripts/jev_real/roughcut_jev_misses.py --in roughcut-jev --out roughcut-jev
"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"
RESULTS_DIR = Path(r"D:\solar-sailer\benchmarks\roughcut\results")

sys.path.insert(0, str(HERE))

# Import order matters: the scoring module installs the cache-only answer-key
# loader and puts the harness on sys.path.
import roughcut_partial_scoring as scoring_mod  # noqa: E402
from roughcut_partial_scoring import (  # noqa: E402
    calibrate_threshold, load_episode, sentence_states_for,
)
import roughcut_jev_report as report_mod  # noqa: E402
import roughcut_jev as pipeline  # noqa: E402

ARM = "jev_a"
NOTRIM_ARM = "jev_b_notrim"
MODULE_ARM = "jev_b_moduleretakes"
T_TRIM = 0.3
SCORE_LEVELS = ["0", "1", "2", "3", "4", "5"]

#: Per-episode result files for the two ladder arms the brief asks for, found
#: by the glob ``bench-page-arms.json`` gives each arm id.
REFERENCE_GLOBS = {
    "luna-chapters-rules5":
        "*-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json",
    "opus5-cc-agentic": None,  # no per-sentence archive needed
}
LUNA_ARM = "luna-chapters-rules5"

#: Veto thresholds swept over ``retake_real``. ``None`` never vetoes.
VETO_SETTINGS = [("none", None), ("0.2", 0.2), ("0.3", 0.3), ("0.4", 0.4),
                 ("0.5 (production Jev)", 0.5), ("module flags", "module")]

KEPT_STATES = ("full", "partial")

#: Hand grouping of the two miss lists, read one sentence at a time with its
#: neighbours. Tied to the decision file this was written against: the
#: fingerprint in the output names it, and any miss the lists surface that is
#: not listed here is reported as unlabelled rather than silently dropped.
MISS_PATTERNS = {
    "jev_removed_editor_kept": [
        ("Half a sentence the transcript split",
         "The transcript breaks one spoken sentence into two rows and the second "
         "row trails off in `..`. Jev reads the fragment on its own, calls it an "
         "abandoned false start, and drops it. The editor kept the whole spoken "
         "sentence, so both rows are in the cut.",
         [("hampton-5.2-shape-demo", 6), ("colman-03.03-muscles-crit", 156),
          ("colman-02.04-skeleton-demo", 151), ("perspective-14e-boxes-critique", 176),
          ("edges-7.01-intro", 361), ("hampton-5.4-assignment-demo", 77),
          ("edges-7.01-intro", 28), ("colman-02.04-skeleton-demo", 6),
          ("colman-02.04-skeleton-demo", 127), ("colman-03.03-muscles-crit", 78),
          ("colman-03.03-muscles-crit", 155)]),
        ("Ordinary connective teaching talk",
         "Level 3 in the prompt's own words: fine, keeps the flow, nothing "
         "memorable. Jev put it at 1 or below. These are the demo lessons, where "
         "the editor keeps almost everything.",
         [("hampton-5.2-shape-demo", 5), ("hampton-5.4-assignment-demo", 109),
          ("colman-03.03-muscles-crit", 273), ("hampton-5.2-shape-demo", 20),
          ("colman-03.03-muscles-crit", 263), ("perspective-14e-boxes-critique", 887),
          ("colman-03.03-muscles-crit", 5), ("perspective-14e-boxes-critique", 464),
          ("colman-03.03-muscles-crit", 230)]),
        ("Retake pair, the wrong side cut",
         "Two attempts at one line. Jev found the pair but crowned the other take, "
         "so the one the editor actually used got cut as the loser.",
         [("edges-7.01-intro", 203), ("edges-7.01-intro", 384),
          ("hampton-5.2-shape-demo", 21), ("edges-7.01-intro", 210),
          ("hampton-5.2-shape-demo", 256), ("edges-7.01-intro", 102)]),
        ("Scripted lesson line read as filler",
         "The written intro asks a question or names a list, one short sentence at "
         "a time. Jev scores a short sentence with no content of its own as "
         "throat-clearing; in a script it is the lesson.",
         [("edges-7.01-intro", 82), ("edges-7.01-intro", 355),
          ("edges-7.01-intro", 277), ("edges-7.01-intro", 242),
          ("edges-7.01-intro", 241), ("edges-7.01-intro", 270)]),
        ("Producer talk and student names in a critique",
         "In a critique the teacher reads a name off the screen or says what the "
         "producer just put up. The prompt's level 0 names exactly that as a cut, "
         "and the editor keeps it, because it is how the critique moves.",
         [("perspective-14e-boxes-critique", 603),
          ("perspective-14e-boxes-critique", 908),
          ("perspective-14e-boxes-critique", 941),
          ("perspective-14e-boxes-critique", 191),
          ("perspective-14e-boxes-critique", 988)]),
        ("Thinking aloud, all ums",
         "A run of pure filler the editor kept anyway, because the pause is part "
         "of the demo's rhythm.",
         [("hampton-5.2-shape-demo", 104), ("hampton-5.2-shape-demo", 103)]),
        ("A joke the editor kept",
         "Self-deprecating aside. The prompt asks for it at level 4 and Jev scored "
         "it at 1.",
         [("hampton-5.4-assignment-demo", 166)]),
    ],
    "jev_kept_editor_removed": [
        ("Teaching line the picture already makes",
         "Real content, said clearly, and the editor still cut it: the drawing on "
         "screen says the same thing, or the point was already made a sentence "
         "earlier. Jev has no way to see the picture and grades the words.",
         [("edges-7.01-intro", 228), ("perspective-14e-boxes-critique", 580),
          ("colman-02.04-skeleton-demo", 178), ("colman-03.03-muscles-crit", 287),
          ("perspective-14e-boxes-critique", 400), ("colman-03.03-muscles-crit", 285),
          ("hampton-5.4-assignment-demo", 150),
          ("perspective-14e-boxes-critique", 583),
          ("perspective-14e-boxes-critique", 350), ("colman-03.03-muscles-crit", 30),
          ("colman-03.03-muscles-crit", 286), ("edges-7.01-intro", 189),
          ("edges-7.01-intro", 27), ("perspective-14e-boxes-critique", 354),
          ("colman-02.04-skeleton-demo", 11)]),
        ("Producer talk and studio logistics",
         "Two people arranging what happens next, or spelling a name out loud. Jev "
         "graded it 3 or above because it is fluent and on topic.",
         [("hampton-5.2-shape-demo", 297), ("perspective-14e-boxes-critique", 957),
          ("hampton-5.2-shape-demo", 298), ("perspective-14e-boxes-critique", 1111),
          ("perspective-14e-boxes-critique", 1138),
          ("perspective-14e-boxes-critique", 1112),
          ("perspective-14e-boxes-critique", 112),
          ("perspective-14e-boxes-critique", 814)]),
        ("Encouragement and wrap-up the editor tightened",
         "The pep talk at the end of a lesson. Warm, well said, and the editor "
         "keeps one line of it and drops the rest.",
         [("perspective-14e-boxes-critique", 1113), ("hampton-5.2-shape-demo", 335),
          ("hampton-5.2-shape-demo", 337), ("hampton-5.2-shape-demo", 334),
          ("perspective-14e-boxes-critique", 816),
          ("perspective-14e-boxes-critique", 343)]),
        ("Half a sentence the transcript split",
         "The mirror of the same problem: the editor cut the whole spoken "
         "sentence, so the trailing fragment goes too, and Jev kept the fragment "
         "because the words it can see read as setup.",
         [("perspective-14e-boxes-critique", 716), ("colman-02.04-skeleton-demo", 10),
          ("hampton-5.2-shape-demo", 39), ("perspective-14e-boxes-critique", 590),
          ("hampton-5.2-shape-demo", 46)]),
        ("Short connective dropped for pacing",
         "`Right?`, `All right?`, `Okay.` Jev scores the tag high because it reads "
         "as ordinary teaching talk; the editor cuts it to keep the cut moving.",
         [("colman-03.03-muscles-crit", 222), ("colman-03.03-muscles-crit", 297),
          ("perspective-14e-boxes-critique", 139)]),
        ("Near-duplicate line, the loser kept",
         "The next sentence says the same thing better and the retake detector "
         "never flagged the pair, so nothing cut it.",
         [("edges-7.01-intro", 261)]),
        ("Operating the drawing or the screen",
         "Level 0 in the prompt. Jev gave it 3.15.",
         [("hampton-5.2-shape-demo", 107)]),
        ("Tangent the editor cut short",
         "The teacher loses the thread and the editor takes the detour out.",
         [("hampton-5.4-assignment-demo", 169)]),
    ],
}


def pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}"


def num(value, places=2):
    return "n/a" if value is None else f"{value:.{places}f}"


def table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return out


def clip(text, limit=260):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ---------------------------------------------------------------------------
# episode text
# ---------------------------------------------------------------------------

def episode_text(name):
    """Sentence texts, corpus order and the word list each SP run indexes into.

    ``unit_words`` is the sentence's words in the order
    ``sentence_scoring.sentence_states`` counts them (``word_scoring.word_units``
    order), so a run ``(first, end)`` slices it directly.
    """
    data = load_episode(name)
    order = [s["id"] for s in data["sentences"]]
    word_text = {w["id"]: w.get("text", "")
                 for w in data["transcript"]["word_segments"]}
    unit_words = defaultdict(list)
    for unit in data["preflight"].word_units:
        unit_words[unit.sentence_id].append(word_text.get(unit.word_id, "?"))
    return {
        "order": order,
        "position": {sid: i for i, sid in enumerate(order)},
        "text": {s["id"]: s.get("text", "") for s in data["sentences"]},
        "unit_words": unit_words,
        "human_kept_frames": data["human_kept_frames"],
        "dialogue_frames": data["dialogue_frames"],
    }


def neighbour_text(texts, sid, step):
    position = texts["position"][sid] + step
    if position < 0 or position >= len(texts["order"]):
        return ""
    return texts["text"][texts["order"][position]]


def render_spans(words, model_runs, human_runs):
    """The sentence with the arm's kept span in ``[]`` and the editor's in ``{}``."""
    def edges(runs):
        return ({run[0] for run in runs or []}, {run[1] for run in runs or []})

    open_m, close_m = edges(model_runs)
    open_h, close_h = edges(human_runs)
    parts = []
    for index, word in enumerate(words):
        closing = ("]" if index in close_m else "") + ("}" if index in close_h else "")
        if closing and parts:
            parts[-1] += closing
        opening = ("[" if index in open_m else "") + ("{" if index in open_h else "")
        parts.append(opening + word)
    tail = (("]" if len(words) in close_m else "")
            + ("}" if len(words) in close_h else ""))
    if tail and parts:
        parts[-1] += tail
    return " ".join(parts)


# ---------------------------------------------------------------------------
# confusion
# ---------------------------------------------------------------------------

def confusion(human, model):
    """The four buckets the brief asks for, plus the partial split."""
    counts = Counter()
    for sid, (human_kind, _runs) in human.items():
        model_kind = model.get(sid, ("removed", None))[0]
        human_kept = human_kind in KEPT_STATES
        model_kept = model_kind in KEPT_STATES
        if human_kept and model_kept:
            counts["both_keep"] += 1
        elif model_kept and not human_kept:
            counts["jev_kept_editor_removed"] += 1
        elif human_kept and not model_kept:
            counts["jev_removed_editor_kept"] += 1
            counts[f"jev_removed_editor_{human_kind}"] += 1
        else:
            counts["both_removed"] += 1
    counts["n"] = len(human)
    return dict(counts)


def accuracy(counts):
    right = counts.get("both_keep", 0) + counts.get("both_removed", 0)
    return right / counts["n"] if counts.get("n") else None


# ---------------------------------------------------------------------------
# retake veto rebuild
# ---------------------------------------------------------------------------

def retake_groups(rows, position):
    """``{group_id: [sentence id, ...]}`` in spoken order, from the stored rows."""
    groups = defaultdict(list)
    for sid, row in rows.items():
        gid = row.get("retake_group")
        if gid is not None:
            groups[gid].append(sid)
    for members in groups.values():
        members.sort(key=lambda sid: position[sid])
    return groups


def jev_winner(row, members):
    """The take Jev's own rule picks, ignoring the real-retake veto.

    Rebuilt from ``retake_choice`` and ``retake_take_probs`` with the pipeline's
    last-take margin, so it is the winner the run would have used at any veto
    threshold. ``None`` means the run never got an answer for the group.
    """
    probs = row.get("retake_take_probs")
    if not probs:
        return None
    keys = [f"take{i + 1}" for i in range(len(members))]
    top = row.get("retake_choice")
    if top not in probs or top not in keys:
        return None
    if probs[top] - probs.get(keys[-1], 0.0) < pipeline.LAST_TAKE_MARGIN:
        top = keys[-1]
    return members[keys.index(top)]


def cut_set_for_veto(rows, position, veto):
    """Sentence ids cut as retake losers at one veto threshold.

    ``veto`` is ``None`` (never veto: always cut the losers of Jev's chosen
    winner) or a float on ``retake_real``. A group the run got no answer for
    keeps the module-winner fallback the pipeline applied, at every setting.
    """
    cut = set()
    for members in retake_groups(rows, position).values():
        row = rows[members[0]]
        winner = jev_winner(row, members)
        if winner is None:
            fallback = row.get("retake_winner")
            if row.get("retake_source") == "module_fallback" and fallback is not None:
                cut.update(m for m in members if m != fallback)
            continue
        if veto is not None:
            real = row.get("retake_real")
            if real is None or real < veto:
                continue
        cut.update(m for m in members if m != winner)
    return cut


# ---------------------------------------------------------------------------
# Luna chapters archive
# ---------------------------------------------------------------------------

def luna_files():
    """``{episode: path}`` for the archived Luna chapters per-episode results."""
    found = {}
    glob = REFERENCE_GLOBS[LUNA_ARM]
    if not RESULTS_DIR.is_dir():
        return found
    for path in sorted(RESULTS_DIR.glob(glob)):
        try:
            with path.open(encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, ValueError):
            continue
        for episode in doc.get("episodes", {}):
            found.setdefault(episode, path)
    return found


def luna_states(episode, path, removals):
    """Luna's human and model states, its own threshold, the same layers on."""
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    saved = doc["episodes"][episode]
    ratings = (saved.get("run_ratings") or [None])[0]
    if not ratings:
        return None, None, None
    decisions = scoring_mod.decisions_from_run_ratings(episode, ratings)
    threshold = float(doc["neutral_threshold"])
    human, model = sentence_states_for(episode, decisions, threshold, removals)
    return human, model, threshold


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------

def fingerprint(paths):
    """md5 and mtime of each input, so a rerun can prove it read the same run.

    The decisions file is rewritten whenever the pipeline patches answers it
    missed, and the numbers move when it does; without this the reader cannot
    tell which version a table came from.
    """
    out = {}
    for key, value in paths.items():
        path = Path(value)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        out[key] = {"path": value, "md5": digest,
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc).isoformat(
                            timespec="seconds")}
    return out


def build(args):
    decision_rows, _requests, _timing, input_paths = report_mod.load_run(args.in_name)
    by_arm, episodes = report_mod.index_decisions(decision_rows)
    for arm in (ARM, NOTRIM_ARM, MODULE_ARM):
        if arm not in by_arm:
            raise SystemExit(f"{args.in_name}-decisions.jsonl has no rows for {arm!r}")

    words = {episode: report_mod.word_ids_by_sentence(episode) for episode in episodes}
    removals, skipped = report_mod.load_removals(episodes)
    if skipped:
        raise SystemExit(f"no cached removals for {skipped}; the layered column is "
                         f"the whole point of this run. Fill the cache with "
                         f"roughcut_partial_scoring.py --removals")
    texts = {episode: episode_text(episode) for episode in episodes}

    decisions, n_missing = report_mod.build_arm_decisions(
        by_arm, episodes, words, ARM, T_TRIM, 0.0)
    notrim, _ = report_mod.build_arm_decisions(
        by_arm, episodes, words, NOTRIM_ARM, None, 0.0)

    print("scoring jev_a t0.3 layered...", file=sys.stderr)
    result = calibrate_threshold(decisions, removals=removals)
    threshold = result["threshold"]
    print(f"  threshold {threshold}", file=sys.stderr)
    print("scoring jev_b_notrim layered...", file=sys.stderr)
    notrim_result = calibrate_threshold(notrim, removals=removals)

    states = {episode: sentence_states_for(episode, decisions[episode], threshold,
                                           removals[episode])
              for episode in episodes}

    # Self-check: the states this script reads must be the ones the metric
    # scored. Recount the cross-tab off them and compare to the scored block.
    state_check = []
    for episode in episodes:
        human, model = states[episode]
        recount = Counter()
        for sid, (human_kind, _runs) in human.items():
            recount[f"{human_kind}/{model.get(sid, ('removed', None))[0]}"] += 1
        scored = result["episodes"][episode]["pair_counts"]
        if {k: v for k, v in scored.items() if v} != {k: v for k, v in recount.items() if v}:
            state_check.append(episode)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_jev_misses.py",
        "inputs": input_paths,
        "input_fingerprint": fingerprint(input_paths),
        "arm": f"{ARM} (t_trim {T_TRIM}) layered with um removal + delete silence",
        "threshold": threshold,
        "notrim_threshold": notrim_result["threshold"],
        "episodes": episodes,
        "missing_answers": n_missing,
        "state_check_failures": state_check,
        "pooled": result["pooled"],
        "notrim_pooled": notrim_result["pooled"],
    }

    summary["per_episode"] = per_episode_table(episodes, result, notrim_result, texts)
    summary["confusion"] = confusion_block(episodes, states, removals, by_arm)
    summary["misses"] = miss_lists(episodes, states, decisions, by_arm, texts)
    summary["trims"] = trim_block(episodes, result, notrim_result, states, decisions,
                                  texts)
    summary["veto"] = veto_sweep(episodes, by_arm, words, removals, texts,
                                 threshold)
    summary["veto_fixed_threshold"] = threshold
    summary["levels"] = level_block(episodes, states, by_arm)
    return summary


def per_episode_table(episodes, result, notrim_result, texts):
    """jev_a layered against the two ladder arms, per episode."""
    reference = {}
    path = RESULTS_DIR / "2026-09-11-model-plus-deterministic.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
        for key in ("luna-chapters-rules5", "opus5-cc-agentic"):
            arm = doc["arms"].get(key)
            if arm:
                reference[key] = arm["episodes"]

    rows = {}
    for episode in episodes:
        mine = result["episodes"][episode]
        row = {
            "sentence_points": mine["sentence_points"],
            "word_score": mine["word_score"],
            "grade": mine["grade"],
            "notrim_sentence_points":
                notrim_result["episodes"][episode]["sentence_points"],
            "human_kept_share": (texts[episode]["human_kept_frames"]
                                 / texts[episode]["dialogue_frames"]
                                 if texts[episode]["dialogue_frames"] else None),
        }
        for key, entry in reference.items():
            found = entry.get(episode)
            if not found:
                continue
            row[f"{key}_sentence_points"] = found["sp_grades"]["umm_silence"]
            row[f"{key}_word_score"] = found["word_grades"]["umm_silence"]
            row[f"{key}_grade"] = found["grades"]["umm_silence"]
        row["gap_vs_luna"] = (
            None if row.get("luna-chapters-rules5_sentence_points") is None
            else row["sentence_points"] - row["luna-chapters-rules5_sentence_points"])
        row["gap_vs_opus"] = (
            None if row.get("opus5-cc-agentic_sentence_points") is None
            else row["sentence_points"] - row["opus5-cc-agentic_sentence_points"])
        rows[episode] = row
    return rows


def confusion_block(episodes, states, removals, by_arm):
    per_episode, pooled = {}, Counter()
    for episode in episodes:
        human, model = states[episode]
        counts = confusion(human, model)
        per_episode[episode] = counts
        pooled.update(counts)

    luna = {"per_episode": {}, "pooled": {}, "note": ""}
    files = luna_files()
    missing = [e for e in episodes if e not in files]
    if missing:
        luna["note"] = (f"no archived per-sentence ratings found for {missing}; "
                        f"Luna's confusion is skipped")
    else:
        luna_pooled = Counter()
        for episode in episodes:
            human, model, threshold = luna_states(episode, files[episode],
                                                  removals[episode])
            if human is None:
                luna["note"] = f"{episode}: archived file has no run_ratings"
                luna_pooled = Counter()
                break
            counts = confusion(human, model)
            counts["threshold"] = threshold
            luna["per_episode"][episode] = counts
            luna_pooled.update({k: v for k, v in counts.items() if k != "threshold"})
        luna["pooled"] = dict(luna_pooled)
        if luna_pooled:
            luna["note"] = (f"from the archived `run_ratings` in "
                            f"`{files[episodes[0]].name}` and its siblings, each at "
                            f"its own file's Neutral threshold, with the same um "
                            f"removal and delete silence layers on")
    return {"per_episode": per_episode, "pooled": dict(pooled), "luna": luna}


def top_two(probabilities):
    if not probabilities:
        return ""
    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])[:2]
    return ", ".join(f"level {k} {v:.2f}" for k, v in ordered)


def miss_lists(episodes, states, decisions, by_arm, texts, limit=40):
    removed_kept, kept_removed = [], []
    for episode in episodes:
        human, model = states[episode]
        for sid, (human_kind, human_runs) in human.items():
            model_kind = model.get(sid, ("removed", None))[0]
            row = by_arm[ARM][episode][sid]
            entry = {
                "episode": episode,
                "id": sid,
                "score": decisions[episode][sid]["score"],
                "raw_score": row.get("score"),
                "probabilities": row.get("score_probabilities"),
                "top_two": top_two(row.get("score_probabilities")),
                "human_state": human_kind,
                "model_state": model_kind,
                "cut_retake": decisions[episode][sid]["cut_retake"],
                "retake_group": row.get("retake_group"),
                "retake_real": row.get("retake_real"),
                "text": texts[episode]["text"].get(sid, ""),
                "prev": neighbour_text(texts[episode], sid, -1),
                "next": neighbour_text(texts[episode], sid, 1),
            }
            if human_kind in KEPT_STATES and model_kind == "removed":
                removed_kept.append(entry)
            elif human_kind == "removed" and model_kind in KEPT_STATES:
                kept_removed.append(entry)
    removed_kept.sort(key=lambda e: (e["score"], e["episode"], e["id"]))
    kept_removed.sort(key=lambda e: (-e["score"], e["episode"], e["id"]))
    out = {
        "jev_removed_editor_kept": removed_kept[:limit],
        "jev_kept_editor_removed": kept_removed[:limit],
        "n_jev_removed_editor_kept": len(removed_kept),
        "n_jev_kept_editor_removed": len(kept_removed),
    }
    for key in ("jev_removed_editor_kept", "jev_kept_editor_removed"):
        out[f"{key}_patterns"] = group_patterns(out[key], key)
    return out


def group_patterns(entries, key):
    """Attach the hand grouping to one miss list and count it.

    Anything the grouping does not name comes back under ``unlabelled``, so a
    rerun on a different decision file cannot quietly present a stale reading.
    """
    index = {(entry["episode"], entry["id"]): entry for entry in entries}
    groups, claimed = [], set()
    for name, note, members in MISS_PATTERNS.get(key, []):
        found = [index[m] for m in members if m in index]
        claimed.update(m for m in members if m in index)
        for entry in found:
            entry["pattern"] = name
        groups.append({"pattern": name, "note": note, "n": len(found),
                       "members": [{"episode": e["episode"], "id": e["id"],
                                    "score": e["score"], "text": e["text"]}
                                   for e in found]})
    leftovers = [entry for entry in entries
                 if (entry["episode"], entry["id"]) not in claimed]
    for entry in leftovers:
        entry["pattern"] = "unlabelled"
    if leftovers:
        groups.append({"pattern": "unlabelled",
                       "note": "in the list but not in the hand grouping; the "
                               "decision file has moved since it was written",
                       "n": len(leftovers),
                       "members": [{"episode": e["episode"], "id": e["id"],
                                    "score": e["score"], "text": e["text"]}
                                   for e in leftovers]})
    groups.sort(key=lambda g: -g["n"])
    return groups


def trim_block(episodes, result, notrim_result, states, decisions, texts, examples=15):
    branch = Counter()
    pairs = Counter()
    emitted = {}
    for episode in episodes:
        row = result["episodes"][episode]
        branch.update(row["partial_branch_counts"] or {})
        pairs.update(row["pair_counts"] or {})
        emitted[episode] = sum(1 for d in decisions[episode].values()
                               if d["keep_words"])

    # Every sentence that reaches the metric as a partial, labelled by the
    # branch it took and by who made it partial: Jev's own ``keep_words``, or
    # the um removal / delete silence frames coming off a whole sentence.
    trims = []
    for episode in episodes:
        human, model = states[episode]
        for sid, (model_kind, model_runs) in model.items():
            if model_kind != "partial":
                continue
            human_kind, human_runs = human[sid]
            jev_trim = decisions[episode][sid]["keep_words"] is not None
            if human_kind == "partial":
                label = scoring_mod.sentence_scoring._partial_branch(human_runs,
                                                                     model_runs)
            elif human_kind == "full":
                label = "editor kept it whole (0.7)"
            else:
                label = "editor removed it (0.0)"
            trims.append({
                "episode": episode, "id": sid, "branch": label,
                "jev_trim": jev_trim,
                "score": decisions[episode][sid]["score"],
                "text": texts[episode]["text"].get(sid, ""),
                "spans": render_spans(texts[episode]["unit_words"].get(sid, []),
                                      model_runs,
                                      human_runs if human_kind == "partial"
                                      else ([(0, len(texts[episode]["unit_words"]
                                                     .get(sid, [])))]
                                            if human_kind == "full" else [])),
            })

    by_branch = defaultdict(list)
    for trim in trims:
        if trim["jev_trim"]:
            by_branch[trim["branch"]].append(trim)
    wanted = ["exact", "subset", "overlap", "disjoint",
              "editor kept it whole (0.7)", "editor removed it (0.0)"]
    picked, index = [], 0
    while len(picked) < examples:
        added = False
        for label in wanted:
            bucket = by_branch.get(label) or []
            if index < len(bucket):
                picked.append(bucket[index])
                added = True
            if len(picked) >= examples:
                break
        if not added:
            break
        index += 1

    jev_branch = Counter(t["branch"] for t in trims if t["jev_trim"])
    layer_branch = Counter(t["branch"] for t in trims if not t["jev_trim"])
    survived = sum(1 for t in trims if t["jev_trim"])

    return {
        "emitted": emitted,
        "emitted_total": sum(emitted.values()),
        "branch_counts": dict(branch),
        "pair_counts": dict(pairs),
        "model_partials_scored": len(trims),
        "jev_trims_scored": survived,
        "jev_trim_branches": dict(jev_branch),
        "layer_only_branches": dict(layer_branch),
        "examples": picked,
        "notrim_pooled_sentence_points":
            notrim_result["pooled"]["sentence_points"],
    }


def pooled_at(episodes, decisions, threshold, removals):
    """Pooled SP / WORD / GRADE at a threshold nobody recalibrated.

    Same weights ``pooling`` uses: sentence count for SENTENCE POINTS, word
    count for WORD SCORE, dialogue frames for GRADE. Lets the veto settings be
    compared without the calibrated threshold moving under them.
    """
    rows = [scoring_mod.score_episode(episode, decisions[episode], threshold,
                                      removals=removals[episode])
            for episode in episodes]

    def weighted(values, weights):
        pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w]
        return (sum(v * w for v, w in pairs) / sum(w for _v, w in pairs)
                if pairs else None)

    return {
        "sentence_points": weighted([r["sentence_points"] for r in rows],
                                    [r["n_sentences"] for r in rows]),
        "word_score": weighted([r["word_score"] for r in rows],
                               [r["word_count"] for r in rows]),
        "grade": weighted([r["grade"] for r in rows],
                          [r["dialogue_frames"] for r in rows]),
    }


def veto_sweep(episodes, by_arm, words, removals, texts, fixed_threshold):
    base, _missing = report_mod.build_arm_decisions(
        by_arm, episodes, words, ARM, T_TRIM, 0.0)
    module_cuts = {episode: {sid for sid, row in by_arm[MODULE_ARM][episode].items()
                             if row["cut_retake"]}
                   for episode in episodes}

    out = []
    for label, veto in VETO_SETTINGS:
        decisions, cuts = {}, {}
        for episode in episodes:
            if veto == "module":
                cut = module_cuts[episode]
            else:
                cut = cut_set_for_veto(by_arm[ARM][episode],
                                       texts[episode]["position"], veto)
            cuts[episode] = cut
            decisions[episode] = {
                sid: {**decision, "cut_retake": sid in cut}
                for sid, decision in base[episode].items()}

        matches_run = all(
            cuts[episode] == {sid for sid, d in base[episode].items() if d["cut_retake"]}
            for episode in episodes)
        print(f"scoring veto {label}...", file=sys.stderr)
        result = calibrate_threshold(decisions, removals=removals)
        threshold = result["threshold"]

        cut_verdict = Counter()
        counts = Counter()
        for episode in episodes:
            human, model = sentence_states_for(episode, decisions[episode], threshold,
                                               removals[episode])
            counts.update(confusion(human, model))
            for sid in cuts[episode]:
                cut_verdict[human[sid][0]] += 1

        out.append({
            "label": label,
            "fixed": pooled_at(episodes, decisions, fixed_threshold, removals),
            "veto": veto if veto != "module" else "production is_retake flags",
            "reproduces_the_run": matches_run,
            "threshold": threshold,
            "losers_cut": sum(len(c) for c in cuts.values()),
            "losers_cut_editor_kept": cut_verdict["full"] + cut_verdict["partial"],
            "losers_cut_editor_removed": cut_verdict["removed"],
            "sentence_points": result["pooled"]["sentence_points"],
            "word_score": result["pooled"]["word_score"],
            "grade": result["pooled"]["grade"],
            "kept_ratio": result["pooled"]["kept_ratio"],
            "confusion": dict(counts),
            "accuracy": accuracy(counts),
        })
    return out


def level_block(episodes, states, by_arm):
    """Mean level probability by what the editor did, plus confidence quartiles."""
    sums = {"kept": defaultdict(float), "removed": defaultdict(float)}
    counts = {"kept": 0, "removed": 0}
    rows = []
    for episode in episodes:
        human, model = states[episode]
        for sid, (human_kind, _runs) in human.items():
            probabilities = by_arm[ARM][episode][sid].get("score_probabilities")
            if not probabilities:
                continue
            side = "kept" if human_kind in KEPT_STATES else "removed"
            counts[side] += 1
            for level in SCORE_LEVELS:
                sums[side][level] += float(probabilities.get(level, 0.0))
            rows.append({
                "confidence": max(probabilities.values()),
                "right": ((model.get(sid, ("removed", None))[0] in KEPT_STATES)
                          == (human_kind in KEPT_STATES)),
            })

    means = {side: {level: (sums[side][level] / counts[side] if counts[side] else None)
                    for level in SCORE_LEVELS}
             for side in ("kept", "removed")}

    quartiles = []
    if rows:
        rows.sort(key=lambda r: r["confidence"])
        cuts = statistics.quantiles([r["confidence"] for r in rows], n=4)
        buckets = defaultdict(list)
        for row in rows:
            index = sum(1 for cut in cuts if row["confidence"] > cut)
            buckets[index].append(row)
        edges = [rows[0]["confidence"]] + cuts + [rows[-1]["confidence"]]
        for index in range(4):
            bucket = buckets[index]
            quartiles.append({
                "quartile": f"Q{index + 1}",
                "confidence_range": [edges[index], edges[index + 1]],
                "n": len(bucket),
                "accuracy": (sum(1 for r in bucket if r["right"]) / len(bucket)
                             if bucket else None),
            })
    return {"means": means, "counts": counts, "quartiles": quartiles,
            "n_without_probabilities":
                sum(len(by_arm[ARM][e]) for e in episodes) - len(rows)}


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def write_markdown(path, summary, json_path):
    lines = []
    episodes = summary["episodes"]
    add = lines.append

    add("# Jev rough cut: where the points go")
    add("")
    add(f"Generated {summary['generated_utc']} by `{summary['script']}` from the stored decisions and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals; the JSON next to this file keeps the raw values.")
    add("")
    add(f"The arm under the microscope is {summary['arm']}, at its calibrated pooled keep threshold {num(summary['threshold'])}. That is the column the published ladder compares on, because every published arm gets the same two modules layered on. `jev_b_notrim` (identical scores, no partial trims, threshold {num(summary['notrim_threshold'])}) appears next to it wherever trimming is the question.")
    add("")
    if summary["state_check_failures"]:
        add(f"WARNING: the per-sentence states recounted here disagree with the scored cross-tab on {summary['state_check_failures']}. Every count below is suspect.")
    else:
        add("Self-check: recounting the full/partial/removed cross-tab from the per-sentence states this file reads reproduces the `pair_counts` block the metric scored, on every episode.")
    add("")
    add(f"{summary['missing_answers']} sentence(s) never got an answer from the run and are scored at 0.0, so they sit in the arm's removed column by construction.")
    add("")
    add(f"The decisions file is fingerprinted at the bottom. It is rewritten whenever the pipeline patches answers it missed, and the numbers move when it does, so a table here will not match a run report generated against an older copy.")
    add("")

    add("## Per episode against the ladder")
    add("")
    add("SENTENCE POINTS, WORD SCORE and GRADE for the same six episodes, all three arms with um removal and delete silence layered on. `human kept` is the share of dialogue frames the editor kept, which is how tight the target cut is.")
    add("")
    rows = []
    for episode in episodes:
        row = summary["per_episode"][episode]
        rows.append([
            episode, pct(row["sentence_points"]), pct(row["word_score"]),
            pct(row["grade"]), pct(row["notrim_sentence_points"]),
            pct(row.get("luna-chapters-rules5_sentence_points")),
            pct(row.get("luna-chapters-rules5_word_score")),
            pct(row.get("opus5-cc-agentic_sentence_points")),
            pct(row.get("opus5-cc-agentic_word_score")),
            pct(row["human_kept_share"]),
            pct(row["gap_vs_luna"]),
        ])
    pooled = summary["pooled"]
    rows.append(["**pooled**", pct(pooled["sentence_points"]), pct(pooled["word_score"]),
                 pct(pooled["grade"]),
                 pct(summary["notrim_pooled"]["sentence_points"]),
                 "—", "—", "—", "—", "—", "—"])
    lines.extend(table(["episode", "jev_a SP", "jev_a WORD", "jev_a GRADE",
                        "notrim SP", "Luna SP", "Luna WORD", "Opus SP", "Opus WORD",
                        "human kept", "SP gap vs Luna"], rows))
    add("")
    gaps = [(summary["per_episode"][e]["gap_vs_luna"], e) for e in episodes
            if summary["per_episode"][e]["gap_vs_luna"] is not None]
    gaps.sort()
    if gaps:
        behind = [g for g in gaps if g[0] < 0]
        ahead = [g for g in gaps if g[0] >= 0]
        add(f"The whole gap sits in {', '.join(f'{name} ({pct(gap)})' for gap, name in behind)}. On the other {len(ahead)} the arm is level with Luna or ahead, by up to {pct(ahead[-1][0]) if ahead else 'n/a'} on {ahead[-1][1] if ahead else 'n/a'}.")
        add("")

    add("## Sentence-level confusion")
    add("")
    add("Every corpus sentence, at the calibrated threshold, by what the editor did and what the arm did. `kept` on either side means full or partial in that cut; `removed` means gone.")
    add("")
    rows = []
    for episode in episodes + ["pooled"]:
        counts = (summary["confusion"]["pooled"] if episode == "pooled"
                  else summary["confusion"]["per_episode"][episode])
        rows.append([
            f"**{episode}**" if episode == "pooled" else episode,
            counts["n"], counts.get("both_keep", 0),
            counts.get("jev_kept_editor_removed", 0),
            counts.get("jev_removed_editor_kept", 0),
            counts.get("jev_removed_editor_full", 0),
            counts.get("jev_removed_editor_partial", 0),
            counts.get("both_removed", 0),
            pct(accuracy(counts)),
        ])
    lines.extend(table(["episode", "sentences", "both keep", "Jev kept, editor removed",
                        "Jev removed, editor kept", "  of which editor full",
                        "  of which editor partial", "both removed", "agreement"], rows))
    add("")

    luna = summary["confusion"]["luna"]
    if luna["pooled"]:
        add(f"Luna chapters on the same six episodes, {luna['note']}.")
        add("")
        rows = []
        for episode in episodes + ["pooled"]:
            counts = (luna["pooled"] if episode == "pooled"
                      else luna["per_episode"][episode])
            rows.append([
                f"**{episode}**" if episode == "pooled" else episode,
                counts["n"], counts.get("both_keep", 0),
                counts.get("jev_kept_editor_removed", 0),
                counts.get("jev_removed_editor_kept", 0),
                counts.get("both_removed", 0),
                pct(accuracy(counts)),
            ])
        lines.extend(table(["episode", "sentences", "both keep",
                            "Luna kept, editor removed", "Luna removed, editor kept",
                            "both removed", "agreement"], rows))
        add("")
    else:
        add(f"Luna chapters: {luna['note']}.")
        add("")

    add("## The misses")
    add("")
    misses = summary["misses"]
    add(f"{misses['n_jev_removed_editor_kept']} sentences the editor kept and the arm dropped, against {misses['n_jev_kept_editor_removed']} the arm kept and the editor dropped. It errs towards cutting. The lists below are the worst of each: the lowest-scoring drops, and the highest-scoring keeps.")
    add("")
    def patterns_section(key, title, lead):
        add(f"### {title}")
        add("")
        add(lead)
        add("")
        groups = misses[f"{key}_patterns"]
        lines.extend(table(["pattern", "of 40"],
                           [[g["pattern"], g["n"]] for g in groups]))
        add("")
        for group in groups:
            add(f"**{group['pattern']}** ({group['n']}). {group['note']}")
            add("")
            for member in group["members"][:3]:
                add(f"- `{member['episode']}` #{member['id']}, score {num(member['score'])}: {clip(member['text'], 200)}")
            add("")
        add(f"The list in full, worst first. `before` and `after` are the neighbouring sentences.")
        add("")
        for entry in misses[key]:
            tail = ", cut as a retake loser" if entry.get("cut_retake") else ""
            add(f"- `{entry['episode']}` #{entry['id']}, score {num(entry['score'])}, {entry['top_two']}{tail}, pattern: {entry['pattern']}. **{clip(entry['text'])}** (before: {clip(entry['prev'], 120)} / after: {clip(entry['next'], 120)})")
        add("")

    patterns_section(
        "jev_removed_editor_kept",
        "Jev removed, editor kept: the 40 lowest scores",
        "Grouped by reading them with their neighbours. The grouping is judgement, "
        "not code; the counts are over these 40, not over all "
        f"{misses['n_jev_removed_editor_kept']}.")
    patterns_section(
        "jev_kept_editor_removed",
        "Jev kept, editor removed: the 40 highest scores",
        "Same treatment from the other side, over these 40 of "
        f"{misses['n_jev_kept_editor_removed']}.")

    add("## Trim quality")
    add("")
    trims = summary["trims"]
    branch = trims["branch_counts"]
    pairs = trims["pair_counts"]
    jev = trims["jev_trim_branches"]
    layer = trims["layer_only_branches"]
    add(f"Jev emitted {trims['emitted_total']} trims across the six episodes and {trims['jev_trims_scored']} of them survive the keep threshold to reach the metric as a partial sentence. They are not the only partials: the um removal and delete silence layers cut inside another {trims['model_partials_scored'] - trims['jev_trims_scored']} sentences Jev had asked to keep whole. Both columns are below, because only the first is Jev's doing.")
    add("")
    rows = [
        ["exact, run for run (pays 2.0)", jev.get("exact", 0), layer.get("exact", 0)],
        ["subset of the editor's runs (1.2)", jev.get("subset", 0), layer.get("subset", 0)],
        ["overlaps, not contained (1.0)", jev.get("overlap", 0), layer.get("overlap", 0)],
        ["disjoint from the editor's runs (0.6)", jev.get("disjoint", 0),
         layer.get("disjoint", 0)],
        ["editor kept the sentence whole (0.7)",
         jev.get("editor kept it whole (0.7)", 0),
         layer.get("editor kept it whole (0.7)", 0)],
        ["editor removed the sentence (0.0)",
         jev.get("editor removed it (0.0)", 0),
         layer.get("editor removed it (0.0)", 0)],
        ["**total**", trims["jev_trims_scored"],
         trims["model_partials_scored"] - trims["jev_trims_scored"]],
    ]
    lines.extend(table(["trim outcome", "Jev's own trims", "layers only"], rows))
    add("")
    add(f"For scale, the editor trimmed inside {pairs.get('partial/full', 0) + pairs.get('partial/partial', 0) + pairs.get('partial/removed', 0)} sentences; the arm came back partial on {branch.get('exact', 0) + branch.get('subset', 0) + branch.get('overlap', 0) + branch.get('disjoint', 0)} of them and kept or dropped the rest whole. Dropping Jev's trims entirely (`jev_b_notrim`) scores {pct(trims['notrim_pooled_sentence_points'])} pooled SENTENCE POINTS against {pct(summary['pooled']['sentence_points'])} with them.")
    add("")
    add("Fifteen trims, the arm's kept span in square brackets and the editor's in braces. A sentence the editor kept whole shows braces around everything; a sentence the editor removed shows none.")
    add("")
    for entry in trims["examples"]:
        add(f"- `{entry['episode']}` #{entry['id']}, {entry['branch']}, score {num(entry['score'])}: {clip(entry['spans'], 400)}")
    add("")

    add("## Retake veto sweep")
    add("")
    add("`cut_retake` rebuilt offline from the stored `retake_real`, `retake_choice` and `retake_take_probs`, then rescored. `none` never vetoes: the losers of Jev's chosen winner are always cut. The last two columns take the sentences each setting cut as retake losers and ask what the editor did with them.")
    add("")
    rows = []
    for entry in summary["veto"]:
        counts = entry["confusion"]
        rows.append([
            entry["label"], num(entry["threshold"]), pct(entry["sentence_points"]),
            pct(entry["word_score"]), pct(entry["grade"]),
            pct(entry["fixed"]["sentence_points"]), pct(entry["fixed"]["word_score"]),
            entry["losers_cut"],
            entry["losers_cut_editor_kept"], entry["losers_cut_editor_removed"],
            counts.get("jev_kept_editor_removed", 0),
            counts.get("jev_removed_editor_kept", 0),
            pct(entry["accuracy"]),
        ])
    lines.extend(table(["veto on real_k", "threshold", "SENTENCE POINTS", "WORD SCORE",
                        "GRADE",
                        f"SP at {num(summary['veto_fixed_threshold'])}",
                        f"WORD at {num(summary['veto_fixed_threshold'])}",
                        "losers cut", "  editor kept them",
                        "  editor cut them", "Jev kept, editor removed",
                        "Jev removed, editor kept", "agreement"], rows))
    add("")
    best = max(summary["veto"], key=lambda e: e["sentence_points"] or 0)
    fixed_best = max(summary["veto"], key=lambda e: e["fixed"]["sentence_points"] or 0)
    add(f"Best on the headline: {best['label']} at {pct(best['sentence_points'])} SENTENCE POINTS, {pct(best['word_score'])} WORD SCORE, {pct(best['grade'])} GRADE. Each setting calibrates its own keep threshold, so the last two columns rescore every setting at {num(summary['veto_fixed_threshold'])}, the threshold the reported arm uses, to show how much of that win is the veto and how much is the threshold landing differently. On the fixed threshold the best is {fixed_best['label']} at {pct(fixed_best['fixed']['sentence_points'])}.")
    add("")

    add("## Do the six score levels separate?")
    add("")
    levels = summary["levels"]
    add(f"Mean probability Jev put on each level, over the {levels['counts']['kept']} sentences the editor kept and the {levels['counts']['removed']} it removed. If the levels carried the signal the prompt asks for, the removed column would load on 0 and 1 and the kept column on 4 and 5.")
    add("")
    rows = []
    for level in SCORE_LEVELS:
        kept = levels["means"]["kept"][level]
        removed = levels["means"]["removed"][level]
        rows.append([f"level {level}", num(removed, 3), num(kept, 3),
                     num(None if kept is None or removed is None else kept - removed, 3)])
    lines.extend(table(["score level", "editor removed", "editor kept", "kept minus removed"],
                       rows))
    add("")
    add("Keep/cut agreement by how confident Jev was, where confidence is the probability mass on its top level.")
    add("")
    rows = [[q["quartile"], f"{q['confidence_range'][0]:.2f}-{q['confidence_range'][1]:.2f}",
             q["n"], pct(q["accuracy"])] for q in levels["quartiles"]]
    lines.extend(table(["quartile", "top-level probability", "sentences", "agreement"],
                       rows))
    add("")
    if levels["n_without_probabilities"]:
        add(f"{levels['n_without_probabilities']} sentence(s) carry no level distribution (the run never got an answer) and are left out of this section.")
        add("")

    add("## Files")
    add("")
    for key, value in summary["input_fingerprint"].items():
        add(f"- input {key}: `{value['path']}`, md5 {value['md5'][:12]}, modified {value['modified_utc']}")
    add(f"- this file: `{path}`")
    add(f"- data: `{json_path}`")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_name", default="roughcut-jev",
                        help="input basename under docs/jev-real")
    parser.add_argument("--out", dest="out_name", default="roughcut-jev",
                        help="output basename under docs/jev-real")
    args = parser.parse_args()

    md_path = OUT_DIR / f"{args.out_name}-misses.md"
    json_path = OUT_DIR / f"{args.out_name}-misses.json"
    existing = [str(p) for p in (md_path, json_path) if p.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    summary = build(args)
    summary["markdown_path"] = str(md_path)
    summary["json_path"] = str(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(md_path, summary, json_path)
    print(json.dumps({
        "markdown": str(md_path), "json": str(json_path),
        "threshold": summary["threshold"],
        "pooled_sentence_points": summary["pooled"]["sentence_points"],
        "confusion": summary["confusion"]["pooled"],
        "state_check_failures": summary["state_check_failures"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
