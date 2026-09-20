"""Export one Jev rough-cut arm as a solar-sailer benchmark result JSON.

Turns the three files ``scripts/jev_real/roughcut_jev.py`` wrote into a file
shaped exactly like the harness's own ``benchmarks/roughcut/results/*.json``,
so a later lead can drop it into that directory and have the bench-page
exporter and the rescore scripts read it without a single Jev call.

What it does
------------
1. Rebuilds the arm's decisions from ``docs/jev-real/<name>-decisions.jsonl``
   through ``roughcut_jev_report``'s own helpers, so the numbers here and the
   numbers in that script's notes come from one rebuild.
2. Calibrates the pooled keep threshold with
   ``roughcut_partial_scoring.calibrate_threshold`` and scores every episode
   PLAIN: the model's cut alone, which is what a harness result JSON stores.
   The um-removal and delete-silence layering is what the ladder page adds on
   top of every arm, so it lands in a separate ``modules_layered`` block, and
   only with ``--modules``.
3. Writes the result JSON: the harness metadata, ``operating_points``, the
   per-episode blocks with all six levels, ``pooled``, ``total_cost``, and
   ``run_ratings`` in the harness's own rating shape.
4. Validates by running the harness's ``scripts/rescore_ratings.py`` on a copy
   of the export in a temp dir inside THIS repo and comparing what that script
   computes against what was exported.

Retakes
-------
Jev's retake decision replaces the corpus ``is_retake`` flag, and a result
JSON has nowhere to put that. So the export does two things: every Jev retake
loser is written with score 0 and category ``repeated_take`` (a rescore that
keeps the corpus flags still cuts it), and ``retake_overrides`` lists both
directions of the disagreement per episode.

READ-ONLY against solar-sailer: it reads the harness package and runs one
harness script with the cwd set there, and writes only under this repo.

Usage::

  python scripts/jev_real/roughcut_jev_export.py \
      --in roughcut-jev-all18-v3 --arm jev_a --t-trim 0.3 \
      --out docs/jev-real/export/2026-09-20-all18-jev-v3-sentence-pass.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(0, str(HERE))

# The scoring module installs the cache-only answer-key loader and puts the
# harness on sys.path; import it before anything that touches the corpus.
import roughcut_partial_scoring as scoring_mod  # noqa: E402
from roughcut_partial_scoring import (  # noqa: E402
    BENCH_DIR, NEUTRAL_LEVEL, calibrate_threshold, detect_removals_for,
    episode_report, load_episode, removals_cache_path,
)
import roughcut_jev_report as report_mod  # noqa: E402

pooling = scoring_mod.pooling
# The level-entry projection and the harness's category menu are imported, not
# restated, so the export tracks the harness if either moves.
from roughcut_bench.experiments import _episode_level_entry  # noqa: E402
from roughcut_bench.parsing import VALID_CATEGORIES  # noqa: E402

LEVELS = list(range(1, 7))
EXPERIMENT_TAG = "jev-roughcut-v3"
WORKFLOW = "jev_sentence_pass"
MODELS = ["jev-1.13.0"]
PROMPTS = ["scripts/jev_real/roughcut_jev_prompts.py v3"]

#: Jev answers a 0-5 score and a retake verdict, never a category, but the
#: harness rating shape carries one and the page's copy reads it. Map the score
#: onto the closest of the seven categories: the menu's own "typically 3-5" is
#: keep, and below that the rubric's wording picks the label (2 "weak,
#: meandering" -> rambling, 1 "junk with mild salvage value" -> tangent,
#: 0 "definite junk, dead air" -> filler). A Jev retake loser is repeated_take.
CATEGORY_BY_SCORE = {5: "keep", 4: "keep", 3: "keep",
                     2: "rambling", 1: "tangent", 0: "filler"}
RETAKE_CATEGORY = "repeated_take"

#: The deterministic layers, keyed as ``2026-09-11-model-plus-deterministic.json``
#: keys them.
MODULE_LAYERS = [
    ("published", "published (no extra modules)", ()),
    ("silence", "+ delete silence", ("silence",)),
    ("umm_silence", "+ um removal + delete silence", ("umm", "silence")),
]

#: Two decimals on the x100 scale, the gate the design spec sets.
TOLERANCE = 5e-5


def category_for(score, cut_retake):
    """The harness category this decision would have carried."""
    if cut_retake:
        return RETAKE_CATEGORY
    level = max(0, min(5, int(round(float(score)))))
    return CATEGORY_BY_SCORE[level]


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------


def build_run_ratings(decisions):
    """One episode's decisions in the harness's ``run_ratings`` rating shape.

    A Jev retake loser is written at score 0: ``rescore_ratings.py`` and the
    page never see ``cut_retake``, so the score has to carry the cut on its
    own. ``retake_overrides`` records what was overridden.
    """
    ratings = {}
    for sid in sorted(decisions):
        decision = decisions[sid]
        cut_retake = bool(decision.get("cut_retake"))
        score = 0.0 if cut_retake else float(decision["score"])
        ratings[str(sid)] = {
            "score": score,
            "category": category_for(decision["score"], cut_retake),
            "keep_words": decision.get("keep_words"),
        }
    return ratings


def retake_overrides(name, decisions):
    """Where Jev's retake verdict and the corpus ``is_retake`` flags disagree.

    ``is_retake_true`` are the sentences this run cuts as retake losers;
    ``is_retake_false`` are corpus-flagged sentences this run kept. Applying
    both lists over the corpus flags reproduces the cut the numbers describe.
    """
    flags = load_episode(name)["retake_flags"]
    jev_cut = {sid for sid, d in decisions.items() if d.get("cut_retake")}
    corpus = {sid for sid, flag in flags.items() if flag}
    return {
        "note": ("Jev's retake pass replaces the corpus is_retake flags. Every "
                 "sentence in is_retake_true is written at score 0 with "
                 "category repeated_take, so a rescore that keeps the corpus "
                 "flags still cuts it; the is_retake_false ids are corpus "
                 "retakes this run kept, and a rescore that keeps the corpus "
                 "flags cuts them where this result keeps them."),
        "is_retake_true": sorted(jev_cut),
        "is_retake_false": sorted(corpus - jev_cut),
        "corpus_flagged": len(corpus),
        "jev_cut": len(jev_cut),
        "agree": len(corpus & jev_cut),
    }


def episode_block(name, decisions, reports, thresholds, latency, timing):
    """One ``episodes[<name>]`` entry, field for field as the harness writes it."""
    data = load_episode(name)
    report = reports[name]
    timing_episode = (timing.get("episodes", {}).get(name) or {})
    entry = latency.get(name) or {}
    cost = timing_episode.get("cost_usd")
    if cost is None:
        cost = entry.get("cost_usd")
    return {
        "media_name": report["media_name"],
        "sentence_frame_offset": report["sentence_frame_offset"],
        "frame_rate": data["frame_rate"],
        "total_dialogue": report["total_dialogue"],
        "human_kept_frames": data["human_kept_frames"],
        "ceiling": report["ceiling"],
        "levels": {str(level): _episode_level_entry(report["levels"][str(level)],
                                                    thresholds[level])
                   for level in LEVELS},
        "cost": cost,
        "per_run_costs": [cost],
        "warnings": report["warnings"],
        "latency": {
            "pass_wall_clock_s": entry.get("pass_wall_clock_s"),
            "wall_clock_s": entry.get("wall_clock_s"),
            "requests": entry.get("requests"),
            "input_tokens": entry.get("input_tokens"),
            "output_tokens": entry.get("output_tokens"),
            "cost_usd": entry.get("cost_usd"),
            "retries": entry.get("retries"),
            "errors": entry.get("errors"),
            "windowed_requests": entry.get("windowed_requests"),
            "sentences": entry.get("sentences"),
        },
        "retake_overrides": retake_overrides(name, decisions),
        "run_ratings": [build_run_ratings(decisions)],
    }


def pooled_block(episodes, reports, ops):
    """The ``pooled`` block: duration weights and all six pooled levels."""
    weights = [reports[name]["total_dialogue"] for name in episodes]
    levels = {}
    for level in LEVELS:
        level_reports = [reports[name]["levels"][str(level)] for name in episodes]
        fair_cms = [lr["fair_cm"] for lr in level_reports]
        ceiling_cms = [reports[name]["ceiling"]["cm"] for name in episodes]
        levels[str(level)] = {
            "threshold": ops["thresholds"][level],
            "kept_ratio": ops["achieved_ratios"][level],
            "grade": pooling.weighted_mean([lr["grade"] for lr in level_reports],
                                           weights),
            "word_grade": pooling.pooled_word_grade(level_reports),
            "sp_grade": pooling.pooled_sp_grade(level_reports),
            "grade_summed_cm": pooling.pooled_cm_grade(fair_cms, ceiling_cms),
            "fair_cm_sum": pooling.sum_cms(fair_cms),
        }
    return {
        "weights": {name: reports[name]["total_dialogue"] for name in episodes},
        "human_kept_frames": sum(load_episode(name)["human_kept_frames"]
                                 for name in episodes),
        "levels": levels,
    }


def modules_layered(episodes, decisions, threshold, plain_reports):
    """The same cut with the shipping modules layered on, ladder-style.

    Mirrors ``2026-09-11-model-plus-deterministic.json``: the model's cut is
    fixed at its own Neutral threshold and the module frames come off it, no
    recalibration, one column per layer. That is the column the ladder page's
    "Deterministic modules applied" view compares on.
    """
    per_episode, skipped = {}, []
    for name in episodes:
        if not removals_cache_path(name).exists():
            skipped.append(name)
            continue
        removals = detect_removals_for(name)
        plain = plain_reports[name]["levels"][str(NEUTRAL_LEVEL)]
        entry = {
            "dialogue_frames": plain_reports[name]["total_dialogue"],
            "grades": {"published": plain["grade"]},
            "word_grades": {"published": plain["word_grade"]},
            "sp_grades": {"published": plain["sp_grade"]},
            "word_count": (plain["word"] or {}).get("count"),
            "sentence_count": (plain["sp"] or {}).get("n_sentences"),
            "trims": {"published": {"sentences_trimmed": 0, "sentences_cut": 0}},
        }
        for key, _label, parts in MODULE_LAYERS:
            if not parts:
                continue
            subset = {part: removals[part] for part in parts}
            layered = episode_report(
                name, decisions[name], {NEUTRAL_LEVEL: threshold},
                levels=(NEUTRAL_LEVEL,), removals=subset,
                gate_threshold=threshold)
            level = layered["levels"][str(NEUTRAL_LEVEL)]
            entry["grades"][key] = level["grade"]
            entry["word_grades"][key] = level["word_grade"]
            entry["sp_grades"][key] = level["sp_grade"]
            entry["trims"][key] = {
                "sentences_trimmed": layered["sentences_trimmed_by_removals"],
                "sentences_cut": layered["sentences_cut_by_removals"],
            }
        per_episode[name] = entry

    scored = list(per_episode)
    frames = [per_episode[n]["dialogue_frames"] for n in scored]
    words = [per_episode[n]["word_count"] for n in scored]
    sentences = [per_episode[n]["sentence_count"] for n in scored]
    weighted, weighted_word, weighted_sp = {}, {}, {}
    for key, _label, _parts in MODULE_LAYERS:
        weighted[key] = pooling.weighted_mean(
            [per_episode[n]["grades"][key] for n in scored], frames)
        weighted_word[key] = pooling.weighted_mean(
            [per_episode[n]["word_grades"][key] for n in scored], words)
        weighted_sp[key] = pooling.weighted_mean(
            [per_episode[n]["sp_grades"][key] for n in scored], sentences)
    return {
        "note": ("The deterministic modules layered onto this arm's cut the way "
                 "scripts/model_plus_deterministic.py layers them onto every "
                 "ladder arm: the module frames come off the model's kept "
                 "ranges at this run's own Neutral threshold, a sentence left "
                 "with no frames is cut whole, and nothing is recalibrated. "
                 "The per-episode blocks above are the plain cut; this block is "
                 "what the page's 'Deterministic modules applied' view shows."),
        "source": ("scripts/jev_real/roughcut_partial_scoring.py, detectors from "
                   "benchmarks/roughcut/scripts/deterministic_baseline.py"),
        "layers": {key: label for key, label, _parts in MODULE_LAYERS},
        "threshold": threshold,
        "episodes_scored": len(scored),
        "skipped_no_removals_cache": skipped,
        "weighted": weighted,
        "weighted_word": weighted_word,
        "weighted_sp": weighted_sp,
        "dialogue_frames": sum(frames),
        "word_count": sum(words),
        "sentence_count": sum(sentences),
        "episodes": per_episode,
    }


def build_export(args):
    """The whole result document, plus the pieces the validation gate needs."""
    decision_rows, request_rows, timing, input_paths = report_mod.load_run(args.in_name)
    by_arm, episodes = report_mod.index_decisions(decision_rows)
    if args.arm not in by_arm and args.arm not in report_mod.DERIVED_ARMS:
        raise SystemExit(f"{args.in_name}-decisions.jsonl has no rows for arm "
                         f"{args.arm!r} (found: {sorted(by_arm)})")

    words = {name: report_mod.word_ids_by_sentence(name) for name in episodes}
    mismatches = report_mod.check_rebuild(by_arm, episodes, words)
    decisions, n_missing = report_mod.build_arm_decisions(
        by_arm, episodes, words, args.arm, args.t_trim, args.missing_score)

    print(f"calibrating {args.arm} at t_trim {args.t_trim:g} over "
          f"{len(episodes)} episode(s)...", file=sys.stderr)
    calibrated = calibrate_threshold(decisions)
    ops = calibrated["operating_points"]
    threshold = calibrated["threshold"]

    reports = {}
    for name in episodes:
        print(f"scoring {name} plain...", file=sys.stderr)
        reports[name] = episode_report(name, decisions[name], ops["thresholds"],
                                       levels=LEVELS)

    latency = report_mod.latency_rows(timing, request_rows, episodes)
    totals = timing.get("totals") or {}
    total_cost = totals.get("cost_usd")
    if total_cost is None:
        total_cost = round(sum(latency[name]["cost_usd"] for name in episodes), 6)

    trims = {name: sum(1 for d in decisions[name].values() if d["keep_words"])
             for name in episodes}
    notes = (
        f"Jev rough cut, sentence pass only (arm {args.arm}, trim-trigger "
        f"t_trim {args.t_trim:g}). Exported by "
        f"scripts/jev_real/roughcut_jev_export.py in the skell-e-router repo "
        f"from {args.in_name}-decisions.jsonl; no model calls were made to "
        f"build this file and no Jev call is needed to rescore it. The cut is "
        f"three Jev passes: a retake pass that cuts the losing take of a "
        f"repeated line, a 0-5 keep score per sentence, and an edge-trim "
        f"question per side whose answer becomes keep_words when Jev puts less "
        f"than {args.t_trim:g} on keeping the sentence whole. "
        f"{sum(trims.values())} of {sum(len(d) for d in decisions.values())} "
        f"sentences carry a word range. SENTENCE POINTS and WORD SCORE are "
        f"native here, not backfilled. Retake overrides: see "
        f"episodes[*].retake_overrides."
    )

    document = {
        "date": date.today().isoformat(),
        "name": args.name,
        "experiment": {
            "name": args.name,
            "experiment": EXPERIMENT_TAG,
            "workflow": WORKFLOW,
            "model": MODELS[0],
            "prompt": PROMPTS[0],
            "arm": args.arm,
            "t_trim": args.t_trim,
            "missing_score": args.missing_score,
            "episodes": list(episodes),
            "notes": notes,
        },
        "workflow": WORKFLOW,
        "models": list(MODELS),
        "prompts": list(PROMPTS),
        "neutral_level": NEUTRAL_LEVEL,
        "neutral_threshold": threshold,
        "operating_points": ops,
        "episodes": {name: episode_block(name, decisions[name], reports,
                                         ops["thresholds"], latency, timing)
                     for name in episodes},
        "pooled": pooled_block(episodes, reports, ops),
        "total_cost": total_cost,
        "source": {
            "script": "scripts/jev_real/roughcut_jev_export.py",
            "repo": "skell-e-router",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_files": input_paths,
            "run_model": timing.get("model"),
            "run_prompt_version": timing.get("prompt_version"),
            "run_concurrency": timing.get("concurrency"),
            "run_wall_clock_s": totals.get("wall_clock_s"),
            "run_requests": totals.get("requests"),
            "merged_from": timing.get("merged_from"),
            "merged_note": timing.get("merged_note"),
            "trims_emitted": trims,
            "sentences_without_an_answer": n_missing,
            "rebuild_mismatches": len(mismatches),
            "missing_fields": {
                "raw_responses_file": (
                    "no raw text sidecar: Jev returns a probability "
                    "distribution over fixed options, not free text. The "
                    "per-sentence distributions are in "
                    f"{args.in_name}-decisions.jsonl in the skell-e-router repo."),
                "disagreements": (
                    "the top-20 disagreement list is built by a private helper "
                    "in roughcut_bench.experiments and nothing reads it off a "
                    "result file; omitted rather than reimplemented."),
                "run_ratings[*].category": (
                    "Jev answers a score and a retake verdict, never a "
                    "category; the category here is derived from the score."),
            },
        },
    }
    if args.modules:
        print("layering the deterministic modules...", file=sys.stderr)
        document["modules_layered"] = modules_layered(episodes, decisions,
                                                      threshold, reports)
    return document, episodes


# ---------------------------------------------------------------------------
# validation gate
# ---------------------------------------------------------------------------

#: Runs the harness's own rescore script on the exported file and records what
#: it computed at full precision. The script prints one decimal and no
#: SENTENCE POINTS, so the wrapper records the level reports it builds instead
#: of parsing its stdout. It patches nothing else: importing the scoring module
#: installs the cache-only answer-key loader, which is what keeps the run
#: read-only against solar-sailer.
RESCORE_WRAPPER = '''
import json
import os
import runpy
import sys

jev_dir, bench_dir, result_path, capture_path = sys.argv[1:5]
extra = sys.argv[5:]
sys.path.insert(0, jev_dir)
import roughcut_partial_scoring  # noqa: F401  (installs the read-only loader)
from roughcut_bench import calibrate, replay

reports, ops_seen = [], []
_score_sentences = replay.score_sentences
_pick = calibrate.pick_operating_points


def score_sentences(*args, **kwargs):
    report = _score_sentences(*args, **kwargs)
    reports.append({
        "levels": {key: {"grade": level["grade"],
                         "word_grade": level["word_grade"],
                         "sp_grade": level["sp_grade"],
                         "n_sentences": (level["sp"] or {}).get("n_sentences"),
                         "word_count": (level["word"] or {}).get("count")}
                   for key, level in report["levels"].items()},
        "total_dialogue": report["total_dialogue"],
    })
    return report


def pick_operating_points(*args, **kwargs):
    ops = _pick(*args, **kwargs)
    ops_seen.append({"thresholds": {str(k): v for k, v in ops["thresholds"].items()},
                     "neutral_threshold": ops["neutral_threshold"],
                     "achieved_ratios": {str(k): v
                                         for k, v in ops["achieved_ratios"].items()}})
    return ops


replay.score_sentences = score_sentences
calibrate.pick_operating_points = pick_operating_points

script = os.path.join(bench_dir, "scripts", "rescore_ratings.py")
sys.argv = [script, result_path] + extra
runpy.run_path(script, run_name="__main__")

with open(capture_path, "w", encoding="utf-8") as handle:
    json.dump({"reports": reports, "operating_points": ops_seen[-1] if ops_seen else None},
              handle)
'''


def run_rescore(result_path, work_dir, extra_args=()):
    """``scripts/rescore_ratings.py`` on ``result_path``, with what it computed.

    Runs from the harness directory so its relative episode and corpus paths
    resolve, on a COPY of the export that lives in this repo's temp dir. The
    script itself writes nothing; the wrapper's capture file lands next to the
    copy, inside this repo.
    """
    wrapper = work_dir / "_rescore_wrapper.py"
    wrapper.write_text(RESCORE_WRAPPER, encoding="utf-8")
    capture = work_dir / "_rescore_capture.json"
    command = [sys.executable, str(wrapper), str(HERE), str(BENCH_DIR),
               str(result_path), str(capture), *extra_args]
    proc = subprocess.run(command, cwd=str(BENCH_DIR), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not capture.exists():
        raise SystemExit(f"rescore_ratings.py failed (exit {proc.returncode}):\n"
                         f"{proc.stdout}\n{proc.stderr}")
    with capture.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["stdout"] = proc.stdout
    return payload


def compare(document, episodes, rescored, label):
    """Exported per-episode SP, WORD and GRADE against the rescore's own."""
    reports = rescored["reports"]
    if len(reports) != len(episodes):
        raise SystemExit(f"rescore produced {len(reports)} episode reports for "
                         f"{len(episodes)} episodes")
    level = str(NEUTRAL_LEVEL)
    rows, failures = [], []
    for name, rescored_report in zip(episodes, reports):
        exported = document["episodes"][name]["levels"][level]
        mine_all = rescored_report["levels"][level]
        for metric, key in (("SENTENCE POINTS", "sp_grade"),
                            ("WORD SCORE", "word_grade"), ("GRADE", "grade")):
            exported_value, rescored_value = exported[key], mine_all[key]
            delta = (None if exported_value is None or rescored_value is None
                     else rescored_value - exported_value)
            ok = delta is not None and abs(delta) <= TOLERANCE
            if not ok:
                failures.append(f"{name} {metric}")
            rows.append({"pass": label, "episode": name, "metric": metric,
                         "exported": exported_value, "rescored": rescored_value,
                         "delta": delta, "ok": ok})
    return {"pass": label, "rows": rows, "failures": failures,
            "threshold_exported": document["neutral_threshold"],
            "threshold_rescored": (rescored["operating_points"] or {}).get(
                "neutral_threshold"),
            "stdout": rescored["stdout"]}


def decisions_from_export(document):
    """The arm's decisions rebuilt from the exported file and nothing else.

    Reads ``run_ratings`` for the score and the word ranges, and applies
    ``retake_overrides`` over the corpus ``is_retake`` flags. If this
    reproduces the exported numbers, every number in the file can be rebuilt
    from the file, which is what "drop it in the results dir and rescore it"
    has to mean for an arm whose retake verdict is its own.
    """
    out = {}
    for name, block in document["episodes"].items():
        flags = dict(load_episode(name)["retake_flags"])
        overrides = block["retake_overrides"]
        for sid in overrides["is_retake_true"]:
            flags[sid] = True
        for sid in overrides["is_retake_false"]:
            flags[sid] = False
        decisions = {}
        for raw_id, rating in block["run_ratings"][0].items():
            sid = int(raw_id)
            decisions[sid] = {
                "score": float(rating["score"]),
                "keep_words": rating.get("keep_words"),
                "cut_retake": bool(flags.get(sid, False)),
            }
        out[name] = decisions
    return out


def self_rescore(document, episodes):
    """Gate 2: rescore the exported file through the scoring module itself.

    ``rescore_ratings.py`` cannot read ``keep_words`` (it rebuilds whole
    sentences from the score alone), so it cannot reproduce ANY partial arm's
    published numbers, this one or the harness's own. This gate closes that
    hole with the same harness code the export used, driven only by what the
    file carries.
    """
    decisions = decisions_from_export(document)
    calibrated = calibrate_threshold(decisions)
    level = str(NEUTRAL_LEVEL)
    rows, failures = [], []
    for name in episodes:
        exported = document["episodes"][name]["levels"][level]
        rescored = calibrated["episodes"][name]
        for metric, exported_key, rescored_key in (
                ("SENTENCE POINTS", "sp_grade", "sentence_points"),
                ("WORD SCORE", "word_grade", "word_score"),
                ("GRADE", "grade", "grade")):
            exported_value = exported[exported_key]
            rescored_value = rescored[rescored_key]
            delta = (None if exported_value is None or rescored_value is None
                     else rescored_value - exported_value)
            ok = delta is not None and abs(delta) <= TOLERANCE
            if not ok:
                failures.append(f"{name} {metric}")
            rows.append({"pass": "self-rescore", "episode": name,
                         "metric": metric, "exported": exported_value,
                         "rescored": rescored_value, "delta": delta, "ok": ok})
    return {"pass": "self-rescore", "rows": rows, "failures": failures,
            "threshold_exported": document["neutral_threshold"],
            "threshold_rescored": calibrated["threshold"],
            "stdout": ""}


def validate(document, episodes, out_path):
    """Copy the export to a temp dir in this repo and rescore it, twice.

    The second pass strips the corpus retake flags (``--no-retakes``), which is
    the counterfactual that isolates how much of any gap is the retake override
    rather than the word ranges the rescore script cannot read.
    """
    parent = out_path.parent / "_validation"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(parent)) as tmp:
        work = Path(tmp)
        copy = work / out_path.name
        shutil.copy2(out_path, copy)
        results = [self_rescore(document, episodes),
                   compare(document, episodes, run_rescore(copy, work),
                           "with-retakes"),
                   compare(document, episodes,
                           run_rescore(copy, work, ["--no-retakes"]),
                           "no-retakes")]
    try:
        parent.rmdir()
    except OSError:
        pass
    return results


PASS_LABELS = {
    "self-rescore": "gate: rescore the export through roughcut_partial_scoring",
    "with-retakes": "harness scripts/rescore_ratings.py (corpus retake flags)",
    "no-retakes": "harness scripts/rescore_ratings.py --no-retakes",
}


def print_validation(results):
    for result in results:
        failures = result["failures"]
        print(f"\n=== {PASS_LABELS.get(result['pass'], result['pass'])} ===")
        print(f"threshold exported {result['threshold_exported']}, "
              f"rescored {result['threshold_rescored']}")
        worst = max((abs(row["delta"]) for row in result["rows"]
                     if row["delta"] is not None), default=0.0)
        print(f"{len(result['rows']) - len(failures)} of {len(result['rows'])} "
              f"comparisons match to two decimals (x100); largest delta "
              f"{worst * 100:+.4f}")
        for row in result["rows"]:
            if not row["ok"]:
                print(f"  MISMATCH {row['episode']} {row['metric']}: exported "
                      f"{(row['exported'] or 0) * 100:.2f} vs rescored "
                      f"{(row['rescored'] or 0) * 100:.2f} "
                      f"({(row['delta'] or 0) * 100:+.2f})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_name", required=True,
                        help="input basename under docs/jev-real")
    parser.add_argument("--arm", default="jev_a",
                        help="arm to export (default: jev_a)")
    parser.add_argument("--t-trim", type=float, default=0.3,
                        help="variant A trim-trigger threshold (default: 0.3)")
    parser.add_argument("--out", required=True, help="output result JSON path")
    parser.add_argument("--name", default=None,
                        help="result name (default: the output file stem "
                             "without its leading date)")
    parser.add_argument("--modules", action="store_true",
                        help="also compute the modules-layered block")
    parser.add_argument("--missing-score", type=float, default=0.0,
                        help="score for a sentence the run got no answer for "
                             "(default: 0.0, which cuts it)")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the rescore_ratings.py gate")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output file")
    args = parser.parse_args()

    for category in list(CATEGORY_BY_SCORE.values()) + [RETAKE_CATEGORY]:
        if category not in VALID_CATEGORIES:
            raise SystemExit(f"category {category!r} is not in the harness's menu "
                             f"{sorted(VALID_CATEGORIES)}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing output: {out_path}")
    if str(out_path).lower().startswith(("d:", str(BENCH_DIR).lower())):
        raise SystemExit(f"refusing to write under solar-sailer: {out_path}")
    if args.name is None:
        stem = out_path.stem
        parts = stem.split("-", 3)
        args.name = parts[3] if len(parts) == 4 and parts[0].isdigit() else stem

    document, episodes = build_export(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    validation = None
    if not args.no_validate:
        validation = validate(document, episodes, out_path)
        print_validation(validation)
        record = out_path.with_name(out_path.stem + ".validation.json")
        record.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(f"\nvalidation record -> {record}", file=sys.stderr)

    pooled = document["pooled"]["levels"][str(NEUTRAL_LEVEL)]
    print(json.dumps({
        "out": str(out_path),
        "episodes": len(episodes),
        "arm": args.arm,
        "t_trim": args.t_trim,
        "neutral_threshold": document["neutral_threshold"],
        "pooled_sentence_points": pooled["sp_grade"],
        "pooled_word_score": pooled["word_grade"],
        "pooled_grade": pooled["grade"],
        "modules_layered_sp": (document.get("modules_layered") or {})
                              .get("weighted_sp"),
        "total_cost": document["total_cost"],
        "validation_failures": None if validation is None else
                               {r["pass"]: len(r["failures"]) for r in validation},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
