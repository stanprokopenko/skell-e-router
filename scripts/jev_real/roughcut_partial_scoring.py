"""Score a Jev rough-cut decision set on the solar-sailer benchmark ladder.

The module takes ``{sentence_id: {"score", "keep_words", "cut_retake"}}`` per
episode and returns the harness's own numbers: SENTENCE POINTS (majority rule,
bonus 2.0, with and without the 0.5 no-partial penalty), WORD SCORE, GRADE,
frame match, the partial counts and the kept ratio.

Nothing here reimplements a metric. Every number comes out of
``roughcut_bench`` (``partial``, ``ranges``, ``sentence_scoring``,
``word_scoring``, ``scoring``, ``calibrate``, ``pooling``, ``replay``), called
the way ``benchmarks/roughcut/scripts/rescore_sentence_points.py``,
``rescore_ratings.py`` and ``model_plus_deterministic.py`` call it.

Three pieces are deliberately copied rather than imported:

* the deterministic layering (``_apply_removals``) is a line-for-line copy of
  ``model_plus_deterministic.apply_layer`` so the module ranges land on the
  same side of the subtraction the published ladder used: they come OFF the
  model's kept ranges, and a sentence left with no frames is cut whole;
* the retake decision arrives as ``cut_retake`` instead of the corpus
  ``is_retake`` flag. The corpus flag is stripped first (the ``--no-retakes``
  pattern in ``rescore_ratings.py``) and ``cut_retake`` is written back into
  the same field, so ``ranges.kept_segments_from_score`` applies it exactly as
  it applies the module's;
* the answer key is read through ``scripts/jev_real/roughcut_bench.py``'s
  ``_cached_answer_key``, which monkeypatches ``episodes.episode_answer_key``
  so nothing under solar-sailer is ever rewritten.

READ-ONLY against solar-sailer. The only file this module writes is the
removals cache under ``docs/jev-real/removals/`` in this repo.

Usage::

  python scripts/jev_real/roughcut_partial_scoring.py --validate
  python scripts/jev_real/roughcut_partial_scoring.py --removals <episode>
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(r"D:\solar-sailer\benchmarks\roughcut")
EPISODES_DIR = BENCH_DIR / "episodes"
BENCH_SCRIPTS_DIR = BENCH_DIR / "scripts"
RESULTS_DIR = BENCH_DIR / "results"

# ``deterministic_baseline`` puts ``<bench>/../../editor`` on sys.path for the
# shipping ``server`` package. That resolves to D:\solar-sailer\editor, which
# does not exist on this machine (the D: tree holds benchmark data only, the
# code lives in the GitHub checkout), so the real checkout is offered first and
# the script's own insert becomes a harmless no-op.
EDITOR_DIRS = [
    Path(r"C:\Users\Stan\Documents\GitHub\solar-sailer\editor"),
    BENCH_DIR.parents[1] / "editor",
]

REMOVALS_DIR = ROOT / "docs" / "jev-real" / "removals"

#: Score written onto a sentence whose frames the deterministic layers ate
#: whole. Same sentinel ``model_plus_deterministic`` uses.
CUT = -1e9

#: The harness's Neutral aggression level.
NEUTRAL_LEVEL = 4

#: Archived run the validation gate reproduces.
VALIDATION_RESULT = RESULTS_DIR / "2026-09-06-all18-partial-single-gpt-5.6-luna.json"
VALIDATION_LAYERED = RESULTS_DIR / "2026-09-11-model-plus-deterministic.json"
VALIDATION_ARM = "luna-single-call"
VALIDATION_EPISODES = [
    "colman-02.04-skeleton-demo",
    "hampton-5.4-assignment-demo",
    "edges-7.01-intro",
]
#: Two decimals on the x100 scale.
VALIDATION_TOLERANCE = 5e-5


# ---------------------------------------------------------------------------
# harness imports (read-only)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(BENCH_DIR))

from roughcut_bench import calibrate, partial as partial_mod, pooling  # noqa: E402
from roughcut_bench import episodes as episodes_mod  # noqa: E402
from roughcut_bench import ranges as ranges_mod  # noqa: E402
from roughcut_bench import replay, scoring, sentence_scoring, word_scoring  # noqa: E402
from roughcut_bench.experiments import _human_kept_frames, _strip_legacy_fields  # noqa: E402
from roughcut_bench.prepass import _merge, subtract  # noqa: E402

# Unused directly, but named in the design as the modules the numbers must come
# from; referencing them here keeps the import list honest and linters quiet.
_HARNESS_MODULES = (scoring, sentence_scoring, word_scoring)


def _load_jev_bench():
    """Import ``scripts/jev_real/roughcut_bench.py`` under a non-clashing name.

    That file is this repo's Jev benchmark runner, not the harness package of
    the same name. Importing it installs ``_cached_answer_key`` over
    ``episodes.episode_answer_key``, which is the whole reason it is loaded:
    the stock loader re-extracts the answer key from the Q: drive and rewrites
    the cache file under solar-sailer.
    """
    if "jev_roughcut_bench" in sys.modules:
        return sys.modules["jev_roughcut_bench"]
    path = Path(__file__).resolve().parent / "roughcut_bench.py"
    spec = importlib.util.spec_from_file_location("jev_roughcut_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["jev_roughcut_bench"] = module
    spec.loader.exec_module(module)
    return module


_JEV_BENCH = _load_jev_bench()
assert episodes_mod.episode_answer_key is _JEV_BENCH._cached_answer_key


# ---------------------------------------------------------------------------
# episode loading
# ---------------------------------------------------------------------------

_EPISODES = {}


def load_episode(name):
    """Everything scoring needs for one episode, built once and cached.

    Returns a dict with the episode config, the pristine corpus ``sentences``
    (never annotated: every scoring call works on copies), the loaded
    ``transcript``, the midpoint word ``tiles`` that map word ids to frames,
    ``frame_rate``, ``media_name``, the ``preflight`` context carrying the
    human answer key and the word units, and ``retake_flags`` (the corpus
    ``is_retake`` per sentence id, kept so a caller can reproduce the
    production retake arm).
    """
    if name in _EPISODES:
        return _EPISODES[name]

    episode = episodes_mod.load_episode(name, str(EPISODES_DIR))
    sentences = episodes_mod.load_corpus(episode)
    _strip_legacy_fields(sentences)
    preflight = episodes_mod.build_episode_preflight(episode, sentences,
                                                     str(EPISODES_DIR))
    corpus_path = episode["corpus"]
    transcript_path = partial_mod.transcript_path_for(corpus_path)
    with open(transcript_path, encoding="utf-8") as handle:
        transcript = json.load(handle)
    tiles = partial_mod.build_word_tiles(sentences, transcript_path,
                                         transcript=transcript)
    retake_flags = {s["id"]: bool(s.get("is_retake")) for s in sentences}

    _EPISODES[name] = {
        "name": name,
        "episode": episode,
        "corpus_path": corpus_path,
        "transcript_path": transcript_path,
        "transcript": transcript,
        "sentences": sentences,
        "tiles": tiles,
        "preflight": preflight,
        "frame_rate": episode.get("frame_rate") or 24,
        "media_name": preflight.media_name,
        "retake_flags": retake_flags,
        "dialogue_frames": preflight.total_dialogue,
        "human_kept_frames": _human_kept_frames(preflight.human_by_media),
    }
    return _EPISODES[name]


# ---------------------------------------------------------------------------
# decisions -> annotated sentences
# ---------------------------------------------------------------------------


def _annotate(data, decisions, warnings):
    """Copy the corpus and write one decision set onto it.

    The corpus ``is_retake`` flag is dropped and replaced by the decision's
    ``cut_retake``, so ``ranges.kept_segments_from_score`` cuts exactly the
    sentences the caller asked to cut and nothing the production module
    flagged leaks in. ``keep_words`` go through the harness's own validator and
    then ``partial.attach_keep_ranges``.
    """
    sentences = [dict(s) for s in data["sentences"]]
    tiles = data["tiles"]
    ratings = {}
    missing = []
    for sentence in sentences:
        sid = sentence["id"]
        sentence.pop("roughcut_keep_ranges", None)
        sentence.pop("is_retake", None)
        decision = decisions.get(sid)
        if decision is None:
            missing.append(sid)
            continue
        sentence["roughcut_score"] = float(decision["score"])
        sentence["is_retake"] = bool(decision.get("cut_retake"))
        allowed = {wid for wid, _text, _sub in tiles.get(sid) or []}
        keep_words = partial_mod._coerce_ranges(
            decision.get("keep_words"), sid, allowed, warnings)
        ratings[sid] = {"keep_words": keep_words}
    if missing:
        raise ValueError(
            f"{data['name']}: {len(missing)} sentence ids missing from the "
            f"decisions: {missing[:10]}")
    n_partial, _n = partial_mod.attach_keep_ranges(sentences, ratings, tiles)
    return sentences, n_partial


def removal_frames(removals, parts=("umm", "silence")):
    """Merged frame ranges for the named deterministic layers.

    ``removals`` is ``{"umm": [(s, e), ...], "silence": [...]}`` in corpus
    frame space; missing keys contribute nothing, so passing only ``silence``
    reproduces the ladder's ``+ delete silence`` column.
    """
    if not removals:
        return []
    return _merge([tuple(interval)
                   for part in parts
                   for interval in (removals.get(part) or [])])


def _apply_removals(sentences, threshold, removed, gate=True):
    """Subtract ``removed`` frames from every kept sentence's frames.

    A line-for-line copy of ``model_plus_deterministic.apply_layer``: the
    ranges come off the MODEL's side, a sentence that loses every frame is cut
    whole, and a sentence the subtraction does not touch is left alone so the
    partial counts stay honest.

    ``gate=False`` drops the "only sentences kept at ``threshold``" guard. That
    is what the threshold sweep needs, and it changes no kept segment at any
    threshold: the gate only skips sentences that contribute nothing anyway.
    """
    if not removed:
        return 0, 0
    field = ranges_mod.pick_timecode_field(sentences)
    n_trimmed = n_emptied = 0
    for sentence in sentences:
        if gate and (sentence.get("is_retake")
                     or sentence["roughcut_score"] < threshold):
            continue
        current = [tuple(iv) for iv in
                   sentence.get("roughcut_keep_ranges") or [tuple(sentence[field])]]
        kept = subtract(current, removed)
        if kept == current:
            continue
        if not kept:
            sentence["roughcut_score"] = CUT
            sentence.pop("roughcut_keep_ranges", None)
            n_emptied += 1
            continue
        sentence["roughcut_keep_ranges"] = [list(iv) for iv in kept]
        n_trimmed += 1
    return n_trimmed, n_emptied


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _metrics_from_level(data, sentences, level, threshold, extra):
    """Shape one ``replay.score_sentences`` level report into a result row."""
    sp = level["sp"] or {}
    word = level["word"] or {}
    kept_frames = calibrate._kept_frames(
        ranges_mod.kept_segments_from_score(
            sentences, threshold, data["media_name"], field="roughcut_score"))
    human_kept = data["human_kept_frames"]
    row = {
        "episode": data["name"],
        "threshold": threshold,
        "sentence_points": level["sp_grade"],
        "sentence_points_raw": sp.get("raw_score"),
        "sentence_points_penalty": bool(sp.get("penalty_applied")),
        "n_sentences": sp.get("n_sentences", len(sentences)),
        "n_partial_human": sp.get("n_partial_human"),
        "n_partial_model": sp.get("n_partial_model"),
        # The full/partial/removed cross-tab and the partial-vs-partial branch
        # split ``sentence_scoring.score_sentences_points`` already computes.
        # Passed straight through so a caller can recount the score without
        # reaching into the level report.
        "pair_counts": sp.get("pair_counts"),
        "partial_branch_counts": sp.get("partial_branch_counts"),
        "word_score": level["word_grade"],
        "word_count": word.get("count", 0),
        "grade": level["grade"],
        "frame_match": level["fair_metrics"]["super_score"],
        "dialogue_frames": data["dialogue_frames"],
        "kept_frames": kept_frames,
        "human_kept_frames": human_kept,
        "kept_ratio": (kept_frames / human_kept) if human_kept else None,
    }
    row.update(extra)
    return row


def _score_annotated(data, sentences, threshold, extra):
    """Score already-annotated sentences at one threshold, Neutral level only."""
    report = replay.score_sentences(
        None, sentences, {NEUTRAL_LEVEL: threshold}, levels=[NEUTRAL_LEVEL],
        field="roughcut_score", precomputed=data["preflight"],
    )
    level = report["levels"][str(NEUTRAL_LEVEL)]
    return _metrics_from_level(data, sentences, level, threshold, extra), level


def score_episode(name, decisions, threshold, removals=None):
    """Harness metrics for one episode's decision set at one keep threshold.

    ``decisions`` maps every corpus sentence id to
    ``{"score": float, "keep_words": [[first_word_id, last_word_id], ...] | None,
    "cut_retake": bool}``. A sentence is kept iff ``cut_retake`` is false and
    ``score >= threshold``; a kept sentence with ``keep_words`` contributes only
    those word runs, otherwise its whole timecode.

    ``removals``, when given, is ``{"umm": [(start, end), ...], "silence": [...]}``
    in corpus frame space. Those frames are subtracted from the model's kept
    ranges before scoring, the way the ladder layers the shipping modules onto
    every arm.
    """
    data = load_episode(name)
    warnings = []
    sentences, n_partial = _annotate(data, decisions, warnings)
    removed = removal_frames(removals)
    n_trimmed, n_emptied = _apply_removals(sentences, threshold, removed)
    extra = {
        "n_partial_decisions": n_partial,
        "removal_frames": sum(end - start for start, end in removed),
        "sentences_trimmed_by_removals": n_trimmed,
        "sentences_cut_by_removals": n_emptied,
        "warnings": warnings,
    }
    row, _level = _score_annotated(data, sentences, threshold, extra)
    return row


def episode_report(name, decisions, threshold_map, levels=(NEUTRAL_LEVEL,),
                   removals=None, gate_threshold=None):
    """The harness's whole report for one episode: the ceiling and every level.

    ``score_episode`` keeps the dozen numbers a sweep needs and throws the rest
    of ``replay.score_sentences``'s report away. Writing a result JSON in the
    harness's own shape needs all of it: the ``ceiling`` block, the media name
    and offset, and per level the frame, word and sentence-points sub-reports.
    The annotation and the removal layering are ``score_episode``'s; only the
    return value is wider.

    ``threshold_map`` is ``{level: threshold}`` and must cover every entry in
    ``levels``. ``gate_threshold`` is the threshold the removal layer is gated
    at, which matters only when ``removals`` is given; it defaults to the
    Neutral entry, or the lowest threshold in the map when Neutral is absent.

    The report carries the same ``n_partial_decisions``, ``removal_frames``,
    ``sentences_trimmed_by_removals``, ``sentences_cut_by_removals`` and
    ``warnings`` extras ``score_episode`` returns.
    """
    data = load_episode(name)
    warnings = []
    sentences, n_partial = _annotate(data, decisions, warnings)
    removed = removal_frames(removals)
    if gate_threshold is None:
        gate_threshold = threshold_map.get(NEUTRAL_LEVEL, min(threshold_map.values()))
    n_trimmed, n_emptied = _apply_removals(sentences, gate_threshold, removed)
    report = replay.score_sentences(
        None, sentences, threshold_map, levels=list(levels),
        field="roughcut_score", precomputed=data["preflight"],
    )
    report.update({
        "episode": name,
        "n_partial_decisions": n_partial,
        "removal_frames": sum(end - start for start, end in removed),
        "sentences_trimmed_by_removals": n_trimmed,
        "sentences_cut_by_removals": n_emptied,
        "warnings": warnings,
    })
    return report


def sentence_states_for(name, decisions, threshold, removals=None):
    """``(human_states, model_states)`` behind one ``score_episode`` call.

    Both sides are ``{sentence_id: (state, runs)}`` straight out of the
    harness's ``sentence_scoring.sentence_states``: the same two maps SENTENCE
    POINTS reads, so "the editor kept it" and "the arm kept it" mean here what
    they mean in the metric. ``runs`` are word ORDER positions inside the
    sentence, ``(first, end_exclusive)``, or ``None`` for a whole or removed
    sentence. ``removals`` is layered exactly as ``score_episode`` layers it.

    Exists so a caller can ask which sentences a pair count is made of without
    reaching into this module's private annotation step.
    """
    data = load_episode(name)
    sentences, _n = _annotate(data, decisions, [])
    _apply_removals(sentences, threshold, removal_frames(removals))
    pre = data["preflight"]
    human = sentence_scoring.sentence_states(
        pre.word_units, pre.human_by_media, pre.media_name, sentences,
        human_rule="majority", offset=pre.offset)
    model_by_media = ranges_mod.kept_segments_from_score(
        sentences, threshold, pre.media_name, field="roughcut_score")
    shifted = {media: [(start + pre.offset, end + pre.offset)
                       for start, end in segments]
               for media, segments in model_by_media.items()}
    model = sentence_scoring.sentence_states(
        pre.word_units, shifted, pre.media_name, sentences, offset=pre.offset)
    return human, model


# ---------------------------------------------------------------------------
# threshold calibration
# ---------------------------------------------------------------------------


def calibrate_threshold(episodes_decisions, removals=None):
    """Pooled keep threshold for one arm, plus the metrics at it.

    The sweep is the harness's: every episode is scored at 0.0..5.0 in 0.1
    steps (``calibrate.sweep``), the rows are pooled by dialogue frames
    (``pooling.pool_sweep_rows``) and ``calibrate.pick_operating_points``
    returns the Neutral threshold, the one that maximises the retired GRADE.
    That is the same call sequence ``score_arm`` in
    ``scripts/jev_real/roughcut_bench.py`` runs for whole-sentence arms; the
    only addition is that the sentences carry word ranges and, optionally, the
    deterministic layers.

    ``episodes_decisions`` is ``{episode_name: decisions}``. ``removals`` is
    ``{episode_name: {"umm": [...], "silence": [...]}}`` or ``None``.

    Returns ``{"threshold", "operating_points", "episodes", "pooled"}``.
    Pooling weights match the ladder: sentence count for SENTENCE POINTS, word
    count for WORD SCORE, dialogue frames for GRADE and frame match.
    """
    if not episodes_decisions:
        raise ValueError("calibrate_threshold: no episodes")

    loaded = []
    for name, decisions in episodes_decisions.items():
        data = load_episode(name)
        warnings = []
        sentences, n_partial = _annotate(data, decisions, warnings)
        removed = removal_frames((removals or {}).get(name))
        # Ungated: the sweep visits every threshold, so the layer cannot be
        # applied relative to one of them.
        _apply_removals(sentences, 0.0, removed, gate=False)
        loaded.append((data, sentences, removed, n_partial, warnings))

    thresholds = calibrate.default_thresholds()
    sweeps = [calibrate.sweep(None, sentences, thresholds=thresholds,
                              field="roughcut_score", precomputed=data["preflight"])
              for data, sentences, _r, _n, _w in loaded]
    weights = [data["dialogue_frames"] for data, *_rest in loaded]
    pooled_rows = pooling.pool_sweep_rows(sweeps, weights)
    human_kept = sum(data["human_kept_frames"] for data, *_rest in loaded)
    ops = calibrate.pick_operating_points(pooled_rows, human_kept)
    threshold = ops["neutral_threshold"]

    per_episode, level_reports = {}, []
    for data, sentences, removed, n_partial, warnings in loaded:
        extra = {
            "n_partial_decisions": n_partial,
            "removal_frames": sum(end - start for start, end in removed),
            "warnings": warnings,
        }
        row, level = _score_annotated(data, sentences, threshold, extra)
        per_episode[data["name"]] = row
        level_reports.append(level)

    sentence_counts = [row["n_sentences"] for row in per_episode.values()]
    pooled = {
        "threshold": threshold,
        "episodes": len(per_episode),
        "sentence_count": sum(sentence_counts),
        "word_count": sum(row["word_count"] for row in per_episode.values()),
        "dialogue_frames": sum(weights),
        "sentence_points": pooling.pooled_sp_grade(level_reports),
        "sentence_points_raw": pooling.weighted_mean(
            [row["sentence_points_raw"] for row in per_episode.values()],
            sentence_counts),
        "word_score": pooling.pooled_word_grade(level_reports),
        "grade": pooling.weighted_mean(
            [row["grade"] for row in per_episode.values()], weights),
        "frame_match": pooling.weighted_mean(
            [row["frame_match"] for row in per_episode.values()], weights),
        "kept_ratio": ops["achieved_ratios"][NEUTRAL_LEVEL],
    }
    return {
        "threshold": threshold,
        "operating_points": ops,
        "episodes": per_episode,
        "pooled": pooled,
    }


# ---------------------------------------------------------------------------
# deterministic removals
# ---------------------------------------------------------------------------


def removals_cache_path(name):
    """Where this repo caches one episode's detected removal ranges."""
    return REMOVALS_DIR / f"{name}.json"


def _umm_word_ids(transcript, umm_ranges, frame_rate):
    """Word ids whose frames are more than half inside an um cut.

    The same majority rule the word metric uses one level down, so the words
    stripped from the transcript Jev reads are exactly the words the Um
    Removal module took out of the audio.
    """
    merged = _merge([tuple(interval) for interval in umm_ranges])
    removed = []
    for word in transcript["word_segments"]:
        start, end = partial_mod._word_bounds(word, frame_rate)
        span = end - start
        if span <= 0:
            inside = any(s <= start < e for s, e in merged)
            if inside:
                removed.append(word["id"])
            continue
        covered = sum(max(0, min(end, e) - max(start, s)) for s, e in merged)
        if covered * 2 > span:
            removed.append(word["id"])
    return removed


def detect_removals_for(name, refresh=False):
    """Run the shipping Um Removal and Delete Silence detectors for one episode.

    Calls ``detect_removals(corpus_path, transcript, frame_rate)`` from
    ``benchmarks/roughcut/scripts/deterministic_baseline.py`` (imported
    read-only) and caches the answer under
    ``docs/jev-real/removals/<episode>.json`` in this repo. An existing cache
    is returned untouched unless ``refresh`` is set.

    The detector needs the conform WAV under
    ``corpus/<episode>/cache/*_conform.wav``; it fails loudly when the cache
    directory is missing. It costs about 12 seconds per episode, which is why
    the result is cached.
    """
    path = removals_cache_path(name)
    if path.exists() and not refresh:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["umm"] = [tuple(iv) for iv in payload["umm"]]
        payload["silence"] = [tuple(iv) for iv in payload["silence"]]
        payload["cached"] = True
        return payload

    for editor_dir in EDITOR_DIRS:
        if editor_dir.is_dir() and str(editor_dir) not in sys.path:
            sys.path.insert(0, str(editor_dir))
    if str(BENCH_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_SCRIPTS_DIR))
    from deterministic_baseline import detect_removals  # noqa: E402

    data = load_episode(name)
    result = detect_removals(data["corpus_path"], data["transcript"],
                             data["frame_rate"])
    word_ids = _umm_word_ids(data["transcript"], result["umm"], data["frame_rate"])
    payload = {
        "episode": name,
        "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "benchmarks/roughcut/scripts/deterministic_baseline.detect_removals",
        "frame_rate": data["frame_rate"],
        "umm": [list(iv) for iv in result["umm"]],
        "silence": [list(iv) for iv in result["silence"]],
        "umm_word_ids": word_ids,
        "stats": result["stats"],
    }
    REMOVALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    payload["umm"] = [tuple(iv) for iv in payload["umm"]]
    payload["silence"] = [tuple(iv) for iv in payload["silence"]]
    payload["cached"] = False
    return payload


# ---------------------------------------------------------------------------
# validation gate
# ---------------------------------------------------------------------------


def decisions_from_run_ratings(name, run_ratings):
    """Archived ``run_ratings`` -> this module's decision dict.

    The archived partial arms return ``{"score", "category", "keep_words"}``
    keyed by sentence id, and their cut was scored with the CORPUS retake
    flags, so ``cut_retake`` takes the corpus flag.
    """
    data = load_episode(name)
    flags = data["retake_flags"]
    decisions = {}
    for raw_id, rating in run_ratings.items():
        sid = int(raw_id)
        decisions[sid] = {
            "score": float(rating["score"]),
            "keep_words": rating.get("keep_words"),
            "cut_retake": flags.get(sid, False),
        }
    return decisions


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}"


def _row(label, mine, published, failures):
    delta = None if (mine is None or published is None) else (mine - published)
    ok = delta is not None and abs(delta) <= VALIDATION_TOLERANCE
    if not ok:
        failures.append(label)
    return (f"| {label} | {_pct(mine)} | {_pct(published)} | "
            f"{'—' if delta is None else f'{delta * 100:+.4f}'} | "
            f"{'ok' if ok else 'MISMATCH'} |")


def validate(episode_names=None):
    """Reproduce the archived Luna partial arm through this module.

    Plain pass: the saved ``run_ratings`` at that file's own Neutral threshold
    against its published per-episode ``sp_grade`` and ``word_grade``.
    Layered pass: the same decisions with the detected ``silence`` and
    ``umm + silence`` ranges subtracted, against the per-episode ``sp_grades``
    and ``word_grades`` ``results/2026-09-11-model-plus-deterministic.json``
    stores for the same arm.

    Returns the number of failed comparisons.
    """
    names = episode_names or VALIDATION_EPISODES
    with open(VALIDATION_RESULT, encoding="utf-8") as handle:
        doc = json.load(handle)
    threshold = float(doc["neutral_threshold"])
    level_key = str(doc["neutral_level"])
    with open(VALIDATION_LAYERED, encoding="utf-8") as handle:
        layered_doc = json.load(handle)
    layered = layered_doc["arms"][VALIDATION_ARM]["episodes"]

    failures = []
    lines = [
        f"# Validation against {VALIDATION_RESULT.name}",
        "",
        f"Neutral level {level_key}, threshold {threshold}. "
        f"cut_retake = the corpus is_retake flag (that arm's own override).",
        "",
        "| metric | mine | published | delta x100 | |",
        "|---|---:|---:|---:|---|",
    ]

    for name in names:
        saved = doc["episodes"][name]
        published = saved["levels"][level_key]
        decisions = decisions_from_run_ratings(name, saved["run_ratings"][0])

        plain = score_episode(name, decisions, threshold)
        lines.append(_row(f"{name} SENTENCE POINTS", plain["sentence_points"],
                          published["sp_grade"], failures))
        lines.append(_row(f"{name} WORD SCORE", plain["word_score"],
                          published["word_grade"], failures))
        lines.append(_row(f"{name} GRADE", plain["grade"],
                          published["grade"], failures))

        removals = detect_removals_for(name)
        published_layers = layered.get(name)
        if not published_layers:
            lines.append(f"| {name} layered | — | not in "
                         f"{VALIDATION_LAYERED.name} | — | skipped |")
            continue
        for key, parts in (("silence", ("silence",)),
                           ("umm_silence", ("umm", "silence"))):
            subset = {part: removals[part] for part in parts}
            layered_row = score_episode(name, decisions, threshold, removals=subset)
            lines.append(_row(f"{name} + {key} SENTENCE POINTS",
                              layered_row["sentence_points"],
                              published_layers["sp_grades"][key], failures))
            lines.append(_row(f"{name} + {key} WORD SCORE",
                              layered_row["word_score"],
                              published_layers["word_grades"][key], failures))
            lines.append(_row(f"{name} + {key} GRADE", layered_row["grade"],
                              published_layers["grades"][key], failures))

    lines.append("")
    if failures:
        lines.append(f"FAILED: {len(failures)} comparison(s) off by more than "
                     f"0.005 x100: {failures}")
    else:
        lines.append("All comparisons match to two decimals (x100).")
    print("\n".join(lines))
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validate", action="store_true",
                        help="reproduce the archived Luna partial arm")
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="episodes for --validate (default: the three in the spec)")
    parser.add_argument("--removals", nargs="*", default=None,
                        help="run the deterministic detectors for these episodes "
                             "and cache the result")
    parser.add_argument("--refresh", action="store_true",
                        help="re-detect removals even when a cache exists")
    args = parser.parse_args()

    if args.removals:
        for name in args.removals:
            payload = detect_removals_for(name, refresh=args.refresh)
            source = "cache" if payload.get("cached") else "detector"
            print(f"{name}: {source}; "
                  f"{len(payload['umm'])} um ranges "
                  f"({payload['stats']['umm_frames']} frames, "
                  f"{len(payload['umm_word_ids'])} words), "
                  f"{len(payload['silence'])} silence ranges "
                  f"({payload['stats']['silence_frames']} frames) -> "
                  f"{removals_cache_path(name)}")

    if args.validate:
        return validate(args.episodes)
    if not args.removals:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
