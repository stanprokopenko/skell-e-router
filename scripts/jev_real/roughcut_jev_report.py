"""Rescore a Jev rough-cut run and write the developer-facing notes.

Reads the three files ``scripts/jev_real/roughcut_jev.py`` wrote
(``docs/jev-real/<name>-decisions.jsonl``, ``-requests.jsonl``,
``-timing.json``), rebuilds every arm's decisions offline and scores them
through ``scripts/jev_real/roughcut_partial_scoring.py``. No model calls, no
detector runs, nothing written outside ``docs/jev-real``.

Arms
----
``jev_a`` is rebuilt once per ``--t-trim`` threshold. ``jev_b``,
``jev_b_moduleretakes`` and ``jev_b_notrim`` come out of the decisions file as
written. Each arm is calibrated and scored twice: plain, and with the step 0
``umm``/``silence`` frames layered on (``--no-modules`` skips the second).

Variant A rebuild, and what the decisions file cannot support
------------------------------------------------------------
The decisions file stores ``first_choice``/``last_choice`` (the TOP option,
which may be ``whole``) and ``first_p_whole``/``last_p_whole``. It does NOT
store the full ``first_probabilities``/``last_probabilities`` distribution, so
the top NON-whole word is unknown whenever ``whole`` won. The sweep therefore
reads: trim that side iff the top option is a word AND P(whole) is below the
threshold. Lowering the threshold withholds trims the run emitted; no threshold
can add a trim to a sentence where ``whole`` was the top option. A threshold of
1.0 reproduces the ``jev_a`` rows exactly, and the rebuild is checked against
them at load time.

Usage::

  python scripts/jev_real/roughcut_jev_report.py --in roughcut-jev --out roughcut-jev
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"
REFERENCE_MD = Path(r"D:\solar-sailer\benchmarks\roughcut\results"
                    r"\2026-09-11-model-plus-deterministic.md")
REFERENCE_JSON = REFERENCE_MD.with_suffix(".json")

sys.path.insert(0, str(HERE))

# The scoring module installs the cache-only answer-key loader and puts the
# harness on sys.path; it has to be imported before anything that touches the
# corpus.
import roughcut_partial_scoring as scoring_mod  # noqa: E402
from roughcut_partial_scoring import (  # noqa: E402
    calibrate_threshold, detect_removals_for, load_episode, removals_cache_path,
    score_episode,
)
import roughcut_jev as pipeline  # noqa: E402

sentence_scoring = scoring_mod.sentence_scoring

BASE_ARMS = ["jev_a", "jev_b", "jev_b_moduleretakes", "jev_b_notrim"]
SWEPT_ARM = "jev_a"
DEFAULT_T_TRIM = [0.3, 0.5, 0.7, 0.9]
PASSES = ["retake", "sentence", "trim_pick"]

#: Published ladder arms quoted at the bottom of the notes, in the order the
#: brief names them. Keys are ``arms`` keys in the reference JSON.
REFERENCE_ARMS = [
    ("luna-chapters-rules5", "best Luna chapters"),
    ("opus5-cc-agentic", "shipped Opus agentic"),
    ("luna-single-call", "Luna single call"),
]
#: Layer keys in the reference JSON: no modules, and um removal + delete silence.
REF_PLAIN, REF_LAYERED = "published", "umm_silence"


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def pct(value):
    """A metric on the x100 scale, two decimals."""
    return "n/a" if value is None else f"{value * 100:.2f}"


def num(value, places=2):
    return "n/a" if value is None else f"{value:.{places}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return out


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(name):
    """The three run files, or a loud failure naming the one that is missing."""
    paths = {
        "decisions": OUT_DIR / f"{name}-decisions.jsonl",
        "requests": OUT_DIR / f"{name}-requests.jsonl",
        "timing": OUT_DIR / f"{name}-timing.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing input file(s): {missing}")
    with paths["timing"].open(encoding="utf-8") as handle:
        timing = json.load(handle)
    return (read_jsonl(paths["decisions"]), read_jsonl(paths["requests"]),
            timing, {k: str(v) for k, v in paths.items()})


def index_decisions(rows):
    """``{arm: {episode: {sentence_id: row}}}`` plus the episode order seen."""
    by_arm = defaultdict(lambda: defaultdict(dict))
    order = []
    for row in rows:
        episode = row["episode"]
        if episode not in order:
            order.append(episode)
        by_arm[row["arm"]][episode][row["id"]] = row
    return by_arm, order


# ---------------------------------------------------------------------------
# decision rebuild
# ---------------------------------------------------------------------------

def word_ids_by_sentence(episode_name):
    """``{sentence_id: [word_id, ...]}`` exactly as the run's trim options saw it.

    Reuses the pipeline's own loader so the um-stripped word list, and with it
    the meaning of "the sentence's first word", matches what step 2 asked about.
    """
    data = pipeline.load_episode_data(episode_name)
    return {sid: [word["w"] for word in words]
            for sid, words in data["words"].items()}


def variant_a_keep_words(word_ids, row, t_trim):
    """``keep_words`` for one sentence at one trim-trigger threshold.

    Trims a side only when the top option on that side is a word (not ``whole``)
    and P(whole) is under ``t_trim``. A trim that would keep the whole sentence,
    or nothing at all, is dropped.
    """
    if not word_ids:
        return None

    def edge(choice, p_whole):
        if choice is None or choice == "whole" or p_whole is None:
            return None
        if p_whole >= t_trim:
            return None
        try:
            return word_ids.index(int(choice))
        except (ValueError, TypeError):
            return None

    lo = edge(row.get("first_choice"), row.get("first_p_whole"))
    hi = edge(row.get("last_choice"), row.get("last_p_whole"))
    if lo is None and hi is None:
        return None
    lo = 0 if lo is None else lo
    hi = len(word_ids) - 1 if hi is None else hi
    if lo > hi or (lo == 0 and hi == len(word_ids) - 1):
        return None
    return [[word_ids[lo], word_ids[hi]]]


def build_arm_decisions(by_arm, episodes, words, arm, t_trim, missing_score):
    """``{episode: {sid: {score, keep_words, cut_retake}}}`` for one arm.

    ``t_trim`` is ignored for every arm but ``jev_a``. A sentence the run never
    got an answer for (``score`` is null) takes ``missing_score``; the count
    comes back so the notes can say how many there were.
    """
    out, n_missing = {}, 0
    for episode in episodes:
        rows = by_arm[arm][episode]
        decisions = {}
        for sid, row in rows.items():
            score = row["score"]
            if score is None:
                score = missing_score
                n_missing += 1
            if arm == SWEPT_ARM:
                keep = variant_a_keep_words(words[episode].get(sid, []), row, t_trim)
            else:
                keep = row["keep_words"]
            decisions[sid] = {"score": float(score), "keep_words": keep,
                              "cut_retake": bool(row["cut_retake"])}
        out[episode] = decisions
    return out, n_missing


def check_rebuild(by_arm, episodes, words):
    """Sentences where the t=1.0 rebuild disagrees with the run's own jev_a rows.

    A non-zero count means this script's reading of the trim fields has drifted
    from ``roughcut_jev.variant_a_keep_words``; every number in the jev_a rows
    below is then suspect.
    """
    mismatches = []
    if SWEPT_ARM not in by_arm:
        return mismatches
    for episode in episodes:
        for sid, row in by_arm[SWEPT_ARM][episode].items():
            rebuilt = variant_a_keep_words(words[episode].get(sid, []), row, 1.0)
            if rebuilt != row["keep_words"]:
                mismatches.append((episode, sid, row["keep_words"], rebuilt))
    return mismatches


# ---------------------------------------------------------------------------
# removals
# ---------------------------------------------------------------------------

def load_removals(episodes):
    """Cached removal ranges per episode; never runs the detector.

    An episode with no cache is dropped from the layered pass with a warning,
    because running the detector here would cost ~12 s per episode and write to
    the cache behind the reader's back.
    """
    removals, skipped = {}, []
    for episode in episodes:
        if not removals_cache_path(episode).exists():
            skipped.append(episode)
            continue
        payload = detect_removals_for(episode)
        removals[episode] = {"umm": payload["umm"], "silence": payload["silence"]}
    return removals, skipped


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_arm(decisions, removals=None):
    """Calibrate one arm's pooled keep threshold and score every episode at it."""
    return calibrate_threshold(decisions, removals=removals)


def metric_row(label, row, seconds):
    return [label, pct(row["sentence_points"]), pct(row["sentence_points_raw"]),
            pct(row["word_score"]), pct(row["grade"]), pct(row["frame_match"]),
            pct(row["kept_ratio"]),
            "yes" if row.get("sentence_points_penalty") else "no",
            num(seconds)]


METRIC_HEADER = ["episode", "SENTENCE POINTS", "SP raw", "WORD SCORE", "GRADE",
                 "frame match", "kept ratio", "penalty", "s/episode"]
POOLED_HEADER = ["arm", "threshold", "SENTENCE POINTS", "SP raw", "WORD SCORE",
                 "GRADE", "frame match", "kept ratio", "s/episode"]


def pooled_row(label, result, seconds_per_episode):
    pooled = result["pooled"]
    return [label, num(result["threshold"], 2), pct(pooled["sentence_points"]),
            pct(pooled["sentence_points_raw"]), pct(pooled["word_score"]),
            pct(pooled["grade"]), pct(pooled["frame_match"]),
            pct(pooled["kept_ratio"]), num(seconds_per_episode)]


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def latency_rows(timing, requests, episodes):
    """One row per episode: wall clock per pass, requests, tokens, cost, retries."""
    per_episode = {}
    by_episode = defaultdict(list)
    for row in requests:
        by_episode[row["episode"]].append(row)
    for episode in episodes:
        passes = (timing.get("episodes", {}).get(episode, {}) or {}).get("passes", {})
        rows = by_episode.get(episode, [])
        per_episode[episode] = {
            "pass_wall_clock_s": {p: (passes.get(p) or {}).get("wall_clock_s")
                                  for p in PASSES},
            "wall_clock_s": (timing.get("episodes", {}).get(episode, {}) or {})
                            .get("episode_wall_clock_s"),
            "requests": len(rows),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
            "cost_usd": round(sum(r.get("cost") or 0.0 for r in rows), 6),
            "retries": sum(1 for r in rows if (r.get("attempt") or 1) > 1),
            "errors": sum(1 for r in rows if r.get("error")),
            "windowed_requests": sum(1 for r in rows if r.get("window_used")),
            "sentences": (timing.get("episodes", {}).get(episode, {}) or {})
                         .get("sentences"),
        }
    return per_episode


def seconds_for(latency, episode):
    return (latency.get(episode) or {}).get("wall_clock_s")


def mean_seconds(latency, episodes):
    values = [seconds_for(latency, e) for e in episodes]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# retake table
# ---------------------------------------------------------------------------

def human_states(episode):
    """``{sentence_id: state}`` for the editor's own cut: full, partial, removed.

    Straight out of the harness's ``sentence_states`` on the human side, the
    same call ``sentence_scoring.score_arm_episode`` makes, so "the editor kept
    it" here means exactly what it means in the SENTENCE POINTS metric.
    """
    data = load_episode(episode)
    pre = data["preflight"]
    states = sentence_scoring.sentence_states(
        pre.word_units, pre.human_by_media, pre.media_name, data["sentences"],
        human_rule="majority", offset=pre.offset)
    return {sid: state for sid, (state, _runs) in states.items()}


def retake_stats(by_arm, episodes):
    """Per-episode retake-pass counts and the editor's verdict on disagreements.

    Reads any arm that carries Jev's own retake cut: jev_b when the trim-pick
    pass ran, jev_a otherwise. Both stamp ``cut_retake`` from the same step 1
    decision, so the counts are identical either way. Returns nothing when the
    file has neither that arm or the module-flag arm.
    """
    jev_arm = next((a for a in ("jev_b", "jev_a", "jev_b_notrim") if a in by_arm), None)
    if jev_arm is None or "jev_b_moduleretakes" not in by_arm:
        return {}
    stats = {}
    for episode in episodes:
        jev_rows = by_arm[jev_arm][episode]
        module_rows = by_arm["jev_b_moduleretakes"][episode]
        states = human_states(episode)

        groups, not_real, fallback = {}, set(), set()
        for sid, row in jev_rows.items():
            gid = row.get("retake_group")
            if gid is None:
                continue
            groups.setdefault(gid, []).append(sid)
            if row.get("retake_source") == "module_fallback":
                fallback.add(gid)
            real = row.get("retake_real")
            if real is not None and real < pipeline.REAL_RETAKE_CUT:
                not_real.add(gid)

        jev_cut = {sid for sid, row in jev_rows.items() if row["cut_retake"]}
        module_cut = {sid for sid, row in module_rows.items() if row["cut_retake"]}
        jev_only = sorted(jev_cut - module_cut)
        module_only = sorted(module_cut - jev_cut)

        def verdict(sids):
            counts = Counter(states.get(sid, "unknown") for sid in sids)
            return {"n": len(sids),
                    "editor_kept": counts["full"] + counts["partial"],
                    "editor_cut": counts["removed"],
                    "full": counts["full"], "partial": counts["partial"],
                    "removed": counts["removed"]}

        stats[episode] = {
            "groups": len(groups),
            "group_members": sum(len(v) for v in groups.values()),
            "not_real": len(not_real),
            "module_fallback": len(fallback),
            "losers_cut_jev": len(jev_cut),
            "losers_cut_module": len(module_cut),
            "agree": len(jev_cut & module_cut),
            "jev_only_cut": verdict(jev_only),
            "module_only_cut": verdict(module_only),
        }
    return stats


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------

def _weighted(pairs):
    num_, den = 0.0, 0.0
    for value, weight in pairs:
        if value is None or not weight:
            continue
        num_ += value * weight
        den += weight
    return (num_ / den) if den else None


def ladder(episodes):
    """The published reference arms, restricted to ``episodes`` when possible.

    Returns ``(rows, restricted, covered)``. ``restricted`` is false when at
    least one reported episode is absent from the reference file, in which case
    the file's own 18-episode pooled numbers are quoted instead.
    """
    if not REFERENCE_JSON.exists():
        return [], False, []
    with REFERENCE_JSON.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    arms = doc["arms"]

    counts = {}
    for arm in arms.values():
        for episode, entry in (arm.get("episodes") or {}).items():
            counts.setdefault(episode, {
                "sentence_count": entry.get("sentence_count"),
                "word_count": entry.get("word_count"),
                "dialogue_frames": entry.get("dialogue_frames"),
            })

    covered = [e for e in episodes if e in counts]
    restricted = len(covered) == len(episodes) and bool(covered)

    rows = []
    for key, label in REFERENCE_ARMS:
        arm = arms.get(key)
        if not arm:
            continue
        entry = {"key": key, "label": label,
                 "episodes": len(covered) if restricted else arm["episodes_scored"]}
        if restricted:
            per = arm["episodes"]
            for layer, tag in ((REF_PLAIN, "plain"), (REF_LAYERED, "layered")):
                entry[f"sentence_points_{tag}"] = _weighted(
                    [(per[e]["sp_grades"][layer], counts[e]["sentence_count"])
                     for e in covered])
                entry[f"word_score_{tag}"] = _weighted(
                    [(per[e]["word_grades"][layer], counts[e]["word_count"])
                     for e in covered])
                entry[f"grade_{tag}"] = _weighted(
                    [(per[e]["grades"][layer], counts[e]["dialogue_frames"])
                     for e in covered])
        else:
            for layer, tag in ((REF_PLAIN, "plain"), (REF_LAYERED, "layered")):
                entry[f"sentence_points_{tag}"] = arm["weighted_sp"][layer]
                entry[f"word_score_{tag}"] = arm["weighted_word"][layer]
                entry[f"grade_{tag}"] = arm["weighted"][layer]
        rows.append(entry)

    base_per = doc.get("baseline_episodes") or {}
    base = {"key": "deterministic-baseline",
            "label": "deterministic baseline (um removal + retakes + delete silence)",
            "episodes": len(covered) if restricted else len(base_per)}
    if restricted and all(e in base_per for e in covered):
        base["sentence_points_plain"] = _weighted(
            [(base_per[e]["sp_grade"], counts[e]["sentence_count"]) for e in covered])
        base["word_score_plain"] = _weighted(
            [(base_per[e]["word_grade"], counts[e]["word_count"]) for e in covered])
        base["grade_plain"] = _weighted(
            [(base_per[e]["grade"], counts[e]["dialogue_frames"]) for e in covered])
    else:
        base["sentence_points_plain"] = doc.get("baseline_weighted_sp")
        base["word_score_plain"] = doc.get("baseline_weighted_word")
        base["grade_plain"] = doc.get("baseline_weighted")
    # The baseline IS the modules; there is no separate layered column.
    base["sentence_points_layered"] = base["sentence_points_plain"]
    base["word_score_layered"] = base["word_score_plain"]
    base["grade_layered"] = base["grade_plain"]
    rows.append(base)
    return rows, restricted, covered


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

def write_notes(path, summary):
    lines = []
    run = summary["run"]
    episodes = summary["episodes"]
    latency = summary["latency"]
    lines.append("# Jev rough cut: run report")
    lines.append("")
    lines.append(f"Generated {summary['generated_utc']} by "
                 f"`scripts/jev_real/roughcut_jev_report.py` from "
                 f"`{run['input_name']}-decisions.jsonl` and its request and timing "
                 f"files. No model calls, no detector runs, $0. Every metric is x100, "
                 f"two decimals; the summary JSON next to this file keeps the raw "
                 f"0-to-1 values.")
    lines.append("")

    lines.append("## What was run")
    lines.append("")
    lines.append(f"Model {run['model']}, prompt version {run['prompt_version']}, "
                 f"concurrency {run['concurrency']}, pick-pass trigger t_trim "
                 f"{run['t_trim']} at run time. "
                 f"{len(episodes)} episode(s): {', '.join(episodes)}.")
    lines.append(f"{run['requests']} requests, {run['errors']} errored, "
                 f"{run['retries']} retried, {run['input_tokens']:,} input tokens, "
                 f"${run['cost_usd']:.4f}, {run['wall_clock_s']} s of wall clock in "
                 f"total.")
    if run.get("aborted"):
        lines.append(f"The run ABORTED: {run['aborted']}")
    lines.append(f"Sentences the run never got an answer for, scored at "
                 f"{run['missing_score']}: {run['missing_answers_per_arm']} per arm "
                 f"out of {run['sentences']}.")
    lines.append("")
    lines.append(f"Variant A is rebuilt per trim-trigger threshold from "
                 f"`first_choice`/`last_choice` and `first_p_whole`/`last_p_whole`. "
                 f"A side is trimmed only when its top option is a word and P(whole) "
                 f"is below the threshold. The decisions file does not store the full "
                 f"choice distribution, so no threshold can add a trim to a sentence "
                 f"where `whole` was the top option; the sweep only withholds trims "
                 f"the run emitted. Threshold 1.0 reproduces the run's own jev_a rows, "
                 f"and it does here on "
                 f"{run['rebuild_check']['checked']} sentences with "
                 f"{run['rebuild_check']['mismatches']} mismatches.")
    lines.append("")
    if summary["warnings"]:
        lines.append("Warnings from this report:")
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    for tag, title in (("plain", "Pooled, plain (the model's cut alone)"),
                       ("layered", "Pooled, with um removal and delete silence "
                                   "layered on")):
        arms = [a for a in summary["arms"] if a.get(tag)]
        if not arms:
            continue
        lines.append(f"## {title}")
        lines.append("")
        if tag == "layered" and summary["layered_episodes"] != episodes:
            lines.append(f"Layered on {len(summary['layered_episodes'])} episode(s) "
                         f"only: {', '.join(summary['layered_episodes']) or 'none'}. "
                         f"The rest have no cached removals and are excluded from "
                         f"this section, so it is not comparable to the plain "
                         f"section above.")
            lines.append("")
        rows = [pooled_row(a["label"], a[tag]["result"],
                           a[tag]["seconds_per_episode"]) for a in arms]
        lines.extend(table(POOLED_HEADER, rows))
        lines.append("")
        if tag == "plain" and summary["sweep_note"]:
            lines.append(summary["sweep_note"])
            lines.append("")

    for tag, title in (("plain", "Per episode, plain"),
                       ("layered", "Per episode, with modules")):
        arms = [a for a in summary["arms"] if a.get(tag)]
        if not arms:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for arm in arms:
            block = arm[tag]
            result = block["result"]
            lines.append(f"### {arm['label']} ({tag})")
            lines.append("")
            lines.append(f"Calibrated pooled keep threshold "
                         f"{num(result['threshold'], 2)}.")
            lines.append("")
            rows = [metric_row(name, result["episodes"][name],
                               seconds_for(latency, name))
                    for name in result["episodes"]]
            pooled = result["pooled"]
            rows.append(["**pooled**", pct(pooled["sentence_points"]),
                         pct(pooled["sentence_points_raw"]),
                         pct(pooled["word_score"]), pct(pooled["grade"]),
                         pct(pooled["frame_match"]), pct(pooled["kept_ratio"]), "—",
                         num(block["seconds_per_episode"])])
            lines.extend(table(METRIC_HEADER, rows))
            lines.append("")

    lines.append("## Latency, tokens and cost per episode")
    lines.append("")
    lines.append("Wall clock is measured around each pass at the run's concurrency, "
                 "retries included, so the three pass columns add up to the total.")
    lines.append("")
    rows = []
    for episode in episodes:
        entry = latency[episode]
        rows.append([episode] +
                    [num(entry["pass_wall_clock_s"][p]) for p in PASSES] +
                    [num(entry["wall_clock_s"]), entry["requests"],
                     f"{entry['input_tokens']:,}", f"{entry['cost_usd']:.4f}",
                     entry["retries"], entry["errors"]])
    totals = summary["latency_totals"]
    rows.append(["**total**"] + [num(totals["pass_wall_clock_s"][p]) for p in PASSES] +
                [num(totals["wall_clock_s"]), totals["requests"],
                 f"{totals['input_tokens']:,}", f"{totals['cost_usd']:.4f}",
                 totals["retries"], totals["errors"]])
    lines.extend(table(["episode", "retake s", "sentence s", "trim pick s", "total s",
                        "requests", "input tokens", "cost $", "retries", "errors"],
                       rows))
    lines.append("")

    lines.append("## Trims")
    lines.append("")
    lines.append("`trims emitted` counts sentences the arm gave a word range to, "
                 "before the keep threshold cuts any of them. `model partials` and "
                 "`human partials` are the SENTENCE POINTS partial counts at the "
                 "calibrated threshold: when the editor trimmed sentences and the "
                 "model trimmed none, the metric takes the 0.5 penalty.")
    lines.append("")
    lines.append(summary["pair_counts_note"])
    lines.append("")
    for arm in summary["arms"]:
        if not arm.get("plain"):
            continue
        rows = []
        for episode, trim in arm["trims"].items():
            row = arm["plain"]["result"]["episodes"][episode]
            rows.append([episode, trim, row["n_partial_model"],
                         row["n_partial_human"],
                         "yes" if row.get("sentence_points_penalty") else "no"])
        rows.append(["**total**", sum(arm["trims"].values()),
                     sum(arm["plain"]["result"]["episodes"][e]["n_partial_model"] or 0
                         for e in arm["trims"]),
                     sum(arm["plain"]["result"]["episodes"][e]["n_partial_human"] or 0
                         for e in arm["trims"]), "—"])
        lines.append(f"### {arm['label']}")
        lines.append("")
        lines.extend(table(["episode", "trims emitted", "model partials",
                            "human partials", "penalty"], rows))
        lines.append("")

    lines.append("## Retake pass")
    lines.append("")
    lines.append("`not real` are groups Jev scored under 0.5 on `real_k`, where "
                 "nothing is cut. The last four columns take the sentences where "
                 "Jev's cut and the production module's flags disagree and ask what "
                 "the editor did with them, using the harness's own human sentence "
                 "state: `kept` is full or partial in the real edit, `cut` is removed.")
    lines.append("")
    rows = []
    for episode, entry in summary["retakes"].items():
        rows.append([episode, entry["groups"], entry["not_real"],
                     entry["module_fallback"], entry["losers_cut_jev"],
                     entry["losers_cut_module"],
                     f"{entry['jev_only_cut']['n']} "
                     f"({entry['jev_only_cut']['editor_kept']} kept / "
                     f"{entry['jev_only_cut']['editor_cut']} cut)",
                     f"{entry['module_only_cut']['n']} "
                     f"({entry['module_only_cut']['editor_kept']} kept / "
                     f"{entry['module_only_cut']['editor_cut']} cut)"])
    lines.extend(table(["episode", "groups", "not real", "module fallback",
                        "losers cut by Jev", "losers cut by the module",
                        "Jev cuts only", "module cuts only"], rows))
    lines.append("")

    lines.append("## Where this lands on the published ladder")
    lines.append("")
    ladder_rows = summary["ladder"]["rows"]
    if not ladder_rows:
        lines.append(f"`{REFERENCE_JSON}` is not readable from here, so there is no "
                     f"ladder to quote.")
    else:
        if summary["ladder"]["restricted"]:
            lines.append(f"Reference arms re-pooled over the same "
                         f"{len(episodes)} episode(s) from the per-episode numbers in "
                         f"`{REFERENCE_JSON.name}`, so they are directly comparable to "
                         f"the tables above.")
        else:
            lines.append(f"Reference arms are the 18-episode pooled numbers from "
                         f"`{REFERENCE_JSON.name}`. NOT THE SAME EPISODE SET as the "
                         f"tables above (missing per-episode numbers for: "
                         f"{', '.join(summary['ladder']['uncovered']) or 'n/a'}), so "
                         f"the comparison is indicative only.")
        lines.append("")
        rows = [[entry["label"], entry["episodes"],
                 pct(entry["sentence_points_plain"]), pct(entry["word_score_plain"]),
                 pct(entry["grade_plain"]), pct(entry["sentence_points_layered"]),
                 pct(entry["word_score_layered"]), pct(entry["grade_layered"])]
                for entry in ladder_rows]
        lines.extend(table(["reference arm", "episodes", "SP plain", "WORD plain",
                            "GRADE plain", "SP + modules", "WORD + modules",
                            "GRADE + modules"], rows))
        lines.append("")
        best = summary["ladder"]["best_jev"]
        if best:
            lines.append(f"Best Jev arm here is {best['label']} at "
                         f"{pct(best['sentence_points'])} SENTENCE POINTS "
                         f"({pct(best['word_score'])} WORD SCORE, "
                         f"{num(best['seconds_per_episode'])} s per episode), "
                         f"{best['placement']}.")
            if best.get("sentence_points_layered") is not None:
                lines.append(f"With um removal and delete silence layered on, the "
                             f"same arm is {pct(best['sentence_points_layered'])} "
                             f"SENTENCE POINTS "
                             f"({pct(best['word_score_layered'])} WORD SCORE), "
                             f"{best['placement_layered']}. That is the column the "
                             f"ladder actually compares on, because every published "
                             f"arm gets the same modules.")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    for key, value in summary["inputs"].items():
        lines.append(f"- input {key}: `{value}`")
    lines.append(f"- this file: `{path}`")
    lines.append(f"- summary JSON: `{summary['summary_path']}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_name", required=True,
                        help="input basename under docs/jev-real")
    parser.add_argument("--out", dest="out_name", required=True,
                        help="output basename under docs/jev-real")
    parser.add_argument("--t-trim", nargs="+", type=float, default=DEFAULT_T_TRIM,
                        help="variant A trim-trigger thresholds to sweep "
                             f"(default: {' '.join(str(t) for t in DEFAULT_T_TRIM)})")
    parser.add_argument("--arms", nargs="+", default=None, choices=BASE_ARMS,
                        help="subset of arms to report (default: every arm the "
                             "decisions file contains)")
    parser.add_argument("--no-modules", action="store_true",
                        help="skip the layered (um removal + delete silence) scoring")
    parser.add_argument("--missing-score", type=float, default=0.0,
                        help="score for a sentence the run got no answer for "
                             "(default: 0.0, which cuts it)")
    args = parser.parse_args()

    notes_path = OUT_DIR / f"{args.out_name}-notes.md"
    summary_path = OUT_DIR / f"{args.out_name}-summary.json"
    existing = [str(p) for p in (notes_path, summary_path) if p.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    decision_rows, request_rows, timing, input_paths = load_run(args.in_name)
    by_arm, episodes = index_decisions(decision_rows)
    warnings = []

    present = [arm for arm in BASE_ARMS if arm in by_arm]
    if not present:
        raise SystemExit(f"{args.in_name}-decisions.jsonl has no rows for any known "
                         f"arm (found: {sorted(by_arm)}, expected some of "
                         f"{BASE_ARMS})")
    if args.arms is None:
        args.arms = present
        absent = [arm for arm in BASE_ARMS if arm not in by_arm]
        if absent:
            warnings.append(
                f"{args.in_name}-decisions.jsonl has no rows for {', '.join(absent)}; "
                f"the run wrote {', '.join(present)} only (a run without "
                f"`--trim-pick` writes no jev_b). Those arms are left out of every "
                f"table below.")
    else:
        asked_absent = [arm for arm in args.arms if arm not in by_arm]
        if asked_absent:
            raise SystemExit(
                f"{args.in_name}-decisions.jsonl has no rows for arm(s) "
                f"{asked_absent} (found: {sorted(by_arm)})")

    words = {episode: word_ids_by_sentence(episode) for episode in episodes}
    mismatches = check_rebuild(by_arm, episodes, words)
    if mismatches:
        warnings.append(
            f"the t=1.0 variant A rebuild disagrees with the run's own jev_a rows on "
            f"{len(mismatches)} sentence(s), first: {mismatches[:3]}. Every jev_a "
            f"number below is suspect.")

    removals, skipped = ({}, list(episodes)) if args.no_modules else load_removals(episodes)
    if skipped and not args.no_modules:
        warnings.append(f"no cached removals for {', '.join(skipped)}; those episodes "
                        f"are excluded from the layered tables. Run "
                        f"`python scripts/jev_real/roughcut_partial_scoring.py "
                        f"--removals {' '.join(skipped)}` to fill the cache.")
    layered_episodes = [e for e in episodes if e in removals]

    latency = latency_rows(timing, request_rows, episodes)

    arm_specs = []
    for arm in args.arms:
        if arm == SWEPT_ARM:
            for t_trim in args.t_trim:
                arm_specs.append({"arm": arm, "t_trim": t_trim,
                                  "key": f"{arm}@t{t_trim:g}",
                                  "label": f"{arm} (t_trim {t_trim:g})"})
        else:
            arm_specs.append({"arm": arm, "t_trim": None, "key": arm, "label": arm})

    missing_total = 0
    arms_out = []
    for spec in arm_specs:
        decisions, n_missing = build_arm_decisions(
            by_arm, episodes, words, spec["arm"], spec["t_trim"], args.missing_score)
        missing_total = max(missing_total, n_missing)
        entry = dict(spec)
        entry["trims"] = {episode: sum(1 for d in decisions[episode].values()
                                       if d["keep_words"])
                          for episode in episodes}
        print(f"scoring {spec['key']} plain...", file=sys.stderr)
        entry["plain"] = {
            "result": score_arm(decisions),
            "seconds_per_episode": mean_seconds(latency, episodes),
        }
        if layered_episodes:
            subset = {e: decisions[e] for e in layered_episodes}
            print(f"scoring {spec['key']} + modules...", file=sys.stderr)
            entry["layered"] = {
                "result": score_arm(subset, removals=removals),
                "seconds_per_episode": mean_seconds(latency, layered_episodes),
            }
        arms_out.append(entry)

    prompt_versions = sorted({r.get("prompt_version") for r in request_rows
                              if r.get("prompt_version")})
    totals = {
        "pass_wall_clock_s": {p: sum((latency[e]["pass_wall_clock_s"][p] or 0.0)
                                     for e in episodes) for p in PASSES},
        "wall_clock_s": sum((latency[e]["wall_clock_s"] or 0.0) for e in episodes),
        "requests": sum(latency[e]["requests"] for e in episodes),
        "input_tokens": sum(latency[e]["input_tokens"] for e in episodes),
        "cost_usd": round(sum(latency[e]["cost_usd"] for e in episodes), 6),
        "retries": sum(latency[e]["retries"] for e in episodes),
        "errors": sum(latency[e]["errors"] for e in episodes),
    }

    ladder_rows, restricted, covered = ladder(episodes)
    best = None
    for entry in arms_out:
        pooled = entry["plain"]["result"]["pooled"]
        if best is None or (pooled["sentence_points"] or 0) > (best["sentence_points"] or 0):
            layered = (entry.get("layered") or {}).get("result")
            best = {"key": entry["key"], "label": entry["label"],
                    "sentence_points": pooled["sentence_points"],
                    "word_score": pooled["word_score"],
                    "seconds_per_episode": entry["plain"]["seconds_per_episode"],
                    "sentence_points_layered":
                        layered["pooled"]["sentence_points"] if layered else None,
                    "word_score_layered":
                        layered["pooled"]["word_score"] if layered else None}
    if best and ladder_rows:
        for tag in ("plain", "layered"):
            mine = best["sentence_points"] if tag == "plain" else \
                best["sentence_points_layered"]
            suffix = "" if tag == "plain" else "_layered"
            if mine is None:
                best[f"placement{suffix}"] = None
                continue
            above = [r["label"] for r in ladder_rows
                     if (r[f"sentence_points_{tag}"] or 0) < mine]
            best[f"placement{suffix}"] = ("above " + ", ".join(above)) if above else \
                "below every reference arm quoted here"

    # A swept threshold that emits the same trim set as its neighbour is a wasted
    # arm; say so rather than leaving four identical rows unexplained.
    swept = [a for a in arms_out if a["arm"] == SWEPT_ARM]
    trim_totals = {a["key"]: sum(a["trims"].values()) for a in swept}
    duplicate = [(swept[i - 1]["label"], swept[i]["label"])
                 for i in range(1, len(swept))
                 if trim_totals[swept[i]["key"]] == trim_totals[swept[i - 1]["key"]]]
    sweep_note = ""
    if duplicate:
        sweep_note = (
            f"{len(duplicate) + 1} of the {len(swept)} swept thresholds emit the same "
            f"number of trims, so their rows are duplicates: "
            f"{'; '.join(f'{a} = {b}' for a, b in duplicate)}. When Jev picks a word "
            f"over `whole` it usually puts well under 0.5 on `whole`, so the "
            f"interesting part of the sweep is below 0.5.")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_jev_report.py",
        "inputs": input_paths,
        "summary_path": str(summary_path),
        "notes_path": str(notes_path),
        "episodes": episodes,
        "layered_episodes": layered_episodes,
        "run": {
            "input_name": args.in_name,
            "model": timing.get("model"),
            "prompt_version": "/".join(prompt_versions) or timing.get("prompt_version"),
            "concurrency": timing.get("concurrency"),
            "t_trim": timing.get("t_trim"),
            "aborted": timing.get("aborted"),
            "sentences": sum((latency[e]["sentences"] or 0) for e in episodes),
            "requests": totals["requests"],
            "errors": totals["errors"],
            "retries": totals["retries"],
            "input_tokens": totals["input_tokens"],
            "cost_usd": totals["cost_usd"],
            "wall_clock_s": round(totals["wall_clock_s"], 3),
            "missing_score": args.missing_score,
            "missing_answers_per_arm": missing_total,
            "rebuild_check": {
                "checked": sum(len(by_arm[SWEPT_ARM][e]) for e in episodes),
                "mismatches": len(mismatches),
                "examples": mismatches[:5],
            },
            "run_warnings": timing.get("warnings", [])[:20],
            "n_run_warnings": timing.get("n_warnings"),
        },
        "t_trim_sweep": args.t_trim,
        "sweep_note": sweep_note,
        "trims_by_arm": {a["key"]: a["trims"] for a in arms_out},
        "arms": arms_out,
        "latency": latency,
        "latency_totals": totals,
        "retakes": retake_stats(by_arm, episodes),
        "pair_counts_note": (
            "The full/partial/removed cross-tab is not in this table: "
            "`score_episode` returns `n_partial_human` and `n_partial_model` but "
            "drops the `pair_counts` block that `sentence_scoring` computes, so "
            "there is no public way to read it without reaching into the scoring "
            "module's internals."),
        "ladder": {
            "source": str(REFERENCE_JSON),
            "restricted": restricted,
            "covered_episodes": covered,
            "uncovered": [e for e in episodes if e not in covered],
            "rows": ladder_rows,
            "best_jev": best,
        },
        "warnings": warnings,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str),
                            encoding="utf-8")
    write_notes(notes_path, summary)

    print(json.dumps({
        "notes": str(notes_path), "summary": str(summary_path),
        "episodes": len(episodes), "arms": [a["key"] for a in arms_out],
        "pooled_sentence_points": {
            a["key"]: a["plain"]["result"]["pooled"]["sentence_points"]
            for a in arms_out},
        "warnings": warnings,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
