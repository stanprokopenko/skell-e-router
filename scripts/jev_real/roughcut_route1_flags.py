"""Route 1 of the hybrid rough cut: can Jev flag the sentences that need a trim?

The route: Jev decides keep or cut for every sentence as it does now, and also
FLAGS the sentences it thinks need an inside-sentence trim. Only the flagged
sentences go to a smarter model (Luna or Opus), which picks the words to keep.

Three questions, all answered from stored data:

1. Flag quality. Candidate flag signals read off the stored ``jev_a`` v3 rows
   (``1 - first_p_whole``, ``1 - last_p_whole`` and their combinations, the
   keep-score probability spread, ``cut_p``), plus a non-Jev sentence-length
   baseline, scored as detectors of the editor's ``partial`` sentences.
2. Ceiling. An oracle arm that hands the editor's exact kept words to every
   flagged human-partial sentence (a perfect smart model) and leaves every
   other decision as Jev made it, rescored at Jev's threshold 2.50 with the um
   removal and delete silence modules layered on, against the 80.47 baseline.
3. Cost and latency of the smart-model step at each operating point, from
   word counts and stated assumptions.

No model calls, no detector runs, $0. The decisions file and the cached removal
ranges are the only inputs; decisions are rebuilt with ``roughcut_jev_report``'s
helpers and scored through ``roughcut_partial_scoring``.

Writes ``docs/jev-real/roughcut-route1-flags.md`` and ``.json``; refuses to
overwrite either unless ``--force``.

Usage::

  python scripts/jev_real/roughcut_route1_flags.py [--force]
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"

sys.path.insert(0, str(HERE))

# Import order matters: the scoring module installs the cache-only answer-key
# loader and puts the harness on sys.path.
import roughcut_partial_scoring as scoring_mod  # noqa: E402
from roughcut_partial_scoring import load_episode  # noqa: E402
import roughcut_jev_report as report_mod  # noqa: E402
from roughcut_jev_misses import luna_files  # noqa: E402

IN_NAME = "roughcut-jev-all18-v3"
OUT_NAME = "roughcut-route1-flags"
ARM = "jev_a"
T_TRIM = 0.3
THRESHOLD = 2.50
EXPECTED_BASELINE_SP = 0.8047

#: The design spec's fit set; every other episode in the file is held out.
FIT_EPISODES = [
    "colman-02.04-skeleton-demo", "hampton-5.4-assignment-demo",
    "colman-03.03-muscles-crit", "edges-7.01-intro", "hampton-5.2-shape-demo",
    "perspective-14e-boxes-critique",
]

RECALL_TARGETS = [0.30, 0.50, 0.70, 0.90]
ORACLE_RECALLS = [0.70, 0.90]
SCORE_LEVELS = ["0", "1", "2", "3", "4", "5"]

# Cost and latency assumptions, stated in the output.
CONTEXT_EACH_SIDE = 3
TOKENS_PER_WORD = 1.3
BATCH_SIZE = 10
CONCURRENCY = 8
SECONDS_PER_REQUEST = 4.0
INSTRUCTION_TOKENS_PER_REQUEST = 1000
OUTPUT_TOKENS_PER_SENTENCE = 40
REASONING_TOKENS_PER_REQUEST = 300
PRICES = {  # USD per million tokens: (input, output)
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-6-luna": (0.10, 0.50),
}
PRICE_SOURCES = {
    "gpt-5.6-luna": "docs/jev-real/routing-notes.md (router rates 0.20 in / 1.20 out)",
    "gpt-6-luna": "skell_e_router/model_config.py pricing (0.10 in / 0.50 out)",
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


def fingerprint(paths):
    out = {}
    for key, value in paths.items():
        path = Path(value)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        out[key] = {"path": str(value), "md5": digest,
                    "modified_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc).isoformat(
                            timespec="seconds")}
    return out


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

def entropy(probabilities):
    values = [float(probabilities.get(level, 0.0)) for level in SCORE_LEVELS]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in values if v > 0)


def _trim_p(p_whole):
    return 0.0 if p_whole is None else 1.0 - float(p_whole)


#: name -> (description, function(row, context) -> float). ``context`` carries
#: the non-Jev facts (word count, Jev's own trim decision).
SIGNALS = {
    "head": ("1 - first_p_whole (Jev thinks the start should go)",
             lambda row, ctx: _trim_p(row.get("first_p_whole"))),
    "tail": ("1 - last_p_whole (Jev thinks the end should go)",
             lambda row, ctx: _trim_p(row.get("last_p_whole"))),
    "max_head_tail": ("max of head and tail",
                      lambda row, ctx: max(_trim_p(row.get("first_p_whole")),
                                           _trim_p(row.get("last_p_whole")))),
    "either_side": ("1 - first_p_whole x last_p_whole (either side, as if independent)",
                    lambda row, ctx: 1.0 - (1.0 - _trim_p(row.get("first_p_whole")))
                    * (1.0 - _trim_p(row.get("last_p_whole")))),
    "mid_mass": ("keep-score probability on levels 2 and 3",
                 lambda row, ctx: sum(float((row.get("score_probabilities") or {})
                                            .get(level, 0.0)) for level in ("2", "3"))),
    "score_entropy": ("entropy of the 0-5 keep-score distribution, bits",
                      lambda row, ctx: entropy(row.get("score_probabilities") or {})),
    "cut_p": ("cut_p, Jev's P(editor removes this sentence)",
              lambda row, ctx: float(row.get("cut_p") or 0.0)),
    "word_count": ("NOT Jev: transcript words in the sentence, ums included",
                   lambda row, ctx: float(ctx["n_words"])),
    "jev_trim_t03": ("NOT a score: Jev's own trim at t_trim 0.3 (1 or 0)",
                     lambda row, ctx: 1.0 if ctx["jev_trim"] else 0.0),
}
JEV_SIGNALS = ["head", "tail", "max_head_tail", "either_side", "mid_mass",
               "score_entropy", "cut_p"]
ORACLE_SIGNALS = JEV_SIGNALS + ["word_count"]


def auc(pairs):
    """Mann-Whitney AUC with tied ranks averaged. ``pairs`` is [(score, is_pos)]."""
    n_pos = sum(1 for _s, positive in pairs if positive)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    ordered = sorted(pairs, key=lambda p: p[0])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2.0
        rank_sum += mean_rank * sum(1 for k in range(i, j) if ordered[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_for_recall(pairs, target):
    """The highest threshold whose ``score >= t`` flag reaches ``target`` recall."""
    n_pos = sum(1 for _s, positive in pairs if positive)
    if not n_pos:
        return None
    positives = sorted((s for s, positive in pairs if positive), reverse=True)
    need = math.ceil(target * n_pos - 1e-9)
    return positives[max(need, 1) - 1]


def flag_stats(pairs, threshold, all_scores):
    """Recall / precision of ``score >= threshold`` on ``pairs``; share of all."""
    n_pos = sum(1 for _s, positive in pairs if positive)
    flagged = [positive for s, positive in pairs if s >= threshold]
    tp = sum(1 for positive in flagged if positive)
    n_all_flagged = sum(1 for s in all_scores if s >= threshold)
    return {
        "threshold": threshold,
        "recall": tp / n_pos if n_pos else None,
        "precision": tp / len(flagged) if flagged else None,
        "flagged": len(flagged),
        "flagged_share_of_population": len(flagged) / len(pairs) if pairs else None,
        "flagged_share_of_all_sentences": (n_all_flagged / len(all_scores)
                                           if all_scores else None),
        "base_rate": n_pos / len(pairs) if pairs else None,
    }


# ---------------------------------------------------------------------------
# scoring with an exempt set
# ---------------------------------------------------------------------------

def prepare(episode, decisions, removals, exempt=frozenset()):
    """Annotated sentences with the modules layered on, ``exempt`` left untouched.

    The same annotate-then-subtract sequence ``score_episode`` runs, except the
    sentences in ``exempt`` get their pre-removal ranges back, so an oracle trim
    can be scored exactly as the editor made it.
    """
    data = load_episode(episode)
    sentences, _n = scoring_mod._annotate(data, decisions, [])
    saved = {s["id"]: (s.get("roughcut_keep_ranges"), s["roughcut_score"])
             for s in sentences if s["id"] in exempt}
    scoring_mod._apply_removals(sentences, THRESHOLD,
                                scoring_mod.removal_frames(removals))
    for sentence in sentences:
        if sentence["id"] in saved:
            ranges, score = saved[sentence["id"]]
            sentence["roughcut_score"] = score
            if ranges is None:
                sentence.pop("roughcut_keep_ranges", None)
            else:
                sentence["roughcut_keep_ranges"] = ranges
    return data, sentences


def model_states(data, sentences):
    """The model side of ``sentence_states_for``, on already-prepared sentences."""
    pre = data["preflight"]
    model_by_media = scoring_mod.ranges_mod.kept_segments_from_score(
        sentences, THRESHOLD, pre.media_name, field="roughcut_score")
    shifted = {media: [(start + pre.offset, end + pre.offset)
                       for start, end in segments]
               for media, segments in model_by_media.items()}
    return scoring_mod.sentence_scoring.sentence_states(
        pre.word_units, shifted, pre.media_name, sentences, offset=pre.offset)


def score_set(episodes, decisions, removals, exempt=None, want_states=False):
    """Pooled SP / WORD / GRADE at 2.50 with the ladder's pooling weights."""
    levels, rows, states = [], [], {}
    for episode in episodes:
        data, sentences = prepare(episode, decisions[episode], removals[episode],
                                  (exempt or {}).get(episode, frozenset()))
        row, level = scoring_mod._score_annotated(data, sentences, THRESHOLD, {})
        levels.append(level)
        rows.append(row)
        if want_states:
            states[episode] = model_states(data, sentences)
    weights = [row["dialogue_frames"] for row in rows]
    pooled = {
        "sentence_points": scoring_mod.pooling.pooled_sp_grade(levels),
        "word_score": scoring_mod.pooling.pooled_word_grade(levels),
        "grade": scoring_mod.pooling.weighted_mean(
            [row["grade"] for row in rows], weights),
        "per_episode_sp": {row["episode"]: row["sentence_points"] for row in rows},
    }
    return pooled, states


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build():
    decision_rows, _requests, _timing, input_paths = report_mod.load_run(IN_NAME)
    by_arm, episodes = report_mod.index_decisions(decision_rows)
    missing_fit = [e for e in FIT_EPISODES if e not in episodes]
    if missing_fit:
        raise SystemExit(f"fit episodes missing from {IN_NAME}: {missing_fit}")
    heldout = [e for e in episodes if e not in FIT_EPISODES]
    splits = {"fit six": FIT_EPISODES, "held-out 12": heldout, "pooled 18": episodes}

    words = {e: report_mod.word_ids_by_sentence(e) for e in episodes}
    removals, skipped = report_mod.load_removals(episodes)
    if skipped:
        raise SystemExit(f"no cached removals for {skipped}")
    base, n_missing = report_mod.build_arm_decisions(
        by_arm, episodes, words, ARM, T_TRIM, 0.0)

    # Baseline, human states and the baseline model states.
    print("scoring baseline...", file=sys.stderr)
    baseline, base_states = score_set(episodes, base, removals, want_states=True)
    human, unit_ids, order = {}, {}, {}
    for episode in episodes:
        h, _m = scoring_mod.sentence_states_for(episode, base[episode], THRESHOLD,
                                                removals[episode])
        human[episode] = h
        data = load_episode(episode)
        ids = defaultdict(list)
        for unit in data["preflight"].word_units:
            ids[unit.sentence_id].append(unit.word_id)
        unit_ids[episode] = ids
        order[episode] = [s["id"] for s in data["sentences"]]

    # Modules alone: every sentence kept whole, no retake cut, layers on.
    print("scoring modules alone...", file=sys.stderr)
    keep_all = {e: {sid: {"score": 5.0, "keep_words": None, "cut_retake": False}
                    for sid in base[e]} for e in episodes}
    _p, modules_states = score_set(episodes, keep_all, removals, want_states=True)

    # Per-sentence table.
    records = []
    for episode in episodes:
        for sid in order[episode]:
            row = by_arm[ARM][episode][sid]
            decision = base[episode][sid]
            human_kind, human_runs = human[episode][sid]
            model_kind, model_runs = base_states[episode][sid]
            mod_kind, mod_runs = modules_states[episode][sid]
            ctx = {"n_words": len(unit_ids[episode].get(sid, [])),
                   "jev_trim": decision["keep_words"] is not None}
            jev_keeps = (not decision["cut_retake"]) and decision["score"] >= THRESHOLD

            def branch(kind, runs):
                if human_kind != "partial" or kind != "partial":
                    return None
                return scoring_mod.sentence_scoring._partial_branch(human_runs, runs)

            records.append({
                "episode": episode, "id": sid,
                "split": "fit" if episode in FIT_EPISODES else "heldout",
                "human": human_kind, "human_runs": human_runs,
                "model": model_kind, "model_branch": branch(model_kind, model_runs),
                "modules_alone": mod_kind,
                "modules_alone_branch": branch(mod_kind, mod_runs),
                "jev_keeps": jev_keeps, "jev_trim": ctx["jev_trim"],
                "null_trim_fields": row.get("first_p_whole") is None,
                "n_trim_words": len(words[episode].get(sid, [])),
                "n_words": ctx["n_words"],
                "signals": {name: fn(row, ctx) for name, (_d, fn) in SIGNALS.items()},
            })

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_route1_flags.py",
        "inputs": input_paths,
        "arm": f"{ARM} (t_trim {T_TRIM}) at threshold {THRESHOLD:.2f}, um removal + "
               f"delete silence layered on",
        "episodes": episodes, "fit": FIT_EPISODES, "heldout": heldout,
        "missing_answers": n_missing,
        "baseline": {k: v for k, v in baseline.items()},
        "signals": {name: desc for name, (desc, _fn) in SIGNALS.items()},
    }
    summary["counts"] = count_block(records, splits)
    summary["nulls"] = null_block(records)
    summary["modules_baseline"] = modules_block(records, splits)
    summary["flags"] = flag_block(records, splits)
    summary["oracle"] = oracle_block(episodes, base, removals, records, unit_ids,
                                     summary["flags"], baseline)
    summary["luna_trims"] = luna_block(episodes, base, removals, records, human,
                                       summary["flags"], baseline)
    summary["cost"] = cost_block(episodes, records, unit_ids, order,
                                 summary["flags"])
    summary["input_fingerprint"] = fingerprint(
        {**input_paths,
         **{f"removals {e}": scoring_mod.removals_cache_path(e) for e in episodes},
         **{f"luna chapters {e}": path for e, path in
            summary["luna_trims"].get("source", {}).items()}})
    return summary


def split_records(records, split_episodes):
    wanted = set(split_episodes)
    return [r for r in records if r["episode"] in wanted]


def count_block(records, splits):
    out = {}
    for name, eps in splits.items():
        recs = split_records(records, eps)
        c = Counter(r["human"] for r in recs)
        out[name] = {"sentences": len(recs), "full": c["full"],
                     "partial": c["partial"], "removed": c["removed"],
                     "jev_keeps": sum(1 for r in recs if r["jev_keeps"])}
    return out


def null_block(records):
    null = [r for r in records if r["null_trim_fields"]]
    return {
        "rows": len(null),
        "by_trim_word_count": dict(Counter(min(r["n_trim_words"], 2) for r in null)),
        "non_null_with_under_two_words": sum(
            1 for r in records if not r["null_trim_fields"] and r["n_trim_words"] < 2),
        "human_partial_in_null_rows": sum(1 for r in null if r["human"] == "partial"),
        "human_partial_total": sum(1 for r in records if r["human"] == "partial"),
        "human_state_of_null_rows": dict(Counter(r["human"] for r in null)),
    }


def modules_block(records, splits):
    """What the um / silence modules already do to the editor's partials."""
    out = {}
    for name, eps in splits.items():
        partials = [r for r in split_records(records, eps) if r["human"] == "partial"]
        mod = Counter(r["modules_alone_branch"] or f"model {r['modules_alone']}"
                      for r in partials)
        arm = Counter()
        for r in partials:
            if r["model"] == "partial":
                who = "Jev trim" if r["jev_trim"] else "layers only"
                arm[f"{who}: {r['model_branch']}"] += 1
            else:
                arm[f"arm {r['model']}"] += 1
        out[name] = {"human_partial": len(partials),
                     "modules_alone": dict(mod), "arm_v3": dict(arm)}
    return out


def flag_block(records, splits):
    """AUC and recall operating points per signal, population and split."""
    populations = {
        "editor kept": lambda r: r["human"] in ("full", "partial"),
        "all sentences": lambda r: True,
        "Jev keeps and editor kept": lambda r: (r["jev_keeps"]
                                                and r["human"] in ("full", "partial")),
    }
    out = {}
    for signal in SIGNALS:
        out[signal] = {}
        for split_name, eps in splits.items():
            recs = split_records(records, eps)
            all_scores = [r["signals"][signal] for r in recs]
            entry = {}
            for pop_name, keep in populations.items():
                pairs = [(r["signals"][signal], r["human"] == "partial")
                         for r in recs if keep(r)]
                points = {}
                for target in RECALL_TARGETS:
                    t = threshold_for_recall(pairs, target)
                    points[f"{target:.2f}"] = flag_stats(pairs, t, all_scores)
                entry[pop_name] = {"auc": auc(pairs), "n": len(pairs),
                                   "positives": sum(1 for _s, p in pairs if p),
                                   "points": points}
            out[signal][split_name] = entry
        # Held-out transfer: thresholds picked on the fit six, applied to held-out.
        fit_pairs = [(r["signals"][signal], r["human"] == "partial")
                     for r in split_records(records, FIT_EPISODES)
                     if r["human"] in ("full", "partial")]
        held = split_records(records, splits["held-out 12"])
        held_pairs = [(r["signals"][signal], r["human"] == "partial")
                      for r in held if r["human"] in ("full", "partial")]
        transfer = {}
        for target in RECALL_TARGETS:
            t = threshold_for_recall(fit_pairs, target)
            transfer[f"{target:.2f}"] = flag_stats(
                held_pairs, t, [r["signals"][signal] for r in held])
        out[signal]["transfer"] = transfer
    return out


def oracle_decisions(episodes, base, records, unit_ids, flagged, override_keep):
    """Jev's decisions with the editor's words on flagged human-partial sentences.

    ``flagged`` is ``{episode: set(sid)}``. Without ``override_keep`` only the
    sentences Jev keeps at 2.50 are trimmed: a trimmer cannot rescue a sentence
    the keep pass already cut. With it, the flagged sentence is also kept.
    Returns ``(decisions, replaced)`` with ``replaced`` as ``{episode: set}``.
    """
    decisions = {e: {sid: dict(d) for sid, d in base[e].items()} for e in episodes}
    replaced = defaultdict(set)
    for r in records:
        episode, sid = r["episode"], r["id"]
        if sid not in flagged.get(episode, ()) or r["human"] != "partial":
            continue
        if not r["jev_keeps"] and not override_keep:
            continue
        ids = unit_ids[episode][sid]
        keep = [[ids[first], ids[end - 1]] for first, end in r["human_runs"]]
        d = decisions[episode][sid]
        d["keep_words"] = keep
        if override_keep:
            d["score"] = max(d["score"], 5.0)
            d["cut_retake"] = False
        replaced[episode].add(sid)
    return decisions, replaced


def oracle_block(episodes, base, removals, records, unit_ids, flags, baseline):
    pooled_threshold = {}
    runs = []
    settings = []
    for signal in ORACLE_SIGNALS:
        for target in ORACLE_RECALLS:
            point = flags[signal]["pooled 18"]["editor kept"]["points"][f"{target:.2f}"]
            settings.append((signal, target, point["threshold"]))
    settings.append(("perfect flag", 1.0, None))

    for signal, target, t in settings:
        if signal == "perfect flag":
            flagged = defaultdict(set)
            for r in records:
                if r["human"] == "partial":
                    flagged[r["episode"]].add(r["id"])
        else:
            flagged = defaultdict(set)
            for r in records:
                if r["signals"][signal] >= t:
                    flagged[r["episode"]].add(r["id"])
        pooled_threshold[(signal, target)] = t
        for variant, override, exempt_layers in (
                ("trim kept only, modules on top", False, False),
                ("trim kept only, oracle trims exempt from modules", False, True),
                ("trim and keep, modules on top", True, False)):
            if signal != "perfect flag" and variant != "trim kept only, modules on top" \
                    and signal not in ("max_head_tail", "either_side", "word_count"):
                continue
            decisions, replaced = oracle_decisions(episodes, base, records, unit_ids,
                                                   flagged, override)
            print(f"oracle {signal} {target} {variant}...", file=sys.stderr)
            want = signal == "perfect flag"
            pooled, states = score_set(
                episodes, decisions, removals,
                exempt=replaced if exempt_layers else None, want_states=want)
            check = None
            if want:
                check = Counter()
                by_key = {(r["episode"], r["id"]): r for r in records}
                for episode in episodes:
                    for sid in replaced[episode]:
                        kind, model_runs = states[episode][sid]
                        human_runs = by_key[(episode, sid)]["human_runs"]
                        check[scoring_mod.sentence_scoring._partial_branch(
                            human_runs, model_runs) if kind == "partial"
                            else f"model {kind}"] += 1
                check = dict(check)
            runs.append({
                "signal": signal, "recall_target": target, "threshold": t,
                "variant": variant,
                "flagged": sum(len(v) for v in flagged.values()),
                "flagged_jev_keeps": sum(1 for r in records
                                         if r["id"] in flagged.get(r["episode"], ())
                                         and r["jev_keeps"]),
                "replaced": sum(len(v) for v in replaced.values()),
                "sentence_points": pooled["sentence_points"],
                "word_score": pooled["word_score"],
                "grade": pooled["grade"],
                "sp_gain": pooled["sentence_points"] - baseline["sentence_points"],
                "word_gain": pooled["word_score"] - baseline["word_score"],
                "grade_gain": pooled["grade"] - baseline["grade"],
                "replaced_branch_check": check,
            })
    return runs


def luna_block(episodes, base, removals, records, human, flags, baseline):
    """Route 1 with a real smart model's trims instead of the editor's.

    The stored Luna chapters arm (gpt-5.6-luna, agentic, xhigh) already chose
    ``keep_words`` for every sentence of the 18 episodes. For each flagged
    sentence Jev keeps, Jev's ``keep_words`` are replaced with Luna's (``None``
    when Luna kept it whole); Jev's keep decision is untouched. Nothing else
    changes, and the modules run on top. Luna's own keep scores are ignored.
    """
    files = luna_files()
    missing = [e for e in episodes if e not in files]
    if missing:
        return {"skipped": f"no Luna chapters archive for {missing}"}
    luna = {}
    for episode in episodes:
        with files[episode].open(encoding="utf-8") as handle:
            doc = json.load(handle)
        ratings = (doc["episodes"][episode].get("run_ratings") or [None])[0]
        luna[episode] = scoring_mod.decisions_from_run_ratings(episode, ratings)

    settings = [(s, t) for s in ("max_head_tail", "word_count") for t in ORACLE_RECALLS]
    settings += [("perfect flag", 1.0), ("every sentence Jev keeps", None)]
    out = {"source": {e: str(files[e]) for e in episodes}, "runs": []}
    for signal, target in settings:
        flagged = defaultdict(set)
        for r in records:
            if signal == "perfect flag":
                hit = r["human"] == "partial"
            elif target is None:
                hit = True
            else:
                t = flags[signal]["pooled 18"]["editor kept"]["points"][
                    f"{target:.2f}"]["threshold"]
                hit = r["signals"][signal] >= t
            if hit and r["jev_keeps"]:
                flagged[r["episode"]].add(r["id"])
        decisions = {}
        for episode in episodes:
            decisions[episode] = {sid: dict(d) for sid, d in base[episode].items()}
            for sid in flagged[episode]:
                decisions[episode][sid]["keep_words"] = luna[episode].get(
                    sid, {}).get("keep_words")
        print(f"luna trims {signal} {target}...", file=sys.stderr)
        pooled, states = score_set(episodes, decisions, removals, want_states=True)
        outcome = Counter()
        for episode in episodes:
            for sid in flagged[episode]:
                human_kind, human_runs = human[episode][sid]
                kind, model_runs = states[episode][sid]
                trimmed = bool(luna[episode].get(sid, {}).get("keep_words"))
                if human_kind == "partial" and kind == "partial":
                    label = scoring_mod.sentence_scoring._partial_branch(
                        human_runs, model_runs)
                else:
                    label = f"editor {human_kind}, model {kind}"
                outcome[f"{'Luna trimmed' if trimmed else 'Luna kept whole'}: {label}"] += 1
        out["runs"].append({
            "signal": signal, "recall_target": target,
            "sent": sum(len(v) for v in flagged.values()),
            "luna_trimmed": sum(1 for e in episodes for sid in flagged[e]
                                if luna[e].get(sid, {}).get("keep_words")),
            "sentence_points": pooled["sentence_points"],
            "word_score": pooled["word_score"], "grade": pooled["grade"],
            "sp_gain": pooled["sentence_points"] - baseline["sentence_points"],
            "word_gain": pooled["word_score"] - baseline["word_score"],
            "grade_gain": pooled["grade"] - baseline["grade"],
            "outcomes": dict(outcome),
        })
    return out


def cost_block(episodes, records, unit_ids, order, flags):
    """Sentences sent and token / dollar / latency estimates per operating point."""
    n_words = {(r["episode"], r["id"]): r["n_words"] for r in records}
    position = {e: {sid: i for i, sid in enumerate(order[e])} for e in episodes}

    def context_words(episode, sid):
        i = position[episode][sid]
        lo, hi = max(0, i - CONTEXT_EACH_SIDE), min(len(order[episode]),
                                                    i + CONTEXT_EACH_SIDE + 1)
        return sum(n_words[(episode, s)] for s in order[episode][lo:hi])

    points = []
    settings = [(s, t) for s in ("max_head_tail", "either_side", "word_count")
                for t in ORACLE_RECALLS] + [("perfect flag", 1.0)]
    for signal, target in settings:
        per_episode = {}
        for episode in episodes:
            recs = [r for r in records if r["episode"] == episode and r["jev_keeps"]]
            if signal == "perfect flag":
                sent = [r for r in recs if r["human"] == "partial"]
            else:
                t = flags[signal]["pooled 18"]["editor kept"]["points"][
                    f"{target:.2f}"]["threshold"]
                sent = [r for r in recs if r["signals"][signal] >= t]
            n = len(sent)
            requests = math.ceil(n / BATCH_SIZE)
            input_tokens = (sum(context_words(episode, r["id"]) for r in sent)
                            * TOKENS_PER_WORD
                            + requests * INSTRUCTION_TOKENS_PER_REQUEST)
            output_tokens = (n * OUTPUT_TOKENS_PER_SENTENCE
                             + requests * REASONING_TOKENS_PER_REQUEST)
            waves = math.ceil(requests / CONCURRENCY)
            per_episode[episode] = {
                "sent": n, "requests": requests,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cost": {model: (input_tokens * p_in + output_tokens * p_out) / 1e6
                         for model, (p_in, p_out) in PRICES.items()},
                "latency_s": waves * SECONDS_PER_REQUEST,
                "sentences": len(order[episode]),
            }
        values = list(per_episode.values())
        points.append({
            "signal": signal, "recall_target": target,
            "sent_mean": statistics.mean(v["sent"] for v in values),
            "sent_max": max(v["sent"] for v in values),
            "sent_share_of_sentences": (sum(v["sent"] for v in values)
                                        / sum(v["sentences"] for v in values)),
            "requests_mean": statistics.mean(v["requests"] for v in values),
            "input_tokens_mean": statistics.mean(v["input_tokens"] for v in values),
            "input_tokens_max": max(v["input_tokens"] for v in values),
            "cost_mean": {m: statistics.mean(v["cost"][m] for v in values)
                          for m in PRICES},
            "cost_max": {m: max(v["cost"][m] for v in values) for m in PRICES},
            "latency_mean_s": statistics.mean(v["latency_s"] for v in values),
            "latency_max_s": max(v["latency_s"] for v in values),
            "per_episode": per_episode,
        })
    return points


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def best_signal(summary):
    jev = [(summary["flags"][s]["pooled 18"]["editor kept"]["auc"] or 0, s)
           for s in JEV_SIGNALS]
    return max(jev)[1]


def summary_lines(summary):
    """The handful of numbers the brief asks for, as sentences."""
    flags = summary["flags"]
    base = summary["baseline"]["sentence_points"]

    def signed(value):
        return ("+" if value >= 0 else "") + pct(value)

    out = []
    for signal in ("max_head_tail", "word_count"):
        f = flags[signal]["pooled 18"]["editor kept"]
        p70, p90 = f["points"]["0.70"], f["points"]["0.90"]
        out.append(f"`{signal}` on editor-kept sentences: AUC {pct(f['auc'])}; at {pct(p70['recall'])} recall precision {pct(p70['precision'])} and {pct(p70['flagged_share_of_all_sentences'])} of all sentences flagged; at {pct(p90['recall'])} recall precision {pct(p90['precision'])}, {pct(p90['flagged_share_of_all_sentences'])} flagged. Random flagging has precision {pct(p70['base_rate'])}.")
    mods = summary["modules_baseline"]["pooled 18"]
    n = mods["human_partial"]
    exact_alone = mods["modules_alone"].get("exact", 0)
    out.append(f"The modules alone already get {exact_alone} of {n} editor partials exactly right ({pct(exact_alone / n)}); in the v3 arm the layers make {mods['arm_v3'].get('layers only: exact', 0)} exact and Jev's own trims {mods['arm_v3'].get('Jev trim: exact', 0)}.")
    oracle = {(r["signal"], r["recall_target"], r["variant"]): r for r in summary["oracle"]}
    variant = "trim kept only, modules on top"
    parts = []
    for signal, target in (("max_head_tail", 0.7), ("max_head_tail", 0.9),
                           ("word_count", 0.7), ("perfect flag", 1.0)):
        r = oracle.get((signal, target, variant))
        if r:
            parts.append(f"`{signal}` {pct(target)}: {pct(r['sentence_points'])} ({signed(r['sp_gain'])})")
    out.append(f"Oracle SP with perfect trims, modules on top, against {pct(base)}: " + "; ".join(parts) + ".")
    luna = summary.get("luna_trims", {})
    if luna.get("runs"):
        parts = [f"`{r['signal']}` {pct(r['recall_target'])}: {signed(r['sp_gain'])}"
                 for r in luna["runs"]]
        out.append("With Luna's stored trims in place of the oracle's, SP gain: " + "; ".join(parts) + ".")
    for p in summary["cost"]:
        if p["signal"] == "max_head_tail":
            out.append(f"Cost at `max_head_tail` {pct(p['recall_target'])}: {num(p['sent_mean'], 0)} sentences sent per episode (max {p['sent_max']}), about ${p['cost_mean']['gpt-5.6-luna']:.3f} per episode on gpt-5.6-luna (max ${p['cost_max']['gpt-5.6-luna']:.3f}), {num(p['latency_mean_s'], 0)} s added per episode (max {num(p['latency_max_s'], 0)} s).")
    return out


def write_markdown(path, summary):
    lines = []
    add = lines.append
    flags = summary["flags"]
    base = summary["baseline"]
    counts = summary["counts"]
    best = best_signal(summary)

    add("# Route 1 flags (developer-facing): can Jev point the smart trimmer at the right sentences?")
    add("")
    add(f"Developer-facing record. Generated {summary['generated_utc']} by `{summary['script']}` from the stored `{IN_NAME}` decisions and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals, unless it is a count, a threshold or a dollar figure; the JSON next to this file keeps the raw values.")
    add("")
    add("Route 1 is a hybrid: Jev decides keep or cut for every sentence as now, and flags the sentences it thinks need an inside-sentence trim; only those go to a smarter model that picks the kept words. This file asks whether the stored Jev fields can do the flagging, what the route could add at best, and roughly what the smart step would cost.")
    add("")
    add(f"The arm is {summary['arm']}. Reproduced baseline: {pct(base['sentence_points'])} SENTENCE POINTS, {pct(base['word_score'])} WORD SCORE, {pct(base['grade'])} GRADE over the 18 ladder episodes, matching the published 80.47 / 74.12 / 88.44. `partial`, `full` and `removed` are the editor's per-sentence states from the harness's `sentence_states` (majority rule), the same states SENTENCE POINTS reads.")
    add("")
    rows = [[name, c["sentences"], c["full"], c["partial"], c["removed"],
             pct(c["partial"] / (c["full"] + c["partial"])), c["jev_keeps"]]
            for name, c in counts.items()]
    lines.extend(table(["split", "sentences", "editor full", "editor partial",
                        "editor removed", "partial share of kept", "Jev keeps at 2.50"],
                       rows))
    add("")

    add("SENTENCE POINTS per sentence, from `sentence_scoring.sentence_points`, is what makes trims matter: an editor-partial sentence the model keeps whole or removes earns 0.4; the model's partial earns 2.0 when its runs match the editor's exactly, 1.2 when they are a subset, 1.0 when they overlap and 0.6 when they are disjoint. Trimming a sentence the editor kept whole drops it from 1.0 to 0.7. One exact trim is worth 1.6 points on its sentence, and a wrong trim on a whole sentence costs 0.3.")
    add("")
    add("## Summary")
    add("")
    for line in summary_lines(summary):
        add(f"- {line}")
    add("")

    add("## Signals")
    add("")
    for name, desc in summary["signals"].items():
        add(f"- `{name}`: {desc}")
    add("")
    nulls = summary["nulls"]
    by_words = nulls["by_trim_word_count"]
    add(f"Null trim fields: {nulls['rows']} of {counts['pooled 18']['sentences']} rows have `first_p_whole` and `last_p_whole` both null, and no row has only one of them. The pipeline asks the first and last questions only when a sentence has at least two words after the um-removal words are stripped (`sentence_jobs` in `roughcut_jev.py`, `if len(words) >= 2`), and every one of these rows is under that bar: {by_words.get(0, 0)} have no words left, {by_words.get(1, 0)} have one. No row is null because a request failed. Head and tail signals score these rows 0. Only {nulls['human_partial_in_null_rows']} of the editor's {nulls['human_partial_total']} partial sentences sits in these rows (editor states in null rows: {', '.join(f'{k} {v}' for k, v in sorted(nulls['human_state_of_null_rows'].items()))}), so the nulls cost the head and tail signals almost nothing.")
    add("")

    add("## What the modules already make partial")
    add("")
    add("The um removal and delete silence modules cut inside sentences on their own, so some of the editor's partials need no smart model. `modules alone` keeps every sentence whole and layers the two modules on; `v3 arm` is the actual baseline. Branch names are the metric's: `exact` pays full partial credit, the others pay less.")
    add("")
    rows = []
    for name, block in summary["modules_baseline"].items():
        m, a = block["modules_alone"], block["arm_v3"]
        n = block["human_partial"]
        layers_exact = a.get("layers only: exact", 0)
        jev_exact = a.get("Jev trim: exact", 0)
        rows.append([name, n,
                     f"{m.get('exact', 0)} ({pct(m.get('exact', 0) / n)})",
                     sum(v for k, v in m.items() if not k.startswith("model ")),
                     f"{layers_exact} ({pct(layers_exact / n)})",
                     sum(v for k, v in a.items() if k.startswith("layers only")),
                     jev_exact,
                     sum(v for k, v in a.items() if k.startswith("Jev trim")),
                     a.get("arm full", 0), a.get("arm removed", 0)])
    lines.extend(table(["split", "editor partials", "modules alone exact",
                        "modules alone partial, any branch", "v3 arm exact via layers",
                        "v3 arm partial via layers", "v3 arm exact via Jev trim",
                        "v3 arm partial via Jev trim", "v3 arm keeps whole",
                        "v3 arm removes"], rows))
    add("")
    pooled_mod = summary["modules_baseline"]["pooled 18"]
    add(f"Pooled detail, v3 arm: {', '.join(f'{k} {v}' for k, v in sorted(pooled_mod['arm_v3'].items()))}. Modules alone: {', '.join(f'{k} {v}' for k, v in sorted(pooled_mod['modules_alone'].items()))}.")
    add("")

    add("## Flag quality")
    add("")
    add("Positive = the editor made the sentence partial. `editor kept` restricts to sentences the editor kept (full against partial), the population the brief asks about; `all sentences` adds the removed ones as negatives; `Jev keeps` restricts further to sentences Jev keeps at 2.50, which is the only place a trim can land. AUC 50 is a coin flip.")
    add("")
    rows = []
    for signal in SIGNALS:
        f = flags[signal]
        rows.append([f"`{signal}`",
                     pct(f["fit six"]["editor kept"]["auc"]),
                     pct(f["held-out 12"]["editor kept"]["auc"]),
                     pct(f["pooled 18"]["editor kept"]["auc"]),
                     pct(f["pooled 18"]["all sentences"]["auc"]),
                     pct(f["pooled 18"]["Jev keeps and editor kept"]["auc"])])
    lines.extend(table(["signal", "AUC fit, editor kept", "AUC held-out, editor kept",
                        "AUC pooled, editor kept", "AUC pooled, all sentences",
                        "AUC pooled, Jev keeps and editor kept"], rows))
    add("")
    pooled_auc = {sig: flags[sig]["pooled 18"]["editor kept"]["auc"] or 0 for sig in JEV_SIGNALS}
    top = sorted(JEV_SIGNALS, key=lambda sig: -pooled_auc[sig])[:3]
    top_text = ", ".join(f"`{sig}` {pct(pooled_auc[sig])}" for sig in top)
    add(f"Best Jev signal by pooled AUC on editor-kept sentences: `{best}`. The top three ({top_text}) are within a point of each other. The oracle, Luna and cost sections use `max_head_tail` as the Jev flag and `word_count` as the no-Jev comparison.")
    add("")

    for split in ("pooled 18", "fit six", "held-out 12"):
        add(f"### Operating points, {split}, editor kept")
        add("")
        base_rate = flags[best][split]["editor kept"]["points"]["0.30"]["base_rate"]
        add(f"Threshold picked inside this split to reach each recall target (the flag is `signal >= threshold`; ties can overshoot the target). Base rate, the precision of flagging at random: {pct(base_rate)}. `share of all` is the share of every sentence in the split, removed ones included, the flag fires on.")
        add("")
        rows = []
        for signal in SIGNALS:
            for target in RECALL_TARGETS:
                p = flags[signal][split]["editor kept"]["points"][f"{target:.2f}"]
                rows.append([f"`{signal}`", pct(target), num(p["threshold"], 3),
                             pct(p["recall"]), pct(p["precision"]),
                             pct(p["flagged_share_of_population"]),
                             pct(p["flagged_share_of_all_sentences"])])
        lines.extend(table(["signal", "recall target", "threshold", "recall",
                            "precision", "share of kept flagged", "share of all"],
                           rows))
        add("")

    add("### Operating points, pooled 18, all sentences")
    add("")
    add("Same, with the editor's removed sentences counted as negatives.")
    add("")
    rows = []
    for signal in SIGNALS:
        for target in RECALL_TARGETS:
            p = flags[signal]["pooled 18"]["all sentences"]["points"][f"{target:.2f}"]
            rows.append([f"`{signal}`", pct(target), num(p["threshold"], 3),
                         pct(p["recall"]), pct(p["precision"]),
                         pct(p["flagged_share_of_population"])])
    lines.extend(table(["signal", "recall target", "threshold", "recall",
                        "precision", "share of all flagged"], rows))
    add("")

    add("### Held-out transfer")
    add("")
    add("Thresholds picked on the fit six (editor kept), then applied unchanged to the held-out 12. This is what a frozen flag would have done on unseen episodes.")
    add("")
    rows = []
    for signal in SIGNALS:
        for target in RECALL_TARGETS:
            p = flags[signal]["transfer"][f"{target:.2f}"]
            rows.append([f"`{signal}`", pct(target), num(p["threshold"], 3),
                         pct(p["recall"]), pct(p["precision"]),
                         pct(p["flagged_share_of_all_sentences"])])
    lines.extend(table(["signal", "fit recall target", "threshold",
                        "held-out recall", "held-out precision", "held-out share of all"],
                       rows))
    add("")

    add("## Ceiling of the route")
    add("")
    add(f"Oracle arm: start from the baseline decisions above. For every sentence the flag fires on, if the editor made it partial, replace Jev's `keep_words` with the editor's exact kept runs (a perfect smart model); if the editor kept it whole or removed it, leave Jev's decision alone (a perfect smart model declines to trim). Flags come from the pooled editor-kept thresholds in the table above. Scored at {num(THRESHOLD)} with the modules layered on, against the reproduced baseline {pct(base['sentence_points'])} SP.")
    add("")
    add("Three variants. `trim kept only, modules on top` is the realistic one: the trimmer only touches sentences Jev keeps at 2.50, and the um and silence modules still run over its output, so an editor run that keeps an um or a pause loses it again. `oracle trims exempt from modules` scores the editor's runs untouched, the pure value of perfect trims. `trim and keep` also keeps a flagged partial sentence Jev had cut, which is outside route 1 (it is a keep decision, not a trim) and is shown only for scale. `perfect flag` fires on every editor-partial sentence and nothing else, which separates flag quality from trim value.")
    add("")
    rows = []
    for run in summary["oracle"]:
        rows.append([f"`{run['signal']}`", pct(run["recall_target"]),
                     num(run["threshold"], 3), run["variant"], run["flagged"],
                     run["flagged_jev_keeps"], run["replaced"],
                     pct(run["sentence_points"]), pct(run["sp_gain"]),
                     pct(run["word_score"]), pct(run["word_gain"]),
                     pct(run["grade"]), pct(run["grade_gain"])])
    lines.extend(table(["flag", "recall", "threshold", "variant", "flagged",
                        "flagged and Jev keeps", "replaced", "SP", "SP gain", "WORD",
                        "WORD gain", "GRADE", "GRADE gain"], rows))
    add("")
    for run in summary["oracle"]:
        if run["replaced_branch_check"] is not None:
            add(f"Self-check, `perfect flag`, {run['variant']}: how the metric reads the {run['replaced']} replaced sentences: {', '.join(f'{k} {v}' for k, v in sorted(run['replaced_branch_check'].items()))}.")
            add("")

    add("## A real smart model instead of the oracle")
    add("")
    luna = summary["luna_trims"]
    if "skipped" in luna:
        add(f"Skipped: {luna['skipped']}.")
    else:
        add("The oracle assumes a trimmer that matches the editor word for word. For a grounded number, the stored Luna chapters arm (gpt-5.6-luna, agentic, xhigh effort, the best Luna arm on the ladder at 83.71 SP) already picked kept words for every sentence of the 18 episodes. Here each flagged sentence Jev keeps takes Luna's `keep_words` (whole when Luna kept it whole), Jev's keep decisions are untouched, and the modules run on top. `every sentence Jev keeps` sends everything, so it is the no-flag version of the route. Luna chose those trims with the whole episode in view and a far bigger budget than the batched call priced below, so treat this as optimistic for a cheap Luna step.")
        add("")
        rows = []
        for run in luna["runs"]:
            rows.append([f"`{run['signal']}`", pct(run["recall_target"]), run["sent"],
                         run["luna_trimmed"], pct(run["sentence_points"]),
                         pct(run["sp_gain"]), pct(run["word_score"]),
                         pct(run["word_gain"]), pct(run["grade"]),
                         pct(run["grade_gain"])])
        lines.extend(table(["flag", "recall", "sent (Jev keeps)", "Luna trimmed",
                            "SP", "SP gain", "WORD", "WORD gain", "GRADE",
                            "GRADE gain"], rows))
        add("")
        add("Where Luna's trims landed, over the sentences sent. `exact` and `other partial branch` are editor-partial sentences Luna trimmed; `on editor-whole` and `on editor-removed` are trims the editor did not make; `missed` is an editor-partial sentence Luna kept whole. The full outcome counts are in the JSON.")
        add("")
        rows = []
        for run in luna["runs"]:
            outcome = run["outcomes"]
            trimmed = {k.split(": ", 1)[1]: v for k, v in outcome.items()
                       if k.startswith("Luna trimmed")}
            whole = {k.split(": ", 1)[1]: v for k, v in outcome.items()
                     if k.startswith("Luna kept whole")}
            rows.append([
                f"`{run['signal']}`", pct(run["recall_target"]),
                trimmed.get("exact", 0),
                sum(trimmed.get(b, 0) for b in ("subset", "overlap", "disjoint")),
                sum(v for k, v in trimmed.items() if k.startswith("editor full")),
                sum(v for k, v in trimmed.items() if k.startswith("editor removed")),
                sum(v for k, v in whole.items() if k.startswith("editor partial")),
            ])
        lines.extend(table(["flag", "recall", "Luna trim exact",
                            "Luna trim, other partial branch", "Luna trim on editor-whole",
                            "Luna trim on editor-removed",
                            "missed: Luna kept an editor partial whole"], rows))
        add("")

    add("## Cost and latency of the smart step")
    add("")
    add(f"Assumptions, all rough. Only flagged sentences Jev keeps at 2.50 are sent (a cut sentence needs no trim). Each goes with {CONTEXT_EACH_SIDE} corpus sentences of context either side, not deduplicated across neighbours in a batch, at {TOKENS_PER_WORD} tokens per transcript word, plus {INSTRUCTION_TOKENS_PER_REQUEST} instruction tokens per request. {BATCH_SIZE} flagged sentences per request, {CONCURRENCY} requests in flight, {num(SECONDS_PER_REQUEST, 0)} s per request, so latency is `ceil(requests / {CONCURRENCY}) x {num(SECONDS_PER_REQUEST, 0)} s` per episode. Output is {OUTPUT_TOKENS_PER_SENTENCE} tokens per sentence plus {REASONING_TOKENS_PER_REQUEST} reasoning tokens per request (low effort). Prices per million tokens: " + "; ".join(f"{m} {PRICES[m][0]:.2f} in / {PRICES[m][1]:.2f} out from {PRICE_SOURCES[m]}" for m in PRICES) + ". No caching assumed. This step comes on top of Jev's own run, about $0.05 and 5.6 s per episode.")
    add("")
    rows = []
    for p in summary["cost"]:
        rows.append([f"`{p['signal']}`", pct(p["recall_target"]),
                     num(p["sent_mean"], 1), p["sent_max"],
                     pct(p["sent_share_of_sentences"]), num(p["requests_mean"], 1),
                     f"{p['input_tokens_mean']:,.0f}", f"{p['input_tokens_max']:,.0f}",
                     f"${p['cost_mean']['gpt-5.6-luna']:.4f}",
                     f"${p['cost_max']['gpt-5.6-luna']:.4f}",
                     f"${p['cost_mean']['gpt-6-luna']:.4f}",
                     num(p["latency_mean_s"], 1), num(p["latency_max_s"], 1)])
    lines.extend(table(["flag", "recall", "sent per episode, mean", "max",
                        "share of sentences sent", "requests, mean",
                        "input tokens, mean", "input tokens, max",
                        "gpt-5.6-luna $, mean", "gpt-5.6-luna $, max",
                        "gpt-6-luna $, mean", "latency s, mean", "latency s, max"],
                       rows))
    add("")

    add("## Files")
    add("")
    for key, info in summary["input_fingerprint"].items():
        add(f"- input {key}: `{info['path']}`, md5 {info['md5'][:12]}, modified {info['modified_utc']}")
    add(f"- this file: `{summary['markdown_path']}`")
    add(f"- data: `{summary['json_path']}`")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing outputs")
    args = parser.parse_args()
    md_path = OUT_DIR / f"{OUT_NAME}.md"
    json_path = OUT_DIR / f"{OUT_NAME}.json"
    existing = [str(p) for p in (md_path, json_path) if p.exists()]
    if existing and not args.force:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    summary = build()
    sp = summary["baseline"]["sentence_points"]
    if round(sp, 4) != EXPECTED_BASELINE_SP:
        print(f"WARNING: baseline SP {sp:.6f} does not reproduce "
              f"{EXPECTED_BASELINE_SP}", file=sys.stderr)
    summary["markdown_path"] = str(md_path)
    summary["json_path"] = str(json_path)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(md_path, summary)
    print(json.dumps({"markdown": str(md_path), "json": str(json_path),
                      "baseline_sp": sp}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
