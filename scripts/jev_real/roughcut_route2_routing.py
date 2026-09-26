"""Route 2 ceiling: Jev scores everything, the least confident slice goes to a bigger model.

Offline estimate from stored data. Jev's ``jev_a`` rows from the 18-episode v3
run are rebuilt exactly the way ``roughcut_jev_report.py`` rebuilds them for the
published 80.47 (trim trigger 0.3, keep threshold 2.50). For each routed share,
the lowest-confidence sentences take the donor's archived per-sentence decision
instead: its keep/cut at its own file's Neutral threshold, its ``keep_words``
trims, and the retake flags the donor's published cut used. The mixed decision
set is then scored with um removal + delete silence layered on, through
``roughcut_partial_scoring.score_episode``.

Donors are the two published ladder arms whose per-sentence ``run_ratings`` are
archived: Luna chapters (``luna-chapters-rules5``) and the shipped Opus agentic
arm (``opus5-cc-agentic``). Their result files are found with the same globs the
bench page's arm registry gives them, and read the way
``benchmarks/roughcut/scripts/model_plus_deterministic.py`` reads them
(``run_ratings[0]``, the file's ``neutral_threshold``, corpus retake flags with
the run's ``retake_overrides`` applied, last file wins).

No model calls, no detector runs, $0. Writes
``docs/jev-real/roughcut-route2-routing.md`` and ``.json``.

Usage::

  python scripts/jev_real/roughcut_route2_routing.py
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
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
    load_episode, removals_cache_path, score_episode, sentence_states_for,
)
import roughcut_jev_report as report_mod  # noqa: E402

pooling = scoring_mod.pooling

IN_NAME = "roughcut-jev-all18-v3"
OUT_NAME = "roughcut-route2-routing"
JEV_ARM = "jev_a"
T_TRIM = 0.3
THRESHOLD = 2.5
SHARES = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
FOCUS_SHARE = 0.25
KEPT = ("full", "partial")

#: Fit set named under "Episode split" in the design spec; the rest are held out.
FIT = ["colman-02.04-skeleton-demo", "hampton-5.4-assignment-demo",
       "colman-03.03-muscles-crit", "edges-7.01-intro", "hampton-5.2-shape-demo",
       "perspective-14e-boxes-critique"]

#: Result globs copied from solar-sailer ``website-docs/bench/rough-cut-arms.json``.
DONORS = {
    "luna": {"ladder_key": "luna-chapters-rules5",
             "label": "Luna chapters (gpt-5.6-luna xhigh, rules5)",
             "globs": ["*-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json"]},
    "opus": {"ladder_key": "opus5-cc-agentic",
             "label": "Opus agentic (claude-opus-5-high, Claude Code, rules1)",
             "globs": ["2026-07-2*-partial-agentic-v1-claude-opus-5.json",
                       "2026-07-25-*-partial-agentic-claude-opus-5.json"]},
}

CONFIDENCE = {
    "top_mass": "top-level probability mass (max of the six score_probabilities)",
    "margin": "distance of the score from the 2.50 threshold, abs(score - 2.5)",
}
SELECTIONS = {
    "per_episode": "bottom X% of each episode",
    "pooled": "one global confidence cutoff that routes X% of all 8,943 sentences",
}

#: Cost and latency assumptions for a windowed donor call.
WINDOW_MAX = 40          # routed sentences per request
WINDOW_SPAN_CAP = 80     # a window closes once it spans this many sentences
CONTEXT = 5              # context sentences either side of the window span
TOKENS_PER_WORD = 1.3
PROMPT_TOKENS = 1500     # rules v5 system prompt is 1,047 words, plus task text
OUTPUT_TOKENS_PER_SENTENCE = 60   # score + keep_words JSON, low reasoning effort
LUNA_PRICE_IN, LUNA_PRICE_OUT = 0.20, 1.20   # USD per million, routing-notes.md
LUNA_SECONDS, CONCURRENCY, JEV_SECONDS = 6.0, 8, 6.0


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}"


def table(header, rows):
    return report_mod.table(header, rows)


def share_label(share):
    return f"{share * 100:g}%"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def fingerprint(path):
    path = Path(path)
    return {"path": str(path), "md5": md5(path),
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            .isoformat(timespec="seconds")}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_donor(key, episodes):
    """``{episode: {"decisions", "threshold", "path", "cost", "seconds", ...}}``."""
    spec = DONORS[key]
    paths = sorted({p for g in spec["globs"] for p in RESULTS_DIR.glob(g)})
    out = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
        threshold = float(doc["neutral_threshold"])
        for name, entry in doc["episodes"].items():
            if name not in episodes:
                continue
            runs = entry.get("run_ratings") or []
            if not runs:
                continue
            data = load_episode(name)
            retakes = dict(data["retake_flags"])
            overrides = entry.get("retake_overrides") or {}
            for sid in overrides.get("is_retake_true") or []:
                retakes[int(sid)] = True
            for sid in overrides.get("is_retake_false") or []:
                retakes[int(sid)] = False
            decisions = {}
            for raw_id, rating in runs[0].items():
                sid = int(raw_id)
                keep = float(rating["score"]) >= threshold
                # The keep decision is carried as 5 or 0 so one 2.50 threshold
                # applies Jev's cut and the donor's own-threshold cut at once.
                decisions[sid] = {"score": 5.0 if keep else 0.0,
                                  "keep_words": rating.get("keep_words"),
                                  "cut_retake": bool(retakes.get(sid, False))}
            session = entry.get("agent_session") or {}
            n_files = len(doc["episodes"])
            raw = {sid: {"score": float(runs[0][str(sid)]["score"]),
                         "keep_words": d["keep_words"], "cut_retake": d["cut_retake"]}
                   for sid, d in decisions.items()}
            out[name] = {
                "decisions": decisions, "raw": raw, "threshold": threshold,
                "path": str(path),
                "cost": entry.get("cost"),
                "seconds": session.get("wall_seconds"),
                "seconds_shared_by": n_files,
                "completion_tokens": session.get("completion_tokens"),
                "prompt_tokens": session.get("prompt_tokens"),
            }
    return out, [str(p) for p in paths]


def confidence(kind, row):
    """Lower means less confident, so it is routed first."""
    if kind == "top_mass":
        probs = row.get("score_probabilities") or {}
        return max(probs.values()) if probs else -1.0
    score = row.get("score")
    return -1.0 if score is None else abs(float(score) - THRESHOLD)


def select(kind, mode, share, by_episode_rows, episodes):
    """``({episode: set(sid)}, cutoff)``: the routed sentences at one share."""
    routed = {e: set() for e in episodes}
    if share <= 0:
        return routed, None
    if mode == "per_episode":
        for episode in episodes:
            rows = by_episode_rows[episode]
            ranked = sorted(rows, key=lambda sid: (confidence(kind, rows[sid]), sid))
            n = int(share * len(ranked) + 0.5)
            routed[episode] = set(ranked[:n])
        return routed, None
    pool = [(confidence(kind, by_episode_rows[e][sid]), episodes.index(e), sid, e)
            for e in episodes for sid in by_episode_rows[e]]
    pool.sort()
    n = int(share * len(pool) + 0.5)
    for _conf, _i, sid, episode in pool[:n]:
        routed[episode].add(sid)
    cutoff = pool[n - 1][0] if n else None
    return routed, cutoff


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

class Scorer:
    """Scores one episode's mixed decision set, cached by the routed set."""

    def __init__(self, jev, donors, removals):
        self.jev, self.donors, self.removals = jev, donors, removals
        self.cache = {}

    def mixed(self, donor, episode, routed):
        base = self.jev[episode]
        if not routed:
            return base
        swap = self.donors[donor][episode]["decisions"]
        return {sid: (swap[sid] if sid in routed else dec) for sid, dec in base.items()}

    def episode(self, donor, episode, routed):
        key = (donor if routed else None, episode, tuple(sorted(routed)))
        if key not in self.cache:
            self.cache[key] = score_episode(
                episode, self.mixed(donor, episode, routed), THRESHOLD,
                self.removals[episode])
        return self.cache[key]


def pool(rows):
    rows = list(rows)
    if not rows:
        return None
    sentences = [r["n_sentences"] for r in rows]
    return {
        "episodes": len(rows),
        "sentence_points": pooling.weighted_mean([r["sentence_points"] for r in rows],
                                                 sentences),
        "word_score": pooling.weighted_mean([r["word_score"] for r in rows],
                                            [r["word_count"] for r in rows]),
        "grade": pooling.weighted_mean([r["grade"] for r in rows],
                                       [r["dialogue_frames"] for r in rows]),
    }


def splits(per_episode, episodes):
    held = [e for e in episodes if e not in FIT]
    return {"fit": pool(per_episode[e] for e in FIT),
            "heldout": pool(per_episode[e] for e in held),
            "all": pool(per_episode[e] for e in episodes)}


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------

def ladder_rows(episodes, jev_sp):
    rows, _restricted, _covered = report_mod.ladder(episodes)
    out = [{"label": r["label"], "key": r["key"],
            "sentence_points": r["sentence_points_layered"]} for r in rows]
    out.append({"label": "Jev jev_a v3 (pure Jev)", "key": "jev_a",
                "sentence_points": jev_sp})
    out.sort(key=lambda r: -r["sentence_points"])
    return out


def placement(sp, ladder):
    """Rank among the ladder arms and a plain 'between A and B' string."""
    above = [r for r in ladder if r["sentence_points"] > sp + 5e-5]
    below = [r for r in ladder if r["sentence_points"] < sp - 5e-5]
    level = [r for r in ladder if abs(r["sentence_points"] - sp) <= 5e-5]
    rank = len(above) + 1
    if level:
        text = f"level with {level[0]['label']}"
    elif not above:
        text = f"top, above {below[0]['label']}"
    elif not below:
        text = f"bottom, below {above[-1]['label']}"
    else:
        text = f"below {above[-1]['label']}, above {below[0]['label']}"
    return {"rank": rank, "of": len(ladder) + (0 if level else 1), "text": text}


# ---------------------------------------------------------------------------
# flips and slice agreement
# ---------------------------------------------------------------------------

def states(episode, decisions, removals):
    human, model = sentence_states_for(episode, decisions, THRESHOLD, removals)
    return ({sid: s for sid, (s, _r) in human.items()},
            {sid: (s, r) for sid, (s, r) in model.items()})


def flip_block(scorer, donor, routed, episodes, human, jev_states):
    counts = Counter()
    for episode in episodes:
        ids = routed[episode]
        if not ids:
            continue
        _h, mixed = states(episode, scorer.mixed(donor, episode, ids),
                           scorer.removals[episode])
        for sid in ids:
            editor = human[episode][sid] in KEPT
            before = jev_states[episode][sid][0] in KEPT
            after = mixed[sid][0] in KEPT
            counts["routed"] += 1
            counts["jev_agreed"] += before == editor
            counts["hybrid_agreed"] += after == editor
            if before != after:
                counts["flips"] += 1
                counts["flips_right" if after == editor else "flips_wrong"] += 1
                counts[("kept_to_cut" if before else "cut_to_kept")
                       + ("_right" if after == editor else "_wrong")] += 1
            elif after and mixed[sid] != jev_states[episode][sid]:
                counts["trim_changed"] += 1
    return dict(counts)


def slice_agreement(routed, episodes, human, jev_states, donor_states):
    n = jev_ok = donor_ok = 0
    for episode in episodes:
        for sid in routed[episode]:
            editor = human[episode][sid] in KEPT
            n += 1
            jev_ok += (jev_states[episode][sid][0] in KEPT) == editor
            donor_ok += (donor_states[episode][sid][0] in KEPT) == editor
    return {"n": n, "jev": jev_ok / n if n else None,
            "donor": donor_ok / n if n else None}


# ---------------------------------------------------------------------------
# cost and latency
# ---------------------------------------------------------------------------

def windows_for(ids):
    """Greedy windows over the routed ids in transcript order: ``[(lo, hi, n)]``."""
    out = []
    for sid in sorted(ids):
        if out and out[-1][2] < WINDOW_MAX and sid - out[-1][0] < WINDOW_SPAN_CAP:
            lo, _hi, n = out[-1]
            out[-1] = (lo, sid, n + 1)
        else:
            out.append((sid, sid, 1))
    return out


def cost_block(routed, episodes, word_counts, out_tokens_per_sentence):
    per_episode = {}
    for episode in episodes:
        ids = routed[episode]
        wins = windows_for(ids)
        words = word_counts[episode]
        last = max(words)
        tokens_in = 0.0
        for lo, hi, _n in wins:
            span = range(max(0, lo - CONTEXT), min(last, hi + CONTEXT) + 1)
            tokens_in += PROMPT_TOKENS + TOKENS_PER_WORD * sum(words.get(s, 0)
                                                               for s in span)
        tokens_out = out_tokens_per_sentence * len(ids)
        waves = math.ceil(len(wins) / CONCURRENCY)
        per_episode[episode] = {
            "sentences": len(ids), "requests": len(wins),
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": (tokens_in * LUNA_PRICE_IN + tokens_out * LUNA_PRICE_OUT) / 1e6,
            "seconds": JEV_SECONDS + waves * LUNA_SECONDS,
        }
    values = list(per_episode.values())

    def stat(key):
        column = [v[key] for v in values]
        return {"mean": statistics.mean(column), "max": max(column)}
    return {"per_episode": per_episode,
            **{k: stat(k) for k in ("sentences", "requests", "input_tokens",
                                    "output_tokens", "cost_usd", "seconds")}}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build():
    decision_rows, requests, timing, run_paths = report_mod.load_run(IN_NAME)
    by_arm, episodes = report_mod.index_decisions(decision_rows)
    jev_rows = {e: dict(by_arm[JEV_ARM][e]) for e in episodes}
    words = {e: report_mod.word_ids_by_sentence(e) for e in episodes}
    jev, n_missing = report_mod.build_arm_decisions(by_arm, episodes, words, JEV_ARM,
                                                    T_TRIM, 0.0)
    removals, skipped = report_mod.load_removals(episodes)
    if skipped:
        raise SystemExit(f"no cached removals for {skipped}")

    donors, donor_paths, missing = {}, {}, {}
    for key in DONORS:
        loaded, paths = load_donor(key, episodes)
        absent = [e for e in episodes if e not in loaded]
        if absent:
            missing[key] = absent
            continue
        donors[key], donor_paths[key] = loaded, paths
    pooled_thresholds = {}
    for key in list(donors):
        raw = {e: donors[key][e]["raw"] for e in episodes}
        cal = scoring_mod.calibrate_threshold(raw, removals=removals)
        t = cal["threshold"]
        pooled_thresholds[key] = {"threshold": t,
                                  "sentence_points": cal["pooled"]["sentence_points"]}
        fair = {}
        for e in episodes:
            entry = dict(donors[key][e])
            entry["decisions"] = {
                sid: {"score": 5.0 if d["score"] >= t else 0.0,
                      "keep_words": d["keep_words"], "cut_retake": d["cut_retake"]}
                for sid, d in entry["raw"].items()}
            entry["threshold"] = t
            fair[e] = entry
        donors[f"{key}@pooled"] = fair
    scorer = Scorer(jev, donors, removals)

    jev_eps = {e: scorer.episode(None, e, set()) for e in episodes}
    jev_pooled = splits(jev_eps, episodes)
    ladder = ladder_rows(episodes, jev_pooled["all"]["sentence_points"])
    ladder_by_key = {r["key"]: r for r in ladder}

    human, jev_states, donor_states = {}, {}, {k: {} for k in donors}
    for episode in episodes:
        human[episode], jev_states[episode] = states(episode, jev[episode],
                                                     removals[episode])
        for key in donors:
            donor_states[key][episode] = states(
                episode, donors[key][episode]["decisions"], removals[episode])[1]

    word_counts = {}
    for episode in episodes:
        word_counts[episode] = {s["id"]: len(s.get("text", "").split())
                                for s in load_episode(episode)["sentences"]}

    sweeps = {}
    for key in donors:
        for kind in CONFIDENCE:
            for mode in SELECTIONS:
                rows = []
                for share in SHARES:
                    routed, cutoff = select(kind, mode, share, jev_rows, episodes)
                    per_ep = {e: scorer.episode(key, e, routed[e]) for e in episodes}
                    pooled = splits(per_ep, episodes)
                    rows.append({
                        "share": share, "cutoff": cutoff,
                        "routed": sum(len(v) for v in routed.values()),
                        "fit": pooled["fit"], "heldout": pooled["heldout"],
                        "all": pooled["all"],
                        "placement": placement(pooled["all"]["sentence_points"], ladder),
                        "per_episode_sp": {e: per_ep[e]["sentence_points"]
                                           for e in episodes},
                        "slice": slice_agreement(routed, episodes, human, jev_states,
                                                 donor_states[key]),
                    })
                sweeps[f"{key}/{kind}/{mode}"] = rows

    def mid_mean(tag):
        return statistics.mean(r["all"]["sentence_points"] for r in sweeps[tag]
                               if 0 < r["share"] < 1)

    best = {}
    for key in donors:
        tags = [f"{key}/{kind}/{mode}" for kind in CONFIDENCE for mode in SELECTIONS]
        best[key] = max(tags, key=mid_mean)
    best_kind = {k: v.split("/")[1] for k, v in best.items()}

    flips = {}
    for key in donors:
        kind = best_kind[key]
        for mode in SELECTIONS:
            routed, _ = select(kind, mode, FOCUS_SHARE, jev_rows, episodes)
            flips[f"{key}/{kind}/{mode}"] = flip_block(scorer, key, routed, episodes,
                                                       human, jev_states)

    archived_out = {}
    for key in [k for k in donors if "@" not in k]:
        comp = sum((donors[key][e]["completion_tokens"] or 0) for e in episodes)
        n_sent = sum(len(jev[e]) for e in episodes)
        archived_out[key] = comp / n_sent if comp else None
    heavy_out = archived_out.get("luna") or OUTPUT_TOKENS_PER_SENTENCE

    kind = best_kind.get("luna", "top_mass")
    costs = []
    for share in SHARES:
        for mode in SELECTIONS:
            routed, _ = select(kind, mode, share, jev_rows, episodes)
            block = cost_block(routed, episodes, word_counts, OUTPUT_TOKENS_PER_SENTENCE)
            heavy = cost_block(routed, episodes, word_counts, heavy_out)
            costs.append({"share": share, "mode": mode, "confidence": kind,
                          **{k: block[k] for k in ("sentences", "requests",
                                                   "input_tokens", "output_tokens",
                                                   "cost_usd", "seconds")},
                          "cost_usd_heavy_output": heavy["cost_usd"],
                          "per_episode": block["per_episode"]})

    jev_cost = statistics.mean(
        (report_mod.latency_rows(timing, requests, episodes)[e]["cost_usd"])
        for e in episodes)
    jev_seconds = report_mod.mean_seconds(
        report_mod.latency_rows(timing, requests, episodes), episodes)

    archive = {}
    for key in [k for k in donors if "@" not in k]:
        per = donors[key]
        costs_ep = [per[e]["cost"] for e in episodes if per[e]["cost"] is not None]
        secs = [per[e]["seconds"] / per[e]["seconds_shared_by"] for e in episodes
                if per[e]["seconds"] is not None]
        archive[key] = {
            "cost_per_episode_mean": statistics.mean(costs_ep) if costs_ep else None,
            "cost_per_episode_max": max(costs_ep) if costs_ep else None,
            "cost_total": sum(costs_ep) if costs_ep else None,
            "seconds_per_episode_mean": statistics.mean(secs) if secs else None,
            "seconds_per_episode_max": max(secs) if secs else None,
            "completion_tokens_per_sentence": archived_out[key],
            "thresholds": {e: per[e]["threshold"] for e in episodes},
        }

    reproduction = {"jev_a": {"mine": jev_pooled["all"]["sentence_points"],
                              "published": 0.8046516828804657}}
    for key in [k for k in donors if "@" not in k]:
        full = sweeps[f"{key}/top_mass/per_episode"][-1]["all"]["sentence_points"]
        reproduction[key] = {
            "mine": full,
            "published": ladder_by_key[DONORS[key]["ladder_key"]]["sentence_points"]}

    inputs = {k: fingerprint(v) for k, v in run_paths.items()}
    inputs["ladder_reference"] = fingerprint(report_mod.REFERENCE_JSON)
    for key, paths in donor_paths.items():
        for path in paths:
            inputs[f"{key}:{Path(path).name}"] = fingerprint(path)
    digest = hashlib.md5()
    for episode in episodes:
        digest.update(md5(removals_cache_path(episode)).encode())
    removals_digest = digest.hexdigest()

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_route2_routing.py",
        "episodes": episodes, "fit": FIT,
        "heldout": [e for e in episodes if e not in FIT],
        "sentences": sum(len(jev[e]) for e in episodes),
        "missing_jev_answers": n_missing, "donors_missing": missing,
        "donor_labels": {k: (DONORS[k]["label"] if "@" not in k else
                             DONORS[k.split("@")[0]]["label"]
                             + f", one pooled threshold {pooled_thresholds[k.split('@')[0]]['threshold']:.1f}")
                         for k in donors},
        "pooled_thresholds": pooled_thresholds,
        "jev": jev_pooled, "ladder": ladder, "reproduction": reproduction,
        "sweeps": sweeps, "best": best, "flips": flips, "costs": costs,
        "jev_cost_per_episode": jev_cost, "jev_seconds_per_episode": jev_seconds,
        "archive": archive, "heavy_output_tokens_per_sentence": heavy_out,
        "assumptions": {
            "window_max": WINDOW_MAX, "window_span_cap": WINDOW_SPAN_CAP,
            "context": CONTEXT, "tokens_per_word": TOKENS_PER_WORD,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens_per_sentence": OUTPUT_TOKENS_PER_SENTENCE,
            "price_in_per_m": LUNA_PRICE_IN, "price_out_per_m": LUNA_PRICE_OUT,
            "luna_seconds": LUNA_SECONDS, "concurrency": CONCURRENCY,
            "jev_seconds": JEV_SECONDS},
        "inputs": inputs, "removals_digest": removals_digest,
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def write_markdown(path, s, json_path):
    lines = []

    def add(text=""):
        lines.append(text)

    donors = list(s["donor_labels"])
    add("# Route 2 routing ceiling (developer-facing notes)")
    add()
    add(f"Generated {s['generated_utc']} by `{s['script']}` from stored decisions, the archived donor ratings and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.")
    add()
    add(f"Question: Jev scores all {s['sentences']:,} sentences of the 18 ladder episodes, the least confident slice goes to a bigger model, and that model's keep/cut and trims replace Jev's on the slice. The ceiling is estimated by swapping in the donor's archived per-sentence decision on the routed slice and rescoring. Jev sentences are the `jev_a` rows of `{IN_NAME}` rebuilt at trim trigger {T_TRIM} and scored at keep threshold {THRESHOLD:.2f}, the setting behind the published 80.47. A donor sentence is kept when its archived score clears its own file's Neutral threshold, carries its own `keep_words`, and takes the retake flags the donor's published cut used (corpus flags plus any `retake_overrides`). The threshold is not recalibrated after mixing.")
    add()
    add("## Reproduction check")
    add()
    rows = [["pure Jev (0%)", pct(s["reproduction"]["jev_a"]["mine"]),
             pct(s["reproduction"]["jev_a"]["published"])]]
    for key in s["reproduction"]:
        if key == "jev_a":
            continue
        rows.append([f"pure donor (100%), {s['donor_labels'][key]}",
                     pct(s["reproduction"][key]["mine"]),
                     pct(s["reproduction"][key]["published"])])
    lines.extend(table(["arm", "SENTENCE POINTS here", "published"], rows))
    add()
    add("Each donor's published number uses the Neutral keep threshold of its own result file, and those files hold one episode each (the Opus hampton-5.5 file holds five), so the threshold was picked per episode against the editor. Jev gets one pooled 2.50 on all 18. For a like-for-like ceiling every donor also appears below as `<donor>@pooled`: its raw scores recalibrated to one pooled threshold across the 18 with the harness's own sweep, everything else unchanged.")
    add()
    rows = [[s["donor_labels"][k], f"{v['threshold']:.1f}", pct(v["sentence_points"])]
            for k, v in s["pooled_thresholds"].items()]
    lines.extend(table(["donor", "pooled threshold", "pure donor SP at it"], rows))
    add()
    if s["donors_missing"]:
        add(f"Donors skipped for missing archives: {s['donors_missing']}.")
        add()

    add("## Sweep, pooled over all 18")
    add()
    add("Pooled SENTENCE POINTS for every confidence definition and selection mode. Confidence definitions: `top_mass` is " + CONFIDENCE["top_mass"] + "; `margin` is " + CONFIDENCE["margin"] + ". Selection: `per_episode` routes the " + SELECTIONS["per_episode"] + "; `pooled` routes everything under " + SELECTIONS["pooled"] + ".")
    add()
    for key in donors:
        tags = [f"{key}/{kind}/{mode}" for kind in CONFIDENCE for mode in SELECTIONS]
        header = ["routed share"] + [t.split("/", 1)[1] for t in tags]
        rows = []
        for i, share in enumerate(SHARES):
            rows.append([share_label(share)]
                        + [pct(s["sweeps"][t][i]["all"]["sentence_points"]) for t in tags])
        add(f"### Donor: {s['donor_labels'][key]}")
        add()
        lines.extend(table(header, rows))
        add()
        add(f"Best variant by mean pooled SP over the 10% to 75% shares: `{s['best'][key].split('/', 1)[1]}`.")
        add()

    add("## Best variant in full, with the ladder")
    add()
    ladder_text = ", ".join(f"{r['label']} {pct(r['sentence_points'])}" for r in s["ladder"])
    add(f"Ladder, with modules, same 18 episodes: {ladder_text}.")
    add()
    for key in donors:
        tag = s["best"][key]
        add(f"### {s['donor_labels'][key]}, `{tag.split('/', 1)[1]}`")
        add()
        rows = []
        for r in s["sweeps"][tag]:
            rows.append([share_label(r["share"]), r["routed"],
                         pct(r["fit"]["sentence_points"]),
                         pct(r["heldout"]["sentence_points"]),
                         pct(r["all"]["sentence_points"]), pct(r["all"]["word_score"]),
                         pct(r["all"]["grade"]),
                         f"{r['placement']['rank']} of {r['placement']['of']}",
                         r["placement"]["text"]])
        lines.extend(table(["share", "routed", "SP fit 6", "SP held-out 12", "SP all 18",
                            "WORD all 18", "GRADE all 18", "ladder rank", "placement"],
                           rows))
        add()

    add("## Per-episode or pooled selection")
    add()
    rows = []
    for key in [k for k in donors if "@" not in k]:
        kind = s["best"][key].split("/")[1]
        for i, share in enumerate(SHARES):
            if share in (0.0, 1.0):
                continue
            a = s["sweeps"][f"{key}/{kind}/per_episode"][i]
            b = s["sweeps"][f"{key}/{kind}/pooled"][i]
            rows.append([key, kind, share_label(share), pct(a["all"]["sentence_points"]),
                         pct(b["all"]["sentence_points"]),
                         f"{(b['all']['sentence_points'] - a['all']['sentence_points']) * 100:+.2f}",
                         "n/a" if b["cutoff"] is None else f"{b['cutoff']:.3f}"])
    lines.extend(table(["donor", "confidence", "share", "SP per-episode", "SP pooled",
                        "pooled minus per-episode", "global cutoff"], rows))
    add()
    add("Pooled selection routes more of the episodes where Jev is least sure overall and fewer of the ones it finds easy, so the per-episode share varies. The per-episode SP for every row is in the JSON under `sweeps`.")
    add()

    add("## Where the gains come from, 25% routed")
    add()
    add("A flip is a routed sentence whose keep/cut changed when the donor's decision replaced Jev's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor (kept means full or partial). Trim changed counts routed sentences both sides kept but trimmed differently.")
    add()
    rows = []
    for tag, c in s["flips"].items():
        routed = c.get("routed", 0)
        rows.append([tag, routed, c.get("flips", 0), c.get("flips_right", 0),
                     c.get("flips_wrong", 0),
                     f"{c.get('cut_to_kept_right', 0)} / {c.get('cut_to_kept_wrong', 0)}",
                     f"{c.get('kept_to_cut_right', 0)} / {c.get('kept_to_cut_wrong', 0)}",
                     c.get("trim_changed", 0),
                     pct(c.get("jev_agreed", 0) / routed if routed else None),
                     pct(c.get("hybrid_agreed", 0) / routed if routed else None)])
    lines.extend(table(["donor/confidence/selection", "routed", "flips", "right", "wrong",
                        "cut to kept right / wrong", "kept to cut right / wrong",
                        "trim changed", "agreement on slice, Jev",
                        "agreement on slice, after routing"], rows))
    add()
    add("Agreement with the editor on the routed slice, Jev against the donor, for the best variant at every share (read from each arm's full-run states):")
    add()
    rows = []
    for key in [k for k in donors if "@" not in k]:
        for r in s["sweeps"][s["best"][key]]:
            if r["share"] == 0:
                continue
            rows.append([key, share_label(r["share"]), r["slice"]["n"],
                         pct(r["slice"]["jev"]), pct(r["slice"]["donor"])])
    lines.extend(table(["donor", "share", "routed", "Jev agrees", "donor agrees"], rows))
    add()

    add("## Cost and latency")
    add()
    a = s["assumptions"]
    add(f"Assumptions, for a windowed Luna call on the routed sentences: routed ids are taken in transcript order and grouped greedily into windows of up to {a['window_max']} routed sentences, a window closing early once it spans {a['window_span_cap']} sentences; each request sends the whole span plus {a['context']} sentences of context either side, at {a['tokens_per_word']} tokens per word, plus {a['prompt_tokens']:,} tokens of prompt (the rules v5 system prompt is 1,047 words). Output is {a['output_tokens_per_sentence']} tokens per routed sentence (a score and optional keep_words, low reasoning effort); the heavy column instead uses the archived Luna chapters run's {s['heavy_output_tokens_per_sentence']:.0f} completion tokens per sentence, which is xhigh reasoning. Price ${a['price_in_per_m']:.2f} in and ${a['price_out_per_m']:.2f} out per million tokens, the gpt-5.6-luna rates `routing-notes.md` used (the router's model table carries no price for gpt-5.6-luna today), no cache discount. Latency is Jev's ~{a['jev_seconds']:.0f} s plus {a['luna_seconds']:.0f} s per wave of {a['concurrency']} concurrent Luna requests. Selection uses the best Luna confidence definition, `{s['costs'][0]['confidence']}`.")
    add()
    rows = []
    for c in s["costs"]:
        rows.append([share_label(c["share"]), c["mode"],
                     f"{c['sentences']['mean']:.0f}", c["sentences"]["max"],
                     f"{c['requests']['mean']:.1f}", c["requests"]["max"],
                     f"{c['input_tokens']['mean'] / 1000:.1f}k",
                     f"${c['cost_usd']['mean']:.4f}", f"${c['cost_usd']['max']:.4f}",
                     f"${c['cost_usd_heavy_output']['mean']:.4f}",
                     f"{c['seconds']['mean']:.0f}", f"{c['seconds']['max']:.0f}"])
    lines.extend(table(["share", "selection", "sentences/ep mean", "max", "requests/ep mean",
                        "max", "input tokens/ep mean", "Luna $/ep mean", "max",
                        "heavy-output $/ep mean", "s/ep mean", "max"], rows))
    add()
    add(f"Jev alone on this run: ${s['jev_cost_per_episode']:.4f} and {s['jev_seconds_per_episode']:.1f} s per episode on average; add it to every row for the hybrid total.")
    add()
    add("Sanity bound from the archived donor runs themselves (whole-episode agentic sessions, not windowed calls):")
    add()
    rows = []
    for key, arc in s["archive"].items():
        rows.append([s["donor_labels"][key],
                     "n/a" if arc["cost_per_episode_mean"] is None
                     else f"${arc['cost_per_episode_mean']:.3f}",
                     "n/a" if arc["cost_per_episode_max"] is None
                     else f"${arc['cost_per_episode_max']:.3f}",
                     "n/a" if arc["seconds_per_episode_mean"] is None
                     else f"{arc['seconds_per_episode_mean']:.0f}",
                     "n/a" if arc["seconds_per_episode_max"] is None
                     else f"{arc['seconds_per_episode_max']:.0f}",
                     "n/a" if arc["completion_tokens_per_sentence"] is None
                     else f"{arc['completion_tokens_per_sentence']:.0f}"])
    lines.extend(table(["donor", "$/ep mean", "$/ep max", "s/ep mean", "s/ep max",
                        "completion tokens/sentence"], rows))
    add()
    add("Seconds are the session wall clock; where one archived file covers several episodes (the Opus hampton-5.5 crits) its wall clock is split evenly across them. Opus cost is the registry's estimated basis, not a bill.")
    add()

    add("## Limits of this estimate")
    add()
    thresholds = {k: sorted(Counter(v["thresholds"].values()).items())
                  for k, v in s["archive"].items()}
    add(f"The donor decisions come from whole-episode agentic runs that read the full transcript, reviewed their own ratings, and were scored at a Neutral threshold calibrated per result file ({'; '.join(f'{k}: ' + ', '.join(f'{t} on {n}' for t, n in v) for k, v in thresholds.items())} episodes). A windowed call on 40 routed sentences with 5 lines of context sees far less, and nothing here says it would reproduce those decisions. Treat every routed number as an upper bound on what the same model gives on a window.")
    add()
    add("The per-file thresholds also flatter the donors against Jev, which uses one pooled 2.50 on every episode.")
    add()

    add("## Files")
    add()
    for key, value in s["inputs"].items():
        add(f"- input {key}: `{value['path']}`, md5 {value['md5'][:12]}, modified {value['modified_utc']}")
    add(f"- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 {s['removals_digest'][:12]}")
    add(f"- this file: `{path}`")
    add(f"- data: `{json_path}`")
    add()
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    md_path = OUT_DIR / f"{OUT_NAME}.md"
    json_path = OUT_DIR / f"{OUT_NAME}.json"
    existing = [str(p) for p in (md_path, json_path) if p.exists()]
    if existing and not args.force:
        parser.error(f"refusing to overwrite {existing}; pass --force")
    summary = build()
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(md_path, summary, json_path)
    print(json.dumps({"markdown": str(md_path), "json": str(json_path),
                      "reproduction": summary["reproduction"],
                      "best": summary["best"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
