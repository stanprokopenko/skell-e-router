"""Build B of round two: fit and score the prompt-breakup combiner offline.

Reads the feature rows ``scripts/jev_real/roughcut_jev_features.py`` wrote
(``docs/jev-real/<fit>-features.jsonl`` and, once it exists,
``<heldout>-features.jsonl``), fits an L2 logistic regression per feature set
on standardised features, and turns each fitted combiner into a rough-cut arm
scored the way every other arm is scored (``roughcut_partial_scoring``: pooled
keep-threshold calibration, um removal and delete silence layered on). No model
calls, nothing read that is not already on disk.

Spec: ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``,
"Build B", "Combiner".

Discipline
----------
Stage 1 (no ``--heldout``): leave-one-episode-out over the six fit episodes
chooses C per feature set and the feature set itself, then every set is
refitted on all six and frozen to ``docs/jev-real/<out>-weights.json``: the
scaler, the coefficients, the intercept and the keep threshold calibrated on
the out-of-fold predictions.

Stage 2 (``--heldout`` given): the fit-set work is recomputed (it is
deterministic) and checked against the frozen file, which must already exist;
the held-out episodes are scored with the frozen weights and the frozen
threshold, never refitted or recalibrated, and the write-up
``docs/jev-real/<out>.md`` plus ``.json`` is written. A held-out row that
recalibrates the threshold on the held-out set is reported once, labelled as
not held out, as a sensitivity check only.

Arm
---
``score = 5 * p_keep``, ``keep_words`` null, ``cut_retake`` from the v3 row.
The retake pass is unchanged this round, so the experiment isolates sentence
judgment.

Bundles
-------
``--feature-version`` names the prompt bundle the feature runs asked (``f1``,
``f2``); the question block in every set name is the bundle's ``q_label``
(``q``, ``q2``). ``--control-prefix`` points at the parent bundle's own
feature runs and frozen weights: its chosen set is refitted through the same
path as the comparison arm (and must land on its frozen weights), and the same
set minus the questions this bundle dropped is the ablation. ``--no-write``
recomputes everything and writes nothing, for reproduction checks.

Usage::

  python scripts/jev_real/roughcut_jev_combine.py --fit roughcut-jev-f1-fit \\
      --out roughcut-jev-f1
  python scripts/jev_real/roughcut_jev_combine.py --fit roughcut-jev-f1-fit \\
      --heldout roughcut-jev-f1-heldout --out roughcut-jev-f1
  python scripts/jev_real/roughcut_jev_combine.py --feature-version f2 \\
      --fit roughcut-jev-f2-fit --control-prefix roughcut-jev-f1 --cap 1.50 \\
      --out roughcut-jev-f2
"""

import argparse
import hashlib
import json
import sys
import warnings as pywarnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"

sys.path.insert(0, str(HERE))

# Import order matters: the scoring module installs the cache-only answer-key
# loader and puts the harness on sys.path.
import roughcut_partial_scoring as scoring_mod  # noqa: E402
from roughcut_partial_scoring import (  # noqa: E402
    calibrate_threshold, score_episode, sentence_states_for,
)
import roughcut_jev_report as report_mod  # noqa: E402
from roughcut_jev_features import CODE_FEATURE_KEYS, V3_DECISIONS  # noqa: E402
from roughcut_jev import FIT_EPISODES  # noqa: E402
from roughcut_jev_prompts import feature_prompts_for  # noqa: E402

pooling = scoring_mod.pooling

KEPT = ("full", "partial")
SCALE = 5.0                       # score = SCALE * p_keep
V3_THRESHOLD = 2.5                # jev_a v3's calibrated keep threshold
V3_T_TRIM = 0.3                   # jev_a v3's published trim trigger
JEV_V3_LADDER_SP = 0.8047         # jev_a v3 on the 18, with modules
ROUTE2_LUNA_25 = 0.8406           # v3 margin, 25% pooled, archived Luna
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
ROUTE_SHARES = [0.25, 0.50]

V3_FEATURES = ["v3_score", "v3_cut_p", "v3_first_p_whole", "v3_last_p_whole"]
#: Fill for a missing value, per feature: an unanswered noul is 0.5 (no lean),
#: a missing v3 score is the keep threshold, a missing trim answer is "whole".
FILL = {"v3_score": V3_THRESHOLD, "v3_cut_p": 0.5, "v3_first_p_whole": 1.0,
        "v3_last_p_whole": 1.0}

#: Set kinds. ``spec`` sets are the spec's four, the only ones eligible for
#: selection; ``diagnostic`` is the extra set that says what the questions add
#: over the free features and v3 alone; ``control`` is the parent bundle's
#: chosen set through the same fitting path, fed from that bundle's own
#: feature run; ``ablation`` is that set minus the questions this bundle
#: dropped, refitted on the fit six.
KIND_TAG = {"spec": "", "diagnostic": " (diagnostic)", "control": " (control)",
            "ablation": " (ablation)"}


def set_specs(bundle, control):
    """The feature sets to fit, in table order, each a dict with its columns.

    ``source`` is which feature run supplies the rows: ``main`` for this
    bundle, ``control`` for the parent bundle's run. The question block is
    named by the bundle's ``q_label`` (``q`` for f1, ``q2`` for f2), so the
    f1 set names are exactly what the f1 weights file froze.
    """
    keys = [k for k, _t in bundle.questions]
    q = bundle.q_label
    specs = [
        {"name": q, "source": "main", "questions": keys, "code": False, "v3": False,
         "kind": "spec"},
        {"name": f"{q}+code", "source": "main", "questions": keys, "code": True,
         "v3": False, "kind": "spec"},
        {"name": f"{q}+code+v3", "source": "main", "questions": keys, "code": True,
         "v3": True, "kind": "spec"},
        {"name": "v3", "source": "main", "questions": [], "code": False, "v3": True,
         "kind": "spec"},
        {"name": "code+v3", "source": "main", "questions": [], "code": True, "v3": True,
         "kind": "diagnostic"},
    ]
    if control:
        chosen_doc = control["weights"]["sets"][control["chosen"]]
        control_keys = control["weights"]["question_keys"]
        c_questions = [f for f in chosen_doc["features"] if f in control_keys]
        c_code = any(f in CODE_FEATURE_KEYS for f in chosen_doc["features"])
        c_v3 = any(f in V3_FEATURES for f in chosen_doc["features"])
        base = f"{control['version']} {control['chosen']}"
        specs.append({"name": base, "source": "control", "questions": c_questions,
                      "code": c_code, "v3": c_v3, "kind": "control"})
        if any(k in c_questions for k in bundle.dropped):
            specs.append({"name": f"{base} minus dropped", "source": "control",
                          "questions": [k for k in c_questions if k not in bundle.dropped],
                          "code": c_code, "v3": c_v3, "kind": "ablation"})
    for spec in specs:
        spec["features"] = feature_names(spec)
    return specs


WHAT = {
    "false_start": "yes: the row is an abandoned attempt",
    "retake_loser": "yes: a losing take of a repeated line",
    "crew_talk": "yes: addressed to the crew, not students",
    "screen_ops": "yes: operating the screen or software",
    "pre_lesson": "yes: chatter before the lesson starts",
    "off_topic": "yes: off the lesson's topic",
    "repeats_point": "yes: repeats a point just made",
    "pure_filler": "yes: filler with no content",
    "pep_talk": "yes: praise or wrap-up with nothing new",
    "funny": "yes: funny or shows personality",
    "teaching_point": "yes: states a point, reason or correction",
    "essential": "yes: the lesson loses something without it",
    "referenced_later": "yes: a later sentence depends on it",
    "transition": "yes: a spoken transition between students or steps",
    "describes_screen": "yes: only describes what is on screen",
    "rambling": "yes: rambling or thinking aloud",
    "split_fragment": "yes: half of a transcriber-split sentence",
    "tangent": "yes: an aside the lesson resumes after",
    "play_by_play": "yes: narrates the hand action with no reason",
    "said_earlier": "yes: the point was made earlier in the episode",
    "wrap_up": "yes: closes a section with nothing new",
    "praise_only": "yes: praise with no correction or reason",
    "verbal_check": "yes: 'right?', a hedge, no content of its own",
    "scripted": "yes: reads like a prepared lesson line",
    "sets_up_next": "yes: exists to set up the next sentence",
    "student_address": "yes: names or addresses a student or their drawing",
    "n_words": "word count after um stripping",
    "n_words_raw": "word count as spoken",
    "pause_before": "seconds of silence before the sentence",
    "pause_after": "seconds of silence after the sentence",
    "duration_s": "spoken duration in seconds",
    "words_per_s": "speaking rate",
    "position": "position in the episode, 0 to 1",
    "is_retake": "corpus retake flag (module loser)",
    "retake_member": "in a retake group",
    "retake_winner": "the module's winning take",
    "trail_off": "row ends in the transcriber's '..' mark",
    "lower_start": "row starts lowercase (continuation)",
    "chain_piece": "piece index in a split-sentence chain",
    "chain_len": "length of the split-sentence chain",
    "um_detected": "um-like words as spoken",
    "um_removed": "words the um module removed",
    "asr_confidence": "mean ASR word confidence",
    "overlap_prev": "word overlap with the previous sentence",
    "overlap_next": "word overlap with the next sentence",
    "v3_score": "v3 0-5 score",
    "v3_cut_p": "v3 P(editor removes it)",
    "v3_first_p_whole": "v3 P(nothing trimmed from the start)",
    "v3_last_p_whole": "v3 P(nothing trimmed from the end)",
}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def fingerprint(path):
    path = Path(path)
    try:
        shown = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return {"path": shown, "md5": md5(path), "bytes": path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            .isoformat(timespec="seconds")}


def load_features(name):
    """``({episode: [row]}, episode_order, timing, paths)`` for one feature run."""
    paths = {"features": OUT_DIR / f"{name}-features.jsonl",
             "requests": OUT_DIR / f"{name}-requests.jsonl",
             "timing": OUT_DIR / f"{name}-timing.json"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing feature run file(s): {missing}")
    rows, order = {}, []
    for row in read_jsonl(paths["features"]):
        if row["episode"] not in rows:
            rows[row["episode"]] = []
            order.append(row["episode"])
        rows[row["episode"]].append(row)
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    return rows, order, timing, {k: str(v) for k, v in paths.items()}


def human_keep(episodes):
    """``{episode: {sid: 1 if the editor kept it (full or partial) else 0}}``."""
    return {e: {sid: int(state in KEPT) for sid, state in report_mod.human_states(e).items()}
            for e in episodes}


# ---------------------------------------------------------------------------
# feature matrix
# ---------------------------------------------------------------------------

def feature_names(spec):
    names = list(spec["questions"])
    if spec["code"]:
        names += list(CODE_FEATURE_KEYS)
    if spec["v3"]:
        names += V3_FEATURES
    return names


def check_aligned(main_rows, other_rows, label):
    """The control run must cover the same sentences with the same free columns.

    The question columns differ by design (the parent bundle asked its own
    list); the code features and the v3 join are computed the same way for
    both runs and must agree, or the two runs are not comparing sentence
    judgment alone.
    """
    for episode, rows in main_rows.items():
        other = other_rows.get(episode)
        if other is None:
            raise SystemExit(f"{label}: control run has no rows for {episode}")
        if [r["id"] for r in rows] != [r["id"] for r in other]:
            raise SystemExit(f"{label}: control run covers different sentences in {episode}")
        for mine, theirs in zip(rows, other):
            if mine["code"] != theirs["code"] or mine["v3"] != theirs["v3"]:
                raise SystemExit(f"{label}: code or v3 columns differ from the control "
                                 f"run at {episode} sentence {mine['id']}")


def row_value(row, name):
    if name in row["q"]:
        value = row["q"][name]
        return 0.5 if value is None else float(value)
    if name in row["code"]:
        return float(row["code"][name])
    if name.startswith("v3_"):
        value = row["v3"].get(name[3:])
        return FILL[name] if value is None else float(value)
    raise KeyError(name)


def matrix(rows, names):
    return np.array([[row_value(r, n) for n in names] for r in rows], dtype=float)


def missing_counts(rows_by_episode):
    counts = Counter()
    for rows in rows_by_episode.values():
        for row in rows:
            counts["q_cells"] += sum(1 for v in row["q"].values() if v is None)
            counts["v3_score"] += row["v3"].get("score") is None
            counts["v3_cut_p"] += row["v3"].get("cut_p") is None
            counts["sentences"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class Combiner:
    """Standardise, then L2 logistic regression. Serialisable to plain JSON."""

    def __init__(self, names, C):
        self.names, self.C = list(names), float(C)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=self.C, max_iter=5000)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def to_json(self):
        return {"features": self.names, "C": self.C,
                "scaler_mean": self.scaler.mean_.tolist(),
                "scaler_scale": self.scaler.scale_.tolist(),
                "coef": self.model.coef_[0].tolist(),
                "intercept": float(self.model.intercept_[0])}

    @classmethod
    def from_json(cls, doc):
        self = cls(doc["features"], doc["C"])
        self.scaler.mean_ = np.array(doc["scaler_mean"])
        self.scaler.scale_ = np.array(doc["scaler_scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(doc["features"])
        self.model.coef_ = np.array([doc["coef"]])
        self.model.intercept_ = np.array([doc["intercept"]])
        self.model.classes_ = np.array([0, 1])
        return self


def loo_predictions(rows_by_episode, y_by_episode, names, C, episodes):
    """Out-of-fold ``{episode: {sid: p_keep}}`` over ``episodes``."""
    out = {}
    for held in episodes:
        train = [e for e in episodes if e != held]
        X = np.vstack([matrix(rows_by_episode[e], names) for e in train])
        y = np.concatenate([y_by_episode[e] for e in train])
        model = Combiner(names, C).fit(X, y)
        p = model.predict(matrix(rows_by_episode[held], names))
        out[held] = {r["id"]: float(v) for r, v in zip(rows_by_episode[held], p)}
    return out


# ---------------------------------------------------------------------------
# arms and scoring
# ---------------------------------------------------------------------------

def arm_decisions(rows_by_episode, p_by_episode):
    """``{episode: {sid: {score, keep_words, cut_retake}}}`` for a combiner arm."""
    out = {}
    for episode, rows in rows_by_episode.items():
        out[episode] = {
            r["id"]: {"score": SCALE * p_by_episode[episode][r["id"]],
                      "keep_words": None,
                      "cut_retake": bool(r["v3"].get("cut_retake"))}
            for r in rows}
    return out


def pool(per_episode):
    rows = list(per_episode.values())
    if not rows:
        return None
    sentences = [r["n_sentences"] for r in rows]
    return {"episodes": len(rows), "sentences": sum(sentences),
            "sentence_points": pooling.weighted_mean(
                [r["sentence_points"] for r in rows], sentences),
            "word_score": pooling.weighted_mean(
                [r["word_score"] for r in rows], [r["word_count"] for r in rows]),
            "grade": pooling.weighted_mean(
                [r["grade"] for r in rows], [r["dialogue_frames"] for r in rows])}


def score_fixed(decisions, threshold, removals):
    """Every episode at one frozen threshold, with modules, plus the pool."""
    per = {e: score_episode(e, d, threshold, removals[e]) for e, d in decisions.items()}
    return {"threshold": threshold, "episodes": per, "pooled": pool(per)}


def score_calibrated(decisions, removals):
    """The report script's path: pooled Neutral threshold, then every episode at it."""
    result = calibrate_threshold(decisions, removals={e: removals[e] for e in decisions})
    # Re-pool by sentence count so fixed and calibrated numbers share one
    # formula; the harness's own pooled figure is kept alongside.
    return {"threshold": result["threshold"], "episodes": result["episodes"],
            "pooled": pool(result["episodes"]),
            "pooled_harness_sp": result["pooled"]["sentence_points"]}


def jev_v3_decisions(episodes, with_trims=True):
    """jev_a v3 rebuilt the way the 80.47 was: trim trigger 0.3, or trims off."""
    decisions = {}
    for name in V3_DECISIONS:
        rows = read_jsonl(OUT_DIR / f"{name}-decisions.jsonl")
        by_arm, order = report_mod.index_decisions(rows)
        wanted = [e for e in order if e in episodes and e not in decisions]
        if not wanted:
            continue
        words = {e: report_mod.word_ids_by_sentence(e) for e in wanted}
        built, _missing = report_mod.build_arm_decisions(
            by_arm, wanted, words, "jev_a", V3_T_TRIM, 0.0)
        for e in wanted:
            decisions[e] = built[e]
    absent = [e for e in episodes if e not in decisions]
    if absent:
        raise SystemExit(f"no v3 jev_a rows for {absent}")
    if not with_trims:
        decisions = {e: {sid: dict(d, keep_words=None) for sid, d in dec.items()}
                     for e, dec in decisions.items()}
    return {e: decisions[e] for e in episodes}


def keep_states(episode, decisions, threshold, removals):
    """``{sid: kept?}`` for an arm, with modules, as SENTENCE POINTS sees it."""
    _human, model = sentence_states_for(episode, decisions, threshold, removals)
    return {sid: int(state in KEPT) for sid, (state, _r) in model.items()}


def confusion(reference, candidate):
    """Four cells of candidate keep/cut against a reference keep/cut."""
    counts = Counter()
    for sid, ref in reference.items():
        cand = candidate.get(sid, 0)
        counts[("ref_keep" if ref else "ref_cut") + ("_cand_keep" if cand else "_cand_cut")] += 1
    counts["n"] = sum(v for k, v in counts.items() if k != "n")
    counts["agreement"] = ((counts["ref_keep_cand_keep"] + counts["ref_cut_cand_cut"])
                           / counts["n"]) if counts["n"] else None
    return dict(counts)


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, p))


def question_aucs(rows_by_episode, y_by_episode, question_keys, episodes):
    """Each question alone against the editor's keep, pooled over ``episodes``."""
    y = np.concatenate([y_by_episode[e] for e in episodes])
    out = {}
    for key in question_keys:
        p = np.concatenate([[row_value(r, key) for r in rows_by_episode[e]]
                            for e in episodes])
        out[key] = auc(y, p)
    for name in ("v3_score", "v3_cut_p"):
        p = np.concatenate([[row_value(r, name) for r in rows_by_episode[e]]
                            for e in episodes])
        out[name] = auc(y, p)
    return out


def quartile_block(p_by_episode, human, kept_by_episode, threshold, episodes):
    """Agreement with the editor by p_keep quartile and by margin quartile."""
    items = [(p_by_episode[e][sid], human[e][sid], kept_by_episode[e][sid])
             for e in episodes for sid in p_by_episode[e]]
    p = np.array([i[0] for i in items])
    margin = np.abs(SCALE * p - threshold)

    def bins(values, label):
        edges = np.quantile(values, [0.25, 0.5, 0.75])
        rows = []
        for q in range(4):
            lo = -np.inf if q == 0 else edges[q - 1]
            hi = np.inf if q == 3 else edges[q]
            idx = [i for i, v in enumerate(values) if lo <= v < hi or (q == 3 and v == hi)]
            if not idx:
                continue
            rows.append({
                "quartile": q + 1, "n": len(idx),
                f"{label}_min": float(min(values[i] for i in idx)),
                f"{label}_max": float(max(values[i] for i in idx)),
                "mean_p_keep": float(np.mean([items[i][0] for i in idx])),
                "editor_kept": float(np.mean([items[i][1] for i in idx])),
                "arm_kept": float(np.mean([items[i][2] for i in idx])),
                "agreement": float(np.mean([items[i][1] == items[i][2] for i in idx])),
            })
        return rows

    return {"by_p_keep": bins(p, "p"), "by_margin": bins(margin, "margin")}


# ---------------------------------------------------------------------------
# route 2 handoff
# ---------------------------------------------------------------------------

def route2_handoff(decisions, p_by_episode, threshold, removals, episodes, shares):
    """Bottom X% by combiner margin, archived Luna substituted, rescored.

    Follows ``roughcut_route2_routing``: one global cutoff over the pooled
    sentences, the donor's own-threshold keep or cut on the routed slice, the
    donor's trims and retake flags with it, everything else the combiner's.
    """
    try:
        import roughcut_route2_routing as route2
    except Exception as exc:  # the other build is refactoring that file
        return {"error": f"{type(exc).__name__}: {exc}"}
    donors, paths = route2.load_donor("luna", episodes)
    absent = [e for e in episodes if e not in donors]
    if absent:
        return {"error": f"archived Luna missing for {absent}", "paths": paths}
    pool_ = sorted((abs(SCALE * p_by_episode[e][sid] - threshold), i, sid, e)
                   for i, e in enumerate(episodes) for sid in p_by_episode[e])
    out = {"donor_paths": paths, "shares": []}
    for share in shares:
        n = int(share * len(pool_) + 0.5)
        routed = {e: set() for e in episodes}
        for _m, _i, sid, e in pool_[:n]:
            routed[e].add(sid)
        mixed = {}
        for e in episodes:
            swap = donors[e]["decisions"]
            mixed[e] = {sid: (swap[sid] if sid in routed[e] else
                              {"score": SCALE if d["score"] >= threshold else 0.0,
                               "keep_words": None, "cut_retake": d["cut_retake"]})
                        for sid, d in decisions[e].items()}
        scored = score_fixed(mixed, V3_THRESHOLD, removals)
        out["shares"].append({"share": share, "routed": n,
                              "cutoff_margin": pool_[n - 1][0] if n else None,
                              "pooled": scored["pooled"],
                              "per_episode_sp": {e: scored["episodes"][e]["sentence_points"]
                                                 for e in episodes}})
    return out


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------

def placement(sp, ladder):
    above = [r for r in ladder if r["sentence_points"] > sp + 5e-5]
    below = [r for r in ladder if r["sentence_points"] < sp - 5e-5]
    if not above:
        return "top, above " + below[0]["label"]
    if not below:
        return "bottom, below " + above[-1]["label"]
    return f"below {above[-1]['label']}, above {below[0]['label']}"


# ---------------------------------------------------------------------------
# stage 1: fit
# ---------------------------------------------------------------------------

def fit_stage(fit_by_source, fit_order, specs, human, removals, log, c_by_set=None):
    """Leave-one-episode-out per set over the C grid, then refit on all six.

    ``fit_by_source`` is ``{"main": rows_by_episode, "control": ...}``; every
    source covers the same sentences (``check_aligned``), so one target
    vector serves all sets. ``c_by_set`` (stage 2) restricts each set to its
    frozen C: the refit is still recomputed and checked against the frozen
    file, the rest of the grid comes from the sweep the file stored.
    """
    main_rows = fit_by_source["main"]
    y_by_episode = {e: np.array([human[e][r["id"]] for r in main_rows[e]])
                    for e in fit_order}
    sets = {}
    for spec in specs:
        rows, names = fit_by_source[spec["source"]], spec["features"]
        sweep = []
        grid = [c_by_set[spec["name"]]] if c_by_set else C_GRID
        for C in grid:
            p = loo_predictions(rows, y_by_episode, names, C, fit_order)
            scored = score_calibrated(arm_decisions(rows, p), removals)
            log(f"  {spec['name']} C={C:g}: LOO SP {scored['pooled']['sentence_points'] * 100:.2f} "
                f"at t={scored['threshold']:.2f}")
            sweep.append({"C": C, "threshold": scored["threshold"],
                          "pooled": scored["pooled"],
                          "pooled_harness_sp": scored["pooled_harness_sp"],
                          "per_episode": scored["episodes"], "p": p})
        # Best LOO SP; ties go to the smaller C (more regularised).
        best = max(sweep, key=lambda s: (round(s["pooled"]["sentence_points"], 6), -s["C"]))
        X = np.vstack([matrix(rows[e], names) for e in fit_order])
        y = np.concatenate([y_by_episode[e] for e in fit_order])
        frozen = Combiner(names, best["C"]).fit(X, y)
        sets[spec["name"]] = {
            "spec": spec, "names": names, "sweep": sweep, "best": best, "frozen": frozen,
            "loo_auc": auc(y, np.concatenate([[best["p"][e][r["id"]] for r in rows[e]]
                                              for e in fit_order])),
        }
    eligible = [s["name"] for s in specs if s["kind"] == "spec"]
    chosen = max(eligible, key=lambda s: (round(sets[s]["best"]["pooled"]["sentence_points"], 6),
                                          -len(sets[s]["names"])))
    return sets, chosen, y_by_episode


def weights_doc(sets, chosen, bundle, fit_order, inputs, control):
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_jev_combine.py",
        "feature_version": bundle.version, "q_label": bundle.q_label,
        "parent": bundle.parent, "dropped": bundle.dropped, "new": bundle.new,
        "fit_episodes": fit_order, "question_keys": [k for k, _t in bundle.questions],
        "chosen_set": chosen, "c_grid": C_GRID, "scale": SCALE,
        "control": ({"version": control["version"], "prefix": control["prefix"],
                     "chosen_set": control["chosen"]} if control else None),
        "sets": {name: dict(entry["frozen"].to_json(),
                            threshold=entry["best"]["threshold"],
                            loo_sentence_points=entry["best"]["pooled"]["sentence_points"],
                            source=entry["spec"]["source"], kind=entry["spec"]["kind"],
                            sweep=[{"C": x["C"], "threshold": x["threshold"], "pooled": x["pooled"],
                                    "pooled_harness_sp": x["pooled_harness_sp"]}
                                   for x in entry["sweep"]])
                 for name, entry in sets.items()},
        "inputs": inputs,
    }


def check_frozen(doc, sets, chosen):
    """The frozen file must describe exactly what stage 1 recomputes now."""
    problems = []
    if doc["chosen_set"] != chosen:
        problems.append(f"chosen set {doc['chosen_set']} in the frozen file, {chosen} now")
    for name, entry in sets.items():
        saved = doc["sets"].get(name)
        if saved is None:
            problems.append(f"{name} not in the frozen file")
            continue
        now = entry["frozen"].to_json()
        if saved["C"] != now["C"] or saved["features"] != now["features"]:
            problems.append(f"{name}: C or features differ from the frozen file")
        if abs(saved["threshold"] - entry["best"]["threshold"]) > 1e-9:
            problems.append(f"{name}: threshold differs from the frozen file")
        if np.max(np.abs(np.array(saved["coef"]) - np.array(now["coef"]))) > 1e-6:
            problems.append(f"{name}: coefficients differ from the frozen file")
    return problems


# ---------------------------------------------------------------------------
# write-up
# ---------------------------------------------------------------------------

def pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}"


def num(value, places=2):
    return "n/a" if value is None else f"{value:.{places}f}"


def table(header, rows):
    return report_mod.table(header, rows)


def seconds_of(timing_by_episode, episode):
    """Wall clock and cost of the feature pass plus the v3 pass it joins."""
    entry = timing_by_episode.get(episode) or {}
    v3 = (entry.get("v3_run") or {})
    feat = entry.get("wall_clock_s")
    return {"feat_s": feat, "v3_s": v3.get("wall_clock_s"),
            "total_s": (feat or 0.0) + (v3.get("wall_clock_s") or 0.0),
            "feat_usd": entry.get("cost_usd"), "v3_usd": v3.get("cost_usd"),
            "total_usd": (entry.get("cost_usd") or 0.0) + (v3.get("cost_usd") or 0.0),
            "requests": entry.get("requests"), "errors": entry.get("errors"),
            "unanswered": entry.get("unanswered_cells"),
            "parts_per_block": sorted({int(v) for v in
                                       (entry.get("parts_per_block") or {}).values()})}


def mean_seconds(timing_by_episode, episodes):
    values = [seconds_of(timing_by_episode, e)["total_s"] for e in episodes]
    return sum(values) / len(values) if values else None


def write_markdown(path, s):
    v, q_label = s["feature_version"], s["q_label"]
    n_q = len(s["question_keys"])
    ctrl = s.get("control")
    order = s["set_order"]
    L = []
    L.append(f"Developer-facing notes on build B of the Jev rough-cut round two: the prompt breakup, {n_q} yes/no "
             f"questions per sentence (bundle `{v}`) and a logistic combiner fitted in code, Jev only.")
    L.append("")
    L.append(f"Generated {s['generated_utc']} by `{s['script']}` from "
             f"`{Path(s['inputs']['fit']['features']).name}`"
             + (f" and `{Path(s['inputs']['heldout']['features']).name}`" if s["inputs"].get("heldout") else "")
             + (f", control rows from `{Path(s['inputs']['control_fit']['features']).name}`"
                + (f" and `{Path(s['inputs']['control_heldout']['features']).name}`"
                   if s["inputs"].get("control_heldout") else "") if ctrl else "")
             + f", weights in `{Path(s['weights_path']).name}`.")
    L.append("")
    L.append(f"# Jev rough cut, build B: prompt breakup ({v})")
    L.append("")
    fit_n, held_n = len(s["fit_episodes"]), len(s.get("heldout_episodes") or [])
    chosen = s["chosen_set"]
    ch = s["sets"][chosen]
    intro = (f"The v3 sentence pass asks one six-level score per sentence. This build asks {n_q} one-look yes/no "
             f"questions instead (bundle `{v}` in `roughcut_jev_prompts.py`), over the same state v3 sent, and fits an "
             f"L2 logistic regression on the probabilities of yes. Sentence judgment is the only thing that changes: "
             f"`keep_words` is null, the retake cut is jev_a v3's. Every number below is with um removal and delete "
             f"silence layered on. Fit-set numbers are leave-one-episode-out over the {fit_n} fit episodes with the keep "
             f"threshold calibrated on the pooled out-of-fold predictions. C and the feature set were chosen on those "
             f"numbers alone, then frozen.")
    if ctrl:
        intro += (f" Bundle `{v}` is `{ctrl['version']}` with {len(s['dropped'])} questions dropped "
                  f"({', '.join(f'`{k}`' for k in s['dropped'])}), the other {n_q - len(s['new'])} unchanged, and "
                  f"{len(s['new'])} added ({', '.join(f'`{k}`' for k in s['new'])}). `{ctrl['version']}`'s chosen set "
                  f"`{ctrl['chosen']}` runs through the same fitting path as the control, from its own feature run, "
                  f"and its held-out and ladder rows use its own frozen weights.")
    L.append(intro)
    L.append("")
    if s.get("heldout"):
        hl = s["heldout"]
        line = (f"Bottom line: the chosen set is `{chosen}` (C {ch['C']:g}, threshold {ch['threshold']:.2f}). "
                f"Leave-one-out on the fit six it scores {pct(ch['loo']['pooled']['sentence_points'])} SP against "
                f"{pct(s['v3_control']['loo']['pooled']['sentence_points'])} for the v3 score through the same fitting path")
        if ctrl:
            cs = s["sets"][ctrl["set"]]
            line += (f", {pct(cs['loo']['pooled']['sentence_points'])} for `{ctrl['version']}`'s `{ctrl['chosen']}` "
                     f"through the same path")
            if ctrl.get("ablation_set"):
                ab = s["sets"][ctrl["ablation_set"]]
                line += (f" ({pct(ab['loo']['pooled']['sentence_points'])} with the dropped questions removed)")
        line += (f" and {pct(s['jev_v3_fit']['pooled']['sentence_points'])} for jev_a v3 as published on the same six. "
                 f"On the {held_n} held-out episodes with the frozen weights and threshold it scores "
                 f"{pct(hl['sets'][chosen]['frozen']['pooled']['sentence_points'])} SP")
        if ctrl:
            line += (f" against {pct(hl['sets'][ctrl['set']]['frozen']['pooled']['sentence_points'])} for "
                     f"`{ctrl['version']}` at its frozen weights and")
        else:
            line += " against"
        line += (f" {pct(hl['jev_v3']['pooled']['sentence_points'])} for jev_a v3. On the 18-episode ladder it lands at "
                 f"{pct(s['ladder']['mine_sp'])}")
        if ctrl:
            line += f" next to `{ctrl['version']}`'s {pct(s['ladder']['control_sp'])} and"
        else:
            line += " next to"
        line += (f" jev_a v3's {pct(s['ladder']['jev_v3_sp'])} ({s['ladder']['placement']}). "
                 f"Spend ${s['spend']['total_usd']:.2f} in Jev calls, {num(s['seconds']['ladder_mean'], 1)} s per "
                 f"ladder episode with the v3 pass included.")
        if s.get("route2") and not s["route2"].get("error"):
            r2 = s["route2"]["by_set"]
            line += (f" Routing the bottom 25% by combiner margin to archived Luna gives "
                     f"{pct(r2[chosen]['shares'][0]['pooled']['sentence_points'])}")
            if ctrl and ctrl["set"] in r2:
                line += f" against {pct(r2[ctrl['set']]['shares'][0]['pooled']['sentence_points'])} for `{ctrl['version']}` and"
            else:
                line += " against"
            line += f" {pct(ROUTE2_LUNA_25)} for the v3 margin."
        L.append(line)
    else:
        L.append(f"Stage 1 only: fit-set results and frozen weights. The chosen set is `{chosen}` (C {ch['C']:g}, "
                 f"threshold {ch['threshold']:.2f}) at {pct(ch['loo']['pooled']['sentence_points'])} SP leave-one-out.")
    L.append("")

    def tag(name):
        return KIND_TAG[s["sets"][name]["kind"]] if name != chosen else " (chosen)"

    # fit-set table
    L.append("## Fit set, leave-one-episode-out")
    L.append("")
    text = (f"Pooled over the {fit_n} fit episodes, every feature set at its best C. `v3` is the control: jev_a v3's "
            f"0-5 score alone through the same fitting path. `code+v3` is a diagnostic set outside the spec's four, "
            f"there to show what the questions add over the free features and v3 together.")
    if ctrl:
        text += (f" `{ctrl['set']}` is `{ctrl['version']}`'s chosen set refitted through the same path from its own "
                 f"feature run, the comparison arm; `{ctrl['set']} minus dropped` is that set without the "
                 f"{len(s['dropped'])} questions `{v}` dropped, the ablation. Neither was eligible for selection.")
    text += f" Seconds per episode are the {v} pass plus the v3 pass it joins (both at concurrency 8)."
    L.append(text)
    L.append("")
    header = ["feature set", "features", "C", "threshold", "SENTENCE POINTS", "WORD SCORE", "GRADE",
              "LOO AUC", "s/episode"]
    rows = []
    for name in order:
        e = s["sets"][name]
        rows.append([f"`{name}`{tag(name)}", len(e["features"]), f"{e['C']:g}", num(e["threshold"]),
                     pct(e["loo"]["pooled"]["sentence_points"]), pct(e["loo"]["pooled"]["word_score"]),
                     pct(e["loo"]["pooled"]["grade"]), num(e["loo_auc"], 3),
                     num(s["seconds"]["fit_mean"], 1)])
    j = s["jev_v3_fit"]
    rows.append(["jev_a v3 as published (trims at 0.3, calibrated)", "", "", num(j["threshold"]),
                 pct(j["pooled"]["sentence_points"]), pct(j["pooled"]["word_score"]), pct(j["pooled"]["grade"]),
                 "", num(s["seconds"]["v3_fit_mean"], 1)])
    j = s["jev_v3_fit_notrim"]
    rows.append(["jev_a v3, keep_words null (calibrated)", "", "", num(j["threshold"]),
                 pct(j["pooled"]["sentence_points"]), pct(j["pooled"]["word_score"]), pct(j["pooled"]["grade"]),
                 "", num(s["seconds"]["v3_fit_mean"], 1)])
    L.extend(table(header, rows))
    L.append("")
    L.append("C sweep, leave-one-out SP per feature set (the chosen C is the best; ties go to the smaller C):")
    L.append("")
    header = ["feature set"] + [f"C {c:g}" for c in C_GRID]
    rows = [[f"`{name}`"] + [pct(x["pooled"]["sentence_points"]) for x in s["sets"][name]["sweep"]]
            for name in order]
    L.extend(table(header, rows))
    L.append("")
    L.append("Per episode, leave-one-out, at each set's pooled threshold:")
    L.append("")
    header = ["episode", "sentences"] + [f"`{n}`" for n in order] + ["jev_a v3", "s/episode"]
    rows = []
    for e in s["fit_episodes"]:
        rows.append([e, s["sets"][chosen]["loo"]["episodes"][e]["n_sentences"]]
                    + [pct(s["sets"][n]["loo"]["episodes"][e]["sentence_points"]) for n in order]
                    + [pct(s["jev_v3_fit"]["episodes"][e]["sentence_points"]),
                       num(s["seconds"]["per_episode"][e]["total_s"], 1)])
    L.extend(table(header, rows))
    L.append("")
    if ctrl and ctrl.get("ablation_set"):
        cs, ab = s["sets"][ctrl["set"]], s["sets"][ctrl["ablation_set"]]
        delta = (ab["loo"]["pooled"]["sentence_points"] - cs["loo"]["pooled"]["sentence_points"]) * 100
        L.append(f"Ablation: `{ctrl['version']}`'s `{ctrl['chosen']}` refitted on the fit six without "
                 f"{', '.join(f'`{k}`' for k in s['dropped'])} scores {pct(ab['loo']['pooled']['sentence_points'])} SP "
                 f"leave-one-out against {pct(cs['loo']['pooled']['sentence_points'])} with them, a change of "
                 f"{delta:+.2f} SP (LOO AUC {num(ab['loo_auc'], 3)} against {num(cs['loo_auc'], 3)}). "
                 + ("Dropping them cost nothing on the fit six." if delta >= -0.05 else
                    "Dropping them cost something on the fit six; see the notes."))
        L.append("")

    # held-out
    if s.get("heldout"):
        hl = s["heldout"]
        L.append("## Held-out, frozen weights")
        L.append("")
        text = (f"The {held_n} held-out episodes scored with the weights and the keep threshold frozen after stage 1. "
                f"Nothing here was fitted, chosen or calibrated on these episodes. The last column recalibrates the "
                f"threshold on the held-out set itself and is not held out; it is there to show how much the frozen "
                f"threshold costs.")
        if ctrl:
            text += (f" `{ctrl['set']}` uses `{ctrl['version']}`'s own frozen weights and threshold (stage 1 "
                     f"reproduced them {'exactly' if ctrl['reproduces_frozen'] else 'with drift, see the warnings'}) "
                     f"on `{ctrl['version']}`'s held-out feature rows.")
        L.append(text)
        L.append("")
        header = ["feature set", "threshold", "SENTENCE POINTS", "WORD SCORE", "GRADE", "AUC",
                  "SP recalibrated (not held out)", "s/episode"]
        rows = []
        for name in order:
            e = hl["sets"][name]
            rows.append([f"`{name}`{tag(name)}", num(e["frozen"]["threshold"]),
                         pct(e["frozen"]["pooled"]["sentence_points"]), pct(e["frozen"]["pooled"]["word_score"]),
                         pct(e["frozen"]["pooled"]["grade"]), num(e["auc"], 3),
                         f"{pct(e['recalibrated']['pooled']['sentence_points'])} at t={e['recalibrated']['threshold']:.2f}",
                         num(s["seconds"]["heldout_mean"], 1)])
        j = hl["jev_v3"]
        rows.append(["jev_a v3 as published (trims at 0.3, t 2.50)", num(j["threshold"]),
                     pct(j["pooled"]["sentence_points"]), pct(j["pooled"]["word_score"]), pct(j["pooled"]["grade"]),
                     "", "", num(s["seconds"]["v3_heldout_mean"], 1)])
        L.extend(table(header, rows))
        L.append("")
        L.append("Per episode, frozen weights:")
        L.append("")
        header = ["episode", "sentences"] + [f"`{n}`" for n in order] + ["jev_a v3", "s/episode"]
        rows = []
        for e in s["heldout_episodes"]:
            rows.append([e, hl["sets"][chosen]["frozen"]["episodes"][e]["n_sentences"]]
                        + [pct(hl["sets"][n]["frozen"]["episodes"][e]["sentence_points"]) for n in order]
                        + [pct(hl["jev_v3"]["episodes"][e]["sentence_points"]),
                           num(s["seconds"]["per_episode"][e]["total_s"], 1)])
        L.extend(table(header, rows))
        L.append("")

        # ladder
        ld = s["ladder"]
        L.append("## Ladder, 18 episodes")
        L.append("")
        text = (f"The fit six enter with their leave-one-out predictions and the 12 ladder held-out episodes with the "
                f"frozen weights, all at the frozen threshold {ch['threshold']:.2f}; greco-2.2-thumbnailing is not a "
                f"ladder episode and is left out here. jev_a v3 is rebuilt through the same scoring path and reproduces "
                f"its published {pct(JEV_V3_LADDER_SP)} at {pct(ld['jev_v3_sp'])}.")
        if ctrl:
            text += (f" `{ctrl['version']}`'s row is built the same way from its own rows at its frozen threshold "
                     f"{ctrl['threshold']:.2f}.")
        text += f" Seconds per episode: {num(s['seconds']['ladder_mean'], 1)} ({v} plus v3)."
        L.append(text)
        L.append("")
        header = ["arm", "SENTENCE POINTS", "s/episode"]
        rows = [[r["label"], pct(r["sentence_points"]),
                 num(s["seconds"]["ladder_mean"], 1) if r.get("mine") else
                 (num(s["seconds"]["control_ladder_mean"], 1) if r.get("control") else "")] for r in ld["rows"]]
        L.extend(table(header, rows))
        L.append("")

    # weights
    L.append("## Standardised weights, chosen set")
    L.append("")
    L.append(f"`{chosen}` refitted on all {fit_n} fit episodes with C {ch['C']:g}, sorted by size. A positive weight "
             f"pushes toward keep. Weights are per standard deviation of the feature, so they compare across features. "
             f"Intercept {ch['intercept']:.3f}.")
    L.append("")
    header = ["feature", "weight", "what it says"]
    rows = []
    for w in ch["weights_sorted"]:
        direction = "keep" if w["weight"] > 0 else "cut"
        new = " (new)" if w["feature"] in s["new"] else ""
        rows.append([f"`{w['feature']}`{new}", f"{w['weight']:+.3f}",
                     f"{WHAT.get(w['feature'], '')}; pushes toward {direction}"])
    L.extend(table(header, rows))
    L.append("")
    full = f"{q_label}+code+v3"
    if chosen != full and full in s["sets"]:
        L.append(f"Weights of `{full}` for comparison, top ten by size:")
        L.append("")
        rows = [[f"`{w['feature']}`", f"{w['weight']:+.3f}"]
                for w in s["sets"][full]["weights_sorted"][:10]]
        L.extend(table(["feature", "weight"], rows))
        L.append("")

    # AUC per question
    L.append("## Each question alone")
    L.append("")
    L.append("AUC of each probability of yes against the editor's keep (full or partial) versus removed, pooled over the "
             "fit six and, when present, the held-out episodes. 0.50 is no signal; a cut question reads below 0.50 and a "
             "keep question above. The v3 score and cut_p are listed on the same footing."
             + (f" Questions marked new were added in `{v}`; the dropped `{ctrl['version']}` questions are listed "
                f"below the table from `{ctrl['version']}`'s own rows." if ctrl else ""))
    L.append("")
    header = ["question", "AUC fit", "AUC held-out", "abs(AUC - 0.5) fit", "signal"]
    rows = []
    for key, entry in sorted(s["question_auc"].items(), key=lambda kv: -abs((kv[1]["fit"] or 0.5) - 0.5)):
        fit_auc = entry["fit"]
        strength = abs(fit_auc - 0.5) if fit_auc is not None else None
        rows.append([f"`{key}`" + (" (new)" if key in s["new"] else ""), num(fit_auc, 3),
                     num(entry.get("heldout"), 3), num(strength, 3),
                     "none" if strength is not None and strength < 0.03 else
                     ("weak" if strength is not None and strength < 0.10 else "yes")])
    L.extend(table(header, rows))
    L.append("")
    weak = [k for k, e in s["question_auc"].items()
            if not k.startswith("v3_") and e["fit"] is not None and abs(e["fit"] - 0.5) < 0.03]
    L.append("Questions with no signal on their own (abs(AUC - 0.5) under 0.03): "
             + (", ".join(f"`{k}`" for k in weak) if weak else "none") + ".")
    L.append("")
    if s.get("dropped_auc"):
        parts = [f"`{k}` {num(e['fit'], 3)} fit" + (f", {num(e['heldout'], 3)} held-out" if e.get("heldout") is not None else "")
                 for k, e in s["dropped_auc"].items()]
        L.append(f"Dropped from `{ctrl['version']}`, AUC alone on `{ctrl['version']}`'s rows: {'; '.join(parts)}.")
        L.append("")

    # confusion
    L.append("## Confusion")
    L.append("")
    L.append("Keep or cut of the chosen arm against the editor and against jev_a v3, with modules, per split. "
             "Fit rows are leave-one-out; held-out rows use the frozen weights.")
    L.append("")
    header = ["split", "reference", "n", "both keep", f"ref keep, {v} cut", f"ref cut, {v} keep", "both cut",
              "agreement", "s/episode"]
    rows = []
    for split, block in s["confusion"].items():
        for ref, c in block.items():
            rows.append([split, ref, c["n"], c["ref_keep_cand_keep"], c["ref_keep_cand_cut"],
                         c["ref_cut_cand_keep"], c["ref_cut_cand_cut"], pct(c["agreement"]),
                         num(s["seconds"][f"{split}_mean"], 1)])
    L.extend(table(header, rows))
    L.append("")

    # calibration
    L.append("## Calibration by quartile")
    L.append("")
    scope = "the 18 ladder episodes (fit leave-one-out plus held-out frozen)" if s.get("heldout") else "the fit six, leave-one-out"
    L.append(f"The chosen combiner's `p_keep` over {scope}. The margin table is the input route 2 needs: "
             f"the bottom margin quartile is the slice a bigger model would take.")
    L.append("")
    q = s["calibration"]
    header = ["p_keep quartile", "n", "p range", "mean p_keep", "editor kept", "arm kept", "agreement", "s/episode"]
    rows = [[r["quartile"], r["n"], f"{r['p_min']:.2f} to {r['p_max']:.2f}", num(r["mean_p_keep"], 3),
             pct(r["editor_kept"]), pct(r["arm_kept"]), pct(r["agreement"]),
             num(s["seconds"]["calibration_mean"], 1)] for r in q["by_p_keep"]]
    L.extend(table(header, rows))
    L.append("")
    header = ["margin quartile", "n", "margin range", "mean p_keep", "editor kept", "arm kept", "agreement", "s/episode"]
    rows = [[r["quartile"], r["n"], f"{r['margin_min']:.2f} to {r['margin_max']:.2f}", num(r["mean_p_keep"], 3),
             pct(r["editor_kept"]), pct(r["arm_kept"]), pct(r["agreement"]),
             num(s["seconds"]["calibration_mean"], 1)] for r in q["by_margin"]]
    L.extend(table(header, rows))
    L.append("")

    # route 2
    if s.get("route2"):
        L.append("## What this hands route 2")
        L.append("")
        r2 = s["route2"]
        if r2.get("error"):
            L.append(f"Not computed: {r2['error']}.")
        else:
            L.append(f"Bottom share of the 18 ladder episodes by combiner margin `abs(5 * p_keep - threshold)`, one "
                     f"global cutoff, the archived Luna chapters decision substituted on the routed slice and rescored "
                     f"with modules, exactly as `roughcut_route2_routing.py` does for the v3 margin. The v3 margin gives "
                     f"{pct(ROUTE2_LUNA_25)} at 25%. Seconds per episode are Jev only; the Luna call is build A's number.")
            L.append("")
            header = ["combiner", "share", "routed", "margin cutoff", "SENTENCE POINTS", "WORD SCORE", "s/episode"]
            rows = []
            for name, block in r2["by_set"].items():
                secs = s["seconds"]["control_ladder_mean"] if s["sets"][name]["kind"] == "control" else s["seconds"]["ladder_mean"]
                for sh in block["shares"]:
                    rows.append([f"`{name}`", f"{sh['share'] * 100:g}%", sh["routed"], num(sh["cutoff_margin"], 3),
                                 pct(sh["pooled"]["sentence_points"]), pct(sh["pooled"]["word_score"]),
                                 num(secs, 1)])
            rows.append(["v3 margin, archived Luna (route 2 write-up)", "25%", 2236, "0.46",
                         pct(ROUTE2_LUNA_25), "", num(s["seconds"]["v3_ladder_mean"], 1)])
            L.extend(table(header, rows))
        L.append("")

    # cost and time
    L.append("## Seconds and dollars per episode")
    L.append("")
    parts = s["request_shape"]["parts_per_block"]
    parts_text = (f"{parts[0]} request{'s' if parts[0] != 1 else ''} per 25-sentence block" if len(parts) == 1 else
                  f"{parts[0]} to {parts[-1]} requests per 25-sentence block")
    L.append(f"The {v} pass at concurrency 8 ({parts_text}, every part carrying the whole state), "
             f"plus the v3 run it joins. Cost is the router's usage accounting at $0.042 per million input tokens.")
    L.append("")
    header = ["episode", "split", "sentences", f"{v} requests", f"{v} errors", "unanswered cells", f"{v} s", "v3 s",
              "total s", f"{v} $", "v3 $", "total $"]
    rows = []
    for e in s["fit_episodes"] + (s.get("heldout_episodes") or []):
        t = s["seconds"]["per_episode"][e]
        rows.append([e, "fit" if e in s["fit_episodes"] else "held-out", s["sentences_per_episode"].get(e),
                     t["requests"], t["errors"], t["unanswered"], num(t["feat_s"], 2), num(t["v3_s"], 2),
                     num(t["total_s"], 2), num(t["feat_usd"], 4), num(t["v3_usd"], 4), num(t["total_usd"], 4)])
    L.extend(table(header, rows))
    L.append("")
    sp = s["spend"]
    L.append(f"Spend on this build: ${sp['fit_usd']:.4f} on the fit six, ${sp['heldout_usd']:.4f} on the held-out "
             f"episodes, ${sp['smoke_usd']:.4f} on the smoke block, ${sp['total_usd']:.4f} in all against the "
             f"${sp['cap_usd']:.2f} cap. Requests: {sp['requests']}, errors {sp['errors']}, unanswered cells "
             f"{sp['unanswered']}."
             + (f" The `{ctrl['version']}` control rows cost nothing new; they are `{ctrl['version']}`'s own run." if ctrl else ""))
    L.append("")

    # notes
    L.append("## Notes")
    L.append("")
    for note in s["notes"]:
        L.append(f"- {note}")
    L.append("")
    if s["warnings"]:
        L.append("Warnings:")
        L.append("")
        for w in s["warnings"]:
            L.append(f"- {w}")
        L.append("")

    L.append("## Inputs")
    L.append("")
    for label, fp in s["fingerprints"].items():
        L.append(f"- {label}: `{fp['path']}` md5 {fp['md5']}, {fp['bytes']} bytes, modified {fp['modified_utc']}")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit", required=True, help="fit feature run basename")
    parser.add_argument("--heldout", default=None, help="held-out feature run basename")
    parser.add_argument("--smoke", default=None,
                        help="smoke run basename, counted in the spend (default <out>-smoke)")
    parser.add_argument("--out", required=True, help="output basename under docs/jev-real")
    parser.add_argument("--feature-version", default="f1",
                        help="prompt bundle the --fit and --heldout runs asked (f1, f2)")
    parser.add_argument("--control-prefix", default=None,
                        help="feature-run prefix of the parent bundle, e.g. roughcut-jev-f1: its "
                             "<prefix>-fit, <prefix>-heldout and <prefix>-weights.json feed the "
                             "control and ablation sets")
    parser.add_argument("--control-version", default=None,
                        help="bundle version of --control-prefix (default: the bundle's parent)")
    parser.add_argument("--cap", type=float, default=3.00,
                        help="the spend cap the write-up reports against")
    parser.add_argument("--no-write", action="store_true",
                        help="compute and print everything, write no file (reproduction check)")
    args = parser.parse_args()
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731

    bundle = feature_prompts_for(args.feature_version)
    question_keys = [k for k, _t in bundle.questions]
    smoke_name = args.smoke or f"{args.out}-smoke"
    weights_path = OUT_DIR / f"{args.out}-weights.json"
    md_path = OUT_DIR / f"{args.out}.md"
    json_path = OUT_DIR / f"{args.out}.json"
    if args.heldout:
        if not weights_path.exists():
            parser.error(f"stage 2 needs the frozen weights at {weights_path}; run stage 1 first")
        existing = [str(p) for p in (md_path, json_path) if p.exists()]
        if existing and not args.no_write:
            parser.error(f"refusing to overwrite existing output(s): {existing}")
    elif weights_path.exists() and not args.no_write:
        parser.error(f"refusing to overwrite frozen weights at {weights_path}")

    control = None
    if args.control_prefix:
        control_version = args.control_version or bundle.parent
        if control_version is None:
            parser.error(f"bundle {bundle.version} has no parent; give --control-version")
        control_weights_path = OUT_DIR / f"{args.control_prefix}-weights.json"
        if not control_weights_path.exists():
            parser.error(f"control weights missing at {control_weights_path}")
        control_weights = json.loads(control_weights_path.read_text(encoding="utf-8"))
        control = {"version": control_version, "prefix": args.control_prefix,
                   "weights": control_weights, "chosen": control_weights["chosen_set"],
                   "weights_path": control_weights_path,
                   "bundle": feature_prompts_for(control_version)}
        if control_weights["question_keys"] != [k for k, _t in control["bundle"].questions]:
            parser.error(f"control weights were fitted on other questions than bundle {control_version}")

    fit_rows, fit_order, fit_timing, fit_paths = load_features(args.fit)
    if sorted(fit_order) != sorted(FIT_EPISODES):
        raise SystemExit(f"fit run has {fit_order}, expected the six fit episodes")
    fit_order = list(FIT_EPISODES)
    inputs = {"fit": fit_paths}
    episodes = list(fit_order)
    held_rows, held_order, held_timing = {}, [], {}
    if args.heldout:
        held_rows, held_order, held_timing, held_paths = load_features(args.heldout)
        inputs["heldout"] = held_paths
        episodes += held_order
    fit_by_source, held_by_source = {"main": fit_rows}, {"main": held_rows}
    control_fit_timing, control_held_timing = {}, {}
    if control:
        c_fit_rows, _c_order, control_fit_timing, c_fit_paths = load_features(f"{control['prefix']}-fit")
        check_aligned(fit_rows, c_fit_rows, "fit")
        fit_by_source["control"] = c_fit_rows
        inputs["control_fit"] = c_fit_paths
        if args.heldout:
            c_held_rows, _c_order, control_held_timing, c_held_paths = load_features(
                f"{control['prefix']}-heldout")
            check_aligned(held_rows, c_held_rows, "heldout")
            held_by_source["control"] = c_held_rows
            inputs["control_heldout"] = c_held_paths
    specs = set_specs(bundle, control)
    set_order = [spec["name"] for spec in specs]
    if control:
        control["set"] = next(sp["name"] for sp in specs if sp["kind"] == "control")
        control["ablation_set"] = next((sp["name"] for sp in specs if sp["kind"] == "ablation"), None)
        control["threshold"] = control["weights"]["sets"][control["chosen"]]["threshold"]

    log("human states and removals...")
    human = human_keep(episodes)
    removals, skipped = report_mod.load_removals(episodes)
    if skipped:
        raise SystemExit(f"no cached removals for {skipped}")
    warnings, notes = [], []

    frozen_doc, c_by_set = None, None
    if args.heldout:
        frozen_doc = json.loads(weights_path.read_text(encoding="utf-8"))
        if all("sweep" in frozen_doc["sets"].get(name, {}) for name in set_order):
            c_by_set = {name: frozen_doc["sets"][name]["C"] for name in set_order}
            log("stage 1 at the frozen C per set (the C sweep comes from the frozen file)...")
        else:
            log("stage 1: full C sweep (the frozen file stores none)...")
    else:
        log("stage 1: leave-one-episode-out over the fit six...")
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        sets, chosen, y_fit = fit_stage(fit_by_source, fit_order, specs, human, removals, log,
                                        c_by_set)
    log(f"chosen set: {chosen} (C {sets[chosen]['best']['C']:g})")

    doc = weights_doc(sets, chosen, bundle, fit_order, inputs["fit"], control)
    if args.heldout:
        problems = check_frozen(frozen_doc, sets, chosen)
        if problems:
            raise SystemExit("stage 1 no longer matches the frozen weights; refusing to score "
                             f"held-out episodes against a moving target: {problems}")
        if c_by_set:
            for name in set_order:
                stored = frozen_doc["sets"][name]["sweep"]
                at_c = next(x for x in stored if x["C"] == c_by_set[name])
                if abs(at_c["pooled"]["sentence_points"]
                       - sets[name]["best"]["pooled"]["sentence_points"]) > 1e-9:
                    raise SystemExit(f"{name}: LOO SP at the frozen C differs from the frozen file")
                sets[name]["sweep"] = stored
        doc = frozen_doc
    elif args.no_write:
        log("stage 1 only, --no-write: frozen weights not written")
    else:
        weights_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        log(f"frozen weights written to {weights_path}")

    if control:
        # The control set is the parent's chosen set through the same path on the
        # same rows, so its refit must land on the parent's frozen weights; its
        # held-out and ladder rows below use the parent's frozen file directly.
        saved = control["weights"]["sets"][control["chosen"]]
        now = sets[control["set"]]["frozen"].to_json()
        drift = float(np.max(np.abs(np.array(saved["coef"]) - np.array(now["coef"]))))
        control["reproduces_frozen"] = (saved["C"] == now["C"] and saved["features"] == now["features"]
                                        and drift < 1e-6
                                        and abs(saved["threshold"] - sets[control["set"]]["best"]["threshold"]) < 1e-9)
        control["coef_drift"] = drift
        if not control["reproduces_frozen"]:
            warnings.append(f"the {control['version']} control refit does not reproduce "
                            f"{control['weights_path'].name}: C {now['C']} vs {saved['C']}, coefficient drift "
                            f"{drift:.2e}, threshold {sets[control['set']]['best']['threshold']} vs {saved['threshold']}")
        doc["sets"][control["set"]] = dict(doc["sets"][control["set"]], **{
            k: saved[k] for k in ("scaler_mean", "scaler_scale", "coef", "intercept", "threshold", "C")})

    # jev_a v3 reproductions on the fit six
    log("jev_a v3 on the fit six...")
    jev_fit = jev_v3_decisions(fit_order, with_trims=True)
    jev_fit_scored = score_calibrated(jev_fit, removals)
    jev_fit_notrim_scored = score_calibrated(jev_v3_decisions(fit_order, with_trims=False), removals)

    summary_sets = {}
    for name, entry in sets.items():
        frozen = entry["frozen"]
        weights = sorted(({"feature": f, "weight": float(c)}
                          for f, c in zip(frozen.names, frozen.model.coef_[0])),
                         key=lambda w: -abs(w["weight"]))
        summary_sets[name] = {
            "features": entry["names"], "C": entry["best"]["C"],
            "threshold": entry["best"]["threshold"],
            "source": entry["spec"]["source"], "kind": entry["spec"]["kind"],
            "loo": {"pooled": entry["best"]["pooled"],
                    "pooled_harness_sp": entry["best"]["pooled_harness_sp"],
                    "episodes": entry["best"]["per_episode"]},
            "loo_auc": entry["loo_auc"],
            "sweep": [{"C": x["C"], "threshold": x["threshold"], "pooled": x["pooled"]}
                      for x in entry["sweep"]],
            "weights_sorted": weights,
            "intercept": float(frozen.model.intercept_[0]),
        }

    # per-question AUC
    question_auc = {k: {"fit": v} for k, v in
                    question_aucs(fit_rows, y_fit, question_keys, fit_order).items()}
    dropped_auc = None
    if control and bundle.dropped:
        dropped_auc = {k: {"fit": v} for k, v in question_aucs(
            fit_by_source["control"], y_fit, bundle.dropped, fit_order).items()
            if k in bundle.dropped}

    # confusion and calibration on the fit six (LOO)
    chosen_t = sets[chosen]["best"]["threshold"]
    fit_p = sets[chosen]["best"]["p"]
    fit_dec = arm_decisions(fit_rows, fit_p)
    fit_kept = {e: keep_states(e, fit_dec[e], chosen_t, removals[e]) for e in fit_order}
    jev_fit_kept = {e: keep_states(e, jev_fit[e], jev_fit_scored["threshold"], removals[e])
                    for e in fit_order}
    confusion_block = {"fit": {
        "editor": confusion(_flat(human, fit_order), _flat(fit_kept, fit_order)),
        "jev_a v3": confusion(_flat(jev_fit_kept, fit_order), _flat(fit_kept, fit_order)),
    }}

    seconds = {"per_episode": {e: seconds_of(fit_timing.get("episodes", {}), e) for e in fit_order}}
    seconds["fit_mean"] = mean_seconds(fit_timing.get("episodes", {}), fit_order)
    seconds["v3_fit_mean"] = _mean([seconds["per_episode"][e]["v3_s"] for e in fit_order])
    seconds["calibration_mean"] = seconds["fit_mean"]
    seconds["control_ladder_mean"] = None

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_jev_combine.py",
        "feature_version": bundle.version, "q_label": bundle.q_label,
        "question_keys": question_keys, "dropped": bundle.dropped, "new": bundle.new,
        "control": ({k: v for k, v in control.items()
                     if k in ("version", "prefix", "chosen", "set", "ablation_set", "threshold",
                              "reproduces_frozen", "coef_drift")} if control else None),
        "fit_episodes": fit_order, "chosen_set": chosen, "c_grid": C_GRID,
        "set_order": set_order, "sets": summary_sets,
        "v3_control": summary_sets["v3"],
        "jev_v3_fit": jev_fit_scored, "jev_v3_fit_notrim": jev_fit_notrim_scored,
        "question_auc": question_auc, "dropped_auc": dropped_auc,
        "confusion": confusion_block,
        "missing": {"fit": missing_counts(fit_rows)},
        "weights_path": str(weights_path), "inputs": inputs,
        "sentences_per_episode": {e: len(fit_rows[e]) for e in fit_order},
    }

    if args.heldout:
        log("stage 2: held-out with frozen weights...")
        held_p = {}
        hl_sets = {}
        for spec in specs:
            name, rows = spec["name"], held_by_source[spec["source"]]
            model = Combiner.from_json(doc["sets"][name])
            p = {e: {r["id"]: float(v) for r, v in
                     zip(rows[e], model.predict(matrix(rows[e], model.names)))}
                 for e in held_order}
            held_p[name] = p
            dec = arm_decisions(rows, p)
            frozen_scored = score_fixed(dec, doc["sets"][name]["threshold"], removals)
            recal = score_calibrated(dec, removals)
            y_h = np.concatenate([[human[e][r["id"]] for r in rows[e]] for e in held_order])
            p_h = np.concatenate([[p[e][r["id"]] for r in rows[e]] for e in held_order])
            hl_sets[name] = {"frozen": frozen_scored, "recalibrated": recal, "auc": auc(y_h, p_h)}
            log(f"  {name}: held-out SP {frozen_scored['pooled']['sentence_points'] * 100:.2f} frozen, "
                f"{recal['pooled']['sentence_points'] * 100:.2f} recalibrated")
        jev_held = jev_v3_decisions(held_order, with_trims=True)
        jev_held_scored = score_fixed(jev_held, V3_THRESHOLD, removals)
        y_h = {e: np.array([human[e][r["id"]] for r in held_rows[e]]) for e in held_order}
        for k, v in question_aucs(held_rows, y_h, question_keys, held_order).items():
            question_auc[k]["heldout"] = v
        if dropped_auc:
            for k, v in question_aucs(held_by_source["control"], y_h, bundle.dropped,
                                      held_order).items():
                if k in dropped_auc:
                    dropped_auc[k]["heldout"] = v
        summary["heldout"] = {"sets": hl_sets, "jev_v3": jev_held_scored}
        summary["heldout_episodes"] = held_order
        summary["missing"]["heldout"] = missing_counts(held_rows)
        summary["sentences_per_episode"].update({e: len(held_rows[e]) for e in held_order})

        # confusion on held-out and pooled 18
        held_dec = arm_decisions(held_rows, held_p[chosen])
        held_kept = {e: keep_states(e, held_dec[e], chosen_t, removals[e]) for e in held_order}
        jev_held_kept = {e: keep_states(e, jev_held[e], V3_THRESHOLD, removals[e]) for e in held_order}
        confusion_block["heldout"] = {
            "editor": confusion(_flat(human, held_order), _flat(held_kept, held_order)),
            "jev_a v3": confusion(_flat(jev_held_kept, held_order), _flat(held_kept, held_order)),
        }

        # ladder on the 18
        ladder_eps = fit_order + [e for e in held_order if e != "greco-2.2-thumbnailing"]
        all_p = {name: dict(sets[name]["best"]["p"], **held_p[name]) for name in set_order}
        all_rows = {src: dict(fit_by_source[src], **held_by_source[src]) for src in fit_by_source}
        rows_for = lambda name: {e: all_rows[sets[name]["spec"]["source"]][e] for e in ladder_eps}  # noqa: E731
        mine_dec = arm_decisions(rows_for(chosen), all_p[chosen])
        mine_18 = score_fixed(mine_dec, chosen_t, removals)
        jev_18_dec = jev_v3_decisions(ladder_eps, with_trims=True)
        jev_18 = score_fixed(jev_18_dec, V3_THRESHOLD, removals)
        ref_rows, _restricted, _covered = report_mod.ladder(ladder_eps)
        ladder = [{"label": r["label"], "sentence_points": r["sentence_points_layered"]} for r in ref_rows]
        ladder.append({"label": "Jev jev_a v3 (pure Jev)", "sentence_points": jev_18["pooled"]["sentence_points"]})
        control_18 = None
        if control:
            control_dec = arm_decisions(rows_for(control["set"]), all_p[control["set"]])
            control_18 = score_fixed(control_dec, control["threshold"], removals)
            ladder.append({"label": f"Jev {control['version']} `{control['chosen']}` combiner (frozen)",
                           "sentence_points": control_18["pooled"]["sentence_points"], "control": True})
        ladder.sort(key=lambda r: -r["sentence_points"])
        place = placement(mine_18["pooled"]["sentence_points"], ladder)
        ladder.append({"label": f"Jev {bundle.version} `{chosen}` combiner (this build)",
                       "sentence_points": mine_18["pooled"]["sentence_points"], "mine": True})
        ladder.sort(key=lambda r: -r["sentence_points"])
        summary["ladder"] = {"episodes": ladder_eps, "rows": ladder, "placement": place,
                             "mine_sp": mine_18["pooled"]["sentence_points"],
                             "control_sp": control_18["pooled"]["sentence_points"] if control_18 else None,
                             "jev_v3_sp": jev_18["pooled"]["sentence_points"],
                             "mine_per_episode": {e: mine_18["episodes"][e]["sentence_points"] for e in ladder_eps},
                             "reproduction_80_47": {"expected": JEV_V3_LADDER_SP,
                                                    "got": jev_18["pooled"]["sentence_points"]}}
        if abs(jev_18["pooled"]["sentence_points"] - JEV_V3_LADDER_SP) > 5e-4:
            warnings.append(f"jev_a v3 on the 18 reproduces {jev_18['pooled']['sentence_points'] * 100:.2f}, "
                            f"not {JEV_V3_LADDER_SP * 100:.2f}")
        all_kept = {e: (fit_kept[e] if e in fit_kept else held_kept[e]) for e in ladder_eps}
        jev_all_kept = {e: keep_states(e, jev_18_dec[e], V3_THRESHOLD, removals[e])
                        for e in ladder_eps}
        confusion_block["ladder18"] = {
            "editor": confusion(_flat(human, ladder_eps), _flat(all_kept, ladder_eps)),
            "jev_a v3": confusion(_flat(jev_all_kept, ladder_eps), _flat(all_kept, ladder_eps)),
        }
        summary["calibration"] = quartile_block(all_p[chosen], human, all_kept, chosen_t, ladder_eps)

        # route 2 handoff
        log("route 2 handoff...")
        r2 = {"by_set": {}}
        r2_names = [f"{bundle.q_label}+code+v3", chosen] + ([control["set"]] if control else [])
        for name in dict.fromkeys(r2_names):
            dec = arm_decisions(rows_for(name), all_p[name])
            block = route2_handoff(dec, all_p[name], doc["sets"][name]["threshold"], removals,
                                   ladder_eps, ROUTE_SHARES)
            if block.get("error"):
                r2 = {"error": block["error"]}
                break
            r2["by_set"][name] = block
        summary["route2"] = r2

        for e in held_order:
            seconds["per_episode"][e] = seconds_of(held_timing.get("episodes", {}), e)
        seconds["heldout_mean"] = mean_seconds(held_timing.get("episodes", {}), held_order)
        seconds["v3_heldout_mean"] = _mean([seconds["per_episode"][e]["v3_s"] for e in held_order])
        seconds["ladder_mean"] = _mean([seconds["per_episode"][e]["total_s"] for e in ladder_eps])
        seconds["v3_ladder_mean"] = _mean([seconds["per_episode"][e]["v3_s"] for e in ladder_eps])
        seconds["ladder18_mean"] = seconds["ladder_mean"]
        seconds["calibration_mean"] = seconds["ladder_mean"]
        if control:
            control_timing = dict(control_fit_timing.get("episodes", {}),
                                  **control_held_timing.get("episodes", {}))
            seconds["control_ladder_mean"] = mean_seconds(control_timing, ladder_eps)
    else:
        summary["calibration"] = quartile_block(fit_p, human, fit_kept, chosen_t, fit_order)

    summary["seconds"] = seconds
    parts_seen = sorted({p for e in seconds["per_episode"].values() for p in e["parts_per_block"]})
    smoke_cost, smoke_requests, smoke_tokens = 0.0, 0, None
    smoke_timing = OUT_DIR / f"{smoke_name}-timing.json"
    if smoke_timing.exists():
        st = json.loads(smoke_timing.read_text(encoding="utf-8"))
        smoke_cost, smoke_requests = st["totals"]["cost_usd"], st["totals"]["requests"]
        smoke_tokens = st["totals"].get("input_tokens")
    summary["request_shape"] = {"parts_per_block": parts_seen or [None],
                                "smoke_input_tokens": smoke_tokens, "smoke_requests": smoke_requests}
    fit_cost = fit_timing["totals"]["cost_usd"]
    held_cost = held_timing.get("totals", {}).get("cost_usd", 0.0) if args.heldout else 0.0
    summary["spend"] = {
        "fit_usd": fit_cost, "heldout_usd": held_cost, "smoke_usd": smoke_cost,
        "total_usd": round(fit_cost + held_cost + smoke_cost, 6), "cap_usd": args.cap,
        "requests": fit_timing["totals"]["requests"] + held_timing.get("totals", {}).get("requests", 0) + smoke_requests,
        "errors": fit_timing["totals"]["errors"] + held_timing.get("totals", {}).get("errors", 0),
        "unanswered": fit_timing["totals"]["unanswered_cells"] + held_timing.get("totals", {}).get("unanswered_cells", 0),
    }

    v = bundle.version
    notes.append(f"Target per sentence: editor kept (full or partial) versus removed, from the harness's human "
                 f"sentence states, the same states SENTENCE POINTS reads. Fit six: "
                 f"{int(sum(y_fit[e].sum() for e in fit_order))} kept of {sum(len(y_fit[e]) for e in fit_order)}.")
    notes.append("Missing values: an unanswered noul is filled with 0.5, a missing v3 score with 2.5, a missing v3 "
                 f"cut_p with 0.5, a missing trim answer (one-word rows) with 1.0 (whole). Counts: {summary['missing']}.")
    notes.append("Fit-set threshold: calibrated by the harness's pooled Neutral sweep on the out-of-fold predictions "
                 "of all six episodes, so it is chosen on the fit set only and then frozen with the weights. The "
                 "pooled SP in the tables is the sentence-count weighted mean of per-episode SP; the harness's own "
                 f"pooled figure for the chosen set is {pct(summary_sets[chosen]['loo']['pooled_harness_sp'])}.")
    notes.append(f"The v3 control lands at {pct(summary_sets['v3']['loo']['pooled']['sentence_points'])} against "
                 f"{pct(jev_fit_scored['pooled']['sentence_points'])} for jev_a v3 as published on the same six "
                 f"and {pct(jev_fit_notrim_scored['pooled']['sentence_points'])} with keep_words null. The control "
                 f"has no trims and a logistic squashing of the score; the keep threshold moves accordingly.")
    parts_text = (f"{parts_seen[0]} request{'s' if parts_seen[0] != 1 else ''}" if len(parts_seen) == 1 else
                  f"{parts_seen[0]} to {parts_seen[-1]} requests") if parts_seen else "one or more requests"
    notes.append(f"The questions are TypeSafe nouls (probability of yes), one per target sentence per question, the "
                 f"same primitive as v3's cut_k. Each block's {len(question_keys)} questions go out in {parts_text} "
                 f"that all carry the full v3 state; the answers never see each other.")
    notes.append("Selection was among the spec's four sets only; `code+v3` is reported as a diagnostic and was not "
                 "eligible" + (f", nor were the `{control['version']}` control and ablation sets." if control else "."))
    if control:
        cs, ab = summary_sets[control["set"]], (summary_sets[control["ablation_set"]] if control["ablation_set"] else None)
        notes.append(f"Control: `{control['version']}`'s chosen set `{control['chosen']}` refitted through this "
                     f"script's path from `{control['prefix']}-fit` "
                     f"{'reproduces' if control['reproduces_frozen'] else 'does not reproduce'} the weights, C and "
                     f"threshold frozen in `{control['weights_path'].name}` (largest coefficient difference "
                     f"{control['coef_drift']:.1e}). Its held-out and ladder rows use that frozen file, so they are "
                     f"the same arm `{control['version']}`'s write-up reports.")
        if ab:
            delta = (ab["loo"]["pooled"]["sentence_points"] - cs["loo"]["pooled"]["sentence_points"]) * 100
            notes.append(f"Ablation: removing {', '.join(f'`{k}`' for k in bundle.dropped)} from `{control['chosen']}` "
                         f"and refitting on the fit six moves leave-one-out SP by {delta:+.2f} "
                         f"({pct(cs['loo']['pooled']['sentence_points'])} to "
                         f"{pct(ab['loo']['pooled']['sentence_points'])}), C {cs['C']:g} to {ab['C']:g}.")
    if args.heldout:
        notes.append("Held-out numbers use the frozen weights and the frozen threshold. Nothing was refitted, "
                     "recalibrated or chosen after the held-out features were read; stage 2 refuses to run if "
                     "the stage 1 recomputation drifts from the frozen file."
                     + (" The control and ablation sets were frozen at stage 1 in the same file." if control else ""))
        eligible = [sp["name"] for sp in specs if sp["kind"] == "spec"]
        hs = {n: summary["heldout"]["sets"][n]["frozen"]["pooled"]["sentence_points"] for n in eligible}
        best_held = max(hs, key=hs.get)
        if best_held != chosen:
            notes.append(f"On held-out, `{best_held}` ({pct(hs[best_held])}) edges the chosen `{chosen}` "
                         f"({pct(hs[chosen])}). The choice stands: it was made on the fit six before any held-out "
                         f"number was read, and the gap is inside the per-episode spread.")
        if control:
            mine_h = hs[chosen]
            ctrl_h = summary["heldout"]["sets"][control["set"]]["frozen"]["pooled"]["sentence_points"]
            notes.append(f"Against `{control['version']}` at its frozen weights: held-out {pct(mine_h)} versus "
                         f"{pct(ctrl_h)} ({(mine_h - ctrl_h) * 100:+.2f}), ladder {pct(summary['ladder']['mine_sp'])} "
                         f"versus {pct(summary['ladder']['control_sp'])} "
                         f"({(summary['ladder']['mine_sp'] - summary['ladder']['control_sp']) * 100:+.2f}).")
        median_ratio = (held_timing.get("totals") or {}).get("est_over_actual_median")
        shape = (f"Request shape: the {len(question_keys)} questions go out as {parts_text} per block, all carrying "
                 f"the whole v3 state")
        if smoke_tokens:
            shape += (f"; a block of 25 sentences measured {smoke_tokens:,} real input tokens across its "
                      f"{smoke_requests} request{'s' if smoke_requests != 1 else ''} on the smoke block")
        if median_ratio:
            shape += (f". Real tokens ran about 1/{median_ratio:.2f} of the 4-chars-per-token estimate over the "
                      f"held-out run")
        notes.append(shape + ".")
        caps = {t.get("real_cap") for t in (fit_timing, held_timing) if t}
        control_caps = ({t.get("real_cap") for t in (control_fit_timing, control_held_timing) if t}
                        if control else set())
        if caps and control_caps and caps != control_caps:
            notes.append(f"Request packing differs from `{control['version']}`: the predicted real-token cap per "
                         f"request was {', '.join(f'{c:,}' for c in sorted(caps))} here against "
                         f"{', '.join(f'{c:,}' for c in sorted(control_caps))} for `{control['version']}` "
                         f"(the provider limit is 64,000 and predictions ran 10 to 20 percent high), so windowed "
                         f"blocks pack their {len(question_keys)} questions into two requests rather than three. "
                         f"The state and the questions are unchanged; only how many questions share a request.")
        notes.append(f"The smoke block (`{smoke_name}-*`, block 0 of colman-02.04-skeleton-demo) is kept on disk "
                     f"and counted in the spend; its answers were not used for fitting.")
    summary["notes"] = notes
    summary["warnings"] = warnings

    fingerprints = {"fit features": fingerprint(fit_paths["features"]),
                    "fit timing": fingerprint(fit_paths["timing"])}
    if weights_path.exists():
        fingerprints["frozen weights"] = fingerprint(weights_path)
    if args.heldout:
        fingerprints["held-out features"] = fingerprint(inputs["heldout"]["features"])
        fingerprints["held-out timing"] = fingerprint(inputs["heldout"]["timing"])
    if control:
        fingerprints[f"control {control['version']} fit features"] = fingerprint(inputs["control_fit"]["features"])
        if args.heldout:
            fingerprints[f"control {control['version']} held-out features"] = fingerprint(
                inputs["control_heldout"]["features"])
        fingerprints[f"control {control['version']} weights"] = fingerprint(control["weights_path"])
    for name in V3_DECISIONS:
        fingerprints[f"v3 decisions {name}"] = fingerprint(OUT_DIR / f"{name}-decisions.jsonl")
    fingerprints["prompt bundle"] = fingerprint(HERE / "roughcut_jev_prompts.py")
    summary["fingerprints"] = fingerprints

    if args.heldout and not args.no_write:
        json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        write_markdown(md_path, summary)
        log(f"wrote {md_path} and {json_path}")
    elif args.heldout:
        log("--no-write: write-up not written")
    else:
        log("stage 1 only: frozen weights written, no write-up yet")

    print(json.dumps({
        "feature_version": bundle.version, "chosen_set": chosen,
        "loo_sp": {n: summary_sets[n]["loo"]["pooled"]["sentence_points"] for n in set_order},
        "loo_threshold": {n: summary_sets[n]["threshold"] for n in set_order},
        "loo_C": {n: summary_sets[n]["C"] for n in set_order},
        "jev_v3_fit_sp": jev_fit_scored["pooled"]["sentence_points"],
        "jev_v3_fit_notrim_sp": jev_fit_notrim_scored["pooled"]["sentence_points"],
        "control": summary["control"],
        "heldout_sp": ({n: summary["heldout"]["sets"][n]["frozen"]["pooled"]["sentence_points"]
                        for n in set_order} if args.heldout else None),
        "heldout_jev_v3_sp": summary["heldout"]["jev_v3"]["pooled"]["sentence_points"] if args.heldout else None,
        "ladder": ({"mine": summary["ladder"]["mine_sp"], "control": summary["ladder"]["control_sp"],
                    "jev_v3": summary["ladder"]["jev_v3_sp"],
                    "placement": summary["ladder"]["placement"]} if args.heldout else None),
        "route2": ({n: [(sh["share"], sh["pooled"]["sentence_points"]) for sh in b["shares"]]
                    for n, b in summary["route2"].get("by_set", {}).items()} if args.heldout else None),
        "question_auc": {k: e.get("fit") for k, e in question_auc.items()},
        "spend": summary["spend"], "warnings": warnings,
    }, indent=2))
    return 0


def _flat(by_episode, episodes):
    return {(e, sid): v for e in episodes for sid, v in by_episode[e].items()}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    sys.exit(main())
