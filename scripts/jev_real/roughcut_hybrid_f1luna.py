"""The stack: the f1 combiner decides, gpt-5.6-luna overrides its least confident slice.

Second pass, step 2 of ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``.
The Jev side is build B's f1 combiner (``roughcut_jev_combine.py``, chosen set
``q+code+v3`` at its frozen C and keep threshold): its keep/cut on every
sentence of the 18 ladder episodes, ``keep_words`` null, ``cut_retake`` from
the v3 row. Its per-sentence ``p_keep`` is recomputed here exactly as the f1
write-up's route-2 section had it (fit six out of fold by leave-one-episode-out
at the frozen C, held-out 12 from the frozen weights), and the recomputation
must reproduce that section's offline ceiling before any call is made.

Selection: margin ``abs(5 * p_keep - threshold)`` (threshold 3.00 for the
chosen set), one global cutoff over the pooled 8,943 sentences giving the
bottom 25%, ties broken by episode order and sentence id the way the offline
number broke them. The Luna call is build A's (``roughcut_hybrid_luna.execute``):
whole transcript as Jev saw it, rules5, groups of up to 40, medium effort. On
routed sentences Luna's verdict replaces the combiner's keep/cut under the keep
rule step 1 froze (read from ``roughcut-hybrid-luna.json``) and, for comparison,
under Luna's ``decision`` field; the retake veto stays; um removal and delete
silence layer on.

Modes::

  python scripts/jev_real/roughcut_hybrid_f1luna.py --check          # p_keep export + 84.81 reproduction, no calls
  python scripts/jev_real/roughcut_hybrid_f1luna.py                  # plan and estimate, no calls
  python scripts/jev_real/roughcut_hybrid_f1luna.py --run --episodes colman-02.04-skeleton-demo --limit-groups 1 --out smoke-hybrid-f1luna
  python scripts/jev_real/roughcut_hybrid_f1luna.py --run            # the 18, writes roughcut-hybrid-f1luna-m25-*
  python scripts/jev_real/roughcut_hybrid_f1luna.py --report         # write-up from the run on disk

READ-ONLY against solar-sailer. No network calls without ``--run``.
"""

import argparse
import hashlib
import json
import statistics
import sys
import warnings as pywarnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"

sys.path.insert(0, str(HERE))

import roughcut_hybrid_luna as hl  # noqa: E402  (imports route2, the scoring module, the pipeline)
import roughcut_route2_routing as r2  # noqa: E402
import roughcut_jev_combine as combine  # noqa: E402
import roughcut_jev_report as report_mod  # noqa: E402

OUT_STEM = "roughcut-hybrid-f1luna"
ARM_PREFIX = "hybrid_f1luna"
SHARE = 0.25
F1_NAME = "roughcut-jev-f1"
F1_FIT, F1_HELDOUT = "roughcut-jev-f1-fit", "roughcut-jev-f1-heldout"
SMOKE_NAME = "smoke-hybrid-f1luna"
BUILD_A_JSON = OUT_DIR / "roughcut-hybrid-luna.json"
BUILD_A_TAG = "m046"                       # the 25% run the stack is compared with
CEILING_TOL = 5e-4                         # reproduction tolerance on the 84.81
DEFAULT_BUDGET_USD = 1.00
FIT, KEPT, THRESHOLD = r2.FIT, r2.KEPT, r2.THRESHOLD
pct, table, num, money, fingerprint = hl.pct, hl.table, hl.num, hl.money, hl.fingerprint


# ---------------------------------------------------------------------------
# the combiner side
# ---------------------------------------------------------------------------

def combiner_p_keep(log=None):
    """Per-sentence ``p_keep`` of the chosen f1 set over the 18 ladder episodes.

    Fit six: out-of-fold predictions, leave-one-episode-out at the frozen C
    (what the f1 write-up's fit-set numbers and its route-2 section used).
    Held-out 12: the frozen weights. Refitting on all six must reproduce the
    frozen coefficients, or the feature files have drifted from the weights.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    weights_path = OUT_DIR / f"{F1_NAME}-weights.json"
    doc = json.loads(weights_path.read_text(encoding="utf-8"))
    chosen = doc["chosen_set"]
    spec = doc["sets"][chosen]
    names, C, threshold = spec["features"], spec["C"], spec["threshold"]
    fit_rows, fit_order, fit_timing, fit_paths = combine.load_features(F1_FIT)
    held_rows, held_order, held_timing, held_paths = combine.load_features(F1_HELDOUT)
    if sorted(fit_order) != sorted(doc["fit_episodes"]):
        raise SystemExit(f"fit feature run has {fit_order}, the weights were fitted on {doc['fit_episodes']}")
    fit_order = list(doc["fit_episodes"])

    human = combine.human_keep(fit_order)
    y = {e: np.array([human[e][r["id"]] for r in fit_rows[e]]) for e in fit_order}
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        loo = combine.loo_predictions(fit_rows, y, names, C, fit_order)
        refit = combine.Combiner(names, C).fit(
            np.vstack([combine.matrix(fit_rows[e], names) for e in fit_order]),
            np.concatenate([y[e] for e in fit_order]))
    drift = float(np.max(np.abs(np.array(refit.to_json()["coef"]) - np.array(spec["coef"]))))
    if drift > 1e-6:
        raise SystemExit(f"refit on the fit six drifts {drift:.2e} from the frozen coefficients")
    frozen = combine.Combiner.from_json(spec)
    held_p = {e: {r["id"]: float(v) for r, v in
                  zip(held_rows[e], frozen.predict(combine.matrix(held_rows[e], names)))}
              for e in held_order}
    ladder_eps = fit_order + [e for e in held_order if e != "greco-2.2-thumbnailing"]
    p = {e: (loo[e] if e in loo else held_p[e]) for e in ladder_eps}
    rows = {e: (fit_rows[e] if e in fit_rows else held_rows[e]) for e in ladder_eps}
    log(f"p_keep: {chosen} C {C:g} threshold {threshold:.2f}, {sum(len(v) for v in p.values())} sentences, "
        f"refit drift {drift:.1e}")
    return {"chosen": chosen, "names": names, "C": C, "threshold": threshold,
            "p": p, "rows": rows, "episodes": ladder_eps, "fit_order": fit_order,
            "held_order": [e for e in held_order if e in ladder_eps],
            "weights_path": weights_path, "weights_doc": doc,
            "feature_paths": {"fit": fit_paths, "heldout": held_paths},
            "timing": {**{e: fit_timing["episodes"][e] for e in fit_order},
                       **{e: held_timing["episodes"][e] for e in held_order}},
            "coef_drift": drift}


def stack_inputs(v3_inputs, cp):
    """Build A's ``inputs`` shape with the combiner in Jev's seat.

    ``jev`` carries the combiner's own keep/cut as 5 or 0 at its frozen
    threshold (what the scorer's 2.50 cut then reads), ``keep_words`` null,
    ``cut_retake`` from the v3 row; ``jev_rows`` carries the raw
    ``5 * p_keep`` and ``p_keep`` for the margin.
    """
    episodes = cp["episodes"]
    if sorted(episodes) != sorted(v3_inputs["episodes"]):
        raise SystemExit("the f1 ladder episodes are not the v3 run's 18")
    scale, threshold = combine.SCALE, cp["threshold"]
    jev, jev_rows = {}, {}
    for e in episodes:
        v3 = v3_inputs["jev"][e]
        jev[e], jev_rows[e] = {}, {}
        for r in cp["rows"][e]:
            sid, p = r["id"], cp["p"][e][r["id"]]
            cut_retake = bool(r["v3"].get("cut_retake"))
            if sid not in v3 or cut_retake != bool(v3[sid]["cut_retake"]):
                raise SystemExit(f"{e} {sid}: cut_retake in the feature row differs from the v3 decision set")
            jev[e][sid] = {"score": scale if scale * p >= threshold else 0.0,
                           "keep_words": None, "cut_retake": cut_retake}
            jev_rows[e][sid] = {"score": scale * p, "p_keep": p, "cut_retake": cut_retake,
                                "v3_score": v3[sid]["score"]}
        if set(jev[e]) != set(v3):
            raise SystemExit(f"{e}: feature rows cover {len(jev[e])} sentences, v3 has {len(v3)}")
    return {**v3_inputs, "episodes": episodes, "jev": jev, "jev_rows": jev_rows,
            "v3_jev": v3_inputs["jev"], "combiner_threshold": threshold}


def margin_of(threshold):
    return lambda row: abs(float(row["score"]) - threshold)


def select_slice(inputs, share):
    """``(routed, cutoff)``: bottom ``share`` by combiner margin, pooled, one cutoff.

    Ties break by episode order then sentence id, exactly as
    ``roughcut_jev_combine.route2_handoff`` ranked them for the offline 84.81.
    """
    episodes, rows = inputs["episodes"], inputs["jev_rows"]
    margin = margin_of(inputs["combiner_threshold"])
    pool = sorted((margin(rows[e][sid]), i, sid, e) for i, e in enumerate(episodes) for sid in rows[e])
    n = int(share * len(pool) + 0.5)
    routed = {e: set() for e in episodes}
    for _m, _i, sid, e in pool[:n]:
        routed[e].add(sid)
    return routed, (pool[n - 1][0] if n else None)


def offline_ceiling(inputs, cp, share):
    """The f1 write-up's route-2 number on this slice, through its own code path."""
    episodes = inputs["episodes"]
    decisions = combine.arm_decisions({e: cp["rows"][e] for e in episodes}, cp["p"])
    return combine.route2_handoff(decisions, cp["p"], cp["threshold"], inputs["removals"],
                                  episodes, [share])


def published_route2(share):
    doc = json.loads((OUT_DIR / f"{F1_NAME}.json").read_text(encoding="utf-8"))
    block = doc["route2"]["by_set"][doc["chosen_set"]]
    row = next(s for s in block["shares"] if abs(s["share"] - share) < 1e-9)
    return {"sentence_points": row["pooled"]["sentence_points"], "routed": row["routed"],
            "cutoff_margin": row["cutoff_margin"], "ladder_f1_sp": doc["ladder"]["f1_sp"],
            "f1_seconds": doc["seconds"]["per_episode"], "f1_spend": doc["spend"]}


def frozen_keep_rule():
    """Step 1's frozen rule, read from build A's write-up JSON; never chosen here."""
    if not BUILD_A_JSON.exists():
        raise SystemExit(f"{BUILD_A_JSON} missing; run roughcut_hybrid_luna.py --report first (step 1)")
    doc = json.loads(BUILD_A_JSON.read_text(encoding="utf-8"))
    frozen = doc.get("frozen_keep_rule")
    if not frozen:
        raise SystemExit("build A's JSON has no frozen_keep_rule; regenerate it with the keep-rule section")
    return frozen


def check(log=None):
    """Recompute ``p_keep``, select the slice, reproduce the offline ceiling. $0."""
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    cp = combiner_p_keep(log)
    inputs = stack_inputs(hl.load_inputs(), cp)
    routed, cutoff = select_slice(inputs, SHARE)
    mine = offline_ceiling(inputs, cp, SHARE)
    if mine.get("error"):
        raise SystemExit(f"offline ceiling failed: {mine['error']}")
    mine_row = mine["shares"][0]
    published = published_route2(SHARE)
    gap = mine_row["pooled"]["sentence_points"] - published["sentence_points"]
    result = {
        "chosen_set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"],
        "coef_drift": cp["coef_drift"], "share": SHARE,
        "routed": sum(len(v) for v in routed.values()), "cutoff_margin": cutoff,
        "offline_ceiling_here": mine_row["pooled"]["sentence_points"],
        "offline_ceiling_published": published["sentence_points"],
        "published_routed": published["routed"], "published_cutoff": published["cutoff_margin"],
        "reproduced": abs(gap) <= CEILING_TOL and mine_row["routed"] == published["routed"]
        and abs((cutoff or 0) - published["cutoff_margin"]) < 1e-9,
    }
    if not result["reproduced"]:
        raise SystemExit("offline ceiling does not reproduce the f1 write-up: " + json.dumps(result, indent=2))
    return result, inputs, cp, routed, cutoff


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(args, parser):
    result, inputs, cp, routed, cutoff = check()
    print(json.dumps({"check": result}, indent=2), flush=True)
    rules = hl.read_rules()
    name = args.out or f"{OUT_STEM}-m{round(SHARE * 100):02d}" + ("" if args.effort == hl.DEFAULT_EFFORT else f"-{args.effort}")
    arm = f"{ARM_PREFIX}_{name.split(OUT_STEM + '-', 1)[-1].replace('-', '_')}"
    frozen = frozen_keep_rule()
    plan_extra = {"selection": "combiner margin abs(5 * p_keep - threshold), pooled, one cutoff",
                  "combiner": {"set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"],
                               "weights": str(cp["weights_path"])},
                  "offline_ceiling": result["offline_ceiling_here"],
                  "frozen_keep_rule": frozen["rule"]}
    return hl.execute(args, parser, inputs, rules, routed, cutoff, name, arm,
                      confidence=margin_of(cp["threshold"]), plan_extra=plan_extra)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report():
    log = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    result, inputs, cp, routed, cutoff = check(log)
    episodes, removals = inputs["episodes"], inputs["removals"]
    f1_jev, v3_jev = inputs["jev"], inputs["v3_jev"]
    frozen = frozen_keep_rule()
    rule = frozen["rule"]
    tag = f"m{round(SHARE * 100):02d}"
    run = hl.load_hybrid_run(tag, episodes, stem=OUT_STEM)
    if run is None:
        raise SystemExit(f"no finished run {OUT_STEM}-{tag} under docs/jev-real")
    for e in episodes:
        if run["routed"][e] != routed[e]:
            raise SystemExit(f"{e}: the run's routed slice is not the slice selected now")
    build_a = hl.load_hybrid_run(BUILD_A_TAG, episodes)
    if build_a is None:
        raise SystemExit(f"build A run {BUILD_A_TAG} missing")
    published = published_route2(SHARE)

    log("donors...")
    donors, donor_paths = {}, {}
    for key in ("luna", "opus"):
        loaded, paths = r2.load_donor(key, episodes)
        absent = [e for e in episodes if e not in loaded]
        if absent:
            raise SystemExit(f"archived {key} decisions missing for {absent}")
        donors[key], donor_paths[key] = loaded, paths
    hl.register_live_donors(donors, run, episodes)
    hl.register_live_donors(donors, build_a, episodes)
    scorer = r2.Scorer(f1_jev, donors, removals)          # combiner in Jev's seat
    scorer_v3 = r2.Scorer(v3_jev, donors, removals)       # build A's seat, for the comparison rows
    everything = {e: set(f1_jev[e]) for e in episodes}

    log("baselines...")
    f1_eps = {e: scorer.episode(None, e, set()) for e in episodes}
    f1_pooled = r2.splits(f1_eps, episodes)
    v3_eps = {e: scorer_v3.episode(None, e, set()) for e in episodes}
    v3_pooled = r2.splits(v3_eps, episodes)
    build_a_key = hl.donor_key(BUILD_A_TAG, "decision")
    build_a_eps = {e: scorer_v3.episode(build_a_key, e, build_a["routed"][e]) for e in episodes}
    build_a_pooled = r2.splits(build_a_eps, episodes)
    build_a_frozen_pooled = r2.splits(
        {e: scorer_v3.episode(hl.donor_key(BUILD_A_TAG, rule), e, build_a["routed"][e]) for e in episodes},
        episodes)
    luna_pure = r2.splits({e: scorer_v3.episode("luna", e, everything[e]) for e in episodes}, episodes)
    reproduction = {
        "jev_a": {"mine": v3_pooled["all"]["sentence_points"], "published": 0.8046516828804657},
        "f1_combiner": {"mine": f1_pooled["all"]["sentence_points"], "published": published["ladder_f1_sp"]},
        "offline_ceiling_25": {"mine": result["offline_ceiling_here"], "published": published["sentence_points"]},
        "build_a_m046_decision": {"mine": build_a_pooled["all"]["sentence_points"], "published": 0.8363},
        "luna_pure": {"mine": luna_pure["all"]["sentence_points"], "published": 0.8371},
    }

    log("the stack...")
    by_rule = {}
    for kr in hl.KEEP_RULES:
        key = hl.donor_key(tag, kr)
        per = {e: scorer.episode(key, e, routed[e]) for e in episodes}
        by_rule[kr] = {"per_episode": per, "pooled": r2.splits(per, episodes)}
    ceiling_ep = {e: scorer.episode("luna", e, routed[e]) for e in episodes}
    ceiling = r2.splits(ceiling_ep, episodes)
    opus_ceiling = r2.splits({e: scorer.episode("opus", e, routed[e]) for e in episodes}, episodes)
    if abs(ceiling["all"]["sentence_points"] - result["offline_ceiling_here"]) > 1e-9:
        raise SystemExit("Scorer ceiling and route2_handoff ceiling disagree on the same slice")

    ladder = r2.ladder_rows(episodes, v3_pooled["all"]["sentence_points"])
    ladder.append({"label": f"Jev f1 `{cp['chosen']}` combiner (build B)", "key": "f1",
                   "sentence_points": f1_pooled["all"]["sentence_points"]})
    ladder.append({"label": "build A m046 (v3 margin 25%, live Luna decision)", "key": "build_a_m046",
                   "sentence_points": build_a_pooled["all"]["sentence_points"]})
    ladder.sort(key=lambda r: -r["sentence_points"])
    placement = {kr: r2.placement(by_rule[kr]["pooled"]["all"]["sentence_points"], ladder)
                 for kr in ("decision", rule)}
    placement["ceiling"] = r2.placement(ceiling["all"]["sentence_points"], ladder)

    log("states, flips, agreement...")
    human, f1_states, v3_states = {}, {}, {}
    for e in episodes:
        human[e], f1_states[e] = r2.states(e, f1_jev[e], removals[e])
        v3_states[e] = r2.states(e, v3_jev[e], removals[e])[1]
    donor_states = {}
    for kr in ("decision", rule):
        key = hl.donor_key(tag, kr)
        donor_states[kr] = {e: r2.states(e, scorer.mixed(key, e, routed[e]), removals[e])[1] for e in episodes}
    donor_states["luna"] = {e: r2.states(e, donors["luna"][e]["decisions"], removals[e])[1] for e in episodes}
    flips = {kr: r2.flip_block(scorer, hl.donor_key(tag, kr), routed, episodes, human, f1_states)
             for kr in ("decision", rule)}
    flips["ceiling"] = r2.flip_block(scorer, "luna", routed, episodes, human, f1_states)
    slices = {kr: r2.slice_agreement(routed, episodes, human, f1_states, donor_states[kr])
              for kr in ("decision", rule, "luna")}
    slices["v3_on_slice"] = r2.slice_agreement(routed, episodes, human, f1_states, v3_states)
    agreement = hl.live_vs_archived(run, donors["luna"], episodes, human, removals)

    # overlap with build A's slice (the v3 margin bottom 25%)
    overlap = sum(len(routed[e] & build_a["routed"][e]) for e in episodes)
    n_routed = sum(len(v) for v in routed.values())
    both = {e: routed[e] & build_a["routed"][e] for e in episodes}
    same = n_agree = 0
    for e in episodes:
        for sid in both[e]:
            a, b = run["rows"][e][sid], build_a["rows"][e][sid]
            if a["luna_answered"] and b["luna_answered"]:
                same += 1
                n_agree += a["luna_decision"] == b["luna_decision"]
    slice_overlap = {"routed": n_routed, "build_a_routed": sum(len(v) for v in build_a["routed"].values()),
                     "shared": overlap, "shared_share": overlap / n_routed if n_routed else None,
                     "shared_both_answered": same,
                     "luna_same_decision_on_shared": n_agree / same if same else None}

    # per episode
    log("per episode...")
    t_eps = run["timing"]["episodes"]
    f1_secs = published["f1_seconds"]
    per_episode = {}
    for e in episodes:
        t = t_eps[e]
        f1_s = f1_secs.get(e, {}).get("total_s")
        f1_usd = f1_secs.get(e, {}).get("total_usd")
        luna_s = t["wall_clock_s"]
        per_episode[e] = {
            "sentences": len(f1_jev[e]), "routed": len(routed[e]), "asked": t["asked"],
            "retake_vetoed": t["retake_vetoed"], "substituted": t["substituted"],
            "unanswered": t["unanswered_targets"], "requests": t["requests"],
            "retries": t["retries"], "errors": t["errors"], "malformed": t["malformed"],
            "v3": v3_eps[e], "f1": f1_eps[e], "build_a": build_a_eps[e],
            "decision": by_rule["decision"]["per_episode"][e], "frozen": by_rule[rule]["per_episode"][e],
            "ceiling": ceiling_ep[e],
            "luna_seconds": luna_s, "f1_seconds": f1_s,
            "seconds": (luna_s + f1_s) if f1_s is not None else None,
            "cost_usd": t["cost_usd"], "cost_listed_usd": t["cost_listed_usd"], "f1_cost_usd": f1_usd,
            "stack_cost_usd": (t["cost_usd"] + f1_usd) if f1_usd is not None else None,
            "prompt_tokens": t["prompt_tokens"], "cached_tokens": t["cached_tokens"],
            "completion_tokens": t["completion_tokens"], "reasoning_tokens": t["reasoning_tokens"],
            "input_tokens_per_request": t["prompt_tokens"] / t["requests"] if t["requests"] else None,
        }
    totals = run["timing"]["totals"]
    secs = [per_episode[e]["seconds"] for e in episodes if per_episode[e]["seconds"] is not None]
    stack_costs = [per_episode[e]["stack_cost_usd"] for e in episodes if per_episode[e]["stack_cost_usd"] is not None]

    smoke_timing = hl.run_paths(SMOKE_NAME)["timing"]
    smoke_spend = (json.loads(smoke_timing.read_text(encoding="utf-8"))["totals"]["cost_usd"]
                   if smoke_timing.exists() else None)

    inputs_fp = {k: fingerprint(v) for k, v in inputs["run_paths"].items()}
    inputs_fp["rules_prompt"] = fingerprint(hl.RULES_PATH)
    inputs_fp["ladder_reference"] = fingerprint(report_mod.REFERENCE_JSON)
    inputs_fp["f1_weights"] = fingerprint(cp["weights_path"])
    for split, paths in cp["feature_paths"].items():
        inputs_fp[f"f1_{split}_features"] = fingerprint(paths["features"])
        inputs_fp[f"f1_{split}_timing"] = fingerprint(paths["timing"])
    inputs_fp["f1_writeup_json"] = fingerprint(OUT_DIR / f"{F1_NAME}.json")
    inputs_fp["build_a_json"] = fingerprint(BUILD_A_JSON)
    for key, paths in donor_paths.items():
        for path in paths:
            inputs_fp[f"{key}:{Path(path).name}"] = fingerprint(path)
    for kind, path in run["paths"].items():
        inputs_fp[f"{tag}:{kind}"] = fingerprint(path)
    for kind, path in build_a["paths"].items():
        inputs_fp[f"build_a_{BUILD_A_TAG}:{kind}"] = fingerprint(path)
    digest = hashlib.md5()
    for e in episodes:
        digest.update(r2.md5(r2.removals_cache_path(e)).encode())

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_hybrid_f1luna.py",
        "model": hl.MODEL, "effort": run["timing"]["effort"], "arm": run["timing"]["arm"], "tag": tag,
        "episodes": episodes, "fit": FIT, "heldout": [e for e in episodes if e not in FIT],
        "sentences": sum(len(f1_jev[e]) for e in episodes),
        "combiner": {"set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"],
                     "features": len(cp["names"]), "coef_drift": cp["coef_drift"]},
        "share": SHARE, "cutoff": cutoff, "routed": n_routed,
        "asked": totals["asked"], "retake_vetoed": totals["retake_vetoed"],
        "substituted": totals["substituted"], "unanswered": totals["unanswered_targets"],
        "requests": totals["requests"], "retries": totals["retries"], "errors": totals["errors"],
        "malformed": totals["malformed"],
        "frozen_keep_rule": frozen, "keep_rules": hl.KEEP_RULES,
        "reproduction": reproduction,
        "f1": f1_pooled, "v3": v3_pooled, "build_a": build_a_pooled, "build_a_frozen": build_a_frozen_pooled,
        "luna_pure": luna_pure,
        "by_rule": {kr: v["pooled"] for kr, v in by_rule.items()},
        "ceiling": ceiling, "opus_ceiling": opus_ceiling["all"],
        "offline_ceiling": result,
        "ladder": ladder, "placement": placement,
        "flips": flips, "slices": slices, "agreement": agreement, "slice_overlap": slice_overlap,
        "per_episode": per_episode,
        "cost_usd": totals["cost_usd"], "cost_listed_usd": totals["cost_listed_usd"],
        "cost_per_episode_mean": totals["cost_usd"] / len(episodes),
        "cost_per_episode_max": max(per_episode[e]["cost_usd"] for e in episodes),
        "stack_cost_per_episode_mean": statistics.mean(stack_costs) if stack_costs else None,
        "f1_cost_per_episode_mean": statistics.mean(
            [per_episode[e]["f1_cost_usd"] for e in episodes if per_episode[e]["f1_cost_usd"] is not None]),
        "seconds_per_episode_mean": statistics.mean(secs) if secs else None,
        "seconds_per_episode_max": max(secs) if secs else None,
        "luna_seconds_per_episode_mean": statistics.mean(per_episode[e]["luna_seconds"] for e in episodes),
        "f1_seconds_per_episode_mean": statistics.mean(
            [per_episode[e]["f1_seconds"] for e in episodes if per_episode[e]["f1_seconds"] is not None]),
        "tokens": {k: totals[k] for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")},
        "estimate": run["timing"].get("estimate"), "wall_clock_s": totals["wall_clock_s"],
        "run_generated_utc": run["timing"]["generated_utc"],
        "smoke_spend_usd": smoke_spend,
        "total_spend_usd": round(totals["cost_usd"] + (smoke_spend or 0.0), 6),
        "settings": {"group_max": hl.GROUP_MAX, "group_span": hl.GROUP_SPAN, "concurrency": hl.CONCURRENCY,
                     "max_attempts": hl.MAX_ATTEMPTS, "malformed_reasks": hl.MALFORMED_REASKS,
                     "timeout_s": hl.TIMEOUT_S, "preamble_version": hl.PREAMBLE_VERSION,
                     "threshold": THRESHOLD, "t_trim": r2.T_TRIM},
        "inputs": inputs_fp, "removals_digest": digest.hexdigest(),
    }


def write_markdown(path, s, json_path):
    lines = []

    def add(text=""):
        lines.append(text)

    rule = s["frozen_keep_rule"]["rule"]
    dec, frz, ceil = s["by_rule"]["decision"], s["by_rule"][rule], s["ceiling"]
    smoke = (f", plus ${s['smoke_spend_usd']:.4f} for the one-group smoke request in `{SMOKE_NAME}-*`"
             if s.get("smoke_spend_usd") is not None else "")
    add("# The f1-Luna stack: combiner decides, Luna overrides its unsure slice (developer-facing notes)")
    add()
    add(f"Generated {s['generated_utc']} by `{s['script']}` from the run files on disk, the f1 feature files and frozen weights, the stored v3 decisions, the archived donor ratings and the cached removal ranges. The report step makes no model calls; the run it reads cost ${s['cost_usd']:.4f} by the router's accounting (${s['cost_listed_usd']:.4f} at list rates 0.20 in, 0.02 cached, 1.20 out per million){smoke}. Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.")
    add()
    add(f"Question: second pass, step 2 of the round-two design. Build B's f1 combiner (`{s['combiner']['set']}`, C {s['combiner']['C']:g}, keep threshold {s['combiner']['threshold']:.2f} on `5 * p_keep`) decides every one of the {s['sentences']:,} sentences of the 18 ladder episodes, `keep_words` null, `cut_retake` from the v3 row. The sentences whose `5 * p_keep` sits closest to the threshold (margin `abs(5 * p_keep - {s['combiner']['threshold']:.2f})` under one global cutoff, {s['cutoff']:.3f}, the bottom {s['share'] * 100:g}%, {s['routed']:,} sentences) go to `{s['model']}` exactly as build A sent Jev's v3 slice: whole episode transcript as Jev saw it, rules5 system prompt, groups of up to {s['settings']['group_max']}, {s['effort']} effort. On routed sentences Luna's verdict replaces the combiner's keep/cut, `keep_words` stays null, the retake veto stays. Two substitutions are reported: Luna's `decision` field, and the keep rule step 1 froze on build A's fit six (`{rule}`, `roughcut-hybrid-luna.md`, chosen before this run was made). The `p_keep` values are the ones the f1 write-up's route-2 section used: fit six out of fold by leave-one-episode-out at the frozen C, held-out 12 from the frozen weights, so the selection on the fit six is not held out in the strict sense (the C and threshold were chosen there), and neither is the cutoff (chosen from the pooled 18); Luna's decisions on the slice are.")
    add()

    add("## Reproduction check")
    add()
    r = s["reproduction"]
    rows = [["pure Jev (jev_a v3)", pct(r["jev_a"]["mine"]), pct(r["jev_a"]["published"])],
            [f"f1 `{s['combiner']['set']}` combiner alone (build B, ladder 18)", pct(r["f1_combiner"]["mine"]), pct(r["f1_combiner"]["published"])],
            ["offline ceiling: combiner margin bottom 25%, archived Luna (f1 write-up)", pct(r["offline_ceiling_25"]["mine"]), pct(r["offline_ceiling_25"]["published"])],
            ["build A m046: v3 margin bottom 25%, live Luna decision", pct(r["build_a_m046_decision"]["mine"]), pct(r["build_a_m046_decision"]["published"])],
            ["pure archived Luna chapters (routed 100%)", pct(r["luna_pure"]["mine"]), pct(r["luna_pure"]["published"])]]
    lines.extend(table(["arm", "SENTENCE POINTS here", "published"], rows))
    add()
    add(f"The combiner's `p_keep` was recomputed from the feature files and the frozen weights (`{Path(str(s['inputs']['f1_weights']['path'])).name}`); refitting on the fit six lands {s['combiner']['coef_drift']:.1e} from the frozen coefficients, the slice has the same {s['routed']:,} sentences and the same cutoff as the f1 write-up's route-2 row, and the archived substitution reproduces its number through the combiner's own `route2_handoff` before any call was made.")
    add()

    add("## Pooled results")
    add()
    add(f"Pooled over the fit six, the held-out 12 and all 18, with modules. `decision` substitutes Luna's decision field on the routed slice; `{rule}` substitutes the frozen keep rule; `ceiling` substitutes the archived Luna chapters decision (the offline number). Build A is the same live call on Jev's v3 slice. The fit-six column is where the combiner's C, threshold and (for build A) the keep rule were chosen; the held-out 12 column is not. Seconds per episode are Luna's wall clock at concurrency {s['settings']['concurrency']} plus the combiner's own per-episode time from the f1 run (v3 sentence pass plus the feature questions); dollars are the router's accounting, Luna only, with the stack total (Luna plus f1) alongside.")
    add()
    rows = []
    luna_cost, f1_cost = s["cost_per_episode_mean"], s["f1_cost_per_episode_mean"]
    for label, p, sp_secs, cost, total in (
            (f"f1 combiner alone (build B)", s["f1"], s["f1_seconds_per_episode_mean"], None, f1_cost),
            (f"stack, Luna `decision`", dec, s["seconds_per_episode_mean"], luna_cost, s["stack_cost_per_episode_mean"]),
            (f"stack, frozen rule `{rule}`", frz, s["seconds_per_episode_mean"], luna_cost, s["stack_cost_per_episode_mean"]),
            (f"stack ceiling (archived Luna on the same slice)", ceil, None, None, None),
            (f"build A m046, Luna `decision`", s["build_a"], None, None, None),
            (f"build A m046, frozen rule `{rule}`", s["build_a_frozen"], None, None, None)):
        rows.append([label, pct(p["fit"]["sentence_points"]), pct(p["heldout"]["sentence_points"]),
                     pct(p["all"]["sentence_points"]), pct(p["all"]["word_score"]), pct(p["all"]["grade"]),
                     num(sp_secs, 1), money(cost), money(total)])
    lines.extend(table(["arm", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18", "GRADE all 18",
                        "s/ep mean", "Luna $/ep mean", "arm total $/ep mean"], rows))
    add()
    ladder_text = ", ".join(f"{r['label']} {pct(r['sentence_points'])}" for r in s["ladder"])
    add(f"Ladder, with modules, same 18 episodes (build A and the f1 combiner added as rows): {ladder_text}. Placement: stack with `decision` {s['placement']['decision']['text']} (rank {s['placement']['decision']['rank']} of {s['placement']['decision']['of']}); stack with `{rule}` {s['placement'][rule]['text']} (rank {s['placement'][rule]['rank']} of {s['placement'][rule]['of']}); the ceiling on this slice {s['placement']['ceiling']['text']}.")
    add()
    gain_dec = dec["all"]["sentence_points"] - s["f1"]["all"]["sentence_points"]
    gain_frz = frz["all"]["sentence_points"] - s["f1"]["all"]["sentence_points"]
    gain_ceil = ceil["all"]["sentence_points"] - s["f1"]["all"]["sentence_points"]
    add(f"Against the offline ceiling {pct(ceil['all']['sentence_points'])} and build A's {pct(s['build_a']['all']['sentence_points'])}: the stack lands at {pct(dec['all']['sentence_points'])} with the decision field ({gain_dec * 100:+.2f} on the combiner alone, {(dec['all']['sentence_points'] - s['build_a']['all']['sentence_points']) * 100:+.2f} on build A) and {pct(frz['all']['sentence_points'])} with `{rule}` ({gain_frz * 100:+.2f} on the combiner alone, {(frz['all']['sentence_points'] - s['build_a_frozen']['all']['sentence_points']) * 100:+.2f} on build A under the same rule). Of the {gain_ceil * 100:.2f} SP the archived substitution adds over the combiner, the live call keeps {(gain_dec / gain_ceil * 100) if gain_ceil else float('nan'):.0f}% with the decision field and {(gain_frz / gain_ceil * 100) if gain_ceil else float('nan'):.0f}% with `{rule}`. Held out: {pct(dec['heldout']['sentence_points'])} and {pct(frz['heldout']['sentence_points'])} against build A's {pct(s['build_a']['heldout']['sentence_points'])} and {pct(s['build_a_frozen']['heldout']['sentence_points'])}, the combiner alone {pct(s['f1']['heldout']['sentence_points'])}, pure Jev v3 {pct(s['v3']['heldout']['sentence_points'])}.")
    add()

    add("## Every keep rule on this slice")
    add()
    add(f"The same stored answers under each keep rule, for the shape of the curve. `{rule}` is the frozen one; nothing here was chosen on these numbers.")
    add()
    rows = []
    for kr in s["keep_rules"]:
        p = s["by_rule"][kr]
        rows.append([("* " if kr == rule else "") + kr, pct(p["fit"]["sentence_points"]),
                     pct(p["heldout"]["sentence_points"]), pct(p["all"]["sentence_points"]),
                     pct(p["all"]["word_score"]), pct(p["all"]["grade"]),
                     f"{(p['all']['sentence_points'] - dec['all']['sentence_points']) * 100:+.2f}",
                     num(s["seconds_per_episode_mean"], 1)])
    lines.extend(table(["keep rule", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18", "GRADE all 18",
                        "all 18 minus decision", "s/ep mean"], rows))
    add()

    add("## The slice")
    add()
    o, sl = s["slice_overlap"], s["slices"]
    add(f"The combiner's bottom 25% shares {o['shared']:,} sentences with build A's v3-margin bottom 25% ({o['shared_share'] * 100:.0f}% of {o['routed']:,}); on the {o['shared_both_answered']:,} shared sentences both runs got an answer for, Luna gave the same decision {o['luna_same_decision_on_shared'] * 100:.0f}% of the time (two separate medium-effort calls on the same transcript). Agreement with the editor on the routed slice (kept means full or partial), read off each arm's own states with modules: combiner {pct(sl['decision']['jev'])}, Jev v3 on the same sentences {pct(sl['v3_on_slice']['donor'])}, live Luna decision {pct(sl['decision']['donor'])}, live Luna `{rule}` {pct(sl[rule]['donor'])}, archived Luna {pct(sl['luna']['donor'])}. Seconds per episode as above, {num(s['seconds_per_episode_mean'], 1)}.")
    add()

    add("## Live Luna against archived Luna on the same sentences")
    add()
    a = s["agreement"]
    rows = [[f"{s['tag']} ({s['share'] * 100:g}% routed, {s['effort']} effort)", a["n"], pct(a["agree_rate"]), pct(a["live_right_rate"]),
             pct(a["archived_right_rate"]), pct(sl["decision"]["jev"]),
             f"{a['disagree_live_right']} / {a['disagree_archived_right']} of {a['disagree']}",
             f"{a['live_keep_archived_cut']} / {a['live_cut_archived_keep']}",
             pct(a["live_keep_rate"]), pct(a["archived_keep_rate"]), pct(a["editor_keep_rate"]),
             pct(dec["all"]["sentence_points"]), pct(ceil["all"]["sentence_points"]),
             num(s["seconds_per_episode_mean"], 1)]]
    lines.extend(table(["run", "answered", "live agrees with archived", "live right", "archived right",
                        "combiner right", "disagreements live right / archived right",
                        "live keep & archived cut / live cut & archived keep",
                        "live keep rate", "archived keep rate", "editor keep rate",
                        "SP live decision", "SP archived (ceiling)", "s/ep mean"], rows))
    add()
    add(f"Same pattern as build A: the live call is cut-heavier than the archive on the unsure slice ({a['live_cut_archived_keep']} live cut where the archive keeps against {a['live_keep_archived_cut']} the other way), the editor keeps more of the slice than either, so the frozen `{rule}` rule, which keeps anything Luna scores 2 or more, recovers part of the gap. Live scores land within one point of the archived score on {pct(a['score_within_1_rate'])}% of the answered sentences.")
    add()

    add("## Where the gains come from")
    add()
    add("A flip is a routed sentence whose keep/cut changed when Luna's verdict replaced the combiner's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor. Trim changed is always zero here because neither side carries trims on the slice.")
    add()
    rows = []
    for label, c in ((f"stack, `decision`", s["flips"]["decision"]), (f"stack, `{rule}`", s["flips"][rule]),
                     ("ceiling (archived Luna)", s["flips"]["ceiling"])):
        routed = c.get("routed", 0)
        rows.append([label, routed, c.get("flips", 0), c.get("flips_right", 0), c.get("flips_wrong", 0),
                     f"{c.get('cut_to_kept_right', 0)} / {c.get('cut_to_kept_wrong', 0)}",
                     f"{c.get('kept_to_cut_right', 0)} / {c.get('kept_to_cut_wrong', 0)}",
                     pct(c.get("jev_agreed", 0) / routed if routed else None),
                     pct(c.get("hybrid_agreed", 0) / routed if routed else None),
                     num(s["seconds_per_episode_mean"], 1)])
    lines.extend(table(["substitution", "routed", "flips", "right", "wrong", "cut to kept right / wrong",
                        "kept to cut right / wrong", "agreement on slice, combiner",
                        "agreement on slice, after routing", "s/ep mean"], rows))
    add()

    add(f"## Per episode, {s['tag']} ({s['share'] * 100:g}% routed, cutoff {s['cutoff']:.3f}, {s['effort']} effort)")
    add()
    rows = []
    for e in s["episodes"]:
        p = s["per_episode"][e]
        rows.append([e + (" (fit)" if e in s["fit"] else ""), p["sentences"], p["routed"], p["asked"], p["unanswered"],
                     pct(p["v3"]["sentence_points"]), pct(p["f1"]["sentence_points"]),
                     pct(p["build_a"]["sentence_points"]), pct(p["decision"]["sentence_points"]),
                     pct(p["frozen"]["sentence_points"]), pct(p["ceiling"]["sentence_points"]),
                     pct(p["frozen"]["word_score"]), pct(p["frozen"]["grade"]),
                     p["requests"], p["retries"], num(p["luna_seconds"], 1), num(p["f1_seconds"], 1),
                     num(p["seconds"], 1), money(p["cost_usd"]), money(p["stack_cost_usd"]),
                     f"{p['input_tokens_per_request'] / 1000:.1f}k" if p["input_tokens_per_request"] else "n/a",
                     f"{p['cached_tokens'] / max(1, p['prompt_tokens']) * 100:.0f}%"])
    lines.extend(table(["episode", "sentences", "routed", "asked", "unanswered", "SP Jev v3", "SP f1", "SP build A",
                        "SP stack decision", f"SP stack {rule}", "SP ceiling", f"WORD stack {rule}",
                        f"GRADE stack {rule}", "requests", "retries", "Luna s", "f1 s", "s/ep", "Luna $",
                        "stack $", "input tokens/request", "cached share"], rows))
    add()
    add(f"Routed sentences the v3 retake pass had already cut were not sent: {s['retake_vetoed']} of {s['routed']:,}. {s['requests']} requests, {s['retries']} retries, {s['errors']} errored attempts, {s['malformed']} malformed or incomplete answers, {s['unanswered']} targets left unanswered after retries (those keep the combiner's own decision). Tokens: {s['tokens']['prompt_tokens']:,} prompt of which {s['tokens']['cached_tokens']:,} cached, {s['tokens']['completion_tokens']:,} completion of which {s['tokens']['reasoning_tokens']:,} reasoning. Router cost ${s['cost_usd']:.4f} (${s['cost_per_episode_mean']:.4f} per episode, max ${s['cost_per_episode_max']:.4f}); the plan estimated ${s['estimate']['cost_usd']:.4f}. Stack cost per episode, Luna plus the f1 combiner's own calls: ${s['stack_cost_per_episode_mean']:.4f}. Seconds per episode {s['seconds_per_episode_mean']:.1f} mean, {s['seconds_per_episode_max']:.1f} max (Luna {s['luna_seconds_per_episode_mean']:.1f}, f1 {s['f1_seconds_per_episode_mean']:.1f}). Run wall clock {s['wall_clock_s']:.0f} s, generated {s['run_generated_utc']}.")
    add()

    add("## How the call was made")
    add()
    st = s["settings"]
    add(f"Identical to build A (`roughcut_hybrid_luna.py`, whose `execute` this script calls): system message the rules5 prompt read from `{s['inputs']['rules_prompt']['path']}` at run time (md5 {s['inputs']['rules_prompt']['md5'][:12]}), never copied into this repo; user message the `{st['preamble_version']}` preamble, the whole transcript as Jev's sentence pass rendered it, then the target ids in groups of up to {st['group_max']} closing early past {st['group_span']} sentences; {st['concurrency']} in flight, {st['max_attempts']} attempts on an error, {st['malformed_reasks']} re-ask on malformed JSON, {st['timeout_s']} s timeout, `reasoning_effort` {s['effort']}. Only the selection and the Jev side differ: the slice is the combiner's margin, and on non-routed sentences the combiner's keep/cut (as 5 or 0 at its {s['combiner']['threshold']:.2f} threshold) stands, with no trims anywhere. Every attempt is a row in the requests file with the raw answer, the parsed verdicts, token counts and cost.")
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


def report(args, parser):
    md_path = OUT_DIR / f"{OUT_STEM}.md"
    json_path = OUT_DIR / f"{OUT_STEM}.json"
    existing = [str(p) for p in (md_path, json_path) if p.exists()]
    if existing and not args.force:
        parser.error(f"refusing to overwrite {existing}; pass --force")
    summary = build_report()
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(md_path, summary, json_path)
    rule = summary["frozen_keep_rule"]["rule"]
    print(json.dumps({"markdown": str(md_path), "json": str(json_path),
                      "reproduction": summary["reproduction"],
                      "frozen_keep_rule": rule,
                      "stack": {kr: {split: summary["by_rule"][kr][split]["sentence_points"]
                                     for split in ("fit", "heldout", "all")}
                                for kr in ("decision", rule)},
                      "ceiling_all": summary["ceiling"]["all"]["sentence_points"],
                      "placement": summary["placement"],
                      "agree": summary["agreement"]["agree_rate"],
                      "seconds_per_episode_mean": summary["seconds_per_episode_mean"],
                      "cost_per_episode_mean": summary["cost_per_episode_mean"],
                      "total_spend_usd": summary["total_spend_usd"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--effort", default=hl.DEFAULT_EFFORT, choices=hl.EFFORTS)
    parser.add_argument("--episodes", nargs="+", default=None, help="subset of the 18 (default: all)")
    parser.add_argument("--limit-groups", type=int, default=None,
                        help="send only the first N groups of each episode (smoke test)")
    parser.add_argument("--out", default=None, help=f"output basename (default {OUT_STEM}-m25)")
    parser.add_argument("--run", action="store_true", help="actually call Luna")
    parser.add_argument("--resume", action="store_true", help="finish a run whose files exist")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD, help="hard spend cap in USD")
    parser.add_argument("--concurrency", type=int, default=hl.CONCURRENCY)
    parser.add_argument("--check", action="store_true",
                        help="recompute p_keep and reproduce the offline ceiling only; no calls")
    parser.add_argument("--report", action="store_true", help="write the markdown and JSON write-up")
    parser.add_argument("--force", action="store_true", help="overwrite the write-up")
    args = parser.parse_args()
    args.share = SHARE
    if args.check:
        result, _inputs, _cp, _routed, _cutoff = check()
        print(json.dumps(result, indent=2))
        return 0
    if args.report:
        return report(args, parser)
    return run(args, parser)


if __name__ == "__main__":
    sys.exit(main())
