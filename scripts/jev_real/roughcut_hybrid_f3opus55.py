"""The f3-Opus stack on claude-opus-5-5: the f3 combiner decides, Opus 5.5 overrides its unsure 25% (and 50%).

Step 6 of ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``.
The call is ``roughcut_hybrid_f3opus.py``'s (rules5 system prompt, preamble plus
whole transcript in the cached prefix, targets uncached, groups of up to 80,
first group alone, medium effort) with the model swapped through its ``Run``;
no Opus 5 answer is reused. The 25% slice is asked fresh
(``roughcut-hybrid-f3opus55-m25-*``); the 50% top-up
(``roughcut-hybrid-f3opus55-m50-*``) reuses the 25% answers and asks only the
rest. Each slice's keep rule is chosen on the fit six and frozen before its
held-out calls.

Modes::

  python scripts/jev_real/roughcut_hybrid_f3opus55.py                                  # plan, 25%, no calls
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --run --episodes colman-03.03-muscles-crit   # probe
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --extrapolate                    # 18-episode 25% cost
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --run --resume --episodes <fit five>
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --keep-rule                      # freeze 25% rule, fit six
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --run --resume                   # the held-out 12
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --share m50 --extrapolate        # top-up plus 25% actual
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --share m50 --run --episodes <fit six>
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --share m50 --keep-rule
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --share m50 --run --resume
  python scripts/jev_real/roughcut_hybrid_f3opus55.py --report                         # write-up, no calls

READ-ONLY against solar-sailer. No network calls without ``--run``.
"""

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import roughcut_hybrid_f3opus as f3o  # noqa: E402  (the run mechanics, slices, Opus 5 run)
import roughcut_hybrid_f1opus as fo  # noqa: E402  (the Opus call, donors)
import roughcut_hybrid_luna as hl  # noqa: E402
import roughcut_route2_routing as r2  # noqa: E402
import roughcut_jev_report as report_mod  # noqa: E402

OUT_DIR = f3o.OUT_DIR
MODEL = "claude-opus-5-5"
OUT_STEM = "roughcut-hybrid-f3opus55"
#: Stop lines from the step-6 brief: 25% projection, and 50% top-up projection plus the 25% actual.
RUNS = {
    "m25": f3o.Run(model=MODEL, name=f"{OUT_STEM}-m25", arm="hybrid_f3opus55_m25", tag="m25", stop_usd=4.50),
    "m50": f3o.Run(model=MODEL, name=f"{OUT_STEM}-m50", arm="hybrid_f3opus55_m50", tag="m50",
                   prior=f"{OUT_STEM}-m25", stop_usd=6.50),
}
LABEL = "f3opus55"
FIT = f3o.FIT
OPUS5_JSON = OUT_DIR / f"{f3o.OUT_STEM}.json"
pct, table, num, money, fingerprint = hl.pct, hl.table, hl.num, hl.money, hl.fingerprint


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def keep_rule_path(tag):
    return OUT_DIR / f"{RUNS[tag].name}-keeprule.json"


def projection_path(tag):
    """The last ``--extrapolate`` result for a slice (the go / no-go record)."""
    return OUT_DIR / f"{RUNS[tag].name}-projection.json"


def run_cost(tag):
    path = f3o.run_paths(RUNS[tag].name)["timing"]
    return json.loads(path.read_text(encoding="utf-8"))["totals"]["cost_usd"] if path.exists() else 0.0


def load_view(tag, episodes, require_all=True):
    views = f3o.load_run(episodes, require_all, name=RUNS[tag].name, tags=(tag,), label=LABEL)
    return views[tag] if views else None


# ---------------------------------------------------------------------------
# keep rule, per slice, fit six only
# ---------------------------------------------------------------------------

def keep_rule_mode(args, parser):
    tag, path = args.share, keep_rule_path(args.share)
    if path.exists() and not args.force:
        parser.error(f"refusing to overwrite {path}")
    st = f3o.setup()
    inputs, fit = st["inputs"], list(FIT)
    view = load_view(tag, fit)
    if view is None:
        raise SystemExit(f"no {RUNS[tag].name} run on disk")
    for e in fit:
        if view["routed"][e] != st["routed"][tag][e]:
            raise SystemExit(f"{e}: the run's {tag} slice is not the slice selected now")
    donors = {}
    hl.register_live_donors(donors, view, fit)
    scorer = r2.Scorer({e: inputs["jev"][e] for e in fit}, donors, {e: inputs["removals"][e] for e in fit})
    chosen = f3o.choose_rules(scorer, st, {tag: view}, fit)[tag]
    heldout_on_disk = [e for e in view["timing"]["episodes"] if e not in fit]
    frozen = {"share": tag, **chosen,
              "chosen_on": f"fit six of {RUNS[tag].name}, before any held-out call of this run",
              "tie_break": "best fit-six SP, ties to the decision field, then the lower score threshold",
              "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "fit_episodes_on_disk": fit, "heldout_episodes_on_disk": heldout_on_disk}
    if heldout_on_disk:
        raise SystemExit(f"held-out episodes already on disk {heldout_on_disk}; the rule would not be blind")
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def opus5_as_archived(view, episodes):
    """The live Opus 5 answers shaped like an archived donor, for ``hl.live_vs_archived``."""
    out = {}
    for e in episodes:
        rows = {sid: r for sid, r in view["rows"][e].items() if r["routed"] and r["opus_answered"]}
        out[e] = {"decisions": {sid: {"score": 5.0 if r["opus_decision"] == "keep" else 0.0} for sid, r in rows.items()},
                  "raw": {sid: {"score": r["opus_score"]} for sid, r in rows.items()}}
    return out


def run_block(tag, view, st, per_sentence_ref=None):
    """Counts, tokens, cache and timing of one Opus 5.5 run."""
    t = view["timing"]["totals"]
    later = [r for r in view["requests"] if r["phase"] != "first" and not r["error"]]
    first_episode = next(iter(view["timing"]["episodes"]))
    first_later = [r for r in later if r["episode"] == first_episode]
    return {
        "name": RUNS[tag].name, "generated_utc": view["timing"]["generated_utc"],
        "counts": {k: t[k] for k in ("asked", "asked_now", "reused", "retake_vetoed", "substituted",
                                     "substituted_new", "groups", "requests", "retries", "errors",
                                     "malformed", "refusals", "fallbacks")},
        "tokens": {k: t[k] for k in ("prompt_tokens", "uncached_input_tokens", "cache_write_tokens",
                                     "cache_read_tokens", "completion_tokens")},
        "cost_usd": t["cost_usd"], "cost_listed_usd": t["cost_listed_usd"], "wall_clock_s": t["wall_clock_s"],
        "per_sentence_usd": t["cost_usd"] / max(1, t["asked_now"]),
        "cache": {"read_share": t["cache_read_tokens"] / max(1, t["prompt_tokens"]),
                  "later_requests": len(later),
                  "later_requests_with_cache_read": sum(1 for r in later if (r["cache_read_tokens"] or 0) > 0),
                  "first_episode": first_episode, "first_episode_later_requests": len(first_later),
                  "first_episode_later_with_cache_read": sum(1 for r in first_later
                                                             if (r["cache_read_tokens"] or 0) > 0)},
        "parse_errors": [{"episode": r["episode"], "group": r["group"], "error": r["parse_error"],
                          "missing": len(r["missing_ids"] or [])}
                         for r in view["requests"] if not r["error"] and r["parse_error"]],
        "provider_models": sorted({r["provider_model"] for r in view["requests"] if r["provider_model"]}),
    }


def build_report():
    st = f3o.setup()
    inputs, cp, cfg = st["inputs"], st["cp"], st["cfg"]
    episodes, removals, f3_jev = inputs["episodes"], inputs["removals"], inputs["jev"]
    heldout = [e for e in episodes if e not in FIT]
    tags = [tag for tag in ("m25", "m50") if f3o.run_paths(RUNS[tag].name)["timing"].exists()]
    frozen, views = {}, {}
    for tag in tags:
        if not keep_rule_path(tag).exists():
            raise SystemExit(f"no frozen {tag} keep rule")
        frozen[tag] = json.loads(keep_rule_path(tag).read_text(encoding="utf-8"))
        views[tag] = load_view(tag, episodes)
        for e in episodes:
            if views[tag]["routed"][e] != st["routed"][tag][e]:
                raise SystemExit(f"{e}: the run's {tag} slice is not the slice selected now")
    o5_views = f3o.load_run(episodes)
    o5_doc = json.loads(OPUS5_JSON.read_text(encoding="utf-8"))
    o5_rules = o5_doc["rules"]
    pub25 = st["published"]["m25"]

    log("donors and scoring...")
    donors, donor_paths = fo.load_donors(episodes)
    for view in list(views.values()) + list(o5_views.values()):
        hl.register_live_donors(donors, view, episodes)
    scorer = r2.Scorer(f3_jev, donors, removals)
    f3_eps = {e: scorer.episode(None, e, set()) for e in episodes}
    f3_pooled = r2.splits(f3_eps, episodes)

    rules = {}
    for tag in tags:
        again = f3o.choose_rules(scorer, st, {tag: views[tag]}, episodes)[tag]
        if again["rule"] != frozen[tag]["rule"]:
            raise SystemExit(f"{tag}: fit-six choice recomputed as {again['rule']}, frozen file says {frozen[tag]['rule']}")
        rules[tag] = frozen[tag]["rule"]
    by_rule, per_ep = {}, {}
    for tag in tags:
        routed = st["routed"][tag]
        by_rule[tag] = {kr: r2.splits({e: scorer.episode(hl.donor_key(views[tag]["tag"], kr), e, routed[e])
                                       for e in episodes}, episodes) for kr in hl.KEEP_RULES}
        per_ep[tag] = {e: scorer.episode(hl.donor_key(views[tag]["tag"], rules[tag]), e, routed[e]) for e in episodes}
    o5 = {tag: {kr: r2.splits({e: scorer.episode(hl.donor_key(o5_views[tag]["tag"], kr), e, st["routed"][tag][e])
                               for e in episodes}, episodes) for kr in hl.KEEP_RULES} for tag in ("m25", "m50")}
    o5_per_ep = {tag: {e: scorer.episode(hl.donor_key(o5_views[tag]["tag"], o5_rules[tag]), e, st["routed"][tag][e])
                       for e in episodes} for tag in ("m25", "m50")}
    reproduction = {"f3_combiner": {"mine": f3_pooled["all"]["sentence_points"], "published": pub25["ladder_sp"]}}
    for tag in ("m25", "m50"):
        reproduction[f"opus5_{tag}_{o5_rules[tag]}"] = {
            "mine": o5[tag][o5_rules[tag]]["all"]["sentence_points"],
            "published": o5_doc["by_rule"][tag][o5_rules[tag]]["all"]["sentence_points"]}
    for key, pair in reproduction.items():
        if abs(pair["mine"] - pair["published"]) > 5e-4:
            raise SystemExit(f"reproduction failed for {key}: {pair}")

    log("ladder, flips, agreement...")
    published = r2.published_rows(episodes)
    quoted, _r, _c = report_mod.ladder(episodes)
    ladder = [{"label": report_mod.quoted_label(r), "key": r["key"], "sentence_points": r["sentence_points_layered"]}
              for r in quoted]
    ladder.append({"label": f"Jev f3 `{cp['chosen']}` combiner alone", "key": "f3",
                   "sentence_points": f3_pooled["all"]["sentence_points"]})
    for tag in ("m25", "m50"):
        ladder.append({"label": f"f3 + Opus 5 {f3o.SHARES[tag] * 100:g}%, `{o5_rules[tag]}`", "key": f"opus5:{tag}",
                       "sentence_points": o5[tag][o5_rules[tag]]["all"]["sentence_points"]})
    for tag in tags:
        ladder.append({"label": f"f3 + Opus 5.5 {f3o.SHARES[tag] * 100:g}%, `{rules[tag]}`", "key": f"opus55:{tag}",
                       "sentence_points": by_rule[tag][rules[tag]]["all"]["sentence_points"]})
    ladder.sort(key=lambda r: -r["sentence_points"])
    placement = {row["key"]: r2.placement(row["sentence_points"], published)
                 for row in ladder if row["key"] == "f3" or ":" in row["key"]}
    for tag in tags:
        if rules[tag] != "decision":
            placement[f"opus55:{tag}:decision"] = r2.placement(by_rule[tag]["decision"]["all"]["sentence_points"],
                                                               published)
    top = published[0]
    top_arm = {"key": top["key"], "label": top["label"], "sentence_points": top["sentence_points_layered"],
               "rank": top["rank"]}

    human, f3_states = {}, {}
    for e in episodes:
        human[e], f3_states[e] = r2.states(e, f3_jev[e], removals[e])
    flips = {}
    for tag in tags:
        flips[f"opus55:{tag}"] = r2.flip_block(scorer, hl.donor_key(views[tag]["tag"], rules[tag]),
                                               st["routed"][tag], episodes, human, f3_states)
    for tag in ("m25", "m50"):
        flips[f"opus5:{tag}"] = r2.flip_block(scorer, hl.donor_key(o5_views[tag]["tag"], o5_rules[tag]),
                                              st["routed"][tag], episodes, human, f3_states)
    agreement = {}
    for tag in tags:
        arch = opus5_as_archived(o5_views[tag], episodes)
        agreement[tag] = {split: hl.live_vs_archived(views[tag], arch, eps, human, removals)
                          for split, eps in (("all", episodes), ("fit", list(FIT)), ("heldout", heldout))}
        agreement[tag]["archived_opus_agentic"] = hl.live_vs_archived(views[tag], donors["opus"], episodes,
                                                                      human, removals)

    log("per episode...")
    f3_secs = pub25["seconds"]
    t25 = views["m25"]["timing"]["episodes"] if "m25" in views else {}
    t50 = views["m50"]["timing"]["episodes"] if "m50" in views else {}
    per_episode = {}
    for e in episodes:
        f3s = f3_secs[e]["total_s"]
        p = {"sentences": len(f3_jev[e]), "f3": f3_eps[e], "f3_seconds": f3s,
             "opus5_m25": o5_per_ep["m25"][e], "opus5_m50": o5_per_ep["m50"][e]}
        if t25:
            a = t25[e]
            p.update(routed_25=a["routed"], asked_25=a["asked"], requests_25=a["requests"], retries_25=a["retries"],
                     refusals_25=a["refusals"], fallbacks_25=a["fallbacks"], errors_25=a["errors"],
                     opus_seconds_25=a["wall_clock_s"], cost_25=a["cost_usd"], seconds_25=a["wall_clock_s"] + f3s,
                     cache_read_share_25=a["cache_read_tokens"] / max(1, a["prompt_tokens"]), m25=per_ep["m25"][e])
        if t50:
            b = t50[e]
            p.update(routed_50=b["routed"], asked_50=b["asked"], reused_50=b["reused"], asked_now_50=b["asked_now"],
                     requests_50=b["requests"], retries_50=b["retries"], refusals_50=b["refusals"],
                     fallbacks_50=b["fallbacks"], errors_50=b["errors"], opus_seconds_50=b["wall_clock_s"],
                     cost_topup_50=b["cost_usd"], cost_50=b["cost_usd"] + p["cost_25"],
                     seconds_50=b["wall_clock_s"] + p["opus_seconds_25"] + f3s, m50=per_ep["m50"][e])
        per_episode[e] = p

    def mean(key):
        return statistics.mean(per_episode[e][key] for e in episodes)

    blocks = {tag: run_block(tag, views[tag], st) for tag in tags}
    rules_text = hl.read_rules()
    for tag in tags:   # the run files keep the probe's one-episode plan; the gate used the 18-episode one
        blocks[tag]["estimate"] = fo.estimate(f3o.plan(st, rules_text, episodes, RUNS[tag])[0], model=MODEL)[1]
    inputs_fp = {k: fingerprint(v) for k, v in inputs["run_paths"].items()}
    inputs_fp["rules_prompt"] = fingerprint(hl.RULES_PATH)
    inputs_fp["ladder_reference"] = fingerprint(report_mod.REFERENCE_JSON)
    inputs_fp["f3_weights"] = fingerprint(cp["weights_path"])
    for split, pth in cp["feature_paths"].items():
        inputs_fp[f"f3_{split}_features"] = fingerprint(pth["features"])
    inputs_fp["f3_writeup_json"] = fingerprint(cfg.writeup_json)
    inputs_fp["opus5_writeup_json"] = fingerprint(OPUS5_JSON)
    for kind, pth in o5_views["m50"]["paths"].items():
        inputs_fp[f"opus5_run:{kind}"] = fingerprint(pth)
    inputs_fp["opus5_keep_rule"] = fingerprint(f3o.KEEP_RULE_PATH)
    for key in ("luna", "opus"):
        for pth in donor_paths[key]:
            inputs_fp[f"{key}:{Path(pth).name}"] = fingerprint(pth)
    for tag in RUNS:
        if projection_path(tag).exists():
            inputs_fp[f"opus55_{tag}_projection"] = fingerprint(projection_path(tag))
    for tag in tags:
        for kind, pth in views[tag]["paths"].items():
            inputs_fp[f"opus55_{tag}_run:{kind}"] = fingerprint(pth)
        inputs_fp[f"opus55_{tag}_keep_rule"] = fingerprint(keep_rule_path(tag))
    digest = hashlib.md5()
    for e in episodes:
        digest.update(r2.md5(r2.removals_cache_path(e)).encode())

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_hybrid_f3opus55.py",
        "model": MODEL, "effort": fo.EFFORT, "episodes": episodes, "fit": FIT, "heldout": heldout,
        "sentences": sum(len(f3_jev[e]) for e in episodes),
        "combiner": {"prefix": f3o.COMBINER, "set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"]},
        "shares": f3o.SHARES, "cutoff": st["cutoff"],
        "routed": {tag: sum(len(v) for v in st["routed"][tag].values()) for tag in f3o.SHARES},
        "tags_run": tags, "m50_ran": "m50" in tags,
        "frozen_keep_rules": frozen, "rules": rules, "keep_rules": hl.KEEP_RULES,
        "reproduction": reproduction, "f3": f3_pooled, "by_rule": by_rule,
        "opus5": {"rules": o5_rules, "by_rule": o5,
                  "seconds_per_episode_mean": {"m50": o5_doc["seconds_per_episode_mean"],
                                               "m25": o5_doc["seconds_25_est_mean"]},
                  "cost_per_episode_mean": {"m50": o5_doc["fresh_cost_per_episode_mean"],
                                            "m25": o5_doc["fresh_cost_25_per_episode_mean"]},
                  "cache_read_share": o5_doc["cache"]["read_share"],
                  "counts": o5_doc["counts"], "counts_25": o5_doc["counts_25"]},
        "top_arm": top_arm, "ladder": ladder, "published_ladder": published, "placement": placement,
        "flips": flips, "agreement": agreement, "per_episode": per_episode, "runs": blocks,
        "cost_usd": sum(b["cost_usd"] for b in blocks.values()),
        "projections": {tag: json.loads(projection_path(tag).read_text(encoding="utf-8"))
                        for tag in RUNS if projection_path(tag).exists()},
        "f3_seconds_per_episode_mean": mean("f3_seconds"),
        "settings": {"group_max": fo.GROUP_MAX, "group_span": fo.GROUP_SPAN, "max_attempts": fo.MAX_ATTEMPTS,
                     "malformed_reasks": fo.MALFORMED_REASKS, "max_tokens": fo.MAX_TOKENS,
                     "preamble_version": hl.PREAMBLE_VERSION, "threshold": f3o.THRESHOLD,
                     "price_per_million_usd": dict(zip(("input", "cache_write", "cache_read", "output"),
                                                       fo.PRICES[MODEL]))},
        "inputs": inputs_fp, "removals_digest": digest.hexdigest(),
    }
    for tag in tags:
        out[f"seconds_per_episode_mean_{tag}"] = mean(f"seconds_{tag[1:]}")
        out[f"seconds_per_episode_max_{tag}"] = max(per_episode[e][f"seconds_{tag[1:]}"] for e in episodes)
        out[f"cost_per_episode_mean_{tag}"] = mean(f"cost_{tag[1:]}")
        out[f"cost_per_episode_max_{tag}"] = max(per_episode[e][f"cost_{tag[1:]}"] for e in episodes)
    return out


def write_markdown(path, s, json_path):
    lines = []
    add = lines.append
    tags, rules, o5 = s["tags_run"], s["rules"], s["opus5"]
    share = {tag: f"{s['shares'][tag] * 100:g}%" for tag in s["shares"]}
    f3 = s["f3"]
    cols = ["SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18", "GRADE all 18"]

    def cells(p):
        return [pct(p["fit"]["sentence_points"]), pct(p["heldout"]["sentence_points"]),
                pct(p["all"]["sentence_points"]), pct(p["all"]["word_score"]), pct(p["all"]["grade"])]

    add("Developer-facing notes on the f3-Opus 5.5 stack: the f3 combiner decides, claude-opus-5-5 overrides its unsure "
        + (" and ".join(share[t] for t in tags)) + ", with the Opus 5 runs alongside.")
    add("")
    add(f"Generated {s['generated_utc']} by `{s['script']}` from the run files on disk, the frozen keep-rule files, the f3 feature files and frozen weights, the Opus 5 f3 run, the archived donor ratings and the cached removal ranges. The report step makes no model calls. Every metric is x100, two decimals, with um removal and delete silence layered on (the ladder column). The JSON next to this file keeps the raw values.")
    add("")
    add("# The f3-Opus 5.5 stack")
    add("")
    add(f"Step 6 of the round-two design. f3 (`{s['combiner']['set']}`, keep threshold {s['combiner']['threshold']:.2f} on `5 * p_keep`) decides all {s['sentences']:,} sentences of the 18 ladder episodes. Its unsure sentences, margin `abs(5 * p_keep - threshold)` at or under a global cutoff, go to `{s['model']}` with the Opus 5 f3 run's call unchanged: rules5 system prompt, preamble plus whole transcript in the cached prefix, targets uncached, groups of up to {s['settings']['group_max']}, first group alone, {s['effort']} effort. The router sends Opus 5.5 adaptive thinking with that effort, as it did for Opus 5. No Opus 5 answer is reused.")
    add("")
    r25 = s["runs"].get("m25")
    if r25:
        text = (f"The 25% slice ({s['routed']['m25']:,} sentences, cutoff {s['cutoff']['m25']:.3f}) was asked fresh: {r25['counts']['asked']:,} sentences sent, the rest already cut by the retake pass. ")
        if "m50" in tags:
            r50 = s["runs"]["m50"]
            text += (f"The 50% top-up ({s['routed']['m50']:,} sentences, cutoff {s['cutoff']['m50']:.3f}) reused the {r50['counts']['reused']:,} 25% answers (same model, system prompt and cached prefix, checked by md5) and sent the other {r50['counts']['asked_now']:,}, so a 50% answer that came from the 25% run was given in a group of different composition. ")
        else:
            text += "The 50% top-up was not run (see the cost section). "
        text += "Each slice's keep rule was chosen on the fit six alone and frozen to disk before that slice's held-out calls: " + "; ".join(
            f"{share[t]} `{rules[t]}` at {s['frozen_keep_rules'][t]['frozen_utc']}" for t in tags) + "."
        add(text)
        add("")

    bl = []
    for t in tags:
        b = s["by_rule"][t][rules[t]]
        bl.append(f"at {share[t]} with `{rules[t]}` f3 plus Opus 5.5 scores {pct(b['all']['sentence_points'])} SP pooled 18, {pct(b['fit']['sentence_points'])} on the fit six and {pct(b['heldout']['sentence_points'])} held-out 12, against Opus 5's {pct(o5['by_rule'][t][o5['rules'][t]]['all']['sentence_points'])} / {pct(o5['by_rule'][t][o5['rules'][t]]['fit']['sentence_points'])} / {pct(o5['by_rule'][t][o5['rules'][t]]['heldout']['sentence_points'])}")
    ranks = "; ".join(f"f3 + Opus 5.5 {share[t]} {s['placement'][f'opus55:{t}']['text']}" for t in tags)
    add(f"Bottom line: {'; '.join(bl)}. f3 alone is {pct(f3['all']['sentence_points'])}. On the full published ladder with modules, {ranks}. The top published arm is {s['top_arm']['label']} at {pct(s['top_arm']['sentence_points'])}.")
    add("")

    add("## Results")
    add("")
    add("Seconds per episode: Opus wall clock (first group alone, then up to 8 in flight) plus f3's own per-episode time."
        + (" For the Opus 5.5 50% row it adds the 25% run's wall clock to the top-up's, since the 50% point needed both; a single fresh 50% run would overlap them." if "m50" in tags else "")
        + " Dollars per episode are model spend: the Opus 5.5 25% row is what that run paid"
        + (", the 50% row adds the top-up" if "m50" in tags else "")
        + ". The Opus 5 rows are the fresh-run figures from its write-up: its 50% seconds are measured, its 25% seconds an estimate from step 5's wall clock, and its dollars what a run asking every sentence would have paid at Opus 5 prices ($5 in, $25 out), since its runs reused step 5's answers.")
    add("")
    rows = []
    for t in tags:
        for kr in ([rules[t]] + (["decision"] if rules[t] != "decision" else [])):
            label = f"f3 + Opus 5.5 {share[t]}, " + (f"fit-chosen `{kr}`" if kr == rules[t] else "Opus `decision`")
            rows.append([label] + cells(s["by_rule"][t][kr])
                        + [num(s[f"seconds_per_episode_mean_{t}"], 1), money(s[f"cost_per_episode_mean_{t}"])])
    for t in ("m25", "m50"):
        rows.append([f"f3 + Opus 5 {share[t]}, fit-chosen `{o5['rules'][t]}` (reference)"]
                    + cells(o5["by_rule"][t][o5["rules"][t]])
                    + [num(o5["seconds_per_episode_mean"][t], 1), money(o5["cost_per_episode_mean"][t])])
    rows.append(["f3 alone"] + cells(f3) + [num(s["f3_seconds_per_episode_mean"], 1), money(0.0)])
    rows.append([f"{s['top_arm']['label']} (top published arm)", "", "", pct(s["top_arm"]["sentence_points"]),
                 "", "", "", ""])
    lines.extend(table(["arm"] + cols + ["s/ep mean", "model $/ep"], rows))
    add("")
    add("f3 alone and every stack score the fit six with f3's leave-one-out `p_keep` and the held-out 12 with its frozen weights. The top published arm has only its ladder number here.")
    add("")
    ladder_text = ", ".join(f"{r['label']} {pct(r['sentence_points'])}" for r in s["ladder"])
    pl = s["placement"]
    places = "; ".join(f"{r['label']} {pl[r['key']]['text']}" for r in s["ladder"] if r["key"] in pl)
    add(f"Ladder, with modules, same 18 episodes (published arms quoted as the top arm, the bench page's headline best, each model family's best and the Opus 5 agentic arm): {ladder_text}. {report_mod.placement_lead(s['published_ladder'])}. Placement: {places}.")
    add("")

    add("## Every keep rule")
    add("")
    add("Chosen on the fit six alone (best fit-six SP, ties to the decision field, then the lower threshold); the starred row is the frozen one. The Opus 5 columns are the same rules on the Opus 5 answers.")
    add("")
    rows = []
    for t in tags:
        secs = s[f"seconds_per_episode_mean_{t}"]
        for kr in s["keep_rules"]:
            p, q = s["by_rule"][t][kr], o5["by_rule"][t][kr]
            rows.append([share[t], ("* " if kr == rules[t] else "") + kr] + cells(p)
                        + [pct(q["fit"]["sentence_points"]), pct(q["heldout"]["sentence_points"]),
                           pct(q["all"]["sentence_points"]), num(secs, 1)])
    lines.extend(table(["share", "keep rule"] + cols + ["Opus 5 SP fit 6", "Opus 5 SP held-out 12",
                                                          "Opus 5 SP all 18", "s/ep mean"], rows))
    add("")

    add("## Opus 5.5 against Opus 5 on the same sentences")
    add("")
    add(f"Keep/cut decision fields on the sentences both live runs answered in each slice, same system prompt and same cached prefix. Group composition differs: {o5['counts_25']['reused']:,} of the {o5['counts_25']['asked']:,} Opus 5 answers on the 25% slice came from step 5's groups, built on the f1 slice. Right means matching the editor, where kept means full or partial. Score within 1 compares the two 0-5 scores. The last row per share compares Opus 5.5 with the archived agentic Opus 5 ratings instead, a different prompt and session.")
    add("")
    rows = []
    for t in tags:
        secs = s[f"seconds_per_episode_mean_{t}"]
        for key, label in (("all", "all 18, Opus 5 live"), ("fit", "fit six, Opus 5 live"),
                           ("heldout", "held-out 12, Opus 5 live"), ("archived_opus_agentic", "all 18, archived agentic Opus 5")):
            a = s["agreement"][t][key]
            rows.append([share[t], label, a.get("n", 0), pct(a["agree_rate"]), pct(a["live_right_rate"]),
                         pct(a["archived_right_rate"]),
                         f"{a.get('disagree_live_right', 0)} / {a.get('disagree_archived_right', 0)} of {a.get('disagree', 0)}",
                         pct(a["live_keep_rate"]), pct(a["archived_keep_rate"]), pct(a["editor_keep_rate"]),
                         pct(a["score_within_1_rate"]), num(secs, 1)])
    lines.extend(table(["share", "compared with", "n", "agree", "Opus 5.5 right", "other right",
                        "disagreements 5.5 / other right", "5.5 keep rate", "other keep rate", "editor keep rate",
                        "score within 1", "s/ep mean"], rows))
    add("")

    add("## Where the gains come from")
    add("")
    add("A flip is a routed sentence whose keep/cut changed when the Opus verdict replaced f3's, read off the scoring module's sentence states with the modules layered. Right means the new state matches the editor.")
    add("")
    rows = []
    for key, label, secs in ([(f"opus55:{t}", f"Opus 5.5 {share[t]} `{rules[t]}`", s[f"seconds_per_episode_mean_{t}"]) for t in tags]
                             + [(f"opus5:{t}", f"Opus 5 {share[t]} `{o5['rules'][t]}`", o5["seconds_per_episode_mean"][t])
                                for t in ("m25", "m50")]):
        c = s["flips"][key]
        routed = c.get("routed", 0)
        rows.append([label, routed, c.get("flips", 0), c.get("flips_right", 0), c.get("flips_wrong", 0),
                     f"{c.get('cut_to_kept_right', 0)} / {c.get('cut_to_kept_wrong', 0)}",
                     f"{c.get('kept_to_cut_right', 0)} / {c.get('kept_to_cut_wrong', 0)}",
                     pct(c.get("jev_agreed", 0) / routed if routed else None),
                     pct(c.get("hybrid_agreed", 0) / routed if routed else None), num(secs, 1)])
    lines.extend(table(["substitution", "routed", "flips", "right", "wrong", "cut to kept right / wrong",
                        "kept to cut right / wrong", "agreement on slice, f3", "agreement on slice, after routing",
                        "s/ep mean"], rows))
    add("")

    add(f"## Per episode (25% cutoff {s['cutoff']['m25']:.3f}, 50% cutoff {s['cutoff']['m50']:.3f})")
    add("")
    rows = []
    has50 = "m50" in tags
    for e in s["episodes"]:
        p = s["per_episode"][e]
        row = [e + (" (fit)" if e in s["fit"] else ""), p["sentences"], p["routed_25"],
               pct(p["f3"]["sentence_points"]), pct(p["opus5_m25"]["sentence_points"]), pct(p["m25"]["sentence_points"])]
        if has50:
            row += [p["asked_now_50"], pct(p["opus5_m50"]["sentence_points"]), pct(p["m50"]["sentence_points"])]
        row += [p["requests_25"] + (p["requests_50"] if has50 else 0), p["retries_25"] + (p["retries_50"] if has50 else 0),
                p["fallbacks_25"] + (p["fallbacks_50"] if has50 else 0), num(p["f3_seconds"], 1), num(p["seconds_25"], 1)]
        row += [num(p["seconds_50"], 1)] if has50 else []
        row += [money(p["cost_25"])] + ([money(p["cost_50"])] if has50 else [])
        row += [f"{p['cache_read_share_25'] * 100:.0f}%"]
        rows.append(row)
    head = ["episode", "sentences", "routed 25%", "SP f3", "SP Opus 5 25%", f"SP Opus 5.5 25% {rules['m25']}"]
    if has50:
        head += ["asked in top-up", "SP Opus 5 50%", f"SP Opus 5.5 50% {rules['m50']}"]
    head += ["requests", "retries", "fallback targets", "f3 s", "s/ep 25%"] + (["s/ep 50%"] if has50 else [])
    head += ["Opus 5.5 $ 25%"] + (["Opus 5.5 $ 50%"] if has50 else []) + ["cache read share 25%"]
    lines.extend(table(head, rows))
    add("")

    add("## Cost, caching and failures")
    add("")
    pr = s["settings"]["price_per_million_usd"]
    for t in tags:
        b = s["runs"][t]
        c, tk, ca = b["counts"], b["tokens"], b["cache"]
        name = "25% run" if t == "m25" else "50% top-up"
        notes = "; ".join(f"{p['episode']} group {p['group']}: {p['error']}, {p['missing']} targets missing"
                          for p in b["parse_errors"]) or "none"
        add(f"The {name} (`{b['name']}`): {c['requests']} requests over {c['groups']} groups, {c['retries']} retries, {c['errors']} errored attempts, {c['malformed']} malformed or incomplete answers ({notes}), {c['refusals']} refusals, {c['fallbacks']} targets left on f3's decision after the re-ask. It paid ${b['cost_usd']:.4f} by the router's accounting (${b['cost_listed_usd']:.4f} at list rates) for {c['asked_now']:,} sentences, ${b['per_sentence_usd']:.5f} each; the plan-mode estimate was ${b['estimate']['cost_usd']:.4f}. Input tokens {tk['prompt_tokens']:,}: {tk['cache_write_tokens']:,} cache writes, {tk['cache_read_tokens']:,} cache reads ({ca['read_share'] * 100:.1f}% of input), {tk['uncached_input_tokens']:,} uncached; {ca['later_requests_with_cache_read']} of the {ca['later_requests']} successful requests after an episode's first read the cache ({ca['first_episode_later_with_cache_read']} of {ca['first_episode_later_requests']} on the first episode run, {ca['first_episode']}). Output {tk['completion_tokens']:,} tokens, thinking included. Provider model {', '.join(b['provider_models'])}. Run wall clock {b['wall_clock_s']:.0f} s, files last written {b['generated_utc']}.")
        add("")
    add(f"Prices used by the router and in the listed column: ${pr['input']:.2f} input, ${pr['cache_write']:.2f} cache write, ${pr['cache_read']:.2f} cache read, ${pr['output']:.2f} output per million tokens. Opus 5.5 spend in this step ${s['cost_usd']:.4f}. Per episode, 25%: ${s['cost_per_episode_mean_m25']:.4f} mean, ${s['cost_per_episode_max_m25']:.4f} max, {s['seconds_per_episode_mean_m25']:.1f} s mean, {s['seconds_per_episode_max_m25']:.1f} s max." + (f" 50%: ${s['cost_per_episode_mean_m50']:.4f} mean, ${s['cost_per_episode_max_m50']:.4f} max, {s['seconds_per_episode_mean_m50']:.1f} s mean, {s['seconds_per_episode_max_m50']:.1f} s max." if "m50" in tags else "") + f" Opus 5's cache read share was {o5['cache_read_share'] * 100:.1f}%.")
    add("")
    pj = s["projections"]
    if "m25" in pj:
        add(f"Budget gates. Before the 25% run, the plan-mode estimate was ${s['runs']['m25']['estimate']['cost_usd']:.2f} (4 characters per token, 2,000 output tokens per request plus 60 per target); the probe on {', '.join(pj['m25']['probe_episodes'])} cost ${pj['m25']['probe_cost_usd']:.4f} and projected ${pj['m25']['projected_18_usd']:.2f} for the 18 against the ${pj['m25']['stop_threshold_usd']:.2f} stop line, so the run went ahead and came in at ${s['runs']['m25']['cost_usd']:.2f}.")
        add("")
    if "m50" in pj:
        top = pj["m50"]["projected_18_usd"] - pj["m50"]["spent_before_usd"]
        verdict = ("so it ran" if "m50" in tags else
                   "so the top-up was skipped and no 50% Opus 5.5 row exists")
        add(f"The 50% top-up, priced at the 25% run's real token rates ({pj['m50']['asked_now_total']:,} new sentences, {pj['m50']['reused_total']:,} reused), projected ${top:.2f}; with the 25% actual ${pj['m50']['spent_before_usd']:.2f} that is ${pj['m50']['projected_18_usd']:.2f} against the ${pj['m50']['stop_threshold_usd']:.2f} line in the step-6 brief, {verdict}.")
        add("")

    add("## Reproduction check")
    add("")
    rows = [[k, pct(v["mine"]), pct(v["published"]), "n/a"] for k, v in s["reproduction"].items()]
    lines.extend(table(["number", "SP here", "published", "s/ep mean"], rows))
    add("")

    add("## Files")
    add("")
    for key, value in s["inputs"].items():
        add(f"- input {key}: `{value['path']}`, md5 {value['md5'][:12]}, modified {value['modified_utc']}")
    add(f"- cached removal ranges for the 18 episodes under `docs/jev-real/removals/`, combined md5 {s['removals_digest'][:12]}")
    add(f"- this file: `{path}`")
    add(f"- data: `{json_path}`")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


def report(args, parser):
    md_path, json_path = OUT_DIR / f"{OUT_STEM}.md", OUT_DIR / f"{OUT_STEM}.json"
    existing = [str(p) for p in (md_path, json_path) if p.exists()]
    if existing and not args.force:
        parser.error(f"refusing to overwrite {existing}; pass --force")
    s = build_report()
    json_path.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    write_markdown(md_path, s, json_path)
    print(json.dumps({"markdown": str(md_path), "rules": s["rules"],
                      "stack": {t: {sp: s["by_rule"][t][s["rules"][t]][sp]["sentence_points"]
                                    for sp in ("fit", "heldout", "all")} for t in s["tags_run"]},
                      "placement": {k: v["text"] for k, v in s["placement"].items()},
                      "top_arm": s["top_arm"],
                      "agreement": {t: {k: v["agree_rate"] for k, v in a.items()} for t, a in s["agreement"].items()},
                      "flips": {k: (v.get("flips_right"), v.get("flips_wrong")) for k, v in s["flips"].items()},
                      "runs": {t: {"cost": b["cost_usd"], "cache": b["cache"], "counts": b["counts"]}
                               for t, b in s["runs"].items()},
                      "per_episode": {k: s[k] for k in s if k.startswith(("seconds_per_episode", "cost_per_episode"))},
                      "cost_usd": s["cost_usd"]}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share", choices=sorted(RUNS), default="m25", help="which slice's run (default m25)")
    parser.add_argument("--episodes", nargs="+", default=None, help="subset of the 18 (default: all)")
    parser.add_argument("--run", action="store_true", help="actually call Opus 5.5")
    parser.add_argument("--resume", action="store_true", help="add episodes to a run whose files exist")
    parser.add_argument("--budget", type=float, default=None,
                        help="hard spend cap in USD for this run (default: its stop line, less the 25%% actual for m50)")
    parser.add_argument("--concurrency", type=int, default=fo.CONCURRENCY)
    parser.add_argument("--extrapolate", action="store_true", help="18-episode cost from real usage; no calls")
    parser.add_argument("--keep-rule", action="store_true", help="choose and freeze this slice's rule on the fit six")
    parser.add_argument("--report", action="store_true", help="write the markdown and JSON write-up")
    parser.add_argument("--force", action="store_true", help="overwrite the write-up or keep-rule file")
    args = parser.parse_args()
    run = RUNS[args.share]
    before = run_cost("m25") if args.share == "m50" else 0.0
    if args.budget is None:
        args.budget = round(run.stop_usd - before, 4)
    if args.extrapolate:
        ref = None if f3o.run_paths(run.name)["timing"].exists() or not run.prior else run.prior
        out = f3o.extrapolate(run, ref=ref, spent_before=before)
        out["computed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        projection_path(args.share).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0 if out["go"] else 3
    if args.keep_rule:
        return keep_rule_mode(args, parser)
    if args.report:
        return report(args, parser)
    return f3o.execute(args, parser, run)


if __name__ == "__main__":
    sys.exit(main())
