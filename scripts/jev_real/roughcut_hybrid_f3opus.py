"""The stack with Opus on the f3 base: the f3 combiner decides, claude-opus-5 overrides its unsure 50%.

Follows the third pass of ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``
after Stan picked f3 as the winning Jev-only combiner on the held-out result.
The Opus call is step 5's (``roughcut_hybrid_f1opus.py``: rules5 system prompt,
preamble plus whole transcript in the cached prefix, targets uncached, groups
of up to 80, first group alone, medium effort, ``claude-opus-5``). The slice is
the f3 combiner's margin bottom 50%, margin ``abs(5 * p_keep - threshold)`` at
f3's frozen threshold, one global cutoff over the 18 ladder episodes, with
``p_keep`` out of fold on the fit six and from the frozen weights on the
held-out 12 (``roughcut_hybrid_f1luna.check`` with the f3 config).

Cost saver: a sentence in the new slice that step 5 answered keeps step 5's
answer (same model, prompt and whole-transcript prefix, checked by md5); only
the rest are asked. Reused answers came from groups of a different
composition. f3's own bottom 25% is nested in its bottom 50%, so the union of
answers also gives the f3 + Opus 25% point.

Modes::

  python scripts/jev_real/roughcut_hybrid_f3opus.py --ceiling       # archived Opus on both f3 slices, $0
  python scripts/jev_real/roughcut_hybrid_f3opus.py                 # plan and estimate, no calls
  python scripts/jev_real/roughcut_hybrid_f3opus.py --run --episodes colman-03.03-muscles-crit   # probe
  python scripts/jev_real/roughcut_hybrid_f3opus.py --extrapolate   # 18-episode cost from the probe
  python scripts/jev_real/roughcut_hybrid_f3opus.py --run --resume --episodes <fit five>
  python scripts/jev_real/roughcut_hybrid_f3opus.py --keep-rule     # choose and freeze on the fit six only
  python scripts/jev_real/roughcut_hybrid_f3opus.py --run --resume  # the held-out 12
  python scripts/jev_real/roughcut_hybrid_f3opus.py --report        # write-up from the run on disk

READ-ONLY against solar-sailer. No network calls without ``--run``.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"

sys.path.insert(0, str(HERE))

import roughcut_hybrid_f1opus as fo  # noqa: E402  (the Opus call, jobs, donors)
import roughcut_hybrid_f1luna as fl  # noqa: E402  (the combiner seat and slice)
import roughcut_hybrid_luna as hl  # noqa: E402
import roughcut_route2_routing as r2  # noqa: E402
import roughcut_jev_report as report_mod  # noqa: E402
import roughcut_jev as pipeline  # noqa: E402
from common import JsonlWriter  # noqa: E402

MODEL, EFFORT = fo.MODEL, fo.EFFORT
COMBINER = "roughcut-jev-f3"
OUT_STEM = "roughcut-hybrid-f3opus"
SHARES = {"m50": 0.50, "m25": 0.25}
MAIN = "m50"
NAME = f"{OUT_STEM}-m50"
ARM = "hybrid_f3opus_m50"
STEP5_NAME = fo.NAME                     # roughcut-hybrid-f1opus-m25
STEP5_JSON = OUT_DIR / f"{fo.OUT_STEM}.json"
F1_JSON = OUT_DIR / "roughcut-jev-f1.json"
PROBE_EPISODE = fo.PROBE_EPISODE
STOP_USD = 9.00                          # projected task total that stops the run
DEFAULT_BUDGET_USD = 9.00
PRICE_IN, PRICE_WRITE, PRICE_READ, PRICE_OUT = fo.PRICE_IN, fo.PRICE_WRITE, fo.PRICE_READ, fo.PRICE_OUT
FIT, KEPT, THRESHOLD = r2.FIT, r2.KEPT, r2.THRESHOLD
KEEP_RULE_PATH = OUT_DIR / f"{NAME}-keeprule.json"
pct, table, num, money, fingerprint, md5_text = hl.pct, hl.table, hl.num, hl.money, hl.fingerprint, hl.md5_text


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run_paths(name=NAME):
    return fo.run_paths(name)


# ---------------------------------------------------------------------------
# the slice and the reusable step-5 answers
# ---------------------------------------------------------------------------

def setup():
    """f3's p_keep, both slices, reproduction of the published offline numbers. $0."""
    cfg = fl.stack_config(COMBINER)
    result, inputs, cp, routed25, cutoff25 = fl.check(log, cfg)
    routed50, cutoff50 = fl.select_slice(inputs, SHARES["m50"])
    published50 = fl.published_route2(SHARES["m50"], cfg)
    n50 = sum(len(v) for v in routed50.values())
    if n50 != published50["routed"] or abs(cutoff50 - published50["cutoff_margin"]) > 1e-9:
        raise SystemExit(f"50% slice {n50} / {cutoff50} does not match the f3 write-up "
                         f"{published50['routed']} / {published50['cutoff_margin']}")
    for e in inputs["episodes"]:
        if not routed25[e] <= routed50[e]:
            raise SystemExit(f"{e}: the 25% slice is not inside the 50% slice")
    return {"cfg": cfg, "check": result, "inputs": inputs, "cp": cp,
            "routed": {"m25": routed25, "m50": routed50}, "cutoff": {"m25": cutoff25, "m50": cutoff50},
            "published": {"m25": fl.published_route2(SHARES["m25"], cfg), "m50": published50}}


def step5_answers(episodes):
    """Step 5's answered sentences and each episode's cached-prefix md5."""
    paths = fo.run_paths(STEP5_NAME)
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    absent = [e for e in episodes if e not in timing["episodes"]]
    if absent:
        raise SystemExit(f"step 5 run missing {absent}")
    answers = {e: {} for e in episodes}
    for row in report_mod.read_jsonl(paths["decisions"]):
        if row["episode"] in answers and row["opus_answered"]:
            answers[row["episode"]][row["id"]] = {"score": row["opus_score"], "decision": row["opus_decision"],
                                                   "reason": row["opus_reason"], "group": row["group"]}
    prefix_md5, system_md5 = {}, {}
    for row in report_mod.read_jsonl(paths["requests"]):
        if row["episode"] in answers and not row["error"]:
            prefix_md5.setdefault(row["episode"], set()).add(row["prefix_md5"])
            system_md5.setdefault(row["episode"], set()).add(row["system_md5"])
    return answers, prefix_md5, system_md5, timing, paths


def plan(st, rules, episodes):
    """Jobs for the sentences step 5 did not answer; the rest are reused."""
    inputs, routed50 = st["inputs"], st["routed"]["m50"]
    old, old_prefix, old_system, _t, _p = step5_answers(inputs["episodes"])
    jobs_by_episode, metas = {}, {}
    for e in episodes:
        reuse = {sid for sid in routed50[e] if sid in old[e]}
        to_ask = {e: routed50[e] - reuse}
        jobs, meta = fo.build_jobs(e, inputs, to_ask, rules)
        # the reused answers saw this exact system prompt and cached prefix
        full_jobs, _m = fo.build_jobs(e, inputs, {e: routed50[e]}, rules)
        if reuse:
            if old_prefix.get(e) != {full_jobs[0]["prefix_md5"]}:
                raise SystemExit(f"{e}: step 5's prefix md5 {old_prefix.get(e)} is not this run's")
            if old_system.get(e) != {md5_text(rules)}:
                raise SystemExit(f"{e}: step 5's system prompt md5 differs from the rules read now")
        losers = set(meta["vetoed_ids"])
        _d, _l, all_losers = hl.episode_context(e, inputs)
        if reuse & all_losers:
            raise SystemExit(f"{e}: step 5 answered a sentence the retake pass cut")
        meta = {**meta, "routed": len(routed50[e]), "routed_25": len(st["routed"]["m25"][e]),
                "reused": len(reuse), "reused_ids": sorted(reuse),
                "retake_vetoed": len(routed50[e] & all_losers), "vetoed_ids": sorted(routed50[e] & all_losers),
                "asked_now": meta["asked"], "asked": meta["asked"] + len(reuse),
                "fresh_groups": len(full_jobs), "fresh_targets": sum(len(j["ids"]) for j in full_jobs)}
        assert not losers - all_losers
        jobs_by_episode[e], metas[e] = jobs, meta
    return jobs_by_episode, metas, old


def decision_rows(episode, st, answers, old, jobs, meta):
    """One row per sentence: f3's decision, the Opus verdict (reused or new), the mix."""
    inputs = st["inputs"]
    base, raw = inputs["jev"][episode], inputs["jev_rows"][episode]
    margin = fl.margin_of(inputs["combiner_threshold"])
    r50, r25 = st["routed"]["m50"][episode], st["routed"]["m25"][episode]
    group_of = {sid: job["group"] for job in jobs for sid in job["ids"]}
    reused = set(meta["reused_ids"])
    rows = []
    for sid in sorted(base):
        jev = base[sid]
        is_routed = sid in r50
        source = "reused" if sid in reused else ("new" if sid in group_of else None)
        answer = old[episode].get(sid) if source == "reused" else answers.get(sid)
        row = {"arm": ARM, "episode": episode, "id": sid, "routed": is_routed, "routed_25": sid in r25,
               "source": source, "asked": source is not None, "group": group_of.get(sid),
               "step5_group": old[episode][sid]["group"] if source == "reused" else None,
               "retake_vetoed": is_routed and sid in meta["vetoed_ids"],
               "jev_score": raw[sid]["score"], "jev_margin": margin(raw[sid]),
               "jev_keep": jev["score"] is not None and jev["score"] >= THRESHOLD,
               "jev_keep_words": jev["keep_words"], "cut_retake": jev["cut_retake"],
               "opus_score": None, "opus_decision": None, "opus_reason": None,
               "opus_answered": False, "fallback": source == "new" and not answer,
               "score": jev["score"], "keep_words": jev["keep_words"],
               "model": MODEL, "effort": EFFORT, "preamble_version": hl.PREAMBLE_VERSION}
        if is_routed and answer:
            row.update(opus_score=answer["score"], opus_decision=answer["decision"],
                       opus_reason=answer["reason"], opus_answered=True,
                       score=5.0 if answer["decision"] == "keep" else 0.0, keep_words=None)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# plan and run
# ---------------------------------------------------------------------------

def execute(args, parser):
    st = setup()
    inputs = st["inputs"]
    rules = hl.read_rules()
    paths = run_paths()
    episodes = args.episodes or inputs["episodes"]
    unknown = [e for e in episodes if e not in inputs["episodes"]]
    if unknown:
        parser.error(f"not in the ladder: {unknown}")
    jobs_by_episode, metas, old = plan(st, rules, episodes)
    per_episode_est, total_est = fo.estimate(jobs_by_episode)
    plan_extra = {"selection": "f3 combiner margin abs(5 * p_keep - threshold), pooled, one cutoff",
                  "combiner": {"prefix": COMBINER, "set": st["cp"]["chosen"], "C": st["cp"]["C"],
                               "threshold": st["cp"]["threshold"]},
                  "offline_luna_ceiling_25": st["check"]["offline_ceiling_here"],
                  "reuse": f"step 5 answers from {STEP5_NAME}, same model, prompt and prefix md5",
                  "caching": "system breakpoint (router) + caller breakpoint on preamble+transcript; targets uncached"}
    strip = ("vetoed_ids", "reused_ids")
    out = {"mode": "plan" if not args.run else "run", "model": MODEL, "effort": EFFORT,
           "share": SHARES[MAIN], "cutoff": st["cutoff"][MAIN],
           "routed_total": sum(len(v) for v in st["routed"][MAIN].values()),
           "reused_total": sum(metas[e]["reused"] for e in episodes),
           "asked_now_total": sum(metas[e]["asked_now"] for e in episodes),
           "vetoed_total": sum(metas[e]["retake_vetoed"] for e in episodes),
           "out": NAME, "arm": ARM, "episodes": episodes, "group_max": fo.GROUP_MAX,
           "group_span": fo.GROUP_SPAN, "concurrency": args.concurrency,
           "rules": {"path": str(hl.RULES_PATH), "md5": md5_text(rules)},
           "budget_cap_usd": args.budget,
           "per_episode": {e: {k: v for k, v in {**metas[e], **per_episode_est[e]}.items() if k not in strip}
                           for e in episodes},
           "totals": total_est, "outputs": [str(p) for p in paths.values()], **plan_extra}
    print(json.dumps(out, indent=2), flush=True)
    if not args.run:
        print("Dry run. Re-run with --run to make the calls.", flush=True)
        return 0

    existing = [str(p) for p in paths.values() if p.exists()]
    done, old_decisions, old_timing = set(), [], None
    if existing and not args.resume:
        parser.error(f"refusing to overwrite existing output(s): {existing}; pass --resume")
    if args.resume and paths["timing"].exists():
        old_timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
        done = set(old_timing.get("episodes", {}))
        old_decisions = [r for r in report_mod.read_jsonl(paths["decisions"]) if r["episode"] in done]
        log(f"resume: skipping {len(done & set(episodes))} finished episode(s)")

    fo.hydrate_anthropic_key()
    import skell_e_router  # noqa: F401  (import before the clock starts)

    budget = pipeline.Budget(args.budget)
    if old_timing:
        budget.add(old_timing["totals"]["cost_usd"])
    timings = dict((old_timing or {}).get("episodes", {}))
    new_decisions, aborted = [], None
    started = time.perf_counter()
    with JsonlWriter(paths["requests"], overwrite_ok=args.resume) as writer:
        for episode in episodes:
            if episode in done:
                continue
            jobs = jobs_by_episode[episode]
            answers, timing = fo.run_episode_jobs(jobs, rules, budget, args.concurrency, writer)
            rows = decision_rows(episode, st, answers, old, jobs, metas[episode])
            new_decisions.extend(rows)
            timings[episode] = {**{k: v for k, v in metas[episode].items() if k not in strip}, **timing,
                                "substituted": sum(1 for r in rows if r["opus_answered"]),
                                "substituted_new": sum(1 for r in rows if r["opus_answered"] and r["source"] == "new"),
                                "fallbacks": sum(1 for r in rows if r["fallback"])}
            share = timing["cache_read_tokens"] / max(1, timing["prompt_tokens"])
            log(f"{episode}: {timing['wall_clock_s']}s, ${timing['cost_usd']:.4f}, {timing['requests']} requests, "
                f"{metas[episode]['reused']} reused, {metas[episode]['asked_now']} asked now, "
                f"{timing['retries']} retries, {timing['errors']} errors, {timing['malformed']} malformed, "
                f"{timing['refusals']} refusals, {timing['unanswered_targets']} fallback targets, "
                f"cache read {share * 100:.0f}% of input; spent ${budget.spent:.4f}")
            if budget.blocked():
                aborted = f"budget cap ${args.budget:.2f} crossed at ${budget.spent:.4f}; stopped after {episode}"
                log(f"ABORT: {aborted}")
                break
            write_run_files(paths, old_decisions + new_decisions, timings, out, args, aborted, old_timing,
                            started, plan_extra)
    write_run_files(paths, old_decisions + new_decisions, timings, out, args, aborted, old_timing,
                    started, plan_extra)
    print(json.dumps(json.loads(paths["timing"].read_text(encoding="utf-8"))["totals"], indent=2))
    return 2 if aborted else 0


def write_run_files(paths, decisions, timings, plan_doc, args, aborted, old_timing, started, plan_extra):
    keys = ("requests", "retries", "errors", "malformed", "refusals", "unanswered_targets", "fallbacks",
            "prompt_tokens", "uncached_input_tokens", "cache_write_tokens", "cache_read_tokens",
            "completion_tokens", "routed", "routed_25", "asked", "asked_now", "reused", "retake_vetoed",
            "substituted", "substituted_new", "groups", "fresh_groups", "fresh_targets")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "effort": EFFORT, "share": SHARES[MAIN], "cutoff": plan_doc["cutoff"], "arm": ARM,
        "out": NAME, "preamble_version": hl.PREAMBLE_VERSION, "rules": plan_doc["rules"],
        "group_max": fo.GROUP_MAX, "group_span": fo.GROUP_SPAN, "concurrency": args.concurrency,
        "max_tokens": fo.MAX_TOKENS, "budget_cap_usd": args.budget, "aborted": aborted,
        "resumed": bool(old_timing), "episodes": timings,
        "totals": {
            "wall_clock_s": round(time.perf_counter() - started
                                  + ((old_timing or {}).get("totals", {}).get("wall_clock_s", 0.0)), 3),
            "episodes": len(timings),
            **{key: sum(t.get(key, 0) for t in timings.values()) for key in keys},
            "cost_usd": round(sum(t["cost_usd"] for t in timings.values()), 6),
            "cost_listed_usd": round(sum(t["cost_listed_usd"] for t in timings.values()), 6),
        },
        "estimate": (old_timing or {}).get("estimate") or plan_doc["totals"],
        "estimate_episodes": (old_timing or {}).get("estimate_episodes") or plan_doc["episodes"],
        **plan_extra,
    }
    paths["decisions"].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions),
                                  encoding="utf-8")
    paths["timing"].write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# $0 modes: ceiling, extrapolation, keep rule
# ---------------------------------------------------------------------------

def ceiling_mode():
    st = setup()
    inputs = st["inputs"]
    donors = fo.load_donors(inputs["episodes"])[0]
    out = {"routed": {k: sum(len(v) for v in r.values()) for k, r in st["routed"].items()},
           "cutoff": st["cutoff"]}
    for tag, routed in st["routed"].items():
        block = fo.ceiling_block(inputs, routed, donors)
        out[tag] = {k: {s: v[s]["sentence_points"] for s in ("fit", "heldout", "all")} for k, v in block.items()}
    print(json.dumps(out, indent=2))
    return 0


def extrapolate_mode():
    """Task total from the finished episodes' real usage: listed prices on the planned asks."""
    paths = run_paths()
    if not paths["timing"].exists():
        raise SystemExit("no run on disk; probe first")
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    requests = report_mod.read_jsonl(paths["requests"])
    done = list(timing["episodes"])
    st = setup()
    rules = hl.read_rules()
    jobs_by_episode, metas, _old = plan(st, rules, st["inputs"]["episodes"])
    firsts = [r for r in requests if r["phase"] == "first" and r["episode"] in done and not r["error"]]
    chars = sum(jobs_by_episode[r["episode"]][0]["prefix_chars"] for r in firsts)
    written = sum((r["cache_write_tokens"] or 0) + (r["cache_read_tokens"] or 0)
                  + (r["uncached_input_tokens"] or 0) for r in firsts)
    tokens_per_char = written / chars
    ok = [r for r in requests if r["episode"] in done and not r["error"]]
    per_target = sum(r["completion_tokens"] or 0 for r in ok) / sum(r["n_targets"] for r in ok)
    per_request = sum(r["completion_tokens"] or 0 for r in ok) / len(ok)
    per_episode, total = {}, 0.0
    for e, jobs in jobs_by_episode.items():
        if e in done:
            cost = timing["episodes"][e]["cost_usd"]
        elif not jobs:
            cost = 0.0
        else:
            prefix = jobs[0]["prefix_chars"] * tokens_per_char
            n_targets = sum(len(j["ids"]) for j in jobs)
            out = max(per_target * n_targets, per_request * len(jobs))
            uncached = sum(j["target_chars"] for j in jobs) * tokens_per_char
            cost = (prefix * PRICE_WRITE + prefix * (len(jobs) - 1) * PRICE_READ
                    + uncached * PRICE_IN + out * PRICE_OUT) / 1e6
        per_episode[e] = round(cost, 4)
        total += cost
    retry_factor = len([r for r in requests if r["episode"] in done]) / max(
        1, sum(timing["episodes"][e]["groups"] for e in done))
    projected = total * max(1.0, retry_factor)
    out = {"probe_episodes": done, "probe_cost_usd": timing["totals"]["cost_usd"],
           "tokens_per_char": tokens_per_char, "completion_per_target": per_target,
           "completion_per_request": per_request, "retry_factor": retry_factor,
           "per_episode_usd": per_episode, "projected_18_usd": round(projected, 4),
           "stop_threshold_usd": STOP_USD, "go": projected <= STOP_USD,
           "asked_now_total": sum(m["asked_now"] for m in metas.values()),
           "reused_total": sum(m["reused"] for m in metas.values())}
    print(json.dumps(out, indent=2))
    return 0 if out["go"] else 3


def load_run(episodes, require_all=True):
    """The run's rows as two views, 50% and f3's 25%, aliased to ``hl``'s ``luna_*`` names."""
    paths = run_paths()
    if not all(p.exists() for p in paths.values()):
        return None
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    absent = [e for e in episodes if e not in timing["episodes"]]
    if absent and require_all:
        raise SystemExit(f"run not finished, missing {absent}")
    base = {e: {} for e in episodes}
    for row in report_mod.read_jsonl(paths["decisions"]):
        if row["episode"] in base:
            base[row["episode"]][row["id"]] = {**row, "luna_score": row["opus_score"],
                                               "luna_decision": row["opus_decision"],
                                               "luna_answered": row["opus_answered"]}
    requests = report_mod.read_jsonl(paths["requests"])
    views = {}
    for tag in SHARES:
        flag = "routed" if tag == "m50" else "routed_25"
        rows = {e: {sid: {**r, "routed": r[flag]} for sid, r in base[e].items()} for e in episodes}
        views[tag] = {"tag": f"f3opus-{tag}", "rows": rows,
                      "routed": {e: {sid for sid, r in rows[e].items() if r["routed"]} for e in episodes},
                      "answered": {e: {sid for sid, r in rows[e].items() if r["routed"] and r["opus_answered"]}
                                   for e in episodes},
                      "timing": timing, "requests": requests, "paths": paths}
    return views


def choose_rules(scorer, st, views, episodes):
    fit = [e for e in episodes if e in FIT]
    out = {}
    for tag, view in views.items():
        sp = {rule: r2.pool(scorer.episode(hl.donor_key(view["tag"], rule), e, st["routed"][tag][e])
                            for e in fit)["sentence_points"] for rule in hl.KEEP_RULES}
        chosen = max(hl.KEEP_RULES, key=lambda rule: (round(sp[rule], 6), -hl.KEEP_RULES.index(rule)))
        out[tag] = {"rule": chosen, "fit_six_sentence_points": sp}
    return out


def keep_rule_mode(args, parser):
    """Choose each share's keep rule on the fit six only and freeze them. Reads no held-out rows."""
    if KEEP_RULE_PATH.exists() and not args.force:
        parser.error(f"refusing to overwrite {KEEP_RULE_PATH}")
    st = setup()
    inputs = st["inputs"]
    fit = list(FIT)
    views = load_run(fit)
    donors = {}
    for tag, view in views.items():
        for e in fit:
            if view["routed"][e] != st["routed"][tag][e]:
                raise SystemExit(f"{e}: the run's {tag} slice is not the slice selected now")
        hl.register_live_donors(donors, view, fit)
    scorer = r2.Scorer({e: inputs["jev"][e] for e in fit}, donors, {e: inputs["removals"][e] for e in fit})
    chosen = choose_rules(scorer, st, views, fit)
    frozen = {"rules": chosen, "headline": MAIN,
              "chosen_on": f"fit six of {NAME}, before any held-out call of this run",
              "tie_break": "best fit-six SP, ties to the decision field, then the lower score threshold",
              "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "fit_episodes_on_disk": fit,
              "heldout_episodes_on_disk": [e for e in views[MAIN]["timing"]["episodes"] if e not in fit]}
    KEEP_RULE_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def agreement_by_source(view, archived, episodes, human, removals, source):
    sub = {**view, "answered": {e: {sid for sid in view["answered"][e]
                                    if source is None or view["rows"][e][sid]["source"] == source}
                                for e in episodes}}
    return hl.live_vs_archived(sub, archived, episodes, human, removals)


def build_report():
    st = setup()
    inputs, cp, cfg = st["inputs"], st["cp"], st["cfg"]
    episodes, removals, f3_jev = inputs["episodes"], inputs["removals"], inputs["jev"]
    if not KEEP_RULE_PATH.exists():
        raise SystemExit("no frozen keep rule; run --keep-rule on the fit six first")
    frozen = json.loads(KEEP_RULE_PATH.read_text(encoding="utf-8"))
    views = load_run(episodes)
    for tag, view in views.items():
        for e in episodes:
            if view["routed"][e] != st["routed"][tag][e]:
                raise SystemExit(f"{e}: the run's {tag} slice is not the slice selected now")
    step5 = json.loads(STEP5_JSON.read_text(encoding="utf-8"))
    s5_rule = step5["frozen_keep_rule"]["rule"]
    f1_doc = json.loads(F1_JSON.read_text(encoding="utf-8"))
    luna_doc = json.loads((OUT_DIR / f"{fl.OUT_STEM}.json").read_text(encoding="utf-8"))
    pub25 = st["published"]["m25"]
    _o, _p, _s, s5_timing, s5_paths = step5_answers(episodes)

    log("donors and scoring...")
    donors, donor_paths = fo.load_donors(episodes)
    for view in views.values():
        hl.register_live_donors(donors, view, episodes)
    scorer = r2.Scorer(f3_jev, donors, removals)
    f3_eps = {e: scorer.episode(None, e, set()) for e in episodes}
    f3_pooled = r2.splits(f3_eps, episodes)

    recomputed = choose_rules(scorer, st, views, episodes)
    for tag in SHARES:
        if recomputed[tag]["rule"] != frozen["rules"][tag]["rule"]:
            raise SystemExit(f"{tag}: fit-six choice recomputed as {recomputed[tag]['rule']}, "
                             f"frozen file says {frozen['rules'][tag]['rule']}")
    rules = {tag: frozen["rules"][tag]["rule"] for tag in SHARES}
    by_rule, per_ep, ceil = {}, {}, {}
    for tag, view in views.items():
        routed = st["routed"][tag]
        by_rule[tag] = {kr: r2.splits({e: scorer.episode(hl.donor_key(view["tag"], kr), e, routed[e])
                                       for e in episodes}, episodes) for kr in hl.KEEP_RULES}
        per_ep[tag] = {kr: {e: scorer.episode(hl.donor_key(view["tag"], kr), e, routed[e]) for e in episodes}
                       for kr in ("decision", rules[tag])}
        ceil[tag] = {key: r2.splits({e: scorer.episode(key, e, routed[e]) for e in episodes}, episodes)
                     for key in ("opus", "opus_notrim", "luna")}
    reproduction = {
        "f3_combiner": {"mine": f3_pooled["all"]["sentence_points"], "published": pub25["ladder_sp"]},
        "luna_ceiling_25": {"mine": ceil["m25"]["luna"]["all"]["sentence_points"],
                            "published": pub25["sentence_points"]},
        "luna_ceiling_50": {"mine": ceil["m50"]["luna"]["all"]["sentence_points"],
                            "published": st["published"]["m50"]["sentence_points"]},
    }
    for key, pair in reproduction.items():
        if abs(pair["mine"] - pair["published"]) > 5e-4:
            raise SystemExit(f"reproduction failed for {key}: {pair}")

    f1_ladder = f1_doc["ladder"].get("mine_sp", f1_doc["ladder"].get("f1_sp"))
    ladder = r2.ladder_rows(episodes, luna_doc["v3"]["all"]["sentence_points"])
    ladder += [
        {"label": f"Jev f3 `{cp['chosen']}` combiner (winning Jev-only run)", "key": "f3",
         "sentence_points": f3_pooled["all"]["sentence_points"]},
        {"label": "Jev f1 `q+code+v3` combiner (reference)", "key": "f1", "sentence_points": f1_ladder},
        {"label": "f1-Opus stack 25%, Opus `decision` (step 5)", "key": "f1opus_decision",
         "sentence_points": step5["by_rule"]["decision"]["all"]["sentence_points"]},
        {"label": f"f1-Opus stack 25%, `{s5_rule}` (step 5)", "key": "f1opus_rule",
         "sentence_points": step5["by_rule"][s5_rule]["all"]["sentence_points"]},
        {"label": f"f1-Luna stack 25%, `{luna_doc['frozen_keep_rule']['rule']}` (step 2)", "key": "f1luna_rule",
         "sentence_points": luna_doc["by_rule"][luna_doc["frozen_keep_rule"]["rule"]]["all"]["sentence_points"]},
    ]
    ladder.sort(key=lambda r: -r["sentence_points"])
    placement = {f"{tag}:{kr}": r2.placement(by_rule[tag][kr]["all"]["sentence_points"], ladder)
                 for tag in SHARES for kr in ("decision", rules[tag])}
    shipped = next(r for r in ladder if r["key"] == "opus5-cc-agentic" or r["label"] == "shipped Opus agentic")

    log("states, flips, agreement...")
    human, f3_states = {}, {}
    for e in episodes:
        human[e], f3_states[e] = r2.states(e, f3_jev[e], removals[e])
    flips = {f"{tag}:{kr}": r2.flip_block(scorer, hl.donor_key(views[tag]["tag"], kr), st["routed"][tag],
                                          episodes, human, f3_states)
             for tag in SHARES for kr in ("decision", rules[tag])}
    flips["m50:ceiling"] = r2.flip_block(scorer, "opus_notrim", st["routed"]["m50"], episodes, human, f3_states)
    agreement = {src or "all": agreement_by_source(views[MAIN], donors["opus"], episodes, human, removals, src)
                 for src in (None, "reused", "new")}

    log("per episode...")
    t_eps = views[MAIN]["timing"]["episodes"]
    f3_secs = pub25["seconds"]
    s5_eps = s5_timing["episodes"]
    new_cost = views[MAIN]["timing"]["totals"]["cost_usd"]
    new_targets = sum(t_eps[e]["asked_now"] for e in episodes)
    per_sentence = new_cost / max(1, new_targets)
    s5_per_sentence = s5_timing["totals"]["cost_usd"] / max(1, s5_timing["totals"]["asked"])
    per_episode = {}
    for e in episodes:
        t = t_eps[e]
        f3s = f3_secs[e]
        src_usd = f3s["v3_usd"] + sum(v["cost_usd"] for v in f3s["source_runs"].values())
        asked25 = sum(1 for sid in st["routed"]["m25"][e] if views["m25"]["rows"][e][sid]["asked"])
        per_episode[e] = {
            "sentences": len(f3_jev[e]), "routed": t["routed"], "routed_25": t["routed_25"],
            "asked": t["asked"], "asked_25": asked25, "reused": t["reused"], "asked_now": t["asked_now"],
            "groups": t["groups"], "requests": t["requests"], "retries": t["retries"], "errors": t["errors"],
            "malformed": t["malformed"], "refusals": t["refusals"], "fallbacks": t["fallbacks"],
            "f3": f3_eps[e],
            **{f"{tag}:{kr}": per_ep[tag][kr][e] for tag in SHARES for kr in ("decision", rules[tag])},
            "opus_seconds": t["wall_clock_s"], "first_group_s": t["first_group_s"],
            "step5_opus_seconds": s5_eps[e]["wall_clock_s"], "f3_seconds": f3s["total_s"],
            "seconds": t["wall_clock_s"] + f3s["total_s"],
            "seconds_fresh_upper": t["wall_clock_s"] + s5_eps[e]["wall_clock_s"] + f3s["total_s"],
            "seconds_25_est": s5_eps[e]["wall_clock_s"] + f3s["total_s"],
            "cost_usd": t["cost_usd"], "f3_source_usd": src_usd,
            "fresh_cost_usd": per_sentence * t["asked"], "fresh_cost_25_usd": per_sentence * asked25,
            "prompt_tokens": t["prompt_tokens"], "cache_read_tokens": t["cache_read_tokens"],
            "cache_read_share": t["cache_read_tokens"] / max(1, t["prompt_tokens"]),
        }
    totals = views[MAIN]["timing"]["totals"]

    def mean(key):
        return statistics.mean(per_episode[e][key] for e in episodes)

    requests = views[MAIN]["requests"]
    later = [r for r in requests if r["phase"] != "first" and not r["error"]]
    inputs_fp = {k: fingerprint(v) for k, v in inputs["run_paths"].items()}
    inputs_fp["rules_prompt"] = fingerprint(hl.RULES_PATH)
    inputs_fp["ladder_reference"] = fingerprint(report_mod.REFERENCE_JSON)
    inputs_fp["f3_weights"] = fingerprint(cp["weights_path"])
    for split, p in cp["feature_paths"].items():
        inputs_fp[f"f3_{split}_features"] = fingerprint(p["features"])
    inputs_fp["f3_writeup_json"] = fingerprint(cfg.writeup_json)
    inputs_fp["f1_writeup_json"] = fingerprint(F1_JSON)
    inputs_fp["f1luna_writeup_json"] = fingerprint(OUT_DIR / f"{fl.OUT_STEM}.json")
    inputs_fp["step5_writeup_json"] = fingerprint(STEP5_JSON)
    for kind, path in s5_paths.items():
        inputs_fp[f"step5_run:{kind}"] = fingerprint(path)
    for key in ("luna", "opus"):
        for path in donor_paths[key]:
            inputs_fp[f"{key}:{Path(path).name}"] = fingerprint(path)
    for kind, path in views[MAIN]["paths"].items():
        inputs_fp[f"opus_run:{kind}"] = fingerprint(path)
    inputs_fp["opus_keep_rule"] = fingerprint(KEEP_RULE_PATH)
    digest = hashlib.md5()
    for e in episodes:
        digest.update(r2.md5(r2.removals_cache_path(e)).encode())

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_hybrid_f3opus.py",
        "model": MODEL, "effort": EFFORT, "arm": ARM, "episodes": episodes, "fit": FIT,
        "heldout": [e for e in episodes if e not in FIT], "sentences": sum(len(f3_jev[e]) for e in episodes),
        "combiner": {"prefix": COMBINER, "set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"]},
        "shares": SHARES, "cutoff": st["cutoff"],
        "routed": {tag: sum(len(v) for v in st["routed"][tag].values()) for tag in SHARES},
        "counts": {k: totals[k] for k in ("asked", "asked_now", "reused", "retake_vetoed", "substituted",
                                          "substituted_new", "groups", "requests", "retries", "errors",
                                          "malformed", "refusals", "fallbacks", "fresh_groups", "fresh_targets")},
        "counts_25": {"asked": sum(p["asked_25"] for p in per_episode.values()),
                      "reused": sum(1 for e in episodes for sid in st["routed"]["m25"][e]
                                    if views["m25"]["rows"][e][sid]["source"] == "reused")},
        "frozen_keep_rule": frozen, "rules": rules, "keep_rules": hl.KEEP_RULES,
        "reproduction": reproduction, "f3": f3_pooled, "by_rule": by_rule, "ceilings": ceil,
        "step5": {"rule": s5_rule, "decision": step5["by_rule"]["decision"], "rule_block": step5["by_rule"][s5_rule],
                  "seconds_per_episode_mean": step5["seconds_per_episode_mean"],
                  "cost_per_episode_mean": step5["cost_per_episode_mean"],
                  "stack_cost_per_episode_mean": step5["stack_cost_per_episode_mean"],
                  "cost_usd": step5["cost_usd"], "per_sentence_usd": s5_per_sentence},
        "f1_alone": {"ladder_sp": f1_ladder, "block": step5["f1"],
                     "seconds_per_episode_mean": step5["f1_seconds_per_episode_mean"]},
        "shipped": shipped, "ladder": ladder, "placement": placement, "flips": flips, "agreement": agreement,
        "per_episode": per_episode,
        "cost_usd": new_cost, "cost_listed_usd": totals["cost_listed_usd"],
        "per_sentence_usd": per_sentence,
        "cost_per_episode_mean": mean("cost_usd"), "cost_per_episode_max": max(p["cost_usd"] for p in per_episode.values()),
        "fresh_cost_per_episode_mean": mean("fresh_cost_usd"), "fresh_cost_25_per_episode_mean": mean("fresh_cost_25_usd"),
        "fresh_cost_total": sum(p["fresh_cost_usd"] for p in per_episode.values()),
        "f3_source_cost_per_episode_mean": mean("f3_source_usd"),
        "seconds_per_episode_mean": mean("seconds"), "seconds_per_episode_max": max(p["seconds"] for p in per_episode.values()),
        "seconds_fresh_upper_mean": mean("seconds_fresh_upper"), "seconds_25_est_mean": mean("seconds_25_est"),
        "opus_seconds_per_episode_mean": mean("opus_seconds"), "f3_seconds_per_episode_mean": mean("f3_seconds"),
        "tokens": {k: totals[k] for k in ("prompt_tokens", "uncached_input_tokens", "cache_write_tokens",
                                          "cache_read_tokens", "completion_tokens")},
        "cache": {"read_share": totals["cache_read_tokens"] / max(1, totals["prompt_tokens"]),
                  "later_requests": len(later),
                  "later_requests_with_cache_read": sum(1 for r in later if (r["cache_read_tokens"] or 0) > 0)},
        "estimate": fo.estimate(plan(st, hl.read_rules(), episodes)[0])[1], "wall_clock_s": totals["wall_clock_s"],
        "run_generated_utc": views[MAIN]["timing"]["generated_utc"],
        "settings": {"group_max": fo.GROUP_MAX, "group_span": fo.GROUP_SPAN,
                     "concurrency": views[MAIN]["timing"]["concurrency"], "max_attempts": fo.MAX_ATTEMPTS,
                     "malformed_reasks": fo.MALFORMED_REASKS, "max_tokens": fo.MAX_TOKENS,
                     "preamble_version": hl.PREAMBLE_VERSION, "threshold": THRESHOLD},
        "inputs": inputs_fp, "removals_digest": digest.hexdigest(),
    }


def write_markdown(path, s, json_path):
    lines = []
    add = lines.append
    r50, r25 = s["rules"]["m50"], s["rules"]["m25"]

    def rule_pair(rule):
        return ["decision"] if rule == "decision" else [rule, "decision"]

    def rule_label(kr, rule):
        if kr == rule:
            return f"fit-chosen `{rule}`" + (" (the decision field)" if rule == "decision" else "")
        return "Opus `decision`"
    b50, b25 = s["by_rule"]["m50"], s["by_rule"]["m25"]
    cnt, c25, cache = s["counts"], s["counts_25"], s["cache"]
    s5 = s["step5"]
    add("Developer-facing notes on the f3-Opus stack: the f3 combiner decides, claude-opus-5 overrides its unsure 50% and 25%.")
    add("")
    add(f"Generated {s['generated_utc']} by `{s['script']}` from the run files on disk, the f3 feature files and frozen weights, step 5's Opus run (whose answers are reused), the archived donor ratings and the cached removal ranges. The report step makes no model calls. Every metric is x100, two decimals, with um removal and delete silence layered on (the ladder column). The JSON next to this file keeps the raw values.")
    add("")
    add("# The f3-Opus stack")
    add("")
    add(f"Stan picked f3 (`{s['combiner']['set']}`, keep threshold {s['combiner']['threshold']:.2f} on `5 * p_keep`) as the winning Jev-only combiner on its held-out result, so this run puts Opus on top of it. f3 decides all {s['sentences']:,} sentences of the 18 ladder episodes; the {s['routed']['m50']:,} whose margin `abs(5 * p_keep - threshold)` sits at or under {s['cutoff']['m50']:.3f} (the pooled bottom 50%, one global cutoff) go to `{s['model']}` with step 5's call unchanged: rules5 system prompt, preamble plus whole transcript in the cached prefix, targets uncached, groups of up to {s['settings']['group_max']}, {s['effort']} effort. f3's own bottom 25% ({s['routed']['m25']:,} sentences, cutoff {s['cutoff']['m25']:.3f}) sits inside that slice, so the same answers give the 25% point too. The fit six carry f3's leave-one-out `p_keep`, the held-out 12 its frozen weights, as in the f1 stack.")
    add("")
    add(f"Cost saver: {cnt['reused']:,} of the {cnt['asked']:,} sentences asked for the 50% slice already had a step 5 answer (same model, same system prompt and same cached prefix, both checked by md5), so only {cnt['asked_now']:,} were sent. The reused answers came from step 5's groups, which held a different mix of targets than a fresh 50% run would have, so a reused answer is what Opus said in a different group. For the 25% point, {c25['reused']:,} of its {c25['asked']:,} asked sentences are reused and the rest come from this run. Keep rules for both shares were chosen on the fit six alone and frozen to `{Path(s['inputs']['opus_keep_rule']['path']).name}` at {s['frozen_keep_rule']['frozen_utc']}, before any held-out call of this run: `{r50}` at 50%, `{r25}` at 25%.")
    add("")
    f3 = s["f3"]
    add(f"Bottom line: f3 plus Opus at 50% with `{r50}` scores {pct(b50[r50]['all']['sentence_points'])} SP pooled 18 ({pct(b50[r50]['heldout']['sentence_points'])} held-out 12), and at 25% with `{r25}` {pct(b25[r25]['all']['sentence_points'])} ({pct(b25[r25]['heldout']['sentence_points'])} held out), against {pct(s5['rule_block']['all']['sentence_points'])} for the f1-Opus 25% stack, {pct(f3['all']['sentence_points'])} for f3 alone and {pct(s['shipped']['sentence_points'])} for shipped Opus agentic.")
    add("")

    add("## Offline ceiling on the f3 slices")
    add("")
    add("Before any call: the archived Opus agentic ratings substituted on the same slices ($0). The archived Opus carries partial keeps the live call does not ask for, so the keep/cut-only row is the like-for-like bound. No live calls behind these rows, so their seconds column is n/a.")
    add("")
    rows = [["f3 combiner alone (winning Jev-only run)", "", pct(f3["fit"]["sentence_points"]), pct(f3["heldout"]["sentence_points"]),
             pct(f3["all"]["sentence_points"]), pct(f3["all"]["word_score"]), pct(f3["all"]["grade"]),
             num(s["f3_seconds_per_episode_mean"], 1)]]
    for tag in ("m25", "m50"):
        for label, key in (("archived Opus, keep/cut only", "opus_notrim"), ("archived Opus with trims", "opus"),
                           ("archived Luna chapters", "luna")):
            p = s["ceilings"][tag][key]
            rows.append([label, f"{s['shares'][tag] * 100:g}%", pct(p["fit"]["sentence_points"]),
                         pct(p["heldout"]["sentence_points"]), pct(p["all"]["sentence_points"]),
                         pct(p["all"]["word_score"]), pct(p["all"]["grade"]), "n/a"])
    lines.extend(table(["substitution on the slice", "share", "SP fit 6", "SP held-out 12", "SP all 18",
                        "WORD all 18", "GRADE all 18", "s/ep mean"], rows))
    add("")

    add("## Results")
    add("")
    add(f"Seconds per episode: the 50% rows are this run's Opus wall clock (new asks only, first group alone then up to {s['settings']['concurrency']} in flight) plus f3's own per-episode time; the 25% rows use step 5's Opus wall clock on a slice of the same size plus f3's time, an estimate. Model $/ep is what this run paid for its new asks; fresh $/ep is what a run asking every sentence of the slice would have cost at this run's per-sentence cost (${s['per_sentence_usd']:.5f}). f1 rows are for reference.")
    add("")
    rows = []

    def row(label, p, secs, paid, fresh):
        rows.append([label, pct(p["fit"]["sentence_points"]), pct(p["heldout"]["sentence_points"]),
                     pct(p["all"]["sentence_points"]), pct(p["all"]["word_score"]), pct(p["all"]["grade"]),
                     num(secs, 1), money(paid), money(fresh)])

    for share, rule, block, secs, paid, fresh in (
            ("50%", r50, b50, s["seconds_per_episode_mean"], s["cost_per_episode_mean"], s["fresh_cost_per_episode_mean"]),
            ("25%", r25, b25, s["seconds_25_est_mean"], None, s["fresh_cost_25_per_episode_mean"])):
        for kr in rule_pair(rule):
            row(f"f3 + Opus {share}, {rule_label(kr, rule)}", block[kr], secs, paid, fresh)
    row(f"f1 + Opus 25%, `{s5['rule']}` (step 5, reference)", s5["rule_block"], s5["seconds_per_episode_mean"], s5["cost_per_episode_mean"], s5["cost_per_episode_mean"])
    row("f1 + Opus 25%, Opus `decision` (step 5, reference)", s5["decision"], s5["seconds_per_episode_mean"], s5["cost_per_episode_mean"], s5["cost_per_episode_mean"])
    row("f3 alone (winning Jev-only run)", f3, s["f3_seconds_per_episode_mean"], 0.0, 0.0)
    row("f1 alone (reference)", s["f1_alone"]["block"], s["f1_alone"]["seconds_per_episode_mean"], 0.0, 0.0)
    rows.append(["shipped Opus agentic (ladder)", "", "", pct(s["shipped"]["sentence_points"]), "", "", "", "", ""])
    lines.extend(table(["arm", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18", "GRADE all 18",
                        "s/ep mean", "model $/ep paid", "model $/ep fresh"], rows))
    add("")
    add("f3 and f1 alone score the fit six with their leave-one-out predictions and the held-out 12 with their frozen weights, the same split as the stacks. Shipped Opus agentic has only its ladder number here; `docs/TASKS.md` puts it at about $5 and an hour per episode. The f3 combiner's $0 is new spend only; its rows are a join of Jev runs already paid for.")
    add("")
    ladder_text = ", ".join(f"{r['label']} {pct(r['sentence_points'])}" for r in s["ladder"])
    pl = s["placement"]
    places = "; ".join(f"{share} `{kr}` {pl[f'{tag}:{kr}']['text']} (rank {pl[f'{tag}:{kr}']['rank']} of {pl[f'{tag}:{kr}']['of']})"
                       for tag, share, rule in (("m50", "50%", r50), ("m25", "25%", r25)) for kr in rule_pair(rule))
    add(f"Ladder, with modules, same 18 episodes: {ladder_text}. Placement: {places}.")
    add("")

    add("## Every keep rule")
    add("")
    add("Chosen on the fit six alone (best fit-six SP, ties to the decision field, then the lower threshold). The other rows show the shape of the curve.")
    add("")
    rows = []
    for tag, rule in (("m50", r50), ("m25", r25)):
        secs = s["seconds_per_episode_mean"] if tag == "m50" else s["seconds_25_est_mean"]
        for kr in s["keep_rules"]:
            p = s["by_rule"][tag][kr]
            rows.append([f"{s['shares'][tag] * 100:g}%", ("* " if kr == rule else "") + kr,
                         pct(p["fit"]["sentence_points"]), pct(p["heldout"]["sentence_points"]),
                         pct(p["all"]["sentence_points"]), pct(p["all"]["word_score"]), pct(p["all"]["grade"]),
                         num(secs, 1)])
    lines.extend(table(["share", "keep rule", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18",
                        "GRADE all 18", "s/ep mean"], rows))
    add("")

    add("## Live Opus against archived Opus")
    add("")
    add("Keep/cut on the 50% slice's answered sentences, split by where the answer came from. The archived Opus ran agentically on rules1 at high effort over whole episodes, so this is a different prompt as well as a different session.")
    add("")
    rows = []
    for key, label in (("all", "all answered"), ("reused", "reused from step 5"), ("new", "asked in this run")):
        a = s["agreement"][key]
        rows.append([label, a.get("n", 0), pct(a["agree_rate"]), pct(a["live_right_rate"]), pct(a["archived_right_rate"]),
                     f"{a.get('disagree_live_right', 0)} / {a.get('disagree_archived_right', 0)} of {a.get('disagree', 0)}",
                     pct(a["live_keep_rate"]), pct(a["archived_keep_rate"]), pct(a["editor_keep_rate"]),
                     pct(a["score_within_1_rate"]), num(s["seconds_per_episode_mean"], 1)])
    lines.extend(table(["answers", "n", "agree", "live right", "archived right", "disagreements live / archived right",
                        "live keep rate", "archived keep rate", "editor keep rate", "score within 1",
                        "s/ep mean"], rows))
    add("")

    add("## Where the gains come from")
    add("")
    add("A flip is a routed sentence whose keep/cut changed when Opus's verdict replaced f3's, read off the scoring module's sentence states with the modules layered. Right means the new state matches the editor (kept means full or partial).")
    add("")
    rows = []
    flip_rows = [(f"{tag}:{kr}", f"{share} `{kr}`", secs)
                 for tag, share, rule, secs in (("m50", "50%", r50, s["seconds_per_episode_mean"]),
                                                ("m25", "25%", r25, s["seconds_25_est_mean"]))
                 for kr in rule_pair(rule)]
    for key, label, secs in flip_rows + [("m50:ceiling", "50% archived Opus keep/cut only (ceiling)", None)]:
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

    add(f"## Per episode (50% cutoff {s['cutoff']['m50']:.3f}, 25% cutoff {s['cutoff']['m25']:.3f})")
    add("")
    rows = []
    for e in s["episodes"]:
        p = s["per_episode"][e]
        rows.append([e + (" (fit)" if e in s["fit"] else ""), p["sentences"], p["routed"], p["reused"], p["asked_now"],
                     pct(p["f3"]["sentence_points"]), pct(p[f"m25:{r25}"]["sentence_points"]),
                     pct(p[f"m50:{r50}"]["sentence_points"]),
                     p["requests"], p["retries"], p["fallbacks"], num(p["opus_seconds"], 1), num(p["f3_seconds"], 1),
                     num(p["seconds"], 1), money(p["cost_usd"]), money(p["fresh_cost_usd"]),
                     f"{p['cache_read_share'] * 100:.0f}%"])
    lines.extend(table(["episode", "sentences", "routed 50%", "reused", "asked now", "SP f3",
                        f"SP 25% {r25}", f"SP 50% {r50}", "requests", "retries",
                        "fallback targets", "Opus s", "f3 s", "s/ep", "Opus $ paid", "Opus $ fresh 50%",
                        "cache read share"], rows))
    add("")

    add("## Cost, caching and failures")
    add("")
    t = s["tokens"]
    add(f"{cnt['requests']} requests over {cnt['groups']} groups, {cnt['retries']} retries, {cnt['errors']} errored attempts, {cnt['malformed']} malformed or incomplete answers, {cnt['refusals']} refusals, {cnt['fallbacks']} targets left on f3's decision after the re-ask. Routed sentences the v3 retake pass had already cut were not sent: {cnt['retake_vetoed']} of {s['routed']['m50']:,}. A fresh 50% run would have sent {cnt['fresh_targets']:,} targets in {cnt['fresh_groups']} groups.")
    add("")
    add(f"This run paid ${s['cost_usd']:.4f} by the router's accounting (${s['cost_listed_usd']:.4f} at list rates) for {cnt['asked_now']:,} new sentences, ${s['per_sentence_usd']:.5f} per sentence against step 5's ${s5['per_sentence_usd']:.5f}. Per episode ${s['cost_per_episode_mean']:.4f} mean, ${s['cost_per_episode_max']:.4f} max. A fresh 50% run at this run's per-sentence cost: ${s['fresh_cost_total']:.2f} in all, ${s['fresh_cost_per_episode_mean']:.4f} per episode; a fresh 25% run ${s['fresh_cost_25_per_episode_mean']:.4f} per episode. The f3 combiner's own Jev spend was $0 new (a join of paid runs; its source passes cost ${s['f3_source_cost_per_episode_mean']:.4f} per episode). The plan-mode estimate for the new asks over all 18, before the probe, was ${s['estimate']['cost_usd']:.4f} (4 characters per token, 2,000 output tokens per request plus 60 per target).")
    add("")
    add(f"Input tokens {t['prompt_tokens']:,}: {t['cache_write_tokens']:,} cache writes, {t['cache_read_tokens']:,} cache reads ({cache['read_share'] * 100:.1f}% of input), {t['uncached_input_tokens']:,} uncached; {cache['later_requests_with_cache_read']} of the {cache['later_requests']} successful requests after an episode's first read the cache. Output {t['completion_tokens']:,} tokens, thinking included. Seconds per episode {s['seconds_per_episode_mean']:.1f} mean, {s['seconds_per_episode_max']:.1f} max (Opus {s['opus_seconds_per_episode_mean']:.1f}, f3 {s['f3_seconds_per_episode_mean']:.1f}); adding step 5's Opus wall clock for the reused part gives {s['seconds_fresh_upper_mean']:.1f}, an upper bound for a fresh 50% run since a fresh run would overlap the two. Run wall clock {s['wall_clock_s']:.0f} s, files last written {s['run_generated_utc']}.")
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
                      "stack": {f"{tag}:{kr}": {sp: s["by_rule"][tag][kr][sp]["sentence_points"]
                                                for sp in ("fit", "heldout", "all")}
                                for tag in SHARES for kr in ("decision", s["rules"][tag])},
                      "ceilings": {tag: {k: v["all"]["sentence_points"] for k, v in c.items()}
                                   for tag, c in s["ceilings"].items()},
                      "placement": s["placement"], "agreement": {k: v["agree_rate"] for k, v in s["agreement"].items()},
                      "flips": {k: (v.get("flips_right"), v.get("flips_wrong")) for k, v in s["flips"].items()},
                      "counts": s["counts"], "counts_25": s["counts_25"], "cache": s["cache"],
                      "seconds": {k: s[k] for k in ("seconds_per_episode_mean", "seconds_fresh_upper_mean",
                                                    "seconds_25_est_mean", "opus_seconds_per_episode_mean")},
                      "cost": {k: s[k] for k in ("cost_usd", "cost_per_episode_mean", "fresh_cost_per_episode_mean",
                                                 "fresh_cost_25_per_episode_mean", "fresh_cost_total",
                                                 "per_sentence_usd")}}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", nargs="+", default=None, help="subset of the 18 (default: all)")
    parser.add_argument("--run", action="store_true", help="actually call Opus")
    parser.add_argument("--resume", action="store_true", help="add episodes to a run whose files exist")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD, help="hard spend cap in USD")
    parser.add_argument("--concurrency", type=int, default=fo.CONCURRENCY)
    parser.add_argument("--ceiling", action="store_true", help="archived Opus on both slices; no calls")
    parser.add_argument("--extrapolate", action="store_true", help="18-episode cost from the probe; no calls")
    parser.add_argument("--keep-rule", action="store_true", help="choose and freeze the rules on the fit six")
    parser.add_argument("--report", action="store_true", help="write the markdown and JSON write-up")
    parser.add_argument("--force", action="store_true", help="overwrite the write-up or keep-rule file")
    args = parser.parse_args()
    if args.ceiling:
        return ceiling_mode()
    if args.extrapolate:
        return extrapolate_mode()
    if args.keep_rule:
        return keep_rule_mode(args, parser)
    if args.report:
        return report(args, parser)
    return execute(args, parser)


if __name__ == "__main__":
    sys.exit(main())
