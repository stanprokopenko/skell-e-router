"""The stack with Opus: the f1 combiner decides, claude-opus-5 overrides its unsure slice.

Third pass, step 5 of ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``.
Same slice as step 2 (``roughcut_hybrid_f1luna.py``: the f1 combiner's margin
bottom 25%, 2,236 sentences), same rules5 system prompt, preamble, whole
transcript and answer format as the Luna call, so the model is the only
intended difference. Cost control for Opus ($5 in, $25 out per million):

- the rules5 system prompt and the preamble plus whole transcript sit in the
  cached prefix (the router's Anthropic prompt caching, ``enable_caching=True``
  with a caller breakpoint on the transcript block, so ``_apply_cache_control``
  adds the system breakpoint and leaves the last-message one out); the target
  ids are a second, uncached text block in the same user message;
- an episode's first group goes alone and writes the cache; the rest go
  concurrently once it returns;
- groups of up to 80 targets (span cap 240, Luna's 40 / 120 ratio kept);
- reasoning effort medium.

A group whose answer is refused, malformed or incomplete is re-asked once;
targets still unanswered keep the combiner's decision and are counted as
fallbacks. Transport errors get up to three attempts, as in the Luna runner.

Modes::

  python scripts/jev_real/roughcut_hybrid_f1opus.py --ceiling       # archived Opus on this slice, $0
  python scripts/jev_real/roughcut_hybrid_f1opus.py                 # plan and estimate, no calls
  python scripts/jev_real/roughcut_hybrid_f1opus.py --run --episodes colman-03.03-muscles-crit   # probe
  python scripts/jev_real/roughcut_hybrid_f1opus.py --extrapolate   # 18-episode cost from the probe's real usage
  python scripts/jev_real/roughcut_hybrid_f1opus.py --run --resume --episodes <fit five>
  python scripts/jev_real/roughcut_hybrid_f1opus.py --keep-rule     # choose and freeze on the fit six only
  python scripts/jev_real/roughcut_hybrid_f1opus.py --run --resume  # the held-out 12
  python scripts/jev_real/roughcut_hybrid_f1opus.py --report        # write-up from the run on disk

READ-ONLY against solar-sailer. No network calls without ``--run``.
"""

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
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

import roughcut_hybrid_f1luna as fl  # noqa: E402  (the slice, the combiner seat, the offline check)
import roughcut_hybrid_luna as hl  # noqa: E402  (preamble, parser, keep rules, donors)
import roughcut_route2_routing as r2  # noqa: E402
import roughcut_jev_report as report_mod  # noqa: E402
import roughcut_jev as pipeline  # noqa: E402
from common import JsonlWriter  # noqa: E402

MODEL = "claude-opus-5"
EFFORT = "medium"
OUT_STEM = "roughcut-hybrid-f1opus"
TAG = "opus-m25"                         # donor tag inside the Scorer
NAME = f"{OUT_STEM}-m25"
ARM = "hybrid_f1opus_m25"
SHARE = fl.SHARE
GROUP_MAX = 80
GROUP_SPAN = 240
CONCURRENCY = 8
MAX_ATTEMPTS = 3
MALFORMED_REASKS = 1
RETRY_BACKOFF_S = (2.0, 6.0, 12.0)
MAX_TOKENS = 32000
PROBE_EPISODE = "colman-03.03-muscles-crit"
STOP_USD = 7.00                          # extrapolated 18-episode total that stops the run
DEFAULT_BUDGET_USD = 7.00                # hard cap on recorded spend
#: USD per million: input, cache write (1.25x), cache read (0.1x), output.
PRICE_IN, PRICE_WRITE, PRICE_READ, PRICE_OUT = 5.00, 6.25, 0.50, 25.00
#: Plan-mode output guess per request, medium effort; the probe replaces it.
OUT_TOKENS_BASE, OUT_TOKENS_PER_TARGET = 2000, 60
LUNA_TAG = "m25"                          # the f1-Luna stack run, roughcut-hybrid-f1luna-m25-*
FIT, KEPT, THRESHOLD = r2.FIT, r2.KEPT, r2.THRESHOLD
pct, table, num, money, fingerprint, md5_text = hl.pct, hl.table, hl.num, hl.money, hl.fingerprint, hl.md5_text


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run_paths(name=NAME):
    return {"decisions": OUT_DIR / f"{name}-decisions.jsonl",
            "requests": OUT_DIR / f"{name}-requests.jsonl",
            "timing": OUT_DIR / f"{name}-timing.json"}


KEEP_RULE_PATH = OUT_DIR / f"{NAME}-keeprule.json"


def hydrate_anthropic_key():
    """ANTHROPIC_API_KEY lives at Machine scope on this box; copy it into the process."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
            os.environ["ANTHROPIC_API_KEY"] = winreg.QueryValueEx(key, "ANTHROPIC_API_KEY")[0]
    except OSError as exc:
        raise SystemExit(f"ANTHROPIC_API_KEY not in the process or Machine environment: {exc}")


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def build_jobs(episode, inputs, routed, rules):
    """One job per group of routed, non-vetoed sentences; the prefix is the cached part."""
    data, lines, losers = hl.episode_context(episode, inputs)
    ids = sorted(routed[episode])
    asked = [sid for sid in ids if sid not in losers]
    vetoed = [sid for sid in ids if sid in losers]
    prefix = hl.PREAMBLE + "\n# TRANSCRIPT\n\n" + "\n".join(lines) + "\n"
    jobs = []
    for group, members in enumerate(r2.group_ids(asked, GROUP_MAX, GROUP_SPAN)):
        targets = "\n# TARGETS\n\n" + ", ".join(str(sid) for sid in members) + "\n"
        jobs.append({"episode": episode, "group": group, "ids": members,
                     "prefix": prefix, "targets": targets,
                     "prefix_chars": len(rules) + len(prefix), "target_chars": len(targets),
                     "user_md5": md5_text(prefix + targets), "prefix_md5": md5_text(prefix)})
    meta = {"sentences": len(data["order"]), "transcript_lines": len(lines),
            "routed": len(ids), "asked": len(asked), "retake_vetoed": len(vetoed),
            "vetoed_ids": vetoed, "groups": len(jobs),
            "prefix_chars": len(rules) + len(prefix)}
    return jobs, meta


def estimate(jobs_by_episode, tokens_per_char=1 / hl.CHARS_PER_TOKEN,
             out_base=OUT_TOKENS_BASE, out_per_target=OUT_TOKENS_PER_TARGET):
    """Cost with caching: first group writes the prefix, later groups read it."""
    per_episode, total = {}, Counter()
    for episode, jobs in jobs_by_episode.items():
        if not jobs:
            per_episode[episode] = {"requests": 0, "targets": 0, "cost_usd": 0.0}
            continue
        prefix = jobs[0]["prefix_chars"] * tokens_per_char
        write, read = prefix, prefix * (len(jobs) - 1)
        uncached = sum(j["target_chars"] for j in jobs) * tokens_per_char
        targets = sum(len(j["ids"]) for j in jobs)
        out = out_base * len(jobs) + out_per_target * targets
        cost = (uncached * PRICE_IN + write * PRICE_WRITE + read * PRICE_READ + out * PRICE_OUT) / 1e6
        per_episode[episode] = {"requests": len(jobs), "targets": targets,
                                "cache_write_tokens": round(write), "cache_read_tokens": round(read),
                                "uncached_tokens": round(uncached), "output_tokens": round(out),
                                "cost_usd": round(cost, 4)}
        for key in ("requests", "targets", "cache_write_tokens", "cache_read_tokens",
                    "uncached_tokens", "output_tokens"):
            total[key] += per_episode[episode][key]
        total["cost_usd"] += cost
    total = dict(total)
    total["cost_usd"] = round(total.get("cost_usd", 0.0), 4)
    return per_episode, total


# ---------------------------------------------------------------------------
# the Opus call
# ---------------------------------------------------------------------------

def _request(job, rules, budget, attempt, phase):
    """One Opus call with the transcript prefix cached. ``(row, answers_or_None)``; never raises."""
    from skell_e_router import ask_ai

    row = {"episode": job["episode"], "group": job["group"], "ids": job["ids"],
           "n_targets": len(job["ids"]), "attempt": attempt, "phase": phase, "model": MODEL,
           "effort": EFFORT, "provider_model": None, "prompt_tokens": None,
           "uncached_input_tokens": None, "cache_write_tokens": None, "cache_read_tokens": None,
           "completion_tokens": None, "cost": None, "cost_listed": None, "finish_reason": None,
           "refusal": False, "elapsed_s": None, "error": None, "parse_error": None,
           "missing_ids": None, "answers": None, "content": None,
           "user_md5": job["user_md5"], "prefix_md5": job["prefix_md5"],
           "system_md5": md5_text(rules), "preamble_version": hl.PREAMBLE_VERSION}
    if budget.blocked():
        row.update(error="BUDGET_STOP: cap reached before this request was sent", elapsed_s=0.0)
        return row, None
    messages = [{"role": "user", "content": [
        {"type": "text", "text": job["prefix"], "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": job["targets"]}]}]
    started = time.perf_counter()
    answers = None
    try:
        response = ask_ai(MODEL, messages, system_message=rules, rich_response=True,
                          reasoning_effort=EFFORT, enable_caching=True, max_tokens=MAX_TOKENS)
        usage = getattr(response.raw_response, "usage", None)

        def tok(attr):
            value = getattr(usage, attr, 0)
            return value if isinstance(value, int) else 0

        uncached, write, read = tok("input_tokens"), tok("cache_creation_input_tokens"), tok("cache_read_input_tokens")
        completion = response.completion_tokens or 0
        row.update(provider_model=response.model, prompt_tokens=response.prompt_tokens,
                   uncached_input_tokens=uncached, cache_write_tokens=write, cache_read_tokens=read,
                   completion_tokens=response.completion_tokens, cost=response.cost,
                   cost_listed=round((uncached * PRICE_IN + write * PRICE_WRITE + read * PRICE_READ
                                      + completion * PRICE_OUT) / 1e6, 6),
                   finish_reason=response.finish_reason, refusal=response.finish_reason == "refusal",
                   content=response.content)
        budget.add(response.cost if response.cost is not None else row["cost_listed"])
        answers, missing, error = hl.parse_answer(response.content, job["ids"])
        if row["refusal"]:
            error = f"refusal; {error}" if error else "refusal"
        row.update(answers={str(k): v for k, v in answers.items()}, missing_ids=missing, parse_error=error)
    except Exception as exc:  # noqa: BLE001 - a failed call is a data point
        code = getattr(exc, "code", None)
        row["error"] = f"{code or type(exc).__name__}: {exc}"[:400]
        row["error_details"] = str(getattr(exc, "details", None))[:300]
    row["elapsed_s"] = round(time.perf_counter() - started, 3)
    row["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row, answers


def run_episode_jobs(jobs, rules, budget, concurrency, writer):
    """First group alone (writes the cache), the rest concurrently, then retries.

    A group gets up to ``MAX_ATTEMPTS`` attempts on an exception and one re-ask
    on a refused, malformed or incomplete answer; answers from every attempt
    merge. Wall clock runs from the first submit to the last result.
    """
    answers = {job["group"]: {} for job in jobs}
    errors, reasks, rows = Counter(), Counter(), []
    started = time.perf_counter()

    def settle(batch, last):
        again = []
        for job in batch:
            group = job["group"]
            if all(sid in answers[group] for sid in job["ids"]):
                continue
            if last[group]["error"]:
                errors[group] += 1
                if errors[group] < MAX_ATTEMPTS:
                    again.append(job)
            else:
                reasks[group] += 1
                if reasks[group] <= MALFORMED_REASKS:
                    again.append(job)
        return again

    def dispatch(batch, phase):
        last = {}
        with futures.ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(batch)))) as pool:
            pending = {pool.submit(_request, job, rules, budget,
                                   errors[job["group"]] + reasks[job["group"]] + 1, phase): job
                       for job in batch}
            for future in futures.as_completed(pending):
                row, got = future.result()
                writer.write(row)
                rows.append(row)
                group = pending[future]["group"]
                last[group] = row
                if got:
                    answers[group].update(got)
        return settle(batch, last)

    todo = []
    if jobs and not budget.blocked():
        todo = dispatch(jobs[:1], "first") + list(jobs[1:])
        phase, round_no = "rest", 1
        while todo and not budget.blocked():
            if round_no > 1:
                time.sleep(RETRY_BACKOFF_S[min(round_no - 2, len(RETRY_BACKOFF_S) - 1)])
            todo = dispatch(todo, phase)
            phase, round_no = "retry", round_no + 1
    wall = time.perf_counter() - started

    unanswered = [sid for job in jobs for sid in job["ids"] if sid not in answers[job["group"]]]
    timing = {
        "wall_clock_s": round(wall, 3),
        "requests": len(rows),
        "retries": sum(1 for r in rows if r["attempt"] > 1),
        "errors": sum(1 for r in rows if r["error"]),
        "malformed": sum(1 for r in rows if not r["error"] and r["parse_error"]),
        "refusals": sum(1 for r in rows if r["refusal"]),
        "unanswered_targets": len(unanswered),
        "unanswered_ids": unanswered,
        "fallback_groups": sorted({job["group"] for job in jobs
                                   if any(sid not in answers[job["group"]] for sid in job["ids"])}),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "uncached_input_tokens": sum(r["uncached_input_tokens"] or 0 for r in rows),
        "cache_write_tokens": sum(r["cache_write_tokens"] or 0 for r in rows),
        "cache_read_tokens": sum(r["cache_read_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "cost_usd": round(sum(r["cost"] or 0.0 for r in rows), 6),
        "cost_listed_usd": round(sum(r["cost_listed"] or 0.0 for r in rows), 6),
        "latency": pipeline.latency_summary([r["elapsed_s"] for r in rows]),
        "first_group_s": next((r["elapsed_s"] for r in rows if r["phase"] == "first"), None),
    }
    return {sid: v for group in answers.values() for sid, v in group.items()}, timing


def decision_rows(episode, inputs, routed, answers, jobs, meta):
    """One row per sentence: the combiner's decision, Opus's verdict, and the mix."""
    base, raw = inputs["jev"][episode], inputs["jev_rows"][episode]
    margin = fl.margin_of(inputs["combiner_threshold"])
    group_of = {sid: job["group"] for job in jobs for sid in job["ids"]}
    rows = []
    for sid in sorted(base):
        jev = base[sid]
        is_routed = sid in routed[episode]
        answer = answers.get(sid)
        row = {"arm": ARM, "episode": episode, "id": sid, "routed": is_routed,
               "asked": sid in group_of, "group": group_of.get(sid),
               "retake_vetoed": is_routed and sid in meta["vetoed_ids"],
               "jev_score": raw[sid]["score"], "jev_margin": margin(raw[sid]),
               "jev_keep": jev["score"] is not None and jev["score"] >= THRESHOLD,
               "jev_keep_words": jev["keep_words"], "cut_retake": jev["cut_retake"],
               "opus_score": None, "opus_decision": None, "opus_reason": None,
               "opus_answered": False, "fallback": sid in group_of and not answer,
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

def plan_jobs(inputs, routed, rules, episodes):
    jobs_by_episode, metas = {}, {}
    for episode in episodes:
        jobs_by_episode[episode], metas[episode] = build_jobs(episode, inputs, routed, rules)
    return jobs_by_episode, metas


def execute(args, parser):
    result, inputs, cp, routed, cutoff = fl.check(log)
    rules = hl.read_rules()
    paths = run_paths()
    episodes = args.episodes or inputs["episodes"]
    unknown = [e for e in episodes if e not in inputs["episodes"]]
    if unknown:
        parser.error(f"not in the ladder: {unknown}")
    jobs_by_episode, metas = plan_jobs(inputs, routed, rules, episodes)
    per_episode_est, total_est = estimate(jobs_by_episode)
    plan_extra = {"selection": "f1 combiner margin abs(5 * p_keep - threshold), pooled, one cutoff",
                  "combiner": {"set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"]},
                  "offline_luna_ceiling": result["offline_ceiling_here"],
                  "caching": "system breakpoint (router) + caller breakpoint on preamble+transcript; targets uncached"}
    plan = {"mode": "plan" if not args.run else "run", "model": MODEL, "effort": EFFORT,
            "share": SHARE, "cutoff": cutoff, "routed_total": sum(len(v) for v in routed.values()),
            "out": NAME, "arm": ARM, "episodes": episodes, "group_max": GROUP_MAX,
            "group_span": GROUP_SPAN, "concurrency": args.concurrency,
            "rules": {"path": str(hl.RULES_PATH), "md5": md5_text(rules)},
            "price_per_million_usd": {"input": PRICE_IN, "cache_write": PRICE_WRITE,
                                      "cache_read": PRICE_READ, "output": PRICE_OUT},
            "budget_cap_usd": args.budget,
            "per_episode": {e: {k: v for k, v in {**metas[e], **per_episode_est[e]}.items() if k != "vetoed_ids"}
                            for e in episodes},
            "totals": total_est, "outputs": [str(p) for p in paths.values()], **plan_extra}
    print(json.dumps(plan, indent=2), flush=True)
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

    hydrate_anthropic_key()
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
            answers, timing = run_episode_jobs(jobs, rules, budget, args.concurrency, writer)
            rows = decision_rows(episode, inputs, routed, answers, jobs, metas[episode])
            new_decisions.extend(rows)
            timings[episode] = {**metas[episode], **timing,
                                "substituted": sum(1 for r in rows if r["opus_answered"]),
                                "fallbacks": sum(1 for r in rows if r["fallback"])}
            share = timing["cache_read_tokens"] / max(1, timing["prompt_tokens"])
            log(f"{episode}: {timing['wall_clock_s']}s, ${timing['cost_usd']:.4f} "
                f"(listed ${timing['cost_listed_usd']:.4f}), {timing['requests']} requests, "
                f"{timing['retries']} retries, {timing['errors']} errors, {timing['malformed']} malformed, "
                f"{timing['refusals']} refusals, {timing['unanswered_targets']} fallback targets, "
                f"cache write {timing['cache_write_tokens']} read {timing['cache_read_tokens']} "
                f"({share * 100:.0f}% of input); spent ${budget.spent:.4f}")
            if budget.blocked():
                aborted = f"budget cap ${args.budget:.2f} crossed at ${budget.spent:.4f}; stopped after {episode}"
                log(f"ABORT: {aborted}")
                break
            # checkpoint after every episode so a killed process loses at most one
            write_run_files(paths, old_decisions + new_decisions, timings, plan, args, cutoff,
                            aborted, old_timing, started, plan_extra)
    write_run_files(paths, old_decisions + new_decisions, timings, plan, args, cutoff,
                    aborted, old_timing, started, plan_extra)
    print(json.dumps(json.loads(paths["timing"].read_text(encoding="utf-8"))["totals"], indent=2))
    return 2 if aborted else 0


def write_run_files(paths, decisions, timings, plan, args, cutoff, aborted, old_timing, started, plan_extra):
    keys = ("requests", "retries", "errors", "malformed", "refusals", "unanswered_targets", "fallbacks",
            "prompt_tokens", "uncached_input_tokens", "cache_write_tokens", "cache_read_tokens",
            "completion_tokens", "routed", "asked", "retake_vetoed", "substituted", "groups")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "effort": EFFORT, "share": SHARE, "cutoff": cutoff, "arm": ARM, "out": NAME,
        "preamble_version": hl.PREAMBLE_VERSION, "rules": plan["rules"], "group_max": GROUP_MAX,
        "group_span": GROUP_SPAN, "concurrency": args.concurrency, "max_tokens": MAX_TOKENS,
        "budget_cap_usd": args.budget, "aborted": aborted, "resumed": bool(old_timing),
        "episodes": timings,
        "totals": {
            "wall_clock_s": round(time.perf_counter() - started
                                  + ((old_timing or {}).get("totals", {}).get("wall_clock_s", 0.0)), 3),
            "episodes": len(timings),
            **{key: sum(t.get(key, 0) for t in timings.values()) for key in keys},
            "cost_usd": round(sum(t["cost_usd"] for t in timings.values()), 6),
            "cost_listed_usd": round(sum(t["cost_listed_usd"] for t in timings.values()), 6),
        },
        "estimate": (old_timing or {}).get("estimate") or plan["totals"],
        "estimate_episodes": (old_timing or {}).get("estimate_episodes") or plan["episodes"],
        **plan_extra,
    }
    paths["decisions"].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions),
                                  encoding="utf-8")
    paths["timing"].write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# $0 modes: ceiling, extrapolation, keep rule
# ---------------------------------------------------------------------------

def load_donors(episodes):
    donors, donor_paths = {}, {}
    for key in ("luna", "opus"):
        loaded, paths = r2.load_donor(key, episodes)
        absent = [e for e in episodes if e not in loaded]
        if absent:
            raise SystemExit(f"archived {key} decisions missing for {absent}")
        donors[key], donor_paths[key] = loaded, paths
    # the live call gives no trims; the fair archived comparison drops Opus's keep_words too
    donors["opus_notrim"] = {e: {**v, "decisions": {sid: {**d, "keep_words": None}
                                                    for sid, d in v["decisions"].items()}}
                             for e, v in donors["opus"].items()}
    return donors, donor_paths


def ceiling_block(inputs, routed, donors=None):
    episodes = inputs["episodes"]
    donors = donors or load_donors(episodes)[0]
    scorer = r2.Scorer(inputs["jev"], donors, inputs["removals"])
    out = {"f1": r2.splits({e: scorer.episode(None, e, set()) for e in episodes}, episodes)}
    for key in ("opus", "opus_notrim", "luna"):
        out[key] = r2.splits({e: scorer.episode(key, e, routed[e]) for e in episodes}, episodes)
    return out


def ceiling_mode():
    result, inputs, cp, routed, cutoff = fl.check(log)
    block = ceiling_block(inputs, routed)
    print(json.dumps({"routed": result["routed"], "cutoff": cutoff,
                      **{k: {s: v[s]["sentence_points"] for s in ("fit", "heldout", "all")}
                         for k, v in block.items()}}, indent=2))
    return 0


def extrapolate_mode():
    """18-episode cost from the finished episodes' real usage (the probe)."""
    paths = run_paths()
    if not paths["timing"].exists():
        raise SystemExit("no run on disk; probe first")
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    requests = report_mod.read_jsonl(paths["requests"])
    done = list(timing["episodes"])
    result, inputs, cp, routed, cutoff = fl.check(log)
    rules = hl.read_rules()
    jobs_by_episode, metas = plan_jobs(inputs, routed, rules, inputs["episodes"])
    firsts = [r for r in requests if r["phase"] == "first" and r["episode"] in done and not r["error"]]
    chars = sum(jobs_by_episode[r["episode"]][0]["prefix_chars"] for r in firsts)
    written = sum((r["cache_write_tokens"] or 0) + (r["cache_read_tokens"] or 0)
                  + (r["uncached_input_tokens"] or 0) for r in firsts)
    tokens_per_char = written / chars
    ok = [r for r in requests if r["episode"] in done and not r["error"]]
    completion = sum(r["completion_tokens"] or 0 for r in ok)
    targets = sum(r["n_targets"] for r in ok)
    per_target = completion / targets
    per_request = completion / len(ok)
    # conservative: each episode pays the larger of target-proportional and request-proportional output
    per_episode = {}
    total = 0.0
    for e, jobs in jobs_by_episode.items():
        if not jobs:
            per_episode[e] = 0.0
            continue
        prefix = jobs[0]["prefix_chars"] * tokens_per_char
        n_targets = sum(len(j["ids"]) for j in jobs)
        out = max(per_target * n_targets, per_request * len(jobs))
        uncached = sum(j["target_chars"] for j in jobs) * tokens_per_char
        cost = (prefix * PRICE_WRITE + prefix * (len(jobs) - 1) * PRICE_READ
                + uncached * PRICE_IN + out * PRICE_OUT) / 1e6
        if e in done:
            cost = timing["episodes"][e]["cost_usd"]
        per_episode[e] = round(cost, 4)
        total += cost
    retry_factor = len([r for r in requests if r["episode"] in done]) / max(
        1, sum(timing["episodes"][e]["groups"] for e in done))
    projected = total * retry_factor if retry_factor > 1 else total
    out = {"probe_episodes": done, "probe_cost_usd": timing["totals"]["cost_usd"],
           "tokens_per_char": tokens_per_char, "completion_per_target": per_target,
           "completion_per_request": per_request, "retry_factor": retry_factor,
           "per_episode_usd": per_episode, "projected_18_usd": round(projected, 4),
           "stop_threshold_usd": STOP_USD, "go": projected <= STOP_USD,
           "cache_read_share_probe": timing["totals"]["cache_read_tokens"] / max(1, timing["totals"]["prompt_tokens"])}
    print(json.dumps(out, indent=2))
    return 0 if out["go"] else 3


def load_run(episodes, require_all=True):
    """The Opus run's rows, aliased to the ``luna_*`` names ``hl``'s donor helpers read."""
    paths = run_paths()
    if not all(p.exists() for p in paths.values()):
        return None
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    absent = [e for e in episodes if e not in timing["episodes"]]
    if absent and require_all:
        raise SystemExit(f"Opus run not finished, missing {absent}")
    by_episode = {e: {} for e in episodes}
    for row in report_mod.read_jsonl(paths["decisions"]):
        if row["episode"] in by_episode:
            by_episode[row["episode"]][row["id"]] = {
                **row, "luna_score": row["opus_score"], "luna_decision": row["opus_decision"],
                "luna_answered": row["opus_answered"]}
    routed = {e: {sid for sid, r in by_episode[e].items() if r["routed"]} for e in episodes}
    answered = {e: {sid for sid, r in by_episode[e].items() if r["opus_answered"]} for e in episodes}
    return {"tag": TAG, "rows": by_episode, "routed": routed, "answered": answered,
            "timing": timing, "requests": report_mod.read_jsonl(paths["requests"]), "paths": paths}


def keep_rule_mode(args, parser):
    """Choose the keep rule on the fit six only and freeze it to disk. Reads no held-out rows."""
    if KEEP_RULE_PATH.exists() and not args.force:
        parser.error(f"refusing to overwrite {KEEP_RULE_PATH}")
    result, inputs, cp, routed, cutoff = fl.check(log)
    fit = list(FIT)
    run = load_run(fit)
    for e in fit:
        if run["routed"][e] != routed[e]:
            raise SystemExit(f"{e}: the run's routed slice is not the slice selected now")
    donors = {}
    hl.register_live_donors(donors, run, fit)
    scorer = r2.Scorer({e: inputs["jev"][e] for e in fit}, donors,
                       {e: inputs["removals"][e] for e in fit})
    fit_sp = {rule: r2.pool(scorer.episode(hl.donor_key(TAG, rule), e, routed[e]) for e in fit)["sentence_points"]
              for rule in hl.KEEP_RULES}
    chosen = max(hl.KEEP_RULES, key=lambda rule: (round(fit_sp[rule], 6), -hl.KEEP_RULES.index(rule)))
    frozen = {"rule": chosen, "chosen_on": "fit six of roughcut-hybrid-f1opus-m25, before any held-out call",
              "fit_six_sentence_points": fit_sp,
              "tie_break": "best fit-six SP, ties to the decision field, then the lower score threshold",
              "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "fit_episodes_on_disk": fit,
              "heldout_episodes_on_disk": [e for e in run["timing"]["episodes"] if e not in fit]}
    KEEP_RULE_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report():
    result, inputs, cp, routed, cutoff = fl.check(log)
    episodes, removals = inputs["episodes"], inputs["removals"]
    f1_jev = inputs["jev"]
    if not KEEP_RULE_PATH.exists():
        raise SystemExit("no frozen keep rule; run --keep-rule on the fit six first")
    frozen = json.loads(KEEP_RULE_PATH.read_text(encoding="utf-8"))
    rule = frozen["rule"]
    run = load_run(episodes)
    for e in episodes:
        if run["routed"][e] != routed[e]:
            raise SystemExit(f"{e}: the run's routed slice is not the slice selected now")
    luna = hl.load_hybrid_run(LUNA_TAG, episodes, stem=fl.OUT_STEM)
    if luna is None:
        raise SystemExit("the f1-Luna stack run is missing")
    luna_doc = json.loads((OUT_DIR / f"{fl.OUT_STEM}.json").read_text(encoding="utf-8"))
    luna_rule = luna_doc["frozen_keep_rule"]["rule"]
    published = fl.published_route2(SHARE)

    log("donors and scoring...")
    donors, donor_paths = load_donors(episodes)
    hl.register_live_donors(donors, run, episodes)
    hl.register_live_donors(donors, luna, episodes)
    scorer = r2.Scorer(f1_jev, donors, removals)
    f1_eps = {e: scorer.episode(None, e, set()) for e in episodes}
    f1_pooled = r2.splits(f1_eps, episodes)

    block = hl.keep_rule_block(scorer, TAG, routed, episodes)
    if block["chosen"] != rule:
        raise SystemExit(f"fit-six choice recomputed as {block['chosen']}, frozen file says {rule}")
    by_rule = block["rules"]
    per_rule_ep = {kr: {e: scorer.episode(hl.donor_key(TAG, kr), e, routed[e]) for e in episodes}
                   for kr in ("decision", rule)}
    luna_stack = {kr: r2.splits({e: scorer.episode(hl.donor_key(LUNA_TAG, kr), e, routed[e]) for e in episodes},
                                episodes) for kr in ("decision", luna_rule)}
    luna_ep = {e: scorer.episode(hl.donor_key(LUNA_TAG, "decision"), e, routed[e]) for e in episodes}
    ceil = {key: r2.splits({e: scorer.episode(key, e, routed[e]) for e in episodes}, episodes)
            for key in ("opus", "opus_notrim", "luna")}
    ceil_ep = {e: scorer.episode("opus", e, routed[e]) for e in episodes}
    reproduction = {
        "f1_combiner": {"mine": f1_pooled["all"]["sentence_points"], "published": published["ladder_f1_sp"]},
        "luna_ceiling": {"mine": ceil["luna"]["all"]["sentence_points"], "published": published["sentence_points"]},
        "opus_ceiling": {"mine": ceil["opus"]["all"]["sentence_points"],
                         "published": luna_doc["opus_ceiling"]["sentence_points"]},
        "luna_stack_decision": {"mine": luna_stack["decision"]["all"]["sentence_points"],
                                "published": luna_doc["by_rule"]["decision"]["all"]["sentence_points"]},
        f"luna_stack_{luna_rule}": {"mine": luna_stack[luna_rule]["all"]["sentence_points"],
                                    "published": luna_doc["by_rule"][luna_rule]["all"]["sentence_points"]},
    }
    for key, pair in reproduction.items():
        if abs(pair["mine"] - pair["published"]) > 5e-4:
            raise SystemExit(f"reproduction failed for {key}: {pair}")

    ladder = r2.ladder_rows(episodes, luna_doc["v3"]["all"]["sentence_points"])
    ladder.append({"label": f"Jev f1 `{cp['chosen']}` combiner (build B)", "key": "f1",
                   "sentence_points": f1_pooled["all"]["sentence_points"]})
    ladder.append({"label": "f1-Luna stack, Luna decision", "key": "f1luna_decision",
                   "sentence_points": luna_stack["decision"]["all"]["sentence_points"]})
    ladder.append({"label": f"f1-Luna stack, `{luna_rule}`", "key": "f1luna_rule",
                   "sentence_points": luna_stack[luna_rule]["all"]["sentence_points"]})
    ladder.sort(key=lambda r: -r["sentence_points"])
    placement = {kr: r2.placement(by_rule[kr]["all"]["sentence_points"], ladder) for kr in ("decision", rule)}
    placement["ceiling"] = r2.placement(ceil["opus"]["all"]["sentence_points"], ladder)
    placement["ceiling_notrim"] = r2.placement(ceil["opus_notrim"]["all"]["sentence_points"], ladder)

    log("states, flips, agreement...")
    human, f1_states = {}, {}
    for e in episodes:
        human[e], f1_states[e] = r2.states(e, f1_jev[e], removals[e])
    flips = {kr: r2.flip_block(scorer, hl.donor_key(TAG, kr), routed, episodes, human, f1_states)
             for kr in ("decision", rule)}
    flips["luna_decision"] = r2.flip_block(scorer, hl.donor_key(LUNA_TAG, "decision"), routed, episodes,
                                           human, f1_states)
    flips["ceiling"] = r2.flip_block(scorer, "opus", routed, episodes, human, f1_states)
    agreement = hl.live_vs_archived(run, donors["opus"], episodes, human, removals)
    agreement_luna_archive = hl.live_vs_archived(run, donors["luna"], episodes, human, removals)

    # live Opus against live Luna on the same routed sentences
    c = Counter()
    for e in episodes:
        for sid in routed[e]:
            o, l = run["rows"][e][sid], luna["rows"][e][sid]
            if not (o["opus_answered"] and l["luna_answered"]):
                continue
            editor = human[e][sid] in KEPT
            ok, lk = o["opus_decision"] == "keep", l["luna_decision"] == "keep"
            c["n"] += 1
            c["agree"] += ok == lk
            c["opus_right"] += ok == editor
            c["luna_right"] += lk == editor
            c["opus_keep"] += ok
            c["luna_keep"] += lk
            c["editor_keep"] += editor
            if ok != lk:
                c["disagree"] += 1
                c["disagree_opus_right"] += ok == editor
    n = c["n"] or 1
    vs_luna = {**dict(c), **{f"{k}_rate": c[k] / n for k in ("agree", "opus_right", "luna_right",
                                                                "opus_keep", "luna_keep", "editor_keep")}}

    log("per episode...")
    t_eps = run["timing"]["episodes"]
    f1_secs = published["f1_seconds"]
    luna_t = luna["timing"]["episodes"]
    per_episode = {}
    for e in episodes:
        t = t_eps[e]
        f1_s = f1_secs.get(e, {}).get("total_s")
        f1_usd = f1_secs.get(e, {}).get("total_usd")
        per_episode[e] = {
            "sentences": len(f1_jev[e]), "routed": len(routed[e]), "asked": t["asked"],
            "groups": t["groups"], "requests": t["requests"], "retries": t["retries"],
            "errors": t["errors"], "malformed": t["malformed"], "refusals": t["refusals"],
            "fallbacks": t["fallbacks"], "substituted": t["substituted"],
            "f1": f1_eps[e], "luna_decision": luna_ep[e], "decision": per_rule_ep["decision"][e],
            "chosen": per_rule_ep[rule][e], "ceiling": ceil_ep[e],
            "opus_seconds": t["wall_clock_s"], "first_group_s": t["first_group_s"],
            "f1_seconds": f1_s, "seconds": (t["wall_clock_s"] + f1_s) if f1_s is not None else None,
            "luna_seconds": luna_t[e]["wall_clock_s"],
            "cost_usd": t["cost_usd"], "cost_listed_usd": t["cost_listed_usd"], "f1_cost_usd": f1_usd,
            "stack_cost_usd": (t["cost_usd"] + f1_usd) if f1_usd is not None else None,
            "luna_cost_usd": luna_t[e]["cost_usd"],
            "prompt_tokens": t["prompt_tokens"], "uncached_input_tokens": t["uncached_input_tokens"],
            "cache_write_tokens": t["cache_write_tokens"], "cache_read_tokens": t["cache_read_tokens"],
            "completion_tokens": t["completion_tokens"],
            "cache_read_share": t["cache_read_tokens"] / max(1, t["prompt_tokens"]),
        }
    totals = run["timing"]["totals"]
    secs = [per_episode[e]["seconds"] for e in episodes if per_episode[e]["seconds"] is not None]
    stack_costs = [per_episode[e]["stack_cost_usd"] for e in episodes if per_episode[e]["stack_cost_usd"] is not None]
    input_cost_uncached_equiv = totals["prompt_tokens"] * PRICE_IN / 1e6
    input_cost_actual = (totals["uncached_input_tokens"] * PRICE_IN + totals["cache_write_tokens"] * PRICE_WRITE
                         + totals["cache_read_tokens"] * PRICE_READ) / 1e6
    multi = [e for e in episodes if per_episode[e]["requests"] > 1]
    later = [r for r in run["requests"] if r["phase"] != "first" and not r["error"]]
    later_with_read = sum(1 for r in later if (r["cache_read_tokens"] or 0) > 0)

    inputs_fp = {k: fingerprint(v) for k, v in inputs["run_paths"].items()}
    inputs_fp["rules_prompt"] = fingerprint(hl.RULES_PATH)
    inputs_fp["ladder_reference"] = fingerprint(report_mod.REFERENCE_JSON)
    inputs_fp["f1_weights"] = fingerprint(cp["weights_path"])
    for split, p in cp["feature_paths"].items():
        inputs_fp[f"f1_{split}_features"] = fingerprint(p["features"])
    inputs_fp["f1_writeup_json"] = fingerprint(OUT_DIR / f"{fl.F1_NAME}.json")
    inputs_fp["f1luna_writeup_json"] = fingerprint(OUT_DIR / f"{fl.OUT_STEM}.json")
    for kind, path in luna["paths"].items():
        inputs_fp[f"f1luna_{LUNA_TAG}:{kind}"] = fingerprint(path)
    for key in ("luna", "opus"):
        for path in donor_paths[key]:
            inputs_fp[f"{key}:{Path(path).name}"] = fingerprint(path)
    for kind, path in run["paths"].items():
        inputs_fp[f"opus_run:{kind}"] = fingerprint(path)
    inputs_fp["opus_keep_rule"] = fingerprint(KEEP_RULE_PATH)
    digest = hashlib.md5()
    for e in episodes:
        digest.update(r2.md5(r2.removals_cache_path(e)).encode())

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_hybrid_f1opus.py",
        "model": MODEL, "effort": EFFORT, "arm": ARM, "tag": TAG,
        "episodes": episodes, "fit": FIT, "heldout": [e for e in episodes if e not in FIT],
        "sentences": sum(len(f1_jev[e]) for e in episodes),
        "combiner": {"set": cp["chosen"], "C": cp["C"], "threshold": cp["threshold"]},
        "share": SHARE, "cutoff": cutoff, "routed": sum(len(v) for v in routed.values()),
        "counts": {k: totals[k] for k in ("asked", "retake_vetoed", "substituted", "groups", "requests",
                                          "retries", "errors", "malformed", "refusals", "fallbacks")},
        "frozen_keep_rule": frozen, "keep_rules": hl.KEEP_RULES, "luna_rule": luna_rule,
        "reproduction": reproduction,
        "f1": f1_pooled, "by_rule": by_rule, "luna_stack": luna_stack, "ceilings": ceil,
        "ladder": ladder, "placement": placement, "flips": flips,
        "agreement_archived_opus": agreement, "agreement_archived_luna": agreement_luna_archive,
        "vs_live_luna": vs_luna, "per_episode": per_episode,
        "cost_usd": totals["cost_usd"], "cost_listed_usd": totals["cost_listed_usd"],
        "cost_per_episode_mean": totals["cost_usd"] / len(episodes),
        "cost_per_episode_max": max(per_episode[e]["cost_usd"] for e in episodes),
        "stack_cost_per_episode_mean": statistics.mean(stack_costs) if stack_costs else None,
        "luna_cost_per_episode_mean": luna_doc["cost_per_episode_mean"],
        "f1_cost_per_episode_mean": luna_doc["f1_cost_per_episode_mean"],
        "seconds_per_episode_mean": statistics.mean(secs) if secs else None,
        "seconds_per_episode_max": max(secs) if secs else None,
        "opus_seconds_per_episode_mean": statistics.mean(per_episode[e]["opus_seconds"] for e in episodes),
        "luna_seconds_per_episode_mean": luna_doc["luna_seconds_per_episode_mean"],
        "luna_stack_seconds_per_episode_mean": luna_doc["seconds_per_episode_mean"],
        "f1_seconds_per_episode_mean": luna_doc["f1_seconds_per_episode_mean"],
        "tokens": {k: totals[k] for k in ("prompt_tokens", "uncached_input_tokens", "cache_write_tokens",
                                          "cache_read_tokens", "completion_tokens")},
        "cache": {"read_share": totals["cache_read_tokens"] / max(1, totals["prompt_tokens"]),
                  "input_cost_actual_usd": input_cost_actual,
                  "input_cost_without_cache_usd": input_cost_uncached_equiv,
                  "later_requests": len(later), "later_requests_with_cache_read": later_with_read,
                  "multi_request_episodes": len(multi)},
        "estimate": run["timing"].get("estimate"), "wall_clock_s": totals["wall_clock_s"],
        "run_generated_utc": run["timing"]["generated_utc"],
        "settings": {"group_max": GROUP_MAX, "group_span": GROUP_SPAN, "concurrency": run["timing"]["concurrency"],
                     "max_attempts": MAX_ATTEMPTS, "malformed_reasks": MALFORMED_REASKS,
                     "max_tokens": MAX_TOKENS, "preamble_version": hl.PREAMBLE_VERSION,
                     "threshold": THRESHOLD},
        "inputs": inputs_fp, "removals_digest": digest.hexdigest(),
    }


def write_markdown(path, s, json_path):
    lines = []

    def add(text=""):
        lines.append(text)

    rule, lrule = s["frozen_keep_rule"]["rule"], s["luna_rule"]
    dec, cho = s["by_rule"]["decision"], s["by_rule"][rule]
    ld, lr = s["luna_stack"]["decision"], s["luna_stack"][lrule]
    co, cn, cl = s["ceilings"]["opus"], s["ceilings"]["opus_notrim"], s["ceilings"]["luna"]
    cnt, cache, sec = s["counts"], s["cache"], s["seconds_per_episode_mean"]
    add("# The f1-Opus stack: combiner decides, Opus overrides its unsure slice (developer-facing notes)")
    add()
    add(f"Generated {s['generated_utc']} by `{s['script']}` from the run files on disk, the f1 feature files and frozen weights, the f1-Luna stack run, the archived donor ratings and the cached removal ranges. The report step makes no model calls; the run it reads cost ${s['cost_usd']:.4f} by the router's accounting (${s['cost_listed_usd']:.4f} at list rates 5.00 in, 6.25 cache write, 0.50 cache read, 25.00 out per million). Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.")
    add()
    add(f"Question: third pass, step 5 of the round-two design. The slice is step 2's: the f1 combiner (`{s['combiner']['set']}`, keep threshold {s['combiner']['threshold']:.2f} on `5 * p_keep`) decides all {s['sentences']:,} sentences of the 18 ladder episodes, and the {s['routed']:,} whose margin sits under {s['cutoff']:.3f} (the pooled bottom {s['share'] * 100:g}%) go to `{s['model']}` with the same rules5 system prompt, preamble, whole transcript and answer format the Luna call used, {s['effort']} effort. Only the model and the request mechanics differ: the prompt is cached, groups hold up to {s['settings']['group_max']} targets instead of 40, and an episode's first group goes alone to write the cache. Two substitutions are reported: Opus's `decision` field, and the keep rule chosen on the fit six (`{rule}`), frozen to `{Path(s['inputs']['opus_keep_rule']['path']).name}` before any held-out call was made.")
    add()

    add("## Offline ceiling on this slice")
    add()
    add("Before any call: the archived Opus agentic ratings (`opus5-cc-agentic`, read the way `roughcut_route2_routing.py` reads its Opus donor, each file's own Neutral threshold) substituted on the same slice. The archived Opus carries partial keeps (`keep_words`), which the live call does not ask for, so the row without them is the like-for-like bound. No live calls behind these rows, so their seconds column is n/a.")
    add()
    rows = []
    for label, p in (("f1 combiner alone", s["f1"]), ("archived Opus with its trims", co),
                     ("archived Opus, keep/cut only (trims dropped)", cn),
                     ("archived Luna chapters (step 2's ceiling)", cl)):
        rows.append([label, pct(p["fit"]["sentence_points"]), pct(p["heldout"]["sentence_points"]),
                     pct(p["all"]["sentence_points"]), pct(p["all"]["word_score"]), pct(p["all"]["grade"]), "n/a"])
    lines.extend(table(["substitution on the slice", "SP fit 6", "SP held-out 12", "SP all 18",
                        "WORD all 18", "GRADE all 18", "s/ep mean"], rows))
    add()

    add("## Pooled results")
    add()
    add(f"Seconds per episode are the model's wall clock (first group alone, then up to {s['settings']['concurrency']} in flight) plus the combiner's own per-episode time from the f1 run; dollars are the router's accounting for the model calls, with the stack total (model plus the f1 combiner's Jev calls) alongside.")
    add()
    rows = []
    for label, p, secs, cost, total in (
            ("f1 combiner alone (build B)", s["f1"], s["f1_seconds_per_episode_mean"], None, s["f1_cost_per_episode_mean"]),
            ("f1-Luna stack, Luna `decision` (step 2)", ld, s["luna_stack_seconds_per_episode_mean"], s["luna_cost_per_episode_mean"],
             s["luna_cost_per_episode_mean"] + s["f1_cost_per_episode_mean"]),
            (f"f1-Luna stack, `{lrule}` (step 2)", lr, s["luna_stack_seconds_per_episode_mean"], s["luna_cost_per_episode_mean"],
             s["luna_cost_per_episode_mean"] + s["f1_cost_per_episode_mean"]),
            ("f1-Opus stack, Opus `decision`", dec, sec, s["cost_per_episode_mean"], s["stack_cost_per_episode_mean"]),
            (f"f1-Opus stack, fit-chosen `{rule}`", cho, sec, s["cost_per_episode_mean"], s["stack_cost_per_episode_mean"]),
            ("ceiling: archived Opus, keep/cut only", cn, None, None, None),
            ("ceiling: archived Opus with trims", co, None, None, None)):
        rows.append([label, pct(p["fit"]["sentence_points"]), pct(p["heldout"]["sentence_points"]),
                     pct(p["all"]["sentence_points"]), pct(p["all"]["word_score"]), pct(p["all"]["grade"]),
                     num(secs, 1), money(cost), money(total)])
    lines.extend(table(["arm", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18", "GRADE all 18",
                        "s/ep mean", "model $/ep mean", "stack $/ep mean"], rows))
    add()
    ladder_text = ", ".join(f"{r['label']} {pct(r['sentence_points'])}" for r in s["ladder"])
    pl = s["placement"]
    add(f"Ladder, with modules, same 18 episodes (the f1 combiner and both f1-Luna stack rows added): {ladder_text}. Placement: Opus stack with `decision` {pl['decision']['text']} (rank {pl['decision']['rank']} of {pl['decision']['of']}); with `{rule}` {pl[rule]['text']} (rank {pl[rule]['rank']} of {pl[rule]['of']}); the keep/cut-only ceiling {pl['ceiling_notrim']['text']}.")
    add()
    g_dec = dec["all"]["sentence_points"] - s["f1"]["all"]["sentence_points"]
    g_cn = cn["all"]["sentence_points"] - s["f1"]["all"]["sentence_points"]
    add(f"Against the Luna stack: Opus with its decision field lands {(dec['all']['sentence_points'] - ld['all']['sentence_points']) * 100:+.2f} SP on Luna's decision field pooled and {(dec['heldout']['sentence_points'] - ld['heldout']['sentence_points']) * 100:+.2f} held out; with each stack's own frozen rule, {(cho['all']['sentence_points'] - lr['all']['sentence_points']) * 100:+.2f} pooled and {(cho['heldout']['sentence_points'] - lr['heldout']['sentence_points']) * 100:+.2f} held out. Of the {g_cn * 100:.2f} SP the archived keep/cut-only Opus substitution adds over the combiner, the live decision field keeps {(g_dec / g_cn * 100) if g_cn else float('nan'):.0f}%.")
    add()

    add("## Every keep rule on this slice")
    add()
    fr = s["frozen_keep_rule"]
    add(f"The rule was chosen on the fit six alone (best fit-six SP, ties to the decision field, then the lower threshold) and written to disk at {fr['frozen_utc']}, when the run on disk held the fit six and {len(fr['heldout_episodes_on_disk'])} held-out episodes. The other rows are for the shape of the curve.")
    add()
    rows = []
    for kr in s["keep_rules"]:
        p = s["by_rule"][kr]
        rows.append([("* " if kr == rule else "") + kr, pct(p["fit"]["sentence_points"]),
                     pct(p["heldout"]["sentence_points"]), pct(p["all"]["sentence_points"]),
                     pct(p["all"]["word_score"]), pct(p["all"]["grade"]),
                     f"{(p['all']['sentence_points'] - dec['all']['sentence_points']) * 100:+.2f}", num(sec, 1)])
    lines.extend(table(["keep rule", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18", "GRADE all 18",
                        "all 18 minus decision", "s/ep mean"], rows))
    add()

    add("## Live Opus against archived Opus and live Luna")
    add()
    a, al, v = s["agreement_archived_opus"], s["agreement_archived_luna"], s["vs_live_luna"]
    add("Keep/cut on the routed sentences Opus answered. The archived Opus decision is its score against its own file's Neutral threshold; the archived Opus ran agentically on rules1 at high effort over whole episodes, so this is a different prompt as well as a different session.")
    add()
    rows = [["live Opus vs archived Opus", a["n"], pct(a["agree_rate"]), pct(a["live_right_rate"]),
             pct(a["archived_right_rate"]), f"{a.get('disagree_live_right', 0)} / {a.get('disagree_archived_right', 0)} of {a.get('disagree', 0)}",
             pct(a["live_keep_rate"]), pct(a["archived_keep_rate"]), pct(a["editor_keep_rate"]), num(sec, 1)],
            ["live Opus vs archived Luna chapters", al["n"], pct(al["agree_rate"]), pct(al["live_right_rate"]),
             pct(al["archived_right_rate"]), f"{al.get('disagree_live_right', 0)} / {al.get('disagree_archived_right', 0)} of {al.get('disagree', 0)}",
             pct(al["live_keep_rate"]), pct(al["archived_keep_rate"]), pct(al["editor_keep_rate"]), num(sec, 1)],
            ["live Opus vs live Luna (step 2)", v["n"], pct(v["agree_rate"]), pct(v["opus_right_rate"]),
             pct(v["luna_right_rate"]), f"{v.get('disagree_opus_right', 0)} / {v.get('disagree', 0) - v.get('disagree_opus_right', 0)} of {v.get('disagree', 0)}",
             pct(v["opus_keep_rate"]), pct(v["luna_keep_rate"]), pct(v["editor_keep_rate"]), num(sec, 1)]]
    lines.extend(table(["pair", "answered by both", "agree", "live Opus right", "other right",
                        "disagreements Opus right / other right", "Opus keep rate", "other keep rate",
                        "editor keep rate", "s/ep mean"], rows))
    add()
    add(f"Live Opus scores land within one point of the archived Opus score on {pct(a['score_within_1_rate'])}% of answered sentences.")
    add()

    add("## Where the gains come from")
    add()
    add("A flip is a routed sentence whose keep/cut changed when the model's verdict replaced the combiner's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor (kept means full or partial).")
    add()
    rows = []
    for label, c in (("Opus `decision`", s["flips"]["decision"]), (f"Opus `{rule}`", s["flips"][rule]),
                     ("Luna `decision` (step 2)", s["flips"]["luna_decision"]),
                     ("archived Opus with trims (ceiling)", s["flips"]["ceiling"])):
        routed = c.get("routed", 0)
        rows.append([label, routed, c.get("flips", 0), c.get("flips_right", 0), c.get("flips_wrong", 0),
                     f"{c.get('cut_to_kept_right', 0)} / {c.get('cut_to_kept_wrong', 0)}",
                     f"{c.get('kept_to_cut_right', 0)} / {c.get('kept_to_cut_wrong', 0)}",
                     pct(c.get("jev_agreed", 0) / routed if routed else None),
                     pct(c.get("hybrid_agreed", 0) / routed if routed else None), num(sec, 1)])
    lines.extend(table(["substitution", "routed", "flips", "right", "wrong", "cut to kept right / wrong",
                        "kept to cut right / wrong", "agreement on slice, combiner",
                        "agreement on slice, after routing", "s/ep mean"], rows))
    add()

    add(f"## Per episode ({s['share'] * 100:g}% routed, cutoff {s['cutoff']:.3f}, {s['effort']} effort)")
    add()
    rows = []
    for e in s["episodes"]:
        p = s["per_episode"][e]
        rows.append([e + (" (fit)" if e in s["fit"] else ""), p["sentences"], p["routed"], p["asked"],
                     pct(p["f1"]["sentence_points"]), pct(p["luna_decision"]["sentence_points"]),
                     pct(p["decision"]["sentence_points"]), pct(p["chosen"]["sentence_points"]),
                     pct(p["ceiling"]["sentence_points"]), p["requests"], p["retries"], p["fallbacks"],
                     num(p["opus_seconds"], 1), num(p["first_group_s"], 1), num(p["f1_seconds"], 1),
                     num(p["seconds"], 1), money(p["cost_usd"]), money(p["luna_cost_usd"]),
                     f"{p['cache_read_share'] * 100:.0f}%"])
    lines.extend(table(["episode", "sentences", "routed", "asked", "SP f1", "SP Luna stack decision",
                        "SP Opus stack decision", f"SP Opus stack {rule}", "SP archived Opus", "requests",
                        "retries", "fallback targets", "Opus s", "first group s", "f1 s", "s/ep",
                        "Opus $", "Luna $ (step 2)", "cache read share"], rows))
    add()

    add("## Cost, caching and failures")
    add()
    t = s["tokens"]
    add(f"{cnt['requests']} requests over {cnt['groups']} groups, {cnt['retries']} retries, {cnt['errors']} errored attempts, {cnt['malformed']} malformed or incomplete answers, {cnt['refusals']} refusals, {cnt['fallbacks']} targets left on the combiner's decision after the re-ask. Routed sentences the v3 retake pass had already cut were not sent: {cnt['retake_vetoed']} of {s['routed']:,}.")
    add()
    add(f"Input tokens {t['prompt_tokens']:,}: {t['cache_write_tokens']:,} cache writes, {t['cache_read_tokens']:,} cache reads ({cache['read_share'] * 100:.1f}% of input), {t['uncached_input_tokens']:,} uncached. {cache['later_requests_with_cache_read']} of the {cache['later_requests']} successful requests after an episode's first read the cache. Input cost ${cache['input_cost_actual_usd']:.4f} against ${cache['input_cost_without_cache_usd']:.4f} had nothing been cached. Output {t['completion_tokens']:,} tokens (thinking included), ${t['completion_tokens'] * PRICE_OUT / 1e6:.4f} of the ${s['cost_usd']:.4f}. Per episode ${s['cost_per_episode_mean']:.4f} mean, ${s['cost_per_episode_max']:.4f} max, against Luna's ${s['luna_cost_per_episode_mean']:.4f}; stack per episode, Opus plus the f1 combiner, ${s['stack_cost_per_episode_mean']:.4f}. The plan estimate was ${s['estimate']['cost_usd']:.4f}. Seconds per episode {sec:.1f} mean, {s['seconds_per_episode_max']:.1f} max (Opus {s['opus_seconds_per_episode_mean']:.1f}, f1 {s['f1_seconds_per_episode_mean']:.1f}; Luna was {s['luna_seconds_per_episode_mean']:.1f}). Run wall clock {s['wall_clock_s']:.0f} s, files last written {s['run_generated_utc']}.")
    add()

    add("## How the call was made")
    add()
    st = s["settings"]
    add(f"`skell_e_router.ask_ai('{s['model']}', ...)` on the direct Anthropic path with `enable_caching=True` and `reasoning_effort='{s['effort']}'` (adaptive thinking), `max_tokens` {st['max_tokens']:,}. System message: the rules5 prompt read from `{s['inputs']['rules_prompt']['path']}` at run time (md5 {s['inputs']['rules_prompt']['md5'][:12]}), which the router marks with a cache breakpoint. User message: two text blocks, the `{st['preamble_version']}` preamble plus the whole transcript as Jev's sentence pass rendered it (caller breakpoint, so the router adds no last-message one), then the target ids. The concatenated text is byte for byte what Luna received. Groups of up to {st['group_max']} targets, closing early past {st['group_span']} sentences (Luna's 40 / 120 ratio). An episode's first group goes alone; the rest go {st['concurrency']} in flight once it returns. {st['max_attempts']} attempts on a transport error, {st['malformed_reasks']} re-ask on a refused, malformed or incomplete answer, then the combiner's decision stands. Every attempt is a row in the requests file with the raw answer, the parsed verdicts, token counts split by cache write and read, and cost.")
    add()

    add("## Reproduction check")
    add()
    rows = [[k, pct(v["mine"]), pct(v["published"]), "n/a"] for k, v in s["reproduction"].items()]
    lines.extend(table(["number", "SP here", "published", "s/ep mean"], rows))
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
    md_path, json_path = OUT_DIR / f"{OUT_STEM}.md", OUT_DIR / f"{OUT_STEM}.json"
    existing = [str(p) for p in (md_path, json_path) if p.exists()]
    if existing and not args.force:
        parser.error(f"refusing to overwrite {existing}; pass --force")
    s = build_report()
    json_path.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    write_markdown(md_path, s, json_path)
    rule = s["frozen_keep_rule"]["rule"]
    print(json.dumps({"markdown": str(md_path), "rule": rule,
                      "opus": {kr: {sp: s["by_rule"][kr][sp]["sentence_points"] for sp in ("fit", "heldout", "all")}
                               for kr in ("decision", rule)},
                      "luna": {kr: {sp: s["luna_stack"][kr][sp]["sentence_points"] for sp in ("fit", "heldout", "all")}
                               for kr in s["luna_stack"]},
                      "ceilings": {k: {sp: v[sp]["sentence_points"] for sp in ("fit", "heldout", "all")}
                                   for k, v in s["ceilings"].items()},
                      "placement": s["placement"], "agree_archived_opus": s["agreement_archived_opus"]["agree_rate"],
                      "vs_live_luna": s["vs_live_luna"]["agree_rate"],
                      "flips": {k: (v.get("flips_right"), v.get("flips_wrong")) for k, v in s["flips"].items()},
                      "counts": s["counts"], "cache": s["cache"],
                      "seconds_per_episode_mean": s["seconds_per_episode_mean"],
                      "opus_seconds_per_episode_mean": s["opus_seconds_per_episode_mean"],
                      "cost_per_episode_mean": s["cost_per_episode_mean"],
                      "stack_cost_per_episode_mean": s["stack_cost_per_episode_mean"],
                      "cost_usd": s["cost_usd"]}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", nargs="+", default=None, help="subset of the 18 (default: all)")
    parser.add_argument("--run", action="store_true", help="actually call Opus")
    parser.add_argument("--resume", action="store_true", help="add episodes to a run whose files exist")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD, help="hard spend cap in USD")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--ceiling", action="store_true", help="archived Opus on this slice; no calls")
    parser.add_argument("--extrapolate", action="store_true", help="18-episode cost from the probe; no calls")
    parser.add_argument("--keep-rule", action="store_true", help="choose and freeze the rule on the fit six")
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
