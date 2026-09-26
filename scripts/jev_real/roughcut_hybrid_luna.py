"""Route 2 for real: Jev decides, gpt-5.6-luna overrides Jev's least confident slice.

Build A of ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``.
The slice is the ``roughcut_route2_routing.py`` selection: Jev's ``jev_a`` rows
from the 18-episode v3 run ranked by margin ``abs(score - 2.5)``, one global
cutoff over all 8,943 sentences (share 0.25 lands at 0.46, share 0.50 at 0.80).
Luna reads the whole episode transcript exactly as Jev saw it (module-cut ums
stripped, Jev's retake losers removed, ``<pause>`` markers) under the rules5
system prompt read from solar-sailer at run time, and gives every routed
sentence a ``score``, a keep/cut ``decision`` and a short ``reason``. Routed
sentences take Luna's decision (score 5 or 0, no trim, Jev's retake veto kept);
everything else stays Jev's. The mixed set is rescored with um removal + delete
silence layered on, the same path the route 2 ceiling used.

Modes::

  python scripts/jev_real/roughcut_hybrid_luna.py --share 0.25              # plan, no calls
  python scripts/jev_real/roughcut_hybrid_luna.py --share 0.25 --run        # 18 episodes
  python scripts/jev_real/roughcut_hybrid_luna.py --share 0.25 --run --effort high --out roughcut-hybrid-luna-m046-high
  python scripts/jev_real/roughcut_hybrid_luna.py --report                  # write-up from the runs on disk
  python scripts/jev_real/roughcut_hybrid_luna.py --keep-rule               # second pass step 1: keep rule on
                                                                            # the fit six, then held out; no calls

The keep-rule sweep (``KEEP_RULES``) scores the routed slice under Luna's
``decision`` field and under ``score >= t`` for t in 1..4, chooses on the six
fit episodes only and freezes that rule; ``--report`` adds the section and the
f1-Luna stack (``roughcut_hybrid_f1luna.py``) reports under the frozen rule.

A run writes ``docs/jev-real/<out>-decisions.jsonl``, ``-requests.jsonl`` and
``-timing.json`` and refuses to overwrite any of them. ``--resume`` finishes a
run that stopped part way: episodes already in the timing file are skipped, new
request rows are appended, decisions and timing are rewritten with the union.

READ-ONLY against solar-sailer. No network calls without ``--run``.
"""

import argparse
import concurrent.futures as futures
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "jev-real"
RULES_PATH = Path(r"D:\solar-sailer\benchmarks\roughcut\prompts\roughcut_system_partial_v5.md")

sys.path.insert(0, str(HERE))

import roughcut_route2_routing as r2  # noqa: E402  (imports the scoring module first)
import roughcut_jev_report as report_mod  # noqa: E402
import roughcut_jev as pipeline  # noqa: E402
from common import JsonlWriter  # noqa: E402

MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "medium"
EFFORTS = ["none", "low", "medium", "high", "xhigh"]
GROUP_MAX = 40            # routed sentences per request
GROUP_SPAN = 120          # a group closes once it would span more than this many sentences
CONCURRENCY = 8
MAX_ATTEMPTS = 3          # attempts per group on a transport or provider error
MALFORMED_REASKS = 1      # extra attempts per group on malformed or incomplete JSON
RETRY_BACKOFF_S = (2.0, 6.0, 12.0)
TIMEOUT_S = 600
DEFAULT_BUDGET_USD = 4.00
OUT_STEM = "roughcut-hybrid-luna"
ARM_PREFIX = "hybrid_luna"
PREAMBLE_VERSION = "hybrid-preamble-v1"
#: Plan-mode pricing, USD per million: gpt-5.6-luna is not in the router's own
#: table, so the estimate uses the list rates ``routing-notes.md`` used; the
#: run records the router's ``AIResponse.cost`` per request as well.
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.20, 0.02, 1.20
#: Plan-mode output guess per request: reasoning plus about 40 tokens of JSON
#: per target at medium effort. The run compares it with the real counts.
OUT_TOKENS_BASE, OUT_TOKENS_PER_TARGET = 1500, 45
CHARS_PER_TOKEN = 4
SCORE_SWEEP = (1, 2, 3, 4)
#: Keep rules for the routed slice: Luna's decision field, or its score at a
#: threshold. The second pass chooses one on the fit six and freezes it.
KEEP_RULES = ["decision"] + [f"score>={t}" for t in SCORE_SWEEP]
HEADLINE_RUN = "m046"                        # the run whose fit-six choice is frozen
CEILING_25, CEILING_SLACK = 0.8406, 0.015   # the high-effort rerun trigger
RUNS = ["m046", "m080", "m046-high"]        # write-up reads whichever exist
SMOKE_NAME = "smoke-hybrid-luna"            # the one-group smoke request's files
FIT, KEPT, THRESHOLD = r2.FIT, r2.KEPT, r2.THRESHOLD

PREAMBLE = """# THIS REQUEST

The rules above are the editing rules. This request differs from their output section in four ways:

1. The transcript below is the whole episode, one sentence per line as `id = text`. Word ids are not shown, filler ums the production module already cuts are already gone, `<pause>` and `<long pause>` mark silences, and the losing takes a retake pass already cut are absent. Read the whole transcript first; the surrounding lines decide what a target is worth.
2. Only the TARGET sentences listed at the end need a verdict. Every other line is context. Do not rate context lines.
3. Give each target a `score` (the 0-5 rubric above), a `decision` (`keep` or `cut`, your final call on whether the sentence stays in the edit) and one short `reason` (under 15 words). Do not give `keep_words`; partial keeps are handled elsewhere.
4. Respond with ONLY a JSON object, no prose, no markdown fences:

{"verdicts": [{"id": 12, "score": 4, "decision": "keep", "reason": "states the teaching point"}, {"id": 13, "score": 0, "decision": "cut", "reason": "false start, restarted in 14"}]}

Every target id must appear exactly once in `verdicts`, and no other id may appear.
"""


# ---------------------------------------------------------------------------
# formatting and small helpers
# ---------------------------------------------------------------------------

pct = r2.pct
table = r2.table
fingerprint = r2.fingerprint


def num(value, places=2):
    return "n/a" if value is None else f"{value:.{places}f}"


def money(value, places=4):
    return "n/a" if value is None else f"${value:.{places}f}"


def md5_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def est_tokens(text):
    return len(text) / CHARS_PER_TOKEN


def read_rules():
    if not RULES_PATH.exists():
        raise SystemExit(f"rules prompt not found at {RULES_PATH}")
    return RULES_PATH.read_text(encoding="utf-8")


def out_name_for(cutoff, effort):
    name = f"{OUT_STEM}-m{round(cutoff * 100):03d}"
    return name if effort == DEFAULT_EFFORT else f"{name}-{effort}"


def run_paths(name):
    return {"decisions": OUT_DIR / f"{name}-decisions.jsonl",
            "requests": OUT_DIR / f"{name}-requests.jsonl",
            "timing": OUT_DIR / f"{name}-timing.json"}


# ---------------------------------------------------------------------------
# selection and jobs
# ---------------------------------------------------------------------------

def routed_slice(inputs, share):
    """``(routed, cutoff)`` from the route 2 sweep's own pooled margin selection.

    The share, not the cutoff, is the primitive: ties at the cutoff are broken
    by episode order and sentence id exactly as the offline sweep broke them,
    so the slice is the one the ceiling was computed on.
    """
    return r2.select("margin", "pooled", share, inputs["jev_rows"], inputs["episodes"])


def episode_context(episode, inputs):
    """Transcript lines as Jev's sentence pass saw them, plus the retake losers."""
    data = pipeline.load_episode_data(episode)
    losers = {sid for sid, row in inputs["jev_rows"][episode].items() if row["cut_retake"]}
    lines = pipeline.transcript_lines(data, losers)
    return data, lines, losers


def build_jobs(episode, inputs, routed, rules):
    """One job per group of routed, non-vetoed sentences, transcript order."""
    data, lines, losers = episode_context(episode, inputs)
    ids = sorted(routed[episode])
    asked = [sid for sid in ids if sid not in losers]
    vetoed = [sid for sid in ids if sid in losers]
    prefix = (PREAMBLE + "\n# TRANSCRIPT\n\n" + "\n".join(lines) + "\n")
    jobs = []
    for group, members in enumerate(r2.group_ids(asked, GROUP_MAX, GROUP_SPAN)):
        user = prefix + "\n# TARGETS\n\n" + ", ".join(str(sid) for sid in members) + "\n"
        jobs.append({"episode": episode, "group": group, "ids": members,
                     "user": user,
                     "est_input_tokens": round(est_tokens(rules) + est_tokens(user)),
                     "user_md5": md5_text(user)})
    meta = {"sentences": len(data["order"]), "transcript_lines": len(lines),
            "routed": len(ids), "asked": len(asked), "retake_vetoed": len(vetoed),
            "vetoed_ids": vetoed, "groups": len(jobs)}
    return jobs, meta


def estimate(jobs_by_episode, effort):
    per_episode, total = {}, {"requests": 0, "targets": 0, "input_tokens": 0,
                              "output_tokens": 0, "cost_usd": 0.0}
    for episode, jobs in jobs_by_episode.items():
        tokens_in = sum(j["est_input_tokens"] for j in jobs)
        targets = sum(len(j["ids"]) for j in jobs)
        tokens_out = sum(OUT_TOKENS_BASE + OUT_TOKENS_PER_TARGET * len(j["ids"]) for j in jobs)
        if effort in ("high", "xhigh"):
            tokens_out *= 2
        cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
        per_episode[episode] = {"requests": len(jobs), "targets": targets,
                                "input_tokens": tokens_in, "output_tokens": tokens_out,
                                "cost_usd": round(cost, 4)}
        for key in total:
            total[key] += per_episode[episode][key]
    total["cost_usd"] = round(total["cost_usd"], 4)
    return per_episode, total


# ---------------------------------------------------------------------------
# the Luna call
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def parse_answer(content, ids):
    """``(answers, missing, error)``: validated verdicts keyed by sentence id."""
    if not content or not content.strip():
        return {}, list(ids), "empty content"
    text = _FENCE.sub("", content.strip())
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}, list(ids), f"not JSON: {exc.msg}"
        try:
            doc = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc2:
            return {}, list(ids), f"not JSON: {exc2.msg}"
    verdicts = doc.get("verdicts") if isinstance(doc, dict) else None
    if verdicts is None and isinstance(doc, dict):
        verdicts = doc.get("ratings")
    if not isinstance(verdicts, list):
        return {}, list(ids), "no verdicts list"
    wanted, answers, problems = set(ids), {}, []
    for item in verdicts:
        if not isinstance(item, dict):
            problems.append("non-object verdict")
            continue
        try:
            sid = int(item.get("id"))
        except (TypeError, ValueError):
            problems.append("bad id")
            continue
        if sid not in wanted:
            problems.append(f"id {sid} not a target")
            continue
        decision = str(item.get("decision", "")).strip().lower()
        score = item.get("score")
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = None
        if decision not in ("keep", "cut") or score is None or not 0 <= score <= 5:
            problems.append(f"id {sid}: score={item.get('score')!r} decision={item.get('decision')!r}")
            continue
        answers[sid] = {"score": score, "decision": decision,
                        "reason": str(item.get("reason", ""))[:200]}
    missing = [sid for sid in ids if sid not in answers]
    error = "; ".join(problems[:5]) if problems else None
    if missing and not error:
        error = f"{len(missing)} target(s) unanswered"
    return answers, missing, error


def _request(job, rules, effort, budget, attempt):
    """One Luna call. Returns ``(row, answers_or_None)``; never raises."""
    from skell_e_router import ask_ai

    row = {"episode": job["episode"], "group": job["group"], "ids": job["ids"],
           "n_targets": len(job["ids"]), "attempt": attempt, "model": MODEL,
           "effort": effort, "provider_model": None, "prompt_tokens": None,
           "cached_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
           "cost": None, "cost_listed": None, "finish_reason": None, "elapsed_s": None,
           "error": None, "parse_error": None, "missing_ids": None, "answers": None,
           "content": None, "est_input_tokens": job["est_input_tokens"],
           "user_md5": job["user_md5"], "system_md5": md5_text(rules),
           "preamble_version": PREAMBLE_VERSION}
    if budget.blocked():
        row.update(error="BUDGET_STOP: cap reached before this request was sent",
                   elapsed_s=0.0)
        return row, None
    started = time.perf_counter()
    answers = None
    try:
        response = ask_ai(MODEL, job["user"], system_message=rules, rich_response=True,
                          reasoning_effort=effort, timeout=TIMEOUT_S)
        usage = getattr(response.raw_response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        cached = cached if isinstance(cached, int) else 0
        prompt = response.prompt_tokens or 0
        completion = response.completion_tokens or 0
        row.update(provider_model=response.model, prompt_tokens=response.prompt_tokens,
                   cached_tokens=cached, completion_tokens=response.completion_tokens,
                   reasoning_tokens=response.reasoning_tokens, cost=response.cost,
                   cost_listed=round(((prompt - cached) * PRICE_IN + cached * PRICE_CACHED
                                      + completion * PRICE_OUT) / 1e6, 6),
                   finish_reason=response.finish_reason, content=response.content)
        budget.add(response.cost if response.cost is not None else row["cost_listed"])
        answers, missing, error = parse_answer(response.content, job["ids"])
        row.update(answers={str(k): v for k, v in answers.items()},
                   missing_ids=missing, parse_error=error)
    except Exception as exc:  # noqa: BLE001 - a failed call is a data point
        code = getattr(exc, "code", None)
        row["error"] = f"{code or type(exc).__name__}: {exc}"[:400]
        row["error_details"] = str(getattr(exc, "details", None))[:300]
    row["elapsed_s"] = round(time.perf_counter() - started, 3)
    row["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row, answers


def run_episode_jobs(jobs, rules, effort, budget, concurrency, writer):
    """Dispatch one episode's groups, retry, return answers and timing.

    A group gets up to ``MAX_ATTEMPTS`` attempts on an exception and
    ``MALFORMED_REASKS`` more on an answer that parses badly or misses targets;
    answers from every attempt merge, later ones winning. Wall clock covers
    the whole episode, submit to last result, at the concurrency used.
    """
    answers = {job["group"]: {} for job in jobs}
    errors, reasks, rows = Counter(), Counter(), []
    todo, round_no = list(jobs), 0
    started = time.perf_counter()
    while todo and not budget.blocked():
        round_no += 1
        if round_no > 1:
            time.sleep(RETRY_BACKOFF_S[min(round_no - 2, len(RETRY_BACKOFF_S) - 1)])
        last = {}
        with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {pool.submit(_request, job, rules, effort, budget,
                                   errors[job["group"]] + reasks[job["group"]] + 1): job
                       for job in todo}
            for future in futures.as_completed(pending):
                row, got = future.result()
                writer.write(row)
                rows.append(row)
                group = pending[future]["group"]
                last[group] = row
                if got:
                    answers[group].update(got)
        next_todo = []
        for job in todo:
            group = job["group"]
            if all(sid in answers[group] for sid in job["ids"]):
                continue
            if last[group]["error"]:
                errors[group] += 1
                if errors[group] < MAX_ATTEMPTS:
                    next_todo.append(job)
            else:
                reasks[group] += 1
                if reasks[group] <= MALFORMED_REASKS:
                    next_todo.append(job)
        todo = next_todo
    wall = time.perf_counter() - started

    unanswered = [sid for job in jobs for sid in job["ids"] if sid not in answers[job["group"]]]
    timing = {
        "wall_clock_s": round(wall, 3),
        "requests": len(rows),
        "retries": sum(1 for r in rows if r["attempt"] > 1),
        "errors": sum(1 for r in rows if r["error"]),
        "malformed": sum(1 for r in rows if not r["error"] and r["parse_error"]),
        "unanswered_targets": len(unanswered),
        "unanswered_ids": unanswered,
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "cached_tokens": sum(r["cached_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
        "reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in rows),
        "cost_usd": round(sum(r["cost"] or 0.0 for r in rows), 6),
        "cost_listed_usd": round(sum(r["cost_listed"] or 0.0 for r in rows), 6),
        "latency": pipeline.latency_summary([r["elapsed_s"] for r in rows]),
    }
    return {sid: v for group in answers.values() for sid, v in group.items()}, timing


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

def decision_rows(episode, inputs, routed, answers, jobs, meta, arm, effort, confidence=None):
    """One row per sentence: Jev's decision, Luna's verdict, and the mix.

    ``confidence(raw_row)`` is the margin the selection ranked on; the default
    is the v3 margin ``abs(score - 2.5)``. ``jev_keep`` is the Jev side's own
    keep/cut at the scorer's threshold, so a reader never has to re-derive it
    from ``jev_score`` (the stack's raw score is ``5 * p_keep`` on a 3.00 cut).
    """
    base, raw = inputs["jev"][episode], inputs["jev_rows"][episode]
    confidence = confidence or (lambda row: r2.confidence("margin", row))
    group_of = {sid: job["group"] for job in jobs for sid in job["ids"]}
    rows = []
    for sid in sorted(base):
        jev = base[sid]
        is_routed = sid in routed[episode]
        answer = answers.get(sid)
        row = {"arm": arm, "episode": episode, "id": sid, "routed": is_routed,
               "asked": sid in group_of, "group": group_of.get(sid),
               "retake_vetoed": is_routed and sid in meta["vetoed_ids"],
               "jev_score": raw[sid]["score"], "jev_margin": confidence(raw[sid]),
               "jev_keep": jev["score"] is not None and jev["score"] >= THRESHOLD,
               "jev_keep_words": jev["keep_words"], "cut_retake": jev["cut_retake"],
               "luna_score": None, "luna_decision": None, "luna_reason": None,
               "luna_answered": False, "score": jev["score"], "keep_words": jev["keep_words"],
               "model": MODEL, "effort": effort, "preamble_version": PREAMBLE_VERSION}
        if is_routed and answer:
            row.update(luna_score=answer["score"], luna_decision=answer["decision"],
                       luna_reason=answer["reason"], luna_answered=True,
                       score=5.0 if answer["decision"] == "keep" else 0.0, keep_words=None)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_inputs():
    return r2.load_jev()


def run(args, parser):
    inputs = load_inputs()
    rules = read_rules()
    routed, cutoff = routed_slice(inputs, args.share)
    name = args.out or out_name_for(cutoff, args.effort)
    arm = f"{ARM_PREFIX}_{name.split(OUT_STEM + '-', 1)[-1].replace('-', '_')}"
    return execute(args, parser, inputs, rules, routed, cutoff, name, arm)


def execute(args, parser, inputs, rules, routed, cutoff, name, arm, confidence=None,
            plan_extra=None):
    """Plan, estimate and (with ``--run``) make the calls for one routed slice.

    Shared with the f1-Luna stack (``roughcut_hybrid_f1luna.py``), which passes
    its own ``inputs`` (combiner keep/cut in ``jev``, ``5 * p_keep`` rows in
    ``jev_rows``), its own ``routed`` slice and ``confidence`` margin, plus
    ``plan_extra`` fields copied into the plan and the timing summary.
    """
    paths = run_paths(name)
    episodes = args.episodes or inputs["episodes"]
    unknown = [e for e in episodes if e not in inputs["episodes"]]
    if unknown:
        parser.error(f"not in the input run: {unknown}")

    jobs_by_episode, metas = {}, {}
    for episode in episodes:
        jobs, meta = build_jobs(episode, inputs, routed, rules)
        if args.limit_groups is not None:
            jobs = jobs[:args.limit_groups]
        jobs_by_episode[episode], metas[episode] = jobs, meta
    per_episode_est, total_est = estimate(jobs_by_episode, args.effort)

    plan = {"mode": "plan" if not args.run else "run", "model": MODEL,
            "effort": args.effort, "share": args.share, "cutoff": cutoff,
            "routed_total": sum(len(v) for v in routed.values()),
            "out": name, "arm": arm, "episodes": episodes,
            "group_max": GROUP_MAX, "group_span": GROUP_SPAN, "concurrency": args.concurrency,
            "rules": {"path": str(RULES_PATH), "md5": md5_text(rules),
                      "est_tokens": round(est_tokens(rules))},
            "price_per_million_usd": {"input": PRICE_IN, "cached_input": PRICE_CACHED,
                                      "output": PRICE_OUT},
            "budget_cap_usd": args.budget,
            "per_episode": {e: {**metas[e], **per_episode_est[e]} for e in episodes},
            "totals": total_est,
            "outputs": [str(p) for p in paths.values()], **(plan_extra or {})}
    print(json.dumps(plan, indent=2), flush=True)
    if total_est["cost_usd"] > args.budget:
        print("Estimate exceeds the budget cap; stopping.", flush=True)
        return 2
    if not args.run:
        print("Dry run. Re-run with --run to make the calls.", flush=True)
        return 0

    existing = [str(p) for p in paths.values() if p.exists()]
    done, old_decisions, old_timing = set(), [], None
    if existing and not args.resume:
        parser.error(f"refusing to overwrite existing output(s): {existing}; "
                     f"pass --resume to finish a partial run")
    if args.resume and paths["timing"].exists():
        old_timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
        done = set(old_timing.get("episodes", {}))
        old_decisions = report_mod.read_jsonl(paths["decisions"]) if paths["decisions"].exists() else []
        old_decisions = [r for r in old_decisions if r["episode"] in done]
        print(f"resume: skipping {len(done)} finished episode(s)", file=sys.stderr)

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
            answers, timing = run_episode_jobs(jobs, rules, args.effort, budget,
                                               args.concurrency, writer)
            rows = decision_rows(episode, inputs, routed, answers, jobs, metas[episode],
                                 arm, args.effort, confidence)
            new_decisions.extend(rows)
            timings[episode] = {**metas[episode], **timing,
                                "substituted": sum(1 for r in rows if r["luna_answered"]),
                                "est_input_tokens": sum(j["est_input_tokens"] for j in jobs)}
            print(f"{episode}: {timing['wall_clock_s']}s, ${timing['cost_usd']:.4f} "
                  f"(listed ${timing['cost_listed_usd']:.4f}), {timing['requests']} requests, "
                  f"{timing['retries']} retries, {timing['errors']} errors, "
                  f"{timing['malformed']} malformed, {timing['unanswered_targets']} unanswered",
                  file=sys.stderr, flush=True)
            if budget.blocked():
                aborted = (f"budget cap ${args.budget:.2f} crossed at ${budget.spent:.4f}; "
                           f"stopped after {episode}")
                print(f"ABORT: {aborted}", file=sys.stderr)
                break

    all_decisions = old_decisions + new_decisions
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "effort": args.effort, "share": args.share, "cutoff": cutoff,
        "arm": arm, "out": name, "preamble_version": PREAMBLE_VERSION,
        "rules": plan["rules"], "group_max": GROUP_MAX, "group_span": GROUP_SPAN,
        "concurrency": args.concurrency, "budget_cap_usd": args.budget,
        "aborted": aborted, "resumed": bool(old_timing),
        "episodes": timings,
        "totals": {
            "wall_clock_s": round(time.perf_counter() - started
                                  + ((old_timing or {}).get("totals", {}).get("wall_clock_s", 0.0)), 3),
            "episodes": len(timings),
            **{key: sum(t[key] for t in timings.values())
               for key in ("requests", "retries", "errors", "malformed", "unanswered_targets",
                           "prompt_tokens", "cached_tokens", "completion_tokens",
                           "reasoning_tokens", "routed", "asked", "retake_vetoed",
                           "substituted")},
            "cost_usd": round(sum(t["cost_usd"] for t in timings.values()), 6),
            "cost_listed_usd": round(sum(t["cost_listed_usd"] for t in timings.values()), 6),
            "est_input_tokens": sum(t["est_input_tokens"] for t in timings.values()),
        },
        "estimate": plan["totals"], **(plan_extra or {}),
    }
    paths["decisions"].write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_decisions), encoding="utf-8")
    paths["timing"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["totals"], indent=2))
    return 2 if aborted else 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def load_hybrid_run(tag, episodes, stem=OUT_STEM):
    """A finished run's decision set, routed slice and per-episode timing."""
    paths = run_paths(f"{stem}-{tag}")
    if not all(p.exists() for p in paths.values()):
        return None
    rows = report_mod.read_jsonl(paths["decisions"])
    timing = json.loads(paths["timing"].read_text(encoding="utf-8"))
    absent = [e for e in episodes if e not in timing["episodes"]]
    if absent:
        print(f"skipping run {tag}: not finished, missing {absent}", file=sys.stderr)
        return None
    by_episode = {e: {} for e in episodes}
    for row in rows:
        by_episode[row["episode"]][row["id"]] = row
    routed = {e: {sid for sid, r in by_episode[e].items() if r["routed"]} for e in episodes}
    asked = {e: {sid for sid, r in by_episode[e].items() if r["luna_answered"]} for e in episodes}
    requests = report_mod.read_jsonl(paths["requests"])
    return {"tag": tag, "rows": by_episode, "routed": routed, "answered": asked,
            "timing": timing, "requests": requests, "paths": paths}


def donor_from_rows(run, episodes, mode="decision", score_min=None):
    """Routed decisions as a ``Scorer`` donor: Luna's decision or a score cut."""
    out = {}
    for episode in episodes:
        decisions = {}
        for sid, row in run["rows"][episode].items():
            if not row["routed"]:
                continue
            if not row["luna_answered"]:
                keep = row.get("jev_keep")
                if keep is None:   # rows written before jev_keep existed (build A's runs)
                    keep = row["jev_score"] is not None and row["jev_score"] >= THRESHOLD
                decisions[sid] = {"score": 5.0 if keep else 0.0,
                                  "keep_words": row["jev_keep_words"],
                                  "cut_retake": row["cut_retake"]}
                continue
            if mode == "decision":
                keep = row["luna_decision"] == "keep"
            else:
                keep = row["luna_score"] >= score_min
            decisions[sid] = {"score": 5.0 if keep else 0.0, "keep_words": None,
                              "cut_retake": row["cut_retake"]}
        out[episode] = {"decisions": decisions}
    return out


def donor_key(tag, rule):
    """Scorer donor name for a live run under one keep rule."""
    return f"live:{tag}" if rule == "decision" else f"live:{tag}:{rule}"


def register_live_donors(donors, run, episodes):
    """Add a live run to ``donors`` under every keep rule in ``KEEP_RULES``."""
    donors[donor_key(run["tag"], "decision")] = donor_from_rows(run, episodes)
    for t in SCORE_SWEEP:
        donors[donor_key(run["tag"], f"score>={t}")] = donor_from_rows(run, episodes, "score", t)


def keep_rule_block(scorer, tag, routed, episodes):
    """Every keep rule on one run's routed slice, split fit / held-out / all.

    The choice reads the fit six only: best fit-six SENTENCE POINTS, ties to
    the decision field, then to the lower score threshold. The held-out and
    pooled numbers of the chosen rule are what the write-up reports as held out.
    """
    rules = {}
    for rule in KEEP_RULES:
        key = donor_key(tag, rule)
        rules[rule] = r2.splits({e: scorer.episode(key, e, routed[e]) for e in episodes},
                                episodes)
    chosen = max(KEEP_RULES, key=lambda rule: (round(rules[rule]["fit"]["sentence_points"], 6),
                                               -KEEP_RULES.index(rule)))
    return {"rules": rules, "chosen": chosen, "chosen_on": "fit six",
            "decision": rules["decision"], "frozen": rules[chosen]}


def keep_rule_sweep(runs, inputs, scorer=None):
    """``(per_run_blocks, frozen)``: the offline step 1 of the second pass, $0.

    The frozen rule is the headline run's fit-six choice (``HEADLINE_RUN``, the
    25% cutoff the stack reuses); the other runs' choices are recorded next to
    it so a disagreement is visible. ``scorer`` must already carry the live
    donors (``register_live_donors``); one is built when none is passed.
    """
    episodes, jev, removals = inputs["episodes"], inputs["jev"], inputs["removals"]
    if scorer is None:
        donors = {}
        for run in runs:
            register_live_donors(donors, run, episodes)
        scorer = r2.Scorer(jev, donors, removals)
    blocks = {run["tag"]: keep_rule_block(scorer, run["tag"], run["routed"], episodes)
              for run in runs}
    head = HEADLINE_RUN if HEADLINE_RUN in blocks else next(iter(blocks))
    frozen = {"rule": blocks[head]["chosen"], "chosen_on": f"fit six of {head}",
              "per_run_choice": {tag: b["chosen"] for tag, b in blocks.items()},
              "runs_agree": len({b["chosen"] for b in blocks.values()}) == 1}
    return blocks, frozen


def keep_rule_mode():
    """``--keep-rule``: print the sweep for the finished runs on disk. No calls."""
    inputs = load_inputs()
    runs = [r for r in (load_hybrid_run(tag, inputs["episodes"]) for tag in RUNS) if r]
    if not runs:
        raise SystemExit("no finished hybrid runs found under docs/jev-real")
    blocks, frozen = keep_rule_sweep(runs, inputs)
    out = {"frozen_keep_rule": frozen, "runs": {}}
    for tag, block in blocks.items():
        out["runs"][tag] = {
            "chosen_on_fit_six": block["chosen"],
            "fit_six": {rule: block["rules"][rule]["fit"]["sentence_points"] for rule in KEEP_RULES},
            "heldout_12": {rule: block["rules"][rule]["heldout"]["sentence_points"] for rule in KEEP_RULES},
            "all_18": {rule: block["rules"][rule]["all"]["sentence_points"] for rule in KEEP_RULES},
        }
    print(json.dumps(out, indent=2))
    return 0


def live_vs_archived(run, archived, episodes, human, removals):
    """Agreement of the live Luna decision with the archived one, on asked targets."""
    c = Counter()
    for episode in episodes:
        arch = archived[episode]["decisions"]
        raw = archived[episode]["raw"]
        for sid in run["answered"][episode]:
            row = run["rows"][episode][sid]
            if sid not in arch:
                c["no_archived"] += 1
                continue
            live = row["luna_decision"] == "keep"
            old = arch[sid]["score"] >= THRESHOLD
            editor = human[episode][sid] in KEPT
            c["n"] += 1
            c["agree"] += live == old
            c["live_right"] += live == editor
            c["archived_right"] += old == editor
            if live != old:
                c["disagree"] += 1
                c["disagree_live_right"] += live == editor
                c["disagree_archived_right"] += old == editor
                c["live_keep_archived_cut" if live else "live_cut_archived_keep"] += 1
            c["live_keep"] += live
            c["archived_keep"] += old
            c["editor_keep"] += editor
            live_score = row["luna_score"]
            c["score_within_1"] += abs(live_score - raw[sid]["score"]) <= 1
    n = c["n"] or 1
    return {**dict(c), "agree_rate": c["agree"] / n, "live_right_rate": c["live_right"] / n,
            "archived_right_rate": c["archived_right"] / n,
            "live_keep_rate": c["live_keep"] / n, "archived_keep_rate": c["archived_keep"] / n,
            "editor_keep_rate": c["editor_keep"] / n,
            "score_within_1_rate": c["score_within_1"] / n}


def episode_seconds(run, jev_latency, episode):
    luna = run["timing"]["episodes"][episode]["wall_clock_s"]
    jev = report_mod.seconds_for(jev_latency, episode)
    return luna, jev, (luna + jev) if jev is not None else None


def build_report():
    inputs = load_inputs()
    episodes, jev, removals = inputs["episodes"], inputs["jev"], inputs["removals"]
    jev_latency = report_mod.latency_rows(inputs["timing"], inputs["requests"], episodes)
    runs = [r for r in (load_hybrid_run(tag, episodes) for tag in RUNS) if r]
    if not runs:
        raise SystemExit("no finished hybrid runs found under docs/jev-real")

    donors, donor_paths = {}, {}
    for key in ("luna", "opus"):
        loaded, paths = r2.load_donor(key, episodes)
        absent = [e for e in episodes if e not in loaded]
        if absent:
            raise SystemExit(f"archived {key} decisions missing for {absent}")
        donors[key], donor_paths[key] = loaded, paths
    for run in runs:
        register_live_donors(donors, run, episodes)
    scorer = r2.Scorer(jev, donors, removals)
    keep_blocks, frozen_keep_rule = keep_rule_sweep(runs, inputs, scorer)

    jev_eps = {e: scorer.episode(None, e, set()) for e in episodes}
    jev_pooled = r2.splits(jev_eps, episodes)
    ladder = r2.ladder_rows(episodes, jev_pooled["all"]["sentence_points"])
    ladder_by_key = {r["key"]: r for r in ladder}
    everything = {e: set(jev[e]) for e in episodes}
    reproduction = {
        "jev_a": {"mine": jev_pooled["all"]["sentence_points"], "published": 0.8046516828804657},
        "luna": {"mine": r2.splits({e: scorer.episode("luna", e, everything[e]) for e in episodes},
                                   episodes)["all"]["sentence_points"],
                 "published": ladder_by_key["luna-chapters-rules5"]["sentence_points"]},
        "opus": {"mine": r2.splits({e: scorer.episode("opus", e, everything[e]) for e in episodes},
                                   episodes)["all"]["sentence_points"],
                 "published": ladder_by_key["opus5-cc-agentic"]["sentence_points"]},
    }

    human, jev_states, donor_states = {}, {}, {}
    for episode in episodes:
        human[episode], jev_states[episode] = r2.states(episode, jev[episode], removals[episode])
    donor_states["luna"] = {e: r2.states(e, donors["luna"][e]["decisions"], removals[e])[1]
                            for e in episodes}
    for run in runs:
        # The live donor only carries the routed sentences, so its states are
        # read off the mixed set (Jev everywhere else), which is the arm itself.
        key = f"live:{run['tag']}"
        donor_states[key] = {
            e: r2.states(e, scorer.mixed(key, e, run["routed"][e]), removals[e])[1]
            for e in episodes}

    results = []
    for run in runs:
        tag, routed = run["tag"], run["routed"]
        key = f"live:{tag}"
        per_ep = {e: scorer.episode(key, e, routed[e]) for e in episodes}
        ceiling_ep = {e: scorer.episode("luna", e, routed[e]) for e in episodes}
        opus_ep = {e: scorer.episode("opus", e, routed[e]) for e in episodes}
        pooled, ceiling = r2.splits(per_ep, episodes), r2.splits(ceiling_ep, episodes)
        sweep = {}
        for t in SCORE_SWEEP:
            k = f"live:{tag}:score>={t}"
            sweep[t] = r2.splits({e: scorer.episode(k, e, routed[e]) for e in episodes},
                                 episodes)["all"]
        t_eps = run["timing"]["episodes"]
        per_episode = {}
        for e in episodes:
            luna_s, jev_s, total_s = episode_seconds(run, jev_latency, e)
            per_episode[e] = {
                "sentences": len(jev[e]), "routed": len(routed[e]),
                "asked": t_eps[e]["asked"], "retake_vetoed": t_eps[e]["retake_vetoed"],
                "substituted": t_eps[e]["substituted"],
                "unanswered": t_eps[e]["unanswered_targets"],
                "requests": t_eps[e]["requests"], "retries": t_eps[e]["retries"],
                "errors": t_eps[e]["errors"], "malformed": t_eps[e]["malformed"],
                "jev": jev_eps[e], "hybrid": per_ep[e], "ceiling": ceiling_ep[e],
                "opus_ceiling": opus_ep[e],
                "luna_seconds": luna_s, "jev_seconds": jev_s, "seconds": total_s,
                "cost_usd": t_eps[e]["cost_usd"], "cost_listed_usd": t_eps[e]["cost_listed_usd"],
                "prompt_tokens": t_eps[e]["prompt_tokens"], "cached_tokens": t_eps[e]["cached_tokens"],
                "completion_tokens": t_eps[e]["completion_tokens"],
                "reasoning_tokens": t_eps[e]["reasoning_tokens"],
                "input_tokens_per_request": (t_eps[e]["prompt_tokens"] / t_eps[e]["requests"]
                                             if t_eps[e]["requests"] else None),
            }
        totals = run["timing"]["totals"]
        secs = [per_episode[e]["seconds"] for e in episodes if per_episode[e]["seconds"] is not None]
        results.append({
            "tag": tag, "arm": run["timing"]["arm"], "effort": run["timing"]["effort"],
            "share": run["timing"]["share"], "cutoff": run["timing"]["cutoff"],
            "routed": sum(len(v) for v in routed.values()),
            "asked": totals["asked"], "retake_vetoed": totals["retake_vetoed"],
            "substituted": totals["substituted"], "unanswered": totals["unanswered_targets"],
            "requests": totals["requests"], "retries": totals["retries"],
            "errors": totals["errors"], "malformed": totals["malformed"],
            "pooled": pooled, "ceiling": ceiling, "score_sweep": sweep,
            "keep_rule": keep_blocks[tag],
            "opus_ceiling": r2.splits(opus_ep, episodes)["all"],
            "placement": r2.placement(pooled["all"]["sentence_points"], ladder),
            "ceiling_placement": r2.placement(ceiling["all"]["sentence_points"], ladder),
            "flips": r2.flip_block(scorer, key, routed, episodes, human, jev_states),
            "ceiling_flips": r2.flip_block(scorer, "luna", routed, episodes, human, jev_states),
            "slice": r2.slice_agreement(routed, episodes, human, jev_states, donor_states[key]),
            "ceiling_slice": r2.slice_agreement(routed, episodes, human, jev_states,
                                                donor_states["luna"]),
            "agreement": live_vs_archived(run, donors["luna"], episodes, human, removals),
            "per_episode": per_episode,
            "cost_usd": totals["cost_usd"], "cost_listed_usd": totals["cost_listed_usd"],
            "cost_per_episode_mean": totals["cost_usd"] / len(episodes),
            "cost_per_episode_max": max(per_episode[e]["cost_usd"] for e in episodes),
            "seconds_per_episode_mean": statistics.mean(secs) if secs else None,
            "seconds_per_episode_max": max(secs) if secs else None,
            "luna_seconds_per_episode_mean": statistics.mean(
                per_episode[e]["luna_seconds"] for e in episodes),
            "tokens": {k: totals[k] for k in ("prompt_tokens", "cached_tokens",
                                                "completion_tokens", "reasoning_tokens")},
            "estimate": run["timing"].get("estimate"),
            "wall_clock_s": totals["wall_clock_s"],
            "generated_utc": run["timing"]["generated_utc"],
        })

    headline = next((r for r in results if r["tag"] == "m046"), None)
    trigger = None
    if headline:
        gap = CEILING_25 - headline["pooled"]["all"]["sentence_points"]
        trigger = {"gap_to_ceiling": gap, "threshold": CEILING_SLACK,
                   "high_effort_required": gap > CEILING_SLACK,
                   "high_effort_ran": any(r["tag"] == "m046-high" for r in results)}

    smoke_timing = run_paths(SMOKE_NAME)["timing"]
    smoke_spend = (json.loads(smoke_timing.read_text(encoding="utf-8"))["totals"]["cost_usd"]
                   if smoke_timing.exists() else None)

    inputs_fp = {k: fingerprint(v) for k, v in inputs["run_paths"].items()}
    inputs_fp["rules_prompt"] = fingerprint(RULES_PATH)
    inputs_fp["ladder_reference"] = fingerprint(report_mod.REFERENCE_JSON)
    for key, paths in donor_paths.items():
        for path in paths:
            inputs_fp[f"{key}:{Path(path).name}"] = fingerprint(path)
    for run in runs:
        for kind, path in run["paths"].items():
            inputs_fp[f"{run['tag']}:{kind}"] = fingerprint(path)
    digest = hashlib.md5()
    for episode in episodes:
        digest.update(r2.md5(r2.removals_cache_path(episode)).encode())

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/jev_real/roughcut_hybrid_luna.py",
        "model": MODEL, "episodes": episodes, "fit": FIT,
        "heldout": [e for e in episodes if e not in FIT],
        "sentences": sum(len(jev[e]) for e in episodes),
        "missing_jev_answers": inputs["n_missing"],
        "jev": jev_pooled, "ladder": ladder, "reproduction": reproduction,
        "jev_cost_per_episode": statistics.mean(jev_latency[e]["cost_usd"] for e in episodes),
        "jev_seconds_per_episode": report_mod.mean_seconds(jev_latency, episodes),
        "runs": results, "trigger": trigger,
        "frozen_keep_rule": frozen_keep_rule, "keep_rules": KEEP_RULES,
        "smoke_spend_usd": smoke_spend,
        "archived_thresholds": {e: donors["luna"][e]["threshold"] for e in episodes},
        "total_spend_usd": round(sum(r["cost_usd"] for r in results), 6),
        "total_spend_listed_usd": round(sum(r["cost_listed_usd"] for r in results), 6),
        "settings": {"group_max": GROUP_MAX, "group_span": GROUP_SPAN,
                     "concurrency": CONCURRENCY, "max_attempts": MAX_ATTEMPTS,
                     "malformed_reasks": MALFORMED_REASKS, "timeout_s": TIMEOUT_S,
                     "price_per_million_usd": {"input": PRICE_IN, "cached_input": PRICE_CACHED,
                                               "output": PRICE_OUT},
                     "preamble_version": PREAMBLE_VERSION, "threshold": THRESHOLD,
                     "t_trim": r2.T_TRIM},
        "inputs": inputs_fp, "removals_digest": digest.hexdigest(),
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def _survival(r, s):
    """Share of the archived substitution's gain over pure Jev the live call kept."""
    jev_sp = s["jev"]["all"]["sentence_points"]
    gain = r["ceiling"]["all"]["sentence_points"] - jev_sp
    return (r["pooled"]["all"]["sentence_points"] - jev_sp) / gain if gain else float("nan")


def run_label(r):
    return f"{r['tag']} ({r['share'] * 100:g}% routed, cutoff {r['cutoff']:.2f}, {r['effort']} effort)"


def write_markdown(path, s, json_path):
    lines = []

    def add(text=""):
        lines.append(text)

    add("# Real Luna routing on Jev's unsure slice (developer-facing notes)")
    add()
    smoke = (f", plus ${s['smoke_spend_usd']:.4f} for the one-group smoke request in `{SMOKE_NAME}-*`"
             if s.get("smoke_spend_usd") is not None else "")
    add(f"Generated {s['generated_utc']} by `{s['script']}` from the hybrid run files on disk, the stored Jev decisions, the archived donor ratings and the cached removal ranges. The report step itself makes no model calls; the runs it reads cost ${s['total_spend_usd']:.4f} by the router's accounting (${s['total_spend_listed_usd']:.4f} at list rates 0.20 in, 0.02 cached, 1.20 out per million){smoke}. Every metric is x100, two decimals, with um removal + delete silence layered on (the ladder column). The JSON next to this file keeps the raw values and every per-episode number.")
    add()
    add(f"Question: route 2 for real. Jev's `jev_a` v3 rows score all {s['sentences']:,} sentences of the 18 ladder episodes; the sentences whose score sits closest to the 2.50 keep threshold (margin `abs(score - 2.5)` under a global cutoff) go to `{s['model']}`, which reads the whole episode transcript as Jev saw it under the rules5 system prompt and returns a score, a keep or cut decision and a reason for each routed sentence. On routed sentences Luna's decision replaces Jev's keep/cut (score 5 or 0), `keep_words` is dropped, Jev's retake veto stays; everything else is Jev's decision rebuilt at trim trigger {s['settings']['t_trim']} and keep threshold {s['settings']['threshold']:.2f}, the setting behind the published 80.47. The cutoffs were chosen from the offline sweep over all 18 episodes (`roughcut-route2-routing.md`), so the routed share is not held out; the model's decisions on the slice are. Ties at the cutoff are broken by episode order and sentence id the way the sweep broke them, so the slice is the one the ceiling was computed on.")
    add()

    add("## Reproduction check")
    add()
    rows = [["pure Jev (jev_a v3)", pct(s["reproduction"]["jev_a"]["mine"]),
             pct(s["reproduction"]["jev_a"]["published"])],
            ["pure archived Luna chapters (routed 100%)", pct(s["reproduction"]["luna"]["mine"]),
             pct(s["reproduction"]["luna"]["published"])],
            ["pure archived Opus agentic (routed 100%)", pct(s["reproduction"]["opus"]["mine"]),
             pct(s["reproduction"]["opus"]["published"])]]
    lines.extend(table(["arm", "SENTENCE POINTS here", "published"], rows))
    add()

    add("## Pooled results")
    add()
    add("Pooled over the fit six, the held-out 12 and all 18, with modules. `ceiling` is the same routed slice substituted with the archived Luna chapters decision (whole-episode agentic run, per-file Neutral threshold), the number `roughcut-route2-routing.md` reported as the upper bound. Seconds per episode are Luna's wall clock at concurrency 8 plus Jev's own per-episode wall clock from the v3 run; dollars are the router's usage accounting summed per episode, Luna only.")
    add()
    rows = []
    for r in s["runs"]:
        rows.append([run_label(r), r["routed"], pct(r["pooled"]["fit"]["sentence_points"]),
                     pct(r["pooled"]["heldout"]["sentence_points"]),
                     pct(r["pooled"]["all"]["sentence_points"]),
                     pct(r["pooled"]["all"]["word_score"]), pct(r["pooled"]["all"]["grade"]),
                     pct(r["ceiling"]["all"]["sentence_points"]),
                     f"{r['placement']['rank']} of {r['placement']['of']}",
                     num(r["seconds_per_episode_mean"], 1), num(r["seconds_per_episode_max"], 1),
                     money(r["cost_per_episode_mean"]), money(r["cost_per_episode_max"])])
    lines.extend(table(["run", "routed", "SP fit 6", "SP held-out 12", "SP all 18", "WORD all 18",
                        "GRADE all 18", "ceiling SP all 18", "ladder rank", "s/ep mean", "s/ep max",
                        "Luna $/ep mean", "$/ep max"], rows))
    add()
    ladder_text = ", ".join(f"{r['label']} {pct(r['sentence_points'])}" for r in s["ladder"])
    add(f"Ladder, with modules, same 18 episodes: {ladder_text}. Placement: " + "; ".join(
        f"{r['tag']} {r['placement']['text']} (ceiling on the same slice: {r['ceiling_placement']['text']})"
        for r in s["runs"]) + ".")
    add()
    if s["trigger"]:
        t = s["trigger"]
        gap = t["gap_to_ceiling"] * 100
        add(f"High-effort trigger: medium effort at 25% routed sits {abs(gap):.2f} SP {'under' if gap >= 0 else 'over'} the 84.06 ceiling; the spec reruns at high effort when the gap is over {t['threshold'] * 100:.1f} SP, so the rerun was {'required' if t['high_effort_required'] else 'not required'} and {'ran' if t['high_effort_ran'] else 'did not run'}.")
        add()
    add("How much of the ceiling survives the live call, as a share of the gain the archived substitution makes over pure Jev: " + "; ".join(
        f"{r['tag']} {_survival(r, s) * 100:.0f}% ({pct(r['pooled']['all']['sentence_points'])} of {pct(r['ceiling']['all']['sentence_points'])} against {pct(s['jev']['all']['sentence_points'])})"
        for r in s["runs"]) + ".")
    add()

    add("## Score sweep on the routed slice")
    add()
    add("Luna's 0-5 score is stored next to its decision, so the routed slice can also be cut at a score threshold instead of the decision field. `decision` is the number above; `score >= t` keeps a routed sentence when Luna's score clears t. Pooled 18, with modules.")
    add()
    rows = []
    for r in s["runs"]:
        rows.append([run_label(r), pct(r["pooled"]["all"]["sentence_points"])]
                    + [pct(r["score_sweep"][t]["sentence_points"]) for t in SCORE_SWEEP]
                    + [pct(r["ceiling"]["all"]["sentence_points"]),
                       num(r["seconds_per_episode_mean"], 1)])
    lines.extend(table(["run", "SP decision"] + [f"SP score >= {t}" for t in SCORE_SWEEP]
                       + ["SP ceiling", "s/ep mean"], rows))
    add()
    best = []
    for r in s["runs"]:
        t_best = max(SCORE_SWEEP, key=lambda t: r["score_sweep"][t]["sentence_points"])
        best.append(f"{r['tag']} keeps at score >= {t_best} for {pct(r['score_sweep'][t_best]['sentence_points'])}, "
                    f"{(r['score_sweep'][t_best]['sentence_points'] - r['pooled']['all']['sentence_points']) * 100:+.2f} on the decision field and "
                    f"{(r['score_sweep'][t_best]['sentence_points'] - r['ceiling']['all']['sentence_points']) * 100:+.2f} on the archived ceiling")
    add("Best score threshold per run: " + "; ".join(best) + ". Luna's own keep/cut decision is cut-heavier than the editor on this slice (keep rates in the next table), so keeping anything it scores 2 or more recovers part of that. The threshold is picked on the same 18 episodes it is reported on, so read it as the shape of the curve, not a held-out number; the decision-field SP above is the number this build set out to measure. The next section redoes the choice with the fit/held-out split.")
    add()

    add("## Keep rule, held out (second pass, step 1)")
    add()
    fk = s["frozen_keep_rule"]
    add(f"Same runs, same stored answers, $0. Each keep rule (Luna's `decision` field, or keep when Luna's score clears 1, 2, 3 or 4) is scored on the six fit episodes first; the best fit-six SENTENCE POINTS is frozen (ties go to the decision field, then to the lower threshold), and only then are the 12 held-out episodes and the pooled 18 read under that rule. The fit-six column is where the choice was made and is not held out; the held-out 12 column is. Pooled 18 mixes the two. The chosen row of each run is marked with `*`. Seconds per episode are the run's Luna wall clock plus Jev's, as above; the rule changes nothing about the call.")
    add()
    for r in s["runs"]:
        kb = r["keep_rule"]
        rows = []
        for rule in s["keep_rules"]:
            sp = kb["rules"][rule]
            rows.append([("* " if rule == kb["chosen"] else "") + rule,
                         pct(sp["fit"]["sentence_points"]), pct(sp["heldout"]["sentence_points"]),
                         pct(sp["all"]["sentence_points"]), pct(sp["all"]["word_score"]),
                         pct(sp["all"]["grade"]),
                         f"{(sp['heldout']['sentence_points'] - kb['decision']['heldout']['sentence_points']) * 100:+.2f}",
                         f"{(sp['all']['sentence_points'] - kb['decision']['all']['sentence_points']) * 100:+.2f}",
                         num(r["seconds_per_episode_mean"], 1)])
        add(f"### {run_label(r)}")
        add()
        lines.extend(table(["keep rule", "SP fit 6 (chosen on)", "SP held-out 12", "SP all 18",
                            "WORD all 18", "GRADE all 18", "held-out minus decision",
                            "all 18 minus decision", "s/ep mean"], rows))
        add()
    summary_bits = []
    for r in s["runs"]:
        kb = r["keep_rule"]
        ch, dec = kb["frozen"], kb["decision"]
        fit_gap = (ch["fit"]["sentence_points"] - dec["fit"]["sentence_points"]) * 100
        summary_bits.append(
            f"{r['tag']} chooses `{kb['chosen']}` on the fit six ({pct(ch['fit']['sentence_points'])} against "
            f"{pct(dec['fit']['sentence_points'])} for the decision field, a {fit_gap:+.2f} margin"
            f"{', close to a tie on its own' if abs(fit_gap) < 0.1 else ''}); held out it gives "
            f"{pct(ch['heldout']['sentence_points'])} against {pct(dec['heldout']['sentence_points'])}, pooled 18 "
            f"{pct(ch['all']['sentence_points'])} against {pct(dec['all']['sentence_points'])}, ceiling on the slice "
            f"{pct(r['ceiling']['all']['sentence_points'])}")
    agree = ("both runs choose the same rule" if fk["runs_agree"]
             else "the runs choose different rules: " + ", ".join(f"{t} {c}" for t, c in fk["per_run_choice"].items()))
    add("Result: " + "; ".join(summary_bits) + f". Frozen rule for the second pass: `{fk['rule']}`, chosen on the {fk['chosen_on']} ({agree}). The f1-Luna stack (`roughcut-hybrid-f1luna.md`) reports its slice under the decision field and under this rule.")
    add()

    add("## Live Luna against archived Luna on the same sentences")
    add()
    add("The ceiling substituted the archived Luna chapters decision (score at the file's own Neutral threshold, 0.1 to 2.1 per episode) on the routed slice. The live run substitutes the decision the windowed call returned. Agreement is measured on the routed sentences the live run got an answer for; right means the decision matches the editor (kept means full or partial). SP columns are the pooled 18 with modules for each substitution.")
    add()
    rows = []
    for r in s["runs"]:
        a = r["agreement"]
        rows.append([run_label(r), a["n"], pct(a["agree_rate"]), pct(a["live_right_rate"]),
                     pct(a["archived_right_rate"]), pct(r["slice"]["jev"]),
                     f"{a['disagree_live_right']} / {a['disagree_archived_right']} of {a['disagree']}",
                     f"{a['live_keep_archived_cut']} / {a['live_cut_archived_keep']}",
                     pct(a["live_keep_rate"]), pct(a["archived_keep_rate"]), pct(a["editor_keep_rate"]),
                     pct(r["pooled"]["all"]["sentence_points"]),
                     pct(r["ceiling"]["all"]["sentence_points"]),
                     num(r["seconds_per_episode_mean"], 1)])
    lines.extend(table(["run", "answered", "live agrees with archived", "live right", "archived right",
                        "Jev right", "disagreements live right / archived right",
                        "live keep & archived cut / live cut & archived keep",
                        "live keep rate", "archived keep rate", "editor keep rate",
                        "SP live", "SP archived (ceiling)", "s/ep mean"], rows))
    add()
    why = []
    for r in s["runs"]:
        a = r["agreement"]
        why.append(f"{r['tag']}: of the {a['disagree']} disagreements the live call cuts where the archive keeps {a['live_cut_archived_keep']} times and keeps where the archive cuts {a['live_keep_archived_cut']} times, and the archive is right on {a['disagree_archived_right']} of them against the live call's {a['disagree_live_right']}")
    n01 = sum(1 for t in s["archived_thresholds"].values() if abs(t - 0.1) < 1e-9)
    add("Why the two differ where they do: " + "; ".join(why) + f". The archived run rated every sentence in one agentic session with a review loop and was then thresholded per episode against the editor, with a 0.1 threshold on {n01} of the 18 files (anything not scored 0 is kept), so it leans keep; the live call answers only the routed ids, has no review loop, and its keep/cut is the model's own call, which leans cut on exactly the sentences Jev was unsure about. The editor keeps more of this slice than either, so the cut-heavy side loses more. Live scores land within one point of the archived score on " + ", ".join(f"{pct(r['agreement']['score_within_1_rate'])}% ({r['tag']})" for r in s["runs"]) + " of the answered sentences.")
    add()

    add("## Where the gains come from")
    add()
    add("A flip is a routed sentence whose keep/cut changed when Luna's decision replaced Jev's, read off the scoring module's own sentence states with the modules layered. Right means the new state matches the editor. Trim changed counts routed sentences kept on both sides whose word runs differ (the live run drops Jev's trims on routed sentences, so this is mostly Jev trims that vanished). The ceiling row is the archived substitution on the same slice.")
    add()
    rows = []
    for r in s["runs"]:
        for label, c in ((run_label(r), r["flips"]), (f"{r['tag']} ceiling (archived Luna)", r["ceiling_flips"])):
            routed = c.get("routed", 0)
            rows.append([label, routed, c.get("flips", 0), c.get("flips_right", 0),
                         c.get("flips_wrong", 0),
                         f"{c.get('cut_to_kept_right', 0)} / {c.get('cut_to_kept_wrong', 0)}",
                         f"{c.get('kept_to_cut_right', 0)} / {c.get('kept_to_cut_wrong', 0)}",
                         c.get("trim_changed", 0),
                         pct(c.get("jev_agreed", 0) / routed if routed else None),
                         pct(c.get("hybrid_agreed", 0) / routed if routed else None),
                         num(r["seconds_per_episode_mean"], 1)])
    lines.extend(table(["run", "routed", "flips", "right", "wrong", "cut to kept right / wrong",
                        "kept to cut right / wrong", "trim changed", "agreement on slice, Jev",
                        "agreement on slice, after routing", "s/ep mean"], rows))
    add()

    for r in s["runs"]:
        add(f"## Per episode, {run_label(r)}")
        add()
        rows = []
        for e in s["episodes"]:
            p = r["per_episode"][e]
            rows.append([e + (" (fit)" if e in s["fit"] else ""), p["sentences"], p["routed"],
                         p["asked"], p["unanswered"], pct(p["jev"]["sentence_points"]),
                         pct(p["hybrid"]["sentence_points"]), pct(p["ceiling"]["sentence_points"]),
                         pct(p["hybrid"]["word_score"]), pct(p["hybrid"]["grade"]),
                         p["requests"], p["retries"], num(p["luna_seconds"], 1),
                         num(p["jev_seconds"], 1), num(p["seconds"], 1), money(p["cost_usd"]),
                         f"{p['input_tokens_per_request'] / 1000:.1f}k" if p["input_tokens_per_request"] else "n/a",
                         f"{p['cached_tokens'] / max(1, p['prompt_tokens']) * 100:.0f}%"])
        lines.extend(table(["episode", "sentences", "routed", "asked", "unanswered", "SP Jev",
                            "SP hybrid", "SP ceiling", "WORD hybrid", "GRADE hybrid", "requests",
                            "retries", "Luna s", "Jev s", "s/ep", "Luna $", "input tokens/request",
                            "cached share"], rows))
        add()
        add(f"Routed sentences that Jev's retake pass had already cut were not sent (they are cut whichever way Luna would vote): {r['retake_vetoed']} of {r['routed']}. {r['requests']} requests in total, {r['retries']} retries, {r['errors']} errored attempts, {r['malformed']} malformed or incomplete answers, {r['unanswered']} targets left unanswered after retries (those keep Jev's own decision). Tokens: {r['tokens']['prompt_tokens']:,} prompt of which {r['tokens']['cached_tokens']:,} cached, {r['tokens']['completion_tokens']:,} completion of which {r['tokens']['reasoning_tokens']:,} reasoning. Router cost ${r['cost_usd']:.4f}, list-rate cost ${r['cost_listed_usd']:.4f}; the plan estimated ${r['estimate']['cost_usd']:.4f}. Run wall clock {r['wall_clock_s']:.0f} s, generated {r['generated_utc']}.")
        add()

    add("## How the call was made")
    add()
    st = s["settings"]
    add(f"System message: the rules5 prompt read from `{s['inputs']['rules_prompt']['path']}` at run time (md5 {s['inputs']['rules_prompt']['md5'][:12]}), never copied into this repo. User message: a short preamble (`{st['preamble_version']}`, in the script) saying the transcript is `id = text` lines without word ids, that only the listed targets need a verdict, and the JSON shape to answer with; then the whole episode transcript as Jev's sentence pass rendered it (module-cut ums stripped, Jev's retake losers removed, pause markers); then the target ids. Targets are the routed ids in transcript order in groups of up to {st['group_max']}, a group closing early once it would span more than {st['group_span']} sentences; {st['concurrency']} requests in flight per episode, {st['max_attempts']} attempts on an error with backoff, {st['malformed_reasks']} re-ask on malformed or incomplete JSON, {st['timeout_s']} s timeout, `reasoning_effort` as labelled. Every attempt is a row in the requests file with the raw answer text, the parsed verdicts, token counts (prompt, cached, completion, reasoning), the router's cost and a list-rate cost. Jev alone on the v3 run: ${s['jev_cost_per_episode']:.4f} and {s['jev_seconds_per_episode']:.1f} s per episode on average.")
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
    print(json.dumps({"markdown": str(md_path), "json": str(json_path),
                      "reproduction": summary["reproduction"],
                      "runs": {r["tag"]: {"sp_all": r["pooled"]["all"]["sentence_points"],
                                          "ceiling": r["ceiling"]["all"]["sentence_points"],
                                          "placement": r["placement"],
                                          "agree": r["agreement"]["agree_rate"]}
                               for r in summary["runs"]},
                      "trigger": summary["trigger"],
                      "total_spend_usd": summary["total_spend_usd"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share", type=float, default=0.25,
                        help="pooled routed share from the route 2 sweep (0.25 -> cutoff 0.46, "
                             "0.50 -> 0.80)")
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORTS,
                        help="gpt-5.6-luna reasoning effort")
    parser.add_argument("--episodes", nargs="+", default=None,
                        help="subset of the 18 (default: all, in run order)")
    parser.add_argument("--limit-groups", type=int, default=None,
                        help="send only the first N groups of each episode (smoke test)")
    parser.add_argument("--out", default=None,
                        help="output basename under docs/jev-real (default from cutoff and effort)")
    parser.add_argument("--run", action="store_true", help="actually call Luna")
    parser.add_argument("--resume", action="store_true",
                        help="finish a run whose files exist: skip episodes in its timing file")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                        help="hard spend cap in USD for this run")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--report", action="store_true",
                        help="write the markdown and JSON write-up from the runs on disk")
    parser.add_argument("--force", action="store_true", help="overwrite the write-up")
    parser.add_argument("--keep-rule", action="store_true",
                        help="offline: score every keep rule on the finished runs, choose on the "
                             "fit six, print fit / held-out / pooled; no calls")
    args = parser.parse_args()
    if args.keep_rule:
        return keep_rule_mode()
    if args.report:
        return report(args, parser)
    return run(args, parser)


if __name__ == "__main__":
    sys.exit(main())
