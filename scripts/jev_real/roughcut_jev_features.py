"""Build B of round two: the prompt-breakup feature pass, Jev only.

Asks one feature bundle from ``scripts/jev_real/roughcut_jev_prompts.py``
(``--feature-version``: ``f1`` is 18 yes/no questions per sentence, ``f2`` is
22) over the same ``{rules, transcript, targets}`` state the v3 sentence pass
sent, and writes one row per sentence with the probabilities of yes, the free
code features, and the v3 answers joined by episode and sentence id. The
combiner that weighs them is ``scripts/jev_real/roughcut_jev_combine.py``;
this script makes no decision.

Spec: ``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``,
"Build B" and "Round two, second pass", "Step 3".

State
-----
``roughcut_jev.sentence_jobs`` builds the v3 blocks (25 targets, whole
transcript when it fits, else the token-fitted window), with the retake losers
Jev's own v3 pass cut read back off the v3 decisions file so the transcript is
the one v3 saw. The 18 questions do not fit next to a 25-target state in one
request, so each block is split into parts: as many questions per request as a
real-token model (``TOKEN_MODEL``) says fit under ``--real-cap``. Every part
carries the full state; the parts of one block answer the same 25 sentences.

Outputs (``docs/jev-real/<out>-features.jsonl``, ``-requests.jsonl``,
``-timing.json``) refuse overwrite unless ``--resume`` adds episodes to an
existing run. No network calls without ``--run``; ``--budget`` is a hard cap.

Usage::

  python scripts/jev_real/roughcut_jev_features.py --out roughcut-jev-f1-fit
  python scripts/jev_real/roughcut_jev_features.py --out roughcut-jev-f1-fit --run
  python scripts/jev_real/roughcut_jev_features.py --out roughcut-jev-f1-heldout \\
      --episodes heldout --run
  python scripts/jev_real/roughcut_jev_features.py --feature-version f2 \\
      --out roughcut-jev-f2-fit --run
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import roughcut_jev as pipeline  # noqa: E402
from roughcut_jev import (  # noqa: E402
    Budget, FIT_EPISODES, JEV_INPUT_PER_MILLION, JEV_MODEL, OUT_DIR, est_tokens,
    jsonl_append, jsonl_read, load_episode_data, median_ratio, run_pass,
    sentence_jobs, transcript_path_for,
)
from roughcut_jev_prompts import (  # noqa: E402
    FEATURE_PROMPT_VERSION, FEATURE_PROMPT_VERSIONS, feature_prompts_for,
)

HELDOUT_EPISODES = [
    "perspective-13d-critique", "hampton-5.5-crit1", "hampton-5.5-crit2",
    "hampton-5.5-crit3", "hampton-5.5-crit4", "hampton-5.5-crit5",
    "flanders-03-thematic-crit", "anatomy-30b-hamstring-crit",
    "colman-04.03-life-crit", "colman-05.02-master-studies-crit",
    "colman-06.06-species-crit", "hampton-7-conclusion", "greco-2.2-thumbnailing",
]

#: v3 decision files, in lookup order. The all-18 file is the fit six plus the
#: 12 ladder held-out episodes; greco-2.2-thumbnailing only exists in the
#: held-out file.
V3_DECISIONS = ["roughcut-jev-all18-v3", "roughcut-jev-heldout-v3"]
V3_ARM = "jev_a"
V3_FIELDS = ("score", "cut_p", "first_p_whole", "last_p_whole", "cut_retake",
             "retake_real")

#: Real input tokens the provider bills, modelled on the v3 sentence-pass log
#: (367 answered requests): ``real = EST_SCALE * est_tokens + PER_QUESTION *
#: n_questions``. The fitted intercept (-1,691) is dropped so the model errs
#: high. ``--real-cap`` is the predicted-real ceiling for one request; the
#: provider rejects a request over 64,000 real tokens with a bare 400.
TOKEN_MODEL = {"est_scale": 1.26, "per_question": 30.0}
DEFAULT_REAL_CAP = 50_000
DEFAULT_BUDGET_USD = 3.00
PASS_NAME = "features"
BLOCK_STRIDE = 100          # request block id = v3 block index * stride + part

UM_WORD = re.compile(r"^(um+|uh+|uhm+|hmm+|mm+|er+|ah+)[,.!?]*$", re.I)
TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------------
# v3 join
# ---------------------------------------------------------------------------

def load_v3(names=V3_DECISIONS):
    """``{(episode, id): jev_a row}`` from the v3 decision files, first file wins."""
    rows, sources = {}, []
    for name in names:
        path = OUT_DIR / f"{name}-decisions.jsonl"
        if not path.exists():
            continue
        sources.append(str(path))
        for row in jsonl_read(path):
            if row["arm"] != V3_ARM:
                continue
            rows.setdefault((row["episode"], row["id"]), row)
    if not rows:
        raise SystemExit(f"no v3 {V3_ARM} rows found in {names} under {OUT_DIR}")
    return rows, sources


def v3_losers(v3_rows, episode):
    """Sentences Jev's v3 retake pass cut: dropped from the transcript v3 saw."""
    return {sid for (ep, sid), row in v3_rows.items()
            if ep == episode and row.get("cut_retake")}


# ---------------------------------------------------------------------------
# code features, free
# ---------------------------------------------------------------------------

def _tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def code_features(data):
    """``{sid: {...}}`` from the corpus, the transcript and the removals cache.

    Pauses, duration and words per second use the raw word bounds (ums
    included, because that is the audio as spoken); word count and um counts
    use the same um-stripped word list the questions see.
    """
    with open(transcript_path_for(data["episode"]["corpus"]), encoding="utf-8") as handle:
        transcript = json.load(handle)
    raw_words = defaultdict(list)
    for word in transcript.get("word_segments", []):
        if word.get("sentence_id") is not None:
            raw_words[word["sentence_id"]].append(word)
    removed, _meta = pipeline.load_removals(data["name"])

    groups = data["retake_groups"]
    winners, members = {}, set()
    for group in groups.values():
        for sid in group.get("members", []):
            members.add(sid)
        if group.get("winner") is not None:
            winners[group["winner"]] = True

    order = data["order"]
    n = len(order)
    bounds = {}
    for sid in order:
        words = [w for w in raw_words.get(sid, [])
                 if isinstance(w.get("start"), (int, float))
                 and isinstance(w.get("end"), (int, float))]
        bounds[sid] = (words[0]["start"], words[-1]["end"]) if words else None

    token_sets = {sid: _tokens(data["rendered"][sid]) for sid in order}
    out = {}
    for idx, sid in enumerate(order):
        sentence = data["by_id"][sid]
        raw = raw_words.get(sid, [])
        kept = data["words"][sid]
        span = bounds[sid]
        prev_span = bounds[order[idx - 1]] if idx > 0 else None
        next_span = bounds[order[idx + 1]] if idx + 1 < n else None
        duration = (span[1] - span[0]) if span else 0.0
        confidences = [w["confidence"] for w in raw
                       if isinstance(w.get("confidence"), (int, float))]
        chain = data["chains"].get(sid)
        piece, chain_len = (0, 1)
        if chain:
            piece, chain_len = (int(x) for x in chain["piece"].split(" of "))
        text = sentence["text"]
        out[sid] = {
            "n_words": len(kept),
            "n_words_raw": len(raw),
            "pause_before": round(max(0.0, span[0] - prev_span[1]), 3)
                            if span and prev_span else 0.0,
            "pause_after": round(max(0.0, next_span[0] - span[1]), 3)
                           if span and next_span else 0.0,
            "duration_s": round(duration, 3),
            "words_per_s": round(len(raw) / duration, 3) if duration > 0 else 0.0,
            "position": round(idx / (n - 1), 4) if n > 1 else 0.0,
            "is_retake": int(bool(sentence.get("is_retake"))),
            "retake_member": int(sid in members),
            "retake_winner": int(sid in winners),
            "trail_off": int(text.rstrip().endswith(pipeline.SPLIT_MARK)),
            "lower_start": int(pipeline._continues_previous(text)),
            "chain_piece": piece,
            "chain_len": chain_len,
            "um_detected": sum(1 for w in raw if UM_WORD.match(w.get("text", ""))),
            "um_removed": sum(1 for w in raw if w["id"] in removed),
            "asr_confidence": round(statistics.fmean(confidences), 4) if confidences else 0.0,
            "overlap_prev": round(_jaccard(token_sets[sid], token_sets[order[idx - 1]]), 4)
                            if idx > 0 else 0.0,
            "overlap_next": round(_jaccard(token_sets[sid], token_sets[order[idx + 1]]), 4)
                            if idx + 1 < n else 0.0,
        }
    return out


CODE_FEATURE_KEYS = [
    "n_words", "n_words_raw", "pause_before", "pause_after", "duration_s",
    "words_per_s", "position", "is_retake", "retake_member", "retake_winner",
    "trail_off", "lower_start", "chain_piece", "chain_len", "um_detected",
    "um_removed", "asr_confidence", "overlap_prev", "overlap_next",
]


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def predicted_real_tokens(state, questions):
    est = est_tokens(state) + est_tokens(questions)
    return TOKEN_MODEL["est_scale"] * est + TOKEN_MODEL["per_question"] * len(questions)


def _part_questions(bundle, keys, n_targets):
    questions = {}
    for k in range(n_targets):
        for key, template in bundle.questions:
            if key in keys:
                questions[f"{key}_{k}"] = {"type": "noul",
                                          "instructions": template.format(k=k)}
    return questions


def pack_questions(bundle, state, n_targets, real_cap):
    """Question keys per part: as many whole questions per request as fit the cap.

    A question is all ``n_targets`` nouls for one key; questions are never
    split across parts, so every part answers every target for its keys. A
    single question that alone breaks the cap still gets its own part (the
    provider, not this estimate, has the last word).
    """
    parts, current = [], []
    for key, _template in bundle.questions:
        trial = current + [key]
        if current and predicted_real_tokens(
                state, _part_questions(bundle, trial, n_targets)) > real_cap:
            parts.append(current)
            current = [key]
        else:
            current = trial
    if current:
        parts.append(current)
    return parts


def feature_jobs(data, bundle, losers, block_size, context_tokens, real_cap,
                 only_blocks=None):
    """One job per (v3 block, part). The state is the v3 sentence job's state."""
    jobs = []
    for v3_job in sentence_jobs(data, losers, block_size, context_tokens):
        block = v3_job["block"]
        if only_blocks is not None and block not in only_blocks:
            continue
        target_ids = v3_job["target_ids"]
        parts = pack_questions(bundle, v3_job["state"], len(target_ids), real_cap)
        if len(parts) >= BLOCK_STRIDE:
            raise AssertionError(f"{data['name']} block {block}: {len(parts)} parts")
        for part, keys in enumerate(parts):
            questions = _part_questions(bundle, keys, len(target_ids))
            job = {"pass": PASS_NAME, "episode": data["name"],
                   "block": block * BLOCK_STRIDE + part,
                   "block_index": block, "part": part, "keys": keys,
                   "ids": list(target_ids), "target_ids": list(target_ids),
                   "window_used": v3_job["window_used"],
                   "window_sentences": v3_job["window_sentences"],
                   "window_radius": v3_job["window_radius"],
                   "block_size": v3_job["block_size"],
                   "state": v3_job["state"], "questions": questions}
            fallback = v3_job["window_job"]
            job["window_job"] = dict(job, state=fallback["state"], window_used=True,
                                     window_sentences=fallback["window_sentences"],
                                     window_radius=fallback["window_radius"],
                                     window_job=None)
            jobs.append(job)
    return jobs


def feature_results(data, bundle, jobs, answers_by_block):
    """``{sid: {key: p_yes}}`` plus warnings for every unanswered (sid, key)."""
    out = {sid: {key: None for key, _t in bundle.questions} for sid in data["order"]}
    warnings = []
    for job in jobs:
        answers = answers_by_block.get(job["block"])
        for k, sid in enumerate(job["target_ids"]):
            for key in job["keys"]:
                answer = (answers or {}).get(f"{key}_{k}")
                if answer is None:
                    warnings.append(f"{data['name']} sentence {sid}: no {key} answer "
                                    f"(block {job['block_index']} part {job['part']})")
                    continue
                out[sid][key] = answer.get("noul")
    return out, warnings


# ---------------------------------------------------------------------------
# plan and run
# ---------------------------------------------------------------------------

def episode_plan(data, jobs):
    est = sum(est_tokens(j["state"]) + est_tokens(j["questions"]) for j in jobs)
    real = sum(predicted_real_tokens(j["state"], j["questions"]) for j in jobs)
    blocks = sorted({j["block_index"] for j in jobs})
    parts = [sum(1 for j in jobs if j["block_index"] == b) for b in blocks]
    return {"episode": data["name"], "sentences": len(data["order"]),
            "blocks": len(blocks), "requests": len(jobs),
            "parts_per_block_min": min(parts) if parts else 0,
            "parts_per_block_max": max(parts) if parts else 0,
            "windowed": sum(1 for j in jobs if j["window_used"]),
            "est_input_tokens": round(est),
            "predicted_real_tokens": round(real),
            "predicted_real_max": round(max(
                (predicted_real_tokens(j["state"], j["questions"]) for j in jobs),
                default=0)),
            "est_cost_usd": round(real * JEV_INPUT_PER_MILLION / 1e6, 4),
            "window_sentences_min": min((j["window_sentences"] for j in jobs), default=0),
            "window_sentences_max": max((j["window_sentences"] for j in jobs), default=0)}


def v3_seconds(episode):
    """The v3 run's wall clock for one episode, so the write-up can add it on."""
    for name in V3_DECISIONS:
        path = OUT_DIR / f"{name}-timing.json"
        if not path.exists():
            continue
        entry = json.loads(path.read_text(encoding="utf-8")).get("episodes", {}).get(episode)
        if entry:
            return {"source": str(path),
                    "wall_clock_s": entry.get("episode_wall_clock_s"),
                    "cost_usd": entry.get("cost_usd")}
    return None


def run_episode(data, bundle, jobs, v3_rows, args, budget, request_rows, warnings):
    rows, answers, timing = run_pass(PASS_NAME, data["name"], jobs, args.concurrency,
                                     budget)
    by_block = {j["block"]: j for j in jobs}
    for row in rows:
        job = by_block[row["block"]]
        row.update(block_index=job["block_index"], part=job["part"],
                   question_keys=job["keys"], prompt_version=bundle.version)
    request_rows.extend(rows)

    probs, warn = feature_results(data, bundle, jobs, answers)
    warnings.extend(warn)
    code = code_features(data)
    covered = sorted({sid for j in jobs for sid in j["target_ids"]},
                     key=data["index_of"].__getitem__)
    feature_rows = []
    for sid in covered:
        v3 = v3_rows.get((data["name"], sid))
        if v3 is None:
            warnings.append(f"{data['name']} sentence {sid}: no v3 row to join")
        feature_rows.append({
            "episode": data["name"], "id": sid,
            "feature_version": bundle.version,
            "q": probs[sid],
            "code": code[sid],
            "v3": {f: (v3.get(f) if v3 else None) for f in V3_FIELDS},
            "text": data["rendered"][sid],
        })

    blocks = sorted({j["block_index"] for j in jobs})
    timing.update({
        "sentences": len(covered), "blocks": len(blocks),
        "parts_per_block": {str(b): sum(1 for j in jobs if j["block_index"] == b)
                            for b in blocks},
        "unanswered_cells": sum(1 for r in feature_rows
                                for p in r["q"].values() if p is None),
        "v3_run": v3_seconds(data["name"]),
        "removals": data["removals"],
    })
    timing["episode_wall_clock_s"] = timing["wall_clock_s"]
    return feature_rows, timing


def existing_run(paths):
    """Episodes already complete in an output set, for ``--resume``."""
    features_path, requests_path, timing_path = paths
    if not timing_path.exists():
        return set(), None
    summary = json.loads(timing_path.read_text(encoding="utf-8"))
    return set(summary.get("episodes", {})), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", nargs="+", default=FIT_EPISODES,
                        help="episode ids (default: the six fit episodes; "
                             "'heldout' expands to the 13 held-out episodes)")
    parser.add_argument("--out", required=True,
                        help="output basename under docs/jev-real, e.g. "
                             "roughcut-jev-f1-fit")
    parser.add_argument("--run", action="store_true", help="actually call Jev")
    parser.add_argument("--resume", action="store_true",
                        help="append episodes missing from an existing run")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                        help="hard spend cap in USD for this invocation")
    parser.add_argument("--feature-version", default=FEATURE_PROMPT_VERSION,
                        choices=FEATURE_PROMPT_VERSIONS)
    parser.add_argument("--block", type=int, default=pipeline.TARGETS_PER_REQUEST)
    parser.add_argument("--context-tokens", type=int,
                        default=pipeline.DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--real-cap", type=int, default=DEFAULT_REAL_CAP,
                        help="predicted real input tokens per request "
                             f"(default {DEFAULT_REAL_CAP}; provider limit 64,000)")
    parser.add_argument("--concurrency", type=int, default=pipeline.DEFAULT_CONCURRENCY)
    parser.add_argument("--smoke", type=int, default=None, metavar="N",
                        help="only the first N block(s) of the first episode")
    args = parser.parse_args()

    if args.episodes == ["heldout"]:
        args.episodes = list(HELDOUT_EPISODES)
    bundle = feature_prompts_for(args.feature_version)
    v3_rows, v3_sources = load_v3()

    features_path = OUT_DIR / f"{args.out}-features.jsonl"
    requests_path = OUT_DIR / f"{args.out}-requests.jsonl"
    timing_path = OUT_DIR / f"{args.out}-timing.json"
    paths = (features_path, requests_path, timing_path)

    done, previous = (existing_run(paths) if args.resume else (set(), None))
    todo = [e for e in args.episodes if e not in done]
    only_blocks = set(range(args.smoke)) if args.smoke else None
    if args.smoke:
        todo = todo[:1]

    plan, episodes_jobs = [], {}
    for name in todo:
        data = load_episode_data(name)
        jobs = feature_jobs(data, bundle, v3_losers(v3_rows, name), args.block,
                            args.context_tokens, args.real_cap, only_blocks)
        episodes_jobs[name] = (data, jobs)
        plan.append(episode_plan(data, jobs))
    totals = {"episodes": len(plan),
              "requests": sum(p["requests"] for p in plan),
              "sentences": sum(p["sentences"] for p in plan),
              "predicted_real_tokens": sum(p["predicted_real_tokens"] for p in plan),
              "est_cost_usd": round(sum(p["est_cost_usd"] for p in plan), 4)}

    if not args.run:
        print(json.dumps({
            "mode": "plan", "model": JEV_MODEL, "feature_version": bundle.version,
            "questions": [k for k, _t in bundle.questions],
            "episodes": todo, "already_done": sorted(done),
            "token_model": TOKEN_MODEL, "real_cap": args.real_cap,
            "block": args.block, "context_tokens": args.context_tokens,
            "budget_cap_usd": args.budget, "v3_sources": v3_sources,
            "plan": plan, "totals": totals,
            "outputs": [str(p) for p in paths],
        }, indent=2))
        return 0

    if totals["est_cost_usd"] > args.budget:
        parser.error(f"estimated ${totals['est_cost_usd']:.4f} exceeds the "
                     f"--budget cap ${args.budget:.2f}; not sending")
    existing = [str(p) for p in paths if p.exists()]
    if existing and not args.resume:
        parser.error(f"refusing to overwrite existing output(s): {existing}")
    if not todo:
        print("nothing to do: every episode is already in the run", file=sys.stderr)
        return 0

    import skell_e_router  # noqa: F401  (import before the clock starts)

    budget = Budget(args.budget)
    request_rows, feature_rows, warnings = [], [], []
    timings, started, aborted = {}, time.perf_counter(), None
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name in todo:
        data, jobs = episodes_jobs[name]
        rows, timing = run_episode(data, bundle, jobs, v3_rows, args, budget,
                                   request_rows, warnings)
        for row in rows:
            row["recorded_utc"] = stamp
        # Write as each episode lands so a run split across calls loses nothing.
        jsonl_append(features_path, rows)
        new_requests = [r for r in request_rows if r["episode"] == name]
        jsonl_append(requests_path, new_requests)
        feature_rows.extend(rows)
        timings[name] = timing
        print(f"{name}: {timing['wall_clock_s']}s, ${timing['cost_usd']:.4f}, "
              f"{timing['requests']} requests, {timing['errors']} errors, "
              f"{timing['failed_blocks']} failed parts, "
              f"{timing['unanswered_cells']} unanswered cells, "
              f"est/actual {timing['est_over_actual_median']}", file=sys.stderr)
        if budget.blocked():
            aborted = (f"budget cap ${args.budget:.2f} crossed at ${budget.spent:.4f}; "
                       f"stopped after {name}")
            print(f"ABORT: {aborted}", file=sys.stderr)
            break

    all_timings = dict((previous or {}).get("episodes", {}), **timings)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": JEV_MODEL, "feature_version": bundle.version,
        "questions": [k for k, _t in bundle.questions],
        "code_features": CODE_FEATURE_KEYS, "v3_fields": list(V3_FIELDS),
        "v3_sources": v3_sources,
        "concurrency": args.concurrency, "block": args.block,
        "context_tokens": args.context_tokens, "real_cap": args.real_cap,
        "token_model": TOKEN_MODEL, "smoke": args.smoke,
        "budget_cap_usd": args.budget, "aborted": aborted,
        "episodes": all_timings,
        "invocations": (previous or {}).get("invocations", []) + [{
            "recorded_utc": stamp, "episodes": list(timings),
            "wall_clock_s": round(time.perf_counter() - started, 3),
            "cost_usd": round(sum(r["cost"] or 0.0 for r in request_rows), 6),
        }],
        "totals": {
            "episodes": len(all_timings),
            "wall_clock_s": round(sum(t["wall_clock_s"] for t in all_timings.values()), 3),
            "requests": sum(t["requests"] for t in all_timings.values()),
            "errors": sum(t["errors"] for t in all_timings.values()),
            "failed_parts": sum(t["failed_blocks"] for t in all_timings.values()),
            "input_tokens": sum(t["input_tokens"] for t in all_timings.values()),
            "output_tokens": sum(t["output_tokens"] for t in all_timings.values()),
            "cost_usd": round(sum(t["cost_usd"] for t in all_timings.values()), 6),
            "sentences": sum(t["sentences"] for t in all_timings.values()),
            "unanswered_cells": sum(t["unanswered_cells"] for t in all_timings.values()),
            "est_over_actual_median": median_ratio(request_rows),
        },
        "warnings": ((previous or {}).get("warnings", []) + warnings)[:200],
        "n_warnings": (previous or {}).get("n_warnings", 0) + len(warnings),
    }
    timing_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"totals": summary["totals"], "aborted": aborted}, indent=2))
    for warning in warnings[:20]:
        print(f"warning: {warning}", file=sys.stderr)
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
