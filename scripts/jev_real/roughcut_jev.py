"""Jev-only rough cut for the solar-sailer benchmark.

Implements ``docs/superpowers/specs/2026-09-20-jev-roughcut-design.md``: three
Jev passes per episode (retakes, sentence scores plus head/tail trim, trim
pick), then a decision row per sentence per arm so every arm can be rescored
offline by ``scripts/jev_real/roughcut_partial_scoring.py``.

Passes
------
step 0  removals cache (``docs/jev-real/removals/<episode>.json``, built by the
        scoring module from the shipping Um Removal detector). Those word ids
        are stripped from every transcript rendering Jev sees. Missing cache =
        loud warning and no stripping.
step 1  retake pass: one ``take_k`` choice and one ``real_k`` noul per retake
        group, 6 groups per request.
step 2  sentence pass: ``score_k`` (0-5), ``first_k`` and ``last_k`` word
        choices, 25 target sentences per request, state carries the rules and
        the whole transcript (windowed to +/-200 sentences when the render is
        over 24,000 estimated tokens or the provider rejects the context).
step 3  trim pick pass: for sentences where ``first_k``/``last_k`` put less than
        ``--t-trim`` on ``whole``, a ``pick_k`` choice over candidate versions,
        10 items per request.

Arms written to the decisions file: jev_a, jev_b, jev_b_moduleretakes,
jev_b_notrim. The ``_mod`` arms (um removal and delete silence layered on) are a
scoring-time option, not separate decisions.

READ-ONLY against solar-sailer: the harness, corpus and prompts are read, never
written. No network calls without ``--run``.
"""

import argparse
import concurrent.futures as futures
import importlib.util
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH_DIR = Path(r"D:\solar-sailer\benchmarks\roughcut")
EPISODES_DIR = BENCH_DIR / "episodes"
OUT_DIR = ROOT / "docs" / "jev-real"
REMOVALS_DIR = OUT_DIR / "removals"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
# The harness package must resolve ahead of this directory: solar-sailer's
# ``roughcut_bench`` package and our sibling runner share a name.
sys.path.insert(0, str(BENCH_DIR))

from roughcut_jev_prompts import (  # noqa: E402
    CUT_VERSION_DESCRIPTION, FIRST_INSTRUCTIONS, LAST_INSTRUCTIONS,
    PICK_INSTRUCTIONS, PROMPT_VERSION, REAL_INSTRUCTIONS, RULES,
    SCORE_INSTRUCTIONS, SCORE_LEVELS, TAKE_INSTRUCTIONS,
    WHOLE_FIRST_DESCRIPTION, WHOLE_LAST_DESCRIPTION, WHOLE_VERSION_DESCRIPTION,
    check_rules_match,
)


def _load_runner():
    """The existing benchmark runner, loaded under a non-colliding name.

    Importing it as ``roughcut_bench`` would shadow (and be shadowed by) the
    harness package of the same name, so it is loaded from its path. Its
    module-level work is wanted: the cache-only answer-key monkeypatch and the
    harness imports.
    """
    spec = importlib.util.spec_from_file_location(
        "roughcut_jev_runner", str(HERE / "roughcut_bench.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["roughcut_jev_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
episodes_mod = runner.episodes_mod
jsonl_append = runner.jsonl_append
jsonl_read = runner.jsonl_read
blocks = runner.blocks
latency_summary = runner.latency_summary

from roughcut_bench.partial import transcript_path_for  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

JEV_MODEL = "jev-1.13.0"
JEV_TIMEOUT = 180
JEV_INPUT_PER_MILLION = 0.042
DEFAULT_CONCURRENCY = 8
DEFAULT_BUDGET_USD = 3.00
DEFAULT_T_TRIM = 0.5
DEFAULT_OUT = "roughcut-jev"

FIT_EPISODES = [
    "colman-02.04-skeleton-demo",
    "hampton-5.4-assignment-demo",
    "colman-03.03-muscles-crit",
    "edges-7.01-intro",
    "hampton-5.2-shape-demo",
    "perspective-14e-boxes-critique",
]

GROUPS_PER_REQUEST = 6
TARGETS_PER_REQUEST = 25
ITEMS_PER_REQUEST = 10
CONTEXT_SENTENCES = 5          # before/after a retake group or a pick item
WINDOW_SENTENCES = 200         # either side of a target block when windowed
TOKEN_CAP = 24_000             # rendered transcript + rules, 4 chars per token
CHARS_PER_TOKEN = 4
PAUSE_S = 0.5
LONG_PAUSE_S = 1.5
REAL_RETAKE_CUT = 0.5          # real_k below this: the group is not a retake
LAST_TAKE_MARGIN = 0.15        # top minus last take below this: the last wins
TRIM_CANDIDATES = 3            # top starts and top ends crossed in step 3
MAX_CHOICE_OPTIONS = 255

PASSES = ["retake", "sentence", "trim_pick"]
ARMS = ["jev_a", "jev_b", "jev_b_moduleretakes", "jev_b_notrim"]
REMOVALS_KEYS = ("umm_word_ids", "um_word_ids", "removed_word_ids", "word_ids",
                 "umm", "ids")
CONTEXT_ERROR_MARKS = ("context", "too long", "too large", "token", "422")


# ---------------------------------------------------------------------------
# step 0: data
# ---------------------------------------------------------------------------

def load_removals(episode_name):
    """``(word_id_set, meta)`` from the um-removal cache, or nothing plus a shout."""
    path = REMOVALS_DIR / f"{episode_name}.json"
    meta = {"path": str(path), "loaded": False, "key": None, "n_word_ids": 0}
    if not path.exists():
        print(
            f"WARNING: no um-removal cache at {path}. Running with NO um stripping: "
            f"Jev sees ums the production module already cuts, so these numbers are "
            f"not the designed arm.", file=sys.stderr)
        return set(), meta
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    ids, key = None, None
    if isinstance(data, list):
        ids, key = data, "<list>"
    elif isinstance(data, dict):
        for candidate in REMOVALS_KEYS:
            value = data.get(candidate)
            if isinstance(value, list):
                ids, key = value, candidate
                break
    if ids is None:
        raise ValueError(
            f"{path}: no word-id list found (looked for {REMOVALS_KEYS}); keys="
            f"{sorted(data) if isinstance(data, dict) else type(data).__name__}")
    meta.update(loaded=True, key=key, n_word_ids=len(ids))
    return {int(i) for i in ids}, meta


def load_episode_data(episode_name, removals_dir_used=True):
    """Everything the passes need for one episode, corpus and transcript only."""
    episode = episodes_mod.load_episode(episode_name, str(EPISODES_DIR))
    sentences = episodes_mod.load_corpus(episode)
    with open(transcript_path_for(episode["corpus"]), encoding="utf-8") as handle:
        transcript = json.load(handle)
    with open(os.path.join(os.path.dirname(episode["corpus"]), "retakes.json"),
              encoding="utf-8") as handle:
        retakes = json.load(handle)

    removed, removals_meta = (load_removals(episode_name) if removals_dir_used
                              else (set(), {"loaded": False}))

    words_by_sentence = defaultdict(list)
    for word in transcript.get("word_segments", []):
        sid = word.get("sentence_id")
        if sid is None:
            continue
        words_by_sentence[sid].append(word)

    order = [s["id"] for s in sentences]
    index_of = {sid: i for i, sid in enumerate(order)}
    rendered, kept_words, tokens_by_sentence = {}, {}, {}
    for sentence in sentences:
        sid = sentence["id"]
        tokens = _render_tokens(words_by_sentence.get(sid, []), removed)
        tokens_by_sentence[sid] = tokens
        kept_words[sid] = [t for t in tokens if "w" in t]
        rendered[sid] = " ".join(t["t"] for t in tokens)
        if not kept_words[sid]:
            # Every word of the sentence is an um the module already cut. Keep the
            # original text visible so the line still reads, but leave the word
            # list empty so no trim question is asked.
            rendered[sid] = sentence["text"]

    return {
        "name": episode_name,
        "episode": episode,
        "sentences": sentences,
        "by_id": {s["id"]: s for s in sentences},
        "order": order,
        "index_of": index_of,
        "tokens": tokens_by_sentence,
        "words": kept_words,
        "rendered": rendered,
        "retake_groups": retakes.get("groups", {}),
        "removals": removals_meta,
    }


def _render_tokens(words, removed):
    """Surviving words plus ``<pause>`` markers, in order.

    A pause marker sits between two surviving words whose silence is at least
    ``PAUSE_S`` (1.5 s gets ``<long pause>``). Time taken up by words the um
    module removes is subtracted from the gap, because that audio is gone from
    the cut Jev is reasoning about.
    """
    out = []
    prev_end, removed_span = None, 0.0
    for word in words:
        start, end = word.get("start"), word.get("end")
        if word["id"] in removed:
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                removed_span += max(0.0, end - start)
            continue
        if prev_end is not None and isinstance(start, (int, float)):
            gap = start - prev_end - removed_span
            if gap >= LONG_PAUSE_S:
                out.append({"t": "<long pause>"})
            elif gap >= PAUSE_S:
                out.append({"t": "<pause>"})
        out.append({"w": word["id"], "t": word["text"]})
        if isinstance(end, (int, float)):
            prev_end = end
        removed_span = 0.0
    return out


def sentence_line(data, sid):
    return f"{sid} = {data['rendered'][sid]}"


def context_lines(data, sid, before=True, n=CONTEXT_SENTENCES):
    idx = data["index_of"][sid]
    lo, hi = (max(0, idx - n), idx) if before else (idx + 1, idx + 1 + n)
    return [sentence_line(data, s) for s in data["order"][lo:hi]]


def transcript_lines(data, drop_ids):
    return [sentence_line(data, sid) for sid in data["order"] if sid not in drop_ids]


def _describe(text):
    """Criteria descriptions must be non-empty strings."""
    text = (text or "").strip()
    return text if text else "(blank)"


def est_tokens(obj):
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return len(text) / CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# request plumbing
# ---------------------------------------------------------------------------

class Budget:
    """Hard spend cap. Recorded cost only; a request with no usage counts 0."""

    def __init__(self, cap):
        self.cap = cap
        self.spent = 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def add(self, cost):
        with self._lock:
            self.spent += cost or 0.0
            if self.spent >= self.cap:
                self.stopped = True
            return self.spent

    def blocked(self):
        return self.stopped


def _request(job, budget, attempt):
    """One classify call. Returns ``(row, answers_or_None)``; never raises."""
    from skell_e_router import classify

    row = {"pass": job["pass"], "episode": job["episode"], "block": job["block"],
           "ids": job["ids"], "n_questions": len(job["questions"]),
           "provider_model": None, "input_tokens": None, "output_tokens": None,
           "cost": None, "elapsed_s": None, "attempt": attempt, "error": None,
           "prompt_version": PROMPT_VERSION, "window_used": bool(job.get("window_used")),
           "est_input_tokens": round(est_tokens(job["state"]) + est_tokens(job["questions"]))}
    if budget.blocked():
        row["error"] = "BUDGET_STOP: cap reached before this request was sent"
        row["elapsed_s"] = 0.0
        return row, None

    started = time.perf_counter()
    answers = None
    try:
        response = classify(JEV_MODEL, job["state"], job["questions"],
                            timeout=JEV_TIMEOUT)
        answers = response.answers
        row.update(provider_model=response.model, input_tokens=response.input_tokens,
                   output_tokens=response.output_tokens, cost=response.cost)
        budget.add(response.cost)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        row["error_details"] = str(getattr(exc, "details", None))[:300]
    row["elapsed_s"] = time.perf_counter() - started
    return row, answers


def _is_context_error(row):
    blob = f"{row.get('error')} {row.get('error_details')}".lower()
    return any(mark in blob for mark in CONTEXT_ERROR_MARKS)


def run_pass(pass_name, episode_name, jobs, concurrency, budget):
    """Dispatch one pass, retry failures once, return rows, answers and timing.

    Wall clock is measured around the whole pass, submit to last result, at the
    concurrency used, retry included: that is the number a per-episode latency
    column has to report.
    """
    request_rows, answers_by_block = [], {}
    started = time.perf_counter()
    if jobs:
        with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {pool.submit(_request, job, budget, 1): job for job in jobs}
            for future in futures.as_completed(pending):
                row, answers = future.result()
                request_rows.append(row)
                answers_by_block[pending[future]["block"]] = answers

        # ~2.5% of requests come back a bare PROVIDER_ERROR that succeeds on a
        # re-request. One retry per failed block; both attempts stay in the log.
        retry_jobs = []
        for job in jobs:
            if answers_by_block.get(job["block"]) is not None:
                continue
            failed = [r for r in request_rows if r["block"] == job["block"]]
            if failed and _is_context_error(failed[0]) and job.get("window_job"):
                retry_jobs.append(job["window_job"])
            else:
                retry_jobs.append(job)
        if retry_jobs and not budget.blocked():
            with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending = {pool.submit(_request, job, budget, 2): job
                           for job in retry_jobs}
                for future in futures.as_completed(pending):
                    row, answers = future.result()
                    request_rows.append(row)
                    if answers is not None:
                        answers_by_block[pending[future]["block"]] = answers
    wall = time.perf_counter() - started

    request_rows.sort(key=lambda r: (r["block"], r["attempt"]))
    timing = {
        "wall_clock_s": round(wall, 3),
        "requests": len(request_rows),
        "input_tokens": sum(r["input_tokens"] or 0 for r in request_rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in request_rows),
        "cost_usd": round(sum(r["cost"] or 0.0 for r in request_rows), 6),
        "errors": sum(1 for r in request_rows if r["error"]),
        "unanswered_blocks": sum(1 for j in jobs
                                 if answers_by_block.get(j["block"]) is None),
        "windowed_requests": sum(1 for r in request_rows if r["window_used"]),
        "latency": latency_summary([r["elapsed_s"] for r in request_rows]),
    }
    return request_rows, answers_by_block, timing


# ---------------------------------------------------------------------------
# step 1: retake pass
# ---------------------------------------------------------------------------

def retake_group_list(data):
    """``[(group_id, members)]`` in corpus order, groups of two or more."""
    out = []
    for gid, group in data["retake_groups"].items():
        members = [m for m in group.get("members", []) if m in data["by_id"]]
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda sid: data["index_of"][sid])
        out.append((str(gid), members))
    out.sort(key=lambda item: data["index_of"][item[1][0]])
    return out


def retake_jobs(data):
    jobs = []
    for block, batch in enumerate(blocks(retake_group_list(data), GROUPS_PER_REQUEST)):
        state, questions, covered, mapping = {"groups": {}}, {}, [], {}
        for k, (gid, members) in enumerate(batch):
            keys = [f"take{i + 1}" for i in range(len(members))]
            state["groups"][str(k)] = {
                "takes": {key: sentence_line(data, sid)
                          for key, sid in zip(keys, members)},
                "before": context_lines(data, members[0], before=True),
                "after": context_lines(data, members[-1], before=False),
            }
            questions[f"take_{k}"] = {
                "type": "choice",
                "instructions": TAKE_INSTRUCTIONS.format(k=k),
                "criteria": {key: _describe(data["rendered"][sid])
                             for key, sid in zip(keys, members)},
            }
            questions[f"real_{k}"] = {
                "type": "noul",
                "instructions": REAL_INSTRUCTIONS.format(k=k),
            }
            mapping[k] = (gid, members, keys)
            covered.extend(members)
        jobs.append({"pass": "retake", "episode": data["name"], "block": block,
                     "ids": covered, "state": state, "questions": questions,
                     "mapping": mapping})
    return jobs


def retake_decisions(data, jobs, answers_by_block):
    """``(per_sentence_info, losers, warnings)`` from the retake answers."""
    info, losers, warnings = {}, set(), []
    for job in jobs:
        answers = answers_by_block.get(job["block"])
        for k, (gid, members, keys) in job["mapping"].items():
            row = {"retake_group": gid, "retake_members": members,
                   "retake_real": None, "retake_take_probs": None,
                   "retake_choice": None, "retake_winner": None,
                   "retake_source": "jev"}
            if not answers or f"take_{k}" not in answers:
                module_winner = data["retake_groups"][gid].get("winner")
                row.update(retake_source="module_fallback", retake_winner=module_winner)
                warnings.append(
                    f"{data['name']} group {gid}: no retake answer, fell back to the "
                    f"module winner {module_winner}")
                if module_winner is not None:
                    losers.update(m for m in members if m != module_winner)
            else:
                take = answers[f"take_{k}"]
                real = (answers.get(f"real_{k}") or {}).get("noul")
                probs = take["probabilities"]
                row.update(retake_real=real, retake_take_probs=probs,
                           retake_choice=take["choice"])
                if real is not None and real < REAL_RETAKE_CUT:
                    row["retake_winner"] = None  # not a retake, nothing is cut
                else:
                    top = take["choice"]
                    last = keys[-1]
                    if probs[top] - probs[last] < LAST_TAKE_MARGIN:
                        top = last
                    winner = members[keys.index(top)]
                    row["retake_winner"] = winner
                    losers.update(m for m in members if m != winner)
            for sid in members:
                info[sid] = row
    return info, losers, warnings


# ---------------------------------------------------------------------------
# step 2: sentence pass
# ---------------------------------------------------------------------------

def _target(data, sid):
    return {"id": sid, "words": [dict(t) for t in data["tokens"][sid]]}


def _trim_criteria(words, whole_description, from_start):
    options = words if from_start else list(reversed(words))
    if len(options) > MAX_CHOICE_OPTIONS - 1:
        options = options[:MAX_CHOICE_OPTIONS - 1]
    if not from_start:
        options = list(reversed(options))
    criteria = {str(w["w"]): _describe(w["t"]) for w in options}
    criteria["whole"] = whole_description
    return criteria


def sentence_jobs(data, drop_ids, force_window=False):
    """One job per block of 25 target sentences, with a windowed fallback job."""
    kept_order = [sid for sid in data["order"] if sid not in drop_ids]
    full_lines = [sentence_line(data, sid) for sid in kept_order]
    over_cap = est_tokens(full_lines) + est_tokens(RULES) > TOKEN_CAP
    windowed = force_window or over_cap

    jobs = []
    for block, target_ids in enumerate(blocks(data["order"], TARGETS_PER_REQUEST)):
        questions, targets = {}, []
        for k, sid in enumerate(target_ids):
            targets.append(_target(data, sid))
            questions[f"score_{k}"] = {
                "type": "score",
                "instructions": SCORE_INSTRUCTIONS.format(k=k),
                "criteria": list(SCORE_LEVELS),
            }
            words = data["words"][sid]
            if len(words) >= 2:
                questions[f"first_{k}"] = {
                    "type": "choice",
                    "instructions": FIRST_INSTRUCTIONS.format(k=k),
                    "criteria": _trim_criteria(words, WHOLE_FIRST_DESCRIPTION, True),
                }
                questions[f"last_{k}"] = {
                    "type": "choice",
                    "instructions": LAST_INSTRUCTIONS.format(k=k),
                    "criteria": _trim_criteria(words, WHOLE_LAST_DESCRIPTION, False),
                }

        def make(window):
            if window:
                lo = max(0, data["index_of"][target_ids[0]] - WINDOW_SENTENCES)
                hi = min(len(data["order"]),
                         data["index_of"][target_ids[-1]] + WINDOW_SENTENCES + 1)
                keep = set(data["order"][lo:hi])
                lines = [line for sid, line in zip(kept_order, full_lines)
                         if sid in keep]
            else:
                lines = full_lines
            return {"pass": "sentence", "episode": data["name"], "block": block,
                    "ids": list(target_ids), "window_used": window,
                    "state": {"rules": RULES, "transcript": lines, "targets": targets},
                    "questions": questions, "target_ids": list(target_ids)}

        job = make(windowed)
        if not windowed:
            job["window_job"] = make(True)
        jobs.append(job)
    return jobs


def sentence_results(data, jobs, answers_by_block):
    """``{sid: {...raw answer fields...}}`` plus warnings."""
    out, warnings = {}, []
    for job in jobs:
        answers = answers_by_block.get(job["block"])
        for k, sid in enumerate(job["target_ids"]):
            row = {"score": None, "score_probabilities": None, "defaulted": True,
                   "first_choice": None, "first_p_whole": None,
                   "last_choice": None, "last_p_whole": None,
                   "first_probabilities": None, "last_probabilities": None}
            if answers and f"score_{k}" in answers:
                score = answers[f"score_{k}"]
                row.update(score=score["score"],
                           score_probabilities=score["probabilities"],
                           defaulted=False)
                first, last = answers.get(f"first_{k}"), answers.get(f"last_{k}")
                if first:
                    row.update(first_choice=first["choice"],
                               first_p_whole=first["probabilities"].get("whole"),
                               first_probabilities=first["probabilities"])
                if last:
                    row.update(last_choice=last["choice"],
                               last_p_whole=last["probabilities"].get("whole"),
                               last_probabilities=last["probabilities"])
            else:
                warnings.append(f"{data['name']} sentence {sid}: no answer "
                                f"(block {job['block']})")
            out[sid] = row
    return out, warnings


# ---------------------------------------------------------------------------
# step 3: trim pick pass
# ---------------------------------------------------------------------------

def _span_text(data, sid, first_id, last_id):
    words = data["words"][sid]
    ids = [w["w"] for w in words]
    lo, hi = ids.index(first_id), ids.index(last_id)
    return " ".join(w["t"] for w in words[lo:hi + 1])


def _top_keys(probabilities, n):
    return [k for k, _ in sorted(probabilities.items(), key=lambda kv: -kv[1])[:n]]


def pick_candidates(data, sid, result):
    """``{version_key: (first_word_id, last_word_id) or None}`` for one sentence.

    ``whole`` maps to None, ``cut`` is added by the caller. Candidate versions
    are the top three starts crossed with the top three ends, dropping any pair
    whose start sits after its end and any span that is the whole sentence.
    """
    words = data["words"][sid]
    ids = [w["w"] for w in words]
    starts = _top_keys(result["first_probabilities"] or {}, TRIM_CANDIDATES)
    ends = _top_keys(result["last_probabilities"] or {}, TRIM_CANDIDATES)
    versions, seen = {}, set()
    for start in starts:
        for end in ends:
            lo = 0 if start == "whole" else ids.index(int(start))
            hi = len(ids) - 1 if end == "whole" else ids.index(int(end))
            if lo > hi or (lo == 0 and hi == len(ids) - 1):
                continue
            span = (ids[lo], ids[hi])
            if span in seen:
                continue
            seen.add(span)
            versions[f"v{len(versions) + 1}"] = span
    return versions


def pick_jobs(data, results, t_trim):
    items = []
    for sid in data["order"]:
        result = results.get(sid) or {}
        if result.get("defaulted") or result.get("first_probabilities") is None:
            continue
        p_first, p_last = result["first_p_whole"], result["last_p_whole"]
        if p_first is None or p_last is None:
            continue
        if p_first >= t_trim and p_last >= t_trim:
            continue
        versions = pick_candidates(data, sid, result)
        if not versions:
            continue
        items.append((sid, versions))

    jobs = []
    for block, batch in enumerate(blocks(items, ITEMS_PER_REQUEST)):
        state, questions, covered, mapping = {"items": {}}, {}, [], {}
        for k, (sid, versions) in enumerate(batch):
            rendered = {key: _span_text(data, sid, *span)
                        for key, span in versions.items()}
            rendered["whole"] = _describe(data["rendered"][sid])
            rendered["cut"] = CUT_VERSION_DESCRIPTION
            state["items"][str(k)] = {
                "before": context_lines(data, sid, before=True),
                "sentence": sentence_line(data, sid),
                "after": context_lines(data, sid, before=False),
                "versions": rendered,
            }
            criteria = {key: _describe(text) for key, text in rendered.items()}
            criteria["whole"] = WHOLE_VERSION_DESCRIPTION + ": " + criteria["whole"]
            questions[f"pick_{k}"] = {
                "type": "choice",
                "instructions": PICK_INSTRUCTIONS.format(k=k),
                "criteria": criteria,
            }
            mapping[k] = (sid, versions)
            covered.append(sid)
        jobs.append({"pass": "trim_pick", "episode": data["name"], "block": block,
                     "ids": covered, "state": state, "questions": questions,
                     "mapping": mapping})
    return jobs


def pick_results(data, jobs, answers_by_block):
    out, warnings = {}, []
    for job in jobs:
        answers = answers_by_block.get(job["block"])
        for k, (sid, versions) in job["mapping"].items():
            if not answers or f"pick_{k}" not in answers:
                warnings.append(f"{data['name']} sentence {sid}: no pick answer "
                                f"(block {job['block']})")
                continue
            pick = answers[f"pick_{k}"]
            choice = pick["choice"]
            span = versions.get(choice)
            out[sid] = {"pick_choice": choice,
                        "pick_probabilities": pick["probabilities"],
                        "pick_keep_words": [list(span)] if span else None,
                        "pick_cut": choice == "cut"}
    return out, warnings


# ---------------------------------------------------------------------------
# decision assembly
# ---------------------------------------------------------------------------

def variant_a_keep_words(data, sid, result):
    """Top ``first_k`` word to top ``last_k`` word; None when nothing is trimmed."""
    first, last = result.get("first_choice"), result.get("last_choice")
    if first is None or last is None or (first == "whole" and last == "whole"):
        return None
    ids = [w["w"] for w in data["words"][sid]]
    lo = 0 if first == "whole" else ids.index(int(first))
    hi = len(ids) - 1 if last == "whole" else ids.index(int(last))
    if lo > hi or (lo == 0 and hi == len(ids) - 1):
        return None
    return [[ids[lo], ids[hi]]]


def build_decisions(data, retake_info, results, picks):
    rows = []
    for sid in data["order"]:
        result = results.get(sid, {})
        retake = retake_info.get(sid, {})
        pick = picks.get(sid, {})
        score = result.get("score")
        cut_jev = bool(retake.get("retake_winner") is not None
                       and retake["retake_winner"] != sid)
        cut_module = bool(data["by_id"][sid].get("is_retake"))

        source = {
            "first_choice": result.get("first_choice"),
            "first_p_whole": result.get("first_p_whole"),
            "last_choice": result.get("last_choice"),
            "last_p_whole": result.get("last_p_whole"),
            "score_probabilities": result.get("score_probabilities"),
            "pick_choice": pick.get("pick_choice"),
            "pick_probabilities": pick.get("pick_probabilities"),
            "retake_group": retake.get("retake_group"),
            "retake_real": retake.get("retake_real"),
            "retake_take_probs": retake.get("retake_take_probs"),
            "retake_choice": retake.get("retake_choice"),
            "retake_winner": retake.get("retake_winner"),
            "retake_source": retake.get("retake_source"),
            "defaulted": result.get("defaulted", True),
            "prompt_version": PROMPT_VERSION,
        }

        keep_a = variant_a_keep_words(data, sid, result)
        score_b = 0.0 if pick.get("pick_cut") else score
        keep_b = None if pick.get("pick_cut") else pick.get("pick_keep_words")

        per_arm = {
            "jev_a": (score, keep_a, cut_jev),
            "jev_b": (score_b, keep_b, cut_jev),
            "jev_b_moduleretakes": (score_b, keep_b, cut_module),
            "jev_b_notrim": (score_b, None, cut_jev),
        }
        for arm, (arm_score, keep_words, cut_retake) in per_arm.items():
            rows.append({"arm": arm, "episode": data["name"], "id": sid,
                         "score": arm_score, "keep_words": keep_words,
                         "cut_retake": cut_retake, **source})
    return rows


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def estimate(episode_names, t_trim):
    """Requests, input tokens and spend per pass per episode, no calls.

    Passes 1 and 2 are built for real, so their numbers are the exact payloads.
    Pass 3 depends on answers that do not exist yet, so it assumes a trim rate.
    """
    assumed_trim_rate = 0.3
    plan, totals = [], {"requests": 0, "est_input_tokens": 0.0, "est_cost_usd": 0.0}
    for name in episode_names:
        data = load_episode_data(name)
        module_losers = {s["id"] for s in data["sentences"] if s.get("is_retake")}
        rows = [("retake", retake_jobs(data)),
                ("sentence", sentence_jobs(data, module_losers))]
        for pass_name, jobs in rows:
            tokens = sum(est_tokens(j["state"]) + est_tokens(j["questions"])
                         for j in jobs)
            plan.append({"episode": name, "pass": pass_name, "requests": len(jobs),
                         "sentences": len(data["order"]),
                         "windowed": sum(1 for j in jobs if j.get("window_used")),
                         "est_input_tokens": round(tokens),
                         "est_cost_usd": round(tokens * JEV_INPUT_PER_MILLION / 1e6, 4)})
        n_items = math.ceil(len(data["order"]) * assumed_trim_rate)
        n_pick = math.ceil(n_items / ITEMS_PER_REQUEST)
        chars = sum(len(data["rendered"][s]) for s in data["order"]) / max(
            1, len(data["order"]))
        pick_tokens = n_pick * (ITEMS_PER_REQUEST * (12 * chars + 600) / CHARS_PER_TOKEN)
        plan.append({"episode": name, "pass": "trim_pick", "requests": n_pick,
                     "sentences": len(data["order"]),
                     "assumed_trim_rate": assumed_trim_rate,
                     "est_input_tokens": round(pick_tokens),
                     "est_cost_usd": round(pick_tokens * JEV_INPUT_PER_MILLION / 1e6, 4)})
        for row in plan[-3:]:
            totals["requests"] += row["requests"]
            totals["est_input_tokens"] += row["est_input_tokens"]
            totals["est_cost_usd"] += row["est_cost_usd"]
    totals["est_cost_usd"] = round(totals["est_cost_usd"], 4)
    totals["est_input_tokens"] = round(totals["est_input_tokens"])
    return plan, totals


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_episode(data, args, budget, request_rows, warnings):
    timing = {"sentences": len(data["order"]), "removals": data["removals"],
              "passes": {}}

    jobs1 = retake_jobs(data)
    rows, answers1, t1 = run_pass("retake", data["name"], jobs1, args.concurrency,
                                  budget)
    request_rows.extend(rows)
    timing["passes"]["retake"] = dict(t1, groups=len(retake_group_list(data)))
    retake_info, losers, warn = retake_decisions(data, jobs1, answers1)
    warnings.extend(warn)
    timing["passes"]["retake"]["losers_cut"] = len(losers)

    jobs2 = sentence_jobs(data, losers)
    rows, answers2, t2 = run_pass("sentence", data["name"], jobs2, args.concurrency,
                                  budget)
    request_rows.extend(rows)
    timing["passes"]["sentence"] = t2
    results, warn = sentence_results(data, jobs2, answers2)
    warnings.extend(warn)

    jobs3 = pick_jobs(data, results, args.t_trim)
    rows, answers3, t3 = run_pass("trim_pick", data["name"], jobs3, args.concurrency,
                                  budget)
    request_rows.extend(rows)
    timing["passes"]["trim_pick"] = dict(t3, items=sum(len(j["mapping"])
                                                       for j in jobs3))
    picks, warn = pick_results(data, jobs3, answers3)
    warnings.extend(warn)

    decisions = build_decisions(data, retake_info, results, picks)
    timing["episode_wall_clock_s"] = round(
        sum(p["wall_clock_s"] for p in timing["passes"].values()), 3)
    timing["cost_usd"] = round(sum(p["cost_usd"] for p in timing["passes"].values()), 6)
    timing["trims"] = {
        arm: sum(1 for d in decisions if d["arm"] == arm and d["keep_words"])
        for arm in ARMS}
    return decisions, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", default=FIT_EPISODES,
                        help="episode ids (default: the six fit episodes)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output basename under docs/jev-real")
    parser.add_argument("--run", action="store_true", help="actually call Jev")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                        help="hard spend cap in USD")
    parser.add_argument("--t-trim", type=float, default=DEFAULT_T_TRIM,
                        help="P(whole) below which a sentence goes to the pick pass")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()

    requests_path = OUT_DIR / f"{args.out}-requests.jsonl"
    decisions_path = OUT_DIR / f"{args.out}-decisions.jsonl"
    timing_path = OUT_DIR / f"{args.out}-timing.json"

    if not args.run:
        plan, totals = estimate(args.episodes, args.t_trim)
        matches, _ = check_rules_match()
        print(json.dumps({
            "mode": "plan", "model": JEV_MODEL, "prompt_version": PROMPT_VERSION,
            "rules_match_source_prompt": matches,
            "episodes": args.episodes, "arms": ARMS,
            "concurrency": args.concurrency, "t_trim": args.t_trim,
            "budget_cap_usd": args.budget,
            "input_price_per_million_usd": JEV_INPUT_PER_MILLION,
            "plan": plan, "totals": totals,
            "outputs": [str(requests_path), str(decisions_path), str(timing_path)],
        }, indent=2))
        return 0

    existing = [p for p in (requests_path, decisions_path, timing_path) if p.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): "
                     f"{[str(p) for p in existing]}")

    # Import the router before the clock starts: otherwise the first pass of the
    # first episode charges ~5 s of module import to its wall-clock number.
    import skell_e_router  # noqa: F401

    budget = Budget(args.budget)
    request_rows, decision_rows, warnings = [], [], []
    timings, started = {}, time.perf_counter()
    aborted = None
    for name in args.episodes:
        data = load_episode_data(name)
        decisions, timing = run_episode(data, args, budget, request_rows, warnings)
        decision_rows.extend(decisions)
        timings[name] = timing
        print(f"{name}: {timing['episode_wall_clock_s']}s, "
              f"${timing['cost_usd']:.4f}, "
              f"{sum(p['requests'] for p in timing['passes'].values())} requests",
              file=sys.stderr)
        if budget.blocked():
            aborted = (f"budget cap ${args.budget:.2f} crossed at "
                       f"${budget.spent:.4f}; stopped after {name}")
            print(f"ABORT: {aborted}", file=sys.stderr)
            break

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": JEV_MODEL, "prompt_version": PROMPT_VERSION,
        "concurrency": args.concurrency, "t_trim": args.t_trim,
        "budget_cap_usd": args.budget, "aborted": aborted,
        "episodes": timings,
        "totals": {
            "wall_clock_s": round(time.perf_counter() - started, 3),
            "requests": len(request_rows),
            "errors": sum(1 for r in request_rows if r["error"]),
            "input_tokens": sum(r["input_tokens"] or 0 for r in request_rows),
            "output_tokens": sum(r["output_tokens"] or 0 for r in request_rows),
            "cost_usd": round(sum(r["cost"] or 0.0 for r in request_rows), 6),
            "decisions": len(decision_rows),
        },
        "pass_wall_clock_s": {
            p: round(sum(t["passes"][p]["wall_clock_s"] for t in timings.values()), 3)
            for p in PASSES},
        "warnings": warnings[:200],
        "n_warnings": len(warnings),
    }

    jsonl_append(requests_path, request_rows)
    jsonl_append(decisions_path, decision_rows)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["totals"], indent=2))
    for warning in warnings[:20]:
        print(f"warning: {warning}", file=sys.stderr)
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
