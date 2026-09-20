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
step 2  sentence pass: ``score_k`` (0-5), ``cut_k`` (a noul, only in prompt
        versions that define one), ``first_k`` and ``last_k`` word choices,
        ``--block`` target sentences per request (25 by default). The state
        carries the rules and as much transcript as fits: the whole transcript
        when it fits ``--context-tokens`` and the provider's request ceiling,
        otherwise the largest symmetric window of sentences around the block
        that does. Every job also carries a half-size window as its fallback,
        sent when the provider rejects the payload.
step 3  trim pick pass, only with ``--trim-pick``: for sentences where
        ``first_k``/``last_k`` put less than ``--t-trim`` on ``whole``, a
        ``pick_k`` choice over candidate versions, 10 items per request.

Arms written to the decisions file: jev_a, jev_b_moduleretakes, jev_b_notrim,
plus jev_b when pass 3 ran. The ``_mod`` arms (um removal and delete silence
layered on) are a scoring-time option, not separate decisions.

``--prompt-version`` picks the prompt set from
``scripts/jev_real/roughcut_jev_prompts.py``; it lands on every request and
decision row, so two runs are never confused for each other.

``--repair NAME`` re-sends only the blocks that have no answer in NAME's request
log, appends the new request rows with ``repair: true`` and rewrites NAME's
decisions file for the repaired sentences. The decisions file is the only output
a repair overwrites; timing.json gains a ``repairs`` entry and keeps the original
wall clock.

READ-ONLY against solar-sailer: the harness, corpus and prompts are read, never
written. No network calls without ``--run``.
"""

import argparse
import bisect
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
    PROMPT_VERSION, PROMPT_VERSIONS, RULES, check_rules_match, prompts_for,
)

#: The prompt bundle every pass reads. ``--prompt-version`` swaps it through
#: ``set_prompt_version`` before any job is built, so the strings a run sends
#: and the ``prompt_version`` it stamps on each row can never disagree.
PROMPT = prompts_for(PROMPT_VERSION)


def set_prompt_version(version):
    global PROMPT
    PROMPT = prompts_for(version)
    return PROMPT


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
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = (0.5, 1.5)   # slept before attempt 2, then before attempt 3
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
TARGETS_PER_REQUEST = 25       # --block default: target sentences per request
ITEMS_PER_REQUEST = 10
CONTEXT_SENTENCES = 5          # before/after a retake group or a pick item
#: ``--context-tokens`` default: estimated tokens one sentence-pass state may
#: use, rules and targets and transcript together. The provider caps the state
#: plus the longest question at 32,000 tokens.
DEFAULT_CONTEXT_TOKENS = 22_000
#: Estimated-token ceiling for a whole sentence-pass request, state plus
#: questions. The provider caps a request at 64,000 tokens
#: (``model_config.ClassificationModel.max_input_tokens``). In the held-out run
#: every sentence-pass request estimated over 49,176 tokens came back a
#: deterministic 400 (``category: invalid_request``), no request under that
#: estimate did, and the largest accepted request measured 64,127 real input
#: tokens. 46,000 leaves room for the estimate running ~22% under the real count.
REQUEST_TOKEN_CAP = 40_000
CHARS_PER_TOKEN = 4
PAUSE_S = 0.5
LONG_PAUSE_S = 1.5
REAL_RETAKE_CUT = 0.5          # real_k below this: the group is not a retake
LAST_TAKE_MARGIN = 0.15        # top minus last take below this: the last wins
TRIM_CANDIDATES = 3            # top starts and top ends crossed in step 3
MAX_CHOICE_OPTIONS = 255

PASSES = ["retake", "sentence", "trim_pick"]
ARMS = ["jev_a", "jev_b", "jev_b_moduleretakes", "jev_b_notrim"]
#: The arm that only exists when pass 3 ran: without a ``pick_k`` answer its
#: rows would be a copy of ``jev_b_notrim`` under a name that promises trims.
TRIM_PICK_ARMS = ["jev_b"]
RETAKE_FIELDS = ("retake_group", "retake_real", "retake_take_probs",
                 "retake_choice", "retake_winner", "retake_source")
REMOVALS_KEYS = ("umm_word_ids", "um_word_ids", "removed_word_ids", "word_ids",
                 "umm", "ids")
#: Error marks that send a failed block to its windowed job. ``invalid_request``
#: is here because the provider rejects an oversized sentence-pass payload with a
#: bare 400 and no message: the same block answers fine once the transcript is
#: windowed, and every question subset of it answers fine at full size. A genuine
#: schema error still fails windowed and is reported, so nothing is hidden.
CONTEXT_ERROR_MARKS = ("context", "too long", "too large", "token", "422",
                       "invalid_request")


def arms_for(trim_pick):
    return ARMS if trim_pick else [a for a in ARMS if a not in TRIM_PICK_ARMS]


def passes_for(trim_pick):
    return PASSES if trim_pick else [p for p in PASSES if p != "trim_pick"]


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
        "chains": split_sentence_chains(
            order, rendered, {s["id"]: s["text"] for s in sentences}),
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


SPLIT_MARK = ".."
#: Stripped off the front of a row before asking whether it starts lowercase.
SPLIT_LEAD_CHARS = " \t\"'‘’“”-–—"


def _continues_previous(text):
    """True when the row opens with a lowercase letter, quotes and dashes aside."""
    stripped = text.lstrip(SPLIT_LEAD_CHARS)
    return bool(stripped) and stripped[0].isalpha() and stripped[0].islower()


def split_sentence_chains(order, rendered, texts):
    """Rows that are pieces of one spoken sentence.

    Row *i* joins row *i+1* when *i* ends in ``..`` (how the transcriber marks a
    split or a trail-off) and *i+1* opens with a lowercase letter, so the second
    row reads as the rest of the first rather than a new sentence. Chains run as
    far as those joins go.

    The ``..`` is read off ``texts``, the corpus sentence text, because the word
    segments the rendering is built from spell the same mark as an em dash. The
    ``spoken_sentence`` it hands back is the rendered text, um-stripped and
    pause-marked like every other line Jev sees.

    Returns ``{sid: {"spoken_sentence": ..., "piece": "n of m"}}`` for every row
    in a chain of two or more, and nothing for a row that stands alone. v1 lost
    11 of its 40 worst drops to exactly this: a fragment read on its own looks
    like an abandoned false start.
    """
    joined = [texts[a].rstrip().endswith(SPLIT_MARK)
              and _continues_previous(texts[b])
              for a, b in zip(order, order[1:])]
    flags, start = {}, 0
    while start < len(order):
        end = start
        while end < len(joined) and joined[end]:
            end += 1
        if end > start:
            chain = order[start:end + 1]
            spoken = " ".join(rendered[sid] for sid in chain)
            for position, sid in enumerate(chain, 1):
                flags[sid] = {"spoken_sentence": spoken,
                              "piece": f"{position} of {len(chain)}"}
        start = end + 1
    return flags


def chain_counts(chains):
    """``{chains, rows, longest}`` for the timing file."""
    lengths = {flags["piece"].split(" of ")[1] for flags in chains.values()}
    sizes = [int(n) for n in lengths]
    return {"rows": len(chains),
            "chains": sum(1 for f in chains.values() if f["piece"].startswith("1 of ")),
            "longest": max(sizes) if sizes else 0}


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
           "prompt_version": PROMPT.version, "window_used": bool(job.get("window_used")),
           "window_sentences": job.get("window_sentences"),
           "window_radius": job.get("window_radius"),
           "block_size": job.get("block_size"),
           "est_over_actual": None,
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
        # The window is fitted on a 4-chars-per-token estimate. Recording it
        # against the provider's own count on every answered request is what
        # lets a later run correct the estimate instead of guessing at it.
        if response.input_tokens:
            row["est_over_actual"] = round(
                row["est_input_tokens"] / response.input_tokens, 4)
        budget.add(response.cost)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        row["error_details"] = str(getattr(exc, "details", None))[:300]
    row["elapsed_s"] = time.perf_counter() - started
    return row, answers


def median_ratio(rows):
    """Median ``est_over_actual`` over the rows that carry one, or None."""
    ratios = sorted(r["est_over_actual"] for r in rows
                    if r.get("est_over_actual") is not None)
    return round(ratios[len(ratios) // 2], 4) if ratios else None


def _is_context_error(row):
    blob = f"{row.get('error')} {row.get('error_details')}".lower()
    return any(mark in blob for mark in CONTEXT_ERROR_MARKS)


def _retry_job(job, request_rows):
    """The job to re-send for a failed block, windowed if the context was rejected."""
    rows = [r for r in request_rows if r["block"] == job["block"]]
    if job.get("window_job") and rows and _is_context_error(rows[-1]):
        return job["window_job"]
    return job


def run_pass(pass_name, episode_name, jobs, concurrency, budget, repair=False):
    """Dispatch one pass, retry failed blocks, return rows, answers and timing.

    ~7% of requests come back a bare PROVIDER_ERROR that succeeds on a
    re-request, and a few fail twice, so a block gets up to ``MAX_ATTEMPTS``
    attempts on any exception, ``RETRY_BACKOFF_S`` apart. Every attempt stays in
    the log as its own row with its ``attempt`` number.

    ``concurrency`` is the worker count for this pass alone: each pass opens and
    closes its own pool, so the passes never overlap.

    Wall clock is measured around the whole pass, submit to last result, at the
    concurrency used, retries and backoff included: that is the number a
    per-episode latency column has to report.
    """
    request_rows, answers_by_block = [], {}
    started = time.perf_counter()
    todo = list(jobs)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not todo or budget.blocked():
            break
        if attempt > 1:
            time.sleep(RETRY_BACKOFF_S[attempt - 2])
        with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {pool.submit(_request, job, budget, attempt): job
                       for job in todo}
            for future in futures.as_completed(pending):
                row, answers = future.result()
                if repair:
                    row["repair"] = True
                request_rows.append(row)
                if answers is not None:
                    answers_by_block[pending[future]["block"]] = answers
        todo = [_retry_job(job, request_rows) for job in todo
                if answers_by_block.get(job["block"]) is None]
    wall = time.perf_counter() - started

    request_rows.sort(key=lambda r: (r["block"], r["attempt"]))
    timing = {
        "wall_clock_s": round(wall, 3),
        "requests": len(request_rows),
        "input_tokens": sum(r["input_tokens"] or 0 for r in request_rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in request_rows),
        "cost_usd": round(sum(r["cost"] or 0.0 for r in request_rows), 6),
        "errors": sum(1 for r in request_rows if r["error"]),
        "failed_blocks": sum(1 for j in jobs
                             if answers_by_block.get(j["block"]) is None),
        "windowed_requests": sum(1 for r in request_rows if r["window_used"]),
        "est_over_actual_median": median_ratio(request_rows),
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
                "instructions": PROMPT.TAKE_INSTRUCTIONS.format(k=k),
                "criteria": {key: _describe(data["rendered"][sid])
                             for key, sid in zip(keys, members)},
            }
            questions[f"real_{k}"] = {
                "type": "noul",
                "instructions": PROMPT.REAL_INSTRUCTIONS.format(k=k),
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
    """One ``targets[k]`` entry, with the split-sentence flags when they apply.

    ``spoken_sentence`` and ``piece`` are only present on a row the transcript
    split, which is what the v2 score instructions key off. The transcript
    rendering stays plain ``id = text`` lines: flagging it too would mean
    re-shaping every line of a state that is re-sent once per 25 sentences, for
    a cue the target already carries.
    """
    target = {"id": sid, "words": [dict(t) for t in data["tokens"][sid]]}
    target.update(data["chains"].get(sid, {}))
    return target


def _trim_criteria(words, whole_description, from_start):
    options = words if from_start else list(reversed(words))
    if len(options) > MAX_CHOICE_OPTIONS - 1:
        options = options[:MAX_CHOICE_OPTIONS - 1]
    if not from_start:
        options = list(reversed(options))
    criteria = {str(w["w"]): _describe(w["t"]) for w in options}
    criteria["whole"] = whole_description
    return criteria


def fit_window(kept_index, kept_lines, first_idx, last_idx, budget):
    """The largest symmetric window around a target block that fits ``budget``.

    ``budget`` is estimated tokens for the rendered transcript alone, and the
    window is measured in corpus sentences either side of the block, counting
    the ones a retake cut already removed. Returns
    ``(lines, radius, whole_transcript)``; ``radius`` is None when the whole
    transcript fits, which is the case this replaces the old fixed 200 with:
    nothing changes for an episode that fits.

    Binary search, because the rendered cost only grows with the radius. The
    block's own lines go in even when they alone are over budget: a target whose
    own text is missing from the transcript is worse than an oversized request.
    """
    if est_tokens(kept_lines) <= budget:
        return list(kept_lines), None, True

    def slice_for(radius):
        lo = bisect.bisect_left(kept_index, first_idx - radius)
        hi = bisect.bisect_right(kept_index, last_idx + radius)
        return lo, hi

    fits, over = 0, max(first_idx, (kept_index[-1] if kept_index else 0) - last_idx) + 1
    while over - fits > 1:
        middle = (fits + over) // 2
        lo, hi = slice_for(middle)
        if est_tokens(kept_lines[lo:hi]) <= budget:
            fits = middle
        else:
            over = middle
    lo, hi = slice_for(fits)
    return kept_lines[lo:hi], fits, False


def sentence_jobs(data, drop_ids, block_size=TARGETS_PER_REQUEST,
                  context_tokens=DEFAULT_CONTEXT_TOKENS):
    """One job per block of ``block_size`` target sentences, plus a fallback job.

    Each job carries the largest transcript window around its own block that
    fits both budgets: ``context_tokens`` for the state (rules, targets and
    transcript), and ``REQUEST_TOKEN_CAP`` for the whole request once the
    questions are counted. The second budget is what keeps a long episode off
    the provider's deterministic 400, and it is why the window shrinks as
    ``block_size`` grows: the trim questions list every word of every target, so
    they take the room the transcript would have had.

    ``window_job`` re-fits at half the rendered cost and is sent when a block
    comes back a context error, including the jobs that already window.
    """
    kept_order = [sid for sid in data["order"] if sid not in drop_ids]
    kept_lines = [sentence_line(data, sid) for sid in kept_order]
    kept_index = [data["index_of"][sid] for sid in kept_order]

    jobs = []
    for block, target_ids in enumerate(blocks(data["order"], block_size)):
        questions, targets = {}, []
        for k, sid in enumerate(target_ids):
            targets.append(_target(data, sid))
            questions[f"score_{k}"] = {
                "type": "score",
                "instructions": PROMPT.SCORE_INSTRUCTIONS.format(k=k),
                "criteria": list(PROMPT.SCORE_LEVELS),
            }
            # A second read on the same sentence, asked as "is it removed?"
            # rather than "how much is it worth?". Every target gets it,
            # one-word rows included, so no sentence is missing a ``cut_p``.
            if PROMPT.CUT_INSTRUCTIONS:
                questions[f"cut_{k}"] = {
                    "type": "noul",
                    "instructions": PROMPT.CUT_INSTRUCTIONS.format(k=k),
                }
            words = data["words"][sid]
            if len(words) >= 2:
                questions[f"first_{k}"] = {
                    "type": "choice",
                    "instructions": PROMPT.FIRST_INSTRUCTIONS.format(k=k),
                    "criteria": _trim_criteria(
                        words, PROMPT.WHOLE_FIRST_DESCRIPTION, True),
                }
                questions[f"last_{k}"] = {
                    "type": "choice",
                    "instructions": PROMPT.LAST_INSTRUCTIONS.format(k=k),
                    "criteria": _trim_criteria(
                        words, PROMPT.WHOLE_LAST_DESCRIPTION, False),
                }

        first_idx, last_idx = (data["index_of"][target_ids[0]],
                               data["index_of"][target_ids[-1]])
        # Everything the request carries whatever the window is, measured the
        # way the request row measures it so the two agree to the token.
        fixed = est_tokens({"rules": RULES, "transcript": [], "targets": targets})
        budget = min(context_tokens - fixed,
                     REQUEST_TOKEN_CAP - fixed - est_tokens(questions))

        def make(line_budget):
            lines, radius, whole = fit_window(kept_index, kept_lines,
                                              first_idx, last_idx, line_budget)
            return {"pass": "sentence", "episode": data["name"], "block": block,
                    "ids": list(target_ids), "window_used": not whole,
                    "window_sentences": len(lines), "window_radius": radius,
                    "block_size": block_size,
                    "state": {"rules": RULES, "transcript": lines, "targets": targets},
                    "questions": questions, "target_ids": list(target_ids)}

        job = make(budget)
        job["window_job"] = make(
            min(budget, est_tokens(job["state"]["transcript"])) / 2)
        jobs.append(job)
    return jobs


def window_stats(jobs):
    """What the fitted window did to one episode's sentence jobs, for the record."""
    sentences = sorted(job["window_sentences"] for job in jobs)
    totals = [round(est_tokens(job["state"]) + est_tokens(job["questions"]))
              for job in jobs]
    if not jobs:
        return {}
    return {
        "block_size": jobs[0]["block_size"],
        "window_sentences_min": sentences[0],
        "window_sentences_median": sentences[len(sentences) // 2],
        "window_sentences_max": sentences[-1],
        "whole_transcript_requests": sum(1 for j in jobs if not j["window_used"]),
        "est_input_tokens_max": max(totals),
        "over_request_cap": sum(1 for t in totals if t > REQUEST_TOKEN_CAP),
    }


def sentence_results(data, jobs, answers_by_block):
    """``{sid: {...raw answer fields...}}`` plus warnings."""
    out, warnings = {}, []
    for job in jobs:
        answers = answers_by_block.get(job["block"])
        for k, sid in enumerate(job["target_ids"]):
            row = {"score": None, "score_probabilities": None, "defaulted": True,
                   "first_choice": None, "first_p_whole": None,
                   "last_choice": None, "last_p_whole": None,
                   "first_probabilities": None, "last_probabilities": None,
                   "cut_p": None}
            # Read independently of the score: a block that answered one and not
            # the other is a partial answer, not a missing one.
            if answers and f"cut_{k}" in answers:
                row["cut_p"] = (answers[f"cut_{k}"] or {}).get("noul")
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
            rendered["cut"] = PROMPT.CUT_VERSION_DESCRIPTION
            state["items"][str(k)] = {
                "before": context_lines(data, sid, before=True),
                "sentence": sentence_line(data, sid),
                "after": context_lines(data, sid, before=False),
                "versions": rendered,
            }
            criteria = {key: _describe(text) for key, text in rendered.items()}
            criteria["whole"] = PROMPT.WHOLE_VERSION_DESCRIPTION + ": " + criteria["whole"]
            questions[f"pick_{k}"] = {
                "type": "choice",
                "instructions": PROMPT.PICK_INSTRUCTIONS.format(k=k),
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


def build_decisions(data, retake_info, results, picks, only_ids=None, arms=None):
    """One row per sentence per arm; ``only_ids`` limits it to repaired sentences.

    ``arms`` defaults to every arm. With the trim-pick pass off the caller passes
    ``arms_for(False)``, which drops ``jev_b``: no ``pick_k`` answer exists, so
    its rows would carry no trims at all.
    """
    arms = list(ARMS if arms is None else arms)
    rows = []
    for sid in data["order"]:
        if only_ids is not None and sid not in only_ids:
            continue
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
            "cut_p": result.get("cut_p"),
            "pick_choice": pick.get("pick_choice"),
            "pick_probabilities": pick.get("pick_probabilities"),
            "retake_group": retake.get("retake_group"),
            "retake_real": retake.get("retake_real"),
            "retake_take_probs": retake.get("retake_take_probs"),
            "retake_choice": retake.get("retake_choice"),
            "retake_winner": retake.get("retake_winner"),
            "retake_source": retake.get("retake_source"),
            "defaulted": result.get("defaulted", True),
            "prompt_version": PROMPT.version,
            "trim_pick_pass": "jev_b" in arms,
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
        for arm in arms:
            arm_score, keep_words, cut_retake = per_arm[arm]
            rows.append({"arm": arm, "episode": data["name"], "id": sid,
                         "score": arm_score, "keep_words": keep_words,
                         "cut_retake": cut_retake, **source})
    return rows


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------

def unanswered_blocks(request_log):
    """``{(episode, pass): {block, ...}}`` for blocks whose every attempt errored.

    An earlier repair's rows count as attempts, so repairing the same run twice
    re-sends nothing that has since come back.
    """
    seen, answered = set(), set()
    for row in request_log:
        key = (row["episode"], row["pass"], row["block"])
        seen.add(key)
        if not row.get("error"):
            answered.add(key)
    gaps = defaultdict(set)
    for episode, pass_name, block in seen - answered:
        gaps[(episode, pass_name)].add(block)
    return gaps


def _retake_info_from_decisions(rows):
    """A previous run's retake answers, read back off its decision rows."""
    return {row["id"]: {key: row.get(key) for key in RETAKE_FIELDS}
            for row in rows if row["arm"] == "jev_a"}


def _patch_retake(row, info):
    """Overwrite one decision row's retake fields with a repaired answer."""
    row.update({key: info.get(key) for key in RETAKE_FIELDS})
    if row["arm"] != "jev_b_moduleretakes":
        winner = info.get("retake_winner")
        row["cut_retake"] = bool(winner is not None and winner != row["id"])


def repair_episode(data, gap_passes, args, budget, previous_rows, pick_block_offset):
    """Re-send one episode's failed blocks and rebuild only what they decide.

    The retake answers and the retake drop set are read back off ``previous_rows``
    (this episode's decision rows), so a re-sent sentence block carries the same
    transcript the original run sent it. Sentences that come back with a score
    also get their trim-pick question asked, numbered past the original run's
    trim_pick blocks so the request log stays unambiguous.

    A failed trim_pick block of the original run cannot be rebuilt: its candidate
    versions come from sentence-pass probabilities the request log does not
    store. Those are reported, not re-sent.

    Returns ``(request_rows, replacements, warnings, report)``. ``replacements``
    supersede previous decision rows by (id, arm); rows that only a repaired
    retake touches are patched in place.
    """
    request_rows, warnings = [], []
    retake_info = _retake_info_from_decisions(previous_rows)
    losers = {row["id"] for row in previous_rows
              if row["arm"] == "jev_a" and row["cut_retake"]}

    if gap_passes.get("retake"):
        jobs = [j for j in retake_jobs(data) if j["block"] in gap_passes["retake"]]
        rows, answers, _timing = run_pass("retake", data["name"], jobs,
                                          args.concurrency, budget, repair=True)
        request_rows.extend(rows)
        info, group_losers, warn = retake_decisions(data, jobs, answers)
        warnings.extend(warn)
        losers = (losers - set(info)) | group_losers
        retake_info.update(info)
        for row in previous_rows:
            if row["id"] in info:
                _patch_retake(row, info[row["id"]])

    results = {}
    if gap_passes.get("sentence"):
        jobs = [j for j in sentence_jobs(data, losers, args.block,
                                         args.context_tokens)
                if j["block"] in gap_passes["sentence"]]
        rows, answers, _timing = run_pass("sentence", data["name"], jobs,
                                          args.concurrency, budget, repair=True)
        request_rows.extend(rows)
        results, warn = sentence_results(data, jobs, answers)
        warnings.extend(warn)

    picks = {}
    pick_batch = pick_jobs(data, results, args.t_trim) if args.trim_pick else []
    for offset, job in enumerate(pick_batch):
        job["block"] = pick_block_offset + offset
    if pick_batch:
        rows, answers, _timing = run_pass("trim_pick", data["name"], pick_batch,
                                          args.concurrency, budget, repair=True)
        request_rows.extend(rows)
        picks, warn = pick_results(data, pick_batch, answers)
        warnings.extend(warn)

    if gap_passes.get("trim_pick"):
        warnings.append(
            f"{data['name']}: trim_pick block(s) "
            f"{sorted(gap_passes['trim_pick'])} cannot be repaired from the log "
            f"(their candidate versions come from sentence-pass probabilities that "
            f"the log does not store); re-run the episode to recover them")

    replacements = build_decisions(data, retake_info, results, picks,
                                   only_ids=set(results),
                                   arms=arms_for(args.trim_pick))
    report = {
        "episode": data["name"],
        "resent_blocks": {p: sorted(gap_passes.get(p, ())) for p in PASSES
                          if gap_passes.get(p)},
        "new_pick_blocks": [j["block"] for j in pick_batch],
        "requests": len(request_rows),
        "errors": sum(1 for r in request_rows if r["error"]),
        "cost_usd": round(sum(r["cost"] or 0.0 for r in request_rows), 6),
        "sentences_repaired": sum(1 for r in results.values() if not r["defaulted"]),
        "still_defaulted": sum(1 for r in results.values() if r["defaulted"]),
    }
    return request_rows, replacements, warnings, report


def run_repair(args, parser, paths):
    """Re-send every block with no answer in an existing run, in place."""
    requests_path, decisions_path, timing_path = paths
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        parser.error(f"--repair {args.repair}: nothing to repair, missing {missing}")

    request_log = jsonl_read(requests_path)
    decisions = jsonl_read(decisions_path)
    summary = json.loads(timing_path.read_text(encoding="utf-8"))
    gaps = unanswered_blocks(request_log)
    if not gaps:
        print("Nothing to repair: every block in the log has an answer.",
              file=sys.stderr)
        return 0

    import skell_e_router  # noqa: F401

    by_episode = defaultdict(list)
    for row in decisions:
        by_episode[row["episode"]].append(row)

    budget = Budget(args.budget)
    started = time.perf_counter()
    new_requests, reports, warnings, replaced = [], [], [], {}
    for name in sorted({episode for episode, _ in gaps}):
        gap_passes = {p: b for (episode, p), b in gaps.items() if episode == name}
        data = load_episode_data(name)
        offset = 1 + max([r["block"] for r in request_log
                          if r["episode"] == name and r["pass"] == "trim_pick"]
                         + [-1])
        rows, replacements, warn, report = repair_episode(
            data, gap_passes, args, budget, by_episode[name], offset)
        stamp = datetime.now(timezone.utc).isoformat()
        for row in rows:
            row["recorded_utc"] = stamp
        new_requests.extend(rows)
        warnings.extend(warn)
        reports.append(report)
        replaced.update({(name, r["id"], r["arm"]): r for r in replacements})
        print(json.dumps(report), flush=True)

    entry = {
        "repaired_utc": datetime.now(timezone.utc).isoformat(),
        "extra_wall_clock_s": round(time.perf_counter() - started, 3),
        "requests": len(new_requests),
        "errors": sum(1 for r in new_requests if r["error"]),
        "cost_usd": round(sum(r["cost"] or 0.0 for r in new_requests), 6),
        "decision_rows_rewritten": len(replaced),
        "episodes": reports,
        "warnings": warnings[:50],
        "n_warnings": len(warnings),
    }
    summary.setdefault("repairs", []).append(entry)

    rebuilt = [replaced.get((r["episode"], r["id"], r["arm"]), r) for r in decisions]
    jsonl_append(requests_path, new_requests)
    decisions_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rebuilt),
        encoding="utf-8")
    timing_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(entry, indent=2))
    for warning in warnings[:20]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def estimate(episode_names, t_trim, trim_pick=True, block_size=TARGETS_PER_REQUEST,
             context_tokens=DEFAULT_CONTEXT_TOKENS):
    """Requests, input tokens and spend per pass per episode, no calls.

    Passes 1 and 2 are built for real, so their numbers are the exact payloads.
    Pass 3 depends on answers that do not exist yet, so it assumes a trim rate;
    with ``trim_pick`` off it is left out of the plan entirely.
    """
    assumed_trim_rate = 0.3
    plan, totals = [], {"requests": 0, "est_input_tokens": 0.0, "est_cost_usd": 0.0}
    chains = {}
    for name in episode_names:
        data = load_episode_data(name)
        chains[name] = chain_counts(data["chains"])
        module_losers = {s["id"] for s in data["sentences"] if s.get("is_retake")}
        episode_rows = []
        for pass_name, jobs in (
                ("retake", retake_jobs(data)),
                ("sentence", sentence_jobs(data, module_losers, block_size,
                                           context_tokens))):
            tokens = sum(est_tokens(j["state"]) + est_tokens(j["questions"])
                         for j in jobs)
            row = {"episode": name, "pass": pass_name, "requests": len(jobs),
                   "sentences": len(data["order"]),
                   "windowed": sum(1 for j in jobs if j.get("window_used")),
                   "est_input_tokens": round(tokens),
                   "est_cost_usd": round(tokens * JEV_INPUT_PER_MILLION / 1e6, 4)}
            if pass_name == "sentence":
                row.update(window_stats(jobs))
            episode_rows.append(row)
        if trim_pick:
            n_items = math.ceil(len(data["order"]) * assumed_trim_rate)
            n_pick = math.ceil(n_items / ITEMS_PER_REQUEST)
            chars = sum(len(data["rendered"][s]) for s in data["order"]) / max(
                1, len(data["order"]))
            pick_tokens = n_pick * (
                ITEMS_PER_REQUEST * (12 * chars + 600) / CHARS_PER_TOKEN)
            episode_rows.append(
                {"episode": name, "pass": "trim_pick", "requests": n_pick,
                 "sentences": len(data["order"]),
                 "assumed_trim_rate": assumed_trim_rate,
                 "est_input_tokens": round(pick_tokens),
                 "est_cost_usd": round(
                     pick_tokens * JEV_INPUT_PER_MILLION / 1e6, 4)})
        for row in episode_rows:
            totals["requests"] += row["requests"]
            totals["est_input_tokens"] += row["est_input_tokens"]
            totals["est_cost_usd"] += row["est_cost_usd"]
        plan.extend(episode_rows)
    totals["est_cost_usd"] = round(totals["est_cost_usd"], 4)
    totals["est_input_tokens"] = round(totals["est_input_tokens"])
    return plan, totals, chains


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_episode(data, args, budget, request_rows, warnings):
    timing = {"sentences": len(data["order"]), "removals": data["removals"],
              "split_chains": chain_counts(data["chains"]),
              "trim_pick": bool(args.trim_pick), "passes": {}}

    jobs1 = retake_jobs(data)
    rows, answers1, t1 = run_pass("retake", data["name"], jobs1, args.concurrency,
                                  budget)
    request_rows.extend(rows)
    timing["passes"]["retake"] = dict(t1, groups=len(retake_group_list(data)))
    retake_info, losers, warn = retake_decisions(data, jobs1, answers1)
    warnings.extend(warn)
    timing["passes"]["retake"]["losers_cut"] = len(losers)

    jobs2 = sentence_jobs(data, losers, args.block, args.context_tokens)
    rows, answers2, t2 = run_pass("sentence", data["name"], jobs2, args.concurrency,
                                  budget)
    request_rows.extend(rows)
    timing["passes"]["sentence"] = dict(t2, **window_stats(jobs2))
    results, warn = sentence_results(data, jobs2, answers2)
    warnings.extend(warn)

    picks = {}
    if args.trim_pick:
        jobs3 = pick_jobs(data, results, args.t_trim)
        rows, answers3, t3 = run_pass("trim_pick", data["name"], jobs3,
                                      args.concurrency, budget)
        request_rows.extend(rows)
        timing["passes"]["trim_pick"] = dict(t3, items=sum(len(j["mapping"])
                                                           for j in jobs3))
        picks, warn = pick_results(data, jobs3, answers3)
        warnings.extend(warn)

    arms = arms_for(args.trim_pick)
    decisions = build_decisions(data, retake_info, results, picks, arms=arms)
    timing["episode_wall_clock_s"] = round(
        sum(p["wall_clock_s"] for p in timing["passes"].values()), 3)
    timing["cost_usd"] = round(sum(p["cost_usd"] for p in timing["passes"].values()), 6)
    timing["trims"] = {
        arm: sum(1 for d in decisions if d["arm"] == arm and d["keep_words"])
        for arm in arms}
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
    parser.add_argument("--prompt-version", default=PROMPT_VERSION,
                        choices=PROMPT_VERSIONS,
                        help=f"prompt set to send (default: {PROMPT_VERSION}); "
                             f"recorded on every request and decision row")
    parser.add_argument("--trim-pick", action="store_true",
                        help="run pass 3, the trim pick, and write the jev_b arm "
                             "(off by default: it is the slowest pass and variant A "
                             "already carries a trim)")
    parser.add_argument("--block", type=int, default=TARGETS_PER_REQUEST,
                        help=f"target sentences per sentence-pass request "
                             f"(default: {TARGETS_PER_REQUEST}); the trim "
                             f"questions grow with it, so a larger block leaves "
                             f"less room for the transcript window")
    parser.add_argument("--context-tokens", type=int,
                        default=DEFAULT_CONTEXT_TOKENS,
                        help=f"estimated-token budget for one sentence-pass "
                             f"state, rules and targets and transcript together "
                             f"(default: {DEFAULT_CONTEXT_TOKENS}); the window is "
                             f"the largest that fits it and the "
                             f"{REQUEST_TOKEN_CAP}-token whole-request ceiling")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="in-flight requests per pass (each pass runs its own "
                             "pool, so passes never overlap)")
    parser.add_argument("--repair", metavar="IN_NAME",
                        help="re-send the blocks with no answer in IN_NAME's request "
                             "log; appends request rows with repair: true and "
                             "rewrites IN_NAME's decisions file in place")
    args = parser.parse_args()
    set_prompt_version(args.prompt_version)

    out_name = args.repair or args.out
    requests_path = OUT_DIR / f"{out_name}-requests.jsonl"
    decisions_path = OUT_DIR / f"{out_name}-decisions.jsonl"
    timing_path = OUT_DIR / f"{out_name}-timing.json"

    if args.repair:
        return run_repair(args, parser,
                          (requests_path, decisions_path, timing_path))

    if not args.run:
        plan, totals, chains = estimate(args.episodes, args.t_trim, args.trim_pick,
                                        args.block, args.context_tokens)
        matches, _ = check_rules_match()
        print(json.dumps({
            "mode": "plan", "model": JEV_MODEL, "prompt_version": PROMPT.version,
            "rules_match_source_prompt": matches,
            "episodes": args.episodes, "arms": arms_for(args.trim_pick),
            "passes": passes_for(args.trim_pick), "split_chains": chains,
            "concurrency": args.concurrency, "t_trim": args.t_trim,
            "block": args.block, "context_tokens": args.context_tokens,
            "request_token_cap": REQUEST_TOKEN_CAP,
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
        per_pass = ", ".join(f"{p} {timing['passes'][p]['wall_clock_s']:.2f}s"
                             for p in PASSES if p in timing["passes"])
        print(f"{name}: {timing['episode_wall_clock_s']}s ({per_pass}), "
              f"${timing['cost_usd']:.4f}, "
              f"{sum(p['requests'] for p in timing['passes'].values())} requests, "
              f"{sum(p['errors'] for p in timing['passes'].values())} errors, "
              f"{sum(p['failed_blocks'] for p in timing['passes'].values())} "
              f"failed blocks", file=sys.stderr)
        if budget.blocked():
            aborted = (f"budget cap ${args.budget:.2f} crossed at "
                       f"${budget.spent:.4f}; stopped after {name}")
            print(f"ABORT: {aborted}", file=sys.stderr)
            break

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": JEV_MODEL, "prompt_version": PROMPT.version,
        "arms": arms_for(args.trim_pick), "trim_pick": bool(args.trim_pick),
        "concurrency": args.concurrency, "t_trim": args.t_trim,
        "block": args.block, "context_tokens": args.context_tokens,
        "request_token_cap": REQUEST_TOKEN_CAP,
        "budget_cap_usd": args.budget, "aborted": aborted,
        "episodes": timings,
        "totals": {
            "wall_clock_s": round(time.perf_counter() - started, 3),
            "requests": len(request_rows),
            "errors": sum(1 for r in request_rows if r["error"]),
            "failed_blocks": sum(t["passes"][p]["failed_blocks"]
                                 for t in timings.values() for p in PASSES
                                 if p in t["passes"]),
            "input_tokens": sum(r["input_tokens"] or 0 for r in request_rows),
            "output_tokens": sum(r["output_tokens"] or 0 for r in request_rows),
            "cost_usd": round(sum(r["cost"] or 0.0 for r in request_rows), 6),
            "decisions": len(decision_rows),
            "est_over_actual_median": median_ratio(request_rows),
            "est_over_actual_sentence_median": median_ratio(
                [r for r in request_rows if r["pass"] == "sentence"]),
        },
        "pass_wall_clock_s": {
            p: round(sum(t["passes"][p]["wall_clock_s"] for t in timings.values()
                         if p in t["passes"]), 3)
            for p in PASSES},
        "warnings": warnings[:200],
        "n_warnings": len(warnings),
    }

    jsonl_append(requests_path, request_rows)
    jsonl_append(decisions_path, decision_rows)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"totals": summary["totals"],
                      "pass_wall_clock_s": summary["pass_wall_clock_s"]}, indent=2))
    for warning in warnings[:20]:
        print(f"warning: {warning}", file=sys.stderr)
    return 2 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
