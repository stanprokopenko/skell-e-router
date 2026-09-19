"""Matched benchmark: TypeSafe Jev vs gpt-5.6-luna on solar-sailer's rough-cut
sentence-rating step.

Arms
----
``luna``        one full-context call per episode, ``roughcut_system_v1.md`` as the
                system prompt, the production ``id = text`` user message, parsed by
                the harness's own ``parse_ratings``.
``jev_full``    Jev score + retake-noul per sentence, state = rules + the WHOLE
                transcript, questions batched 25 sentences per request.
``jev_window``  same questions, state = a +/-15 sentence excerpt around each block
                of 10 target sentences.

``*_retake`` variants rescore the same answers with the score forced to 0 whenever
the retake noul is >= 0.6. They cost nothing extra.

Everything is scored by the benchmark harness's own functions
(``roughcut_bench.calibrate`` + ``roughcut_bench.replay.score_sentences``) at the
harness's own calibrated operating points: SENTENCE POINTS, WORD SCORE and MCC.

READ-ONLY against solar-sailer: the harness and corpus are imported and read, never
written. ``episodes.episode_answer_key`` is monkeypatched to read the committed
``episodes/<name>.answerkey.json`` cache instead of re-extracting from the Q: drive,
because the stock function would rewrite that cache file.

No network calls without ``--run``.
"""

import argparse
import concurrent.futures as futures
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(r"D:\solar-sailer\benchmarks\roughcut")
EPISODES_DIR = BENCH_DIR / "episodes"
PROMPT_PATH = BENCH_DIR / "prompts" / "roughcut_system_v1.md"
OUT_DIR = ROOT / "docs" / "jev-real"
RESULTS_PATH = OUT_DIR / "roughcut-results.jsonl"
REQUESTS_PATH = OUT_DIR / "roughcut-requests.jsonl"
SUMMARY_PATH = OUT_DIR / "roughcut-summary.json"
NOTES_PATH = OUT_DIR / "roughcut-notes.md"

EPISODES = [
    "colman-02.04-skeleton-demo",
    "hampton-5.4-assignment-demo",
    "colman-03.03-muscles-crit",
    "edges-7.01-intro",
    "hampton-5.2-shape-demo",
]
API_ARMS = ["luna", "jev_full", "jev_window"]
SCORED_ARMS = ["luna", "jev_full", "jev_full_retake", "jev_window", "jev_window_retake"]

LUNA_MODEL = "gpt-5.6-luna"
LUNA_REASONING_EFFORT = "medium"   # not recorded in the archived run; brief's default
LUNA_MAX_TOKENS = 32000            # from results/2026-07-15-13d-v1-single-gpt-5.6-luna.json
LUNA_TIMEOUT = 900
# Luna pricing per million tokens, same formula as scripts/benchmark_jev_classification.py.
LUNA_IN, LUNA_CACHED, LUNA_OUT = 0.20, 0.02, 1.20

JEV_MODEL = "jev-1.13.0"
JEV_INPUT_PER_MILLION = 0.042
JEV_TIMEOUT = 180
JEV_CONCURRENCY = 4
JEV_BLOCK_FULL = 25     # target sentences per request, jev_full
JEV_BLOCK_WINDOW = 10   # target sentences per request, jev_window
JEV_WINDOW_PAD = 15     # sentences of context either side of a window block
RETAKE_NOUL_CUT = 0.6

# The harness defaults a still-missing id to 2.5; the brief asks for 3 ("fine").
STILL_MISSING_SCORE = 3
STILL_MISSING_CATEGORY = "keep"

NEUTRAL_LEVEL = "4"
BUDGET_CAP_USD = 2.50


# ---------------------------------------------------------------------------
# harness imports (read-only)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(BENCH_DIR))
from roughcut_bench import calibrate, episodes as episodes_mod, replay  # noqa: E402
from roughcut_bench.parsing import ParseError, parse_ratings  # noqa: E402
from roughcut_bench.sentence_scoring import sentence_states  # noqa: E402


def _cached_answer_key(episode, episodes_dir=str(EPISODES_DIR)):
    """Read the committed answer-key cache. Never extracts, never writes.

    The stock ``episodes.episode_answer_key`` stats the source .prproj on the Q:
    drive and rewrites ``episodes/<name>.answerkey.json`` whenever the stamp
    misses. This benchmark must not write anywhere under solar-sailer, so it
    reads the cache and fails loudly if the cached spec disagrees with the
    episode config.
    """
    path = os.path.join(episodes_dir, f"{episode['name']}.answerkey.json")
    with open(path, encoding="utf-8") as handle:
        cache = json.load(handle)
    cached_ak = cache["answer_key"]
    same = (cached_ak.get("type") == episode["answer_key"].get("type")
            and cached_ak.get("sequence") == episode["answer_key"].get("sequence")
            and cached_ak["path"].replace("\\", "/").lower()
            == episode["answer_key"]["path"].replace("\\", "/").lower())
    if not same:
        raise ValueError(
            f"answer-key cache {path} does not match the episode config; refusing "
            f"to re-extract (this script is read-only against solar-sailer)"
        )
    return cache["timecodes"]


episodes_mod.episode_answer_key = _cached_answer_key


# ---------------------------------------------------------------------------
# prompt -> rules text + score criteria
# ---------------------------------------------------------------------------

def _sections(text):
    out, current, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("# "):
            if current:
                out[current] = "\n".join(buf).strip()
            current, buf = line[2:].strip(), []
        else:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf).strip()
    return out


def load_prompt_parts():
    """``(system_prompt, rules_text, score_criteria)`` from roughcut_system_v1.md.

    ``rules_text`` is what Jev's state carries: the mission, the editing rules and
    the 0-5 rubric, with every output-format instruction removed (the JSON schema
    section, the category menu, and the mission line describing the ``id = text``
    input format, which is false for the windowed arm).
    """
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    sec = _sections(system_prompt)

    mission = sec["MISSION"].split("\n\n")[0].strip()
    rules = sec["EDITING RULES"].strip()

    rubric = sec["SCORING RUBRIC"]
    levels = {}
    for line in rubric.splitlines():
        match = re.match(r"^-\s*([0-5])\s*[—-]\s*(.+)$", line.strip())
        if match:
            levels[int(match.group(1))] = f"{match.group(1)} — {match.group(2).strip()}"
    if sorted(levels) != [0, 1, 2, 3, 4, 5]:
        raise ValueError(f"expected six rubric levels 0-5, parsed {sorted(levels)}")
    criteria = [{"what": levels[i]} for i in range(6)]

    rubric_head = rubric.split("\n\n")[0].strip()
    rubric_body = "\n".join(levels[i] and f"- {levels[i]}" for i in range(5, -1, -1))
    rules_text = (
        f"# MISSION\n\n{mission}\n\n"
        f"# EDITING RULES\n\n{rules}\n\n"
        f"# SCORING RUBRIC\n\n{rubric_head}\n\n{rubric_body}"
    )
    return system_prompt, rules_text, criteria


SCORE_INSTRUCTIONS = (
    "How much does the sentence with id {sid} in `transcript` earn its place in the "
    "final edit, following `rules`? Consider the sentences around it: repeated takes "
    "(keep the best take, default the last one, cut the others), false starts, "
    "continuity, humor and personality."
)
RETAKE_INSTRUCTIONS = (
    "The sentence with id {sid} in `transcript` is one of several attempts at the same "
    "line, and it is NOT the take an editor would keep (the last take is kept unless "
    "clearly worse)."
)
WINDOW_NOTE = ("transcript is an excerpt of a longer lesson; ids are the original "
               "positions")


def build_questions(ids, criteria):
    questions = {}
    for sid in ids:
        questions[f"s{sid}"] = {
            "type": "score",
            "instructions": SCORE_INSTRUCTIONS.format(sid=sid),
            "criteria": criteria,
        }
        questions[f"r{sid}"] = {
            "type": "noul",
            "instructions": RETAKE_INSTRUCTIONS.format(sid=sid),
        }
    return questions


def build_user_message(sentences):
    """Byte-identical to the harness / production ``_build_user_message``."""
    body = "\n".join(f"{s['id']} = {s['text']}" for s in sentences)
    return "# TRANSCRIPT\n\n" + body


def build_retry_message(missing_sentences, missing_ids):
    header = (
        "# TRANSCRIPT (MISSING SENTENCES)\n\n"
        f"The following sentence ids were missing from your previous ratings: "
        f"{missing_ids}. Rate ONLY these sentences, using the exact same JSON schema.\n\n"
    )
    return header + "\n".join(f"{s['id']} = {s['text']}" for s in missing_sentences)


# ---------------------------------------------------------------------------
# minimal run helpers (scripts/jev_real/common.py did not exist when this was
# written, so these stay local rather than creating it)
# ---------------------------------------------------------------------------

def jsonl_append(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_read(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def blocks(ids, size):
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def latency_summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "total_s": sum(ordered),
        "median_s": statistics.median(ordered),
        "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_s": ordered[-1],
    }


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def run_luna(episode_name, sentences, system_prompt):
    """One full-context call plus the production missing-id retry."""
    from skell_e_router import ask_ai

    request_rows = []
    expected_ids = [s["id"] for s in sentences]
    warnings = []

    def call(user_message, label):
        started = time.perf_counter()
        row = {"arm": "luna", "episode": episode_name, "block": label,
               "n_sentences": None, "provider_model": None, "input_tokens": None,
               "output_tokens": None, "reasoning_tokens": None,
               "cached_input_tokens": None, "cost_usd": None, "elapsed_s": None,
               "error": None, "finish_reason": None, "attempt": 1}
        content = None
        try:
            response = ask_ai(
                LUNA_MODEL, user_message, system_message=system_prompt,
                reasoning_effort=LUNA_REASONING_EFFORT, max_tokens=LUNA_MAX_TOKENS,
                timeout=LUNA_TIMEOUT, rich_response=True,
            )
            usage = getattr(response.raw_response, "usage", None)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            inp, out = response.prompt_tokens, response.completion_tokens
            cost = (None if inp is None or out is None else
                    ((inp - cached) * LUNA_IN + cached * LUNA_CACHED + out * LUNA_OUT)
                    / 1_000_000)
            row.update(provider_model=response.model, input_tokens=inp,
                       output_tokens=out, reasoning_tokens=response.reasoning_tokens,
                       cached_input_tokens=cached, cost_usd=cost,
                       finish_reason=response.finish_reason)
            content = response.content
        except Exception as exc:  # recorded, never swallowed silently
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["elapsed_s"] = time.perf_counter() - started
        request_rows.append(row)
        return content

    content = call(build_user_message(sentences), "full")
    ratings, missing_ids = {}, list(expected_ids)
    if content is not None:
        try:
            ratings, missing_ids, warn = parse_ratings(content, expected_ids)
            warnings.extend(warn)
        except ParseError as exc:
            warnings.append(f"primary parse failed: {exc}")
            missing_ids = list(expected_ids)

    if missing_ids:
        missing_set = set(missing_ids)
        missing_sentences = [s for s in sentences if s["id"] in missing_set]
        content2 = call(build_retry_message(missing_sentences, missing_ids),
                        "missing_ids_retry")
        if content2 is not None:
            try:
                ratings2, _, warn2 = parse_ratings(content2, missing_ids)
                warnings.extend(warn2)
                ratings.update(ratings2)
            except ParseError as exc:
                warnings.append(f"retry parse failed: {exc}")

    defaulted = []
    for sid in expected_ids:
        if sid not in ratings:
            ratings[sid] = {"score": STILL_MISSING_SCORE,
                            "category": STILL_MISSING_CATEGORY}
            defaulted.append(sid)
    if defaulted:
        warnings.append(
            f"{len(defaulted)} id(s) still missing after retry; defaulted to score "
            f"{STILL_MISSING_SCORE} {STILL_MISSING_CATEGORY}: {defaulted[:20]}"
        )

    by_id = {s["id"]: s for s in sentences}
    sentence_rows = [{
        "arm": "luna", "episode": episode_name, "id": sid,
        "score": ratings[sid]["score"], "category": ratings[sid]["category"],
        "defaulted": sid in set(defaulted), "text": by_id[sid]["text"],
    } for sid in expected_ids]
    return sentence_rows, request_rows, warnings


def _jev_request(arm, episode_name, block_index, state, target_ids, criteria):
    from skell_e_router import classify

    questions = build_questions(target_ids, criteria)
    started = time.perf_counter()
    row = {"arm": arm, "episode": episode_name, "block": block_index,
           "n_sentences": len(target_ids), "first_id": target_ids[0],
           "last_id": target_ids[-1], "n_questions": len(questions),
           "state_sentences": len(state["transcript"]), "provider_model": None,
           "input_tokens": None, "output_tokens": None, "cost_usd": None,
           "elapsed_s": None, "error": None}
    answers = None
    try:
        response = classify(JEV_MODEL, state, questions, timeout=JEV_TIMEOUT)
        answers = response.answers
        row.update(provider_model=response.model, input_tokens=response.input_tokens,
                   output_tokens=response.output_tokens, cost_usd=response.cost)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        row["error_details"] = str(getattr(exc, "details", None))[:300]
    row["elapsed_s"] = time.perf_counter() - started
    return row, answers


def _dispatch_jev(arm, episode_name, jobs, request_rows, answers_by_block, attempt):
    with futures.ThreadPoolExecutor(max_workers=JEV_CONCURRENCY) as pool:
        pending = {
            pool.submit(_jev_request, arm, episode_name, bi, state, tids, criteria): bi
            for bi, tids, state, criteria in jobs
        }
        for future in futures.as_completed(pending):
            row, answers = future.result()
            row["attempt"] = attempt
            request_rows.append(row)
            if answers is not None or pending[future] not in answers_by_block:
                answers_by_block[pending[future]] = answers


def jev_jobs(arm, sentences, rules_text, criteria, only_blocks=None):
    """``[(block_index, target_ids, state, criteria)]`` for one arm and episode."""
    ids = [s["id"] for s in sentences]
    index_of = {sid: i for i, sid in enumerate(ids)}
    full_transcript = [{"id": s["id"], "text": s["text"]} for s in sentences]
    size = JEV_BLOCK_FULL if arm == "jev_full" else JEV_BLOCK_WINDOW

    jobs = []
    for block_index, target_ids in enumerate(blocks(ids, size)):
        if only_blocks is not None and block_index not in only_blocks:
            continue
        if arm == "jev_full":
            state = {"rules": rules_text, "transcript": full_transcript}
        else:
            lo = max(0, index_of[target_ids[0]] - JEV_WINDOW_PAD)
            hi = min(len(ids), index_of[target_ids[-1]] + JEV_WINDOW_PAD + 1)
            state = {"rules": rules_text, "note": WINDOW_NOTE,
                     "transcript": full_transcript[lo:hi]}
        jobs.append((block_index, target_ids, state, criteria))
    return jobs


def jev_rows_from_answers(arm, episode_name, jobs, answers_by_block, by_id,
                          repair=False):
    sentence_rows, warnings = [], []
    for block_index, target_ids, _state, _criteria in jobs:
        answers = answers_by_block.get(block_index)
        for sid in target_ids:
            row = {"arm": arm, "episode": episode_name, "id": sid,
                   "text": by_id[sid]["text"]}
            if repair:
                row["repair"] = True
            if not answers or f"s{sid}" not in answers:
                warnings.append(f"id {sid}: no answer (block {block_index})")
                row.update(score=None, probabilities=None, confidence=None,
                           retake_noul=None, defaulted=True)
            else:
                score_answer = answers[f"s{sid}"]
                row.update(score=score_answer["score"],
                           probabilities=score_answer["probabilities"],
                           confidence=score_answer["confidence"],
                           retake_noul=(answers.get(f"r{sid}") or {}).get("noul"),
                           defaulted=False)
            sentence_rows.append(row)
    return sentence_rows, warnings


def run_jev(arm, episode_name, sentences, rules_text, criteria, only_blocks=None,
            repair=False):
    """``jev_full`` re-sends the whole transcript per block; ``jev_window`` sends a
    +/-15 sentence excerpt around each block of 10 targets."""
    by_id = {s["id"]: s for s in sentences}
    jobs = jev_jobs(arm, sentences, rules_text, criteria, only_blocks)

    request_rows, answers_by_block = [], {}
    _dispatch_jev(arm, episode_name, jobs, request_rows, answers_by_block, attempt=1)
    # About 3% of requests come back as a bare PROVIDER_ERROR with no HTTP status
    # and do not reproduce. One re-attempt per failed block clears them; both
    # attempts stay in the request log.
    retry_jobs = [job for job in jobs if answers_by_block.get(job[0]) is None]
    if retry_jobs:
        _dispatch_jev(arm, episode_name, retry_jobs, request_rows, answers_by_block,
                      attempt=2)
    request_rows.sort(key=lambda r: (r["block"], r["attempt"]))

    sentence_rows, warnings = jev_rows_from_answers(
        arm, episode_name, jobs, answers_by_block, by_id, repair=repair)
    return sentence_rows, request_rows, warnings


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def load_episode_context(name):
    episode = episodes_mod.load_episode(name, str(EPISODES_DIR))
    sentences = episodes_mod.load_corpus(episode)
    preflight = episodes_mod.build_episode_preflight(episode, sentences,
                                                     str(EPISODES_DIR))
    return episode, sentences, preflight


def arm_scores_from_rows(rows):
    """``{arm: {episode: {id: score}}}`` for every scored variant."""
    raw = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        arm, ep, sid = row["arm"], row["episode"], row["id"]
        raw[arm][ep][sid] = row
    out = {}
    for arm, episodes in raw.items():
        if arm == "luna":
            out["luna"] = {ep: {sid: float(r["score"]) for sid, r in s.items()}
                           for ep, s in episodes.items()}
            continue
        plain, retake = {}, {}
        for ep, by_id in episodes.items():
            plain[ep], retake[ep] = {}, {}
            for sid, r in by_id.items():
                score = STILL_MISSING_SCORE if r["score"] is None else float(r["score"])
                plain[ep][sid] = score
                noul = r.get("retake_noul")
                retake[ep][sid] = 0.0 if (noul is not None and noul >= RETAKE_NOUL_CUT) \
                    else score
        out[arm] = plain
        out[f"{arm}_retake"] = retake
    return out


def score_arm(arm_scores, contexts, episode_names):
    """Harness scoring for one arm over ``episode_names``, pooled calibration."""
    loaded = []
    for name in episode_names:
        episode, sentences, preflight = contexts[name]
        annotated = [dict(s) for s in sentences]
        scores = arm_scores[name]
        for sentence in annotated:
            sentence["roughcut_score"] = float(scores[sentence["id"]])
        loaded.append((episode, annotated, preflight))

    thresholds = calibrate.default_thresholds()
    sweeps = [calibrate.sweep(ep, sents, thresholds=thresholds,
                              field="roughcut_score", precomputed=pre)
              for ep, sents, pre in loaded]
    weights = [pre.total_dialogue for _e, _s, pre in loaded]
    from roughcut_bench import pooling
    pooled_rows = pooling.pool_sweep_rows(sweeps, weights)
    human_kept = sum(
        end - start
        for _e, _s, pre in loaded
        for segs in pre.human_by_media.values()
        for start, end in segs
    )
    ops = calibrate.pick_operating_points(pooled_rows, human_kept)
    threshold_map = ops["thresholds"]
    neutral_threshold = ops["neutral_threshold"]

    per_episode, level_reports = {}, []
    for (episode, sentences, preflight), _w in zip(loaded, weights):
        report = replay.score_sentences(episode, sentences, threshold_map,
                                        field="roughcut_score", precomputed=preflight)
        level = report["levels"][NEUTRAL_LEVEL]
        level_reports.append(level)
        human_states = sentence_states(preflight.word_units, preflight.human_by_media,
                                       preflight.media_name, sentences,
                                       offset=preflight.offset)
        correct = kept_model = kept_human = 0
        cells = {"keep_keep": 0, "kept_but_editor_cut": 0,
                 "cut_but_editor_kept": 0, "cut_cut": 0}
        per_sentence = {}
        for sentence in sentences:
            sid = sentence["id"]
            model_keep = (sentence["roughcut_score"] >= neutral_threshold
                          and not sentence.get("is_retake"))
            human_keep = human_states[sid][0] != "removed"
            per_sentence[sid] = (model_keep, human_keep)
            correct += int(model_keep == human_keep)
            kept_model += int(model_keep)
            kept_human += int(human_keep)
            if model_keep and human_keep:
                cells["keep_keep"] += 1
            elif model_keep:
                cells["kept_but_editor_cut"] += 1
            elif human_keep:
                cells["cut_but_editor_kept"] += 1
            else:
                cells["cut_cut"] += 1
        per_episode[episode["name"]] = {
            "n_sentences": len(sentences),
            "sentence_points": level["sp_grade"],
            "sentence_points_raw": level["sp"]["raw_score"],
            "sentence_points_penalty": level["sp"]["penalty_applied"],
            "n_partial_human": level["sp"]["n_partial_human"],
            "word_score": level["word_grade"],
            "mcc": level["fair_metrics"]["mcc"],
            "frame_match": level["fair_metrics"]["super_score"],
            "grade": level["grade"],
            "keep_cut_accuracy": correct / len(sentences),
            "kept_by_model": kept_model,
            "kept_by_human": kept_human,
            "keep_cells": cells,
            "_per_sentence": per_sentence,
        }

    pooled = {
        "neutral_threshold": neutral_threshold,
        "kept_ratio": ops["achieved_ratios"][4],
        "sentence_points": pooling.pooled_sp_grade(level_reports),
        "word_score": pooling.pooled_word_grade(level_reports),
        "mcc": pooling.weighted_mean([lr["fair_metrics"]["mcc"] for lr in level_reports],
                                     weights),
        "frame_match": pooling.weighted_mean(
            [lr["fair_metrics"]["super_score"] for lr in level_reports], weights),
        "grade": pooling.weighted_mean([lr["grade"] for lr in level_reports], weights),
        "keep_cut_accuracy": (
            sum(per_episode[n]["keep_cut_accuracy"] * per_episode[n]["n_sentences"]
                for n in per_episode)
            / sum(per_episode[n]["n_sentences"] for n in per_episode)),
        "keep_cells": {
            key: sum(per_episode[n]["keep_cells"][key] for n in per_episode)
            for key in ("keep_keep", "kept_but_editor_cut", "cut_but_editor_kept",
                        "cut_cut")},
    }
    cells = pooled["keep_cells"]
    kept = cells["keep_keep"] + cells["kept_but_editor_cut"]
    editor_kept = cells["keep_keep"] + cells["cut_but_editor_kept"]
    pooled["keep_precision"] = cells["keep_keep"] / kept if kept else None
    pooled["keep_recall"] = cells["keep_keep"] / editor_kept if editor_kept else None
    return {"pooled": pooled, "episodes": per_episode}


def confidence_quartiles(rows, arm, keep_flags):
    """Keep/cut accuracy by Jev score-confidence quartile."""
    items = []
    for row in rows:
        if row["arm"] != arm or row.get("confidence") is None:
            continue
        flags = keep_flags.get(row["episode"], {}).get(row["id"])
        if flags is None:
            continue
        items.append((row["confidence"], int(flags[0] == flags[1])))
    if len(items) < 8:
        return None
    items.sort(key=lambda x: x[0])
    size = len(items) / 4
    out = []
    for q in range(4):
        chunk = items[int(q * size):int((q + 1) * size)]
        if not chunk:
            continue
        out.append({
            "quartile": q + 1,
            "n": len(chunk),
            "confidence_min": chunk[0][0],
            "confidence_max": chunk[-1][0],
            "accuracy": sum(c for _conf, c in chunk) / len(chunk),
        })
    return out


def retake_noul_analysis(rows, arm, keep_flags, contexts):
    """How the retake noul lines up with the editor's cuts and the corpus retakes."""
    nouls, human_cut, corpus_retake = [], [], []
    for row in rows:
        if row["arm"] != arm or row.get("retake_noul") is None:
            continue
        flags = keep_flags.get(row["episode"], {}).get(row["id"])
        if flags is None:
            continue
        nouls.append(row["retake_noul"])
        human_cut.append(0 if flags[1] else 1)
        sentences = contexts[row["episode"]][1]
        corpus_retake.append(int(bool(
            next(s for s in sentences if s["id"] == row["id"]).get("is_retake"))))
    if len(nouls) < 8:
        return None

    def corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return num / den if den else None

    fired = [i for i, v in enumerate(nouls) if v >= RETAKE_NOUL_CUT]
    n_cut = sum(human_cut)
    return {
        "n": len(nouls),
        "mean_noul_human_cut": (sum(v for v, c in zip(nouls, human_cut) if c) / n_cut
                                if n_cut else None),
        "mean_noul_human_kept": (
            sum(v for v, c in zip(nouls, human_cut) if not c) / (len(nouls) - n_cut)
            if len(nouls) - n_cut else None),
        "point_biserial_r_vs_human_cut": corr(nouls, human_cut),
        "point_biserial_r_vs_corpus_is_retake": corr(nouls, corpus_retake),
        "fired_at_0.6": len(fired),
        "precision_vs_human_cut": (sum(human_cut[i] for i in fired) / len(fired)
                                   if fired else None),
        "recall_vs_human_cut": (sum(human_cut[i] for i in fired) / n_cut
                                if n_cut else None),
        "precision_vs_corpus_is_retake": (sum(corpus_retake[i] for i in fired) / len(fired)
                                          if fired else None),
        "corpus_is_retake_count": sum(corpus_retake),
    }


def position_accuracy(keep_flags, contexts, bins=5):
    """Keep/cut accuracy by where the sentence sits in the episode.

    If Jev is losing the thread as the transcript grows, later bins score worse.
    """
    buckets = [{"bin": i + 1, "n": 0, "correct": 0} for i in range(bins)]
    for name, flags in keep_flags.items():
        ids = [s["id"] for s in contexts[name][1]]
        for position, sid in enumerate(ids):
            pair = flags.get(sid)
            if pair is None:
                continue
            bucket = buckets[min(bins - 1, position * bins // len(ids))]
            bucket["n"] += 1
            bucket["correct"] += int(pair[0] == pair[1])
    for bucket in buckets:
        bucket["accuracy"] = bucket["correct"] / bucket["n"] if bucket["n"] else None
    return buckets


def retake_sentence_accuracy(keep_flags, contexts):
    """Keep/cut accuracy split by the corpus's own ``is_retake`` flag."""
    out = {"is_retake": {"n": 0, "correct": 0}, "not_retake": {"n": 0, "correct": 0}}
    for name, flags in keep_flags.items():
        for sentence in contexts[name][1]:
            pair = flags.get(sentence["id"])
            if pair is None:
                continue
            key = "is_retake" if sentence.get("is_retake") else "not_retake"
            out[key]["n"] += 1
            out[key]["correct"] += int(pair[0] == pair[1])
    for block in out.values():
        block["accuracy"] = block["correct"] / block["n"] if block["n"] else None
    return out


def category_breakdown(rows, keep_flags):
    """Luna's own categories against the editor's decision."""
    out = defaultdict(lambda: {"n": 0, "human_kept": 0, "model_kept": 0, "correct": 0})
    for row in rows:
        if row["arm"] != "luna":
            continue
        flags = keep_flags.get(row["episode"], {}).get(row["id"])
        if flags is None:
            continue
        bucket = out[row["category"]]
        bucket["n"] += 1
        bucket["human_kept"] += int(flags[1])
        bucket["model_kept"] += int(flags[0])
        bucket["correct"] += int(flags[0] == flags[1])
    return {k: v for k, v in sorted(out.items(), key=lambda kv: -kv[1]["n"])}


def archived_reference(episode_names):
    path = BENCH_DIR / "results" / "2026-09-06-all18-partial-single-gpt-5.6-luna.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name in episode_names:
        entry = data.get("episodes", {}).get(name)
        if not entry:
            continue
        level = entry["levels"][NEUTRAL_LEVEL]
        out[name] = {
            "sentence_points": level["sp_grade"],
            "sentence_points_raw": (level.get("sp") or {}).get("raw_score"),
            "word_score": level["word_grade"],
            "mcc": level["fair_metrics"]["mcc"],
            "frame_match": level["fair_metrics"]["super_score"],
        }
    return {
        "run": path.name,
        "workflow": data.get("workflow"),
        "note": ("partial-capable arm: it emits sub-sentence keeps, so it escapes the "
                 "flat 0.5 no-partial penalty that every whole-sentence arm here pays"),
        "episodes": out,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(value, places=3):
    return "n/a" if value is None else f"{value:.{places}f}"


def build_notes(summary):
    lines = []
    lines.append("# Jev vs Luna on the rough-cut sentence rating")
    lines.append("")
    lines.append(f"Generated {summary['generated_utc']} by `scripts/jev_real/roughcut_bench.py`.")
    lines.append("")
    lines.append("## What was run")
    lines.append("")
    lines.append(f"Episodes: {', '.join(summary['episodes'])}.")
    lines.append("")
    lines.append(
        f"`luna` is one full-context call per episode against {LUNA_MODEL} "
        f"(reasoning_effort={LUNA_REASONING_EFFORT}, max_tokens={LUNA_MAX_TOKENS}), with "
        f"`prompts/roughcut_system_v1.md` as the system prompt and the production "
        f"`id = text` user message. Missing ids get one targeted retry; anything still "
        f"missing is filled with score {STILL_MISSING_SCORE}."
    )
    lines.append("")
    lines.append(
        f"`jev_full` sends state `{{rules, transcript}}` with the whole episode and asks "
        f"{JEV_BLOCK_FULL} sentences' worth of questions per request (a score and a "
        f"retake noul each), so the transcript is re-sent about n/{JEV_BLOCK_FULL} times. "
        f"`jev_window` asks {JEV_BLOCK_WINDOW} sentences per request over a "
        f"+/-{JEV_WINDOW_PAD} sentence excerpt. Jev requests run at concurrency "
        f"{JEV_CONCURRENCY}. The `_retake` variants reuse the same answers and force the "
        f"score to 0 when the retake noul is >= {RETAKE_NOUL_CUT}."
    )
    lines.append("")
    lines.append(
        "Scoring calls the benchmark harness itself "
        "(`roughcut_bench.calibrate` + `roughcut_bench.replay.score_sentences`), pooled "
        "calibration across the scored episodes, read at the harness's Neutral level 4."
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    header = ("| arm | episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | "
              "frame match | keep/cut acc |")
    for arm in summary["arms"]:
        block = summary["scores"][arm]
        lines.append(f"### {arm}")
        lines.append("")
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|")
        for name in summary["episodes"]:
            row = block["episodes"].get(name)
            if not row:
                continue
            lines.append(
                f"| {arm} | {name} | {_fmt(row['sentence_points'])} | "
                f"{_fmt(row['sentence_points_raw'])} | {_fmt(row['word_score'])} | "
                f"{_fmt(row['mcc'])} | {_fmt(row['frame_match'])} | "
                f"{_fmt(row['keep_cut_accuracy'])} |")
        pooled = block["pooled"]
        lines.append(
            f"| **{arm}** | **weighted** | **{_fmt(pooled['sentence_points'])}** | -- | "
            f"**{_fmt(pooled['word_score'])}** | **{_fmt(pooled['mcc'])}** | "
            f"**{_fmt(pooled['frame_match'])}** | "
            f"**{_fmt(pooled['keep_cut_accuracy'])}** |")
        lines.append("")
        lines.append(f"Calibrated Neutral threshold: {pooled['neutral_threshold']}, "
                     f"kept ratio vs the editor: {_fmt(pooled['kept_ratio'])}.")
        lines.append("")

    if summary.get("archived_reference"):
        ref = summary["archived_reference"]
        lines.append("## Archived reference")
        lines.append("")
        lines.append(f"`{ref['run']}` ({ref['workflow']}). {ref['note']}.")
        lines.append("")
        lines.append("| episode | SENTENCE POINTS | SP raw | WORD SCORE | MCC | frame match |")
        lines.append("|---|---|---|---|---|---|")
        for name, row in ref["episodes"].items():
            lines.append(
                f"| {name} | {_fmt(row['sentence_points'])} | "
                f"{_fmt(row['sentence_points_raw'])} | {_fmt(row['word_score'])} | "
                f"{_fmt(row['mcc'])} | {_fmt(row['frame_match'])} |")
        lines.append("")

    lines.append("## Cost and latency")
    lines.append("")
    lines.append("| arm | requests | input tokens | cost USD | median s | p95 s | wall-clock s |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm, block in summary["operations"]["by_arm"].items():
        lat = block["latency"] or {}
        lines.append(
            f"| {arm} | {block['requests']} | {block['input_tokens']} | "
            f"{block['cost_usd']:.4f} | {_fmt(lat.get('median_s'), 1)} | "
            f"{_fmt(lat.get('p95_s'), 1)} | {_fmt(block['wall_clock_s'], 1)} |")
    lines.append("")
    lines.append(f"Total recorded spend: ${summary['operations']['total_cost_usd']:.4f}.")
    lines.append("")
    lines.append("| arm | episode | wall-clock s | requests | cost USD |")
    lines.append("|---|---|---|---|---|")
    for key, block in summary["operations"]["by_arm_episode"].items():
        arm, name = key.split("|", 1)
        lines.append(f"| {arm} | {name} | {_fmt(block['wall_clock_s'], 1)} | "
                     f"{block['requests']} | {block['cost_usd']:.4f} |")
    lines.append("")

    if summary.get("confidence_quartiles"):
        lines.append("## Jev accuracy by confidence quartile")
        lines.append("")
        lines.append("| arm | quartile | n | confidence range | keep/cut accuracy |")
        lines.append("|---|---|---|---|---|")
        for arm, quartiles in summary["confidence_quartiles"].items():
            for q in quartiles or []:
                lines.append(
                    f"| {arm} | Q{q['quartile']} | {q['n']} | "
                    f"{_fmt(q['confidence_min'])}-{_fmt(q['confidence_max'])} | "
                    f"{_fmt(q['accuracy'])} |")
        lines.append("")

    if summary.get("retake_noul"):
        lines.append("## Retake noul vs the editor's cuts")
        lines.append("")
        lines.append("| arm | n | mean noul (human cut) | mean noul (human kept) | r vs human cut | "
                     "r vs corpus is_retake | fired >=0.6 | precision | recall |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for arm, block in summary["retake_noul"].items():
            if not block:
                continue
            lines.append(
                f"| {arm} | {block['n']} | {_fmt(block['mean_noul_human_cut'])} | "
                f"{_fmt(block['mean_noul_human_kept'])} | "
                f"{_fmt(block['point_biserial_r_vs_human_cut'])} | "
                f"{_fmt(block['point_biserial_r_vs_corpus_is_retake'])} | "
                f"{block['fired_at_0.6']} | {_fmt(block['precision_vs_human_cut'])} | "
                f"{_fmt(block['recall_vs_human_cut'])} |")
        lines.append("")

    lines.append("## Where each arm disagrees with the editor")
    lines.append("")
    lines.append("| arm | both keep | kept, editor cut | cut, editor kept | both cut | "
                 "keep precision | keep recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in summary["arms"]:
        pooled = summary["scores"][arm]["pooled"]
        cells = pooled["keep_cells"]
        lines.append(
            f"| {arm} | {cells['keep_keep']} | {cells['kept_but_editor_cut']} | "
            f"{cells['cut_but_editor_kept']} | {cells['cut_cut']} | "
            f"{_fmt(pooled['keep_precision'])} | {_fmt(pooled['keep_recall'])} |")
    lines.append("")

    if summary.get("position_accuracy"):
        lines.append("## Keep/cut accuracy by position in the episode")
        lines.append("")
        lines.append("| arm | first fifth | 2nd | 3rd | 4th | last fifth |")
        lines.append("|---|---|---|---|---|---|")
        for arm, buckets in summary["position_accuracy"].items():
            cells = " | ".join(_fmt(b["accuracy"]) for b in buckets)
            lines.append(f"| {arm} | {cells} |")
        lines.append("")

    if summary.get("retake_sentence_accuracy"):
        lines.append("## Keep/cut accuracy on the corpus's retake sentences")
        lines.append("")
        lines.append("| arm | retake n | retake accuracy | other n | other accuracy |")
        lines.append("|---|---|---|---|---|")
        for arm, block in summary["retake_sentence_accuracy"].items():
            lines.append(
                f"| {arm} | {block['is_retake']['n']} | "
                f"{_fmt(block['is_retake']['accuracy'])} | {block['not_retake']['n']} | "
                f"{_fmt(block['not_retake']['accuracy'])} |")
        lines.append("")

    if summary.get("luna_categories"):
        lines.append("## Luna categories vs the editor")
        lines.append("")
        lines.append("| category | n | model kept | human kept | keep/cut accuracy |")
        lines.append("|---|---|---|---|---|")
        for name, row in summary["luna_categories"].items():
            lines.append(f"| {name} | {row['n']} | {row['model_kept']} | "
                         f"{row['human_kept']} | {_fmt(row['correct'] / row['n'])} |")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


CAVEATS = [
    "SENTENCE POINTS punishes every whole-sentence arm here with the harness's flat 0.5 "
    "no-partial penalty on any episode where the editor trimmed inside sentences, because "
    "none of these arms can emit sub-sentence keeps. The penalty is identical across arms, "
    "so the ranking holds; the `SP raw` column is the same score before the penalty.",
    "The corpus already carries `is_retake` from solar-sailer's own retakes pass, and "
    "`ranges.kept_segments_from_score` cuts those sentences for EVERY arm regardless of "
    "model score. That deterministic layer sits underneath all five arms and dampens the "
    "measurable effect of Jev's retake noul, especially on edges-7.01-intro (169 of 389 "
    "sentences are flagged).",
    "The keep threshold is calibrated per arm by the harness (the threshold that maximises "
    "the retired GRADE, pooled across the scored episodes), so the arms are compared at "
    "each one's own best operating point rather than at a fixed cut-off.",
    "The archived Luna run 2026-07-15-13d-v1-single-gpt-5.6-luna.json records "
    "model/prompt/max_tokens but not reasoning effort, so the Luna arm here uses medium.",
    "Costs are token-based estimates from the recorded usage, not an invoice. Failed and "
    "retried requests may add unreported provider charges.",
    "Read-only against solar-sailer: the answer-key loader is monkeypatched to read the "
    "committed cache rather than re-extract and rewrite it.",
    "This benchmark could not run until a router bug was fixed. "
    "`skell_e_router.classification._parse` required a score answer's probabilities to "
    "sum to 1.000 +/- 0.001 and to reproduce the reported score within 0.001 per level. "
    "Jev rounds both to two decimals, so a six-level score legitimately sums to 0.99 and "
    "drifts up to 0.04 from its score. About a third of live requests were rejected as "
    "malformed. Both tolerances now derive from that rounding.",
    "6 of 239 requests came back as a bare PROVIDER_ERROR with no HTTP status and did not "
    "reproduce on a re-request. Each was re-issued once; all succeeded, and both attempts "
    "are in roughcut-requests.jsonl (the repaired rows carry `repair: true`). Cause "
    "unknown, roughly a 2.5% sporadic failure rate.",
]


def build_summary(rows, request_rows, contexts, episode_names):
    scores_by_arm = arm_scores_from_rows(rows)
    arms = [a for a in SCORED_ARMS if a in scores_by_arm]

    scored, keep_flags_by_arm = {}, {}
    for arm in arms:
        covered = [n for n in episode_names if n in scores_by_arm[arm]
                   and len(scores_by_arm[arm][n]) == len(contexts[n][1])]
        if not covered:
            continue
        result = score_arm(scores_by_arm[arm], contexts, covered)
        keep_flags_by_arm[arm] = {
            name: result["episodes"][name].pop("_per_sentence")
            for name in result["episodes"]
        }
        result["episodes_scored"] = covered
        scored[arm] = result
    arms = list(scored)

    by_arm = defaultdict(lambda: {"requests": 0, "errors": 0, "input_tokens": 0,
                                  "output_tokens": 0, "cost_usd": 0.0,
                                  "elapsed": [], "wall_clock_s": 0.0})
    by_arm_episode = defaultdict(lambda: {"requests": 0, "cost_usd": 0.0,
                                          "wall_clock_s": 0.0, "errors": 0})
    for row in request_rows:
        arm = row["arm"]
        target = by_arm[arm]
        target["requests"] += 1
        target["errors"] += int(bool(row.get("error")))
        target["input_tokens"] += row.get("input_tokens") or 0
        target["output_tokens"] += row.get("output_tokens") or 0
        target["cost_usd"] += row.get("cost_usd") or 0.0
        target["elapsed"].append(row.get("elapsed_s") or 0.0)
        key = f"{arm}|{row['episode']}"
        by_arm_episode[key]["requests"] += 1
        by_arm_episode[key]["cost_usd"] += row.get("cost_usd") or 0.0
        by_arm_episode[key]["errors"] += int(bool(row.get("error")))
        by_arm_episode[key]["wall_clock_s"] = max(
            by_arm_episode[key]["wall_clock_s"], row.get("wall_clock_s") or 0.0)
    for arm, block in by_arm.items():
        block["latency"] = latency_summary(block.pop("elapsed"))
        block["wall_clock_s"] = sum(
            v["wall_clock_s"] for k, v in by_arm_episode.items()
            if k.split("|", 1)[0] == arm)

    quartiles, noul = {}, {}
    for arm in ("jev_full", "jev_window"):
        if arm in keep_flags_by_arm:
            quartiles[arm] = confidence_quartiles(rows, arm, keep_flags_by_arm[arm])
            noul[arm] = retake_noul_analysis(rows, arm, keep_flags_by_arm[arm], contexts)
        retake_arm = f"{arm}_retake"
        if retake_arm in keep_flags_by_arm:
            noul[retake_arm] = retake_noul_analysis(
                rows, arm, keep_flags_by_arm[retake_arm], contexts)

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": [n for n in episode_names if n in contexts],
        "arms": arms,
        "settings": {
            "luna_model": LUNA_MODEL, "luna_reasoning_effort": LUNA_REASONING_EFFORT,
            "luna_max_tokens": LUNA_MAX_TOKENS, "prompt": str(PROMPT_PATH),
            "jev_model": JEV_MODEL, "jev_block_full": JEV_BLOCK_FULL,
            "jev_block_window": JEV_BLOCK_WINDOW, "jev_window_pad": JEV_WINDOW_PAD,
            "jev_concurrency": JEV_CONCURRENCY, "retake_noul_cut": RETAKE_NOUL_CUT,
            "still_missing_score": STILL_MISSING_SCORE,
            "neutral_level": NEUTRAL_LEVEL,
        },
        "scores": scored,
        "operations": {
            "by_arm": dict(by_arm),
            "by_arm_episode": dict(by_arm_episode),
            "total_cost_usd": sum(b["cost_usd"] for b in by_arm.values()),
            "total_requests": sum(b["requests"] for b in by_arm.values()),
            "total_errors": sum(b["errors"] for b in by_arm.values()),
        },
        "confidence_quartiles": quartiles,
        "retake_noul": noul,
        "position_accuracy": {arm: position_accuracy(flags, contexts)
                              for arm, flags in keep_flags_by_arm.items()},
        "retake_sentence_accuracy": {arm: retake_sentence_accuracy(flags, contexts)
                                     for arm, flags in keep_flags_by_arm.items()},
        "luna_categories": (category_breakdown(rows, keep_flags_by_arm["luna"])
                            if "luna" in keep_flags_by_arm else None),
        "archived_reference": archived_reference(episode_names),
        "caveats": CAVEATS,
    }


# ---------------------------------------------------------------------------
# plan / main
# ---------------------------------------------------------------------------

def estimate(arms, episode_names, contexts, rules_text):
    """Conservative token/spend estimate. 4 chars per token, no cache credit."""
    rules_tokens = len(rules_text) / 4
    plan, total = [], 0.0
    for name in episode_names:
        sentences = contexts[name][1]
        n = len(sentences)
        transcript_chars = sum(len(str(s["id"])) + len(s["text"]) + 20 for s in sentences)
        transcript_tokens = transcript_chars / 4
        user_tokens = sum(len(str(s["id"])) + len(s["text"]) + 4 for s in sentences) / 4
        for arm in arms:
            if arm == "luna":
                inp = user_tokens + len(PROMPT_PATH.read_text(encoding="utf-8")) / 4
                out = n * 22 + 6000  # ratings plus a generous reasoning allowance
                cost = (inp * LUNA_IN + out * LUNA_OUT) / 1_000_000
                requests, tokens = 1, inp
            else:
                if arm == "jev_full":
                    n_blocks = math.ceil(n / JEV_BLOCK_FULL)
                    state_tokens = rules_tokens + transcript_tokens
                    q_tokens = JEV_BLOCK_FULL * 230
                else:
                    n_blocks = math.ceil(n / JEV_BLOCK_WINDOW)
                    span = min(n, JEV_BLOCK_WINDOW + 2 * JEV_WINDOW_PAD)
                    state_tokens = rules_tokens + transcript_tokens * span / n + 20
                    q_tokens = JEV_BLOCK_WINDOW * 230
                tokens = n_blocks * (state_tokens + q_tokens)
                cost = tokens * JEV_INPUT_PER_MILLION / 1_000_000
                requests = n_blocks
            plan.append({"arm": arm, "episode": name, "sentences": n,
                         "requests": requests, "est_input_tokens": round(tokens),
                         "est_cost_usd": round(cost, 4)})
            total += cost
    return plan, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", default=EPISODES,
                        help="episode ids to call (default: all five)")
    parser.add_argument("--arms", nargs="+", default=API_ARMS, choices=API_ARMS)
    parser.add_argument("--run", action="store_true", help="actually call the APIs")
    parser.add_argument("--score-only", action="store_true",
                        help="rebuild the summary and notes from the recorded jsonl")
    parser.add_argument("--repair", action="store_true",
                        help="re-request only the Jev blocks whose sentences have no "
                             "answer; appends corrected rows, never rewrites history")
    parser.add_argument("--budget", type=float, default=BUDGET_CAP_USD)
    args = parser.parse_args()

    unknown = [e for e in args.episodes if e not in EPISODES]
    if unknown:
        parser.error(f"unknown episode(s): {unknown}")

    system_prompt, rules_text, criteria = load_prompt_parts()

    contexts = {}
    for name in EPISODES:
        try:
            contexts[name] = load_episode_context(name)
        except Exception as exc:
            print(f"episode {name}: preflight failed: {exc}", file=sys.stderr)

    if not args.run and not args.score_only and not args.repair:
        plan, total = estimate(args.arms, args.episodes, contexts, rules_text)
        print(json.dumps({
            "mode": "plan",
            "episodes": args.episodes,
            "api_arms": args.arms,
            "scored_variants": SCORED_ARMS,
            "plan": plan,
            "estimated_spend_usd": round(total, 4),
            "budget_cap_usd": args.budget,
            "outputs": [str(RESULTS_PATH), str(REQUESTS_PATH), str(SUMMARY_PATH),
                        str(NOTES_PATH)],
        }, indent=2))
        print("\nRules text sent to Jev as state.rules:\n", file=sys.stderr)
        print(rules_text, file=sys.stderr)
        return

    existing_rows = jsonl_read(RESULTS_PATH)
    existing_requests = jsonl_read(REQUESTS_PATH)

    if args.repair:
        latest = {}
        for row in existing_rows:
            latest[(row["arm"], row["episode"], row["id"])] = row
        gaps = defaultdict(set)
        for (arm, name, sid), row in latest.items():
            if arm.startswith("jev") and row.get("score") is None:
                size = JEV_BLOCK_FULL if arm == "jev_full" else JEV_BLOCK_WINDOW
                ids = [s["id"] for s in contexts[name][1]]
                gaps[(arm, name)].add(ids.index(sid) // size)
        if not gaps:
            print("Nothing to repair.", file=sys.stderr)
        for (arm, name), block_set in sorted(gaps.items()):
            _episode, sentences, _pre = contexts[name]
            started = time.perf_counter()
            sentence_rows, request_rows, warnings = run_jev(
                arm, name, sentences, rules_text, criteria,
                only_blocks=block_set, repair=True)
            wall = time.perf_counter() - started
            stamp = datetime.now(timezone.utc).isoformat()
            for row in request_rows:
                row["wall_clock_s"] = 0.0  # repair time is not the arm's wall clock
                row["repair"] = True
                row["recorded_utc"] = stamp
            jsonl_append(REQUESTS_PATH, request_rows)
            jsonl_append(RESULTS_PATH, sentence_rows)
            existing_rows.extend(sentence_rows)
            existing_requests.extend(request_rows)
            print(json.dumps({
                "repaired": f"{arm}/{name}", "blocks": sorted(block_set),
                "sentences": len(sentence_rows), "elapsed_s": round(wall, 1),
                "still_missing": len([r for r in sentence_rows if r["score"] is None]),
                "warnings": warnings[:3],
            }), flush=True)

    if args.run:
        done = {(r["arm"], r["episode"]) for r in existing_rows}
        todo = [(arm, name) for arm in args.arms for name in args.episodes
                if (arm, name) not in done and name in contexts]
        skipped = [(arm, name) for arm in args.arms for name in args.episodes
                   if (arm, name) in done]
        if skipped:
            print(f"Already recorded, refusing to overwrite: {skipped}", file=sys.stderr)
        if not todo:
            print("Nothing new to run.", file=sys.stderr)
        else:
            plan, total = estimate(
                sorted({a for a, _ in todo}), sorted({e for _, e in todo}),
                contexts, rules_text)
            spent = sum(r.get("cost_usd") or 0.0 for r in existing_requests)
            if total + spent > args.budget:
                parser.error(
                    f"estimated ${total:.4f} plus ${spent:.4f} already spent exceeds the "
                    f"${args.budget:.2f} cap")
            print(json.dumps({"running": todo, "estimated_spend_usd": round(total, 4),
                              "already_spent_usd": round(spent, 4)}), flush=True)

        for arm, name in todo:
            _episode, sentences, _pre = contexts[name]
            started = time.perf_counter()
            if arm == "luna":
                sentence_rows, request_rows, warnings = run_luna(
                    name, sentences, system_prompt)
            else:
                sentence_rows, request_rows, warnings = run_jev(
                    arm, name, sentences, rules_text, criteria)
            wall = time.perf_counter() - started
            stamp = datetime.now(timezone.utc).isoformat()
            for row in request_rows:
                row["wall_clock_s"] = wall
                row["recorded_utc"] = stamp
            jsonl_append(REQUESTS_PATH, request_rows)
            jsonl_append(RESULTS_PATH, sentence_rows)
            existing_rows.extend(sentence_rows)
            existing_requests.extend(request_rows)
            cost = sum(r.get("cost_usd") or 0.0 for r in request_rows)
            errors = [r["error"] for r in request_rows if r.get("error")]
            print(json.dumps({
                "arm": arm, "episode": name, "sentences": len(sentence_rows),
                "requests": len(request_rows), "cost_usd": round(cost, 5),
                "wall_clock_s": round(wall, 1), "errors": errors[:3],
                "warnings": warnings[:3],
            }), flush=True)

    if not existing_rows:
        print("No recorded rows to score.", file=sys.stderr)
        return

    present = [n for n in EPISODES if n in contexts
               and any(r["episode"] == n for r in existing_rows)]
    summary = build_summary(existing_rows, existing_requests, contexts, present)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    NOTES_PATH.write_text(build_notes(summary), encoding="utf-8")
    print(json.dumps({
        "scored_arms": summary["arms"],
        "episodes": summary["episodes"],
        "total_cost_usd": round(summary["operations"]["total_cost_usd"], 4),
        "wrote": [str(SUMMARY_PATH), str(NOTES_PATH)],
    }, indent=2))


if __name__ == "__main__":
    main()
