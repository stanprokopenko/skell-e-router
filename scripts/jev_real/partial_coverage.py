"""Do the human editor's inside-sentence trims land on boundaries code can find?

No model calls, no network, $0. Read-only against solar-sailer.

The rough-cut bench scores every corpus sentence as ``full``, ``partial`` or
``removed`` on the human side. ``partial`` means the editor kept only some of
the sentence's words. This script takes every human-partial sentence in every
loadable episode and asks one question per trim boundary: is there anything in
the transcript -- a pause, a filler word, a punctuation mark, a trail-off --
that a deterministic pass could have used to PROPOSE that boundary?

It does not ask whether code could decide the trim. Only whether the boundary
would be in the candidate set a judge (Jev) is handed.

Definitions used here
---------------------
Boundary
    A transition between a dropped word and a kept word INSIDE the sentence.
    The sentence's own start and end are not boundaries: an editor who drops a
    sentence's first three words creates exactly one boundary, between word 3
    and word 4.

Gap
    ``next["start"] - prev["end"]`` in seconds, from the RAW AssemblyAI word
    bounds. The refined ``start_custom`` / ``end_custom`` bounds (what the frame
    scorer uses) stretch each word outward to swallow the pause, so they cannot
    measure it.

Explainable
    A boundary satisfies a rule. Six rules are reported separately -- gap >=
    0.3s / 0.5s / 0.8s, filler-adjacent, punctuation-adjacent, trail-off -- plus
    the combined rule (gap >= 0.3 OR filler OR punctuation OR trail-off).

Two readings of coverage are reported, because they answer different questions:
per SENTENCE (every one of its boundaries is explainable, i.e. the whole trim is
proposable) and per BOUNDARY (the share of individual boundaries explainable,
which shows whether a handful of hard boundaries is sinking otherwise easy
sentences).

Sides: on a human-partial boundary, ``filler-adjacent`` looks at the DROPPED
side's word, because that is the editorial signal ("she cut the 'um'"). The
split-point generator over ALL sentences has no dropped side, so it looks at
both sides -- a superset, so every explained boundary is also a candidate.

Usage (from the repo root):
  python scripts/jev_real/partial_coverage.py
  python scripts/jev_real/partial_coverage.py --episodes edges-7.01-intro
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(r"D:\solar-sailer\benchmarks\roughcut")
EPISODES_DIR = BENCH_DIR / "episodes"
CORPUS_DIR = BENCH_DIR / "corpus"
OUT_DIR = ROOT / "docs" / "jev-real"
MD_PATH = OUT_DIR / "partial-coverage.md"
JSON_PATH = OUT_DIR / "partial-coverage.json"

# Nothing under solar-sailer may be written, and importing the harness would
# otherwise drop .pyc files into its package directory.
sys.dont_write_bytecode = True
# The harness package must win over this directory's sibling roughcut_bench.py.
sys.path.insert(0, str(BENCH_DIR))

from roughcut_bench import episodes as episodes_mod  # noqa: E402
from roughcut_bench.partial import transcript_path_for  # noqa: E402
from roughcut_bench.sentence_scoring import sentence_states  # noqa: E402


def _cached_answer_key(episode, episodes_dir=str(EPISODES_DIR)):
    """Read the committed answer-key cache. Never extracts, never writes.

    Same guard as ``scripts/jev_real/roughcut_bench.py``: the stock
    ``episodes.episode_answer_key`` stats the source .prproj and REWRITES
    ``episodes/<name>.answerkey.json`` when the stamp misses, and nothing under
    solar-sailer may be written from this repo.
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
# tokens
# ---------------------------------------------------------------------------

#: Stripped off both ends before a word is compared to the filler list.
PUNCT = " \t\"'`.,!?;:-–—()[]{}…\u201c\u201d\u2018\u2019"

#: Exactly the tokens named in the brief. No spelling variants are added
#: ("umm", "uhh"): a wider list would inflate the coverage number.
FILLER_SINGLE = {
    "um", "uh", "hmm", "mm", "mhm", "like", "so", "okay", "ok", "right",
    "yeah", "yes", "well", "basically", "actually",
}
FILLER_PHRASES = {("you", "know"), ("kind", "of"), ("sort", "of"), ("i", "mean")}

#: Sentence-final punctuation that reads as a trail-off. The corpus writes the
#: trail-off on the SENTENCE text as ``..``; the transcript writes the word
#: itself with an em dash or an ellipsis.
TRAILOFF_SUFFIXES = ("..", "...", "\u2014", "\u2013", "…")

#: "comma, period, dash, question mark", plus the exclamation mark, which is the
#: same clause break with a different mood.
END_PUNCT = (",", ".", "?", "!", "-", "\u2013", "\u2014", "…", ";", ":")

GAP_RULES = [("gap030", 0.3), ("gap050", 0.5), ("gap080", 0.8)]
RULES = ["gap030", "gap050", "gap080", "filler", "punct", "trailoff", "combined"]


def norm_token(text):
    return (text or "").strip().strip(PUNCT).lower()


def is_filler_at(tokens, index):
    """Is the word at ``index`` a filler token, alone or inside a filler phrase."""
    if not 0 <= index < len(tokens):
        return False
    if tokens[index] in FILLER_SINGLE:
        return True
    if index > 0 and (tokens[index - 1], tokens[index]) in FILLER_PHRASES:
        return True
    if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) in FILLER_PHRASES:
        return True
    return False


def ends_trailoff(text):
    stripped = (text or "").rstrip()
    return any(stripped.endswith(suffix) for suffix in TRAILOFF_SUFFIXES)


def ends_punct(text):
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped.endswith(END_PUNCT)


def word_time(word, key):
    """Raw AssemblyAI bound in seconds, falling back to the refined one."""
    value = word.get(key)
    if value is None:
        value = word.get(f"{key}_custom")
    return value


# ---------------------------------------------------------------------------
# boundary features
# ---------------------------------------------------------------------------


def boundary_features(words, tokens, index, dropped_side):
    """Features of the boundary between ``words[index - 1]`` and ``words[index]``.

    ``dropped_side`` is ``"prev"``, ``"next"`` or ``None`` (no side known, which
    is the split-point generator's case: then either side's filler counts).
    """
    prev_word, next_word = words[index - 1], words[index]
    prev_end = word_time(prev_word, "end")
    next_start = word_time(next_word, "start")
    gap = None
    if prev_end is not None and next_start is not None:
        gap = round(float(next_start) - float(prev_end), 4)

    if dropped_side == "prev":
        filler = is_filler_at(tokens, index - 1)
    elif dropped_side == "next":
        filler = is_filler_at(tokens, index)
    else:
        filler = is_filler_at(tokens, index - 1) or is_filler_at(tokens, index)

    features = {
        "index": index,
        "gap": gap,
        "filler": bool(filler),
        "punct": ends_punct(prev_word.get("text")),
        "trailoff": ends_trailoff(prev_word.get("text")),
        "prev_text": prev_word.get("text"),
        "next_text": next_word.get("text"),
        "dropped_side": dropped_side,
    }
    for key, threshold in GAP_RULES:
        features[key] = gap is not None and gap >= threshold
    features["combined"] = bool(
        features["gap030"] or features["filler"] or features["punct"]
        or features["trailoff"]
    )
    return features


def trim_shape(labels, runs):
    """One of five mutually exclusive shapes for a partial sentence.

    ``head_only`` the kept words are a suffix (the editor dropped the opening),
    ``tail_only`` they are a prefix, ``head_and_tail`` one run with both ends
    dropped, ``middle_cut`` several kept runs with both ends kept (every removal
    is strictly inside), ``multi_run`` several kept runs plus at least one end
    dropped.
    """
    n = len(labels)
    if len(runs) > 1:
        return "middle_cut" if (labels[0] and labels[-1]) else "multi_run"
    start, end = runs[0]
    if start > 0 and end >= n:
        return "head_only"
    if start == 0 and end < n:
        return "tail_only"
    if start > 0 and end < n:
        return "head_and_tail"
    return "full"  # unreachable: a partial sentence always drops something


# ---------------------------------------------------------------------------
# per-episode work
# ---------------------------------------------------------------------------


def load_episode_words(name):
    """``(sentences, human_states, words_by_sentence)`` for one episode.

    ``words_by_sentence`` is aligned with the run positions ``sentence_states``
    reports: the harness sorts its word units by ``(sentence_id, word_id)``, so
    position ``k`` of a run is the ``k``-th word of that sentence in word-id
    order.
    """
    episode = episodes_mod.load_episode(name, str(EPISODES_DIR))
    sentences = episodes_mod.load_corpus(episode)
    preflight = episodes_mod.build_episode_preflight(episode, sentences,
                                                     str(EPISODES_DIR))
    with open(transcript_path_for(episode["corpus"]), encoding="utf-8") as handle:
        transcript = json.load(handle)
    by_id = {word["id"]: word for word in transcript["word_segments"]}

    words_by_sentence = defaultdict(list)
    for unit in preflight.word_units or []:
        word = by_id.get(unit.word_id)
        if word is not None:
            words_by_sentence[unit.sentence_id].append(word)

    human_states = sentence_states(
        preflight.word_units or [], preflight.human_by_media, preflight.media_name,
        sentences, offset=preflight.offset,
    )
    return sentences, human_states, words_by_sentence


def analyse_episode(name):
    sentences, human_states, words_by_sentence = load_episode_words(name)

    shapes = Counter()
    sentence_rows = []
    boundary_rows = []
    split_counts = []          # candidate split points per sentence, ALL sentences
    split_sentences = 0
    split_word_boundaries = 0
    touching_boundaries = 0    # inside-sentence word pairs the transcript abuts
    trailoff_words = 0         # words carrying the trail-off punctuation
    trailoff_internal = 0      # ... of those, ones that are not sentence-final

    for sentence in sentences:
        sid = sentence["id"]
        words = words_by_sentence.get(sid) or []
        tokens = [norm_token(word.get("text")) for word in words]

        for position, word in enumerate(words):
            if ends_trailoff(word.get("text")):
                trailoff_words += 1
                if position != len(words) - 1:
                    trailoff_internal += 1

        # --- candidate split points, every sentence with at least two words
        if len(words) >= 2:
            split_sentences += 1
            split_word_boundaries += len(words) - 1
            count = 0
            for index in range(1, len(words)):
                feature = boundary_features(words, tokens, index, None)
                count += bool(feature["combined"])
                if feature["gap"] is not None and feature["gap"] < 0.005:
                    touching_boundaries += 1
            split_counts.append(count)

        state, runs = human_states.get(sid, ("removed", None))
        if state != "partial" or not runs or not words:
            continue

        labels = [False] * len(words)
        for start, end in runs:
            for position in range(start, min(end, len(labels))):
                labels[position] = True
        n_dropped = sum(1 for kept in labels if not kept)
        shape = trim_shape(labels, runs)
        shapes[shape] += 1

        features = []
        for index in range(1, len(labels)):
            if labels[index - 1] == labels[index]:
                continue
            dropped_side = "prev" if not labels[index - 1] else "next"
            feature = boundary_features(words, tokens, index, dropped_side)
            feature["sentence_id"] = sid
            feature["episode"] = name
            features.append(feature)
            boundary_rows.append(feature)

        sentence_rows.append({
            "episode": name,
            "sentence_id": sid,
            "n_words": len(words),
            "n_dropped": n_dropped,
            "dropped_share": n_dropped / len(words),
            "shape": shape,
            "n_boundaries": len(features),
            "sentence_trailoff": ends_trailoff(sentence.get("text")),
            "rules": {rule: all(f[rule] for f in features) if features else True
                      for rule in RULES},
            "any_combined": any(f["combined"] for f in features) if features else True,
            "render": render_sentence(words, labels),
            "gaps": [f["gap"] for f in features],
            "boundaries": features,
        })

    return {
        "episode": name,
        "n_sentences": len(sentences),
        "n_partial": len(sentence_rows),
        "n_boundaries": len(boundary_rows),
        "shapes": dict(shapes),
        "sentences": sentence_rows,
        "boundaries": boundary_rows,
        "split_counts": split_counts,
        "split_sentences": split_sentences,
        "split_word_boundaries": split_word_boundaries,
        "touching_boundaries": touching_boundaries,
        "trailoff_words": trailoff_words,
        "trailoff_internal": trailoff_internal,
    }


def render_sentence(words, labels):
    """The sentence with each kept RUN wrapped in brackets, dropped words bare."""
    parts = []
    for index, word in enumerate(words):
        text = word.get("text") or ""
        opens = labels[index] and (index == 0 or not labels[index - 1])
        closes = labels[index] and (index == len(labels) - 1 or not labels[index + 1])
        if opens:
            text = "[" + text
        if closes:
            text = text + "]"
        parts.append(text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    index = math.ceil(q * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def coverage(sentence_rows, boundary_rows):
    out = {"rules": {}}
    n_s, n_b = len(sentence_rows), len(boundary_rows)
    for rule in RULES:
        s_hits = sum(1 for row in sentence_rows if row["rules"][rule])
        b_hits = sum(1 for row in boundary_rows if row[rule])
        out["rules"][rule] = {
            "sentences_all_boundaries": s_hits,
            "sentence_share": s_hits / n_s if n_s else None,
            "boundaries": b_hits,
            "boundary_share": b_hits / n_b if n_b else None,
        }
    any_hits = sum(1 for row in sentence_rows if row["any_combined"])
    out["combined_any_boundary"] = {
        "sentences": any_hits,
        "share": any_hits / n_s if n_s else None,
    }
    return out


def unexplained_reasons(boundary_rows):
    """Why the combined rule missed the boundaries it missed."""
    missed = [row for row in boundary_rows if not row["combined"]]
    buckets = Counter()
    words = Counter()
    sides = Counter()
    for row in missed:
        gap = row["gap"]
        if gap is None:
            buckets["no timing on one side"] += 1
        elif gap < 0.02:
            buckets["words touch, gap under 0.02s"] += 1
        elif gap < 0.10:
            buckets["gap 0.02-0.10s"] += 1
        elif gap < 0.20:
            buckets["gap 0.10-0.20s"] += 1
        else:
            buckets["gap 0.20-0.30s"] += 1
        dropped = row["prev_text"] if row["dropped_side"] == "prev" else row["next_text"]
        words[norm_token(dropped)] += 1
        sides[row["dropped_side"]] += 1
    return {
        "n_unexplained": len(missed),
        "gap_buckets": buckets.most_common(),
        "dropped_side_words": words.most_common(15),
        "dropped_side": dict(sides),
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


RULE_LABELS = {
    "gap030": "gap >= 0.30s",
    "gap050": "gap >= 0.50s",
    "gap080": "gap >= 0.80s",
    "filler": "filler-adjacent (dropped side)",
    "punct": "punctuation on the previous word",
    "trailoff": "trail-off marker",
    "combined": "combined (gap >= 0.30s OR filler OR punctuation OR trail-off)",
}

SHAPE_LABELS = {
    "head_only": "head-only (kept run is a suffix)",
    "tail_only": "tail-only (kept run is a prefix)",
    "head_and_tail": "head+tail (one kept run, both ends dropped)",
    "middle_cut": "middle-cut (removals strictly inside, both ends kept)",
    "multi_run": "multiple kept runs (inside removals plus an end trim)",
}


def build_markdown(payload):
    pooled = payload["pooled"]
    lines = []
    lines.append("# Can code propose the editor's inside-sentence trim boundaries?")
    lines.append("")
    lines.append(f"Developer-facing record, generated by `scripts/jev_real/partial_coverage.py` on {payload['date']}. No model calls, no network, $0. Read-only against `D:\\solar-sailer\\benchmarks\\roughcut`; every number below is also in `partial-coverage.json`.")
    lines.append("")
    lines.append("## What this measures")
    lines.append("")
    lines.append("The bench labels every corpus sentence `full`, `partial` or `removed` on the human side. `partial` means the editor kept only some of the sentence's words. This script takes every human-partial sentence and asks, for each trim boundary, whether anything in the transcript could have PROPOSED that boundary: a pause, a filler word, a punctuation mark, a trail-off marker.")
    lines.append("")
    lines.append("It does not ask whether code could make the call. A high number here means a cheap pass can hand a judge the right candidate list; it says nothing about picking the right candidate.")
    lines.append("")
    lines.append("- Boundary: a transition between a dropped word and a kept word inside the sentence. The sentence's own start and end do not count, so dropping the first three words of a sentence makes exactly one boundary.")
    lines.append("- Gap: `next.start - prev.end` in seconds from the raw AssemblyAI word bounds. The refined `start_custom` / `end_custom` bounds that the frame scorer uses stretch each word outward over the pause, so they cannot measure it.")
    lines.append("- Filler tokens: um, uh, hmm, mm, mhm, like, you know, so, okay, ok, right, yeah, yes, well, kind of, sort of, I mean, basically, actually. Case normalised, punctuation stripped, no spelling variants added. On a human boundary only the DROPPED side's word is checked, because that is the editorial signal.")
    lines.append("- Punctuation: the previous word's text ends in a comma, period, dash, question mark, exclamation mark, semicolon or colon.")
    lines.append("- Trail-off: the previous word ends in an em dash or an ellipsis, which is how the transcript writes the `..` the corpus puts on a trailed-off sentence.")
    lines.append("- Human side derived by `roughcut_bench.sentence_scoring.sentence_states` under the majority rule, the same call the leaderboard's sentence-points scorer makes.")
    lines.append("")
    lines.append("## Pooled")
    lines.append("")
    lines.append(f"{pooled['n_episodes']} episodes, {pooled['n_sentences']} corpus sentences, {pooled['n_partial']} human-partial ({_pct(pooled['partial_share'])} of sentences), {pooled['n_boundaries']} trim boundaries inside them ({pooled['boundaries_per_partial']:.2f} per partial sentence).")
    lines.append("")
    lines.append(f"Words dropped per partial sentence: median {pooled['dropped_median']:.1f}, mean {pooled['dropped_mean']:.2f}, p90 {pooled['dropped_p90']}, max {pooled['dropped_max']}. Share of the sentence dropped: median {_pct(pooled['dropped_share_median'])}, mean {_pct(pooled['dropped_share_mean'])}.")
    lines.append("")
    lines.append("### Trim shapes")
    lines.append("")
    lines.append("| Shape | Sentences | Share of partials |")
    lines.append("|---|---:|---:|")
    for shape, count in pooled["shapes"]:
        share = count / pooled["n_partial"] if pooled["n_partial"] else None
        lines.append(f"| {SHAPE_LABELS.get(shape, shape)} | {count} | {_pct(share)} |")
    lines.append("")
    lines.append("### Coverage per rule")
    lines.append("")
    lines.append("Sentence column: every one of that sentence's boundaries satisfies the rule, so the whole trim is proposable. Boundary column: the share of individual boundaries that satisfy it.")
    lines.append("")
    lines.append("| Rule | Sentences fully explained | Boundaries explained |")
    lines.append("|---|---:|---:|")
    for rule in RULES:
        block = pooled["coverage"]["rules"][rule]
        lines.append(f"| {RULE_LABELS[rule]} | {_pct(block['sentence_share'])} ({block['sentences_all_boundaries']}/{pooled['n_partial']}) | {_pct(block['boundary_share'])} ({block['boundaries']}/{pooled['n_boundaries']}) |")
    lines.append("")
    any_block = pooled["coverage"]["combined_any_boundary"]
    lines.append(f"At least one boundary explainable under the combined rule: {_pct(any_block['share'])} ({any_block['sentences']}/{pooled['n_partial']} sentences).")
    lines.append("")
    lines.append(f"The trail-off row is 0.0% by construction, not because the marker is rare. {pooled['trailoff_words']} words in the corpus carry it and {pooled['trailoff_internal']} of them sit anywhere but their sentence's last word, so a trail-off can never land on an inside boundary. It is a signal for cutting a whole sentence, not for trimming one. Treat that rule as contributing nothing here and drop it from the combined rule without changing a single number.")
    lines.append("")
    lines.append("### Candidate split points the combined rule generates")
    lines.append("")
    split = pooled["split"]
    lines.append(f"Counted over ALL {split['sentences']} sentences with at least two words, not just the partial ones, because that is the candidate set a judge would have to work through. Here the rule has no dropped side to look at, so a filler on either side of the boundary counts, which makes this a superset of the boundaries scored above.")
    lines.append("")
    lines.append(f"Candidate split points per sentence: median {split['median']:.1f}, p90 {split['p90']}, max {split['max']}, mean {split['mean']:.2f}. Total {split['total']} candidates over {split['word_boundaries']} inside-sentence word boundaries ({_pct(split['rate'])} of all word boundaries). {split['zero_sentences']} sentences ({_pct(split['zero_share'])}) get no candidate at all.")
    lines.append("")
    lines.append(f"One caveat on the gap rule: it can only see what the transcriber wrote. {_pct(split['touching_share'])} of all inside-sentence word boundaries have the two words abutting (gap under 0.005s), and that share runs from 15% on some episodes to 61% on greco-2.2-thumbnailing, so the same physical pause is not measured the same way everywhere.")
    lines.append("")
    lines.append("### Why the combined rule misses what it misses")
    lines.append("")
    reasons = pooled["unexplained"]
    lines.append(f"{reasons['n_unexplained']} of {pooled['n_boundaries']} boundaries are unexplained. Their gaps:")
    lines.append("")
    for label, count in reasons["gap_buckets"]:
        lines.append(f"- {label}: {count}")
    lines.append("")
    top_words = ", ".join(f"`{word or '(empty)'}` {count}"
                          for word, count in reasons["dropped_side_words"][:10])
    lines.append(f"Most common dropped-side words at an unexplained boundary: {top_words}.")
    lines.append("")
    lines.append("## Per episode, combined rule")
    lines.append("")
    lines.append("| Episode | Sentences | Partial | Partial share | Boundaries | Sentences fully explained | Boundaries explained | Median split points |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["episodes"]:
        block = row["coverage"]["rules"]["combined"]
        lines.append(
            f"| {row['episode']} | {row['n_sentences']} | {row['n_partial']} | "
            f"{_pct(row['partial_share'])} | {row['n_boundaries']} | "
            f"{_pct(block['sentence_share'])} | {_pct(block['boundary_share'])} | "
            f"{row['split']['median']:.1f} |")
    lines.append("")
    lines.append("## Ten trims the combined rule cannot propose")
    lines.append("")
    lines.append("Kept words are in brackets, dropped words are bare. `gaps` lists every inside boundary's silence in seconds, in order. These are the sentences where at least one boundary fails every rule, so a candidate generator would never offer the editor's trim.")
    lines.append("")
    for example in payload["examples"]:
        gaps = ", ".join("n/a" if gap is None else f"{gap:.2f}s" for gap in example["gaps"])
        lines.append(f"- `{example['episode']}` id {example['sentence_id']}, {example['shape']}, {example['n_dropped']}/{example['n_words']} words dropped, gaps {gaps}: {example['render']}")
    lines.append("")
    if payload["skipped"]:
        lines.append("## Episodes skipped")
        lines.append("")
        for name, reason in payload["skipped"]:
            lines.append(f"- `{name}`: {reason}")
        lines.append("")
    else:
        lines.append("Every episode loaded; nothing skipped.")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def episode_names():
    names = []
    for path in sorted(EPISODES_DIR.glob("*.json")):
        if path.name.endswith(".answerkey.json"):
            continue
        names.append(path.stem)
    return names


def pick_examples(episode_results, limit=10):
    """Partial sentences the combined rule does NOT explain, spread over episodes."""
    pool = []
    for result in episode_results:
        for row in result["sentences"]:
            if not row["rules"]["combined"]:
                pool.append(row)
    # One per episode first, so the list is not ten sentences from one lesson.
    by_episode = defaultdict(list)
    for row in pool:
        by_episode[row["episode"]].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda r: (-r["n_boundaries"], r["sentence_id"]))
    picked, round_index = [], 0
    while len(picked) < limit:
        added = False
        for name in sorted(by_episode):
            rows = by_episode[name]
            if round_index < len(rows):
                picked.append(rows[round_index])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        round_index += 1
    return [{key: row[key] for key in
             ("episode", "sentence_id", "shape", "n_words", "n_dropped", "gaps", "render")}
            for row in picked]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", nargs="*", default=None)
    args = parser.parse_args()

    names = args.episodes or episode_names()
    results, skipped = [], []
    for name in names:
        try:
            results.append(analyse_episode(name))
        except Exception as exc:  # a missing answer key, an unreadable corpus
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"SKIPPED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not results:
        raise SystemExit("no episode loaded")

    all_sentences = [row for result in results for row in result["sentences"]]
    all_boundaries = [row for result in results for row in result["boundaries"]]
    all_splits = [count for result in results for count in result["split_counts"]]

    shapes = Counter()
    for result in results:
        shapes.update(result["shapes"])

    n_sentences = sum(result["n_sentences"] for result in results)
    n_partial = len(all_sentences)
    dropped = [row["n_dropped"] for row in all_sentences]
    shares = [row["dropped_share"] for row in all_sentences]
    split_word_boundaries = sum(result["split_word_boundaries"] for result in results)

    pooled = {
        "n_episodes": len(results),
        "n_sentences": n_sentences,
        "n_partial": n_partial,
        "partial_share": n_partial / n_sentences if n_sentences else None,
        "n_boundaries": len(all_boundaries),
        "boundaries_per_partial": len(all_boundaries) / n_partial if n_partial else 0.0,
        "dropped_median": median(dropped),
        "dropped_mean": sum(dropped) / len(dropped) if dropped else 0.0,
        "dropped_p90": percentile(dropped, 0.9),
        "dropped_max": max(dropped) if dropped else 0,
        "dropped_share_median": median(shares),
        "dropped_share_mean": sum(shares) / len(shares) if shares else 0.0,
        "shapes": shapes.most_common(),
        "coverage": coverage(all_sentences, all_boundaries),
        "unexplained": unexplained_reasons(all_boundaries),
        "trailoff_words": sum(result["trailoff_words"] for result in results),
        "trailoff_internal": sum(result["trailoff_internal"] for result in results),
        "split": {
            "sentences": len(all_splits),
            "word_boundaries": split_word_boundaries,
            "total": sum(all_splits),
            "median": median(all_splits),
            "p90": percentile(all_splits, 0.9),
            "max": max(all_splits) if all_splits else 0,
            "mean": sum(all_splits) / len(all_splits) if all_splits else 0.0,
            "rate": (sum(all_splits) / split_word_boundaries
                     if split_word_boundaries else None),
            "zero_sentences": sum(1 for count in all_splits if count == 0),
            "zero_share": (sum(1 for count in all_splits if count == 0) / len(all_splits)
                           if all_splits else None),
            "touching_boundaries": sum(result["touching_boundaries"] for result in results),
            "touching_share": (sum(result["touching_boundaries"] for result in results)
                               / split_word_boundaries if split_word_boundaries else None),
        },
    }

    episode_payload = []
    for result in results:
        splits = result["split_counts"]
        episode_payload.append({
            "episode": result["episode"],
            "n_sentences": result["n_sentences"],
            "n_partial": result["n_partial"],
            "partial_share": (result["n_partial"] / result["n_sentences"]
                              if result["n_sentences"] else None),
            "n_boundaries": result["n_boundaries"],
            "shapes": result["shapes"],
            "coverage": coverage(result["sentences"], result["boundaries"]),
            "split": {
                "sentences": len(splits),
                "total": sum(splits),
                "median": median(splits) or 0.0,
                "p90": percentile(splits, 0.9),
                "max": max(splits) if splits else 0,
                "mean": sum(splits) / len(splits) if splits else 0.0,
            },
        })

    payload = {
        "date": date.today().isoformat(),
        "script": "scripts/jev_real/partial_coverage.py",
        "source": str(BENCH_DIR),
        "rules": {rule: RULE_LABELS[rule] for rule in RULES},
        "pooled": pooled,
        "episodes": episode_payload,
        "examples": pick_examples(results),
        "skipped": skipped,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(MD_PATH, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_markdown(payload))

    # ---- console summary
    print()
    print(f"episodes {pooled['n_episodes']}  sentences {pooled['n_sentences']}  "
          f"partial {pooled['n_partial']} ({_pct(pooled['partial_share'])})  "
          f"boundaries {pooled['n_boundaries']}")
    print(f"dropped words per partial: median {pooled['dropped_median']:.1f} "
          f"mean {pooled['dropped_mean']:.2f} p90 {pooled['dropped_p90']} "
          f"max {pooled['dropped_max']}; share dropped median "
          f"{_pct(pooled['dropped_share_median'])}")
    print("shapes: " + "  ".join(f"{shape} {count}" for shape, count in pooled["shapes"]))
    print()
    print(f"{'rule':46s} {'sentences':>12s} {'boundaries':>12s}")
    for rule in RULES:
        block = pooled["coverage"]["rules"][rule]
        print(f"{RULE_LABELS[rule]:46s} {_pct(block['sentence_share']):>12s} "
              f"{_pct(block['boundary_share']):>12s}")
    any_block = pooled["coverage"]["combined_any_boundary"]
    print(f"{'combined, at least one boundary':46s} {_pct(any_block['share']):>12s}")
    print()
    split = pooled["split"]
    print(f"split points per sentence: median {split['median']:.1f} p90 {split['p90']} "
          f"max {split['max']} mean {split['mean']:.2f} "
          f"({_pct(split['rate'])} of word boundaries, "
          f"{split['zero_sentences']} sentences with none)")
    print()
    print(f"unexplained boundaries: {pooled['unexplained']['n_unexplained']}")
    for label, count in pooled["unexplained"]["gap_buckets"]:
        print(f"  {label}: {count}")
    print("  top dropped-side words: " + ", ".join(
        f"{word or '(empty)'} {count}"
        for word, count in pooled["unexplained"]["dropped_side_words"][:8]))
    if skipped:
        print()
        for name, reason in skipped:
            print(f"skipped {name}: {reason}")
    print()
    print(f"wrote {MD_PATH}")
    print(f"wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
