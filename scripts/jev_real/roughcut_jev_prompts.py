"""Prompt text for the Jev rough-cut pipeline.

Every instruction, criteria and level string the pipeline sends lives here,
verbatim from ``docs/superpowers/specs/2026-09-20-jev-roughcut-design.md``.
Every version ever sent stays in ``PROMPTS``; ``PROMPT_VERSION`` is the
default the pipeline uses and is recorded on every request row, so a run can
always be traced back to the exact strings it sent. Add a version rather than
editing one in place.

``RULES`` is the EDITING RULES section of
``D:\\solar-sailer\\benchmarks\\roughcut\\prompts\\roughcut_system_partial_v5.md``
with the output-format material dropped (the JSON schema, the category menu and
the ``keep_words`` encoding are the text model's answer format, not editing
rules). It is copied rather than read from D: so a run is reproducible from this
repo alone; ``check_rules_match()`` re-reads the source and reports drift.
"""

import os
import re
from types import SimpleNamespace

PROMPT_VERSION = "v2"

RULES_SOURCE = os.path.join(
    r"D:\solar-sailer\benchmarks\roughcut\prompts", "roughcut_system_partial_v5.md")

RULES = """- Cut narration that is off topic or irrelevant to the lesson.
- Off topic includes the speaker operating the screen or software (opening files, scrolling, fixing a window), talking to the editor or producer about what to insert or show, and any conversation before the lesson starts, even when the lesson is mentioned. These are cut. The speaker's spoken transitions between pieces or students ("Let's move on to Adam") are content and stay.
- If the speaker repeats a line, they're attempting multiple takes. Keep the best take and cut the rest. Default to the LAST take unless it's clearly worse.
- A take is also "repeated" when the speaker abandons an explanation ("I didn't explain that well", "let me say that again") and restarts it. The earlier attempt is the losing take even if it was longer or fluent. Never keep both.
- Keep continuity intact: don't cut a sentence that a later sentence refers back to, and don't create a jump or gap in the train of thought.
- Keep entertaining and humorous sentences, and content showcasing the speaker's personality or unique perspective.
- Cut distracting rambling — unless it's funny or entertaining.
- **You may keep PART of a sentence.** Nothing can be rewritten, reordered or summarized — but within a sentence you keep, you may drop words from the start, from the end, or from the middle, as long as what survives is made of the original words in their original order.
- Assume appropriate visuals will accompany the narration."""


def check_rules_match():
    """``(matches, source_text_or_error)`` against the solar-sailer prompt file.

    Read-only. Used by the pipeline's plan output so a silently edited source
    prompt is visible without making the run depend on drive D:.
    """
    try:
        with open(RULES_SOURCE, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    match = re.search(r"^# EDITING RULES\s*\n(.*?)(?=^# )", text, re.S | re.M)
    if not match:
        return False, "EDITING RULES section not found"
    source = match.group(1).strip()
    return source == RULES, source


# ---------------------------------------------------------------------------
# shared across every version: the retake pass, the head/tail trim questions and
# the trim-pick pass. v2 changes the score question and nothing else.
# ---------------------------------------------------------------------------

TAKE_INSTRUCTIONS = (
    "`groups[{k}].takes` are sentences the retake detector flagged as repeated "
    "attempts at the same line, in the order spoken, with the surrounding sentences "
    "in `groups[{k}].before` and `groups[{k}].after`. Which take does the video "
    "editor keep in the final cut? Prefer the take that is complete, fluent and best "
    "worded. When two takes are equally good, choose the later one."
)

REAL_INSTRUCTIONS = (
    "The takes in `groups[{k}].takes` are repeated attempts at the same line, where "
    "the speaker said it again to get it right, rather than different sentences that "
    "happen to share words, a deliberate repetition for emphasis, or a callback to an "
    "earlier point."
)

FIRST_INSTRUCTIONS = (
    "Assume the editor keeps sentence `targets[{k}]`. Editors often trim the start of "
    "a kept sentence: a wind-up like 'So, um, yeah, okay', a stranded false start "
    "before the real sentence begins, or a first attempt that the rest of the sentence "
    "replaces. Which word is the first word the editor keeps? Choose `whole` when "
    "nothing should be trimmed. Never trim only to remove an 'um', 'uh' or another "
    "single filler word inside otherwise good speech: those are handled elsewhere and "
    "stay. What survives must read as a complete, grammatical line."
)

LAST_INSTRUCTIONS = (
    "Assume the editor keeps sentence `targets[{k}]`. Editors often trim the end of a "
    "kept sentence: a fade-out like 'you know, kind of, right?', a trail-off, or a "
    "dangling 'and' or 'so' the next sentence does not need. Which word is the last "
    "word the editor keeps? Choose `whole` when nothing should be trimmed. Never trim "
    "only to remove an 'um', 'uh' or another single filler word inside otherwise good "
    "speech: those are handled elsewhere and stay. What survives must read as a "
    "complete, grammatical line."
)

WHOLE_FIRST_DESCRIPTION = "nothing is trimmed from the start"
WHOLE_LAST_DESCRIPTION = "nothing is trimmed from the end"

PICK_INSTRUCTIONS = (
    "`items[{k}].versions` are candidate ways to keep sentence `items[{k}].sentence`, "
    "each made of the sentence's own words in order. `whole` keeps every word; `cut` "
    "removes the sentence. Given the surrounding sentences, which version does the "
    "video editor keep? Prefer `whole` unless a version removes something that clearly "
    "does not belong: a wind-up, a fade-out, or an abandoned first attempt. Do not "
    "prefer a version because it drops an 'um' or a single filler word. Choose `cut` "
    "only when nothing in the sentence is worth keeping."
)

WHOLE_VERSION_DESCRIPTION = "keep the whole sentence"
CUT_VERSION_DESCRIPTION = "cut the sentence entirely"

_SHARED = {
    "RULES": RULES,
    "TAKE_INSTRUCTIONS": TAKE_INSTRUCTIONS,
    "REAL_INSTRUCTIONS": REAL_INSTRUCTIONS,
    "FIRST_INSTRUCTIONS": FIRST_INSTRUCTIONS,
    "LAST_INSTRUCTIONS": LAST_INSTRUCTIONS,
    "WHOLE_FIRST_DESCRIPTION": WHOLE_FIRST_DESCRIPTION,
    "WHOLE_LAST_DESCRIPTION": WHOLE_LAST_DESCRIPTION,
    "PICK_INSTRUCTIONS": PICK_INSTRUCTIONS,
    "WHOLE_VERSION_DESCRIPTION": WHOLE_VERSION_DESCRIPTION,
    "CUT_VERSION_DESCRIPTION": CUT_VERSION_DESCRIPTION,
}

# ---------------------------------------------------------------------------
# step 2: the score question, one bundle per version
# ---------------------------------------------------------------------------

V1_SCORE_LEVELS = [
    "A false start the speaker abandons, the losing take of a line said again, dead "
    "air, or the speaker operating the screen or software or talking to the producer "
    "about what to show. Examples: 'So the, uh..', 'Let me just scroll down here.', "
    "'Can you put that on screen?'",
    "Filler with no lesson content: throat-clearing, 'okay so', 'um yeah', or a "
    "sentence that only announces what is about to be said. Example: 'Okay, um, so "
    "yeah, let's see.'",
    "Rambling or a tangent that is not funny: the point was already made, or it "
    "wanders away from the lesson. Cut unless the next sentence depends on it.",
    "Ordinary connective teaching talk: fine, keeps the flow, nothing memorable. "
    "Example: 'So that's the first thing to look at.'",
    "Clear teaching content or a genuine personality moment: it explains a point, "
    "gives a reason, or is funny. The edit is weaker without it.",
    "An essential teaching point or a great moment: the core idea of the lesson, the "
    "punchline of a joke, or the key correction on a student's work.",
]

V1_SCORE_INSTRUCTIONS = (
    "How much does sentence `targets[{k}]` earn its place in the final edit of this "
    "art lesson, given the whole transcript in `transcript` and the editing rules in "
    "`rules`? Later sentences change what earlier ones are worth: a line said again "
    "means the earlier attempt loses, and a sentence a later one refers back to must "
    "stay."
)

# v2 goes after the two miss patterns that cost v1 the most in
# ``docs/jev-real/roughcut-jev-misses.md``: a spoken sentence the transcript
# split across rows, read as an abandoned false start, and a scripted set-up
# line read as filler. The split-sentence half of the fix needs the
# ``spoken_sentence`` and ``piece`` fields the pipeline adds to a target.
V2_SCORE_LEVELS = [
    "A false start the speaker abandons and then restarts, the losing take of a line "
    "said again, dead air, the speaker operating the screen or software ('let me open "
    "that file', 'let me scroll down'), or talk to the crew that is not part of the "
    "lesson ('is this recording?', 'can you put that on screen?'). Example of a false "
    "start: 'So the, uh..' followed by 'So the shoulder connects here.'",
    "Pure filler with nothing of the lesson in it: throat-clearing, 'okay so', 'um "
    "yeah', 'let's see', 'all right'. Only for a row that carries no content at all, "
    "never for a short line that sets up the next one.",
    "Rambling that repeats a point already made, a tangent away from the lesson that "
    "is not funny, or a closing pep talk or recap that adds nothing new. Cut unless "
    "the next sentence depends on it.",
    "Ordinary teaching talk that keeps the flow, even when plain, long or loosely "
    "worded: a transition between topics or between students' pieces ('Let's go to "
    "Dermot', 'Anthony Hernandez.'), a scripted set-up line the next line answers "
    "('So, how do you learn the rules?', 'When do you use a sharp edge?'), a list read "
    "one item per row ('Sharp, firm, soft, and lost.'), or an instruction to the "
    "student. Not memorable, but the lesson reads wrong without it.",
    "Clear teaching content or a genuine personality moment: it explains a point, "
    "gives a reason, corrects a student's work, or is funny ('But wait, there's "
    "more!'). The edit is weaker without it.",
    "An essential teaching point or a great moment: the core idea of the lesson, the "
    "punchline of a joke, or the key correction on a student's work.",
]

V2_SCORE_INSTRUCTIONS = (
    "How much does sentence `targets[{k}]` earn its place in the final edit of this "
    "art lesson, given the whole transcript in `transcript` and the editing rules in "
    "`rules`? Later sentences change what earlier ones are worth: a line said again "
    "means the earlier attempt loses, and a sentence a later one refers back to must "
    "stay. When `targets[{k}].spoken_sentence` is present, the transcript split one "
    "spoken sentence across several rows and this row is one piece of it: judge the "
    "piece by the whole spoken sentence it belongs to, not as a fragment on its own. "
    "A row ending in `..` is where the transcriber marked a trail-off or a split, not "
    "proof of a false start."
)

PROMPTS = {
    "v1": dict(_SHARED, SCORE_LEVELS=V1_SCORE_LEVELS,
               SCORE_INSTRUCTIONS=V1_SCORE_INSTRUCTIONS),
    "v2": dict(_SHARED, SCORE_LEVELS=V2_SCORE_LEVELS,
               SCORE_INSTRUCTIONS=V2_SCORE_INSTRUCTIONS),
}

PROMPT_VERSIONS = list(PROMPTS)

#: Every version must carry every field, so a half-written version fails at
#: import rather than halfway through a paid run.
PROMPT_FIELDS = tuple(sorted(set(_SHARED) | {"SCORE_LEVELS", "SCORE_INSTRUCTIONS"}))

for _name, _bundle in PROMPTS.items():
    _gaps = [_f for _f in PROMPT_FIELDS if _f not in _bundle]
    if _gaps:
        raise AssertionError(f"prompt version {_name} is missing {_gaps}")
if PROMPT_VERSION not in PROMPTS:
    raise AssertionError(f"PROMPT_VERSION {PROMPT_VERSION!r} has no entry in PROMPTS")


def prompts_for(version):
    """The whole prompt set for one version, as an attribute bag.

    Raises on an unknown version: a typo must not quietly fall back to the
    default and mislabel every request row it writes.
    """
    if version not in PROMPTS:
        raise KeyError(f"unknown prompt version {version!r}; have {sorted(PROMPTS)}")
    return SimpleNamespace(version=version, **PROMPTS[version])
