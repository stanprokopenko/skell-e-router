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

PROMPT_VERSION = "v3"

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

# v3 keeps v2's split-sentence work and its level 0 and level 1 rewrites and
# narrows level 3 back towards v1: the "even when plain, long or loosely worded"
# phrase and the instruction clause are gone, the transition, set-up and list
# cases the misses list asked for stay. It also adds ``cut_k``, a second read on
# the same sentence asked as a direct removal question instead of a 0-5 score,
# so the two can be compared and blended offline.
V3_SCORE_LEVELS = list(V2_SCORE_LEVELS)
V3_SCORE_LEVELS[3] = (
    "Ordinary connective teaching talk: fine, keeps the flow, nothing memorable. "
    "This includes a transition between topics or between students' pieces ('Let's "
    "go to Dermot', 'Anthony Hernandez.'), a scripted set-up line the next line "
    "answers ('So, how do you learn the rules?', 'When do you use a sharp edge?'), "
    "and a list read one item per row ('Sharp, firm, soft, and lost.'). Example: "
    "'So that's the first thing to look at.'"
)

V3_SCORE_INSTRUCTIONS = V2_SCORE_INSTRUCTIONS

V3_CUT_INSTRUCTIONS = (
    "The video editor removes sentence `targets[{k}]` from the final cut entirely, "
    "given the whole transcript in `transcript` and the editing rules in `rules`. "
    "Losing takes of a line said again, abandoned false starts, pure filler, "
    "off-topic talk and operating the screen are removed. Teaching content, "
    "transitions between students' pieces, scripted set-up lines the next line "
    "answers, and funny or personal moments stay. When `targets[{k}].spoken_sentence` "
    "is present, judge this row as a piece of that whole spoken sentence."
)

#: ``None`` means the version asks no ``cut_k`` question. Every version carries
#: the field so a version that forgets it fails at import, not mid-run.
PROMPTS = {
    "v1": dict(_SHARED, SCORE_LEVELS=V1_SCORE_LEVELS,
               SCORE_INSTRUCTIONS=V1_SCORE_INSTRUCTIONS,
               CUT_INSTRUCTIONS=None),
    "v2": dict(_SHARED, SCORE_LEVELS=V2_SCORE_LEVELS,
               SCORE_INSTRUCTIONS=V2_SCORE_INSTRUCTIONS,
               CUT_INSTRUCTIONS=None),
    "v3": dict(_SHARED, SCORE_LEVELS=V3_SCORE_LEVELS,
               SCORE_INSTRUCTIONS=V3_SCORE_INSTRUCTIONS,
               CUT_INSTRUCTIONS=V3_CUT_INSTRUCTIONS),
}

PROMPT_VERSIONS = list(PROMPTS)

#: Every version must carry every field, so a half-written version fails at
#: import rather than halfway through a paid run.
PROMPT_FIELDS = tuple(sorted(set(_SHARED) | {"SCORE_LEVELS", "SCORE_INSTRUCTIONS",
                                             "CUT_INSTRUCTIONS"}))

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


# ---------------------------------------------------------------------------
# feature bundles: the prompt-breakup questions of round two
# (``docs/superpowers/specs/2026-09-26-jev-roughcut-round-two-design.md``,
# "Build B"). One yes/no noul per target sentence per question, asked over the
# same ``{rules, transcript, targets}`` state as the v3 sentence pass and sent
# by ``scripts/jev_real/roughcut_jev_features.py``. The questions never see
# each other's answers; a combiner fitted in code does the weighing. Edit the
# list as a new version (``f2``), never in place.
# ---------------------------------------------------------------------------

F1_QUESTIONS = [
    ("false_start",
     "The speaker abandons sentence `targets[{k}]` part-way and restarts the same "
     "thought right after it, so this row is the dropped attempt. The rules say: "
     "\"A take is also 'repeated' when the speaker abandons an explanation ('I "
     "didn't explain that well', 'let me say that again') and restarts it. The "
     "earlier attempt is the losing take even if it was longer or fluent.\" A row "
     "that merely ends in `..` and continues in the next row is not a false start."),
    ("retake_loser",
     "Sentence `targets[{k}]` is one attempt at a line that is said again nearby, "
     "and it is not the best attempt. The rules say: \"If the speaker repeats a "
     "line, they're attempting multiple takes. Keep the best take and cut the rest. "
     "Default to the LAST take unless it's clearly worse.\""),
    ("crew_talk",
     "Sentence `targets[{k}]` is addressed to the editor, producer or crew rather "
     "than to the students, such as 'is this recording?' or 'can you put that on "
     "screen?'. The rules say such talk is off topic: \"talking to the editor or "
     "producer about what to insert or show\" is cut."),
    ("screen_ops",
     "Sentence `targets[{k}]` is about operating the screen, the software or the "
     "recording: opening a file, scrolling, fixing a window, finding a brush. The "
     "rules say: \"Off topic includes the speaker operating the screen or software "
     "(opening files, scrolling, fixing a window)\", and that is cut."),
    ("pre_lesson",
     "Sentence `targets[{k}]` is chatter from before the lesson has started, such "
     "as settling in, checking the setup, or small talk. The rules say: \"any "
     "conversation before the lesson starts, even when the lesson is mentioned\" "
     "is cut."),
    ("off_topic",
     "Sentence `targets[{k}]` is off the topic of this lesson. The rules say: "
     "\"Cut narration that is off topic or irrelevant to the lesson.\" A spoken "
     "transition between students or sections is on topic."),
    ("repeats_point",
     "Sentence `targets[{k}]` repeats a point already made in the last few "
     "sentences of `transcript` and adds nothing new to it. A restatement that adds "
     "a reason, an example or a correction is not a repeat."),
    ("pure_filler",
     "Sentence `targets[{k}]` is filler with no lesson content at all: 'ok', "
     "'alright', 'yeah', 'so', 'let's see', throat-clearing. A short line that sets "
     "up the next one ('So, how do you learn the rules?') is not filler."),
    ("pep_talk",
     "Sentence `targets[{k}]` is encouragement, praise or a wrap-up with nothing "
     "new in it: 'great job', 'keep practising', 'that's it for today'. A "
     "compliment that names what was done well is teaching, not pep talk."),
    ("funny",
     "Sentence `targets[{k}]` is funny, or shows the instructor's personality or "
     "unique perspective. The rules say: \"Keep entertaining and humorous "
     "sentences, and content showcasing the speaker's personality or unique "
     "perspective.\""),
    ("teaching_point",
     "Sentence `targets[{k}]` states a teaching point, gives a reason, or corrects "
     "something in a student's work. It explains rather than only announcing, "
     "narrating or reacting."),
    ("essential",
     "The lesson would lose something if sentence `targets[{k}]` were cut from the "
     "final edit: a core idea, a key correction, the punchline of a joke, or a step "
     "the student needs. Ordinary connective talk that keeps the flow is not "
     "essential."),
    ("referenced_later",
     "A later sentence in `transcript` depends on sentence `targets[{k}]` having "
     "been heard: it refers back to it, answers it, or continues its train of "
     "thought. The rules say: \"Keep continuity intact: don't cut a sentence that "
     "a later sentence refers back to, and don't create a jump or gap in the train "
     "of thought.\""),
    ("transition",
     "Sentence `targets[{k}]` is a spoken transition between students, sections or "
     "steps of the lesson, such as 'Let's move on to Adam' or 'Now the second "
     "layer'. The rules say: \"The speaker's spoken transitions between pieces or "
     "students ('Let's move on to Adam') are content and stay.\""),
    ("describes_screen",
     "Sentence `targets[{k}]` only describes what is visible on screen ('here is a "
     "line', 'this part is darker') rather than explaining why or what to do about "
     "it. The rules say: \"Assume appropriate visuals will accompany the "
     "narration.\" A description that carries a reason or a correction is not only "
     "descriptive."),
    ("rambling",
     "Sentence `targets[{k}]` is rambling or thinking aloud: it wanders, circles "
     "or stalls without landing a point. The rules say: \"Cut distracting rambling "
     "unless it's funny or entertaining.\""),
    ("split_fragment",
     "Sentence `targets[{k}]` is half of one spoken sentence that continues in the "
     "next row or from the previous one, split by the transcriber rather than "
     "abandoned by the speaker. A row ending in `..` followed by a row that starts "
     "lowercase is the usual shape; `targets[{k}].spoken_sentence`, when present, "
     "shows the whole spoken sentence."),
    ("tangent",
     "Sentence `targets[{k}]` is a tangent: it leaves the lesson for an aside, an "
     "anecdote or a side topic and the lesson resumes after it. A funny or "
     "entertaining aside is still a tangent here; whether it stays is judged "
     "elsewhere."),
]

# f2 (round two, second pass, spec "Step 3"): the four f1 questions with no
# signal alone are dropped (``split_fragment`` is covered by the code
# features), the other fourteen are carried over by reference so they cannot
# drift, and eight new questions aim at f1's largest remaining misses:
# teaching lines the editor cut because the drawing already shows it,
# encouragement and wrap-up the editor tightens, and scripted lines Jev reads
# as filler.
F2_DROPPED = ["funny", "referenced_later", "describes_screen", "split_fragment"]

F2_NEW_QUESTIONS = [
    ("play_by_play",
     "Sentence `targets[{k}]` narrates the instructor's own hand action as it "
     "happens ('I'm going to put a line here', 'let me just darken this') without "
     "giving a reason a viewer could not see for themselves. The rules say: \"Assume "
     "appropriate visuals will accompany the narration.\" A line that says why the "
     "mark goes there, or what to avoid, is not play-by-play."),
    ("said_earlier",
     "The same point as sentence `targets[{k}]` was already made earlier in this "
     "episode's `transcript`, anywhere before it and not only in the last few "
     "sentences, and this row adds nothing to it. A callback that adds a reason, an "
     "example or a correction is not a repeat."),
    ("wrap_up",
     "Sentence `targets[{k}]` closes a section or the episode ('so that's the arm', "
     "'alright, moving on', 'that's it for today') and adds nothing new. A closing "
     "line that states the takeaway of the section is content, not a wrap-up."),
    ("praise_only",
     "Sentence `targets[{k}]` praises a student's work ('nice job', 'this is looking "
     "good') with no correction, reason or next step attached. A compliment that "
     "names what was done well and why is a teaching point, not praise only."),
    ("verbal_check",
     "Sentence `targets[{k}]` is a check on the listener ('right?', 'you know?', "
     "'does that make sense?') or a hedge ('I think', 'sort of', 'or whatever') with "
     "no content of its own. A question the next sentence answers is a set-up line, "
     "not a check."),
    ("scripted",
     "Sentence `targets[{k}]` reads like a prepared or scripted lesson line rather "
     "than spontaneous talk: a clean, complete statement of the kind a written lesson "
     "would open with ('So, how do you learn the rules?', 'There are four kinds of "
     "edges.'). Scripted lines are content; the rules' cuts are for off-topic talk, "
     "losing takes, filler and rambling."),
    ("sets_up_next",
     "Sentence `targets[{k}]` is a short line that exists only to set up the "
     "sentence that follows it in `transcript`, such as a question the next line "
     "answers or a lead-in like 'here's the thing', and the next sentence would land "
     "oddly without it. The rules say: \"don't create a jump or gap in the train of "
     "thought.\""),
    ("student_address",
     "Sentence `targets[{k}]` names or addresses a specific student or their drawing "
     "in a critique ('Adam, your torso is too long', 'looking at Maria's piece'). "
     "Naming the student while giving feedback is content; whether the feedback "
     "itself is worth keeping is judged elsewhere."),
]

F2_QUESTIONS = [q for q in F1_QUESTIONS if q[0] not in F2_DROPPED] + F2_NEW_QUESTIONS

#: ``Q_LABEL`` names the question block in the combiner's feature-set names
#: (``q`` for f1, ``q2`` for f2). ``PARENT`` is the bundle a version was
#: edited from, with the keys it dropped and the keys it added, so the
#: combiner can report each change against the parent without guessing.
FEATURE_PROMPTS = {
    "f1": {"QUESTIONS": F1_QUESTIONS, "RULES": RULES, "Q_LABEL": "q",
           "PARENT": None, "DROPPED": [], "NEW": []},
    "f2": {"QUESTIONS": F2_QUESTIONS, "RULES": RULES, "Q_LABEL": "q2",
           "PARENT": "f1", "DROPPED": F2_DROPPED,
           "NEW": [key for key, _text in F2_NEW_QUESTIONS]},
}
FEATURE_PROMPT_VERSION = "f1"
FEATURE_PROMPT_VERSIONS = list(FEATURE_PROMPTS)

for _name, _bundle in FEATURE_PROMPTS.items():
    _keys = [key for key, _text in _bundle["QUESTIONS"]]
    if len(_keys) != len(set(_keys)):
        raise AssertionError(f"feature bundle {_name} repeats a question key")
    for _key, _text in _bundle["QUESTIONS"]:
        if "{k}" not in _text:
            raise AssertionError(f"feature bundle {_name}: {_key} never names targets[k]")
    if _bundle["PARENT"] is not None:
        _parent_keys = [key for key, _t in FEATURE_PROMPTS[_bundle["PARENT"]]["QUESTIONS"]]
        _expected = [k for k in _parent_keys if k not in _bundle["DROPPED"]] + _bundle["NEW"]
        if _keys != _expected:
            raise AssertionError(f"feature bundle {_name} is not its parent minus DROPPED "
                                 f"plus NEW: {_keys} != {_expected}")
        for _key in _bundle["DROPPED"]:
            if _key not in _parent_keys:
                raise AssertionError(f"feature bundle {_name} drops {_key}, which "
                                     f"{_bundle['PARENT']} never asked")


def feature_prompts_for(version):
    """The question list for one feature bundle, as an attribute bag.

    ``questions`` is ``[(key, instructions_template)]`` in the spec's order;
    every template names its sentence as ``targets[{k}]``. ``q_label`` is the
    combiner's name for the question block, ``parent`` the bundle this one was
    edited from (or ``None``), ``dropped`` and ``new`` the keys it removed and
    added against that parent.
    """
    if version not in FEATURE_PROMPTS:
        raise KeyError(f"unknown feature bundle {version!r}; have {sorted(FEATURE_PROMPTS)}")
    bundle = FEATURE_PROMPTS[version]
    return SimpleNamespace(version=version, questions=list(bundle["QUESTIONS"]),
                           rules=bundle["RULES"], q_label=bundle["Q_LABEL"],
                           parent=bundle["PARENT"], dropped=list(bundle["DROPPED"]),
                           new=list(bundle["NEW"]))
