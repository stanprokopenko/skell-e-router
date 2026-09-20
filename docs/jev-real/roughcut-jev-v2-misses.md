# Jev rough cut: where the points go

Generated 2026-09-20T22:16:07+00:00 by `scripts/jev_real/roughcut_jev_misses.py` from the stored decisions and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals; the JSON next to this file keeps the raw values.

The arm under the microscope is jev_a (t_trim 0.3) layered with um removal + delete silence, at its calibrated pooled keep threshold 2.40. That is the column the published ladder compares on, because every published arm gets the same two modules layered on. `jev_b_notrim` (identical scores, no partial trims, threshold 2.40) appears next to it wherever trimming is the question.

Self-check: recounting the full/partial/removed cross-tab from the per-sentence states this file reads reproduces the `pair_counts` block the metric scored, on every episode.

0 sentence(s) never got an answer from the run and are scored at 0.0, so they sit in the arm's removed column by construction.

The decisions file is fingerprinted at the bottom. It is rewritten whenever the pipeline patches answers it missed, and the numbers move when it does, so a table here will not match a run report generated against an older copy.

## Per episode against the ladder

SENTENCE POINTS, WORD SCORE and GRADE for the same six episodes, all three arms with um removal and delete silence layered on. `human kept` is the share of dialogue frames the editor kept, which is how tight the target cut is.

| episode | jev_a SP | jev_a WORD | jev_a GRADE | notrim SP | Luna SP | Luna WORD | Opus SP | Opus WORD | human kept | SP gap vs Luna |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 85.36 | 73.88 | 105.60 | 83.81 | 82.47 | 76.08 | 86.49 | 76.10 | 76.17 | 2.89 |
| hampton-5.4-assignment-demo | 85.63 | 76.15 | 94.94 | 85.10 | 90.57 | 85.60 | 94.60 | 89.47 | 72.88 | -4.93 |
| colman-03.03-muscles-crit | 74.95 | 58.74 | 72.77 | 75.58 | 69.01 | 68.34 | 73.63 | 66.40 | 97.65 | 5.94 |
| edges-7.01-intro | 78.56 | 78.98 | 79.35 | 78.56 | 89.59 | 89.34 | 90.39 | 89.71 | 47.89 | -11.03 |
| hampton-5.2-shape-demo | 91.31 | 83.75 | 95.16 | 91.53 | 90.78 | 81.60 | 95.06 | 87.60 | 68.99 | 0.54 |
| perspective-14e-boxes-critique | 81.40 | 79.60 | 82.74 | 81.43 | 89.01 | 85.03 | 90.54 | 86.36 | 52.35 | -7.61 |
| **pooled** | 82.52 | 76.31 | 86.40 | 82.46 | — | — | — | — | — | — |

The whole gap sits in edges-7.01-intro (-11.03), perspective-14e-boxes-critique (-7.61), hampton-5.4-assignment-demo (-4.93). On the other 3 the arm is level with Luna or ahead, by up to 5.94 on colman-03.03-muscles-crit.

## Sentence-level confusion

Every corpus sentence, at the calibrated threshold, by what the editor did and what the arm did. `kept` on either side means full or partial in that cut; `removed` means gone.

| episode | sentences | both keep | Jev kept, editor removed | Jev removed, editor kept |   of which editor full |   of which editor partial | both removed | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 157 | 11 | 8 | 4 | 4 | 18 | 90.21 |
| hampton-5.4-assignment-demo | 300 | 214 | 30 | 11 | 9 | 2 | 45 | 86.33 |
| colman-03.03-muscles-crit | 303 | 245 | 18 | 30 | 22 | 8 | 10 | 84.16 |
| edges-7.01-intro | 389 | 98 | 33 | 52 | 44 | 8 | 206 | 78.15 |
| hampton-5.2-shape-demo | 411 | 266 | 32 | 17 | 14 | 3 | 96 | 88.08 |
| perspective-14e-boxes-critique | 1146 | 326 | 131 | 81 | 64 | 17 | 608 | 81.50 |
| **pooled** | 2743 | 1306 | 255 | 199 | 157 | 42 | 983 | 83.45 |

Luna chapters on the same six episodes, from the archived `run_ratings` in `2026-09-08-colman0204-partial-agentic-router-v2-xhigh-chapters-r5-gpt-5.6-luna.json` and its siblings, each at its own file's Neutral threshold, with the same um removal and delete silence layers on.

| episode | sentences | both keep | Luna kept, editor removed | Luna removed, editor kept | both removed | agreement |
|---|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 139 | 6 | 26 | 23 | 83.51 |
| hampton-5.4-assignment-demo | 300 | 204 | 5 | 21 | 70 | 91.33 |
| colman-03.03-muscles-crit | 303 | 213 | 9 | 62 | 19 | 76.57 |
| edges-7.01-intro | 389 | 114 | 6 | 36 | 233 | 89.20 |
| hampton-5.2-shape-demo | 411 | 263 | 27 | 20 | 101 | 88.56 |
| perspective-14e-boxes-critique | 1146 | 353 | 69 | 54 | 670 | 89.27 |
| **pooled** | 2743 | 1286 | 122 | 219 | 1116 | 87.57 |

## The misses

199 sentences the editor kept and the arm dropped, against 255 the arm kept and the editor dropped. It errs towards cutting. The lists below are the worst of each: the lowest-scoring drops, and the highest-scoring keeps.

### Jev removed, editor kept: the 40 lowest scores

Grouped by reading them with their neighbours. The grouping is judgement, not code; the counts are over these 40, not over all 199.

| pattern | of 40 |
|---|---:|
| unlabelled | 20 |
| Half a sentence the transcript split | 6 |
| Retake pair, the wrong side cut | 4 |
| Producer talk and student names in a critique | 4 |
| Ordinary connective teaching talk | 3 |
| Thinking aloud, all ums | 2 |
| Scripted lesson line read as filler | 1 |
| A joke the editor kept | 0 |

**unlabelled** (20). in the list but not in the hand grouping; the decision file has moved since it was written

- `hampton-5.4-assignment-demo` #54, score 0.22: its modification early on.
- `colman-02.04-skeleton-demo` #21, score 0.59: Remember as I went over in the demo as well?
- `perspective-14e-boxes-critique` #989, score 0.77: using atmospheric perspective on some of these planes.

**Half a sentence the transcript split** (6). The transcript breaks one spoken sentence into two rows and the second row trails off in `..`. Jev reads the fragment on its own, calls it an abandoned false start, and drops it. The editor kept the whole spoken sentence, so both rows are in the cut.

- `colman-03.03-muscles-crit` #156, score 0.71: You have to kind of..
- `perspective-14e-boxes-critique` #176, score 1.14: If the..
- `edges-7.01-intro` #361, score 0.75: um, see how edges behave differently under various lighting conditions.

**Retake pair, the wrong side cut** (4). Two attempts at one line. Jev found the pair but crowned the other take, so the one the editor actually used got cut as the loser.

- `edges-7.01-intro` #203, score 1.04: "They're all soft edges with a different flavor!" Hmm...
- `edges-7.01-intro` #384, score 0.86: So let's get going.
- `edges-7.01-intro` #210, score 0.71: all shapes have edges.

**Producer talk and student names in a critique** (4). In a critique the teacher reads a name off the screen or says what the producer just put up. The prompt's level 0 names exactly that as a cut, and the editor keeps it, because it is how the critique moves.

- `perspective-14e-boxes-critique` #603, score 0.47: Jayoshi points out the principle behind overdoing it.
- `perspective-14e-boxes-critique` #908, score 1.00: What commitment.
- `perspective-14e-boxes-critique` #941, score 0.96: Anthony Hernandez.

**Ordinary connective teaching talk** (3). Level 3 in the prompt's own words: fine, keeps the flow, nothing memorable. Jev put it at 1 or below. These are the demo lessons, where the editor keeps almost everything.

- `hampton-5.2-shape-demo` #5, score 0.69: And now we'll try that with a little bit more directness in terms of looking at a reference and seeing if we could practice that from that picture.
- `colman-03.03-muscles-crit` #273, score 1.00: As opposed to just kind of drawing it for what it is and getting a little bit more of a kind of a, uh, worried about like the silhouette and almost again, instead of looking at patterns, look at it i…
- `colman-03.03-muscles-crit` #5, score 1.34: Uh, so weight is one thing, and I like seeing that everyone's trying to draw from their imagination, the weight of the pose and putting it together, and also the flow of the muscles together.

**Thinking aloud, all ums** (2). A run of pure filler the editor kept anyway, because the pause is part of the demo's rhythm.

- `hampton-5.2-shape-demo` #104, score 1.22: so, I mean, hmm, mm-hmm.
- `hampton-5.2-shape-demo` #103, score 1.22: Um, uh, like, you know,

**Scripted lesson line read as filler** (1). The written intro asks a question or names a list, one short sentence at a time. Jev scores a short sentence with no content of its own as throat-clearing; in a script it is the lesson.

- `edges-7.01-intro` #82, score 0.38: Sharp, firm, soft, and lost.

**A joke the editor kept** (0). Self-deprecating aside. The prompt asks for it at level 4 and Jev scored it at 1.


The list in full, worst first. `before` and `after` are the neighbouring sentences.

- `hampton-5.4-assignment-demo` #54, score 0.22, level 0 0.93, level 3 0.05, pattern: unlabelled. **its modification early on.** (before: So, here's my head and what we can start to do in some of these too and totally understandable if you didn't do this in… / after: So, maybe, you know, instead of..)
- `edges-7.01-intro` #82, score 0.38, level 0 0.88, level 3 0.04, pattern: Scripted lesson line read as filler. **Sharp, firm, soft, and lost.** (before: To be able to keep this stuff in our head and make decisions, it's a lot easier to think about 5 values and 4 edges. / after: Lost is actually off the spectrum a little.)
- `perspective-14e-boxes-critique` #603, score 0.47, level 0 0.82, level 3 0.10, pattern: Producer talk and student names in a critique. **Jayoshi points out the principle behind overdoing it.** (before: Sure. / after: Command F.)
- `colman-02.04-skeleton-demo` #21, score 0.59, level 0 0.83, level 4 0.08, pattern: unlabelled. **Remember as I went over in the demo as well?** (before: See how loose and almost bendy I'm being with it? / after: Don't get..)
- `hampton-5.2-shape-demo` #5, score 0.69, level 0 0.81, level 4 0.11, pattern: Ordinary connective teaching talk. **And now we'll try that with a little bit more directness in terms of looking at a reference and seeing if we could practice that from that picture.** (before: Okay, so we've been now through our, our alternative or secondary way of thinking about gesture, putting our emphasis m… / after: So, I used, uh, these to make sure that what we have..)
- `colman-03.03-muscles-crit` #156, score 0.71, level 0 0.78, level 3 0.12, pattern: Half a sentence the transcript split. **You have to kind of..** (before: you draw from your head, um, remember to continue and find a balance, find weight as well and make the adjustments, you… / after: you'd have to..)
- `edges-7.01-intro` #210, score 0.71, level 0 0.78, level 3 0.11, cut as a retake loser, pattern: Retake pair, the wrong side cut. **all shapes have edges.** (before: All shapes.. / after: And this doesn't..)
- `edges-7.01-intro` #361, score 0.75, level 0 0.77, level 3 0.15, pattern: Half a sentence the transcript split. **um, see how edges behave differently under various lighting conditions.** (before: See how edges behave differently.. / after: Start by learning..)
- `perspective-14e-boxes-critique` #989, score 0.77, level 0 0.75, level 3 0.15, pattern: unlabelled. **using atmospheric perspective on some of these planes.** (before: There's Sandra Susser / after: Let me see which other ones I wanna click on.)
- `perspective-14e-boxes-critique` #988, score 0.79, level 0 0.74, level 3 0.18, pattern: Producer talk and student names in a critique. **There's Sandra Susser** (before: Okay, then I'm gonna just show a few others and mention their name. / after: using atmospheric perspective on some of these planes.)
- `edges-7.01-intro` #384, score 0.86, level 0 0.66, level 3 0.16, cut as a retake loser, pattern: Retake pair, the wrong side cut. **So let's get going.** (before: Up next, your first edge assignment. / after: Up next, your first edge assignment.)
- `edges-7.01-intro` #253, score 0.96, level 0 0.74, level 4 0.14, pattern: unlabelled. **edges are influenced by things like the light source, the composition, our preferences and style, but understanding their connection to form, that's the foundation.** (before: edges are influenced by things.. / after: They indicate how round..)
- `perspective-14e-boxes-critique` #941, score 0.96, level 0 0.68, level 3 0.21, pattern: Producer talk and student names in a critique. **Anthony Hernandez.** (before: Yeah. / after: It is kind of complicated, isn't it?)
- `colman-03.03-muscles-crit` #157, score 0.97, level 0 0.70, level 3 0.16, pattern: unlabelled. **you'd have to..** (before: You have to kind of.. / after: a lot more balance.)
- `colman-03.03-muscles-crit` #273, score 1.00, level 0 0.66, level 3 0.19, pattern: Ordinary connective teaching talk. **As opposed to just kind of drawing it for what it is and getting a little bit more of a kind of a, uh, worried about like the silhouette and almost again, instead of looking at patterns, look at it in terms of forms, right?** (before: So, think about things like that. / after: Look at it in terms of volumes and building blocks and almost like you're carving it out of wood.)
- `perspective-14e-boxes-critique` #908, score 1.00, level 0 0.68, level 3 0.19, pattern: Producer talk and student names in a critique. **What commitment.** (before: "Rupert, 'Dddd.' Unbelievable. / after: 'SB.' And I can't make it bigger in here, right?)
- `perspective-14e-boxes-critique` #11, score 1.01, level 0 0.65, level 3 0.17, pattern: unlabelled. **Hi everybody.** (before: Nope. / after: We are in part 2 of this course.)
- `colman-03.03-muscles-crit` #274, score 1.02, level 0 0.68, level 3 0.17, pattern: unlabelled. **Look at it in terms of volumes and building blocks and almost like you're carving it out of wood.** (before: As opposed to just kind of drawing it for what it is and getting a little bit more of a kind of a, uh, worried about li… / after: That's one thing I want you to kind of think about is you're like almost as if you're carving this out of wood and, you…)
- `perspective-14e-boxes-critique` #181, score 1.03, level 0 0.66, level 3 0.19, pattern: unlabelled. **And I do want to say something about using the Zolli app, which is great, but there is something about extremizing this.** (before: Hard at first, easy soon. / after: One of you..)
- `edges-7.01-intro` #203, score 1.04, level 0 0.71, level 4 0.17, cut as a retake loser, pattern: Retake pair, the wrong side cut. **"They're all soft edges with a different flavor!" Hmm...** (before: So let me try that again. / after: they're all soft edges with a different flavor!)
- `edges-7.01-intro` #255, score 1.07, level 0 0.70, level 4 0.15, pattern: unlabelled. **they indicate how rounded or sudden a transition is between two planes.** (before: They indicate how round.. / after: This value change..)
- `hampton-5.2-shape-demo` #205, score 1.08, level 0 0.71, level 4 0.21, pattern: unlabelled. **And then foot.** (before: And then we have our ground. / after: A fun homework assignment could even be that you'd..)
- `edges-7.01-intro` #29, score 1.11, level 0 0.68, level 4 0.20, pattern: unlabelled. **oh, like, you know, or overly smudged tones that hide uncertainty and mistakes instead of communicating form.** (before: You'll find random sharp edges everywhere or overly smudged tones that hide uncertainty and mistakes instead.. / after: God, this thing's squeaky again.)
- `hampton-5.4-assignment-demo` #77, score 1.12, level 0 0.67, level 3 0.16, pattern: Half a sentence the transcript split. **Maybe we could even..** (before: And then if I'm good with that, then I'm just going to start to put some of these more perspective-driven lines across. / after: yeah, I don't think on her specifically, but you could broaden the hip or, or change the shape in any way that you see …)
- `edges-7.01-intro` #359, score 1.14, level 0 0.67, level 3 0.16, pattern: unlabelled. **Pay attention to edges, try to replicate them as close as possible to reality.** (before: Pay attention to edges, try to replicate them as close as possible to reality. / after: See how edges behave differently..)
- `perspective-14e-boxes-critique` #176, score 1.14, level 0 0.69, level 4 0.14, pattern: Half a sentence the transcript split. **If the..** (before: And if it's above us, the opposite is the case. / after: if we can see the bottom plane, that means that bottom corner is closer to us than the top corner.)
- `colman-03.03-muscles-crit` #78, score 1.18, level 0 0.63, level 3 0.24, pattern: Half a sentence the transcript split. **It doesn't..** (before: All just kind of flows together. / after: not really worried once again about the, the names so much.)
- `colman-02.04-skeleton-demo` #4, score 1.19, level 0 0.63, level 3 0.25, pattern: unlabelled. **Let's talk about your assignment.** (before: 3, 2, 1. / after: Uh, I want you to..)
- `hampton-5.2-shape-demo` #103, score 1.22, level 1 0.78, level 0 0.08, pattern: Thinking aloud, all ums. **Um, uh, like, you know,** (before: So, it comes this way, back here. / after: so, I mean, hmm, mm-hmm.)
- `hampton-5.2-shape-demo` #104, score 1.22, level 1 0.79, level 0 0.08, pattern: Thinking aloud, all ums. **so, I mean, hmm, mm-hmm.** (before: Um, uh, like, you know, / after: You probably need to bring this foot down or this foot up.)
- `perspective-14e-boxes-critique` #623, score 1.29, level 1 0.61, level 0 0.17, pattern: unlabelled. **Hmm?** (before: If the vanishing point is right here and this line converges—like so—and like so—it can tell us that this plane is actu… / after: You know, so, I mean, hmm, um, uh, like, you know, so, I mean, hmm, mm-hmm.)
- `edges-7.01-intro` #28, score 1.30, level 0 0.67, level 4 0.18, pattern: Half a sentence the transcript split. **You'll find random sharp edges everywhere or overly smudged tones that hide uncertainty and mistakes instead..** (before: You can usually spot an amateur by looking at their edges. / after: oh, like, you know, or overly smudged tones that hide uncertainty and mistakes instead of communicating form.)
- `perspective-14e-boxes-critique` #922, score 1.32, level 0 0.58, level 3 0.23, pattern: unlabelled. **Very clean and professional looking and it looks like you more than paid your dues.** (before: Um, Espe. / after: How do I get out of here?)
- `edges-7.01-intro` #102, score 1.33, level 0 0.65, level 4 0.20, pattern: Retake pair, the wrong side cut. **But lost edges don't always mean that it's super blurry between the two shapes.** (before: But lost edges don't always mean that it's super blurry between two shapes. / after: Hmm, sometimes it just means you've made the values of neighboring shapes identical.)
- `colman-03.03-muscles-crit` #5, score 1.34, level 0 0.62, level 4 0.17, pattern: Ordinary connective teaching talk. **Uh, so weight is one thing, and I like seeing that everyone's trying to draw from their imagination, the weight of the pose and putting it together, and also the flow of the muscles together.** (before: Mistakes carry such a, such a heavy stigma. / after: We talked about flow..)
- `perspective-14e-boxes-critique` #363, score 1.34, level 0 0.64, level 4 0.12, pattern: unlabelled. **If they do, you practice it the other way.** (before: You check to see if lines diverge as they recede. / after: I could be a taskmaster and tell you that for every receding divergence error you made..)
- `perspective-14e-boxes-critique` #844, score 1.34, level 0 0.54, level 3 0.30, pattern: unlabelled. **Spirit on.** (before: Let's look at a related comment. / after: Am I not in..)
- `perspective-14e-boxes-critique` #624, score 1.37, level 1 0.52, level 0 0.22, pattern: unlabelled. **You know, so, I mean, hmm, um, uh, like, you know, so, I mean, hmm, mm-hmm.** (before: Hmm? / after: And it looks more like a cube if)
- `perspective-14e-boxes-critique` #839, score 1.37, level 0 0.55, level 3 0.27, pattern: unlabelled. **It's knowing why you're doing it.** (before: Edward de Bono's Blue Hat. / after: It can help you know when to grind down and keep working at this.)
- `perspective-14e-boxes-critique` #372, score 1.38, level 0 0.62, level 3 0.13, pattern: unlabelled. **I suggest you slow down and pay attention to when these things happen.** (before: You are not overthinking. / after: —so that you don't have to do 1,000 more like you apparently did before we began.)

### Jev kept, editor removed: the 40 highest scores

Same treatment from the other side, over these 40 of 255.

| pattern | of 40 |
|---|---:|
| unlabelled | 18 |
| Teaching line the picture already makes | 10 |
| Producer talk and studio logistics | 3 |
| Half a sentence the transcript split | 3 |
| Encouragement and wrap-up the editor tightened | 2 |
| Short connective dropped for pacing | 2 |
| Near-duplicate line, the loser kept | 1 |
| Operating the drawing or the screen | 1 |
| Tangent the editor cut short | 0 |

**unlabelled** (18). in the list but not in the hand grouping; the decision file has moved since it was written

- `perspective-14e-boxes-critique` #140, score 4.04: Alright.
- `edges-7.01-intro` #252, score 3.77: edges are influenced by things..
- `perspective-14e-boxes-critique` #1017, score 3.65: The problem with this, as some of you I think have mentioned, is Charlie has mentioned it, uh, in editing this course is that you start to go to sleep at night and see these things.

**Teaching line the picture already makes** (10). Real content, said clearly, and the editor still cut it: the drawing on screen says the same thing, or the point was already made a sentence earlier. Jev has no way to see the picture and grades the words.

- `edges-7.01-intro` #228, score 3.88: This highlight is sharp here, firm here, and soft here as it slowly fades away.
- `colman-02.04-skeleton-demo` #178, score 3.65: Um, once again, making sure there's room for all the teeth even though it's completely grid-based.
- `colman-03.03-muscles-crit` #287, score 3.41: If you're starting to think about kind of what's going on on the other side, even though I don't see it..

**Producer talk and studio logistics** (3). Two people arranging what happens next, or spelling a name out loud. Jev graded it 3 or above because it is fluent and on topic.

- `hampton-5.2-shape-demo` #297, score 3.26: That was great.
- `perspective-14e-boxes-critique` #957, score 3.37: F-M-I-N-K-E-E.
- `perspective-14e-boxes-critique` #1112, score 3.27: I noticed that with Yves, uh, who did a piano, that some of you did pianos too.

**Half a sentence the transcript split** (3). The mirror of the same problem: the editor cut the whole spoken sentence, so the trailing fragment goes too, and Jev kept the fragment because the words it can see read as setup.

- `colman-02.04-skeleton-demo` #10, score 3.37: Skeleton that we provided is..
- `perspective-14e-boxes-critique` #590, score 3.32: that looks right enough.
- `hampton-5.2-shape-demo` #46, score 3.31: So, you're..

**Encouragement and wrap-up the editor tightened** (2). The pep talk at the end of a lesson. Warm, well said, and the editor keeps one line of it and drops the rest.

- `perspective-14e-boxes-critique` #1113, score 3.54: Piano is a great challenge.
- `hampton-5.2-shape-demo` #337, score 3.27: it's going to be uncomfortable at first, especially if you've been doing the other approach to kind of reorganize your thoughts and then, you know, redevelop a new way of representing and organizing …

**Short connective dropped for pacing** (2). `Right?`, `All right?`, `Okay.` Jev scores the tag high because it reads as ordinary teaching talk; the editor cuts it to keep the cut moving.

- `colman-03.03-muscles-crit` #222, score 3.67: Right?
- `perspective-14e-boxes-critique` #139, score 3.96: Okay.

**Near-duplicate line, the loser kept** (1). The next sentence says the same thing better and the retake detector never flagged the pair, so nothing cut it.

- `edges-7.01-intro` #261, score 3.63: A firm edge.

**Operating the drawing or the screen** (1). Level 0 in the prompt. Jev gave it 3.15.

- `hampton-5.2-shape-demo` #107, score 3.30: So, I could just drop that for now.

**Tangent the editor cut short** (0). The teacher loses the thread and the editor takes the detour out.


The list in full, worst first. `before` and `after` are the neighbouring sentences.

- `perspective-14e-boxes-critique` #140, score 4.04, level 5 0.46, level 4 0.37, pattern: unlabelled. **Alright.** (before: Okay. / after: Doobie, this rapid-fire page where you labeled it as such, uh, were some of your best.)
- `perspective-14e-boxes-critique` #139, score 3.96, level 5 0.47, level 4 0.33, pattern: Short connective dropped for pacing. **Okay.** (before: So, at the very top will be the oldest. / after: Alright.)
- `edges-7.01-intro` #228, score 3.88, level 4 0.72, level 5 0.15, pattern: Teaching line the picture already makes. **This highlight is sharp here, firm here, and soft here as it slowly fades away.** (before: Even a single small shape can have a changing edge type around it. / after: How do you know when to use each type of edge?)
- `edges-7.01-intro` #252, score 3.77, level 4 0.67, level 5 0.18, pattern: unlabelled. **edges are influenced by things..** (before: So, edges are.. / after: edges are influenced by things like the light source, the composition, our preferences and style, but understanding the…)
- `colman-03.03-muscles-crit` #222, score 3.67, level 4 0.52, level 5 0.22, pattern: Short connective dropped for pacing. **Right?** (before: now go to the zoo and draw gorillas, you know, after you're going through this course or any ape, you're gonna have a r… / after: You can see how the flow of the muscles are mimicking and almost, um, being directed by what the skeletal structure is …)
- `colman-02.04-skeleton-demo` #178, score 3.65, level 4 0.70, level 3 0.22, pattern: Teaching line the picture already makes. **Um, once again, making sure there's room for all the teeth even though it's completely grid-based.** (before: I'm just kind of carving it back into it to create more, um, clearer shapes. / after: And can I walk away from here gaining what I needed to gain, which is the understanding of all the different intricacie…)
- `perspective-14e-boxes-critique` #1017, score 3.65, level 4 0.59, level 3 0.16, pattern: unlabelled. **The problem with this, as some of you I think have mentioned, is Charlie has mentioned it, uh, in editing this course is that you start to go to sleep at night and see these things.** (before: Max Long. / after: But if you're going to sleep at night and seeing these things and you can make the jump from seeing these things and wh…)
- `perspective-14e-boxes-critique` #1018, score 3.65, level 4 0.63, level 3 0.14, pattern: unlabelled. **But if you're going to sleep at night and seeing these things and you can make the jump from seeing these things and why you're doing it, it may be a good thing that they're..** (before: The problem with this, as some of you I think have mentioned, is Charlie has mentioned it, uh, in editing this course i… / after: they're going into your brain.)
- `colman-02.04-skeleton-demo` #66, score 3.64, level 4 0.62, level 5 0.15, pattern: unlabelled. **Um, going back and adjusting this angle here.** (before: um, but it's not so busy. / after: So I really wanna see how you analyze the skeleton from the angle that you choose and how you approach the simplificati…)
- `edges-7.01-intro` #261, score 3.63, level 4 0.68, level 3 0.15, pattern: Near-duplicate line, the loser kept. **A firm edge.** (before: A sharp corner. / after: A firm edge would suggest a kind of rounded corner.)
- `perspective-14e-boxes-critique` #1019, score 3.58, level 4 0.61, level 3 0.18, pattern: unlabelled. **they're going into your brain.** (before: But if you're going to sleep at night and seeing these things and you can make the jump from seeing these things and wh… / after: Oh, this was very impressive.)
- `perspective-14e-boxes-critique` #1016, score 3.55, level 4 0.54, level 3 0.21, pattern: unlabelled. **Max Long.** (before: Extremely close up in three-point perspective, crazy things happen. / after: The problem with this, as some of you I think have mentioned, is Charlie has mentioned it, uh, in editing this course i…)
- `perspective-14e-boxes-critique` #1113, score 3.54, level 4 0.61, level 3 0.20, pattern: Encouragement and wrap-up the editor tightened. **Piano is a great challenge.** (before: I noticed that with Yves, uh, who did a piano, that some of you did pianos too. / after: Of course, if it's something that you wanted to do anyway because in the first part of this course, you collected stuff…)
- `colman-02.04-skeleton-demo` #184, score 3.51, level 4 0.65, level 3 0.21, pattern: unlabelled. **—just** (before: it's good to use, you know, different line weights, um, to show value, uh, mm-hmm, as well as weight, the actual weight… / after: kind of solidifying and designing, uh, not just drawing the skull itself, right?)
- `edges-7.01-intro` #254, score 3.50, level 4 0.52, level 5 0.26, pattern: unlabelled. **They indicate how round..** (before: edges are influenced by things like the light source, the composition, our preferences and style, but understanding the… / after: they indicate how rounded or sudden a transition is between two planes.)
- `hampton-5.4-assignment-demo` #150, score 3.49, level 4 0.56, level 3 0.37, pattern: Teaching line the picture already makes. **So maybe I'll just extend that a little bit lower.** (before: And on this one, we have some of that top view on the foot. / after: This —So it looks like it's going away, but I would almost always do this unless I'm really working with that top-down …)
- `colman-03.03-muscles-crit` #29, score 3.45, level 4 0.53, level 3 0.33, pattern: unlabelled. **So, you can see the difference there.** (before: hooking everything together, right? / after: So, I'm just trying to connect all the parts, right?)
- `colman-03.03-muscles-crit` #287, score 3.41, level 4 0.43, level 3 0.40, pattern: Teaching line the picture already makes. **If you're starting to think about kind of what's going on on the other side, even though I don't see it..** (before: and when I talk about dynamic proportions, but they'll be so great, there'll be a great sense of of a dynamic energy in… / after: what's the weight like?)
- `colman-03.03-muscles-crit` #54, score 3.40, level 4 0.67, level 2 0.18, pattern: unlabelled. **and not putting myself through college but working at my dad's law firm during the day and I would, um, you know, draw and, uh, study anatomy in 9 different classes.** (before: when I was going to my dad's law firm and putting my.. / after: I'd have my anatomy book open when I was driving in traffic, not safe, but when I was stuck in traffic, I'd flip throug…)
- `colman-03.03-muscles-crit` #30, score 3.39, level 4 0.51, level 3 0.32, pattern: Teaching line the picture already makes. **So, I'm just trying to connect all the parts, right?** (before: So, you can see the difference there. / after: And not getting too caught up in the details right now, I just wanna show you how we can really connect the parts.)
- `edges-7.01-intro` #27, score 3.38, level 4 0.51, level 3 0.17, pattern: Teaching line the picture already makes. **You can usually spot an amateur by looking at their edges.** (before: Um, uh, like, you know, so, I mean, hmm, mm-hmm... / after: You'll find random sharp edges everywhere or overly smudged tones that hide uncertainty and mistakes instead..)
- `perspective-14e-boxes-critique` #350, score 3.38, level 4 0.36, level 5 0.31, pattern: Teaching line the picture already makes. **If they do, you practice it the other way.** (before: You see if lines diverge as they recede. / after: I think this might be one to show while I'm saying that.)
- `colman-02.04-skeleton-demo` #10, score 3.37, level 4 0.50, level 3 0.41, pattern: Half a sentence the transcript split. **Skeleton that we provided is..** (before: So one skeleton, one skull. / after: we only have one, which is chimpanzee.)
- `colman-02.04-skeleton-demo` #11, score 3.37, level 4 0.54, level 3 0.33, pattern: Teaching line the picture already makes. **we only have one, which is chimpanzee.** (before: Skeleton that we provided is.. / after: But try a different angle, not an angle that I, I demoed, uh, during the episode.)
- `perspective-14e-boxes-critique` #271, score 3.37, level 5 0.47, level 0 0.24, pattern: unlabelled. **Dermot!** (before: Somewhere out there. / after: Look at this!)
- `perspective-14e-boxes-critique` #957, score 3.37, level 3 0.48, level 4 0.40, pattern: Producer talk and studio logistics. **F-M-I-N-K-E-E.** (before: At least Minkee. / after: Laid out so clearly and with such economy of line.)
- `perspective-14e-boxes-critique` #562, score 3.33, level 4 0.51, level 3 0.30, pattern: unlabelled. **and, you know." In other words, this could be the top plane.** (before: "This could be the top plane and the.. / after: "Yeah, because these look like they're coming together." But it wouldn't be dotted lines that way.)
- `perspective-14e-boxes-critique` #744, score 3.33, level 4 0.47, level 3 0.37, pattern: unlabelled. **That would be..** (before: We'll do one more. / after: Sita.)
- `edges-7.01-intro` #276, score 3.32, level 4 0.52, level 3 0.23, pattern: unlabelled. **and the object receiving the cast shadow.** (before: The edge can also describe the distance between the object casting / after: But wait, there's more!)
- `perspective-14e-boxes-critique` #590, score 3.32, level 4 0.39, level 3 0.32, pattern: Half a sentence the transcript split. **that looks right enough.** (before: Away, where they converge, get the other set or the other ones in that line set to converge at the same point and then,… / after: When we get that together, then if we want a ramp on there, that ramp will find its hidden VP up here and will be able …)
- `hampton-5.2-shape-demo` #46, score 3.31, level 4 0.44, level 3 0.34, pattern: Half a sentence the transcript split. **So, you're..** (before: sometimes I like to include these 3 points, sometimes helps me see the early on even that quality of rotation, right? / after: if you're always thinking about that center and the distance to one side and the distance to the other, it does solve t…)
- `hampton-5.2-shape-demo` #107, score 3.30, level 4 0.47, level 3 0.44, pattern: Operating the drawing or the screen. **So, I could just drop that for now.** (before: So, that's the only thing I'm noticing here is my, my leg got a little bit long on one side. / after: Here would be one of the, though, with the fallbacks or the kind of the negative aspects of this approach that I'm so u…)
- `perspective-14e-boxes-critique` #565, score 3.30, level 4 0.48, level 3 0.34, pattern: unlabelled. **Is that what that dotted line's doing?** (before: "Yeah, it doesn't look like that." Oh, yeah, yeah, yeah. / after: I'm going to go into that mode.)
- `perspective-14e-boxes-critique` #400, score 3.29, level 4 0.40, level 3 0.30, pattern: Teaching line the picture already makes. **You've got your absolute reference.** (before: And watch me dare on this next one." Again, you hardly need a teacher for that. / after: You do that 100 times and you will have what you need for seeing what up to 300, even up to 600 squares in perspective …)
- `colman-03.03-muscles-crit` #286, score 3.28, level 4 0.39, level 3 0.33, pattern: Teaching line the picture already makes. **and when I talk about dynamic proportions, but they'll be so great, there'll be a great sense of of a dynamic energy in your work.** (before: This is the way of conceptual thinking that you need that will really help you communicate volume and structure, weight… / after: If you're starting to think about kind of what's going on on the other side, even though I don't see it..)
- `hampton-5.2-shape-demo` #337, score 3.27, level 4 0.56, level 3 0.21, pattern: Encouragement and wrap-up the editor tightened. **it's going to be uncomfortable at first, especially if you've been doing the other approach to kind of reorganize your thoughts and then, you know, redevelop a new way of representing and organizing the same information you've been looking at.** (before: You're.. / after: So, try to treat it with some levity, have fun.)
- `hampton-5.4-assignment-demo` #94, score 3.27, level 4 0.44, level 3 0.37, pattern: unlabelled. **Again, here's the..** (before: So, my shape is going to get overlapped, come forward. / after: so, thinking of all those things from the last approach, weight, you know, what parts are holding the weight, but I'm d…)
- `perspective-14e-boxes-critique` #1112, score 3.27, level 4 0.50, level 3 0.35, pattern: Producer talk and studio logistics. **I noticed that with Yves, uh, who did a piano, that some of you did pianos too.** (before: Yeah, there, there's something. / after: Piano is a great challenge.)
- `hampton-5.2-shape-demo` #297, score 3.26, level 4 0.46, level 3 0.34, pattern: Producer talk and studio logistics. **That was great.** (before: Okay. / after: So, then our next thing will just be giving people their assignment.)
- `perspective-14e-boxes-critique` #138, score 3.25, level 4 0.27, level 5 0.25, pattern: unlabelled. **So, at the very top will be the oldest.** (before: See, it's arranged oldest to newest. / after: Okay.)

## Trim quality

Jev emitted 77 trims across the six episodes and 27 of them survive the keep threshold to reach the metric as a partial sentence. They are not the only partials: the um removal and delete silence layers cut inside another 288 sentences Jev had asked to keep whole. Both columns are below, because only the first is Jev's doing.

| trim outcome | Jev's own trims | layers only |
|---|---:|---:|
| exact, run for run (pays 2.0) | 4 | 105 |
| subset of the editor's runs (1.2) | 0 | 0 |
| overlaps, not contained (1.0) | 0 | 29 |
| disjoint from the editor's runs (0.6) | 3 | 35 |
| editor kept the sentence whole (0.7) | 14 | 100 |
| editor removed the sentence (0.0) | 6 | 19 |
| **total** | 27 | 288 |

For scale, the editor trimmed inside 388 sentences; the arm came back partial on 176 of them and kept or dropped the rest whole. Dropping Jev's trims entirely (`jev_b_notrim`) scores 82.46 pooled SENTENCE POINTS against 82.52 with them.

Fifteen trims, the arm's kept span in square brackets and the editor's in braces. A sentence the editor kept whole shows braces around everything; a sentence the editor removed shows none.

- `colman-02.04-skeleton-demo` #77, exact, score 3.83: [{But when does searching get caught up on]} details?" [{"In the class demo, I was going a little bit further because I wanted to show you as I was speaking through the approaches and my thought process."]} Right?
- `colman-02.04-skeleton-demo` #141, disjoint, score 2.44: [{the way I would like you to think for the creating more of a dynamic,] uh, [alive creature]} but also—
- `colman-02.04-skeleton-demo` #41, editor kept it whole (0.7), score 2.90: [{Dealing with that tube form we discussed] that—}
- `hampton-5.4-assignment-demo` #227, editor removed it (0.0), score 2.52: So, [part—]
- `colman-02.04-skeleton-demo` #93, exact, score 3.77: [{I always wanna see you draw through the form]} because just—
- `colman-03.03-muscles-crit` #83, disjoint, score 3.36: [{This is a common error that I'm seeing,] you} know?
- `colman-02.04-skeleton-demo` #159, editor kept it whole (0.7), score 3.21: {I just [think—]}
- `hampton-5.2-shape-demo` #39, editor removed it (0.0), score 3.22: So, [that's—]
- `colman-02.04-skeleton-demo` #127, exact, score 2.86: [{Now,]} um, [{for the assignment,]} I would—
- `perspective-14e-boxes-critique` #50, disjoint, score 2.59: Um, and [moving away from instrumental perspective {and toward the kind of freehand skills that we've seen some of our guest artists like Peter Han and Rembert showcase for us.]}
- `colman-03.03-muscles-crit` #90, editor kept it whole (0.7), score 3.24: [{So, everything's kinda locking together,] right?}
- `hampton-5.2-shape-demo` #46, editor removed it (0.0), score 3.31: So, [you're—]
- `hampton-5.4-assignment-demo` #153, exact, score 2.81: [{Here,]} here,
- `colman-03.03-muscles-crit` #154, editor kept it whole (0.7), score 3.43: {So, [as you create your—]}
- `hampton-5.2-shape-demo` #285, editor removed it (0.0), score 2.57: Here's [the—]

## Retake veto sweep

`cut_retake` rebuilt offline from the stored `retake_real`, `retake_choice` and `retake_take_probs`, then rescored. `none` never vetoes: the losers of Jev's chosen winner are always cut. The last two columns take the sentences each setting cut as retake losers and ask what the editor did with them.

| veto on real_k | threshold | SENTENCE POINTS | WORD SCORE | GRADE | SP at 2.40 | WORD at 2.40 | losers cut |   editor kept them |   editor cut them | Jev kept, editor removed | Jev removed, editor kept | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 2.40 | 82.88 | 76.23 | 86.29 | 82.88 | 76.23 | 304 | 56 | 248 | 229 | 218 | 83.70 |
| 0.2 | 2.40 | 82.81 | 76.46 | 86.52 | 82.81 | 76.46 | 278 | 49 | 229 | 235 | 213 | 83.67 |
| 0.3 | 2.40 | 82.66 | 76.40 | 86.47 | 82.66 | 76.40 | 250 | 40 | 210 | 240 | 212 | 83.52 |
| 0.4 | 2.40 | 82.66 | 76.42 | 86.49 | 82.66 | 76.42 | 225 | 31 | 194 | 247 | 204 | 83.56 |
| 0.5 (production Jev) | 2.40 | 82.52 | 76.31 | 86.40 | 82.52 | 76.31 | 183 | 22 | 161 | 255 | 199 | 83.45 |
| module flags | 2.40 | 82.73 | 76.18 | 86.22 | 82.73 | 76.18 | 304 | 57 | 247 | 229 | 221 | 83.59 |

Best on the headline: none at 82.88 SENTENCE POINTS, 76.23 WORD SCORE, 86.29 GRADE. Each setting calibrates its own keep threshold, so the last two columns rescore every setting at 2.40, the threshold the reported arm uses, to show how much of that win is the veto and how much is the threshold landing differently. On the fixed threshold the best is none at 82.88.

## Do the six score levels separate?

Mean probability Jev put on each level, over the 1505 sentences the editor kept and the 1238 it removed. If the levels carried the signal the prompt asks for, the removed column would load on 0 and 1 and the kept column on 4 and 5.

| score level | editor removed | editor kept | kept minus removed |
|---|---:|---:|---:|
| level 0 | 0.397 | 0.132 | -0.265 |
| level 1 | 0.080 | 0.030 | -0.050 |
| level 2 | 0.057 | 0.027 | -0.030 |
| level 3 | 0.260 | 0.337 | 0.078 |
| level 4 | 0.172 | 0.399 | 0.227 |
| level 5 | 0.035 | 0.074 | 0.039 |

Keep/cut agreement by how confident Jev was, where confidence is the probability mass on its top level.

| quartile | top-level probability | sentences | agreement |
|---|---:|---:|---:|
| Q1 | 0.23-0.42 | 727 | 74.83 |
| Q2 | 0.42-0.49 | 657 | 84.63 |
| Q3 | 0.49-0.61 | 696 | 85.63 |
| Q4 | 0.61-0.99 | 663 | 89.44 |

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-decisions.jsonl`, md5 d7e0589a2aaf, modified 2026-09-20T22:15:15+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-requests.jsonl`, md5 6982b25a7236, modified 2026-09-20T22:15:15+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-timing.json`, md5 f8b2871dea16, modified 2026-09-20T22:15:15+00:00
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-misses.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-v2-misses.json`
