# Jev rough cut: where the points go

Generated 2026-09-20T22:02:59+00:00 by `scripts/jev_real/roughcut_jev_misses.py` from the stored decisions and the cached removal ranges. No model calls, no detector runs, $0. Every metric is x100, two decimals; the JSON next to this file keeps the raw values.

The arm under the microscope is jev_a (t_trim 0.3) layered with um removal + delete silence, at its calibrated pooled keep threshold 2.40. That is the column the published ladder compares on, because every published arm gets the same two modules layered on. `jev_b_notrim` (identical scores, no partial trims, threshold 2.40) appears next to it wherever trimming is the question.

Self-check: recounting the full/partial/removed cross-tab from the per-sentence states this file reads reproduces the `pair_counts` block the metric scored, on every episode.

0 sentence(s) never got an answer from the run and are scored at 0.0, so they sit in the arm's removed column by construction.

The decisions file is fingerprinted at the bottom. It is rewritten whenever the pipeline patches answers it missed, and the numbers move when it does, so a table here will not match a run report generated against an older copy.

## Per episode against the ladder

SENTENCE POINTS, WORD SCORE and GRADE for the same six episodes, all three arms with um removal and delete silence layered on. `human kept` is the share of dialogue frames the editor kept, which is how tight the target cut is.

| episode | jev_a SP | jev_a WORD | jev_a GRADE | notrim SP | Luna SP | Luna WORD | Opus SP | Opus WORD | human kept | SP gap vs Luna |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 83.66 | 73.69 | 105.79 | 81.96 | 82.47 | 76.08 | 86.49 | 76.10 | 76.17 | 1.19 |
| hampton-5.4-assignment-demo | 86.33 | 74.99 | 94.39 | 86.33 | 90.57 | 85.60 | 94.60 | 89.47 | 72.88 | -4.23 |
| colman-03.03-muscles-crit | 75.41 | 62.22 | 74.96 | 75.91 | 69.01 | 68.34 | 73.63 | 66.40 | 97.65 | 6.40 |
| edges-7.01-intro | 77.56 | 74.27 | 75.18 | 77.56 | 89.59 | 89.34 | 90.39 | 89.71 | 47.89 | -12.03 |
| hampton-5.2-shape-demo | 92.04 | 84.30 | 95.69 | 92.41 | 90.78 | 81.60 | 95.06 | 87.60 | 68.99 | 1.27 |
| perspective-14e-boxes-critique | 84.62 | 80.29 | 83.29 | 84.49 | 89.01 | 85.03 | 90.54 | 86.36 | 52.35 | -4.39 |
| **pooled** | 83.84 | 76.46 | 86.41 | 83.77 | — | — | — | — | — | — |

The whole gap sits in edges-7.01-intro (-12.03), perspective-14e-boxes-critique (-4.39), hampton-5.4-assignment-demo (-4.23). On the other 3 the arm is level with Luna or ahead, by up to 6.40 on colman-03.03-muscles-crit.

## Sentence-level confusion

Every corpus sentence, at the calibrated threshold, by what the editor did and what the arm did. `kept` on either side means full or partial in that cut; `removed` means gone.

| episode | sentences | both keep | Jev kept, editor removed | Jev removed, editor kept |   of which editor full |   of which editor partial | both removed | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| colman-02.04-skeleton-demo | 194 | 146 | 7 | 19 | 9 | 10 | 22 | 86.60 |
| hampton-5.4-assignment-demo | 300 | 206 | 19 | 19 | 15 | 4 | 56 | 87.33 |
| colman-03.03-muscles-crit | 303 | 240 | 17 | 35 | 27 | 8 | 11 | 82.84 |
| edges-7.01-intro | 389 | 79 | 17 | 71 | 61 | 10 | 222 | 77.38 |
| hampton-5.2-shape-demo | 411 | 260 | 23 | 23 | 20 | 3 | 105 | 88.81 |
| perspective-14e-boxes-critique | 1146 | 303 | 74 | 104 | 84 | 20 | 665 | 84.47 |
| **pooled** | 2743 | 1234 | 157 | 271 | 216 | 55 | 1081 | 84.40 |

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

271 sentences the editor kept and the arm dropped, against 157 the arm kept and the editor dropped. It errs towards cutting. The lists below are the worst of each: the lowest-scoring drops, and the highest-scoring keeps.

### Jev removed, editor kept: the 40 lowest scores

Grouped by reading them with their neighbours. The grouping is judgement, not code; the counts are over these 40, not over all 271.

| pattern | of 40 |
|---|---:|
| Half a sentence the transcript split | 11 |
| Ordinary connective teaching talk | 9 |
| Retake pair, the wrong side cut | 6 |
| Scripted lesson line read as filler | 6 |
| Producer talk and student names in a critique | 5 |
| Thinking aloud, all ums | 2 |
| A joke the editor kept | 1 |

**Half a sentence the transcript split** (11). The transcript breaks one spoken sentence into two rows and the second row trails off in `..`. Jev reads the fragment on its own, calls it an abandoned false start, and drops it. The editor kept the whole spoken sentence, so both rows are in the cut.

- `hampton-5.2-shape-demo` #6, score 0.35: So, I used, uh, these to make sure that what we have..
- `colman-03.03-muscles-crit` #156, score 0.44: You have to kind of..
- `colman-02.04-skeleton-demo` #151, score 0.84: I always..

**Ordinary connective teaching talk** (9). Level 3 in the prompt's own words: fine, keeps the flow, nothing memorable. Jev put it at 1 or below. These are the demo lessons, where the editor keeps almost everything.

- `hampton-5.2-shape-demo` #5, score 0.58: And now we'll try that with a little bit more directness in terms of looking at a reference and seeing if we could practice that from that picture.
- `hampton-5.4-assignment-demo` #109, score 0.59: So here's my leg up and then down and then back on this side.
- `colman-03.03-muscles-crit` #273, score 0.67: As opposed to just kind of drawing it for what it is and getting a little bit more of a kind of a, uh, worried about like the silhouette and almost again, instead of looking at patterns, look at it i…

**Retake pair, the wrong side cut** (6). Two attempts at one line. Jev found the pair but crowned the other take, so the one the editor actually used got cut as the loser.

- `edges-7.01-intro` #203, score 0.70: "They're all soft edges with a different flavor!" Hmm...
- `edges-7.01-intro` #384, score 0.70: So let's get going.
- `hampton-5.2-shape-demo` #21, score 0.87: So, in this, I'm taking my part and just to..

**Scripted lesson line read as filler** (6). The written intro asks a question or names a list, one short sentence at a time. Jev scores a short sentence with no content of its own as throat-clearing; in a script it is the lesson.

- `edges-7.01-intro` #82, score 0.37: Sharp, firm, soft, and lost.
- `edges-7.01-intro` #355, score 0.75: So, how do you learn the rules?
- `edges-7.01-intro` #277, score 0.81: But wait, there's more!

**Producer talk and student names in a critique** (5). In a critique the teacher reads a name off the screen or says what the producer just put up. The prompt's level 0 names exactly that as a cut, and the editor keeps it, because it is how the critique moves.

- `perspective-14e-boxes-critique` #603, score 0.06: Jayoshi points out the principle behind overdoing it.
- `perspective-14e-boxes-critique` #908, score 0.62: What commitment.
- `perspective-14e-boxes-critique` #941, score 0.86: Anthony Hernandez.

**Thinking aloud, all ums** (2). A run of pure filler the editor kept anyway, because the pause is part of the demo's rhythm.

- `hampton-5.2-shape-demo` #104, score 0.86: so, I mean, hmm, mm-hmm.
- `hampton-5.2-shape-demo` #103, score 0.90: Um, uh, like, you know,

**A joke the editor kept** (1). Self-deprecating aside. The prompt asks for it at level 4 and Jev scored it at 1.

- `hampton-5.4-assignment-demo` #166, score 1.21: it's always the 6-line figure but I can never remember what the 6 lines are.

The list in full, worst first. `before` and `after` are the neighbouring sentences.

- `perspective-14e-boxes-critique` #603, score 0.06, level 0 0.98, level 1 0.01, pattern: Producer talk and student names in a critique. **Jayoshi points out the principle behind overdoing it.** (before: Sure. / after: Command F.)
- `hampton-5.2-shape-demo` #6, score 0.35, level 0 0.91, level 4 0.05, pattern: Half a sentence the transcript split. **So, I used, uh, these to make sure that what we have..** (before: And now we'll try that with a little bit more directness in terms of looking at a reference and seeing if we could prac… / after: or try to show these to make sure that you have that comparison of one that we've looked at with the first approach and…)
- `edges-7.01-intro` #82, score 0.37, level 0 0.90, level 4 0.05, pattern: Scripted lesson line read as filler. **Sharp, firm, soft, and lost.** (before: To be able to keep this stuff in our head and make decisions, it's a lot easier to think about 5 values and 4 edges. / after: Lost is actually off the spectrum a little.)
- `colman-03.03-muscles-crit` #156, score 0.44, level 0 0.80, level 1 0.06, pattern: Half a sentence the transcript split. **You have to kind of..** (before: you draw from your head, um, remember to continue and find a balance, find weight as well and make the adjustments, you… / after: you'd have to..)
- `hampton-5.2-shape-demo` #5, score 0.58, level 0 0.81, level 3 0.07, pattern: Ordinary connective teaching talk. **And now we'll try that with a little bit more directness in terms of looking at a reference and seeing if we could practice that from that picture.** (before: Okay, so we've been now through our, our alternative or secondary way of thinking about gesture, putting our emphasis m… / after: So, I used, uh, these to make sure that what we have..)
- `hampton-5.4-assignment-demo` #109, score 0.59, level 0 0.68, level 1 0.20, pattern: Ordinary connective teaching talk. **So here's my leg up and then down and then back on this side.** (before: it takes a bit to kind of settle into it for me sometimes, but that goes with the mindset also of like doing a longer d… / after: I'm..)
- `perspective-14e-boxes-critique` #908, score 0.62, level 0 0.80, level 4 0.08, pattern: Producer talk and student names in a critique. **What commitment.** (before: "Rupert, 'Dddd.' Unbelievable. / after: 'SB.' And I can't make it bigger in here, right?)
- `colman-03.03-muscles-crit` #273, score 0.67, level 0 0.73, level 1 0.09, pattern: Ordinary connective teaching talk. **As opposed to just kind of drawing it for what it is and getting a little bit more of a kind of a, uh, worried about like the silhouette and almost again, instead of looking at patterns, look at it in terms of forms, right?** (before: So, think about things like that. / after: Look at it in terms of volumes and building blocks and almost like you're carving it out of wood.)
- `edges-7.01-intro` #203, score 0.70, level 0 0.81, level 4 0.13, cut as a retake loser, pattern: Retake pair, the wrong side cut. **"They're all soft edges with a different flavor!" Hmm...** (before: So let me try that again. / after: they're all soft edges with a different flavor!)
- `edges-7.01-intro` #384, score 0.70, level 0 0.64, level 1 0.17, cut as a retake loser, pattern: Retake pair, the wrong side cut. **So let's get going.** (before: Up next, your first edge assignment. / after: Up next, your first edge assignment.)
- `hampton-5.2-shape-demo` #20, score 0.74, level 0 0.73, level 3 0.09, pattern: Ordinary connective teaching talk. **Here, I can look at the image and see that it's tilted and stretched and pinched maybe.** (before: Right, so, we thought of that as this, that was our pillar shape. / after: So, in this, I'm taking my part and just to..)
- `edges-7.01-intro` #355, score 0.75, level 0 0.73, level 3 0.13, pattern: Scripted lesson line read as filler. **So, how do you learn the rules?** (before: Draw from life. / after: Well, I think it's through observation and practice.)
- `edges-7.01-intro` #277, score 0.81, level 0 0.61, level 1 0.18, pattern: Scripted lesson line read as filler. **But wait, there's more!** (before: and the object receiving the cast shadow. / after: The edge type of a cast shadow can describe the distance between the object casting and the object receiving the cast s…)
- `colman-02.04-skeleton-demo` #151, score 0.84, level 0 0.58, level 1 0.18, pattern: Half a sentence the transcript split. **I always..** (before: Um, you know, —there's a lot of fun angles to try but I would like you to be simple first and then you could try some c… / after: extra credit is really just more for you.)
- `hampton-5.2-shape-demo` #104, score 0.86, level 1 0.77, level 0 0.20, pattern: Thinking aloud, all ums. **so, I mean, hmm, mm-hmm.** (before: Um, uh, like, you know, / after: You probably need to bring this foot down or this foot up.)
- `perspective-14e-boxes-critique` #941, score 0.86, level 0 0.68, level 3 0.14, pattern: Producer talk and student names in a critique. **Anthony Hernandez.** (before: Yeah. / after: It is kind of complicated, isn't it?)
- `hampton-5.2-shape-demo` #21, score 0.87, level 0 0.70, level 4 0.11, cut as a retake loser, pattern: Retake pair, the wrong side cut. **So, in this, I'm taking my part and just to..** (before: Here, I can look at the image and see that it's tilted and stretched and pinched maybe. / after: just to help you see this to begin, I'm thinking behind the head, it would be doing something like this.)
- `edges-7.01-intro` #210, score 0.88, level 0 0.73, level 4 0.12, cut as a retake loser, pattern: Retake pair, the wrong side cut. **all shapes have edges.** (before: All shapes.. / after: And this doesn't..)
- `hampton-5.2-shape-demo` #103, score 0.90, level 1 0.82, level 0 0.15, pattern: Thinking aloud, all ums. **Um, uh, like, you know,** (before: So, it comes this way, back here. / after: so, I mean, hmm, mm-hmm.)
- `perspective-14e-boxes-critique` #176, score 1.01, level 0 0.69, level 4 0.13, pattern: Half a sentence the transcript split. **If the..** (before: And if it's above us, the opposite is the case. / after: if we can see the bottom plane, that means that bottom corner is closer to us than the top corner.)
- `edges-7.01-intro` #361, score 1.07, level 0 0.68, level 4 0.16, pattern: Half a sentence the transcript split. **um, see how edges behave differently under various lighting conditions.** (before: See how edges behave differently.. / after: Start by learning..)
- `edges-7.01-intro` #242, score 1.08, level 0 0.67, level 4 0.12, pattern: Scripted lesson line read as filler. **When do you use a firm edge?** (before: When do you use a sharp edge? / after: When do you use a soft edge?)
- `colman-03.03-muscles-crit` #263, score 1.10, level 0 0.60, level 3 0.16, pattern: Ordinary connective teaching talk. **Interlocking everything.** (before: You're getting very contour heavy here. / after: See how everything is kinda..)
- `perspective-14e-boxes-critique` #887, score 1.10, level 0 0.67, level 4 0.11, pattern: Ordinary connective teaching talk. **Then it was worth it.** (before: Even if you never draw a single accurate cube, did you move forward in navigating in that pretend space? / after: Keep that in mind.)
- `hampton-5.4-assignment-demo` #77, score 1.15, level 0 0.38, level 1 0.34, pattern: Half a sentence the transcript split. **Maybe we could even..** (before: And then if I'm good with that, then I'm just going to start to put some of these more perspective-driven lines across. / after: yeah, I don't think on her specifically, but you could broaden the hip or, or change the shape in any way that you see …)
- `edges-7.01-intro` #28, score 1.16, level 0 0.71, level 4 0.17, pattern: Half a sentence the transcript split. **You'll find random sharp edges everywhere or overly smudged tones that hide uncertainty and mistakes instead..** (before: You can usually spot an amateur by looking at their edges. / after: oh, like, you know, or overly smudged tones that hide uncertainty and mistakes instead of communicating form.)
- `edges-7.01-intro` #241, score 1.17, level 0 0.64, level 4 0.12, pattern: Scripted lesson line read as filler. **When do you use a sharp edge?** (before: Okay. / after: When do you use a firm edge?)
- `perspective-14e-boxes-critique` #191, score 1.17, level 0 0.64, level 4 0.12, pattern: Producer talk and student names in a critique. **Let's go to Dermot.** (before: Okay, I think that's enough for Doobie for the time being. / after: We're down here at the bottom, which is the first people, right?)
- `colman-02.04-skeleton-demo` #6, score 1.19, level 0 0.62, level 4 0.16, pattern: Half a sentence the transcript split. **seeing what I did with the skeleton and skulls, I want to do..** (before: Uh, I want you to.. / after: you should do your own studies using the resource material we've provided.)
- `colman-02.04-skeleton-demo` #127, score 1.19, level 0 0.44, level 1 0.25, pattern: Half a sentence the transcript split. **Now, um, for the assignment, I would..** (before: So, very quick and loose but a pretty good, uh, breakdown, um, of a skeleton in a very objective streamlined approach, … / after: I actually would like to..)
- `colman-03.03-muscles-crit` #5, score 1.19, level 0 0.52, level 3 0.29, pattern: Ordinary connective teaching talk. **Uh, so weight is one thing, and I like seeing that everyone's trying to draw from their imagination, the weight of the pose and putting it together, and also the flow of the muscles together.** (before: Mistakes carry such a, such a heavy stigma. / after: We talked about flow..)
- `perspective-14e-boxes-critique` #464, score 1.20, level 0 0.61, level 4 0.15, pattern: Ordinary connective teaching talk. **This will help when I mention, uh, you've got your absolute true, not to be argued with reference that you know is a cube and that means that you can trust it and then compare of what you do and how it's different from that true reference.** (before: Okay. / after: Okay, that's what can be shown because she did these GIFs which are wonderful and will be illustrating what I say.)
- `hampton-5.4-assignment-demo` #166, score 1.21, level 0 0.64, level 4 0.19, pattern: A joke the editor kept. **it's always the 6-line figure but I can never remember what the 6 lines are.** (before: So, the reason I use this one is the way I remember the Reilly figure starting is always like.. / after: I think it's like 1, 2, 3, 4, uh, that's not it, something, something.)
- `colman-03.03-muscles-crit` #78, score 1.22, level 0 0.53, level 3 0.25, pattern: Half a sentence the transcript split. **It doesn't..** (before: All just kind of flows together. / after: not really worried once again about the, the names so much.)
- `hampton-5.2-shape-demo` #256, score 1.24, level 0 0.58, level 3 0.24, cut as a retake loser, pattern: Retake pair, the wrong side cut. **Where does it begin?** (before: Let's do this. / after: Where does that part attach?)
- `edges-7.01-intro` #102, score 1.27, level 0 0.69, level 4 0.18, pattern: Retake pair, the wrong side cut. **But lost edges don't always mean that it's super blurry between the two shapes.** (before: But lost edges don't always mean that it's super blurry between two shapes. / after: Hmm, sometimes it just means you've made the values of neighboring shapes identical.)
- `colman-03.03-muscles-crit` #155, score 1.29, level 0 0.58, level 3 0.14, pattern: Half a sentence the transcript split. **you draw from your head, um, remember to continue and find a balance, find weight as well and make the adjustments, you know, if this is actually a great pose but this once again, this wouldn't really work.** (before: So, as you create your.. / after: You have to kind of..)
- `edges-7.01-intro` #270, score 1.29, level 0 0.68, level 4 0.22, pattern: Scripted lesson line read as filler. **The edge on a cast shadow can describe the type of light source.** (before: But remember, we also have cast shadows. / after: A diffused light, like a cloudy sky, creates very soft cast shadows.)
- `perspective-14e-boxes-critique` #988, score 1.30, level 0 0.54, level 3 0.19, pattern: Producer talk and student names in a critique. **There's Sandra Susser** (before: Okay, then I'm gonna just show a few others and mention their name. / after: using atmospheric perspective on some of these planes.)
- `colman-03.03-muscles-crit` #230, score 1.32, level 0 0.51, level 3 0.21, pattern: Ordinary connective teaching talk. **And really where you can really gotta focus on weight, you know, and what really as you're dealing with these muscles, um, You think about how the pose really works and this one, there's just not a lot of weight, so I want you to still think about that even t…** (before: your strong design skills are gonna be a sum of all these parts. / after: You're posing and kind of the..)

### Jev kept, editor removed: the 40 highest scores

Same treatment from the other side, over these 40 of 157.

| pattern | of 40 |
|---|---:|
| Teaching line the picture already makes | 15 |
| Producer talk and studio logistics | 8 |
| Encouragement and wrap-up the editor tightened | 6 |
| Half a sentence the transcript split | 5 |
| Short connective dropped for pacing | 3 |
| Near-duplicate line, the loser kept | 1 |
| Operating the drawing or the screen | 1 |
| Tangent the editor cut short | 1 |

**Teaching line the picture already makes** (15). Real content, said clearly, and the editor still cut it: the drawing on screen says the same thing, or the point was already made a sentence earlier. Jev has no way to see the picture and grades the words.

- `edges-7.01-intro` #228, score 4.11: This highlight is sharp here, firm here, and soft here as it slowly fades away.
- `perspective-14e-boxes-critique` #580, score 3.83: In other words, if we were to look at that in an ortho, it would be shaped like this.
- `colman-02.04-skeleton-demo` #178, score 3.80: Um, once again, making sure there's room for all the teeth even though it's completely grid-based.

**Producer talk and studio logistics** (8). Two people arranging what happens next, or spelling a name out loud. Jev graded it 3 or above because it is fluent and on topic.

- `hampton-5.2-shape-demo` #297, score 3.57: That was great.
- `perspective-14e-boxes-critique` #957, score 3.42: F-M-I-N-K-E-E.
- `hampton-5.2-shape-demo` #298, score 3.29: So, then our next thing will just be giving people their assignment.

**Encouragement and wrap-up the editor tightened** (6). The pep talk at the end of a lesson. Warm, well said, and the editor keeps one line of it and drops the rest.

- `perspective-14e-boxes-critique` #1113, score 3.59: Piano is a great challenge.
- `hampton-5.2-shape-demo` #335, score 3.38: Don't be too hard on yourself.
- `hampton-5.2-shape-demo` #337, score 3.35: it's going to be uncomfortable at first, especially if you've been doing the other approach to kind of reorganize your thoughts and then, you know, redevelop a new way of representing and organizing …

**Half a sentence the transcript split** (5). The mirror of the same problem: the editor cut the whole spoken sentence, so the trailing fragment goes too, and Jev kept the fragment because the words it can see read as setup.

- `perspective-14e-boxes-critique` #716, score 3.55: if it were a..
- `colman-02.04-skeleton-demo` #10, score 3.53: Skeleton that we provided is..
- `hampton-5.2-shape-demo` #39, score 3.26: So, that's..

**Short connective dropped for pacing** (3). `Right?`, `All right?`, `Okay.` Jev scores the tag high because it reads as ordinary teaching talk; the editor cuts it to keep the cut moving.

- `colman-03.03-muscles-crit` #222, score 3.98: Right?
- `colman-03.03-muscles-crit` #297, score 3.53: All right?
- `perspective-14e-boxes-critique` #139, score 3.13: Okay.

**Near-duplicate line, the loser kept** (1). The next sentence says the same thing better and the retake detector never flagged the pair, so nothing cut it.

- `edges-7.01-intro` #261, score 3.22: A firm edge.

**Operating the drawing or the screen** (1). Level 0 in the prompt. Jev gave it 3.15.

- `hampton-5.2-shape-demo` #107, score 3.15: So, I could just drop that for now.

**Tangent the editor cut short** (1). The teacher loses the thread and the editor takes the detour out.

- `hampton-5.4-assignment-demo` #169, score 3.12: So, it's like 1, 2, 3, 4.

The list in full, worst first. `before` and `after` are the neighbouring sentences.

- `edges-7.01-intro` #228, score 4.11, level 4 0.66, level 5 0.28, pattern: Teaching line the picture already makes. **This highlight is sharp here, firm here, and soft here as it slowly fades away.** (before: Even a single small shape can have a changing edge type around it. / after: How do you know when to use each type of edge?)
- `colman-03.03-muscles-crit` #222, score 3.98, level 4 0.57, level 5 0.30, pattern: Short connective dropped for pacing. **Right?** (before: now go to the zoo and draw gorillas, you know, after you're going through this course or any ape, you're gonna have a r… / after: You can see how the flow of the muscles are mimicking and almost, um, being directed by what the skeletal structure is …)
- `perspective-14e-boxes-critique` #580, score 3.83, level 4 0.52, level 5 0.28, pattern: Teaching line the picture already makes. **In other words, if we were to look at that in an ortho, it would be shaped like this.** (before: and I have these wedges that are sticking out on either side and then we know it was meant to be that way. / after: That's what we're going to deal with next.)
- `colman-02.04-skeleton-demo` #11, score 3.81, level 4 0.78, level 5 0.11, pattern: Teaching line the picture already makes. **we only have one, which is chimpanzee.** (before: Skeleton that we provided is.. / after: But try a different angle, not an angle that I, I demoed, uh, during the episode.)
- `colman-02.04-skeleton-demo` #178, score 3.80, level 4 0.81, level 3 0.11, pattern: Teaching line the picture already makes. **Um, once again, making sure there's room for all the teeth even though it's completely grid-based.** (before: I'm just kind of carving it back into it to create more, um, clearer shapes. / after: And can I walk away from here gaining what I needed to gain, which is the understanding of all the different intricacie…)
- `perspective-14e-boxes-critique` #1113, score 3.59, level 4 0.69, level 3 0.11, pattern: Encouragement and wrap-up the editor tightened. **Piano is a great challenge.** (before: I noticed that with Yves, uh, who did a piano, that some of you did pianos too. / after: Of course, if it's something that you wanted to do anyway because in the first part of this course, you collected stuff…)
- `hampton-5.2-shape-demo` #297, score 3.57, level 4 0.67, level 3 0.17, pattern: Producer talk and studio logistics. **That was great.** (before: Okay. / after: So, then our next thing will just be giving people their assignment.)
- `perspective-14e-boxes-critique` #716, score 3.55, level 4 0.48, level 3 0.25, pattern: Half a sentence the transcript split. **if it were a..** (before: But you can see this is.. / after: if it's a cube, that's too tall.)
- `colman-02.04-skeleton-demo` #10, score 3.53, level 4 0.68, level 3 0.17, pattern: Half a sentence the transcript split. **Skeleton that we provided is..** (before: So one skeleton, one skull. / after: we only have one, which is chimpanzee.)
- `colman-03.03-muscles-crit` #297, score 3.53, level 4 0.52, level 3 0.16, pattern: Short connective dropped for pacing. **All right?** (before: So I'm doing this, this video earlier, and hopefully if you guys see, um, this before you get to this lesson, it'll hel… / after: But I commend all of you for doing..)
- `colman-03.03-muscles-crit` #287, score 3.46, level 4 0.52, level 3 0.21, pattern: Teaching line the picture already makes. **If you're starting to think about kind of what's going on on the other side, even though I don't see it..** (before: and when I talk about dynamic proportions, but they'll be so great, there'll be a great sense of of a dynamic energy in… / after: what's the weight like?)
- `perspective-14e-boxes-critique` #400, score 3.45, level 4 0.41, level 5 0.23, pattern: Teaching line the picture already makes. **You've got your absolute reference.** (before: And watch me dare on this next one." Again, you hardly need a teacher for that. / after: You do that 100 times and you will have what you need for seeing what up to 300, even up to 600 squares in perspective …)
- `colman-03.03-muscles-crit` #285, score 3.42, level 4 0.46, level 5 0.21, pattern: Teaching line the picture already makes. **This is the way of conceptual thinking that you need that will really help you communicate volume and structure, weight, and even your posing, and even the energy in your work will really have a sense of dynamic, um, proportions and have a sense of..** (before: It's a very tough challenge and the way to do that is to draw with volume, to draw as if you're sculpting, to draw arou… / after: and when I talk about dynamic proportions, but they'll be so great, there'll be a great sense of of a dynamic energy in…)
- `perspective-14e-boxes-critique` #957, score 3.42, level 4 0.55, level 3 0.25, pattern: Producer talk and studio logistics. **F-M-I-N-K-E-E.** (before: At least Minkee. / after: Laid out so clearly and with such economy of line.)
- `hampton-5.4-assignment-demo` #150, score 3.40, level 4 0.48, level 3 0.46, pattern: Teaching line the picture already makes. **So maybe I'll just extend that a little bit lower.** (before: And on this one, we have some of that top view on the foot. / after: This —So it looks like it's going away, but I would almost always do this unless I'm really working with that top-down …)
- `hampton-5.2-shape-demo` #335, score 3.38, level 4 0.58, level 3 0.24, pattern: Encouragement and wrap-up the editor tightened. **Don't be too hard on yourself.** (before: So, when you're doing these, make sure to, you know, have some patience, have some grace. / after: You're..)
- `perspective-14e-boxes-critique` #583, score 3.38, level 4 0.39, level 3 0.38, pattern: Teaching line the picture already makes. **And find out the logic of the vanishing points.** (before: I mean, in our next lesson group. / after: Okay,)
- `hampton-5.2-shape-demo` #337, score 3.35, level 4 0.63, level 0 0.12, pattern: Encouragement and wrap-up the editor tightened. **it's going to be uncomfortable at first, especially if you've been doing the other approach to kind of reorganize your thoughts and then, you know, redevelop a new way of representing and organizing the same information you've been looking at.** (before: You're.. / after: So, try to treat it with some levity, have fun.)
- `perspective-14e-boxes-critique` #350, score 3.32, level 5 0.36, level 4 0.35, pattern: Teaching line the picture already makes. **If they do, you practice it the other way.** (before: You see if lines diverge as they recede. / after: I think this might be one to show while I'm saying that.)
- `hampton-5.2-shape-demo` #334, score 3.31, level 4 0.54, level 3 0.26, pattern: Encouragement and wrap-up the editor tightened. **So, when you're doing these, make sure to, you know, have some patience, have some grace.** (before: Uh, okay. / after: Don't be too hard on yourself.)
- `perspective-14e-boxes-critique` #354, score 3.30, level 4 0.51, level 3 0.27, pattern: Teaching line the picture already makes. **Anthony Hernandez did a superb job.** (before: This is Anthony Hern.. / after: You sketch, you check to see if lines diverge as they go away, you practice it the other way.)
- `colman-03.03-muscles-crit` #30, score 3.29, level 3 0.52, level 4 0.33, pattern: Teaching line the picture already makes. **So, I'm just trying to connect all the parts, right?** (before: So, you can see the difference there. / after: And not getting too caught up in the details right now, I just wanna show you how we can really connect the parts.)
- `hampton-5.2-shape-demo` #298, score 3.29, level 4 0.57, level 3 0.18, pattern: Producer talk and studio logistics. **So, then our next thing will just be giving people their assignment.** (before: That was great. / after: Mm-hmm.)
- `hampton-5.2-shape-demo` #39, score 3.26, level 4 0.42, level 3 0.23, pattern: Half a sentence the transcript split. **So, that's..** (before: Pelvis is this way. / after: look, it's almost the same as the neck, right?)
- `perspective-14e-boxes-critique` #816, score 3.25, level 4 0.36, level 5 0.29, pattern: Encouragement and wrap-up the editor tightened. **Do it with curiosity.** (before: Don't judge yourself harshly. / after: It is to remember the reason you are doing this.)
- `perspective-14e-boxes-critique` #590, score 3.24, level 4 0.46, level 5 0.18, pattern: Half a sentence the transcript split. **that looks right enough.** (before: Away, where they converge, get the other set or the other ones in that line set to converge at the same point and then,… / after: When we get that together, then if we want a ramp on there, that ramp will find its hidden VP up here and will be able …)
- `edges-7.01-intro` #261, score 3.22, level 4 0.60, level 0 0.18, pattern: Near-duplicate line, the loser kept. **A firm edge.** (before: A sharp corner. / after: A firm edge would suggest a kind of rounded corner.)
- `hampton-5.2-shape-demo` #46, score 3.22, level 4 0.42, level 3 0.24, pattern: Half a sentence the transcript split. **So, you're..** (before: sometimes I like to include these 3 points, sometimes helps me see the early on even that quality of rotation, right? / after: if you're always thinking about that center and the distance to one side and the distance to the other, it does solve t…)
- `perspective-14e-boxes-critique` #1111, score 3.22, level 4 0.53, level 3 0.23, pattern: Producer talk and studio logistics. **Yeah, there, there's something.** (before: I'm just saying, like, if there was something. / after: I noticed that with Yves, uh, who did a piano, that some of you did pianos too.)
- `colman-03.03-muscles-crit` #286, score 3.21, level 4 0.47, level 3 0.19, pattern: Teaching line the picture already makes. **and when I talk about dynamic proportions, but they'll be so great, there'll be a great sense of of a dynamic energy in your work.** (before: This is the way of conceptual thinking that you need that will really help you communicate volume and structure, weight… / after: If you're starting to think about kind of what's going on on the other side, even though I don't see it..)
- `perspective-14e-boxes-critique` #343, score 3.21, level 5 0.36, level 4 0.26, pattern: Encouragement and wrap-up the editor tightened. **I don't know how I can make that point more emphatically and you hardly need a teacher's feedback for this.** (before: no receding divergence. / after: Let's close.)
- `perspective-14e-boxes-critique` #1138, score 3.21, level 4 0.36, level 5 0.24, pattern: Producer talk and studio logistics. **Yeah, but let me, let me say one thing about the next episode.** (before: Yeah. / after: One of you asked, "How important is it to get the angles right?" That's, that's what we're doing, but we're doing it on…)
- `perspective-14e-boxes-critique` #1112, score 3.20, level 4 0.50, level 3 0.27, pattern: Producer talk and studio logistics. **I noticed that with Yves, uh, who did a piano, that some of you did pianos too.** (before: Yeah, there, there's something. / after: Piano is a great challenge.)
- `perspective-14e-boxes-critique` #112, score 3.17, level 3 0.34, level 4 0.31, pattern: Producer talk and studio logistics. **Okay, got it.** (before: Unless you open the original. / after: Let's start with Cheyenne who got first post and you did a beautiful job on these cubes.)
- `edges-7.01-intro` #189, score 3.16, level 4 0.49, level 0 0.22, pattern: Teaching line the picture already makes. **But they're all just shapes transitioning from one to the other.** (before: All of these have different effects and they feel different. / after: They're all soft edges with a different flavor.)
- `hampton-5.2-shape-demo` #107, score 3.15, level 3 0.61, level 4 0.28, pattern: Operating the drawing or the screen. **So, I could just drop that for now.** (before: So, that's the only thing I'm noticing here is my, my leg got a little bit long on one side. / after: Here would be one of the, though, with the fallbacks or the kind of the negative aspects of this approach that I'm so u…)
- `edges-7.01-intro` #27, score 3.13, level 4 0.55, level 0 0.22, pattern: Teaching line the picture already makes. **You can usually spot an amateur by looking at their edges.** (before: Um, uh, like, you know, so, I mean, hmm, mm-hmm... / after: You'll find random sharp edges everywhere or overly smudged tones that hide uncertainty and mistakes instead..)
- `perspective-14e-boxes-critique` #139, score 3.13, level 4 0.32, level 3 0.23, pattern: Short connective dropped for pacing. **Okay.** (before: So, at the very top will be the oldest. / after: Alright.)
- `hampton-5.4-assignment-demo` #169, score 3.12, level 4 0.46, level 3 0.31, pattern: Tangent the editor cut short. **So, it's like 1, 2, 3, 4.** (before: Hmm, hmm, hmm. / after: It's some weird thing like this where I was like never able to remember all the different crossing rhythms in there, so…)
- `perspective-14e-boxes-critique` #814, score 3.11, level 4 0.37, level 5 0.25, pattern: Producer talk and studio logistics. **So, instead of having just, you know, the paragraph that I said, it was very specifically for what he's not doing.** (before: yeah, and maybe an inset with the actual text of it. / after: Don't judge yourself harshly.)

## Trim quality

Jev emitted 90 trims across the six episodes and 32 of them survive the keep threshold to reach the metric as a partial sentence. They are not the only partials: the um removal and delete silence layers cut inside another 275 sentences Jev had asked to keep whole. Both columns are below, because only the first is Jev's doing.

| trim outcome | Jev's own trims | layers only |
|---|---:|---:|
| exact, run for run (pays 2.0) | 5 | 103 |
| subset of the editor's runs (1.2) | 0 | 0 |
| overlaps, not contained (1.0) | 0 | 27 |
| disjoint from the editor's runs (0.6) | 3 | 38 |
| editor kept the sentence whole (0.7) | 21 | 95 |
| editor removed the sentence (0.0) | 3 | 12 |
| **total** | 32 | 275 |

For scale, the editor trimmed inside 388 sentences; the arm came back partial on 176 of them and kept or dropped the rest whole. Dropping Jev's trims entirely (`jev_b_notrim`) scores 83.77 pooled SENTENCE POINTS against 83.84 with them.

Fifteen trims, the arm's kept span in square brackets and the editor's in braces. A sentence the editor kept whole shows braces around everything; a sentence the editor removed shows none.

- `colman-02.04-skeleton-demo` #77, exact, score 3.83: [{But when does searching get caught up on]} details?" [{"In the class demo, I was going a little bit further because I wanted to show you as I was speaking through the approaches and my thought process."]} Right?
- `colman-02.04-skeleton-demo` #141, disjoint, score 2.83: [{the way I would like you to think for the creating more of a dynamic,] uh, [alive creature]} but also—
- `colman-02.04-skeleton-demo` #41, editor kept it whole (0.7), score 2.90: [{Dealing with that tube form we discussed] that—}
- `hampton-5.4-assignment-demo` #8, editor removed it (0.0), score 2.52: [He does, but he'll explain it in more abstract language,] like in—
- `colman-02.04-skeleton-demo` #93, exact, score 3.49: [{I always wanna see you draw through the form]} because just—
- `colman-03.03-muscles-crit` #83, disjoint, score 2.63: [{This is a common error that I'm seeing,] you} know?
- `colman-02.04-skeleton-demo` #102, editor kept it whole (0.7), score 3.58: [{is attached to the scapula,] right?}
- `colman-03.03-muscles-crit` #185, editor removed it (0.0), score 2.70: [It's looking pretty good] as I can—
- `colman-02.04-skeleton-demo` #149, exact, score 2.84: and [{you can choose a different angle if you want to.]}
- `perspective-14e-boxes-critique` #50, disjoint, score 2.40: Um, and [moving away from instrumental perspective {and toward the kind of freehand skills that we've seen some of our guest artists like Peter Han and Rembert showcase for us.]}
- `colman-02.04-skeleton-demo` #156, editor kept it whole (0.7), score 3.02: [{Start with our ball here,] right?}
- `perspective-14e-boxes-critique` #564, editor removed it (0.0), score 2.66: ["Yeah, it doesn't look] like that." Oh, yeah, yeah, yeah.
- `colman-03.03-muscles-crit` #281, exact, score 3.82: [{So, just like you're chiseling in on a piece of wood, you know, whittling away or sculpting,]} you're kind of—
- `colman-03.03-muscles-crit` #32, editor kept it whole (0.7), score 2.86: [{Even something like that, even like this one here, yeah, the pose,] right?}
- `perspective-14e-boxes-critique` #373, exact, score 2.90: —so [{that you don't have to do 1,000 more like you apparently did before we began.]}

## Retake veto sweep

`cut_retake` rebuilt offline from the stored `retake_real`, `retake_choice` and `retake_take_probs`, then rescored. `none` never vetoes: the losers of Jev's chosen winner are always cut. The last two columns take the sentences each setting cut as retake losers and ask what the editor did with them.

| veto on real_k | threshold | SENTENCE POINTS | WORD SCORE | GRADE | SP at 2.40 | WORD at 2.40 | losers cut |   editor kept them |   editor cut them | Jev kept, editor removed | Jev removed, editor kept | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 2.10 | 83.57 | 76.21 | 86.34 | 83.94 | 76.35 | 304 | 54 | 250 | 231 | 202 | 84.21 |
| 0.2 | 2.10 | 83.57 | 76.41 | 86.46 | 83.98 | 76.56 | 281 | 48 | 233 | 235 | 198 | 84.21 |
| 0.3 | 2.10 | 83.43 | 76.38 | 86.48 | 83.87 | 76.53 | 253 | 39 | 214 | 239 | 197 | 84.10 |
| 0.4 | 2.10 | 83.54 | 76.46 | 86.56 | 84.02 | 76.60 | 231 | 31 | 200 | 242 | 190 | 84.25 |
| 0.5 (production Jev) | 2.40 | 83.84 | 76.46 | 86.41 | 83.84 | 76.46 | 186 | 20 | 166 | 157 | 271 | 84.40 |
| module flags | 2.10 | 83.54 | 76.23 | 86.35 | 83.87 | 76.35 | 304 | 57 | 247 | 228 | 206 | 84.18 |

Best on the headline: 0.5 (production Jev) at 83.84 SENTENCE POINTS, 76.46 WORD SCORE, 86.41 GRADE. Each setting calibrates its own keep threshold, so the last two columns rescore every setting at 2.40, the threshold the reported arm uses, to show how much of that win is the veto and how much is the threshold landing differently. On the fixed threshold the best is 0.4 at 84.02.

## Do the six score levels separate?

Mean probability Jev put on each level, over the 1505 sentences the editor kept and the 1238 it removed. If the levels carried the signal the prompt asks for, the removed column would load on 0 and 1 and the kept column on 4 and 5.

| score level | editor removed | editor kept | kept minus removed |
|---|---:|---:|---:|
| level 0 | 0.470 | 0.144 | -0.326 |
| level 1 | 0.120 | 0.057 | -0.063 |
| level 2 | 0.062 | 0.035 | -0.027 |
| level 3 | 0.139 | 0.249 | 0.110 |
| level 4 | 0.166 | 0.420 | 0.253 |
| level 5 | 0.042 | 0.095 | 0.053 |

Keep/cut agreement by how confident Jev was, where confidence is the probability mass on its top level.

| quartile | top-level probability | sentences | agreement |
|---|---:|---:|---:|
| Q1 | 0.20-0.41 | 692 | 74.28 |
| Q2 | 0.41-0.51 | 683 | 80.53 |
| Q3 | 0.51-0.64 | 711 | 88.33 |
| Q4 | 0.64-0.98 | 657 | 94.82 |

## Files

- input decisions: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-decisions.jsonl`, md5 a55f5b40cc88, modified 2026-09-20T21:53:34+00:00
- input requests: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-requests.jsonl`, md5 ee303ea5881a, modified 2026-09-20T21:53:33+00:00
- input timing: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-timing.json`, md5 1da7aeb624cb, modified 2026-09-20T21:53:34+00:00
- this file: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-misses.md`
- data: `C:\Users\Stan\Documents\GitHub\skell-e-router\docs\jev-real\roughcut-jev-misses.json`
