# Where Fable 5.1 is still hard-coded (Nov 9, 2026)

Scan of every git repo under `Documents\GitHub` (64 folders) and `Documents\GitLab` (31 folders) for `claude-fable-5-1`, `fable-5-1`, `fable 5.1` and similar spellings, tracked files only. GitLab has no references at all. Benchmark result files, logs, Slack-thread archives, handoff docs and the Obsidian vault mention Fable 5.1 as history and are not listed here. What follows is code and config that would actually send traffic to Fable 5.1 or offer it as a choice, plus the orchestrator briefs.

## Pins that send real traffic to Fable 5.1

**houston, six cron jobs.** Each launches a lead on `claude-fable-5-1[1m]` by name. These are the only places where Fable 5.1 is the model actually used, not just offered.

| File | Line |
|---|---|
| `crons/jobs/aingel-morning-brief.ps1` | 35, `-Model 'claude-fable-5-1[1m]'` |
| `crons/jobs/meeting-notes.ps1` | 43 |
| `crons/jobs/second-brain-refresh.ps1` | 25 (comment cites Stan's 2026-09-07 order) |
| `crons/jobs/shakuro-copy-review.ps1` | 34 |
| `crons/jobs/solar-telemetry-weekly.ps1` | 31 |
| `crons/jobs/team-feedback-sweep.ps1` | 28 |

**skell-e-web, the chat routing nickname.** `backend/rag/routing.py` line 243 maps the word "fable" typed in a chat message to `claude-fable-5-1` at medium effort. Nothing defaults to it: the chat default is Haiku 5.5, the big tier is Opus 5.5, and the general default is Gemini 3.1 Pro.

## Menu entries that offer Fable 5.1 as a choice

- **skell-e-web.** `backend/constants.py` line 104 (allowed-model list), `proko-app/src/app/model-options.constants.ts` lines 38, 90, 140 and 161 (picker labels and the "Frontier" tier tag). Tests in `backend/tests/` reference it too.
- **solar-sailer.** `editor/server/services/chat_agent/api_models.py` line 69 (API chat model catalog; default is GPT-6 Luna), `editor/server/services/chat_agent/cli_backends/selection.py` lines 83 and 100 (Claude Code CLI curated list and label; default is still `claude-fable-5`), `editor/src/services/chatBackends.ts` lines 42 and 69 (the frontend mirror of that list, drift-guarded against selection.py). Comments only in `model.py`, `usage.py` and `roughcut_agent.py`.

## claude-orchestrator

**The default is already moving to Fable 5.2, uncommitted.** `relay/src/models.js` in the working tree sets `DEFAULT_MODEL` to `claude-fable-5-2[1m]` with a comment citing Stan's order of 2026-11-09, and adds both Fable 5.2 ids to the allowlist. `briefings/orchestrator.md` is also modified. Neither change is committed yet (last commit touching models.js is from 2026-09-30), and `relay/test/models.test.js` lines 15 to 17 still assert the default is Fable 5.1, as does `relay/test/leads.test.js` line 5272. Whoever is doing that switch has the tests and two docs left.

**Standing instructions that name Fable 5.1 as the model to use:**

- `CLAUDE.md` line 47: "Leads get `DEFAULT_MODEL`, Fable 5.1". Needs to say Fable 5.2 once the switch lands.
- `briefings/orchestrator.md` line 80: the launch_lead row's parenthetical "(Fable 5.1, retained Fable 5 pins, and Opus 5.5)". The committed version; the working-tree version may already be updated.
- `state/NOTES.md` line 49: "Driving Solar Sailer for real edits for Stan = Fable 5.1, never gpt-6-astra (Stan, 2026-10-30)." This is the orchestrator's own standing note and will keep steering Solar Sailer leads to 5.1 unless updated.
- `state/lead-briefs/1788323115.034389.md` line 3: a staffing rule that every subagent runs on Fable 5.1. This is a past lead's brief, so it only matters if it is reused as a template.

The `state/model-windows.json` entry for `claude-fable-5-1` and the retained allowlist entries are fine to keep: existing leads pinned to 5.1 need them to resume.

## Not a problem

- `skell-e-router` itself: Fable 5.1 stays registered alongside 5.2, as it should. Nothing defaults to it.
- `benchmark`: config entries and result docs, historical.
- `skell-e-agent`: one changelog mention.
- `aingel`: one plan doc.
- `stan-vault`: copies of benchmark docs and a daily note.

## If you want these moved to Fable 5.2

The router accepts `claude-fable-5-2` as of 3.48.0 and the benchmark shows 5.2 at low effort matches 5.1's best score at half the cost. The edits are small: six one-line changes in houston, one line in skell-e-web routing plus the two picker files, three list files in solar-sailer (the two curated lists must change together or their drift test fails), and the orchestrator's CLAUDE.md, briefing parenthetical, tests and NOTES.md once the in-flight models.js change is committed. Each repo should get its own lead because the houston crons and the orchestrator relay are live systems.
