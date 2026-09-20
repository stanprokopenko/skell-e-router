# Solar Sailer scout: retakes panel, preferences and keys, rough cut bench page

Read-only scout of `C:/Users/Stan/Documents/GitHub/solar-sailer` on 2026-09-20. Nothing was written or modified in that repo or on `D:`. All paths below are absolute; line numbers are from the files as of this date.

Note on drives: `C:/Users/Stan/Documents/GitHub/solar-sailer/benchmarks` is a directory junction to `D:/solar-sailer/benchmarks`. The two paths are the same bytes (verified: `bench-page-arms.json` has identical md5 `848ba3aa7ee94b4ba922228e1199e608`, size and mtime on both). Editing one edits both.

## 1. The retakes panel

### Verdict on the AI button

The "AI selection of best takes" button **exists in the UI but is permanently disabled and wired to nothing**. It is a placeholder, labelled `AI auto-pick` with a `soon` tag.

`C:/Users/Stan/Documents/GitHub/solar-sailer/editor/src/components/RetakeReview/RetakeReviewPanel.tsx:689-699` renders it: a `<span title="AI winner picking is coming later">` wrapping a `<Button variant="ghost" size="sm" className={styles.aiBtn} disabled iconLeft={<Sparkles size={12} />}>AI auto-pick <span className={styles.soonTag}>soon</span></Button>`. There is no `onClick`. The CSS comment at `C:/Users/Stan/Documents/GitHub/solar-sailer/editor/src/components/RetakeReview/RetakeReview.module.css:125-138` calls it "the honest seam for the future AI auto-pick — the ONLY AI affordance". The panel-chrome doc says the same: `C:/Users/Stan/Documents/GitHub/solar-sailer/docs/editor/components.md:133` ("a disabled 'AI auto-pick — soon' affordance").

No server endpoint, module, or model call backs it. Grep over `editor/src` and `editor/server` for `ai_select`, `select_best`, `smart_select`, `best_take` turns up only the manual op `selectBestTake` and its tests. The retakes module itself never calls an LLM: `C:/Users/Stan/Documents/GitHub/solar-sailer/editor/server/modules/retakes.py:37-38` imports only `embed_batch` and `LLMAuthError`; grouping is embedding similarity, and `retakes_diagnostic.py` does not even embed (it replays the cache). The only LLM-driven module in this neighbourhood is rough cut (`C:/Users/Stan/Documents/GitHub/solar-sailer/editor/server/modules/roughcut.py:56, 243-341`, which uses `call_llm`).

### Component files

- `.../editor/src/components/RetakeReview/RetakeReviewHost.tsx` (118 lines) — modeless floating host. Size constants at L17-23; the header-fit comment at L18-20 names the "AI/?/X" cluster. Empty-state Dialog when no media has groups.
- `.../editor/src/components/RetakeReview/RetakeReviewPanel.tsx` (846 lines) — the panel. Header L673-719 (title, media `Select`, AI placeholder, shortcuts tooltip, close). Stats chips L720-747. Auto-play toggle L749-768. "Reset all" L769-777. Group list L779-806. Footer with progress, "Remove unused takes" and "Done" L808-833. Remove confirm dialog L835-845.
- `.../editor/src/components/RetakeReview/GroupCard.tsx` (278 lines) — one group; winner pip logic L59-79, per-take text mode (kept/diff/same) L98-122, take rows L250-256, correction affordances around L229.
- `.../editor/src/components/RetakeReview/TakeRow.tsx` (182 lines) — one take row, `isKept` styling L74, "same words" chip L86.
- `.../editor/src/components/RetakeReview/takeSequencer.ts`, `WaveformStrip.tsx`, `waveformCache.ts`, `wordDiff.ts`, `panelKeys.ts` — audition, waveforms, diff, keyboard.
- Entry points: View menu `view:retake-review` and the timeline pill right-click, routed through `.../editor/src/services/retakes/retakeReviewController.ts`; pill menu at `.../editor/src/components/Timeline/TranscriptRetakeMenu.tsx:94-95, 149` ("Select as best take").

### State shape for groups and selections

Groups are **derived**, not stored. `.../editor/src/services/retakes/deriveRetakeGroups.ts` builds `DerivedRetakeGroup` (L192-229) from the transcript doc: `groupId`, `letter`, `colorHex`, `takes: DerivedRetakeTake[]` (L170-190: `displayNumber`, `takeIndex`, `sids`, `startSec`, `endSec`, `isFalseStart`, `isRemoved`, `sentences`), `defaultTakeIdx`, `winnerTakeIdx`, `isChanged`, `notARetake`, `removedSids`, `memberSids`, `anchorSec`.

The **default** winner is the chronologically last real, non-removed take (`deriveRetakeGroups.ts:322-328`). A user pick overrides it. `isChanged = winnerTakeIdx !== defaultTakeIdx` (L352).

Two persisted, user-owned arrays live on the transcript sidecar, both typed in `.../editor/src/store/slices/transcriptsSlice.ts`:

- `RetakeGroupEntry` (L64-71): `{ id, winner_take_index, winner_sids, member_sids, picked_by: 'user' | 'ai', picked_at }`. `winner_take_index === null` means the winner is a false start identified by `winner_sids` alone. The doc comment at L61-62 states outright: "The AI seam reserves `picked_by: \"ai\"` plus confidence/reason/model fields — not implemented, extra JSON fields are tolerated structurally."
- `RetakeCorrectionEntry` (L86-93): `{ group_id, member_sids, not_a_retake?, removed_sids?, corrected_by: 'user' | 'ai', corrected_at }`.

Both hang off `TranscriptDoc` at L103-104. Staleness rule: an entry applies only while the group id exists and its current sorted member sids equal `member_sids`.

### How a user changes the chosen take, end to end

1. Click a take row (or press Enter) → `pickTake` in `RetakeReviewPanel.tsx:293-329`. It refuses removed takes (L300-301), pushes a `PickHistoryEntry` for panel-local Ctrl+Z (L304-312), marks the group heard, then calls `selectBestTake(store, { mediaId, groupId, targetTake: { takeIndex, sids } })` (L316-320) and auto-advances to the next group (L327).
2. `.../editor/src/services/retakes/selectBestTake.ts` — refuses not-a-retake groups (L244-247) and removed takes (L249-252), builds a plan via `selectBestTakePlan` (L253), then runs **one** `executeCommands` batch labelled `'Select best take'` = one undo entry (`runPlanCommands` L100-114, call at L254).
3. The plan (`.../editor/src/services/retakes/selectBestTakePlan.ts:1-13, 123-127, 199`) is a pair: `setClipEnabled` to re-enable whatever this feature or the retakes module previously disabled, then `disableRanges { source: 'retake-review' }` over the new losers. Its `persistPayload` is `RetakeWinnerPersistPayload` (L45-51).
4. Optimistic store write of the `RetakeGroupEntry` with `picked_by: 'user'` (`selectBestTake.ts:256-266`), then `POST /retakes/winner` (`.../editor/src/services/retakesApi.ts:38-52`), then replace the store array with the server's echo. Failure → error toast plus transcript resync (L277-284).
5. Server: `.../editor/server/routers/retakes.py:511-531` (`post_retakes_winner`) → `_write_winner` (L359-473). It validates membership, then **hardcodes** `"picked_by": "user"` at L438 and a UTC timestamp at L439-441, does a self-healing upsert under `transcript_lock`, and writes atomically. The corrections endpoints mirror this and hardcode `corrected_by = "user"` at L658.
6. Corrections ("Not a retake", "Remove take from group") go through `.../editor/src/services/retakes/corrections.ts` (`setGroupNotARetake` L~240-260, `setTakeRemoved`), same one-batch discipline, `POST /retakes/correction` / `/clear`.

### How the user keeps editing afterwards

The panel is modeless and never blocks the timeline. Losers stay on the timeline **disabled**, not deleted, so every pick is reversible. Panel-local Ctrl+Z reverts the last pick through the same op rather than global undo (`RetakeReviewPanel.tsx:381-397`; rationale in `docs/editor/components.md:135`). "Reset all" returns changed groups to the default last-take rule. Committing is a separate explicit step: "Remove unused takes" builds one `deleteRanges` command with `ripple: true` behind a `ConfirmDialog` (`RetakeReviewPanel.tsx:818-824, 835-845`; `buildRemoveUnusedTakesCommand` / `removeUnusedTakes` in `selectBestTake.ts`). Downstream, rough cut honours persisted picks by forcing loser scores to 0.0 (`docs/editor/processing-modules.md:230`).

### Background docs worth reading before building on this

- `C:/Users/Stan/Documents/GitHub/solar-sailer/docs/editor/processing-modules.md:181-203` — the Retakes section: schema, staleness, carry-forward across module re-runs, corrections composition rule, apply paths.
- `C:/Users/Stan/Documents/GitHub/solar-sailer/docs/editor/components.md:129-137` — panel chrome, entry points, keyboard, corrections.
- `C:/Users/Stan/Documents/GitHub/solar-sailer/docs/superpowers/plans/2026-08-07-retake-review-panel-build-contract.md` — the D1/D2/D4/D6/D13 contract the code cites throughout.
- `C:/Users/Stan/Documents/GitHub/solar-sailer/docs/superpowers/plans/2026-08-24-retake-review-corrections.md`.

## 2. Preferences and API keys

### Preferences UI

`.../editor/src/components/Dialogs/PreferencesDialog.tsx` — section union at L138 (`'account' | 'autosave' | 'import' | 'sync' | 'agent' | 'models' | 'apiKeys' | 'mobile' | 'cache' | 'diagnostics' | 'scripting' | 'plugins' | 'licenses'`), nav list L560-577, `<ApiKeysSection />` at L694, `<ModelsSection>` at L681, deep-link handling L297-305.

`.../editor/src/components/Dialogs/ApiKeysSection.tsx` (248 lines) — per-provider rows at L121; `ApiKeyRow` L135-244 with a `<input type="password">` at L207-217, Save at L226-231, masked status `Saved. Ends in ••••{last4}` at L156-164, live/restart notice at L42 and L77.

`.../editor/src/services/apiKeys.ts:43-96` — `API_KEY_DEFS`: AssemblyAI, OpenAI, Anthropic, Gemini, Groq, xAI, Jina. Each carries label, "used for" text, a `looksValid` check and `expectedShape`; the non-blocking format check is `checkApiKeyFormat` at L103-110. **There is no TypeSafe/Jev row.**

### Server-side storage

- **Keys are encrypted, not plaintext.** `.../editor/electron/apiKeyStore.ts` (266 lines): `API_KEY_NAMES` at L39-47; file is `userData/api-keys.json` (path built in `.../editor/electron/index.ts:393-400`); format `{keys: {NAME: {ciphertext: base64}}}`; `readStoreFile` L118-140, `writeStoreFile` L142-152 (tmp+rename, `mode: 0o600`); encryption via Electron `safeStorage` (DPAPI on Windows, Keychain on macOS) in `setApiKey` L177-215 (encrypt at L205), `getDecryptedApiKeys` L238-266, `deleteApiKey` L217-236. The renderer only ever sees `{isSet, last4, inShellEnv}` (L56-69, `statusFor` L154-164).
- **General preferences are plain JSON**: `.../editor/electron/preferencesFile.ts` writes `userData/preferences.json` (path at `index.ts:374`); `readPreferencesFileState` L26-31, `writePreferencesFileCrashSafe` L45-53.
- **Model tier overrides**: `.../editor/server/services/model_registry.py:39-53` reads `%APPDATA%/solar-sailer/user_models.json`, loads at L56-66, merges at L78-103.
- Server-side "preferences" modules are in-RAM mirrors only, not files: `.../editor/server/services/analysis_preferences.py:22-85`, `.../editor/server/services/import_preferences.py:13-64`.

### How a key reaches a server module at call time

**Environment variables only.** No per-call config dict, no key singleton. Two delivery paths:

1. **Spawn merge** — `.../editor/electron/index.ts:2248-2262` spreads `getDecryptedApiKeys(apiKeyStoreDeps())` into the Python process env at start.
2. **Live push** — IPC `api-keys:set` / `api-keys:delete` (`index.ts:1453-1467`) → `.../editor/electron/apiKeyPush.ts` (`effectiveKeyValue` L27-37, saved key overrides shell env; `pushApiKeyToServer` L39-70) POSTs `http://127.0.0.1:{port}/preferences/api-keys` → `.../editor/server/routers/preferences_router.py:64-88` (`push_api_keys`), which sets `os.environ[name] = value` at L79-84, gated by the allowlist `_PUSHABLE_KEY_NAMES` at L46-54 (unknown name → 400 at L76-78).

Consumers read `os.environ` at call time: `.../editor/server/services/key_preflight.py:53-69` (`PROVIDER_ENV`), `_key_status` L237-241, `preflight_module_keys` L301-320. Leak guard for child processes: `.../editor/server/services/subprocess_env.py:45-95` (`ENV_DENY_LIST`) plus `clean_subprocess_env` from L99.

### skell-e-router in the editor today

Declared in `.../editor/server/requirements.txt:22`; pinned `skell-e-router==3.27.2+solar1` in `requirements.lock:3302` and `requirements.windows.in:125`; wheel manifest `.../editor/server/locked-wheels/sources.json:49-78`. A vendored runtime copy sits at `.../editor/.staging/python/Lib/site-packages/skell_e_router`.

Exactly two router functions are used, both behind wrappers, both **keyless at the call site**:

- `ask_ai` — `.../editor/server/services/llm_router.py:20-24` (lazy import), called at L225, L258, L364 inside `call_llm` (L195-318) and `call_llm_tools` (L338-375). The kwargs built at L219-227 are model parameters only; **no `config=` argument is ever passed.**
- `get_embedding` — `.../editor/server/services/embeddings_router.py:37-41`, used by `embed_batch` L44-57.
- Import warm-up only: `.../editor/server/services/ai_runtime.py:29`.

`classify` is **never** called anywhere under `editor/`. `TYPESAFE`, `typesafe` and `jev` have zero hits under `editor/`. Jev usage lives only in the benchmarks tree (`.../solar-sailer/benchmarks/jev-chapter-split-probe/score_sentences.py:8`, `probe_clusters.py:16`, `README.md:15,33`), where the key comes from `TYPESAFE_API_KEY` in the process env.

### Could `classify(..., config={"typesafe_api_key": ...})` take a per-user key?

Yes, and most of the plumbing already exists, but four pieces are missing.

Upstream semantics, from `skell_e_router/classification.py:208-229` (read in a local venv copy): the signature is `classify(model, state, questions, *, config: dict | None = None, timeout: float = 30)`, and the key resolution at L223 is `config.get("typesafe_api_key") if config is not None else os.getenv("TYPESAFE_API_KEY")`. That is **either/or**: passing any dict disables the env fallback, so `config={}` raises `RouterError("MISSING_ENV", ...)` at L224-225.

Reusable as-is: encrypted per-user storage, save/delete/status IPC, and the masked UI row all generalize. Adding one entry to `apiKeyStore.ts:39-47` and one `ApiKeyDef` to `apiKeys.ts:43-96` renders a `TYPESAFE_API_KEY` row with no new UI code. Both delivery paths (`index.ts:2258` spawn merge, `preferences_router.py:79-84` live push) already handle any allowlisted name.

Missing:

1. **A six-way name mirror.** The contract is documented at `apiKeyStore.ts:28-38`: `src/types/electron.d.ts` `ApiKeyName`, `src/services/apiKeys.ts` `API_KEY_DEFS`, `preferences_router.py` `_PUSHABLE_KEY_NAMES` (L46-54), `key_preflight.py` `PROVIDER_ENV`/`PROVIDER_LABEL` (L53-69), and `subprocess_env.py` `ENV_DENY_LIST` (L45-95, which has a parity test that fails when a name is absent).
2. **No `classify` wrapper.** `llm_router.py` exposes only `ask_ai`; there is no `call_classify`. Importing `skell_e_router` outside the wrappers is forbidden by the plugin contract (`.../editor/plugin-docs/plugin-development-guide.md:34`) and pinned by a test that blocks it from the startup import graph (`.../editor/server/tests/test_ai_runtime_startup.py:30`). A wrapper is mandatory.
3. **No seam for a config dict.** Since keys arrive as env vars, the simplest correct call is `classify(model, state, questions)` with no `config`, letting L223's `os.getenv` branch pick up the pushed value. A `config={"typesafe_api_key": ...}` form would need the wrapper to read `os.environ["TYPESAFE_API_KEY"]` itself and to **omit** `config` entirely when the key is absent, or L223-225 raises.
4. **No cost attribution path.** `_record_cost` (`llm_router.py:150-170`) and `.../editor/server/services/api_pricing.py:8` assume the `ask_ai` rich-response shape; a `ClassificationResponse` (parsed at `classification.py:156-206`) needs its own ledger path.

## 3. The rough cut benchmark page

### Site and page

Docs site source is `C:/Users/Stan/Documents/GitHub/solar-sailer/website-docs/`, a **Docusaurus 3.10.2** site (`website-docs/docusaurus.config.ts`; version pinned at `website-docs/package.json:20-22`). No mkdocs, astro, vitepress, next or jekyll config exists in the repo.

The page is a hand-written React page, not generated markdown: `C:/Users/Stan/Documents/GitHub/solar-sailer/website-docs/src/pages/rough-cut-bench.tsx` (1787 lines) plus `rough-cut-bench.module.css`. Docusaurus `src/pages/` routing publishes it at `docs.solarsailer.com/rough-cut-bench`. It fetches its data at runtime: `rough-cut-bench.tsx:1739` does `const url = useBaseUrl('/data/rough-cut-bench.json')`. The header comment at L7-13 explains why it is hand-rolled inline SVG with no chart library: a strict `default-src 'self'` CSP.

**What regenerates is the data file**, `website-docs/static/data/rough-cut-bench.json`. Never hand-edit it.

A stale duplicate of the page sits at `.../solar-sailer/.free-tier-b/refactor-check/website-docs/src/pages/rough-cut-bench.tsx`. Ignore it.

### Generation pipeline

Script: `C:/Users/Stan/Documents/GitHub/solar-sailer/benchmarks/roughcut/scripts/export_bench_page.py` (972 lines).

- Input constants L58-62: `BENCH_DIR` (58), `RESULTS_DIR = BENCH_DIR/results` (59), `CODENAMES_PATH = BENCH_DIR/episode-codenames.json` (60), `REGISTRY_PATH = BENCH_DIR/bench-page-arms.json` (61), `WINNER_PATH = RESULTS_DIR/WINNER.md` (62, the legacy-arm ladder source).
- Output: `find_output_path()` L130-152 returns `<repo>/website-docs/static/data/rough-cut-bench.json` (L147). It deliberately walks up six levels without resolving symlinks because `benchmarks/` is a junction to another drive (docstring L132-135). Override with `--out`.
- Registry load in `main()` at L904-905 (`main()` spans L898-970).
- Arms loop: `build()` at L610, `for entry in registry["current"]:` at L621. Inside: `hidden` skip L622-626, `load_arm_rows()` L627, model/workflow/variant derivation cross-checked against the result JSONs with a hard failure on mismatch L633-649, per-episode pooling from L661.
- Per-result reader: `load_arm_rows()` L157-209, globbing `RESULTS_DIR` at L162 and pulling `levels[neutral]` fields at L186-207.
- Write gates in `main()`: `mojibake_check` L918-925, `leak_check` L927-932 (aborts on internal episode ids, artist names, or local paths), write at L962-966. `--check` computes and writes nothing (L958-960).

### bench-page-arms.json

Location: `C:/Users/Stan/Documents/GitHub/solar-sailer/benchmarks/roughcut/bench-page-arms.json` (= `D:/solar-sailer/benchmarks/roughcut/bench-page-arms.json`, same file via the junction).

Top level: `_comment`, `_fields` (a self-documenting field dictionary at L3-20), `current` (L21, 21 entries), `_legacy_comment`, `legacy` (L282, 20 entries), `baselines` (L504, 1 entry), `views` (L518).

A `current` entry has: `id`, `result_globs[]`, `model`, `effort_source` (optional), `workflow`, `harness`, `variant`, `variant_details`, `cost_basis`, `shipped`, plus optional `hidden` / `hidden_reason`. `legacy` entries add `date`; `baselines` entries add `baseline`, `result_file`, `variant_key`. `base_model` is derived, not stored. **There is no `notes` field anywhere in the file.**

Full example entry, verbatim from `bench-page-arms.json:22-36`:

```json
    {
      "id": "opus5-cc-agentic",
      "result_globs": [
        "2026-07-2*-partial-agentic-v1-claude-opus-5.json",
        "2026-07-25-*-partial-agentic-claude-opus-5.json"
      ],
      "model": "claude-opus-5-high",
      "effort_source": "cli-default-recorded",
      "workflow": "agentic",
      "harness": "Claude Code",
      "variant": "rules1",
      "variant_details": "Rules v1, the original partial-keep ruleset, paired with the v1 task brief: one session per episode rates the whole transcript, then reads its own partial ranges back and revises them. No chapter split and no extra tools. No effort flag was set; the run used the Claude Code CLI defaults, and its session logs record extended thinking with effort high on every message, so it publishes as -high.",
      "cost_basis": "estimated",
      "shipped": true
    },
```

### Fields the page reads and shows

Arm type declared at `rough-cut-bench.tsx:23-63`, normalization at L340-347, ladder row render at L763-812:

| Shown | Line |
| --- | --- |
| `model` (config name) | 765 |
| best / shipped / baseline chips | 766-768 |
| `variant` cell with `variant_details` as a hover/focus tooltip | 771-783 (tooltip set at 714-719) |
| `workflow` | 785 |
| `harness` | 786 |
| `weighted_grade` = SP (sentence points), `.toFixed(2)` plus a bar | 788, 746 |
| `frame_match` (column appears only when some non-baseline arm has one) | 797, gate at 673 |
| `cost_per_episode_hour_usd`, with an "est." superscript when `cost_basis === 'estimated'` | 747, 800-806 |
| `wall_minutes_total` (latency) | 748, 808 |
| `episodes_scored` | 810 |
| `date` | 811 |
| `grades` per episode, `views` (layered), `era`, `base_model` (family filter) | 287-294, 342, 1505-1508 |

Sort keys at L525-545: `score`→`weighted_grade`, `frame`→`frame_match`, `cost`→`cost_per_episode_hour_usd`, `time`→`wall_minutes_total`. Summary tiles at L1432-1449.

Published but never displayed: `word_score` (WORD) and `grade_v1` (the retired ceiling-divided GRADE), carried for continuity only (`rough-cut-bench.tsx:48-50`; exporter docstring L28-34). No `notes` field reaches the page.

### Adding an arm

Canonical procedure: `C:/Users/Stan/Documents/GitHub/solar-sailer/benchmarks/roughcut/README.md:204-216`.

1. Put the result JSONs under `benchmarks/roughcut/results/` first. A new episode needs a codename in `episode-codenames.json` before anything else (README:216).
2. Append an object to the `current` array in `benchmarks/roughcut/bench-page-arms.json` (array starts L21) with `id`, `result_globs`, `model` (including the effort suffix), `workflow`, `harness`, `variant`, `variant_details` (required, never empty), `cost_basis`, `shipped` (README:209).
3. From `benchmarks/roughcut/`, run `python scripts/model_plus_deterministic.py` to rebuild the layered "Deterministic modules applied" view. The exporter fails if the new arm is absent from it (README:210).
4. If the run predates 2026-09-11, backfill scores: `python scripts/backfill_sp_grades.py --arm <arm-id>` and `python scripts/backfill_word_grades.py --arm <arm-id>` (README:211). The exporter raises and names the missing one at `export_bench_page.py:666-687`.
5. Regenerate: `python benchmarks/roughcut/scripts/export_bench_page.py` (dry run with `--check`). This writes `website-docs/static/data/rough-cut-bench.json`.
6. Build and deploy **manually**. From `website-docs/DEPLOY.md:18-24`:

```
cd website-docs
npm install        # first time only
npm run build
node C:/Users/Stan/Documents/GitHub/claude-orchestrator/scripts/agent-firebase.mjs deploy --only hosting --project=solar-sailer-web
```

Firebase Hosting site `solar-sailer-docs` is pinned in `website-docs/firebase.json`. README:212 gives the human shorthand `npm run build && firebase deploy --only hosting`; the `agent-firebase.mjs` wrapper is required for agent sessions on Stan's PC (`DEPLOY.md:7-15`). `npm run build` runs the prebuild `compile-go-links.mjs` and `check-subscription-policy.mjs` plus a postbuild go-link check (`package.json:8-10`).

**No CI publishes this page.** `.github/workflows/` contains only `build-native.yml` and `linux-derisk.yml`; neither mentions `website-docs`, `rough-cut-bench`, or `export_bench_page`. The only automated route is the docs-cron pipeline, which merges to main and then runs the same manual build and deploy (`DEPLOY.md:28`), so committing and letting an imminent cron publish ride along is acceptable (README:212).
