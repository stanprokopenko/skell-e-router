# Cache marker review

Developer reference. The review found and fixed one bug: the LiteLLM route changed cache settings in the caller's saved conversation. Version 3.43.2 applies those adjustments to a private request copy. The final blind review found no remaining issues in the reviewed scope.

## Scope and finding

Reviewed commit `81ec817`, version 3.43.1, against skell-e-agent's `docs/TASKS.md` section "Tool-message cache marker contract, 2026-11-06". The scope covered tool-result block-list conversion, message-level and inner cache marker hoisting, caller marker counting, and TTL ordering on the direct and LiteLLM routes.

Two independent gpt-6-astra reviewers performed round one. The tool-result review was clean. The TTL review found a P2 caller-history mutation in `utils._apply_cache_control_litellm`: `_construct_messages` retains caller-owned message objects, and `_raise_ttls_before_1h` replaced their marker dictionaries. Reusing history after removing a later one-hour marker still requested hour-long caching. The existing test protected a separately retained marker dictionary but missed replacement inside its parent message.

The fix copies the messages before adding or normalizing cache settings. This also prevents the existing default-breakpoint placement from changing caller history. Public arguments and returned request content remain compatible. No direct-path behavior changed.

Anthropic's [prompt caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) defines the ordering of mixed cache durations. This patch retains the existing policy of raising earlier five-minute markers before a later one-hour marker.

## Verification

- Five new regression cases fail against the original helper extracted from `81ec817` and pass against the fix. Four exercise `ask_ai` with reused histories, covering tool-message and content-block markers with omitted or explicit five-minute durations. The fifth checks default system and trailing breakpoints.
- Final suite: 1,164 tests and 17 subtests passed. Anthropic subset: 120 tests passed. Commands used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, as required by the documented local plugin issue.
- The agent's existing `spikes/cache_control_passthrough.py all` passed against this worktree through `PYTHONPATH`, without changing installed packages. Defaults and caller placement passed offline. The live request wrote 5,295 tokens to the one-hour cache, with 25 uncached input tokens. The response text was empty, so this probe establishes cache passthrough, not answer correctness.
- Separate live tool exchanges through `ask_ai`, with `direct_sdk=True` and `False`, each returned the expected code `ORCHID42`. Each included an earlier five-minute marker and a message-level one-hour tool marker. Both preserved the original conversation and reported 414 prompt tokens and 8 output tokens.
- A fresh blind gpt-6-astra reviewer checked the combined change in round two and found no concrete failures. The loop stopped after that clean round.
- Estimated API spend was $0.054, based on returned usage and configured prices. No consumer installations or capability floors changed.

Version 3.43.2 contains the behavior fix, regression tests, and documentation. No follow-up findings remain from this review.
