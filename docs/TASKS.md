# Active Tasks & Deferred Follow-ups

## Model registry follow-ups (from 2026-07-21 model-gap work)

- [x] **Swap `kimi-k2.6` → Kimi K3** — done 2026-07-22 (v3.16.0): Moonshot's K3 API launched early (Jul 16), so `kimi-k3` now routes first-party via `api.moonshot.ai` (MOONSHOT_API_KEY, $3/$15 per 1M) and the `kimi-k2.6` DeepInfra stand-in was removed.
- [ ] **Re-evaluate kimi-k3 hosting** once the K3 open weights land on DeepInfra (promised ~Jul 27, 2026) — DeepInfra may undercut Moonshot's $3/$15 first-party pricing, but check latency first (DeepInfra ran Kimi-K2.6 at ~3.5 min/answer).
- [ ] **Remove or replace the dead Groq entries** `qwen3-32b` and `kimi-k2-0905` — Groq deprecated both (Jun 17 / Mar 23, 2026); calls fail on free/developer tiers. Removal needs Stan's sign-off. `qwen3.5-397b` (DeepInfra) and `kimi-k3` (Moonshot first-party) are the successors.
- [ ] **Add Gemini 3.5 Pro when it goes GA** — still in limited Vertex preview as of Jul 21, 2026; Google shipped 3.6 Flash instead and teased Gemini 4.
- [ ] **Consider DeepSeek first-party API** — DeepInfra's DeepSeek-V4-Pro is ~3× pricier than api.deepseek.com ($1.30/$2.60 vs ~$0.44/$0.87 per 1M). Needs a DEEPSEEK_API_KEY if we want first-party pricing.

## Benchmark notes

- MiniMax-M3 (4×) and Kimi-K2.6 (1×) time out on the largest ~30k-token clipping prompts via DeepInfra (see benchmark run 20260722_030307_4e84ca). Re-run those pairs if DeepInfra latency improves.
- Kimi-K3 (Moonshot first-party) also failed 2 of those clipping prompts (`clipper_v30`/`v31`, run 20260723_013149_c03f00) after ~90 min of retries at reasoning high; the ones that succeeded took 20–90 min each. Re-run those two if Moonshot capacity improves, or try reasoning low for that prompt family.
