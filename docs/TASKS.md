# Active Tasks & Deferred Follow-ups

## Model registry follow-ups (from 2026-07-21 model-gap work)

- [ ] **Swap `kimi-k2.6` → Kimi K3** once the K3 open weights land on DeepInfra (Moonshot promised weights ~Jul 27, 2026). K3 is the current flagship; K2.6 was added as the best available substitute.
- [ ] **Remove or replace the dead Groq entries** `qwen3-32b` and `kimi-k2-0905` — Groq deprecated both (Jun 17 / Mar 23, 2026); calls fail on free/developer tiers. Removal needs Stan's sign-off. `qwen3.5-397b` and `kimi-k2.6` (both via DeepInfra) are the successors.
- [ ] **Add Gemini 3.5 Pro when it goes GA** — still in limited Vertex preview as of Jul 21, 2026; Google shipped 3.6 Flash instead and teased Gemini 4.
- [ ] **Consider DeepSeek first-party API** — DeepInfra's DeepSeek-V4-Pro is ~3× pricier than api.deepseek.com ($1.30/$2.60 vs ~$0.44/$0.87 per 1M). Needs a DEEPSEEK_API_KEY if we want first-party pricing.

## Benchmark notes

- MiniMax-M3 (4×) and Kimi-K2.6 (1×) time out on the largest ~30k-token clipping prompts via DeepInfra (see benchmark run 20260722_030307_4e84ca). Re-run those pairs if DeepInfra latency improves.
