# Needle 2 — 5-cycle empirical investigation (summary)

Five self-directed experiment cycles, each chosen to answer something the **web/docs
cannot** — only running the model reveals it. Every experiment runs from a clean state
and is reproducible via the `cN_*.py` scripts here.

| Cycle | Question | Headline result |
|------:|----------|-----------------|
| 1 | Does `confidence` predict correctness? | *(later found confounded)* — apparent 55% acc, confidence looked uninformative |
| 2 | Can better tool schemas rescue failures? | *(later found confounded)* — schema tweaks barely moved anything |
| 3 | Is `.complete()` deterministic? | **The key finding.** Deterministic from a clean state (sd=0), but **stateful** — no auto-reset. Cycles 1–2 were measuring context contamination. With `.reset()`: **100% acc, median conf 0.889** |
| 4 | How does it scale with tool count? | Accuracy robust to 64 tools; cost decoupled from catalogue size; sharp **top-5 retrieval threshold** (≤5 tools ~75ms/call, ≥6 ~380ms flat) |
| 5 | Does it survive messy input? | 79% overall; robust to typos/filler/**multilingual**; **ALL-CAPS is a hard failure** (caps words read as tickers); `validation.ungrounded` catches hallucinated args |

## The one thing to remember

**`Needle.complete()` is stateful (256-token sliding window) and does not reset between
calls.** Call `.reset()` between independent requests. This single detail moved the same
eval from **55% → 100%** and was invisible until identical inputs were repeated (Cycle 3).
It is the difference between concluding "45M/2-bit model is weak" (wrong) and "this model
is strong when driven correctly" (right).

## Corrected picture of Needle 2

- **Genuinely accurate** at tool calling + structured extraction from a clean state
  (100% on a 20-item eval), and **perfectly deterministic** (no sampling noise).
- **Well-calibrated confidence** after all — high (≈0.89) on correct calls from a clean
  state. Gate high for precision.
- **Scales cleanly**: large tool catalogues cost only a one-time init and a flat ~300ms
  retrieval tax; per-token speed (~300 tps) and RAM (~40MB) stay constant.
- **Robust** to typos, politeness, and several non-English languages; **not** robust to
  ALL-CAPS input. Use `validation.ungrounded` (not just `confidence`) to catch bad args.

## Operating checklist (derived empirically)
1. `reset()` between independent requests (or use a fresh engine).
2. Normalize input case — never pass ALL CAPS.
3. Keep the tool catalogue ≤5 when latency matters (avoids the retrieval tax).
4. Gate on `confidence` for precision; also drop calls flagged in `validation.ungrounded`.
5. Expect deterministic output — cache/verify freely, there's no run-to-run noise.

## Reproduce
```bash
python -m venv .venv-needle && . .venv-needle/bin/activate
pip install cactus-needle pydantic
for f in c1_calibration c2_schema_quality c3_determinism c4_scaling c4_threshold c5_robustness; do
  python needle2_investigation/experiments/$f.py
done
```
