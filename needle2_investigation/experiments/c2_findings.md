> ⚠️ **SUPERSEDED BY CYCLE 3.** The premise here — that the paraphrases "failed" — was
> wrong. They failed only because this harness reused one engine without `.reset()`,
> contaminating each query with prior context. With a clean state the model gets them all
> right, so there was nothing for schema tweaks to "rescue." The "45M/2-bit capability
> ceiling" conclusion below is an artifact. See `c3_findings.md`.

# Cycle 2 — Can schema quality rescue the failed paraphrases?

**Question (not on the web):** Cycle 1 failed ~56% of paraphrased tool calls. Does
improving the *tool schema* (verbose descriptions, "use this when...", examples,
synonym hints in arg docs) recover them — holding the inputs fixed?

**Method:** same 16 answerable queries, three schema variants for the same 4 tools:
A=minimal (Cycle-1 style), B=rich (verbose + examples), C=rich+arg-synonym-hints.

## Results

```
Variant A MINIMAL   : accuracy 7/16 = 44%   conf mean=0.077 median=0.008
Variant B RICH      : accuracy 8/16 = 50%   conf mean=0.074 median=0.017
Variant C RICH+HINT : accuracy 8/16 = 50%   conf mean=0.081 median=0.013
```

Per-item accuracy pattern (O=correct) A→B→C stayed identical for 15/16 items.
The only change: "What's the price of AAPL?" flipped x→O→O under richer schemas.
Every paraphrase that failed in Cycle 1 ("weather in Tokyo", "How hot is it in Dubai",
"How much is TSLA", "Put on The Beatles", "Play Radiohead", "Give me a 10 minute timer",
"Start a 30 min countdown") **stayed failing across all three variants.**

## Findings

1. **Prompt/schema engineering barely helps: +1 item (+6 pts), and no further gain
   from C over B.** The synonym hints in argument docs (variant C) added nothing over
   plain verbose descriptions.

2. **Confidence is nearly schema-invariant.** Median crept 0.008 → 0.017, mean flat
   (~0.077). Richer text does not make the model *more sure* of the calls it was already
   unsure about.

3. **Verbose schemas slightly *hurt* the one strong case.** The canonical
   "What's the weather in Cairo?" dropped 0.876 → 0.725 (B) / 0.799 (C). Adding
   description text dilutes the sharp signal on the phrasing the model already nails —
   a mild but real tradeoff.

4. **Conclusion: this is a capability ceiling, not a prompting problem.** At 45M params /
   2-bit, single-shot paraphrase robustness is ~44–50% and cannot be meaningfully lifted
   by better tool docs. If you need those paraphrases, the fix is architectural
   (retry/agent loop, few-shot, or escalate) — not wording. That directly sets up a
   later cycle: does the multi-step `.run()` loop recover any of these?
