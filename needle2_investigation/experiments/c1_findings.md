# Cycle 1 — Confidence calibration study

**Question (not on the web):** does Needle 2's reported `confidence` actually
predict whether the emitted tool call is correct, and what gating threshold is best?

**Method:** 20 labeled queries (16 answerable across 4 tools + 4 that should abstain),
run through `.complete()`, each graded against expected tool + key argument.

## Results

```
Raw accuracy (no gating): 11/20 = 55%
  confidence when CORRECT : min=0.005 mean=0.103 max=0.876
  confidence when WRONG   : min=0.002 mean=0.021 max=0.134

Gating (execute call iff confidence >= T):
    T   executed  correct  precision  recall(of answerable)
 0.00       8        7       0.88        0.44
 0.05       2        2       1.00        0.12
 0.20       1        1       1.00        0.06
 0.50       1        1       1.00        0.06
```

## Findings — and a correction to the earlier writeup

1. **Confidence is NOT a clean correctness signal.** The mean confidence is higher for
   correct calls (0.103 vs 0.021), but that gap is carried almost entirely by a single
   item — "What's the weather in Cairo?" at 0.876. Strip that one and the distributions
   overlap almost completely: plenty of *correct* calls sit at 0.005–0.015, the same band
   as wrong calls and even abstentions (0.013–0.037).

2. **High confidence is phrasing-locked, not task-locked.** The weather tool only fired
   at high confidence for the exact template *"What's the weather in <City>?"* (0.876).
   Paraphrases collapsed it: "weather in Tokyo" 0.134, "Is it raining in London?" 0.116,
   "How hot is it in Dubai right now?" 0.010. Same tool, same intent — confidence swings 80×.

3. **My earlier claim was a small-sample artifact.** In the first pass I saw Berlin/Cairo
   weather at ~0.85 and concluded "confidence tracks reliability sharply." A 20-item set
   shows that only holds for one canonical phrasing. **Corrected takeaway:** confidence
   gating buys you near-perfect *precision* (1.00 at T≥0.05) but brutal *recall* — you'd
   execute only 2 of 16 answerable queries. It's a "fire only when almost certain" knob,
   not a general reliability score.

4. **Abstention is the model's real strength.** All 4 out-of-scope queries correctly
   produced no call. The failures are on *answerable* queries it silently drops (44% recall).

**Net:** raw single-shot accuracy on paraphrased tool calls is ~44%, and confidence can't
rescue the dropped ones because they look identical to noise. This motivates Cycle 2:
can richer tool schemas (enums, examples, better descriptions) lift both the calls AND
their confidence on the exact paraphrases that failed here?
