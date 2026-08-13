# Cycle 3 — Determinism & stability (this overturns Cycles 1 & 2)

**Question (not on the web):** is Needle 2's `.complete()` deterministic? Earlier runs
showed the same query giving different confidence (Cairo 0.876 vs 0.868), hinting at
nondeterminism that would add noise to every measurement.

## Results

```
T1 — same engine instance, query repeated 12x:
  'weather in Cairo?'      conf mean=0.455 sd=0.156  -> DECISION VARIED across 2 outcomes
  'How much is TSLA...'    conf mean=0.315 sd=0.035  -> DECISION VARIED across 3 outcomes
  'Play jazz'              conf mean=0.190 sd=0.094  -> DECISION VARIED across 2 outcomes

T2 — FRESH engine each call, query repeated 12x:
  'weather in Cairo?'      conf sd=0.0000 (always 0.8528)  -> DETERMINISTIC
  'How much is TSLA...'    conf sd=0.0000 (always 0.9855)  -> DETERMINISTIC
  'Play jazz'              conf sd=0.0000 (always 0.7456)  -> DETERMINISTIC

T3 — state leakage:
  'weather in Cairo?' fresh              -> conf 0.8528
  'weather in Cairo?' after 4 prior msgs -> conf 0.2826   (STATE LEAKAGE)
  'weather in Cairo?' after .reset()     -> conf 0.8528   (restored)
```

## The finding

**Needle 2's `.complete()` is stateful and fully deterministic — but only from a clean
state.** The engine keeps a running conversation in its 256-token sliding window and does
**not** auto-reset between calls. So:

- From a **fresh engine** (or after `.reset()`), output is *bit-identical* run to run
  (sd = 0.0000). There is no sampling noise at all.
- On a **shared engine without reset**, each query is contaminated by the accumulated
  history of previous queries. Confidence swings 0.23–0.85 for the *same* input, and the
  decision itself can flip. `.reset()` fully restores the clean-state result.

## Consequence: Cycles 1 and 2 were measuring a harness bug, not the model

Cycles 1 and 2 looped `m.complete(q)` over 20 queries on a single engine **without
resetting**. Re-running the exact Cycle-1 eval under three state regimes:

```
shared-noreset  (what Cycle 1 did) : acc 11/20 =  55%   conf median=0.010
shared-reset                       : acc 20/20 = 100%   conf median=0.889
fresh engine per query             : acc 20/20 = 100%   conf median=0.889
```

**Corrected conclusions:**

- Needle 2 scores **20/20 (100%)** on this eval — all 16 tool calls correct *and* all 4
  abstentions correct — when state is handled properly. The "~44% paraphrase ceiling"
  from Cycle 1 and the "capability ceiling, can't prompt past it" from Cycle 2 were
  **both artifacts of context contamination**, not real limits.
- **Confidence IS well-calibrated after all** — median 0.889 on correct calls from a clean
  state, vs the near-zero values I saw under contamination. My Cycle-1 claim that
  "confidence doesn't separate correct from incorrect" was itself caused by the bug.
- The earlier "phrasing-locked confidence" observation was really "position-in-conversation
  locked": the first query (Cairo) looked strong only because it ran before contamination
  built up.

## Practical guidance (the real usage rule)

Call `.reset()` between independent requests (or use a fresh `Needle` per request).
Only *keep* state when you genuinely want multi-turn context. This single detail is the
difference between 55% and 100% on the same inputs — and it is not something the docs or
web spell out; it took repeating identical inputs to surface it.
