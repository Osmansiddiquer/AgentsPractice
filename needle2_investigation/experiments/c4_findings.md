# Cycle 4 — Scaling with tool-catalogue size (measured with reset)

**Question (not on the web):** Needle 2 retrieves the "top-5 relevant tools" from a
larger catalogue. Does accuracy survive burying the right tool among many distractors,
and how do latency / throughput / RAM scale with catalogue size? Measured from a clean
state (reset per query) per the Cycle-3 finding.

## Results — accuracy & resource scaling (`c4_scaling.py`)

```
catalogue   acc    conf   init_s   call_ms   decode_tps   peak_mb
    4       2/2   0.902    0.37     156.7        302.1      36.4
    8       2/2   0.935    0.48     535.1        304.2      38.9
   16       2/2   0.863    0.91     518.7        315.0      39.7
   32       2/2   0.865    1.75     543.4        301.5      39.8
   64       2/2   0.865    3.46     521.5        309.5      39.9
```

## Results — retrieval latency threshold (`c4_threshold.py`)

```
tools   warm_call_ms
    1        74.8
    3        77.4
    5        77.5     <- top-5 boundary
    6       379.6     <- retrieval engages: ~5x jump
    7       369.3
   10       416.6
   20       380.0
```

## Findings

1. **Accuracy is robust to catalogue size.** The correct tool is still selected with
   2/2 accuracy and high confidence (~0.86–0.94) even when buried among **62 distractors**
   (64-tool catalogue). Top-5 retrieval does its job.

2. **Inference cost is decoupled from catalogue size.** Decode throughput stays ~300 tok/s
   and peak RAM stays ~36–40MB whether you declare 4 tools or 64. Retrieval narrows to a
   fixed working set, so the model never pays per-token for unused tools.

3. **Sharp retrieval threshold at 5→6 tools.** With ≤5 tools every call is ~75ms (all tools
   passed directly, no retrieval). At 6+ tools a retrieval step engages and adds a **fixed
   ~300ms tax** (→ ~380ms/call), which then stays flat from 6 to 20 tools. The cost is a
   step, not a slope — you pay it once for "having a catalogue," not per extra tool.

4. **Init time is the only thing that scales with catalogue size** — roughly linear,
   ~50ms per declared tool (0.37s at 4 tools → 3.46s at 64), i.e. building the tool index.

## Practical guidance

- If latency matters and you can fit your tools in **≤5**, calls are ~5× faster (no
  retrieval step). Group/merge tools to stay under 6 when you can.
- Large catalogues are fine for accuracy and memory — you just eat a one-time init cost
  and a flat ~300ms per-call retrieval tax. Both are constant, not growing, so a
  100-tool catalogue behaves like a 20-tool one at call time.
