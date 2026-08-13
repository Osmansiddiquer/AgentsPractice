# Needle 2 — Investigation

Hands-on investigation of **Needle 2**, run locally in this sandbox on 2026-08-13.

## What it is

**Needle 2** is a new agentic LLM released by **[Cactus Compute](https://cactuscompute.com/needle)** in 2026.
It's built for **tool/function calling, device use, and structured extraction** on edge
devices (phones, wearables, smart home, robots) — not for open-ended chat.

| Spec | Value |
|---|---|
| Parameters | ~45M (35M matmul-active) |
| On-disk size | single ~14MB binary |
| Peak session RAM | ~28MB (measured ~33–38MB here) |
| Quantization | CQ2-bit (2-bit), from pretraining onward |
| Architecture | "Simple Attention Network": 27 layers × 512-wide, Hadamard MLPs (Walsh transforms), engram n-gram memory, 256-token sliding window |
| Compute | ~70 MFLOPs/token |
| Speed (claimed) | 500+ tok/s on Raspberry Pi 5; 300–700 on sub-$200 phones; up to 1,500 on VR |
| Package | `pip install cactus-needle` (v2.0.2); weights: HF `Cactus-Compute/needle2` |

> Note: sources disagree on the license (the site says Apache 2.0, the GitHub README says MIT). Worth confirming before any downstream use.

## How I ran it

```bash
python -m venv .venv-needle
. .venv-needle/bin/activate
pip install cactus-needle pydantic
python needle2_investigation/demo.py   # first run downloads weights from HF
```

The Python API is small: `needle.Needle(tools=[...])`, `.complete()` (single-turn,
returns raw tool calls), `.run()` (multi-step agent loop), and `needle.extract(text, schema)`.
Tools are plain functions with Google-style docstrings, decorated with `@needle.tool` —
the JSON schema is derived automatically from the signature + docstring.

## What I observed (measured here, CPU x86_64)

**Init:** ~3.5s to load weights. **Latency:** each query ~0.04–0.6s. **Decode:** ~180–275 tok/s
on a plain cloud CPU. **Peak RAM:** ~33–38MB per session — matches the "tiny footprint" claim.

**1. Tool calling works, and confidence is well-calibrated.** A clean single-tool query
lands with high confidence; anything it can't map reliably comes back at very low confidence:

```
[EXECUTE ] conf=0.849  [get_weather(city='Berlin')]        <- What's the weather in Berlin?
[ESCALATE] conf=0.018  []                                  <- Turn on the office light
[ESCALATE] conf=0.008  [get_weather(city='Berlin')]        <- Compose a haiku about the sea
```

The confidence score is the important part: the one reliable call scored ~0.85–0.98 across
runs, while every wrong, abstained, or out-of-scope result scored **below 0.2** (often <0.02).
The intended pattern is **gate on a threshold** — execute high-confidence calls, escalate the
rest. When it mis-fires (e.g. it once emitted `set_timer` for a "turn on the light" request),
the confidence was ~0.03, so a threshold would have caught it.

**2. Structured extraction is strong.** Free-text → typed Pydantic model, exactly right:

```
"Hi, I'm Dr. Sarah Chen, Head of Robotics at Cactus Compute. Reach me at sarah.chen@cactus.ai."
-> name='Dr. Sarah Chen' company='Cactus Compute' email='sarah.chen@cactus.ai' role='Head of Robotics'
```

**3. It is genuinely small and honest about its limits.** At 45M params / 2-bit, raw
tool-selection accuracy on harder or multi-tool prompts is shaky (parallel-call prompts often
came back empty). The calibrated confidence is what makes it usable in practice — it knows when
it doesn't know, which for an on-device escalation-to-cloud design is the right tradeoff.

## Example runs (input → output)

Captured live in this sandbox. Four tools were declared: `get_weather(city)`,
`send_email(to, subject, body)`, `play_music(artist, song="")`, `get_stock_price(ticker)`.

### Tool calling

| Input | Output (function call) | Confidence |
|---|---|---|
| `How's the weather in Cairo?` | `get_weather(city='Cairo')` | **0.868** |
| `What's the weather like in New York City right now?` | *abstained* | 0.052 |
| `Play some Miles Davis` | `play_music(artist='Miles Davis')` | 0.086 |
| `Play the song Bohemian Rhapsody by Queen` | *abstained* | 0.005 |
| `Email jane@corp.com with subject Lunch and tell her I'll be 10 minutes late` | `send_email(to='jane@corp.com', subject='Lunch', body="I'll be 10 minutes late")` | 0.005 |
| `What's Apple's stock trading at?` | *abstained* | 0.003 |
| `How much is TSLA?` | `get_stock_price(ticker='TSLA')` | 0.013 |

Two patterns stand out. When it does fill arguments they're usually correct (the email
parsed into all three fields; `TSLA` extracted cleanly). And confidence tracks reliability
sharply — only the clean single-slot weather query scored high (0.87). Anything needing a
mapping (`Apple`→`AAPL`), multiple args (song + artist), or extra phrasing dropped near zero.
That low score is the model flagging "don't trust this — escalate."

### Structured extraction

```
IN : Team sync moved to Thursday at 2:30pm in the Blue Room
OUT: title='Team sync' day='Thursday' time='2:30pm' location='Blue Room'      (perfect)

IN : Can I get 3 large pepperoni pizzas please
OUT: item='large pepperoni pizzas' quantity=3 size='large pepperoni pizzas'   (quantity right,
     but 'size' wrongly copied the whole item instead of 'large')
```

Well-separated fields (the event) come out spot-on; when fields overlap semantically
(item vs. size both touching "large"), a 45M/2-bit model smears them.

## Takeaway

Needle 2 delivers on its pitch: a ~14MB, ~28MB-RAM tool-calling / extraction model that runs
fast on commodity CPU with no GPU. It is **not** a chat model — treat it as a cheap, local
first-pass router: high-confidence calls execute on-device, low-confidence ones escalate to a
bigger model. Structured extraction is the standout capability. Confidence gating is not
optional — it's the mechanism that makes a model this small safe to act on.

## Files
- `demo.py` — reproducible script covering tool calling + confidence gating + extraction.

## Sources
- [Needle 2 — Cactus Compute](https://cactuscompute.com/needle)
- [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle)
- HuggingFace: `Cactus-Compute/needle2`
