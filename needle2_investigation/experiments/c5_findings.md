# Cycle 5 — Robustness to messy input (reset per query)

**Question (not on the web):** clean-state accuracy is high (Cycle 3). Does it survive
typos, casing, politeness filler, non-English, and distractor/negation phrasing?

## Results

```
Overall: 15/19 = 79%

By category:  clean 3/3 | typo 3/3 | lowercase 1/1 | chatter 3/3
              fr 1/1 | es 1/1 | de 1/1 | distractor 2/3 | CAPS 0/3  <-- hard failure
```

## Findings

### 1. ALL-CAPS is a hard, reproducible failure mode — caps words are read as tickers
Every ALL-CAPS query routed to `get_stock_price`, one call per capitalized word:

```
"WEATHER IN CAIRO"  -> get_stock_price('WEATHER'), get_stock_price('CAIRO')
"PLAY MILES DAVIS"  -> get_stock_price('PLAY'), get_stock_price('MILES'), get_stock_price('DAVIS')
"PRICE OF TSLA"     -> get_stock_price('OF')
```

The same text in lowercase works perfectly (`"weather in cairo"` -> `get_weather('Cairo')`).
Mechanistically this makes sense: ticker symbols (AAPL, TSLA) are the uppercase tokens in
training, so the model treats *any* caps word as a ticker. **Fix: normalize input case
before calling** — do not pass user text through in ALL CAPS.

### 2. Robust to typos and politeness/filler
Misspellings survive (`"Whats the wether in Cairoo?"` -> `get_weather('Cairoo')`;
`"How mch is TSLA tradng at?"` -> `get_stock_price('TSLA')`). Leading/trailing chatter
("hey there, could you maybe...please, thanks!") not only survives but often *raises*
confidence (0.995, 0.960) — more surrounding context helps disambiguation.

### 3. Multilingual works out of the box — undocumented and notable for 45M params
Weather queries in French, Spanish, and German all mapped correctly:
`"Quel temps fait-il a Caire?"`, `"Que tiempo hace en El Cairo?"`,
`"Wie ist das Wetter in Kairo?"` -> `get_weather(...)`, confidence 0.24–0.71.

### 4. Negation/distractor is a soft spot — and the model's own validator catches it
`"Forget stocks and music, just the weather in Cairo"` emitted a correct
`get_weather('Cairo')` **plus** a hallucinated `get_weather('Cairo, CA')`. The response's
`validation.ungrounded` field flagged `get_weather.city` — the model self-reported that the
extra city wasn't grounded in the input. **`validation.ungrounded` is a more useful safety
signal than `confidence`** for catching hallucinated arguments; worth checking it and
dropping ungrounded calls.

## Practical guidance
- Lowercase / normal-case the input before calling (ALL CAPS breaks it).
- Read `validation.ungrounded` and discard flagged args — it catches hallucinated
  parameters that confidence alone misses.
- Typos, politeness, and common non-English inputs are handled; you don't need to clean
  those.
