"""
Cycle 1 — Confidence calibration study.

Question (not answerable from docs): does Needle 2's reported `confidence`
actually predict whether the emitted tool call is *correct*? If yes, what
threshold best separates good calls from bad ones?

We build a labeled eval set (query -> expected tool + key arg, or None for
"should abstain"), run each through .complete(), grade it, and report:
  - overall raw accuracy (no gating)
  - confidence distribution for correct vs incorrect decisions
  - accuracy/precision/recall of an "execute if confidence >= T" policy sweep
"""
import needle

# ---- tools ----
@needle.tool
def get_weather(city: str):
    """Get the current weather for a city.
    Args:
        city: Name of the city.
    """
    return {}

@needle.tool
def get_stock_price(ticker: str):
    """Look up the current stock price for a ticker symbol.
    Args:
        ticker: The stock ticker symbol, e.g. AAPL.
    """
    return {}

@needle.tool
def play_music(artist: str):
    """Play music by an artist.
    Args:
        artist: The artist name.
    """
    return {}

@needle.tool
def set_timer(minutes: int):
    """Start a countdown timer.
    Args:
        minutes: Duration in minutes.
    """
    return {}

TOOLS = [get_weather, get_stock_price, play_music, set_timer]

# ---- labeled eval set: (query, expected_tool_or_None, expected_key, expected_val_substr) ----
EVAL = [
    ("What's the weather in Cairo?",              "get_weather",     "city", "cairo"),
    ("weather in Tokyo",                          "get_weather",     "city", "tokyo"),
    ("Is it raining in London?",                  "get_weather",     "city", "london"),
    ("How hot is it in Dubai right now?",         "get_weather",     "city", "dubai"),
    ("What's the price of AAPL?",                 "get_stock_price", "ticker", "aapl"),
    ("How much is TSLA trading at?",              "get_stock_price", "ticker", "tsla"),
    ("Look up NVDA stock",                        "get_stock_price", "ticker", "nvda"),
    ("Quote for MSFT please",                     "get_stock_price", "ticker", "msft"),
    ("Play some Miles Davis",                     "play_music",      "artist", "miles"),
    ("Put on The Beatles",                        "play_music",      "artist", "beatles"),
    ("I want to listen to Taylor Swift",          "play_music",      "artist", "taylor"),
    ("Play Radiohead",                            "play_music",      "artist", "radiohead"),
    ("Set a timer for 5 minutes",                 "set_timer",       "minutes", "5"),
    ("Give me a 10 minute timer",                 "set_timer",       "minutes", "10"),
    ("Timer, 2 minutes",                          "set_timer",       "minutes", "2"),
    ("Start a 30 min countdown",                  "set_timer",       "minutes", "30"),
    # ---- should ABSTAIN (no matching tool) ----
    ("What's the meaning of life?",               None, None, None),
    ("Write me a poem about autumn",              None, None, None),
    ("Translate hello into French",              None, None, None),
    ("What's 17 times 23?",                       None, None, None),
]

def grade(expected_tool, exp_key, exp_val, calls):
    """Return True if the model's output matches the label."""
    if expected_tool is None:
        return len(calls) == 0
    if len(calls) != 1:
        return False
    c = calls[0]
    if c["name"] != expected_tool:
        return False
    val = str(c["arguments"].get(exp_key, "")).lower()
    return exp_val in val

def main():
    m = needle.Needle(tools=TOOLS)
    rows = []
    for q, tool, k, v in EVAL:
        r = m.complete(q)
        ok = grade(tool, k, v, r["function_calls"])
        rows.append((q, ok, r["confidence"], r["function_calls"]))

    correct = [c for _, ok, c, _ in rows if ok]
    wrong   = [c for _, ok, c, _ in rows if not ok]
    n = len(rows)
    print(f"Raw accuracy (no gating): {len(correct)}/{n} = {len(correct)/n:.0%}")
    if correct:
        print(f"  confidence when CORRECT : min={min(correct):.3f} mean={sum(correct)/len(correct):.3f} max={max(correct):.3f}")
    if wrong:
        print(f"  confidence when WRONG   : min={min(wrong):.3f} mean={sum(wrong)/len(wrong):.3f} max={max(wrong):.3f}")

    print("\nGating policy sweep (execute a call only if confidence >= T):")
    print(f"{'T':>5} {'exec':>5} {'exec_correct':>12} {'precision':>10} {'recall_of_answerable':>20}")
    answerable = sum(1 for _, t, _, _ in EVAL if t is not None)
    for T in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        executed = [(q, ok, c, calls) for q, ok, c, calls in rows if calls and c >= T]
        exec_correct = sum(1 for _, ok, _, _ in executed if ok)
        prec = exec_correct / len(executed) if executed else 0.0
        rec = exec_correct / answerable
        print(f"{T:>5.2f} {len(executed):>5} {exec_correct:>12} {prec:>10.2f} {rec:>20.2f}")

    print("\nPer-item:")
    for q, ok, c, calls in rows:
        mark = "OK " if ok else "XX "
        print(f"  {mark} conf={c:.3f}  {q}")

if __name__ == "__main__":
    main()
