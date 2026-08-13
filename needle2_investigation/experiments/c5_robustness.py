"""
Cycle 5 — Robustness to messy input (measured from a clean state, reset per query).

Cycle 3 established the true clean-state accuracy is high. Does that survive real-world
noise? We take canonical queries that work, then perturb them and check whether the tool
call + key argument still come out right.

Perturbations:
  - clean (baseline)
  - typos (misspellings)
  - casing (ALL CAPS / no caps)
  - leading/trailing chatter (politeness, filler)
  - non-English (French, Spanish, German)
  - distractor clause (mention another domain that should NOT trigger)
"""
import needle

@needle.tool
def get_weather(city: str):
    """Get the current weather for a city.
    Args:
        city: Name of the city.
    """
@needle.tool
def get_stock_price(ticker: str):
    """Look up the current stock price for a ticker symbol.
    Args:
        ticker: The stock ticker symbol, e.g. AAPL.
    """
@needle.tool
def play_music(artist: str):
    """Play music by an artist.
    Args:
        artist: The artist name.
    """

TOOLS = [get_weather, get_stock_price, play_music]

# Each row: (category, query, expected_tool, key, expected_val_substr)
CASES = [
    ("clean",        "What's the weather in Cairo?",                 "get_weather", "city", "cairo"),
    ("typo",         "Whats the wether in Cairoo?",                  "get_weather", "city", "cair"),
    ("caps",         "WHAT IS THE WEATHER IN CAIRO",                 "get_weather", "city", "cairo"),
    ("lowercase",    "what is the weather in cairo",                 "get_weather", "city", "cairo"),
    ("chatter",      "hey there, could you maybe tell me the weather in Cairo please, thanks!", "get_weather", "city", "cairo"),
    ("fr",           "Quel temps fait-il a Caire?",                  "get_weather", "city", "cair"),
    ("es",           "Que tiempo hace en El Cairo?",                 "get_weather", "city", "cairo"),
    ("de",           "Wie ist das Wetter in Kairo?",                 "get_weather", "city", "kairo"),
    ("distractor",   "Forget stocks and music, just the weather in Cairo", "get_weather", "city", "cairo"),

    ("clean",        "How much is TSLA trading at?",                 "get_stock_price", "ticker", "tsla"),
    ("typo",         "How mch is TSLA tradng at?",                   "get_stock_price", "ticker", "tsla"),
    ("caps",         "PRICE OF TSLA",                                "get_stock_price", "ticker", "tsla"),
    ("chatter",      "sorry to bother you, what's TSLA trading at right now?", "get_stock_price", "ticker", "tsla"),
    ("distractor",   "not the weather, I want the TSLA stock price", "get_stock_price", "ticker", "tsla"),

    ("clean",        "Play some Miles Davis",                        "play_music", "artist", "miles"),
    ("typo",         "Ply some Myles Davis",                         "play_music", "artist", "davis"),
    ("caps",         "PLAY MILES DAVIS",                             "play_music", "artist", "miles"),
    ("chatter",      "if you don't mind, please play some Miles Davis for me", "play_music", "artist", "miles"),
    ("distractor",   "no weather please, just play Miles Davis",     "play_music", "artist", "miles"),
]

def grade(t, k, v, calls):
    return len(calls) == 1 and calls[0]["name"] == t and v in str(calls[0]["arguments"].get(k, "")).lower()

def main():
    m = needle.Needle(tools=TOOLS)
    from collections import defaultdict
    bycat = defaultdict(lambda: [0, 0])
    rows = []
    for cat, q, t, k, v in CASES:
        m.reset()
        r = m.complete(q)
        ok = grade(t, k, v, r["function_calls"])
        bycat[cat][0] += ok; bycat[cat][1] += 1
        got = r["function_calls"][0] if r["function_calls"] else None
        rows.append((cat, ok, r["confidence"], q, got))

    total_ok = sum(ok for _, ok, _, _, _ in rows)
    print(f"Overall robustness: {total_ok}/{len(rows)} = {total_ok/len(rows):.0%}\n")
    print("By category:")
    for cat in ["clean","typo","caps","lowercase","chatter","fr","es","de","distractor"]:
        if cat in bycat:
            ok, n = bycat[cat]
            print(f"  {cat:11} {ok}/{n}")
    print("\nPer-item:")
    for cat, ok, c, q, got in rows:
        mark = "OK " if ok else "XX "
        g = f"{got['name']}({got['arguments']})" if got else "(abstained)"
        print(f"  {mark} [{cat:10}] conf={c:.3f}  {q}\n         -> {g}")

if __name__ == "__main__":
    main()
