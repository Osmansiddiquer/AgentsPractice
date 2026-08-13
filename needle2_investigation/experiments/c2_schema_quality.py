"""
Cycle 2 — Can schema quality lift the calls (and confidence) that failed in Cycle 1?

Cycle 1 showed ~44% recall on paraphrased tool calls, with confidence collapsing
off the canonical phrasing. Here we hold the *inputs* fixed and vary only the
tool schema quality, then measure whether accuracy/confidence improve.

Three schema variants for the SAME four tools:
  A. MINIMAL  : bare one-line description, terse arg docs (Cycle-1 style)
  B. RICH     : verbose description + explicit "use this when..." + examples in docstring
  C. RICH+HINT: RICH plus argument descriptions that name synonyms/paraphrases

Same 16 answerable queries from Cycle 1. Report accuracy + mean/median confidence
per variant, and per-item confidence deltas.
"""
import needle
from statistics import mean, median

QUERIES = [
    ("What's the weather in Cairo?",        "get_weather",     "city",   "cairo"),
    ("weather in Tokyo",                    "get_weather",     "city",   "tokyo"),
    ("Is it raining in London?",            "get_weather",     "city",   "london"),
    ("How hot is it in Dubai right now?",   "get_weather",     "city",   "dubai"),
    ("What's the price of AAPL?",           "get_stock_price", "ticker", "aapl"),
    ("How much is TSLA trading at?",        "get_stock_price", "ticker", "tsla"),
    ("Look up NVDA stock",                  "get_stock_price", "ticker", "nvda"),
    ("Quote for MSFT please",               "get_stock_price", "ticker", "msft"),
    ("Play some Miles Davis",               "play_music",      "artist", "miles"),
    ("Put on The Beatles",                  "play_music",      "artist", "beatles"),
    ("I want to listen to Taylor Swift",    "play_music",      "artist", "taylor"),
    ("Play Radiohead",                      "play_music",      "artist", "radiohead"),
    ("Set a timer for 5 minutes",           "set_timer",       "minutes","5"),
    ("Give me a 10 minute timer",           "set_timer",       "minutes","10"),
    ("Timer, 2 minutes",                    "set_timer",       "minutes","2"),
    ("Start a 30 min countdown",            "set_timer",       "minutes","30"),
]

def grade(exp_tool, k, v, calls):
    if len(calls) != 1: return False
    c = calls[0]
    return c["name"] == exp_tool and v in str(c["arguments"].get(k, "")).lower()

# ---------- Variant A: MINIMAL ----------
def build_minimal():
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
    @needle.tool
    def set_timer(minutes: int):
        """Start a countdown timer.
        Args:
            minutes: Duration in minutes.
        """
    return [get_weather, get_stock_price, play_music, set_timer]

# ---------- Variant B: RICH ----------
def build_rich():
    @needle.tool
    def get_weather(city: str):
        """Get the current weather, temperature, and conditions for a city.
        Use this whenever the user asks about weather, temperature, rain, sun,
        how hot or cold it is, or forecast for a place. Example: "weather in Paris".
        Args:
            city: Name of the city.
        """
    @needle.tool
    def get_stock_price(ticker: str):
        """Look up the current stock market price for a ticker symbol.
        Use this whenever the user asks how much a stock is, its price, quote,
        or what it is trading at. Example: "price of AAPL".
        Args:
            ticker: The stock ticker symbol, e.g. AAPL.
        """
    @needle.tool
    def play_music(artist: str):
        """Start playing music by a given artist or band.
        Use this whenever the user wants to listen to, play, or put on music.
        Example: "play some Coldplay".
        Args:
            artist: The artist or band name.
        """
    @needle.tool
    def set_timer(minutes: int):
        """Start a countdown timer for a number of minutes.
        Use this whenever the user asks to set, start, or give them a timer or
        countdown. Example: "set a timer for 3 minutes".
        Args:
            minutes: Duration in minutes.
        """
    return [get_weather, get_stock_price, play_music, set_timer]

# ---------- Variant C: RICH + arg hints (synonyms in arg docs) ----------
def build_rich_hint():
    @needle.tool
    def get_weather(city: str):
        """Get the current weather, temperature, and conditions for a city.
        Use this for weather / temperature / rain / how hot or cold it is.
        Args:
            city: The place to check. Phrases like "in Tokyo", "raining in London",
                "hot in Dubai" all refer to this city.
        """
    @needle.tool
    def get_stock_price(ticker: str):
        """Look up the current stock price for a ticker symbol.
        Use this for stock price / quote / how much a stock is trading at.
        Args:
            ticker: The ticker symbol. "AAPL", "TSLA", "NVDA", "MSFT" are tickers;
                "look up X stock" and "quote for X" both provide the ticker.
        """
    @needle.tool
    def play_music(artist: str):
        """Start playing music by an artist or band.
        Use this for play / listen to / put on music.
        Args:
            artist: The artist or band. "put on The Beatles", "listen to Taylor Swift",
                "play Radiohead" all provide the artist.
        """
    @needle.tool
    def set_timer(minutes: int):
        """Start a countdown timer for a number of minutes.
        Use this for set/start/give me a timer or countdown.
        Args:
            minutes: Number of minutes. "10 minute timer", "timer, 2 minutes",
                "30 min countdown" all provide the minutes.
        """
    return [get_weather, get_stock_price, play_music, set_timer]

def run_variant(name, tools):
    m = needle.Needle(tools=tools)
    accs, confs, per = 0, [], []
    for q, tool, k, v in QUERIES:
        r = m.complete(q)
        ok = grade(tool, k, v, r["function_calls"])
        accs += ok
        confs.append(r["confidence"])
        per.append((q, ok, r["confidence"]))
    print(f"\n### Variant {name}")
    print(f"accuracy = {accs}/{len(QUERIES)} = {accs/len(QUERIES):.0%}"
          f"   conf mean={mean(confs):.3f} median={median(confs):.3f}")
    return {q: (ok, c) for q, ok, c in per}

def main():
    A = run_variant("A MINIMAL", build_minimal())
    B = run_variant("B RICH", build_rich())
    C = run_variant("C RICH+HINT", build_rich_hint())
    print("\n### Per-item accuracy (A -> B -> C) and confidence")
    for q, *_ in QUERIES:
        a, b, c = A[q], B[q], C[q]
        flags = f"{'O' if a[0] else 'x'}{'O' if b[0] else 'x'}{'O' if c[0] else 'x'}"
        print(f"  [{flags}]  {a[1]:.3f} -> {b[1]:.3f} -> {c[1]:.3f}   {q}")

if __name__ == "__main__":
    main()
