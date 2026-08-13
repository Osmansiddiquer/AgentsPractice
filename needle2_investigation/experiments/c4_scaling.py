"""
Cycle 4 — Scaling with tool count (measured correctly, with reset).

Needle 2 is documented to retrieve the top-5 relevant tools from a larger catalogue.
Questions (not on the web):
  - Does accuracy on a fixed target query hold as we bury its correct tool among
    N distractor tools (N = 4, 8, 16, 32, 64)?
  - How do latency / decode_tps / peak_ram scale with the declared catalogue size?
All queries run from a CLEAN state (reset per query) per the Cycle-3 finding.
"""
import needle, time
from statistics import mean

def make_tool(i):
    """Build a distinct distractor tool with index i."""
    def fn(query: str):
        return {}
    fn.__name__ = f"distractor_{i}"
    fn.__doc__ = (f"Perform domain action number {i} on a query.\n"
                  f"    Args:\n        query: The input for action {i}.\n")
    return needle.tool(fn)

# The real target tool + a fixed probe that should map to it.
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

PROBES = [
    ("What's the weather in Cairo?",   "get_weather",     "city",   "cairo"),
    ("How much is TSLA trading at?",   "get_stock_price", "ticker", "tsla"),
]

def grade(t, k, v, calls):
    return len(calls) == 1 and calls[0]["name"] == t and v in str(calls[0]["arguments"].get(k, "")).lower()

def main():
    print(f"{'catalogue':>9} {'acc':>7} {'conf':>7} {'init_s':>7} {'call_ms':>8} {'decode_tps':>11} {'peak_mb':>8}")
    for n_distractors in [2, 6, 14, 30, 62]:
        real = [get_weather, get_stock_price]
        catalogue = real + [make_tool(i) for i in range(n_distractors)]
        size = len(catalogue)

        t0 = time.time()
        m = needle.Needle(tools=catalogue)
        init_s = time.time() - t0

        ok, confs, lat, tps, ram = 0, [], [], [], []
        for q, t, k, v in PROBES:
            m.reset()
            t1 = time.time()
            r = m.complete(q)
            lat.append((time.time() - t1) * 1000)
            ok += grade(t, k, v, r["function_calls"])
            confs.append(r["confidence"]); tps.append(r["decode_tps"]); ram.append(r["peak_ram_mb"])
        print(f"{size:>9} {ok}/{len(PROBES):>5} {mean(confs):>7.3f} {init_s:>7.2f} "
              f"{mean(lat):>8.1f} {mean(tps):>11.1f} {max(ram):>8.1f}")

if __name__ == "__main__":
    main()
