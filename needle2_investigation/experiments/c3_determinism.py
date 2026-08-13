"""
Cycle 3 — Determinism & stability.

Question (not on the web): is Needle 2's .complete() deterministic? Across earlier
runs the same query gave slightly different confidence (Cairo 0.876 vs 0.868;
Berlin 0.849 vs 0.886). Is that real run-to-run nondeterminism, or an artifact of
different tool sets / call order? If nondeterministic, how big is the noise — and
does it ever flip the *decision* (which tool / whether it abstains)?

Tests:
  T1. Same engine instance, same query repeated N times -> vary?
  T2. Fresh engine each time, same query -> vary?
  T3. Does call order / preceding queries change a later query's result? (state leakage)
"""
import needle
from statistics import mean, pstdev

def tools():
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
    return [get_weather, get_stock_price]

def sig(r):
    """Compact signature of a result: (tool_name or '-', frozenset args)."""
    if not r["function_calls"]:
        return ("-", ())
    c = r["function_calls"][0]
    return (c["name"], tuple(sorted(c["arguments"].items())))

QUERIES = ["What's the weather in Cairo?", "How much is TSLA trading at?", "Play jazz"]
N = 12

def summarize(label, confs, sigs):
    uniq = set(sigs)
    cs = f"mean={mean(confs):.4f} sd={pstdev(confs):.4f} min={min(confs):.4f} max={max(confs):.4f}"
    dec = "DETERMINISTIC decision" if len(uniq) == 1 else f"DECISION VARIED across {len(uniq)} outcomes"
    det = "identical" if pstdev(confs) == 0 else "varies"
    print(f"{label}: conf {det}  [{cs}]  -> {dec}")
    return len(uniq)

def main():
    print(f"### T1 — same engine instance, query repeated N={N}")
    m = needle.Needle(tools=tools())
    for q in QUERIES:
        confs, sigs = [], []
        for _ in range(N):
            r = m.complete(q)
            confs.append(r["confidence"]); sigs.append(sig(r))
        summarize(f"  '{q}'", confs, sigs)

    print(f"\n### T2 — FRESH engine each call, query repeated N={N}")
    for q in QUERIES:
        confs, sigs = [], []
        for _ in range(N):
            m2 = needle.Needle(tools=tools())
            r = m2.complete(q)
            confs.append(r["confidence"]); sigs.append(sig(r))
        summarize(f"  '{q}'", confs, sigs)

    print(f"\n### T3 — state leakage: does a preceding query change a later one?")
    target = "What's the weather in Cairo?"
    # baseline: fresh engine, ask target first
    m3 = needle.Needle(tools=tools())
    base = m3.complete(target)
    # same engine, run distractors first, then target again
    m4 = needle.Needle(tools=tools())
    for d in ["How much is TSLA trading at?", "Play jazz", "weather in Oslo", "price of NVDA"]:
        m4.complete(d)
    after = m4.complete(target)
    print(f"  target fresh-first : conf={base['confidence']:.4f}  {sig(base)}")
    print(f"  target after 4 msgs: conf={after['confidence']:.4f}  {sig(after)}")
    print(f"  -> {'NO state leakage' if sig(base)==sig(after) and abs(base['confidence']-after['confidence'])<1e-9 else 'STATE LEAKAGE (result depends on history)'}")

    # also test explicit reset()
    m4.reset()
    resetd = m4.complete(target)
    print(f"  target after reset(): conf={resetd['confidence']:.4f}  {sig(resetd)}")

if __name__ == "__main__":
    main()
