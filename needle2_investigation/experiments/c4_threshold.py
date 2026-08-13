"""
Cycle 4 (addendum) — locate the tool-retrieval latency threshold.

Measures warm per-call latency (5 calls, drop the first, mean of the next 4;
reset between calls) as catalogue size crosses the documented top-5 boundary.
"""
import needle, time
from statistics import mean

@needle.tool
def get_weather(city: str):
    """Get the current weather for a city.
    Args:
        city: Name of the city.
    """

def make_tool(i):
    def fn(query: str):
        return {}
    fn.__name__ = f"distractor_{i}"
    fn.__doc__ = f"Domain action {i}.\n    Args:\n        query: input {i}.\n"
    return needle.tool(fn)

def main():
    print(f"{'tools':>5} {'warm_call_ms':>12}")
    for size in [1, 3, 5, 6, 7, 10, 20]:
        cat = [get_weather] + [make_tool(i) for i in range(size - 1)]
        m = needle.Needle(tools=cat)
        lat = []
        for j in range(5):
            m.reset()
            t = time.time()
            m.complete("What's the weather in Cairo?")
            lat.append((time.time() - t) * 1000)
        print(f"{size:>5} {mean(lat[1:]):>12.1f}")

if __name__ == "__main__":
    main()
