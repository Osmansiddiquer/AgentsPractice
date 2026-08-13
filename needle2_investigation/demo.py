"""
Needle 2 investigation — reproducible demo.

Needle 2 (Cactus Compute) is a ~45M-parameter, 2-bit agentic LLM that ships as a
single ~14MB binary and runs a session in ~28MB RAM. It is built for tool/function
calling, device use, and structured extraction on edge devices.

Setup:
    python -m venv .venv-needle
    . .venv-needle/bin/activate
    pip install cactus-needle pydantic

Run:
    python needle2_investigation/demo.py

First run downloads the weights from HuggingFace (Cactus-Compute/needle2).
"""

import needle
import pydantic


# ---------------------------------------------------------------------------
# 1. Declare tools with Google-style docstrings — Needle builds the JSON schema
#    from the signature + docstring automatically.
# ---------------------------------------------------------------------------
@needle.tool
def get_weather(city: str):
    """Get the current weather for a city.

    Args:
        city: Name of the city.
    """
    return {"city": city, "temp_c": 21, "condition": "sunny"}


@needle.tool
def turn_on_light(room: str):
    """Turn on the smart light in a room.

    Args:
        room: The room name.
    """
    return {"room": room, "state": "on"}


def main():
    m = needle.Needle(tools=[get_weather, turn_on_light])

    # -----------------------------------------------------------------------
    # 2. Single-turn tool calling with calibrated confidence.
    #    Gate on confidence: execute high-confidence calls, escalate the rest.
    # -----------------------------------------------------------------------
    THRESHOLD = 0.5
    print("=== Tool calling + confidence gating ===")
    for q in [
        "What's the weather in Berlin?",
        "Turn on the office light",
        "Compose a haiku about the sea",  # out of scope -> should abstain
    ]:
        # IMPORTANT: .complete() is stateful (256-token sliding window). Reset
        # between independent requests or history contaminates later ones and
        # tanks accuracy/confidence. See experiments/c3_findings.md.
        m.reset()
        r = m.complete(q)
        do = "EXECUTE " if (r["confidence"] >= THRESHOLD and r["function_calls"]) else "ESCALATE"
        print(f"[{do}] conf={r['confidence']:.3f}  {r['function_calls']}  <- {q}")

    # -----------------------------------------------------------------------
    # 3. Structured extraction into a Pydantic model.
    # -----------------------------------------------------------------------
    print("\n=== Structured extraction ===")

    class Contact(pydantic.BaseModel):
        name: str
        company: str
        email: str
        role: str

    text = (
        "Hi, I'm Dr. Sarah Chen, Head of Robotics at Cactus Compute. "
        "Reach me at sarah.chen@cactus.ai."
    )
    print(needle.extract(text, Contact))


if __name__ == "__main__":
    main()
