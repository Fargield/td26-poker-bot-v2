"""Generate runnable league bot files (league/bots/<kind>.py) from the baseline template.

Usage:  python league/make_bots.py
Each generated bot is a full server (same plumbing as baseline-bots/) whose
choose_action delegates to league/strategies.py. Port comes from $PORT.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(ROOT, "baseline-bots", "baseline_calling_station.py")
OUT = os.path.join(HERE, "bots")

START, END = "# ===== YOUR CODE HERE =====", "# ===== END OF YOUR CODE ====="

BODY = '''# ===== YOUR CODE HERE =====
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from league.strategies import KINDS
_ACT = KINDS["{kind}"]


def new_half(config: AgentConfig, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    return _ACT(obs, memory)


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass

# ===== END OF YOUR CODE ====='''


GENBODY = '''# ===== YOUR CODE HERE =====
import json as _json
from league.genome import make_act
_G = _json.loads(os.environ.get("BOT_GENES", "{}")) or None
_ACT = make_act(_G) if _G else None


def new_half(config: AgentConfig, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    return _ACT(obs, memory)


def on_hand_end(result: HandResult, memory: dict) -> None:
    if _G and _G.get("tilt"):
        if result.my_net <= -30:
            memory["tilt_until"] = result.hand_index + 15

# ===== END OF YOUR CODE ====='''


def gen_template(text):
    body = GENBODY
    out = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: body, text, flags=re.S)
    out = out.replace('sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
                      'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))', 1)
    # genome.py lives in league/, imported as league.genome; make sure the repo root is importable
    return out.replace('AGENT_NAME = "CallingStation"', 'AGENT_NAME = "GenBot"')


def main():
    with open(os.path.join(OUT, "genbot.py"), "w", encoding="utf-8") as f:
        os.makedirs(OUT, exist_ok=True)
        f.write(gen_template(open(TEMPLATE, encoding="utf-8").read()))
    print("wrote genbot")
    sys.path.insert(0, ROOT)
    from league.strategies import KINDS
    text = open(TEMPLATE, encoding="utf-8").read()
    os.makedirs(OUT, exist_ok=True)
    for kind in KINDS:
        body = BODY.replace("{kind}", kind)
        # this file lives in league/bots/, so the root is two levels up
        body = body.replace("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
                            "os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
        out = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: body, text, flags=re.S)
        out = out.replace('sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
                          'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))', 1)
        out = out.replace('AGENT_NAME = "CallingStation"', f'AGENT_NAME = "{kind}"')
        with open(os.path.join(OUT, f"{kind}.py"), "w", encoding="utf-8") as f:
            f.write(out)
        print("wrote", kind)


import sys  # noqa: E402

if __name__ == "__main__":
    main()
