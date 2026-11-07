"""Random bot genomes: a wide space of poker styles, played by one parametric bot (league/genbot.py).

  python league/genome.py --sample 20     # print 20 genomes and fuzz-test each
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equity as E
from game import Action

FOLD, CALL, HALF, POT = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT


def random_genome(rng: random.Random) -> dict:
    reader = rng.choice(["equity", "equity", "equity", "pf_only", "blind"])
    g = {
        "reader": reader,
        "p_random": rng.choice([0.0, 0.0, 0.05, 0.15, 0.3]),
        "blind_mode": rng.choice(["raise", "call", "mixed", "check_fold"]),
        "blind_raise_p": rng.uniform(0.2, 1.0),
        "samples": rng.choice([100, 300]),
        # thresholds per phase: preflop / postflop
        "bet_pf": rng.uniform(0.45, 0.70), "bet_post": rng.uniform(0.45, 0.72),
        "raise_pf": rng.uniform(0.55, 0.78), "raise_post": rng.uniform(0.55, 0.80),
        "call_slack": rng.uniform(-0.08, 0.10),
        "pot_odds_aware": rng.random() < 0.8,
        "bluff": rng.choice([0.0, 0.0, 0.1, 0.25, 0.4]),
        "slowplay": rng.choice([0.0, 0.0, 0.2, 0.4]),
        "pot_pref": rng.uniform(0.0, 1.0),          # 0 = always half, 1 = always pot
        "ip_shift": rng.uniform(-0.05, 0.05),       # threshold shift when on the button
        "adaptive": rng.random() < 0.35,
        "tilt": rng.random() < 0.2,
        "seed": rng.randrange(1 << 30),
    }
    return g


def _pick(legal, *prefs):
    for a in prefs:
        if a in legal:
            return a
    return CALL if CALL in legal else legal[0]


def make_act(g: dict):
    def act(obs, memory):
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        r = memory.get("rng")
        if r is None:
            r = memory["rng"] = random.Random(g["seed"])
            memory["st"] = {"faced": 0, "folds": 0, "lost": 0.0}
        st = memory["st"]
        pre = len(obs.board) == 0
        facing = obs.to_call > 0
        # opponent counters (adaptive): crude fold-rate / aggression tracking
        bluff, slack = g["bluff"], g["call_slack"]
        if g["adaptive"]:
            hist = obs.action_history
            opp = [a for a in hist if not a.is_me]
            n_raise = sum(1 for a in opp if a.action in (HALF, POT))
            memory["opp_n"] = memory.get("opp_n", 0) + len(opp)
            memory["opp_r"] = memory.get("opp_r", 0) + n_raise
            if memory["opp_n"] > 60:
                agg = memory["opp_r"] / memory["opp_n"]
                if agg > 0.45:
                    slack += 0.05          # call lighter vs aggressive opponents
                if agg < 0.15:
                    bluff = min(0.5, bluff + 0.15)   # steal vs passive ones
        if g["tilt"] and memory.get("tilt_until", -1) >= obs.hand_index:
            slack += 0.06
            bluff = min(0.6, bluff + 0.2)
        if r.random() < g["p_random"]:
            return r.choice(list(legal))
        rd = g["reader"]
        if rd == "blind":
            m = g["blind_mode"]
            if m == "check_fold":
                return CALL if not facing else _pick(legal, FOLD)
            if m == "call":
                return CALL
            if m == "raise" or (m == "mixed" and r.random() < g["blind_raise_p"]):
                return _pick(legal, POT if r.random() < g["pot_pref"] else HALF, HALF, POT)
            return CALL
        if rd == "pf_only":
            e = E.preflop_strength(tuple(obs.hole_cards))
        elif pre:
            e = E.preflop_strength(tuple(obs.hole_cards))
        else:
            e = E.equity(tuple(obs.hole_cards), tuple(obs.board), samples=g["samples"])
        shift = g["ip_shift"] if obs.is_button else -g["ip_shift"]
        e = e + shift
        bet_t = g["bet_pf"] if pre else g["bet_post"]
        rai_t = g["raise_pf"] if pre else g["raise_post"]
        big = r.random() < g["pot_pref"]
        if facing:
            odds = obs.to_call / (obs.pot + obs.to_call)
            if e >= rai_t and not (g["slowplay"] and r.random() < g["slowplay"] and e >= 0.8):
                return _pick(legal, POT if big or e >= rai_t + 0.1 else HALF, HALF)
            need = odds - slack if g["pot_odds_aware"] else 0.5 - slack
            if e >= need:
                return CALL
            if bluff and r.random() < bluff * 0.3:
                return _pick(legal, HALF, CALL)
            return _pick(legal, FOLD)
        if e >= bet_t and not (g["slowplay"] and r.random() < g["slowplay"] and e >= 0.8):
            return _pick(legal, POT if big or e >= bet_t + 0.1 else HALF, HALF)
        if bluff and r.random() < bluff:
            return _pick(legal, HALF, POT)
        return CALL
    return act


def fuzz(g: dict, n: int = 300) -> bool:
    """Smoke test: the genome's bot always returns a legal action on random states."""
    from game import Observation  # noqa: F401
    import types
    rng = random.Random(1)
    act = make_act(g)
    mem: dict = {}
    for i in range(n):
        deck = list(range(36))
        rng.shuffle(deck)
        nb = rng.choice([0, 3, 4, 5])
        legal = tuple(rng.choice([(FOLD, CALL, HALF, POT), (CALL, HALF, POT), (FOLD, CALL), (CALL,)]))
        pot = rng.randint(3, 160)
        obs = types.SimpleNamespace(
            hole_cards=tuple(deck[:5]), board=tuple(deck[5:5 + nb]), legal_actions=legal,
            to_call=0 if FOLD not in legal else rng.randint(1, 40), pot=pot,
            is_button=bool(rng.getrandbits(1)), action_history=(), hand_index=i)
        a = act(obs, mem)
        if a not in legal:
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    a = ap.parse_args()
    rng = random.Random(7)
    bad = 0
    for i in range(a.sample):
        g = random_genome(rng)
        ok = fuzz(g)
        bad += not ok
        print(i, "OK " if ok else "BAD", json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                                    for k, v in g.items() if k != "seed"}))
    print("illegal-action genomes:", bad)
