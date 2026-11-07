"""League opponent strategies: choose_action(obs, memory) for each KIND.

These imitate what mid-tier teams will plausibly submit. Used by league/make_bots.py,
which wraps each one in the standard server template.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equity as E
from game import Action

FOLD, CALL, HALF, POT = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT


def _eq(obs, samples=300):
    if len(obs.board) == 0:
        return E.preflop_strength(tuple(obs.hole_cards))
    return E.equity(tuple(obs.hole_cards), tuple(obs.board), samples=samples)


def _pick(legal, *prefs):
    for a in prefs:
        if a in legal:
            return a
    return CALL if CALL in legal else legal[0]


def _rng(memory):
    r = memory.get("rng")
    if r is None:
        r = memory["rng"] = random.Random(12345)
    return r


def threshold_bot(bet_thr, raise_thr, call_slack, bluff=0.0):
    """Equity-threshold bot. call_slack: extra equity below pot odds still called (+) or needed (-)."""
    def act(obs, memory):
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        e = _eq(obs)
        r = _rng(memory)
        if obs.to_call > 0:
            odds = obs.to_call / (obs.pot + obs.to_call)
            if e >= raise_thr:
                return _pick(legal, POT if e >= raise_thr + 0.1 else HALF, HALF)
            if e + call_slack >= odds:
                return CALL
            return _pick(legal, FOLD)
        if e >= bet_thr:
            return _pick(legal, POT if e >= bet_thr + 0.1 else HALF, HALF)
        if bluff and r.random() < bluff:
            return _pick(legal, HALF)
        return CALL
    return act


def exploiter(obs, memory):
    """Built to punish a transparent bot: stab every check, fold to bets without a strong hand."""
    legal = obs.legal_actions
    if len(legal) == 1:
        return legal[0]
    e = _eq(obs)
    r = _rng(memory)
    if obs.to_call > 0:
        odds = obs.to_call / (obs.pot + obs.to_call)
        # opponent bets mean strength: continue only when clearly ahead of a strong range
        if e >= 0.62:
            return _pick(legal, POT, HALF)
        if e >= max(odds + 0.12, 0.5):
            return CALL
        return _pick(legal, FOLD)
    # checked to us (or we act first): bet most of the time
    if e >= 0.5 or r.random() < 0.65:
        return _pick(legal, HALF if e < 0.6 else POT, HALF)
    return CALL


def tag(obs, memory):
    return threshold_bot(0.60, 0.68, 0.0)(obs, memory)


def lag(obs, memory):
    return threshold_bot(0.52, 0.62, 0.03, bluff=0.25)(obs, memory)


KINDS = {
    "eq_tight": threshold_bot(0.62, 0.72, -0.03),
    "eq_medium": threshold_bot(0.55, 0.66, 0.0),
    "eq_loose": threshold_bot(0.50, 0.62, 0.05),
    "eq_bluffer": threshold_bot(0.55, 0.66, 0.0, bluff=0.3),
    "tag": tag,
    "lag": lag,
    "exploiter": exploiter,
}
