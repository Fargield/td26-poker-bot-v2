"""pytest checks for opponent.py (the heads-up opponent tracker).

Run from the bundle folder:

    python -m pytest -q tests/test_opponent.py

The tracker is fed the same Observations/HandResults the judge would send,
using in-process Games only (no HTTP, no judge).
"""

import os
import random
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)  # tests/ sits one level below bundle/
sys.path.insert(0, BUNDLE)

from game import (  # noqa: E402
    Action,
    AgentConfig,
    Game,
    HandResult,
    Observation,
    ObservedAction,
    Street,
    make_deal,
)
from opponent import OpponentModel, get_model  # noqa: E402


KINDS = ("pot", "half", "station", "random", "checkfold")
EXPECTED_CLS = {
    "pot": "maniac_pot",
    "half": "maniac_half",
    "station": "calling_station",
    "checkfold": "check_fold",
    "random": "random",
}


# ---------------------------------------------------------------------------
# Opponent policies and the probe policy
# ---------------------------------------------------------------------------

def _opponent_policy(kind, rng):
    """Return a callable Observation -> Action for one opponent archetype."""

    if kind == "pot":
        def policy(obs):
            if Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            return Action.CHECK_CALL

    elif kind == "half":
        def policy(obs):
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL

    elif kind == "station":
        def policy(obs):
            return Action.CHECK_CALL

    elif kind == "random":
        def policy(obs):
            return rng.choice(obs.legal_actions)

    elif kind == "checkfold":
        def policy(obs):
            if obs.to_call == 0:
                return Action.CHECK_CALL
            return Action.FOLD

    else:  # pragma: no cover - protects against a typo in a test
        raise ValueError(f"unknown opponent kind: {kind!r}")

    return policy


def _probe_policy(obs, rng):
    """Our own (probe) policy; rng is seeded with the simulation seed."""
    r = rng.random()
    if Action.RAISE_HALF in obs.legal_actions and r < 0.3:
        return rng.choice((Action.RAISE_HALF, Action.RAISE_POT))
    if Action.FOLD in obs.legal_actions and r > 0.85:
        return Action.FOLD
    return Action.CHECK_CALL


def simulate(opp_kind, n_hands, seed):
    """Play n_hands against opp_kind; return (model, seat-1 action counts)."""
    model = OpponentModel(AgentConfig())
    rng = random.Random(seed)
    # The opponent policy takes only an Observation, so it owns its own RNG;
    # the probe RNG is independent (as in tools/probe_bot.py).
    opponent = _opponent_policy(opp_kind, random.Random(seed))
    truth = [0, 0, 0, 0]

    for d in range(n_hands):
        game = Game(make_deal(seed, d), button_seat=d % 2, hand_index=d)
        while not game.is_terminal:
            if game.to_act == 0:
                obs = game.observation(0)
                model.update_from_obs(obs)
                action = _probe_policy(obs, rng)
            else:
                obs = game.observation(1)
                action = opponent(obs)
            game.apply(action)

        for rec in game.history:
            if rec.seat == 1:
                truth[int(rec.action)] += 1
        model.on_hand_end(game.hand_result(0))

    return model, truth


# ---------------------------------------------------------------------------
# 1. Completeness: every opponent action is filed exactly once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", KINDS)
def test_completeness(kind, seed):
    model, truth = simulate(kind, 300, seed)
    totals = [sum(model.cnt[i] for i in range(128) if i % 4 == a) for a in range(4)]
    assert totals == truth
    assert model.profile()["errors"] == 0


# ---------------------------------------------------------------------------
# 2. Classification and confidence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", KINDS)
def test_classification(kind, seed):
    model, _ = simulate(kind, 200, seed)
    prof = model.profile()
    assert prof["cls"] == EXPECTED_CLS[kind]
    assert prof["conf"] >= 0.9


# ---------------------------------------------------------------------------
# 3. How quickly the class is identified
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", ("pot", "half", "station", "checkfold"))
def test_speed_deterministic(kind, seed):
    model, _ = simulate(kind, 15, seed)
    assert model.profile()["cls"] == EXPECTED_CLS[kind]


@pytest.mark.parametrize("seed", range(5))
def test_speed_random(seed):
    model, _ = simulate("random", 60, seed)
    assert model.profile()["cls"] == EXPECTED_CLS["random"]


# ---------------------------------------------------------------------------
# 4. Stats sanity after 200 hands, seed 0
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def profiles_200():
    return {kind: simulate(kind, 200, 0)[0].profile() for kind in KINDS}


def test_stats_station(profiles_200):
    p = profiles_200["station"]
    assert p["fold_to_bet"] == 0
    assert p["agg"] == 0
    assert p["pfr"] == 0
    assert p["open_call"] == 1.0
    assert "never_folds" in p["tags"]


def test_stats_checkfold(profiles_200):
    p = profiles_200["checkfold"]
    assert p["fold_to_bet"] == 1.0
    assert p["vpip"] == 0
    assert p["open_fold"] == 1.0
    assert p["check_free"] == 1.0


def test_stats_pot_and_half(profiles_200):
    pot = profiles_200["pot"]
    assert pot["half_share"] == 0.0
    assert pot["pfr"] == 1.0
    assert pot["fold_to_bet"] == 0

    half = profiles_200["half"]
    assert half["half_share"] == 1.0


def test_stats_random(profiles_200):
    p = profiles_200["random"]
    assert 0.15 <= p["fold_to_bet"] <= 0.40
    assert 0.3 <= p["half_share"] <= 0.7


@pytest.mark.parametrize("kind", KINDS)
def test_stats_shape(profiles_200, kind):
    p = profiles_200[kind]
    assert len(p["by_street"]) == 4
    assert set(p["by_pos"]) == {"btn", "bb"}
    assert len(p["sd_cat"]) == 9
    assert p["hands"] == 200


# ---------------------------------------------------------------------------
# 5. Hand-written synthetic histories
# ---------------------------------------------------------------------------

def _obs(hand_index, street, pot, to_call, my_bet, opp_bet, my_stack,
         opp_stack, is_button, history, legal_actions=(Action.FOLD, Action.CHECK_CALL)):
    """Build an Observation with dummies for fields the tracker never reads."""
    return Observation(
        hand_index=hand_index,
        street=street,
        hole_cards=(),
        board=(),
        pot=pot,
        to_call=to_call,
        my_bet=my_bet,
        opp_bet=opp_bet,
        my_stack=my_stack,
        opp_stack=opp_stack,
        is_button=is_button,
        legal_actions=tuple(legal_actions),
        raise_to={},
        min_raise_to=0,
        raises_this_street=0,
        action_history=tuple(history),
        big_blind=2,
    )


def test_sb_folds_before_we_act():
    model = OpponentModel(AgentConfig())
    result = HandResult(hand_index=0, my_net=1, pot=3, went_to_showdown=False,
                        board=(), opp_hole=None, winner="me")
    model.on_hand_end(result)
    prof = model.profile()
    assert prof["n_open"] == 1
    assert prof["open_fold"] == 1.0
    assert prof["n_pf"] == 1
    assert prof["vpip"] == 0.0


def test_incremental_history_is_not_double_counted():
    model = OpponentModel(AgentConfig())
    opp_raise = ObservedAction(Street.PREFLOP, False, Action.RAISE_POT, 5, 6)

    obs1 = _obs(0, Street.PREFLOP, pot=8, to_call=4, my_bet=2, opp_bet=6,
                my_stack=78, opp_stack=74, is_button=False, history=(opp_raise,))
    model.update_from_obs(obs1)
    model.update_from_obs(obs1)  # same observation delivered twice
    prof = model.profile()
    assert prof["n_dec"] == 1
    assert prof["n_open"] == 1
    assert prof["open_raise"] == 1.0

    my_call = ObservedAction(Street.PREFLOP, True, Action.CHECK_CALL, 4, 6)
    obs2 = _obs(0, Street.FLOP, pot=12, to_call=0, my_bet=0, opp_bet=0,
                my_stack=74, opp_stack=74, is_button=False,
                history=(opp_raise, my_call), legal_actions=(Action.CHECK_CALL,))
    model.update_from_obs(obs2)
    assert model.profile()["n_dec"] == 1  # our own call is not an opponent action

    result = HandResult(hand_index=0, my_net=6, pot=18, went_to_showdown=False,
                        board=(0, 1, 2), opp_hole=None, winner="me")
    model.on_hand_end(result)  # we bet 6 on the flop and the opponent folded
    prof = model.profile()
    assert prof["n_dec"] == 2
    assert prof["fold_to_bet"] == 1.0
    assert prof["by_street"][1]["n_face"] == 1
    assert prof["hands"] == 1


def test_duplicate_on_hand_end_is_ignored():
    model = OpponentModel(AgentConfig())
    result = HandResult(hand_index=0, my_net=1, pot=3, went_to_showdown=False,
                        board=(), opp_hole=None, winner="me")
    model.on_hand_end(result)
    before = model.profile()
    hands_before, ndec_before = before["hands"], before["n_dec"]

    model.on_hand_end(result)  # duplicate delivery for the same hand_index
    after = model.profile()
    assert after["hands"] == hands_before
    assert after["n_dec"] == ndec_before


# ---------------------------------------------------------------------------
# 6. Robustness: junk input never escapes
# ---------------------------------------------------------------------------

def test_robustness():
    model = OpponentModel(AgentConfig())
    model.update_from_obs(None)
    model.update_from_obs(object())
    prof = model.profile()
    assert isinstance(prof, dict) and "cls" in prof
    assert prof["errors"] >= 1

    model2 = OpponentModel(AgentConfig())
    model2.on_hand_end(None)
    model2.on_hand_end("junk")
    prof2 = model2.profile()
    assert isinstance(prof2, dict) and "cls" in prof2
    assert prof2["errors"] >= 1

    for cfg in (None, {"starting_stack": "x"}):
        m = OpponentModel(cfg)
        assert isinstance(m.profile(), dict)
        assert "cls" in m.profile()

    assert isinstance(get_model({}), OpponentModel)
    memory = {}
    first = get_model(memory)
    second = get_model(memory)
    assert isinstance(first, OpponentModel)
    assert first is second


# ---------------------------------------------------------------------------
# 7. Timing: profile() at every one of our decisions
# ---------------------------------------------------------------------------

def test_timing_1000_hands_profile_every_decision():
    model = OpponentModel(AgentConfig())
    rng = random.Random(0)
    opponent = _opponent_policy("random", random.Random(0))
    elapsed = 0.0

    for d in range(1000):
        game = Game(make_deal(0, d), button_seat=d % 2, hand_index=d)
        while not game.is_terminal:
            if game.to_act == 0:
                obs = game.observation(0)
                t0 = time.perf_counter()
                model.update_from_obs(obs)
                model.profile()
                elapsed += time.perf_counter() - t0
                action = _probe_policy(obs, rng)
            else:
                action = opponent(game.observation(1))
            game.apply(action)

        t0 = time.perf_counter()
        model.on_hand_end(game.hand_result(0))
        elapsed += time.perf_counter() - t0

    assert elapsed < 0.6, f"model calls took {elapsed:.3f}s for 1000 hands"


# Regression: with a fixed mistake rate the deterministic archetypes lost to
# the Dirichlet "unknown" model after a few hundred hands (cost linear in n).
@pytest.mark.parametrize("kind", KINDS)
def test_class_stays_locked_full_half(kind):
    model, _ = simulate(kind, 1000, 3)
    prof = model.profile()
    assert prof["cls"] == EXPECTED_CLS[kind]
    assert prof["conf"] >= 0.99
