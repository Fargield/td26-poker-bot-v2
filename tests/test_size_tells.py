"""Bet-size tells: learn P(strong | pot bet) vs P(strong | half bet).

In-process simulations with game.Game (no HTTP / judge), like
tests/test_opponent_reraise.py. Our probe simply check/calls, so the opponent
gets many postflop FREE decisions and almost every hand reaches showdown.
"""
import random

import equity as E
from game import Action, AgentConfig, Game, make_deal
from opponent import OpponentModel


def _play(opp_policy, hands=300, seed=0):
    """Play `hands` against opp_policy; return the tracked profile."""
    model = OpponentModel(AgentConfig())
    rng = random.Random(seed)
    for h in range(hands):
        g = Game(make_deal(seed, h), button_seat=h % 2, hand_index=h)
        while not g.is_terminal:
            if g.to_act == 0:
                model.update_from_obs(g.observation(0))
                action = Action.CHECK_CALL       # never fold/raise: max showdowns
            else:
                action = opp_policy(g.observation(1), rng)
            g.apply(action)
        model.on_hand_end(g.hand_result(0))
    return model.profile()


def _equity_sizer(obs, rng):
    """Postflop: pot-bet only with equity > 0.7, otherwise half-bet."""
    if obs.to_call > 0 or not obs.board:
        return Action.CHECK_CALL
    e = E.equity(obs.hole_cards, obs.board, samples=400)
    if e > 0.7 and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    if Action.RAISE_HALF in obs.legal_actions:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def _random_sizer(obs, rng):
    """Postflop: pick half or pot at random, independent of the cards."""
    if obs.to_call > 0 or not obs.board:
        return Action.CHECK_CALL
    if Action.RAISE_HALF in obs.legal_actions and Action.RAISE_POT in obs.legal_actions:
        return rng.choice((Action.RAISE_HALF, Action.RAISE_POT))
    return Action.CHECK_CALL


def test_pot_strong_half_medium_size_tell():
    p = _play(_equity_sizer, hands=300, seed=0)
    assert p["n_size_half"] >= 8
    assert p["n_size_pot"] >= 8
    assert p["size_tell"] > 0.2, p["size_tell"]


def test_random_sizes_have_no_tell():
    # The max size of a hand is pot whenever any street used pot, so half-max
    # showdowns are only ~12% of hands. More hands keep the noise below 0.1.
    p = _play(_random_sizer, hands=2000, seed=0)
    assert p["n_size_half"] >= 8
    assert p["n_size_pot"] >= 8
    assert abs(p["size_tell"]) < 0.1, p["size_tell"]


def test_default_profile_has_size_fields_and_never_raises():
    p = OpponentModel(AgentConfig()).profile()
    assert p["size_tell"] == 0.0
    assert p["n_size_half"] == 0
    assert p["n_size_pot"] == 0
    assert p["bet_half_rate"] == 0.2
    assert p["bet_pot_rate"] == 0.2
    assert p["n_post_free"] == 0
