"""fold_to_reraise: opponent folds after raising and then facing our re-raise."""
import random

from game import Action, AgentConfig, Game, make_deal
from opponent import OpponentModel


def _play(opp_policy, hands=200, seed=1):
    m = OpponentModel(AgentConfig())
    rng = random.Random(seed)
    for h in range(hands):
        g = Game(make_deal(seed, h), button_seat=h % 2, hand_index=h)
        while not g.is_terminal:
            legal = g.legal_actions()
            if g.to_act == 0:
                obs = g.observation(0)
                m.update_from_obs(obs)
                opp_raised = any((not a.is_me) and a.street == obs.street and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                                 for a in obs.action_history)
                a = Action.RAISE_POT if (opp_raised and Action.RAISE_POT in legal) else Action.CHECK_CALL
            else:
                a = opp_policy(g.observation(1), legal, rng)
            g.apply(a)
        m.on_hand_end(g.hand_result(0))
    return m.profile()


def _raise_then_fold(obs, legal, rng):
    me_raised = any(a.is_me and a.street == obs.street and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                    for a in obs.action_history)
    if me_raised and Action.FOLD in legal:
        return Action.FOLD
    if Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def _always_pot(obs, legal, rng):
    return Action.RAISE_POT if Action.RAISE_POT in legal else Action.CHECK_CALL


def test_raise_then_fold_detected():
    p = _play(_raise_then_fold)
    assert p["n_reraise"] >= 50
    assert p["fold_to_reraise"] > 0.95


def test_always_pot_never_folds_to_reraise():
    p = _play(_always_pot)
    assert p["n_reraise"] >= 50
    assert p["fold_to_reraise"] < 0.05


def test_default_profile_has_fields():
    p = OpponentModel(AgentConfig()).profile()
    assert "fold_to_reraise" in p and "n_reraise" in p
