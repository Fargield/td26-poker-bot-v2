"""
strategy.py - decision logic for the heads-up Short Deck PLO5 bot.

Entry point:  decide(obs, prof, memory) -> Action   (may raise; agent.py wraps it)

Idea (equity vs price, opponent-aware):
  * e = our equity vs the opponent's likely range (equity.py). Against bots
    whose actions do not depend on their cards (always-raise, calling
    station, random) the range is "any hand" -> the equity is exact.
    Against other bots the range is narrowed by their preflop action
    (opp_top) and a margin is added when they bet postflop.
  * Facing a bet we continue if e > threshold, where the threshold blends
    plain pot odds (opponent will not bet again) with the "all-in or fold"
    break-even (opponent keeps betting every street until all-in), weighted
    by w = how likely the opponent is to keep putting money in.
  * Value-raise when e is clearly above 0.5 vs that range; pot-size raises
    against opponents that never fold (EV per chip = 2e - 1 > 0).
  * Bluff only against opponents that fold a lot (measured fold_to_bet),
    never against stations / maniacs / random.
  * Time bank: memory["t_used"] is the thinking time spent so far this half
    (measured in agent.py); sample counts shrink as the bank empties.
"""

from __future__ import annotations

import json
import os

import equity as E
from game import Action

try:
    import range_equity as RE
except Exception:          # module broken -> fall back to equity margins
    RE = None

# Tunable parameters (override for experiments with env STRAT_PARAMS='{"key": value}').
# Card-blind opponent groups: "maniac" (always raise), "station", "random", "folder" (check/fold).
# Per group: w = how likely they keep betting later (blends pot odds with all-in break-even),
# raise_* = equity to raise when facing a bet, bet_* = equity to bet when checked to.
P = {
    "maniac_w": 0.4, "maniac_raise_pf": 0.66, "maniac_raise_post": 0.58,
    "maniac_bet_pf": 0.66, "maniac_bet_post": 0.58,
    "station_w": 0.0, "station_raise_pf": 0.46, "station_raise_post": 0.44,
    "station_bet_pf": 0.46, "station_bet_post": 0.44,
    "random_w": 0.0, "random_raise_pf": 0.60, "random_raise_post": 0.53,
    "random_bet_pf": 0.60, "random_bet_post": 0.53,
    "folder_w": 0.0, "folder_raise_pf": 0.53, "folder_raise_post": 0.52,
    "folder_bet_pf": 0.53, "folder_bet_post": 0.52,
    "raise_pf": 0.62, "raise_post": 0.66,           # raise vs other (card-aware) opponents
    "bet_pf": 0.60, "bet_post": 0.60,
    "pot_step_raise": 0.08, "pot_step_bet": 0.10,   # equity above thr -> pot instead of half
    "call_margin_rr": 0.0,                          # extra equity needed to call, card-blind opp
    # card-aware opponents betting postflop: equity vs their top-X% made hands (range_equity.py)
    "range_on": 1, "range_hedge": 0.5, "range_min": 0.12, "range_per_raise": 0.6,
    "range_default_freq": 0.35, "range_samples": 1000,
    # bet-size tell: use showdown-calibrated P(strong | pot) vs P(strong | half)
    "size_on": 1, "size_tell_min": 0.15,
    # re-raise steal vs opponents that fold to our re-raises
    "steal_on": 1, "steal_min_n": 15, "steal_edge": 0.15,
}
# Trained overrides (written by tools/train.py), then the env var on top.
try:
    _pf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
    if os.path.isfile(_pf) and not os.environ.get("STRAT_IGNORE_FILE"):
        with open(_pf, encoding="utf-8") as _f:
            P.update({k: v for k, v in json.load(_f).items() if k in P})
except Exception:
    pass
try:
    P.update(json.loads(os.environ.get("STRAT_PARAMS", "") or "{}"))
except Exception:
    pass

FOLD, CALL, HALF, POT = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT

# Opponent classes from opponent.py whose actions ignore their cards -> parameter group.
GROUP = {"maniac_pot": "maniac", "maniac_half": "maniac", "maniac": "maniac",
         "calling_station": "station", "random": "random", "check_fold": "folder"}
RANDOM_RANGE = set(GROUP)

# Sample counts per street (index = len(board) bucket) by time-bank level.
SAMPLES_NORMAL = {3: 1000, 4: 1500, 5: 2000}
SAMPLES_LOW = {3: 400, 4: 500, 5: 800}
SAMPLES_CRIT = {3: 120, 4: 150, 5: 300}


def _samples(board_len: int, memory: dict) -> int:
    used = memory.get("t_used", 0.0)
    bank = memory.get("t_bank", 8.0)
    if used > 0.75 * bank:
        table = SAMPLES_CRIT
    elif used > 0.5 * bank:
        table = SAMPLES_LOW
    else:
        table = SAMPLES_NORMAL
    return table.get(board_len, 1000)


def _opp_preflop_action(obs) -> int:
    """0 = no voluntary action seen yet / checked, 1 = called (limped), 2 = raised."""
    best = 0
    for a in obs.action_history:
        if a.street == 0 and not a.is_me:
            if a.action in (HALF, POT):
                return 2
            if a.action == CALL:
                best = 1
    return best


def _opp_bet_this_street(obs) -> int:
    """Number of raises the opponent made on the current street."""
    st = int(obs.street)
    return sum(1 for a in obs.action_history
               if a.street == st and not a.is_me and a.action in (HALF, POT))


def _opp_last_raise(obs) -> int:
    """Action int (HALF/POT) of the opponent's last raise this street, else -1."""
    st = int(obs.street)
    for a in reversed(obs.action_history):
        if a.street == st and not a.is_me and a.action in (HALF, POT):
            return int(a.action)
    return -1


def _opp_top(obs, prof, rr: bool) -> float:
    """Fraction of starting hands the opponent may hold."""
    if rr:
        return 1.0
    act = _opp_preflop_action(obs)
    if act == 2:
        top = prof.get("pfr", 0.3)
    elif act == 1:
        top = prof.get("vpip", 0.7)
    else:
        return 1.0
    # be conservative: never assume tighter than top 20%
    return min(1.0, max(0.2, float(top)))


def _equity(obs, memory, opp_top: float) -> float:
    hole = tuple(obs.hole_cards)
    board = tuple(obs.board)
    if len(board) == 0:
        if opp_top >= 0.999:
            return E.preflop_strength(hole)
        return E.equity(hole, (), samples=max(100, int(600 * opp_top) if memory.get("t_used", 0) < 0.5 * memory.get("t_bank", 8.0) else 100), opp_top=opp_top)
    n = _samples(len(board), memory)
    if opp_top < 0.999:
        # rejection sampling costs ~samples/opp_top: keep the work constant
        n = max(100, int(n * opp_top))
    return E.equity(hole, board, samples=n, opp_top=opp_top)


def _pick(legal, *prefs):
    for a in prefs:
        if a in legal:
            return a
    return CALL


def decide(obs, prof: dict, memory: dict) -> Action:
    legal = obs.legal_actions
    if len(legal) == 1:
        return legal[0]
    can_raise = HALF in legal or POT in legal

    cls = prof.get("cls", "unknown") if prof else "unknown"
    rr = cls in RANDOM_RANGE
    street = int(obs.street)
    preflop = street == 0
    river = street == 3

    S = memory.get("stack", 80)
    my_in = S - obs.my_stack            # our chips in this hand (stacks reset each hand)
    opp_in = S - obs.opp_stack
    eff = min(my_in + obs.my_stack, opp_in + obs.opp_stack)  # max each can put in
    pot = obs.pot
    to_call = obs.to_call

    opp_top = _opp_top(obs, prof, rr)
    e = _equity(obs, memory, opp_top)

    # ---- opponent tendencies -------------------------------------------
    grp = GROUP.get(cls)
    if grp:
        w = P[grp + "_w"]
    else:
        agg = prof.get("agg", 0.3) if prof else 0.3
        w = min(1.0, max(0.0, 1.2 * agg))
    fold_to_bet = prof.get("fold_to_bet", 0.4) if prof else 0.4
    n_face = prof.get("n_face", 0) if prof else 0
    never_folds = cls in ("maniac_pot", "maniac_half", "maniac", "calling_station") or (
        n_face >= 20 and fold_to_bet < 0.08)

    # a card-aware opponent's postflop bets mean strength: narrow its range
    margin = 0.0
    e_bet = e                          # equity used when facing their bet
    post_top = 1.0                     # their current-street range (fraction of made hands)
    if not rr and not preflop:
        nb = _opp_bet_this_street(obs)
        if nb:
            bs = prof["by_street"][street] if prof and "by_street" in prof else {}
            agg_s = bs.get("agg", 0.3) if bs.get("n_agg", 0) >= 10 else 0.3
            margin = (0.04 + 0.06 * (1.0 - min(1.0, agg_s * 2))) * min(nb, 2)
            used = memory.get("t_used", 0.0)
            if (P["range_on"] and RE is not None
                    and used < 0.6 * memory.get("t_bank", 8.0)):
                size_tell_on = (
                    P["size_on"] and prof
                    and prof.get("n_size_half", 0) >= 8
                    and prof.get("n_size_pot", 0) >= 8
                    and prof.get("size_tell", 0.0) >= P["size_tell_min"])
                if size_tell_on:
                    # Pot bets come from the top chunk; half bets are the
                    # medium chunk plus a possible top-chunk sizing mix.
                    last = _opp_last_raise(obs)
                    if last == POT:
                        base = prof.get("bet_pot_rate", 0.2)
                    elif last == HALF:
                        base = prof.get("bet_half_rate", 0.2) + prof.get("bet_pot_rate", 0.2)
                    else:
                        base = 0.0
                else:
                    base = agg_s if bs.get("n_agg", 0) >= 10 else P["range_default_freq"]
                post_top = base * (P["range_per_raise"] ** (nb - 1))
                post_top = min(1.0, max(P["range_min"], post_top))
                try:
                    er = RE.range_equity(obs.hole_cards, obs.board, opp_top, post_top,
                                         samples=P["range_samples"])
                    e_bet = P["range_hedge"] * e + (1.0 - P["range_hedge"]) * er
                    margin = 0.0
                except Exception:
                    e_bet = e

    # ---- facing a bet ----------------------------------------------------
    if to_call > 0:
        pot_odds = to_call / (pot + to_call)
        allin_thr = (eff - my_in) / (2.0 * eff) if eff > 0 else pot_odds
        if river or obs.opp_stack == 0 or to_call >= obs.my_stack:
            thr = pot_odds
        else:
            thr = w * max(allin_thr, pot_odds) + (1.0 - w) * pot_odds
        e_adj = e_bet - margin
        if rr:
            thr += P["call_margin_rr"]

        # value raise
        if can_raise:
            if rr:
                r_thr = P[grp + ("_raise_pf" if preflop else "_raise_post")]
            else:
                r_thr = P["raise_pf"] if preflop else P["raise_post"]
            if e_adj >= r_thr:
                if never_folds or e_adj >= r_thr + P["pot_step_raise"]:
                    return _pick(legal, POT, HALF)
                return _pick(legal, HALF, POT)

        # re-raise steal vs opponents that fold to our re-raises (EV comparison)
        if can_raise and not rr and P["steal_on"]:
            n_rr = prof.get("n_reraise", 0) if prof else 0
            f = prof.get("fold_to_reraise", 0.3) if prof else 0.3
            e_called = None
            if n_rr >= P["steal_min_n"] and f > 0.3:
                # if they do not fold they hold the top (1 - f) of their current range
                try:
                    if preflop:
                        top = max(0.1, (1.0 - f) * opp_top)
                        e_called = E.equity(obs.hole_cards, (), samples=max(100, int(600 * top)),
                                            opp_top=top)
                    elif RE is not None:
                        pt = max(0.05, (1.0 - f) * post_top)
                        e_called = RE.range_equity(obs.hole_cards, obs.board, opp_top, pt,
                                                   samples=P["range_samples"])
                except Exception:
                    e_called = None
            if e_called is not None:
                ev_cont = max(0.0, e_adj * (pot + to_call) - to_call) if e_adj >= thr else 0.0
                best, best_ev = None, ev_cont
                for a in (HALF, POT):
                    if a not in legal:
                        continue
                    risk = obs.raise_to[a] - obs.my_bet
                    if risk <= 0:
                        continue
                    fe = max(0.0, f - P["steal_edge"])
                    ev = fe * pot + (1.0 - fe) * (e_called * (pot + 2 * risk - to_call) - risk)
                    if ev > best_ev:
                        best, best_ev = a, ev
                if best is not None:
                    return best

        if e_adj >= thr:
            return CALL
        return _pick(legal, FOLD)

    # ---- nothing to call: check or bet -------------------------------------
    if not can_raise:
        return CALL
    if rr:
        b_thr = P[grp + ("_bet_pf" if preflop else "_bet_post")]
    else:
        b_thr = P["bet_pf"] if preflop else P["bet_post"]
    if e >= b_thr:
        if never_folds or rr or e >= b_thr + P["pot_step_bet"]:
            return _pick(legal, POT, HALF)
        return _pick(legal, HALF, POT)

    # bluff / steal vs opponents that fold a lot (never vs stations/maniacs/random)
    if not never_folds and cls not in ("random",) and n_face >= 12 and fold_to_bet >= 0.45:
        if cls == "check_fold" or fold_to_bet >= 0.6 or e >= 0.35:
            return _pick(legal, HALF, POT)
    return CALL
