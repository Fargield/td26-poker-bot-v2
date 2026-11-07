"""
opponent.py - cheap, crash-proof opponent tracker for the heads-up PLO5 bot.

Usage from agent.py (all calls are safe: they never raise):

    from opponent import OpponentModel

    def new_half(config, memory):
        memory["opp"] = OpponentModel(config)

    def choose_action(obs, memory):
        opp = memory["opp"]
        opp.update_from_obs(obs)          # first thing every decision
        prof = opp.profile()              # cached dict, see PROFILE FIELDS below
        ...

    def on_hand_end(result, memory):
        memory["opp"].on_hand_end(result)

How it works
------------
Every opponent decision is filed under (street, opp position, context, action):

  context  FREE    nothing to call (check / bet half / bet pot)
           FACE    facing a voluntary bet/raise, raising allowed (fold/call/half/pot)
           CAPPED  facing a bet, raising NOT allowed (cap reached or all-in) (fold/call)
           OPEN    preflop, opp is the small blind facing only the big blind

Opponent decisions we never see in an Observation (its answer to OUR last
action of the hand) are inferred in on_hand_end() from the final pot:
fold to our raise, call of our raise, river check behind, and the small
blind folding preflop before we ever act.

Classification is a Bayesian comparison of fixed archetypes (always pot,
always half, always raise, calling station, check/fold, uniform random)
against an "unknown" catch-all (Dirichlet-multinomial, any stationary
mix). Deterministic bots are identified after a handful of decisions;
uniform random needs ~20-40 opponent decisions.

PROFILE FIELDS (profile() -> dict; rates are k/n, with a neutral default when n == 0)
  cls, conf          class label and posterior of the best archetype; cls is
                     'unknown' (or 'nit' by rule) while conf < CLASS_MIN_CONF
  best, post         best archetype regardless of conf; {archetype: posterior}
  hands, n_dec       finished hands, opponent decisions recorded
  vpip, pfr, n_pf    preflop voluntarily-put-in / raise rate over hands where
                     the opponent made a preflop decision
  vpip_btn, n_pf_btn, vpip_bb, n_pf_bb   same split by the opponent's position
  agg, n_agg         raises / (raises + calls + checks), all streets
  fold_to_bet, n_face   folds / decisions facing a bet (FACE + CAPPED, excl. OPEN)
  call_vs_bet        calls / decisions facing a bet
  raise_vs_bet, n_face_rl   raises / FACE decisions (raising allowed)
  check_free, bet_free, n_free   check / bet rate when nothing to call
  open_fold, open_call, open_raise, n_open   SB first action preflop
  half_share, n_raises   RAISE_HALF / all raises
  by_street          list of 4 dicts (PREFLOP..RIVER): agg, fold_to_bet,
                     raise_vs_bet, check_free, bet_free, n_agg, n_face, n_free
  by_pos             {'btn': {...}, 'bb': {...}} same keys, opp on button / big blind
  saw_flop, n_saw_flop, wtsd, n_sd   flop seen / went to showdown (given flop)
  sd_cat             list[9]: showdown counts by opp hand category (score >> 20)
  sd_weak_raise, n_sd_weak     P(opp raised postflop | showdown with high card/pair)
  sd_strong_raise, n_sd_strong P(opp raised postflop | showdown with straight+)
  my_net_bb          our cumulative net this half, in big blinds
  tags               list of style tags: aggressive, passive, folds_a_lot,
                     never_folds, loose, tight, bluffer, value_only
  errors             internal exceptions swallowed (should stay 0)
"""

from __future__ import annotations

import math

try:  # evaluator is optional: without it showdown categories are skipped
    from evaluator import omaha_score as _omaha_score
except Exception:  # pragma: no cover
    _omaha_score = None

# Action ints (game.Action): FOLD=0, CHECK_CALL=1, RAISE_HALF=2, RAISE_POT=3
FOLD, CC, HALF, POT = 0, 1, 2, 3
# Contexts
FREE, FACE, CAPPED, OPEN = 0, 1, 2, 3
CTX_NAMES = ("free", "face", "capped", "open")
# Opponent position index
POS_BB, POS_BTN = 0, 1
STREET_NAMES = ("preflop", "flop", "turn", "river")

CLASS_MIN_CONF = 0.90
DEV_A, DEV_B = 1.0, 30.0  # Beta prior on a deterministic archetype's deviation rate (mean ~3%)

ME, OPP = 0, 1


def _idx(street: int, pos: int, ctx: int, act: int) -> int:
    return ((street * 2 + pos) * 4 + ctx) * 4 + act


# Legal actions per context
_LEGAL = {
    FREE: (CC, HALF, POT),
    FACE: (FOLD, CC, HALF, POT),
    CAPPED: (FOLD, CC),
    OPEN: (FOLD, CC, HALF, POT),
}


def _det(main_by_ctx: dict) -> tuple:
    """Deterministic archetype: per context a 4-tuple of (is_main, log_weight).

    Main actions carry log(share of the main mass); any other legal action is
    a "deviation" with log(1 / #other legal actions). The deviation rate itself
    is integrated out with a Beta(DEV_A, DEV_B) prior (see _classify), so the
    cost of a perfectly consistent opponent grows only like log(n).
    """
    table = []
    for ctx in (FREE, FACE, CAPPED, OPEN):
        legal = _LEGAL[ctx]
        main = main_by_ctx[ctx]
        if not isinstance(main, dict):
            main = {main: 1.0}
        others = [a for a in legal if a not in main]
        row = [(False, -1e9)] * 4
        for a, p in main.items():
            row[a] = (True, math.log(p))
        for a in others:
            row[a] = (False, math.log(1.0 / len(others)))
        table.append(tuple(row))
    return tuple(table)


def _uniform() -> tuple:
    table = []
    for ctx in (FREE, FACE, CAPPED, OPEN):
        legal = _LEGAL[ctx]
        row = [(True, -1e9)] * 4
        for a in legal:
            row[a] = (True, math.log(1.0 / len(legal)))
        table.append(tuple(row))
    return tuple(table)


# log-probability tables [ctx][act]
ARCHETYPES = {
    "maniac_pot": _det({FREE: POT, FACE: POT, CAPPED: CC, OPEN: POT}),
    "maniac_half": _det({FREE: HALF, FACE: HALF, CAPPED: CC, OPEN: HALF}),
    "maniac": _det({FREE: {HALF: 0.5, POT: 0.5}, FACE: {HALF: 0.5, POT: 0.5},
                    CAPPED: CC, OPEN: {HALF: 0.5, POT: 0.5}}),
    "calling_station": _det({FREE: CC, FACE: CC, CAPPED: CC, OPEN: CC}),
    "check_fold": _det({FREE: CC, FACE: FOLD, CAPPED: FOLD, OPEN: FOLD}),
    "random": _uniform(),
}
LOG_PRIOR = {"maniac_pot": -2.0, "maniac_half": -2.0, "maniac": -3.0,
             "calling_station": -2.0, "check_fold": -2.0, "random": -2.0,
             "unknown": 0.0}
_LGAMMA_K = {k: math.lgamma(k) for k in (2, 3, 4)}


def _lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


_LBETA0 = _lbeta(DEV_A, DEV_B)


def _rate(k: float, n: float, default: float) -> float:
    return k / n if n > 0 else default


class _Hand:
    """Replay state of the current hand (bets/stacks per street)."""

    __slots__ = ("hand_index", "opp_pos", "street", "bets", "stacks", "acted",
                 "raises", "idx", "last_obs", "pf_decision", "vpip", "pfr",
                 "raises_post", "saw_flop", "done")

    def __init__(self, hand_index: int, opp_is_button: bool, stack: int, bb: int, sb: int):
        self.hand_index = hand_index
        self.opp_pos = POS_BTN if opp_is_button else POS_BB
        self.street = 0
        if opp_is_button:
            self.bets = [bb, sb]
        else:
            self.bets = [sb, bb]
        self.stacks = [stack - self.bets[ME], stack - self.bets[OPP]]
        self.acted = [False, False]
        self.raises = 0
        self.idx = 0
        self.last_obs = None
        self.pf_decision = False
        self.vpip = False
        self.pfr = False
        self.raises_post = 0
        self.saw_flop = False
        self.done = False

    def set_street(self, s: int) -> None:
        if s != self.street:
            self.street = s
            self.bets = [0, 0]
            self.acted = [False, False]
            self.raises = 0
            if s >= 1:
                self.saw_flop = True


class OpponentModel:
    """Per-half opponent tracker. Keep one instance in `memory`."""

    def __init__(self, config=None):
        self.stack = 80
        self.bb = 2
        self.sb = 1
        self.cap = 4
        try:
            if config is not None:
                get = config.get if isinstance(config, dict) else (lambda k, d: getattr(config, k, d))
                self.stack = int(get("starting_stack", 80))
                self.bb = int(get("big_blind", 2))
                self.sb = int(get("small_blind", 1))
                self.cap = int(get("raise_cap", 4))
        except Exception:
            pass
        self.cnt = [0] * 128
        self.hands = 0
        self.pf_n = [0, 0]     # by opp pos
        self.vpip_n = [0, 0]
        self.pfr_n = [0, 0]
        self.saw_flop_n = 0
        self.sd_n = 0           # showdowns among hands that saw a flop
        self.sd_cat = [0] * 9
        self.sd_cat_raised = [0] * 9
        self.my_net = 0
        self.errors = 0
        self._hand = None
        self._last_done = -1
        self._ver = 0
        self._cache_ver = -1
        self._cache = None

    # ------------------------------------------------------------------ input

    def update_from_obs(self, obs) -> None:
        """Process opponent actions in obs.action_history not seen yet."""
        try:
            hi = int(obs.hand_index)
            h = self._hand
            if h is None or h.hand_index != hi:
                if h is not None and not h.done:
                    self._finish_hand(h, None)
                h = _Hand(hi, not bool(obs.is_button), self.stack, self.bb, self.sb)
                self._hand = h
            hist = obs.action_history
            if len(hist) < h.idx:  # should not happen; restart this hand
                h = _Hand(hi, not bool(obs.is_button), self.stack, self.bb, self.sb)
                self._hand = h
            for i in range(h.idx, len(hist)):
                self._apply(h, hist[i])
            h.idx = len(hist)
            h.set_street(int(obs.street))
            h.last_obs = obs
        except Exception:
            self.errors += 1

    def on_hand_end(self, result) -> None:
        """Infer the opponent's unseen last action and close the hand."""
        try:
            hi = int(result.hand_index)
            if hi == self._last_done:
                return  # duplicate delivery
            h = self._hand
            if h is not None and h.hand_index != hi:
                if not h.done:
                    self._finish_hand(h, None)
                h = None
            if h is None or h.done:
                # We never acted this hand: the SB opponent folded first-in.
                h = _Hand(hi, True, self.stack, self.bb, self.sb)
                self._hand = h
                if not result.went_to_showdown and int(result.my_net) > 0:
                    self._count(h, 0, OPEN, FOLD)
                    h.pf_decision = True
            else:
                self._infer_last(h, result)
            self._finish_hand(h, result)
        except Exception:
            self.errors += 1

    # --------------------------------------------------------------- internals

    def _count(self, h: _Hand, street: int, ctx: int, act: int) -> None:
        self.cnt[_idx(street, h.opp_pos, ctx, act)] += 1
        self._ver += 1
        if street == 0:
            h.pf_decision = True
            if act == HALF or act == POT:
                h.vpip = True
                h.pfr = True
            elif act == CC and ctx != FREE:
                h.vpip = True
        elif act == HALF or act == POT:
            h.raises_post += 1

    def _ctx(self, h: _Hand, p: int) -> int:
        to_call = h.bets[1 - p] - h.bets[p]
        if to_call <= 0:
            return FREE
        if (h.street == 0 and p == OPP and h.opp_pos == POS_BTN
                and h.raises == 0 and not h.acted[OPP]):
            return OPEN
        if h.raises < self.cap and h.stacks[p] > to_call and h.stacks[1 - p] > 0:
            return FACE
        return CAPPED

    def _apply(self, h: _Hand, rec) -> None:
        h.set_street(int(rec.street))
        p = ME if rec.is_me else OPP
        act = int(rec.action)
        if p == OPP:
            self._count(h, h.street, self._ctx(h, p), act)
        if act == FOLD:
            h.acted[p] = True
            return
        amount = int(rec.amount)
        h.stacks[p] -= amount
        h.bets[p] = int(rec.street_total)
        if act == HALF or act == POT:
            h.raises += 1
            h.acted[1 - p] = False
        h.acted[p] = True

    def _infer_last(self, h: _Hand, result) -> None:
        obs = h.last_obs
        if obs is None:
            return
        h.set_street(int(obs.street))
        net = int(result.my_net)
        delta = int(result.pot) - int(obs.pot)
        to_call = max(0, int(obs.to_call))
        cc = min(to_call, int(obs.my_stack))
        showdown = bool(result.went_to_showdown)
        if not showdown:
            if net <= 0 or delta <= cc:
                return  # we folded (or nothing to infer)
            # we raised by `delta`, opponent folded
            self._my_raise(h, delta, h.bets[ME] + delta)
            self._count(h, h.street, self._ctx(h, OPP), FOLD)
            return
        if delta <= cc:
            # we checked / called
            h.stacks[ME] -= cc
            h.bets[ME] += cc
            h.acted[ME] = True
            round_over = h.acted[OPP] and h.bets[ME] == h.bets[OPP]
            if (not round_over and h.bets[ME] == h.bets[OPP]
                    and h.stacks[ME] > 0 and h.stacks[OPP] > 0):
                self._count(h, h.street, FREE, CC)  # opp checked behind
            return
        # we raised, opponent called (else there would be another obs)
        my_bet, opp_bet, opp_stack = int(obs.my_bet), int(obs.opp_bet), int(obs.opp_stack)
        new_total = None
        try:
            for rt in obs.raise_to.values():
                rt = int(rt)
                opp_add = min(rt - opp_bet, opp_stack)
                if (rt - my_bet) + opp_add == delta:
                    new_total = rt
                    break
        except Exception:
            new_total = None
        if new_total is None:
            new_total = my_bet + max(cc + 1, delta // 2)
        self._my_raise(h, new_total - my_bet, new_total)
        self._count(h, h.street, self._ctx(h, OPP), CC)

    def _my_raise(self, h: _Hand, add: int, new_total: int) -> None:
        h.stacks[ME] -= add
        h.bets[ME] = new_total
        h.raises += 1
        h.acted[ME] = True
        h.acted[OPP] = False

    def _finish_hand(self, h: _Hand, result) -> None:
        h.done = True
        self._last_done = h.hand_index
        self.hands += 1
        pos = h.opp_pos
        if h.pf_decision:
            self.pf_n[pos] += 1
            self.vpip_n[pos] += h.vpip
            self.pfr_n[pos] += h.pfr
        if result is not None:
            self.my_net += int(result.my_net)
            showdown = bool(result.went_to_showdown)
            if showdown:
                h.saw_flop = True
            if h.saw_flop:
                self.saw_flop_n += 1
                if showdown:
                    self.sd_n += 1
            opp_hole = result.opp_hole
            board = result.board
            if (showdown and opp_hole and board and len(board) == 5
                    and _omaha_score is not None):
                try:
                    cat = _omaha_score(tuple(opp_hole), tuple(board)) >> 20
                    if 0 <= cat <= 8:
                        self.sd_cat[cat] += 1
                        self.sd_cat_raised[cat] += 1 if h.raises_post > 0 else 0
                except Exception:
                    self.errors += 1
        self._ver += 1

    # ----------------------------------------------------------------- output

    def _sum(self, streets, poss, ctxs, acts) -> int:
        c = self.cnt
        t = 0
        for s in streets:
            for p in poss:
                for x in ctxs:
                    base = ((s * 2 + p) * 4 + x) * 4
                    for a in acts:
                        t += c[base + a]
        return t

    def _block(self, streets, poss) -> dict:
        S = self._sum
        raises = S(streets, poss, (FREE, FACE, OPEN), (HALF, POT))
        calls = S(streets, poss, (FACE, CAPPED, OPEN), (CC,))
        checks = S(streets, poss, (FREE,), (CC,))
        n_agg = raises + calls + checks
        n_face = S(streets, poss, (FACE, CAPPED), (FOLD, CC, HALF, POT))
        folds = S(streets, poss, (FACE, CAPPED), (FOLD,))
        n_face_rl = S(streets, poss, (FACE,), (FOLD, CC, HALF, POT))
        face_raise = S(streets, poss, (FACE,), (HALF, POT))
        n_free = S(streets, poss, (FREE,), (CC, HALF, POT))
        return {
            "agg": _rate(raises, n_agg, 0.3), "n_agg": n_agg,
            "fold_to_bet": _rate(folds, n_face, 0.4), "n_face": n_face,
            "raise_vs_bet": _rate(face_raise, n_face_rl, 0.15), "n_face_rl": n_face_rl,
            "check_free": _rate(checks, n_free, 0.6),
            "bet_free": _rate(n_free - checks, n_free, 0.4), "n_free": n_free,
        }

    def _classify(self):
        tot = [[0] * 4 for _ in range(4)]
        c = self.cnt
        for s in range(4):
            for p in range(2):
                for x in range(4):
                    base = ((s * 2 + p) * 4 + x) * 4
                    row = tot[x]
                    for a in range(4):
                        row[a] += c[base + a]
        ll = {}
        for name, table in ARCHETYPES.items():
            v = LOG_PRIOR[name]
            n_main = n_dev = 0
            for x in range(4):
                row = tot[x]
                lt = table[x]
                for a in range(4):
                    k = row[a]
                    if k:
                        is_main, lw = lt[a]
                        v += k * lw
                        if is_main:
                            n_main += k
                        else:
                            n_dev += k
            if name != "random":
                v += _lbeta(n_dev + DEV_A, n_main + DEV_B) - _LBETA0
            ll[name] = v
        # unknown: Dirichlet(1) marginal likelihood per context
        v = LOG_PRIOR["unknown"]
        for x in range(4):
            legal = _LEGAL[x]
            n = 0
            for a in legal:
                n += tot[x][a]
                v += math.lgamma(tot[x][a] + 1)
            k = len(legal)
            v += _LGAMMA_K[k] - math.lgamma(n + k)
        ll["unknown"] = v
        m = max(ll.values())
        z = sum(math.exp(x - m) for x in ll.values())
        post = {k: math.exp(x - m) / z for k, x in ll.items()}
        best = max(post, key=post.get)
        return best, post[best], post

    def profile(self) -> dict:
        """Opponent summary (cached until new data). Never raises."""
        try:
            if self._cache_ver == self._ver and self._cache is not None:
                return self._cache
            prof = self._build_profile()
            self._cache = prof
            self._cache_ver = self._ver
            return prof
        except Exception:
            self.errors += 1
            return self._default_profile()

    def _build_profile(self) -> dict:
        ALL_S, ALL_P = (0, 1, 2, 3), (0, 1)
        prof = self._block(ALL_S, ALL_P)
        S = self._sum
        n_dec = sum(self.cnt)
        pf = self.pf_n[0] + self.pf_n[1]
        vp = self.vpip_n[0] + self.vpip_n[1]
        pr = self.pfr_n[0] + self.pfr_n[1]
        prof.update({
            "hands": self.hands, "n_dec": n_dec,
            "vpip": _rate(vp, pf, 0.7), "pfr": _rate(pr, pf, 0.3), "n_pf": pf,
            "vpip_btn": _rate(self.vpip_n[POS_BTN], self.pf_n[POS_BTN], 0.7),
            "n_pf_btn": self.pf_n[POS_BTN],
            "vpip_bb": _rate(self.vpip_n[POS_BB], self.pf_n[POS_BB], 0.7),
            "n_pf_bb": self.pf_n[POS_BB],
        })
        n_face = prof["n_face"]
        prof["call_vs_bet"] = _rate(S(ALL_S, ALL_P, (FACE, CAPPED), (CC,)), n_face, 0.45)
        n_open = S((0,), (POS_BTN,), (OPEN,), (FOLD, CC, HALF, POT))
        prof["n_open"] = n_open
        prof["open_fold"] = _rate(S((0,), (POS_BTN,), (OPEN,), (FOLD,)), n_open, 0.2)
        prof["open_call"] = _rate(S((0,), (POS_BTN,), (OPEN,), (CC,)), n_open, 0.4)
        prof["open_raise"] = _rate(S((0,), (POS_BTN,), (OPEN,), (HALF, POT)), n_open, 0.4)
        n_half = S(ALL_S, ALL_P, (FREE, FACE, OPEN), (HALF,))
        n_raises = n_half + S(ALL_S, ALL_P, (FREE, FACE, OPEN), (POT,))
        prof["half_share"] = _rate(n_half, n_raises, 0.5)
        prof["n_raises"] = n_raises
        prof["by_street"] = [self._block((s,), ALL_P) for s in range(4)]
        prof["by_pos"] = {"btn": self._block(ALL_S, (POS_BTN,)),
                          "bb": self._block(ALL_S, (POS_BB,))}
        prof["saw_flop"] = _rate(self.saw_flop_n, self.hands, 0.5)
        prof["n_saw_flop"] = self.saw_flop_n
        prof["wtsd"] = _rate(self.sd_n, self.saw_flop_n, 0.4)
        prof["n_sd"] = self.sd_n
        prof["sd_cat"] = list(self.sd_cat)
        weak_n = self.sd_cat[0] + self.sd_cat[1]
        weak_r = self.sd_cat_raised[0] + self.sd_cat_raised[1]
        strong_n = sum(self.sd_cat[3:])
        strong_r = sum(self.sd_cat_raised[3:])
        prof["sd_weak_raise"] = _rate(weak_r, weak_n, 0.2)
        prof["n_sd_weak"] = weak_n
        prof["sd_strong_raise"] = _rate(strong_r, strong_n, 0.5)
        prof["n_sd_strong"] = strong_n
        prof["my_net_bb"] = self.my_net / self.bb if self.bb else float(self.my_net)

        best, conf, post = self._classify()
        tags = []
        if prof["n_agg"] >= 15:
            if prof["agg"] >= 0.6:
                tags.append("aggressive")
            elif prof["agg"] <= 0.15:
                tags.append("passive")
        if n_face >= 10:
            if prof["fold_to_bet"] >= 0.6:
                tags.append("folds_a_lot")
            elif prof["fold_to_bet"] <= 0.05:
                tags.append("never_folds")
        if pf >= 15:
            if prof["vpip"] >= 0.85:
                tags.append("loose")
            elif prof["vpip"] <= 0.45:
                tags.append("tight")
        if weak_n >= 8 and strong_n >= 5:
            if prof["sd_weak_raise"] >= 0.5 * max(prof["sd_strong_raise"], 0.1) and weak_r >= 3:
                tags.append("bluffer")
            elif prof["sd_weak_raise"] <= 0.1 and prof["sd_strong_raise"] >= 0.4:
                tags.append("value_only")
        cls = best if conf >= CLASS_MIN_CONF else "unknown"
        if cls == "unknown" and n_face >= 15 and prof["fold_to_bet"] >= 0.6 and prof["agg"] <= 0.25:
            cls = "nit"
        prof["cls"] = cls
        prof["best"] = best
        prof["conf"] = conf
        prof["post"] = post
        prof["tags"] = tags
        prof["errors"] = self.errors
        return prof

    def _default_profile(self) -> dict:
        blk = {"agg": 0.3, "n_agg": 0, "fold_to_bet": 0.4, "n_face": 0,
               "raise_vs_bet": 0.15, "n_face_rl": 0, "check_free": 0.6,
               "bet_free": 0.4, "n_free": 0}
        prof = dict(blk)
        prof.update({
            "cls": "unknown", "best": "unknown", "conf": 0.0, "post": {"unknown": 1.0},
            "hands": 0, "n_dec": 0, "vpip": 0.7, "pfr": 0.3, "n_pf": 0,
            "vpip_btn": 0.7, "n_pf_btn": 0, "vpip_bb": 0.7, "n_pf_bb": 0,
            "call_vs_bet": 0.45, "open_fold": 0.2, "open_call": 0.4, "open_raise": 0.4,
            "n_open": 0, "half_share": 0.5, "n_raises": 0,
            "by_street": [dict(blk) for _ in range(4)],
            "by_pos": {"btn": dict(blk), "bb": dict(blk)},
            "saw_flop": 0.5, "n_saw_flop": 0, "wtsd": 0.4, "n_sd": 0, "sd_cat": [0] * 9,
            "sd_weak_raise": 0.2, "n_sd_weak": 0, "sd_strong_raise": 0.5, "n_sd_strong": 0,
            "my_net_bb": 0.0, "tags": [], "errors": self.errors if hasattr(self, "errors") else 0,
        })
        return prof


def get_model(memory: dict, config=None) -> OpponentModel:
    """Return memory['opp'] (create it if missing). Never raises."""
    try:
        m = memory.get("opp")
        if not isinstance(m, OpponentModel):
            m = OpponentModel(config)
            memory["opp"] = m
        return m
    except Exception:
        return OpponentModel(config)
