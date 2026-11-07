"""
equity.py - fast hand-strength queries for Short Deck 5-card PLO.

Public API (all card arguments are iterables of card ints 0..35, see evaluator.py):

    preflop_strength(hole) -> float
        Equity (win + 0.5 * tie) of the 5 hole cards vs ONE uniformly random
        opponent hand, board still to come. Table lookup (~2 us). Built
        offline by tools/build_preflop.py into data/preflop.npy.
        Falls back to Monte Carlo if the table file is missing.

    preflop_percentile(hole) -> float
        Fraction of all 376,992 starting hands with strictly lower
        preflop_strength (0.0 = worst hand, ~1.0 = best).

    equity(hole, board=(), samples=None, rng=None, opp_top=1.0, exact=False) -> float
        Equity vs one random opponent hand given the board so far
        (len(board) in {0, 3, 4, 5}). Monte Carlo over the unseen cards
        (runout + opponent hand), card removal handled exactly.
          samples : number of (runout, opponent) samples; None = per-street
                    default (DEFAULT_SAMPLES). More samples = slower/more exact.
          rng     : None -> fast path: rows from a precomputed random bank
                    (seeded, deterministic per process). A numpy Generator
                    -> fresh random draws (slower; used for offline builds/tests).
          opp_top : restrict the opponent to the top `opp_top` fraction of
                    starting hands by preflop_strength (1.0 = any hand,
                    0.25 = top 25%). Done by rejection: internally draws
                    ~samples/opp_top candidates (capped), so cost grows.
          exact   : river only: enumerate all C(26,5) = 65,780 opponent hands.
        Results are cached on (hole, board, samples, opp_top, exact) when rng
        is None, so repeated queries within a street are free.

    omaha_scores(holes, boards) -> np.ndarray
        Vectorised equivalent of evaluator.omaha_score: holes (N,5), boards (N,5)
        (or (5,) broadcast) -> (N,) int scores, identical values to omaha_score.

    clear_cache()

Accuracy (std error of the MC estimate) is ~0.5/sqrt(samples) at worst;
see tools/bench_equity.py for measured timings and errors per sample count.

How it is fast:
  * A 5-card hand's score depends only on its rank multiset + flush flag.
    key = sum(5 ** rank) is a unique code for the multiset (each rank occurs
    <= 4 times), so a flat int32 table of 2 * 5**9 entries maps
    key (+ 5**9 if flush) -> the exact evaluator score. No sorting needed.
  * Turn/river: the board is fixed (per river card), so the best score for
    each possible opponent hole PAIR is computed once; an opponent's Omaha
    score is then the max over its 10 pairs (10 lookups instead of 100).

Depends on numpy (present in the tournament base image) and evaluator.score_five.
Import cost: a few tens of ms (table fill + random banks + .npy load).
"""

from __future__ import annotations

import os
import random as _pyrandom
from itertools import combinations, combinations_with_replacement
from math import comb

import numpy as np

from evaluator import score_five

__all__ = [
    "preflop_strength",
    "preflop_percentile",
    "equity",
    "omaha_scores",
    "clear_cache",
    "DEFAULT_SAMPLES",
    "PREFLOP_TABLE_LOADED",
]

NUM_CARDS = 36

# Per-street default sample counts, keyed by len(board).
# Chosen from tools/bench_equity.py so the average query stays well under 1 ms.
DEFAULT_SAMPLES = {0: 1000, 3: 1000, 4: 4000, 5: 4000}

# ---------------------------------------------------------------------------
# Score table: key = sum(5**rank) (+ _KS if flush) -> evaluator score
# ---------------------------------------------------------------------------
_KS = 5 ** 9  # 1,953,125
_TABLE = np.zeros(2 * _KS, dtype=np.int32)


def _fill_table() -> None:
    for ranks in combinations_with_replacement(range(9), 5):
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue
        key = sum(5 ** r for r in ranks)
        # Non-flush version: position i gets suit i % 4. Equal ranks are
        # adjacent (<= 4 of them), so their suits differ; 4 suits appear.
        cards = [(r << 2) | (i % 4) for i, r in enumerate(ranks)]
        _TABLE[key] = score_five(cards)
        if len(set(ranks)) == 5:
            _TABLE[_KS + key] = score_five([r << 2 for r in ranks])


_fill_table()

_CKEY = np.array([5 ** (c >> 2) for c in range(NUM_CARDS)], dtype=np.int32)
_CSUIT = np.array([c & 3 for c in range(NUM_CARDS)], dtype=np.int32)
_KS32 = np.int32(_KS)

_PAIRS5 = list(combinations(range(5), 2))
_TRIPS5 = list(combinations(range(5), 3))
_PI = np.array([p[0] for p in _PAIRS5], dtype=np.intp)
_PJ = np.array([p[1] for p in _PAIRS5], dtype=np.intp)
_TI = np.array([t[0] for t in _TRIPS5], dtype=np.intp)
_TJ = np.array([t[1] for t in _TRIPS5], dtype=np.intp)
_TK = np.array([t[2] for t in _TRIPS5], dtype=np.intp)


def _pair_keys(cards: np.ndarray):
    """cards (..., 5) -> (pair key (...,10), pair suit (...,10); suit 4 = offsuit)."""
    k = _CKEY[cards]
    s = _CSUIT[cards]
    a = s[..., _PI]
    pk = k[..., _PI] + k[..., _PJ]
    ps = np.where(a == s[..., _PJ], a, 4)
    return pk, ps


def _trip_keys(board: np.ndarray):
    """board (..., 5) -> (triple key (...,10), triple suit (...,10); 5 = not monotone)."""
    k = _CKEY[board]
    s = _CSUIT[board]
    a = s[..., _TI]
    tk = k[..., _TI] + k[..., _TJ] + k[..., _TK]
    ts = np.where((a == s[..., _TJ]) & (a == s[..., _TK]), a, 5)
    return tk, ts


def _combine(pk, ps, tk, ts):
    """Best score over pairs x triples. pk/ps (..., P), tk/ts (..., T) -> (...)."""
    idx = pk[..., :, None] + tk[..., None, :]
    idx += (ps[..., :, None] == ts[..., None, :]) * _KS32
    return _TABLE[idx].max(axis=(-1, -2))


def omaha_scores(holes, boards) -> np.ndarray:
    """Vectorised evaluator.omaha_score. holes (N,5) or (5,), boards (N,5) or (5,).
    Returns int32 array of shape (N,) (or 0-d for two 1-D inputs)."""
    h = np.asarray(holes, dtype=np.intp)
    b = np.asarray(boards, dtype=np.intp)
    pk, ps = _pair_keys(h)
    tk, ts = _trip_keys(b)
    return _combine(pk, ps, tk, ts)


# ---------------------------------------------------------------------------
# Colex index of a 5-card set (preflop table index), 0 .. C(36,5)-1
# ---------------------------------------------------------------------------
_NUM_HANDS = comb(NUM_CARDS, 5)  # 376,992
_COLEX = [np.array([comb(c, j + 1) for c in range(NUM_CARDS)], dtype=np.int64) for j in range(5)]
_COLEX_PY = [[comb(c, j + 1) for c in range(NUM_CARDS)] for j in range(5)]


def _colex_py(sorted5) -> int:
    return (_COLEX_PY[0][sorted5[0]] + _COLEX_PY[1][sorted5[1]] + _COLEX_PY[2][sorted5[2]]
            + _COLEX_PY[3][sorted5[3]] + _COLEX_PY[4][sorted5[4]])


def _colex_np(sorted5: np.ndarray) -> np.ndarray:
    """(N,5) ascending cards -> (N,) colex indices."""
    return (_COLEX[0][sorted5[:, 0]] + _COLEX[1][sorted5[:, 1]] + _COLEX[2][sorted5[:, 2]]
            + _COLEX[3][sorted5[:, 3]] + _COLEX[4][sorted5[:, 4]])


# ---------------------------------------------------------------------------
# Preflop table
# ---------------------------------------------------------------------------
_PF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preflop.npy")
_PF: np.ndarray | None = None      # float32 equities, len 376,992
_PF_SORTED: np.ndarray | None = None
_PF_THRESH: dict = {}


def _load_preflop() -> None:
    global _PF
    try:
        raw = np.load(_PF_PATH, allow_pickle=False)
        if raw.shape == (_NUM_HANDS,):
            if raw.dtype == np.uint16:
                _PF = raw.astype(np.float32) / np.float32(65535.0)
            else:
                _PF = raw.astype(np.float32)
    except Exception:
        _PF = None


_load_preflop()
PREFLOP_TABLE_LOADED = _PF is not None


# ---------------------------------------------------------------------------
# Random banks: rows of positions into the sorted list of unseen cards.
# Row layout: [runout positions (k), opponent positions (5, ascending)].
# ---------------------------------------------------------------------------
_BANK_ROWS = 8192
_BANK_RNG = np.random.default_rng(20261107)
_BANKS: dict[int, np.ndarray] = {}   # keyed by len(board)
_PICK = _pyrandom.Random(7)


def _make_rows(n_unseen: int, k: int, n: int, gen: np.random.Generator) -> np.ndarray:
    rows = np.argsort(gen.random((n, n_unseen)), axis=1)[:, : k + 5]
    rows[:, k:] = np.sort(rows[:, k:], axis=1)
    return rows.astype(np.intp)


for _L in (0, 3, 4, 5):
    _BANKS[_L] = _make_rows(31 - _L, 5 - _L, _BANK_ROWS, _BANK_RNG)

# Pair ids among R unseen positions (a < b): id table and pair lists.
def _pair_index(R: int):
    pa, pb = zip(*combinations(range(R), 2))
    pid = np.full((R, R), -1, dtype=np.intp)
    pa = np.array(pa, dtype=np.intp)
    pb = np.array(pb, dtype=np.intp)
    pid[pa, pb] = np.arange(len(pa))
    return pa, pb, pid


_PA26, _PB26, _PID26 = _pair_index(26)
_PA27, _PB27, _PID27 = _pair_index(27)
_NP27 = len(_PA27)  # 351

# River bank: opponent pair ids (B,10).
_RIVER_PIDS = _PID26[_BANKS[5][:, _PI], _BANKS[5][:, _PJ]]
# Turn bank: flat ids river_pos * 351 + pair_id (B,10), plus river pos (B,).
_TURN_RPOS = _BANKS[4][:, 0].copy()
_TURN_FLAT = _TURN_RPOS[:, None] * _NP27 + _PID27[_BANKS[4][:, 1 + _PI], _BANKS[4][:, 1 + _PJ]]
_RIVER_ALL: np.ndarray | None = None      # (65780, 5) all opponent position sets
_RIVER_ALL_PIDS: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CACHE: dict = {}
_CACHE_MAX = 50000


def clear_cache() -> None:
    _CACHE.clear()


def _check(hole, board):
    h = tuple(sorted(int(c) for c in hole))
    b = tuple(int(c) for c in board)
    if len(h) != 5:
        raise ValueError("hole must have 5 cards")
    if len(b) not in (0, 3, 4, 5):
        raise ValueError("board must have 0, 3, 4 or 5 cards")
    used = h + b
    if len(set(used)) != len(used) or min(used) < 0 or max(used) >= NUM_CARDS:
        raise ValueError("cards must be distinct ints in 0..35")
    return h, b


def _opp_threshold(opp_top: float) -> float:
    """Minimum preflop strength of the top `opp_top` fraction of hands."""
    global _PF_SORTED
    t = _PF_THRESH.get(opp_top)
    if t is None:
        if _PF_SORTED is None:
            _PF_SORTED = np.sort(_PF)
        i = int((1.0 - opp_top) * _NUM_HANDS)
        i = min(max(i, 0), _NUM_HANDS - 1)
        t = float(_PF_SORTED[i])
        _PF_THRESH[opp_top] = t
    return t


def _rows_for(L: int, n: int, rng):
    """Return (rows (n, k+5) positions, bank slice start or None)."""
    if rng is None and n <= _BANK_ROWS:
        start = _PICK.randrange(_BANK_ROWS - n + 1)
        return _BANKS[L][start:start + n], start
    gen = rng if rng is not None else np.random.default_rng()
    return _make_rows(31 - L, 5 - L, n, gen), None


def _weights(opp_cards_sorted: np.ndarray, opp_top: float):
    if opp_top >= 1.0 or _PF is None:
        return None
    w = _PF[_colex_np(opp_cards_sorted)] >= _opp_threshold(opp_top)
    if not w.any():
        return None
    return w


def _result(my, opp, w) -> float:
    r = (my > opp).astype(np.float32) + np.float32(0.5) * (my == opp)
    if w is None:
        return float(r.mean())
    return float(r[w].mean())


def _n_draw(samples: int, opp_top: float) -> int:
    if opp_top >= 1.0 or _PF is None:
        return samples
    return min(int(samples / max(opp_top, 1e-3)) + 1, 4 * _BANK_ROWS)


# ---------------------------------------------------------------------------
# Street-specific estimators
# ---------------------------------------------------------------------------
def _eq_generic(h, b, unseen, n, rng, opp_top) -> float:
    """Preflop / flop: sample runout + opponent; 100 + 100 lookups per sample."""
    L = len(b)
    k = 5 - L
    rows, _ = _rows_for(L, n, rng)
    cards = unseen[rows]                       # (n, k+5)
    if L:
        board = np.concatenate([np.broadcast_to(np.array(b, dtype=np.intp), (len(cards), L)),
                                cards[:, :k]], axis=1)
    else:
        board = cards[:, :5]
    opp = cards[:, k:]
    tk, ts = _trip_keys(board)
    mpk, mps = _pair_keys(np.array(h, dtype=np.intp))
    my = _combine(mpk[None, :], mps[None, :], tk, ts)
    opk, ops = _pair_keys(opp)
    op = _combine(opk, ops, tk, ts)
    return _result(my, op, _weights(opp, opp_top))


def _eq_turn(h, b, unseen, n, rng, opp_top) -> float:
    """Turn: all 27 river cards x all 351 unseen pairs scored once, then each
    sample (river, opponent) costs 10 lookups."""
    boards = np.empty((27, 5), dtype=np.intp)
    boards[:, :4] = b
    boards[:, 4] = unseen
    tk, ts = _trip_keys(boards)                                # (27,10)
    mpk, mps = _pair_keys(np.array(h, dtype=np.intp))
    my_r = _combine(mpk[None, :], mps[None, :], tk, ts)        # (27,)
    ca, cb = unseen[_PA27], unseen[_PB27]
    pk = _CKEY[ca] + _CKEY[cb]
    sa = _CSUIT[ca]
    ps = np.where(sa == _CSUIT[cb], sa, 4)
    idx = pk[None, :, None] + tk[:, None, :]                   # (27,351,10)
    idx += (ps[None, :, None] == ts[:, None, :]) * _KS32
    pv = _TABLE[idx].max(axis=2).ravel()                       # (27*351,)
    if rng is None and n <= _BANK_ROWS:
        start = _PICK.randrange(_BANK_ROWS - n + 1)
        flat = _TURN_FLAT[start:start + n]
        rpos = _TURN_RPOS[start:start + n]
        opp_pos = _BANKS[4][start:start + n, 1:]
    else:
        gen = rng if rng is not None else np.random.default_rng()
        rows = _make_rows(27, 1, n, gen)
        rpos = rows[:, 0]
        opp_pos = rows[:, 1:]
        flat = rpos[:, None] * _NP27 + _PID27[opp_pos[:, _PI], opp_pos[:, _PJ]]
    op = pv[flat].max(axis=1)
    my = my_r[rpos]
    w = _weights(unseen[opp_pos], opp_top) if opp_top < 1.0 else None
    return _result(my, op, w)


def _eq_river(h, b, unseen, n, rng, opp_top, exact) -> float:
    """River: score each of the 325 unseen pairs once; opponent = max of 10."""
    global _RIVER_ALL, _RIVER_ALL_PIDS
    bb = np.array(b, dtype=np.intp)
    tk, ts = _trip_keys(bb)                                    # (10,)
    mpk, mps = _pair_keys(np.array(h, dtype=np.intp))
    my = _combine(mpk, mps, tk, ts)                            # scalar
    ca, cb = unseen[_PA26], unseen[_PB26]
    pk = _CKEY[ca] + _CKEY[cb]
    sa = _CSUIT[ca]
    ps = np.where(sa == _CSUIT[cb], sa, 4)
    idx = pk[:, None] + tk[None, :]
    idx += (ps[:, None] == ts[None, :]) * _KS32
    pv = _TABLE[idx].max(axis=1)                               # (325,)
    if exact:
        if _RIVER_ALL is None:
            _RIVER_ALL = np.array(list(combinations(range(26), 5)), dtype=np.intp)
            _RIVER_ALL_PIDS = _PID26[_RIVER_ALL[:, _PI], _RIVER_ALL[:, _PJ]]
        pids, opp_pos = _RIVER_ALL_PIDS, _RIVER_ALL
    elif rng is None and n <= _BANK_ROWS:
        start = _PICK.randrange(_BANK_ROWS - n + 1)
        pids = _RIVER_PIDS[start:start + n]
        opp_pos = _BANKS[5][start:start + n]
    else:
        gen = rng if rng is not None else np.random.default_rng()
        opp_pos = _make_rows(26, 0, n, gen)
        pids = _PID26[opp_pos[:, _PI], opp_pos[:, _PJ]]
    op = pv[pids].max(axis=1)
    w = _weights(unseen[opp_pos], opp_top) if opp_top < 1.0 else None
    return _result(my, op, w)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def equity(hole, board=(), samples=None, rng=None, opp_top: float = 1.0,
           exact: bool = False) -> float:
    """Equity (P(win) + 0.5 P(tie)) of `hole` vs one random opponent hand.

    See the module docstring for parameters. Raises ValueError on bad cards.
    """
    h, b = _check(hole, board)
    L = len(b)
    opp_top = float(opp_top)
    if not 0.0 < opp_top <= 1.0:
        raise ValueError("opp_top must be in (0, 1]")
    n = int(samples) if samples is not None else DEFAULT_SAMPLES[L]
    if n < 1:
        raise ValueError("samples must be >= 1")
    exact = bool(exact) and L == 5
    if L == 0 and opp_top >= 1.0 and samples is None and rng is None and _PF is not None:
        return float(_PF[_colex_py(h)])
    key = None
    if rng is None:
        key = (h, b, n, opp_top, exact)
        v = _CACHE.get(key)
        if v is not None:
            return v
    used = set(h) | set(b)
    unseen = np.array([c for c in range(NUM_CARDS) if c not in used], dtype=np.intp)
    nd = _n_draw(n, opp_top)
    if L == 5:
        v = _eq_river(h, b, unseen, nd, rng, opp_top, exact)
    elif L == 4:
        v = _eq_turn(h, b, unseen, nd, rng, opp_top)
    else:
        v = _eq_generic(h, b, unseen, nd, rng, opp_top)
    if key is not None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = v
    return v


def preflop_strength(hole) -> float:
    """Preflop equity vs a uniformly random opponent hand (table lookup)."""
    h, _ = _check(hole, ())
    if _PF is not None:
        return float(_PF[_colex_py(h)])
    return equity(h, (), samples=DEFAULT_SAMPLES[0])


def preflop_percentile(hole) -> float:
    """Fraction of all starting hands with strictly lower preflop strength."""
    global _PF_SORTED
    h, _ = _check(hole, ())
    if _PF is None:
        # Rough fallback: equity is roughly linear in percentile around 0.5.
        return min(max((preflop_strength(h) - 0.35) / 0.3, 0.0), 1.0)
    if _PF_SORTED is None:
        _PF_SORTED = np.sort(_PF)
    v = _PF[_colex_py(h)]
    return float(np.searchsorted(_PF_SORTED, v, side="left")) / _NUM_HANDS
