"""
range_equity.py - postflop equity vs a NARROWED opponent range.

    range_equity(hole, board, opp_top=1.0, post_top=1.0, samples=1500) -> float

Monte Carlo over (runout, opponent hand) like equity.equity, but each sampled
opponent hand is kept only if
  * its preflop strength is in the top `opp_top` fraction (as equity.opp_top), and
  * its CURRENT made-hand strength on `board` (best 2 hole + 3 of the board cards
    dealt so far) is in the top `post_top` fraction of the (preflop-filtered)
    samples.
Use it when a card-aware opponent bets/raises postflop: a bot that bets ~30%
of the time on this street holds roughly the top 30% of hands here.

Only made-hand strength is used for the ranking (draws are not), so callers
should blend the result with the unnarrowed equity as a hedge.

Cost: one pass of n x 100 table lookups for the current board plus one for the
showdown, ~0.5-1.5 ms for n = 1500. Results are cached. Raises ValueError on
bad input (callers wrap it).
"""

from __future__ import annotations

import numpy as np

import equity as E

_RNG = np.random.default_rng(20261108)
_CACHE: dict = {}
_CACHE_MAX = 20000
# Precomputed random rows per board length (positions into the unseen cards).
_ROWS = {L: E._make_rows(31 - L, 5 - L, 4096, _RNG) for L in (3, 4, 5)}
_START = [0]
_TRIPLES4 = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _current_scores(opp: np.ndarray, board: tuple) -> np.ndarray:
    """Best made hand of each opponent (n,5) using 2 hole + 3 of the dealt board."""
    pk, ps = E._pair_keys(opp)                                  # (n,10)
    b = np.array(board, dtype=np.intp)
    if len(board) == 3:
        k = E._CKEY[b]
        s = E._CSUIT[b]
        tk = np.array([k.sum()], dtype=np.int64)
        ts = np.array([s[0] if (s[0] == s[1] == s[2]) else 5])
    elif len(board) == 4:
        tks, tss = [], []
        for t in _TRIPLES4:
            k = E._CKEY[b[list(t)]]
            s = E._CSUIT[b[list(t)]]
            tks.append(k.sum())
            tss.append(s[0] if (s[0] == s[1] == s[2]) else 5)
        tk = np.array(tks, dtype=np.int64)
        ts = np.array(tss)
    else:
        tk, ts = E._trip_keys(b)
    return E._combine(pk, ps, tk[None, :], ts[None, :])


def range_equity(hole, board, opp_top: float = 1.0, post_top: float = 1.0,
                 samples: int = 1500) -> float:
    h, b = E._check(hole, board)
    L = len(b)
    if L not in (3, 4, 5):
        raise ValueError("range_equity needs a flop, turn or river board")
    opp_top = min(1.0, max(0.05, round(float(opp_top) * 20) / 20))
    post_top = min(1.0, max(0.05, round(float(post_top) * 20) / 20))
    n = max(200, min(4096, int(samples)))
    key = (h, b, opp_top, post_top, n)
    v = _CACHE.get(key)
    if v is not None:
        return v

    used = set(h) | set(b)
    unseen = np.array([c for c in range(E.NUM_CARDS) if c not in used], dtype=np.intp)
    rows_all = _ROWS[L]
    span = len(rows_all) - n + 1
    st = _START[0] % span
    _START[0] = (st + 997) % span
    rows = rows_all[st:st + n]
    k = 5 - L
    cards = unseen[rows]
    opp = np.sort(cards[:, k:], axis=1)
    if k:
        full = np.concatenate([np.broadcast_to(np.array(b, dtype=np.intp), (n, L)),
                               cards[:, :k]], axis=1)
    else:
        full = np.broadcast_to(np.array(b, dtype=np.intp), (n, 5))

    keep = np.ones(n, dtype=bool)
    if opp_top < 1.0:
        w = E._weights(opp, opp_top)
        if w is not None:
            keep &= w
    if post_top < 1.0 and keep.sum() >= 20:
        cs = _current_scores(opp, b)
        thr = np.quantile(cs[keep], 1.0 - post_top)
        keep2 = keep & (cs >= thr)
        if keep2.sum() >= 20:
            keep = keep2

    tk, ts = E._trip_keys(full)
    mpk, mps = E._pair_keys(np.array(h, dtype=np.intp))
    my = E._combine(mpk[None, :], mps[None, :], tk, ts)
    opk, ops = E._pair_keys(opp)
    op = E._combine(opk, ops, tk, ts)
    r = (my > op).astype(np.float32) + np.float32(0.5) * (my == op)
    v = float(r[keep].mean())
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = v
    return v
