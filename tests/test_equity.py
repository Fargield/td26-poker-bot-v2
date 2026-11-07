"""Pytest suite for bundle/equity.py.

Checks the vectorised evaluator, all-street equity behaviour, input validation,
exact river/turn enumeration, suit-permutation invariance, the (optional)
preflop table, caching and speed budgets.

Card ints are 0..35: rank = c >> 2 (0 = six ... 8 = ace), suit = c & 3.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from itertools import combinations

import numpy as np
import pytest

# Make the bundle directory importable no matter where pytest is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equity as E  # noqa: E402
from evaluator import cards_from_str, omaha_score, warmup  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _evaluator_warmup():
    """Build the evaluator lookup table once (takes a few seconds)."""
    warmup()


SE = lambda n: 0.5 / math.sqrt(n)  # noqa: E731


def _deal(rng: random.Random, board_len: int = 5):
    """Return (hole5, board) of distinct random cards."""
    cards = rng.sample(range(36), 5 + board_len)
    return cards[:5], cards[5:]


# ---------------------------------------------------------------------------
# 1. Vectorised evaluator matches the reference evaluator.
# ---------------------------------------------------------------------------
def test_vectorised_matches_omaha_score():
    rng = random.Random(101)
    n = 100_000
    holes = np.empty((n, 5), dtype=np.intp)
    boards = np.empty((n, 5), dtype=np.intp)
    ref = np.empty(n, dtype=np.int64)

    for i in range(n):
        cards = rng.sample(range(36), 10)
        h, b = cards[:5], cards[5:]
        holes[i] = h
        boards[i] = b
        ref[i] = omaha_score(h, b)

    got = np.asarray(E.omaha_scores(holes, boards))
    mismatches = int(np.count_nonzero(got != ref))
    assert mismatches == 0, f"{mismatches} of {n} rows disagree with omaha_score"


# ---------------------------------------------------------------------------
# 2. Default equity is a probability on every street.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("board_len", [0, 3, 4, 5])
def test_equity_range_all_streets(board_len):
    rng = random.Random(200 + board_len)
    for _ in range(50):
        hole, board = _deal(rng, board_len)
        v = E.equity(hole, board)
        assert isinstance(v, float), f"expected a Python float, got {type(v)!r}"
        assert 0.0 <= v <= 1.0, f"equity out of range on {board_len}-card board: {v}"


# ---------------------------------------------------------------------------
# 3. Bad input raises ValueError.
# ---------------------------------------------------------------------------
def test_bad_input_raises():
    good_hole = [0, 1, 2, 3, 4]
    good_board = [5, 6, 7]

    with pytest.raises(ValueError):  # 4 hole cards
        E.equity([0, 1, 2, 3], good_board)
    with pytest.raises(ValueError):  # duplicate in hole
        E.equity([0, 0, 1, 2, 3], good_board)
    with pytest.raises(ValueError):  # card shared between hole and board
        E.equity(good_hole, [4, 5, 6])
    with pytest.raises(ValueError):  # board of length 2
        E.equity(good_hole, [5, 6])
    with pytest.raises(ValueError):  # card out of range
        E.equity([0, 1, 2, 3, 36], good_board)
    with pytest.raises(ValueError):  # opp_top == 0
        E.equity(good_hole, good_board, opp_top=0)
    with pytest.raises(ValueError):  # samples == 0
        E.equity(good_hole, good_board, samples=0)


# ---------------------------------------------------------------------------
# 4. River exact enumeration matches brute force over C(26, 5) opponents.
# ---------------------------------------------------------------------------
def test_river_exact_vs_bruteforce():
    rng = random.Random(404)
    for _ in range(3):
        hole, board = _deal(rng, 5)
        used = set(hole) | set(board)
        unseen = [c for c in range(36) if c not in used]

        s_me = omaha_score(hole, board)
        total = 0.0
        for opp in combinations(unseen, 5):
            s_op = omaha_score(opp, board)
            if s_me > s_op:
                total += 1.0
            elif s_me == s_op:
                total += 0.5
        brute = total / math.comb(len(unseen), 5)

        exact = E.equity(hole, board, exact=True)
        assert abs(exact - brute) <= 1e-6, (
            f"exact={exact} brute={brute} diff={abs(exact - brute)} "
            f"hole={hole} board={board}"
        )


# ---------------------------------------------------------------------------
# 5. Obvious river nuts and trash.
# ---------------------------------------------------------------------------
def test_river_nuts_and_trash():
    board_nuts = cards_from_str("Ah Kh Qh 7c 6d")
    hole_nuts = cards_from_str("Jh Th 9s 9d 8c")  # royal flush, unbeatable
    assert E.equity(hole_nuts, board_nuts, exact=True) == 1.0

    board_trash = cards_from_str("Ah Kh Qh Jd 9c")
    hole_trash = cards_from_str("6c 6d 7s 7c 8d")
    eq = E.equity(hole_trash, board_trash, exact=True)
    assert eq < 0.3, f"trash river equity should be < 0.3, got {eq}"


# ---------------------------------------------------------------------------
# 6. Default river sampling agrees with exact enumeration.
# ---------------------------------------------------------------------------
def test_river_sampled_vs_exact():
    rng = random.Random(606)
    tol = 4 * SE(E.DEFAULT_SAMPLES[5])
    for i in range(10):
        hole, board = _deal(rng, 5)
        sampled = E.equity(hole, board)
        exact = E.equity(hole, board, exact=True)
        diff = abs(sampled - exact)
        assert diff < tol, (
            f"deal {i}: sampled={sampled} exact={exact} diff={diff} tol={tol}"
        )


# ---------------------------------------------------------------------------
# 7. Turn: default fast path and a fresh 20k-sample run vs exact enumeration.
# ---------------------------------------------------------------------------
def test_turn_vs_exact_enumeration():
    rng = random.Random(707)
    for i in range(5):
        hole, board = _deal(rng, 4)
        used = set(hole) | set(board)
        unseen = [c for c in range(36) if c not in used]
        assert len(unseen) == 27

        exact = sum(
            E.equity(hole, board + [r], exact=True) for r in unseen
        ) / len(unseen)

        fast = E.equity(hole, board)
        fast_tol = 4 * SE(E.DEFAULT_SAMPLES[4])
        assert abs(fast - exact) < fast_tol, (
            f"deal {i}: fast={fast} exact={exact} diff={abs(fast - exact)} "
            f"tol={fast_tol}"
        )

        fresh = E.equity(
            hole, board, samples=20000, rng=np.random.default_rng(i)
        )
        fresh_tol = 4 * SE(20000)
        assert abs(fresh - exact) < fresh_tol, (
            f"deal {i}: fresh={fresh} exact={exact} diff={abs(fresh - exact)} "
            f"tol={fresh_tol}"
        )


# ---------------------------------------------------------------------------
# 8. Flop: default fast path vs a fresh 40k-sample reference.
# ---------------------------------------------------------------------------
def test_flop_fast_vs_fresh():
    rng = random.Random(808)
    tol = 4 * SE(E.DEFAULT_SAMPLES[3]) + 4 * SE(40000)
    for i in range(5):
        hole, board = _deal(rng, 3)
        reference = E.equity(
            hole, board, samples=40000, rng=np.random.default_rng(i)
        )
        fast = E.equity(hole, board)
        diff = abs(fast - reference)
        assert diff < tol, (
            f"deal {i}: fast={fast} reference={reference} diff={diff} tol={tol}"
        )


# ---------------------------------------------------------------------------
# 9. Suit relabelling must not change exact equity (or preflop strength).
# ---------------------------------------------------------------------------
def test_suit_permutation_invariance():
    rng = random.Random(909)
    for i in range(10):
        hole, board = _deal(rng, 5)
        base = E.equity(hole, board, exact=True)
        for _ in range(3):
            perm = rng.sample(range(4), 4)
            hole2 = [(c & ~3) | perm[c & 3] for c in hole]
            board2 = [(c & ~3) | perm[c & 3] for c in board]
            other = E.equity(hole2, board2, exact=True)
            assert abs(base - other) <= 1e-9, (
                f"deal {i}: perm={perm} base={base} permuted={other}"
            )

    if E.PREFLOP_TABLE_LOADED:
        for _ in range(50):
            hole, _ = _deal(rng, 0)
            base = E.preflop_strength(hole)
            perm = rng.sample(range(4), 4)
            hole2 = [(c & ~3) | perm[c & 3] for c in hole]
            other = E.preflop_strength(hole2)
            assert abs(base - other) <= 1e-9, (
                f"preflop perm={perm} base={base} permuted={other}"
            )


# ---------------------------------------------------------------------------
# 10. Preflop table agrees with Monte Carlo and orders aces over junk.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not E.PREFLOP_TABLE_LOADED, reason="preflop table not loaded")
def test_preflop_table_vs_mc():
    rng = random.Random(1010)
    for i in range(8):
        hole, _ = _deal(rng, 0)
        table = E.preflop_strength(hole)
        mc = E.equity(hole, (), samples=20000, rng=np.random.default_rng(i))
        assert abs(table - mc) < 0.025, (
            f"hand {i}: table={table} mc={mc} diff={abs(table - mc)}"
        )

    assert E.preflop_strength(cards_from_str("As Ah Ks Kh Qd")) > E.preflop_strength(
        cards_from_str("6c 7d 8h 9s Jc")
    )

    rng = random.Random(1011)
    for _ in range(200):
        hole, _ = _deal(rng, 0)
        v = E.preflop_strength(hole)
        assert 0.2 <= v <= 0.8, f"preflop strength {v} outside [0.2, 0.8]"


# ---------------------------------------------------------------------------
# 11. Preflop percentile is in [0, 1] and monotone in strength.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not E.PREFLOP_TABLE_LOADED, reason="preflop table not loaded")
def test_preflop_percentile():
    rng = random.Random(1111)
    hands = [_deal(rng, 0)[0] for _ in range(200)]

    strengths = [E.preflop_strength(h) for h in hands]
    percentiles = [E.preflop_percentile(h) for h in hands]
    for p in percentiles:
        assert 0.0 <= p <= 1.0, f"percentile {p} outside [0, 1]"

    order = sorted(range(len(hands)), key=lambda j: strengths[j])
    ordered_p = [percentiles[j] for j in order]
    assert all(
        ordered_p[k] <= ordered_p[k + 1] + 1e-12
        for k in range(len(ordered_p) - 1)
    ), "preflop_percentile is not non-decreasing in preflop_strength"


# ---------------------------------------------------------------------------
# 12. Cached repeated query: identical value, second call < 50 us.
# ---------------------------------------------------------------------------
def test_cache():
    E.clear_cache()
    rng = random.Random(1212)
    hole, board = _deal(rng, 3)

    first = E.equity(hole, board)
    t0 = time.perf_counter()
    second = E.equity(hole, board)
    elapsed = time.perf_counter() - t0

    assert first == second
    assert elapsed < 50e-6, f"cached call took {elapsed * 1e6:.1f} us"


# ---------------------------------------------------------------------------
# 13. opp_top=0.2 must lower the average equity.
# ---------------------------------------------------------------------------
def test_opp_top_lowers_equity():
    rng = random.Random(1313)
    deals = []
    tries = 0
    while len(deals) < 30 and tries < 20000:
        tries += 1
        hole, board = _deal(rng, 3)
        if (not E.PREFLOP_TABLE_LOADED) or E.preflop_strength(hole) > 0.55:
            deals.append((hole, board))

    assert len(deals) == 30, f"only found {len(deals)} qualifying flop deals"

    default_vals = [E.equity(h, b) for h, b in deals]
    top_vals = [E.equity(h, b, opp_top=0.2) for h, b in deals]

    for v in top_vals:
        assert 0.0 <= v <= 1.0, f"opp_top equity out of range: {v}"

    mean_default = sum(default_vals) / len(default_vals)
    mean_top = sum(top_vals) / len(top_vals)
    assert mean_top < mean_default, (
        f"mean(opp_top=0.2)={mean_top} not below mean(default)={mean_default}"
    )


# ---------------------------------------------------------------------------
# 14. Speed budget: 200 default calls per street.
# ---------------------------------------------------------------------------
def test_speed_budget():
    E.clear_cache()
    for board_len in (3, 4, 5):
        rng = random.Random(1400 + board_len)
        deals = [_deal(rng, board_len) for _ in range(200)]

        # First call of the street is a warm-up and is not measured.
        E.equity(*deals[0])

        times = []
        for hole, board in deals[1:]:
            t0 = time.perf_counter()
            E.equity(hole, board)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        worst = max(times)
        assert avg < 1500e-6, (
            f"street {board_len}: average {avg * 1e6:.1f} us exceeds 1500 us"
        )
        assert worst < 50e-3, (
            f"street {board_len}: max {worst * 1e3:.2f} ms exceeds 50 ms"
        )
