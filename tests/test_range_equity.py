"""Tests for range_equity.py (postflop equity vs a narrowed opponent range)."""
import random
import time
from itertools import combinations

import numpy as np
import pytest

import equity as E
import range_equity as R
from evaluator import score_five


def _deal(rng, L):
    cards = rng.sample(range(36), 5 + L)
    return tuple(cards[:5]), tuple(cards[5:])


@pytest.mark.parametrize("L", [3, 4, 5])
def test_current_scores_match_bruteforce(L):
    rng = random.Random(L)
    for _ in range(30):
        cards = rng.sample(range(36), 5 + L)
        opp, board = cards[:5], tuple(cards[5:])
        got = int(R._current_scores(np.array([sorted(opp)], dtype=np.intp), board)[0])
        want = max(score_five(list(p) + list(t)) for p in combinations(opp, 2)
                   for t in combinations(board, 3))
        assert got == want


@pytest.mark.parametrize("L", [3, 4, 5])
def test_unrestricted_matches_equity(L):
    rng = random.Random(100 + L)
    for _ in range(10):
        hole, board = _deal(rng, L)
        a = R.range_equity(hole, board, 1.0, 1.0, samples=4096)
        b = E.equity(hole, board, samples=4000)
        assert abs(a - b) < 0.05


def test_narrowing_lowers_medium_hand_and_is_bounded():
    rng = random.Random(7)
    drops = []
    for _ in range(40):
        hole, board = _deal(rng, 3)
        full = R.range_equity(hole, board, 1.0, 1.0)
        nar = R.range_equity(hole, board, 1.0, 0.2)
        assert 0.0 <= nar <= 1.0
        drops.append(full - nar)
    assert np.mean(drops) > 0.05      # top-20% made hands beat us more often


def test_opp_top_and_bad_input():
    hole, board = (0, 5, 10, 15, 20), (1, 6, 11)
    v = R.range_equity(hole, board, opp_top=0.3, post_top=0.5)
    assert 0.0 <= v <= 1.0
    with pytest.raises(ValueError):
        R.range_equity(hole, (), 1.0, 0.5)
    with pytest.raises(ValueError):
        R.range_equity(hole, (0, 1, 2), 1.0, 0.5)   # duplicate card


def test_speed():
    rng = random.Random(3)
    worst = 0.0
    for L in (3, 4, 5):
        for _ in range(20):
            hole, board = _deal(rng, L)
            t = time.perf_counter()
            R.range_equity(hole, board, 0.5, 0.3, samples=1500)
            worst = max(worst, time.perf_counter() - t)
    assert worst < 0.02


def test_varying_sample_counts_do_not_break_slicing():
    rng = random.Random(11)
    for n in (1000, 4096, 200, 4096, 1500, 4096):
        hole, board = _deal(rng, 4)
        v = R.range_equity(hole, board, 1.0, 0.5, samples=n)
        assert 0.0 <= v <= 1.0
