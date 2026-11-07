"""Tunable parameters for tools/train.py: key -> (min, max, step). Booleans are flipped."""

NUMERIC = {
    "maniac_raise_pf": (0.5, 0.8, 0.02), "maniac_raise_post": (0.45, 0.75, 0.02),
    "maniac_bet_pf": (0.5, 0.8, 0.02), "maniac_bet_post": (0.45, 0.75, 0.02),
    "station_raise_pf": (0.35, 0.65, 0.02), "station_raise_post": (0.35, 0.65, 0.02),
    "station_bet_pf": (0.35, 0.65, 0.02), "station_bet_post": (0.35, 0.65, 0.02),
    "random_raise_pf": (0.5, 0.75, 0.02), "random_raise_post": (0.45, 0.7, 0.02),
    "random_bet_pf": (0.5, 0.75, 0.02), "random_bet_post": (0.45, 0.7, 0.02),
    "raise_pf": (0.5, 0.8, 0.02), "raise_post": (0.5, 0.85, 0.02),
    "bet_pf": (0.45, 0.75, 0.02), "bet_post": (0.45, 0.75, 0.02),
    "pot_step_raise": (0.02, 0.2, 0.02), "pot_step_bet": (0.02, 0.2, 0.02),
    "call_margin_rr": (-0.03, 0.06, 0.01),
    "range_hedge": (0.2, 0.9, 0.05), "range_min": (0.05, 0.3, 0.02),
    "range_per_raise": (0.3, 0.9, 0.05), "range_default_freq": (0.15, 0.6, 0.03),
    "size_tell_min": (0.05, 0.4, 0.03),
    "steal_min_n": (8, 40, 4), "steal_edge": (0.0, 0.3, 0.03),
}
BOOLS = ["range_on", "size_on", "steal_on"]
