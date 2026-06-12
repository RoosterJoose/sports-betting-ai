"""Unit tests for src/scripts/oos_test_ufc_mov.py.

Tests the pure-function helpers that have no external dependencies:
  - encode_actual_mov: maps (winner, finish) -> 6-outcome MoV target
  - make_fighter_stats: converts a leak-free rate row into a fighter_stats
    dict consumable by method_of_victory_probabilities()
  - compute_leak_free_career_mov: per-fighter career MoV rates from PRIOR
    fights only (no leakage from the current fight)
"""
import numpy as np
import pandas as pd
import pytest

from src.scripts.oos_test_ufc_mov import (
    encode_actual_mov,
    make_fighter_stats,
    compute_leak_free_career_mov,
    MOV_OUTCOMES,
)


# ── encode_actual_mov ───────────────────────────────────────────────


def test_encode_actual_mov_basic_ko_sub_dec():
    """The 6 base outcomes encode correctly from (winner, finish)."""
    fights = pd.DataFrame({
        "winner": ["Red", "Blue", "Red", "Blue", "Red", "Blue"],
        "finish": ["KO/TKO", "KO/TKO", "SUB", "SUB", "U-DEC", "S-DEC"],
    })
    out = encode_actual_mov(fights).tolist()
    assert out == ["red_ko", "blue_ko", "red_sub", "blue_sub", "red_dec", "blue_dec"]


def test_encode_actual_mov_case_insensitive():
    """Winner and finish are case-insensitive."""
    fights = pd.DataFrame({
        "winner": ["red", "BLUE", "  Red  ", "blue"],
        "finish": ["ko/tko", "SUB", "m-dec", "U-DEC"],
    })
    out = encode_actual_mov(fights).tolist()
    assert out == ["red_ko", "blue_sub", "red_dec", "blue_dec"]


def test_encode_actual_mov_excludes_dq_nan_overturned():
    """DQ, Overturned, NaN, and Draw fights are excluded (return None)."""
    fights = pd.DataFrame({
        "winner": ["Red", "Blue", "Red", "Red", "Draw", "Red"],
        "finish": ["DQ", "Overturned", "", None, "U-DEC", np.nan],
    })
    out = encode_actual_mov(fights).tolist()
    assert out == [None, None, None, None, None, None]


def test_encode_actual_mov_dq_with_ko_keyword_treated_as_dq():
    """If 'DQ' is the finish, it should be excluded even if it contains 'KO' or 'DEC' substrings.

    The check uses str.contains which is substring-based. 'DQ' doesn't contain
    'KO' or 'DEC', but 'KO/TKO DQ' would contain 'KO'. Test that a finish string
    like 'DQ/Overturned' (no KO) is excluded correctly.
    """
    fights = pd.DataFrame({
        "winner": ["Red", "Red"],
        "finish": ["DQ/Overturned", "Overturned"],
    })
    out = encode_actual_mov(fights).tolist()
    assert out == [None, None]


def test_encode_actual_mov_all_decision_variants():
    """U-DEC, S-DEC, M-DEC all map to *_*_dec."""
    fights = pd.DataFrame({
        "winner": ["Red", "Blue", "Red", "Blue"],
        "finish": ["U-DEC", "S-DEC", "M-DEC", "DEC"],
    })
    out = encode_actual_mov(fights).tolist()
    assert out == ["red_dec", "blue_dec", "red_dec", "blue_dec"]


# ── make_fighter_stats ──────────────────────────────────────────────


def test_make_fighter_stats_preserves_rates_to_001():
    """A 40/25/35 rate row should produce a fighter_stats dict that
    compute_mov_rates() reproduces as 40/25/35 (within 1%)."""
    # Build a Series with the leak-free rate columns
    row = pd.Series({
        "r_ko_rate_lf": 0.40,
        "r_sub_rate_lf": 0.25,
        "r_dec_rate_lf": 0.35,
        "b_ko_rate_lf": 0.50,
        "b_sub_rate_lf": 0.20,
        "b_dec_rate_lf": 0.30,
    })
    # Lazy import to avoid pulling src.features into module-level imports
    from src.models.ufc_prop_probabilities import compute_mov_rates

    r_stats = make_fighter_stats(row, "red", "middleweight")
    b_stats = make_fighter_stats(row, "blue", "lightweight")

    # Check the structure: wins=100, ko+sub+dec_uni+dec_split+dec_maj = 100
    assert r_stats["wins"] == 100
    assert r_stats["win_by_ko_tko"] + r_stats["win_by_submission"] + \
        r_stats["win_by_decision_unanimous"] + r_stats["win_by_decision_split"] + \
        r_stats["win_by_decision_majority"] == 100

    # The resulting rates (after compute_mov_rates normalization) should be
    # close to the inputs.
    r_rates = compute_mov_rates(r_stats)
    assert abs(r_rates["ko"] - 0.40) < 0.01
    assert abs(r_rates["sub"] - 0.25) < 0.01
    assert abs(r_rates["dec"] - 0.35) < 0.01

    b_rates = compute_mov_rates(b_stats)
    assert abs(b_rates["ko"] - 0.50) < 0.01
    assert abs(b_rates["sub"] - 0.20) < 0.01
    assert abs(b_rates["dec"] - 0.30) < 0.01


def test_make_fighter_stats_uses_wc_default_when_nan():
    """NaN rates (fighter had no prior wins) should fall back to 40/25/35."""
    row = pd.Series({
        "r_ko_rate_lf": np.nan,
        "r_sub_rate_lf": np.nan,
        "r_dec_rate_lf": np.nan,
    })
    from src.models.ufc_prop_probabilities import compute_mov_rates

    stats = make_fighter_stats(row, "red", "middleweight")
    rates = compute_mov_rates(stats)
    # Default is 40/25/35
    assert abs(rates["ko"] - 0.40) < 0.01
    assert abs(rates["sub"] - 0.25) < 0.01
    assert abs(rates["dec"] - 0.35) < 0.01


def test_make_fighter_stats_corner_prefix_red_vs_blue():
    """Red corner reads r_*_rate_lf, blue corner reads b_*_rate_lf."""
    row = pd.Series({
        "r_ko_rate_lf": 0.10,
        "r_sub_rate_lf": 0.10,
        "r_dec_rate_lf": 0.80,
        "b_ko_rate_lf": 0.70,
        "b_sub_rate_lf": 0.20,
        "b_dec_rate_lf": 0.10,
    })
    from src.models.ufc_prop_probabilities import compute_mov_rates

    r_stats = make_fighter_stats(row, "red", "middleweight")
    b_stats = make_fighter_stats(row, "blue", "middleweight")
    # Red should be KO-poor (decision-heavy), blue should be KO-rich
    r_rates = compute_mov_rates(r_stats)
    b_rates = compute_mov_rates(b_stats)
    assert r_rates["dec"] > 0.70
    assert b_rates["ko"] > 0.60


# ── module-level constants ──────────────────────────────────────────


def test_mov_outcomes_is_six_keys():
    """MOV_OUTCOMES is the canonical 6-key set (order matters for indexing)."""
    assert len(MOV_OUTCOMES) == 6
    assert "red_ko" in MOV_OUTCOMES
    assert "red_sub" in MOV_OUTCOMES
    assert "red_dec" in MOV_OUTCOMES
    assert "blue_ko" in MOV_OUTCOMES
    assert "blue_sub" in MOV_OUTCOMES
    assert "blue_dec" in MOV_OUTCOMES
    # Each corner has 3 method outcomes
    for corner in ["red", "blue"]:
        assert sum(1 for k in MOV_OUTCOMES if k.startswith(corner)) == 3


# ── compute_leak_free_career_mov (leak-freeness) ─────────────────────


def test_compute_leak_free_career_mov_is_leak_free():
    """The function's entire purpose: fight F's outcome must NOT appear in
    F's own leak-free rates. A regression where someone uses
    expanding().sum() instead of shift(1).expanding().sum() would fail this.

    Setup: Fighter A wins fight[0] by KO, then loses fight[1] by KO.
    At fight[1], A's career rate should be 100% KO (from fight[0] only).
    If leak-free is broken, A's rate at fight[1] would be 50% KO (averaged
    with the loss).
    """
    fights = pd.DataFrame({
        "r_fighter": ["Alice", "Bob"],
        "b_fighter": ["Bob", "Alice"],
        "weight_class": ["middleweight", "middleweight"],
        "game_date": pd.to_datetime(["2020-01-01", "2020-06-01"]),
        "winner": ["Red", "Blue"],
        "finish": ["KO/TKO", "KO/TKO"],
    })
    out = compute_leak_free_career_mov(fights)

    # Fight 0: Alice (red) wins by KO. Before this fight, Alice has 0 wins
    # so her rates are NaN. Same for Bob.
    r0 = out.loc[out["r_fighter"] == "Alice"].iloc[0]
    assert pd.isna(r0["r_ko_rate_lf"]), (
        f"Alice should have no prior wins at fight 0, got r_ko_rate_lf={r0['r_ko_rate_lf']!r}"
    )
    assert pd.isna(r0["r_sub_rate_lf"])
    assert pd.isna(r0["r_dec_rate_lf"])

    # Fight 1: Bob (red) wins by KO over Alice. Before this fight:
    # - Bob has 0 wins (he lost fight 0) → NaN
    # - Alice has 1 win (fight 0, by KO) → 100% KO
    r1 = out.loc[out["r_fighter"] == "Bob"].iloc[0]
    assert pd.isna(r1["r_ko_rate_lf"]), (
        f"Bob should have no prior wins at fight 1, got {r1['r_ko_rate_lf']!r}"
    )

    b1 = out.loc[out["b_fighter"] == "Alice"].iloc[0]
    assert b1["b_ko_rate_lf"] == 1.0, (
        f"Alice's leak-free KO rate at fight 1 should be 1.0 (from her fight 0 KO), "
        f"got {b1['b_ko_rate_lf']}. If this is 0.5, the function is leaking fight 1's loss."
    )
    # Alice has 1 prior win (KO), so sub/dec rates are 0/1 = 0.0, NOT NaN
    assert b1["b_sub_rate_lf"] == 0.0, (
        f"Alice's leak-free Sub rate at fight 1 should be 0.0 (1 prior win, 0 subs), "
        f"got {b1['b_sub_rate_lf']}"
    )
    assert b1["b_dec_rate_lf"] == 0.0, (
        f"Alice's leak-free Dec rate at fight 1 should be 0.0 (1 prior win, 0 decs), "
        f"got {b1['b_dec_rate_lf']}"
    )


def test_compute_leak_free_career_mov_cumulative():
    """Verify rates accumulate correctly across multiple fights.

    Setup: Alice wins 4 fights (KO, SUB, DEC, KO), then loses fight[4].
    At fight[4], Alice's leak-free rates should be 50/25/25 (from fights 0-3):
    4 prior wins total, 2 KOs, 1 sub, 1 dec.
    Fight 4's own loss must NOT be included.
    """
    fights = pd.DataFrame({
        "r_fighter": ["Alice", "Alice", "Alice", "Alice", "Bob"],
        "b_fighter": ["Bob", "Carl", "Dee", "Eve", "Alice"],
        "weight_class": ["middleweight"] * 5,
        "game_date": pd.to_datetime([
            "2020-01-01", "2020-06-01", "2020-12-01", "2021-06-01", "2021-12-01"
        ]),
        "winner": ["Red", "Red", "Red", "Red", "Blue"],
        "finish": ["KO/TKO", "SUB", "U-DEC", "KO/TKO", "SUB"],
    })
    out = compute_leak_free_career_mov(fights)

    # Fight 4: Bob (red) wins by SUB over Alice.
    # Alice (blue) at fight 4: 4 prior wins (KO, SUB, DEC, KO) → 2/4=50% KO, 1/4=25% sub, 1/4=25% dec
    b4 = out.loc[out["b_fighter"] == "Alice"].iloc[0]
    assert abs(b4["b_ko_rate_lf"] - 0.50) < 0.01, (
        f"Alice's KO rate at fight 4 should be ~50% (2 KOs out of 4 prior wins), got {b4['b_ko_rate_lf']}"
    )
    assert abs(b4["b_sub_rate_lf"] - 0.25) < 0.01, (
        f"Alice's Sub rate at fight 4 should be ~25% (1 sub out of 4 prior wins), got {b4['b_sub_rate_lf']}"
    )
    assert abs(b4["b_dec_rate_lf"] - 0.25) < 0.01, (
        f"Alice's Dec rate at fight 4 should be ~25% (1 dec out of 4 prior wins), got {b4['b_dec_rate_lf']}"
    )
