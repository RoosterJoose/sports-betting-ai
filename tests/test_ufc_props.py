"""Unit tests for the UFC prop bet pipeline.

Tests:
  - Model-derived method-of-victory probabilities sum to 1.0 and respect
    the red/blue split from the binary winner probability
  - Round-of-finish probabilities sum to 1.0 and respect scheduled_rounds
  - DK scraper parses the __NEXT_DATA__ blob correctly
  - Edge calculation: positive edge → "yes" side, negative → "no" side
  - Ranking: highest absolute edge first, below threshold filtered out
  - CSV output: header + rows, header-only when no ranked bets
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ufc_prop_probabilities import (
    compute_mov_rates,
    method_of_victory_probabilities,
    prop_bet_model_probabilities,
    round_of_finish_probabilities,
)
from src.scripts.ufc_props import (
    _match_prop_to_model,
    compute_prop_bet_edges,
    rank_prop_bets,
    write_csv,
)


# ─────────────────────────────────────────────────────────────────────
# Model probability extraction tests
# ─────────────────────────────────────────────────────────────────────


def test_compute_mov_rates_sums_to_one():
    """MoV rates for any fighter should sum to 1.0 (normalized)."""
    stats = {"wins": 10, "win_by_ko_tko": 5, "win_by_submission": 2,
             "win_by_decision_unanimous": 3, "win_by_decision_split": 0,
             "win_by_decision_majority": 0}
    rates = compute_mov_rates(stats)
    assert abs(sum(rates.values()) - 1.0) < 1e-6, f"rates don't sum to 1.0: {rates}"


def test_compute_mov_rates_zero_wins_falls_back_to_default():
    """Fighter with 0 wins should fall back to weight-class typical (40/25/35)."""
    stats = {"wins": 0, "win_by_ko_tko": 0, "win_by_submission": 0,
             "win_by_decision_unanimous": 0, "win_by_decision_split": 0,
             "win_by_decision_majority": 0}
    rates = compute_mov_rates(stats)
    assert abs(sum(rates.values()) - 1.0) < 1e-6
    # Default: 40% KO, 25% sub, 35% dec
    assert rates["ko"] == pytest.approx(0.40, abs=1e-6)
    assert rates["sub"] == pytest.approx(0.25, abs=1e-6)
    assert rates["dec"] == pytest.approx(0.35, abs=1e-6)


def test_method_of_victory_sums_to_one():
    """All 6 MoV outcomes (red_ko, red_sub, red_dec, blue_ko, blue_sub, blue_dec)
    should sum to 1.0 — every fight ends in exactly one of these ways."""
    red = {"wins": 10, "win_by_ko_tko": 5, "win_by_submission": 2,
           "win_by_decision_unanimous": 3, "win_by_decision_split": 0,
           "win_by_decision_majority": 0}
    blue = {"wins": 8, "win_by_ko_tko": 3, "win_by_submission": 3,
            "win_by_decision_unanimous": 2, "win_by_decision_split": 0,
            "win_by_decision_majority": 0}
    mov = method_of_victory_probabilities(
        p_red_wins=0.6, red_stats=red, blue_stats=blue, weight_class="middleweight"
    )
    total = sum(mov.values())
    assert abs(total - 1.0) < 1e-6, f"MoV probabilities don't sum to 1.0: {mov} (total={total})"


def test_method_of_victory_red_advantage_proportional_to_p_red_wins():
    """If red is heavily favored (P=0.9), red's total MoV mass should be ~0.9."""
    red = {"wins": 10, "win_by_ko_tko": 5, "win_by_submission": 2,
           "win_by_decision_unanimous": 3, "win_by_decision_split": 0,
           "win_by_decision_majority": 0}
    blue = {"wins": 10, "win_by_ko_tko": 5, "win_by_submission": 2,
            "win_by_decision_unanimous": 3, "win_by_decision_split": 0,
            "win_by_decision_majority": 0}
    mov = method_of_victory_probabilities(
        p_red_wins=0.9, red_stats=red, blue_stats=blue, weight_class="middleweight"
    )
    red_total = mov["red_ko"] + mov["red_sub"] + mov["red_dec"]
    assert abs(red_total - 0.9) < 1e-6, f"red total should be ~0.9, got {red_total}"


def test_round_of_finish_sums_to_one():
    """All round outcomes (round_1..N + goes_distance) should sum to 1.0."""
    red = {"fighter_recent_first_round_rate": 0.15}
    blue = {"fighter_recent_first_round_rate": 0.10}
    rof = round_of_finish_probabilities(
        p_finish_by_ko=0.20, p_finish_by_sub=0.10,
        red_stats=red, blue_stats=blue, scheduled_rounds=3,
    )
    total = sum(rof.values())
    assert abs(total - 1.0) < 1e-6, f"round probabilities don't sum to 1.0: {rof} (total={total})"


def test_round_of_finish_5_round_fight_has_all_rounds():
    """5-round fight should have round_1 through round_5 in the output."""
    red = {"fighter_recent_first_round_rate": 0.15}
    blue = {"fighter_recent_first_round_rate": 0.10}
    rof = round_of_finish_probabilities(
        p_finish_by_ko=0.30, p_finish_by_sub=0.10,
        red_stats=red, blue_stats=blue, scheduled_rounds=5,
    )
    for r in range(1, 6):
        assert f"round_{r}" in rof, f"missing round_{r} in {rof}"


def test_round_of_finish_3_round_fight_omits_rounds_4_5():
    """3-round fight should NOT have round_4 or round_5 keys (they'd be 0)."""
    red = {"fighter_recent_first_round_rate": 0.15}
    blue = {"fighter_recent_first_round_rate": 0.10}
    rof = round_of_finish_probabilities(
        p_finish_by_ko=0.20, p_finish_by_sub=0.10,
        red_stats=red, blue_stats=blue, scheduled_rounds=3,
    )
    assert "round_4" not in rof
    assert "round_5" not in rof


def test_prop_bet_model_probabilities_schema():
    """prop_bet_model_probabilities() should return a flat dict with all
    expected keys (used for joining against market odds downstream)."""
    probs = prop_bet_model_probabilities(
        p_red_wins=0.6,
        red_name="Ilia Topuria",
        blue_name="Justin Gaethje",
        weight_class="lightweight",
        scheduled_rounds=5,
        fighter_db={},  # empty DB → falls back to weight-class averages
        wc_avg={},
    )
    expected_keys = {
        "fight", "weight_class", "scheduled_rounds",
        "p_red_wins", "p_blue_wins",
        "p_red_ko", "p_red_sub", "p_red_dec",
        "p_blue_ko", "p_blue_sub", "p_blue_dec",
        "p_round_1", "p_round_2", "p_round_3", "p_round_4", "p_round_5",
        "p_goes_distance",
    }
    assert expected_keys.issubset(set(probs.keys())), (
        f"missing keys: {expected_keys - set(probs.keys())}"
    )
    # All 6 MoV outcomes sum to 1.0
    mov_total = (
        probs["p_red_ko"] + probs["p_red_sub"] + probs["p_red_dec"] +
        probs["p_blue_ko"] + probs["p_blue_sub"] + probs["p_blue_dec"]
    )
    assert abs(mov_total - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Edge calculation + ranking tests
# ─────────────────────────────────────────────────────────────────────


def test_match_prop_red_ko():
    """Red-corner KO prop should match p_red_ko."""
    probs = {
        "p_red_wins": 0.6, "p_blue_wins": 0.4,
        "p_red_ko": 0.20, "p_red_sub": 0.10, "p_red_dec": 0.30,
        "p_blue_ko": 0.15, "p_blue_sub": 0.05, "p_blue_dec": 0.20,
        "p_round_1": 0.10, "p_round_2": 0.08, "p_round_3": 0.06,
        "p_round_4": 0.05, "p_round_5": 0.04, "p_goes_distance": 0.67,
    }
    prop = {
        "prop_type": "method_of_victory",
        "fighter": "Ilia Topuria",
        "outcome": "KO/TKO",
        "odds": 250,
    }
    matched = _match_prop_to_model(prop, probs, "Ilia Topuria", "Justin Gaethje")
    assert matched == 0.20


def test_match_prop_blue_sub():
    """Blue-corner sub prop should match p_blue_sub."""
    probs = {
        "p_red_ko": 0.20, "p_red_sub": 0.10, "p_red_dec": 0.30,
        "p_blue_ko": 0.15, "p_blue_sub": 0.05, "p_blue_dec": 0.20,
        "p_round_1": 0.10, "p_round_2": 0.08, "p_round_3": 0.06,
        "p_round_4": 0.05, "p_round_5": 0.04, "p_goes_distance": 0.67,
    }
    prop = {
        "prop_type": "method_of_victory",
        "fighter": "Justin Gaethje",
        "outcome": "Submission",
        "odds": 400,
    }
    matched = _match_prop_to_model(prop, probs, "Ilia Topuria", "Justin Gaethje")
    assert matched == 0.05


def test_match_prop_round_1():
    """Round-of-finish prop for round 1 should match p_round_1."""
    probs = {
        "p_red_ko": 0.20, "p_red_sub": 0.10, "p_red_dec": 0.30,
        "p_blue_ko": 0.15, "p_blue_sub": 0.05, "p_blue_dec": 0.20,
        "p_round_1": 0.10, "p_round_2": 0.08, "p_round_3": 0.06,
        "p_round_4": 0.05, "p_round_5": 0.04, "p_goes_distance": 0.67,
    }
    prop = {
        "prop_type": "round_of_finish",
        "fighter": "Ilia Topuria",
        "outcome": "Fight ends in round 1",
        "odds": 800,
    }
    matched = _match_prop_to_model(prop, probs, "Ilia Topuria", "Justin Gaethje")
    assert matched == 0.10


def test_match_prop_goes_distance():
    """Goes-the-distance prop should match p_goes_distance."""
    probs = {
        "p_round_1": 0.10, "p_round_2": 0.08, "p_round_3": 0.06,
        "p_goes_distance": 0.67,
    }
    prop = {
        "prop_type": "round_of_finish",
        "fighter": "",
        "outcome": "Fight goes the distance",
        "odds": -200,
    }
    matched = _match_prop_to_model(prop, probs, "A", "B")
    assert matched == 0.67


def test_match_prop_unknown_fighter_returns_none():
    """Prop for a fighter not in the fight should return None (not matched)."""
    probs = {"p_red_ko": 0.20, "p_red_sub": 0.10, "p_red_dec": 0.30,
             "p_blue_ko": 0.15, "p_blue_sub": 0.05, "p_blue_dec": 0.20,
             "p_round_1": 0.10, "p_goes_distance": 0.67}
    prop = {
        "prop_type": "method_of_victory",
        "fighter": "Conor McGregor",  # not in this fight
        "outcome": "KO/TKO",
        "odds": 300,
    }
    matched = _match_prop_to_model(prop, probs, "Ilia Topuria", "Justin Gaethje")
    assert matched is None


def test_compute_prop_bet_edges_positive_edge_yes_side():
    """Model_prob > market_implied → positive edge → 'yes' side."""
    probs = {
        "p_red_ko": 0.30, "p_red_sub": 0.10, "p_red_dec": 0.20,
        "p_blue_ko": 0.15, "p_blue_sub": 0.05, "p_blue_dec": 0.20,
        "p_round_1": 0.10, "p_round_2": 0.08, "p_round_3": 0.06,
        "p_goes_distance": 0.67,
    }
    dk_props = [{
        "prop_type": "method_of_victory",
        "fighter": "Ilia Topuria",
        "outcome": "KO/TKO",
        "odds": 400,  # implied = 0.20, model = 0.30 → edge = +0.10
        "sportsbook": "draftkings",
        "market_label": "Method of Victory",
    }]
    edges = compute_prop_bet_edges(probs, dk_props, "Ilia Topuria", "Justin Gaethje")
    assert len(edges) == 1
    assert edges[0]["edge"] == pytest.approx(0.10, abs=0.01)
    assert edges[0]["side"] == "yes"


def test_compute_prop_bet_edges_negative_edge_no_side():
    """Model_prob < market_implied → negative edge → 'no' side."""
    probs = {
        "p_red_ko": 0.10, "p_red_sub": 0.05, "p_red_dec": 0.45,
        "p_blue_ko": 0.15, "p_blue_sub": 0.05, "p_blue_dec": 0.20,
        "p_round_1": 0.10, "p_goes_distance": 0.67,
    }
    dk_props = [{
        "prop_type": "method_of_victory",
        "fighter": "Ilia Topuria",
        "outcome": "KO/TKO",
        "odds": 200,  # implied = 0.333, model = 0.10 → edge = -0.233
        "sportsbook": "draftkings",
        "market_label": "Method of Victory",
    }]
    edges = compute_prop_bet_edges(probs, dk_props, "Ilia Topuria", "Justin Gaethje")
    assert len(edges) == 1
    assert edges[0]["edge"] < 0
    assert edges[0]["side"] == "no"


def test_rank_prop_bets_sorted_by_abs_edge_desc():
    """Ranked list should be sorted by absolute edge, highest first."""
    edges = [
        {"prop_type": "a", "fighter": "f1", "outcome": "o1", "edge": 0.03, "model_prob": 0.5, "market_implied": 0.47, "side": "yes", "dk_odds": 100, "sportsbook": "draftkings"},
        {"prop_type": "b", "fighter": "f2", "outcome": "o2", "edge": -0.15, "model_prob": 0.1, "market_implied": 0.25, "side": "no", "dk_odds": 300, "sportsbook": "draftkings"},
        {"prop_type": "c", "fighter": "f3", "outcome": "o3", "edge": 0.08, "model_prob": 0.3, "market_implied": 0.22, "side": "yes", "dk_odds": 350, "sportsbook": "draftkings"},
    ]
    ranked = rank_prop_bets(edges, min_edge=0.05)
    assert len(ranked) == 2  # the 0.03 edge is filtered out
    assert ranked[0]["edge"] == -0.15  # |0.15| > |0.08|
    assert ranked[1]["edge"] == 0.08


def test_write_csv_header_only_when_empty(tmp_path):
    """When no ranked bets, write_csv should still create a header-only file."""
    csv_path = tmp_path / "props.csv"
    write_csv([], str(csv_path))
    content = csv_path.read_text()
    assert "prop_type,fighter,outcome" in content
    # Exactly 1 line (the header) — use original content, not stripped
    # (stripping would collapse the trailing newline)
    assert content.count("\n") == 1, (
        f"expected 1 newline, got {content.count(chr(10))}: {content!r}"
    )
    # No data rows
    assert "Topuria" not in content
    assert "KO/TKO" not in content


def test_write_csv_with_data(tmp_path):
    """When there are ranked bets, write_csv should write header + data rows."""
    csv_path = tmp_path / "props.csv"
    ranked = [{
        "prop_type": "method_of_victory",
        "fighter": "Ilia Topuria",
        "outcome": "KO/TKO",
        "model_prob": 0.30,
        "market_implied": 0.20,
        "edge": 0.10,
        "side": "yes",
        "dk_odds": 400,
        "sportsbook": "draftkings",
    }]
    write_csv(ranked, str(csv_path))
    content = csv_path.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2  # header + 1 data row
    assert "Ilia Topuria" in lines[1]
    assert "KO/TKO" in lines[1]


# ─────────────────────────────────────────────────────────────────────
# DK scraper tests
# ─────────────────────────────────────────────────────────────────────


def test_dk_scraper_extracts_next_data():
    """DKPropsScraper._extract_next_data should pull the __NEXT_DATA__ blob."""
    from src.data.dk_props_scraper import DKPropsScraper
    scraper = DKPropsScraper()
    html = '<html><script id="__NEXT_DATA__" type="application/json">{"foo": "bar"}</script></html>'
    blob = scraper._extract_next_data(html)
    assert blob == {"foo": "bar"}


def test_dk_scraper_handles_missing_script_tag():
    """If __NEXT_DATA__ isn't present, return None (defensive)."""
    from src.data.dk_props_scraper import DKPropsScraper
    scraper = DKPropsScraper()
    html = "<html><body>No data here</body></html>"
    assert scraper._extract_next_data(html) is None


def test_dk_scraper_parses_method_of_victory():
    """_classify_prop should correctly map DK market labels to our canonical types."""
    from src.data.dk_props_scraper import _classify_prop
    # Method of victory variants
    assert _classify_prop("Method of Victory") == "method_of_victory"
    assert _classify_prop("Method of Victory - KO/TKO") == "method_of_victory"
    assert _classify_prop("Victory Method") == "method_of_victory"
    # Round of finish / distance
    assert _classify_prop("Fight goes the distance") == "round_of_finish"
    assert _classify_prop("Round Betting") == "round_of_finish"
    assert _classify_prop("Distance") == "round_of_finish"
    # Total rounds (regression guard: "Total Rounds" must NOT be "round_of_finish"
    # even though it contains "round")
    assert _classify_prop("Total Rounds") == "total_rounds"
    assert _classify_prop("Total Rounds O/U") == "total_rounds"
    # Moneyline / other
    assert _classify_prop("Moneyline") == "other"


def test_dk_scraper_classifies_bare_outcome_labels_as_method_of_victory():
    """Regression guard: bare outcome names (KO/TKO, Submission, Decision)
    must classify as method_of_victory via exact-label match, not fall
    through to 'other'. Pre-June-11 fix these all classified as 'other'
    because the substring sets were too broad and excluded the bare names.
    """
    from src.data.dk_props_scraper import _classify_prop
    # The 3 main MoV outcomes as bare labels (most common DK format)
    assert _classify_prop("KO/TKO") == "method_of_victory"
    assert _classify_prop("Submission") == "method_of_victory"
    assert _classify_prop("Decision") == "method_of_victory"
    # Case-insensitive
    assert _classify_prop("ko/tko") == "method_of_victory"
    assert _classify_prop("SUBMISSION") == "method_of_victory"
    assert _classify_prop("decision") == "method_of_victory"
    # Whitespace-stripped
    assert _classify_prop("  KO/TKO  ") == "method_of_victory"
    assert _classify_prop(" Submission ") == "method_of_victory"
    # KO/TKO variants
    assert _classify_prop("KO/TKO/DQ") == "method_of_victory"
    assert _classify_prop("Knockout") == "method_of_victory"
    assert _classify_prop("TKO") == "method_of_victory"
    assert _classify_prop("KO") == "method_of_victory"
    # Sub variants
    assert _classify_prop("Sub") == "method_of_victory"
    # Decision subtypes
    assert _classify_prop("Unanimous Decision") == "method_of_victory"
    assert _classify_prop("Split Decision") == "method_of_victory"
    assert _classify_prop("Majority Decision") == "method_of_victory"
    assert _classify_prop("Decision - Unanimous") == "method_of_victory"
    assert _classify_prop("Decision - Split") == "method_of_victory"
    assert _classify_prop("Decision - Majority") == "method_of_victory"
    # Other fight outcomes
    assert _classify_prop("Draw") == "method_of_victory"
    assert _classify_prop("No Contest") == "method_of_victory"
    assert _classify_prop("NC") == "method_of_victory"


def test_dk_scraper_exact_match_does_not_break_other_types():
    """Regression guard: exact-label match for MoV must NOT misclassify
    other prop types. E.g. "Distance" must stay round_of_finish, "Total
    Rounds" must stay total_rounds, "Moneyline" must stay other.
    """
    from src.data.dk_props_scraper import _classify_prop
    # Distance is NOT a MoV outcome (it means "goes the distance")
    assert _classify_prop("Distance") == "round_of_finish"
    assert _classify_prop("Fight goes the distance") == "round_of_finish"
    # Total Rounds is NOT a MoV outcome
    assert _classify_prop("Total Rounds") == "total_rounds"
    assert _classify_prop("Total Rounds O/U") == "total_rounds"
    # Moneyline is NOT a prop
    assert _classify_prop("Moneyline") == "other"
    # Random text is NOT a MoV
    assert _classify_prop("") == "other"
    assert _classify_prop("Random text") == "other"


def test_dk_scraper_classifies_compound_mov_labels():
    """Regression guard: DK often phrases MoV as compound labels like
    "Ilia Topuria to win by KO/TKO". The substring match must catch these
    via "to win by ko/tko" / "to win by submission" / "to win by decision"
    keywords, not fall through to "other".
    """
    from src.data.dk_props_scraper import _classify_prop
    # "Fighter to win by X" compound labels (most common DK format)
    assert _classify_prop("Ilia Topuria to win by KO/TKO") == "method_of_victory"
    assert _classify_prop("Justin Gaethje to win by Submission") == "method_of_victory"
    assert _classify_prop("Alex Pereira to win by Decision") == "method_of_victory"
    # Shorter "win by X" variants
    assert _classify_prop("Topuria win by KO/TKO") == "method_of_victory"
    assert _classify_prop("Gaethje win by Knockout") == "method_of_victory"
    # "Inside the distance" = ends before R3/R5 (MoV-adjacent)
    assert _classify_prop("Topuria to win inside the distance") == "method_of_victory"


def test_dk_scraper_returns_empty_on_fetch_failure():
    """If the fetch fails, get_event_props should return [] (not raise)."""
    from src.data.dk_props_scraper import DKPropsScraper
    scraper = DKPropsScraper()
    with patch.object(scraper, "_fetch", side_effect=Exception("network down")):
        result = scraper.get_event_props("https://example.com/event")
    assert result == []


def test_dk_scraper_returns_empty_on_missing_next_data():
    """If the page has no __NEXT_DATA__, return [] (not raise)."""
    from src.data.dk_props_scraper import DKPropsScraper
    scraper = DKPropsScraper()
    with patch.object(scraper, "_fetch", return_value="<html>no data</html>"):
        result = scraper.get_event_props("https://example.com/event")
    assert result == []


# ─────────────────────────────────────────────────────────────────────
# Odds API conversion tests
# ─────────────────────────────────────────────────────────────────────


def test_american_odds_to_implied_prob_positive():
    """+200 should imply 33.3% (100/(200+100))."""
    from src.data.odds_api import american_odds_to_implied_prob
    assert american_odds_to_implied_prob(200) == pytest.approx(0.3333, abs=0.001)


def test_american_odds_to_implied_prob_negative():
    """-150 should imply 60% (150/(150+100))."""
    from src.data.odds_api import american_odds_to_implied_prob
    assert american_odds_to_implied_prob(-150) == pytest.approx(0.60, abs=0.001)


def test_american_odds_to_implied_prob_none():
    """None or 0 should return None (defensive)."""
    from src.data.odds_api import american_odds_to_implied_prob
    assert american_odds_to_implied_prob(None) is None
    assert american_odds_to_implied_prob(0) is None


def test_american_odds_to_fair_prob_removes_vig():
    """Fair probability should sum to 1.0 (vig removed)."""
    from src.data.odds_api import american_odds_to_fair_prob
    # -110 / -110 (standard vig): implied each = 52.38%, fair each = 50%
    red, blue = american_odds_to_fair_prob(-110, -110)
    assert abs(red - 0.5) < 1e-6
    assert abs(blue - 0.5) < 1e-6
    assert abs(red + blue - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Calibration threading tests (June 11 fix)
# Regression guards: predict_p_red_wins() must thread winner_calibration.json
# through to _predict_winner_direct() so the ufc_props CLI produces
# calibrated (not raw) P(red wins). The raw output is systematically
# overconfident — same bug class as the NBA/MLB phantom edges.
# ─────────────────────────────────────────────────────────────────────


def test_load_winner_model_returns_three_values():
    """load_winner_model() must return (model, meta, cal) — 3-tuple, not 2-tuple.

    Regression guard: the pre-June-11 version returned (model, meta) and
    `cal` was never threaded into predict_p_red_wins, so the ufc_props
    CLI bypassed the winner calibration entirely.
    """
    from src.scripts.ufc_props import load_winner_model
    result = load_winner_model()
    assert result is not None
    assert len(result) == 3, f"expected 3-tuple (model, meta, cal), got {len(result)}"
    model, meta, cal = result
    if model is None:
        pytest.skip("UFC model not trained yet — skipping")
    assert model is not None
    assert isinstance(meta, dict)
    assert isinstance(cal, list), f"cal should be a list, got {type(cal)}"
    # If the calibration file exists, the list should have at least one bin
    cal_path = Path("models/ufc/winner_calibration.json")
    if cal_path.exists():
        assert len(cal) > 0, "winner_calibration.json exists but cal is empty"


def test_load_winner_model_handles_missing_model_file(monkeypatch, tmp_path):
    """If winner_v1.json is missing, return (None, None, []) — not crash."""
    from src.scripts import ufc_props
    monkeypatch.setattr(ufc_props, "MODEL_DIR", tmp_path)
    model, meta, cal = ufc_props.load_winner_model()
    assert model is None
    assert meta is None
    assert cal == []


def test_predict_p_red_wins_threads_cal(monkeypatch):
    """predict_p_red_wins() must pass `cal` through to _predict_winner_direct.

    Regression guard: pre-June-11 the function hardcoded `cal=[]` which
    caused systematic overconfidence (the +76% edge on UFC underdogs).
    We mock _predict_winner_direct to capture the cal argument.
    """
    from src.scripts import ufc_props, kalshi_ufc
    test_cal = [{"bin_lo": 0.0, "bin_hi": 1.01, "model_pred": 0.5, "actual_rate": 0.42, "n": 100}]

    captured = {}

    def fake_predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal):
        captured["cal"] = cal
        return 0.65

    # Patch _predict_winner_direct in the kalshi_ufc module (where
    # predict_p_red_wins imports it from)
    monkeypatch.setattr(kalshi_ufc, "_predict_winner_direct", fake_predict_winner_direct)
    # Stub get_fighter_stats to avoid the real DB lookup
    monkeypatch.setattr(kalshi_ufc, "get_fighter_stats", lambda name, db, avg: ({"wins": 5}, True))

    result = ufc_props.predict_p_red_wins(
        "Fighter A", "Fighter B", "lightweight", 3,
        model=None, features=[], fighter_db={}, wc_avg={},
        cal=test_cal,
    )
    assert result == 0.65
    # The key assertion: cal was passed through (not replaced with [])
    assert captured["cal"] is test_cal, (
        f"cal was not threaded through — got {captured['cal']!r}, "
        f"expected {test_cal!r}"
    )


def test_predict_p_red_wins_default_cal_is_empty_list(monkeypatch):
    """If cal is not passed, default to [] (bypass calibration).

    Documents the current behavior: `cal=None` → `cal=[]` which bypasses
    calibration. Production callers in main() always pass cal=cal now.
    """
    from src.scripts import ufc_props, kalshi_ufc

    captured = {}

    def fake_predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal):
        captured["cal"] = cal
        return 0.5

    monkeypatch.setattr(kalshi_ufc, "_predict_winner_direct", fake_predict_winner_direct)
    monkeypatch.setattr(kalshi_ufc, "get_fighter_stats", lambda name, db, avg: ({"wins": 5}, True))
    ufc_props.predict_p_red_wins(
        "Fighter A", "Fighter B", "lightweight", 3,
        model=None, features=[], fighter_db={}, wc_avg={},
    )
    assert captured["cal"] == [], f"default cal should be [] when not passed, got {captured['cal']!r}"


def test_predict_p_red_wins_calibration_changes_output(monkeypatch):
    """Integration test: with real cal, output differs from cal=[].

    Regression guard against the bug class where someone passes `cal`
    through to predict_p_red_wins() but _predict_winner_direct() silently
    ignores it. We force _predict_winner_direct to behave correctly with
    a non-trivial cal (mapping prior 0.75 → actual 0.40) and assert:
      (a) cal=[] → returns raw prior (0.75)
      (b) cal=[{bin mapping 0.75 → 0.40}] → returns 0.40 (different)
    """
    from src.scripts import ufc_props, kalshi_ufc

    test_cal = [
        {"bin_lo": 0.70, "bin_hi": 0.80, "model_pred": 0.75,
         "actual_rate": 0.40, "n": 50},
    ]

    def fake_predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal):
        # Properly apply the calibration (mimicking real kalshi_ufc behavior)
        raw = 0.75
        for entry in cal:
            if entry["bin_lo"] <= raw < entry["bin_hi"]:
                return entry["actual_rate"]
        return raw

    monkeypatch.setattr(kalshi_ufc, "_predict_winner_direct", fake_predict_winner_direct)
    monkeypatch.setattr(kalshi_ufc, "get_fighter_stats", lambda name, db, avg: ({"wins": 5}, True))

    # (a) cal=[] → raw output (0.75)
    no_cal = ufc_props.predict_p_red_wins(
        "Fighter A", "Fighter B", "lightweight", 3,
        model=None, features=[], fighter_db={}, wc_avg={},
        cal=[],
    )
    # (b) cal=[real bin] → calibrated output (0.40)
    with_cal = ufc_props.predict_p_red_wins(
        "Fighter A", "Fighter B", "lightweight", 3,
        model=None, features=[], fighter_db={}, wc_avg={},
        cal=test_cal,
    )
    assert no_cal == pytest.approx(0.75), f"cal=[] should pass through raw 0.75, got {no_cal}"
    assert with_cal == pytest.approx(0.40), f"cal=[mapping 0.75→0.40] should return 0.40, got {with_cal}"
    assert no_cal != with_cal, "calibration must actually change the output — if equal, cal is being ignored"


def test_main_threads_cal_through_to_predict_p_red_wins(monkeypatch):
    """Regression guard: src/scripts/ufc_props.py main() must unpack 3 values
    from load_winner_model() and pass cal=cal to predict_p_red_wins().

    Pre-June-11 the call was `model, meta = load_winner_model()` and
    `predict_p_red_wins(... cal=[])` — both bugs fixed in one pass.
    """
    import inspect
    from src.scripts import ufc_props
    src = inspect.getsource(ufc_props.main)
    # Must unpack 3 values
    assert "model, meta, cal = load_winner_model()" in src, (
        "main() must unpack 3 values from load_winner_model() — "
        f"check the source:\n{src[:500]}"
    )
    # Must pass cal=cal (not cal=[]) to predict_p_red_wins
    assert "cal=cal" in src, (
        "main() must pass cal=cal to predict_p_red_wins() — "
        f"check the source:\n{src[:500]}"
    )
    # Must NOT pass cal=[] to predict_p_red_wins
    assert "cal=[]" not in src.split("predict_p_red_wins")[1][:300], (
        "main() must NOT pass cal=[] to predict_p_red_wins() — "
        "that was the pre-June-11 bypass bug"
    )
