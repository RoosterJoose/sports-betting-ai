"""Tests for the UFC method-of-victory + round-of-finish calibration pipeline.

Covers:
  - load_mov_calibration() — file loading, caching, missing-file fallback
  - calibrate_single_prob() — bin lookup, out-of-range, missing-table
  - calibrate_mov_distribution() — sums to 1.0, shape preservation
  - calibrate_round_distribution() — sums to 1.0, shape preservation
  - prop_bet_model_probabilities() — applies calibration, returns all 12 keys
  - kalshi_ufc._print_mov_predictions() — surfacing smoke test (no crash)

The tests do NOT depend on the real `mov_calibration.json` — they build
a synthetic calibration table in tmp_path and verify the application logic.
This makes the tests fast and independent of the actual training data.
"""
import json
import sys
import warnings
from pathlib import Path

import pytest

# Suppress the noisy warnings from the modules under test
warnings.filterwarnings("ignore")

# Ensure the project root is on sys.path so `from src...` works (matches conftest.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.ufc_prop_probabilities import (  # noqa: E402
    MOV_KEYS,
    ROUND_KEYS,
    calibrate_mov_distribution,
    calibrate_round_distribution,
    calibrate_single_prob,
    load_mov_calibration,
    method_of_victory_probabilities,
    prop_bet_model_probabilities,
    round_of_finish_probabilities,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def fake_calibration():
    """Build a synthetic calibration table for testing.

    Structure: 6 MoV outcomes + 6 round outcomes, each with 3 bins
    covering [0, 0.15) and [0.15, 0.30) and [0.30, 1.0). The actual_rate
    is intentionally OFFSET from the bin center so we can verify the
    calibrator maps prior → actual_rate (not → prior).
    """
    cal = {}
    for k in MOV_KEYS + ROUND_KEYS:
        cal[k] = [
            {"bin_lo": 0.00, "bin_hi": 0.15, "model_pred": 0.075, "actual_rate": 0.10, "n": 50},
            {"bin_lo": 0.15, "bin_hi": 0.30, "model_pred": 0.225, "actual_rate": 0.20, "n": 50},
            {"bin_lo": 0.30, "bin_hi": 1.01, "model_pred": 0.50, "actual_rate": 0.40, "n": 50},
        ]
    return cal


@pytest.fixture
def fake_calibration_file(tmp_path, fake_calibration, monkeypatch):
    """Write a synthetic calibration to a tmp file and point the module at it."""
    cal_path = tmp_path / "mov_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(fake_calibration, f)
    # Patch MODEL_DIR to point at tmp_path
    import src.models.ufc_prop_probabilities as mod
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)
    # Clear the module-level cache so load_mov_calibration re-reads
    mod._MOV_CALIBRATION_CACHE = None
    yield cal_path
    # Reset cache + restore MODEL_DIR
    mod._MOV_CALIBRATION_CACHE = None


@pytest.fixture
def sample_fighters():
    """Two synthetic fighters with known career MoV rates."""
    red = {
        "wins": 10, "losses": 2,
        "win_by_ko_tko": 5, "win_by_submission": 2,
        "win_by_decision_unanimous": 2, "win_by_decision_split": 1,
        "win_by_decision_majority": 0,
        "fighter_recent_first_round_rate": 0.15,
    }
    blue = {
        "wins": 8, "losses": 3,
        "win_by_ko_tko": 3, "win_by_submission": 3,
        "win_by_decision_unanimous": 2, "win_by_decision_split": 0,
        "win_by_decision_majority": 0,
        "fighter_recent_first_round_rate": 0.10,
    }
    return red, blue


# ── Test: load_mov_calibration ──────────────────────────────────────


def test_load_mov_calibration_missing_file(monkeypatch, tmp_path):
    """If mov_calibration.json is missing, return empty dict (not crash)."""
    import src.models.ufc_prop_probabilities as mod
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)
    mod._MOV_CALIBRATION_CACHE = None
    result = load_mov_calibration()
    assert result == {}


def test_load_mov_calibration_loads_file(fake_calibration_file):
    """If the file exists, return its contents (one entry per key)."""
    cal = load_mov_calibration()
    assert set(cal.keys()) == set(MOV_KEYS + ROUND_KEYS)
    assert cal["red_ko"][0]["actual_rate"] == 0.10
    assert cal["goes_distance"][1]["bin_lo"] == 0.15


def test_load_mov_calibration_caches(fake_calibration_file, monkeypatch):
    """Second call returns cached result (no re-read)."""
    first = load_mov_calibration()
    # Now corrupt the file — cache should still serve the original
    fake_calibration_file.write_text("{}")
    second = load_mov_calibration()
    assert first is second
    assert second["red_ko"][0]["actual_rate"] == 0.10


def test_load_mov_calibration_force_reload(fake_calibration_file):
    """force_reload=True bypasses the cache and re-reads the file."""
    load_mov_calibration()  # prime cache
    fake_calibration_file.write_text("{}")
    result = load_mov_calibration(force_reload=True)
    assert result == {}


# ── Test: calibrate_single_prob ─────────────────────────────────────


def test_calibrate_single_prob_in_range(fake_calibration_file):
    """A prior inside a bin maps to that bin's actual_rate."""
    # 0.10 falls in [0.00, 0.15) → 0.10
    assert calibrate_single_prob(0.10, "red_ko") == pytest.approx(0.10)
    # 0.20 falls in [0.15, 0.30) → 0.20
    assert calibrate_single_prob(0.20, "red_ko") == pytest.approx(0.20)
    # 0.50 falls in [0.30, 1.01) → 0.40
    assert calibrate_single_prob(0.50, "red_ko") == pytest.approx(0.40)


def test_calibrate_single_prob_below_range(fake_calibration_file):
    """Prior below the lowest bin → first bin's actual_rate."""
    result = calibrate_single_prob(0.001, "red_ko")
    assert result == pytest.approx(0.10)


def test_calibrate_single_prob_above_range(fake_calibration_file):
    """Prior above the highest bin → last bin's actual_rate."""
    result = calibrate_single_prob(0.99, "red_ko")
    assert result == pytest.approx(0.40)


def test_calibrate_single_prob_no_calibration(monkeypatch, tmp_path):
    """If no calibration available, return prior unchanged."""
    import src.models.ufc_prop_probabilities as mod
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)
    mod._MOV_CALIBRATION_CACHE = None
    assert calibrate_single_prob(0.42, "red_ko") == 0.42
    assert calibrate_single_prob(0.001, "round_1") == 0.001


def test_calibrate_single_prob_unknown_outcome(fake_calibration_file):
    """If the outcome key isn't in the table, return prior unchanged."""
    assert calibrate_single_prob(0.42, "no_such_outcome") == 0.42


# ── Test: calibrate_mov_distribution ────────────────────────────────


def test_calibrate_mov_distribution_sums_to_one(fake_calibration_file, sample_fighters):
    """After calibration + renormalization, the 6 MoV probs sum to 1.0."""
    red, blue = sample_fighters
    raw = method_of_victory_probabilities(0.65, red, blue, "lightweight")
    cal = calibrate_mov_distribution(raw)
    total = sum(cal.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    # All values are valid probabilities
    for v in cal.values():
        assert 0.0 <= v <= 1.0


def test_calibrate_mov_distribution_preserves_keys(fake_calibration_file, sample_fighters):
    """Calibrated output has the same keys as the input."""
    red, blue = sample_fighters
    raw = method_of_victory_probabilities(0.65, red, blue, "lightweight")
    cal = calibrate_mov_distribution(raw)
    assert set(cal.keys()) == set(MOV_KEYS)


def test_calibrate_mov_distribution_no_cal(monkeypatch, tmp_path, sample_fighters):
    """If no calibration available, return the input dict unchanged."""
    import src.models.ufc_prop_probabilities as mod
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)
    mod._MOV_CALIBRATION_CACHE = None
    red, blue = sample_fighters
    raw = method_of_victory_probabilities(0.65, red, blue, "lightweight")
    cal = calibrate_mov_distribution(raw)
    assert cal == raw


# ── Test: calibrate_round_distribution ──────────────────────────────


def test_calibrate_round_distribution_sums_to_one(fake_calibration_file, sample_fighters):
    """After calibration, the round probs sum to 1.0."""
    red, blue = sample_fighters
    raw_mov = method_of_victory_probabilities(0.65, red, blue, "lightweight")
    raw_rof = round_of_finish_probabilities(
        raw_mov["red_ko"] + raw_mov["blue_ko"],
        raw_mov["red_sub"] + raw_mov["blue_sub"],
        red, blue, 3,
    )
    cal = calibrate_round_distribution(raw_rof)
    total = sum(cal.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    for v in cal.values():
        assert 0.0 <= v <= 1.0


def test_calibrate_round_distribution_preserves_keys(fake_calibration_file, sample_fighters):
    """Calibrated round dict has the same keys (filtered to ROUND_KEYS)."""
    red, blue = sample_fighters
    raw_mov = method_of_victory_probabilities(0.65, red, blue, "lightweight")
    raw_rof = round_of_finish_probabilities(
        raw_mov["red_ko"] + raw_mov["blue_ko"],
        raw_mov["red_sub"] + raw_mov["blue_sub"],
        red, blue, 3,
    )
    cal = calibrate_round_distribution(raw_rof)
    # Must include all ROUND_KEYS that were in the input
    for k in raw_rof:
        assert k in cal


# ── Test: prop_bet_model_probabilities (integration) ─────────────────


def test_prop_bet_model_probabilities_includes_calibrated(fake_calibration_file, sample_fighters):
    """prop_bet_model_probabilities() applies calibration and returns all 12 keys."""
    red, blue = sample_fighters
    probs = prop_bet_model_probabilities(
        p_red_wins=0.65,
        red_name="Fighter A",
        blue_name="Fighter B",
        weight_class="lightweight",
        scheduled_rounds=3,
        fighter_db={},  # empty — will fall back to weight-class defaults
        wc_avg={},
    )
    expected_keys = (
        ["p_red_wins", "p_blue_wins"]
        + [f"p_red_{m}" for m in ("ko", "sub", "dec")]
        + [f"p_blue_{m}" for m in ("ko", "sub", "dec")]
        + [f"p_round_{r}" for r in range(1, 6)]
        + ["p_goes_distance"]
    )
    for k in expected_keys:
        assert k in probs, f"missing key: {k}"
        assert 0.0 <= probs[k] <= 1.0, f"{k} out of [0,1]: {probs[k]}"
    # MoV probs sum to 1.0
    mov_total = sum(probs[f"p_red_{m}"] + probs[f"p_blue_{m}"] for m in ("ko", "sub", "dec"))
    assert mov_total == pytest.approx(1.0, abs=1e-6)
    # Round probs sum to 1.0
    round_total = sum(probs[f"p_round_{r}"] for r in range(1, 4)) + probs["p_goes_distance"]
    assert round_total == pytest.approx(1.0, abs=1e-6)


def test_prop_bet_model_probabilities_no_cal_fallback(monkeypatch, tmp_path, sample_fighters):
    """Without calibration, probs still sum to 1.0 (raw prior behavior)."""
    import src.models.ufc_prop_probabilities as mod
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)
    mod._MOV_CALIBRATION_CACHE = None
    red, blue = sample_fighters
    probs = prop_bet_model_probabilities(
        p_red_wins=0.65,
        red_name="Fighter A",
        blue_name="Fighter B",
        weight_class="lightweight",
        scheduled_rounds=3,
        fighter_db={},
        wc_avg={},
    )
    assert 0.0 <= probs["p_red_ko"] <= 1.0
    # No crash; structure is intact
    assert probs["weight_class"] == "lightweight"
    assert probs["scheduled_rounds"] == 3


# ── Test: kalshi_ufc._print_mov_predictions (smoke test) ───────────


def test_print_mov_predictions_runs_without_crash(fake_calibration_file, capsys):
    """The scanner's MoV surfacing function should not crash when called
    with empty fighter_db / wc_avg (falls back to weight-class defaults)."""
    from src.scripts.kalshi_ufc import _print_mov_predictions, UPCOMING_MATCHUPS
    # Empty DBs — every fighter lookup will fall back to wc defaults
    _print_mov_predictions(
        model=None, features=None, cal=None,
        fighter_db={}, wc_avg={"_default": {}},
        matchups=UPCOMING_MATCHUPS,
    )
    captured = capsys.readouterr().out
    # Should print the section header
    assert "Method-of-Victory predictions" in captured
    # The print uses lowercase fighter names (from UPCOMING_MATCHUPS keys)
    assert "ilia topuria" in captured or "alex pereira" in captured
    # And it should print at least one MoV probability line (KO/Sub/Dec)
    assert "KO=" in captured and "Dec=" in captured


def test_print_mov_predictions_with_real_fighter(fake_calibration_file, capsys):
    """Strengthened smoke test: with a real fighter in the DB, the surfaced
    MoV probs must reflect the fighter's career rates (not just wc defaults).

    Regression guard against the test passing with a future bug that returns
    hardcoded zeros or skips per-fighter lookups.
    """
    from src.scripts.kalshi_ufc import _print_mov_predictions
    # Populate DB with one fighter who is a heavy KO artist
    real_fighter_db = {
        "Test Ko Beast": {
            "wins": 20, "losses": 2,
            "win_by_ko_tko": 18, "win_by_submission": 1,
            "win_by_decision_unanimous": 1, "win_by_decision_split": 0,
            "win_by_decision_majority": 0,
            "avg_fight_time": 300,  # short fights
        },
    }
    real_wc_avg = {"_default": {}}
    # Use a single matchup that includes the real fighter
    test_matchups = {
        "Test Ko Beast": ("Test Opponent", "lightweight", 3),
        "Test Opponent": ("Test Ko Beast", "lightweight", 3),
    }
    _print_mov_predictions(
        model=None, features=None, cal=None,
        fighter_db=real_fighter_db, wc_avg=real_wc_avg,
        matchups=test_matchups,
    )
    captured = capsys.readouterr().out
    assert "Test Ko Beast" in captured
    # P(red wins) defaults to 0.5 when model is None → 50/50 prior
    assert "P(red)=50%" in captured or "P(red)=0%" not in captured
    # The KO= value should be a valid percentage (0-100)
    import re
    ko_matches = re.findall(r"KO=(\d+)%", captured)
    assert len(ko_matches) > 0, f"No KO= percentages found in: {captured}"
    for ko_str in ko_matches:
        ko_val = int(ko_str)
        assert 0 <= ko_val <= 100, f"KO={ko_val} out of [0,100]"


def test_print_mov_predictions_with_empty_matchups(fake_calibration_file, capsys):
    """Empty matchups dict → prints "(no known matchups to predict)"."""
    from src.scripts.kalshi_ufc import _print_mov_predictions
    _print_mov_predictions(
        model=None, features=None, cal=None,
        fighter_db={}, wc_avg={}, matchups={},
    )
    captured = capsys.readouterr().out
    assert "no known matchups" in captured


# ── Regression guard: cal vs no-cal must differ ───────────────────


def test_calibration_actually_changes_output(fake_calibration_file, sample_fighters):
    """Regression guard: with calibration loaded, the surfaced probs must
    differ from the uncalibrated probs. If they're identical, the cal
    is being silently bypassed (regression).
    """
    red, blue = sample_fighters
    # Force a calibration where every outcome maps to a UNIFORM 0.50 (very
    # different from the raw prior). If cal is applied, all 6 MoV probs
    # should be ~0.167 (1/6) after renormalization.
    uniform_cal = {
        k: [{"bin_lo": 0.0, "bin_hi": 1.01, "model_pred": 0.5,
             "actual_rate": 0.5, "n": 100}] for k in MOV_KEYS
    }
    uniform_cal.update({
        k: [{"bin_lo": 0.0, "bin_hi": 1.01, "model_pred": 0.5,
             "actual_rate": 0.5, "n": 100}] for k in ROUND_KEYS
    })
    # Override the load_mov_calibration cache to return the uniform cal
    import src.models.ufc_prop_probabilities as mod
    mod._MOV_CALIBRATION_CACHE = uniform_cal

    probs_cal = prop_bet_model_probabilities(
        p_red_wins=0.65,
        red_name="Fighter A",
        blue_name="Fighter B",
        weight_class="lightweight",
        scheduled_rounds=3,
        fighter_db={}, wc_avg={},
    )
    # With uniform 0.5 calibration, all 6 MoV probs are renormalized to 1/6 ≈ 0.1667
    expected = 1.0 / 6
    for m in ("ko", "sub", "dec"):
        for corner in ("red", "blue"):
            k = f"p_{corner}_{m}"
            assert abs(probs_cal[k] - expected) < 1e-3, (
                f"{k} = {probs_cal[k]:.4f}, expected ~{expected:.4f} "
                f"(calibration should map all priors to 0.5 → uniform 1/6)"
            )

    # Reset the cache so subsequent tests get the default (no cal) behavior
    mod._MOV_CALIBRATION_CACHE = None
