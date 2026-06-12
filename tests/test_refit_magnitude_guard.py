"""Unit tests for the magnitude guard in the 4 live-refit scripts.

The guard refuses to save BetaCal calibrations where |a| > 3, |b| > 3, or
|c| > 3. These "dangerous-magnitude" calibrators are in the overfit regime
and systematically destroy the model_prob distribution (e.g. NBA pts a=8.06,
ast a=-1.62 — live-fitted values that crushed predictions and produced
negative edges on every pick).

The guard can be bypassed with --force-save.

Tested scripts (all import the same _check_betacal_magnitude helper):
  - scripts/refit_nba_beta_cal_live.py
  - scripts/refit_mlb_beta_cal_live.py
  - scripts/refit_wnba_beta_cal_live.py
  - scripts/refit_nhl_beta_cal_live.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.calibrator import BetaCalibrator


# Helper to import the guard from each of the 4 refit scripts
def _import_guard(sport: str):
    """Import the _check_betacal_magnitude helper from a refit script."""
    if sport == "nba":
        from scripts.refit_nba_beta_cal_live import _check_betacal_magnitude, MAX_PARAM_MAGNITUDE
    elif sport == "mlb":
        from scripts.refit_mlb_beta_cal_live import _check_betacal_magnitude, MAX_PARAM_MAGNITUDE
    elif sport == "wnba":
        from scripts.refit_wnba_beta_cal_live import _check_betacal_magnitude, MAX_PARAM_MAGNITUDE
    elif sport == "nhl":
        from scripts.refit_nhl_beta_cal_live import _check_betacal_magnitude, MAX_PARAM_MAGNITUDE
    else:
        raise ValueError(f"unknown sport: {sport}")
    return _check_betacal_magnitude, MAX_PARAM_MAGNITUDE


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_returns_none_for_safe_params(sport):
    """Normal calibrator (|a|,|b|,|c| < 3) passes the guard."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 1.0
    bc.b = 0.5
    bc.c = 0.2
    # Should return None (safe to save)
    result = guard(bc, "test_stat", force_save=False)
    assert result is None, f"{sport}: expected None for safe params, got {result}"


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_rejects_extreme_a(sport, capsys):
    """Calibrator with |a| > 3 is rejected with a clear CRITICAL message."""
    guard, threshold = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 8.06  # exactly the NBA pts extreme value
    bc.b = 0.5
    bc.c = 0.2
    result = guard(bc, "test_stat", force_save=False)
    assert result is not None, f"{sport}: expected a skip-result, got None"
    assert result.get("skipped") is True
    assert "DANGEROUS MAGNITUDE" in result["reason"]
    assert "a=8.0600" in result["reason"]
    # Should print CRITICAL to stdout
    captured = capsys.readouterr()
    assert "CRITICAL" in captured.out, f"{sport}: missing CRITICAL warning in stdout"
    assert "REFUSED" in captured.out or "Refusing" in captured.out


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_rejects_negative_extreme_a(sport):
    """Calibrator with |a| > 3 (negative direction) is rejected (NBA ast a=-1.62 case)."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = -1.62  # exactly the NBA ast extreme value (still |a| < 3 actually, but the principle)
    # The original NBA ast was a=-1.62 which is BELOW threshold. Test a value that IS extreme.
    bc.a = -3.5
    bc.b = 0.5
    bc.c = 0.2
    result = guard(bc, "test_stat", force_save=False)
    assert result is not None
    assert result.get("skipped") is True
    assert "DANGEROUS MAGNITUDE" in result["reason"]


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_rejects_extreme_b(sport):
    """Calibrator with |b| > 3 is rejected."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 1.0
    bc.b = 4.0  # extreme
    bc.c = 0.2
    result = guard(bc, "test_stat", force_save=False)
    assert result is not None
    assert result.get("skipped") is True
    assert "DANGEROUS MAGNITUDE" in result["reason"]
    assert "b=4.0000" in result["reason"]


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_rejects_extreme_c(sport):
    """Calibrator with |c| > 3 is rejected."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 1.0
    bc.b = 0.5
    bc.c = -5.0  # extreme
    result = guard(bc, "test_stat", force_save=False)
    assert result is not None
    assert result.get("skipped") is True
    assert "DANGEROUS MAGNITUDE" in result["reason"]
    assert "c=-5.0000" in result["reason"]


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_at_threshold_boundary(sport):
    """At exactly |param| = 3.0 the guard should still allow (≤)."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 3.0  # exactly at threshold
    bc.b = 0.5
    bc.c = 0.2
    result = guard(bc, "test_stat", force_save=False)
    assert result is None, f"{sport}: 3.0 should be allowed (≤ threshold)"


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_just_above_threshold(sport):
    """Just above 3.0 (e.g. 3.001) is rejected."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 3.001
    bc.b = 0.5
    bc.c = 0.2
    result = guard(bc, "test_stat", force_save=False)
    assert result is not None
    assert result.get("skipped") is True


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_force_save_bypasses_check(sport, capsys):
    """force_save=True bypasses the guard (no rejection, no CRITICAL message)."""
    guard, _ = _import_guard(sport)
    bc = BetaCalibrator()
    bc.a = 8.06
    bc.b = 0.5
    bc.c = 0.2
    result = guard(bc, "test_stat", force_save=True)
    assert result is None, f"{sport}: force_save should bypass guard"
    # No CRITICAL message printed
    captured = capsys.readouterr()
    assert "CRITICAL" not in captured.out


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_uses_threshold_3(sport):
    """The threshold is exactly 3.0 (matches the project policy)."""
    _, threshold = _import_guard(sport)
    assert threshold == 3.0, f"{sport}: threshold changed from 3.0 to {threshold}"


@pytest.mark.parametrize("sport", ["nba", "mlb", "wnba", "nhl"])
def test_guard_catches_historical_nba_extremes(sport, capsys):
    """Regression test: the historical NBA pts a=8.06 and ast a=-1.62 cases.

    NBA pts a=8.06 is in the dangerous-magnitude regime (|a| > 3) and the
    guard should reject it. The ast a=-1.62 case is BELOW the threshold
    but is still a negative a; verify the guard's behavior is correct.
    """
    guard, _ = _import_guard(sport)

    # NBA pts a=8.06 — should be REJECTED
    bc = BetaCalibrator()
    bc.a = 8.06
    bc.b = -0.64
    bc.c = 3.20
    result = guard(bc, "pts", force_save=False)
    assert result is not None and result.get("skipped") is True
    captured = capsys.readouterr()
    assert "CRITICAL" in captured.out

    # NBA ast a=-1.62 — is below threshold (|a| < 3) so it would NOT be
    # rejected by the magnitude guard. The ast case was a SEPARATE issue
    # (negative-a inversion bug). This test documents that the magnitude
    # guard's scope is |a|>3, not the negative-a bug class.
    bc2 = BetaCalibrator()
    bc2.a = -1.62
    bc2.b = 3.61
    bc2.c = -5.33
    result2 = guard(bc2, "ast", force_save=False)
    # |a|=1.62 is allowed; |b|=3.61 IS extreme (b=3.61 > 3.0); |c|=5.33 IS extreme
    # So this should still be REJECTED due to b and c being extreme.
    assert result2 is not None and result2.get("skipped") is True
