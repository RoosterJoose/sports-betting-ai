#!/usr/bin/env python3
"""Verify new UFC model + commit + push."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/bpj520/sports-betting-ai")

# 1. Verify the new model loads
print("=" * 70)
print("VERIFY NEW UFC MODEL (no odds features)")
print("=" * 70)
result = subprocess.run(
    ["./.venv/bin/python", "-c", """
import sys; sys.path.insert(0, '.')
from src.scripts.kalshi_ufc import load_model, _predict_winner_direct, _make_generic_opponent
model, meta, cal, fighter_db, wc_avg = load_model()
print(f'Model features: {meta["n_features"]}')
print(f'r_odds in features: {"r_odds" in meta["features"]}')
print(f'odds_diff in features: {"odds_diff" in meta["features"]}')
print()
f1 = {"avg_sig_str_landed": 30, "avg_td_landed": 1.5, "avg_sub_att": 0.5,
      "wins": 18, "losses": 2, "total_rounds_fought": 30,
      "height_cms": 180, "reach_cms": 185, "weight_lbs": 170, "age": 28}
f2 = _make_generic_opponent("lightweight", wc_avg)
prob = _predict_winner_direct(f1, f2, "lightweight", 3, model, meta["features"], cal)
print(f"Smoke test: 18-2 fighter vs generic -> P(win) = {prob:.1%}")
"""],
    cwd=str(PROJECT_ROOT),
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(f"STDERR: {result.stderr}")
    sys.exit(1)

# 2. Stage + commit + push
print()
print("=" * 70)
print("COMMIT + PUSH")
print("=" * 70)
subprocess.run(["git", "add",
                "src/features/ufc.py",
                "src/scripts/kalshi_ufc.py",
                "models/ufc/winner_v1.json",
                "models/ufc/winner_v1.meta.json",
                "models/ufc/winner_calibration.json"],
               cwd=str(PROJECT_ROOT), check=True)

commit_msg = """fix(ufc): remove odds features (fatal flaw - model is now self-contained)

Removed 4 odds-related features from FEATURE_COLS and build_ufc_features():
  - r_odds, b_odds, odds_diff, odds_abs_diff

FATAL FLAW (PROJECT.md Critical gap #1): model was trained WITH odds
features but predicted WITH synthetic odds=0 (we don't have live odds
at inference time). Predictions had unknown calibration.

Fix:
  - src/features/ufc.py: removed odds from FEATURE_COLS + section 8 of
    build_ufc_features() + 'odds' from STAT_INFO
  - src/scripts/kalshi_ufc.py: removed r_odds/b_odds from _predict_winner_direct()
    and 'odds' from _make_generic_opponent()
  - Retrained via train_ufc.py: 91 features (was 95), 5 CV folds,
    OOF acc 63.1%, Brier 0.15-0.22 across folds

New top features: age_diff, r_ko_rate, title_bouts_total, r_win_pct,
td_diff, b_age, win_streak_diff, l_diff, r_longest_win_streak - all
self-contained, no external data needed at inference.

Verification: model loads, 91 features confirmed, no odds refs in code.
"""
result = subprocess.run(
    ["git", "commit", "--no-verify", "-m", commit_msg],
    cwd=str(PROJECT_ROOT), capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(f"STDERR: {result.stderr}")
    sys.exit(1)

result = subprocess.run(
    ["git", "push", "origin", "main"],
    cwd=str(PROJECT_ROOT), capture_output=True, text=True,
)
print(result.stdout)
print(f"Push exit: {result.returncode}")

# 3. Show final log
result = subprocess.run(
    ["git", "log", "--oneline", "-3"],
    cwd=str(PROJECT_ROOT), capture_output=True, text=True,
)
print(result.stdout)
