"""Build MoV (method-of-victory) + round-of-finish calibration tables from historical UFC fights.

For each historical fight with an OOF (out-of-fold) prediction from the
TimeSeriesSplit CV, compute the model's prior P(red wins by KO),
P(red wins by sub), P(red wins by dec), P(blue wins by KO), etc., and the
prior round-of-finish distribution. Compare to actual outcomes. Bin by
prior probability (5% bins) and compute actual rate per bin.

Saves to: models/ufc/mov_calibration.json
Structure:
  {
    "red_ko":    [{"bin_lo", "bin_hi", "model_pred", "actual_rate", "n"}, ...],
    "red_sub":   [...],
    "red_dec":   [...],
    "blue_ko":   [...],
    "blue_sub":  [...],
    "blue_dec":  [...],
    "round_1":   [...],
    "round_2":   [...],
    "round_3":   [...],
    "round_4":   [...],
    "round_5":   [...],
    "goes_distance": [...],
  }

Usage:
    python -m src.scripts.train_ufc_mov_cal
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data.ufc import UFCDataSource
from src.features.ufc import build_ufc_features, FEATURE_COLS
from src.models.ufc_prop_probabilities import (
    method_of_victory_probabilities,
    round_of_finish_probabilities,
)

MODEL_DIR = Path("models/ufc")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# All 12 calibration targets (6 MoV + 6 round)
MOV_KEYS = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]
ROUND_KEYS = ["round_1", "round_2", "round_3", "round_4", "round_5", "goes_distance"]
ALL_KEYS = MOV_KEYS + ROUND_KEYS


def outcome_to_mov_key(winner: str, finish) -> str | None:
    """Map the actual fight outcome to one of 6 MoV outcome keys.

    Returns None for unknown outcomes (draw, no contest, missing method, etc.).
    """
    if pd.isna(winner) or pd.isna(finish):
        return None
    finish_str = str(finish).upper()
    is_red = str(winner).strip().lower() == "red"
    if "KO" in finish_str or "TKO" in finish_str:
        return "red_ko" if is_red else "blue_ko"
    if "SUB" in finish_str:
        return "red_sub" if is_red else "blue_sub"
    if "DEC" in finish_str:
        return "red_dec" if is_red else "blue_dec"
    return None


def outcome_to_round_key(finish_round, scheduled_rounds: int) -> str | None:
    """Map the actual fight to one of the 6 round keys.

    Returns None for unknown (e.g., NaN finish_round).
    """
    if pd.isna(finish_round):
        return None
    try:
        r = int(finish_round)
    except (ValueError, TypeError):
        return None
    if 1 <= r <= 5 and r <= scheduled_rounds:
        return f"round_{r}"
    return None


def build_calibration_table(
    priors: np.ndarray, actuals: np.ndarray, bin_width: float = 0.05, min_n: int = 5
) -> list[dict]:
    """Bin prior probs and compute actual rate per bin.

    Args:
        priors: 1D array of prior probabilities (one per fight)
        actuals: 1D array of binary actuals (0/1) — did this outcome happen?
        bin_width: size of each probability bin (default 5%)
        min_n: minimum number of samples to include a bin

    Returns:
        List of dicts: [{"bin_lo", "bin_hi", "model_pred", "actual_rate", "n"}]
    """
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    table = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (priors >= lo) & (priors < hi)
        n = int(mask.sum())
        if n < min_n:
            continue
        actual_rate = float(actuals[mask].mean())
        table.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "model_pred": round(float(priors[mask].mean()), 4),
            "actual_rate": round(actual_rate, 4),
            "n": n,
        })
    return table


def main():
    print("=" * 70)
    print("  UFC MoV + ROUND CALIBRATION BUILDER")
    print("=" * 70)

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n1. Loading UFC dataset (MikeSpa master CSV via UFCDataSource)...")
    ds = UFCDataSource()
    df = ds.fetch_player_game_logs(["all"])
    if df.empty:
        print("   No data loaded. Check data/cache/ufc/ufc-master.csv")
        return
    print(f"   {len(df)} rows loaded")

    # Take unique fights (Red corner rows are the canonical ones)
    if "is_red" in df.columns:
        fights = df[df["is_red"] == 1].copy()
    else:
        fights = df.drop_duplicates(subset=["r_fighter", "game_date"]).copy()
    fights["game_date"] = pd.to_datetime(fights["game_date"], errors="coerce")
    fights = fights.sort_values("game_date").reset_index(drop=True)
    print(f"   {len(fights)} unique fights")
    print(f"   Date range: {fights['game_date'].min().date()} → {fights['game_date'].max().date()}")

    # ── 2. Build features ──────────────────────────────────────────
    print("\n2. Building features...")
    featured = build_ufc_features(fights)
    available = [c for c in FEATURE_COLS if c in featured.columns]
    print(f"   {len(available)} of {len(FEATURE_COLS)} features available")

    if "winner" in featured.columns:
        winner_col = "winner"
    elif "Winner" in featured.columns:
        winner_col = "Winner"
    else:
        print("   No winner column found!")
        return
    featured = featured.dropna(subset=[winner_col]).reset_index(drop=True)
    print(f"   {len(featured)} fights with known winner")

    X = featured[available].fillna(0)
    y = (featured[winner_col].astype(str) == "Red").astype(int).values
    base_rate = float(y.mean())
    print(f"   Base rate (Red wins): {base_rate:.1%}")

    # ── 3. Run TimeSeriesSplit CV to get OOF predictions ───────────
    print("\n3. Running TimeSeriesSplit CV (5 folds, 15% test each)...")
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.15))
    oof_preds = np.full(len(X), np.nan)
    fold_sizes = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=(1 - y[train_idx].mean()) / max(y[train_idx].mean(), 1e-3),
            random_state=42 + fold, eval_metric="logloss",
            early_stopping_rounds=30,
        )
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[test_idx], y[test_idx])],
            verbose=False,
        )
        oof_preds[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        fold_sizes.append(len(test_idx))
        print(f"   Fold {fold}: {len(test_idx)} test fights")

    oof_mask = ~np.isnan(oof_preds)
    n_oof = int(oof_mask.sum())
    print(f"   Total OOF predictions: {n_oof} ({n_oof / len(X):.0%} of dataset)")

    # ── 4. Compute prior MoV + round probs for each OOF fight ─────
    print("\n4. Computing prior MoV + round probs for each OOF fight...")
    indices = np.where(oof_mask)[0]

    # Allocate storage
    priors_mov = {k: np.zeros(n_oof) for k in MOV_KEYS}
    priors_round = {k: np.zeros(n_oof) for k in ROUND_KEYS}
    actuals_mov = {k: np.zeros(n_oof) for k in MOV_KEYS}
    actuals_round = {k: np.zeros(n_oof) for k in ROUND_KEYS}

    skipped = 0
    # ── 4a. Load fighter_lookup for consistent prior computation ────
    # We use fighter_lookup.json (the same lookup prop_bet_model_probabilities
    # uses at inference time) to compute priors, so the calibration table
    # corrects biases in the SAME prior distribution that it will be applied
    # to. Per-row stats from the featured DataFrame would be "more correct"
    # for each historical fight (include the current outcome) but would be
    # a different distribution than what inference sees — the calibration
    # would then be a near-identity (the prior already encodes the answer).
    print("   Loading fighter_lookup.json for prior computation (matches inference)...")
    fighter_db = _load_fighter_lookup()
    wc_avg = _load_wc_averages()
    print(f"   {len(fighter_db)} fighters in lookup")

    for i, idx in enumerate(indices):
        row = featured.iloc[idx]
        p_red = float(oof_preds[idx])

        # Get weight class and scheduled rounds
        weight_class = str(row.get("weight_class", "middleweight")).lower().strip()
        if not weight_class or weight_class == "nan":
            weight_class = "middleweight"
        try:
            scheduled_rounds = int(row.get("no_of_rounds", 3))
        except (ValueError, TypeError):
            scheduled_rounds = 3
        scheduled_rounds = max(1, min(5, scheduled_rounds))

        # Build fighter stats from the lookup (NOT from per-row stats) so the
        # prior distribution matches what inference will compute. Falls back
        # to weight-class defaults via get_fighter_stats if the fighter isn't
        # in the lookup.
        r_name = str(row.get("r_fighter", "")).strip()
        b_name = str(row.get("b_fighter", "")).strip()
        r_stats = _lookup_fighter_stats(r_name, fighter_db, wc_avg)
        b_stats = _lookup_fighter_stats(b_name, fighter_db, wc_avg)

        try:
            mov = method_of_victory_probabilities(p_red, r_stats, b_stats, weight_class)
            rof = round_of_finish_probabilities(
                mov["red_ko"] + mov["blue_ko"],
                mov["red_sub"] + mov["blue_sub"],
                r_stats, b_stats, scheduled_rounds,
            )
        except Exception as e:
            skipped += 1
            continue

        for k in MOV_KEYS:
            priors_mov[k][i] = mov[k]
        for k in ROUND_KEYS:
            priors_round[k][i] = rof.get(k, 0.0)

        # Map actual outcome
        winner = row.get(winner_col, "")
        finish = row.get("finish", row.get("Finish", ""))
        finish_round = row.get("finish_round", row.get("Finish_round", None))

        mov_key = outcome_to_mov_key(winner, finish)
        if mov_key:
            actuals_mov[mov_key][i] = 1.0

        round_key = outcome_to_round_key(finish_round, scheduled_rounds)
        if round_key is not None:
            actuals_round[round_key][i] = 1.0
        else:
            # Could not determine round — assume went to distance (safe default)
            actuals_round["goes_distance"][i] = 1.0

    print(f"   Skipped {skipped} fights (errors during feature build)")
    print(f"   Computed priors/actuals for {n_oof - skipped} fights")

    # ── 5. Build calibration tables ────────────────────────────────
    print("\n5. Building calibration tables (5% bins, min_n=5)...")
    calibration = {}
    print("\n   --- MoV calibration (6 outcomes) ---")
    for k in MOV_KEYS:
        table = build_calibration_table(priors_mov[k], actuals_mov[k])
        calibration[k] = table
        avg_actual = float(actuals_mov[k].mean())
        print(f"   {k:8s}: {len(table):>2d} bins, base rate = {avg_actual:.1%}")

    print("\n   --- Round calibration (6 outcomes) ---")
    for k in ROUND_KEYS:
        table = build_calibration_table(priors_round[k], actuals_round[k])
        calibration[k] = table
        avg_actual = float(actuals_round[k].mean())
        print(f"   {k:14s}: {len(table):>2d} bins, base rate = {avg_actual:.1%}")

    # ── 6. Save ─────────────────────────────────────────────────────
    out_path = MODEL_DIR / "mov_calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n   Calibration saved to {out_path}")

    # ── 7. Print sample calibration table (red_ko + round_1) ─────
    print("\n6. Sample calibration tables:")
    for k in ["red_ko", "blue_ko", "round_1", "goes_distance"]:
        print(f"\n   --- {k} ---")
        print(f"   {'bin':>10s}  {'N':>4s}  {'pred':>6s}  {'actual':>7s}  {'gap':>6s}")
        for entry in calibration.get(k, []):
            gap = entry["actual_rate"] - entry["model_pred"]
            print(f"   {entry['bin_lo']:.0%}-{entry['bin_hi']:.0%}  {entry['n']:>4d}  "
                  f"{entry['model_pred']:>5.1%}  {entry['actual_rate']:>6.1%}  {gap:>+5.1%}")


def _load_fighter_lookup() -> dict:
    """Load fighter_lookup.json built by train_ufc.py."""
    f = MODEL_DIR / "fighter_lookup.json"
    if not f.exists():
        print(f"   WARNING: {f} not found — calibration will use wc defaults")
        return {}
    with open(f) as fh:
        return json.load(fh)


def _load_wc_averages() -> dict:
    """Load wc_averages.json built by train_ufc.py."""
    f = MODEL_DIR / "wc_averages.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        return json.load(fh)


def _lookup_fighter_stats(name: str, fighter_db: dict, wc_avg: dict) -> dict:
    """Look up fighter stats from fighter_db (the inference-time lookup),
    falling back to weight-class defaults if the fighter isn't in the DB.

    Uses src.models.ufc_prop_probabilities.get_fighter_stats for consistency
    with the inference code path.
    """
    from src.models.ufc_prop_probabilities import get_fighter_stats
    stats, _ = get_fighter_stats(name, fighter_db, wc_avg)
    return stats


if __name__ == "__main__":
    main()
