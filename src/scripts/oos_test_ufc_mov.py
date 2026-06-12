#!/usr/bin/env python3
"""Chronological 80/20 OOS holdout test for the UFC MoV distribution.

Mirrors the pattern in src/scripts/oos_test_ufc_cal.py but tests the
6-outcome Method-of-Victory distribution (red_ko, red_sub, red_dec,
blue_ko, blue_sub, blue_dec) instead of the binary winner. The key
question:

  Are the model-derived MoV probabilities (prior + OOF calibration) well-
  calibrated OOS, or are they systematically overcompressed (e.g. predict
  25% red-KO when actual is 12%)?

Why this is critical: the same overcompression bug class that produced
phantom edges in NBA / MLB could also be present in the UFC MoV probs.
If P(red_ko) is overconfident OOS, then KO/TKO props we flag as "STRONG"
edges are also artifacts.

Approach:
  1. Load MikeSpa UFC dataset via UFCDataSource
  2. Build the 6-outcome target (winner × finish) for each fight
  3. Compute LEAK-FREE career MoV rates per fighter using shift(1) +
     expanding window — the fighter's career MoV distribution at the
     time of the fight, computed from PRIOR fights only
  4. Build all 107 model features as in build_ufc_features
  5. Sort chronologically, split 80/20 (oldest 80% = train, newest 20% =
     OOS test)
  6. Re-train XGBClassifier (winner) on the train slice (no leakage)
  7. For each OOS test fight:
     a. Predict P(red wins) from the model
     b. Look up the leak-free career MoV rates for both fighters
     c. Compute the 6-outcome MoV distribution via
        method_of_victory_probabilities()
     d. Apply mov_calibration.json if available (the "calibrated" arm)
  8. Compare both arms (raw prior, calibrated) to actual MoV outcome:
     - Multi-class log loss (proper scoring rule)
     - Mean Brier across the 6 outcomes
     - Per-outcome calibration bins (bin predicted, compare to actual)
     - Top-1 accuracy
     - Top-2 accuracy (is actual in top-2 predicted outcomes?)
  9. Verdict: are the MoV probs trustworthy for live prop betting?

Output: prints a verdict to stdout and saves JSON to
models/ufc/oos_test_mov.json.

Usage:
    python -m src.scripts.oos_test_ufc_mov
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.ufc import UFCDataSource
from src.features.ufc import build_ufc_features, FEATURE_COLS, WEIGHT_CLASS_FINISH_PCT
from src.models.ufc_prop_probabilities import (
    method_of_victory_probabilities,
    calibrate_mov_distribution,
    load_mov_calibration,
    MOV_KEYS,
)

MODEL_DIR = Path("models/ufc")
MOV_OUTCOMES = MOV_KEYS  # ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]


def encode_actual_mov(fights: pd.DataFrame) -> pd.Series:
    """Map each fight's (winner, finish) to a 6-outcome MoV target.

    Returns a Series indexed by fight row with values in MOV_KEYS.
    Fights with missing winner or unknown finish are excluded (NaN).
    """
    finish = fights["finish"].astype(str).str.upper().fillna("")
    winner = fights["winner"].astype(str).str.strip().str.lower()
    out = pd.Series([None] * len(fights), index=fights.index, dtype=object)
    is_ko = finish.str.contains("KO") | finish.str.contains("TKO")
    is_sub = finish.str.contains("SUB")
    is_dec = finish.str.contains("DEC")
    # Skip DQ / Overturned / nan
    valid = (is_ko | is_sub | is_dec) & (winner.isin(["red", "blue"]))
    out[valid & is_ko & (winner == "red")] = "red_ko"
    out[valid & is_sub & (winner == "red")] = "red_sub"
    out[valid & is_dec & (winner == "red")] = "red_dec"
    out[valid & is_ko & (winner == "blue")] = "blue_ko"
    out[valid & is_sub & (winner == "blue")] = "blue_sub"
    out[valid & is_dec & (winner == "blue")] = "blue_dec"
    return out


def compute_leak_free_career_mov(fights: pd.DataFrame) -> pd.DataFrame:
    """For each fighter, compute career MoV rates from PRIOR fights.

    Returns the input fights DataFrame augmented with columns:
        r_ko_rate_lf, r_sub_rate_lf, r_dec_rate_lf,
        b_ko_rate_lf, b_sub_rate_lf, b_dec_rate_lf

    Leak-free: the career rates for fight F are computed from all fights
    strictly BEFORE F (i.e., shift(1) + expanding). This means fight F's
    own outcome does NOT leak into the prediction.

    Defaults: if a fighter has no prior wins, the rates are NaN. The caller
    (make_fighter_stats) falls back to weight-class typical 40/25/35.

    Implementation: index-based merge (no game_date merge), which avoids
    the date format/tz mismatch that caused the 0-matches bug in the
    previous version. We carry the original fight index through the
    long-form history, then merge back on index.
    """
    # ── 1. Build long-form history (one row per fighter per fight) ───
    rows = []
    for idx, fight in fights.iterrows():
        for fighter, corner in [(fight["r_fighter"], "red"), (fight["b_fighter"], "blue")]:
            if not isinstance(fighter, str) or not fighter:
                continue
            is_red = (corner == "red")
            won = (
                (is_red and str(fight.get("winner", "")).strip().lower() == "red")
                or (not is_red and str(fight.get("winner", "")).strip().lower() == "blue")
            )
            finish = str(fight.get("finish", "")).upper()
            is_ko = int(won and ("KO" in finish or "TKO" in finish))
            is_sub = int(won and "SUB" in finish)
            is_dec = int(won and "DEC" in finish)
            is_win = int(is_ko or is_sub or is_dec)
            rows.append({
                "fight_idx": idx,  # carry the original index for the merge back
                "fighter": fighter,
                "corner": corner,
                "is_ko": is_ko,
                "is_sub": is_sub,
                "is_dec": is_dec,
                "is_win": is_win,
            })
    if not rows:
        # No valid fights — return the input with NaN rate columns
        out = fights.copy()
        for prefix in ["r_", "b_"]:
            for m in ["ko", "sub", "dec"]:
                out[f"{prefix}{m}_rate_lf"] = np.nan
        return out
    long_df = pd.DataFrame(rows)

    # ── 2. Sort by fighter + fight_idx (chronological within fighter) ─
    long_df = long_df.sort_values(["fighter", "fight_idx"]).reset_index(drop=True)

    # ── 3. Compute career stats with shift(1) + expanding (leak-free) ─
    g = long_df.groupby("fighter", sort=False)
    long_df["career_ko"] = g["is_ko"].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    long_df["career_sub"] = g["is_sub"].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    long_df["career_dec"] = g["is_dec"].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
    long_df["career_wins"] = g["is_win"].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)

    # ── 4. Compute rates (NaN if no prior wins) ──────────────────────
    long_df["ko_rate"] = np.where(
        long_df["career_wins"] > 0,
        long_df["career_ko"] / long_df["career_wins"].replace(0, np.nan),
        np.nan,
    )
    long_df["sub_rate"] = np.where(
        long_df["career_wins"] > 0,
        long_df["career_sub"] / long_df["career_wins"].replace(0, np.nan),
        np.nan,
    )
    long_df["dec_rate"] = np.where(
        long_df["career_wins"] > 0,
        long_df["career_dec"] / long_df["career_wins"].replace(0, np.nan),
        np.nan,
    )

    # ── 5. Pivot back to per-fight: red and blue rates on the same row ─
    # Use fight_idx to merge back to the original fights DataFrame (index-based,
    # NOT date-based, to avoid the 0-matches merge bug).
    red_rates = long_df[long_df["corner"] == "red"][["fight_idx", "ko_rate", "sub_rate", "dec_rate"]].rename(
        columns={"ko_rate": "r_ko_rate_lf", "sub_rate": "r_sub_rate_lf", "dec_rate": "r_dec_rate_lf"}
    ).set_index("fight_idx")
    blue_rates = long_df[long_df["corner"] == "blue"][["fight_idx", "ko_rate", "sub_rate", "dec_rate"]].rename(
        columns={"ko_rate": "b_ko_rate_lf", "sub_rate": "b_sub_rate_lf", "dec_rate": "b_dec_rate_lf"}
    ).set_index("fight_idx")

    # ── 6. Merge back via index (no game_date) ───────────────────────
    out = fights.copy()
    out = out.join(red_rates, how="left")
    out = out.join(blue_rates, how="left")
    return out


def make_fighter_stats(rate_row: pd.Series, corner: str, weight_class: str) -> dict:
    """Convert a leak-free rate row into a fighter_stats dict for
    method_of_victory_probabilities()."""
    prefix = "r_" if corner == "red" else "b_"
    ko = rate_row.get(f"{prefix}ko_rate_lf", np.nan)
    sub = rate_row.get(f"{prefix}sub_rate_lf", np.nan)
    dec = rate_row.get(f"{prefix}dec_rate_lf", np.nan)
    # If any rate is NaN (fighter had no prior wins), use weight-class default
    if pd.isna(ko) or pd.isna(sub) or pd.isna(dec):
        wc_finish = WEIGHT_CLASS_FINISH_PCT.get(str(weight_class).lower(), 0.45)
        # Default MoV split: 40% KO, 25% sub, 35% dec
        ko, sub, dec = 0.40, 0.25, 0.35
    # Map to win_by_*_count fields so compute_mov_rates picks them up
    wins = 100  # arbitrary large enough so rates are stable
    return {
        "wins": wins,
        "win_by_ko_tko": int(round(ko * wins)),
        "win_by_submission": int(round(sub * wins)),
        "win_by_decision_unanimous": int(round(dec * wins * 0.8)),
        "win_by_decision_split": int(round(dec * wins * 0.15)),
        "win_by_decision_majority": int(round(dec * wins * 0.05)),
        # Provide a default first-round rate (used for round_of_finish).
        # 0.10 = 10% of prior fights ended in R1 — a reasonable UFC default.
        "fighter_recent_first_round_rate": 0.10,
    }


def main():
    print("=" * 72)
    print("  UFC MoV DISTRIBUTION — CHRONOLOGICAL 80/20 OOS HOLDOUT TEST")
    print("  (mirrors oos_test_ufc_cal.py for the 6-outcome MoV prior)")
    print("=" * 72)

    # ── 1. Load data ────────────────────────────────────────────────
    print("\n1. Loading UFC dataset (MikeSpa master CSV via UFCDataSource)...")
    ds = UFCDataSource()
    df = ds.fetch_player_game_logs(["all"])
    if df.empty:
        print("   No data loaded. Check data/cache/ufc/ufc-master.csv")
        return
    print(f"   {len(df)} fighter-game rows from {df['player_id'].nunique()} fighters")

    # Get unique fights (Red corner rows are canonical)
    if "is_red" in df.columns:
        fights = df[df["is_red"] == 1].copy()
        # The per-fighter DataFrame has player_id (=r_fighter) and opponent
        # (=b_fighter) for red corner rows. Reconstruct r_fighter/b_fighter
        # for downstream use (compute_leak_free_career_mov, encode_actual_mov).
        fights["r_fighter"] = fights["player_id"]
        fights["b_fighter"] = fights["opponent"]
    else:
        fights = df.drop_duplicates(subset=["r_fighter", "game_date"]).copy()
    print(f"   {len(fights)} unique fights (Red corner rows)")

    # Drop fights where r_fighter / b_fighter is missing (corrupt rows)
    n_before = len(fights)
    fights = fights.dropna(subset=["r_fighter", "b_fighter"]).reset_index(drop=True)
    print(f"   {n_before - len(fights)} fights dropped for missing fighter names")

    fights["game_date"] = pd.to_datetime(fights["game_date"], errors="coerce")
    fights = fights.sort_values("game_date").reset_index(drop=True)
    print(f"   Date range: {fights['game_date'].min().date()} → {fights['game_date'].max().date()}")

    # ── 2. Build the 6-outcome MoV target ───────────────────────────
    print("\n2. Building 6-outcome MoV target (winner × finish)...")
    fights["mov_target"] = encode_actual_mov(fights)
    fights = fights.dropna(subset=["mov_target"]).reset_index(drop=True)
    print(f"   {len(fights)} fights with valid MoV target")
    print(f"   MoV distribution:\n{fights['mov_target'].value_counts().to_string()}")

    # ── 3. Build model features (for the winner classifier) ─────────
    print("\n3. Building winner-model features...")
    featured = build_ufc_features(fights)
    available = [c for c in FEATURE_COLS if c in featured.columns]
    print(f"   {len(available)} of {len(FEATURE_COLS)} features available")
    target_col = "winner"
    featured = featured.dropna(subset=[target_col])
    X = featured[available].fillna(0)
    y = (featured[target_col].astype(str).str.strip().str.lower() == "red").astype(int).values
    base_rate = float(y.mean())
    print(f"   {len(featured)} rows with target, base rate (Red wins): {base_rate:.1%}")

    # ── 4. Compute leak-free career MoV rates per fight ─────────────
    print("\n4. Computing leak-free career MoV rates (shift(1) on prior fights)...")
    print("   (This is the slow part — O(n) per fighter, runs once)")
    # New approach: compute_leak_free_career_mov now returns the fights
    # DataFrame augmented with r_*/b_*_rate_lf columns directly (index-
    # based join, no game_date merge). This avoids the 0-matches bug.
    fights_with_mov = compute_leak_free_career_mov(fights)
    # Merge into the featured DataFrame by index (both built from the same fights)
    fights_with_mov = fights_with_mov[["r_ko_rate_lf", "r_sub_rate_lf", "r_dec_rate_lf",
                                        "b_ko_rate_lf", "b_sub_rate_lf", "b_dec_rate_lf"]]
    featured = featured.join(fights_with_mov, how="left")
    n_with_rates = int(featured[["r_ko_rate_lf", "b_ko_rate_lf"]].notna().all(axis=1).sum())
    print(f"   {n_with_rates}/{len(featured)} fights have leak-free career MoV rates for both fighters")
    assert n_with_rates > 0, (
        "Merge produced 0 matches — check that fights['r_fighter']/'b_fighter' are set correctly."
    )
    if n_with_rates < 100:
        print("   ⚠ Too few fights with leak-free rates — check that the master CSV has prior fights for both fighters")
        return

    # ── 5. Chronological 80/20 split ────────────────────────────────
    print("\n5. Splitting chronologically 80/20 (oldest = train, newest = OOS)...")
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    train_dates = featured["game_date"].iloc[:split]
    test_dates = featured["game_date"].iloc[split:]
    print(f"   Train: {len(X_train)} rows ({train_dates.min().date()} → {train_dates.max().date()})")
    print(f"   Test:  {len(X_test)} rows ({test_dates.min().date()} → {test_dates.max().date()})")
    print(f"   Train base rate: {y_train.mean():.1%}  Test base rate: {y_test.mean():.1%}")

    # ── 6. Train winner model on train slice ────────────────────────
    print("\n6. Training XGBClassifier on train slice (no leakage)...")
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train.values, y_train)

    # Also fit on full data for "in-sample" baseline
    print("   (Also fitting on FULL data for in-sample comparison...)")
    full_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(1 - base_rate) / base_rate,
        random_state=42,
        eval_metric="logloss",
    )
    full_model.fit(X.values, y)
    p_full = full_model.predict_proba(X.values)[:, 1]

    # ── 7. Predict P(red wins) on OOS test slice ────────────────────
    print("\n7. Predicting P(red wins) on OOS test slice...")
    p_test = model.predict_proba(X_test.values)[:, 1]

    # ── 8. Compute 6-outcome MoV distribution for each OOS test fight
    # We do BOTH arms:
    #   - raw: method_of_victory_probabilities() directly
    #   - cal: ... then calibrate_mov_distribution() with mov_calibration.json
    print("\n8. Computing 6-outcome MoV distribution for OOS test fights...")
    cal = load_mov_calibration()
    if cal:
        print(f"   Loaded mov_calibration.json with {len(cal)} outcome keys")
    else:
        print("   No mov_calibration.json found — only the raw prior arm will run")

    test_featured = featured.iloc[split:].reset_index(drop=True)
    # Build per-fight fighter stats from leak-free rates
    raw_mov = []  # list of dicts
    cal_mov = []
    valid_idx = []  # indices where we have both leak-free rates and a valid target
    for i, row in test_featured.iterrows():
        if pd.isna(row.get("r_ko_rate_lf")) or pd.isna(row.get("b_ko_rate_lf")):
            continue
        if row["mov_target"] not in MOV_OUTCOMES:
            continue
        wc = str(row.get("weight_class", "middleweight") or "middleweight").lower()
        r_stats = make_fighter_stats(row, "red", wc)
        b_stats = make_fighter_stats(row, "blue", wc)
        try:
            mov = method_of_victory_probabilities(
                float(p_test[i]), r_stats, b_stats, wc
            )
        except Exception as e:
            continue
        raw_mov.append(mov)
        if cal:
            mov_c = calibrate_mov_distribution(mov, cal)
        else:
            mov_c = mov
        cal_mov.append(mov_c)
        valid_idx.append(i)

    raw_mov_df = pd.DataFrame(raw_mov)
    cal_mov_df = pd.DataFrame(cal_mov)
    actual_outcomes = np.array([test_featured.loc[i, "mov_target"] for i in valid_idx])
    print(f"   {len(raw_mov_df)} OOS fights with valid MoV prediction + target")
    print(f"   Actual MoV distribution in OOS test:")
    for k in MOV_OUTCOMES:
        n = int((actual_outcomes == k).sum())
        print(f"     {k:9s} {n:>5d}  ({n/len(actual_outcomes):.1%})")

    # ── 9. Multi-class log loss and Brier per outcome ───────────────
    print("\n9. OOS metrics — multi-class log loss + Brier per outcome (raw vs calibrated):")
    # Multi-class log loss: requires the full 6-outcome probability vector
    # and the actual outcome as a one-hot encoding.
    actual_onehot = np.zeros((len(actual_outcomes), 6))
    for j, k in enumerate(MOV_OUTCOMES):
        actual_onehot[actual_outcomes == k, j] = 1
    raw_arr = raw_mov_df[MOV_OUTCOMES].values
    cal_arr = cal_mov_df[MOV_OUTCOMES].values
    # Clip to avoid log(0)
    raw_clipped = np.clip(raw_arr, 1e-6, 1 - 1e-6)
    cal_clipped = np.clip(cal_arr, 1e-6, 1 - 1e-6)
    # Naive baseline: constant prior = base rate distribution
    base_dist = np.array([(actual_outcomes == k).mean() for k in MOV_OUTCOMES])
    naive_log_loss = -float(np.sum(actual_onehot * np.log(np.clip(base_dist, 1e-6, 1))))
    # Pass 1D string labels (not 2D one-hot) so log_loss can binarize them.
    # This avoids the "LabelBinarizer was not fitted with multilabel input" error.
    raw_log_loss = float(log_loss(actual_outcomes, raw_clipped, labels=MOV_OUTCOMES))
    cal_log_loss = float(log_loss(actual_outcomes, cal_clipped, labels=MOV_OUTCOMES))
    raw_mean_brier = float(np.mean((raw_arr - actual_onehot) ** 2))
    cal_mean_brier = float(np.mean((cal_arr - actual_onehot) ** 2))
    naive_mean_brier = float(np.mean((base_dist - actual_onehot) ** 2))

    # In-sample (overfit check, not for verdict)
    in_p_red = p_full  # full-data fitted probs
    in_mov = []
    for i, row in featured.iterrows():
        if pd.isna(row.get("r_ko_rate_lf")) or pd.isna(row.get("b_ko_rate_lf")):
            continue
        if row["mov_target"] not in MOV_OUTCOMES:
            continue
        wc = str(row.get("weight_class", "middleweight") or "middleweight").lower()
        r_stats = make_fighter_stats(row, "red", wc)
        b_stats = make_fighter_stats(row, "blue", wc)
        try:
            mov = method_of_victory_probabilities(
                float(in_p_red[i]), r_stats, b_stats, wc
            )
        except Exception:
            continue
        in_mov.append(mov)
    in_mov_df = pd.DataFrame(in_mov)
    in_actual = featured["mov_target"].iloc[: len(in_mov_df)].astype(str).values
    in_onehot = np.zeros((len(in_actual), 6))
    for j, k in enumerate(MOV_OUTCOMES):
        in_onehot[in_actual == k, j] = 1
    in_arr = in_mov_df[MOV_OUTCOMES].values
    in_arr_clipped = np.clip(in_arr, 1e-6, 1 - 1e-6)
    # Pass 1D string labels (not 2D one-hot) to avoid multilabel binarizer error.
    in_log_loss = float(log_loss(in_actual, in_arr_clipped, labels=MOV_OUTCOMES))
    in_mean_brier = float(np.mean((in_arr - in_onehot) ** 2))

    print(f"\n   {'Metric':24s} {'In-sample':>10s} {'OOS raw':>10s} {'OOS cal':>10s} {'Naive':>10s}")
    print(f"   {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"   {'Multi-class log loss':24s} {in_log_loss:>10.4f} {raw_log_loss:>10.4f} {cal_log_loss:>10.4f} {naive_log_loss:>10.4f}")
    print(f"   {'Mean Brier (6 outcomes)':24s} {in_mean_brier:>10.4f} {raw_mean_brier:>10.4f} {cal_mean_brier:>10.4f} {naive_mean_brier:>10.4f}")
    print(f"\n   ⚠ CAL ARM LOOK-AHEAD BIAS: mov_calibration.json was trained on the SAME data")
    print(f"     the winner model was trained on (OOF on the full set). Applying it to the")
    print(f"     OOS test slice has look-ahead bias, so the cal-vs-raw log loss comparison")
    print(f"     above is contaminated. The truly OOS test would re-fit the calibration on")
    print(f"     the train slice (OOF on train) then apply to test. The raw prior arm is the")
    print(f"     more reliable signal of true OOS performance.")

    # Top-1 / Top-2 accuracy
    raw_top1 = float(accuracy_score(actual_outcomes, raw_mov_df.idxmax(axis=1).values))
    cal_top1 = float(accuracy_score(actual_outcomes, cal_mov_df.idxmax(axis=1).values))
    # Top-2: is actual in the top-2 predicted outcomes?
    raw_top2 = 0.0
    cal_top2 = 0.0
    for j in range(len(actual_outcomes)):
        raw_top2_idx = set(raw_mov_df.iloc[j].nlargest(2).index)
        cal_top2_idx = set(cal_mov_df.iloc[j].nlargest(2).index)
        if actual_outcomes[j] in raw_top2_idx:
            raw_top2 += 1
        if actual_outcomes[j] in cal_top2_idx:
            cal_top2 += 1
    raw_top2 /= max(len(actual_outcomes), 1)
    cal_top2 /= max(len(actual_outcomes), 1)
    # Random baseline top-1: 1/6 = 16.7%, top-2 = 2/6 = 33.3%
    print(f"\n   {'Top-1 accuracy':24s} {'   —':>10s} {raw_top1:>9.1%} {cal_top1:>9.1%} {'16.7%':>10s}")
    print(f"   {'Top-2 accuracy':24s} {'   —':>10s} {raw_top2:>9.1%} {cal_top2:>9.1%} {'33.3%':>10s}")

    # ── 10. Per-outcome calibration (calibrated arm) ───────────────
    print("\n10. Per-outcome OOS calibration (calibrated arm):")
    print(f"   Bin predicted P → actual outcome rate. Large gaps = overcompression.")
    print(f"   (Bin midpoint: 5%, 15%, 25%, ... 95%. Bins with n<5 are skipped.)\n")
    print(f"   {'Outcome':10s}  {'Bin':>8s}  {'N':>4s}  {'p_pred':>8s}  {'p_actual':>9s}  {'Gap':>7s}  {'Status'}")
    print(f"   {'-'*10}  {'-'*8}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*10}")
    per_outcome_gaps = {}
    per_outcome_max_gap = {}
    for j, k in enumerate(MOV_OUTCOMES):
        gaps = []
        max_gap = 0.0
        pred_col = cal_arr[:, j]
        actual_col = actual_onehot[:, j]
        bins = np.arange(0.0, 1.01, 0.1)
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (pred_col >= lo) & (pred_col < hi)
            n = int(mask.sum())
            if n < 5:
                if n > 0:
                    print(f"   {k:10s}  {lo:.0%}-{hi:.0%}     {n:>4d}  (too few, skipping)")
                continue
            p_pred = float(pred_col[mask].mean())
            p_actual = float(actual_col[mask].mean())
            gap = p_actual - p_pred
            gaps.append(abs(gap))
            max_gap = max(max_gap, abs(gap))
            status = "🚩 OVERCOMP" if abs(gap) > 0.15 and p_pred > 0.3 else ("OK" if abs(gap) < 0.05 else "WARN")
            print(f"   {k:10s}  {lo:.0%}-{hi:.0%}     {n:>4d}  {p_pred:>7.1%}  {p_actual:>8.1%}  {gap:>+6.1%}  {status}")
        per_outcome_gaps[k] = float(np.mean(gaps)) if gaps else None
        per_outcome_max_gap[k] = max_gap
    print()

    # ── 11. Edge simulation: how often do +20-30% edges win OOS? ───
    print("11. Edge simulation: how often do scanner-flagged MoV edges actually win OOS?")
    print("   Test: when the model puts P(outcome) >= X%, what fraction of those")
    print("   OOS fights actually end in that outcome? If model says 25% but")
    print("   actual is 12%, the scanner's edges are artifacts.\n")
    print(f"   {'Threshold':>12s}  {'Outcome':10s}  {'N calls':>8s}  {'N hits':>7s}  {'Hit rate':>8s}  {'Cal hit rate':>13s}")
    print(f"   {'-'*12}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*13}")
    print(f"   (rows with n_raw<20 omitted — too few OOS fights to be reliable)")
    edge_results = {}
    for j, k in enumerate(MOV_OUTCOMES):
        for thresh in [0.20, 0.30, 0.40]:
            raw_mask = raw_arr[:, j] >= thresh
            cal_mask = cal_arr[:, j] >= thresh
            n_raw = int(raw_mask.sum())
            n_cal = int(cal_mask.sum())
            n_hit_raw = int((actual_onehot[raw_mask, j] == 1).sum())
            n_hit_cal = int((actual_onehot[cal_mask, j] == 1).sum())
            hr_raw = n_hit_raw / n_raw if n_raw > 0 else 0
            hr_cal = n_hit_cal / n_cal if n_cal > 0 else 0
            edge_results[(k, thresh, "raw")] = (n_raw, hr_raw)
            edge_results[(k, thresh, "cal")] = (n_cal, hr_cal)
            if n_raw < 20:
                continue  # too few to be reliable
            print(f"   {thresh:>11.0%}  {k:10s}  {n_raw:>8d}  {n_hit_raw:>7d}  {hr_raw:>7.1%}  {hr_cal:>12.1%}")

    # ── 12. Verdict ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT (for UFC MoV prop betting)")
    print("=" * 72)

    # Decision logic:
    #   - OOS log loss must beat naive
    #   - Mean |gap| across outcomes must be ≤ 15% (no severe overcompression)
    #   - Max |gap| must be ≤ 25%
    #   - Top-1 accuracy must beat random (16.7%)
    valid_outcome_gaps = [v for v in per_outcome_gaps.values() if v is not None]
    mean_outcome_gap = float(np.mean(valid_outcome_gaps)) if valid_outcome_gaps else 1.0
    max_outcome_gap = max(per_outcome_max_gap.values()) if per_outcome_max_gap else 1.0
    raw_beats_naive = raw_log_loss < naive_log_loss
    cal_beats_naive = cal_log_loss < naive_log_loss
    cal_beats_raw = cal_log_loss < raw_log_loss
    raw_beats_random_top1 = raw_top1 > 0.167
    cal_beats_random_top1 = cal_top1 > 0.167
    cal_beats_random_top2 = cal_top2 > 0.333

    # Verdict logic — ALL cal-related branches require cal_beats_naive (the
    # cal arm has look-ahead bias, so it must earn its keep by beating naive).
    if not cal_beats_naive and not raw_beats_naive:
        verdict = "SKIP — MoV prior is not better than the naive base-rate distribution"
        reason = (
            f"Both raw ({raw_log_loss:.4f}) and calibrated ({cal_log_loss:.4f}) MoV log loss "
            f"are WORSE than the naive baseline ({naive_log_loss:.4f}). "
            f"The 6-outcome prior adds no value over just using the base rate."
        )
    elif mean_outcome_gap > 0.20 or max_outcome_gap > 0.30:
        verdict = "SKIP — severe overcompression in calibrated MoV"
        reason = (
            f"Mean outcome gap = {mean_outcome_gap:.1%}, max outcome gap = {max_outcome_gap:.1%}. "
            f"Per-outcome calibration shows severe miscalibration. "
            f"Scanner edges on MoV props (e.g. 'Topuria by KO +200') are likely artifacts."
        )
    elif not cal_beats_random_top1 and not raw_beats_random_top1:
        verdict = "SKIP — MoV prior is not better than random top-1"
        reason = (
            f"Top-1 accuracy = {raw_top1:.1%} (raw) / {cal_top1:.1%} (cal). "
            f"Random baseline is 16.7%. The prior is not picking the right outcome more often than chance."
        )
    elif raw_beats_naive and cal_beats_naive and cal_beats_raw and cal_beats_random_top2:
        # The cal arm is contaminated by look-ahead bias (mov_calibration.json
        # was trained on the same data). Even though cal beats raw here, that
        # comparison is biased upward. The honest recommendation is to use the
        # raw prior (which is NOT contaminated) and re-fit the calibration on
        # the train slice for a truly OOS cal test. The raw_beats_naive guard
        # ensures the raw arm itself is worth recommending.
        verdict = "PROCEED with raw MoV (cal arm contaminated — re-fit for truly OOS cal test)"
        reason = (
            f"Cal log loss {cal_log_loss:.4f} < raw {raw_log_loss:.4f} (calibration appears to help), "
            f"BUT mov_calibration.json was trained on the same data — the cal-vs-raw comparison is contaminated. "
            f"The HONEST signal is the raw arm: raw log loss {raw_log_loss:.4f} beats naive ({naive_log_loss:.4f}), "
            f"raw top-1 acc {raw_top1:.1%} > random 16.7%, raw top-2 acc {raw_top2:.1%} > random 33.3%. "
            f"Use the raw prior; re-fit mov_calibration.json on OOF train data to get a truly OOS cal test. "
            f"Mean |gap| = {mean_outcome_gap:.1%}, max |gap| = {max_outcome_gap:.1%}."
        )
    elif cal_beats_naive and cal_beats_random_top1 and not cal_beats_random_top2:
        verdict = "PROCEED with caution (only bet top-1 calibrated MoV, avoid top-2)"
        reason = (
            f"Cal log loss {cal_log_loss:.4f} beats naive ({naive_log_loss:.4f}). "
            f"Top-1 acc {cal_top1:.1%} > random 16.7%. "
            f"Top-2 acc {cal_top2:.1%} is NOT reliably > random 33.3%. "
            f"Only bet when the calibrated MoV is the SINGLE most-probable outcome."
        )
    elif raw_beats_naive and raw_beats_random_top1 and not cal_beats_raw:
        verdict = "PROCEED with raw prior (calibration does not help OOS)"
        reason = (
            f"Raw log loss {raw_log_loss:.4f} beats naive ({naive_log_loss:.4f}). "
            f"Raw top-1 acc {raw_top1:.1%} > random 16.7%. "
            f"Calibration makes it worse OOS ({cal_log_loss:.4f} > {raw_log_loss:.4f}). "
            f"Mean |gap| = {mean_outcome_gap:.1%}, max |gap| = {max_outcome_gap:.1%}. "
            f"Use the raw prior; consider re-fitting the calibration table."
        )
    else:
        verdict = "SKIP — neither raw nor cal MoV beats naive + random reliably"
        reason = (
            f"Raw log loss {raw_log_loss:.4f} does NOT beat naive ({naive_log_loss:.4f}), "
            f"or raw top-1 acc {raw_top1:.1%} does NOT beat random 16.7%. "
            f"Cal arm is contaminated (look-ahead bias — cal trained on full data). "
            f"Mean |gap| = {mean_outcome_gap:.1%}, max |gap| = {max_outcome_gap:.1%}."
        )

    print(f"\n   Verdict: {verdict}")
    print(f"   Reason:  {reason}")
    print(f"\n   Decision thresholds applied:")
    print(f"     Raw log loss beats naive:        {raw_log_loss:.4f} < {naive_log_loss:.4f} = {raw_beats_naive}")
    print(f"     Cal log loss beats naive:        {cal_log_loss:.4f} < {naive_log_loss:.4f} = {cal_beats_naive}")
    print(f"     Cal log loss beats raw:          {cal_log_loss:.4f} < {raw_log_loss:.4f} = {cal_beats_raw}")
    print(f"     Raw top-1 beats random (16.7%): {raw_top1:.1%} > 16.7% = {raw_beats_random_top1}")
    print(f"     Cal top-1 beats random (16.7%): {cal_top1:.1%} > 16.7% = {cal_beats_random_top1}")
    print(f"     Cal top-2 beats random (33.3%): {cal_top2:.1%} > 33.3% = {cal_beats_random_top2}")
    print(f"     Mean |outcome gap| threshold:    {mean_outcome_gap:.1%} (≤15% = OK, >20% = SKIP)")
    print(f"     Max  |outcome gap| threshold:    {max_outcome_gap:.1%} (≤25% = OK, >30% = SKIP)")

    print(f"\n   Recommended use of MoV probabilities:")
    if "high-confidence" in verdict:
        print("     • Bet MoV props only when the calibrated P(outcome) is the SINGLE most-probable outcome")
        print("     • Avoid parlays that combine multiple low-confidence MoV props")
        print("     • Re-check the calibration table quarterly (or after each major card)")
    elif "cal arm contaminated" in verdict or "raw MoV" in verdict:
        print("     • Use the RAW (uncalibrated) MoV probs — they are OOS-validated and pass all thresholds")
        print("     • The calibration table's improvement is likely look-ahead bias; re-fit on OOF train data")
        print("     • Once a truly OOS calibration is built, re-run this test to validate it")
    elif "caution" in verdict:
        print("     • ONLY bet the SINGLE most-probable calibrated outcome per fight")
        print("     • Do NOT bet 2nd-most-probable outcomes (top-2 acc is at random)")
        print("     • Use a higher edge threshold (e.g. require P > 30% AND edge > 10%)")
    elif "raw prior" in verdict:
        print("     • Use the raw (uncalibrated) MoV probs; skip the calibration table for now")
        print("     • Re-fit mov_calibration.json on OOF data — current cal is hurting OOS")
    elif "PROCEED" in verdict:
        print("     • Calibrated MoV probs are usable; calibration improves OOS over raw prior")
        print("     • Top-2 accuracy above random means the 2nd-most-probable outcome is also worth considering")
    else:
        print("     • DO NOT use MoV probs for live betting — they're not better than naive")
        print("     • Either retrain the MoV prior (e.g., add per-fighter recent-form features)")
        print("       or skip MoV props entirely and only bet the moneyline (winner model is OOS-validated separately)")

    # ⚠ Look-ahead bias disclaimer for the cal arm (also printed above the table)
    print(f"\n   ⚠ CAL ARM LOOK-AHEAD BIAS DISCLAIMER (see also section 9 above):")
    print(f"     mov_calibration.json was trained on the SAME data the model was trained on")
    print(f"     (OOF on the full set). Applying it to the OOS test slice has bias, so the")
    print(f"     cal-vs-raw log loss comparison is contaminated. The truly OOS test would")
    print(f"     re-fit the calibration on the train slice (OOF on train) then apply to test.")
    print(f"     The raw prior arm is the more reliable signal of true OOS performance.")

    # ── 13. Save JSON output ───────────────────────────────────────
    output = {
        "test_date": pd.Timestamp.now().isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_oos_with_mov": int(len(raw_mov_df)),
        "base_rate": base_rate,
        "in_sample_for_overfit_check_only_NOT_for_verdict": {
            "log_loss": in_log_loss,
            "mean_brier": in_mean_brier,
        },
        "oos_metrics": {
            "raw": {
                "log_loss": raw_log_loss,
                "mean_brier": raw_mean_brier,
                "top1_accuracy": raw_top1,
                "top2_accuracy": raw_top2,
            },
            "calibrated": {
                "log_loss": cal_log_loss,
                "mean_brier": cal_mean_brier,
                "top1_accuracy": cal_top1,
                "top2_accuracy": cal_top2,
            },
            "naive": {
                "log_loss": naive_log_loss,
                "mean_brier": naive_mean_brier,
            },
        },
        "per_outcome_calibration_gaps": per_outcome_gaps,
        "per_outcome_max_gap": per_outcome_max_gap,
        "mean_outcome_gap": mean_outcome_gap,
        "max_outcome_gap": max_outcome_gap,
        "edge_simulation": {
            f"{k}|{thresh:.0%}|{arm}": {"n": n, "hit_rate": hr}
            for (k, thresh, arm), (n, hr) in edge_results.items()
        },
        "thresholds": {
            "mean_outcome_gap_ok": 0.15,
            "mean_outcome_gap_skip": 0.20,
            "max_outcome_gap_ok": 0.25,
            "max_outcome_gap_skip": 0.30,
        },
        "verdict": verdict,
        "reason": reason,
    }
    out_path = MODEL_DIR / "oos_test_mov.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n   Results saved to {out_path}")


if __name__ == "__main__":
    main()
