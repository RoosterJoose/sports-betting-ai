import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data.nascar import NASCARDataSource    from src.data.nascar_loop import fetch_multiyear_loop_data, _normalize_driver
from src.features.nascar import NASCARFeatureEngineer

MODEL_DIR = Path("models/nascar")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "avg_finish_position_5", "avg_standings_position_5",
    "avg_finish_position_10", "avg_standings_position_10",
    "avg_finish_position_20", "avg_standings_position_20",
    "rate_is_winner_10", "rate_pole_position_10", "rate_laps_led_most_10",
    "tt_superspeedway_avg", "tt_short_avg", "tt_intermediate_avg",
    "tt_road_avg", "tt_triangle_avg", "tt_speedway_avg",
    "form_recent", "finish_std_10",
    "team_avg_finish", "manufacturer_avg_finish",
    "race_number", "season_experience",
    "avg_start_pos_5", "avg_start_pos_10", "avg_start_pos_20",
    "avg_starting_position_5", "avg_starting_position_10", "avg_starting_position_20",
    # Loop data features
    "roll_driver_rating_3", "roll_driver_rating_5", "roll_driver_rating_10",
    "roll_avg_running_pos_3", "roll_avg_running_pos_5", "roll_avg_running_pos_10",
    "roll_laps_led_rate_3", "roll_laps_led_rate_5", "roll_laps_led_rate_10",
    "roll_top15_laps_3", "roll_top15_laps_5", "roll_top15_laps_10",
    "roll_quality_passes_3", "roll_quality_passes_5", "roll_quality_passes_10",
    "roll_pass_diff_3", "roll_pass_diff_5", "roll_pass_diff_10",
    "roll_pct_quality_passes_3", "roll_pct_quality_passes_5", "roll_pct_quality_passes_10",
    "roll_pct_top15_laps_3", "roll_pct_top15_laps_5", "roll_pct_top15_laps_10",
]


def train_model(df, target_col, model_name, base_rate):
    available = [c for c in FEATURES if c in df.columns]
    X = df[available].fillna(0).values
    y = df[target_col].values

    tscv = TimeSeriesSplit(n_splits=5, test_size=max(200, int(len(df) * 0.08)))
    oof_preds = np.zeros(len(df))
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=(1 - base_rate) / max(base_rate, 0.01),
            random_state=42 + fold,
            eval_metric="logloss",
            early_stopping_rounds=30,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict_proba(X_test)[:, 1]
        oof_preds[test_idx] = preds
        acc = accuracy_score(y_test, (preds >= 0.5).astype(int))
        brier = brier_score_loss(y_test, preds)
        cv_scores.append({"fold": fold, "accuracy": round(acc, 4), "brier": round(brier, 4)})

    overall_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
    overall_brier = brier_score_loss(y, oof_preds)

    median_prob = np.median(oof_preds)
    positive_rate = (oof_preds >= 0.5).mean()

    # Platt calibration on OOF predictions
    log_odds = np.log(np.clip(oof_preds, 1e-6, 1 - 1e-6) / (1 - np.clip(oof_preds, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e10, solver="lbfgs")
    platt.fit(log_odds.reshape(-1, 1), y)
    platt_slope = float(platt.coef_[0][0])
    platt_intercept = float(platt.intercept_[0])
    calibrated = 1 / (1 + np.exp(-(log_odds * platt_slope + platt_intercept)))
    cal_brier = brier_score_loss(y, calibrated)
    cal_acc = accuracy_score(y, (calibrated >= 0.5).astype(int))

    final_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=(1 - base_rate) / max(base_rate, 0.01),
        random_state=42, eval_metric="logloss",
    )
    final_model.fit(X, y)

    model_file = MODEL_DIR / f"{model_name}.json"
    final_model.save_model(str(model_file))

    meta = {
        "model_name": model_name,
        "train_date": pd.Timestamp.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(available),
        "base_rate": float(base_rate),
        "cv_scores": cv_scores,
        "oof_accuracy": float(overall_acc),
        "oof_brier": float(overall_brier),
        "calibrated_accuracy": float(cal_acc),
        "calibrated_brier": float(cal_brier),
        "platt_slope": platt_slope,
        "platt_intercept": platt_intercept,
        "features": available,
        "median_prob": float(median_prob),
        "positive_rate": float(positive_rate),
    }
    meta_file = MODEL_DIR / f"{model_name}.meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return model, meta


def main():
    print("=== NASCAR Model Training ===")
    ds = NASCARDataSource()
    df = ds.fetch_player_game_logs(["2020", "2021", "2022", "2023", "2024", "2025"])
    if df.empty:
        print("No data")
        return
    print(f"  {len(df)} driver-race rows, {df['driver_name'].nunique()} drivers")
    
    # Merge loop data from Racing-Reference (driver rating, avg running position, etc.)
    print(f"\n  Merging loop data from Racing-Reference...")
    loop_years = [2020, 2021, 2022, 2023, 2024, 2025]
    loop_df = fetch_multiyear_loop_data(loop_years)
    if not loop_df.empty:
        # Normalize driver names for merge (use same function as scraper)
        df["driver_norm"] = df["driver_name"].apply(_normalize_driver)
        df["race_number"] = df["race_number"].astype(int)
        df["season"] = df["season"].astype(str)
        loop_merge = loop_df[["driver_norm", "race_number", "season",
                              "driver_rating", "avg_running_position", "laps_led",
                              "total_laps", "quality_passes", "top15_laps",
                              "passing_differential", "pct_quality_passes",
                              "pct_top15_laps", "fastest_laps",
                              "start_position", "finish_position"]].copy()
        loop_merge["race_number"] = loop_merge["race_number"].astype(int)
        loop_merge["season"] = loop_merge["season"].astype(str)
        before = len(df)
        df = df.merge(loop_merge, on=["driver_norm", "race_number", "season"], how="left",
                      suffixes=("", "_loop"))
        matched = df["driver_rating"].notna().sum()
        print(f"    Matched {matched}/{before} driver-race rows with loop data ({matched/before:.1%})")
        # Drop loop suffix columns and driver_norm
        for c in df.columns:
            if c.endswith("_loop"):
                df.drop(columns=[c], inplace=True)
        df.drop(columns=["driver_norm"], inplace=True, errors="ignore")
    else:
        print(f"    No loop data available — models will train without driver rating features")

    from types import SimpleNamespace
    cfg = SimpleNamespace(rolling_windows=[3, 5, 10], recency_decay=0.003)
    fe = NASCARFeatureEngineer(cfg)
    # Compute target columns from raw data BEFORE feature engineering (which strips raw columns)
    targets_raw = df[["driver_name", "race_number", "season", "finish_position", "is_winner"]].copy()
    targets_raw["top_5"] = (targets_raw["finish_position"] <= 5).astype(int)
    targets_raw["top_10"] = (targets_raw["finish_position"] <= 10).astype(int)

    featured = fe.build_features(df)

    # Merge targets back into featured data (player_id = driver_name, preserved by feature engineer)
    # Include season in merge keys to prevent Cartesian products across years
    featured = featured.merge(
        targets_raw[["driver_name", "race_number", "season", "top_5", "top_10", "is_winner"]],
        left_on=["player_id", "race_number", "season"],
        right_on=["driver_name", "race_number", "season"],
        how="left"
    )
    featured = featured.drop(columns=[col for col in ["driver_name_y", "driver_name_x"] if col in featured.columns], errors="ignore")

    available = [c for c in FEATURES if c in featured.columns]
    featured = featured.dropna(subset=available + ["is_winner", "top_5", "top_10"])
    print(f"  {len(featured)} featured rows, {len(available)} features")

    targets = [
        ("win", "is_winner", featured["is_winner"].mean()),
        ("top5", "top_5", featured["top_5"].mean()),
        ("top10", "top_10", featured["top_10"].mean()),
    ]

    for model_name, target_col, base_rate in targets:
        print(f"\nTraining {model_name} model (base rate={base_rate:.1%})...")
        model, meta = train_model(featured, target_col, model_name, base_rate)
        print(f"  OOF acc={meta['oof_accuracy']:.1%}, brier={meta['oof_brier']:.4f}")

    print(f"\nModels saved to {MODEL_DIR}/")
    for f in sorted(MODEL_DIR.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
