"""
F5 Plate Appearance Outcome Model — 8-class LightGBM Classifier.

Trains on Statcast PA-level data predicting 8 discrete outcomes:
    0: OUT (field_out, force_out, ground_into_dp, sac_fly, sac_bunt, etc.)
    1: 1B  (single)
    2: 2B  (double)
    3: 3B  (triple)
    4: HR  (home_run)
    5: BB  (walk, intent_walk)
    6: HBP (hit_by_pitch)
    7: K   (strikeout, strikeout_double_play)

Features per PA:
    - Pitcher rolling K%, BB%, HR/9 from MLB feature engineer
    - Batter rolling stats (launch speed, barrel rate) from Statcast
    - Game context: inning, outs, balls, strikes, runners on
    - Park factors
    - Batter/pitcher handedness matchup
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
from src.config.settings import PROJECT_ROOT
from src.data.mlb_pitching_stats import fetch_multiyear_pitching_stats, merge_pitching_stats_into_pa

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
STATCAST_DIR = PROJECT_ROOT / "data" / "cache" / "mlb" / "statcast"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mlb"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 8 outcome classes
OUTCOME_MAP = {
    "OUT": 0, "1B": 1, "2B": 2, "3B": 3, "HR": 4,
    "BB": 5, "HBP": 6, "K": 7,
}
OUTCOME_NAMES = {v: k for k, v in OUTCOME_MAP.items()}

# Park factors for K — using Statcast team codes (verified from actual data)
# Higher = more pitcher-friendly (more Ks)
PARK_FACTOR_K = {
    "SD": 1.08, "SEA": 1.06, "NYM": 1.04, "MIA": 1.03, "CLE": 1.02,
    "ATH": 1.02, "TB": 1.01, "SF": 1.01, "WSH": 1.00, "DET": 1.00,
    "MIL": 0.99, "BAL": 0.99, "KC": 0.99, "MIN": 0.99, "PIT": 0.99,
    "LAA": 0.98, "PHI": 0.98, "CIN": 0.98, "ATL": 0.97, "CHC": 0.97,
    "TEX": 0.97, "BOS": 0.97, "TOR": 0.96, "HOU": 0.96, "STL": 0.96,
    "AZ": 0.95, "NYY": 0.95, "LAD": 0.94, "CWS": 0.93, "COL": 0.88,
}

# Park factors for HR (lower = more HR-suppressing) — Statcast codes
PARK_FACTOR_HR = {
    "COL": 1.28, "CIN": 1.14, "BOS": 1.12, "NYY": 1.10, "BAL": 1.09,
    "CHC": 1.07, "MIL": 1.06, "MIN": 1.05, "TEX": 1.04, "HOU": 1.04,
    "CLE": 1.04, "LAA": 1.03, "PHI": 1.03, "AZ": 1.02, "TB": 1.01,
    "ATL": 1.01, "WSH": 1.00, "DET": 1.00, "KC": 1.00, "STL": 1.00,
    "PIT": 0.99, "NYY": 0.99, "MIA": 0.98, "SEA": 0.98, "LAD": 0.97,
    "SD": 0.96, "SF": 0.95, "ATH": 0.95, "TOR": 0.94, "NYM": 0.93,
}

MLB_TEAM_CODES = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "CWS": "CWS", "DET": "DET",
    "HOU": "HOU", "KC": "KC", "KCR": "KC", "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "SD", "SEA": "SEA",
    "SF": "SF", "STL": "STL", "TB": "TB", "TEX": "TEX", "TOR": "TOR",
    "WSH": "WSH", "WSN": "WSH",
}


def map_statcast_events(events_series: pd.Series) -> pd.Series:
    """Map Statcast events to 8-class outcome codes."""
    mapping = {
        "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
        "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "HBP",
        "strikeout": "K", "strikeout_double_play": "K",
    }
    # Everything else maps to OUT
    return events_series.map(mapping).fillna("OUT")


def build_pa_dataset(seasons: list[int] = None) -> pd.DataFrame:
    """Build PA-level dataset from cached Statcast data with features.
    
    Returns DataFrame with features + outcome labels for training.
    """
    if seasons is None:
        seasons = [2024, 2025, 2026]

    all_pas = []
    for season in seasons:
        cache_file = STATCAST_DIR / f"statcast_{season}.parquet"
        if not cache_file.exists():
            print(f"  No Statcast cache for {season}, skipping", flush=True)
            continue
        print(f"  Loading {season} Statcast data...", flush=True)
        df = pd.read_parquet(cache_file)
        
        # Filter to regular season games only
        df = df[df["game_type"].isin(["R", "r"])].copy()
        
        # Filter out rows without events (non-PA rows like pitch sequences)
        df = df[df["events"].notna()].copy()
        
        # Map events to 8 classes
        df["outcome"] = map_statcast_events(df["events"])
        df["outcome_code"] = df["outcome"].map(OUTCOME_MAP)
        
        # Drop any unmapped outcomes
        df = df[df["outcome_code"].notna()].copy()
        
        print(f"    {len(df)} PAs with outcomes", flush=True)
        
        # Build features
        # Pitcher features (rolling K%, BB% from existing MLB pipeline)
        # We'll merge these from the MLBFeatureEngineer later
        # For now, use basic per-PA features
        
        # Game context features
        df["is_home"] = (df["inning_topbot"] == "Bot").astype(int)
        df["runners_on"] = (
            df["on_1b"].notna().astype(int) +
            df["on_2b"].notna().astype(int) +
            df["on_3b"].notna().astype(int)
        )
        
        # Park factor (map home_team to code)
        df["park_factor_k"] = df["home_team"].map(PARK_FACTOR_K).fillna(1.0)
        df["park_factor_hr"] = df["home_team"].map(PARK_FACTOR_HR).fillna(1.0)
        
        # Matchup features
        df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
        
        # Umpire zone (simplified: fixed 1.0 since we don't have per-umpire data)
        df["umpire_zone_factor"] = 1.0
        
        # Batter features (rolling averages from Statcast agg cache)
        # These are merged separately
        
        # Keep only the columns we need
        keep_cols = [
            "game_pk", "game_date", "pitcher", "batter", "inning",
            "inning_topbot", "outs_when_up", "balls", "strikes",
            "on_1b", "on_2b", "on_3b", "home_team", "away_team",
            "stand", "p_throws", "player_name",
            "outcome", "outcome_code",
            "is_home", "runners_on", "park_factor_k", "park_factor_hr",
            "same_hand", "umpire_zone_factor",
        ]
        available = [c for c in keep_cols if c in df.columns]
        all_pas.append(df[available].copy())
        
        print(f"    Features: {len(available)} columns", flush=True)

    if not all_pas:
        print("  No data loaded!", flush=True)
        return pd.DataFrame()

    result = pd.concat(all_pas, ignore_index=True)
    print(f"  Total: {len(result)} PAs across {len(seasons)} seasons", flush=True)
    print(f"  Outcome distribution:")
    for code in sorted(result["outcome_code"].unique()):
        count = (result["outcome_code"] == code).sum()
        print(f"    {OUTCOME_NAMES[code]:4s}: {count:6d} ({count/len(result):.1%})")
    
    return result


def add_rolling_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-PA features — game context, park factors, and batter/pitcher stats.
    
    V1 features are all from the Statcast PA-level data itself:
    - Game context: inning, outs, balls, strikes, runners_on
    - Park factors (from home_team)
    - Batter/pitcher handedness matchup
    - Batter and pitcher as categorical features (via aggregation)
    
    V2 will add rolling pitcher K/9, BB/9 from game logs once ID mapping
    between Statcast (MLBAM IDs) and game logs (internal IDs) is built.
    """
    if pa_df.empty:
        return pa_df
    
    print(f"  Adding rolling features for {len(pa_df)} PAs", flush=True)
    
    # Sort by game_date globally first for proper temporal ordering
    pa_df = pa_df.sort_values("game_date").reset_index(drop=True)
    
    # Compute indicator columns
    pa_df["pitcher_is_k"] = (pa_df["outcome_code"] == 7).astype(int)
    pa_df["pitcher_is_bb"] = (pa_df["outcome_code"] == 5).astype(int)
    pa_df["pitcher_is_hr"] = (pa_df["outcome_code"] == 4).astype(int)
    
    # Build rolling features with explicit per-group computation
    # to avoid pandas groupby/transform/index alignment edge cases
    
    # Per-pitcher rolling K rate
    pa_df["pitcher_k_rate_prior"] = 0.22  # default
    for pid in pa_df["pitcher"].unique():
        mask = pa_df["pitcher"] == pid
        idx = pa_df.index[mask]
        sorted_idx = idx.sort_values()  # already sorted by game_date
        vals = pa_df.loc[sorted_idx, "pitcher_is_k"].values
        for i, actual_idx in enumerate(sorted_idx):
            if i == 0:
                pa_df.at[actual_idx, "pitcher_k_rate_prior"] = 0.22
            else:
                prior = vals[:i].mean()
                pa_df.at[actual_idx, "pitcher_k_rate_prior"] = prior
    
    # Per-pitcher rolling BB rate
    pa_df["pitcher_bb_rate_prior"] = 0.08
    for pid in pa_df["pitcher"].unique():
        mask = pa_df["pitcher"] == pid
        idx = pa_df.index[mask]
        sorted_idx = idx.sort_values()
        vals = pa_df.loc[sorted_idx, "pitcher_is_bb"].values
        for i, actual_idx in enumerate(sorted_idx):
            if i == 0:
                pa_df.at[actual_idx, "pitcher_bb_rate_prior"] = 0.08
            else:
                prior = vals[:i].mean()
                pa_df.at[actual_idx, "pitcher_bb_rate_prior"] = prior
    
    # Per-pitcher rolling HR rate
    pa_df["pitcher_hr_rate_prior"] = 0.03
    for pid in pa_df["pitcher"].unique():
        mask = pa_df["pitcher"] == pid
        idx = pa_df.index[mask]
        sorted_idx = idx.sort_values()
        vals = pa_df.loc[sorted_idx, "pitcher_is_hr"].values
        for i, actual_idx in enumerate(sorted_idx):
            if i == 0:
                pa_df.at[actual_idx, "pitcher_hr_rate_prior"] = 0.03
            else:
                prior = vals[:i].mean()
                pa_df.at[actual_idx, "pitcher_hr_rate_prior"] = prior
    
    # Per-batter rolling K rate (sorted by batter, then game_date)
    pa_df["batter_k_rate_prior"] = 0.22
    bat_sort = pa_df.sort_values(["batter", "game_date"])
    for bid in pa_df["batter"].unique():
        mask = bat_sort["batter"] == bid
        idx = bat_sort.index[mask]
        vals = pa_df.loc[idx, "pitcher_is_k"].values  # batter's K indicator
        for j, actual_idx in enumerate(idx):
            if j == 0:
                pa_df.at[actual_idx, "batter_k_rate_prior"] = 0.22
            else:
                prior = vals[:j].mean()
                pa_df.at[actual_idx, "batter_k_rate_prior"] = prior
    
    # Per-batter rolling BB rate
    pa_df["batter_bb_rate_prior"] = 0.08
    for bid in pa_df["batter"].unique():
        mask = bat_sort["batter"] == bid
        idx = bat_sort.index[mask]
        vals = pa_df.loc[idx, "pitcher_is_bb"].values
        for j, actual_idx in enumerate(idx):
            if j == 0:
                pa_df.at[actual_idx, "batter_bb_rate_prior"] = 0.08
            else:
                prior = vals[:j].mean()
                pa_df.at[actual_idx, "batter_bb_rate_prior"] = prior
    
    # Drop intermediate indicator columns
    for col in ["pitcher_is_k", "pitcher_is_bb", "pitcher_is_hr"]:
        if col in pa_df.columns:
            pa_df.drop(columns=[col], inplace=True)
    
    # Handle missing values
    numeric_cols = pa_df.select_dtypes(include=[np.number]).columns
    pa_df[numeric_cols] = pa_df[numeric_cols].fillna(0)
    
    return pa_df


def get_feature_columns(pa_df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns for the model."""
    # Exclude target, ID, and redundant columns
    exclude = {
        "outcome", "outcome_code", "game_pk", "game_date", "pitcher", "batter",
        "player_name", "home_team", "away_team", "home_score", "away_score",
        "on_1b", "on_2b", "on_3b",  # encoded as runners_on
    }
    return [c for c in pa_df.columns if c not in exclude and pa_df[c].dtype in [np.int64, np.float64, np.int32, np.float32]]


def train_pa_model(pa_df: pd.DataFrame) -> dict:
    """Train 8-class LightGBM model on PA data."""
    
    feature_cols = get_feature_columns(pa_df)
    print(f"\n  Training features ({len(feature_cols)}):")
    for c in feature_cols[:20]:
        print(f"    {c}")
    if len(feature_cols) > 20:
        print(f"    ... and {len(feature_cols)-20} more")

    X = pa_df[feature_cols].values
    y = pa_df["outcome_code"].values.astype(int)

    # Time-based split (by game_date)
    pa_df = pa_df.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(pa_df) * 0.85)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    # Class weights: inverse sqrt frequency capped at 10.0
    # This gives rare events (3B=0.4%, HBP=1.1%) more importance
    # without completely overwhelming the model
    class_counts = np.bincount(y_train, minlength=8)
    total = len(y_train)
    n_classes = 8
    sample_weights = np.ones(len(y_train), dtype=float)
    for i in range(n_classes):
        # Inverse sqrt weighting: sqrt(total / (n_classes * count))
        weight = float(np.sqrt(total / (n_classes * max(class_counts[i], 1))))
        weight = min(weight, 10.0)  # Cap at 10x to prevent overfitting to rare events
        mask = y_train == i
        sample_weights[mask] = weight
        print(f"  {OUTCOME_NAMES[i]:4s}: count={class_counts[i]:6d} weight={weight:.2f}")

    # LightGBM dataset with per-sample weights
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 8,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.08,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "min_child_weight": 5.0,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "verbose": -1,
        "num_threads": 4,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    # Evaluate
    preds = model.predict(X_test)
    pred_classes = np.argmax(preds, axis=1)
    
    accuracy = np.mean(pred_classes == y_test)
    
    # Per-class metrics
    print(f"\n  Test accuracy: {accuracy:.4f}")
    print(f"  Per-class results:")
    for cls_idx in range(8):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_acc = np.mean(pred_classes[mask] == cls_idx)
            print(f"    {OUTCOME_NAMES[cls_idx]:4s}: acc={cls_acc:.3f} n={mask.sum():6d}")

    # Brier score (multiclass)
    y_onehot = np.zeros((len(y_test), 8))
    y_onehot[np.arange(len(y_test)), y_test] = 1
    brier = np.mean(np.sum((preds - y_onehot) ** 2, axis=1))
    print(f"\n  Multiclass Brier: {brier:.4f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importance})
    feat_imp = feat_imp.sort_values("importance", ascending=False)
    print(f"\n  Top 15 features by gain:")
    for _, r in feat_imp.head(15).iterrows():
        print(f"    {r['feature']:45s} {r['importance']:.1f}")

    # Save model
    model_file = MODEL_DIR / "f5_pa_outcome.txt"
    model.save_model(str(model_file))
    print(f"\n  Model saved to {model_file}")

    # Save metadata
    meta = {
        "train_date": datetime.now().isoformat(),
        "n_samples": len(pa_df),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_accuracy": round(float(accuracy), 4),
        "test_brier": round(float(brier), 4),
        "best_iteration": model.best_iteration,
        "feature_cols": feature_cols,
        "outcome_names": OUTCOME_NAMES,
        "top_features": feat_imp.head(20)["feature"].tolist(),
        "class_distribution": {
            OUTCOME_NAMES[i]: int(np.bincount(y, minlength=8)[i])
            for i in range(8)
        },
    }
    meta_file = MODEL_DIR / "f5_pa_outcome.meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_file}")

    return {"model": model, "meta": meta, "feature_cols": feature_cols}


def main():
    print("=" * 65)
    print("  F5 PA OUTCOME MODEL — 8-Class LightGBM")
    print("=" * 65)

    print("\n1. Building PA dataset...")
    pa_df = build_pa_dataset([2024, 2025])
    
    if pa_df.empty:
        print("  No data available!")
        return
    
    print(f"\n2. Fetching MLB pitcher stats (FIP, K/9, BB/9)...")
    pitching_df = fetch_multiyear_pitching_stats([2024, 2025], min_ip=30)
    if not pitching_df.empty:
        print(f"\n3. Merging FIP data into PA dataset...")
        pa_df = merge_pitching_stats_into_pa(pa_df, pitching_df)
    else:
        print("  No pitching stats available, skipping FIP features")
    
    print(f"\n4. Adding rolling features...")
    pa_df = add_rolling_features(pa_df)
    
    print(f"\n  Final dataset: {len(pa_df)} rows, {pa_df.select_dtypes(include=[np.number]).shape[1]} numeric columns")
    
    print(f"\n5. Training model...")
    result = train_pa_model(pa_df)
    
    print(f"\n  Done! Model ready for F5 Monte Carlo simulation.")


if __name__ == "__main__":
    main()
