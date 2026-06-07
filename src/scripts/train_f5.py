#!/usr/bin/env python3
"""Train F5 (First 5 Innings) multiclass model for MLB - Enhanced v2.

Key improvements over v1:
- class_weight="balanced" to handle HOME bias (45% HOME, 40% AWAY, 15% TIE)
- More trees (1000 vs 500), lower LR (0.02 vs 0.03)
- L1/L2 regularization
- Feature importance tracking
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.features.mlb import MLBFeatureEngineer, TEAM_IDS
import lightgbm as lgb
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
CACHE_DIR = PROJECT_ROOT / "data/cache/mlb"

PARK_FACTOR_K = {
    "SD": 1.08, "SEA": 1.06, "NYM": 1.04, "MIA": 1.03, "CLE": 1.02,
    "OAK": 1.02, "TB": 1.01, "SF": 1.01, "WSH": 1.00, "DET": 1.00,
    "MIL": 0.99, "BAL": 0.99, "KC": 0.99, "MIN": 0.99, "PIT": 0.99,
    "LAA": 0.98, "PHI": 0.98, "CIN": 0.98, "ATL": 0.97, "CHC": 0.97,
    "TEX": 0.97, "BOS": 0.97, "TOR": 0.96, "HOU": 0.96, "STL": 0.96,
    "ARI": 0.95, "NYY": 0.95, "LAD": 0.94, "CWS": 0.93, "COL": 0.88,
}


def load_f5_outcomes():
    path = CACHE_DIR / "f5_outcomes.csv"
    if not path.exists():
        print("F5 outcomes CSV not found.")
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"  Loaded {len(df)} F5 outcomes")
    return df


def load_pitcher_features():
    cache_files = sorted(CACHE_DIR.glob("game_logs_*.parquet"))
    if not cache_files:
        print("No cached MLB game logs found.")
        return None

    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(name="mlb", display_name="MLB",
                       rolling_windows=cfg["features"]["rolling_windows"],
                       recency_decay=0.001)
    fe = MLBFeatureEngineer(scfg)
    all_games = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    featured = fe.build_features(all_games)

    if "position" in featured.columns:
        pitchers = featured[featured["position"] == "P"].copy()
    else:
        print("  No position column")
        return None

    print(f"  {len(pitchers)} pitcher-game rows")
    return pitchers


def build_training_data(pitchers, f5_outcomes):
    print("  Building training data...")
    if "game_pk" not in f5_outcomes.columns:
        return None, None, None
    if "game_pk" not in pitchers.columns:
        return None, None, None
    game_teams = pitchers[["game_pk", "team_abbr", "game_date"]].drop_duplicates(subset=["game_pk", "team_abbr"])
    game_lookup = {}
    for gpk, grp in game_teams.groupby("game_pk"):
        teams = grp["team_abbr"].unique().tolist()
        if len(teams) >= 2:
            game_lookup[gpk] = {"away": teams[0], "home": teams[1], "date": grp["game_date"].iloc[0]}
    print(f"  Game lookup: {len(game_lookup)} games")
    X_rows, y_labels = [], []
    matched, no_game, no_pitcher = 0, 0, 0
    for _, game in f5_outcomes.iterrows():
        gpk = game["game_pk"]
        if gpk not in game_lookup:
            no_game += 1
            continue
        info = game_lookup[gpk]
        away_code, home_code = info["away"], info["home"]
        game_date = info["date"]
        gs = game_date - pd.Timedelta(hours=4)
        ge = game_date + pd.Timedelta(hours=3)
        ap = pitchers[(pitchers["game_date"] >= gs) & (pitchers["game_date"] <= ge) & (pitchers["team_abbr"].str.upper() == away_code.upper())].sort_values("game_date")
        hp = pitchers[(pitchers["game_date"] >= gs) & (pitchers["game_date"] <= ge) & (pitchers["team_abbr"].str.upper() == home_code.upper())].sort_values("game_date")
        if ap.empty or hp.empty:
            no_pitcher += 1
            continue
        away_p, home_p = ap.iloc[-1], hp.iloc[-1]
        outcome_str = str(game.get("f5_outcome", "TIE")).strip().upper()
        outcome = 0 if outcome_str == "AWAY" else (1 if outcome_str == "HOME" else 2)
        row = {"park_factor_k": PARK_FACTOR_K.get(home_code, 1.0)}
        for col in away_p.index:
            if isinstance(col, str):
                if any(col.endswith(f"_avg_{w}") for w in [7, 14, 30]):
                    row[f"a_{col}"] = away_p[col]
                    row[f"h_{col}"] = home_p[col]
                elif col.endswith("_ewm"):
                    row[f"a_{col}"] = away_p[col]
                    row[f"h_{col}"] = home_p[col]
        if len(row) >= 10:
            X_rows.append(row)
            y_labels.append(outcome)
            matched += 1
    print(f"  Matched: {matched}, No game: {no_game}, No pitcher: {no_pitcher}")
    if not X_rows:
        return None, None, None
    X = pd.DataFrame(X_rows).fillna(0)
    y = np.array(y_labels)
    feature_cols = sorted([c for c in X.columns if c != "park_factor_k"])
    X = X[["park_factor_k"] + feature_cols]
    counts = np.bincount(y, minlength=3)
    print(f"  Examples: {len(X)}, features: {len(feature_cols)}")
    print(f"  Dist: AWAY={counts[0]/len(y):.1%} HOME={counts[1]/len(y):.1%} TIE={counts[2]/len(y):.1%}")
    return X, y, feature_cols


def train_model(X, y, feature_cols):
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    model = lgb.LGBMModel(
        objective="multiclass", num_class=3,
        n_estimators=1000, num_leaves=31, learning_rate=0.02,
        subsample=0.8, feature_fraction=0.7,
        reg_alpha=0.5, reg_lambda=1.0, min_child_samples=20,
        class_weight="balanced",
        random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    preds = model.predict(X_test)
    pred_classes = np.argmax(preds, axis=1)
    accuracy = np.mean(pred_classes == y_test)
    print(f"  Test accuracy: {accuracy:.1%}")

    for cls_idx, cls_name in enumerate(["AWAY", "HOME", "TIE"]):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_acc = np.mean(pred_classes[mask] == cls_idx)
            print(f"  {cls_name:5s}: acc={cls_acc:.1%} n={mask.sum()}")

    y_onehot = np.zeros((len(y_test), 3))
    y_onehot[np.arange(len(y_test)), y_test] = 1
    brier = np.mean(np.sum((preds - y_onehot) ** 2, axis=1))
    print(f"  Brier: {brier:.4f}")

    # Feature importance
    imp = pd.DataFrame({"feature": ["park_factor_k"] + feature_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"  Top 10 features:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['feature']:40s} {r['importance']}")

    # Save as v2
    model_file = MODEL_DIR / "f5_multiclass_v2.txt"
    model.booster_.save_model(str(model_file))
    print(f"  Model saved to {model_file}")

    meta = {
        "train_date": datetime.now().isoformat(),
        "n_samples": len(X), "n_features": len(feature_cols),
        "n_train": len(X_train), "n_test": len(X_test),
        "test_accuracy": round(float(accuracy), 4),
        "test_brier": round(float(brier), 4),
        "best_iteration": int(model.best_iteration_ or 0),
        "features": ["park_factor_k"] + feature_cols,
        "top_features": imp.head(10)["feature"].tolist(),
    }
    with open(MODEL_DIR / "f5_multiclass_v2.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Metadata saved")

    # Calibration per outcome
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    for cls_idx, cls_name in enumerate(["away", "home", "tie"]):
        cal_table = []
        class_preds = preds[:, cls_idx]
        class_actual = (y_test == cls_idx).astype(int)
        for lo in np.arange(0, 1.0, 0.05):
            hi = lo + 0.05
            mask = (class_preds >= lo) & (class_preds < hi)
            if mask.sum() >= 5:
                cal_table.append({
                    "p_pred_min": float(lo), "p_pred_max": float(hi),
                    "p_actual": float(class_actual[mask].mean()), "n": int(mask.sum()),
                })
        with open(CALIB_DIR / f"f5_v2_{cls_name}_empirical.json", "w") as f:
            json.dump(cal_table, f, indent=2)
        print(f"  Calibration saved for {cls_name}")

    return model


def main():
    print("=" * 60)
    print("  F5 MULTICLASS MODEL v2")
    print("=" * 60)
    print("\n1. Loading F5 outcomes...")
    f5_outcomes = load_f5_outcomes()
    if f5_outcomes is None:
        return
    print("\n2. Loading pitcher features...")
    pitchers = load_pitcher_features()
    if pitchers is None:
        return
    print("\n3. Building training data...")
    X, y, feature_cols = build_training_data(pitchers, f5_outcomes)
    if X is None:
        return
    print("\n4. Training model...")
    train_model(X, y, feature_cols)
    print("\n  Done!")


if __name__ == "__main__":
    main()
