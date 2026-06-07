import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data.ufc import UFCDataSource
from src.features.ufc import build_ufc_features, FEATURE_COLS

MODEL_DIR = Path("models/ufc")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== UFC Fight-Winner Model Training ===")
    print("Loading MikeSpa dataset...")
    ds = UFCDataSource()
    df = ds.fetch_player_game_logs(["all"])
    if df.empty:
        print("No data loaded")
        return
    print(f"  {len(df)} fights loaded")

    df = df.sort_values("game_date").reset_index(drop=True)

    print("Building features...")
    featured = build_ufc_features(df)
    available = [c for c in FEATURE_COLS if c in featured.columns]
    print(f"  {len(available)} features available")

    X = featured[available].fillna(0)

    if "winner" in featured.columns:
        winner = featured["winner"]
    elif "Winner" in featured.columns:
        winner = featured["Winner"]
    else:
        print("  No winner column found in featured data!")
        print(f"  Columns: {list(featured.columns[:30])}")
        return
    y = (winner == "Red").astype(int).values

    base_rate = y.mean()
    print(f"  Base rate (Red wins): {base_rate:.1%}")
    print(f"  Blue wins: {1 - base_rate:.1%}")

    tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(df) * 0.15))
    models = []
    oof_preds = np.zeros(len(df))
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=(1 - base_rate) / base_rate,
            random_state=42 + fold,
            eval_metric="logloss",
            early_stopping_rounds=30,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict_proba(X_test)[:, 1]
        oof_preds[test_idx] = preds

        acc = accuracy_score(y_test, (preds >= 0.5).astype(int))
        brier = brier_score_loss(y_test, preds)
        auc = roc_auc_score(y_test, preds)
        cv_scores.append({"fold": fold, "accuracy": round(acc, 4), "brier": round(brier, 4), "auc": round(auc, 4)})
        print(f"  Fold {fold}: acc={acc:.1%}, brier={brier:.4f}, auc={auc:.3f}")

    oof_train = oof_preds[y == 1].mean()
    oof_test = 1 - oof_preds[y == 0].mean()
    overall_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
    print(f"\n  OOF Accuracy: {overall_acc:.1%}")
    print(f"  OOF Mean prob (Red wins): {oof_preds.mean():.1%}")
    print(f"  OOF Mean prob when Red actually wins: {oof_train:.1%}")
    print(f"  OOF Mean prob when Blue actually wins: {1 - oof_test:.1%}")

    print("\nCalibration by confidence bin:")
    bins = np.arange(0, 1.05, 0.05)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (oof_preds >= lo) & (oof_preds < hi)
        if mask.sum() >= 10:
            actual = y[mask].mean()
            print(f"  [{lo:.0%}-{hi:.0%}): pred={lo + 0.025:.1%} actual={actual:.1%}  n={mask.sum()}")

    # Retrain on full data
    print("\nRetraining on full dataset...")
    final_model = XGBClassifier(
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
    final_model.fit(X, y)

    final_preds = final_model.predict_proba(X)[:, 1]
    final_acc = accuracy_score(y, (final_preds >= 0.5).astype(int))
    final_brier = brier_score_loss(y, final_preds)
    print(f"  In-sample accuracy: {final_acc:.1%}")
    print(f"  In-sample brier: {final_brier:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": available,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 features:")
    for _, row in importance.head(20).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    # Save model
    model_file = MODEL_DIR / "winner_v1.json"
    final_model.save_model(str(model_file))
    print(f"\nModel saved to {model_file}")

    # Save metadata
    meta = {
        "train_date": pd.Timestamp.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(available),
        "base_rate": float(base_rate),
        "cv_scores": cv_scores,
        "oof_accuracy": float(overall_acc),
        "features": available,
    }
    meta_file = MODEL_DIR / "winner_v1.meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_file}")

    # Save calibration table for scanner
    cal_table = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (oof_preds >= lo) & (oof_preds < hi)
        if mask.sum() >= 10:
            cal_table.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "model_pred": float(lo + 0.025),
                "actual_rate": float(y[mask].mean()),
                "n": int(mask.sum()),
            })
    cal_file = MODEL_DIR / "winner_calibration.json"
    with open(cal_file, "w") as f:
        json.dump(cal_table, f, indent=2)
    print(f"Calibration saved to {cal_file}")


if __name__ == "__main__":
    main()
