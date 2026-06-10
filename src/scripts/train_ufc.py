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
        models.append(model)  # Keep CV-fold models in memory; saved below
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

    # ── Save CV fold models (used by scanner for calibration-consistent predictions) ──
    print("\nSaving CV fold models...")
    for fold, m in enumerate(models):
        fold_path = MODEL_DIR / f"winner_v1_fold{fold}.json"
        m.save_model(str(fold_path))
        print(f"  Fold {fold} saved to {fold_path.name}")

    # Save winner_v1.json = fold 0 (backward compat, calibration-consistent)
    # The calibration was built from OOF predictions across all CV folds.
    # Using a CV model (trained on 4/5 of data) instead of the final retrained
    # model (trained on all data) keeps the prediction distribution consistent
    # with the calibration table, preventing the [0-5%] bin mismatch.
    model_file = MODEL_DIR / "winner_v1.json"
    models[0].save_model(str(model_file))
    print(f"Main model (CV fold 0) saved to {model_file}")

    # Save metadata
    meta = {
        "train_date": pd.Timestamp.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(available),
        "base_rate": float(base_rate),
        "cv_scores": cv_scores,
        "oof_accuracy": float(overall_acc),
        "features": available,
        "n_cv_models": len(models),
        "model_source": "cv_fold_0",
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

    # ── Build fighter lookup table ────────────────────────────────────────
    print("\nBuilding fighter lookup table...")
    # Read raw CSV directly for per-fighter stats
    raw_path = Path("data/cache/ufc/ufc-master.csv")
    if not raw_path.exists():
        print("  Raw CSV not found, skipping fighter lookup")
    else:
        raw = pd.read_csv(raw_path)
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

        fighter_stats = {}
        # For each fighter, collect stats from both red and blue corner appearances
        for _, row in raw.iterrows():
            for prefix, label in [("r_", "r_fighter"), ("b_", "b_fighter")]:
                fname = str(row.get(label, "")).strip()
                if not fname or fname == "nan":
                    continue
                if fname not in fighter_stats:
                    fighter_stats[fname] = {
                        "avg_sig_str_landed": [], "avg_td_landed": [], "avg_sub_att": [],
                        "wins": None, "losses": None,
                        "total_rounds_fought": [],
                        "height_cms": [], "reach_cms": [], "weight_lbs": [], "age": [],
                        "weight_class": None, "total_fight_time_secs": [],
                    }
                f = fighter_stats[fname]
                # Collect numeric stats
                for col, key in [
                    (f"{prefix}avg_sig_str_landed", "avg_sig_str_landed"),
                    (f"{prefix}avg_td_landed", "avg_td_landed"),
                    (f"{prefix}avg_sub_att", "avg_sub_att"),
                    (f"{prefix}total_rounds_fought", "total_rounds_fought"),
                    (f"{prefix}height_cms", "height_cms"),
                    (f"{prefix}reach_cms", "reach_cms"),
                    (f"{prefix}weight_lbs", "weight_lbs"),
                    (f"{prefix}age", "age"),
                ]:
                    val = row.get(col, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            f[key].append(float(val))
                        except (ValueError, TypeError):
                            pass

                # Total fight time
                tft = row.get("total_fight_time_secs", row.get("total_fight_time", None))
                if tft is not None and not (isinstance(tft, float) and np.isnan(tft)):
                    try:
                        f["total_fight_time_secs"].append(float(tft))
                    except (ValueError, TypeError):
                        pass

                # Wins/losses (team-level, shared by both corners)
                if f["wins"] is None:
                    w = row.get(f"{prefix}wins", None)
                    if w is not None and not (isinstance(w, float) and np.isnan(w)):
                        try:
                            f["wins"] = int(float(w))
                        except (ValueError, TypeError):
                            pass
                if f["losses"] is None:
                    l = row.get(f"{prefix}losses", None)
                    if l is not None and not (isinstance(l, float) and np.isnan(l)):
                        try:
                            f["losses"] = int(float(l))
                        except (ValueError, TypeError):
                            pass

                # Weight class
                wc = row.get("weight_class", None)
                if wc is not None and str(wc).strip().lower() != "nan":
                    f["weight_class"] = str(wc).strip().lower()

        # Collapse lists to means
        fighter_db = {}
        for fname, stats in fighter_stats.items():
            entry = {}
            for key in ["avg_sig_str_landed", "avg_td_landed", "avg_sub_att",
                        "total_rounds_fought", "height_cms", "reach_cms",
                        "weight_lbs", "age", "total_fight_time_secs"]:
                vals = stats[key]
                entry[key] = round(float(np.mean(vals)), 1) if vals else None
            entry["wins"] = stats["wins"] if stats["wins"] is not None else 5
            entry["losses"] = stats["losses"] if stats["losses"] is not None else 5
            entry["weight_class"] = stats["weight_class"] or "middleweight"
            entry["avg_fight_time"] = entry["total_fight_time_secs"] or 652
            fighter_db[fname] = entry

        # Save fighter lookup
        fighter_file = MODEL_DIR / "fighter_lookup.json"
        with open(fighter_file, "w") as f:
            json.dump(fighter_db, f, indent=2)
        print(f"  Saved fighter lookup: {len(fighter_db)} fighters to {fighter_file}")

        # ── Merge augmented fighters (not in CSV) ─────────────────────
        aug_path = MODEL_DIR / "fighter_augment.json"
        if aug_path.exists():
            with open(aug_path) as f:
                aug = json.load(f)
            new = 0
            updated = 0
            for name, stats in aug.items():
                if name not in fighter_db:
                    new += 1
                else:
                    updated += 1
                fighter_db[name] = stats  # always apply (overwrites stale CSV)
            if new or updated:
                print(f"  Merged {new} new + {updated} updated fighters from {aug_path}")
                with open(fighter_file, "w") as f:
                    json.dump(fighter_db, f, indent=2)

        # Build weight-class averages
        wc_data = {}
        for fname, entry in fighter_db.items():
            wc = entry["weight_class"]
            if wc not in wc_data:
                wc_data[wc] = {"avg_fight_time": [], "height_cms": [], "weight_lbs": [], "age": []}
            for key in ["avg_fight_time", "height_cms", "weight_lbs", "age"]:
                if entry.get(key) is not None:
                    wc_data[wc][key].append(entry[key])

        wc_avg = {}
        for wc, data in wc_data.items():
            wc_avg[wc] = {k: round(float(np.mean(v)), 1) for k, v in data.items() if v}

        # Add _default from middleweight or first available
        default = wc_avg.get("middleweight", {})
        if not default:
            default = next(iter(wc_avg.values()), {})
        wc_avg["_default"] = default

        wc_file = MODEL_DIR / "wc_averages.json"
        with open(wc_file, "w") as f:
            json.dump(wc_avg, f, indent=2)
        print(f"  Saved weight-class averages: {len(wc_avg)} classes to {wc_file}")


if __name__ == "__main__":
    main()
