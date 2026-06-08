#!/usr/bin/env python3
"""Backtest World Cup match outcome model against 2022 World Cup results.

Feeds 2022 World Cup matches through the trained ML model and compares
predictions to actual outcomes. Reports accuracy, Brier, and calibration.

WARNING: The model was trained on international match data that includes
the 2022 World Cup matches (no temporal train/test split). Results here
are NOT clean out-of-sample and likely inflated.

Usage:
    python -m src.scripts.backtest_wc
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.world_cup import fetch_all_matches, compute_elo
from src.models.calibrator import EmpiricalCalibrator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "worldcup"
CALIB_DIR = MODEL_DIR / "calibration"

OUTCOME_LABELS = ["home", "draw", "away"]


def _build_form_features(elo_df, elo_ratings, cutoff_date=None):
    """Build recent form features for all teams from ELO data (same as scan_wc.py)."""
    form = {}
    for team in elo_ratings:
        team_matches = elo_df[(elo_df["home_team"] == team) | (elo_df["away_team"] == team)]
        if cutoff_date:
            team_matches = team_matches[team_matches["match_date"] < pd.Timestamp(cutoff_date)]
        team_matches = team_matches.sort_values("match_date").tail(10)

        if team_matches.empty:
            form[team] = {"wr": 0.0, "dr": 0.0, "gs": 0.0, "gc": 0.0, "n": 0}
            continue

        wins, draws, gs, gc = 0, 0, 0, 0
        for _, r in team_matches.iterrows():
            is_home = r["home_team"] == team
            home_score = int(r["home_score"])
            away_score = int(r["away_score"])
            if is_home:
                gs += home_score; gc += away_score
                if home_score > away_score: wins += 1
                elif home_score == away_score: draws += 1
            else:
                gs += away_score; gc += home_score
                if away_score > home_score: wins += 1
                elif away_score == home_score: draws += 1
        n = len(team_matches)
        form[team] = {"wr": wins / n, "dr": draws / n, "gs": gs / n, "gc": gc / n, "n": n}
    return form


def main():
    print("=" * 65)
    print("  WORLD CUP 2022 BACKTEST")
    print("=" * 65)
    print("  WARNING: Model was trained on data that includes these matches.")
    print("  Results are NOT clean out-of-sample and are likely inflated.")

    # 1. Load model
    print("\n1. Loading model...")
    import lightgbm as lgb
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    if not model_path.exists() or not meta_path.exists():
        print("  No trained model found. Run train_worldcup.py first.")
        return
    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    features = meta.get("features", [])
    calibrator = EmpiricalCalibrator(CALIB_DIR) if CALIB_DIR.exists() else None
    print(f"  Model loaded: {meta.get('n_features', '?')} features, Brier={meta.get('test_brier', '?'):.4f}")

    # 2. Fetch all data
    print("\n2. Fetching match data...")
    df = fetch_all_matches()
    elo_df = compute_elo(df)
    # Merge tournament_code back (only in raw df, not carried through compute_elo)
    elo_df = elo_df.merge(
        df[["match_date", "home_team", "away_team", "tournament_code"]],
        on=["match_date", "home_team", "away_team"],
        how="left"
    )
    print(f"  {len(elo_df)} matches with ELO")

    # 3. Filter to 2022 World Cup matches
    print("\n3. Filtering 2022 World Cup matches...")
    wc2022 = elo_df[(elo_df["match_date"].dt.year == 2022) & (elo_df["tournament_code"] == "WC")].copy()
    print(f"  {len(wc2022)} World Cup matches in 2022")

    # 4. Predict each match
    print("\n4. Predicting outcomes...")
    predictions = []

    for _, match in wc2022.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["match_date"]

        # ELO ratings as of match date (no lookahead in features)
        pre_wc_data = elo_df[elo_df["match_date"] < match_date]
        elo_ratings = {}
        for _, row in pre_wc_data.iterrows():
            elo_ratings[row["home_team"]] = row["elo_home_post"]
            elo_ratings[row["away_team"]] = row["elo_away_post"]

        elo_h = elo_ratings.get(home, 1500)
        elo_a = elo_ratings.get(away, 1500)

        # Form features from data before match date only
        form_features = _build_form_features(elo_df, elo_ratings, cutoff_date=match_date)
        hf = form_features.get(home, {"wr": 0, "dr": 0, "gs": 0, "gc": 0, "n": 0})
        af = form_features.get(away, {"wr": 0, "dr": 0, "gs": 0, "gc": 0, "n": 0})

        # Build feature vector
        vec = {}
        for c in features:
            if c == "elo_home": vec[c] = elo_h
            elif c == "elo_away": vec[c] = elo_a
            elif c == "elo_diff": vec[c] = elo_h - elo_a
            elif c in ("h_wr", "h_dr", "h_gs", "h_gc", "h_n"): vec[c] = hf.get(c[2:], 0)
            elif c in ("a_wr", "a_dr", "a_gs", "a_gc", "a_n"): vec[c] = af.get(c[2:], 0)
            else: vec[c] = 0

        x = np.array([vec.get(c, 0) for c in features]).reshape(1, -1).astype(float)
        probs = model.predict(x)[0]

        # Apply calibration
        if calibrator:
            for ci, cn in enumerate(["home", "draw", "away"]):
                cal_p = calibrator.calibrate(cn, 0, probs[ci])
                if cal_p != probs[ci]:
                    probs[ci] = cal_p
            total = probs.sum()
            if total > 0:
                probs = probs / total

        # Actual outcome
        hs, as_ = int(match["home_score"]), int(match["away_score"])
        if hs > as_: actual = 0
        elif as_ > hs: actual = 2
        else: actual = 1

        predictions.append({
            "match_date": str(match_date.date()),
            "home": home,
            "away": away,
            "home_score": hs,
            "away_score": as_,
            "actual": actual,
            "actual_label": OUTCOME_LABELS[actual],
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
            "pred_class": int(np.argmax(probs)),
            "pred_label": OUTCOME_LABELS[int(np.argmax(probs))],
            "correct": actual == int(np.argmax(probs)),
        })

    # 5. Results
    results = pd.DataFrame(predictions)
    if results.empty:
        print("  No predictions generated.")
        return

    n = len(results)
    correct = results["correct"].sum()
    accuracy = correct / n
    print(f"\n  Predictions: {n}, Correct: {correct} ({accuracy:.1%})")

    # Brier score
    y_onehot = np.zeros((n, 3))
    y_onehot[np.arange(n), results["actual"].values] = 1
    preds_array = results[["p_home", "p_draw", "p_away"]].values
    brier = float(np.mean(np.sum((preds_array - y_onehot) ** 2, axis=1)))

    # Naive baseline: always predict HOME (most common outcome)
    majority_class = np.argmax(np.bincount(results["actual"].values))
    naive_preds = np.zeros((n, 3))
    naive_preds[:, majority_class] = 1
    naive_brier = float(np.mean(np.sum((naive_preds - y_onehot) ** 2, axis=1)))

    print(f"  Brier: {brier:.4f} (naive: {naive_brier:.4f})")
    if brier < naive_brier:
        print(f"  ✅ Beats naive baseline by {(naive_brier - brier) / naive_brier:.0%}")
    else:
        print(f"  ❌ Worse than naive baseline")

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    for cls_idx, cls_name in enumerate(OUTCOME_LABELS):
        mask = results["actual"] == cls_idx
        if mask.sum() > 0:
            cls_acc = (results["pred_class"][mask] == cls_idx).mean()
            print(f"    {cls_name:12s}: {cls_acc:.1%} (n={mask.sum()})")

    # Calibration per outcome
    print(f"\n  Calibration bins:")
    for cls_idx, cls_name in enumerate(OUTCOME_LABELS):
        col = f"p_{cls_name}"
        class_preds = results[col].values
        class_actual = (results["actual"].values == cls_idx).astype(int)
        print(f"    {cls_name:10s}: ", end="")
        bins_reported = 0
        for lo in np.arange(0, 1.0, 0.1):
            hi = min(lo + 0.1, 1.0)
            mask = (class_preds >= lo) & (class_preds < hi)
            if mask.sum() >= 3:
                avg_pred = class_preds[mask].mean()
                actual_rate = class_actual[mask].mean()
                err = actual_rate - avg_pred
                marker = "✅" if abs(err) < 0.05 else ("⚠️" if abs(err) < 0.10 else "❌")
                print(f"{lo:.0%}-{hi:.0%}: pred={avg_pred:.0%} actual={actual_rate:.0%} err={err:+.0%} {marker}  ", end="")
                bins_reported += 1
        if bins_reported == 0:
            print("no bins with 3+ samples", end="")
        print()

    # Edge threshold analysis — simulate betting at different edge cutoffs
    # WARNING: Uses uniform market prior (33%) since 2022 Kalshi market prices
    # are not available. Also, model saw these matches during training.
    print(f"\n  Edge threshold analysis (quarter-Kelly, 1 bet per match):")
    print(f"  ⚠️  Uniform 33% market prior — no 2022 Kalshi prices available")
    print(f"  ⚠️  Not out-of-sample — model was trained on this data")
    print(f"  {'Threshold':>10s} {'Bets':>5s} {'Wins':>5s} {'WR%':>6s} {'ROI%':>6s}")
    print(f"  {'-'*10} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")

    for min_edge_pct in [10, 25, 50, 75, 100, 150, 200]:
        bankroll = 100.0
        trades = 0
        wins = 0
        for _, r in results.iterrows():
            preds_3way = np.array([r["p_home"], r["p_draw"], r["p_away"]])

            # Best edge among outcomes (1 bet per match max)
            best_edge = 0
            best_outcome = -1
            best_model_p = 0
            for outcome_idx in range(3):
                model_p = preds_3way[outcome_idx]
                if model_p < 0.10:
                    continue
                fair_p = 1.0 / 3
                edge_pct = (model_p - fair_p) / fair_p * 100
                if edge_pct > best_edge:
                    best_edge = edge_pct
                    best_outcome = outcome_idx
                    best_model_p = model_p

            if best_outcome == -1 or best_edge < min_edge_pct:
                continue

            # Quarter-Kelly (morning_scan.py formula: quarter THEN cap)
            market_prob = 1.0 / 3
            edge = best_edge / 100.0
            kelly_full = edge / max(0.001, 1 - market_prob)
            kelly_quarter = kelly_full * 0.25
            kelly_capped = min(kelly_quarter, 0.03)
            stake = kelly_capped * bankroll
            cost = market_prob
            contracts = max(1, int(stake / cost))
            cost_total = contracts * cost
            if cost_total > bankroll * 0.05:
                continue
            trades += 1
            if r["actual"] == best_outcome:
                bankroll += contracts * 1.0 - cost_total
                wins += 1
            else:
                bankroll -= cost_total
        roi = (bankroll - 100.0) / 100.0 * 100 if trades > 0 else 0
        wr = wins / trades * 100 if trades > 0 else 0
        print(f"  {f'>{min_edge_pct:.0f}%':>10s} {trades:5d} {wins:5d} {wr:5.1f}% {roi:5.1f}%")

    print(f"\n  Results saved to: models/worldcup/backtest_2022.json")
    results.to_json(MODEL_DIR / "backtest_2022.json", orient="records", indent=2)
    print(f"  Done.")


if __name__ == "__main__":
    main()
