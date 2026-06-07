import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

from src.data.prizepicks import PrizePicksScraper
from src.features.ufc import build_ufc_features, FEATURE_COLS, WEIGHT_CLASS_FINISH_PCT, STAT_INFO

MODEL_DIR = Path("models/ufc")


def load_models():
    reg = XGBRegressor()
    reg.load_model(str(MODEL_DIR / "total_rounds_reg.json"))
    cls = XGBClassifier()
    cls.load_model(str(MODEL_DIR / "total_rounds_cls.json"))
    with open(MODEL_DIR / "fighter_lookup.json") as f:
        fighter_db = json.load(f)
    with open(MODEL_DIR / "wc_averages.json") as f:
        wc_avg = json.load(f)
    with open(MODEL_DIR / "feature_stats.json") as f:
        feat_stats = json.load(f)
    with open(MODEL_DIR / "total_rounds.meta.json") as f:
        meta = json.load(f)
    return reg, cls, fighter_db, wc_avg, feat_stats, meta


def get_fighter_stats(fighter_name: str, fighter_db: dict, wc_avg: dict) -> dict:
    """Get fighter stats from lookup table or defaults."""
    if fighter_name in fighter_db:
        return fighter_db[fighter_name]

    # Try fuzzy match
    matches = [(name, stats) for name, stats in fighter_db.items()
               if fighter_name.lower() in name.lower()
               or any(
                   all(part.lower() in name.lower() for part in fighter_name.split())
                   for _ in [1]
               )]

    if matches:
        return matches[0][1]

    # Fall back to defaults
    default_wc = wc_avg.get("_default", wc_avg.get("middleweight", {}))
    return {
        "avg_sig_str_landed": 27.0,
        "avg_td_landed": 1.3,
        "avg_sub_att": 0.5,
        "wins": 5,
        "losses": 5,
        "total_rounds_fought": 10,
        "height_cms": 178.0,
        "reach_cms": 183.0,
        "weight_lbs": 170.0,
        "age": 30,
        "weight_class": "middleweight",
        "avg_fight_time": default_wc.get("avg_fight_time", 652),
    }


def build_fight_row(fighter1: dict, fighter2: dict, wc: str, scheduled_rounds: int) -> pd.DataFrame:
    stats = {
        "r_avg_sig_str_landed": fighter1.get("avg_sig_str_landed", 27.0),
        "b_avg_sig_str_landed": fighter2.get("avg_sig_str_landed", 27.0),
        "r_avg_td_landed": fighter1.get("avg_td_landed", 1.3),
        "b_avg_td_landed": fighter2.get("avg_td_landed", 1.3),
        "r_avg_sub_att": fighter1.get("avg_sub_att", 0.5),
        "b_avg_sub_att": fighter2.get("avg_sub_att", 0.5),
        "r_wins": fighter1.get("wins", 5),
        "b_wins": fighter2.get("wins", 5),
        "r_losses": fighter1.get("losses", 5),
        "b_losses": fighter2.get("losses", 5),
        "r_total_rounds_fought": fighter1.get("total_rounds_fought", 10),
        "b_total_rounds_fought": fighter2.get("total_rounds_fought", 10),
        "r_height_cms": fighter1.get("height_cms", 178.0),
        "b_height_cms": fighter2.get("height_cms", 178.0),
        "r_reach_cms": fighter1.get("reach_cms", 183.0),
        "b_reach_cms": fighter2.get("reach_cms", 183.0),
        "r_weight_lbs": fighter1.get("weight_lbs", 170.0),
        "b_weight_lbs": fighter2.get("weight_lbs", 170.0),
        "r_age": fighter1.get("age", 30),
        "b_age": fighter2.get("age", 30),
        "weight_class": wc,
        "no_of_rounds": scheduled_rounds,
        "r_fighter": "red_corner",
        "b_fighter": "blue_corner",
        "game_id": "0",
        "game_date": pd.Timestamp.now(),
        "total_fight_time_secs": 652,
        "finish_round": 3,
    }
    df = pd.DataFrame([stats])
    featured = build_ufc_features(df)
    available = [c for c in FEATURE_COLS if c in featured.columns]
    X = featured[available].fillna(0)
    return X, featured


def predict_over_rounds(
    fighter_name: str,
    opponent_name: str,
    line: float,
    wc: str,
    scheduled_rounds: int,
    reg, cls, fighter_db, wc_avg, feat_stats, meta
) -> dict:
    """Predict P(OVER line) for a UFC Total Rounds bet."""
    f1 = get_fighter_stats(fighter_name, fighter_db, wc_avg)
    f2 = get_fighter_stats(opponent_name, fighter_db, wc_avg)

    X, featured = build_fight_row(f1, f2, wc, scheduled_rounds)
    if X.empty:
        return None

    reg_pred = reg.predict(X)[0]
    cls_proba = cls.predict_proba(X)[0]
    decision_prob = cls_proba[1]

    # Compute P(OVER line) based on line threshold
    line_seconds = line * 300
    reg_std = feat_stats.get("target_std", {}).get("std", 356)

    # Simple: use normal CDF with regression prediction
    p_over_reg = 1 - norm.cdf((line_seconds - reg_pred) / reg_std)

    # For line=2.5 in 3-round fights, use the decision classifier directly
    if line == 2.5 and scheduled_rounds <= 3:
        p_over_cls = decision_prob
        p_over = 0.6 * p_over_cls + 0.4 * p_over_reg
    elif line == 3.5 and scheduled_rounds >= 5:
        p_over_cls = decision_prob
        p_over = 0.6 * p_over_cls + 0.4 * p_over_reg
    elif line <= 1.5:
        p_over = min(0.95, 0.5 + 0.5 * (1 - norm.cdf((line_seconds - reg_pred) / reg_std)))
    else:
        p_over = p_over_reg

    p_over = np.clip(p_over, 0.05, 0.95)

    return {
        "fighter": fighter_name,
        "opponent": opponent_name,
        "line": line,
        "weight_class": wc,
        "scheduled_rounds": scheduled_rounds,
        "reg_pred_secs": round(reg_pred, 1),
        "reg_pred_rounds": round(reg_pred / 300, 2),
        "decision_prob": round(decision_prob, 3),
        "p_over": round(p_over, 3),
        "f1_in_db": fighter_name in fighter_db,
        "f2_in_db": opponent_name in fighter_db,
    }


def get_prizepicks_ufc():
    scraper = PrizePicksScraper()
    lines = scraper.fetch_lines("ufc")
    if lines.empty:
        print("No UFC lines on PrizePicks")
        return []
    return lines


def classify_weight_class(team_str: str) -> str:
    team_lower = team_str.strip().lower()

    wc_map = {
        "flyweight": "flyweight", "bantamweight": "bantamweight",
        "featherweight": "featherweight", "lightweight": "lightweight",
        "welterweight": "welterweight", "middleweight": "middleweight",
        "light heavyweight": "light heavyweight", "heavyweight": "heavyweight",
        "women's strawweight": "women's strawweight",
        "women's flyweight": "women's flyweight",
        "women's bantamweight": "women's bantamweight",
    }
    for key, val in wc_map.items():
        if key in team_lower:
            return val
    return "middleweight"


def main():
    reg, cls, fighter_db, wc_avg, feat_stats, meta = load_models()

    print("=== UFC Predictor ===")
    print(f"Model: 57.4% decision classifier, R²≈0 regressor")
    print(f"Fighters in DB: {len(fighter_db)}")
    print()

    lines = get_prizepicks_ufc()
    if lines.empty:
        print("No UFC lines available on PrizePicks")
        return

    # Group by fight (fighter + opponent)
    fights = {}
    for _, row in lines.iterrows():
        fighter = str(row.get("player_name", "")).strip()
        opponent = str(row.get("description", "")).strip()
        stat_type = str(row.get("stat_type", ""))
        line = float(row.get("line_score", 2.5))

        if not fighter or not opponent:
            continue

        key = tuple(sorted([fighter, opponent]))
        if key not in fights:
            fights[key] = {
                "fighter": fighter,
                "opponent": opponent,
                "total_rounds_lines": [],
                "stat_type": stat_type,
            }
        fights[key]["total_rounds_lines"].append(line)

    predictions = []
    for key, fight in fights.items():
        fighter = fight["fighter"]
        opponent = fight["opponent"]

        # Try to determine weight class
        wc = "middleweight"
        for name in [fighter, opponent]:
            fw = get_fighter_stats(name, fighter_db, wc_avg)
            if fw.get("weight_class") and fw["weight_class"] not in ("unknown", "middleweight"):
                wc = fw["weight_class"]
                break

        scheduled_rounds = 3 if "women" not in wc.lower() else 3

        for line_val in fight["total_rounds_lines"]:
            result = predict_over_rounds(
                fighter, opponent, line_val, wc, scheduled_rounds,
                reg, cls, fighter_db, wc_avg, feat_stats, meta
            )
            if result:
                predictions.append(result)

    predictions.sort(key=lambda p: abs(p["p_over"] - 0.5), reverse=True)

    print(f"Found {len(fights)} UFC fights, {len(predictions)} predictions")
    print()
    print(f"{'Fighter':25s} {'Opponent':25s} {'Line':>5s} {'P(OVER)':>8s} {'Decision':>8s} {'DB':>4s}")
    print("-" * 80)
    for p in predictions[:20]:
        pct = f"{p['p_over']*100:.0f}%"
        dec = f"{p['decision_prob']*100:.0f}%"
        db = "Y" if p["f1_in_db"] else "N"
        print(f"{p['fighter']:25s} {p['opponent']:25s} {p['line']:>4.1f}  {pct:>7s}  {dec:>7s}  {db:>3s}")

    strong_over = [p for p in predictions if p["p_over"] > 0.65]
    strong_under = [p for p in predictions if p["p_over"] < 0.35]
    print()
    print(f"Strong OVER (P>65%): {len(strong_over)} predictions")
    for p in strong_over[:5]:
        print(f"  {p['fighter']:25s} vs {p['opponent']:25s} line={p['line']:.1f} P={p['p_over']:.1%}")
    print(f"Strong UNDER (P<35%): {len(strong_under)} predictions")
    for p in strong_under[:5]:
        print(f"  {p['fighter']:25s} vs {p['opponent']:25s} line={p['line']:.1f} P={p['p_over']:.1%}")

    if not strong_over and not strong_under:
        print("No strong predictions. Model is near-random for these matchups.")

        avg_over = np.mean([p["p_over"] for p in predictions])
        print(f"\nAverage P(OVER) across all: {avg_over:.1%}")
        print("Tip: Only bet if P(OVER) deviates significantly from 50%")


if __name__ == "__main__":
    main()
