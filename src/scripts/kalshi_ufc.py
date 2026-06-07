"""
Kalshi UFC market scanner.
Scans for UFC-related markets, maps fighter names to model predictions,
and computes edges. All UFC series are currently planned (markets=0) —
this scanner is infrastructure ready for when they launch.
"""
import json
import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data.kalshi import KalshiClient
from src.data.ufc import UFCDataSource
from src.features.ufc import build_ufc_features, FEATURE_COLS, WEIGHT_CLASS_FINISH_PCT, STAT_INFO

MODEL_DIR = Path("models/ufc")


def load_model():
    model_file = MODEL_DIR / "winner_v1.json"
    meta_file = MODEL_DIR / "winner_v1.meta.json"
    cal_file = MODEL_DIR / "winner_calibration.json"
    fighter_file = MODEL_DIR / "fighter_lookup.json"
    wc_file = MODEL_DIR / "wc_averages.json"

    if not model_file.exists():
        print("  UFC model not found. Run: python -m src.scripts.train_ufc")
        return None, None, None, None, None

    model = XGBClassifier()
    model.load_model(str(model_file))

    with open(meta_file) as f:
        meta = json.load(f)

    cal = []
    if cal_file.exists():
        with open(cal_file) as f:
            cal = json.load(f)

    fighter_db = {}
    if fighter_file.exists():
        with open(fighter_file) as f:
            fighter_db = json.load(f)

    wc_avg = {}
    if wc_file.exists():
        with open(wc_file) as f:
            wc_avg = json.load(f)

    return model, meta, cal, fighter_db, wc_avg


def get_fighter_stats(fighter_name, fighter_db, wc_avg):
    if fighter_name in fighter_db:
        return fighter_db[fighter_name]
    matches = [(name, stats) for name, stats in fighter_db.items()
               if fighter_name.lower() in name.lower()]
    if matches:
        return sorted(matches, key=lambda x: abs(len(x[0]) - len(fighter_name)))[0][1]
    default_wc = wc_avg.get("_default", wc_avg.get("middleweight", {}))
    return {
        "avg_sig_str_landed": 27.0, "avg_td_landed": 1.3,
        "avg_sub_att": 0.5, "wins": 5, "losses": 5,
        "total_rounds_fought": 10, "height_cms": 178.0,
        "reach_cms": 183.0, "weight_lbs": 170.0, "age": 30,
        "weight_class": "middleweight",
        "avg_fight_time": default_wc.get("avg_fight_time", 652),
    }


def predict_winner(fighter_a, fighter_b, wc, scheduled_rounds, model, features, cal, fighter_db, wc_avg):
    f1 = get_fighter_stats(fighter_a, fighter_db, wc_avg)
    f2 = get_fighter_stats(fighter_b, fighter_db, wc_avg)

    stats = {
        "r_avg_sig_str_landed": f1.get("avg_sig_str_landed", 27.0),
        "b_avg_sig_str_landed": f2.get("avg_sig_str_landed", 27.0),
        "r_avg_td_landed": f1.get("avg_td_landed", 1.3),
        "b_avg_td_landed": f2.get("avg_td_landed", 1.3),
        "r_avg_sub_att": f1.get("avg_sub_att", 0.5),
        "b_avg_sub_att": f2.get("avg_sub_att", 0.5),
        "r_wins": f1.get("wins", 5), "b_wins": f2.get("wins", 5),
        "r_losses": f1.get("losses", 5), "b_losses": f2.get("losses", 5),
        "r_total_rounds_fought": f1.get("total_rounds_fought", 10),
        "b_total_rounds_fought": f2.get("total_rounds_fought", 10),
        "r_height_cms": f1.get("height_cms", 178.0),
        "b_height_cms": f2.get("height_cms", 178.0),
        "r_reach_cms": f1.get("reach_cms", 183.0),
        "b_reach_cms": f2.get("reach_cms", 183.0),
        "r_weight_lbs": f1.get("weight_lbs", 170.0),
        "b_weight_lbs": f2.get("weight_lbs", 170.0),
        "r_age": f1.get("age", 30), "b_age": f2.get("age", 30),
        "weight_class": wc, "no_of_rounds": scheduled_rounds,
        "r_fighter": fighter_a, "b_fighter": fighter_b,
        "game_id": "0", "game_date": pd.Timestamp.now(),
        "total_fight_time_secs": 652, "finish_round": 3,
    }
    row = pd.DataFrame([stats])
    featured = build_ufc_features(row)
    available = [c for c in features if c in featured.columns]
    X = featured[available].fillna(0)

    prob = float(model.predict_proba(X)[0, 1])

    # Calibration correction
    calibrated = prob
    for entry in cal:
        lo, hi = entry["bin_lo"], entry["bin_hi"]
        if lo <= prob < hi:
            calibrated = entry["actual_rate"]
            break

    return calibrated, prob, fighter_a in fighter_db, fighter_b in fighter_db


def parse_fighters_from_title(title: str) -> tuple:
    title_clean = title.strip()
    # Pattern: "Fighter A vs Fighter B" or "Fighter A to defeat/beats Fighter B"
    sep = r"(?:vs\.?|VS\.?|to\s+defeat|to\s+beat|defeats?|beats?)"
    vs_match = re.search(rf"(.+?)\s+{sep}\s+(.+?)(?:\s+wins?|$)", title_clean)
    if vs_match and not vs_match.group(1).strip().endswith(("wins", "win")):
        return vs_match.group(1).strip(), vs_match.group(2).strip()

    return None, None


def get_weight_class(title: str) -> str:
    title_lower = title.lower()
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
        if key in title_lower:
            return val
    return "middleweight"


def get_scheduled_rounds(wc: str) -> int:
    return 5 if "heavy" in wc or "championship" in wc or "title" in wc else 3


def scan():
    kc = KalshiClient()
    model, meta, cal, fighter_db, wc_avg = load_model()
    if model is None:
        return

    # For women's fights, swap wc to generic
    features = meta["features"] if meta else FEATURE_COLS

    # Scan ALL markets for UFC/fight content
    mkts = kc.list_markets(limit=500)
    if mkts is None or mkts.empty:
        print("  No markets available")
        return

    # Filter for fight-related content
    txt = mkts["ticker"].str.cat(mkts["title"], sep=" ", na_rep="")
    mask = txt.str.contains(
        r"UFC|MMA|FIGHT(?:ER)?|BOXING|FIGHTING|WINNER\b|VS\.?\s",
        case=False, na=False, regex=True
    )
    # Exclude multi-event combos without clear fighter names
    ufc = mkts[mask]
    if ufc.empty:
        print("  No UFC/fight markets found on Kalshi (all 25 series still at 0 markets)")
        return

    print(f"  Found {len(ufc)} potential fight markets")

    balance = kc.get_balance()
    results = []

    for _, m in ufc.iterrows():
        ticker = m["ticker"]
        title = m.get("title", "")
        yb = float(m.get("yes_bid_dollars", 0) or 0)
        ya = float(m.get("yes_ask_dollars", 1) or 1)
        if yb <= 0 and ya >= 1.0:
            continue
        yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

        fighter_a, fighter_b = parse_fighters_from_title(title)
        if not fighter_a or not fighter_b:
            continue

        wc = get_weight_class(title)
        sched = get_scheduled_rounds(wc)

        cal_prob, raw_prob, a_in_db, b_in_db = predict_winner(
            fighter_a, fighter_b, wc, sched,
            model, features, cal, fighter_db, wc_avg
        )

        # Prefer_b odds: P(fighter_b wins) = 1 - P(fighter_a wins)
        # Market prices: YES price for {fighter_a} wins = yes_mid
        # If market is on fighter_b, invert
        fighter_in_title = fighter_a.lower() in title.lower()
        opponent_in_title = fighter_b.lower() in title.lower()

        if opponent_in_title and not fighter_in_title:
            p_model = 1 - cal_prob
            market_prob = yes_mid
        else:
            p_model = cal_prob
            market_prob = yes_mid

        edge = p_model - market_prob
        found_a = "✅" if a_in_db else "❌"
        found_b = "✅" if b_in_db else "❌"

        results.append({
            "ticker": ticker, "title": title,
            "fighter_a": fighter_a, "fighter_b": fighter_b,
            "p_model": p_model, "market_prob": market_prob,
            "edge": edge, "price_cents": max(1, int(yes_mid * 100)),
            "a_in_db": a_in_db, "b_in_db": b_in_db,
            "weight_class": wc,
        })

        print(f"  {fighter_a:25s} vs {fighter_b:25s}  "
              f"model={p_model:.0%} mkt={market_prob:.0%} "
              f"edge={edge:+.0%}  {found_a} {found_b}")

    if not results:
        print("  No fight markets could be parsed (all series still at 0 markets)")
        return

    print(f"\n  Balance: ${balance:.2f}")
    qualifying = [r for r in results if r["edge"] >= 0.05 and 0.10 <= r["price_cents"] / 100 <= 0.80 and r["a_in_db"] and r["b_in_db"]]
    print(f"  Qualifying bets (edge≥5%, both fighters in DB): {len(qualifying)}")

    if qualifying:
        print(f"\n  {'Fighter':25s} {'Opponent':25s} {'Model':>6s} {'Mkt':>6s} {'Edge':>6s} {'Price'}")
        print(f"  {'─'*75}")
        for r in sorted(qualifying, key=lambda x: -x["edge"])[:10]:
            print(f"  {r['fighter_a']:25s} {r['fighter_b']:25s} "
                  f"{r['p_model']:.0%}  {r['market_prob']:.0%}  "
                  f"{r['edge']:+.0%}  {r['price_cents']}¢")

    print()
    portfolio = kc._request("GET", "/portfolio/orders", params={"limit": 50}).get("orders", [])
    ufc_orders = [o for o in portfolio if "UFC" in o.get("ticker", "").upper()
                  or "FIGHT" in o.get("ticker", "").upper()]
    if ufc_orders:
        print(f"  Active UFC orders: {len(ufc_orders)}")
        for o in ufc_orders:
            print(f"    {o['ticker']}  {o['status']}  {o.get('yes_price_dollars','?')}")
    else:
        print("  No active UFC orders")


if __name__ == "__main__":
    scan()
