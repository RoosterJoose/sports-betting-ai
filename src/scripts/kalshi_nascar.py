"""
Kalshi NASCAR market scanner.
Scans NASCAR markets (KXNASCARRACE, KXNASCARTOP5, KXNASCARTOP10, etc.),
maps drivers to model predictions, and computes edges against Kalshi prices.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data.kalshi import KalshiClient
from src.data.nascar import NASCARDataSource
from src.features.nascar import NASCARFeatureEngineer

MODEL_DIR = Path("models/nascar")

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
]

SERIES_MAP = {
    "KXNASCARRACE": ("race", "win"),
    "KXNASCARTOP3": ("top3", "top5"),
    "KXNASCARTOP5": ("top5", "top5"),
    "KXNASCARTOP10": ("top10", "top10"),
    "KXNASCARTOP20": ("top20", "top10"),
}


def load_models():
    models = {}
    for name in ["win", "top5", "top10"]:
        model_file = MODEL_DIR / f"{name}.json"
        meta_file = MODEL_DIR / f"{name}.meta.json"
        if not model_file.exists():
            return None, {}
        model = XGBClassifier()
        model.load_model(str(model_file))
        with open(meta_file) as f:
            meta = json.load(f)
        models[name] = (model, meta)
    return models, meta.get("features", FEATURES)


def get_driver_stats(driver_name, featured):
    match = featured[featured["driver_name"].str.lower().str.contains(driver_name.lower(), na=False)]
    if match.empty:
        return None
    latest = match.sort_values("race_number", ascending=False).iloc[0]
    return {k: latest.get(k, 0) for k in FEATURES}


def parse_driver_from_title(title):
    title_lower = title.lower()
    title_lower = re.sub(r"\bwill\s+", "", title_lower)
    title_lower = re.sub(r"\s+finish\s+in\s+the\s+top\s+\d+\s+at\s+.*", "", title_lower)
    title_lower = re.sub(r"\s+be\s+the\s+.*", "", title_lower)
    title_lower = re.sub(r"\s+", " ", title_lower).strip()
    return title_lower


def infer_event_code(ticker):
    parts = ticker.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def scan():
    kc = KalshiClient()
    models_and_features = load_models()
    if models_and_features is None:
        print("  NASCAR models not trained. Run: python -m src.scripts.train_nascar")
        return

    models, features = models_and_features

    try:
        ds = NASCARDataSource()
        df = ds.fetch_player_game_logs(["2024", "2025"])
        from types import SimpleNamespace
        cfg = SimpleNamespace(rolling_windows=[3, 5, 10], recency_decay=0.003)
        fe = NASCARFeatureEngineer(cfg)
        featured = fe.build_features(df)
    except Exception as e:
        print(f"  Data error: {e}")
        featured = pd.DataFrame()

    all_rows = []
    for series_ticker, (market_label, model_name) in SERIES_MAP.items():
        mkts = kc.list_markets(series_ticker=series_ticker, limit=100)
        if mkts is not None and not mkts.empty:
            mkts = mkts.copy()
            mkts["series_label"] = market_label
            mkts["model_name"] = model_name
            all_rows.append(mkts)

    if not all_rows:
        print("  No NASCAR markets available on Kalshi")
        return

    nascar = pd.concat(all_rows, ignore_index=True)
    print(f"  Found {len(nascar)} NASCAR markets")

    balance = kc.get_balance()
    results = []

    for _, m in nascar.iterrows():
        ticker = m["ticker"]
        title = m.get("title", "")
        market_label = m["series_label"]
        model_name = m["model_name"]
        yb = float(m.get("yes_bid_dollars", 0) or 0)
        ya = float(m.get("yes_ask_dollars", 1) or 1)
        if yb <= 0 and ya >= 1.0:
            continue
        yes_mid = round((yb + ya) / 2.0, 4)
        event_code = infer_event_code(ticker)

        driver_str = parse_driver_from_title(title)
        if not driver_str:
            continue

        if featured.empty:
            continue

        stats = get_driver_stats(driver_str, featured)
        if stats is None or any(v is None for v in stats.values()):
            results.append({
                "ticker": ticker, "title": title, "driver": driver_str,
                "event": event_code, "type": market_label,
                "found": False,
            })
            continue

        X = pd.DataFrame([stats]).fillna(0)
        available = [c for c in features if c in X.columns]

        model_obj = models.get(model_name)
        if model_obj is None:
            continue

        model, meta = model_obj
        prob = float(model.predict_proba(X[available])[0, 1])

        # Apply Platt calibration
        platt_slope = meta.get("platt_slope")
        platt_intercept = meta.get("platt_intercept")
        if platt_slope is not None:
            log_odds = np.log(max(prob, 1e-6) / max(1 - prob, 1e-6))
            prob = 1.0 / (1.0 + np.exp(-(log_odds * platt_slope + platt_intercept)))

        edge = prob - yes_mid
        price_cents = max(1, min(99, int(yes_mid * 100)))

        results.append({
            "ticker": ticker, "title": title,
            "driver": driver_str, "event": event_code,
            "type": market_label,
            "p_model": prob, "market_prob": yes_mid,
            "edge": edge, "price_cents": price_cents,
            "found": True,
        })

    cup_results = [r for r in results if r.get("found") and r["event"] == "FIRC26"]
    truck_results = [r for r in results if r.get("found") and r["event"] != "FIRC26"]
    not_found = [r for r in results if not r.get("found")]

    print(f"\n=== FireKeepers Casino 400 (Michigan, Sun June 7) ===")

    for mtype in ["top5", "top10", "top20", "race", "top3"]:
        subset = [r for r in cup_results if r["type"] == mtype]
        if not subset:
            continue
        base_rate_map = {"top5": "11.0%", "top10": "24.5%", "top20": "45%", "race": "1.4%"}
        br = base_rate_map.get(mtype, "")
        print(f"\n--- {mtype.upper()} (model base rate={br}) ---")
        for r in sorted(subset, key=lambda x: -x.get("edge", 0))[:8]:
            print(f"  {r['driver']:30s} model={r['p_model']:.0%} mkt={r['market_prob']:.0%} "
                  f"edge={r['edge']:+.0%}  {r['price_cents']}¢")

    qualifying = [r for r in cup_results
                  if r["edge"] >= 0.05 and 0.10 <= r["price_cents"] / 100 <= 0.80]
    print(f"\n  Qualifying bets (edge≥5%, 10-80¢): {len(qualifying)}")
    if qualifying:
        for r in sorted(qualifying, key=lambda x: -x["edge"]):
            mtype = r['type']
            print(f"  {r['driver']:30s} {mtype:6s} model={r['p_model']:.0%} "
                  f"mkt={r['market_prob']:.0%} edge={r['edge']:+.0%}  {r['price_cents']}¢")

    print(f"\n  Balance: ${balance:.2f}")


if __name__ == "__main__":
    scan()
