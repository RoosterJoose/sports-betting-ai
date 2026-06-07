"""
NASCAR Weekly Pipeline — Friday/Saturday qualifying → Sunday race.

Schedule:
  Friday:   Practice sessions → scrape practice speeds from Wikipedia
  Saturday: Qualifying → scrape starting grid from Wikipedia
  Sunday:   Run model with live qualifying position, execute bets

Usage:
  python -m src.main kalshi nascar              # paper trade
  python -m src.main kalshi nascar --live        # live trade
  python -m src.main kalshi nascar --bankroll 50 # custom bankroll
"""
import json
import re
import sys
import time
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kalshi import KalshiClient
from src.data.nascar import NASCARDataSource, WIKI_HEADERS, _parse_finish
from src.data.nascar_feed import (
    fetch_wikipedia_starting_grid,
    get_race_schedule,
)
from src.features.nascar import NASCARFeatureEngineer
from src.utils.trade_tracker import TradeTracker

MODEL_DIR = Path("models/nascar")
PAPER_LOG = Path("data/paper_trades/nascar.csv")

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
    "KXNASCARTOP5": ("top5", "top5"),
    "KXNASCARTOP10": ("top10", "top10"),
    "KXNASCARTOP3": ("top3", "top5"),
    "KXNASCARTOP20": ("top20", "top10"),
}

WIKI_HEADERS_LOCAL = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def detect_current_race():
    """Detect the current year's upcoming cup race.
    Returns (year, race_number, race_name, wiki_title, race_date) or None."""
    today = date.today()
    for year in [today.year, today.year - 1]:
        schedule = get_race_schedule(year)
        if schedule.empty:
            continue
        for _, r in schedule.iterrows():
            race_no = int(r.get("race_number", 0))
            wiki = r.get("wiki_title", "")
            if not wiki:
                continue
            url = f"https://en.wikipedia.org/wiki/{wiki.replace(' ', '_')}"
            try:
                resp = requests.get(url, headers=WIKI_HEADERS_LOCAL, timeout=8)
                if resp.status_code != 200:
                    continue
            except Exception:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            has_race_results = False
            has_qualifying = False
            for table in soup.find_all("table", class_="wikitable"):
                rows_in_table = table.find_all("tr")
                if len(rows_in_table) < 3:
                    continue
                hdrs = [h.get_text(strip=True).lower()[:10] for h in rows_in_table[0].find_all(["th", "td"])]
                hdrs_str = " ".join(hdrs)
                # Race results = has grid + laps columns + many rows
                if "grid" in hdrs_str and "lap" in hdrs_str and len(rows_in_table) > 20:
                    has_race_results = True
                    break
            if has_race_results:
                continue
            # Check for qualifying (pos + no + driver + time + speed)
            for table in soup.find_all("table", class_="wikitable"):
                rows_in_table = table.find_all("tr")
                if len(rows_in_table) < 5:
                    continue
                hdrs = [h.get_text(strip=True).lower()[:10] for h in rows_in_table[0].find_all(["th", "td"])]
                hdrs_str = " ".join(hdrs)
                if "pos" in hdrs_str and "time" in hdrs_str and "speed" in hdrs_str and "driver" in hdrs_str:
                    has_qualifying = True
                    break
            return {
                "year": year,
                "race_number": race_no,
                "race_name": r.get("race_name", ""),
                "wiki_title": wiki,
                "has_qualifying": has_qualifying,
            }
    return None


def scrape_qualifying_results(wiki_title):
    """Scrape qualifying results from Wikipedia race article.
    Returns {driver_lower: {'position': int, 'time': str, 'speed': float}}."""
    url = f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers=WIKI_HEADERS_LOCAL, timeout=10)
        if resp.status_code != 200:
            return {}
    except Exception:
        return {}
    soup = BeautifulSoup(resp.text, "html.parser")
    for table in soup.find_all("table", class_="wikitable"):
        rows = table.find_all("tr")
        if len(rows) < 5:
            continue
        hdrs = [h.get_text(strip=True).lower()[:10] for h in rows[0].find_all(["th", "td"])]
        # Qualifying table has: pos, no., driver, time, speed (NOT grid, laps, points)
        has_pos = "pos" in hdrs or "pos." in hdrs
        has_no = "no" in hdrs or "no." in hdrs or "#" in hdrs
        has_driver = "driver" in hdrs
        has_time = "time" in hdrs
        has_speed = "speed" in hdrs
        has_grid = "grid" in hdrs
        has_laps = "lap" in hdrs
        if not (has_pos and has_no and has_driver and has_time and has_speed):
            continue
        if has_grid or has_laps:
            continue
        # Find column indices
        pos_idx = next(i for i, h in enumerate(hdrs) if h in ("pos", "pos."))
        drv_idx = next(i for i, h in enumerate(hdrs) if "driver" in h)
        time_idx = next(i for i, h in enumerate(hdrs) if "time" in h)
        speed_idx = next(i for i, h in enumerate(hdrs) if "speed" in h)
        results = {}
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) <= max(pos_idx, drv_idx, time_idx, speed_idx):
                continue
            try:
                pos = int(re.match(r"(\d+)", texts[pos_idx])[1])
            except (ValueError, TypeError, AttributeError):
                continue
            driver = re.sub(r"\s*\([^)]*\)", "", texts[drv_idx]).strip().lower()
            lap_time = texts[time_idx] if time_idx < len(texts) else ""
            try:
                speed = float(re.sub(r"[^0-9.]", "", texts[speed_idx]))
            except (ValueError, TypeError):
                speed = 0.0
            if pos >= 1 and pos <= 45:
                results[driver] = {"position": pos, "time": lap_time, "speed": speed}
        if len(results) >= 10:
            return results
    return {}


def scrape_practice_results(wiki_title):
    """Scrape practice results from Wikipedia race article.
    Returns {driver_lower: {'position': int, 'speed': float}}."""
    url = f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers=WIKI_HEADERS_LOCAL, timeout=10)
    except Exception:
        return {}
    soup = BeautifulSoup(resp.text, "html.parser")
    for table in soup.find_all("table", class_="wikitable"):
        rows = table.find_all("tr")
        if len(rows) < 5:
            continue
        hdrs_before = (table.find_previous("h3") or table.find_previous("h2") or table.find_previous("span"))
        section_text = hdrs_before.get_text(strip=True).lower() if hdrs_before else ""
        if "practice" not in section_text:
            continue
        hdrs = [h.get_text(strip=True).lower()[:10] for h in rows[0].find_all(["th", "td"])]
        if "pos" not in hdrs or "driver" not in hdrs or "speed" not in hdrs:
            continue
        pos_idx = next(i for i, h in enumerate(hdrs) if h in ("pos", "pos."))
        drv_idx = next(i for i, h in enumerate(hdrs) if "driver" in h)
        speed_idx = next(i for i, h in enumerate(hdrs) if "speed" in h)
        results = {}
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) <= max(pos_idx, drv_idx, speed_idx):
                continue
            try:
                pos = int(re.match(r"(\d+)", texts[pos_idx])[1])
            except (ValueError, TypeError, AttributeError):
                continue
            driver = re.sub(r"\s*\([^)]*\)", "", texts[drv_idx]).strip().lower()
            try:
                speed = float(re.sub(r"[^0-9.]", "", texts[speed_idx]))
            except (ValueError, TypeError):
                speed = 0.0
            if pos >= 1 and pos <= 45 and speed > 0:
                results[driver] = {"position": pos, "speed": speed}
        if len(results) >= 10:
            return results
    return {}


# Models with Brier >= this threshold have no predictive signal (effectively random)
BRIER_THRESHOLD = 0.22

def load_models():
    models = {}
    featured_list = None
    for name in ["win", "top5", "top10"]:
        model_file = MODEL_DIR / f"{name}.json"
        meta_file = MODEL_DIR / f"{name}.meta.json"
        if not model_file.exists():
            return None, {}
        model = XGBClassifier()
        model.load_model(str(model_file))
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Skip models with no predictive signal
        brier = meta.get("oof_brier", meta.get("calibrated_brier", 0.5))
        base_rate = meta.get("base_rate", 0.5)
        
        # Naive Brier: what you'd get by always predicting the base rate
        # If model can't beat this, it's worse than useless
        brier_naive = base_rate * (1 - base_rate)**2 + (1 - base_rate) * base_rate**2
        
        if brier >= BRIER_THRESHOLD:
            print(f"  ⚠ SKIPPING {name}: Brier={brier:.4f} >= threshold={BRIER_THRESHOLD}")
            continue
        if brier >= brier_naive:
            print(f"  ⚠ SKIPPING {name}: Brier={brier:.4f} >= naive={brier_naive:.4f} (worse than constant prediction)")
            continue
        
        print(f"  ✓ LOADED {name}: Brier={brier:.4f} (naive={brier_naive:.4f})")
        models[name] = (model, meta)
        if featured_list is None:
            featured_list = meta.get("features", FEATURES)
    
    if not models:
        print(f"  ✗ No models passed Brier threshold (all >= {BRIER_THRESHOLD})")
        return None, {}
    return models, featured_list or FEATURES


def get_driver_stats(driver_name, featured):
    match = featured[featured["driver_name"].str.lower().str.contains(driver_name.lower(), na=False)]
    if match.empty:
        return None
    latest = match.sort_values("race_number", ascending=False).iloc[0]
    return {k: latest.get(k, 0) for k in FEATURES}


def parse_driver_from_title(title):
    t = title.lower()
    t = re.sub(r"\bwill\s+", "", t)
    t = re.sub(r"\s+finish\s+in\s+the\s+top\s+\d+\s+at\s+.*", "", t)
    t = re.sub(r"\s+be\s+the\s+.*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def platt_calibrate(prob, meta):
    slope = meta.get("platt_slope")
    intercept = meta.get("platt_intercept")
    if slope is None or intercept is None:
        return prob
    log_odds = np.log(max(prob, 1e-6) / max(1 - prob, 1e-6))
    return 1.0 / (1.0 + np.exp(-(log_odds * slope + intercept)))


def log_paper_trade(ticker, driver, market_type, side, price, size, model_prob, market_prob, edge, race_name):
    PAPER_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "race": race_name,
        "ticker": ticker,
        "driver": driver,
        "type": market_type,
        "side": side,
        "price": price,
        "size": size,
        "model_prob": round(model_prob, 4),
        "market_prob": round(market_prob, 4),
        "edge": round(edge, 4),
        "pnl": 0.0,
    }
    df = pd.DataFrame([record])
    if PAPER_LOG.exists():
        existing = pd.read_csv(PAPER_LOG)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(PAPER_LOG, index=False)
    return record


def run_weekly_scan(bankroll: float = 100.0, paper_only: bool = True):
    print(f"\n{'='*65}")
    print(f"  NASCAR WEEKLY PIPELINE — {datetime.now().strftime('%A %B %d, %Y')}")
    print(f"  Bankroll: ${bankroll:.2f}  |  Mode: {'PAPER' if paper_only else 'LIVE'}")
    print(f"{'='*65}")

    # Step 1: Detect current race
    race_info = detect_current_race()
    if race_info is None:
        print("  No upcoming race detected")
        return

    print(f"\n  Current race: R{race_info['race_number']} — {race_info['race_name']}")
    print(f"  Wikipedia article exists: {race_info['wiki_title']}")

    # Step 2: Scrape qualifying results (available Saturday after qualifying)
    print(f"\n  ── Scraping qualifying data ──")
    qualifying = scrape_qualifying_results(race_info["wiki_title"])
    if qualifying:
        # Show top qualifiers
        sorted_drivers = sorted(qualifying.items(), key=lambda x: x[1]["position"])
        print(f"  ✅ Found qualifying results for {len(qualifying)} drivers!")
        for driver, data in sorted_drivers[:5]:
            print(f"     P{data['position']:2d}: {driver:25s} {data['speed']:.1f} mph")
        # Show a few backmarkers
        for driver, data in sorted_drivers[-3:]:
            print(f"     P{data['position']:2d}: {driver:25s} {data['speed']:.1f} mph")
    else:
        print(f"  ❌ No qualifying results yet (available after qualifying Saturday)")

    # Step 3: Scrape practice results (available Friday after practice)
    practice = scrape_practice_results(race_info["wiki_title"])
    if practice:
        sorted_practice = sorted(practice.items(), key=lambda x: x[1]["position"])
        print(f"\n  ✅ Found practice results for {len(practice)} drivers!")
        for driver, data in sorted_practice[:3]:
            print(f"     P{data['position']:2d}: {driver:25s} {data['speed']:.1f} mph")
    else:
        print(f"\n  ❌ No practice results yet (available after practice Friday)")

    # Step 4: Load historical data and features
    ds = NASCARDataSource()

    # Load current season (2026) + previous season (2025) for feature stability
    data_years = ["2025", "2026"] if race_info["year"] == 2026 else ["2024", "2025"]
    df = ds.fetch_player_game_logs(data_years)
    from types import SimpleNamespace
    cfg = SimpleNamespace(rolling_windows=[3, 5, 10], recency_decay=0.003)
    fe = NASCARFeatureEngineer(cfg)
    featured = fe.build_features(df)

    # Step 5: Load models and scan Kalshi markets
    loaded_models = load_models()
    if loaded_models is None or loaded_models[0] is None:
        print("\n  No usable models. Either models not trained or all failed quality checks.")
        if loaded_models is None:
            print("  Run: python -m src.scripts.train_nascar")
        return
    models, feature_list = loaded_models

    kc = KalshiClient()
    balance = kc.get_balance() if not paper_only else bankroll
    available = balance

    all_markets = []
    for series_ticker, (market_label, model_name) in SERIES_MAP.items():
        mkts = kc.list_markets(series_ticker=series_ticker, limit=100)
        if mkts is not None and not mkts.empty:
            mkts = mkts.copy()
            mkts["series_label"] = market_label
            mkts["model_name"] = model_name
            all_markets.append(mkts)

    if not all_markets:
        print("  No NASCAR markets available on Kalshi")
        return

    nascar = pd.concat(all_markets, ignore_index=True)
    print(f"\n  Kalshi markets: {len(nascar)} total")

    # Step 6: Scan each market
    results = []
    no_targets = []

    for _, m in nascar.iterrows():
        ticker = m["ticker"]
        title = m.get("title", "")
        market_label = m["series_label"]
        model_name = m["model_name"]
        yb = float(m.get("yes_bid_dollars", 0) or 0)
        ya = float(m.get("yes_ask_dollars", 1) or 1)
        if yb <= 0 and ya >= 1.0:
            continue

        driver_str = parse_driver_from_title(title)
        if not driver_str:
            continue

        stats = get_driver_stats(driver_str, featured)
        if stats is None:
            continue

        X = pd.DataFrame([stats]).fillna(0)
        avail = [c for c in feature_list if c in X.columns]

        model_obj = models.get(model_name)
        if model_obj is None:
            continue
        model, meta = model_obj

        # Model probability
        prob_yes = float(model.predict_proba(X[avail])[0, 1])
        prob_yes = platt_calibrate(prob_yes, meta)

        # Override with live qualifying position if available
        driver_lower = driver_str.lower()
        if driver_lower in qualifying:
            actual_start = qualifying[driver_lower]["position"]
            # Qualifying position bonus: top-5 starters get +, backmarkers get -
            start_advantage = max(-0.10, (21 - actual_start) * 0.004)
            prob_yes = max(0.01, min(0.95, prob_yes + start_advantage))

        # Market prices
        yes_mid = round((yb + ya) / 2.0, 4)
        no_bid = round(1.0 - ya, 4)
        no_ask = round(1.0 - yb, 4)
        no_mid = round((no_bid + no_ask) / 2.0, 4)

        yes_edge = prob_yes - yes_mid
        no_edge = (1 - prob_yes) - no_mid
        price_cents = max(1, min(99, int(yes_mid * 100)))

        has_qualifying = driver_lower in qualifying
        record = {
            "ticker": ticker,
            "title": title,
            "driver": driver_str,
            "type": market_label,
            "prob_yes": prob_yes,
            "prob_no": 1 - prob_yes,
            "yes_mid": yes_mid,
            "no_mid": no_mid,
            "yes_edge": round(yes_edge, 4),
            "no_edge": round(no_edge, 4),
            "price_cents": price_cents,
            "has_q": has_qualifying,
        }
        results.append(record)

        # YES-side: model says probability > market, edge >= 5%, price 10-80¢
        if yes_edge >= 0.05 and 0.10 <= yes_mid <= 0.80:
            no_targets.append(record)

    # Display YES-side targets
    no_targets.sort(key=lambda x: -x["yes_edge"])
    print(f"\n  {'='*55}")
    print(f"  YES-SIDE OPPORTUNITIES (edge ≥5%, 10-80¢): {len(no_targets)}")
    print(f"  {'='*55}")
    if no_targets:
        for r in no_targets[:12]:
            q_flag = "⚑" if r["has_q"] else " "
            print(f"  {q_flag} {r['driver']:28s} {r['type']:6s} "
                  f"model={r['prob_yes']:.0%} mkt={r['yes_mid']:.0%} "
                  f"edge={r['yes_edge']:+.0%}  ¢{r['price_cents']}")
        if len(no_targets) > 12:
            print(f"  ... and {len(no_targets) - 12} more")
    else:
        print("  No qualifying YES-side opportunities found")

    # Step 7: Execute
    print(f"\n  {'='*55}")
    print(f"  EXECUTION: {'PAPER TRADE' if paper_only else 'LIVE'}")
    print(f"  Available: ${available:.2f}")
    print(f"  {'='*55}")

    placed = 0
    max_trades = 8
    max_exposure = available * 0.25
    cumulative_cost = 0
    tracker_trades = []

    for r in no_targets:
        if placed >= max_trades:
            break
        if cumulative_cost >= max_exposure:
            break

        yes_price = r["yes_mid"]
        if yes_price <= 0 or yes_price >= 0.90:
            continue

        # Position sizing: 3% of bankroll per position
        max_per_position = available * 0.03
        position_size = min(max_per_position, max_exposure - cumulative_cost)
        if position_size < 2.00:
            continue

        size = max(1, int(position_size / yes_price))
        position_cost = size * yes_price
        if position_cost > max_exposure - cumulative_cost:
            continue

        side = "BUY_YES"

        if paper_only:
            log_paper_trade(
                r["ticker"], r["driver"], r["type"],
                side, yes_price, size,
                r["prob_yes"], r["yes_mid"], r["yes_edge"],
                race_info["race_name"]
            )
            tracker_trades.append({
                "sport": "nascar",
                "model_name": r["type"],
                "ticker": r["ticker"],
                "title": r["title"],
                "side": "yes",
                "price_cents": int(yes_price * 100),
                "size": size,
                "model_prob": r["prob_yes"],
                "market_prob": r["yes_mid"],
                "edge": r["yes_edge"],
                "live": 0,
            })
            qf = "*" if r["has_q"] else " "
            print(f"  [PAPER] {qf} {side:10s} {r['driver']:25s} {r['type']:6s} "
                  f"{size:2d}x${yes_price:.2f}=${position_cost:.2f} "
                  f"(model={r['prob_yes']:.0%} edge={r['yes_edge']:+.0%})")
        else:
            try:
                order = kc.place_order(
                    ticker=r["ticker"],
                    side=side,
                    price=yes_price,
                    count=size,
                )
                if order:
                    tracker_trades.append({
                        "sport": "nascar",
                        "model_name": r["type"],
                        "ticker": r["ticker"],
                        "title": r["title"],
                        "side": "yes",
                        "price_cents": int(yes_price * 100),
                        "size": size,
                        "model_prob": r["prob_yes"],
                        "market_prob": r["yes_mid"],
                        "edge": r["yes_edge"],
                        "live": 1,
                    })
                    print(f"  [LIVE]  {side:10s} {r['driver']:25s} {r['type']:6s} "
                          f"{size:2d}x${yes_price:.2f}=${position_cost:.2f}")
            except Exception as e:
                print(f"  [FAIL]  {r['driver']:25s}: {e}")

        placed += 1
        cumulative_cost += position_cost

    print(f"\n  Summary: {placed} positions placed (${cumulative_cost:.2f} of ${max_exposure:.2f} exposure)")
    print(f"  Remaining: ${available - cumulative_cost:.2f}")

    # Paper trade history
    if PAPER_LOG.exists():
        log_df = pd.read_csv(PAPER_LOG)
        nascar_log = log_df[log_df["type"].isin(["top3", "top5", "top10", "top20", "race"])]
        if not nascar_log.empty:
            total = len(nascar_log)
            print(f"\n  Paper trade history: {total} total trades")
            last_race = nascar_log["race"].iloc[-1] if "race" in nascar_log.columns else ""
            print(f"  Last trade race: {last_race}")

    # Log all trades to unified tracker
    if tracker_trades:
        tracker = TradeTracker()
        tracker.log_batch(tracker_trades)
        print(f"  Logged {len(tracker_trades)} trades to tracker")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--bankroll", type=float, default=100.0)
    args = parser.parse_args()
    run_weekly_scan(bankroll=args.bankroll, paper_only=not args.live)
