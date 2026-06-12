#!/usr/bin/env python3
"""Unified Kalshi NHL bettor — covers NHL player prop markets.

Scans Kalshi for NHL player prop markets, loads the corresponding XGBoost
regressor, computes edge with distribution-appropriate probability + Wang.

Market types (Kalshi series tickers TBD — inferred from league conventions):
  KXNHLGOALS   → goals
  KXNHLASSISTS → assists
  KXNHLPOINTS  → points
  KXNHLSHOTS   → shots on goal
  KXNHLPIM     → penalty minutes
  KXNHLGOALS+ASSISTS → goals+assists

Usage:
    python -m src.scripts.kalshi_nhl_unified --scan
"""
import sys, re, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.kalshi import KalshiClient
from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
from scipy.stats import norm as _norm
import toml

MODEL_DIR = PROJECT_ROOT / "models" / "nhl"
WANG_LAMBDA = 0.20  # lighter calibration for NHL

# Market type configuration
# Series tickers are best-guess based on Kalshi conventions.
# NHL player prop markets are currently off-season; update tickers when they launch.
MARKET_TYPES = [
    {
        "name": "GOALS",
        "model_name": "GOALS",
        "series_ticker": "KXNHLGOALS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*goals?\??$",
        "desc": "goals",
        "info_only": False,
    },
    {
        "name": "ASSISTS",
        "model_name": "ASSISTS",
        "series_ticker": "KXNHLASSISTS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*assists?\??$",
        "desc": "assists",
        "info_only": False,
    },
    {
        "name": "POINTS",
        "model_name": "POINTS",
        "series_ticker": "KXNHLPOINTS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*points?\??$",
        "desc": "points",
        "info_only": False,
    },
    {
        "name": "SHOTS",
        "model_name": "SHOTS",
        "series_ticker": "KXNHLSHOTS",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*shots?\s*(?:on\s*goal)?\??$",
        "desc": "shots",
        "info_only": False,
    },
    {
        "name": "PIM",
        "model_name": "PIM",
        "series_ticker": "KXNHLPIM",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*penalty\s*minutes?\??$",
        "desc": "penalty minutes",
        "info_only": False,
    },
    {
        "name": "G+A",
        "model_name": "GOALS+ASSISTS",
        "series_ticker": "KXNHLLGOALS",  # placeholder — update when Kalshi lists
        "pattern": r"^(.+?):\s*(\d+)\+?\s*(?:goals?\s*\+\s*assists?|G\+A)\??$",
        "desc": "goals+assists",
        "info_only": False,
    },
]


def _load_regressor(model_name: str):
    if model_name is None:
        return None, None, None, BetaCalibrator()
    mn = model_name.lower()
    # Handle model name with +: GOALS+ASSISTS → goals+assists.json
    model_path = MODEL_DIR / f"{mn}.json"
    meta_path = MODEL_DIR / f"{mn}.metrics.json"
    if not model_path.exists():
        return None, None, None, BetaCalibrator()
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    try:
        with open(model_path) as f:
            mdata = json.load(f)
        feature_names = mdata.get('learner', {}).get('feature_names', [])
    except Exception:
        feature_names = []
    std = 1.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        std = meta.get("residual_std", meta.get("mae", 1.0))
    cal_path = MODEL_DIR / f"{mn}_beta_cal.json"
    beta_cal = BetaCalibrator.load(cal_path)
    return model, float(std), feature_names, beta_cal


def _match_player(title: str, latest: pd.DataFrame) -> pd.Series:
    """Match player name from Kalshi title to NHL feature data."""
    if not title or latest is None or latest.empty:
        return None
    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]
    if "player_name" in latest.columns:
        exact = latest[latest["player_name"].str.lower() == clean.lower()]
        if len(exact) >= 1:
            return exact.iloc[-1]
        lm = latest[latest["player_name"].str.lower().str.contains(last.lower(), na=False)]
        if len(lm) >= 1:
            fi = lm[lm["player_name"].str.lower().str[0] == first[0].lower()]
            if len(fi) >= 1:
                return fi.iloc[-1]
            return lm.iloc[-1]
    return None


def _p_ge_line(row, model, residual_std, line_val, feature_names, stat_name="", beta_cal=None):
    feat_dict = {}
    for c in feature_names:
        if c in row.index:
            val = row[c]
            if pd.isna(val):
                val = 0.0
            feat_dict[c] = float(val)
        else:
            feat_dict[c] = 0.0
    X_pred = pd.DataFrame([feat_dict]).fillna(0)
    mu = model.predict(X_pred)[0]
    sigma = max(residual_std, 0.3)
    p_raw = p_ge_stat(stat_name, mu, sigma, line_val)
    if beta_cal is not None and beta_cal._fitted:
        p_corrected = beta_cal(p_raw)
    else:
        z = _norm.ppf(p_raw)
        p_corrected = _norm.cdf(z - WANG_LAMBDA)
    p_corrected = min(0.75, float(p_corrected))
    return max(0.001, p_corrected), float(mu)


def load_features():
    """Load NHL data and build features using NHLDataSource + NHLFeatureEngineer."""
    from src.features.nhl import NHLFeatureEngineer
    from src.config.settings import SportConfig

    cfg_path = CONFIG_DIR / "nhl.toml"
    if cfg_path.exists():
        cfg = toml.load(cfg_path)
    else:
        cfg = {"features": {"rolling_windows": [5, 10, 20], "recency_decay": 0.001}}

    scfg = SportConfig(
        name="nhl", display_name="NHL",
        rolling_windows=cfg["features"]["rolling_windows"],
        recency_decay=cfg["features"].get("recency_decay", 0.001),
    )
    fe = NHLFeatureEngineer(scfg)

    cache_path = PROJECT_ROOT / "data" / "nhl_cache" / "game_logs_v2.parquet"
    if not cache_path.exists():
        print("  No cached NHL data. Run data pipeline first.")
        return None

    all_games = pd.read_parquet(cache_path)
    print(f"  Loaded {len(all_games)} raw rows", flush=True)

    featured = fe.build_features(all_games)
    print(f"  Feature engineering: {len(featured)} rows, {len(featured.columns)} cols", flush=True)

    # Merge player_name back
    if "player_name" in all_games.columns and "player_id" in all_games.columns and "game_date" in all_games.columns:
        merge_df = all_games[["player_id", "game_date", "player_name"]] \
            .drop_duplicates(subset=["player_id", "game_date"])
        merge_df["game_date"] = pd.to_datetime(merge_df["game_date"])
        featured["game_date"] = pd.to_datetime(featured["game_date"])
        featured = featured.merge(merge_df, on=["player_id", "game_date"], how="left")

    return featured


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--bet", action="store_true")
    args = parser.parse_args()

    client = KalshiClient()
    print(f"Balance: ${client.get_balance():.2f}\n")

    latest = load_features()
    if latest is None or latest.empty:
        print("No feature data. Run data pipeline first.")
        return

    if "game_date" in latest.columns:
        latest = latest.sort_values("game_date").groupby("player_id").last().reset_index()
    print(f"Loaded features for {len(latest)} players\n")

    all_opps = []
    model_cache = {}

    for mt in MARKET_TYPES:
        name = mt["name"]
        model_name = mt["model_name"]
        series = mt["series_ticker"]
        pattern = mt["pattern"]
        desc = mt["desc"]
        info_only = mt.get("info_only", False)

        print(f"Scanning {name} ({series})...", flush=True)
        try:
            mkts = client.list_markets(series_ticker=series, limit=500)
            if mkts is None or mkts.empty:
                print(f"  No markets (off-season?)")
                continue
        except Exception as e:
            print(f"  Cannot reach Kalshi: {e}")
            continue
        print(f"  {len(mkts)} markets", flush=True)

        if model_name not in model_cache:
            m, s, feats, cal = _load_regressor(model_name)
            if m is None and not info_only:
                print(f"  No regressor for {model_name} — skipping")
                continue
            model_cache[model_name] = (m, s, feats, cal)
        reg_model, reg_std, feature_names, beta_cal = model_cache.get(model_name, (None, None, None, None))

        count = 0
        for _, mrow in mkts.iterrows():
            try:
                ticker = mrow["ticker"]
                title = mrow.get("title", "")
                yb_v = mrow.get("yes_bid_dollars", 0)
                ya_v = mrow.get("yes_ask_dollars", 1)
                yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
                ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
                if yb <= 0 and ya >= 1.0:
                    continue
                yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

                lm = re.match(pattern, title, re.IGNORECASE)
                if not lm:
                    continue
                pname = lm.group(1).strip()
                line_val = int(lm.group(2))
                if line_val <= 0:
                    continue

                if info_only:
                    label = f"{pname} {line_val}+ {desc}"
                    all_opps.append({
                        "type": name, "ticker": ticker,
                        "side": "yes",
                        "price_cents": max(1, int(yes_mid * 100)),
                        "model_prob": 0.5,
                        "market_prob": round(yes_mid, 4),
                        "edge": 0.0,
                        "contracts": 1,
                        "player": pname,
                        "team": "",
                        "line_val": line_val,
                        "stat_desc": desc,
                        "label": label,
                    })
                    count += 1
                    continue

                row_match = _match_player(pname, latest)
                if row_match is None:
                    continue

                avg_cols = [c for c in row_match.index
                            if c.endswith("_avg_5") and isinstance(row_match[c], (int, float))]
                if avg_cols and all(pd.isna(row_match[c]) for c in avg_cols):
                    continue

                p_yes, mu = _p_ge_line(
                    row_match, reg_model, reg_std, line_val, feature_names,
                    stat_name=name, beta_cal=beta_cal,
                )
                yes_edge = p_yes - yes_mid
                no_edge = (1.0 - p_yes) - (1.0 - yes_mid)

                label = f"{pname} {line_val}+ {desc}"
                all_opps.append({
                    "type": name, "ticker": ticker,
                    "side": "yes",
                    "price_cents": max(1, int(yes_mid * 100)),
                    "model_prob": round(p_yes, 4),
                    "market_prob": round(yes_mid, 4),
                    "edge": round(yes_edge, 4),
                    "yes_edge": round(yes_edge, 4),
                    "no_edge": round(no_edge, 4),
                    "contracts": 1,
                    "player": pname,
                    "team": "",
                    "line_val": line_val,
                    "stat_desc": desc,
                    "label": label,
                })
                count += 1
            except Exception:
                pass

        label = f"  {name:4s} ({series:11s}): {count} markets matched"
        if info_only:
            label += " (info_only)"
        print(label)

    print(f"\nTotal matched opportunities: {len(all_opps)}")
    all_opps.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)

    if all_opps:
        print(f"\nTop 10:")
        print(f"  {'Type':5s} {'Player':25s} {'Bet':20s} {'Edge':>8s} {'Price':>6s}")
        print(f"  " + "-" * 66)
        for o in all_opps[:10]:
            bt = o.get("type", "?")
            player = o.get("player", "")[:24]
            bet_str = f"{o.get('line_val', 0)}+ {o.get('stat_desc', '')}" if o.get("line_val", 0) else o.get("stat_desc", "")
            edge_str = f"{o.get('edge', 0):+.0%}" if o.get("edge", 0) != 0 else "N/A"
            print(f"  {bt:5s} {player:25s} {bet_str:20s} {edge_str:>8s} {o.get('price_cents', 0):3d}c")

    if args.bet:
        # NO-side logic (mirrors kalshi_mlb_unified.py):
        # For each opportunity, compute both yes_edge and no_edge, take the
        # side with the larger absolute edge. Sort by max(|yes_edge|, |no_edge|)
        # descending so the most aggressive opportunities fill first.
        # Cap per-bet dollar risk at 5% of bankroll, max 25 contracts.
        print(f"\n--- PLACING ORDERS (max-edge side per opportunity) ---")
        bet_opps = [o for o in all_opps if not o.get("info_only", False)]
        bet_opps.sort(key=lambda o: max(abs(o.get("yes_edge", o.get("edge", 0))),
                                         abs(o.get("no_edge", 0))), reverse=True)

        # Daily loss circuit breaker
        starting_balance = client.get_balance()
        daily_loss_limit = 0.10  # 10% max daily loss
        daily_pnl = 0.0
        placed = 0
        for o in bet_opps:
            if placed >= 12:
                break
            yes_edge = o.get("yes_edge", o.get("edge", 0))
            no_edge = o.get("no_edge", 0)
            p_y = o.get("model_prob", 0.5)
            mkt_y = o.get("market_prob", 0.5)
            mkt_n = 1.0 - mkt_y

            if daily_pnl <= -starting_balance * daily_loss_limit:
                print(f"  DAILY LOSS LIMIT HIT (-${abs(daily_pnl):.2f}), stopping")
                break

            # Pick the side with the larger edge
            if no_edge > yes_edge and no_edge > 0:
                side = "no"
                direction = "BUY NO"
                edge = no_edge
                bet_mid = mkt_n
            elif yes_edge > 0:
                side = "yes"
                direction = "BUY YES"
                edge = yes_edge
                bet_mid = mkt_y
            else:
                continue

            if edge < 0.05:
                continue
            if bet_mid < 0.10 or bet_mid > 0.90:
                continue

            # Bid pricing: sit 1¢ inside the bet side's mid
            if side == "yes":
                bid = min(98, max(1, int(mkt_y * 100) + 1))
            else:
                bid = max(2, int(mkt_n * 100) - 1)

            # Cost per contract = bid cents (symmetric for YES/NO)
            cost_per = bid / 100.0
            b = client.get_balance()
            target_risk = b * 0.05
            count = int(target_risk / cost_per)
            count = min(count, 25)
            if count < 1:
                continue
            try:
                client.create_order(
                    ticker=o["ticker"], side=side,
                    yes_price=bid if side == "yes" else (100 - bid),
                    count=str(count),
                )
                daily_pnl -= cost_per * count
                print(f"  {direction:8s} {o['type']:5s} {o['player'][:25]:25s} "
                      f"{o.get('line_val', 0)}+ @ {bid}c x{count} "
                      f"(model={p_y:.0%} mkt_y={mkt_y:.0%} mkt_n={mkt_n:.0%} "
                      f"edge={edge:+.1%} risk=${cost_per * count:.2f})", flush=True)
                placed += 1
            except Exception as e:
                print(f"  FAILED {o['player']}: {e}", flush=True)
        print(f"  Placed {placed} | Balance: ${client.get_balance():.2f}")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
