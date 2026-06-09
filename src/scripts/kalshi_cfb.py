#!/usr/bin/env python3
"""
Kalshi College Football market scanner.

Scans KXNCAAFGAME (game winner), KXNCAAFTOTAL (total points), and
KXNCAAFSPREAD (spread) markets, loads trained CFB models, builds
features from CFB data, and computes edges.

Usage:
    python -m src.scripts.kalshi_cfb              # dry-run scan
    python -m src.scripts.kalshi_cfb --bet         # place qualifying orders
"""
import json
import re
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

from src.config.settings import Settings
from src.data.kalshi import KalshiClient
from src.data.cfb import CFBDataSource
from src.features.cfb import CFBFeatureEngineer, FEATURE_COLS, CFB_STATS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "cfb"

# ── Team name normalisation (Kalshi title → CFBD API name) ──────────────
TEAM_NAME_FIXES = {
    "St.": "State",
    "Ala": "Alabama",
    "Ariz": "Arizona",
    "Ark": "Arkansas",
    "Boise St.": "Boise State",
    "Cal": "California",
    "Cent. Florida": "UCF",
    "Colo": "Colorado",
    "Fla": "Florida",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Ga. Southern": "Georgia Southern",
    "Ga. Tech": "Georgia Tech",
    "Geo": "Georgia",
    "Ill": "Illinois",
    "Ind": "Indiana",
    "Iowa St.": "Iowa State",
    "Kansas St.": "Kansas State",
    "Michigan St.": "Michigan State",
    "Miami (FL)": "Miami",
    "Minn": "Minnesota",
    "Miss": "Ole Miss",
    "Miss St.": "Mississippi State",
    "Ohio St.": "Ohio State",
    "Oklahoma St.": "Oklahoma State",
    "Oregon St.": "Oregon State",
    "Penn St.": "Penn State",
    "San Jose St.": "San Jose State",
    "South Fla.": "South Florida",
    "UNC": "North Carolina",
    "Utah St.": "Utah State",
    "Washington St.": "Washington State",
}

TITLE_PATTERN = re.compile(
    r"^Will\s+(.+?)\s+win\s+the\s+(.+?)\s+vs\s+(.+?)\s+college\s+football\s+game\??$",
    re.IGNORECASE,
)

# ── Cached data & models ────────────────────────────────────────────────
_cfb_data = None  # raw fetched DataFrame
_cfb_features = None  # featured DataFrame (per-game per-team rows)
_team_profiles = None  # dict: team -> latest feature row (offensive profile)
_team_def_profiles = None  # dict: team -> defensive stats dict
_loaded_models = None  # cached loaded models


def _normalise_team(name: str) -> str:
    """Normalise a Kalshi title team name to CFBD API team name."""
    name = name.strip()
    # Direct lookup
    if name in TEAM_NAME_FIXES:
        return TEAM_NAME_FIXES[name]
    # Try the fix map values (already correct)
    for canon in TEAM_NAME_FIXES.values():
        if canon == name:
            return name
    # Try partial match
    for k, v in TEAM_NAME_FIXES.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            return v
    return name


def _load_models() -> dict:
    """Load all trained CFB models (cached). Returns dict of target -> (model, meta)."""
    global _loaded_models
    if _loaded_models is not None:
        return _loaded_models

    models = {}
    for target in ("win", "spread_margin", "total_points"):
        model_path = MODEL_DIR / f"{target}.json"
        meta_path = MODEL_DIR / f"{target}.meta.json"
        if not model_path.exists() or not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if target == "win":
                m = XGBClassifier()
            else:
                m = XGBRegressor()
            m.load_model(str(model_path))
            models[target] = (m, meta)
        except Exception as e:
            print(f"  Failed to load {target} model: {e}")

    _loaded_models = models
    return models


def _fetch_and_featurize() -> tuple:
    """Fetch CFB data and build features once. Returns (raw_df, featured_df)."""
    global _cfb_data, _cfb_features
    if _cfb_data is not None and _cfb_features is not None:
        return _cfb_data, _cfb_features

    cfg = Settings().load_sport_config("cfb")
    if cfg is None:
        print("  CFB config not found")
        return pd.DataFrame(), pd.DataFrame()

    ds = CFBDataSource()
    fe = CFBFeatureEngineer(cfg)

    # Fetch current season and previous season for rolling stats
    years = [str(datetime.now().year - i) for i in range(cfg.season_lookback)]
    raw = ds.fetch_player_game_logs(years)
    if raw.empty:
        print("  No CFB data — check CFBD_API_KEY")
        return pd.DataFrame(), pd.DataFrame()

    _cfb_data = raw

    # Build features
    featured = fe.build_features(raw)
    _cfb_features = featured
    return raw, featured


def _build_team_profiles(featured: pd.DataFrame):
    """Build per-team offensive and defensive profiles from featured data.

    Returns (team_profiles, team_def_profiles) where:
      team_profiles: dict[team_name -> latest feature row as Series]
      team_def_profiles: dict[team_name -> defensive rolling stats dict]
    """
    global _team_profiles, _team_def_profiles
    if _team_profiles is not None and _team_def_profiles is not None:
        return _team_profiles, _team_def_profiles

    if featured.empty:
        return {}, {}

    # Sort by game_date so latest is last
    sorted_df = featured.sort_values(["team", "game_date"])

    # ── Team offensive profiles: latest feature row per team ──
    profiles = {}
    for team, grp in sorted_df.groupby("team"):
        latest = grp.iloc[-1]
        # Convert to dict, dropping NaNs
        row = {}
        for col, val in latest.items():
            if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                row[col] = float(val)
        profiles[team] = row

    # ── Team defensive profiles: rolling averages of what the team allows ──
    # From raw data, compute defensive stats for each team
    if _cfb_data is not None:
        raw = _cfb_data
        def_rows = []
        for team, grp in raw.sort_values(["team", "game_date"]).groupby("team"):
            # Build what this team's defense allows: points_against, total_yards (opponent's yards), etc.
            for _, game in grp.iterrows():
                def_rows.append({
                    "team": team,
                    "game_date": game.get("game_date"),
                    "pts_allowed_def": game.get("points_against", 0),
                    "yds_allowed_def": game.get("total_yards", 0),
                    "pass_yds_allowed_def": game.get("passing_yards", 0),
                    "rush_yds_allowed_def": game.get("rushing_yards", 0),
                    "fd_allowed_def": game.get("first_downs", 0),
                    "to_forced_def": game.get("turnovers", 0),
                })

        if not def_rows:
            return profiles, {}

        def_df = pd.DataFrame(def_rows)
        if "game_date" in def_df.columns:
            def_df["game_date"] = pd.to_datetime(def_df["game_date"], errors="coerce")
        def_df = def_df.sort_values(["team", "game_date"])
        windows = [4, 8, 12]

        def_profiles = {}
        for team, grp in def_df.groupby("team"):
            latest = grp.iloc[-1]
            profile = {}
            for col in ["pts_allowed_def", "yds_allowed_def", "pass_yds_allowed_def",
                        "rush_yds_allowed_def", "fd_allowed_def", "to_forced_def"]:
                if col not in grp.columns:
                    continue
                vals = grp[col].shift(1).dropna()
                for w in windows:
                    if len(vals) >= 2:
                        profile[f"{col}_avg_{w}"] = float(vals.tail(w).mean())
                    else:
                        profile[f"{col}_avg_{w}"] = 0.0
            if profile:
                def_profiles[team] = profile

        _team_def_profiles = def_profiles
    else:
        _team_def_profiles = {}

    _team_profiles = profiles
    return _team_profiles, _team_def_profiles


def _build_matchup_features(
    team_name: str,
    opponent_name: str,
    team_profiles: dict,
    def_profiles: dict,
) -> np.ndarray:
    """Build a feature vector for 'team' vs 'opponent' as the model expects.

    Uses team's offensive profile + opponent's defensive profile.
    Returns a numpy array matching the model's feature order.
    """
    # Find the trained win classifier model to get feature list
    models = _load_models()
    if "win" not in models:
        return None
    _, meta = models["win"]
    features = meta.get("features", [])

    team_row = team_profiles.get(team_name, {})
    opp_def = def_profiles.get(opponent_name, {})

    # If opponent has no defensive profile, fall back to their offensive/defensive
    # stats from the main profile (points_against averages)
    if not opp_def:
        opp_profile = team_profiles.get(opponent_name, {})
        opp_def = {
            f"pts_allowed_def_avg_{w}": opp_profile.get(f"points_against_avg_{w}", 0.0)
            for w in [4, 8, 12]
        }
        for col_root, col_prefix in [
            ("total_yards", "yds_allowed_def"),
            ("passing_yards", "pass_yds_allowed_def"),
            ("rushing_yards", "rush_yds_allowed_def"),
        ]:
            for w in [4, 8, 12]:
                opp_def[f"{col_prefix}_avg_{w}"] = opp_profile.get(f"{col_root}_avg_{w}", 0.0)

    # Start with team's own feature values
    vec = {}
    # Build def_map for differential lookup
    def_map = {
        "total_yards": "yds_allowed_def",
        "passing_yards": "pass_yds_allowed_def",
        "rushing_yards": "rush_yds_allowed_def",
        "turnovers": "to_forced_def",
    }
    # Pre-extract differential columns and their w values
    diff_cols = {}  # col_name -> (inner_col, w)
    for c in features:
        if c.startswith("off_def_") and "_diff_" in c:
            # c = "off_def_total_yards_diff_4"
            # Split on "_diff_" to get ["off_def_{inner}", "{w}"]
            parts = c.split("_diff_")
            if len(parts) == 2:
                inner = parts[0][len("off_def_"):]  # "total_yards"
                try:
                    w = int(parts[1])
                    diff_cols[c] = (inner, w)
                except ValueError:
                    pass

    for c in features:
        # Skip differential features — handled below
        if c in diff_cols:
            continue
        # Opponent defense features come from opp_def
        if c.startswith("opp_") and "_def_" in c:
            # e.g., opp_pts_allowed_def_avg_4 → strip opp_ → pts_allowed_def_avg_4
            key = c.replace("opp_", "", 1)
            vec[c] = opp_def.get(key, 0.0)
        elif c in ("win_streak", "win_pct_4", "win_pct_8", "days_rest", "home",
                   "spread_line", "total_line"):
            # Game context features
            if c == "home":
                vec[c] = 1.0  # Assume the market team is "home"
            elif c == "spread_line":
                vec[c] = team_row.get(c, 0.0)
            elif c == "total_line":
                vec[c] = team_row.get(c, 0.0)
            else:
                vec[c] = team_row.get(c, 0.0)
        else:
            # Team offensive feature
            vec[c] = team_row.get(c, 0.0)

    # Now compute differential features using extracted inner/w
    for c, (inner, w) in diff_cols.items():
        # Team offensive avg
        off_avg = f"{inner}_avg_{w}"
        team_off = team_row.get(off_avg, 0.0)

        # Opponent defensive avg
        def_key = def_map.get(inner)
        if def_key:
            opp_val = opp_def.get(f"{def_key}_avg_{w}", 0.0)
            vec[c] = team_off - opp_val
        else:
            vec[c] = team_off

    # Build final feature vector in model feature order
    result = np.array([vec.get(c, 0.0) for c in features], dtype=np.float32)
    return result


def _predict_win(team_name: str, opponent_name: str) -> tuple:
    """Predict win probability for team vs opponent.
    Returns (p_team_wins, p_opponent_wins, raw_model_prob).
    """
    team_profiles, def_profiles = _build_team_profiles(_cfb_features)
    models = _load_models()
    if "win" not in models:
        return 0.5, 0.5, 0.5

    model, _ = models["win"]

    # Feature vector for team vs opponent
    x_team = _build_matchup_features(team_name, opponent_name, team_profiles, def_profiles)
    x_opp = _build_matchup_features(opponent_name, team_name, team_profiles, def_profiles)

    if x_team is None or x_opp is None:
        return 0.5, 0.5, 0.5

    try:
        p_team = float(model.predict_proba(x_team.reshape(1, -1))[0, 1])
        p_opp = float(model.predict_proba(x_opp.reshape(1, -1))[0, 1])
    except Exception:
        return 0.5, 0.5, 0.5

    # Normalise: if model was trained on balanced classes, these may not sum to 1
    total = p_team + p_opp
    if total > 0:
        p_team_norm = p_team / total
        p_opp_norm = p_opp / total
    else:
        p_team_norm = p_opp_norm = 0.5

    return p_team_norm, p_opp_norm, p_team


def parse_game_market(market) -> dict:
    """Parse a KXNCAAFGAME market row. Returns parsed info dict or None."""
    ticker = market.get("ticker", "")
    title = market.get("title", "")

    m = TITLE_PATTERN.match(title.strip())
    if not m:
        return None

    market_team = m.group(1).strip()
    team_a = m.group(2).strip()
    team_b = m.group(3).strip()

    # Determine opponent
    opponent = team_b if market_team == team_a else team_a

    return {
        "market_team": market_team,
        "team_a": team_a,
        "team_b": team_b,
        "opponent": opponent,
        "ticker": ticker,
        "title": title,
    }


def get_cfb_bets(kc=None, min_edge=0.05) -> list:
    """Return structured list of qualifying CFB bets for morning_scan integration.

    Each bet dict has the same schema as other morning_scan bet dicts:
      type, ticker, side, price_cents, model_prob, market_prob, edge,
      contracts, player, team, line_val, stat_desc, label
    """
    kc = kc or KalshiClient()
    models = _load_models()
    if "win" not in models:
        print("  CFB: win model not found — run train_cfb_models.py first")
        return []

    # Fetch data
    raw, featured = _fetch_and_featurize()
    if featured.empty:
        print("  CFB: no data available")
        return []

    # Build profiles (cached internally)
    _build_team_profiles(featured)

    # Get game winner markets
    mkts = kc.list_markets(series_ticker="KXNCAAFGAME", limit=500)
    if mkts is None or mkts.empty:
        print("  CFB: no game winner markets")
        return []

    balance = kc.get_balance()
    results = []

    for _, m in mkts.iterrows():
        try:
            yb = float(m.get("yes_bid_dollars", 0) or 0)
            ya = float(m.get("yes_ask_dollars", 1) or 1)
            if yb <= 0 and ya >= 1.0:
                continue
            yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))
            if yes_mid < 0.02 or yes_mid > 0.98:
                continue

            parsed = parse_game_market(m)
            if parsed is None:
                continue

            market_team = parsed["market_team"]
            opponent = parsed["opponent"]

            # Normalise team names
            team_norm = _normalise_team(market_team)
            opp_norm = _normalise_team(opponent)

            # Check if teams exist in profiles
            team_profiles, _ = _build_team_profiles(featured)
            if team_norm not in team_profiles or opp_norm not in team_profiles:
                print(f"  Skipping {market_team} vs {opponent}: teams not in CFB data")
                continue

            p_team, p_opp, raw_p = _predict_win(team_norm, opp_norm)
            edge = p_team - yes_mid

            if edge >= min_edge and 0.10 <= yes_mid <= 0.80:
                results.append({
                    "type": "CFB",
                    "ticker": parsed["ticker"],
                    "side": "yes",
                    "price_cents": max(1, int(yes_mid * 100)),
                    "model_prob": round(p_team, 4),
                    "market_prob": round(yes_mid, 4),
                    "edge": round(edge, 4),
                    "contracts": 1,
                    "player": market_team,
                    "team": opponent,
                    "line_val": 0,
                    "stat_desc": "win",
                    "label": f"CFB-{market_team} vs {opponent}",
                })
                print(f"  {market_team:25s} vs {opponent:25s}  "
                      f"model={p_team:.0%} mkt={yes_mid:.0%} edge={edge:+.0%}")

        except Exception as e:
            print(f"  Error processing {m.get('ticker','?')}: {e}")
            continue

    return results


def scan():
    """Run the full CFB market scan and display results."""
    kc = KalshiClient()
    balance = kc.get_balance()

    # Load models
    models = _load_models()
    if not models:
        print("  No CFB models found. Run: python -m src.scripts.train_cfb_models")
        return

    available = ", ".join(sorted(models.keys()))
    print(f"  Models loaded: {available}")

    # Fetch data
    _, featured = _fetch_and_featurize()
    if featured.empty:
        print("  No CFB data. Check CFBD_API_KEY in .env")
        return

    team_profiles, def_profiles = _build_team_profiles(featured)
    print(f"  Team profiles: {len(team_profiles)} teams")

    # Get game markets
    game_mkts = kc.list_markets(series_ticker="KXNCAAFGAME", limit=500)
    if game_mkts is None or game_mkts.empty:
        print("  No KXNCAAFGAME markets found")
        return

    print(f"\n  {'='*70}")
    print(f"  KXNCAAFGAME — Game Winner Markets ({len(game_mkts)} total)")
    print(f"  {'='*70}")

    # Process each market individually using title parsing (like get_cfb_bets)
    # then group by matchup for display
    game_data = {}  # key -> list of {team, opp, p_win, mid, ticker, yb, ya}

    for _, m in game_mkts.iterrows():
        try:
            yb = float(m.get("yes_bid_dollars", 0) or 0)
            ya = float(m.get("yes_ask_dollars", 1) or 1)
            if yb <= 0 and ya >= 1.0:
                continue
            yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))
            if yes_mid < 0.02 or yes_mid > 0.98:
                continue

            parsed = parse_game_market(m)
            if parsed is None:
                continue

            market_team = parsed["market_team"]
            opponent = parsed["opponent"]
            team_norm = _normalise_team(market_team)
            opp_norm = _normalise_team(opponent)

            if team_norm not in team_profiles or opp_norm not in team_profiles:
                continue

            p_win, _, _ = _predict_win(team_norm, opp_norm)

            key = f"{parsed['team_a']}_vs_{parsed['team_b']}"
            if key not in game_data:
                game_data[key] = {
                    "team_a": parsed["team_a"],
                    "team_b": parsed["team_b"],
                    "team_a_norm": team_norm if market_team == parsed["team_a"] else opp_norm,
                    "team_b_norm": opp_norm if market_team == parsed["team_a"] else team_norm,
                    "entries": [],
                }
            game_data[key]["entries"].append({
                "team": market_team,
                "norm": team_norm,
                "p_win": float(p_win),
                "mid": yes_mid,
                "yb": yb,
                "ya": ya,
                "ticker": parsed["ticker"],
            })
        except Exception:
            continue

    print(f"  Matchups: {len(game_data)}")

    qualifying = []
    for key, g in sorted(game_data.items()):
        entries = g["entries"]
        team_a, team_b = g["team_a"], g["team_b"]

        # Find price for each team
        p_a = p_b = 0.5
        mkt_a = mkt_b = 0
        ticker_a = ticker_b = ""

        for e in entries:
            if e["norm"] == g["team_a_norm"]:
                p_a = e["p_win"]
                mkt_a = e["mid"]
                ticker_a = e["ticker"]
            else:
                p_b = e["p_win"]
                mkt_b = e["mid"]
                ticker_b = e["ticker"]

        # If only one market found, derive the other side
        if mkt_b == 0 and mkt_a > 0:
            mkt_b = 1.0 - mkt_a
            p_b = 1.0 - p_a

        edge_a = p_a - mkt_a if mkt_a > 0 else 0
        edge_b = p_b - mkt_b if mkt_b > 0 else 0
        best_edge = max(edge_a, edge_b)

        if edge_a >= edge_b and edge_a >= 0.05 and 0.10 <= mkt_a <= 0.80:
            qualifying.append({
                "match": f"{team_a} vs {team_b}",
                "team": team_a, "opponent": team_b,
                "model_p": float(p_a), "mkt_p": float(mkt_a),
                "edge": float(edge_a), "ticker": ticker_a,
            })
        elif edge_b > edge_a and edge_b >= 0.05 and 0.10 <= mkt_b <= 0.80:
            qualifying.append({
                "match": f"{team_a} vs {team_b}",
                "team": team_b, "opponent": team_a,
                "model_p": float(p_b), "mkt_p": float(mkt_b),
                "edge": float(edge_b), "ticker": ticker_b,
            })

        print(f"  {team_a:25s} vs {team_b:25s}  "
              f"P({team_a})={p_a:.0%}  mkt={mkt_a:.0%}  "
              f"P({team_b})={p_b:.0%}  mkt={mkt_b:.0%}  "
              f"edge={best_edge:+.0%}")

    if qualifying:
        qualifying.sort(key=lambda x: -x["edge"])
        print(f"\n  {'='*70}")
        print(f"  QUALIFYING BETS (edge ≥ 5%)")
        print(f"  {'='*70}")
        for q in qualifying[:10]:
            print(f"\n  {q['match']:40s}")
            print(f"  Pick: {q['team']:30s}  Model: {q['model_p']:.0%}")
            print(f"  Market: {q['mkt_p']:.0%}  Edge: {q['edge']:+.0%}")
            if q['ticker']:
                print(f"  Ticker: {q['ticker']}")
    else:
        print(f"\n  No qualifying bets found (min_edge=5%)")

    # Also check total/spread markets (currently likely 0)
    for series_ticker, label in [("KXNCAAFTOTAL", "Total Points"),
                                   ("KXNCAAFSPREAD", "Spread")]:
        mkts = kc.list_markets(series_ticker=series_ticker, limit=50)
        if mkts is not None and not mkts.empty:
            print(f"\n  {label}: {len(mkts)} markets")

    print(f"\n  Balance: ${balance:.2f}")# ── Standalone entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    auto = "--bet" in sys.argv
    bets = get_cfb_bets(min_edge=0.05)
    if bets:
        print(f"\n  Found {len(bets)} qualifying CFB bets")
        if auto:
            print("  --bet mode: not yet implemented for standalone CFB scanner")
            print("  Use morning_scan --bet to place orders")
    else:
        scan()
