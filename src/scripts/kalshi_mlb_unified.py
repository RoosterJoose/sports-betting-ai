#!/usr/bin/env python3
"""Unified Kalshi MLB bettor — covers all available MLB market types.

Scans Kalshi for MLB markets across every stat type Kalshi offers,
loads the corresponding regressor, computes edge with Wang calibration,
and places limit orders where edge >= 7%.

Market types:
  KXMLBKS → strikeouts          (SO,  pitcher)
  KXMLBHR → home runs           (HR,  hitter)
  KXMLBTB → total bases         (TB,  hitter)
  KXMLBHRR → H+R+RBI            (HRR, hitter)
  KXMLBIP → pitching outs       (IP,  pitcher) — no active markets yet
  KXMLBER → earned runs allowed  (ER,  pitcher) — no active markets yet
  KXMLBH → hits allowed          (H,   pitcher) — no active markets yet
  KXMLBBB → walks allowed        (BB,  pitcher) — no active markets yet
  KXMLBR → runs                  (R,   hitter)  — no active markets yet
  KXMLBRBI → RBIs                (RBI, hitter)  — no active markets yet
  KXMLBSB → stolen bases         (SB,  hitter)  — no active markets yet

Usage:
    python -m src.scripts.kalshi_mlb_unified --scan
    python -m src.scripts.kalshi_mlb_unified --bet
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
from src.models.calibrator import EmpiricalCalibrator, BetaCalibrator
from src.models.distributions import p_ge_stat
import toml, lightgbm as lgb
from scipy.stats import norm as _norm

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"

WANG_LAMBDA = 0.30  # Fallback when empirical calibration not available

# Per Phase 4: Kalshi fee-zone awareness.
# Per NotebookLM: Kalshi taker fees in 40-60c "dead zone" consume up to
# 3.5% of capital at risk. Boost min_edge requirement in that range.
FEE_ZONE_LOW = 0.40
FEE_ZONE_HIGH = 0.60
FEE_ZONE_MIN_EDGE = 0.075  # 7.5% edge required in 40-60c range (vs 5% elsewhere)

# Liquidity-aware price gate (relaxed 2026-06-11)
# Core range 0.15 < mkt < 0.75 — the original gate, where all markets are
# assumed to have meaningful liquidity. This covers ~60% of Kalshi markets.
# Extended range 0.05 <= mkt <= 0.95 — previously blocked, now allowed
# only if the side we'd lift has >= MIN_LIQUIDITY_CONTRACTS contracts of
# resting depth. Per the 2026-06-11 diagnostic:
#   - 0-10c YES markets have median ask_size=500 (deep), 30% have bid_size>=10
#   - 90-100c NO markets have median ask_size=0 (thin), only 12% have ask_size>=10
# So the extended range unlocks the deep 0-10c YES asks while still filtering
# out the thin 90-100c NO asks.
MIN_LIQUIDITY_CONTRACTS = 10
EXTENDED_GATE_LOW = 0.05
EXTENDED_GATE_HIGH = 0.95

# Module-level calibration singletons
_calibrator: EmpiricalCalibrator | None = None
_beta_calibrators: dict[str, BetaCalibrator] = {}
_isotonic_calibrators: dict[str, "IsotonicCalibrator"] = {}

# Stats that benefit from Isotonic over BetaCal (low-count, tail-sensitive).
# Per NotebookLM: low-count props have severe tail overconfidence that
# Isotonic Regression handles better than parametric BetaCal.
ISOTONIC_PREFERRED: set[str] = {"ip", "r", "rbi", "sb", "hr", "blk", "stl"}


def _get_cal():
    # NOTE (2026-06-11): Empirical cal files were moved to
    # models/mlb/calibration/.empirical_backup/ because they were crushing
    # model_prob to 2-17% on markets priced at 56-99%. Beta cals are now
    # the active path. The EmpiricalCalibrator returns None if no
    # *_empirical.json files exist in CALIB_DIR, so this function is a
    # no-op until the empirical cals are rebuilt from fresh 2026 data.
    global _calibrator
    if _calibrator is None and CALIB_DIR.exists():
        _calibrator = EmpiricalCalibrator(CALIB_DIR)
    return _calibrator


def _get_beta_cal(stat_name: str) -> BetaCalibrator | None:
    """Load per-stat BetaCalibrator from MODEL_DIR (cached).

    Per-stat BetaCal files live at ``models/mlb/{stat}_beta_cal.json``
    (root, NOT in the ``calibration/`` subdir — that's reserved for the
    empirical calibrator which IS read via CALIB_DIR above). This was
    a pre-existing path mismatch where the scanner read from
    ``calibration/`` but the writers (refit_mlb_beta_cal_live.py and
    fit_pa_k_calibration.py) write to MODEL_DIR root, so all beta cals
    were silently ignored and the scanner fell through to Wang.
    """
    key = stat_name.lower()
    if key in _beta_calibrators:
        bc = _beta_calibrators[key]
        return bc if bc._fitted else None
    path = MODEL_DIR / f"{key}_beta_cal.json"
    bc = BetaCalibrator.load(path)
    _beta_calibrators[key] = bc
    return bc if bc._fitted else None


def _get_isotonic_cal(stat_name: str):
    """Load per-stat IsotonicCalibrator from MODEL_DIR (cached).

    Same path fix as _get_beta_cal — reads from MODEL_DIR root where
    the isotonic cal files actually live. The ``calibration/`` subdir
    is only used for the empirical calibrator.
    """
    from src.models.calibrator import IsotonicCalibrator
    key = stat_name.lower()
    if key in _isotonic_calibrators:
        ic = _isotonic_calibrators[key]
        return ic if ic._fitted else None
    path = MODEL_DIR / f"{key}_isotonic_cal.json"
    ic = IsotonicCalibrator.load(path)
    _isotonic_calibrators[key] = ic
    return ic if ic._fitted else None

# registry: (model_name, raw_col, series_ticker, position_filter, title_pattern)
# pattern groups: player_name, line_value
# info_only: True means scan but don't bet (model lacks signal for that market type)
MARKET_TYPES = [
    {
        "name": "KS",
        "model_name": "SO",
        "series_ticker": "KXMLBKS",
        "position": "pitcher",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*strikeouts?\??$",
        "desc": "strikeouts",
        "info_only": False,
    },
    {
        "name": "HR",
        "model_name": "HR",
        "series_ticker": "KXMLBHR",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*home\s*runs?\??$",
        "desc": "home runs",
        "info_only": False,  # Phase 1: 3/4 backtest — sharpest market, kept live with caution
    },
    {
        "name": "TB",
        "model_name": "TB",
        "series_ticker": "KXMLBTB",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*total\s*bases?\??$",
        "desc": "total bases",
        "info_only": False,  # Phase 1: 5/5 (100%) backtest
    },
    {
        "name": "HRR",
        "model_name": "H_R_RBI",
        "series_ticker": "KXMLBHRR",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*hits\s*\+\s*runs\s*\+\s*RBIs?\??$",
        "desc": "hits+runs+RBIs",
        "info_only": False,  # Phase 1: 5/5 (100%) backtest
    },
    # New market types (no active markets as of June 2026 — patterns inferred)
    {
        "name": "IP",
        "model_name": "IP",
        "series_ticker": "KXMLBIP",
        "position": "pitcher",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*outs?\??$",
        "desc": "pitching outs",
        "info_only": False,  # Phase 1: 5/5 (100%) backtest — uses Isotonic cal
    },
    {
        "name": "ER",
        "model_name": "ER",
        "series_ticker": "KXMLBER",
        "position": "pitcher",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*earned\s*runs?\??$",
        "desc": "earned runs allowed",
        "info_only": False,  # Phase 1: 4/4 (100%) backtest
    },
    {
        "name": "H",
        "model_name": "H",
        "series_ticker": "KXMLBH",
        "position": "pitcher",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*hits?\??$",
        "desc": "hits allowed",
        "info_only": False,  # Phase 1: 5/5 (100%) backtest
    },
    {
        "name": "BB",
        "model_name": "BB",
        "series_ticker": "KXMLBBB",
        "position": "pitcher",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*walks?\??$",
        "desc": "walks allowed",
        "info_only": False,  # Phase 1: 4/4 (100%) backtest
    },
    {
        "name": "RBI",
        "model_name": "RBI",
        "series_ticker": "KXMLBRBI",
        "position": "hitter",
        "pattern": r"^(.+?):\s*(\d+)\+?\s*RBIs?\??$",
        "desc": "RBIs",
        "info_only": False,  # Phase 1: 4/4 (100%) backtest — uses Isotonic cal
    },
    # R dropped: 2/4 (50%) backtest — fails naive baseline
    # SB dropped: 1/4 (25%) backtest — worst, fails naive baseline
]

def _load_regressor(model_name):
    # Try LightGBM first, fall back to XGBoost
    mn = model_name.lower()
    lgb_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"{'lgb' if lgb_path.exists() else 'reg'}_{mn}.meta.json"
    if lgb_path.exists():
        model = lgb.Booster(model_file=str(lgb_path))
    else:
        xgb_path = MODEL_DIR / f"reg_{mn}.json"
        if not xgb_path.exists() or not meta_path.exists():
            return None, None
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(xgb_path))
    if not meta_path.exists():
        return model, 1.0
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta.get("residual_std", 1.0)

def _match_player(title, lc, position_filter=None):
    """Match player by name, optionally filtering by position (hitter/pitcher).
    
    position_filter=None: no filter
    position_filter="hitter": exclude pitchers (position != "P")
    position_filter="pitcher": only pitchers (position == "P")
    """
    if not title or lc is None or lc.empty:
        return None
    
    df = lc
    if position_filter == "hitter":
        df = lc[lc.get("position", "") != "P"]
    elif position_filter == "pitcher":
        df = lc[lc.get("position", "") == "P"]
    
    if df.empty:
        return None
    
    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None
    first, last = parts[0], parts[-1]
    exact = df[df["player_name"].str.lower() == clean.lower()]
    if len(exact) == 1:
        return exact.iloc[0]
    lm = df[df["player_name"].str.lower().str.endswith(last.lower(), na=False)]
    if len(lm) == 1:
        return lm.iloc[0]
    la = df[df["player_name"].str.lower().str.contains(last.lower(), na=False)]
    if len(la) >= 1:
        fi = la[la["player_name"].str.lower().str[0] == first[0].lower()]
        return fi.iloc[0] if len(fi) >= 1 else la.iloc[0]
    return None

def _p_ge_line(row, model, residual_std, line_val, stat_name=None):
    """Predict P(stat >= line_val) using empirical calibration → Beta Calibration → Wang.

    Calibration cascade:
      1. Multi-bin empirical calibration (built from test set) — best when available
      2. Beta Calibration (per-stat, fitted from test set residuals)
      3. Wang Transform (global fallback, λ=0.30)
    """
    # Handle both LightGBM (Booster) and XGBoost (XGBRegressor)
    if hasattr(model, 'feature_name'):
        feats = model.feature_name()
    elif hasattr(model, 'feature_names_in_'):
        feats = model.feature_names_in_
    else:
        feats = [c for c in row.index if isinstance(row[c], (int, float))]
    mu = model.predict(pd.DataFrame([{c: row.to_dict().get(c, 0) for c in feats}]).fillna(0))[0]
    sigma = max(residual_std, 0.3)
    # Use distribution-appropriate mapping (NB for SO/TB/H, Poisson for HR/SB)
    p_raw = float(p_ge_stat(stat_name or "SO", mu, sigma, line_val))

    # ── Step 1: Multi-bin empirical calibration ────────────────────────────
    cal = _get_cal()
    if cal is not None and stat_name is not None:
        stat_key = stat_name.lower()
        line_key = str(line_val)
        bins = cal.calibration.get(stat_key, {}).get(line_key, {}).get("bins", [])
        if len(bins) > 1:  # multi-bin = properly calibrated
            p_cal = cal.calibrate(stat_key, line_val, p_raw)
            p_cal = min(0.999, max(0.001, float(p_cal)))
            return p_cal, float(mu)
        # Single bin = old format (flat per-line rate) — fall through

    # ── Step 2: Isotonic (preferred for low-count) or Beta Calibration ────
    if stat_name is not None:
        if stat_name.lower() in ISOTONIC_PREFERRED:
            iso_cal = _get_isotonic_cal(stat_name)
            if iso_cal is not None and iso_cal._fitted:
                p_cor = iso_cal(p_raw)
                p_cor = min(0.999, max(0.001, float(p_cor)))
                return p_cor, float(mu)
        beta_cal = _get_beta_cal(stat_name)
        if beta_cal is not None and beta_cal._fitted:
            p_cor = beta_cal(p_raw)
            p_cor = min(0.999, max(0.001, float(p_cor)))
            return p_cor, float(mu)

    # ── Step 3: Wang Transform (fallback) ──────────────────────────────────
    z = _norm.ppf(p_raw)
    p_corrected = _norm.cdf(z - WANG_LAMBDA)
    p_corrected = min(0.999, max(0.001, float(p_corrected)))
    return p_corrected, float(mu)

# Module-level cache for recency check data
_recency_df: pd.DataFrame | None = None

def _get_recency_df():
    """Load and cache the game logs parquet for recency checks.

    Without caching, this function reads a 2.5MB parquet file every
    time it's called (~400 times per scan), which dominates runtime.
    """
    global _recency_df
    if _recency_df is not None:
        return _recency_df
    cache_path = PROJECT_ROOT / "data" / "cache" / "mlb" / "game_logs_2026_2025_2024.parquet"
    if cache_path.exists():
        _recency_df = pd.read_parquet(cache_path)
    return _recency_df


def _recency_check(player_name: str, line_val: int, stat_col: str = "so") -> tuple[float, float, bool]:
    """Compare model prediction to actual rate for a player.

    Uses 2026 data first (prefer ≥3 games for stability), falls back to
    2025, then 2024 if 2026 sample is too small. This prevents the live
    scanner from overreacting to 1-game extremes (0% or 100%) while still
    providing ground truth rates for backtesting.

    stat_col: which stat to check ("so", "hr", "tb", "h_r_rbi", "ip", "h", "bb",
               "er", "r", "rbi", "sb").
    Returns (actual_rate, -1, True) where actual_rate=-1 means insufficient data.
    """
    try:
        df = _get_recency_df()
        if df is None:
            return -1, -1, True

        # Pitcher stats: filter by position=="P" and gs==1 (starts only)
        # Hitter stats: filter by position!="P"
        pitcher_stats = {"so", "ip", "h", "bb", "er"}
        is_pitcher_stat = stat_col in pitcher_stats
        
        if is_pitcher_stat:
            combined = (df["gs"] == 1) & (df["position"] == "P")
        else:
            combined = df["position"] != "P"

        # Tiered fallback: 2026 ≥3 → 2025 → 2024 → give up
        player_name_mask = df["player_name"].str.contains(player_name, case=False, na=False)
        for season in ["2026", "2025", "2024"]:
            player_games = df[player_name_mask & (df["season"] == season) & combined]
            if len(player_games) >= 3:
                break
        if len(player_games) < 3:
            return -1, -1, True

        # Generic stat rate computation
        if stat_col == "so":
            actual_rate = (player_games["so"] >= line_val).mean()
        elif stat_col == "hr":
            actual_rate = (player_games["hr"] >= line_val).mean()
        elif stat_col == "tb":
            tb = player_games["1b"] + 2 * player_games["2b"] + 3 * player_games["3b"] + 4 * player_games["hr"]
            actual_rate = (tb >= line_val).mean()
        elif stat_col == "h_r_rbi":
            hrr = player_games["h"] + player_games["r"] + player_games["rbi"]
            actual_rate = (hrr >= line_val).mean()
        elif stat_col in ("ip", "h", "bb", "er", "r", "rbi", "sb"):
            # Generic: use the column directly
            if stat_col in player_games.columns:
                actual_rate = (player_games[stat_col] >= line_val).mean()
            else:
                actual_rate = -1
        else:
            actual_rate = -1
        return float(actual_rate), -1, True
    except Exception:
        return -1, -1, True


def _game_is_pregame(ticker):
    """Check if the market's game is pre-game (not in progress/final)."""
    import json, requests
    try:
        map_file = Path("/tmp/mlb_game_status.json")
        if map_file.exists():
            with open(map_file) as f:
                status_map = json.load(f)
        else:
            return True

        # Known Kalshi MLB team codes (2-3 letters)
        TEAM_CODES = {"MIA","WSH","DET","TB","MIN","CWS","NYM","SEA","SD","PHI",
                      "BAL","BOS","CLE","NYY","KC","CIN","TOR","ATL","SF","MIL",
                      "TEX","STL","ATH","CHC","PIT","HOU","COL","LAA","LAD","ARI"}

        # Extract the player part: after first dash, before the last -N
        m1 = re.search(r"-([A-Z]+)\d+-", ticker)
        if not m1:
            return True
        player_part = m1.group(1)

        # Find known team code at start of player_part
        player_team = ""
        for t_len in [3, 2]:
            prefix = player_part[:t_len]
            if prefix in TEAM_CODES:
                player_team = prefix
                break
        if not player_team:
            return True

        # Extract combined team string from the first part
        m2 = re.match(r"\w+-\d{2}[A-Z]{3}\d{2}\d{4}([A-Z]+)-", ticker)
        if not m2:
            return True
        combined = m2.group(1)

        # The other team is the remainder
        other = combined.replace(player_team, "", 1) if player_team in combined else ""
        if not other:
            return True

        key1 = f"{other}@{player_team}"
        key2 = f"{player_team}@{other}"
        status = status_map.get(key1, status_map.get(key2, ""))
        return status in ("", "Pre-Game", "Scheduled", "Warmup")
    except Exception:
        return True

def load_features():
    cfg = toml.load(CONFIG_DIR / "mlb.toml")
    from src.config.settings import SportConfig
    scfg = SportConfig(name="mlb", display_name="MLB",
                       rolling_windows=cfg["features"]["rolling_windows"], recency_decay=0.001)
    from src.execution.mlb_predictor import MLBLinePredictor
    predictor = MLBLinePredictor(scfg)
    predictor.load_data()
    return predictor._latest_features

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
        print("No feature data. Run training first.")
        return
    print(f"Loaded {len(latest)} players\n")

    all_opps = []
    model_cache = {}

    for mt in MARKET_TYPES:
        name = mt["name"]
        model_name = mt["model_name"]
        series = mt["series_ticker"]
        pos = mt["position"]
        pattern = mt["pattern"]
        desc = mt["desc"]

        info_only = mt.get("info_only", True)

        if model_name not in model_cache:
            m, s = _load_regressor(model_name)
            if m is None:
                print(f"  Skipping {name}: no regressor for {model_name}")
                continue
            model_cache[model_name] = (m, s)
        reg_model, reg_std = model_cache[model_name]

        print(f"Scanning {name} ({series})...", flush=True)
        mkts = client.list_markets(series_ticker=series, limit=1000)
        if mkts is None or mkts.empty:
            print(f"  No markets found")
            continue
        print(f"  {len(mkts)} markets", flush=True)

        opps = []
        for _, m in mkts.iterrows():
            try:
                ticker = m["ticker"]
                title = m["title"]
                yb = float(m["yes_bid_dollars"]) if m["yes_bid_dollars"] not in ("", "nan", "0.0000", None) else 0
                ya = float(m["yes_ask_dollars"]) if m["yes_ask_dollars"] not in ("", "nan", None) else 1
                if yb <= 0 and ya >= 1.0:
                    continue
                if yb <= 0 and ya <= 0:
                    continue
                yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

                line_match = re.match(pattern, title, re.IGNORECASE)
                if not line_match:
                    continue
                player_name = line_match.group(1).strip()
                line_val = int(line_match.group(2))
                if line_val <= 0:
                    continue

                row = _match_player(player_name, latest, position_filter=pos)
                if row is None:
                    continue

                # Skip players with insufficient data (all rolling averages NaN)
                avg_cols = [c for c in row.index if c.endswith("_avg_7") and isinstance(row[c], (int, float))]
                if not avg_cols or all(pd.isna(row[c]) for c in avg_cols):
                    continue

                p_yes, mu = _p_ge_line(row, reg_model, reg_std, line_val, stat_name=model_name)

                # Check recency: if player's actual rate (2024-2026, pref ≥3 games) differs from model,
                # use the more conservative rate (min = lower edge for YES betting)
                recency_rate, _, _ = _recency_check(player_name, line_val, stat_col=model_name.lower())
                recency_used = ""
                if recency_rate >= 0:
                    # Use recency rate as primary when available (2026 actual data beats model)
                    # min = conservative for YES betting (lower prob = less edge)
                    p_final = min(p_yes, recency_rate)
                    if abs(p_yes - recency_rate) > 0.10:
                        recency_used = f" (recency={recency_rate:.0%} model={p_yes:.0%})"
                        p_yes = p_final
                else:
                    p_final = p_yes

                yes_edge = p_yes - yes_mid
                no_edge = (1 - p_yes) - (1 - yes_mid)

                # Phase 4: Per-stat confidence gate (only flag as live-bettable
                # if the backtest says it's reliable). These flags match
                # PROJECT.md and the June 9 formal backtest.
                # Stats that fail naive OR have broken BetaCal stay info_only.
                STAT_LIVE_QUALITY = {
                    "SO":  True,   # 5/5 (100%), gentle cal
                    "HR":  True,   # 3/4 (75%), sharpest market — small bet size
                    "TB":  True,   # 5/5 (100%)
                    "H_R_RBI": True,  # 5/5 (100%)
                    "IP":  True,   # 5/5 (100%), Isotonic cal
                    "ER":  True,   # 4/4 (100%)
                    "H":   True,   # 5/5 (100%)
                    "BB":  True,   # 4/4 (100%)
                    "RBI": True,   # 4/4 (100%), Isotonic cal
                    "R":   False,  # 2/4 (50%) fails
                    "SB":  False,  # 1/4 (25%) worst
                }
                quality_pass = STAT_LIVE_QUALITY.get(model_name, False)

                # AND both gates: info_only flag (per market type) AND quality gate
                effective_info_only = info_only or (not quality_pass)

                opps.append({
                    "type": name,
                    "player": row.get("player_name", player_name),
                    "stat": desc,
                    "line": line_val,
                    "mu": round(mu, 2),
                    "sigma": round(reg_std, 2),
                    "p_yes": round(p_yes, 3),
                    "mkt_yes": round(yes_mid, 3),
                    "yes_edge": round(yes_edge, 3),
                    "no_edge": round(no_edge, 3),
                    "recency_rate": round(recency_rate, 3) if recency_rate >= 0 else None,
                    "recency_used": recency_used,
                    "ticker": ticker,
                    "info_only": effective_info_only,
                })
            except Exception:
                pass

        opps.sort(key=lambda x: max(abs(x["yes_edge"]), abs(x["no_edge"])), reverse=True)
        all_opps.extend(opps)

        if opps:
            print(f"  {'Player':25s} {'Line':>5s} {'P(Y)':>5s} {'Mkt':>5s} {'Edge(Y)':>7s} {'Edge(N)':>7s} {'2026':>6s} {'Note':>10s}")
            print(f"  " + "-" * 78)
            for o in opps[:5]:
                r = o.get("recency_rate")
                r_str = f"{r:.0%}" if r is not None else "N/A"
                note = o.get("recency_used", "")
                print(f"  {o['player'][:25]:25s} {o['line']:2d}+ {o['p_yes']:.0%} {o['mkt_yes']:.0%} "
                      f"{o['yes_edge']:>+6.1%} {o['no_edge']:>+6.1%} {r_str:>6s} {note:>10s}")
        else:
            print(f"  No matched opportunities")

    all_opps.sort(key=lambda x: max(abs(x["yes_edge"]), abs(x["no_edge"])), reverse=True)

    print(f"\nTotal matched opportunities: {len(all_opps)}")
    if all_opps:
        info_only_count = sum(1 for o in all_opps if o.get("info_only", True))
        if info_only_count == len(all_opps):
            print(f"  ⚠ ALL models are info_only (failed backtest — worse than naive baseline)")
            print(f"  ⚠ Edges shown below are NOISE — do not bet on them.")
        print(f"\nTop 10 overall:")
        print(f"  {'Type':4s} {'Player':25s} {'Stat':20s} {'Line':>5s} {'P(Y)':>5s} {'Mkt':>5s} {'Edge(Y)':>7s} {'2026':>6s} {'Note':>10s}")
        print(f"  " + "-" * 90)
        for o in all_opps[:10]:
            r = o.get("recency_rate")
            r_str = f"{r:.0%}" if r is not None else "N/A"
            note = o.get("recency_used", "")
            print(f"  {o['type']:4s} {o['player'][:25]:25s} {o['stat'][:20]:20s} {o['line']:2d}+ "
                  f"{o['p_yes']:.0%} {o['mkt_yes']:.0%} "
                  f"{o['yes_edge']:>+6.1%} {r_str:>6s} {note:>10s}")

    # Daily loss circuit breaker — track starting balance
    starting_balance = client.get_balance()
    daily_loss_limit = 0.10  # 10% max daily loss

    if args.bet:
        # Filter out info_only markets before placing bets
        bet_opps = [o for o in all_opps if not o.get("info_only", True)]
        if not bet_opps:
            print("\n  No non-info_only opportunities to bet on.")
        else:
            # Per 2026-06-11 diagnostic: NO-side opportunities (model < market
            # by 5%+) outnumber YES-side ~2.7x. Sort by max(|yes_edge|, |no_edge|)
            # so we surface the most aggressive opportunities first. Cap raised
            # 6→12: per-bet dollar cap is the real risk control, count cap is
            # just a sanity backstop.
            bet_opps.sort(key=lambda o: max(abs(o.get("yes_edge", 0)), abs(o.get("no_edge", 0))), reverse=True)
            print(f"\n--- PLACING ORDERS ({len(bet_opps)} candidates after info_only filter) ---")
            placed = 0
            daily_pnl = 0.0
            for o in bet_opps:
                if placed >= 12:  # Raised 2026-06-11: 692 NO opportunities vs 260 YES;
                    break         # per-bet dollar cap is the real risk control.
                # Skip info-only markets (ALL models failed backtest)
                if o.get("info_only", False):
                    continue
                # Skip in-progress games (model uses pre-game data)
                if not _game_is_pregame(o.get("ticker", "")):
                    continue
                # Circuit breaker: stop if daily loss > 10%
                if daily_pnl <= -starting_balance * daily_loss_limit:
                    print(f"  DAILY LOSS LIMIT HIT (-${abs(daily_pnl):.2f}), stopping")
                    break

                yes_edge = o["yes_edge"]
                no_edge = o["no_edge"]
                mkt_y = o["mkt_yes"]
                p_y = o["p_yes"]
                no_mid = 1.0 - mkt_y

                # ── Pick the side with the larger edge (NO wins ~2.7x more often) ──
                # `lift_size` is the resting depth on the side we'd be lifting
                # to fill this bet. For YES, that's the YES ask; for NO, that's
                # the NO ask (which equals 100 - YES bid). Read from the
                # market row's *_size_fp fields (V2 API fixed-point strings).
                if no_edge > yes_edge:
                    side = "no"
                    direction = "BUY NO"
                    bet_mid = no_mid      # relevant mid for the bet side
                    edge = no_edge
                    # NO ask liquidity ≈ contracts resting at the YES bid
                    try:
                        lift_size = float(m.get("yes_bid_size_fp") or 0)
                    except (TypeError, ValueError):
                        lift_size = 0
                else:
                    side = "yes"
                    direction = "BUY YES"
                    bet_mid = mkt_y
                    edge = yes_edge
                    # YES ask liquidity = contracts resting at the YES ask
                    try:
                        lift_size = float(m.get("yes_ask_size_fp") or 0)
                    except (TypeError, ValueError):
                        lift_size = 0

                # ── Fee-zone edge boost (40-60c on the BET side's mid) ────────
                if FEE_ZONE_LOW <= bet_mid <= FEE_ZONE_HIGH:
                    required_edge = FEE_ZONE_MIN_EDGE  # 7.5%
                else:
                    required_edge = 0.05                # 5%

                # ── Liquidity-aware price gate (relaxed 2026-06-11) ───────────
                # Core range: any market in 0.15 < bet_mid < 0.75 (was the old gate).
                # Extended range: 0.05 <= bet_mid <= 0.95 ONLY if the side we'd
                # lift has >= MIN_LIQUIDITY_CONTRACTS contracts of resting depth.
                in_core = 0.15 < bet_mid < 0.75
                in_extended = (EXTENDED_GATE_LOW <= bet_mid <= EXTENDED_GATE_HIGH
                               and lift_size >= MIN_LIQUIDITY_CONTRACTS)
                if not (in_core or in_extended):
                    continue
                if edge <= required_edge:
                    continue

                # ── Compute bid: sit just inside the side's mid ───────────────
                if side == "yes":
                    # YES bid: 1¢ above YES mid, capped at 98¢
                    bid = min(98, int(mkt_y * 100) + 1)
                else:
                    # NO bid: 1¢ below NO mid, floored at 2¢
                    bid = max(2, int(no_mid * 100) - 1)

                # ── Cost: pay `bid` cents per contract regardless of side ────
                # FIX (2026-06-11): prior code used (100-bid)/100 for NO, which
                # is the *profit if NO wins*, not the cost. The cost is bid/100
                # for both sides — buying YES at 80¢ costs 80¢/contract, same as
                # buying NO at 80¢.
                cost_per = bid / 100.0
                b = client.get_balance()
                target_risk = b * 0.05
                count = int(target_risk / cost_per)
                # Liquidity cap: at bid=10¢ a 5%-of-bankroll bet = 50 contracts,
                # which is unlikely to fill fully on a thin book. Cap at 25.
                count = min(count, 25)
                if count < 1:
                    print(f"  SKIP {o['player']}: can't risk <5% "
                          f"(1 contract = ${cost_per:.2f} > ${target_risk:.2f} limit)")
                    continue
                try:
                    client.create_order(ticker=o["ticker"], side=side,
                                        yes_price=bid if side == "yes" else (100 - bid),
                                        count=str(count))
                    daily_pnl -= cost_per * count
                    print(f"  {direction:8s} {o['type']:4s} {o['player'][:25]:25s} "
                          f"{o['stat'][:15]:15s} {o['line']}+ @ {bid}¢ x{count} "
                          f"(model={p_y:.0%} mkt_y={mkt_y:.0%} mkt_n={no_mid:.0%} "
                          f"edge={edge:+.1%} risk=${cost_per*count:.2f})", flush=True)
                    placed += 1
                except Exception as e:
                    print(f"  FAILED {o['player']}: {e}", flush=True)
        print(f"  Placed {placed} | Balance: ${client.get_balance():.2f}")

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
