#!/usr/bin/env python3
"""Counterfactual: what would the kalshi_mlb_unified NO-side bet have made today?

# MUST STAY IN SYNC WITH src/scripts/kalshi_mlb_unified.py:
#   - _p_ge_line / cascade order  (Isotonic → BetaCal → Wang)
#   - _recency_check  (2026 ≥3 → 2025 → 2024)
#   - _game_is_pregame  (status map from /tmp/mlb_game_status.json)
#   - bet-side selection logic  (no_edge > yes_edge → BUY NO)
#   - fee zone (40-60c on BET side mid, requires 7.5% edge)
#   - price gate (0.15 < bet_mid < 0.75)
# If you change the production logic, mirror the change here.


Mirrors the production scanner's calibration cascade (Isotonic for HR/R/RBI/SB/IP,
BetaCal for others, Wang fallback) + recency floor + pre-game filter, then
applies the NEW bet-side selection logic to identify every NO-bet candidate
and estimates expected PnL using the model's predicted probability.

NO outcomes are not yet known (games haven't been played), so we report
*expected* PnL per contract = no_edge (the model's edge over the market).
Sum across all contracts is the expected value of the bet set under the
assumption that the model is well-calibrated.

Usage:
    python -m scripts.counterfactual_mlb_no_betting
    python -m scripts.counterfactual_mlb_no_betting --simulate  # add a Bernoulli
                                                             # draw around the
                                                             # model prob to get
                                                             # a noisy PnL band
"""
import sys, json, warnings, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config.settings import CONFIG_DIR, PROJECT_ROOT
from src.models.calibrator import BetaCalibrator
from src.models.distributions import p_ge_stat
import toml, lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
WANG_LAMBDA = 0.30
ISOTONIC_PREFERRED = {"ip", "r", "rbi", "sb", "hr", "blk", "stl"}

FEE_ZONE_LOW = 0.40
FEE_ZONE_HIGH = 0.60
FEE_ZONE_MIN_EDGE = 0.075
PRICE_GATE_LOW = 0.15
PRICE_GATE_HIGH = 0.75
# Liquidity-aware extended range (2026-06-11): unlocks 0-10c and 90-100c
# markets IF the side we'd lift has >= MIN_LIQUIDITY_CONTRACTS of resting depth.
EXTENDED_GATE_LOW = 0.05
EXTENDED_GATE_HIGH = 0.95
MIN_LIQUIDITY_CONTRACTS = 10
MIN_EDGE = 0.05
MAX_BETS = 6
BANKROLL = 100.0  # counterfactual bankroll
RISK_PCT = 0.05   # 5% of bankroll per bet, mirrors production


def _get_beta_cal(stat):
    p = MODEL_DIR / f"{stat.lower()}_beta_cal.json"
    return BetaCalibrator.load(p) if p.exists() else None


def _get_iso_cal(stat):
    from src.models.calibrator import IsotonicCalibrator
    p = MODEL_DIR / f"{stat.lower()}_isotonic_cal.json"
    return IsotonicCalibrator.load(p) if p.exists() else None


def _load_reg(name):
    mn = name.lower()
    p = MODEL_DIR / f"lgb_{mn}.txt"
    m = lgb.Booster(model_file=str(p))
    meta = json.load(open(MODEL_DIR / f"lgb_{mn}.meta.json"))
    return m, meta.get("residual_std", 1.0)


def _prod_p_ge(row, model, std, line, stat):
    """Production-exact cascade: Isotonic (preferred) -> BetaCal -> Wang."""
    if hasattr(model, "feature_name"):
        feats = model.feature_name()
    else:
        feats = [c for c in row.index if isinstance(row[c], (int, float))]
    mu = model.predict(pd.DataFrame([{c: row.to_dict().get(c, 0) for c in feats}]).fillna(0))[0]
    sigma = max(std, 0.3)
    p_raw = float(p_ge_stat(stat, mu, sigma, line))
    p_used = None
    used_cal = "raw"
    if stat.lower() in ISOTONIC_PREFERRED:
        ic = _get_iso_cal(stat)
        if ic is not None and ic._fitted:
            p_used = min(0.999, max(0.001, float(ic(p_raw))))
            used_cal = "isotonic"
    if p_used is None:
        bc = _get_beta_cal(stat)
        if bc is not None and bc._fitted:
            p_used = min(0.999, max(0.001, float(bc(p_raw))))
            used_cal = "beta"
    if p_used is None:
        z = _norm.ppf(p_raw)
        p_used = min(0.999, max(0.001, float(_norm.cdf(z - WANG_LAMBDA))))
        used_cal = "wang"
    return mu, p_raw, p_used, used_cal


# Recency check
_recency_df = None
def _get_recency_df():
    global _recency_df
    if _recency_df is not None: return _recency_df
    cp = PROJECT_ROOT / "data" / "cache" / "mlb" / "game_logs_2026_2025_2024.parquet"
    if cp.exists():
        _recency_df = pd.read_parquet(cp)
    return _recency_df


def _recency_check(player_name, line_val, stat_col):
    try:
        df = _get_recency_df()
        if df is None: return -1
        pitcher_stats = {"so", "ip", "h", "bb", "er"}
        is_pitcher = stat_col in pitcher_stats
        if is_pitcher:
            combined = (df["gs"] == 1) & (df["position"] == "P")
        else:
            combined = df["position"] != "P"
        pname_mask = df["player_name"].str.contains(player_name, case=False, na=False)
        for season in ["2026", "2025", "2024"]:
            games = df[pname_mask & (df["season"] == season) & combined]
            if len(games) >= 3: break
        if len(games) < 3: return -1
        if stat_col == "so":
            return float((games["so"] >= line_val).mean())
        if stat_col == "hr":
            return float((games["hr"] >= line_val).mean())
        if stat_col == "tb":
            tb = games["1b"] + 2*games["2b"] + 3*games["3b"] + 4*games["hr"]
            return float((tb >= line_val).mean())
        if stat_col == "h_r_rbi":
            hrr = games["h"] + games["r"] + games["rbi"]
            return float((hrr >= line_val).mean())
        if stat_col in games.columns:
            return float((games[stat_col] >= line_val).mean())
        return -1
    except Exception:
        return -1


# Pre-game filter
import json as _json
def _game_is_pregame(ticker):
    try:
        map_file = Path("/tmp/mlb_game_status.json")
        if not map_file.exists(): return True
        with open(map_file) as f: status_map = _json.load(f)
        TEAM_CODES = {"MIA","WSH","DET","TB","MIN","CWS","NYM","SEA","SD","PHI",
                      "BAL","BOS","CLE","NYY","KC","CIN","TOR","ATL","SF","MIL",
                      "TEX","STL","ATH","CHC","PIT","HOU","COL","LAA","LAD","ARI"}
        m1 = re.search(r"-([A-Z]+)\d+-", ticker)
        if not m1: return True
        player_part = m1.group(1)
        player_team = ""
        for t_len in [3, 2]:
            prefix = player_part[:t_len]
            if prefix in TEAM_CODES: player_team = prefix; break
        if not player_team: return True
        m2 = re.match(r"\w+-\d{2}[A-Z]{3}\d{2}\d{4}([A-Z]+)-", ticker)
        if not m2: return True
        combined = m2.group(1)
        other = combined.replace(player_team, "", 1) if player_team in combined else ""
        if not other: return True
        k1 = f"{other}@{player_team}"; k2 = f"{player_team}@{other}"
        st = status_map.get(k1, status_map.get(k2, ""))
        return st in ("", "Pre-Game", "Scheduled", "Warmup")
    except Exception:
        return True


# Per-stat backtest confidence gate (mirrors production STAT_LIVE_QUALITY)
STAT_LIVE_QUALITY = {
    "SO": True, "HR": True, "TB": True, "H_R_RBI": True,
    "IP": True, "ER": True, "H": True, "BB": True, "RBI": True,
    "R": False, "SB": False,
}


def _match(title, lc, pos=None):
    if not title or lc is None or lc.empty: return None
    df = lc
    if pos == "hitter": df = lc[lc.get("position", "") != "P"]
    elif pos == "pitcher": df = lc[lc.get("position", "") == "P"]
    if df.empty: return None
    clean = title.replace("?", "").replace(":", "").strip()
    parts = clean.split()
    if len(parts) < 2: return None
    first, last = parts[0], parts[-1]
    exact = df[df["player_name"].str.lower() == clean.lower()]
    if len(exact) == 1: return exact.iloc[0]
    lm = df[df["player_name"].str.lower().str.endswith(last.lower(), na=False)]
    if len(lm) == 1: return lm.iloc[0]
    la = df[df["player_name"].str.lower().str.contains(last.lower(), na=False)]
    if len(la) >= 1:
        fi = la[la["player_name"].str.lower().str[0] == first[0].lower()]
        return fi.iloc[0] if len(fi) >= 1 else la.iloc[0]
    return None


# Load features
from src.execution.mlb_predictor import MLBLinePredictor
from src.config.settings import SportConfig
cfg = toml.load(CONFIG_DIR / "mlb.toml")
scfg = SportConfig(name="mlb", display_name="MLB",
                   rolling_windows=cfg["features"]["rolling_windows"], recency_decay=0.001)
predictor = MLBLinePredictor(scfg)
predictor.load_data()
latest = predictor._latest_features

MARKET_TYPES = [
    ("KS", "SO", "KXMLBKS", "pitcher", r"^(.+?):\s*(\d+)\+?\s*strikeouts?\??$"),
    ("HR", "HR", "KXMLBHR", "hitter",  r"^(.+?):\s*(\d+)\+?\s*home\s*runs?\??$"),
    ("TB", "TB", "KXMLBTB", "hitter",  r"^(.+?):\s*(\d+)\+?\s*total\s*bases?\??$"),
    ("HRR","H_R_RBI","KXMLBHRR","hitter",r"^(.+?):\s*(\d+)\+?\s*hits\s*\+\s*runs\s*\+\s*RBIs?\??$"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Bernoulli-draw PnL using model prob to get a noisy band")
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--include-yes", action="store_true",
                        help="Also count YES-side bets (for comparison)")
    args = parser.parse_args()

    from src.data.kalshi import KalshiClient
    client = KalshiClient()
    print(f"Bankroll (counterfactual): ${BANKROLL:.2f}\n")

    # Collect every opportunity with full info
    opps = []
    for name, model_name, series, pos, pattern in MARKET_TYPES:
        mkts = client.list_markets(series_ticker=series, limit=1000)
        if mkts is None or mkts.empty:
            print(f"{series}: 0 markets"); continue
        print(f"{series}: {len(mkts)} markets, scanning...", flush=True)
        model, std = _load_reg(model_name)
        quality_pass = STAT_LIVE_QUALITY.get(model_name, False)
        for _, m in mkts.iterrows():
            try:
                ticker = m["ticker"]
                if not _game_is_pregame(ticker): continue
                yb = float(m.get("yes_bid_dollars") or 0)
                ya = float(m.get("yes_ask_dollars") or 1)
                if yb <= 0 and ya >= 1.0: continue
                if yb <= 0 and ya <= 0: continue
                yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))
                lm = re.match(pattern, m["title"], re.IGNORECASE)
                if not lm: continue
                player = lm.group(1).strip()
                line = int(lm.group(2))
                if line <= 0: continue
                row = _match(player, latest, pos)
                if row is None: continue
                avg_cols = [c for c in row.index if c.endswith("_avg_7") and isinstance(row[c], (int, float))]
                if not avg_cols or all(pd.isna(row[c]) for c in avg_cols): continue
                mu, p_raw, p_prod, used_cal = _prod_p_ge(row, model, std, line, model_name)
                rec = _recency_check(player, line, model_name.lower())
                p_floored = min(p_prod, rec) if rec >= 0 else p_prod
                yes_edge = p_floored - yes_mid
                no_edge = (1 - p_floored) - (1 - yes_mid)
                no_mid = 1 - yes_mid

                # Apply the NEW bet-side selection (mirrors production):
                if not quality_pass: continue
                if no_edge > yes_edge:
                    side = "no"; bet_mid = no_mid; edge = no_edge
                else:
                    side = "yes"; bet_mid = yes_mid; edge = yes_edge

                if FEE_ZONE_LOW <= bet_mid <= FEE_ZONE_HIGH:
                    req = FEE_ZONE_MIN_EDGE
                else:
                    req = MIN_EDGE
                # Side-relevant top-of-book depth for the extended gate
                if side == "yes":
                    try:
                        lift_size = float(m.get("yes_ask_size_fp") or 0)
                    except (TypeError, ValueError):
                        lift_size = 0
                else:
                    try:
                        lift_size = float(m.get("yes_bid_size_fp") or 0)
                    except (TypeError, ValueError):
                        lift_size = 0
                in_core = PRICE_GATE_LOW < bet_mid < PRICE_GATE_HIGH
                in_extended = (EXTENDED_GATE_LOW <= bet_mid <= EXTENDED_GATE_HIGH
                               and lift_size >= MIN_LIQUIDITY_CONTRACTS)
                if not (in_core or in_extended): continue
                if edge <= req: continue

                # Compute bid + count + cost (mirrors production)
                if side == "yes":
                    bid = min(98, int(yes_mid * 100) + 1)
                else:
                    bid = max(2, int(no_mid * 100) - 1)
                cost_per = bid / 100.0
                target_risk = BANKROLL * RISK_PCT
                count = max(1, int(target_risk / cost_per))

                # Expected PnL per contract (under model calibration):
                # If we buy YES at bid: profit = +1.0 - bid/100 if YES, -bid/100 if NO
                #   E[profit] = p_yes*(1 - bid/100) + (1-p_yes)*(-bid/100) = p_yes - bid/100 = edge
                # If we buy NO at bid: profit = +1.0 - bid/100 if NO, -bid/100 if YES
                #   E[profit] = (1-p_yes)*(1 - bid/100) + p_yes*(-bid/100) = (1-p_yes) - bid/100 = no_edge
                exp_pnl_per = edge
                exp_pnl_total = exp_pnl_per * count
                cost = cost_per * count

                opps.append({
                    "type": name, "player": row.get("player_name", player),
                    "line": line, "p_yes": p_floored, "used_cal": used_cal,
                    "mkt_yes": yes_mid, "mkt_no": no_mid,
                    "yes_edge": yes_edge, "no_edge": no_edge,
                    "side": side, "edge": edge, "bid": bid, "count": count,
                    "cost": cost, "exp_pnl": exp_pnl_total,
                    "ticker": ticker,
                })
            except Exception:
                pass

    # Sort by edge, take top MAX_BETS
    opps.sort(key=lambda o: o["edge"], reverse=True)
    selected = opps[:MAX_BETS]
    yes_picks = [o for o in selected if o["side"] == "yes"]
    no_picks = [o for o in selected if o["side"] == "no"]

    print(f"\n=== COUNTERFACTUAL: top {MAX_BETS} bets by edge ===\n")
    print(f"Total qualifying opportunities: {len(opps)}")
    print(f"  YES-side: {sum(1 for o in opps if o['side']=='yes')}")
    print(f"  NO-side:  {sum(1 for o in opps if o['side']=='no')}")
    print(f"Selected top {MAX_BETS}: {len(yes_picks)} YES, {len(no_picks)} NO\n")

    print(f"  {'Side':4s} {'Type':4s} {'Player':22s} {'Stat':14s} {'Line':>4s} "
          f"{'p_yes':>6s} {'mkt_y':>6s} {'mkt_n':>6s} {'edge':>6s} {'bid':>4s} "
          f"{'cnt':>3s} {'cost':>6s} {'E[PnL]':>7s}")
    print("  " + "-" * 105)
    total_cost = 0.0
    total_exp_pnl = 0.0
    for o in selected:
        s = "YES" if o["side"] == "yes" else "NO "
        print(f"  {s:4s} {o['type']:4s} {o['player'][:22]:22s} "
              f"{o['type']:14s} {o['line']:>2d}+ {o['p_yes']:>5.1%} "
              f"{o['mkt_yes']:>5.1%} {o['mkt_no']:>5.1%} {o['edge']:>+5.1%} "
              f"{o['bid']:>3d}¢ {o['count']:>3d} ${o['cost']:>5.2f} "
              f"${o['exp_pnl']:>+6.2f}")
        total_cost += o["cost"]
        total_exp_pnl += o["exp_pnl"]

    print("  " + "-" * 105)
    print(f"  {'TOTAL':4s} {'':4s} {'':22s} {'':14s} {'':>4s} "
          f"{'':>6s} {'':>6s} {'':>6s} {'':>6s} {'':>4s} "
          f"{sum(o['count'] for o in selected):>3d} "
          f"${total_cost:>5.2f} ${total_exp_pnl:>+6.2f}")
    roi = total_exp_pnl / total_cost if total_cost > 0 else 0
    print(f"\n  Total expected PnL:  ${total_exp_pnl:+.2f}  on  ${total_cost:.2f}  risked  (ROI {roi:+.1%})")

    # Run a noisy Monte Carlo if requested
    if args.simulate:
        print(f"\n=== MONTE CARLO: {args.n_sims} sims around model prob ===\n")
        rng = np.random.default_rng(42)
        sim_pnls = []
        for _ in range(args.n_sims):
            sim_pnl = 0.0
            for o in selected:
                p_y = o["p_yes"]
                # Bernoulli: did YES actually happen?
                yes_happened = rng.random() < p_y
                if o["side"] == "yes":
                    # Bought YES at bid cents
                    profit_per = (1.0 - o["bid"]/100) if yes_happened else (-o["bid"]/100)
                else:
                    # Bought NO at bid cents
                    profit_per = (1.0 - o["bid"]/100) if not yes_happened else (-o["bid"]/100)
                sim_pnl += profit_per * o["count"]
            sim_pnls.append(sim_pnl)
        sim_pnls = np.array(sim_pnls)
        print(f"  PnL distribution over {args.n_sims:,} sims:")
        print(f"    Mean:        ${sim_pnls.mean():+.2f}")
        print(f"    Median:      ${np.median(sim_pnls):+.2f}")
        print(f"    Std:         ${sim_pnls.std():.2f}")
        print(f"    5th %ile:    ${np.percentile(sim_pnls, 5):+.2f}  (worst-case-ish)")
        print(f"    25th %ile:   ${np.percentile(sim_pnls, 25):+.2f}")
        print(f"    75th %ile:   ${np.percentile(sim_pnls, 75):+.2f}")
        print(f"    95th %ile:   ${np.percentile(sim_pnls, 95):+.2f}  (best-case-ish)")
        print(f"    P(profit):   {(sim_pnls > 0).mean():.1%}")
        print(f"    P(>=$5):     {(sim_pnls >= 5).mean():.1%}")


if __name__ == "__main__":
    main()
