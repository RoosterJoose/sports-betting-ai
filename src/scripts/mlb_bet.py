#!/usr/bin/env python3
"""MLB PrizePicks pipeline — uses all-position XGBRegressor models.

Predicts P(≥PrizePicks_line) via normal CDF on regressor output, then
Wang-corrects for market calibration bias. No more classifier target mismatch.

Usage:
    python -m src.scripts.mlb_bet
"""
import sys, json, re, warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.prizepicks import get_prizepicks_client
from src.features.mlb import MLBFeatureEngineer
from src.config.settings import SportConfig, CONFIG_DIR
from src.models.calibrator import BetaCalibrator
import toml

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "mlb"
CALIB_DIR = MODEL_DIR / "calibration"
BREAKEVEN = 0.542  # 5/6 Flex Play breakeven

# PrizePicks stat_type → ALL-position regressor model name
PP_REG_MAP = {
    "Pitcher Strikeouts":   "ALL_SO",
    "Earned Runs Allowed":  "ALL_ER",
    "Hits Allowed":         "ALL_H",
    "Walks Allowed":        "ALL_BB",
    "Pitching Outs":        "ALL_IP",
    "Hitter Strikeouts":    "ALL_SO",
    "Hits":                 "ALL_H",
    "RBIs":                 "ALL_RBI",
    "Walks":                "ALL_BB",
    "Home Runs":            "ALL_HR",
    "Stolen Bases":         "ALL_SB",
    "Total Bases":          "ALL_TB",
    "Runs":                 "ALL_R",
    "Hits+Runs+RBIs":       "ALL_H_R_RBI",
}

POSITION_MAP = {
    "pitcher": ["Pitcher Strikeouts", "Earned Runs Allowed", "Hits Allowed",
                "Walks Allowed", "Pitching Outs", "Pitcher Fantasy Score"],
    "hitter":  ["Total Bases", "Hitter Strikeouts", "Hits",
                "RBIs", "Walks", "Home Runs", "Stolen Bases", "Runs",
                "Singles", "Doubles", "Triples", "Hitter Fantasy Score",
                "Hits+Runs+RBIs"],
}


def load_regressor(model_name):
    """Load a LightGBM regressor (lgb_<stat>.txt) + meta + BetaCal.

    train_mlb_regression.py writes LightGBM models as `lgb_<stat>.txt` with
    companion `lgb_<stat>.meta.json`. BetaCal lives at the model-dir root
    (models/mlb/<stat>_beta_cal.json), not in models/mlb/calibration/ (which
    is the legacy empirical-cal dir, .gitignored).
    """
    name = model_name.lower()
    path = MODEL_DIR / f"lgb_{name}.txt"
    meta_path = MODEL_DIR / f"lgb_{name}.meta.json"
    if not path.exists() or not meta_path.exists():
        return None, None, None, None
    with open(meta_path) as f:
        meta = json.load(f)
    booster = lgb.Booster(model_file=str(path))
    # Load BetaCal from model-dir root (where fit_mlb_beta_cal.py writes)
    beta_cal = BetaCalibrator.load(MODEL_DIR / f"{name}_beta_cal.json")
    return booster, meta.get("residual_std", 1.0), meta.get("r2", 0), beta_cal


def p_ge_line(feat_row, booster, residual_std, line_val, beta_cal=None):
    """P(stat ≥ line_val) using normal CDF → Beta Calibration.

    Falls back to raw normal-CDF probability when Beta Calibration is unavailable.
    """
    feature_names = booster.feature_name()
    row = {c: feat_row.get(c, 0) for c in feature_names}
    X = pd.DataFrame([row]).fillna(0)
    # Ensure column order matches the booster (LightGBM is order-sensitive for prediction)
    X = X[feature_names]
    mu = booster.predict(X)[0]
    sigma = max(residual_std, 0.3)
    p_raw = 1.0 - norm.cdf((line_val - 0.5 - mu) / sigma)
    p_raw = max(0.001, min(0.999, float(p_raw)))

    if beta_cal is not None and beta_cal._fitted:
        p_corr = beta_cal(p_raw)
    else:
        p_corr = p_raw

    return max(0.001, min(0.999, float(p_corr))), float(mu)


def main():
    print("=" * 100)
    print(f"MLB PrizePicks Picks (Regressor) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)

    # ── 1. Load config ──
    with open(CONFIG_DIR / "mlb.toml") as f:
        raw = toml.load(f)
    cfg = SportConfig(name="mlb", display_name="MLB",
                      rolling_windows=raw["features"]["rolling_windows"],
                      recency_decay=0.001)
    fe = MLBFeatureEngineer(cfg)

    # ── 2. Load regressors ──
    loaded = {}
    for model_name in set(PP_REG_MAP.values()):
        m, s, r2, bc = load_regressor(model_name)
        if m is not None:
            loaded[model_name] = (m, s, r2, bc)

    print(f"Regressors: {list(loaded.keys())}", flush=True)
    for name, (_, s, r2, bc) in loaded.items():
        bc_str = f" + BetaCal" if bc._fitted else ""
        print(f"  {name:12s} σ_res={s:.3f}  R²={r2:.3f}{bc_str}", flush=True)

    # ── 3. Load MLB historical data → features ──
    cache_files = sorted(Path("data/cache/mlb").glob("game_logs_*.parquet"))
    if not cache_files:
        print("No cached MLB data. Run: python -m src.main train mlb")
        return
    all_games = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    print(f"Raw data: {len(all_games)} player-game rows", flush=True)

    all_featured = fe.build_features(all_games)
    print(f"Featured: {len(all_featured)} rows", flush=True)
    has_name = all_featured["player_name"].notna().sum() if "player_name" in all_featured.columns else 0
    print(f"Rows with player_name: {has_name}/{len(all_featured)}", flush=True)

    all_featured = all_featured.sort_values(["player_id", "game_date"])
    latest = all_featured.groupby("player_id").last().reset_index()

    # ── 4. Fetch PrizePicks lines ──
    scraper = get_prizepicks_client()
    lines = scraper.fetch_lines("mlb", league_id=2)
    print(f"PrizePicks: {len(lines)} total lines", flush=True)

    line_stats = set(lines["stat_type"].unique())
    mapped = set(PP_REG_MAP.keys())
    modeled = set(PP_REG_MAP.keys()) & line_stats
    print(f"PrizePicks stat types: {sorted(line_stats)}", flush=True)
    print(f"Mapped to regressors: {sorted(modeled)}", flush=True)

    # ── 5. Predict each line ──
    results = []
    errors = []

    for _, row in lines.iterrows():
        pp_stat = str(row.get("stat_type", "")).strip()
        reg_name = PP_REG_MAP.get(pp_stat)
        if reg_name is None:
            errors.append(("unmapped", pp_stat))
            continue
        if reg_name not in loaded:
            errors.append(("no_regressor", pp_stat))
            continue
        reg_model, reg_std, _, beta_cal = loaded[reg_name]

        pname = str(row.get("player_name", "")).strip()
        if not pname or pname == "nan":
            errors.append(("no_name", pp_stat))
            continue

        try:
            pp_line = float(row["line_score"])
        except (ValueError, TypeError):
            errors.append(("bad_line", pname))
            continue

        # Match player in featured data by name
        last_name = pname.split()[-1].replace("'", "").replace(".", "")
        match = latest[latest["player_name"].str.contains(last_name, case=False, na=False)]
        if match.empty:
            errors.append(("no_match", pname))
            continue
        if len(match) > 1:
            first_initial = pname.split()[0][0]
            exact = match[match["player_name"].str.lower().str.startswith(pname.split()[0].lower(), na=False)]
            if len(exact) == 1:
                match = exact
            else:
                # Narrow by position
                is_pitcher_prop = pp_stat in POSITION_MAP["pitcher"]
                if is_pitcher_prop:
                    pitchers = match[match["position"].isin(["P", "RP", "SP", "CP"])]
                    if len(pitchers) == 1:
                        match = pitchers
                if len(match) > 1:
                    match = match.iloc[[0]]

        feat_row = match.iloc[0]
        position = str(feat_row.get("position", ""))
        is_pitcher = position in ("P", "RP", "SP", "CP")

        # Position guard — only predict pitcher props for pitchers
        if pp_stat in POSITION_MAP["pitcher"] and not is_pitcher:
            errors.append(("hitter_no_pitcher_prop", pname))
            continue
        if pp_stat in POSITION_MAP["hitter"] and is_pitcher:
            errors.append(("pitcher_no_hitter_prop", pname))
            continue

        # Predict P(≥line) via regressor + normal CDF + Beta Calibration
        prob, mu = p_ge_line(feat_row.to_dict(), reg_model, reg_std, pp_line, beta_cal=beta_cal)

        # Edge vs PrizePicks breakeven
        edge = prob - BREAKEVEN
        if edge <= 0:
            continue

        results.append({
            "player": pname,
            "position": position,
            "stat": pp_stat,
            "line": round(pp_line, 2),
            "mu": round(mu, 2),
            "sigma": round(reg_std, 2),
            "prob": round(prob, 4),
            "edge": round(edge, 4),
        })

    # ── 6. Sort and display ──
    results.sort(key=lambda x: x["edge"], reverse=True)

    print(f"\nErrors: {len(errors)}")
    for e_type in sorted(set(e[0] for e in errors)):
        count = sum(1 for e in errors if e[0] == e_type)
        print(f"  {e_type}: {count}")

    print(f"\nValid predictions: {len(results)} ({len(set(r['player'] for r in results))} unique players)")

    # ── 7. Actionable filter ──
    def is_actionable(r):
        return r["edge"] > 0.02 and r["prob"] >= BREAKEVEN

    actionable = [r for r in results if is_actionable(r)]

    pitcher_act = [r for r in actionable if r["stat"] in POSITION_MAP["pitcher"]]
    hitter_act = [r for r in actionable if r["stat"] in POSITION_MAP["hitter"]]

    if pitcher_act:
        print(f"\n{'='*120}")
        print(f"PITCHER PROPS ({len(pitcher_act)} actionable)")
        print(f"{'Player':25s} {'Stat':26s} {'Line':>6s} {'μ':>5s} {'σ':>5s} {'P(≥)':>6s} {'Edge':>8s}")
        print("-" * 120)
        for r in pitcher_act[:15]:
            print(f"{r['player']:25s} {r['stat']:26s} {r['line']:6.2f} {r['mu']:5.2f} "
                  f"{r['sigma']:5.2f} {r['prob']:6.1%} {r['edge']:>+7.1%}")

    if hitter_act:
        print(f"\n{'='*120}")
        print(f"HITTER PROPS ({len(hitter_act)} actionable)")
        print(f"{'Player':25s} {'Stat':26s} {'Line':>6s} {'μ':>5s} {'σ':>5s} {'P(≥)':>6s} {'Edge':>8s}")
        print("-" * 120)
        for r in hitter_act[:20]:
            print(f"{r['player']:25s} {r['stat']:26s} {r['line']:6.2f} {r['mu']:5.2f} "
                  f"{r['sigma']:5.2f} {r['prob']:6.1%} {r['edge']:>+7.1%}")

    # ── 8. Slip builder (6 picks, mixed stats, 2+ teams) ──
    CALIBRATED_STATS = {"Earned Runs Allowed", "Hits Allowed", "Walks Allowed",
                        "Pitcher Strikeouts", "Hits", "Walks", "RBIs",
                        "Total Bases", "Hitter Strikeouts"}

    slip_candidates = [r for r in actionable if r["stat"] in CALIBRATED_STATS]
    slip_candidates.sort(key=lambda x: x["edge"], reverse=True)

    # Build slip: one per player, diverse stat types
    best_per_player = {}
    for r in slip_candidates:
        if r["player"] not in best_per_player or r["edge"] > best_per_player[r["player"]]["edge"]:
            best_per_player[r["player"]] = r
    sorted_by_player = sorted(best_per_player.values(), key=lambda x: x["edge"], reverse=True)

    slip = []
    stat_types_used = {}
    for r in sorted_by_player:
        if len(slip) >= 6:
            break
        if stat_types_used.get(r["stat"], 0) >= 2:
            continue
        slip.append(r)
        stat_types_used[r["stat"]] = stat_types_used.get(r["stat"], 0) + 1

    if slip:
        print(f"\n{'='*100}")
        print(f"PRIZEPICKS 5/6 FLEX SLIP")
        print("-" * 100)
        for r in slip:
            print(f"  {r['player']:25s} — {r['stat']:26s} OVER {r['line']:.1f} "
                  f"(P={r['prob']:.0%}, edge={r['edge']:.1%})")
        avg_prob = sum(r["prob"] for r in slip) / len(slip)
        print(f"\n  Avg P(≥line)/leg: {avg_prob:.0%}  |  Breakeven: {BREAKEVEN:.0%}")
        print(f"  Bankroll: $10 → bet $2-3 at app.prizepicks.com")

    # ── 9. Breakdown by stat type ──
    print(f"\n{'='*80}")
    print(f"BREAKDOWN BY STAT TYPE:")
    print("-" * 80)
    stat_breakdown = {}
    for r in results:
        stat_breakdown.setdefault(r["stat"], {"count": 0, "avg_edge": 0})
        stat_breakdown[r["stat"]]["count"] += 1
        stat_breakdown[r["stat"]]["avg_edge"] += r["edge"]
    for stat, info in sorted(stat_breakdown.items(), key=lambda x: -x[1]["count"]):
        info["avg_edge"] /= info["count"] if info["count"] else 1
        print(f"  {stat:30s} {info['count']:4d} predictions, avg edge {info['avg_edge']:.1%}")

    return results


if __name__ == "__main__":
    main()
