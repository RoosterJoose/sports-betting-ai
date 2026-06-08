#!/usr/bin/env python3
"""Compare simulated K props against Kalshi market prices to quantify bias.

Runs the MLB Monte Carlo simulator, fetches live Kalshi KXMLBKS strikeout
markets, and computes:
  - P_model = NB-smoothed P(K >= line) from the simulator
  - P_market = Kalshi midpoint price (market-implied probability)
  - Bias = P_model - P_market  (positive = model thinks over is more likely than market)

Usage:
    python -m src.scripts.compare_sim_vs_market
"""
from __future__ import annotations

import sys
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.settings import PROJECT_ROOT
from src.mlb.mlb_simulator import MLBSimulator, DEFAULT_RELIEVER, DEFAULT_BATTER
from src.data.kalshi import KalshiClient
from src.models.distributions import p_ge_stat

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
CACHE_DIR = PROJECT_ROOT / "data/cache/mlb"
STATCAST_DIR = CACHE_DIR / "statcast"

# MLB Stats API team IDs → 3-letter codes
MLB_API_TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS",
    112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
    116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT",
    135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA",
    147: "NYY", 158: "MIL",
}

KALSHI_K_PATTERN = re.compile(r"^(.+?):\s*(\d+)\+?\s*strikeouts?\??$", re.IGNORECASE)


# ── Schedule ─────────────────────────────────────────────────────

def fetch_schedule(date_str: str | None = None) -> list[dict]:
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team,lineup"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Schedule fetch error: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("codedState") == "I":
                continue
            teams = game.get("teams", {})
            away, home = teams.get("away", {}), teams.get("home", {})
            away_id = away.get("team", {}).get("id")
            home_id = home.get("team", {}).get("id")
            ap = away.get("probablePitcher", {})
            hp = home.get("probablePitcher", {})
            games.append({
                "game_pk": game.get("gamePk"),
                "away_team": MLB_API_TEAM_IDS.get(away_id, ""),
                "home_team": MLB_API_TEAM_IDS.get(home_id, ""),
                "away_pitcher_name": ap.get("fullName", "") if ap else "",
                "away_pitcher_id": ap.get("id") if ap else None,
                "home_pitcher_name": hp.get("fullName", "") if hp else "",
                "home_pitcher_id": hp.get("id") if hp else None,
            })
    return games


# ── Feature loaders (from scan_mlb_sim) ──────────────────────────

def load_statcast_agg() -> pd.DataFrame:
    files = sorted(STATCAST_DIR.glob("statcast_agg_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_raw_statcast_pitcher_agg() -> pd.DataFrame:
    files = sorted(STATCAST_DIR.glob("statcast_202*.parquet"))
    if not files:
        return pd.DataFrame()
    recent = files[-2:]
    chunks = []
    for f in recent:
        df = pd.read_parquet(f, columns=[
            "pitcher", "game_pk", "game_date", "player_name",
            "launch_speed", "launch_angle", "events", "description", "bb_type",
        ])
        chunks.append(df)
    raw = pd.concat(chunks, ignore_index=True)
    if raw.empty:
        return pd.DataFrame()

    has_contact = raw["launch_speed"].notna() & raw["launch_angle"].notna()
    agg_list = []
    for (pitcher_id, game_pk), group in raw.groupby(["pitcher", "game_pk"]):
        n_pa = len(group)
        n_contact = int(group["launch_speed"].notna().sum())
        contact_mask = has_contact.loc[group.index]
        contact_group = group[contact_mask]
        n_gb = int((group["bb_type"] == "ground_ball").sum()) if "bb_type" in group.columns else 0
        n_fb = int((group["bb_type"].str.contains("fly_ball", na=False)).sum()) if "bb_type" in group.columns else 0
        n_ld = int((group["bb_type"] == "line_drive").sum()) if "bb_type" in group.columns else 0
        row = {
            "pitcher": pitcher_id, "game_pk": game_pk,
            "game_date": group["game_date"].iloc[0] if "game_date" in group.columns else None,
            "n_pa_faced": n_pa,
            "launch_speed_avg": float(contact_group["launch_speed"].mean()) if n_contact > 0 else None,
            "launch_angle_avg": float(contact_group["launch_angle"].mean()) if n_contact > 0 else None,
            "hard_hit_rate": float((contact_group["launch_speed"] >= 95).sum() / n_pa) if n_pa > 0 else 0.0,
            "barrel_rate": float(
                ((contact_group["launch_speed"] >= 98) &
                 (contact_group["launch_angle"] >= 26) &
                 (contact_group["launch_angle"] <= 30)).sum() / n_pa
            ) if n_pa > 0 else 0.0,
            "gb_rate": n_gb / max(n_pa, 1), "fb_rate": n_fb / max(n_pa, 1),
            "ld_rate": n_ld / max(n_pa, 1),
            "k_rate_raw": float((group["events"] == "strikeout").sum() / n_pa) if n_pa > 0 else 0.0,
            "bb_rate_raw": float((group["events"] == "walk").sum() / n_pa) if n_pa > 0 else 0.0,
        }
        agg_list.append(row)
    pitcher_game = pd.DataFrame(agg_list)
    if pitcher_game.empty:
        return pitcher_game
    pitcher_game = pitcher_game.sort_values(["pitcher", "game_pk"])
    roll_cols = ["launch_speed_avg", "launch_angle_avg", "hard_hit_rate",
                 "barrel_rate", "gb_rate", "fb_rate", "ld_rate",
                 "k_rate_raw", "bb_rate_raw"]
    for col in roll_cols:
        if col not in pitcher_game.columns:
            continue
        vals = pd.to_numeric(pitcher_game[col], errors="coerce")
        pitcher_game[f"{col}_ewm"] = vals.groupby(pitcher_game["pitcher"]).transform(
            lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean()
        )
    return pitcher_game


def load_pitching_stats() -> pd.DataFrame:
    f = CACHE_DIR / "pitching_stats_2024_2025_2026.parquet"
    if not f.exists():
        files = sorted(CACHE_DIR.glob("pitching_stats_*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(f)
    return df


def build_pitcher_features(
    pitcher_name: str, pitcher_id: int | None,
    statcast_pitcher_df: pd.DataFrame, pitching_df: pd.DataFrame,
) -> dict:
    """Build feature dict for a pitcher (minimal — just enough for sim)."""
    features = dict(DEFAULT_RELIEVER)
    features["name"] = pitcher_name
    if pitcher_id is None:
        return features
    pid_str = str(int(pitcher_id))

    # Statcast rolling stats
    if not statcast_pitcher_df.empty and "pitcher" in statcast_pitcher_df.columns:
        try:
            p_rows = statcast_pitcher_df[statcast_pitcher_df["pitcher"] == int(pid_str)]
        except (ValueError, TypeError):
            p_rows = pd.DataFrame()
        if not p_rows.empty:
            last = (p_rows.sort_values("game_pk").iloc[-1]
                    if "game_pk" in p_rows.columns else p_rows.iloc[-1])
            for key, col in [
                ("pitcher_avg_ev_against", "launch_speed_avg_ewm"),
                ("pitcher_avg_la_against", "launch_angle_avg_ewm"),
                ("pitcher_hard_hit_against", "hard_hit_rate_ewm"),
                ("pitcher_gb_rate_against", "gb_rate_ewm"),
                ("pitcher_fb_rate_against", "fb_rate_ewm"),
            ]:
                if col in last.index and pd.notna(last[col]):
                    features[key] = float(last[col])

    # Pitching stats
    if not pitching_df.empty:
        if "player_id" in pitching_df.columns:
            match = pitching_df[pitching_df["player_id"].astype(str) == pid_str]
        else:
            match = pd.DataFrame()
        if match.empty and "player_name" in pitching_df.columns:
            match = pitching_df[
                pitching_df["player_name"].str.contains(
                    pitcher_name.split()[-1], case=False, na=False
                )
            ]
        if not match.empty:
            last = match.sort_values("season").iloc[-1] if "season" in match.columns else match.iloc[-1]
            for key, col in [
                ("pitcher_fip", "fip"), ("pitcher_k9", "k9"), ("pitcher_bb9", "bb9"),
                ("pitcher_hr9", "hr9"), ("pitcher_whip", "whip"),
                ("pitcher_k_bb_pct", "k_bb_pct"),
            ]:
                if col in last.index and pd.notna(last[col]):
                    features[key] = float(last[col])
    return features


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 85)
    print(f"  MLB Simulator vs Market Comparison — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 85)

    # ── 1. Load simulator ──
    print("\n1. Loading PA outcome model...")
    try:
        sim = MLBSimulator()
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    # ── 2. Fetch schedule ──
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n2. Fetching schedule for {today}...")
    games = fetch_schedule(today)
    if not games:
        print("  No games found.")
        return
    print(f"  {len(games)} games")

    # ── 3. Load feature data ──
    print("\n3. Loading feature data...")
    statcast_pitcher_df = load_raw_statcast_pitcher_agg()
    pitching_df = load_pitching_stats()

    # Build default batters (9 per team)
    away_batters = [dict(DEFAULT_BATTER) | {"name": f"Away #{i+1}"} for i in range(9)]
    home_batters = [dict(DEFAULT_BATTER) | {"name": f"Home #{i+1}"} for i in range(9)]

    # ── 4. Kalshi market prices ──
    print("\n4. Fetching Kalshi KXMLBKS markets...")
    client = KalshiClient()
    mkts = client.list_markets(series_ticker="KXMLBKS", limit=500)
    if mkts is None or mkts.empty:
        print("  No Kalshi strikeout markets found.")
        return
    print(f"  {len(mkts)} markets")

    # Build market lookup: {pitcher_name: {line: mid_price}}
    market_lookup: dict[str, dict[int, float]] = {}
    for _, m in mkts.iterrows():
        title = m.get("title", "")
        match = KALSHI_K_PATTERN.match(title)
        if not match:
            continue
        player = match.group(1).strip()
        line_val = int(match.group(2))
        yb = float(m["yes_bid_dollars"]) if m.get("yes_bid_dollars") not in ("", "nan", "0.0000", None) else 0
        ya = float(m["yes_ask_dollars"]) if m.get("yes_ask_dollars") not in ("", "nan", None) else 1
        if yb <= 0 and ya >= 1.0:
            continue
        mid = max(0.01, min(0.99, (yb + ya) / 2.0))
        market_lookup.setdefault(player, {})[line_val] = mid

    print(f"  Unique pitchers with markets: {len(market_lookup)}")

    # ── 5. Simulate each game and compare ──
    print("\n5. Simulating games and comparing to market...")

    all_comparisons = []

    for game in games:
        away = game["away_team"]
        home = game["home_team"]
        ap_name = game["away_pitcher_name"]
        hp_name = game["home_pitcher_name"]

        if not ap_name or not hp_name:
            continue

        # Build features
        ap_feats = build_pitcher_features(ap_name, game["away_pitcher_id"],
                                           statcast_pitcher_df, pitching_df)
        hp_feats = build_pitcher_features(hp_name, game["home_pitcher_id"],
                                           statcast_pitcher_df, pitching_df)

        # Run simulation (1000 sims is enough for mean/std)
        result = sim.simulate_game(
            ap_feats, hp_feats, away_batters, home_batters, n_sims=1000,
        )

        # Compare each pitcher to market
        for side, pitcher_key, pitcher_name in [
            ("AWAY", "away_pitcher", ap_name),
            ("HOME", "home_pitcher", hp_name),
        ]:
            p = result[pitcher_key]
            mu = p["k_mean"]
            sigma = p["k_std"]

            if pitcher_name not in market_lookup:
                continue

            market_lines = market_lookup[pitcher_name]
            print(f"\n  {away if side=='AWAY' else home:4s} {pitcher_name[:25]:25s} "
                  f"K/game={mu:.1f}  σ={sigma:.1f}")

            header = f"  {'':10s} {'Line':>4s} {'P_model':>8s} {'P_mkt':>8s} {'Bias':>8s} {'Interp':>25s}"
            print(header)
            print(f"  {'':10s} {'─'*4:4s} {'─'*8:8s} {'─'*8:8s} {'─'*8:8s} {'─'*25:25s}")

            for line in sorted(market_lines.keys()):
                p_model = p_ge_stat("SO", mu, sigma, line)
                p_mkt = market_lines[line]
                bias = p_model - p_mkt

                # Interpretation
                if abs(bias) < 0.03:
                    interp = "Calibrated ✓"
                elif bias > 0:
                    if bias > 0.15:
                        interp = f"Model OVER by {bias:.0%}"
                    elif bias > 0.08:
                        interp = f"Model over {bias:.0%}"
                    else:
                        interp = f"Slightly over {bias:.0%}"
                else:
                    abias = abs(bias)
                    if abias > 0.15:
                        interp = f"Model UNDER by {abias:.0%}"
                    elif abias > 0.08:
                        interp = f"Model under {abias:.0%}"
                    else:
                        interp = f"Slightly under {abias:.0%}"

                print(f"  {'':10s} K≥{line:2d}  {p_model:>7.0%}  {p_mkt:>7.0%}  "
                      f"{bias:>+7.1%}  {interp:>25s}")

                all_comparisons.append({
                    "pitcher": pitcher_name,
                    "team": away if side == "AWAY" else home,
                    "opponent": home if side == "AWAY" else away,
                    "mu": round(mu, 2), "sigma": round(sigma, 2),
                    "line": line,
                    "p_model": round(p_model, 4),
                    "p_market": round(p_mkt, 4),
                    "bias": round(bias, 4),
                })

    # ── 6. Aggregate bias summary ──
    print(f"\n{'='*85}")
    print("  AGGREGATE BIAS SUMMARY")
    print(f"{'='*85}")

    if not all_comparisons:
        print("  No matched comparisons.")
        return

    df = pd.DataFrame(all_comparisons)

    # Overall stats
    mean_bias = df["bias"].mean()
    mean_abs_bias = df["bias"].abs().mean()
    print(f"\n  Overall (n={len(df)} line-pitcher pairs):")
    print(f"    Mean bias:      {mean_bias:+.1%}")
    print(f"    Mean |bias|:    {mean_abs_bias:.1%}")
    print(f"    Within ±3%:     {(df['bias'].abs() < 0.03).mean():.1%}")
    print(f"    Within ±5%:     {(df['bias'].abs() < 0.05).mean():.1%}")
    print(f"    Within ±10%:    {(df['bias'].abs() < 0.10).mean():.1%}")

    # By line value
    print(f"\n  By Line Value:")
    for line in sorted(df["line"].unique()):
        sub = df[df["line"] == line]
        print(f"    K≥{line:2d}:  mean bias={sub['bias'].mean():+.1%}  "
              f"|bias|={sub['bias'].abs().mean():.1%}  "
              f"n={len(sub)}")

    # Bias distribution by pitcher
    print(f"\n  By Pitcher (sorted by |bias|, top 10):")
    pitcher_bias = df.groupby("pitcher").agg(
        mean_bias=("bias", "mean"),
        mean_abs_bias=("bias", lambda x: x.abs().mean()),
        n_lines=("line", "count"),
        mean_mu=("mu", "mean"),
    ).sort_values("mean_abs_bias", ascending=False)

    for pitcher, row in pitcher_bias.head(10).iterrows():
        print(f"    {pitcher[:25]:25s}  "
              f"bias={row['mean_bias']:+.1%}  "
              f"|bias|={row['mean_abs_bias']:.1%}  "
              f"n_lines={int(row['n_lines'])}  "
              f"μ={row['mean_mu']:.1f}")

    # ── 7. Save to JSON ──
    out_path = Path("/tmp/mlb_sim_vs_market.json")
    df.to_json(out_path, orient="records", indent=2)
    print(f"\n  Saved to {out_path}")

    print(f"\n  Done at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
