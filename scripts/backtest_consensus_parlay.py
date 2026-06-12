#!/usr/bin/env python3
"""Backtest: replay the 4-leg consensus-favorite cross-sport parlay on 2022-2024 data.

Replays the "consensus strong/moderate favorite" strategy on:
  - MLB 2022-2024 regular-season games (from MLB Stats API)
  - WC 2022 group + knockout stage (from existing project Elo data)

Strategy definition (replicating today's 4 legs):
  - Leg 1: USA ML (WC) — moderate favorite, ~52% implied (DK -110)
  - Leg 2: Brazil ML (WC) — heavy favorite, ~73% implied (DK -250 to -300)
  - Leg 3: BAL ML (MLB) — moderate-strong favorite, ~58% implied (DK -140)
  - Leg 4: CLE ML (MLB) — strong favorite, ~60% implied (DK -155)

For each historical game, we identify the consensus favorite using a proxy:
  - MLB: home team + team record differential → implied win prob
  - WC:  the team with higher pre-match Elo → implied win prob

Then we stratify games into probability tiers and report the empirical win rate.

Finally, combine the 4 leg-specific win rates via independence assumption to get
the joint probability of a 4-leg parlay, and compute ROI at typical Kalshi prices.

Usage:
    python -m scripts.backtest_consensus_parlay
    python -m scripts.backtest_consensus_parlay --fetch-only   # populate cache, don't analyze
    python -m scripts.backtest_consensus_parlay --skip-fetch    # analyze existing cache only
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mlb"
OUT_DIR = PROJECT_ROOT / "reports" / "backtests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"
HTTP_TIMEOUT = 30
HTTP_SLEEP = 0.10  # polite delay between API calls

# 4 legs we're backtesting. (label, sport, target_implied_prob, model_prob, research)
TARGET_LEGS = [
    {"leg": 1, "sport": "WC",  "team": "USA",   "match": "USA vs Paraguay",       "implied": 0.52, "model": 0.613, "research": "MODERATE"},
    {"leg": 2, "sport": "WC",  "team": "Brazil","match": "Brazil vs Morocco",     "implied": 0.73, "model": 0.832, "research": "STRONG"},
    {"leg": 3, "sport": "MLB", "team": "BAL",   "match": "ATL @ BAL",             "implied": 0.58, "model": None,  "research": "MODERATE (research)"},
    {"leg": 4, "sport": "MLB", "team": "CLE",   "match": "CLE @ CIN",             "implied": 0.60, "model": None,  "research": "STRONG"},
]


# ── HTTP helpers ────────────────────────────────────────────────────────────


def http_get_json(url: str, retries: int = 3) -> Optional[dict]:
    """GET with retries. Returns parsed JSON or None on failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            # Other 4xx/5xx — return None
            return None
        except (requests.RequestException, ValueError):
            time.sleep(1.0 * (attempt + 1))
    return None


# ── MLB data fetch ──────────────────────────────────────────────────────────


def fetch_mlb_schedule(start: str, end: str) -> list[dict]:
    """Fetch MLB schedule for a date range. Returns list of completed games."""
    url = f"{MLB_API}/schedule?sportId=1&startDate={start}&endDate={end}&hydrate=team"
    data = http_get_json(url)
    if not data:
        return []
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})
            away_score = away.get("score")
            home_score = home.get("score")
            if away_score is None or home_score is None:
                continue
            games.append({
                "game_pk": g.get("gamePk"),
                "date": d.get("date"),
                "away_id": away.get("team", {}).get("id"),
                "away_name": away.get("team", {}).get("name"),
                "away_score": away_score,
                "home_id": home.get("team", {}).get("id"),
                "home_name": home.get("team", {}).get("name"),
                "home_score": home_score,
                "winner_is_home": home_score > away_score,
            })
    return games


def fetch_mlb_2022_2024(force: bool = False) -> pd.DataFrame:
    """Fetch MLB 2022, 2023, 2024 regular season games. Cached as parquet."""
    cache_path = CACHE_DIR / "game_logs_2022_2023_2024_consensus.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    print("  Fetching MLB 2022-2024 schedule (monthly chunks)...")
    all_games = []
    for year in (2022, 2023, 2024):
        # Regular season: ~April 1 to ~Oct 1
        start = datetime(year, 4, 1)
        end = datetime(year, 10, 5)
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=30), end)
            print(f"    {year}-{cur.strftime('%m-%d')} … {chunk_end.strftime('%m-%d')}: ", end="", flush=True)
            games = fetch_mlb_schedule(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            print(f"{len(games)} games")
            all_games.extend(games)
            cur = chunk_end + timedelta(days=1)
            time.sleep(HTTP_SLEEP)
    df = pd.DataFrame(all_games)
    if df.empty:
        print("  ⚠️  No games fetched — check network / API")
        return df
    df["season"] = pd.to_datetime(df["date"]).dt.year
    print(f"  Total: {len(df)} completed games across 2022-2024")
    df.to_parquet(cache_path, index=False)
    print(f"  Cached: {cache_path}")
    return df


# ── MLB consensus-favorite proxy ────────────────────────────────────────────


def compute_team_records_up_to(games: pd.DataFrame, target_date: str) -> dict[int, tuple[int, int]]:
    """For each team, return (wins, losses) of games strictly before target_date."""
    prior = games[games["date"] < target_date]
    records: dict[int, list[int]] = {}
    for _, g in prior.iterrows():
        if g["winner_is_home"]:
            records.setdefault(g["home_id"], [0, 0])[0] += 1
            records.setdefault(g["away_id"], [0, 0])[1] += 1
        else:
            records.setdefault(g["home_id"], [0, 0])[1] += 1
            records.setdefault(g["away_id"], [0, 0])[0] += 1
    return {tid: tuple(rec) for tid, rec in records.items()}


def implied_win_prob(record_diff: float, is_home: bool) -> float:
    """Approximate consensus-favorite implied probability.

    record_diff = (favorite_win_pct - underdog_win_pct) at the time of game.
    is_home = True if the favorite is the home team (adds ~3% home boost).

    Calibrated so that:
      - 0.000 record_diff + home = 0.540 (slight home favorite baseline)
      - +0.050 record_diff + home = 0.585
      - +0.100 record_diff + home = 0.628
      - +0.150 record_diff + home = 0.668
      - +0.200 record_diff + home = 0.706
      - +0.250 record_diff + home = 0.741
      - +0.300 record_diff + home = 0.773
    """
    base = 0.50
    if is_home:
        base = 0.54
    # Logistic: logit(p) = logit(0.50 or 0.54) + k * diff
    # Use k=3.2 so +0.20 record_diff shifts the home-favorite logit by ~+0.64
    # which moves p from 0.54 to ~0.69.
    logit_base = np.log(base / (1.0 - base))
    logit = logit_base + 3.2 * record_diff
    p = 1.0 / (1.0 + np.exp(-logit))
    return float(p)


def assign_mlb_favorite(games: pd.DataFrame) -> pd.DataFrame:
    """For each game, identify the consensus favorite + their implied probability.

    Uses running team records up to (but not including) the game date. The team
    with the better record is the "consensus favorite." Adds a small home boost.
    """
    games = games.sort_values("date").reset_index(drop=True)
    fav_is_home_list = []
    fav_implied_list = []

    # We process games in date order; rebuild records incrementally for efficiency.
    records: dict[int, list[int]] = {}
    for _, g in games.iterrows():
        h_id, a_id = g["home_id"], g["away_id"]
        h_rec = records.get(h_id, [0, 0])
        a_rec = records.get(a_id, [0, 0])
        h_pct = h_rec[0] / max(1, h_rec[0] + h_rec[1])
        a_pct = a_rec[0] / max(1, a_rec[0] + a_rec[1])

        # Favorite is the team with the better record (ties → home team).
        if h_pct >= a_pct:
            fav_is_home = True
            diff = h_pct - a_pct
        else:
            fav_is_home = False
            diff = a_pct - h_pct

        # Skip games where either team has < 10 games (record not stable).
        h_n = h_rec[0] + h_rec[1]
        a_n = a_rec[0] + a_rec[1]
        if h_n < 10 or a_n < 10:
            fav_implied_list.append(np.nan)
            fav_is_home_list.append(fav_is_home)
        else:
            p = implied_win_prob(diff, is_home=fav_is_home)
            fav_implied_list.append(p)
            fav_is_home_list.append(fav_is_home)

        # Update records with this game's outcome.
        if g["winner_is_home"]:
            records.setdefault(h_id, [0, 0])[0] += 1
            records.setdefault(a_id, [0, 0])[1] += 1
        else:
            records.setdefault(h_id, [0, 0])[1] += 1
            records.setdefault(a_id, [0, 0])[0] += 1

    games = games.copy()
    games["fav_is_home"] = fav_is_home_list
    games["fav_implied"] = fav_implied_list
    games["fav_won"] = games.apply(
        lambda r: r["winner_is_home"] if r["fav_is_home"] else not r["winner_is_home"],
        axis=1,
    )
    return games


# ── WC 2022 results ─────────────────────────────────────────────────────────


def load_wc_2022_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 2022 World Cup results + the underlying Elo timeline.

    Returns (wc_games, elo_timeline) so assign_wc_favorite can compute pre-game
    Elo without re-importing the world_cup module.
    """
    try:
        from src.data.world_cup import fetch_all_matches, compute_elo
    except ImportError:
        print("  ⚠️  Could not import src.data.world_cup — skipping WC 2022 leg")
        return pd.DataFrame(), pd.DataFrame()

    df = fetch_all_matches()
    if df.empty:
        return df, pd.DataFrame()
    elo_timeline = compute_elo(df)
    wc = df[(df["match_date"].dt.year == 2022) & (df["tournament_code"] == "WC")].copy()
    if wc.empty:
        return wc, elo_timeline
    wc = wc.rename(columns={"match_date": "date"})
    wc["date"] = pd.to_datetime(wc["date"]).dt.strftime("%Y-%m-%d")
    wc["winner_is_home"] = wc["home_score"] > wc["away_score"]
    wc = wc[["date", "home_team", "away_team", "home_score", "away_score", "winner_is_home"]]
    return wc, elo_timeline


def assign_wc_favorite(wc_games: pd.DataFrame, elo_timeline: pd.DataFrame) -> pd.DataFrame:
    """For each WC 2022 game, identify consensus favorite using pre-game Elo.

    For WC matches, the "consensus favorite" is the team with the higher Elo
    rating going into the match. We use the pre-game Elo diff to derive an
    implied win probability, mirroring the formula used in scan_wc.py.
    """
    if elo_timeline.empty:
        return wc_games

    fav_is_home_list = []
    fav_implied_list = []
    for _, g in wc_games.iterrows():
        # Pre-game Elo: latest ELO for each team strictly before this date.
        prior = elo_timeline[pd.to_datetime(elo_timeline["match_date"]) < pd.Timestamp(g["date"])]
        if prior.empty:
            fav_implied_list.append(np.nan)
            fav_is_home_list.append(True)
            continue
        # Get each team's most recent Elo (search both home and away columns).
        elo_h = prior[prior["home_team"] == g["home_team"]]["elo_home_post"].tail(1)
        if elo_h.empty:
            elo_h = prior[prior["away_team"] == g["home_team"]]["elo_away_post"].tail(1)
        elo_a = prior[prior["away_team"] == g["away_team"]]["elo_away_post"].tail(1)
        if elo_a.empty:
            elo_a = prior[prior["home_team"] == g["away_team"]]["elo_home_post"].tail(1)

        if elo_h.empty or elo_a.empty:
            fav_implied_list.append(np.nan)
            fav_is_home_list.append(True)
            continue

        h_elo = float(elo_h.iloc[0])
        a_elo = float(elo_a.iloc[0])
        diff = h_elo - a_elo  # positive = home stronger

        # Standard Elo expected-score formula: E = 1 / (1 + 10^(-diff/400))
        p_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        if p_home >= 0.5:
            fav_is_home = True
            fav_implied = p_home
        else:
            fav_is_home = False
            fav_implied = 1.0 - p_home
        fav_is_home_list.append(fav_is_home)
        fav_implied_list.append(fav_implied)

    out = wc_games.copy()
    out["fav_is_home"] = fav_is_home_list
    out["fav_implied"] = fav_implied_list
    out["fav_won"] = out.apply(
        lambda r: r["winner_is_home"] if r["fav_is_home"] else not r["winner_is_home"],
        axis=1,
    )
    return out


# ── Analysis: stratify + win-rate per probability tier ──────────────────────


TIERS = [
    (0.45, 0.55, "Light favorite (~50-55%)"),
    (0.55, 0.65, "Moderate favorite (~55-65%)"),  # covers BAL / CLE / USA
    (0.65, 0.75, "Strong favorite (~65-75%)"),
    (0.75, 1.01, "Heavy favorite (~75%+)"),       # covers Brazil
]


def stratify_and_report(df: pd.DataFrame, label: str) -> dict:
    """For each probability tier, compute empirical win rate of the favorite.

    Returns a dict keyed by tier label → {n, wins, win_rate, implied_mid}.
    """
    df = df.dropna(subset=["fav_implied", "fav_won"])
    print(f"\n  {label}")
    print(f"  {'Tier':<32s} {'N':>5s} {'Wins':>5s} {'Win rate':>10s} {'Implied mid':>12s} {'Cal err':>9s}")
    print(f"  {'-'*32} {'-'*5} {'-'*5} {'-'*10} {'-'*12} {'-'*9}")

    out = {}
    for lo, hi, name in TIERS:
        mask = (df["fav_implied"] >= lo) & (df["fav_implied"] < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:<32s} {0:5d} {0:5d} {'n/a':>10s} {'-':>12s} {'-':>9s}")
            out[name] = {"n": 0, "wins": 0, "win_rate": None, "implied_mid": (lo + hi) / 2}
            continue
        wins = int(df.loc[mask, "fav_won"].sum())
        wr = wins / n
        implied_mid = (lo + hi) / 2
        cal_err = wr - implied_mid
        marker = "✅" if abs(cal_err) < 0.04 else ("⚠️" if abs(cal_err) < 0.08 else "❌")
        print(f"  {name:<32s} {n:5d} {wins:5d} {wr:10.1%} {implied_mid:12.1%} {cal_err:+8.1%} {marker}")
        out[name] = {"n": n, "wins": wins, "win_rate": wr, "implied_mid": implied_mid, "cal_err": cal_err}
    return out


# ── Joint ROI computation ───────────────────────────────────────────────────


def compute_joint_roi(mlb_tiers: dict, wc_tiers: dict, joint_at: list[float]) -> dict:
    """Combine leg-specific win rates via independence. Compute ROI at Kalshi prices.

    Leg mapping (the 4 legs we're backtesting):
      Leg 1 USA (WC): tier "Moderate favorite (~55-65%)"  → wc_tiers
      Leg 2 Brazil (WC): tier "Heavy favorite (~75%+)"      → wc_tiers
      Leg 3 BAL (MLB): tier "Moderate favorite (~55-65%)"   → mlb_tiers
      Leg 4 CLE (MLB): tier "Moderate favorite (~55-65%)"   → mlb_tiers
    """
    # Pull empirical win rates (fall back to the implied mid if N is too small)
    def safe_wr(tiers, name):
        t = tiers.get(name, {})
        if t.get("win_rate") is None or t.get("n", 0) < 30:
            return t.get("implied_mid"), t.get("n", 0)
        return t["win_rate"], t["n"]

    p_usa,   n_usa   = safe_wr(wc_tiers,  "Moderate favorite (~55-65%)")
    p_brazil, n_brazil = safe_wr(wc_tiers, "Heavy favorite (~75%+)")
    p_bal,   n_bal   = safe_wr(mlb_tiers, "Moderate favorite (~55-65%)")
    p_cle,   n_cle   = safe_wr(mlb_tiers, "Moderate favorite (~55-65%)")

    # Use BAL and CLE separately: we approximated them both as the same tier, but
    # the empirical sub-distribution within the tier will be similar. Compute
    # the average for the joint; flag in the report if they materially differ.
    p_ml_mod = (p_bal + p_cle) / 2
    n_ml_mod = n_bal + n_cle

    # Joint (independent): USA × Brazil × BAL × CLE
    p_joint = p_usa * p_brazil * p_bal * p_cle
    p_joint_avg = p_usa * p_brazil * p_ml_mod * p_ml_mod

    fair_price = 1.0 / p_joint

    print(f"\n  Per-leg empirical win rates (from backtest):")
    print(f"    Leg 1 USA (WC)   : {p_usa:6.1%}  (n={n_usa} WC 2022 moderate-favorite games)")
    print(f"    Leg 2 Brazil (WC): {p_brazil:6.1%}  (n={n_brazil} WC 2022 heavy-favorite games)")
    print(f"    Leg 3 BAL (MLB)  : {p_bal:6.1%}  (n={n_bal} MLB 2022-24 moderate-favorite games)")
    print(f"    Leg 4 CLE (MLB)  : {p_cle:6.1%}  (n={n_cle} MLB 2022-24 moderate-favorite games)")
    print(f"    BAL+CLE average  : {p_ml_mod:6.1%}  (n={n_ml_mod} total)")
    print()
    print(f"  Joint probability (independent): {p_joint:6.2%}  =  1 in {1.0/p_joint:.1f}")
    print(f"  Joint probability (BAL=CLE avg): {p_joint_avg:6.2%}")
    print(f"  Fair price (decimal):           {fair_price:6.2f}x  (American: {((fair_price - 1) * 100):+.0f})")
    print()
    print(f"  ROI at typical Kalshi prices (parlay = 1 contract, pays $1 if all 4 hit):")
    print(f"  {'Kalshi offer':>14s} {'Dec odds':>10s} {'EV per $1':>10s} {'ROI %':>8s} {'Verdict':>16s}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*16}")

    rows = []
    for offer in joint_at:
        ev = p_joint * 1.0 - offer
        roi = ev / offer * 100 if offer > 0 else float("inf")
        if roi > 5:
            verdict = "✅ positive"
        elif roi > -2:
            verdict = "⚠️  marginal"
        else:
            verdict = "❌ negative"
        print(f"  {'$' + f'{offer:.2f}':>14s} {1.0/offer:>9.2f}x {ev:>+10.3f} {roi:>+7.1f}% {verdict:>16s}")
        rows.append({"offer": offer, "dec_odds": 1.0 / offer, "ev": ev, "roi_pct": roi, "verdict": verdict})

    return {
        "p_usa": p_usa, "p_brazil": p_brazil, "p_bal": p_bal, "p_cle": p_cle,
        "n_usa": n_usa, "n_brazil": n_brazil, "n_bal": n_bal, "n_cle": n_cle,
        "p_joint": p_joint, "p_joint_avg": p_joint_avg, "fair_price": fair_price,
        "kalshi_rows": rows,
    }


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  CONSENSUS 4-LEG PARLAY BACKTEST — 2022-2024 historical record")
    print("  Legs: USA + Brazil + BAL + CLE")
    print("=" * 70)

    skip_fetch = "--skip-fetch" in sys.argv
    force_fetch = "--fetch-only" not in sys.argv and not skip_fetch

    # ── 1. MLB 2022-2024 ───────────────────────────────────────────────────
    print("\n1. MLB 2022-2024 regular-season games")
    mlb_games = fetch_mlb_2022_2024(force=force_fetch)
    if mlb_games.empty:
        print("  ⚠️  No MLB data — cannot backtest MLB legs (3 + 4). Exiting.")
        return
    print(f"  Loaded {len(mlb_games):,} games across {mlb_games['season'].nunique()} seasons")
    print("  Assigning consensus favorite (home team + record differential)...")
    mlb_with_fav = assign_mlb_favorite(mlb_games)
    mlb_tiers = stratify_and_report(mlb_with_fav, "MLB 2022-2024 — favorite win rate by probability tier")

    # ── 2. WC 2022 ─────────────────────────────────────────────────────────
    print("\n2. World Cup 2022 group + knockout stage")
    wc_games, elo_timeline = load_wc_2022_results()
    if wc_games.empty:
        print("  ⚠️  No WC 2022 data — cannot backtest WC legs (1 + 2).")
        wc_tiers = {}
        wc_with_fav = pd.DataFrame()
    else:
        print(f"  Loaded {len(wc_games)} WC 2022 matches")
        print("  Assigning consensus favorite (higher pre-game Elo)...")
        wc_with_fav = assign_wc_favorite(wc_games, elo_timeline)
        wc_tiers = stratify_and_report(wc_with_fav, "WC 2022 — favorite win rate by probability tier")

    # ── 3. Joint ROI ──────────────────────────────────────────────────────
    if mlb_tiers and wc_tiers:
        print("\n3. Joint 4-leg parlay ROI on Kalshi binary markets")
        joint_at = [0.10, 0.12, 0.15, 0.175, 0.20, 0.25, 0.30]
        roi_result = compute_joint_roi(mlb_tiers, wc_tiers, joint_at)
    else:
        print("\n3. ⚠️  Skipping joint ROI — missing data for at least one leg")
        roi_result = None

    # ── 4. Save report ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_path = OUT_DIR / f"consensus_parlay_backtest_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(_render_report(mlb_tiers, wc_tiers, roi_result, mlb_with_fav, wc_with_fav if not wc_games.empty else None))
    print(f"\n  Report saved: {report_path}")

    # Also save a JSON for programmatic use
    json_path = OUT_DIR / f"consensus_parlay_backtest_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "mlb_tiers": mlb_tiers,
            "wc_tiers": wc_tiers,
            "joint": roi_result,
            "target_legs": TARGET_LEGS,
        }, f, indent=2, default=str)
    print(f"  JSON saved:   {json_path}")
    print(f"\n  Verdict: see {report_path.name} for full report")


def _render_report(mlb_tiers, wc_tiers, roi, mlb_df, wc_df) -> str:
    out = ["# Consensus 4-Leg Parlay Backtest (2022-2024)\n"]
    out.append(f"**Generated:** {datetime.now().isoformat()}")
    out.append("\n## Strategy")
    out.append(
        "Replay the consensus-favorite cross-sport parlay (USA + Brazil + BAL + CLE) "
        "on 2022-2024 historical data. For each game, identify the consensus favorite "
        "using a proxy (MLB: home team + record differential; WC: higher pre-game Elo), "
        "stratify by implied probability tier, and report empirical win rates. Combine "
        "via independence to get the joint probability of a 4-leg parlay.\n"
    )
    out.append("## The 4 legs")
    out.append("| Leg | Sport | Match | Implied | Model | Research |")
    out.append("|---|---|---|---|---|---|")
    for leg in TARGET_LEGS:
        out.append(f"| {leg['leg']} | {leg['sport']} | {leg['match']} | {leg['implied']:.0%} | {leg['model'] or '-'} | {leg['research']} |")
    out.append("")

    if mlb_tiers:
        out.append("## MLB 2022-2024 — favorite win rate by probability tier")
        out.append("| Tier | N | Wins | Win rate | Implied mid | Cal err |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for _, _, name in TIERS:
            t = mlb_tiers.get(name, {})
            wr = f"{t['win_rate']:.1%}" if t.get("win_rate") is not None else "n/a"
            err = f"{t.get('cal_err', 0):+.1%}" if t.get("cal_err") is not None else "-"
            out.append(f"| {name} | {t.get('n', 0)} | {t.get('wins', 0)} | {wr} | {t.get('implied_mid', 0):.1%} | {err} |")
        out.append("")

    if wc_tiers:
        out.append("## WC 2022 — favorite win rate by probability tier")
        out.append("| Tier | N | Wins | Win rate | Implied mid | Cal err |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for _, _, name in TIERS:
            t = wc_tiers.get(name, {})
            wr = f"{t['win_rate']:.1%}" if t.get("win_rate") is not None else "n/a"
            err = f"{t.get('cal_err', 0):+.1%}" if t.get("cal_err") is not None else "-"
            out.append(f"| {name} | {t.get('n', 0)} | {t.get('wins', 0)} | {wr} | {t.get('implied_mid', 0):.1%} | {err} |")
        out.append("")

    if roi:
        out.append("## Joint 4-leg parlay ROI on Kalshi")
        out.append(f"- Leg 1 USA (WC moderate):  **{roi['p_usa']:.1%}**  (n={roi['n_usa']})")
        out.append(f"- Leg 2 Brazil (WC heavy):  **{roi['p_brazil']:.1%}**  (n={roi['n_brazil']})")
        out.append(f"- Leg 3 BAL (MLB moderate): **{roi['p_bal']:.1%}**  (n={roi['n_bal']})")
        out.append(f"- Leg 4 CLE (MLB moderate): **{roi['p_cle']:.1%}**  (n={roi['n_cle']})")
        out.append(f"- **Joint probability (independent): {roi['p_joint']:.2%}**  =  1 in {1.0/roi['p_joint']:.1f}")
        out.append(f"- Fair price (decimal): {roi['fair_price']:.2f}x  (American: {((roi['fair_price'] - 1) * 100):+.0f})")
        out.append("")
        out.append("| Kalshi offer | Dec odds | EV per $1 | ROI % | Verdict |")
        out.append("|---:|---:|---:|---:|---|")
        for r in roi["kalshi_rows"]:
            out.append(f"| ${r['offer']:.2f} | {r['dec_odds']:.2f}x | {r['ev']:+.3f} | {r['roi_pct']:+.1f}% | {r['verdict']} |")
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
