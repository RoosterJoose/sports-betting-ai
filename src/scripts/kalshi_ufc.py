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


# ── Non-fighter outcomes to filter from combo titles ────────────────
NON_FIGHTER_OUTCOMES = {
    "ko/tko/dq", "submission", "decision", "fig", "tko", "ko",
    "ko/tko", "draw", "no contest", "nc", "majority decision",
    "split decision", "unanimous decision", "dq", "doctor stoppage",
    "via ko/tko", "via submission", "via decision",
    # Method-of-victory patterns
    "fight ends before round 3", "fight ends before round 4",
    "fight ends before round 2", "fight ends before round 5",
    "fight ends before round 1",
    "ko/tko/dq in round 1", "ko/tko/dq in round 2",
    "ko/tko/dq in round 3", "fight goes the distance",
}

# ── UFC Freedom 250 (June 14, 2026) full card ────────────────────────
# Format: fighter_name_lower → (opponent, weight_class, scheduled_rounds)
# Sourced from Tapology: https://www.tapology.com/fightcenter/events/137848-ufc-white-house
UPCOMING_MATCHUPS = {
    # Main event — Lightweight Championship (5 rounds)
    "ilia topuria": ("Justin Gaethje", "lightweight", 5),
    "justin gaethje": ("Ilia Topuria", "lightweight", 5),
    # Co-main — Interim Heavyweight Championship (5 rounds)
    "alex pereira": ("Ciryl Gane", "heavyweight", 5),
    "ciryl gane": ("Alex Pereira", "heavyweight", 5),
    # Bantamweight
    "sean o'malley": ("Aiemann Zahabi", "bantamweight", 3),
    "aiemann zahabi": ("Sean O'Malley", "bantamweight", 3),
    # Heavyweight
    "derrick lewis": ("Josh Hokit", "heavyweight", 3),
    "josh hokit": ("Derrick Lewis", "heavyweight", 3),
    # Lightweight
    "mauricio ruffy": ("Michael Chandler", "lightweight", 3),
    "michael chandler": ("Mauricio Ruffy", "lightweight", 3),
    # Middleweight
    "bo nickal": ("Kyle Daukaus", "middleweight", 3),
    "kyle daukaus": ("Bo Nickal", "middleweight", 3),
    # Featherweight
    "diego lopes": ("Steve Garcia", "featherweight", 3),
    "steve garcia": ("Diego Lopes", "featherweight", 3),
}


# ── Patterns for non-fighter entries in combo titles ────────────────
# All-caps abbreviation as first word (e.g. "CAR Hurricanes", "UFC Fighter")
_CAPS_PREFIX_TEAM = re.compile(r"^[A-Z]{2,5}\s+")
# Single-letter suffix (e.g. "Los Angeles D", "Minnesota W")
_SINGLE_LETTER_TEAM = re.compile(r"^.+\s+[A-Z]$")
# Standalone all-caps abbreviations (e.g. "CAR", "UFC", "NFL")
_ALL_CAPS_ONLY = re.compile(r"^[A-Z]{2,5}$")


def parse_combo_title(title: str) -> list[str]:
    """Parse comma-separated Kalshi UFC combo title.

    'yes Diego Lopes,yes Bo Nickal,yes KO/TKO/DQ,yes Fig'
    → ['Diego Lopes', 'Bo Nickal']

    Filters out non-fighter outcomes (KO/TKO, Submission, Decision, etc.)
    and non-fighter entities (team names, city abbreviations).
    """
    title_clean = title.strip()
    fighters = []

    # Handle comma-separated format
    if "," in title_clean:
        parts = title_clean.split(",")
        for part in parts:
            part = part.strip()
            # Strip leading 'yes ' or 'no '
            cleaned = re.sub(r"^(yes|no)\s+", "", part, flags=re.IGNORECASE).strip()
            if not cleaned:
                continue
            if cleaned.lower() in NON_FIGHTER_OUTCOMES:
                continue
            # Filter out team names / abbreviations
            if _ALL_CAPS_ONLY.match(cleaned):
                continue
            if _CAPS_PREFIX_TEAM.match(cleaned):
                continue
            if _SINGLE_LETTER_TEAM.match(cleaned):
                continue
            # Must look like a name (has at least one space, not a method/finish)
            if " " in cleaned and not cleaned.lower().startswith(("via ", "by ")):
                fighters.append(cleaned)
        if fighters:
            return fighters

    # Fallback: try legacy "vs" pattern
    sep = r"(?:vs\.?|VS\.?|to\s+defeat|to\s+beat|defeats?|beats?)"
    vs_match = re.search(rf"(.+?)\s+{sep}\s+(.+?)(?:\s+wins?|$)", title_clean)
    if vs_match and not vs_match.group(1).strip().endswith(("wins", "win")):
        return [vs_match.group(1).strip(), vs_match.group(2).strip()]

    return []


def get_opponent(fighter_name: str) -> tuple:
    """Look up upcoming opponent for a fighter.
    Returns (opponent_name_or_None, weight_class, scheduled_rounds).
    If opponent unknown, returns (None, wc, rounds) for generic-opponent fallback.
    """
    key = fighter_name.lower().strip()
    if key in UPCOMING_MATCHUPS:
        return UPCOMING_MATCHUPS[key]
    # Fuzzy match
    for k, v in UPCOMING_MATCHUPS.items():
        if key in k or k in key:
            return v
    return (None, "middleweight", 3)


def _make_generic_opponent(wc: str, wc_avg: dict) -> dict:
    """Build a weight-class-average opponent stats dict for fighters
    whose opponent is unknown."""
    wc_key = wc.lower().replace(" ", "_")
    wc_entry = wc_avg.get(wc_key, wc_avg.get("middleweight", {}))
    return {
        "avg_sig_str_landed": wc_entry.get("avg_sig_str_landed", 27.0),
        "avg_sig_str_pct": wc_entry.get("avg_sig_str_pct", 0.48),
        "avg_td_landed": wc_entry.get("avg_td_landed", 1.3),
        "avg_td_pct": wc_entry.get("avg_td_pct", 0.35),
        "avg_sub_att": wc_entry.get("avg_sub_att", 0.5),
        "current_win_streak": 1, "current_lose_streak": 0,
        "longest_win_streak": 3,
        "wins": wc_entry.get("avg_wins", 8),
        "losses": wc_entry.get("avg_losses", 4),
        "total_rounds_fought": wc_entry.get("avg_rounds", 12),
        "total_title_bouts": 0,
        "height_cms": wc_entry.get("avg_height", 178.0),
        "reach_cms": wc_entry.get("avg_reach", 183.0),
        "weight_lbs": wc_entry.get("avg_weight", 170.0),
        "age": 30, "odds": 0,
        "win_by_ko_tko": 3, "win_by_submission": 2,
        "win_by_decision_unanimous": 2,
        "win_by_decision_split": 0, "win_by_decision_majority": 0,
        "match_weightclass_rank": 50,
        "stance": "orthodox",
    }


def _predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal):
    """Predict P(fighter_a wins) given pre-fetched stats for both fighters."""
    f1, f2 = f_stats, opp_stats
    stats = {
        "r_avg_sig_str_landed": f1.get("avg_sig_str_landed", 27.0),
        "b_avg_sig_str_landed": f2.get("avg_sig_str_landed", 27.0),
        "r_avg_sig_str_pct": f1.get("avg_sig_str_pct", 0.48),
        "b_avg_sig_str_pct": f2.get("avg_sig_str_pct", 0.48),
        "r_avg_td_landed": f1.get("avg_td_landed", 1.3),
        "b_avg_td_landed": f2.get("avg_td_landed", 1.3),
        "r_avg_td_pct": f1.get("avg_td_pct", 0.35),
        "b_avg_td_pct": f2.get("avg_td_pct", 0.35),
        "r_avg_sub_att": f1.get("avg_sub_att", 0.5),
        "b_avg_sub_att": f2.get("avg_sub_att", 0.5),
        "r_current_win_streak": f1.get("current_win_streak", 1),
        "b_current_win_streak": f2.get("current_win_streak", 1),
        "r_current_lose_streak": f1.get("current_lose_streak", 0),
        "b_current_lose_streak": f2.get("current_lose_streak", 0),
        "r_longest_win_streak": f1.get("longest_win_streak", 3),
        "b_longest_win_streak": f2.get("longest_win_streak", 3),
        "r_wins": f1.get("wins", 5), "b_wins": f2.get("wins", 5),
        "r_losses": f1.get("losses", 5), "b_losses": f2.get("losses", 5),
        "r_total_rounds_fought": f1.get("total_rounds_fought", 10),
        "b_total_rounds_fought": f2.get("total_rounds_fought", 10),
        "r_total_title_bouts": f1.get("total_title_bouts", 0),
        "b_total_title_bouts": f2.get("total_title_bouts", 0),
        "r_height_cms": f1.get("height_cms", 178.0),
        "b_height_cms": f2.get("height_cms", 178.0),
        "r_reach_cms": f1.get("reach_cms", 183.0),
        "b_reach_cms": f2.get("reach_cms", 183.0),
        "r_weight_lbs": f1.get("weight_lbs", 170.0),
        "b_weight_lbs": f2.get("weight_lbs", 170.0),
        "r_age": f1.get("age", 30), "b_age": f2.get("age", 30),
        "r_odds": f1.get("odds", 0), "b_odds": f2.get("odds", 0),
        "r_win_by_ko_tko": f1.get("win_by_ko_tko", 3),
        "b_win_by_ko_tko": f2.get("win_by_ko_tko", 3),
        "r_win_by_submission": f1.get("win_by_submission", 2),
        "b_win_by_submission": f2.get("win_by_submission", 2),
        "r_win_by_decision_unanimous": f1.get("win_by_decision_unanimous", 2),
        "b_win_by_decision_unanimous": f2.get("win_by_decision_unanimous", 2),
        "r_win_by_decision_split": f1.get("win_by_decision_split", 0),
        "b_win_by_decision_split": f2.get("win_by_decision_split", 0),
        "r_win_by_decision_majority": f1.get("win_by_decision_majority", 0),
        "b_win_by_decision_majority": f2.get("win_by_decision_majority", 0),
        "r_match_weightclass_rank": f1.get("match_weightclass_rank", 50),
        "b_match_weightclass_rank": f2.get("match_weightclass_rank", 50),
        "r_stance": f1.get("stance", "orthodox"),
        "b_stance": f2.get("stance", "orthodox"),
        "weight_class": wc, "no_of_rounds": rounds,
        "game_id": "0", "game_date": pd.Timestamp.now(),
        "total_fight_time_secs": 652, "finish_round": 3,
        "title_bout": 1 if rounds >= 5 else 0,
        "gender": "Womens" if "women" in wc else "Male",
        "better_rank": 0,
    }
    row = pd.DataFrame([stats])
    featured = build_ufc_features(row)
    for c in features:
        if c not in featured.columns:
            featured[c] = 0.0
    X = featured[[c for c in features if c in featured.columns]].fillna(0)
    prob = float(model.predict_proba(X)[0, 1])
    # Calibration
    calibrated = prob
    for entry in cal:
        lo, hi = entry["bin_lo"], entry["bin_hi"]
        if lo <= prob < hi:
            calibrated = entry["actual_rate"]
            break
    return calibrated


def scan():
    kc = KalshiClient()
    model, meta, cal, fighter_db, wc_avg = load_model()
    if model is None:
        return

    features = meta["features"] if meta else FEATURE_COLS

    mkts = kc.list_markets(limit=500)
    if mkts is None or mkts.empty:
        print("  No markets available")
        return

    # Filter for fight-related content (broader: catch multi-outcome combos)
    txt = mkts["ticker"].str.cat(mkts["title"], sep=" ", na_rep="")
    mask = txt.str.contains(
        r"UFC|MMA|FIGHT(?:ER)?|BOXING|FIGHTING|WINNER\b|VS\.?\s"
        r"|MULTIGAMESPORTS|MULTIGAMEEXTENDED.*KO|MULTIGAMEEXTENDED.*TKO|MULTIGAMEEXTENDED.*Submission|MULTIGAMEEXTENDED.*decision",
        case=False, na=False, regex=True
    )
    # Also catch markets where the title has known UFC fighter names
    ufc_fighter_mask = mkts["title"].str.contains(
        r"Ilia Topuria|Alex Pereira|Sean O'Malley|Diego Lopes|Bo Nickal|Ciryl Gane|"
        r"Justin Gaethje|Derrick Lewis|Michael Chandler|Mauricio Ruffy|"
        r"Aiemann Zahabi|Steve Garcia|Josh Hokit|Kyle Daukaus",
        case=False, na=False, regex=True
    )
    mask = mask | ufc_fighter_mask
    ufc = mkts[mask]
    if ufc.empty:
        print("  No UFC/fight markets found on Kalshi")
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

        # Parse fighters from title
        fighters = parse_combo_title(title)
        if not fighters:
            continue

        # ── Determine if this is a multi-outcome combo or single fight ──
        num_fighters = len(fighters)

        # Look up opponents and compute individual win probabilities
        indv_probs = {}
        indv_in_db = {}
        missing_opponents = []
        for fn in fighters:
            opp, wc, rounds = get_opponent(fn)
            f_stats = get_fighter_stats(fn, fighter_db, wc_avg)

            if opp:
                opp_stats = get_fighter_stats(opp, fighter_db, wc_avg)
                prob = _predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal)
            else:
                opp_stats = _make_generic_opponent(wc, wc_avg)
                prob = _predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal)
                missing_opponents.append(fn)

            indv_probs[fn] = prob
            indv_in_db[fn] = fn in fighter_db

        # Joint probability: product of individual win probabilities (independence)
        joint_prob = 1.0
        for p in indv_probs.values():
            joint_prob *= p

        # Apply a mild correlation penalty for multi-leg combos
        if num_fighters > 2:
            # Reduce joint prob by 2% per additional leg (conservative)
            joint_prob *= 0.98 ** (num_fighters - 2)

        edge = joint_prob - yes_mid
        price_cents = max(1, int(yes_mid * 100))
        all_in_db = all(indv_in_db.values())

        # Build display string
        fighter_display = ", ".join(fighters[:6])
        db_status = " ".join("✅" if indv_in_db.get(f, False) else "❌" for f in fighters[:4])
        if len(fighters) > 4:
            db_status += f" +{len(fighters)-4}"

        results.append({
            "ticker": ticker, "title": title,
            "fighters": fighters,
            "p_model": joint_prob, "market_prob": yes_mid,
            "edge": edge, "price_cents": price_cents,
            "all_in_db": all_in_db,
            "num_legs": num_fighters,
            "indv_probs": indv_probs,
        })

        print(f"  [{num_fighters} legs] {fighter_display:60s}  "
              f"model={joint_prob:.0%} mkt={yes_mid:.0%} "
              f"edge={edge:+.0%}  {db_status}")
        if missing_opponents:
            print(f"    ⚠  generic opponents for: {', '.join(missing_opponents[:3])}")

    if not results:
        print("  No fight markets could be parsed")
        return

    print(f"\n  Balance: ${balance:.2f}")
    # Qualifying: edge≥5%, all fighters in DB, price 10-80¢
    qualifying = [r for r in results
                  if r["edge"] >= 0.05
                  and 0.10 <= r["price_cents"] / 100 <= 0.80
                  and r["all_in_db"]]
    print(f"  Qualifying bets (edge≥5%, all fighters in DB, price 10-80¢): {len(qualifying)}")

    if qualifying:
        print(f"\n  {'#Legs':5s} {'Fighters':60s} {'Model':>6s} {'Mkt':>6s} {'Edge':>6s} {'Price'}")
        print(f"  {'─'*90}")
        for r in sorted(qualifying, key=lambda x: -x["edge"])[:10]:
            f_str = ", ".join(r["fighters"][:6])
            print(f"  {r['num_legs']:5d} {f_str:60s} "
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


def get_ufc_bets(kc=None, min_edge=0.05) -> list:
    """Return structured list of qualifying UFC bets for morning_scan integration.

    Handles both single-fight markets (legacy "vs" format) and multi-outcome
    combo markets (comma-separated "yes FighterA,yes FighterB,..." format).

    Each bet dict has the same schema as other morning_scan bet dicts:
      type, ticker, side, price_cents, model_prob, market_prob, edge,
      contracts, player, team, line_val, stat_desc, label
    """
    kc = kc or KalshiClient()
    model, meta, cal, fighter_db, wc_avg = load_model()
    if model is None:
        return []

    features = meta["features"] if meta else FEATURE_COLS

    mkts = kc.list_markets(limit=500)
    if mkts is None or mkts.empty:
        return []

    txt = mkts["ticker"].str.cat(mkts["title"], sep=" ", na_rep="")
    mask = txt.str.contains(
        r"UFC|MMA|FIGHT(?:ER)?|BOXING|FIGHTING|WINNER\b|VS\.?\s"
        r"|MULTIGAMESPORTS|MULTIGAMEEXTENDED.*KO|MULTIGAMEEXTENDED.*TKO|MULTIGAMEEXTENDED.*Submission|MULTIGAMEEXTENDED.*decision",
        case=False, na=False, regex=True
    )
    # Also catch markets where the title has known UFC fighter names
    ufc_fighter_mask = mkts["title"].str.contains(
        r"Ilia Topuria|Alex Pereira|Sean O'Malley|Diego Lopes|Bo Nickal|Ciryl Gane|"
        r"Justin Gaethje|Derrick Lewis|Michael Chandler|Mauricio Ruffy|"
        r"Aiemann Zahabi|Steve Garcia|Josh Hokit|Kyle Daukaus",
        case=False, na=False, regex=True
    )
    mask = mask | ufc_fighter_mask
    ufc = mkts[mask]
    if ufc.empty:
        return []

    results = []
    for _, m in ufc.iterrows():
        try:
            ticker = m["ticker"]
            title = m.get("title", "")
            yb_v = m.get("yes_bid_dollars", 0)
            ya_v = m.get("yes_ask_dollars", 1)
            yb = 0.0 if (isinstance(yb_v, float) and (yb_v != yb_v)) else float(yb_v or 0)
            ya = 1.0 if (isinstance(ya_v, float) and (ya_v != ya_v)) else float(ya_v or 1)
            if yb <= 0 and ya >= 1.0:
                continue
            yes_mid = max(0.01, min(0.99, (yb + ya) / 2.0))

            fighters = parse_combo_title(title)
            if not fighters:
                continue

            num_fighters = len(fighters)

            # Compute individual win probabilities
            indv_probs = {}
            indv_in_db = {}
            for fn in fighters:
                opp, wc, rounds = get_opponent(fn)
                f_stats = get_fighter_stats(fn, fighter_db, wc_avg)

                if opp:
                    opp_stats = get_fighter_stats(opp, fighter_db, wc_avg)
                    prob = _predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal)
                else:
                    opp_stats = _make_generic_opponent(wc, wc_avg)
                    prob = _predict_winner_direct(f_stats, opp_stats, wc, rounds, model, features, cal)

                indv_probs[fn] = prob
                indv_in_db[fn] = fn in fighter_db

            # Joint probability
            joint_prob = 1.0
            for p in indv_probs.values():
                joint_prob *= p
            if num_fighters > 2:
                joint_prob *= 0.98 ** (num_fighters - 2)

            edge = joint_prob - yes_mid
            all_in_db = all(indv_in_db.values())

            if edge < min_edge or not all_in_db:
                continue
            if yes_mid < 0.10 or yes_mid > 0.80:
                continue

            f_list = ", ".join(fighters[:4])
            results.append({
                "type": "UFC",
                "ticker": ticker,
                "side": "yes",
                "price_cents": max(1, int(yes_mid * 100)),
                "model_prob": round(joint_prob, 4),
                "market_prob": round(yes_mid, 4),
                "edge": round(edge, 4),
                "contracts": 1,
                "player": f_list if num_fighters <= 3 else f"{f_list} +{num_fighters-3}",
                "team": "",
                "line_val": num_fighters,
                "stat_desc": "combo_win",
                "label": f"UFC {num_fighters}-leg: {f_list}",
            })
        except Exception:
            pass

    return results


if __name__ == "__main__":
    scan()
