"""UFC prop bet pipeline: fetch odds, compute model probabilities, rank by edge.

Ties together:
  - The Odds API (moneyline odds for UFC events)
  - DK public event page scraper (method of victory, round of finish props)
  - Model-derived MoV + round-of-finish probabilities (from
    src/models/ufc_prop_probabilities.py)
  - Edge calculation + ranking

⚠️  WARNING: The MoV and round-of-finish probabilities are derived from
career win-method rates and weight-class finish rates, then OOF-calibrated
via `models/ufc/mov_calibration.json` (built by
`src/scripts/train_ufc_mov_cal.py`). The calibration corrects the raw
prior overcompression but still has look-ahead bias in the fighter
stats. Use with caution for live betting. The binary winner probability
(from `winner_v1.json`) IS calibrated via `winner_calibration.json`.

Usage:
    # Full pipeline for one event (DK URL + Odds API event id):
    python -m src.scripts.ufc_props \\
        --red "Ilia Topuria" --blue "Justin Gaethje" \\
        --wc lightweight --rounds 5 \\
        --dk-url "https://sportsbook.draftkings.com/event/u-fight-12345" \\
        --odds-api-event-id "abc123..."

    # Model-only mode (no market fetch — just rank model probabilities):
    python -m src.scripts.ufc_props \\
        --red "Ilia Topuria" --blue "Justin Gaethje" \\
        --wc lightweight --rounds 5 \\
        --model-only

    # CSV output (for downstream consumption):
    python -m src.scripts.ufc_props \\
        --red "..." --blue "..." --wc ... --rounds ... \\
        --dk-url "..." --csv > props.csv

The output is a ranked list of prop bets, sorted by absolute edge
(highest edge first). Bets with positive edge (model > market) are
candidates to bet YES; negative edge means the model thinks the line
is wrong (consider NO if the book offers it).
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from src.data.dk_props_scraper import DKPropsScraper
from src.data.odds_api import (
    OddsAPIClient,
    OddsAPIError,
    american_odds_to_implied_prob,
)
from src.features.ufc import FEATURE_COLS
from src.models.ufc_prop_probabilities import (
    load_fighter_db,
    load_wc_averages,
    prop_bet_model_probabilities,
)

MODEL_DIR = Path("models/ufc")

# Minimum edge to surface a prop bet. 5% is the standard threshold used
# elsewhere in the project (kalshi_ufc.py, kalshi_mlb_unified.py, etc.).
MIN_EDGE = 0.05


def load_winner_model():
    """Load the UFC winner model + meta + binary-winner calibration.

    Returns (model, meta, cal) where cal is the list of
    {bin_lo, bin_hi, model_pred, actual_rate, n} entries from
    models/ufc/winner_calibration.json. Returns (None, None, []) if the
    model file is missing (cal is [] if the calibration file is missing).
    """
    model_file = MODEL_DIR / "winner_v1.json"
    meta_file = MODEL_DIR / "winner_v1.meta.json"
    cal_file = MODEL_DIR / "winner_calibration.json"
    if not model_file.exists():
        return None, None, []
    model = XGBClassifier()
    model.load_model(str(model_file))
    with open(meta_file) as f:
        meta = json.load(f)
    cal = []
    if cal_file.exists():
        with open(cal_file) as f:
            cal = json.load(f)
    return model, meta, cal


def predict_p_red_wins(
    red_name: str,
    blue_name: str,
    weight_class: str,
    scheduled_rounds: int,
    model,
    features: list[str],
    fighter_db: dict,
    wc_avg: dict,
    cal: list | None = None,
) -> float | None:
    """Run the binary winner model and return P(red wins).

    Threading `cal` through to `_predict_winner_direct()` is critical —
    without it, the raw model output is systematically overconfident
    (the +76% edge on UFC underdogs bug class). Pass `None` (default)
    or `[]` to deliberately bypass calibration (e.g., for a
    calibration-validation test).
    """
    from src.scripts.kalshi_ufc import (
        get_fighter_stats,
        _predict_winner_direct,
    )
    red_stats, _ = get_fighter_stats(red_name, fighter_db, wc_avg)
    blue_stats, _ = get_fighter_stats(blue_name, fighter_db, wc_avg)
    return _predict_winner_direct(
        red_stats, blue_stats, weight_class, scheduled_rounds,
        model, features, cal=cal if cal is not None else [],
    )


def compute_prop_bet_edges(
    model_probs: dict,
    dk_props: list[dict],
    red_name: str,
    blue_name: str,
) -> list[dict]:
    """Match model probabilities to DK prop outcomes and compute edges.

    Returns a list of dicts, one per matched prop bet:
        {
            "prop_type": "method_of_victory",
            "fighter": "Ilia Topuria",
            "outcome": "KO/TKO",
            "model_prob": 0.20,
            "market_implied": 0.28,    # from DK American odds
            "edge": -0.08,             # model_prob - market_implied
            "dk_odds": +250,
            "side": "no",              # model < market → bet NO
        }
    """
    results = []
    for prop in dk_props:
        matched_prob = _match_prop_to_model(prop, model_probs, red_name, blue_name)
        if matched_prob is None:
            continue
        market_imp = american_odds_to_implied_prob(prop.get("odds"))
        if market_imp is None or market_imp <= 0:
            continue
        edge = matched_prob - market_imp
        results.append({
            "prop_type": prop.get("prop_type", "other"),
            "fighter": prop.get("fighter", ""),
            "outcome": prop.get("outcome", ""),
            "market_label": prop.get("market_label", ""),
            "model_prob": round(matched_prob, 4),
            "market_implied": round(market_imp, 4),
            "edge": round(edge, 4),
            "dk_odds": prop.get("odds", 0),
            "side": "yes" if edge > 0 else "no",
            "sportsbook": prop.get("sportsbook", "draftkings"),
        })
    return results


def _match_prop_to_model(
    prop: dict, model_probs: dict, red_name: str, blue_name: str
) -> float | None:
    """Match one DK prop outcome to the corresponding model probability.

    Returns None if the prop can't be matched (unrecognized outcome, etc.).
    """
    prop_type = prop.get("prop_type", "")
    outcome = (prop.get("outcome") or "").lower()
    fighter = (prop.get("fighter") or "").lower()

    # Determine which corner the prop is about
    is_red = red_name.lower() in fighter or fighter in red_name.lower()
    is_blue = blue_name.lower() in fighter or fighter in blue_name.lower()
    if not is_red and not is_blue:
        return None

    corner_prefix = "p_red_" if is_red else "p_blue_"

    if prop_type == "method_of_victory":
        if "ko" in outcome or "tko" in outcome:
            return model_probs[corner_prefix + "ko"]
        if "sub" in outcome:
            return model_probs[corner_prefix + "sub"]
        if "dec" in outcome:
            return model_probs[corner_prefix + "dec"]
        return None

    if prop_type == "round_of_finish":
        # Try to parse the round number from the outcome
        import re
        m = re.search(r"round\s*(\d)", outcome)
        if m:
            round_key = f"p_round_{m.group(1)}"
            if round_key in model_probs:
                return model_probs[round_key]
        if "distance" in outcome or "goes" in outcome:
            return model_probs["p_goes_distance"]
        return None

    if prop_type == "total_rounds":
        # "Over 1.5 rounds" → P(round_1 + round_2) for OVER, else goes_distance
        m = re.search(r"(over|under)\s*(\d+\.?\d*)", outcome)
        if not m:
            return None
        side, line = m.group(1), float(m.group(2))
        # Sum round probabilities up to the line
        n_full = int(line)  # rounds 1..n are fully included if line == n.0
        cum = 0.0
        for r in range(1, n_full + 1):
            cum += model_probs.get(f"p_round_{r}", 0.0)
        if line != n_full:  # fractional line like 1.5
            cum += model_probs.get(f"p_round_{n_full + 1}", 0.0) * (line - n_full)
        return cum if side == "over" else (1.0 - cum)

    return None


def rank_prop_bets(edges: list[dict], min_edge: float = MIN_EDGE) -> list[dict]:
    """Sort prop bets by absolute edge (highest first), filter by min_edge.

    Returns a new list (does not mutate input).
    """
    filtered = [e for e in edges if abs(e["edge"]) >= min_edge]
    return sorted(filtered, key=lambda e: abs(e["edge"]), reverse=True)


def print_ranked_bets(ranked: list[dict], model_probs: dict) -> None:
    """Print the ranked prop bet list to stdout in a human-readable format."""
    print(f"\n{'='*80}")
    print(f"  UFC PROP BET RANKING — {model_probs.get('fight', '?')}")
    print(f"  Weight class: {model_probs.get('weight_class', '?')}  |  "
          f"Rounds: {model_probs.get('scheduled_rounds', '?')}")
    print(f"  Model: P(red wins)={model_probs.get('p_red_wins', 0):.1%}  "
          f"P(blue wins)={model_probs.get('p_blue_wins', 0):.1%}")
    print(f"{'='*80}")
    if not ranked:
        print("  No prop bets meet the minimum edge threshold.")
        print(f"  (min_edge = {MIN_EDGE:.0%})")
        return
    print(f"  {'#':>3s}  {'Prop':<20s}  {'Fighter':<22s}  {'Outcome':<22s}  "
          f"{'Model':>6s}  {'Mkt':>6s}  {'Edge':>6s}  {'Side':>4s}  Odds")
    print(f"  {'-'*110}")
    for i, bet in enumerate(ranked, 1):
        print(f"  {i:>3d}  {bet['prop_type']:<20s}  {bet['fighter']:<22s}  "
              f"{bet['outcome']:<22s}  {bet['model_prob']:>5.1%}  "
              f"{bet['market_implied']:>5.1%}  {bet['edge']:>+5.1%}  "
              f"{bet['side']:>4s}  {bet['dk_odds']:+d}")


def write_csv(ranked: list[dict], path: str) -> None:
    """Write ranked bets to a CSV file at the given path."""
    if not ranked:
        # Write header-only CSV so downstream consumers don't break
        with open(path, "w", newline="") as f:
            f.write("prop_type,fighter,outcome,model_prob,market_implied,edge,side,dk_odds,sportsbook\n")
        return
    keys = ["prop_type", "fighter", "outcome", "model_prob",
            "market_implied", "edge", "side", "dk_odds", "sportsbook"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(ranked)


def main():
    parser = argparse.ArgumentParser(
        description="UFC prop bet pipeline: fetch odds, compute model probabilities, rank by edge."
    )
    parser.add_argument("--red", required=True, help="Red corner fighter name")
    parser.add_argument("--blue", required=True, help="Blue corner fighter name")
    parser.add_argument("--wc", default="middleweight", help="Weight class (e.g. lightweight, heavyweight)")
    parser.add_argument("--rounds", type=int, default=3, help="Scheduled rounds (3 or 5)")
    parser.add_argument("--dk-url", default="", help="DraftKings event URL (for prop scraping)")
    parser.add_argument("--odds-api-event-id", default="",
                        help="The Odds API event id (for moneyline odds)")
    parser.add_argument("--model-only", action="store_true",
                        help="Skip market fetch — just print model probabilities")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE,
                        help=f"Minimum edge to surface (default {MIN_EDGE})")
    parser.add_argument("--csv", default="", help="Write ranked bets to this CSV file")
    args = parser.parse_args()

    # ⚠️  User-facing warning about uncalibrated MoV/round probs
    print("⚠️  WARNING: MoV and round-of-finish probabilities are PRIOR-BASED")
    print("    ESTIMATES with no OOS validation. Binary winner prob IS calibrated.")
    print("    Do NOT bet these uncalibrated prop probs without further review.")
    print()

    # 1. Load model + DBs
    model, meta, cal = load_winner_model()
    if model is None:
        print("ERROR: UFC model not found. Run: python -m src.scripts.train_ufc", file=sys.stderr)
        sys.exit(1)
    features = meta.get("features", FEATURE_COLS)
    fighter_db = load_fighter_db()
    wc_avg = load_wc_averages()

    # 2. Get P(red wins) from the binary model (with calibration applied)
    p_red = predict_p_red_wins(
        args.red, args.blue, args.wc, args.rounds,
        model, features, fighter_db, wc_avg, cal=cal,
    )
    if p_red is None:
        print("ERROR: Failed to compute P(red wins)", file=sys.stderr)
        sys.exit(1)

    # 3. Compute model-derived MoV + round-of-finish probabilities
    model_probs = prop_bet_model_probabilities(
        p_red_wins=p_red,
        red_name=args.red,
        blue_name=args.blue,
        weight_class=args.wc,
        scheduled_rounds=args.rounds,
        fighter_db=fighter_db,
        wc_avg=wc_avg,
    )

    # 4. (Optional) Fetch DK props
    dk_props: list[dict] = []
    if args.dk_url and not args.model_only:
        try:
            scraper = DKPropsScraper()
            dk_props = scraper.get_event_props(args.dk_url)
            print(f"  Fetched {len(dk_props)} DK prop outcomes from {args.dk_url}")
        except Exception as e:
            warnings.warn(f"DK scrape failed: {e}")
            dk_props = []

    # 5. (Optional) Fetch The Odds API moneyline
    if args.odds_api_event_id and not args.model_only and os.environ.get("ODDS_API_KEY"):
        try:
            client = OddsAPIClient()
            odds = client.get_ufc_odds(args.odds_api_event_id)
            print(f"  Fetched moneyline from {len(odds)} bookmakers via The Odds API")
        except OddsAPIError as e:
            warnings.warn(f"Odds API fetch failed: {e}")

    # 6. Compute edges (if we have DK props)
    if dk_props:
        edges = compute_prop_bet_edges(model_probs, dk_props, args.red, args.blue)
        ranked = rank_prop_bets(edges, min_edge=args.min_edge)
    else:
        ranked = []

    # 7. Output
    print_ranked_bets(ranked, model_probs)
    if args.csv:
        write_csv(ranked, args.csv)
        print(f"\n  Wrote {len(ranked)} ranked bets to {args.csv}")

    if args.model_only and not dk_props:
        # Model-only mode: print the raw MoV + round probabilities too
        print(f"\n  --- Model-derived probabilities (no market to compare) ---")
        for k, v in model_probs.items():
            if isinstance(v, float):
                print(f"    {k:25s} = {v:.4f}")


if __name__ == "__main__":
    main()
