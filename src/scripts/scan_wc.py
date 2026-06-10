import sys, json, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.data.kalshi import KalshiClient
from src.data.world_cup import (fetch_all_matches, compute_elo, get_elo_for_teams,
                                  get_known_elo_teams, WC2026_TEAMS, CONF_MAP, CONF_ELO,
                                  build_feature_vector)
from src.data.fotmob import (LineupCache, FotMobScraper, compute_key_player_out,
                              map_kalshi_to_fotmob_id)
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "worldcup"
CALIB_DIR = MODEL_DIR / "calibration"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Lineup cache + scraper (lazy init — scraper takes ~2s to spin up Chromium)
_lineup_cache: Optional[LineupCache] = None
_fotmob_scraper: Optional[FotMobScraper] = None


def _get_lineup_cache() -> LineupCache:
    global _lineup_cache
    if _lineup_cache is None:
        _lineup_cache = LineupCache()
    return _lineup_cache


def _get_fotmob_scraper() -> FotMobScraper:
    global _fotmob_scraper
    if _fotmob_scraper is None:
        _fotmob_scraper = FotMobScraper()
    return _fotmob_scraper


def _full_name_to_code(full_name: str) -> str:
    """Reverse of TICKER_TEAM_MAP: full country name -> 3-letter code (e.g. 'Argentina' -> 'ARG')."""
    for code, name in TICKER_TEAM_MAP.items():
        if name == full_name:
            return code
    return full_name


def _get_lineups_for_ticker(ticker: str) -> dict:
    """Fetch lineups for a match ticker, using cache first then scraping FotMob."""
    if not ticker:
        return {}
    cache = _get_lineup_cache()
    if cache.is_fresh(ticker):
        cached = cache.get(ticker)
        if cached:
            return cached
    fotmob_id = map_kalshi_to_fotmob_id(ticker)
    if fotmob_id is None:
        return {}
    try:
        scraper = _get_fotmob_scraper()
        lineups = scraper.fetch_lineups(fotmob_id)
    except Exception:
        return {}
    if lineups.get("status") == "ok" and (lineups.get("home") or lineups.get("away")):
        cache.set(ticker, lineups)
        return lineups
    return {}

# Trained ML model cache
_loaded_model = None
_loaded_meta = None

# Neutral-venue empirical offset (computed by backtest_wc.py).
# Cached at module load to avoid reading the file per-match.
_neutral_offset = None
_neutral_offset_loaded = False

def _load_neutral_offset():
    """Load the empirical offset from neutral_offset.json. Returns dict or empty."""
    global _neutral_offset, _neutral_offset_loaded
    if _neutral_offset_loaded:
        return _neutral_offset
    _neutral_offset_loaded = True
    offset_path = CALIB_DIR / "neutral_offset.json"
    if not offset_path.exists():
        _neutral_offset = {}
        return _neutral_offset
    try:
        with open(offset_path) as f:
            _neutral_offset = json.load(f)
        return _neutral_offset
    except Exception:
        _neutral_offset = {}
        return _neutral_offset

def _load_match_model():
    """Load trained match outcome model. Returns (model, meta) or (None, None)."""
    global _loaded_model, _loaded_meta
    if _loaded_model is not None:
        return _loaded_model, _loaded_meta
    
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    
    if not model_path.exists() or not meta_path.exists():
        return None, None
    
    try:
        import lightgbm as lgb
        _loaded_model = lgb.Booster(model_file=str(model_path))
        with open(meta_path) as f:
            _loaded_meta = json.load(f)
        return _loaded_model, _loaded_meta
    except Exception:
        return None, None


    # Team name normalization for Kalshi tickers
TICKER_TEAM_MAP = {
    "KOR": "Korea Republic", "CZE": "Czechia", "CAN": "Canada", "BIH": "Bosnia and Herzegovina",
    "SWE": "Sweden", "TUN": "Tunisia", "AUS": "Australia", "TUR": "Turkiye",
    "PAR": "Paraguay", "JPN": "Japan", "SEN": "Senegal", "IRQ": "Iraq",
    "NOR": "Norway", "FRA": "France", "COL": "Colombia", "POR": "Portugal",
    "COD": "Congo DR", "URU": "Uruguay", "ESP": "Spain", "EGY": "Egypt",
    "IRI": "Iran", "NZL": "New Zealand", "BEL": "Belgium", "CPV": "Cape Verde",
    "KSA": "Saudi Arabia", "NED": "Netherlands", "CUW": "Curacao", "CIV": "Ivory Coast",
    "ECU": "Ecuador", "GER": "Germany", "USA": "USA", "RSA": "South Africa",
    "SCO": "Scotland", "BRA": "Brazil", "MEX": "Mexico",
    "QAT": "Qatar", "SUI": "Switzerland",
    "ARG": "Argentina", "AUT": "Austria", "DZA": "Algeria", "CRO": "Croatia",
    "ENG": "England", "GHA": "Ghana", "PAN": "Panama", "JOR": "Jordan",
    "UZB": "Uzbekistan", "JAM": "Jamaica",
    "MAR": "Morocco", "NGA": "Nigeria", "CMR": "Cameroon", "MLI": "Mali",
    "BFA": "Burkina Faso", "GUI": "Guinea", "CRC": "Costa Rica",
    "HON": "Honduras", "SLV": "El Salvador", "FIJ": "Fiji",
    "UAE": "United Arab Emirates", "PRK": "Korea DPR",
    "VEN": "Venezuela", "PER": "Peru", "CHI": "Chile",
    "BOL": "Bolivia", "DEN": "Denmark", "SVN": "Slovenia", "GRC": "Greece",
    "NCL": "New Caledonia", "SUR": "Suriname", "HTI": "Haiti",
}


def elo_expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def shin_devig(prices_3way):
    p = np.array(prices_3way, dtype=float)
    lo, hi = 0.0, 0.999
    for _ in range(100):
        z = (lo + hi) / 2
        denom = 2 * (1 - z)
        if denom <= 0:
            break
        terms = np.sqrt(np.maximum(0, z**2 + 4 * (1 - z) * p))
        total = float(np.sum((terms - z) / denom))
        if total > 1:
            lo = z
        else:
            hi = z
        if hi - lo < 1e-10:
            break
    z = (lo + hi) / 2
    if z >= 0.999:
        return p / p.sum()
    denom = 2 * (1 - z)
    out = (np.sqrt(np.maximum(0, z**2 + 4 * (1 - z) * p)) - z) / denom
    return out / out.sum()


# Build form features from historical ELO data for ML model input
_team_form_cache = None

def _build_form_features(elo_df, elo_ratings):
    """Build Elo-adjusted form features for all teams from ELO data.

    Uses the same logic as build_feature_dataset() in training: each match's
    result is compared to its Elo-expected win probability, producing a
    "performance vs expectation" metric that is comparable across confederations.

    Returns dict: team -> {perf, opp_elo, gs, gc, n}
        perf     = avg(actual_points - elo_expected) over last 5 matches
        opp_elo  = average opponent Elo of last 5 opponents
    """
    global _team_form_cache
    if _team_form_cache is not None:
        return _team_form_cache

    from src.data.world_cup import _elo_expected

    form = {}
    for team in elo_ratings:
        # Get recent matches involving this team
        team_matches = elo_df[(elo_df["home_team"] == team) | (elo_df["away_team"] == team)]
        team_matches = team_matches.sort_values("match_date").tail(5)

        if team_matches.empty:
            form[team] = {"perf": 0.0, "opp_elo": elo_ratings.get(team, 1500),
                          "gs": 0.0, "gc": 0.0, "n": 0}
            continue

        perf_sum, opp_elo_sum, gs_sum, gc_sum = 0.0, 0.0, 0.0, 0.0
        k = len(team_matches)

        for _, r in team_matches.iterrows():
            is_home = r["home_team"] == team
            home_score = int(r["home_score"])
            away_score = int(r["away_score"])
            team_elo = r["elo_home_pre"] if is_home else r["elo_away_pre"]
            opp_elo = r["elo_away_pre"] if is_home else r["elo_home_pre"]

            # Actual points: 1 for win, 0.5 for draw, 0 for loss
            if home_score > away_score:
                actual = 1.0 if is_home else 0.0
            elif away_score > home_score:
                actual = 0.0 if is_home else 1.0
            else:
                actual = 0.5

            expected = _elo_expected(team_elo, opp_elo)
            perf_sum += actual - expected
            opp_elo_sum += opp_elo

            if is_home:
                gs_sum += home_score
                gc_sum += away_score
            else:
                gs_sum += away_score
                gc_sum += home_score

        form[team] = {
            "perf": perf_sum / k,
            "opp_elo": opp_elo_sum / k,
            "gs": gs_sum / k,
            "gc": gc_sum / k,
            "n": k,
        }

    _team_form_cache = form
    return form


def predict_match(home_team, away_team, elo_ratings, form_features=None,
                  ticker: str = None, lineups: dict = None):
    """Predict match outcome using trained ML model, falling back to Elo formula.
    Returns array of [p_home, p_draw, p_away].

    Parameters
    ----------
    home_team, away_team : str
        Full country name (e.g. "Argentina", "USA").
    elo_ratings : dict
        Mapping of country name to Elo rating.
    form_features : dict, optional
        Pre-computed form features per team.
    ticker : str, optional
        Kalshi WC ticker (e.g. "KXWCGAME-26JUN11MEXCUB-MEX"). Used to compute
        key_player_out via the LineupCache (or a FotMob scrape on miss).
    lineups : dict, optional
        Pre-fetched lineups {"home": [...], "away": [...]}. Overrides ticker.
    """
    elo_h = elo_ratings.get(home_team, CONF_ELO.get(CONF_MAP.get(home_team, ""), 1500))
    elo_a = elo_ratings.get(away_team, CONF_ELO.get(CONF_MAP.get(away_team, ""), 1500))

    # Compute key_player_out (per NotebookLM Q2d). Lineups published ~60 min
    # before kickoff; before that the feature stays 0 (treated as OOD by model).
    key_player_out = 0
    if lineups is None and ticker:
        lineups = _get_lineups_for_ticker(ticker)
    if lineups:
        try:
            key_player_out = compute_key_player_out(
                _full_name_to_code(home_team),
                _full_name_to_code(away_team),
                lineups,
            )
        except Exception:
            key_player_out = 0

    # Try trained ML model first
    model, meta = _load_match_model()
    if model is not None and meta is not None and form_features is not None:
        try:
            hf = form_features.get(home_team, {"perf": 0, "opp_elo": elo_h, "gs": 0, "gc": 0, "n": 0})
            af = form_features.get(away_team, {"perf": 0, "opp_elo": elo_a, "gs": 0, "gc": 0, "n": 0})

            # Build feature vector via shared utility (matches training data order)
            # Pass "WC" so is_neutral=1 — World Cup matches are at neutral venues
            features = meta.get("features", [])
            x = build_feature_vector(elo_h, elo_a, hf, af, "WC", features,
                                      key_player_out=key_player_out)
            probs = model.predict(x)[0]

            # === Apply neutral-venue empirical offset (per NotebookLM rec) ===
            # Δ_class = mean(P_model) - actual_rate on 2022 WC val, capped at ±0.15.
            # Closes the 67-70% home bias at neutral venues with a single
            # additive shift per class. Renormalized to sum to 1.
            offset = _load_neutral_offset()
            if offset.get("applied", False):
                cap = offset.get("cap", 0.15)
                dh = max(-cap, min(cap, offset.get("delta_home", 0.0)))
                dd = max(-cap, min(cap, offset.get("delta_draw", 0.0)))
                da = max(-cap, min(cap, offset.get("delta_away", 0.0)))
                probs[0] -= dh
                probs[1] -= dd
                probs[2] -= da
                probs = np.maximum(probs, 0.001)
                probs = probs / probs.sum()

            # Renormalize to sum to 1
            probs = probs / probs.sum()
            return probs
        except Exception:
            pass  # Fall through to Elo formula
    
    # Fallback: Elo-based formula
    eh = elo_expected(elo_h, elo_a)
    ea = 1.0 - eh
    
    diff = abs(elo_h - elo_a)
    draw_prob = max(0.12, 0.28 - diff * 0.0004)
    
    p_draw = draw_prob
    p_home = eh * (1 - draw_prob)
    p_away = ea * (1 - draw_prob)
    
    return np.array([p_home, p_draw, p_away])


MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


def extract_match(market):
    ticker = market["ticker"]
    title = market.get("title", "")

    m = re.match(r"KXWCGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})-([A-Z]+)", ticker)
    if not m:
        return None

    yr, mon_s, day_s, t1, t2, outcome = m.groups()
    year = 2000 + int(yr)
    month = MONTHS.get(mon_s, 6)
    day = int(day_s)

    home_team = TICKER_TEAM_MAP.get(t1, t1)
    away_team = TICKER_TEAM_MAP.get(t2, t2)

    if outcome == "TIE":
        outcome_type = "tie"
    elif outcome == t1:
        outcome_type = "home"
    elif outcome == t2:
        outcome_type = "away"
    else:
        return None

    return {
        "match_date": f"{year}-{month:02d}-{day:02d}",
        "match_key": f"{t1}_{t2}",
        "home_team": home_team,
        "away_team": away_team,
        "short_home": t1,
        "short_away": t2,
        "outcome_type": outcome_type,
        "ticker": ticker,
        "title": title,
    }


def get_wc_markets(kc):
    mkts = kc.list_markets(series_ticker="KXWCGAME", limit=500)
    if mkts is None or mkts.empty:
        return pd.DataFrame()
    return mkts


def scan(args=None):
    kc = KalshiClient()
    balance = kc.get_balance()

    # Compute latest Elo ratings
    df = fetch_all_matches()
    elo_df = compute_elo(df)
    known_elo_teams = get_known_elo_teams(elo_df)

    # Get Elo ratings for all WC2026 teams
    all_teams = list(WC2026_TEAMS)
    elo_ratings = get_elo_for_teams(elo_df, all_teams)

    print(f"\n{'='*80}")
    print(f"  WORLD CUP 2026 — MATCH SCANNER")
    print(f"  Balance: ${balance:.2f}  |  Teams with Elo: {sum(1 for t in all_teams if t in elo_ratings)}/{len(all_teams)}")
    print(f"{'='*80}")

    # Print top Elo ratings
    print(f"\n  Top 10 Elo ratings:")
    for t, e in sorted(elo_ratings.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t:20s} {e:.0f}")

    # Load ML model if available (for display)
    wc_model, wc_meta = _load_match_model()
    if wc_model is not None:
        print(f"  Trained ML match model loaded ({wc_meta.get('n_features', '?')} features, "
              f"Brier={wc_meta.get('test_brier', '?'):.4f})")
    else:
        print(f"  No trained ML model — using Elo formula")

    # Load neutral-venue empirical offset (per NotebookLM rec)
    offset = _load_neutral_offset()
    if offset.get("applied", False):
        print(f"  Neutral-venue offset APPLIED: "
              f"Δ_H={offset.get('delta_home', 0):+.3f}  "
              f"Δ_D={offset.get('delta_draw', 0):+.3f}  "
              f"Δ_A={offset.get('delta_away', 0):+.3f}  "
              f"(cap=±{offset.get('cap', 0.15)}, n_val={offset.get('n_val', 0)})")
    else:
        print(f"  Neutral-venue offset: NOT APPLIED (neutral_offset.json missing or applied=False)")
    
    # Build form features for ML model
    form_features = _build_form_features(elo_df, elo_ratings)

    # Get WC markets
    wc_mkts = get_wc_markets(kc)
    print(f"\n  KXWCGAME markets found: {len(wc_mkts)}")

    # Group markets by match
    matches = {}
    for _, m in wc_mkts.iterrows():
        info = extract_match(m)
        if info is None:
            continue
        key = info["match_key"]
        if key not in matches:
            matches[key] = {
                "match_date": info["match_date"],
                "home_team": info["home_team"],
                "away_team": info["away_team"],
                "short_home": info["short_home"],
                "short_away": info["short_away"],
            }
        try:
            yb = float(m.get("yes_bid_dollars", m.get("yes_bid", "0")))
            ya = float(m.get("yes_ask_dollars", m.get("yes_ask", "1")))
            mid = (yb + ya) / 2
        except (ValueError, TypeError):
            continue
        matches[key][f"{info['outcome_type']}_ticker"] = info["ticker"]
        matches[key][f"{info['outcome_type']}_mid"] = mid
        matches[key][f"{info['outcome_type']}_yb"] = yb
        matches[key][f"{info['outcome_type']}_ya"] = ya

    print(f"  Match groups: {len(matches)}")

    # Predict and evaluate
    qualifying = []
    match_list = list(matches.items())
    # Remove incomplete matches (missing all 3 outcomes)
    filtered = []
    for key, m in match_list:
        if all(f"{p}_mid" in m for p in ["home", "tie", "away"]):
            filtered.append((key, m))

    for key, m in sorted(filtered):
        home = m["home_team"]
        away = m["away_team"]

        # Skip non-World Cup matches (teams not in 2026 WC) or unknown Elo
        if home not in WC2026_TEAMS or away not in WC2026_TEAMS:
            continue
        if home not in known_elo_teams or away not in known_elo_teams:
            continue

        # Use the home-team ticker as the lineup cache key (one entry per match)
        match_ticker = m.get("home_ticker", "")
        model_probs = predict_match(home, away, elo_ratings, form_features,
                                    ticker=match_ticker)
        mkt_probs = np.array([m["home_mid"], m["tie_mid"], m["away_mid"]])
        fair_probs = shin_devig(mkt_probs)

        labels = ["home", "tie", "away"]
        outcomes = [f"{home} wins", "Draw", f"{away} wins"]

        for idx, label in enumerate(labels):
            model_p = model_probs[idx]
            mkt_p = mkt_probs[idx]
            fair_p = fair_probs[idx]

            if mkt_p <= 0 or fair_p <= 0:
                continue

            if mkt_p > 0.90 or mkt_p < 0.02:
                continue

            edge = model_p - fair_p
            edge_pct = (model_p - fair_p) / fair_p * 100

            if edge_pct > 15 and model_p > 0.15:
                ticker_key = f"{label}_ticker"
                if ticker_key not in m:
                    continue

                yb = m[f"{label}_yb"]
                ya = m[f"{label}_ya"]

                cost = float(ya)
                cnt = int(balance * 0.01 / cost) if cost > 0 else 0
                qualifying.append({
                    "match": f"{home} vs {away}",
                    "pick": outcomes[idx],
                    "outcome": label,
                    "home_team": home,
                    "away_team": away,
                    "model_p": float(model_p),
                    "mkt_p": float(mkt_p),
                    "fair_p": float(fair_p),
                    "edge_pct": float(edge_pct),
                    "yb": float(yb),
                    "ya": float(ya),
                    "contracts": cnt,
                    "ticker": m[ticker_key],
                })

    # Print results
    if qualifying:
        qualifying.sort(key=lambda x: -x["edge_pct"])

        print(f"\n  {'='*80}")
        print(f"  QUALIFYING BETS (edge > 15%, model_prob > 15%)")
        print(f"  {'='*80}")

        for q in qualifying:
            print(f"\n  {q['match']:40s}")
            print(f"  Pick: {q['pick']:30s}  Model: {q['model_p']:.1%}")
            print(f"  Market: {q['mkt_p']:.1%} (bid={q['yb']:.2f}/ask={q['ya']:.2f})")
            print(f"  Fair: {q['fair_p']:.1%}  Edge: {q['edge_pct']:.0f}%")

            cost_per_contract = q["ya"]
            max_contracts = int(balance * 0.01 / cost_per_contract) if cost_per_contract > 0 else 0
            if max_contracts > 0:
                print(f"  → BUY {max_contracts} contract(s) @ ${cost_per_contract:.2f} (1% of bankroll)")
            else:
                print(f"  → Insufficient bankroll (need ${cost_per_contract * 100:.2f})")
    else:
        print(f"\n  No qualifying bets found.")

    # Show all WC matches
    print(f"\n\n  {'='*80}")
    print(f"  ALL WORLD CUP MATCHES (references)")
    print(f"  {'='*80}")

    wc_matches = [(k, m) for k, m in sorted(matches.items())
                  if m.get("home_team") in WC2026_TEAMS and m.get("away_team") in WC2026_TEAMS
                  and m.get("home_team") in known_elo_teams and m.get("away_team") in known_elo_teams]
    for key, m in wc_matches:
        home = m["home_team"]
        away = m["away_team"]
        mkt_probs = np.array([m.get("home_mid", 0), m.get("tie_mid", 0), m.get("away_mid", 0)])
        match_ticker_ref = m.get("home_ticker", "")
        model_probs = predict_match(home, away, elo_ratings, form_features,
                                    ticker=match_ticker_ref)

        date_str = m.get("match_date", "TBD")
        print(f"\n  {home:20s} vs {away:20s}  [{date_str}]")
        print(f"  Model:    HW={model_probs[0]:.1%}  D={model_probs[1]:.1%}  AW={model_probs[2]:.1%}")
        print(f"  Market:   HW={mkt_probs[0]:.1%}  D={mkt_probs[1]:.1%}  AW={mkt_probs[2]:.1%}")

        for idx, label in enumerate(["home", "tie", "away"]):
            p = mkt_probs[idx]
            if p > 0:
                edge_pct = (model_probs[idx] / p - 1) * 100
                arrow = "↑" if edge_pct > 10 else ("↓" if edge_pct < -10 else "─")
                print(f"    {label:5s}: mkt={p:.0%} model={model_probs[idx]:.0%} edge={edge_pct:+.0f}% {arrow}")

    return qualifying


if __name__ == "__main__":
    scan()
