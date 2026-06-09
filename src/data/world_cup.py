import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import requests
import csv
import io
from pathlib import Path
from datetime import datetime, timedelta


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "worldcup"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ELO_BASE = "https://www.eloratings.net"

ELO_TEAM_URL = f"{ELO_BASE}/en.teams.tsv"
ELO_TOURNAMENT_URL = f"{ELO_BASE}/en.tournaments.tsv"

def _fetch_tsv(url):
    r = requests.get(url, timeout=15)
    lines = [l for l in r.text.strip().split("\n") if not l.startswith("#")]
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter="\t")
    data = [row for row in reader if row]
    return data

def _build_team_code_map():
    data = _fetch_tsv(ELO_TEAM_URL)
    raw_map = {row[0]: row[1] for row in data if len(row) >= 2}
    code_to_name = {}
    for code, raw in raw_map.items():
        name = raw.strip()
        n = name
        if n == "United States": n = "USA"
        elif n == "South Korea": n = "Korea Republic"
        elif n == "Turkey": n = "Turkiye"
        elif n == "Czech Republic": n = "Czechia"
        elif n == "Czechoslovakia": n = "Czechia"
        elif n == "Côte d'Ivoire": n = "Ivory Coast"
        elif n == "Cape Verde Islands": n = "Cape Verde"
        elif n == "IR Iran": n = "Iran"
        elif n == "Korea DPR": n = "Korea DPR"
        elif n == "DR Congo": n = "Congo DR"
        elif n == "Bosnia and Herzegovina": n = "Bosnia and Herzegovina"
        elif n == "Scotland": n = "Scotland"
        elif n == "Congo": n = "Congo DR"
        code_to_name[code] = n
    return code_to_name

TEAM_CODE_MAP = None

def _get_team_code_map():
    global TEAM_CODE_MAP
    if TEAM_CODE_MAP is None:
        TEAM_CODE_MAP = _build_team_code_map()
    return TEAM_CODE_MAP

def _code_to_name(code):
    return _get_team_code_map().get(code, code)

# Fetch from 2010 to get 12+ years of pre-2022 training data (was 2018–2021 only).
# eloratings.net has results back to the 1990s; 2010 gives rich historical
# coverage without going so far back that team identities are unrecognisable.
YEARS_TO_FETCH = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
                  2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

COLUMNS = ["year","month","day","team1","team2","score1","score2","tournament",
           "venue","elo_change1","elo1","elo2","elo_change2","rank_change1","rank_change2"]

def fetch_all_matches(force_refetch=False):
    cache_file = CACHE_DIR / "all_matches.parquet"
    if cache_file.exists() and not force_refetch:
        df = pd.read_parquet(cache_file)
        return df

    team_map = _get_team_code_map()
    all_rows = []
    for year in YEARS_TO_FETCH:
        url = f"{ELO_BASE}/{year}_results.tsv"
        try:
            data = _fetch_tsv(url)
        except Exception as e:
            continue
        for row in data:
            if len(row) < 15:
                continue
            try:
                y, m, d, t1, t2, s1, s2, tourn = row[0], int(row[1]), int(row[2]), row[3], row[4], row[5], row[6], row[7]
                e1, e2, ec1, ec2 = float(row[10]), float(row[11]), float(row[9]) if row[9] else 0, float(row[12]) if row[12] else 0
            except (ValueError, IndexError):
                continue
            home = team_map.get(t1, t1)
            away = team_map.get(t2, t2)
            try:
                hs, aws = int(s1), int(s2)
            except ValueError:
                continue
            all_rows.append({
                "match_date": f"{y}-{m:02d}-{d:02d}",
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": aws,
                "tournament_code": tourn,
                "elo_home_pre": e1,
                "elo_away_pre": e2,
                "elo_home_post": e1 + ec1,
                "elo_away_post": e2 + ec2,
                "source": "eloratings",
            })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df.to_parquet(cache_file)
    return df


def compute_elo(df, k_factor=24):
    if df.empty:
        return pd.DataFrame()
    if "elo_home_pre" in df.columns and "elo_home_post" in df.columns and "source" in df.columns:
        result = df.rename(columns={
            "elo_home_pre": "elo_home_pre",
            "elo_away_pre": "elo_away_pre",
            "elo_home_post": "elo_home_post",
            "elo_away_post": "elo_away_post",
        })
        return result[["match_date","home_team","away_team","home_score","away_score",
                        "elo_home_pre","elo_away_pre","elo_home_post","elo_away_post"]].copy()

    elo = {}
    history = []
    K = k_factor

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])

        if home not in elo:
            elo[home] = CONF_ELO.get(CONF_MAP.get(home, ""), 1500)
        if away not in elo:
            elo[away] = CONF_ELO.get(CONF_MAP.get(away, ""), 1500)

        eh = 1.0 / (1.0 + 10.0 ** ((elo[away] - elo[home]) / 400.0))
        ea = 1.0 - eh

        if hs > as_:
            sh, sa = 1.0, 0.0
        elif as_ > hs:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        elo[home] += K * (sh - eh)
        elo[away] += K * (sa - ea)

        history.append({
            "match_date": row["match_date"],
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "elo_home_pre": eh, "elo_away_pre": ea,
            "elo_home_post": elo[home], "elo_away_post": elo[away],
        })

    return pd.DataFrame(history)


CONF_ELO = {
    "UEFA": 1600, "CONMEBOL": 1580,
    "CONCACAF": 1420, "CAF": 1440, "AFC": 1380, "OFC": 1300,
}

CONF_MAP = {
    "USA":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Costa Rica":"CONCACAF","Jamaica":"CONCACAF",
    "Honduras":"CONCACAF","El Salvador":"CONCACAF","Suriname":"CONCACAF","Curacao":"CONCACAF",
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Uruguay":"CONMEBOL","Colombia":"CONMEBOL",
    "Ecuador":"CONMEBOL","Peru":"CONMEBOL","Paraguay":"CONMEBOL","Venezuela":"CONMEBOL",
    "Chile":"CONMEBOL","Bolivia":"CONMEBOL",
    "England":"UEFA","Spain":"UEFA","Germany":"UEFA","France":"UEFA","Portugal":"UEFA",
    "Netherlands":"UEFA","Belgium":"UEFA","Croatia":"UEFA","Italy":"UEFA","Switzerland":"UEFA",
    "Denmark":"UEFA","Austria":"UEFA","Turkiye":"UEFA","Sweden":"UEFA","Poland":"UEFA",
    "Czechia":"UEFA","Ukraine":"UEFA","Serbia":"UEFA","Norway":"UEFA","Scotland":"UEFA",
    "Slovakia":"UEFA","Romania":"UEFA","Hungary":"UEFA","Slovenia":"UEFA","Greece":"UEFA",
    "Bosnia and Herzegovina":"UEFA",
    "Japan":"AFC","Korea Republic":"AFC","Australia":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Uzbekistan":"AFC","Iraq":"AFC","Jordan":"AFC","Qatar":"AFC","United Arab Emirates":"AFC","Korea DPR":"AFC",
    "Senegal":"CAF","Morocco":"CAF","Nigeria":"CAF","Egypt":"CAF","Tunisia":"CAF",
    "Algeria":"CAF","Congo DR":"CAF","Ghana":"CAF","Cameroon":"CAF","South Africa":"CAF",
    "Mali":"CAF","Ivory Coast":"CAF","Burkina Faso":"CAF","Guinea":"CAF",
    "New Zealand":"OFC","Fiji":"OFC","New Caledonia":"OFC",
    "Cape Verde":"CAF",
}

WC2026_TEAMS = [
    "USA", "Mexico", "Canada",
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador",
    "Peru", "Paraguay", "Venezuela", "Chile", "Bolivia",
    "England", "Spain", "Germany", "France", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Italy", "Switzerland",
    "Denmark", "Austria", "Turkiye", "Sweden", "Poland",
    "Czechia", "Ukraine", "Serbia", "Norway", "Scotland",
    "Slovakia", "Romania", "Hungary", "Slovenia", "Greece",
    "Japan", "Korea Republic", "Australia", "Iran", "Saudi Arabia",
    "Uzbekistan", "Iraq", "Jordan", "Qatar", "United Arab Emirates",
    "Senegal", "Morocco", "Nigeria", "Egypt", "Tunisia",
    "Algeria", "Congo DR", "Ghana", "Cameroon", "South Africa",
    "Mali", "Ivory Coast", "Burkina Faso", "Guinea",
    "Panama", "Costa Rica", "Jamaica", "Honduras", "El Salvador",
    "New Zealand", "Fiji",
    "Suriname", "Cape Verde", "New Caledonia", "Curacao",
    "Korea DPR", "Bosnia and Herzegovina",
]

def _elo_expected(team_elo, opp_elo):
    """Elo expected win probability: 1/(1+10^((opp_elo - team_elo)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((opp_elo - team_elo) / 400.0))


def build_feature_dataset(elo_df):
    """Build feature dataset with Elo-adjusted form.

    Replaces raw win/draw rate with *performance vs Elo expectation*
    so that beating a minnow (expected) ≠ beating a giant (upset).
    Also adds average opponent Elo as context for interpreting goal stats.
    """
    if elo_df.empty:
        return pd.DataFrame()
    records = []
    team_cache = {}

    for _, row in elo_df.iterrows():
        date = row["match_date"]
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        elo_h = row["elo_home_pre"]
        elo_a = row["elo_away_pre"]

        team_cache.setdefault(home, [])
        team_cache.setdefault(away, [])

        def elo_adjusted_form(team, team_elo_pre_match, opp_elo_pre_match, before, n=5):
            """Return (perf, opp_elo, gs, gc, n) — Elo-adjusted form.

            perf = average of (actual_points - elo_expected) over last n matches.
                   Positive = outperforming Elo; negative = underperforming.
            opp_elo = average opponent Elo in last n (context for goal stats).
            """
            mlist = [m for m in reversed(team_cache.get(team, [])) if m["date"] < before][:n]
            if not mlist:
                # No history: use current opponent's Elo as best guess for "who they face"
                return 0.0, opp_elo_pre_match, 0.0, 0.0, 0

            perf_sum = 0.0
            opp_elo_sum = 0.0
            gs_sum = 0.0
            gc_sum = 0.0
            k = len(mlist)

            for m in mlist:
                actual = 1.0 if m["won"] else (0.5 if m["draw"] else 0.0)
                perf_sum += actual - m["elo_expected"]
                opp_elo_sum += m["opp_elo"]
                gs_sum += m["gs"]
                gc_sum += m["gc"]

            return perf_sum / k, opp_elo_sum / k, gs_sum / k, gc_sum / k, k

        # Home team's expected win prob vs this away opponent
        home_expected = _elo_expected(elo_h, elo_a)
        away_expected = _elo_expected(elo_a, elo_h)

        h_perf, h_opp_elo, hgs, hgc, hn = elo_adjusted_form(home, elo_h, elo_a, date)
        a_perf, a_opp_elo, ags, agc, an = elo_adjusted_form(away, elo_a, elo_h, date)

        hw, aw, d_ = (1, 0, 0) if hs > as_ else ((0, 1, 0) if as_ > hs else (0, 0, 1))

        records.append({
            "match_date": date,
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "elo_home": elo_h,
            "elo_away": elo_a,
            "elo_diff": elo_h - elo_a,
            "h_perf": h_perf, "h_opp_elo": h_opp_elo, "h_gs": hgs, "h_gc": hgc, "h_n": hn,
            "a_perf": a_perf, "a_opp_elo": a_opp_elo, "a_gs": ags, "a_gc": agc, "a_n": an,
            "home_won": hw, "draw": d_, "away_won": aw,
        })

        # Store match in cache with opponent Elo + expected for future form lookups
        team_cache[home].append({
            "date": date, "won": hw, "draw": d_, "gs": hs, "gc": as_,
            "opp_elo": elo_a, "elo_expected": home_expected,
        })
        team_cache[away].append({
            "date": date, "won": aw, "draw": d_, "gs": as_, "gc": hs,
            "opp_elo": elo_h, "elo_expected": away_expected,
        })

    return pd.DataFrame(records)


def get_elo_for_teams(elo_df, target_teams, as_of_date=None):
    latest = {}
    if as_of_date is None and not elo_df.empty:
        as_of_date = elo_df["match_date"].max()

    for _, row in elo_df.iterrows():
        if as_of_date and row["match_date"] > pd.Timestamp(as_of_date):
            continue
        latest[row["home_team"]] = row["elo_home_post"]
        latest[row["away_team"]] = row["elo_away_post"]

    result = {}
    for team in target_teams:
        if team in latest:
            result[team] = latest[team]
        else:
            result[team] = CONF_ELO.get(CONF_MAP.get(team, ""), 1500)
    return result


def get_known_elo_teams(elo_df):
    teams = set()
    for _, row in elo_df.iterrows():
        teams.add(row["home_team"])
        teams.add(row["away_team"])
    return teams


def build_feature_vector(elo_home, elo_away, hf, af, tournament_code, features):
    """Build a model-ready feature vector from Elo ratings and form dictionaries.

    Shared by backtest_wc.py and scan_wc.py.  The *features* list must match
    the ordered column names the model was trained on (stored in the model
    metadata file, e.g. ``wc_match_outcome.meta.json``).

    Parameters
    ----------
    elo_home : float
        Home team's Elo rating (raw, e.g. 1850).
    elo_away : float
        Away team's Elo rating.
    hf : dict
        Home team recent form: ``{"perf", "opp_elo", "gs", "gc", "n"}``.
        (Elo-adjusted: perf = avg actual - expected; opp_elo = avg opponent Elo)
    af : dict
        Away team recent form.
    tournament_code : str | None
        Tournament code for the match (e.g. ``"WC"``, ``"FR"`` for friendly).
    features : list[str]
        Ordered list of feature names the model expects.

    Returns
    -------
    np.ndarray
        Feature vector shaped ``(1, len(features))`` ready for ``model.predict()``.
    """
    elo_diff = elo_home - elo_away
    tc = str(tournament_code or "").upper()
    is_friendly = 1 if "FR" in tc else 0

    vec = {}
    for c in features:
        if c == "elo_home":
            vec[c] = elo_home
        elif c == "elo_away":
            vec[c] = elo_away
        elif c == "elo_diff":
            vec[c] = elo_diff
        elif c == "elo_diff_abs":
            vec[c] = abs(elo_diff)
        elif c in ("h_perf", "h_opp_elo", "h_gs", "h_gc", "h_n"):
            vec[c] = hf.get(c[2:], 0)
        elif c in ("a_perf", "a_opp_elo", "a_gs", "a_gc", "a_n"):
            vec[c] = af.get(c[2:], 0)
        elif c == "h_goal_diff":
            vec[c] = hf.get("gs", 0) - hf.get("gc", 0)
        elif c == "a_goal_diff":
            vec[c] = af.get("gs", 0) - af.get("gc", 0)
        elif c == "is_friendly":
            vec[c] = is_friendly
        else:
            # Backward compat: old models may have h_wr, h_dr, etc.
            if c in ("h_wr", "h_dr"):
                vec[c] = 0  # deprecated, replaced by h_perf
            elif c in ("a_wr", "a_dr"):
                vec[c] = 0
            else:
                vec[c] = 0

    return np.array([vec.get(c, 0) for c in features], dtype=float).reshape(1, -1)

class WorldCupDataSource:
    """Wrapper class for World Cup ELO data source.
    Each match is split into two rows (home_team, away_team) so
    player_id = team name and each team has many historical games.
    """
    
    def fetch_player_game_logs(self, seasons: list[str]) -> pd.DataFrame:
        from src.data.world_cup import fetch_all_matches
        df = fetch_all_matches()
        if df.empty:
            return df
        
        # Split each match into home and away rows
        home_rows = df.copy()
        home_rows["player_id"] = home_rows["home_team"]
        home_rows["goals_for"] = home_rows["home_score"]
        home_rows["goals_against"] = home_rows["away_score"]
        home_rows["is_home"] = 1
        home_rows["opponent"] = home_rows["away_team"]
        home_rows["elo_pre"] = home_rows["elo_home_pre"]
        home_rows["opponent_elo_pre"] = home_rows["elo_away_pre"]
        home_rows["total_goals"] = home_rows["home_score"] + home_rows["away_score"]
        home_rows["game_date"] = home_rows["match_date"]
        home_rows["season"] = home_rows["match_date"].dt.year.astype(str)
        
        away_rows = df.copy()
        away_rows["player_id"] = away_rows["away_team"]
        away_rows["goals_for"] = away_rows["away_score"]
        away_rows["goals_against"] = away_rows["home_score"]
        away_rows["is_home"] = 0
        away_rows["opponent"] = away_rows["home_team"]
        away_rows["elo_pre"] = away_rows["elo_away_pre"]
        away_rows["opponent_elo_pre"] = away_rows["elo_home_pre"]
        away_rows["total_goals"] = away_rows["home_score"] + away_rows["away_score"]
        away_rows["game_date"] = away_rows["match_date"]
        away_rows["season"] = away_rows["match_date"].dt.year.astype(str)
        
        result = pd.concat([home_rows, away_rows], ignore_index=True)
        result = result.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        
        print(f"  World Cup: {len(result)} team-game rows from {len(df)} matches, {result['player_id'].nunique()} teams")
        return result
    
    def fetch_schedule(self, season: str) -> pd.DataFrame:
        return self.fetch_player_game_logs([season])
    
    def fetch_player_stats(self, player_id: str, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()
    
    def fetch_team_stats(self, team_id, start_date, end_date) -> pd.DataFrame:
        return pd.DataFrame()
