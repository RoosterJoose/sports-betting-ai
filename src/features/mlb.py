import numpy as np
import pandas as pd
from pathlib import Path

from src.features.base import FeatureEngineer

# Statcast directory for handedness lookups
STATCAST_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "mlb" / "statcast"

# ── Handedness caches (loaded once per process) ──
_PITCHER_HAND_CACHE: dict[str, str] = {}  # player_id -> "R" or "L"
_BATTER_HAND_CACHE: dict[str, str] = {}   # player_id -> "R" or "L"
_HANDEDNESS_LOADED = False


def _load_handedness():
    """Load pitcher and batter handedness from Statcast parquet files.
    
    Cached globally so multiple feature builds don't re-read the files.
    Returns (pitcher_hand_map, batter_hand_map) where maps are player_id -> "R"|"L".
    """
    global _PITCHER_HAND_CACHE, _BATTER_HAND_CACHE, _HANDEDNESS_LOADED
    if _HANDEDNESS_LOADED:
        return _PITCHER_HAND_CACHE, _BATTER_HAND_CACHE
    
    _HANDEDNESS_LOADED = True
    files = sorted(STATCAST_DIR.glob("statcast_202*.parquet"))
    if not files:
        return {}, {}
    
    for fpath in files:
        try:
            df = pd.read_parquet(fpath, columns=["pitcher", "p_throws", "batter", "stand"])
        except Exception:
            continue
        if df.empty:
            continue
        pit_hand = df.groupby("pitcher")["p_throws"].first().to_dict()
        bat_hand = df.groupby("batter")["stand"].first().to_dict()
        _PITCHER_HAND_CACHE.update({str(k): v for k, v in pit_hand.items() if pd.notna(v)})
        _BATTER_HAND_CACHE.update({str(k): v for k, v in bat_hand.items() if pd.notna(v)})
    
    return _PITCHER_HAND_CACHE, _BATTER_HAND_CACHE

# Statcast park factors for K/9 (higher = more pitcher-friendly = more Ks)
PARK_FACTOR_K = {
    "SD": 1.08,  "SEA": 1.06, "NYM": 1.04, "MIA": 1.03, "CLE": 1.02,
    "OAK": 1.02, "TB":  1.01, "SF":  1.01, "WSH": 1.00, "DET": 1.00,
    "MIL": 0.99, "BAL": 0.99, "KC":  0.99, "MIN": 0.99, "PIT": 0.99,
    "LAA": 0.98, "PHI": 0.98, "CIN": 0.98, "ATL": 0.97, "CHC": 0.97,
    "TEX": 0.97, "BOS": 0.97, "TOR": 0.96, "HOU": 0.96, "STL": 0.96,
    "ARI": 0.95, "NYY": 0.95, "LAD": 0.94, "CWS": 0.93, "COL": 0.88,
}

# Park factor for HR (higher = more HR-friendly)
# Based on 3-year Statcast HR park factor data
PARK_FACTOR_HR = {
    "COL": 1.32, "CIN": 1.16, "BAL": 1.14, "BOS": 1.13, "NYY": 1.11,
    "MIL": 1.10, "HOU": 1.09, "LAD": 1.08, "PHI": 1.07, "MIN": 1.06,
    "TEX": 1.05, "WSH": 1.04, "ARI": 1.03, "CHC": 1.02, "DET": 1.02,
    "CLE": 1.01, "ATL": 1.00, "STL": 1.00, "SD":  0.99, "SEA": 0.98,
    "MIA": 0.97, "LAA": 0.97, "OAK": 0.96, "PIT": 0.96, "NYM": 0.95,
    "CWS": 0.94, "KC":  0.94, "TB":  0.93, "TOR": 0.93, "SF":  0.92,
}

# Park factor for TB (total bases)
PARK_FACTOR_TB = {
    "COL": 1.25, "BOS": 1.15, "CIN": 1.14, "BAL": 1.13, "NYY": 1.12,
    "MIL": 1.10, "HOU": 1.09, "LAD": 1.08, "PHI": 1.07, "MIN": 1.06,
    "TEX": 1.05, "WSH": 1.04, "ARI": 1.03, "CHC": 1.02, "DET": 1.01,
    "CLE": 1.01, "ATL": 1.00, "STL": 1.00, "SD":  0.99, "SEA": 0.99,
    "MIA": 0.98, "LAA": 0.98, "OAK": 0.97, "PIT": 0.97, "NYM": 0.96,
    "CWS": 0.96, "KC":  0.95, "TB":  0.94, "TOR": 0.95, "SF":  0.93,
}

TEAM_IDS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


class MLBFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()
        if df.empty:
            return df

        if "game_date" not in df.columns or df["game_date"].dtype == "object":
            df["game_date"] = pd.to_datetime(df["game_date"])

        # Team abbreviation mapping for park factors
        df["team_abbr"] = df["team_id"].astype(int).map(TEAM_IDS).fillna("UNK")
        df["opp_abbr"] = df.get("opponent_id", df["team_id"]).fillna(-1).astype(int).map(TEAM_IDS).fillna("UNK")

        # ── Opponent quality: lineup K% (temporally correct) ────────────
        # Research: lineup K% varies 15-25% across teams. Sportsbooks price
        # off pitcher season averages, ignoring the specific lineup's K rate.
        # Fix: compute cumulative team batting K% from games BEFORE each row,
        #       so early-season games don't see future data (no leakage).
        #       Only meaningful for pitchers; hitters get NaN.
        if all(c in df.columns for c in ["so", "ab", "position", "opponent_id", "team_id"]):
            # Sort by date (already sorted above, but be explicit)
            df = df.sort_values(["game_date", "player_id"]).reset_index(drop=True)
            # Cumulative team batting K%: expanding window per team, shifted
            cum_so = {}
            cum_ab = {}
            opp_k_vals = []
            for idx, row in df.iterrows():
                opp_id = str(row["opponent_id"])
                if opp_id in cum_so and cum_ab.get(opp_id, 0) > 0:
                    k_pct = cum_so[opp_id] / cum_ab[opp_id]
                else:
                    k_pct = 0.22  # MLB average batting K%
                opp_k_vals.append(k_pct)
                # Update cumulative stats from this row (hitters only)
                if str(row.get("position", "")) != "P":
                    tid = str(row["team_id"])
                    so = float(row.get("so", 0) or 0)
                    ab = float(row.get("ab", 0) or 0)
                    cum_so[tid] = cum_so.get(tid, 0) + so
                    cum_ab[tid] = cum_ab.get(tid, 0) + ab
            df["opp_k_pct"] = opp_k_vals
            # Only pitchers face opponent batting K%; hitters get 0
            is_pitcher = df.get("position", "").astype(str).isin(["P", "SP", "RP", "CP"])
            df.loc[~is_pitcher, "opp_k_pct"] = 0.0
        else:
            df["opp_k_pct"] = 0.0

        # ── Platoon handedness ─────────────────────────────────────────
        # Research: LHP vs RHP measurably affects K%, HR%, and TB allowed.
        # Extreme platoon situations (6+ same-side batters vs starter)
        # are underpriced by sportsbooks.
        # NOTE: opponent_id in game logs is a TEAM ID, not a player ID,
        # so per-game platoon matchup requires team-level LHB% aggregation
        # (future). For now: player_is_lefty captures the individual
        # handedness advantage, which is predictive on its own.
        if "player_id" in df.columns and "position" in df.columns:
            pitcher_hand, batter_hand = _load_handedness()
            if pitcher_hand or batter_hand:
                # Map individual player handedness (player_id IS a player ID)
                df["player_hand"] = df["player_id"].astype(str).map(
                    lambda pid: pitcher_hand.get(pid, batter_hand.get(pid, ""))
                )
                df["player_is_lefty"] = (df["player_hand"] == "L").astype(int)
                df.drop(columns=["player_hand"], inplace=True)
                n_known = (df["player_is_lefty"].notna()).sum()
                if n_known > 0:
                    pct_l = df["player_is_lefty"].mean() * 100
                    print(f"  Player handedness: {n_known}/{len(df)} rows, {pct_l:.1f}% lefty", flush=True)
            else:
                df["player_is_lefty"] = 0
        else:
            df["player_is_lefty"] = 0

        # Park factors: home team's park (pre-game knowledge, no leakage)
        park_k = []
        park_hr = []
        park_tb = []
        for _, row in df.iterrows():
            home_abbr = row["team_abbr"] if row.get("home_or_away", "A") == "H" else row["opp_abbr"]
            park_k.append(PARK_FACTOR_K.get(home_abbr, 1.0))
            park_hr.append(PARK_FACTOR_HR.get(home_abbr, 1.0))
            park_tb.append(PARK_FACTOR_TB.get(home_abbr, 1.0))
        df["park_factor_k"] = park_k
        df["park_factor_hr"] = park_hr
        df["park_factor_tb"] = park_tb

        # Split and process separately
        hitters = df[df["position"] != "P"].copy() if "position" in df.columns else df.copy()
        pitchers = df[df["position"] == "P"].copy() if "position" in df.columns else pd.DataFrame()

        if not hitters.empty:
            hitters["tb"] = hitters.get("1b", 0) + 2 * hitters.get("2b", 0) + 3 * hitters.get("3b", 0) + 4 * hitters.get("hr", 0)
            hitters["woba"] = (0.69 * hitters.get("bb", 0) + 0.888 * hitters.get("1b", 0) + 1.271 * hitters.get("2b", 0) + 1.616 * hitters.get("3b", 0) + 2.101 * hitters.get("hr", 0)) / (
                hitters["ab"] + hitters.get("bb", 0) + hitters.get("sf", 0) + hitters.get("hbp", 0)
            ).replace(0, 1)
            
            # Cross-stat features
            ab_safe = hitters["ab"].replace(0, 1)
            hitters["iso"] = hitters.get("slg", hitters["tb"] / ab_safe) - hitters.get("avg", hitters["h"] / ab_safe)
            hitters["contact_rate"] = (hitters["ab"] - hitters["so"]) / ab_safe
            hitters["bb_rate"] = hitters["bb"] / ab_safe
            hitters["hr_per_h"] = hitters["hr"] / hitters["h"].replace(0, 1)
            hitters["xbh_per_h"] = (hitters.get("2b", 0) + hitters.get("3b", 0) + hitters["hr"]) / hitters["h"].replace(0, 1)
            
            hitter_cols = ["h", "ab", "r", "rbi", "bb", "so", "2b", "3b", "hr", "sb", "tb", "woba",
                          "iso", "contact_rate", "bb_rate", "hr_per_h", "xbh_per_h"]
            available = [c for c in hitter_cols if c in hitters.columns]
            hitters = self.rolling_averages(hitters, available)
            hitters = self.rolling_medians(hitters, available)
            hitters = self.recency_weighted_avg(hitters, available)
            hitters = self.streak_features(hitters, available)
            hitters = self.consistency_features(hitters, available)
            hitters = self.schedule_density(hitters)

        if not pitchers.empty:
            ip_clip = pitchers.get("ip", 1).clip(lower=0.1)
            pitchers["k_9"] = pitchers["so"] / ip_clip * 9
            pitchers["bb_9"] = pitchers["bb"] / ip_clip * 9
            pitchers["hr_9"] = pitchers["hr"] / ip_clip * 9
            pitchers["whip"] = (pitchers["bb"] + pitchers["h"]) / ip_clip
            pitchers["fip_9"] = np.where(
                ip_clip > 0,
                (13 * pitchers["hr"] + 3 * pitchers["bb"] - 2 * pitchers["so"]) / ip_clip + 3.10,
                5.00
            )
            pitchers["k_rate"] = pitchers["so"] / pitchers["bf"].replace(0, 1)
            pitchers["babip"] = (pitchers["h"] - pitchers["hr"]) / (pitchers["bf"] - pitchers["so"] - pitchers["hr"] - pitchers.get("bb", 0)).replace(0, 1)
            pitchers["gb_rate"] = 1.0  # placeholder - would need batted ball data
            
            pitcher_cols = ["ip", "er", "so", "bb", "h", "hr", "k_9", "bb_9", "hr_9", "whip", "fip_9", "k_rate", "babip"]
            available = [c for c in pitcher_cols if c in pitchers.columns]
            pitchers = self.rolling_averages(pitchers, available)
            pitchers = self.rolling_medians(pitchers, available)
            pitchers = self.recency_weighted_avg(pitchers, available)
            pitchers = self.streak_features(pitchers, available)
            pitchers = self.consistency_features(pitchers, available)
            pitchers = self.schedule_density(pitchers)

        df = pd.concat([hitters, pitchers], ignore_index=True)
        df = df.sort_values(["game_date", "player_id"]).reset_index(drop=True)

        keep_cols = ["player_id", "game_date", "game_pk", "position", "team_id", "player_name", "team_abbr", "gs",
                     "home_or_away", "park_factor_k", "park_factor_hr", "park_factor_tb",
                     "opp_k_pct", "player_is_lefty"]
        for c in df.columns:
            if any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            elif any(c.endswith(f"_med_{w}") for w in self.windows):
                keep_cols.append(c)
            elif c.endswith("_ewm"):
                keep_cols.append(c)
            elif c.endswith(("_streak", "_consistency")):
                keep_cols.append(c)
            elif c in ("days_rest", "b2b", "four_in_six"):
                keep_cols.append(c)

        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
