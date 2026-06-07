import numpy as np
import pandas as pd

from src.data.pipeline import MODEL_DIR
from src.models.trainer import ModelTrainer

# PrizePicks stat_type -> (model_name, position_filter, column_name, is_team)
# is_team=True means team aggregate (falls back to team avg when no player match)
# is_team=False means individual player prop (returns None when no player match)
STAT_MAP = {
    # Hitter stats with individual classifier models
    "Hits":                  ("H",  "hitter", "h",  False),
    "Home Runs":             ("HR", "hitter", "hr", False),
    "RBIs":                  ("RBI","hitter", "rbi",False),
    "Stolen Bases":          ("SB", "hitter", "sb", False),
    "Walks":                 ("BB", "hitter", "bb", False),
    "Hitter Strikeouts":     ("SO", "hitter", "so", False),
    "Total Bases":           ("TB", "hitter", "tb", False),
    # Hitter stats without individual models (falls to team average when matched)
    "Singles":               ("1b", "hitter", "1b", True),
    "Doubles":               ("2b", "hitter", "2b", True),
    "Runs":                  ("R",  "hitter", "r",  True),
    "Hits+Runs+RBIs":        ("HRR","hitter", "h_r_rbi", True),
    # Pitcher stats with individual classifier models
    "Earned Runs Allowed":   ("ER",  "pitcher", "er", False),
    "Pitcher Strikeouts":    ("SO",  "pitcher", "so", False),
    "Hits Allowed":          ("H",   "pitcher", "h",  False),
    "Walks Allowed":         ("BB",  "pitcher", "bb", False),
    "Pitching Outs":         ("IP",  "pitcher", "ip", False),
}

TEAM_ABBR_TO_ID = {
    "ARI": 109, "AZ": 109,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,
    "CHC": 112,
    "CWS": 145, "CHW": 145,
    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,
    "KC": 118, "KCR": 118,
    "LAA": 108,
    "LAD": 119,
    "MIA": 146,
    "MIL": 158,
    "MIN": 142,
    "NYM": 121,
    "NYY": 147,
    "OAK": 133,
    "PHI": 143,
    "PIT": 134,
    "SD": 135, "SDP": 135,
    "SEA": 136,
    "SF": 137, "SFG": 137,
    "STL": 138,
    "TB": 139, "TBR": 139,
    "TEX": 140,
    "TOR": 141,
    "WSH": 120, "WSN": 120,
}


def compute_fantasy_score(df):
    return (df.get("1b", 0) * 1 + df.get("2b", 0) * 2 +
            df.get("3b", 0) * 3 + df.get("hr", 0) * 4 +
            df.get("rbi", 0) * 1 + df.get("r", 0) * 1 +
            df.get("bb", 0) * 1 + df.get("sb", 0) * 2 +
            df.get("hbp", 0) * 1)


def match_player(name, player_df):
    """Match PrizePicks player name to player in feature DataFrame."""
    if not name or player_df is None or player_df.empty:
        return None
    name_clean = name.strip()
    parts = name_clean.split()
    if len(parts) < 2:
        return None
    last = parts[-1]

    exact = player_df[player_df["player_name"].str.lower() == name_clean.lower()]
    if len(exact) == 1:
        return exact.iloc[0]

    last_end = player_df[player_df["player_name"].str.lower().str.endswith(last.lower(), na=False)]
    if len(last_end) == 1:
        return last_end.iloc[0]

    last_any = player_df[player_df["player_name"].str.lower().str.contains(last.lower(), na=False)]
    if not last_any.empty:
        if len(last_any) == 1:
            return last_any.iloc[0]
        first_initial = parts[0][0].lower()
        fi = last_any[last_any["player_name"].str.lower().str[0] == first_initial]
        if len(fi) == 1:
            return fi.iloc[0]
        return last_any.iloc[0]

    return None


class MLBLinePredictor:
    def __init__(self, sport_cfg):
        self.cfg = sport_cfg
        self._featured = None
        self._models = {}
        self._latest_features = None
        self._team_data = None

    def load_data(self, pipeline=None):
        """Load cached MLB features directly from parquet files."""
        cache_dir = MODEL_DIR.parent / "data" / "cache" / "mlb"
        cache_files = sorted(cache_dir.glob("game_logs_*.parquet"))
        if not cache_files:
            if pipeline is not None:
                pipeline.seasons = 2
                result = pipeline.build_training_data("H")
                if result is None:
                    return False
                featured = pipeline._cached_featured
            else:
                return False
        else:
            from src.features.mlb import MLBFeatureEngineer
            featured = MLBFeatureEngineer(self.cfg).build_features(
                pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
            )

        if featured is None or featured.empty:
            return False

        self._featured = featured
        self._latest_features = (
            featured.sort_values("game_date")
            .groupby("player_id")
            .last()
            .reset_index()
        )
        # Precompute team aggregates
        self._build_team_aggregates()
        return True

    def _build_team_aggregates(self):
        """Precompute team-level rolling averages for fallback predictions."""
        df = self._featured
        if df is None:
            return
        hitters = df[df["position"] != "P"].copy()
        hitters["1b"] = (hitters.get("h", 0) - hitters.get("2b", 0) -
                         hitters.get("3b", 0) - hitters.get("hr", 0))
        hitters["tb"] = (hitters.get("1b", 0) + 2 * hitters.get("2b", 0) +
                         3 * hitters.get("3b", 0) + 4 * hitters.get("hr", 0))
        hitters["h_r_rbi"] = hitters.get("h", 0) + hitters.get("r", 0) + hitters.get("rbi", 0)
        hitters["fantasy_score"] = compute_fantasy_score(hitters)

        team_cols = ["h", "hr", "rbi", "sb", "bb", "so", "r", "tb", "1b", "2b", "h_r_rbi", "fantasy_score"]
        avail = [c for c in team_cols if c in hitters.columns]
        game_id_col = "game_pk" if "game_pk" in hitters.columns else "game_id"
        team_games = hitters.groupby(["team_id", game_id_col, "game_date"])[avail].sum().reset_index()
        team_games = team_games.sort_values(["team_id", "game_date"])

        for col in avail:
            team_games[f"{col}_avg"] = team_games.groupby("team_id")[col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            team_games[f"{col}_std"] = team_games.groupby("team_id")[col].transform(
                lambda x: x.shift(1).expanding().std()
            )
        self._team_data = team_games

    def _team_avg_for_stat(self, team_id, raw_col):
        if self._team_data is None:
            return None, None
        team = self._team_data[self._team_data["team_id"] == str(team_id)].sort_values("game_date")
        if team.empty:
            return None, None
        avg = team[f"{raw_col}_avg"].iloc[-1]
        std = team[f"{raw_col}_std"].iloc[-1]
        if pd.isna(avg):
            return None, None
        return avg, (std if not pd.isna(std) else None)

    def _load_model(self, model_name):
        if model_name not in self._models:
            model_path = MODEL_DIR / "mlb" / f"{model_name}.json"
            if not model_path.exists():
                return None
            trainer = ModelTrainer(model_dir=MODEL_DIR, sport="mlb", stat_type=model_name)
            model = trainer.load()
            if model is None:
                return None
            self._models[model_name] = model
        return self._models[model_name]

    def _player_std(self, player_id, raw_col):
        """Compute standard deviation of a stat for a player from raw data."""
        df = self._featured
        if df is None:
            return None
        pdata = df[df["player_id"] == player_id].sort_values("game_date")
        if pdata.empty or raw_col not in pdata.columns:
            return None
        vals = pdata[raw_col].dropna().tail(20)
        if len(vals) < 3:
            return None
        return float(vals.std())

    def predict_line(self, line: dict):
        pp_stat = line.get("stat_type", "")
        line_val = line.get("line_score")
        player_name = line.get("player_name", "")
        team_code = line.get("description", "")

        try:
            line_val = float(line_val) if line_val is not None else None
        except (ValueError, TypeError):
            return None

        if not pp_stat or line_val is None:
            return None

        mapping = STAT_MAP.get(pp_stat)
        if mapping is None:
            return None

        model_name, pos_filter, raw_col, is_team = mapping

        # Try individual player model prediction first
        if self._latest_features is not None and player_name:
            # Filter by position before matching
            search_df = self._latest_features
            if pos_filter == "hitter":
                search_df = self._latest_features[self._latest_features.get("position", "") != "P"]
            elif pos_filter == "pitcher":
                search_df = self._latest_features[self._latest_features.get("position", "") == "P"]
            if search_df.empty:
                search_df = self._latest_features
            row = match_player(player_name, search_df)
            if row is not None:
                model = self._load_model(model_name)
                if model is not None:
                    try:
                        model_features = model.get_booster().feature_names
                        feat_dict = row.to_dict()
                        available = {c: feat_dict.get(c, 0) for c in model_features}
                        x = pd.DataFrame([available]).fillna(0)
                        model_prob = model.predict_proba(x)[0, 1]

                        avg = row.get(f"{raw_col}_avg_14", row.get(f"{raw_col}_avg_7", None))
                        pid = row.get("player_id")
                        std = self._player_std(pid, raw_col) if avg else None

                        if avg is not None:
                            avg_f = float(avg)
                            std_f = float(std) if std is not None and std > 0 else None

                            if std_f and std_f > 0:
                                dist = abs(line_val - avg_f) / std_f
                                if dist > 2.5:
                                    return None
                                if dist > 0.5:
                                    from scipy.stats import norm
                                    z = (line_val - avg_f) / std_f
                                    p_normal = 1.0 - norm.cdf(z)
                                    blend = min(dist / 2.0, 0.67)
                                    prob = model_prob * (1.0 - blend) + p_normal * blend
                                else:
                                    prob = model_prob
                            else:
                                prob = model_prob

                            prob = max(0.01, min(0.99, prob))

                            return {
                                "team": team_code,
                                "player": row.get("player_name", player_name),
                                "stat_type": pp_stat,
                                "line": line_val,
                                "avg": round(avg_f, 1),
                                "std": round(std_f, 1) if std_f else None,
                                "model_prob": round(prob, 4),
                                "effective_prob": round(prob, 4),
                                "direction": "over",
                                "is_model": True,
                            }
                    except Exception:
                        pass

        # Team-level fallback: only use for actual team-level stats (is_team=True)
        if is_team:
            team_id = TEAM_ABBR_TO_ID.get(team_code.upper())
            if team_id is not None and self._team_data is not None:
                team_games = self._team_data[self._team_data["team_id"] == str(team_id)].sort_values("game_date")
                if not team_games.empty:
                    avg = team_games[f"{raw_col}_avg"].iloc[-1] if f"{raw_col}_avg" in team_games.columns else None
                    std = team_games[f"{raw_col}_std"].iloc[-1] if f"{raw_col}_avg" in team_games.columns else None
                    if avg is not None and not pd.isna(avg):
                        if std is not None and not pd.isna(std) and std > 0:
                            from scipy.stats import norm
                            z = (line_val - avg) / std
                            prob = 1.0 - norm.cdf(z)
                        else:
                            prob = 0.5 + (float(avg) - line_val) / (2 * max(float(avg), 1))
                            prob = min(max(prob, 0.05), 0.95)
                        return {
                            "team": team_code,
                            "player": player_name,
                            "stat_type": pp_stat,
                            "line": line_val,
                            "avg": round(float(avg), 1),
                            "std": round(float(std), 1) if std is not None and not pd.isna(std) and std > 0 else None,
                            "model_prob": round(prob, 4),
                            "effective_prob": round(prob, 4),
                            "direction": "over",
                            "is_model": False,
                        }
        return None

    def _player_std(self, player_id, raw_col):
        """Compute standard deviation of a stat for a player from raw data."""
        df = self._featured
        if df is None:
            return None
        pdata = df[df["player_id"] == player_id].sort_values("game_date")
        if pdata.empty or raw_col not in pdata.columns:
            return None
        vals = pdata[raw_col].dropna().tail(20)
        if len(vals) < 3:
            return None
        return float(vals.std())


def predict_mlb_edges(pipeline, lines):
    from src.data.kalshi import KalshiClient
    from src.execution.risk import RiskManager
    from src.config.settings import settings

    cfg = settings.load_sport_config("mlb")
    predictor = MLBLinePredictor(cfg)
    if not predictor.load_data(pipeline):
        print("  Failed to load MLB data for prediction")
        return

    try:
        actual_bankroll = KalshiClient().get_balance()
    except Exception:
        actual_bankroll = 10.0

    risk = RiskManager(
        bankroll=actual_bankroll,
        kelly_fraction=0.25,
        max_bet_pct=0.03,
    )

    opportunities = []
    printed = set()
    for _, row in lines.iterrows():
        line_dict = {k: str(v) if isinstance(v, (pd.Timestamp, np.integer, np.floating))
                     else v for k, v in row.items()}
        result = predictor.predict_line(line_dict)
        if result is None:
            continue

        prob = result.get("model_prob", result.get("effective_prob", 0.5))
        avg = result.get("avg", 0)
        lv = result.get("line", 0)

        if avg and avg > 0 and lv > 0:
            distance_ratio = abs(float(avg) - lv) / float(avg)
            if distance_ratio > 2.0:
                continue

        breakeven = 0.542
        edge = prob - breakeven
        if edge <= 0:
            continue

        result["edge_pct"] = edge
        kelly_frac = risk.kelly_size(prob, 1 / (1 - breakeven))
        bet = risk.size_bet(prob, lv, "over")
        result["bet_size"] = round(bet, 2)

        opportunities.append(result)

    opportunities.sort(key=lambda x: x["edge_pct"], reverse=True)

    print(f"  {'Player':<22} {'Stat':<22} {'Line':<6} {'Avg':<6} "
          f"{'P(over)':<8} {'Edge%':<8} {'Bet':<6}")
    print(f"  {'-'*75}")
    for opp in opportunities[:30]:
        player = opp.get("player", opp.get("team", "?"))
        dedup_key = (player, opp["stat_type"], opp.get("line"))
        if dedup_key in printed:
            continue
        printed.add(dedup_key)
        print(f"  {player:<22} {opp['stat_type']:<22} "
              f"{opp['line']:<6} {opp.get('avg', 'N/A') or 'N/A':<6} "
              f"{opp['effective_prob']:.1%}  {opp['edge_pct']:.1%}  "
              f"${opp['bet_size']:<.2f}")

    print(f"\n  {len(opportunities)} total edge opportunities ({len(printed)} unique)")
