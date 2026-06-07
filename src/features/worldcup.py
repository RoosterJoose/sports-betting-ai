import pandas as pd

from src.features.base import FeatureEngineer


class WorldCupFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()
        if df.empty:
            return df

        if "game_date" not in df.columns:
            if "match_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["match_date"])
            else:
                df["game_date"] = pd.to_datetime(df.index)

        # ELO features (team's ELO relative to opponent)
        if "elo_pre" in df.columns and "opponent_elo_pre" in df.columns:
            df["elo_diff"] = df["elo_pre"] - df["opponent_elo_pre"]
            df["elo_total"] = df["elo_pre"] + df["opponent_elo_pre"]
            # Win probability based on ELO
            df["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** ((df["opponent_elo_pre"] - df["elo_pre"]) / 400.0))

        # Rolling averages: form, goals scored, goals conceded (all with shift(1) to prevent leakage)
        stat_cols = []
        for col in ["elo_pre", "elo_diff", "goals_for", "goals_against"]:
            if col in df.columns:
                stat_cols.append(col)
        
        if stat_cols:
            df = self.rolling_averages(df, stat_cols)
            df = self.recency_weighted_avg(df, stat_cols)

        # Keep only feature columns + identity
        keep_cols = ["player_id", "game_date", "season", "is_home"]
        for col in df.columns:
            if any(col.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(col)
            elif col.endswith("_ewm"):
                keep_cols.append(col)
            elif col in ["elo_diff", "elo_total", "elo_win_prob"]:
                keep_cols.append(col)

        return df[[c for c in keep_cols if c in df.columns]].copy()