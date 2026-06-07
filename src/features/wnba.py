import pandas as pd
import numpy as np

from src.features.base import FeatureEngineer

WNBA_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
    "min",
]

WNBA_COMBINED = {
    "pr": ["pts", "reb"],
    "pa": ["pts", "ast"],
    "pra": ["pts", "reb", "ast"],
}

WNBA_SCARCE = ["stl", "blk"]


class WNBAFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()
        if "game_date" not in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        for col in WNBA_SCARCE:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        stats_for_features = WNBA_STATS + [f"{s}_log" for s in WNBA_SCARCE]

        df = self.rolling_averages(df, stats_for_features)
        df = self.rolling_medians(df, stats_for_features)
        df = self.recency_weighted_avg(df, stats_for_features)
        df = self.streak_features(df, stats_for_features)
        df = self.consistency_features(df, stats_for_features)
        df = self.expected_possessions(df)
        df = self.home_away_split(df)

        for name, parts in WNBA_COMBINED.items():
            df[f"{name}_combined"] = df[parts].sum(axis=1)
            df = self.rolling_averages(df, [f"{name}_combined"])
            df = self.rolling_medians(df, [f"{name}_combined"])
            df = self.streak_features(df, [f"{name}_combined"])
            df = self.consistency_features(df, [f"{name}_combined"])
            df = df.drop(columns=[f"{name}_combined"], errors="ignore")

        df = self.schedule_density(df)

        for stat in ["pts", "reb", "ast", "stl", "blk"]:
            df = self.opponent_adjustment(df, stat)

        keep_cols = ["player_id", "game_date", "game_id", "season", "matchup"]
        for c in df.columns:
            if any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            elif any(c.endswith(f"_med_{w}") for w in self.windows):
                keep_cols.append(c)
            elif c.endswith("_ewm"):
                keep_cols.append(c)
            elif c.endswith(("_streak", "_consistency", "_adj")):
                keep_cols.append(c)
            elif c in ("days_rest", "b2b", "four_in_six", "is_home", "exp_poss"):
                keep_cols.append(c)

        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
