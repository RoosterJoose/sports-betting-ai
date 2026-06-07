import pandas as pd
import numpy as np

from src.features.base import FeatureEngineer

NBA_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
    "min", "plus_minus",
]

NBA_SCARCE = ["stl", "blk"]  # Poisson/low-frequency, need log-transform

NBA_COMBINED = {
    "pr": ["pts", "reb"],
    "pa": ["pts", "ast"],
    "ra": ["reb", "ast"],
    "pra": ["pts", "reb", "ast"],
    "sb": ["stl", "blk"],
}


class NBAFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()
        if "game_date" not in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        # Log-transform scarce stats for Poisson stability
        for col in NBA_SCARCE:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        stats_for_features = NBA_STATS + [f"{s}_log" for s in NBA_SCARCE]

        # Rolling features (all use shift(1) = no leak)
        df = self.rolling_averages(df, stats_for_features)
        df = self.rolling_medians(df, stats_for_features)
        df = self.recency_weighted_avg(df, stats_for_features)
        df = self.streak_features(df, stats_for_features)
        df = self.consistency_features(df, stats_for_features)
        df = self.expected_possessions(df)
        df = self.home_away_split(df)

        # Combined stats — rolling averages, then drop raw combined
        for name, parts in NBA_COMBINED.items():
            combined = f"{name}_combined"
            df[combined] = df[parts].sum(axis=1)
            df = self.rolling_averages(df, [combined])
            df = self.rolling_medians(df, [combined])
            df = self.streak_features(df, [combined])
            df = self.consistency_features(df, [combined])
            df = df.drop(columns=[combined], errors="ignore")

        # Schedule density
        df = self.schedule_density(df)

        # Opponent adjustments at team level
        for stat in ["pts", "reb", "ast", "stl", "blk"]:
            df = self.opponent_adjustment(df, stat)

        # Strip raw current-game stat columns. Keep only lagged features + schedule + identity.
        keep_cols = ["player_id", "game_date", "game_id", "season", "matchup"]
        for c in df.columns:
            # Rolling averages
            if any(c.endswith(f"_avg_{w}") for w in self.windows):
                keep_cols.append(c)
            # Rolling medians
            elif any(c.endswith(f"_med_{w}") for w in self.windows):
                keep_cols.append(c)
            # EWMs
            elif c.endswith("_ewm"):
                keep_cols.append(c)
            # Derived features
            elif c.endswith(("_streak", "_consistency", "_adj")):
                keep_cols.append(c)
            # Schedule + meta
            elif c in ("days_rest", "b2b", "four_in_six", "is_home", "exp_poss"):
                keep_cols.append(c)

        df = df[[c for c in keep_cols if c in df.columns]].copy()
        return df
