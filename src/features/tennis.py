import pandas as pd

from src.features.base import FeatureEngineer

TENNIS_STATS = [
    "ace", "df", "svpt", "first_in", "first_won",
    "second_won", "bp_saved", "bp_faced", "sv_gms",
]

TENNIS_OPP_STATS = [
    "ace_opp", "df_opp", "svpt_opp", "first_in_opp",
    "first_won_opp", "second_won_opp",
]


class TennisFeatureEngineer(FeatureEngineer):
    def build_features(self, games: pd.DataFrame, opponent_games: pd.DataFrame = None) -> pd.DataFrame:
        df = games.copy()

        if "tourney_date" in df.columns:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

        player_id = df.get("player_id", df.get("winner_id"))
        if player_id is not None:
            df["is_winner"] = (df["winner_id"].astype(str) == df["player_id"].astype(str)).astype(int)
            player_cols = {
                "ace": "w_ace" if "w_ace" in df.columns else "ace",
                "df": "w_df" if "w_df" in df.columns else "df",
                "first_in": "w_first_in" if "w_first_in" in df.columns else "first_in",
                "first_won": "w_first_won" if "w_first_won" in df.columns else "first_won",
                "second_won": "w_second_won" if "w_second_won" in df.columns else "second_won",
                "bp_saved": "w_bp_saved" if "w_bp_saved" in df.columns else "bp_saved",
                "bp_faced": "w_bp_faced" if "w_bp_faced" in df.columns else "bp_faced",
            }
            rename = {v: k for k, v in player_cols.items() if v != k}
            df = df.rename(columns=rename)

        if "surface" in df.columns:
            df["surface_hard"] = (df["surface"] == "Hard").astype(int)
            df["surface_clay"] = (df["surface"] == "Clay").astype(int)
            df["surface_grass"] = (df["surface"] == "Grass").astype(int)

        if "best_of" in df.columns:
            df["is_best_of_5"] = (df["best_of"] == 5).astype(int)

        available = [c for c in TENNIS_STATS if c in df.columns]
        if available:
            df = self.rolling_averages(df, available, group_col="player_id")
            df = self.recency_weighted_avg(df, available, group_col="player_id")

        if "ace" in df.columns and "df" in df.columns:
            df["ace_to_df_ratio"] = df["ace"] / df["df"].replace(0, 1)
            df = self.rolling_averages(df, ["ace_to_df_ratio"], group_col="player_id")

        if "first_in" in df.columns and "first_won" in df.columns:
            df["first_serve_pct"] = df["first_in"] / df["svpt"].replace(0, 1)
            df["first_serve_won_pct"] = df["first_won"] / df["first_in"].replace(0, 1)

        if "bp_faced" in df.columns and "bp_saved" in df.columns:
            df["bp_save_pct"] = df["bp_saved"] / df["bp_faced"].replace(0, 1)

        df = self.remove_leakage(df)
        return df
