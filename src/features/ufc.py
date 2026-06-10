import numpy as np
import pandas as pd

WEIGHT_CLASS_FINISH_PCT = {
    "heavyweight": 0.55, "light heavyweight": 0.50,
    "middleweight": 0.48, "welterweight": 0.46,
    "lightweight": 0.44, "featherweight": 0.42,
    "bantamweight": 0.40, "flyweight": 0.38,
    "women's bantamweight": 0.36, "women's flyweight": 0.34,
    "women's strawweight": 0.32, "women's featherweight": 0.34,
    "catch weight": 0.50,
}

# ── Defaults for career-average stats ──────────────────────────────────
STAT_INFO = [
    ("avg_sig_str_landed", 27.0),
    ("avg_sig_str_pct", 0.48),
    ("avg_td_landed", 1.3),
    ("avg_td_pct", 0.35),
    ("avg_sub_att", 0.5),
    ("current_win_streak", 1),
    ("current_lose_streak", 0),
    ("longest_win_streak", 3),
    ("wins", 0),
    ("losses", 0),
    ("total_rounds_fought", 5),
    ("total_title_bouts", 0),
    ("height_cms", 178.0),
    ("reach_cms", 183.0),
    ("weight_lbs", 170.0),
    ("age", 30),
    ("win_by_ko_tko", 3),
    ("win_by_submission", 2),
    ("win_by_decision_unanimous", 2),
    ("win_by_decision_split", 0),
    ("win_by_decision_majority", 0),
    ("win_by_tko_doctor_stoppage", 0),
    ("match_weightclass_rank", 50),
]

# Stance encoding
STANCE_ENCODER = {"orthodox": 0, "southpaw": 1, "switch": 2, "open stance": 3}

# ── All feature column names ──────────────────────────────────────────
FEATURE_COLS = [
    # Physical (existing)
    "r_height_cms", "b_height_cms",
    "r_reach_cms", "b_reach_cms",
    "r_weight_lbs", "b_weight_lbs",
    "r_age", "b_age",
    "height_diff", "height_abs_diff",
    "reach_diff", "reach_abs_diff",
    "weight_diff", "weight_abs_diff",
    "age_diff", "age_abs_diff",
    # Career averages (existing)
    "r_avg_sig_str_landed", "b_avg_sig_str_landed",
    "r_avg_td_landed", "b_avg_td_landed",
    "r_avg_sub_att", "b_avg_sub_att",
    "r_wins", "b_wins", "r_losses", "b_losses",
    "r_total_rounds_fought", "b_total_rounds_fought",
    "sig_str_diff", "sig_str_abs_diff", "sig_str_total",
    "td_diff", "td_abs_diff", "td_total",
    "sub_diff", "sub_abs_diff", "sub_total",
    "w_diff", "w_abs_diff", "w_total",
    "l_diff", "l_abs_diff", "l_total",
    "rounds_diff", "rounds_abs_diff", "rounds_total",
    "combined_sig_str", "combined_td_att",
    "r_win_pct", "b_win_pct",
    "r_experience", "b_experience",
    "exp_diff", "exp_total",
    # NEW: Striking accuracy
    "r_avg_sig_str_pct", "b_avg_sig_str_pct",
    "sig_str_pct_diff", "sig_str_pct_abs_diff",
    # NEW: Takedown accuracy
    "r_avg_td_pct", "b_avg_td_pct",
    "td_pct_diff", "td_pct_abs_diff",
    # NEW: Current streaks
    "r_current_win_streak", "b_current_win_streak",
    "r_current_lose_streak", "b_current_lose_streak",
    "win_streak_diff", "lose_streak_diff",
    # NEW: Longest streaks
    "r_longest_win_streak", "b_longest_win_streak",
    "longest_win_streak_diff",
    # NEW: Title / ranking
    "r_total_title_bouts", "b_total_title_bouts",
    "title_bouts_diff", "title_bouts_total",
    "r_match_weightclass_rank", "b_match_weightclass_rank",
    "rank_diff", "rank_abs_diff", "better_rank",
    # NEW: Stance matchup
    "r_stance_code", "b_stance_code",
    "same_stance", "both_orthodox",
    # NEW: Win method breakdowns
    "r_ko_rate", "b_ko_rate",
    "r_sub_rate", "b_sub_rate",
    "r_dec_rate", "b_dec_rate",
    "ko_rate_diff", "sub_rate_diff", "dec_rate_diff",
    # NEW: Rolling form features (computed in build_features)
    "fighter_avg_sig_str_landed_avg_3",
    "fighter_avg_td_landed_avg_3",
    "opponent_avg_sig_str_landed_avg_3",
    # NEW: Opponent quality
    "opponent_avg_sig_str_landed",
    "opponent_avg_td_landed",
    "fighter_def_rating",
    "opponent_quality_adj",
    # Context
    "wc_finish_rate", "scheduled_rounds",
    "is_title_fight", "is_female",
    # NEW: Layoff / fight pace
    "days_since_last_fight",
    "fighting_freq_365",
    # NEW: Win method from last 3 fights (rolling)
    "fighter_recent_ko_rate",
    "fighter_recent_first_round_rate",
]


def build_ufc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all UFC features from per-fighter-per-fight DataFrame.

    The input DataFrame must have:
      - Raw CSV columns (r_*, b_*) preserved from data source
      - fighter_* / opponent_* normalized columns for rolling features
      - Fight-level columns (winner, finish_round, weight_class, etc.)
    """
    result = df.copy()
    if result.empty:
        return result

    result = result.reset_index(drop=True)

    # ── 1. Fill missing stat defaults ─────────────────────────────────
    for col, default in STAT_INFO:
        for prefix in ["r_", "b_"]:
            c = f"{prefix}{col}"
            if c not in result.columns:
                result[c] = default
            result[c] = pd.to_numeric(result[c], errors="coerce").fillna(default)

    # ── 2. Physical diff features ─────────────────────────────────────
    for col, short, default in [
        ("avg_sig_str_landed", "sig_str", 27.0),
        ("avg_td_landed", "td", 1.3),
        ("avg_sub_att", "sub", 0.5),
        ("wins", "w", 0),
        ("losses", "l", 0),
        ("total_rounds_fought", "rounds", 5),
        ("height_cms", "height", 178.0),
        ("reach_cms", "reach", 183.0),
        ("weight_lbs", "weight", 170.0),
        ("age", "age", 30),
    ]:
        r_c = f"r_{col}"
        b_c = f"b_{col}"
        result[f"{short}_diff"] = result[r_c] - result[b_c]
        result[f"{short}_abs_diff"] = result[f"{short}_diff"].abs()
        result[f"{short}_total"] = result[r_c] + result[b_c]

    # ── 3. NEW: Striking & TD accuracy diffs ─────────────────────────
    for col, short in [
        ("avg_sig_str_pct", "sig_str_pct"),
        ("avg_td_pct", "td_pct"),
    ]:
        r_c = f"r_{col}"
        b_c = f"b_{col}"
        result[f"{short}_diff"] = result[r_c] - result[b_c]
        result[f"{short}_abs_diff"] = result[f"{short}_diff"].abs()

    # ── 4. NEW: Streak features ──────────────────────────────────────
    result["win_streak_diff"] = result["r_current_win_streak"] - result["b_current_win_streak"]
    result["lose_streak_diff"] = result["r_current_lose_streak"] - result["b_current_lose_streak"]
    result["longest_win_streak_diff"] = result["r_longest_win_streak"] - result["b_longest_win_streak"]

    # ── 5. NEW: Title / rank features ────────────────────────────────
    result["title_bouts_diff"] = result["r_total_title_bouts"] - result["b_total_title_bouts"]
    result["title_bouts_total"] = result["r_total_title_bouts"] + result["b_total_title_bouts"]
    result["rank_diff"] = result["r_match_weightclass_rank"] - result["b_match_weightclass_rank"]
    result["rank_abs_diff"] = result["rank_diff"].abs()
    # better_rank from CSV (1 = red has better rank)
    if "better_rank" in result.columns:
        result["better_rank"] = pd.to_numeric(result["better_rank"], errors="coerce").fillna(0)

    # ── 6. NEW: Stance encoding ──────────────────────────────────────
    for prefix in ["r_", "b_"]:
        stance_col = f"{prefix}stance"
        if stance_col in result.columns:
            result[f"{prefix}stance_code"] = (
                result[stance_col].astype(str).str.strip().str.lower()
                .map(STANCE_ENCODER).fillna(0)
            )
        else:
            result[f"{prefix}stance_code"] = 0
    result["same_stance"] = (
        result["r_stance_code"] == result["b_stance_code"]
    ).astype(int)
    result["both_orthodox"] = (
        (result["r_stance_code"] == 0) & (result["b_stance_code"] == 0)
    ).astype(int)

    # ── 7. NEW: Win method rates (career) ────────────────────────────
    for prefix in ["r_", "b_"]:
        total_wins = result[f"{prefix}wins"].clip(lower=1)
        ko = result.get(f"{prefix}win_by_ko_tko", 0)
        sub = result.get(f"{prefix}win_by_submission", 0)
        dec_uni = result.get(f"{prefix}win_by_decision_unanimous", 0)
        dec_split = result.get(f"{prefix}win_by_decision_split", 0)
        dec_maj = result.get(f"{prefix}win_by_decision_majority", 0)
        total_dec = pd.to_numeric(dec_uni, errors="coerce").fillna(0) \
                  + pd.to_numeric(dec_split, errors="coerce").fillna(0) \
                  + pd.to_numeric(dec_maj, errors="coerce").fillna(0)
        result[f"{prefix}ko_rate"] = pd.to_numeric(ko, errors="coerce").fillna(0) / total_wins
        result[f"{prefix}sub_rate"] = pd.to_numeric(sub, errors="coerce").fillna(0) / total_wins
        result[f"{prefix}dec_rate"] = total_dec / total_wins

    for rate in ["ko_rate", "sub_rate", "dec_rate"]:
        result[f"{rate}_diff"] = result[f"r_{rate}"] - result[f"b_{rate}"]

    # ── 8. REMOVED: Historical odds features (June 10 fix — fatal flaw) ──
    # Model was trained WITH odds but predicted WITH synthetic odds=0.
    # Removed r_odds, b_odds, odds_diff, odds_abs_diff from FEATURE_COLS
    # so the model is now self-contained at inference time. Predictions
    # no longer depend on betting-market info we don't have at trade time.

    # ── 9. Combined features ─────────────────────────────────────────
    result["combined_sig_str"] = result["r_avg_sig_str_landed"] + result["b_avg_sig_str_landed"]
    result["combined_td_att"] = result["r_avg_td_landed"] + result["b_avg_td_landed"]

    # ── 10. Experience & win% ────────────────────────────────────────
    for prefix in ["r_", "b_"]:
        total_fights = result[f"{prefix}wins"] + result[f"{prefix}losses"]
        result[f"{prefix}win_pct"] = result[f"{prefix}wins"] / total_fights.replace(0, 1)
        result[f"{prefix}experience"] = total_fights
        result[f"{prefix}win_pct"] = result[f"{prefix}win_pct"].fillna(0.5)
        result[f"{prefix}experience"] = result[f"{prefix}experience"].fillna(0)

    result["exp_diff"] = result["r_experience"] - result["b_experience"]
    result["exp_total"] = result["r_experience"] + result["b_experience"]

    # ── 11. Weight class metadata ────────────────────────────────────
    wc = result.get("weight_class", "").astype(str).str.strip().str.lower()
    result["wc_finish_rate"] = wc.map(WEIGHT_CLASS_FINISH_PCT).fillna(0.45)
    result["scheduled_rounds"] = pd.to_numeric(
        result.get("no_of_rounds", 3), errors="coerce"
    ).fillna(3)
    result["is_title_fight"] = (
        pd.to_numeric(result.get("no_of_rounds", 3), errors="coerce").fillna(3) >= 5
    ).astype(int)

    if "title_bout" in result.columns:
        try:
            result["is_title_fight"] = pd.to_numeric(result["title_bout"], errors="coerce").fillna(0).astype(int)
        except (ValueError, TypeError):
            pass

    # ── 12. Gender ───────────────────────────────────────────────────
    if "gender" in result.columns:
        result["is_female"] = (
            result["gender"].fillna("male").astype(str).str.lower().str.contains("female")
        ).astype(int)
    else:
        result["is_female"] = 0

    # ── 13. Rolling form features (using normalized fighter_* cols) ──
    # If fighter_* columns exist, compute rolling averages
    fighter_rolling_cols = [
        "fighter_avg_sig_str_landed",
        "fighter_avg_sig_str_pct",
        "fighter_avg_td_landed",
        "fighter_avg_td_pct",
        "fighter_avg_sub_att",
    ]

    avail_fighter_rolling = [c for c in fighter_rolling_cols if c in result.columns]
    if avail_fighter_rolling and "game_date" in result.columns:
        result = _add_rolling_features(result, avail_fighter_rolling)

        # Opponent quality: rolling opponent striking allowed
        if "opponent_avg_sig_str_landed" in result.columns:
            result = _add_opponent_quality(result)

    # ── 14. Layoff & fight pace ──────────────────────────────────────
    if "game_date" in result.columns and "player_id" in result.columns:
        result = result.sort_values(["player_id", "game_date"])
        result["days_since_last_fight"] = (
            result.groupby("player_id")["game_date"].diff().dt.days
        )
        result["days_since_last_fight"] = result["days_since_last_fight"].fillna(180)

        # Fighting frequency: fights in last 365 days
        # Use a simple loop per fighter (dataset is only ~10K rows)
        result = result.sort_values(["player_id", "game_date"])
        result["fighting_freq_365"] = 0
        for fighter in result["player_id"].unique():
            mask = result["player_id"] == fighter
            idx = result.index[mask]
            dates = result.loc[idx, "game_date"]
            valid = dates.notna()
            if valid.any():
                fighter_dates = dates[valid]
                counts = []
                for i in range(len(fighter_dates)):
                    cutoff = fighter_dates.iloc[i] - pd.Timedelta(days=365)
                    count = (fighter_dates.iloc[:i] >= cutoff).sum()
                    counts.append(count)
                result.loc[fighter_dates.index, "fighting_freq_365"] = counts

        # Recent finish rate (last 3 fights)
        if "is_ko" in result.columns:
            result["fighter_recent_ko_rate"] = (
                result.groupby("player_id")["is_ko"]
                .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            )
            result["fighter_recent_ko_rate"] = result["fighter_recent_ko_rate"].fillna(
                result.get("wc_finish_rate", 0.45)
            )

        # Recent first-round finish rate
        if "finish_round" in result.columns:
            first_round = (result["finish_round"] == 1).astype(int)
            result["fighter_recent_first_round_rate"] = (
                result.groupby("player_id")["finish_round"]
                .transform(lambda x: (x.shift(1) == 1).rolling(3, min_periods=1).mean())
            )
            result["fighter_recent_first_round_rate"] = result[
                "fighter_recent_first_round_rate"
            ].fillna(0.1)

    # ── 15. Ensure game_date present ─────────────────────────────────
    date_col = result.get("game_date", result.get("date"))
    if date_col is not None:
        result["game_date"] = pd.to_datetime(date_col, errors="coerce")
    else:
        result["game_date"] = pd.Timestamp.now()
    result["game_date"] = result["game_date"].fillna(pd.Timestamp.now())

    result["total_fight_time_secs"] = pd.to_numeric(
        result.get("total_fight_time_secs", 900), errors="coerce"
    ).fillna(900)

    # ── 16. Ensure player_id exists ──────────────────────────────────
    if "player_id" not in result.columns:
        result["player_id"] = result.get("game_id", result.index.astype(str))

    available_features = [c for c in FEATURE_COLS if c in result.columns]
    return result


def _add_rolling_features(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """Add rolling averages, medians, and EWMA for fighter stats."""
    df = df.sort_values(["player_id", "game_date"])

    for w in [3, 5]:
        for col in stat_cols:
            if col not in df.columns:
                continue
            df[f"{col}_avg_{w}"] = (
                df.groupby("player_id")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    # EWMA (recency-weighted)
    for col in stat_cols:
        if col not in df.columns:
            continue
        df[f"{col}_ewm"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).ewm(alpha=0.3).mean())
        )
    # Opponent rolling stats
    opp_cols = [c for c in df.columns if c.startswith("opponent_") and c in stat_cols]
    for w in [3]:
        for col in opp_cols:
            if col not in df.columns:
                continue
            df[f"{col}_avg_{w}"] = (
                df.groupby("player_id")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    return df


def _add_opponent_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust opponent striking stats for opponent quality.
    Tracks what average level of striking a fighter allows their opponents
    (defensive rating). Then adjusts opponent quality relative to the field.
    """
    df = df.sort_values(["player_id", "game_date"])

    stat = "opponent_avg_sig_str_landed"
    if stat not in df.columns:
        return df

    # Rolling avg of what each fighter's opponents achieve
    # (high value = fighter allows lots of strikes = weak defense)
    df["fighter_def_rating"] = (
        df.groupby("player_id")[stat]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    league_avg = df[stat].mean()
    df["fighter_def_rating"] = df["fighter_def_rating"].fillna(league_avg)

    # Opponent quality adjustment for current opponent
    # >1.0 means the fighter faces a tougher-than-average opponent
    # <1.0 means the fighter faces a weaker-than-average opponent
    df["opponent_quality_adj"] = (
        df["fighter_def_rating"] / league_avg
    ).clip(0.5, 2.0)

    return df
