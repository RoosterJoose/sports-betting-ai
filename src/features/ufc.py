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

STAT_INFO = [
    ("avg_sig_str_landed", 27.0),
    ("avg_td_landed", 1.3),
    ("avg_sub_att", 0.5),
    ("wins", 0),
    ("losses", 0),
    ("total_rounds_fought", 5),
    ("height_cms", 178.0),
    ("reach_cms", 183.0),
    ("weight_lbs", 170.0),
    ("age", 30),
]


FEATURE_COLS = [
    "r_avg_sig_str_landed", "b_avg_sig_str_landed",
    "r_avg_td_landed", "b_avg_td_landed",
    "r_avg_sub_att", "b_avg_sub_att",
    "r_wins", "b_wins", "r_losses", "b_losses",
    "r_total_rounds_fought", "b_total_rounds_fought",
    "r_height_cms", "b_height_cms",
    "r_reach_cms", "b_reach_cms",
    "r_weight_lbs", "b_weight_lbs",
    "r_age", "b_age",
    "sig_str_diff", "sig_str_abs_diff", "sig_str_total",
    "td_diff", "td_abs_diff", "td_total",
    "sub_diff", "sub_abs_diff", "sub_total",
    "w_diff", "w_abs_diff", "w_total",
    "l_diff", "l_abs_diff", "l_total",
    "rounds_diff", "rounds_abs_diff", "rounds_total",
    "height_diff", "height_abs_diff",
    "reach_diff", "reach_abs_diff",
    "weight_diff", "weight_abs_diff",
    "age_diff", "age_abs_diff",
    "combined_sig_str", "combined_td_att",
    "r_win_pct", "b_win_pct",
    "r_experience", "b_experience",
    "exp_diff", "exp_total",
    "wc_finish_rate", "scheduled_rounds",
    "is_title_fight", "is_female",
]


def build_ufc_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        return result

    result = result.reset_index(drop=True)

    for col, default in STAT_INFO:
        for prefix in ["r_", "b_"]:
            c = f"{prefix}{col}"
            if c not in result.columns:
                result[c] = default
            result[c] = pd.to_numeric(result[c], errors="coerce").fillna(default)

    for col, default in STAT_INFO:
        short = col.replace("avg_", "").replace("_landed", "").replace("_att", "")
        if short == col:
            short = col.split("_")[0] if "_" in col else col
        if col == "total_rounds_fought":
            short = "rounds"
        r_c = f"r_{col}"
        b_c = f"b_{col}"
        result[f"{short}_diff"] = result[r_c] - result[b_c]
        result[f"{short}_abs_diff"] = result[f"{short}_diff"].abs()
        result[f"{short}_total"] = result[r_c] + result[b_c]

    result["combined_sig_str"] = (
        result["r_avg_sig_str_landed"] + result["b_avg_sig_str_landed"]
    )
    result["combined_td_att"] = (
        result["r_avg_td_landed"] + result["b_avg_td_landed"]
    )

    for prefix in ["r_", "b_"]:
        total_fights = result[f"{prefix}wins"] + result[f"{prefix}losses"]
        result[f"{prefix}win_pct"] = result[f"{prefix}wins"] / total_fights.replace(0, 1)
        result[f"{prefix}experience"] = total_fights
        result[f"{prefix}win_pct"] = result[f"{prefix}win_pct"].fillna(0.5)
        result[f"{prefix}experience"] = result[f"{prefix}experience"].fillna(0)

    result["exp_diff"] = result["r_experience"] - result["b_experience"]
    result["exp_total"] = result["r_experience"] + result["b_experience"]

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
            result["is_title_fight"] = result["title_bout"].fillna(0).astype(int)
        except (ValueError, TypeError):
            pass

    if "gender" in result.columns:
        result["is_female"] = (
            result["gender"].fillna("male").astype(str).str.lower().str.contains("female")
        ).astype(int)
    else:
        result["is_female"] = 0

    if "player_id" not in result.columns:
        result["player_id"] = result.get("game_id", result.index.astype(str))
    date_col = result.get("game_date", result.get("date"))
    if date_col is not None:
        result["game_date"] = pd.to_datetime(date_col, errors="coerce")
    else:
        result["game_date"] = pd.Timestamp.now()
    result["game_date"] = result["game_date"].fillna(pd.Timestamp.now())

    result["total_fight_time_secs"] = pd.to_numeric(
        result.get("total_fight_time_secs", 900), errors="coerce"
    ).fillna(900)

    available_features = [c for c in FEATURE_COLS if c in result.columns]

    return result
