#!/usr/bin/env python3
"""Validate the empirical neutral-venue offset on the 2023+ OOS test set.

The offset was computed on the 2022 WC val set (57 neutral-venue matches).
This script answers: does it generalize to held-out 2023+ matches?

Key checks:
  1. Full 2023+ test set: with offset vs without (offset only fires at
     is_neutral=1 matches, so non-neutral rows are unchanged).
  2. Neutral-venue subset only (the rows where offset actually fires):
     with vs without — the apples-to-apples test.
  3. Non-neutral subset: with vs without — control (should be identical
     modulo floating-point).
  4. Per-tournament breakdown.

If Brier improves on the neutral-venue subset, the offset is real.
If it gets worse, it's 2022-WC-specific overfitting and should not be
trusted for WC 2026.

Usage:
    python -m src.scripts.validate_offset_oos
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.world_cup import fetch_all_matches, compute_elo, build_feature_vector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "worldcup"
CALIB_DIR = MODEL_DIR / "calibration"

NEUTRAL_TOURNAMENTS = {"WC", "EC", "AC"}
OUTCOME_LABELS = ["home", "draw", "away"]


def _build_form_features(elo_df, elo_ratings, cutoff_date):
    """Build recent form features for all teams from ELO data BEFORE cutoff_date.

    Mirrors build_feature_dataset() in src/data/world_cup.py: performance vs
    Elo-expected win probability, last 5 matches.
    """
    form = {}
    for team in elo_ratings:
        team_matches = elo_df[
            ((elo_df["home_team"] == team) | (elo_df["away_team"] == team))
            & (elo_df["match_date"] < cutoff_date)
        ]
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

            if home_score > away_score:
                actual = 1.0 if is_home else 0.0
            elif away_score > home_score:
                actual = 0.0 if is_home else 1.0
            else:
                actual = 0.5

            from src.data.world_cup import _elo_expected
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
    return form


def _build_team_history(elo_df):
    """Precompute per-team chronological match history as a dict.

    Returns {team: sorted_dates_array}. We then use date-bounded slicing
    via np.searchsorted to get the last 5 matches before any cutoff in O(log n)
    per query — way faster than re-scanning elo_df inside the per-match loop.
    """
    from src.data.world_cup import _elo_expected

    # Pre-convert to numpy for speed
    dates = elo_df["match_date"].values  # datetime64[ns]
    home = elo_df["home_team"].values
    away = elo_df["away_team"].values
    home_score = elo_df["home_score"].values.astype(int)
    away_score = elo_df["away_score"].values.astype(int)
    elo_home_pre = elo_df["elo_home_pre"].values
    elo_away_pre = elo_df["elo_away_pre"].values

    # Group by team: list of (date_idx, is_home, home_score, away_score, team_elo, opp_elo, actual_pts)
    team_hist = {}
    n = len(elo_df)
    for i in range(n):
        h, a = home[i], away[i]
        # home team
        if h not in team_hist:
            team_hist[h] = []
        team_hist[h].append((dates[i], 1, int(home_score[i]), int(away_score[i]),
                              float(elo_home_pre[i]), float(elo_away_pre[i])))
        # away team
        if a not in team_hist:
            team_hist[a] = []
        team_hist[a].append((dates[i], 0, int(home_score[i]), int(away_score[i]),
                              float(elo_away_pre[i]), float(elo_home_pre[i])))

    # Sort each team's history by date (stable since we walked elo_df in date order)
    for t in team_hist:
        team_hist[t].sort(key=lambda r: r[0])
    return team_hist


def _form_for_team(team_hist, team, cutoff_date, default_elo=1500):
    """Compute form features for one team using precomputed history.

    O(log n) date lookup + O(5) record aggregation.
    """
    if team not in team_hist:
        return {"perf": 0.0, "opp_elo": default_elo, "gs": 0.0, "gc": 0.0, "n": 0}
    hist = team_hist[team]
    # Find last 5 records with date < cutoff_date
    dates_arr = np.array([r[0] for r in hist])
    # Binary search for first index >= cutoff_date
    cutoff_ns = np.datetime64(cutoff_date)
    end = int(np.searchsorted(dates_arr, cutoff_ns, side="left"))
    start = max(0, end - 5)
    window = hist[start:end]
    if not window:
        return {"perf": 0.0, "opp_elo": default_elo, "gs": 0.0, "gc": 0.0, "n": 0}

    from src.data.world_cup import _elo_expected
    perf_sum, opp_elo_sum, gs_sum, gc_sum = 0.0, 0.0, 0.0, 0.0
    for _, is_home, hs, as_, team_elo, opp_elo in window:
        if hs > as_:
            actual = 1.0 if is_home else 0.0
        elif as_ > hs:
            actual = 0.0 if is_home else 1.0
        else:
            actual = 0.5
        expected = _elo_expected(team_elo, opp_elo)
        perf_sum += actual - expected
        opp_elo_sum += opp_elo
        if is_home:
            gs_sum += hs; gc_sum += as_
        else:
            gs_sum += as_; gc_sum += hs
    k = len(window)
    return {
        "perf": perf_sum / k,
        "opp_elo": opp_elo_sum / k,
        "gs": gs_sum / k,
        "gc": gc_sum / k,
        "n": k,
    }


def main():
    print("=" * 70)
    print("  EMPIRICAL OFFSET — OUT-OF-SAMPLE VALIDATION (2023+ test set)")
    print("=" * 70)

    # 1. Load model + offset
    print("\n1. Loading model + offset...")
    import lightgbm as lgb
    model_path = MODEL_DIR / "wc_match_outcome.txt"
    meta_path = MODEL_DIR / "wc_match_outcome.meta.json"
    offset_path = CALIB_DIR / "neutral_offset.json"

    if not model_path.exists() or not meta_path.exists():
        print("  No trained model found. Run train_worldcup.py first.")
        return
    if not offset_path.exists():
        print("  No offset file found. Run train_worldcup.py or backtest_wc.py first.")
        return

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    with open(offset_path) as f:
        offset = json.load(f)
    features = meta.get("features", [])

    print(f"  Model: {meta.get('n_features', '?')} features, val_brier={meta.get('val_brier', '?'):.4f}")
    print(f"  Offset: Δ_H={offset.get('delta_home', 0):+.3f}  "
          f"Δ_D={offset.get('delta_draw', 0):+.3f}  "
          f"Δ_A={offset.get('delta_away', 0):+.3f}  "
          f"(cap=±{offset.get('cap', 0.15)}, computed on {offset.get('n_val', 0)} val rows)")
    print(f"  Offset val Brier: {offset.get('val_brier_before', '?'):.4f} → "
          f"{offset.get('val_brier_after', '?'):.4f}")

    # 2. Fetch data + filter to 2023+ test set
    print("\n2. Loading 2023+ test set...")
    df = fetch_all_matches()
    elo_df = compute_elo(df)
    elo_df = elo_df.merge(
        df[["match_date", "home_team", "away_team", "tournament_code"]],
        on=["match_date", "home_team", "away_team"],
        how="left",
    )
    test_df = elo_df[elo_df["match_date"] >= pd.Timestamp("2023-01-01")].copy()
    test_df = test_df.sort_values("match_date").reset_index(drop=True)
    print(f"  {len(test_df)} matches in 2023+ test set")

    # Breakdown by tournament
    tc_counts = test_df["tournament_code"].fillna("?").value_counts()
    print(f"\n  By tournament code:")
    for tc, n in tc_counts.head(10).items():
        marker = " [NEUTRAL]" if tc in NEUTRAL_TOURNAMENTS else ""
        print(f"    {tc:6s}  {n:5d}{marker}")

    n_neutral_test = int(test_df["tournament_code"].isin(NEUTRAL_TOURNAMENTS).sum())
    print(f"\n  Neutral-venue rows: {n_neutral_test}/{len(test_df)} "
          f"({n_neutral_test / len(test_df):.1%})")

    # 3. Predict each match (temporal ELO, no look-ahead)
    print("\n3. Precomputing team history (one-time cost)...")
    team_hist = _build_team_history(elo_df)
    print(f"  {len(team_hist)} teams in history index")
    print("\n4. Predicting with proper temporal ELO...")
    raw_probs = []
    offset_probs = []
    actuals = []
    is_neutrals = []
    match_info = []

    cap = offset.get("cap", 0.15)
    dh = max(-cap, min(cap, offset.get("delta_home", 0.0)))
    dd = max(-cap, min(cap, offset.get("delta_draw", 0.0)))
    da = max(-cap, min(cap, offset.get("delta_away", 0.0)))

    for i, match in test_df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_date = match["match_date"]
        tc = match.get("tournament_code", "") or ""

        # ELO ratings as of match date (no look-ahead) — vectorized
        pre_mask = elo_df["match_date"] < match_date
        pre = elo_df[pre_mask]
        elo_ratings = {}
        if not pre.empty:
            elo_ratings.update(dict(zip(pre["home_team"].values, pre["elo_home_post"].values)))
            elo_ratings.update(dict(zip(pre["away_team"].values, pre["elo_away_post"].values)))

        elo_h = elo_ratings.get(home, 1500)
        elo_a = elo_ratings.get(away, 1500)

        # Form features via precomputed team history (O(log n) per team)
        hf = _form_for_team(team_hist, home, match_date, default_elo=elo_h)
        af = _form_for_team(team_hist, away, match_date, default_elo=elo_a)

        # Build feature vector (tournament_code determines is_neutral)
        x = build_feature_vector(elo_h, elo_a, hf, af, tc, features)
        probs = model.predict(x)[0]

        # Actual outcome
        hs, as_ = int(match["home_score"]), int(match["away_score"])
        if hs > as_:
            actual = 0
        elif as_ > hs:
            actual = 2
        else:
            actual = 1

        is_neutral = 1 if tc in NEUTRAL_TOURNAMENTS else 0

        # Save raw probs
        raw_probs.append(probs.copy())

        # Apply offset only at neutral venues (matches scan_wc.py logic)
        offset_p = probs.copy()
        if is_neutral:
            offset_p[0] -= dh
            offset_p[1] -= dd
            offset_p[2] -= da
            offset_p = np.maximum(offset_p, 0.001)
            offset_p = offset_p / offset_p.sum()
        offset_probs.append(offset_p)

        actuals.append(actual)
        is_neutrals.append(is_neutral)
        match_info.append({
            "date": str(match_date.date()),
            "home": home, "away": away, "tc": tc,
            "is_neutral": is_neutral, "actual": actual,
        })

        if (len(raw_probs) % 500) == 0:
            print(f"  ... {len(raw_probs)}/{len(test_df)} matches predicted")

    print(f"  {len(raw_probs)} matches predicted.")

    raw = np.array(raw_probs)
    off = np.array(offset_probs)
    act = np.array(actuals)
    neu = np.array(is_neutrals)

    # 4. Compute Brier scores
    y_onehot = np.zeros((len(act), 3))
    y_onehot[np.arange(len(act)), act] = 1

    def brier(probs):
        return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    def acc(probs):
        return float((np.argmax(probs, axis=1) == act).mean())

    print(f"\n{'='*70}")
    print(f"  RESULTS — 2023+ TEST SET (n={len(act)})")
    print(f"{'='*70}")

    # 4a. Full test set
    brier_raw_full = brier(raw)
    brier_off_full = brier(off)
    acc_raw_full = acc(raw)
    acc_off_full = acc(off)

    print(f"\n  FULL TEST SET (all tournaments):")
    print(f"    Raw    : Brier={brier_raw_full:.4f}  Acc={acc_raw_full:.1%}")
    print(f"    Offset : Brier={brier_off_full:.4f}  Acc={acc_off_full:.1%}")
    print(f"    Δ Brier: {brier_off_full - brier_raw_full:+.4f}  "
          f"Δ Acc: {acc_off_full - acc_raw_full:+.1%}")

    # 4b. Neutral-venue subset
    raw_n = raw[neu == 1]
    off_n = off[neu == 1]
    act_n = act[neu == 1]
    y_onehot_n = np.zeros((len(act_n), 3))
    y_onehot_n[np.arange(len(act_n)), act_n] = 1

    def brier_arr(probs, yo):
        return float(np.mean(np.sum((probs - yo) ** 2, axis=1)))

    brier_raw_n = brier_arr(raw_n, y_onehot_n)
    brier_off_n = brier_arr(off_n, y_onehot_n)
    acc_raw_n = float((np.argmax(raw_n, axis=1) == act_n).mean())
    acc_off_n = float((np.argmax(off_n, axis=1) == act_n).mean())

    print(f"\n  NEUTRAL-VENUE SUBSET (n={neu.sum()}, where offset actually fires):")
    print(f"    Raw    : Brier={brier_raw_n:.4f}  Acc={acc_raw_n:.1%}")
    print(f"    Offset : Brier={brier_off_n:.4f}  Acc={acc_off_n:.1%}")
    print(f"    Δ Brier: {brier_off_n - brier_raw_n:+.4f}  "
          f"Δ Acc: {acc_off_n - acc_raw_n:+.1%}")

    # 4c. Non-neutral control
    raw_h = raw[neu == 0]
    off_h = off[neu == 0]
    act_h = act[neu == 0]
    y_onehot_h = np.zeros((len(act_h), 3))
    y_onehot_h[np.arange(len(act_h)), act_h] = 1
    brier_raw_h = brier_arr(raw_h, y_onehot_h)
    brier_off_h = brier_arr(off_h, y_onehot_h)
    print(f"\n  NON-NEUTRAL SUBSET (n={(neu == 0).sum()}, control — should be identical):")
    print(f"    Raw    : Brier={brier_raw_h:.4f}")
    print(f"    Offset : Brier={brier_off_h:.4f}  (should equal raw, offset not applied)")
    print(f"    Δ      : {brier_off_h - brier_raw_h:+.6f}  (floating-point only)")

    # 5. Per-tournament breakdown (only tournaments with ≥10 matches)
    print(f"\n  PER-TOURNAMENT BREAKDOWN (≥10 matches):")
    print(f"  {'Tournament':<14}{'n':>6}{'Neutral':>10}{'Brier_raw':>12}{'Brier_off':>12}{'Δ Brier':>10}{'Verdict':>12}")
    print(f"  {'-'*14}{'-'*6}{'-'*10}{'-'*12}{'-'*12}{'-'*10}{'-'*12}")
    for tc, n in tc_counts.items():
        if n < 10:
            continue
        mask = np.array([m["tc"] == tc for m in match_info])
        raw_t = raw[mask]
        off_t = off[mask]
        act_t = act[mask]
        if len(act_t) < 5:
            continue
        yo_t = np.zeros((len(act_t), 3))
        yo_t[np.arange(len(act_t)), act_t] = 1
        b_r = brier_arr(raw_t, yo_t)
        b_o = brier_arr(off_t, yo_t)
        is_n = "YES" if tc in NEUTRAL_TOURNAMENTS else "no"
        delta = b_o - b_r
        if tc in NEUTRAL_TOURNAMENTS:
            verdict = "✅ helps" if delta < 0 else "⚠️ hurts" if delta > 0.005 else "─ neutral"
        else:
            verdict = "(skipped)"
            if abs(delta) > 1e-6:
                verdict = f"❌ WHY?"
        print(f"  {tc:<14}{n:>6}{is_n:>10}{b_r:>12.4f}{b_o:>12.4f}{delta:>+10.4f}{verdict:>12}")

    # 6. Verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    if brier_off_n < brier_raw_n:
        improvement = (brier_raw_n - brier_off_n) / brier_raw_n * 100
        print(f"  ✅ Offset GENERALIZES — Brier improves on 2023+ neutral subset by {improvement:+.1f}%")
        print(f"     Raw {brier_raw_n:.4f} → Offset {brier_off_n:.4f}")
        print(f"     Safe to keep applied for WC 2026.")
    elif brier_off_n > brier_raw_n + 0.005:
        worsening = (brier_off_n - brier_raw_n) / brier_raw_n * 100
        print(f"  ❌ Offset is OVERFITTING — Brier WORSENS on 2023+ neutral subset by {worsening:+.1f}%")
        print(f"     Raw {brier_raw_n:.4f} → Offset {brier_off_n:.4f}")
        print(f"     Recommend setting applied=false in neutral_offset.json before WC 2026.")
    else:
        print(f"  ─ Offset is NEUTRAL — Brier change < 0.005 on neutral subset")
        print(f"     Raw {brier_raw_n:.4f} → Offset {brier_off_n:.4f}")
        print(f"     Safe to keep; benefit may be small but no harm.")

    # 7. Save OOS results for the record
    out_path = MODEL_DIR / "offset_oos_2023plus.json"
    out = {
        "n_test": int(len(act)),
        "n_neutral": int(neu.sum()),
        "n_non_neutral": int((neu == 0).sum()),
        "offset_used": {"delta_home": float(dh), "delta_draw": float(dd),
                        "delta_away": float(da), "cap": float(cap)},
        "full": {
            "brier_raw": brier_raw_full, "brier_offset": brier_off_full,
            "acc_raw": acc_raw_full, "acc_offset": acc_off_full,
        },
        "neutral_subset": {
            "brier_raw": brier_raw_n, "brier_offset": brier_off_n,
            "acc_raw": acc_raw_n, "acc_offset": acc_off_n,
            "brier_delta": brier_off_n - brier_raw_n,
            "verdict": "generalizes" if brier_off_n < brier_raw_n else
                       "overfits" if brier_off_n > brier_raw_n + 0.005 else "neutral",
        },
        "non_neutral_subset": {
            "brier_raw": brier_raw_h, "brier_offset": brier_off_h,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
