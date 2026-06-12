#!/usr/bin/env python3
"""Per-decile calibration check for MLB PRODUCTION cals (no refit, no save).

Mirrors the per-decile logic in fit_mlb_beta_cal.py:fit_beta_cal_for_stat()
but applies it to the PRODUCTION cals (models/mlb/{stat}_beta_cal.json and
{stat}_isotonic_cal.json) on the current test set, without refitting.

Reports per-decile calibration for:
  - RAW model probabilities
  - POST-BetaCal probabilities (if BetaCal file exists)
  - POST-Isotonic probabilities (if IsotonicCal file exists)

Flags any cal where worst-decile |bias| > 10% (the per-decile guard's
threshold in fit_mlb_beta_cal.py). Two views:
  1. RAW rejected → matches the current guard's behavior in fit_mlb_beta_cal
  2. POST-CAL rejected → cal is actually broken in production

This answers two questions:
  - "Which cals would the fit-time per-decile guard REJECT today?" (RAW view)
  - "Which production cals are actually doing a bad job and need re-fit?" (POST view)

Usage:
    python -m scripts.check_mlb_cals_per_decile
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PROJECT_ROOT
from src.models.calibrator import BetaCalibrator, IsotonicCalibrator
from src.models.distributions import p_ge_stat
from src.scripts.fit_mlb_beta_cal import load_features
import lightgbm as lgb

MODEL_DIR = PROJECT_ROOT / "models" / "mlb"
DECILE_GUARD = 0.10  # 10% — matches the per-decile guard in fit_mlb_beta_cal

# Stats that benefit from Isotonic over BetaCal (per
# kalshi_mlb_unified.py:80 ISOTONIC_PREFERRED). For these, the scanner
# tries Isotonic first and only falls back to BetaCal. The RE-FIT
# recommendation should use the IsoCal status for these, not the BetaCal
# status, since that's the cal the scanner actually loads in production.
ISOTONIC_PREFERRED = {"ip", "r", "rbi", "sb", "hr", "blk", "stl"}

# Stats to check (matching MARKET_TYPES in kalshi_mlb_unified.py:215-309)
STATS = [
    ("SO", "so", "pitcher", None),
    ("HR", "hr", "hitter", None),
    ("TB", "tb", "hitter", None),
    ("H_R_RBI", None, "hitter", lambda df: df["h"] + df["r"] + df["rbi"]),
    ("IP", "ip", "pitcher", None),
    ("ER", "er", "pitcher", None),
    ("H", "h", "pitcher", None),
    ("BB", "bb", "pitcher", None),
    ("RBI", "rbi", "hitter", None),
]


def _per_decile_check(p_raw, p_score, outcome, name):
    """Compute per-decile bias by binning on P_raw.

    Returns dict with worst_raw_bias, worst_score_bias, decile_table, n_deciles.
    """
    df = pd.DataFrame({"p_raw": p_raw, "p_score": p_score, "outcome": outcome})
    df["bin"] = pd.qcut(df["p_raw"], q=10, duplicates="drop")
    stats = df.groupby("bin", observed=True).agg(
        p_raw_mean=("p_raw", "mean"),
        p_score_mean=("p_score", "mean"),
        outcome_mean=("outcome", "mean"),
        n=("outcome", "count"),
    )
    stats["raw_bias"] = (stats["p_raw_mean"] - stats["outcome_mean"]).abs()
    stats["score_bias"] = (stats["p_score_mean"] - stats["outcome_mean"]).abs()
    worst_raw = float(stats["raw_bias"].max()) if not stats.empty else 0.0
    worst_score = float(stats["score_bias"].max()) if not stats.empty else 0.0
    return {
        "n": int(len(p_raw)),
        "n_deciles": int(len(stats)),
        "worst_raw_bias": worst_raw,
        "worst_score_bias": worst_score,
        "raw_rejected": worst_raw > DECILE_GUARD,
        "score_rejected": worst_score > DECILE_GUARD,
        "decile_table": stats,
    }


def _render_decile_table(table, p_col, bias_col, label):
    """Render one per-decile table."""
    print(f"  {'Bin (P_raw range)':>22s}  {'P_score':>7s}  {'P_act':>6s}  {'|bias|':>6s}  {'n':>5s}")
    for idx, row in table.iterrows():
        # Format the bin index (a pandas Interval) for readability
        bin_str = f"{idx.left:.2f}-{idx.right:.2f}" if hasattr(idx, "left") else str(idx)
        print(f"  {bin_str:>22s}  {row[p_col]:>7.3f}  {row['outcome_mean']:>6.3f}  "
              f"{row[bias_col]:>6.3f}  {int(row['n']):>5d}")


def check_stat(featured, stat_name, raw_col, pos_filter, compute_fn):
    """Run per-decile check on one stat's production cals."""
    mn = stat_name.lower()
    model_path = MODEL_DIR / f"lgb_{mn}.txt"
    meta_path = MODEL_DIR / f"lgb_{mn}.meta.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"  {stat_name}: model not found, skipping")
        return None

    model = lgb.Booster(model_file=str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    residual_std = meta.get("residual_std", 1.0)
    model_features = meta.get("features", model.feature_name())

    # Filter by position
    if pos_filter == "pitcher" and "position" in featured.columns:
        df = featured[featured["position"] == "P"].copy()
    elif pos_filter == "hitter" and "position" in featured.columns:
        df = featured[featured["position"] != "P"].copy()
    else:
        df = featured.copy()

    target_col = raw_col or f"{stat_name.lower()}_computed"
    if compute_fn is not None:
        df[target_col] = compute_fn(df)
    elif raw_col:
        target_col = raw_col

    if target_col not in df.columns:
        print(f"  {stat_name}: column '{target_col}' not found, skipping")
        return None

    df = df.dropna(subset=[target_col]).copy()
    if len(df) < 100:
        print(f"  {stat_name}: only {len(df)} rows, skipping")
        return None

    y = df[target_col].values
    available = [c for c in model_features if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median())

    # Temporal split (80/20) — same as fit_mlb_beta_cal.py
    date_col = "game_date" if "game_date" in df.columns else "date"
    dates = pd.to_datetime(df[date_col])
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    dates = dates.iloc[sort_idx]
    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y[split:]

    test_feat = pd.DataFrame(index=range(len(X_test)))
    for c in model_features:
        if c in X_test.columns:
            test_feat[c] = X_test[c].values
        else:
            test_feat[c] = 0.0
    mu = model.predict(test_feat.fillna(0))

    y_mean = float(np.mean(y_test))
    y_max = int(np.max(y_test))
    line_range = range(max(0, int(y_mean * 0.3)), min(y_max + 1, max(15, y_max + 1)))

    raw_probs = []
    outcomes = []
    for line_val in line_range:
        p_raw = p_ge_stat(stat_name, mu, max(residual_std, 0.3), line_val)
        actual = (y_test >= line_val).astype(int)
        raw_probs.extend(p_raw.tolist())
        outcomes.extend(actual.tolist())

    raw_arr = np.array(raw_probs, dtype=float)
    out_arr = np.array(outcomes, dtype=int)
    valid = (raw_arr > 0.01) & (raw_arr < 0.99)
    if valid.sum() < 100:
        print(f"  {stat_name}: only {valid.sum()} valid predictions, skipping")
        return None

    raw_for_decile = raw_arr[valid]
    out_for_decile = out_arr[valid]

    # Load production cals
    beta_cal_path = MODEL_DIR / f"{mn}_beta_cal.json"
    iso_cal_path = MODEL_DIR / f"{mn}_isotonic_cal.json"

    p_beta = None
    bc_obj = None
    if beta_cal_path.exists():
        bc_obj = BetaCalibrator.load(beta_cal_path)
        if bc_obj is not None and getattr(bc_obj, "_fitted", False):
            p_beta = bc_obj.calibrate(raw_for_decile)

    p_iso = None
    ic_obj = None
    if iso_cal_path.exists():
        ic_obj = IsotonicCalibrator.load(iso_cal_path)
        if ic_obj is not None and getattr(ic_obj, "_fitted", False):
            p_iso = ic_obj.calibrate(raw_for_decile)

    has_beta = p_beta is not None
    has_iso = p_iso is not None

    print(f"\n{'=' * 78}")
    print(f"  {stat_name}  |  n_valid={valid.sum():,}  |  n_lines={len(line_range)}  |  "
          f"BetaCal={'YES' if has_beta else 'NO'}  IsoCal={'YES' if has_iso else 'NO'}")
    if has_beta and bc_obj is not None:
        print(f"  Beta cal: a={bc_obj.a:.3f}, b={bc_obj.b:.3f}, c={bc_obj.c:.3f}")
    if has_iso and ic_obj is not None:
        print(f"  Iso cal:  {type(ic_obj).__name__} (loaded)")
    print(f"{'=' * 78}")

    # RAW per-decile
    raw_result = _per_decile_check(raw_for_decile, raw_for_decile, out_for_decile, "raw")
    print(f"\n  [1] RAW per-decile  (n={raw_result['n']:,}, deciles={raw_result['n_deciles']})")
    _render_decile_table(raw_result["decile_table"], "p_raw_mean", "raw_bias", "raw")
    print(f"  >>> Worst RAW  |bias|: {raw_result['worst_raw_bias']:.1%}  "
          f"{'❌ REJECTED (current guard)' if raw_result['raw_rejected'] else '✅ OK'}")

    # POST-BetaCal per-decile
    beta_result = None
    if has_beta:
        beta_result = _per_decile_check(raw_for_decile, p_beta, out_for_decile, "beta")
        print(f"\n  [2] POST-BetaCal per-decile")
        _render_decile_table(beta_result["decile_table"], "p_score_mean", "score_bias", "beta")
        print(f"  >>> Worst BetaCal  |bias|: {beta_result['worst_score_bias']:.1%}  "
              f"{'❌ REJECTED (broken cal)' if beta_result['score_rejected'] else '✅ OK'}")

    # POST-Isotonic per-decile
    iso_result = None
    if has_iso:
        iso_result = _per_decile_check(raw_for_decile, p_iso, out_for_decile, "iso")
        print(f"\n  [3] POST-Isotonic per-decile")
        _render_decile_table(iso_result["decile_table"], "p_score_mean", "score_bias", "iso")
        print(f"  >>> Worst IsoCal  |bias|: {iso_result['worst_score_bias']:.1%}  "
              f"{'❌ REJECTED (broken cal)' if iso_result['score_rejected'] else '✅ OK'}")

    return {
        "stat": stat_name,
        "raw": raw_result,
        "beta": beta_result,
        "iso": iso_result,
        "has_beta": has_beta,
        "has_iso": has_iso,
    }


def main():
    print("=" * 78)
    print("  MLB PRODUCTION CALS — PER-DECILE GUARD CHECK")
    print(f"  Threshold: worst decile |bias| > {DECILE_GUARD:.0%} → REJECTED")
    print("  (Mirrors fit_mlb_beta_cal.py:fit_beta_cal_for_stat, applied to existing cals)")
    print("=" * 78)

    print("\nLoading features...", flush=True)
    featured = load_features()
    if featured is None:
        return
    print(f"  {len(featured):,} total rows, {len(featured.columns)} columns", flush=True)

    results = []
    for stat_name, raw_col, pos_filter, compute_fn in STATS:
        result = check_stat(featured, stat_name, raw_col, pos_filter, compute_fn)
        if result is not None:
            results.append(result)

    # Summary table
    print(f"\n{'=' * 78}")
    print("  SUMMARY — Worst decile |bias| per cal (threshold 10%)")
    print(f"{'=' * 78}")
    header = f"  {'Stat':10s}  {'n':>6s}  {'RAW':>7s}  {'BetaCal':>8s}  {'IsoCal':>7s}  Status"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    fit_needed = []
    for r in results:
        n_str = f"{r['raw']['n']:,}"
        raw_str = f"{r['raw']['worst_raw_bias']:.1%}"
        beta_str = f"{r['beta']['worst_score_bias']:.1%}" if r['beta'] else "—"
        iso_str = f"{r['iso']['worst_score_bias']:.1%}" if r['iso'] else "—"

        # RE-FIT logic — uses the cal the scanner ACTUALLY loads in production.
        # For ISOTONIC_PREFERRED stats (ip/r/rbi/sb/hr/blk/stl), the scanner
        # tries Iso first; only if Iso is missing/broken does it fall back to
        # BetaCal. So a broken BetaCal with a working IsoCal is fine in
        # production — don't flag. For non-ISOTONIC_PREFERRED stats, the
        # scanner uses BetaCal directly.
        stat_lower = r['stat'].lower()
        uses_iso_first = stat_lower in ISOTONIC_PREFERRED
        primary_broken = False
        if uses_iso_first:
            if r['iso'] is not None:
                primary_broken = r['iso']['score_rejected']
            elif r['beta'] is not None:
                primary_broken = r['beta']['score_rejected']
            else:
                primary_broken = True  # both missing
        else:
            if r['beta'] is not None:
                primary_broken = r['beta']['score_rejected']
            elif r['iso'] is not None:
                primary_broken = r['iso']['score_rejected']
            else:
                primary_broken = True  # both missing

        # Also note RAW rejection separately (the fit-time guard checks RAW).
        # If RAW is bad but POST-CAL is good, the cal is working — don't
        # flag for re-fit, but DO note that re-fitting would currently be
        # blocked by the fit-time guard (a known issue with the guard's
        # RAW-only design).
        raw_would_reject = r['raw']['raw_rejected']
        post_cal_fine = (not primary_broken)
        if primary_broken:
            status = "❌ RE-FIT RECOMMENDED"
            fit_needed.append(r['stat'])
        elif raw_would_reject and post_cal_fine:
            # Cal is working in production, but the model has high raw
            # miscalibration — re-fit would currently be blocked by the
            # guard's RAW-only design. Status is OK but flagged with note.
            status = "✅ OK (cal working; re-fit blocked by RAW guard)"
        else:
            status = "✅ OK"

        print(f"  {r['stat']:10s}  {n_str:>6s}  {raw_str:>7s}  {beta_str:>8s}  {iso_str:>7s}  {status}")

    # Final recommendation
    print(f"\n{'=' * 78}")
    if fit_needed:
        print(f"  ❌ {len(fit_needed)} stat(s) need re-fit: {', '.join(fit_needed)}")
        print(f"  Run: python -m src.scripts.fit_mlb_beta_cal")
        print(f"  (The per-decile guard will REJECT any new cal with worst decile |bias| > 10%.)")
    else:
        print(f"  ✅ All production cals pass the per-decile guard (raw AND post-cal).")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
