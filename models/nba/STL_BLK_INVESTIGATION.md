# STL/BLK NBA Model Investigation

**Date:** 2026-06-09
**Author:** Audit session following formal NBA backtest
**Status:** Investigation complete, recommendations listed (not yet implemented)

---

## TL;DR

Both STL and BLK models underperform the naive baseline (STL 4/7 = 57%, BLK 5/7 = 71% lines beat naive). The **distribution choice is correct** (Poisson for low-rate events) — the regressor is what fails. Root causes are: (1) no opponent defensive features, (2) zero-inflation is treated as a single outcome, (3) 231 features overfit on rare-event noise, (4) playing time dominates everything.

**Both models remain `info_only=True` in the scanner** (gated correctly). No NBA bets are placed on STL or BLK today.

---

## Current State

### Backtest Results (formal `backtest_nba.py`, June 9, 2026)

| Stat | Lines tested | Beats naive | R² | MAE | Residual σ | Dir acc |
|------|-------------|-------------|-----|-----|-----------|---------|
| **STL** | 7 | 4/7 (57%) | 0.1106 | 0.71 | 0.92 | 62.4% |
| **BLK** | 7 | 5/7 (71%) | 0.1891 | 0.51 | 0.71 | 69.2% |
| PTS (for comparison) | 30 | 30/30 (100%) | ~0.45 | 3.0 | 6.15 | — |

### BetaCal Parameters

| Stat | a | b | c | Interpretation |
|------|---|---|---|---------------|
| **STL** | 0.91 | 1.17 | **−0.25** | c=−0.25 is steep — raw model is over-confident, calibrator crushes toward 0.5 |
| **BLK** | 1.22 | 0.70 | +0.25 | More reasonable, but still shifts probabilities substantially |
| PTS | 1.02 | 0.48 | +0.43 | Model is conservative; calibrator expands |

The **STL `c=−0.25`** is the smoking gun: the regressor produces extreme probabilities (0.0 or 1.0) and the calibrator is forced to remap them toward 0.5, destroying signal.

### Top Features (from `*_importance.csv`)

**STL top 5:**
1. `stl_ewm` (0.069) — recent STL rate
2. `stl_log_ewm` (0.051) — log of recent STL
3. `min_avg_3` (0.031) — recent playing time
4. `stl_avg_20` (0.015) — 20-game STL average
5. `min_med_3` (0.012) — recent median minutes

**BLK top 5:**
1. `blk_ewm` (0.093) — recent BLK rate
2. `blk_avg_20` (0.060) — 20-game average
3. `blk_log_ewm` (0.058)
4. `blk_avg_10` (0.010)
5. `pra_combined_avg_20` (0.009) — composite PTS+REB+AST

**Critical observation:** Top features are *all self-statistics*. No opponent defensive vulnerability features appear in the top 20. The existing `opponent_adjustment` only adjusts for the opponent's *average stat* — not the defense's tendency to give up steals/blocks.

---

## Root Causes

### 1. Missing opponent defensive context (HIGHEST IMPACT)
Steals and blocks are opponent-dependent:
- **Steals** correlate strongly with opponent **turnover rate** (TOV%)
- **Blocks** correlate with opponent **rim shot attempts**

The current model has no way to capture "this pitcher/batter faces a steal-prone defense today." Top features are all self-statistics.

### 2. Zero-inflation confuses the model
A 0-STL game could mean:
- Player didn't play (DNP-CD, 0 minutes)
- Player played 30 minutes and recorded 0 steals

Treating these as the same outcome dilutes the signal. The model can't distinguish "didn't get the chance" from "had chance, didn't happen."

### 3. Overdispersion is intrinsic
For STL: mean=0.8, σ=0.92. Pure Poisson would have σ=√0.8=0.89 — so the variance matches Poisson well, but the **count is so low that each steal flips the prediction dramatically**. Predictions in this regime are inherently noisy.

### 4. 231 features = overfitting
Top STL feature has only 6.8% importance. Top 50 features account for ~50% of total importance. The other 181 features are contributing noise. The model has more parameters than signal.

### 5. Playing time dominates everything
`min_avg_3` is the #3 STL feature and a top-5 BLK feature. STL/BLK scale with minutes, but the model treats them as if they were PTS-style volume stats. A bench player with 8 minutes is treated like a starter with 32 minutes.

---

## Distribution Choice — Not the Problem

`src/models/distributions.py:POISSON_STATS` correctly classifies STL, BLK, TOV as **Poisson** (low-rate, count-based events). For low-rate counts, Poisson is the right parametric choice. The issue is **upstream**: the regressor can't produce a good `μ` to feed the Poisson.

**Do not change the distribution** without first addressing the regressor quality.

---

## Concrete Recommendations (ranked by ROI)

### 🥇 1. Add opponent defensive features (HIGH IMPACT, 1-day work)
Compute from the same game logs data, joined by opponent team:
- `opp_stl_allowed_avg_10` — opponent STL given up per game, last 10
- `opp_tov_pct_10` — opponent turnover rate (steals correlate strongly)
- `opp_blk_allowed_avg_10`
- `opp_rim_fga_10` — opponent attempts at rim (drives BLK opportunities; needs shot location data)
- `opp_pace_10` — opponent possessions per game

**Estimated lift:** 5-10% R² for STL, 3-5% for BLK.

### 🥈 2. Two-stage minutes → STL/BLK (MEDIUM-HIGH IMPACT, half-day)
Predict minutes first, then STL/BLK per-minute:
```
STL_pred = (predicted_min / 48) × STL_per_min_predicted
```
Stops playing time from masking the defensive signal. A bench player with 8 min should not be treated like a starter with 32 min.

### 🥉 3. Zero-inflated hurdle model (MEDIUM IMPACT, full day)
Two-stage:
1. Classifier: P(STL > 0 | minutes > 0)
2. NB regression on the non-zero values

Currently the model treats "0 steals in 0 min" and "0 steals in 30 min" as identical — they aren't.

### 4. Drop low-importance features (QUICK WIN, 1 hour)
Filter to top 30-50 features by gain. Current 231-feature models overfit on rare-event noise. Should bump STL R² from 0.11 → ~0.15-0.18.

### 5. Pace-normalized targets (MEDIUM IMPACT)
Train on STL/100 possessions and BLK/100 possessions instead of raw counts. Removes the possession-rate confound from the labels.

### 6. Different distribution (LOW ROI — skip for now)
Poisson is correct. ZINB would help slightly over a hurdle model but adds complexity. Don't pursue until #1 and #2 are in.

---

## Recommended Action Plan

| Step | Effort | Impact | Order |
|------|--------|--------|-------|
| Quick win: feature selection (top 50) | 1 hour | Medium | **Do first** |
| Add opponent defensive features | Half day | High | **Do second** |
| Re-backtest STL/BLK | 30 min | Validation | After #1, #2 |
| Decide on two-stage minutes model | Half day | High | If #1, #2 insufficient |
| Hurdle model | Full day | Medium | Only if above fails |

**Stop condition:** If after #1 + #2 the STL R² is still < 0.20, declare STL "model has no edge" and remove from scanner.

---

## Would I Bet My Own Money?

| Stat | Verdict | Reasoning |
|------|---------|-----------|
| **STL** | ❌ NO | 57% beats naive, R²=0.11, extreme BetaCal c=−0.25 — model has no edge |
| **BLK** | ❌ NO | 71% beats naive, R²=0.19 — slightly better but still unreliable |
| **SB (STL+BLK composite)** | ❌ NO | 75% beats naive but R²=0.17 — composite of two weak models |

**Both models stay `info_only=True`.** The scanner correctly gates them; the bet loop in `nba_bet.py` does not pass them to morning_scan's bet placement.

---

## File Locations

- Model: `models/nba/STL.json`, `models/nba/BLK.json`
- Metrics: `models/nba/STL.metrics.json`, `models/nba/BLK.metrics.json`
- Calibration: `models/nba/stl_beta_cal.json`, `models/nba/blk_beta_cal.json`
- Calibration diagnostics: `models/nba/stl_calibration_diag.json`, `models/nba/blk_calibration_diag.json`
- Feature importances: `models/nba/STL_importance.csv`, `models/nba/BLK_importance.csv`
- Distribution mapping: `src/models/distributions.py` (POISSON_STATS set)
- Feature engineering: `src/features/nba.py` (`NBA_SCARCE = ["stl", "blk"]`)
- Backtest: `src/scripts/backtest_nba.py`

---

## Related Issues

- Same investigation pattern applies to TOV (88% beats naive, R²=0.22, no BetaCal)
- The opponent defensive feature gap likely affects HR (75% beats naive) and possibly PTS/REB/AST to a lesser degree
- The 231-feature overfitting is a general issue across all NBA models — feature selection may help everywhere
