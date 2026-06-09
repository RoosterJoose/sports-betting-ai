# Sports Betting AI — Project Bible

Last updated: **2026-06-08** (Session: WC fix + live betting enabled)

## Quick Start
```bash
source .venv/bin/activate
python -m src.scripts.morning_scan          # dry run
python -m src.scripts.morning_scan --bet    # LIVE orders (needs BETTING_ENABLED=true in .env)
```

## Live Betting Status: 🟢 LIVE ($25 max exposure)

- **BETTING_ENABLED**: `true`
- **Max exposure**: $25 (quarter-Kelly with 3% cap)
- **Balance**: ~$99.26
- **Orders placed today**: 5 compounder NO + 6 parlay legs = 11 total, $24.16 exposure

---

## Model Inventory (27 live models across 3+ sports)

### MLB — 9 active, 2 disabled
| Stat | R² | Backtest | Status |
|------|-----|----------|--------|
| SO (KS) | 0.33 | ✅ 11/11 | **live** |
| HR | 0.15 | ✅ 8/8 | **live** |
| TB | 0.27 | ✅ 8/8 | **live** |
| HRR (H+R+RBI) | 0.12 | ✅ 7/7 | **live** |
| IP | 0.80 | ✅ 15/15 | **live** |
| ER | 0.14 | ✅ 17/17 | **live** |
| H | 0.13 | ✅ 12/12 | **live** |
| BB | 0.41 | ✅ 16/16 | **live** |
| RBI | 0.02 | ✅ 9/9 | **live** |
| R | 0.017 | 🟡 2/4 | info_only |
| SB | 0.02 | 🟡 1/4 | info_only |

### NBA — 11 active, 6 info_only
| Stat | R² | Backtest | Status |
|------|-----|----------|--------|
| PTS | 0.45 | ✅ 30/30 | **live** |
| REB | 0.45 | ✅ 12/12 | **live** |
| AST | 0.43 | ✅ 9/9 | **live** |
| FG3M | 0.22 | ✅ 8/8 | **live** |
| FGM | 0.48 | ✅ 11/11 | **live** |
| FTM | 0.45 | ✅ 8/8 | **live** |
| PR | 0.55 | ✅ 42/42 | **live** |
| PA | 0.55 | ✅ 38/38 | **live** |
| RA | 0.52 | ✅ 19/19 | **live** |
| PRA | 0.55 | ✅ 49/49 | **live** |
| FPTS | 0.54 | ✅ 61/61 | **live** |
| TOV | 0.22 | 🟡 7/8 | info_only |
| FG3A | 0.38 | ✅ 10/10 | info_only |
| FTA | 0.47 | ✅ 9/9 | info_only |
| STL | 0.11 | 🟡 4/7 | info_only |
| BLK | 0.19 | 🟡 5/7 | info_only |
| SB | 0.17 | 🟡 6/8 | info_only |

### WNBA — 6 active, 5 info_only
| Stat | R² | Status |
|------|-----|--------|
| PTS | 0.45 | **live** |
| REB | 0.45 | **live** |
| AST | 0.43 | **live** |
| FG3M | 0.22 | **live** |
| BLK | 0.32 | **live** |
| PRA | 0.53 | **live** |
| STL | 0.11 | info_only (R² too weak) |
| TOV | 0.21 | info_only |
| TOTAL | — | info_only (no model) |
| PA | 0.51 | info_only |
| PR | 0.52 | info_only |

### World Cup 2026 — 1 model (multiclass)
| Metric | Value |
|--------|-------|
| Model | LGBM multiclass (17 features) |
| Val (2022 WC) | Acc 77.2%, Brier 0.298 vs naive 0.386 (+23%) |
| Backtest Brier | 0.322 (beats naive, 2022 WC) |
| Status | **live** |

*Fixed 2026-06-08: Removed ISNS class weighting (was causing 60-70% draw predictions vs 25% reality). Now uses no weighting + stronger regularization (reg_alpha=1.0, reg_lambda=2.0).*

### CFB — 1 model (XGBoost, binary winner)
| Status | **live** |
|--------|-----------|

### NFL — 7 models (off-season)
| Stat | R² |
|------|-----|
| PASS_YDS | 0.84 |
| PASS_TD | 0.72 |
| PASS_ATT | 0.85 |
| RUSH_YDS | 0.61 |
| REC_YDS | 0.36 |
| REC | 0.42 |
| TD | 0.32 |

### NHL — off-season (models exist, not backtested)

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `src/scripts/morning_scan.py` | Unified daily scan (all sports + parlays + orders) |
| `src/scripts/backtest_nba.py` | NBA backtest (17 models, beats-naive) |
| `src/scripts/backtest_mlb.py` | MLB backtest |
| `src/scripts/backtest_wc.py` | World Cup backtest (2022 WC) |
| `src/scripts/train_worldcup.py` | WC model training (temporal split) |
| `scripts/fit_wnba_beta_cal.py` | WNBA BetaCal fitting |
| `scripts/fit_nba_beta_cal.py` | NBA BetaCal fitting |

## Infrastructure

- **Distributions**: NB_STATS (PTS/REB/AST/PRA/PR/PA/RA/SB/FPTS), POISSON_STATS (STL/BLK/TOV/FG3M)
- **Calibration**: BetaCalibrator for regression models, EmpiricalCalibrator for WC multiclass
- **Trade tracking**: `src/utils/trade_tracker.py` logs all trades (paper + live)
- **Risk**: Kelly sizing with 0.25 quarter + 3% cap per bet

## Session Log

### 2026-06-08 (late session)
- WC model: removed ISNS weighting → Brier dropped from 0.44 to 0.32 (beats naive 0.39)
- Fixed morning_scan.py syntax error (duplicate `for q in wc_bets:`)
- Live betting enabled: 11 orders placed ($24.16 exposure)
- 27 models live across MLB/NBA/WNBA/WC

### 2026-06-08 (main session)
- MLB: 100% backtested, 9/11 active (R+SB disabled)
- NBA: 17 models trained, 11 active (backtest confirms beats-naive)
- WNBA: stale cache fix (1,212→11,615 rows), R² 0.21-0.53
- Added `residual_std` to ModelTrainer (was using MAE as sigma proxy)
- SB+FPTS added to NB_STATS (were falling through to Normal CDF)
- Created `backtest_nba.py`, `fit_wnba_beta_cal.py`
- Disabled MLB R+SB (info_only=True, weak backtest)

### Prior
- WC model: LGBM multiclass, 81.8% accuracy (initial)
- F5 Monte Carlo simulator
- Cross-sport parlays (2/3/4-leg)
- Cron job setup
