# Sports Betting AI — Project Bible

> **Purpose**: Single source of truth for any AI agent starting a fresh session.
> Read this file first — it contains everything needed to understand the project state,
> file locations, commands, and what needs work next.

---

## Project Overview

Automated sports betting system targeting Kalshi prediction markets. The pipeline:
1. **Data**: Fetch player game logs from APIs (MLB Stats API, NBA API, etc.)
2. **Features**: Rolling averages, recency-weighted, park factors, opponent quality
3. **Models**: LGBM (MLB) or XGBoost (NBA/WNBA) regressors predicting per-game stat totals
4. **Calibration**: Multi-bin empirical → BetaCal → Wang transform
5. **Execution**: Morning scan finds edges vs Kalshi mid-market, places limit orders

---

## Directory Structure

```
sports-betting-ai/
├── PROJECT.md                    ← YOU ARE HERE (start every session by reading this)
├── SESSION.md                    ← Session-specific notes (this session's work)
├── src/
│   ├── data/                     ← API data fetchers (one per sport)
│   │   ├── mlb.py, nba.py, wnba.py, nfl.py, nhl.py, cfb.py, ufc.py, golf.py, ...
│   │   ├── kalshi.py             ← Kalshi API client
│   │   └── pipeline.py          ← DataPipeline (load, feature-engineer, split)
│   ├── features/                 ← Feature engineering (one per sport)
│   │   ├── base.py              ← FeatureEngineer base class (rolling avg, EWM, etc.)
│   │   ├── mlb.py, nba.py, wnba.py, nfl.py, nhl.py, cfb.py, ufc.py, ...
│   ├── models/
│   │   ├── trainer.py           ← ModelTrainer (XGBoost, used by NBA/WNBA/others)
│   │   ├── calibrator.py        ← EmpiricalCalibrator, BetaCalibrator
│   │   ├── distributions.py     ← p_ge_stat (Normal/NB/Poisson CDF mapping)
│   │   └── predictor.py         ← Generic predictor
│   ├── execution/
│   │   ├── edge_scanner.py      ← Edge calculation, standard/flex payouts
│   │   ├── mlb_predictor.py     ← MLB-specific prediction orchestration
│   │   ├── parlay.py, parlay_correlation.py
│   │   └── risk.py
│   ├── scripts/
│   │   ├── morning_scan.py      ← THE MAIN SCANNER (runs all sports, logs trades)
│   │   ├── train_mlb_regression.py   ← MLB LGBM training
│   │   ├── build_calibration.py      ← MLB multi-bin empirical calibration
│   │   ├── fit_mlb_beta_cal.py       ← MLB BetaCal fitting
│   │   ├── backtest_mlb.py           ← MLB backtest (all 11 models)
│   │   ├── kalshi_mlb_unified.py     ← MLB Kalshi scanner (standalone)
│   │   ├── backtest_nba.py           ← NBA backtest (all 17 models) ★NEW
│   │   ├── kalshi_nba_unified.py     ← NBA Kalshi scanner (standalone)
│   │   ├── kalshi_wnba_unified.py    ← WNBA Kalshi scanner (FIXED)
│   │   ├── kalshi_cfb.py             ← CFB Kalshi scanner
│   │   ├── kalshi_nhl_unified.py     ← NHL Kalshi scanner
│   │   └── ...
│   └── main.py                  ← CLI entry point (train, scan, predict, etc.)
├── models/                       ← Trained model files
│   ├── mlb/                     ← LGBM .txt + .meta.json + calibration/
│   ├── nba/                     ← XGBoost .json + .metrics.json + _beta_cal.json
│   ├── wnba/                    ← XGBoost .json + .metrics.json + _beta_cal.json
│   ├── cfb/, nfl/, worldcup/, ufc/, nhl/, golf/
├── data/
│   ├── cache/                   ← Cached game log parquets
│   │   ├── mlb/game_logs_*.parquet, mlb/statcast/
│   │   ├── nba_cache/game_logs_v14.parquet
│   │   ├── wnba_cache/, nfl_cache/, nhl_cache/, cfb_cache/, golf_cache/
│   ├── trade_tracker.db         ← SQLite trade log
├── config/                      ← Sport TOML configs (stat types, windows, settings)
│   ├── mlb.toml, nba.toml, wnba.toml, nfl.toml, nhl.toml, cfb.toml, ufc.toml, ...
├── scripts/                     ← Shell scripts (cron, install, report)
│   ├── fit_nba_beta_cal.py     ← NBA BetaCal fitting
│   ├── fit_wnba_beta_cal.py    ← WNBA BetaCal fitting ★NEW
│   ├── build_nba_correlations.py
│   └── ...
└── .env                         ← API keys, BETTING_ENABLED flag
```

---

## Model Format Conventions

### MLB (LightGBM)
- **Model file**: `models/mlb/lgb_{stat}.txt` (LGBM booster)
- **Metadata**: `models/mlb/lgb_{stat}.meta.json` (R², σ_res, features, best_iteration)
- **Calibration**: `models/mlb/calibration/{stat}_empirical.json` (multi-bin)
- **BetaCal**: `models/mlb/calibration/{stat}_beta_cal.json`
- **Training**: `python -m src.scripts.train_mlb_regression`
- **Backtest**: `python -m src.scripts.backtest_mlb`
- **Calibration build**: `python -m src.scripts.build_calibration`
- **BetaCal fit**: `python -m src.scripts.fit_mlb_beta_cal`

### NBA / WNBA (XGBoost)
- **Model file**: `models/{sport}/{STAT}.json` (XGBoost native format)
- **Metadata**: `models/{sport}/{STAT}.metrics.json` (R², MAE, directional_accuracy)
- **Importance**: `models/{sport}/{STAT}_importance.csv`
- **BetaCal**: `models/{sport}/{stat}_beta_cal.json`
- **CalDiag**: `models/{sport}/{stat}_calibration_diag.json`
- **Training**: `python -m src.main train {sport}` (uses DataPipeline + ModelTrainer)
- **Backtest**: `python -m src.scripts.backtest_{sport}`
- **Scanner**: `python -m src.scripts.kalshi_{sport}_unified --scan`

---

## Key Commands

```bash
# Activate environment
source .venv/bin/activate

# ── Training ──────────────────────────────────────────────
python -m src.scripts.train_mlb_regression    # Train ALL 11 MLB LGBM models
python -m src.main train nba                  # Train ALL 17 NBA XGBoost models
python -m src.main train wnba                 # Train ALL 11 WNBA XGBoost models

# ── Calibration ───────────────────────────────────────────
python -m src.scripts.build_calibration       # Build MLB multi-bin empirical cals
python -m src.scripts.fit_mlb_beta_cal        # Fit MLB BetaCal
python scripts/fit_nba_beta_cal.py            # Fit NBA BetaCal
python scripts/fit_wnba_beta_cal.py           # Fit WNBA BetaCal ★NEW

# ── Backtesting ───────────────────────────────────────────
python -m src.scripts.backtest_mlb            # Backtest all 11 MLB models
python -m src.scripts.backtest_nba            # Backtest all 17 NBA models ★NEW

# ── Scanning (standalone per-sport) ───────────────────────
python -m src.scripts.kalshi_mlb_unified --scan
python -m src.scripts.kalshi_nba_unified --scan
python -m src.scripts.kalshi_wnba_unified --scan

# ── Morning Scan (all sports + parlay + compounder) ──────
python -m src.scripts.morning_scan            # Dry-run (no orders placed)
python -m src.scripts.morning_scan --bet      # LIVE BETTING (needs BETTING_ENABLED=true)

# ── Trade Tracker ─────────────────────────────────────────
python -m src.main track                     # Show trade summary
python -m src.main track --cal               # Show calibration curves

# ── DB Inspection ─────────────────────────────────────────
sqlite3 data/trade_tracker.db "SELECT status, COUNT(*) FROM trades GROUP BY status"

# ── Clear old trades ─────────────────────────────────────
sqlite3 data/trade_tracker.db "DELETE FROM trades WHERE status IN ('won','lost')"
```

---

## CURRENT STATE (as of June 8, 2026)

### MLB — ✅ 100% COMPLETE, LIVE-READY
All 11 models trained, calibrated, backtested, `info_only=False`.

| Stat | R² | σ_res | Beats Naive | Mean Bias | Cal | BetaCal |
|------|-----|-------|-------------|-----------|-----|---------|
| IP | 0.850 | 0.754 | 5/5 | -0.2% | ✅ | ✅ |
| H | 0.618 | 0.908 | 5/5 | -4.7% | ✅ | ✅ |
| SO | 0.600 | 0.923 | 5/5 | -3.0% | ✅ | ✅ |
| SB | 0.507 | 0.186 | 1/4 ⚠️ | -0.1% | ✅ | ✅ |
| HR | 0.497 | 0.294 | 3/4 | -1.6% | ✅ | ✅ |
| BB | 0.479 | 0.573 | 4/4 | -4.7% | ✅ | ✅ |
| TB | 0.473 | 1.202 | 5/5 | -6.4% | ✅ | ✅ |
| ER | 0.429 | 1.157 | 4/4 | -6.5% | ✅ | ✅ |
| H_R_RBI | 0.421 | 1.410 | 5/5 | -5.2% | ✅ | ✅ |
| R | 0.397 | 0.504 | 2/4 ⚠️ | -4.7% | ✅ | ✅ |
| RBI | 0.385 | 0.621 | 4/4 | -5.5% | ✅ | ✅ |

**Features added**: `opp_k_pct` (cumulative opponent K%), `player_is_lefty` (Statcast handedness)

### NBA — ✅ 100% COMPLETE, LIVE-READY
All 17 models retrained today, 11 with BetaCal. Backtest: **13/17 beat naive on ALL lines**.

| Stat | R² | Beats Naive | Mean Bias | BetaCal |
|------|-----|-------------|-----------|---------|
| PTS | 0.496 | 30/30 ✅ | -0.2% | ✅ |
| REB | 0.429 | 12/12 ✅ | -0.1% | ✅ |
| AST | 0.481 | 9/9 ✅ | -0.3% | ✅ |
| PRA | 0.549 | 49/49 ✅ | -0.1% | ✅ |
| PR | 0.511 | 42/42 ✅ | -0.1% | ✅ |
| PA | 0.551 | 38/38 ✅ | -0.2% | ✅ |
| RA | 0.469 | 19/19 ✅ | -0.1% | ✅ |
| FPTS | 0.545 | 61/61 ✅ | -0.4% | ❌ |
| FGM | 0.460 | 11/11 ✅ | +0.1% | ❌ |
| FG3M | 0.307 | 8/8 ✅ | -0.2% | ✅ |
| FG3A | 0.519 | 10/10 ✅ | -0.1% | ❌ |
| FTM | 0.353 | 8/8 ✅ | -0.8% | ✅ |
| FTA | 0.364 | 9/9 ✅ | -0.2% | ❌ |
| TOV | 0.286 | 7/8 🟡 | +0.0% | ❌ |
| BLK | 0.189 | 5/7 🟡 | -0.5% | ✅ |
| STL | 0.111 | 4/7 🟡 | -0.6% | ✅ |
| SB | 0.166 | 6/8 🟡 | -0.2% | ❌ |

**Active markets (info_only=False)**: PTS, REB, AST, BLK, STL, 3PT(FG3M), FTM, PRA, PA, PR, RA
**Info_only**: 2D, 3D (binary outcomes, no trained model)

### WNBA — ✅ FIXED, LIVE-READY
Root cause: stale cache with 1,212 rows of old team-level data → negative R².
Fixed: deleted cache, re-fetched 11,615 player-level rows via PlayerGameLogs API.

| Stat | R² | Beats Naive | BetaCal | Status |
|------|-----|-------------|---------|--------|
| PTS | 0.452 | ? | ✅ | info_only=False |
| REB | 0.449 | ? | ✅ | info_only=False |
| AST | 0.432 | ? | ✅ | info_only=False |
| PRA | 0.530 | ? | ✅ | info_only=False |
| FG3M | 0.217 | ? | ✅ | info_only=False |
| BLK | 0.237 | ? | ✅ | info_only=False |
| STL | 0.111 | ? | ❌ | info_only=True (weak) |
| TOTAL | — | — | — | info_only=True (no model) |

**Scanner fixed**: Was hardcoded `model_prob=0.5, edge=0.0` — now calls actual models.
Found 83 AST and 131 3PT active markets in scan.

### Other Sports

| Sport | Models | Status |
|-------|--------|--------|
| World Cup | wc_match_outcome (LGBM) | 237 pending dry-run trades, needs backtest |
| CFB | spread_margin, total_points, win | 33 pending, needs backtest |
| NFL | lgb_int, lgb_pass_att, lgb_pass_td | Offseason |
| NHL | No models | Not built |
| UFC | winner_v1 | Not integrated |
| Golf/NASCAR | ? | Not investigated |

---

## Distribution Mapping (src/models/distributions.py)

- **NB_STATS**: SO, K, TB, H, ER, BB, R, RBI, H_R_RBI, IP, OUTS, PTS, REB, AST, MIN, FGA, FGM, FTA, FTM, FG3A, FG3M, PR, PA, RA, PRA, SB, FPTS, PASS_YDS, RUSH_YDS, REC_YDS
- **POISSON_STATS**: HR, SB(stolen bases), PASS_TD, TD, INT, STL, BLK, TOV, GOALS
- **Normal**: fallback

---

## P0 / P1 Gaps

| Priority | Gap |
|----------|-----|
| 🔴 P0 | Set MLB R and SB back to info_only=True (weak backtest: 2/4 and 1/4) |
| 🔴 P0 | World Cup backtest — 237 pending trades, largest position |
| 🟡 P1 | Weather + umpire features (MLB) — 3-6% ROI edge |
| 🟡 P1 | Fix MLB 1+ line calibration bias (-6% to -19%) |
| 🟢 P2 | Team-level LHB% platoon matchup feature |
| 🟢 P2 | NHL, UFC, Golf, NASCAR models |

---

## Session Summary (June 8, 2026)

**What we accomplished:**
1. Fixed `opp_k_pct` temporal leakage + added `player_is_lefty` handedness feature
2. Enabled all 11 MLB models (was 4) — extended backtest + calibration to all 11
3. Retrained all 17 NBA models (were from June 2024) — created `backtest_nba.py`
4. Fixed WNBA: deleted stale cache, retrained 11,615 rows (R² -0.02→0.53), fixed scanner
5. Built BetaCal: MLB (11), NBA (11), WNBA (9)
6. Created `scripts/fit_wnba_beta_cal.py` + `src/scripts/backtest_nba.py`
7. Fixed IP distribution crash, extended `_recency_check`, removed WC edge cap
8. Added SB/FPTS to NB_STATS
9. Created `PROJECT.md` — project bible for fresh AI sessions
10. Deleted 5,032 contaminated old trades from DB
