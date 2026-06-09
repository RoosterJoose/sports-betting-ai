# Session State — Sports Betting AI

> **⚠️ First command in a new Codebuff session:**
> Read `PROJECT.md` for the complete project bible, then read this file for session context.

Repo: https://github.com/RoosterJoose/sports-betting-ai

## To restore in a new terminal:
```bash
git clone https://github.com/RoosterJoose/sports-betting-ai.git
cd sports-betting-ai
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Files to restore from original machine:
- `.env` — Kalshi API key, DB path
- `kalshi-key.pem` — RSA private key for Kalshi auth
- `data/cache/` — cached game log parquets (gitignored, ~300MB+)

## Kalshi Status
- **Balance**: $54.09 (check: `python3 -c 'from src.data.kalshi import KalshiClient; c=KalshiClient(); print(f"${c.get_balance():.2f}")'`)

---

## Session Work — June 8, 2026

### What We Built Today

**MLB — 100% Complete ✅**
- All 11 models trained, calibrated, backtested, and `info_only=False`
- Fixed critical bugs: `opp_k_pct` temporal leakage, `player_is_lefty` distribution crash
- Added opponent lineup K% and platoon handedness features from Statcast
- Extended backtest + calibration scripts from 4 to 11 models
- 18/19 lines beat naive across SO/HR/TB/H_R_RBI; 8 of 11 models beat naive on ALL lines
- Deleted 5,032 contaminated old trades from the DB (fake edges, 0% win rate)
- IP model is the surprise star: R²=0.850, beats naive 5/5, bias only -0.2%

**NBA — Retrained ✅**
- All 17 XGBoost models were from **June 2024** (2 years stale)
- Retrained with 97K rows (121K total, 4 seasons, 1,162 players)
- Refitted BetaCal for 11 core stats (bias zeroed)
- Scanner loads fresh models correctly (1,476 markets matched across 8 types)
- **Still needs backtest verification before live betting**

**Pipeline Fixes**
- Added IP/OUTS to NB_STATS (was crashing backtest on Normal CDF + arrays)
- Extended `_recency_check` to all 11 MLB models (was only 4)
- Removed WC edge cap (0.35) from morning_scan.py
- Refactored WC feature-building into shared `build_feature_vector()` in `src/data/world_cup.py`

**Documentation**
- Created `PROJECT.md` — complete project bible for fresh AI sessions
- See `PROJECT.md` for: directory structure, model status, commands, gaps, distribution mapping

---

## Current Bottlenecks (P0)

1. **NBA backtest** — models are fresh but not verified to beat naive
2. **WNBA broken** — all models have negative R² (need retraining)
3. **World Cup backtest** — 237 pending trades, model quality unknown
4. **Weather/umpire features** — research says 3-6% ROI, not yet built

---

## Quick Verification Commands

```bash
source .venv/bin/activate

# Verify MLB: all 11 models info_only=False
python3 -c "from src.scripts.kalshi_mlb_unified import MARKET_TYPES; [print(f'{m[\"name\"]:5s} info_only={m.get(\"info_only\",True)}') for m in MARKET_TYPES]"

# Verify NBA: models fresh (today's date)
stat -f "%Sm" models/nba/pts.json

# Verify trade tracker: pending count
sqlite3 data/trade_tracker.db "SELECT status, COUNT(*) FROM trades GROUP BY status"

# Run dry-run morning scan
python -m src.scripts.morning_scan

# Run MLB backtest
python -m src.scripts.backtest_mlb
```
