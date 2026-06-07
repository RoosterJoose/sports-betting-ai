# Session State — Sports Betting AI

Last updated: 2026-06-07

## Active Build Tracks

### Track 1: F5 Monte Carlo Rebuild (In Progress)
- **Status**: 3 files created, training script has bugs to fix
- **Files created**:
  - `src/mlb/f5_pa_outcome.py` — 8-class LightGBM PA outcome model
  - `src/mlb/f5_simulator.py` — Markov Chain Monte Carlo F5 simulator
- **Known bugs to fix**: 
  - Park factor codes don't match Statcast (3-letter vs 2-letter codes)
  - Batter rolling stats sorted by pitcher, not batter (look-ahead bias)
  - Walk resolution in simulator has bases-loaded bug
- **Next step**: Fix bugs → run training → update `scan_f5.py`

### Track 2: NASCAR Loop Data Rebuild (In Progress)
- **Status**: Scraper created, failed on HTML parsing
- **Files created**:
  - `src/data/nascar_loop.py` — Racing-Reference loop data scraper
- **Known bugs to fix**:
  - Schedule table HTML parsing doesn't match Racing-Reference structure
- **Next step**: Fix URL/parsing → fetch loop data → update feature engineer → re-train

### Track 3: World Cup Launch (Ready, Blocked)
- **Status**: ✅ Scanner works. Models trained. Morning scan integrated.
- **Blocked on**: Needs Kalshi funds ($0.09→$50.09 ✅ Deposited!)
- **Launch**: Auto-activates June 11

### Track 4: NASCAR Re-Train (Not Started)
- **Depends on**: Track 2 completing (need loop data features)

## Kalshi Status
- Balance: $50.09
- API key in .env (not pushed to git)

## Research from NotebookLM
Answers received for all 7 questions covering:
- F5 Monte Carlo PA simulation framework
- NASCAR driver rating as primary feature (r≈0.614)
- Feature validation methodology (walk-forward, shift(1) audit)
- Model decay monitoring (Brier, ECE, CLV tracking)

## How to Continue
```bash
# When starting a new session, run:
cd sports-betting-ai

# To train F5 PA model:
python3 -m src.mlb.f5_pa_outcome

# To test NASCAR loop data scraper:
python3 -m src.data.nascar_loop

# To run F5 scanner:
python3 -m src.scripts.scan_f5

# To re-train NASCAR:
python3 -m src.scripts.train_nascar
```
