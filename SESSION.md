# Session State — Sports Betting AI

Repo: https://github.com/RoosterJoose/sports-betting-ai

## First command in a new Codebuff session:
> Read SESSION.md for the full context on where we left off

Then say "Continue building from SESSION.md" to resume work.

## To restore in a new terminal:
```bash
git clone https://github.com/RoosterJoose/sports-betting-ai.git
cd sports-betting-ai
```

## After cloning — restore these files from the original machine:
- `sports-betting-ai/.env` — Kalshi API key, DB path
- `sports-betting-ai/kalshi-key.pem` — RSA private key for Kalshi auth
(These are gitignored for security — they don't come with the clone)

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

### Track 3: World Cup Launch (Ready)
- **Status**: ✅ Scanner works. Models trained. Morning scan integrated.
- **Kalshi**: $50.09 deposited ✅ can trade
- **Launch**: Auto-activates June 11

### Track 4: NASCAR Re-Train (Not Started)
- **Depends on**: Track 2 completing (need loop data features)

## Kalshi Status
- Balance: $50.09 (check with `python3 -c 'from src.data.kalshi import KalshiClient; c=KalshiClient(); print(f"${c.get_balance():.2f}")'`)
- API key in .env (not pushed to git — on local machine at sports-betting-ai/.env)

## Key Research Findings from NotebookLM

### F5 Monte Carlo (replaces current 39.6% accuracy multiclass model)
- **Architecture**: PA-level Monte Carlo simulator, not game-level classifier
- **8 outcomes**: OUT, 1B, 2B, 3B, HR, BB, HBP, K — predict via LightGBM
- **Top pitcher features**: xFIP, SIERA, K%, BB% (normalize HR luck, isolate skill)
- **Catcher framing**: +2 framing runs → 1-2% K probability shift per PA
- **Umpire zone**: 0.5-1.0 run swing per game
- **Park factors**: K/9 park factor (SD=1.08, COL=0.88), HR factor (COL=1.32, SF=0.92)
- **For F5**: Drop bullpen ERA (zero signal), use only starter + catcher + ump

### NASCAR (current models worse than naive baseline)
- **Driver Rating** (year-to-date loop data): correlation r≈0.614 with finish
- **Avg Running Position**: best live/in-race metric (isolates car speed from pit strategy variance)
- **Data sources**: Racing-Reference (historical loop data) + cf.nascar.com Swagger API (live qualifying)
- **Track types**: superspeedway, intermediate, short, road, triangle(Pocono), speedway

### Feature Validation & Decay
- **Must use walk-forward validation** (not random k-fold — creates look-ahead bias)
- **`.shift(1)` audit**: all rolling features must use shift(1) to prevent leakage
- **Label shuffling test**: accuracy should drop 7-10% when outcomes shuffled within game dates
- **Decay thresholds**: recalibrate when CLV goes negative, ECE > 0.015, or Brier stagnates
- **Minimum significance**: 1,000 out-of-sample games for statistical significance

## How to Continue from Here
### Fix critical bugs first:
1. Fix park factor codes in `f5_pa_outcome.py` (use Statcast 3-letter codes)
2. Fix batter rolling stats sort order in `f5_pa_outcome.py`
3. Fix walk resolution in `f5_simulator.py` (base-loaded check before setting on_1b)
4. Fix NASCAR scraper HTML parsing in `nascar_loop.py`

### Then run:
```bash
python3 -m src.mlb.f5_pa_outcome           # Train PA outcome model (~10 min)
python3 -m src.data.nascar_loop             # Fetch NASCAR loop data
python3 -m src.mlb.f5_simulator             # Test simulator
```

### After fixes verified:
```bash
python3 -m src.scripts.scan_f5               # F5 scanner with MC pipeline
python3 -m src.scripts.train_nascar          # Re-train NASCAR with loop data
python3 -m src.scripts.nascar_weekly         # Weekly NASCAR scan (Pocono June 14)
```

## Tools Available
- **Deep research skill**: `.codebuff/skills/deep-research/` (loaded at session start)
- **Web search agent**: `.codebuff/agents/web-search-agent.md`
- **NotebookLM**: 7 questions answered (stored above) — ask Codebuff for details
