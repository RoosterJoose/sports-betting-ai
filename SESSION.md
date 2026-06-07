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

## Before running F5/nascar training, regenerate Statcast cache:
```bash
python3 -m src.scripts.fetch_mlb_statcast --seasons 2024 2025 2026
```
(Cached data is gitignored — 302MB)

## Kalshi Status
- **Balance**: $54.09 (check: `python3 -c 'from src.data.kalshi import KalshiClient; c=KalshiClient(); print(f"${c.get_balance():.2f}")'`)
- API key in .env (restore from original machine after clone)

---

## Pipeline Status (June 7, 2026)

### ✅ F5 Monte Carlo Pipeline — Research Prototype
| File | Purpose | Status |
|---|---|---|
| `src/mlb/f5_pa_outcome.py` | 8-class LightGBM PA outcome model | ✅ Trained: 43.76% acc, Brier 0.736 |
| `src/mlb/f5_simulator.py` | Markov Chain MC F5 simulator | ✅ Works: loads model, runs sims |
| `src/scripts/scan_f5.py` | Kalshi F5 scanner (old game model) | ✅ Still uses game-level classifier |

**Current model quality:**
- PA model accuracy: 43.76% (8 classes, class imbalance addressed with sqrt weights)
- Top features: batter_k_rate_prior, batter_bb_rate_prior, pitcher_k_rate_prior
- **Problem**: Rare events (3B=0.4%, HR=3%, HBP=1%) still have near-zero accuracy
- **Monte Carlo approach needs**: better features (xFIP, SIERA, catcher framing) from MLB API — not just Statcast rolling rates
- **For betting**: Use the existing game-level classifier in scan_f5.py (proven 39.6% accuracy on F5 outcomes)

**Next steps for F5:**
- Integrate xFIP/SIERA from MLB stats API (not Statcast)
- Add catcher framing data
- Calibrate Monte Carlo output vs Kalshi markets

### ✅ NASCAR Loop Data Pipeline — Research Prototype
| File | Purpose | Status |
|---|---|---|
| `src/data/nascar_loop.py` | Racing-Reference loop data scraper | ✅ Works: 36 races/season, 19 cols incl Driver Rating |
| `src/features/nascar.py` | Feature engineer with loop data features | ✅ 24 new loop features (driver_rating, avg_running_pos, etc.) |
| `src/scripts/train_nascar.py` | NASCAR model training | ✅ Loop data merged (78% match rate), 48 features |
| `src/scripts/nascar_weekly.py` | Kalshi NASCAR scanner | ✅ Ready |

**Current model quality (with loop data):**
| Model | Accuracy | Brier | Naive Brier | Beats Naive? |
|---|---|---|---|---|
| Win | 92.0% | 0.0511 | 0.026 | ❌ No |
| Top 5 | 83.7% | 0.1362 | 0.110 | ❌ No |
| Top 10 | 72.8% | 0.2251 | 0.191 | ❌ No |

**Analysis**: Models still don't beat naive baseline. Driver Rating (r≈0.614) is predictive per research, but either:
- 78% match rate is too low (22% of rows have missing loop data → default values → noise)
- The rolling features from 3/5/10 race windows are too noisy for such a small season (36 races)
- NASCAR is fundamentally harder to predict than MLB (fewer races, more variance)

**Next steps:**
- Fix the 22% mismatch rate (likely Wikipedia → Racing-Reference name discrepancies)
- Try simpler features (just last-race driver rating, not rolling averages)
- Add qualifying position as a boost factor

### ✅ World Cup Pipeline — Production Ready
- Scanner works, models trained, morning scan integrated
- Auto-activates June 11
- Kalshi balance is $54.09

---

## Key Research Findings (from NotebookLM)

### F5 Architecture
- PA-level Monte Carlo > game-level classifier for fine-grained probability
- Top pitcher features: xFIP, SIERA, K%, BB%
- Catcher framing: +2 runs → 1-2% K probability shift per PA
- For F5: only starter pitcher matters (drop bullpen features)

### NASCAR
- Driver Rating r≈0.614 with finish position
- Avg Running Position is best live/in-race metric
- Sources: Racing-Reference (historical), cf.nascar.com (live qualifying)

### Feature Validation
- Walk-forward validation required (not random k-fold)
- All rolling features must use .shift(1)
- Decay thresholds: recalibrate when ECE > 0.015

## How to Continue
```bash
# Run F5 scan (uses established game-level model)
python3 -m src.scripts.scan_f5 2026-06-07

# Run NASCAR weekly scan
python3 -m src.scripts.nascar_weekly --bankroll 50

# Re-train NASCAR with fixes
python3 -m src.scripts.train_nascar

# Check Kalshi balance
python3 -c 'from src.data.kalshi import KalshiClient; c=KalshiClient(); print(f"${c.get_balance():.2f}")'
```

## Known Issues
1. **PrizePicks API keys** — never found in .env (user said they should be there)
2. **F5 PA model** — rare events (3B, 2B, HBP) still poorly predicted; needs xFIP/SIERA features
3. **NASCAR models** — don't beat naive baseline; need better name matching or simpler features
4. **Statcast cache** — 302MB, gitignored, must regenerate on fresh clone
