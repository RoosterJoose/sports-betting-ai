# Consensus 4-Leg Parlay Backtest (2022-2024)

**Generated:** 2026-06-12T10:04:03.647762

## Strategy
Replay the consensus-favorite cross-sport parlay (USA + Brazil + BAL + CLE) on 2022-2024 historical data. For each game, identify the consensus favorite using a proxy (MLB: home team + record differential; WC: higher pre-game Elo), stratify by implied probability tier, and report empirical win rates. Combine via independence to get the joint probability of a 4-leg parlay.

## The 4 legs
| Leg | Sport | Match | Implied | Model | Research |
|---|---|---|---|---|---|
| 1 | WC | USA vs Paraguay | 52% | 0.613 | MODERATE |
| 2 | WC | Brazil vs Morocco | 73% | 0.832 | STRONG |
| 3 | MLB | ATL @ BAL | 58% | - | MODERATE (research) |
| 4 | MLB | CLE @ CIN | 60% | - | STRONG |

## MLB 2022-2024 — favorite win rate by probability tier
| Tier | N | Wins | Win rate | Implied mid | Cal err |
|---|---:|---:|---:|---:|---:|
| Light favorite (~50-55%) | 1677 | 871 | 51.9% | 50.0% | +1.9% |
| Moderate favorite (~55-65%) | 4267 | 2432 | 57.0% | 60.0% | -3.0% |
| Strong favorite (~65-75%) | 1161 | 735 | 63.3% | 70.0% | -6.7% |
| Heavy favorite (~75%+) | 52 | 31 | 59.6% | 88.0% | -28.4% |

## WC 2022 — favorite win rate by probability tier
| Tier | N | Wins | Win rate | Implied mid | Cal err |
|---|---:|---:|---:|---:|---:|
| Light favorite (~50-55%) | 7 | 5 | 71.4% | 50.0% | +21.4% |
| Moderate favorite (~55-65%) | 6 | 4 | 66.7% | 60.0% | +6.7% |
| Strong favorite (~65-75%) | 17 | 9 | 52.9% | 70.0% | -17.1% |
| Heavy favorite (~75%+) | 27 | 21 | 77.8% | 88.0% | -10.2% |

## Joint 4-leg parlay ROI on Kalshi
- Leg 1 USA (WC moderate):  **60.0%**  (n=6)
- Leg 2 Brazil (WC heavy):  **88.0%**  (n=27)
- Leg 3 BAL (MLB moderate): **57.0%**  (n=4267)
- Leg 4 CLE (MLB moderate): **57.0%**  (n=4267)
- **Joint probability (independent): 17.15%**  =  1 in 5.8
- Fair price (decimal): 5.83x  (American: +483)

| Kalshi offer | Dec odds | EV per $1 | ROI % | Verdict |
|---:|---:|---:|---:|---|
| $0.10 | 10.00x | +0.072 | +71.5% | ✅ positive |
| $0.12 | 8.33x | +0.052 | +42.9% | ✅ positive |
| $0.15 | 6.67x | +0.022 | +14.3% | ✅ positive |
| $0.17 | 5.71x | -0.003 | -2.0% | ⚠️  marginal |
| $0.20 | 5.00x | -0.028 | -14.2% | ❌ negative |
| $0.25 | 4.00x | -0.078 | -31.4% | ❌ negative |
| $0.30 | 3.33x | -0.128 | -42.8% | ❌ negative |

