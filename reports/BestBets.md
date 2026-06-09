# Best Bets — Audit Report (June 9, 2026)

## Portfolio Status

| Sport | Orders | Status |
|-------|--------|--------|
| **MLB** | 81 | Resting (not filled — low liquidity) |
| **NBA** | 49 | Resting (not filled — low liquidity) |
| **World Cup** | 45 | Executed ✅ (filled) |
| **CFB** | 1 (Clemson) | Resting (disabled — off-season) |

**Balance:** $78.58

## MLB Scanner Health

✅ **Working.** 9 active stat types (KS, HR, TB, HRR, IP, ER, H, BB, RBI), 2 info-only (R, SB). 1,937 players loaded.

Sample KS scan for today (JUN09): 210 markets → 21 qualifying:
- Lucas Giolito 2+ KS: edge=+20% (model=72% vs mkt=52%)
- Gerrit Cole 7+ KS: edge=+28% (model=53% vs mkt=24%)
- Walbert Ureña 5+ KS: edge=+6%
- Kai-Wei Teng 7+ KS: edge=+12%
- Gerrit Cole 8+ KS: edge=+10%

## NBA Scanner Health

✅ **Working** (via nba_bet.py). NBA Finals SAS vs NYK active for June 10. Fixed `-m` module loading bug — morning_scan now imports from nba_bet.py.

## Known Issues

- **MLB orders resting**: Most markets have low liquidity — orders sit unfilled until a counterparty takes them
- **UFC**: Title parser fixed for comma-separated combo format. Model trained (85.2% acc). 0 qualifying — fighters missing from DB (Diego Lopes, Bo Nickal, Alex Pereira)
- **CFB**: Off-season, disabled from betting
- **NFL**: Off-season

## Next Steps

1. Run `python3 src/scripts/nba_bet.py --bet` to place NBA bets for tomorrow's Finals
2. Run `python -m src.scripts.morning_scan --scan` to test full morning scan pipeline
3. Retrain UFC model with updated fighter data to get Diego Lopes, Bo Nickal, Alex Pereira
