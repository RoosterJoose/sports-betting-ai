# Best Bets — Audit Report

---

## June 10, 2026 — Today's Plays (Game 4 of NBA Finals + MLB slate)

**Generated:** 2026-06-10 08:04 PDT | **Staleness:** NBA OK (2d), MLB OK (1d) | **Status:** CANDIDATES, not placed

### 🏀 Top 3 NBA Plays (Kalshi, Game 4 tonight)

| # | Player | Stat | Line | Model P | Mkt P | Edge | Price | ★ |
|---|--------|------|------|---------|-------|------|-------|---|
| 1 | **Mitchell Robinson** | REB | **6+** | 79% | 36% | **+43%** | 36c | ★★★★★ |
| 2 | **Mitchell Robinson** | REB | **5+** | 87% | 49% | **+38%** | 49c | ★★★★½ |
| 3 | **Miles McBride** | 3PT | **2+** | 56% | 27% | **+29%** | 26c | ★★★★ |

### ⚾ Top 3 MLB Plays (PrizePicks, MIN_LINE ≥ 1.0 filter applied)

| # | Player | Stat | Line | Model P | Edge | Side | ★ |
|---|--------|------|------|---------|------|------|---|
| 1 | **Davis Martin** | Pitcher SO | 3.5 | 99.8% | +45.6% | OVER | ★★★★★ |
| 2 | **Chris Sale** | Hits Allowed | 3.5 | 99.1% | +44.9% | OVER | ★★★★★ |
| 3 | **Ian Happ** | H+R+RBI | 1.5 | 80.5% | +26.4% | OVER | ★★★★ |

### 🎰 Best Parlays

**2-Leg (PrizePicks Power Play, 3× payout):**
- Davis Martin SO 3.5 OVER (P=99.8%) + Chris Sale H allowed 3.5 OVER (P=99.1%)
- Joint (independent): 98.9% | EV @ 3×: +197%
- ⚠️ **Correlated** — verify not same game before locking both

**2-Leg NBA (Kalshi combined):**
- Robinson REB 5+ @ 49c + McBride 3PT 2+ @ 26c → 0.87 × 0.56 = **48.7% joint**

**3-Leg Mixed (Power Play 5×):**
- Robinson REB 5+ (Kalshi 87%) + Davis Martin SO 3.5 (PP 99.8%) + Ian Happ H+R+RBI 1.5 (PP 80.5%)
- Joint: **69.9%** | EV @ 5×: **+250%**

**4-Leg Mixed (Power Play 10×):**
- Above 3 + McBride 3PT 2+ (Kalshi 56%)
- Joint: **39.1%** | EV @ 10×: **+291%**

### 💵 Recommended Portfolio ($9.11 total exposure)

| # | Pick | Amount | Notes |
|---|------|--------|-------|
| 1 | Robinson REB 6+ YES @ 36c | $0.36 | 5★ |
| 2 | Robinson REB 5+ YES @ 49c | $0.49 | 4.5★ |
| 3 | McBride 3PT 2+ YES @ 26c | $0.26 | 4★ |
| 4 | Davis Martin SO 3.5 OVER | $2.00 | 5★ |
| 5 | Chris Sale H allowed 3.5 OVER | $2.00 | 5★ |
| 6 | Ian Happ H+R+RBI 1.5 OVER | $2.00 | 4★ |
| 7 | 3-leg parlay (Robinson + Martin + Happ) | $2.00 | Cross-platform |
| | **Total** | **$9.11** | within $30/day limit |

Expected return if all 6 singles hit ≈ $50. Per PROJECT.md Hard Rule #3, **no orders placed** — these are candidates for user approval.

---

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
