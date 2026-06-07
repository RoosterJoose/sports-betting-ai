#!/bin/bash
# Integration wrapper: Our ML + Octagon CLI
# Division of labor:
#   1. Our system → MLB pitcher strikeouts (trained regressors, 61-72% accuracy)
#   2. Octagon CLI  → All other Kalshi markets (politics, crypto, finance)
#
# Usage:
#   ./integration_wrapper.sh         # scan everything
#   ./integration_wrapper.sh --bet   # scan + bet on MLB

OUR_MLB="python3 -m src.scripts.kalshi_mlb_unified --scan"
OUR_BET="python3 -m src.scripts.kalshi_mlb_unified --bet"
OCTAGON="kalshi search edge --min-edge 7"

echo "======= Our System: MLB Pitcher Markets ======="
$OUR_MLB

if [[ "$1" == "--bet" ]]; then
    echo ""
    echo "======= Placing MLB Bets ======="
    $OUR_BET
fi

echo ""
echo "======= Octagon CLI: All Other Markets ======="
echo "Run manually after setup: kalshi search edge --min-edge 7"
echo "Setup: bunx kalshi-trading-bot-cli@latest (interactive wizard)"
echo "Config: ~/.kalshi-bot/.env needs your LLM API key"

echo ""
echo "======= Combined Portfolio ======="
python3 -c "
from src.data.kalshi import KalshiClient
kc = KalshiClient()
print(f'Kalshi Balance: \${kc.get_balance():.2f}')
try:
    pos = kc.get_positions()
    if pos is not None and not pos.empty:
        print(f'Positions: {len(pos)}')
        for _, p in pos.iterrows():
            print(f'  {p.get(\"ticker\",\"?\")}: {p.get(\"position\",\"?\")}')
    else:
        print('No open positions')
except Exception as e:
    print(f'Positions check: {e}')
"
