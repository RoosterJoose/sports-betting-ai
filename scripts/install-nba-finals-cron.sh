#!/usr/bin/env bash
#=============================================================================
#  install-nba-finals-cron.sh — Install cron entries to auto-refresh NBA cache
#  4 hours after each game of the 2026 NBA Finals (SAS @ NYK)
#
#  Schedule: 21:00 (9:00 PM) local time on the day of each game
#    (4 hours after 5:00 PM PT tipoff for weeknight games)
#
#  Usage:
#    ./scripts/install-nba-finals-cron.sh              # install Games 4-7
#    ./scripts/install-nba-finals-cron.sh --status     # show installed entries
#    ./scripts/install-nba-finals-cron.sh --remove     # remove all entries
#    ./scripts/install-nba-finals-cron.sh --all-games  # include Games 1-3 (already played)
#
#  Note: Game 3 (June 8) is already past — included only with --all-games.
#  Games 5-7 are conditional — the cron will still run; the script is
#  idempotent and safe to run on days with no game.
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
REFRESH_SCRIPT="${PROJECT_ROOT}/bin/refresh_nba_cache.sh"
CRON_ID="# nba-finals-auto-refresh"

# ── 2026 NBA Finals schedule ───────────────────────────────────────────────
# Game 3 confirmed June 8, 2026 (Mon). Standard NBA Finals cadence is
# every 2-3 days; tipoff 8:00 PM ET (5:00 PM PT) for weeknights, 8:30 PM
# ET (5:30 PM PT) for weekends. +4hr from tipoff = 9:00 PM PT local time.
declare -a GAMES=(
  "1:2026-06-04:Thu"   # Game 1
  "2:2026-06-06:Sat"   # Game 2
  "3:2026-06-08:Mon"   # Game 3 (confirmed)
  "4:2026-06-10:Wed"   # Game 4 (most likely)
  "5:2026-06-13:Sat"   # Game 5 (if needed)
  "6:2026-06-16:Tue"   # Game 6 (if needed)
  "7:2026-06-19:Fri"   # Game 7 (if needed)
)

# ── Parse args ──────────────────────────────────────────────────────────────
DO_STATUS=false
DO_REMOVE=false
DO_ALL_GAMES=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status) DO_STATUS=true; shift ;;
        --remove) DO_REMOVE=true; shift ;;
        --all-games) DO_ALL_GAMES=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Status ──────────────────────────────────────────────────────────────────
if $DO_STATUS; then
    echo "=== Installed NBA Finals cron entries ==="
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l | grep -A 0 -B 0 "$CRON_ID" | head -20
    else
        echo "  (none installed)"
    fi
    echo ""
    echo "=== Refresh log (last 10 lines) ==="
    tail -n 10 "${HOME}/logs/refresh_nba_cache.log" 2>/dev/null || echo "  (no log yet)"
    exit 0
fi

# ── Remove ──────────────────────────────────────────────────────────────────
if $DO_REMOVE; then
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$CRON_ID" | crontab -
        echo "✅ All NBA Finals cron entries removed"
    else
        echo "No NBA Finals cron entries to remove"
    fi
    exit 0
fi

# ── Prereq checks ───────────────────────────────────────────────────────────
if [ ! -x "$REFRESH_SCRIPT" ]; then
    chmod +x "$REFRESH_SCRIPT"
    echo "  Made ${REFRESH_SCRIPT} executable"
fi

# ── Determine which games to install ───────────────────────────────────────
# Default: only future games (4, 5, 6, 7). With --all-games: all 7.
TODAY=$(date '+%Y-%m-%d')
CRON_LINES=""

# Header line for the new block
CRON_LINES+="${CRON_ID} (installed $(date '+%Y-%m-%d %H:%M'))${CRON_ID}"$'\n'

for entry in "${GAMES[@]}"; do
    IFS=':' read -r NUM DATE DOW <<< "$entry"
    # Parse month and day
    MONTH=$(echo "$DATE" | cut -d- -f2 | sed 's/^0//')
    DAY=$(echo "$DATE" | cut -d- -f3 | sed 's/^0//')

    # Skip past games unless --all-games
    if [ "$DATE" \< "$TODAY" ] && ! $DO_ALL_GAMES; then
        echo "  Skipping Game $NUM ($DATE $DOW — already past)"
        continue
    fi

    # 21:00 (9:00 PM) local time on the day of the game = 4hr after 5:00 PM PT tipoff
    CRON_LINES+="${MINUTE:-0} 21 $DAY $MONTH * cd ${PROJECT_ROOT} && ${REFRESH_SCRIPT} # game-$NUM${CRON_ID}"$'\n'
    echo "  Will install: Game $NUM ($DATE $DOW) at 21:00 local time"
done

# ── Install ────────────────────────────────────────────────────────────────
# Remove existing entries, then append new ones
(crontab -l 2>/dev/null | grep -v "$CRON_ID" || true
echo "$CRON_LINES"
) | crontab -

echo ""
echo "✅ NBA Finals cron entries INSTALLED"
echo ""
echo "  Script:  ${REFRESH_SCRIPT}"
echo "  Log:     ${HOME}/logs/refresh_nba_cache.log"
echo ""
echo "  Verify:  ./scripts/install-nba-finals-cron.sh --status"
echo "  Remove:  ./scripts/install-nba-finals-cron.sh --remove"
echo ""
echo "  Note: Games 5-7 are conditional on the series going that long."
echo "  The refresh script is idempotent — it will run but harmlessly if"
echo "  no new game data exists."
