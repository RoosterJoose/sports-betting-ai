#!/usr/bin/env bash
#=============================================================================
#  install-nhl-cron.sh — Install a monthly cron job to refresh the NHL cache
#
#  Schedule: 3:00 AM local time on the 1st of each month
#    The NHL 2025-26 regular season ended April 16, 2026. The 2026-27 regular
#    season starts in early October 2026. During the off-season (May-Sept),
#    a monthly refresh keeps the model trained on the latest available data
#    (finals games, late-season injuries, roster changes, etc.).
#
#    In-season (Oct-April), the same monthly refresh is fine because games
#    happen infrequently between months. Switch to daily in-season cron when
#    regular-season play resumes.
#
#  Usage:
#    ./scripts/install-nhl-cron.sh              # install
#    ./scripts/install-nhl-cron.sh --status     # show status
#    ./scripts/install-nhl-cron.sh --remove     # uninstall
#
#  Calls: bin/refresh_everything.sh nhl --no-scan
#    --no-scan: cron shouldn't generate live dry-run output; run manually
#    if you want to see edges.
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
DISPATCHER="${PROJECT_ROOT}/bin/refresh_everything.sh"
LOG_FILE_NAME="refresh_nhl_everything.log"
CRON_ID="# nhl-monthly-auto-refresh"
SCHEDULE="0 3 1 * *"   # min=0, hour=3, day=1, *=every month, *=every weekday

# ── Parse args ──────────────────────────────────────────────────────────────
DO_STATUS=false
DO_REMOVE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status) DO_STATUS=true; shift ;;
        --remove) DO_REMOVE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Status ──────────────────────────────────────────────────────────────────
if $DO_STATUS; then
    echo "=== Installed NHL monthly cron entries ==="
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l | grep "$CRON_ID" -A 1
    else
        echo "  (none installed)"
    fi
    echo ""
    echo "=== Refresh log (last 10 lines) ==="
    tail -n 10 "${HOME}/logs/${LOG_FILE_NAME}" 2>/dev/null || echo "  (no log yet)"
    exit 0
fi

# ── Remove ──────────────────────────────────────────────────────────────────
if $DO_REMOVE; then
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$CRON_ID" | crontab -
        echo "✅ NHL monthly cron removed"
    else
        echo "No NHL monthly cron to remove"
    fi
    exit 0
fi

# ── Prereq ──────────────────────────────────────────────────────────────────
if [ ! -x "$DISPATCHER" ]; then
    chmod +x "$DISPATCHER"
    echo "  Made ${DISPATCHER} executable"
fi

# ── Build cron line ─────────────────────────────────────────────────────────
# 0 3 1 * *  = 03:00 on day 1 of every month
CRON_LINE="${CRON_ID}
${SCHEDULE} cd ${PROJECT_ROOT} && ${DISPATCHER} nhl --no-scan ${CRON_ID}"

# ── Install ─────────────────────────────────────────────────────────────────
(crontab -l 2>/dev/null | grep -v "$CRON_ID" || true
echo "$CRON_LINE"
) | crontab -

echo "✅ NHL monthly refresh cron INSTALLED"
echo ""
echo "  Schedule:  03:00 on the 1st of each month"
echo "  Script:    ${DISPATCHER} nhl --no-scan"
echo "  Log:       ${HOME}/logs/${LOG_FILE_NAME}"
echo ""
echo "  Verify:    ./scripts/install-nhl-cron.sh --status"
echo "  Remove:    ./scripts/install-nhl-cron.sh --remove"
echo ""
echo "  Note: When NHL regular season starts (early Oct 2026), you'll want"
echo "  to switch to daily refresh. Use --remove and add a daily entry, or"
echo "  ask the agent to wire up an in-season daily cron."
