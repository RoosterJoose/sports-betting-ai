#!/usr/bin/env bash
#=============================================================================
#  install-nfl-cron.sh — Install cron entries to refresh the NFL cache
#
#  Two-phase schedule (handled by month-restricted cron expressions so the
#  transition is automatic — no manual edits needed when the season starts):
#
#    1. Off-season monthly refresh (Mar 1 - Aug 1, 04:00 local)
#         "0 4 1 3-8 *"
#         The 2025 NFL regular season ended Jan 4, 2026 and Super Bowl LX
#         was Feb 8, 2026. The 2026 regular season starts Sept 7, 2026.
#         During the off-season (March - August), a monthly refresh keeps
#         the model trained on final 2025 season data + draft/free agency.
#
#    2. In-season daily refresh (Sept 1 - Feb 28/29, 04:00 local)
#         "0 4 * 9-2 *"
#         Starts Sept 1 (a few days before opening day Sept 7) so the model
#         is fresh for Week 1. Daily refreshes run through the end of the
#         playoffs (Super Bowl typically early February).
#
#  Both phases coexist in the crontab; cron itself filters by month.
#
#  Usage:
#    ./scripts/install-nfl-cron.sh              # install
#    ./scripts/install-nfl-cron.sh --status     # show status
#    ./scripts/install-nfl-cron.sh --remove     # uninstall
#
#  Calls: bin/refresh_everything.sh nfl --no-scan
#    --no-scan: cron shouldn't generate live dry-run output; run manually
#    if you want to see edges.
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
DISPATCHER="${PROJECT_ROOT}/bin/refresh_everything.sh"
LOG_FILE_NAME="refresh_nfl_everything.log"
CRON_ID="# nfl-auto-refresh"

# Monthly off-season (March 1 - Aug 1)  -- 04:00 on the 1st
MONTHLY_SCHEDULE="0 4 1 3-8 *"
MONTHLY_TAG="monthly-offseason"

# Daily regular season (Sept 1 - Feb 28/29)  -- 04:00 every day
DAILY_SCHEDULE="0 4 * 9-2 *"
DAILY_TAG="daily-regular-season"

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
    echo "=== Installed NFL cron entries ==="
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
        echo "✅ All NFL cron entries removed"
    else
        echo "No NFL cron entries to remove"
    fi
    exit 0
fi

# ── Prereq ──────────────────────────────────────────────────────────────────
if [ ! -x "$DISPATCHER" ]; then
    chmod +x "$DISPATCHER"
    echo "  Made ${DISPATCHER} executable"
fi

# ── Build cron lines ────────────────────────────────────────────────────────
CRON_BLOCK="${CRON_ID} (installed $(date '+%Y-%m-%d %H:%M'))
${MONTHLY_SCHEDULE} cd ${PROJECT_ROOT} && ${DISPATCHER} nfl --no-scan # ${MONTHLY_TAG}
${DAILY_SCHEDULE} cd ${PROJECT_ROOT} && ${DISPATCHER} nfl --no-scan # ${DAILY_TAG}
${CRON_ID}"

# ── Install ─────────────────────────────────────────────────────────────────
(crontab -l 2>/dev/null | grep -v "$CRON_ID" || true
echo "$CRON_BLOCK"
) | crontab -

echo "✅ NFL refresh cron entries INSTALLED"
echo ""
echo "  Two entries (cron filters by month — no manual switching needed):"
echo ""
echo "    1. Monthly off-season    ${MONTHLY_SCHEDULE}"
echo "         04:00 on the 1st, March through August"
echo "         (current phase — runs next on July 1, 2026)"
echo ""
echo "    2. Daily regular season  ${DAILY_SCHEDULE}"
echo "         04:00 every day, September through February"
echo "         (auto-activates Sept 1, 2026 for opening day Sept 7)"
echo ""
echo "  Script:    ${DISPATCHER} nfl --no-scan"
echo "  Log:       ${HOME}/logs/${LOG_FILE_NAME}"
echo ""
echo "  Verify:    ./scripts/install-nfl-cron.sh --status"
echo "  Remove:    ./scripts/install-nfl-cron.sh --remove"
