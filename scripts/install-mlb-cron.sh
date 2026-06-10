#!/usr/bin/env bash
#=============================================================================
#  install-mlb-cron.sh — Install a daily cron job to refresh the MLB cache
#
#  Schedule: 6:00 AM local time daily (after most west-coast games complete)
#  The refresh script is idempotent and safe to run any day.
#
#  Usage:
#    ./scripts/install-mlb-cron.sh              # install
#    ./scripts/install-mlb-cron.sh --status     # show status
#    ./scripts/install-mlb-cron.sh --remove     # uninstall
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
REFRESH_SCRIPT="${PROJECT_ROOT}/bin/refresh_mlb_cache.sh"
CRON_ID="# mlb-daily-auto-refresh"
SCHEDULE_HOUR=6
SCHEDULE_MIN=0

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
    echo "=== Installed MLB daily cron entries ==="
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l | grep "$CRON_ID" -A 1
    else
        echo "  (none installed)"
    fi
    echo ""
    echo "=== Refresh log (last 10 lines) ==="
    tail -n 10 "${HOME}/logs/refresh_mlb_cache.log" 2>/dev/null || echo "  (no log yet)"
    exit 0
fi

# ── Remove ──────────────────────────────────────────────────────────────────
if $DO_REMOVE; then
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$CRON_ID" | crontab -
        echo "✅ MLB daily cron removed"
    else
        echo "No MLB daily cron to remove"
    fi
    exit 0
fi

# ── Prereq ──────────────────────────────────────────────────────────────────
if [ ! -x "$REFRESH_SCRIPT" ]; then
    chmod +x "$REFRESH_SCRIPT"
    echo "  Made ${REFRESH_SCRIPT} executable"
fi

# ── Build cron line ─────────────────────────────────────────────────────────
CRON_LINE="${CRON_ID}
${SCHEDULE_MIN} ${SCHEDULE_HOUR} * * * cd ${PROJECT_ROOT} && ${REFRESH_SCRIPT} ${CRON_ID}"

# ── Install ─────────────────────────────────────────────────────────────────
(crontab -l 2>/dev/null | grep -v "$CRON_ID" || true
echo "$CRON_LINE"
) | crontab -

echo "✅ MLB daily refresh cron INSTALLED"
echo ""
echo "  Schedule:  ${SCHEDULE_HOUR}:$(printf '%02d' $SCHEDULE_MIN) daily"
echo "  Script:    ${REFRESH_SCRIPT}"
echo "  Log:       ${HOME}/logs/refresh_mlb_cache.log"
echo ""
echo "  Verify:    ./scripts/install-mlb-cron.sh --status"
echo "  Remove:    ./scripts/install-mlb-cron.sh --remove"
