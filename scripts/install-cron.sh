#!/usr/bin/env bash
#=============================================================================
#  install-cron.sh — Install cron jobs for the morning scan report
#                    and the WC lineup auto-populator (WC season only).
#
#  Usage:
#    ./scripts/install-cron.sh                     # install at 9am (paper) + WC lineups
#    ./scripts/install-cron.sh --bet               # install at 9am (live) + WC lineups
#    ./scripts/install-cron.sh --time "8:30"       # custom time (paper) + WC lineups
#    ./scripts/install-cron.sh --no-wc-lineups     # skip WC lineups cron
#    ./scripts/install-cron.sh --remove            # uninstall both cron jobs
#    ./scripts/install-cron.sh --status            # check if installed
#
#  Requires:
#    REPORT_EMAIL_TO in .env to receive email reports
#
#  The WC lineups cron is hard-coded to the FIFA World Cup 2026 window:
#    hourly 7am-11pm, days 11-19 of June + July 2026
#    (cron: 0 7-23 11-19 6,7 *) -- only fires during the 39-day tournament.
#    Outside WC season it's a no-op (the script fetches zero KXWCGAME markets).
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
SCRIPT="${PROJECT_ROOT}/scripts/daily-report.sh"
WC_LINEUPS_BIN="${PROJECT_ROOT}/bin/populate_wc_lineups.py"
CRON_ID="# sports-betting-ai-daily-report"
WC_LINEUPS_CRON_ID="# sports-betting-ai-wc-lineups"

# ── Parse args ──────────────────────────────────────────────────────────────
BET_FLAG=""
SCHEDULE_TIME="9:00"
DO_REMOVE=false
DO_STATUS=false
INSTALL_WC_LINEUPS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bet) BET_FLAG="--bet"; shift ;;
        --time) SCHEDULE_TIME="$2"; shift 2 ;;
        --no-wc-lineups) INSTALL_WC_LINEUPS=false; shift ;;
        --remove) DO_REMOVE=true; shift ;;
        --status) DO_STATUS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Convert schedule time to cron format ────────────────────────────────────
HOUR=$(echo "$SCHEDULE_TIME" | cut -d: -f1)
MINUTE=$(echo "$SCHEDULE_TIME" | cut -d: -f2)
# Remove leading zeros for cron (cron handles them fine, but be safe)
HOUR=$((10#$HOUR))
MINUTE=$((10#$MINUTE))

# ── Status check ────────────────────────────────────────────────────────────
if $DO_STATUS; then
    echo "================================================================"
    echo "  Cron status"
    echo "================================================================"
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        echo "✅ Daily report cron is INSTALLED"
        crontab -l | grep "$CRON_ID"
        echo ""
        echo "  Script: $SCRIPT"
        echo "  Logs:   $PROJECT_ROOT/reports/"
    else
        echo "❌ Daily report cron is NOT installed"
    fi
    echo ""
    if crontab -l 2>/dev/null | grep -q "$WC_LINEUPS_CRON_ID"; then
        echo "✅ WC lineups cron is INSTALLED"
        crontab -l | grep "$WC_LINEUPS_CRON_ID"
        echo ""
        echo "  Script: $WC_LINEUPS_BIN"
        echo "  Schedule: hourly 7am-11pm, days 11-19 of June+July (WC 2026 window)"
        echo "  Logs:     $PROJECT_ROOT/reports/wc_lineups_cron.log"
    else
        echo "❌ WC lineups cron is NOT installed"
    fi
    exit 0
fi

# ── Remove ──────────────────────────────────────────────────────────────────
if $DO_REMOVE; then
    removed=0
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$CRON_ID" | crontab -
        echo "✅ Daily report cron removed"
        removed=1
    fi
    if crontab -l 2>/dev/null | grep -q "$WC_LINEUPS_CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$WC_LINEUPS_CRON_ID" | crontab -
        echo "✅ WC lineups cron removed"
        removed=1
    fi
    if [ "$removed" -eq 0 ]; then
        echo "No cron jobs found to remove"
    fi
    exit 0
fi

# ── Check prerequisites ─────────────────────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Script not found: $SCRIPT"
    exit 1
fi

chmod +x "$SCRIPT"

# Warn about email config
ENV_FILE="${PROJECT_ROOT}/.env"
EMAIL_CONFIGURED=false
if [ -f "$ENV_FILE" ] && grep -q "REPORT_EMAIL_TO" "$ENV_FILE" 2>/dev/null; then
    EMAIL_CONFIGURED=true
fi

# ── Build cron line for daily report ────────────────────────────────────────
# Runs at the specified time, redirects stdout/stderr to a daily log
LOG_DIR="${PROJECT_ROOT}/reports"
CRON_LOG="${LOG_DIR}/cron.log"
CRON_LINE="${CRON_ID}
${MINUTE} ${HOUR} * * * cd ${PROJECT_ROOT} && ${SCRIPT} ${BET_FLAG} >> ${CRON_LOG} 2>&1 ${CRON_ID}"

INSTALLED_TIME=$(printf "%02d:%02d" "$HOUR" "$MINUTE")

# ── Build cron line for WC lineups (hourly, WC 2026 window only) ───────────
# Schedule: 0 7-23 11-19 6,7 *
#   - minute 0 (top of the hour)
#   - hours 7-23 (7am-11pm local time)
#   - days 11-19 of months 6,7 (June 11 - July 19, WC 2026 window)
#   - any day of week
# This is a 39-day window. Outside it, the script fetches zero KXWCGAME
# markets and exits immediately (no-op cost ~1s).
WC_LINEUPS_LOG="${LOG_DIR}/wc_lineups_cron.log"
WC_LINEUPS_CRON_LINE="${WC_LINEUPS_CRON_ID}
0 7-23 11-19 6,7 * cd ${PROJECT_ROOT} && ${WC_LINEUPS_BIN} >> ${WC_LINEUPS_LOG} 2>&1 ${WC_LINEUPS_CRON_ID}"

# ── Install (replace existing entries with our IDs, then add new ones) ─────
crontab -l 2>/dev/null | grep -v -E "$CRON_ID|$WC_LINEUPS_CRON_ID" > /tmp/cron.tmp
{
    cat /tmp/cron.tmp
    echo "$CRON_LINE"
    if $INSTALL_WC_LINEUPS && [ -f "$WC_LINEUPS_BIN" ]; then
        echo "$WC_LINEUPS_CRON_LINE"
    fi
} | crontab -
rm -f /tmp/cron.tmp

echo "✅ Daily report cron INSTALLED"
echo ""
echo "  Schedule:  ${INSTALLED_TIME} daily"
echo "  Script:    ${SCRIPT}"
echo "  Mode:      ${BET_FLAG:---paper (dry run)}"
echo "  Log:       ${CRON_LOG}"
echo "  Reports:   ${LOG_DIR}/"
echo ""
if $EMAIL_CONFIGURED; then
    echo "  📧 Email will be sent to the address in .env"
else
    echo "  ⚠️  No REPORT_EMAIL_TO set in .env — reports saved to disk only"
    echo "     Add REPORT_EMAIL_TO=you@example.com to .env for email delivery"
fi

if $INSTALL_WC_LINEUPS; then
    if [ -f "$WC_LINEUPS_BIN" ]; then
        echo ""
        echo "✅ WC lineups cron INSTALLED (WC 2026 window: June 11 - July 19)"
        echo ""
        echo "  Schedule:  hourly 7am-11pm, days 11-19 of June + July (cron: 0 7-23 11-19 6,7 *)"
        echo "  Script:    ${WC_LINEUPS_BIN}"
        echo "  Log:       ${WC_LINEUPS_LOG}"
        echo "  Purpose:   pre-populate data/cache/worldcup/lineups.json + fotmob_ids.json"
        echo "             60-90 min before each WC kickoff, so key_player_out fires at scan time"
    else
        echo ""
        echo "⚠️  --no-wc-lineups-equivalent: bin/populate_wc_lineups.py not found, skipping"
    fi
else
    echo ""
    echo "⏭️  WC lineups cron NOT installed (--no-wc-lineups flag)"
fi

echo ""
echo "  To verify: ./scripts/install-cron.sh --status"
echo "  To remove: ./scripts/install-cron.sh --remove"

# ── Also create .env template if missing ────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'EOF'
# ── Email Configuration ─────────────────────────────────────────────────
# Set this to receive daily reports via email
# REPORT_EMAIL_TO=you@example.com

# For SMTP (Gmail, etc.) — uncomment and fill in:
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=your@gmail.com
# SMTP_PASSWORD=your-app-password
# REPORT_EMAIL_FROM=sports-betting-ai@example.com
EOF
    echo "  Created .env template at ${ENV_FILE}"
fi
