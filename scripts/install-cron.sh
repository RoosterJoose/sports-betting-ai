#!/usr/bin/env bash
#=============================================================================
#  install-cron.sh — Install a daily cron job for the morning scan report
#
#  Usage:
#    ./scripts/install-cron.sh                     # install at 9am (paper)
#    ./scripts/install-cron.sh --bet               # install at 9am (live)
#    ./scripts/install-cron.sh --time "8:30"       # custom time (paper)
#    ./scripts/install-cron.sh --remove            # uninstall the cron job
#    ./scripts/install-cron.sh --status            # check if installed
#
#  Requires:
#    REPORT_EMAIL_TO in .env to receive email reports
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
SCRIPT="${PROJECT_ROOT}/scripts/daily-report.sh"
CRON_ID="# sports-betting-ai-daily-report"

# ── Parse args ──────────────────────────────────────────────────────────────
BET_FLAG=""
SCHEDULE_TIME="9:00"
DO_REMOVE=false
DO_STATUS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bet) BET_FLAG="--bet"; shift ;;
        --time) SCHEDULE_TIME="$2"; shift 2 ;;
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
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        echo "✅ Daily report cron is INSTALLED"
        crontab -l | grep "$CRON_ID"
        echo ""
        echo "  Script: $SCRIPT"
        echo "  Logs:   $PROJECT_ROOT/reports/"
    else
        echo "❌ Daily report cron is NOT installed"
    fi
    exit 0
fi

# ── Remove ──────────────────────────────────────────────────────────────────
if $DO_REMOVE; then
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$CRON_ID" | crontab -
        echo "✅ Cron job removed"
    else
        echo "No cron job found to remove"
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

# ── Build cron line ─────────────────────────────────────────────────────────
# Runs at the specified time, redirects stdout/stderr to a daily log
LOG_DIR="${PROJECT_ROOT}/reports"
CRON_LOG="${LOG_DIR}/cron.log"
CRON_LINE="${CRON_ID}
${MINUTE} ${HOUR} * * * cd ${PROJECT_ROOT} && ${SCRIPT} ${BET_FLAG} >> ${CRON_LOG} 2>&1 ${CRON_ID}"

INSTALLED_TIME=$(printf "%02d:%02d" "$HOUR" "$MINUTE")

# ── Install ─────────────────────────────────────────────────────────────────
# Remove existing job with our ID, then add new one
(crontab -l 2>/dev/null | grep -v "$CRON_ID"
echo "$CRON_LINE"
) | crontab -

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
