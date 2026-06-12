#!/usr/bin/env bash
#=============================================================================
#  install-cron.sh — Install cron jobs for the morning scan report,
#                    the WC lineup auto-populator (WC season only),
#                    the WNBA daily cache refresh (WNBA season only),
#                    and the WC offset re-fit (early WC tournament window).
#
#  Usage:
#    ./scripts/install-cron.sh                     # install at 9am (paper) + WC lineups + WNBA refresh + WC offset
#    ./scripts/install-cron.sh --bet               # install at 9am (live) + WC lineups + WNBA refresh + WC offset
#    ./scripts/install-cron.sh --time "8:30"       # custom time (paper) + WC lineups + WNBA refresh + WC offset
#    ./scripts/install-cron.sh --no-wc-lineups     # skip WC lineups cron
#    ./scripts/install-cron.sh --no-wnba-refresh   # skip WNBA daily refresh cron
#    ./scripts/install-cron.sh --no-wc-offset      # skip WC offset re-fit cron
#    ./scripts/install-cron.sh --remove            # uninstall all cron jobs
#    ./scripts/install-cron.sh --status            # check if installed
#
#  Requires:
#    REPORT_EMAIL_TO in .env to receive email reports
#
#  The WC lineups cron is hard-coded to the FIFA World Cup 2026 window:
#    hourly 7am-11pm, days 11-19 of June + July 2026
#    (cron: 0 7-23 11-19 6,7 *) -- only fires during the 39-day tournament.
#    Outside WC season it's a no-op (the script fetches zero KXWCGAME markets).
#
#  The WNBA refresh cron runs once a day at 5am local time. WNBA season runs
#    May 1 - Oct 31 (cron: 0 5 1-31 5-10 *) so the cron is effectively dormant
#    outside season. Refreshing at 5am ensures the cache is fresh by the time
#    the morning scan runs at 9am.
#
#  The WC offset re-fit cron runs daily at 6am on days 14-19 of June (cron:
#    0 6 14-19 6 *). The wrapper has a built-in n_2026>=10 guard, so it
#    no-ops until ~10 WC 2026 matches complete, then auto-regenerates
#    models/worldcup/offset_oos_2023plus.json if the 2026-included pooled
#    delta shifts by >5pp vs the current offset. --no-wc-offset skips it.
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
SCRIPT="${PROJECT_ROOT}/scripts/daily-report.sh"
WC_LINEUPS_BIN="${PROJECT_ROOT}/bin/populate_wc_lineups.py"
WNBA_REFRESH_BIN="${PROJECT_ROOT}/bin/refresh_wnba_cache.sh"
WC_OFFSET_BIN="${PROJECT_ROOT}/scripts/refit_wc_offset_2026.py"
CRON_ID="# sports-betting-ai-daily-report"
WC_LINEUPS_CRON_ID="# sports-betting-ai-wc-lineups"
WNBA_REFRESH_CRON_ID="# sports-betting-ai-wnba-refresh"
WC_OFFSET_CRON_ID="# sports-betting-ai-wc-offset"

# ── Parse args ──────────────────────────────────────────────────────────────
BET_FLAG=""
SCHEDULE_TIME="9:00"
DO_REMOVE=false
DO_STATUS=false
INSTALL_WC_LINEUPS=true
INSTALL_WNBA_REFRESH=true
INSTALL_WC_OFFSET=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bet) BET_FLAG="--bet"; shift ;;
        --time) SCHEDULE_TIME="$2"; shift 2 ;;
        --no-wc-lineups) INSTALL_WC_LINEUPS=false; shift ;;
        --no-wnba-refresh) INSTALL_WNBA_REFRESH=false; shift ;;
        --no-wc-offset) INSTALL_WC_OFFSET=false; shift ;;
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
    echo ""
    if crontab -l 2>/dev/null | grep -q "$WNBA_REFRESH_CRON_ID"; then
        echo "✅ WNBA refresh cron is INSTALLED"
        crontab -l | grep "$WNBA_REFRESH_CRON_ID"
        echo ""
        echo "  Script: $WNBA_REFRESH_BIN"
        echo "  Schedule: daily 5am, May 1 - Oct 31 (WNBA season)"
        echo "  Logs:     $HOME/logs/refresh_wnba_cache.log"
    else
        echo "❌ WNBA refresh cron is NOT installed"
    fi
    echo ""
    if crontab -l 2>/dev/null | grep -q "$WC_OFFSET_CRON_ID"; then
        echo "✅ WC offset re-fit cron is INSTALLED"
        crontab -l | grep "$WC_OFFSET_CRON_ID"
        echo ""
        echo "  Script: $WC_OFFSET_BIN"
        echo "  Schedule: daily 6am, June 14-19 (early WC tournament window)"
        echo "  Logs:     reports/wc_offset_refit_cron.log"
    else
        echo "❌ WC offset re-fit cron is NOT installed"
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
    if crontab -l 2>/dev/null | grep -q "$WNBA_REFRESH_CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$WNBA_REFRESH_CRON_ID" | crontab -
        echo "✅ WNBA refresh cron removed"
        removed=1
    fi
    if crontab -l 2>/dev/null | grep -q "$WC_OFFSET_CRON_ID"; then
        crontab -l 2>/dev/null | grep -v "$WC_OFFSET_CRON_ID" | crontab -
        echo "✅ WC offset re-fit cron removed"
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
chmod +x "$WNBA_REFRESH_BIN" 2>/dev/null || true
chmod +x "$WC_OFFSET_BIN" 2>/dev/null || true

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

# ── Build cron line for WNBA refresh (daily 5am, May-Oct only) ─────────────
# Schedule: 0 5 * 5-10 *
#   - minute 0
#   - hour 5 (5am local time, finishes well before the 9am daily report)
#   - any day of month
#   - months 5-10 (May 1 - Oct 31, WNBA regular season + playoffs)
#   - any day of week
# Outside the WNBA season (Nov-Apr) the cron schedule will still fire, but
# refresh_wnba_cache.sh will fetch zero games and exit immediately (no-op).
# Refreshing at 5am ensures the cache is fresh by the time the morning
# scan runs at 9am.
WNBA_REFRESH_CRON_LINE="${WNBA_REFRESH_CRON_ID}
0 5 * 5-10 * cd ${PROJECT_ROOT} && ${WNBA_REFRESH_BIN} >> /dev/null 2>&1 ${WNBA_REFRESH_CRON_ID}"

# ── Build cron line for WC offset re-fit (daily 6am, June 14-19 only) ──────
# Schedule: 0 6 14-19 6 *
#   - minute 0
#   - hour 6 (6am local time, finishes before the 9am daily report)
#   - days 14-19 of month 6 (June 14-19, ~10 matches in for the 2026 WC)
#   - any year
#   - any day of week
# The wrapper has a built-in n_2026>=10 guard, so it no-ops until enough
# 2026 matches accumulate, then auto-regenerates
# models/worldcup/offset_oos_2023plus.json if the pooled delta shifts by >5pp
# vs the current offset. Outside June 14-19 the cron won't fire.
WC_OFFSET_LOG="${LOG_DIR}/wc_offset_refit_cron.log"
WC_OFFSET_CRON_LINE="${WC_OFFSET_CRON_ID}
0 6 14-19 6 * cd ${PROJECT_ROOT} && ${VENV_PY:-${PROJECT_ROOT}/.venv/bin/python} -m scripts.refit_wc_offset_2026 --commit >> ${WC_OFFSET_LOG} 2>&1 ${WC_OFFSET_CRON_ID}"

# ── Install (replace existing entries with our IDs, then add new ones) ─────
crontab -l 2>/dev/null | grep -v -E "$CRON_ID|$WC_LINEUPS_CRON_ID|$WNBA_REFRESH_CRON_ID|$WC_OFFSET_CRON_ID" > /tmp/cron.tmp
{
    cat /tmp/cron.tmp
    echo "$CRON_LINE"
    if $INSTALL_WC_LINEUPS && [ -f "$WC_LINEUPS_BIN" ]; then
        echo "$WC_LINEUPS_CRON_LINE"
    fi
    if $INSTALL_WNBA_REFRESH && [ -f "$WNBA_REFRESH_BIN" ]; then
        echo "$WNBA_REFRESH_CRON_LINE"
    fi
    if $INSTALL_WC_OFFSET && [ -f "$WC_OFFSET_BIN" ]; then
        echo "$WC_OFFSET_CRON_LINE"
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

if $INSTALL_WNBA_REFRESH; then
    if [ -f "$WNBA_REFRESH_BIN" ]; then
        echo ""
        echo "✅ WNBA refresh cron INSTALLED (WNBA season: May 1 - Oct 31)"
        echo ""
        echo "  Schedule:  daily 5am, May-Oct (cron: 0 5 * 5-10 *)"
        echo "  Script:    ${WNBA_REFRESH_BIN}"
        echo "  Log:       ${HOME}/logs/refresh_wnba_cache.log"
        echo "  Purpose:   refresh data/wnba_cache/wnba_games.parquet + refit WNBA BetaCal"
        echo "             so the staleness guard passes during WNBA season (May-Oct)"
    else
        echo ""
        echo "⚠️  bin/refresh_wnba_cache.sh not found, skipping WNBA refresh cron"
    fi
else
    echo ""
    echo "⏭️  WNBA refresh cron NOT installed (--no-wnba-refresh flag)"
fi

if $INSTALL_WC_OFFSET; then
    if [ -f "$WC_OFFSET_BIN" ]; then
        echo ""
        echo "✅ WC offset re-fit cron INSTALLED (early WC tournament: June 14-19)"
        echo ""
        echo "  Schedule:  daily 6am, June 14-19 (cron: 0 6 14-19 6 *)"
        echo "  Script:    ${VENV_PY:-${PROJECT_ROOT}/.venv/bin/python} -m scripts.refit_wc_offset_2026 --commit"
        echo "  Log:       ${WC_OFFSET_LOG}"
        echo "  Purpose:   re-fit models/worldcup/offset_oos_2023plus.json from the 2026 WC sample."
        echo "             Wrapper has built-in n_2026>=10 + 5pp-shift guards; no-ops if not ready."
    else
        echo ""
        echo "⚠️  scripts/refit_wc_offset_2026.py not found, skipping WC offset cron"
    fi
else
    echo ""
    echo "⏭️  WC offset re-fit cron NOT installed (--no-wc-offset flag)"
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
