#!/usr/bin/env bash
#=============================================================================
#  daily-report.sh — Run the full morning scan and save a timestamped report
#
#  Usage:
#    ./scripts/daily-report.sh                      # paper/dry-run mode
#    ./scripts/daily-report.sh --bet                # live-bet mode
#    ./scripts/daily-report.sh --paper --bankroll 100  # custom bankroll
#
#  Output:
#    reports/YYYY-MM-DD_HHMM.txt          — full scan log
#    reports/latest.txt                   — symlink to most recent report
#    reports/daily-report.log             — append-only run log
#=============================================================================

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

# ── Config ──────────────────────────────────────────────────────────────────
TIMEOUT_MINUTES=30
REPORTS_DIR="${PROJECT_ROOT}/reports"
LOG_FILE="${REPORTS_DIR}/daily-report.log"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
SCAN_MODULE="src.scripts.morning_scan"

# ── Ensure dirs exist ───────────────────────────────────────────────────────
mkdir -p "$REPORTS_DIR"

# ── Timestamp ───────────────────────────────────────────────────────────────
TIMESTAMP=$(date "+%Y-%m-%d_%H%M")
DATE_LABEL=$(date "+%Y-%m-%d %H:%M")
REPORT_FILE="${REPORTS_DIR}/${TIMESTAMP}.txt"

# ── Header ──────────────────────────────────────────────────────────────────
{
    echo "================================================================================"
    echo "  DAILY REPORT — ${DATE_LABEL}"
    echo "  Args: $*"
    echo "  Project: $(basename "$PROJECT_ROOT")"
    echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
    echo "================================================================================"
    echo ""
} > "$REPORT_FILE"

# ── Run scan ─────────────────────────────────────────────────────────────────
echo "Running morning scan... ($(date))"
echo "  Output → ${REPORT_FILE}"

# Run with a timeout so we always get partial output if something hangs
if command -v timeout &>/dev/null; then
    timeout "${TIMEOUT_MINUTES}m" \
        "$PYTHON" -m "$SCAN_MODULE" "$@" 2>&1 \
        | tee -a "$REPORT_FILE"
    EXIT_CODE=$?
else
    # macOS (no timeout by default) — use perl or just let it run
    "$PYTHON" -m "$SCAN_MODULE" "$@" 2>&1 \
        | tee -a "$REPORT_FILE"
    EXIT_CODE=$?
fi

# ── Footer ───────────────────────────────────────────────────────────────────
{
    echo ""
    echo "================================================================================"
    echo "  SCAN FINISHED — Exit code: ${EXIT_CODE}"
    echo "  Duration: started ${DATE_LABEL}, finished $(date '+%Y-%m-%d %H:%M')"
    echo "================================================================================"
} >> "$REPORT_FILE"

# ── Symlink latest ───────────────────────────────────────────────────────────
LATEST_LINK="${REPORTS_DIR}/latest.txt"
ln -sf "${TIMESTAMP}.txt" "$LATEST_LINK"

# ── Log entry ────────────────────────────────────────────────────────────────
REPORT_SIZE=$(wc -c < "$REPORT_FILE" 2>/dev/null || echo 0)
echo "[${TIMESTAMP}] exit=${EXIT_CODE} size=${REPORT_SIZE} file=${REPORT_FILE##*/} args=$*" >> "$LOG_FILE"

# ── Email (optional) ──────────────────────────────────────────────────────────
EMAIL_TO="${REPORT_EMAIL_TO:-}"
if [ -n "$EMAIL_TO" ] && [ -f "$LATEST_LINK" ]; then
    echo ""
    echo "  Sending email to ${EMAIL_TO}..."
    if "$PYTHON" "${PROJECT_ROOT}/scripts/send-report.py" --to "$EMAIL_TO" 2>&1; then
        echo "  Email sent."
    else
        echo "  Email failed (sendmail may not be configured). Set SMTP_* vars in .env to use SMTP."
    fi
fi

# ── Summary to stdout ────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "  REPORT SAVED → ${REPORT_FILE}  (${REPORT_SIZE} bytes)"
echo "  View via: cat ${LATEST_LINK}"
echo "  Exit code: ${EXIT_CODE}"
echo "================================================================================"

exit $EXIT_CODE
