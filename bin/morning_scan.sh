#!/bin/bash
# Morning Scan Cron Wrapper
# Runs morning_scan --paper at 8 AM daily and logs results.
# Used by crontab: 0 8 * * * /Users/bpj520/sports-betting-ai/bin/morning_scan.sh

set -euo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/morning_scan.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python3"

mkdir -p "${LOG_DIR}"

{
    echo ""
    echo "============================================"
    echo "  MORNING SCAN — $(date '+%Y-%m-%d %H:%M')"
    echo "============================================"

    cd "${PROJECT_DIR}"

    echo ""
    echo "  Pre-flight: per-sport data + model + cal freshness"
    if ! "${PYTHON}" -m src.utils.preflight 2>&1; then
        echo "  [preflight] CRASHED — see trace above"
    fi

    echo ""
    "${PYTHON}" -m src.scripts.morning_scan --paper

    echo ""
    echo "── Live calibration refit (every ${REFIT_INTERVAL_DAYS:-7} days) ───"
    if [ -x "${PROJECT_DIR}/bin/refit_calibrations.sh" ]; then
        "${PROJECT_DIR}/bin/refit_calibrations.sh" || true
    else
        echo "  (refit_calibrations.sh not executable, skipping)"
    fi

    echo ""
    echo "  Done at $(date '+%H:%M:%S')"
    echo "============================================"
} >> "${LOG_FILE}" 2>&1

# Keep log manageable — truncate to 2000 lines
tail -n 2000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true
