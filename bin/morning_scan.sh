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
    "${PYTHON}" -m src.scripts.morning_scan --paper

    echo ""
    echo "  Done at $(date '+%H:%M:%S')"
    echo "============================================"
} >> "${LOG_FILE}" 2>&1

# Keep log manageable — truncate to 2000 lines
tail -n 2000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true
