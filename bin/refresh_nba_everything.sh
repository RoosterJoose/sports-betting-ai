#!/usr/bin/env bash
#=============================================================================
#  refresh_nba_everything.sh — Full nightly NBA refresh chain
#
#  Replaces manual train+recal+scan workflow with one cron-callable script.
#  Prevents the regression-model staleness issue we hit on June 9, 2026
#  (cache refreshed, but models weren't retrained on the new data).
#
#  Usage (manual):
#    ./bin/refresh_nba_everything.sh              # full chain
#    ./bin/refresh_nba_everything.sh --no-scan    # skip the final live scan
#
#  What it does (in order):
#    1. Delete the existing NBA parquet cache
#    2. Re-fetch 2025-26 Regular Season + Playoffs via nba_api
#    3. Re-train all 11 NBA LightGBM regression models on the new cache
#       (this is the step that was missing in the stale-models bug)
#    4. Re-fit all 11 BetaCal calibrations
#    5. Run the staleness check (should pass)
#    6. (Optional) Run the live dry-run scan to surface the top edges
#       in the log for next-morning review
#
#  Idempotent: safe to run multiple times.
#  Errors are non-fatal — logged but exit 0 so cron doesn't spam alerts.
#
#  Total runtime: 10-20 minutes (dominated by Step 3 retrain).
#=============================================================================

set -uo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/refresh_nba_everything.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
CACHE_FILE="${PROJECT_DIR}/data/nba_cache/game_logs_v14.parquet"
SCAN_OUTPUT="${LOG_DIR}/refresh_nba_scan_output.txt"

mkdir -p "${LOG_DIR}"

DO_SCAN=true
for arg in "$@"; do
    case "$arg" in
        --no-scan) DO_SCAN=false ;;
    esac
done

log() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "${LOG_FILE}"
}

# ── Header ──────────────────────────────────────────────────────────────────
log "==================================================================="
log "  REFRESH START — ${0##*/}  (scan=$([ \"$DO_SCAN\" = true ] && echo yes || echo no))"
log "==================================================================="

cd "${PROJECT_DIR}"

# ── Step 1: Delete old cache ────────────────────────────────────────────────
if [ -f "${CACHE_FILE}" ]; then
    OLD_BYTES=$(stat -f%z "${CACHE_FILE}" 2>/dev/null || echo "?")
    log "  Step 1: Deleting old cache (${OLD_BYTES} bytes)..."
    rm -f "${CACHE_FILE}"
else
    log "  Step 1: No existing cache to delete"
fi

# ── Step 2: Re-fetch 2025-26 RS + Playoffs ──────────────────────────────────
log "  Step 2: Re-fetching 2025-26 RS + Playoffs from nba_api..."
if "${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
from src.data.nba import NBADataSource
ds = NBADataSource()
df = ds.fetch_player_game_logs()
print(f'  Fetched: {len(df)} rows')
print(f'  Date range: {df.game_date.min()} to {df.game_date.max()}')
" >> "${LOG_FILE}" 2>&1; then
    NEW_BYTES=$(stat -f%z "${CACHE_FILE}" 2>/dev/null || echo "?")
    log "  Step 2: SUCCESS — new cache ${NEW_BYTES} bytes"
else
    log "  Step 2: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 2 (fetch)"
    log "==================================================================="
    exit 0
fi

# ── Step 3: Re-train all 11 NBA LightGBM models (CRITICAL) ─────────────────
log "  Step 3: Re-training all 11 NBA LightGBM models (this takes 10-15 min)..."
if "${PYTHON}" -m src.main train nba >> "${LOG_FILE}" 2>&1; then
    log "  Step 3: SUCCESS — models retrained at $(date '+%H:%M:%S')"
else
    log "  Step 3: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 3 (train)"
    log "==================================================================="
    exit 0
fi

# ── Step 4: Re-fit calibrations ─────────────────────────────────────────────
log "  Step 4: Re-fitting NBA BetaCal calibrations..."
if "${PYTHON}" -m scripts.fit_nba_beta_cal >> "${LOG_FILE}" 2>&1; then
    log "  Step 4: SUCCESS"
else
    log "  Step 4: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 4 (calibration)"
    log "==================================================================="
    exit 0
fi

# ── Step 5: Staleness check ─────────────────────────────────────────────────
log "  Step 5: Running staleness check..."
if "${PYTHON}" -m src.utils.staleness >> "${LOG_FILE}" 2>&1; then
    log "  Step 5: SUCCESS — all sports fresh"
else
    log "  Step 5: WARN — staleness check failed (other sports may be stale)"
fi

# ── Step 6: Live dry-run scan (optional) ───────────────────────────────────
if [ "$DO_SCAN" = true ]; then
    log "  Step 6: Running live dry-run scan (top edges to ${SCAN_OUTPUT})..."
    if "${PYTHON}" -m src.scripts.nba_bet --scan > "${SCAN_OUTPUT}" 2>&1; then
        EDGE_COUNT=$(grep -cE 'Edge=' "${SCAN_OUTPUT}" 2>/dev/null || echo "?")
        log "  Step 6: SUCCESS — ${EDGE_COUNT} edges logged to ${SCAN_OUTPUT}"
    else
        log "  Step 6: FAILED — see ${SCAN_OUTPUT} for details"
    fi
else
    log "  Step 6: Skipped (--no-scan)"
fi

# ── Footer ──────────────────────────────────────────────────────────────────
log ""
log "  REFRESH COMPLETE at $(date '+%H:%M:%S')"
log "==================================================================="

# Keep log manageable — truncate to 3000 lines (this script is verbose)
tail -n 3000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true

exit 0
