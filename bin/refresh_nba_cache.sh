#!/usr/bin/env bash
#=============================================================================
#  refresh_nba_cache.sh — Refresh NBA cache + calibrations after a Finals game
#
#  Used by the 2026 NBA Finals cron schedule to auto-refresh 4hr after
#  each game so the staleness guard passes for live betting.
#
#  Usage (manual):
#    ./bin/refresh_nba_cache.sh
#
#  What it does:
#    1. Deletes the existing NBA parquet
#    2. Re-fetches 2025-26 Regular Season + Playoffs via nba_api
#    3. Re-fits all 11 BetaCal calibrations
#    4. Runs the staleness check (should pass after refresh)
#    5. Logs everything to logs/refresh_nba_cache.log
#
#  Idempotent: safe to run multiple times (overwrites cache).
#  Errors are non-fatal — logged but exit 0 so cron doesn't spam alerts.
#=============================================================================

set -uo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/refresh_nba_cache.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
CACHE_FILE="${PROJECT_DIR}/data/nba_cache/game_logs_v14.parquet"

mkdir -p "${LOG_DIR}"

log() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "${LOG_FILE}"
}

# ── Header ──────────────────────────────────────────────────────────────────
log "==================================================================="
log "  REFRESH START — ${0##*/}"
log "==================================================================="

cd "${PROJECT_DIR}"

# ── Step 1: Delete old cache ────────────────────────────────────────────────
if [ -f "${CACHE_FILE}" ]; then
    OLD_ROWS=$(stat -f%z "${CACHE_FILE}" 2>/dev/null || echo "?")
    log "  Step 1: Deleting old cache (${OLD_ROWS} bytes)..."
    rm -f "${CACHE_FILE}"
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
    NEW_ROWS=$(stat -f%z "${CACHE_FILE}" 2>/dev/null || echo "?")
    log "  Step 2: SUCCESS — new cache ${NEW_ROWS} bytes"
else
    log "  Step 2: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 2 (fetch)"
    log "==================================================================="
    exit 0
fi

# ── Step 3: Re-fit calibrations ─────────────────────────────────────────────
log "  Step 3: Re-fitting NBA BetaCal calibrations..."
if "${PYTHON}" -m scripts.fit_nba_beta_cal >> "${LOG_FILE}" 2>&1; then
    log "  Step 3: SUCCESS"
else
    log "  Step 3: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 3 (calibration)"
    log "==================================================================="
    exit 0
fi

# ── Step 4: Staleness check ─────────────────────────────────────────────────
log "  Step 4: Running staleness check..."
if "${PYTHON}" -m src.utils.staleness >> "${LOG_FILE}" 2>&1; then
    log "  Step 4: SUCCESS — all sports fresh"
else
    log "  Step 4: WARN — staleness check failed (other sports may be stale)"
fi

# ── Footer ──────────────────────────────────────────────────────────────────
log ""
log "  REFRESH COMPLETE"
log "==================================================================="

# Keep log manageable — truncate to 2000 lines
tail -n 2000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true

exit 0
