#!/usr/bin/env bash
#=============================================================================
#  refresh_wnba_cache.sh — Refresh WNBA cache + calibrations during season
#
#  Used by the daily WNBA cron schedule to auto-refresh once per day during
#  the WNBA regular season (May 1 - Oct 31) so the staleness guard passes
#  for live betting. 3-day stale data is the same risk class as NBA — WNBA
#  games happen daily during the season and an in-season pause is rare.
#
#  Usage (manual):
#    ./bin/refresh_wnba_cache.sh
#    ./bin/refresh_wnba_cache.sh --no-cal          # skip the cal refit
#
#  What it does:
#    1. Deletes the existing WNBA parquet
#    2. Re-fetches current season via the WNBA data source
#    3. Re-fits all BetaCal calibrations
#    4. Runs the staleness check (should pass after refresh)
#    5. Logs everything to logs/refresh_wnba_cache.log
#
#  Idempotent: safe to run multiple times (overwrites cache).
#  Errors are non-fatal — logged but exit 0 so cron doesn't spam alerts.
#=============================================================================

set -uo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/refresh_wnba_cache.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
CACHE_FILE="${PROJECT_DIR}/data/wnba_cache/wnba_games.parquet"

mkdir -p "${LOG_DIR}"

DO_CAL=true
for arg in "$@"; do
    case "$arg" in
        --no-cal) DO_CAL=false ;;
    esac
done

log() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "${LOG_FILE}"
}

# ── Header ──────────────────────────────────────────────────────────────────
log "==================================================================="
log "  REFRESH START — ${0##*/}  (cal=$([ "$DO_CAL" = true ] && echo yes || echo no))"
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

# ── Step 2: Re-fetch current WNBA season ────────────────────────────────────
log "  Step 2: Re-fetching WNBA games from data source..."
if "${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
from src.data.wnba import WNBADataSource
ds = WNBADataSource()
df = ds.fetch_player_game_logs()
print(f'  Fetched: {len(df)} rows')
if 'game_date' in df.columns and len(df) > 0:
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

# ── Step 3: Re-fit calibrations (optional) ──────────────────────────────────
if [ "$DO_CAL" = true ]; then
    log "  Step 3: Re-fitting WNBA BetaCal calibrations..."
    if "${PYTHON}" -m scripts.fit_wnba_beta_cal >> "${LOG_FILE}" 2>&1; then
        log "  Step 3: SUCCESS"
    else
        log "  Step 3: FAILED — see above for trace"
        log "==================================================================="
        log "  REFRESH FAILED at Step 3 (calibration)"
        log "==================================================================="
        exit 0
    fi
else
    log "  Step 3: Skipped (--no-cal)"
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
log "  REFRESH COMPLETE at $(date '+%H:%M:%S')"
log "==================================================================="

# Keep log manageable — truncate to 2000 lines
tail -n 2000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true

exit 0
