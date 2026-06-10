#!/usr/bin/env bash
#=============================================================================
#  refresh_mlb_cache.sh — Refresh MLB cache daily
#
#  Used by the daily cron to auto-refresh MLB data so the staleness guard
#  stays green during the regular season (Mar-Nov).
#
#  Usage (manual):
#    ./bin/refresh_mlb_cache.sh
#
#  What it does:
#    1. Deletes the current-season MLB game_logs parquet
#    2. Re-fetches the current season (2026) from statsapi.mlb.com
#    3. Runs the staleness check (should pass after refresh)
#    4. Logs everything to logs/refresh_mlb_cache.log
#
#  Idempotent: safe to run multiple times (overwrites cache).
#  Errors are non-fatal — logged but exit 0 so cron doesn't spam alerts.
#
#  Schedule: daily 6:00 AM PT (after most west-coast games complete)
#=============================================================================

set -uo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/refresh_mlb_cache.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
CACHE_DIR="${PROJECT_DIR}/data/cache/mlb"
CURRENT_SEASON="2026"

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

# ── Step 1: Delete current-season cache files ───────────────────────────────
log "  Step 1: Deleting current-season cache (${CURRENT_SEASON})..."
DELETED_BYTES=0
for f in "${CACHE_DIR}/game_logs_${CURRENT_SEASON}"*.parquet; do
    if [ -f "$f" ]; then
        SZ=$(stat -f%z "$f" 2>/dev/null || echo 0)
        DELETED_BYTES=$((DELETED_BYTES + SZ))
        rm -f "$f"
        log "    Deleted: $f (${SZ} bytes)"
    fi
done
log "  Step 1: Deleted ${DELETED_BYTES} bytes of stale cache"

# ── Step 2: Re-fetch current season ─────────────────────────────────────────
log "  Step 2: Re-fetching MLB ${CURRENT_SEASON} from statsapi.mlb.com..."
if "${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
from src.data.mlb import MLBDataSource
ds = MLBDataSource()
df = ds.fetch_player_game_logs(['${CURRENT_SEASON}'])
print(f'  Fetched: {len(df)} rows')
if 'game_date' in df.columns and not df.empty:
    import pandas as pd
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    print(f'  Date range: {df.game_date.min()} to {df.game_date.max()}')
" >> "${LOG_FILE}" 2>&1; then
    NEW_BYTES=0
    for f in "${CACHE_DIR}/game_logs_${CURRENT_SEASON}"*.parquet; do
        if [ -f "$f" ]; then
            SZ=$(stat -f%z "$f" 2>/dev/null || echo 0)
            NEW_BYTES=$((NEW_BYTES + SZ))
        fi
    done
    log "  Step 2: SUCCESS — new cache ${NEW_BYTES} bytes"
else
    log "  Step 2: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 2 (fetch)"
    log "==================================================================="
    exit 0
fi

# ── Step 3: Staleness check ─────────────────────────────────────────────────
log "  Step 3: Running staleness check..."
if "${PYTHON}" -m src.utils.staleness >> "${LOG_FILE}" 2>&1; then
    log "  Step 3: SUCCESS — all sports fresh"
else
    log "  Step 3: WARN — staleness check failed (other sports may be stale)"
fi

# ── Footer ──────────────────────────────────────────────────────────────────
log ""
log "  REFRESH COMPLETE"
log "==================================================================="

# Keep log manageable — truncate to 2000 lines
tail -n 2000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true

exit 0
