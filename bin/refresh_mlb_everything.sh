#!/usr/bin/env bash
#=============================================================================
#  refresh_mlb_everything.sh — Full nightly MLB refresh chain
#
#  Mirrors bin/refresh_nba_everything.sh but for MLB. Runs daily via cron
#  so the cache, models, and live scanner all stay in sync during the
#  regular season (Mar-Nov).
#
#  Usage (manual):
#    ./bin/refresh_mlb_everything.sh              # full chain
#    ./bin/refresh_mlb_everything.sh --no-scan    # skip the final live scan
#
#  What it does (in order):
#    1. Delete the current-season MLB game_logs parquet
#    2. Re-fetch the current season (2026) from statsapi.mlb.com
#    3. Re-train all MLB LightGBM regression models via train_mlb_regression.py
#    4. (No BetaCal step for MLB — calibration files don't exist yet; the
#       model is used directly. If/when calibrations are added, add a
#       step 4 here.)
#    5. Run the staleness check (should pass)
#    6. (Optional) Run the live dry-run scan to surface the top edges
#       in the log for next-morning review
#
#  Idempotent: safe to run multiple times.
#  Errors are non-fatal — logged but exit 0 so cron doesn't spam alerts.
#
#  Total runtime: 5-15 minutes (dominated by Step 3 retrain).
#=============================================================================

set -uo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/refresh_mlb_everything.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
CACHE_DIR="${PROJECT_DIR}/data/cache/mlb"
CURRENT_SEASON="2026"
SCAN_OUTPUT="${LOG_DIR}/refresh_mlb_scan_output.txt"

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

# ── Step 3: Re-train all MLB LightGBM models ───────────────────────────────
log "  Step 3: Re-training MLB LightGBM models (this takes 5-10 min)..."
if "${PYTHON}" -m src.scripts.train_mlb_regression >> "${LOG_FILE}" 2>&1; then
    log "  Step 3: SUCCESS — models retrained at $(date '+%H:%M:%S')"
else
    log "  Step 3: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 3 (train)"
    log "==================================================================="
    exit 0
fi

# ── Step 4: (No BetaCal for MLB — calibrations not yet implemented) ─────────
log "  Step 4: Skipped (MLB BetaCal not yet implemented — no calibration files)"

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
    if "${PYTHON}" -m src.scripts.mlb_bet --scan > "${SCAN_OUTPUT}" 2>&1; then
        EDGE_COUNT=$(grep -cE 'Edge=|edge=|YES|NO' "${SCAN_OUTPUT}" 2>/dev/null || echo "?")
        log "  Step 6: SUCCESS — ${EDGE_COUNT} edge lines logged to ${SCAN_OUTPUT}"
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

# Keep log manageable — truncate to 3000 lines
tail -n 3000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true

exit 0
