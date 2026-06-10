#!/usr/bin/env bash
#=============================================================================
#  refresh_everything.sh — Sport-aware dispatcher for the full refresh chain
#
#  Replaces bin/refresh_nba_everything.sh and bin/refresh_mlb_everything.sh
#  with a single dispatcher. Takes a sport as the first argument and runs
#  the appropriate cache → fetch → train → cal → scan chain.
#
#  Usage:
#    ./bin/refresh_everything.sh <sport> [--no-scan]
#
#  Supported sports:
#    nba   cache+train+cal+scan (via src.main train nba, fit_nba_beta_cal, nba_bet)
#    mlb   cache+train+scan     (no cal — files don't exist; mlb_bet scan)
#    nhl   cache+train+scan     (via src.main train nhl, kalshi_nhl_unified)
#    nfl   cache+train+scan     (via src.main train nfl, kalshi_nfl_unified)
#
#  Examples:
#    ./bin/refresh_everything.sh nba              # full NBA chain
#    ./bin/refresh_everything.sh nba --no-scan    # skip the final live scan
#    ./bin/refresh_everything.sh mlb --no-scan    # MLB cache+train only
#
#  Cron can call: ./bin/refresh_everything.sh <sport> [--no-scan]
#  per the appropriate schedule for each sport's season window.
#
#  Errors are non-fatal — logged but exit 0 so cron doesn't spam alerts.
#=============================================================================

set -uo pipefail

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

# ── Args ────────────────────────────────────────────────────────────────────
SPORT="${1:-}"
shift || true  # remove sport from $@

DO_SCAN=true
for arg in "$@"; do
    case "$arg" in
        --no-scan) DO_SCAN=false ;;
    esac
done

if [ -z "$SPORT" ]; then
    echo "Usage: $0 <sport> [--no-scan]"
    echo "  Supported sports: nba, mlb, nhl, nfl"
    exit 1
fi

LOG_FILE="${LOG_DIR}/refresh_${SPORT}_everything.log"
SCAN_OUTPUT="${LOG_DIR}/refresh_${SPORT}_scan_output.txt"

mkdir -p "${LOG_DIR}"

log() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "${LOG_FILE}"
}

# ── Sport-specific config ───────────────────────────────────────────────────
# Per-sport: CACHE_GLOB, FETCH_CMD, TRAIN_CMD, CAL_CMD, SCAN_CMD
case "$SPORT" in
    nba)
        CACHE_GLOB="data/nba_cache/game_logs_v14.parquet"
        FETCH_CMD="from src.data.nba import NBADataSource; ds = NBADataSource(); ds.fetch_player_game_logs()"
        TRAIN_CMD="${PYTHON} -m src.main train nba"
        CAL_CMD="${PYTHON} -m scripts.fit_nba_beta_cal"
        SCAN_CMD="${PYTHON} -m src.scripts.nba_bet --scan"
        CURRENT_SEASON="2025-26"
        ;;
    mlb)
        CACHE_GLOB="data/cache/mlb/game_logs_*.parquet"
        FETCH_CMD="from src.data.mlb import MLBDataSource; ds = MLBDataSource(); ds.fetch_player_game_logs(['2026'])"
        TRAIN_CMD="${PYTHON} -m src.scripts.train_mlb_regression"
        CAL_CMD="${PYTHON} -m scripts.fit_mlb_beta_cal"
        SCAN_CMD="${PYTHON} -m src.scripts.mlb_bet --scan"
        CURRENT_SEASON="2026"
        ;;
    nhl)
        CACHE_GLOB="data/nhl_cache/*.parquet"
        FETCH_CMD="from src.data.nhl import NHLDataSource; ds = NHLDataSource(); ds.fetch_player_game_logs([2025, 2026])"
        TRAIN_CMD="${PYTHON} -m src.main train nhl"
        CAL_CMD=""  # NHL cal files don't exist yet
        SCAN_CMD="${PYTHON} -m src.scripts.kalshi_nhl_unified"
        CURRENT_SEASON="2025-26"
        ;;
    nfl)
        CACHE_GLOB="data/nfl_cache/*.parquet"
        FETCH_CMD="from src.data.nfl import NFLDataSource; ds = NFLDataSource(); ds.fetch_player_game_logs([2024, 2025])"
        TRAIN_CMD="${PYTHON} -m src.main train nfl"
        CAL_CMD="${PYTHON} -m src.scripts.refit_nfl_beta_cal"
        SCAN_CMD="${PYTHON} -m src.scripts.kalshi_nfl_unified"
        CURRENT_SEASON="2025"
        ;;
    *)
        echo "Unknown sport: $SPORT"
        echo "  Supported: nba, mlb, nhl, nfl"
        exit 1
        ;;
esac

# ── Header ──────────────────────────────────────────────────────────────────
log "==================================================================="
log "  REFRESH START — ${SPORT}  (scan=$([ \"$DO_SCAN\" = true ] && echo yes || echo no))"
log "==================================================================="

cd "${PROJECT_DIR}"

# ── Step 1: Delete cache ────────────────────────────────────────────────────
log "  Step 1: Deleting cache matching ${CACHE_GLOB}..."
DELETED_BYTES=0
if [[ "$CACHE_GLOB" == *"*"* ]]; then
    for f in $CACHE_GLOB; do
        if [ -f "$f" ]; then
            SZ=$(stat -f%z "$f" 2>/dev/null || echo 0)
            DELETED_BYTES=$((DELETED_BYTES + SZ))
            rm -f "$f"
        fi
    done
else
    if [ -f "$CACHE_GLOB" ]; then
        DELETED_BYTES=$(stat -f%z "$CACHE_GLOB" 2>/dev/null || echo 0)
        rm -f "$CACHE_GLOB"
    fi
fi
log "  Step 1: Deleted ${DELETED_BYTES} bytes"

# ── Step 2: Re-fetch ────────────────────────────────────────────────────────
log "  Step 2: Re-fetching ${SPORT} (${CURRENT_SEASON})..."
if "${PYTHON}" -c "
import sys
sys.path.insert(0, '.')
${FETCH_CMD}
print('  Fetched OK')
" >> "${LOG_FILE}" 2>&1; then
    log "  Step 2: SUCCESS"
else
    log "  Step 2: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 2 (fetch)"
    log "==================================================================="
    exit 0
fi

# ── Step 3: Re-train ───────────────────────────────────────────────────────
log "  Step 3: Re-training ${SPORT} models..."
if ${TRAIN_CMD} >> "${LOG_FILE}" 2>&1; then
    log "  Step 3: SUCCESS at $(date '+%H:%M:%S')"
else
    log "  Step 3: FAILED — see above for trace"
    log "==================================================================="
    log "  REFRESH FAILED at Step 3 (train)"
    log "==================================================================="
    exit 0
fi

# ── Step 4: Re-fit calibrations (optional per sport) ───────────────────────
if [ -n "${CAL_CMD}" ]; then
    log "  Step 4: Re-fitting ${SPORT} calibrations..."
    if ${CAL_CMD} >> "${LOG_FILE}" 2>&1; then
        log "  Step 4: SUCCESS"
    else
        log "  Step 4: FAILED — see above for trace"
        log "==================================================================="
        log "  REFRESH FAILED at Step 4 (calibration)"
        log "==================================================================="
        exit 0
    fi
else
    log "  Step 4: Skipped (no cal refit for ${SPORT})"
fi

# ── Step 5: Staleness check ─────────────────────────────────────────────────
log "  Step 5: Running staleness check..."
if "${PYTHON}" -m src.utils.staleness >> "${LOG_FILE}" 2>&1; then
    log "  Step 5: SUCCESS — all sports fresh"
else
    log "  Step 5: WARN — staleness check failed (other sports may be stale)"
fi

# ── Step 6: Live dry-run scan (optional) ───────────────────────────────────
if [ "$DO_SCAN" = true ] && [ -n "${SCAN_CMD}" ]; then
    log "  Step 6: Running live dry-run scan (top edges to ${SCAN_OUTPUT})..."
    if ${SCAN_CMD} > "${SCAN_OUTPUT}" 2>&1; then
        EDGE_COUNT=$(grep -cE 'Edge=|YES|NO' "${SCAN_OUTPUT}" 2>/dev/null || echo "?")
        log "  Step 6: SUCCESS — ${EDGE_COUNT} edge lines logged"
    else
        log "  Step 6: FAILED — see ${SCAN_OUTPUT} for details"
    fi
elif [ "$DO_SCAN" = false ]; then
    log "  Step 6: Skipped (--no-scan)"
else
    log "  Step 6: Skipped (no SCAN_CMD for ${SPORT})"
fi

# ── Footer ──────────────────────────────────────────────────────────────────
log ""
log "  REFRESH COMPLETE at $(date '+%H:%M:%S')"
log "==================================================================="

# Keep log manageable — truncate to 3000 lines
tail -n 3000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "${LOG_FILE}" || true

exit 0
