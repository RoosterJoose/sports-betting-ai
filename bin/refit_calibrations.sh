#!/bin/bash
# Live Calibration Refit Cron Wrapper
#
# Refits BetaCal calibrators (NBA + MLB) from the latest resolved trade
# outcomes in data/trade_tracker.db so calibration doesn't drift over
# time. Designed to be called from bin/morning_scan.sh but can also be
# run standalone for manual refits.
#
# Frequency control via state file: data/.last_calibration_refit
# records the date of the last successful refit. The wrapper only runs
# the (slow) refit scripts if N+ days have elapsed since that date.
#
# Configuration:
#   REFIT_INTERVAL_DAYS  — refit cadence in days (default 7)
#   REFIT_FORCE=1         — bypass the frequency check (manual refit)
#   REFIT_MIN_SAMPLES=30  — min resolved trades required per stat
#
# Usage:
#   ./bin/refit_calibrations.sh                 # respect cadence (default weekly)
#   REFIT_FORCE=1 ./bin/refit_calibrations.sh  # force refit now
#   REFIT_INTERVAL_DAYS=3 ./bin/refit_calibrations.sh  # every 3 days
#
# Always returns 0 so a failed refit doesn't break the morning scan.

set -uo pipefail  # NOT -e: we want to continue past per-script failures

PROJECT_DIR="/Users/bpj520/sports-betting-ai"
LOG_DIR="${HOME}/logs"
LOG_FILE="${LOG_DIR}/refit_calibrations.log"
PYTHON="${PROJECT_DIR}/.venv/bin/python3"

REFIT_INTERVAL_DAYS="${REFIT_INTERVAL_DAYS:-7}"
# Validate REFIT_INTERVAL_DAYS is a positive integer; fall back to 7
# if unset/zero/negative/non-numeric. The `2>/dev/null` suppresses the
# `[` error when the value isn't a number (e.g. REFIT_INTERVAL_DAYS=foo).
if ! [ "${REFIT_INTERVAL_DAYS}" -gt 0 ] 2>/dev/null; then
    REFIT_INTERVAL_DAYS=7
fi
REFIT_MIN_SAMPLES="${REFIT_MIN_SAMPLES:-30}"
STATE_FILE="${PROJECT_DIR}/data/.last_calibration_refit"

mkdir -p "${LOG_DIR}"

# ── Frequency check ──────────────────────────────────────────────────────
should_refit() {
    if [ "${REFIT_FORCE:-0}" = "1" ]; then
        echo "REFIT_FORCE=1 set — bypassing frequency check"
        return 0
    fi
    if [ ! -f "${STATE_FILE}" ]; then
        echo "No state file (${STATE_FILE}) — running first refit"
        return 0
    fi
    local last_run
    last_run=$(cat "${STATE_FILE}" 2>/dev/null | tr -d '[:space:]')
    if [ -z "${last_run}" ]; then
        echo "Empty state file — running refit"
        return 0
    fi
    # macOS uses `date -j -f`, Linux uses `date -d`. Try both.
    local last_epoch
    last_epoch=$(date -j -f "%Y-%m-%d" "${last_run}" +%s 2>/dev/null \
                 || date -d "${last_run}" +%s 2>/dev/null \
                 || echo 0)
    local now_epoch
    now_epoch=$(date +%s)
    local interval_seconds=$((REFIT_INTERVAL_DAYS * 86400))
    local elapsed=$((now_epoch - last_epoch))
    local elapsed_days=$((elapsed / 86400))
    if [ "${elapsed}" -ge "${interval_seconds}" ]; then
        echo "Last refit: ${last_run} (${elapsed_days} days ago, threshold: ${REFIT_INTERVAL_DAYS})"
        return 0
    else
        echo "Skipped (last refit: ${last_run}, ${elapsed_days} days ago, threshold: ${REFIT_INTERVAL_DAYS})"
        return 1
    fi
}

# ── Run wrapper ──────────────────────────────────────────────────────────
{
    echo ""
    echo "============================================"
    echo "  LIVE CALIBRATION REFIT — $(date '+%Y-%m-%d %H:%M')"
    echo "  Interval: ${REFIT_INTERVAL_DAYS} days  |  Min samples: ${REFIT_MIN_SAMPLES}"
    echo "============================================"

    if ! should_refit; then
        echo "  No refit due."
        echo ""
        echo "  Done at $(date '+%H:%M:%S')"
        echo "============================================"
        exit 0
    fi

    cd "${PROJECT_DIR}"
    nba_ok=1
    mlb_ok=1
    wnba_ok=1
    nhl_ok=1

    echo ""
    echo "── Refitting NBA calibrations ─────────────────────────────"
    if "${PYTHON}" scripts/refit_nba_beta_cal_live.py \
            --min-samples "${REFIT_MIN_SAMPLES}" 2>&1; then
        echo "  ✓ NBA refit complete"
    else
        echo "  ✗ NBA refit failed (continuing to MLB)"
        nba_ok=0
    fi

    echo ""
    echo "── Refitting MLB calibrations ─────────────────────────────"
    if "${PYTHON}" scripts/refit_mlb_beta_cal_live.py \
            --min-samples "${REFIT_MIN_SAMPLES}" 2>&1; then
        echo "  ✓ MLB refit complete"
    else
        echo "  ✗ MLB refit failed"
        mlb_ok=0
    fi

    echo ""
    echo "── Refitting WNBA calibrations ────────────────────────────"
    if [ -f "${PROJECT_DIR}/scripts/refit_wnba_beta_cal_live.py" ]; then
        if "${PYTHON}" scripts/refit_wnba_beta_cal_live.py \
                --min-samples "${REFIT_MIN_SAMPLES}" 2>&1; then
            echo "  ✓ WNBA refit complete"
        else
            echo "  ✗ WNBA refit failed"
            wnba_ok=0
        fi
    else
        echo "  (refit_wnba_beta_cal_live.py not found, skipping)"
        wnba_ok=0
    fi

    echo ""
    echo "── Refitting NHL calibrations ─────────────────────────────"
    if [ -f "${PROJECT_DIR}/scripts/refit_nhl_beta_cal_live.py" ]; then
        if "${PYTHON}" scripts/refit_nhl_beta_cal_live.py \
                --min-samples "${REFIT_MIN_SAMPLES}" 2>&1; then
            echo "  ✓ NHL refit complete"
        else
            echo "  ✗ NHL refit failed"
            nhl_ok=0
        fi
    else
        echo "  (refit_nhl_beta_cal_live.py not found, skipping)"
        nhl_ok=0
    fi

    # Update state file only if at least one refit succeeded
    if [ "${nba_ok}" = "1" ] || [ "${mlb_ok}" = "1" ] || \
       [ "${wnba_ok}" = "1" ] || [ "${nhl_ok}" = "1" ]; then
        date '+%Y-%m-%d' > "${STATE_FILE}"
        echo ""
        echo "  ✓ State file updated: ${STATE_FILE} = $(cat ${STATE_FILE})"
        if [ "${nba_ok}" = "0" ]; then
            echo "  ⚠ NBA refit failed; calibration may drift until next attempt"
        fi
        if [ "${mlb_ok}" = "0" ]; then
            echo "  ⚠ MLB refit failed; calibration may drift until next attempt"
        fi
        if [ "${wnba_ok}" = "0" ]; then
            echo "  ⚠ WNBA refit failed; calibration may drift until next attempt"
        fi
        if [ "${nhl_ok}" = "0" ]; then
            echo "  ⚠ NHL refit failed; calibration may drift until next attempt"
        fi
    else
        echo ""
        echo "  ✗ All 4 refits failed — state file NOT updated (will retry tomorrow)"
    fi

    echo ""
    echo "  Done at $(date '+%H:%M:%S')"
    echo "============================================"
} >> "${LOG_FILE}" 2>&1

# Keep log manageable — truncate to 2000 lines (matches morning_scan.sh pattern)
tail -n 2000 "${LOG_FILE}" > "${LOG_FILE}.tmp" 2>/dev/null \
    && mv "${LOG_FILE}.tmp" "${LOG_FILE}" \
    || true

# Always return 0 — a failed refit must not break the morning scan
exit 0
