#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Shared helpers for run_evaluate.sh, run_pretrain.sh, run_surgery.sh
# ═══════════════════════════════════════════════════════════════════════
# Source this from each pipeline script:
#   source "$(dirname "$0")/lib/common.sh"
#
# Required variables before sourcing:
#   MASTER_LOG  — path to master log file
#   MODE        — SANITY | POC | FULL
#   OUT_DIR     — outputs/sanity | outputs/poc | outputs/full
#   LOGDIR      — logs/
# ═══════════════════════════════════════════════════════════════════════

# ── Counters ──────────────────────────────────────────────────────────
STEP_COUNT=0
STEP_PASS=0
STEP_FAIL=0
PIPELINE_START=$(date +%s)

# ── Logging ───────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg" | tee -a "$MASTER_LOG"
}

banner() {
    local step_num="$1"
    local step_name="$2"
    local est_time="$3"
    echo "" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "  STEP ${step_num}: ${step_name}" | tee -a "$MASTER_LOG"
    echo "  Mode: ${MODE} | Est: ${est_time}" | tee -a "$MASTER_LOG"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
}

# ── Background HF upload (zero GPU idle time between steps) ──────────
UPLOAD_PID=""
bg_upload() {
    if [[ -n "$UPLOAD_PID" ]]; then
        wait "$UPLOAD_PID" 2>/dev/null
        UPLOAD_PID=""
    fi
    python -u src/utils/hf_outputs.py upload "$OUT_DIR" >> "$LOGDIR/hf_upload.log" 2>&1 &
    UPLOAD_PID=$!
    log "HF upload started in background (PID=$UPLOAD_PID)"
}

# ── Run a pipeline step (correct PIPESTATUS capture) ─────────────────
run_step() {
    local step_num="$1"; shift
    local step_name="$1"; shift
    local est_time="$1"; shift
    local log_file="$1"; shift
    local cmd=("$@")

    STEP_COUNT=$((STEP_COUNT + 1))
    banner "$step_num" "$step_name" "$est_time"
    log "CMD: python -u ${cmd[*]}"

    local step_start=$(date +%s)

    python -u "${cmd[@]}" 2>&1 | tee "$log_file" | tee -a "$MASTER_LOG"
    local exit_code=${PIPESTATUS[0]}  # capture Python's exit code (not tee's)

    local step_end=$(date +%s)
    local elapsed=$(( step_end - step_start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [[ $exit_code -eq 0 ]]; then
        log "PASSED: ${step_name} (${mins}m ${secs}s)"
        STEP_PASS=$((STEP_PASS + 1))
        bg_upload
        return 0
    else
        log "FAILED: ${step_name} (${mins}m ${secs}s) — exit code ${exit_code}"
        STEP_FAIL=$((STEP_FAIL + 1))
        return 1
    fi
}

# ── Verify step output (default: non-fatal; pass --fatal for hard stop)
verify() {
    local desc="$1"
    local code="$2"
    local fatal="${3:-}"

    if python -c "$code" 2>&1 | tee -a "$MASTER_LOG"; then
        log "VERIFY OK: $desc"
    else
        if [[ "$fatal" == "--fatal" ]]; then
            log "FATAL: VERIFY FAIL: $desc"
            exit 1
        else
            log "WARNING: VERIFY FAIL: $desc (continuing)"
        fi
    fi
}

# ── GPU watchdog (background, alerts if GPU drops below threshold) ────
WATCHDOG_PID=""
start_watchdog() {
    if [[ -f "src/utils/gpu_watchdog.py" ]]; then
        python src/utils/gpu_watchdog.py &
        WATCHDOG_PID=$!
        log "GPU watchdog started (PID=$WATCHDOG_PID)"
    fi
}

stop_watchdog() {
    if [[ -n "$WATCHDOG_PID" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null
        log "GPU watchdog stopped"
    fi
}

# ── Print pipeline summary ───────────────────────────────────────────
print_summary() {
    local pipeline_name="$1"
    local pipeline_end=$(date +%s)
    local total_elapsed=$(( pipeline_end - PIPELINE_START ))
    local total_hours=$(( total_elapsed / 3600 ))
    local total_mins=$(( (total_elapsed % 3600) / 60 ))
    local total_secs=$(( total_elapsed % 60 ))

    echo "" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "  ${pipeline_name} PIPELINE COMPLETE" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
    echo "  Mode:       ${MODE}" | tee -a "$MASTER_LOG"
    echo "  Total:      ${total_hours}h ${total_mins}m ${total_secs}s" | tee -a "$MASTER_LOG"
    echo "  Steps:      ${STEP_PASS} passed, ${STEP_FAIL} failed, ${STEP_COUNT} total" | tee -a "$MASTER_LOG"
    echo "  Outputs:    ${OUT_DIR}/" | tee -a "$MASTER_LOG"
    echo "  Master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
    echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
}

# ── Wait for final background upload + cleanup ───────────────────────
finalize() {
    local pipeline_name="${1:?FATAL: finalize requires pipeline_name argument (e.g., finalize \"ExPLoRA\")}"
    print_summary "$pipeline_name"

    if [[ -n "$UPLOAD_PID" ]]; then
        log "Waiting for final HF upload..."
        wait "$UPLOAD_PID" 2>/dev/null
        log "Final HF upload complete"
    fi

    stop_watchdog

    if [[ $STEP_FAIL -gt 0 ]]; then
        echo "" | tee -a "$MASTER_LOG"
        echo "  WARNING: ${STEP_FAIL} step(s) failed. Check individual logs above." | tee -a "$MASTER_LOG"
        echo "  All steps with checkpoints can be RE-RUN safely (auto-resume)." | tee -a "$MASTER_LOG"
        exit 1
    fi
    exit 0
}
