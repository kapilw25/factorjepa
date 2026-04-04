#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch10: V-JEPA Continual Pretraining (FactorJEPA)
# Pipeline: m09(train) → m05(re-embed) → m06 → m06b → m05(shuffled) → m07 → m08 → m08b
#
# USAGE:
#   ./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log  # ~5 min
#   ./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log      # ~67h (115K, 1 epoch)
#
# CACHE CONTROL (output_guard skips completed steps automatically):
#   Use all cached:    ./scripts/run_pretrain.sh --FULL
#   Re-run everything: rm -rf outputs/full/m09_* && ./scripts/run_pretrain.sh --FULL
#   Re-run training:   rm outputs/full/m09_lambda*/student_encoder.pt && ...
#   Re-run re-embed:   rm outputs/full/embeddings_vjepa_lambda*.npy && ...
#
# PREREQUISITES:
#   1. ./setup_env_uv.sh --gpu --from-wheels
#   2. rsync -avhP data/ <gpu-host>:/workspace/factorjepa/data/   # from Mac (~17 min)
#   3. data/full_local/ exists (from m00d or rsync)
#   4. data/val_1k_local/ exists (for validation during training)
#   5. Ch9 complete: outputs/full/m06_metrics.json (needed for m08b comparison)
#   6. tmux new -s ch10   # run inside tmux for long runs
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# Reduce CUDA memory fragmentation (recovers ~10-14GB on long training runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
SUBSET_FLAG=""
CONFIG="configs/pretrain/vitg16_indian.yaml"

usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL [--config CONFIG]"
    echo ""
    echo "  --SANITY   Quick validation (~5 min, 900 train clips from config)"
    echo "  --POC      10K subset: 1-epoch ablation + 5-epoch winner (~5h)"
    echo "  --FULL     115K full corpus: no subset, dataset size from manifest (~28h)"
    echo "  --config   Override YAML config (default: $CONFIG)"
    exit 1
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --SANITY)
            MODE="SANITY"
            MODE_FLAG="--SANITY"
            SUBSET_FLAG=""
            OUT_DIR="outputs/sanity"
            shift
            ;;
        --POC)
            MODE="POC"
            MODE_FLAG="--FULL"
            SUBSET_FLAG="--subset data/subset_10k.json"
            OUT_DIR="outputs/poc"
            shift
            ;;
        --FULL)
            MODE="FULL"
            MODE_FLAG="--FULL"
            SUBSET_FLAG=""
            OUT_DIR="outputs/full"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[[ -z "$MODE" ]] && usage

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Read MIN_CLIPS from configs/pipeline.yaml (no hardcoding in shell)
PIPELINE_YAML="configs/pipeline.yaml"
if [[ "$MODE" == "SANITY" ]]; then
    MIN_CLIPS=$(python -u src/utils/config.py get-yaml "$PIPELINE_YAML" verify.sanity_min_clips)
else
    MIN_CLIPS=$(python -u src/utils/config.py get-yaml "$PIPELINE_YAML" verify.full_min_clips)
fi

LOGDIR="logs"
mkdir -p "$LOGDIR" "$OUT_DIR"

MASTER_LOG="$LOGDIR/ch10_${MODE}_$(date +%Y%m%d_%H%M%S).log"

# Activate venv
if [[ -d "venv_walkindia" ]]; then
    source venv_walkindia/bin/activate
else
    echo "ERROR: venv_walkindia not found. Run: ./setup_env_uv.sh --gpu"
    exit 1
fi

# ── Helper functions (same as run_evaluate.sh) ────────────────────────

STEP_COUNT=0
STEP_PASS=0
STEP_FAIL=0
PIPELINE_START=$(date +%s)

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg" | tee -a "$MASTER_LOG"
}

# ── GPU watchdog (background, emails alert if GPU < 50% for 5 min) ───
WATCHDOG_PID=""
if [[ -f "scripts/gpu_watchdog.py" ]]; then
    python scripts/gpu_watchdog.py &
    WATCHDOG_PID=$!
    log "GPU watchdog started (PID=$WATCHDOG_PID)"
fi

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

# Background HF upload — runs between steps, zero GPU idle time
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

verify() {
    local desc="$1"
    local code="$2"
    if python -c "$code" 2>&1 | tee -a "$MASTER_LOG"; then
        log "VERIFY OK: $desc"
    else
        log "FATAL: VERIFY FAIL: $desc"
        exit 1
    fi
}

# ── Time estimates ────────────────────────────────────────────────────
if [[ "$MODE" == "SANITY" ]]; then
    T_M09="~3 min (8 steps)"
    T_M05_RE="~30 sec"
    T_M06_RE="~10 sec"
    T_M08B_RE="~5 sec"
elif [[ "$MODE" == "POC" ]]; then
    T_M09="~35 min/lambda (80 steps)"
    T_M05_RE="~1h 47min (10K clips at 1.55 clips/s)"
    T_M06_RE="~1 min"
    T_M08B_RE="~30 sec"
else  # FULL
    T_M09="~6.4h/lambda (929 steps)"
    T_M05_RE="~20h (115K clips at 1.55 clips/s)"
    T_M06_RE="~5 min"
    T_M08B_RE="~30 sec"
fi

# ── Auto-detect optimal batch size from profiler ─────────────────────
PROFILE_JSON="outputs/profile/profile_data.json"
BATCH_FLAG=""

if [[ ! -f "$PROFILE_JSON" ]]; then
    log "No profiler data found. Running VRAM profiler (~5 min)..."
    python -u scripts/profile_vram.py 2>&1 | tee "$LOGDIR/profile_vram.log" | tee -a "$MASTER_LOG"
    if [[ ! -f "$PROFILE_JSON" ]]; then
        log "FATAL: Profiler failed. Cannot determine optimal batch size."
        exit 1
    fi
fi

if [[ -f "$PROFILE_JSON" ]]; then
    OPTIMAL_BS=$(python -u src/utils/gpu_batch.py optimal-bs --profile-json "$PROFILE_JSON")
    BATCH_FLAG="--batch-size $OPTIMAL_BS"
    log "Auto batch size: ${OPTIMAL_BS} (from profiler, <=75% VRAM)"
fi

# ── Local data (mode-dependent) ──────────────────────────────────────
if [[ "$MODE" == "FULL" ]]; then
    LOCAL_DATA="data/full_local"
else
    LOCAL_DATA="data/subset_10k_local"
fi
LOCAL_FLAG=""

if [[ -d "$LOCAL_DATA" && -f "$LOCAL_DATA/manifest.json" ]]; then
    log "Local data exists: $LOCAL_DATA"
    LOCAL_FLAG="--local-data $LOCAL_DATA"
elif [[ "$MODE" == "SANITY" ]]; then
    # SANITY can stream (small clip count)
    log "No local data — SANITY will use HF streaming"
else
    log "FATAL: Local data not found at $LOCAL_DATA"
    if [[ "$MODE" == "FULL" ]]; then
        log "Run: python -u src/m00d_download_subset.py --FULL"
    else
        log "Run: python -u src/m00d_download_subset.py --FULL --subset data/subset_10k.json"
    fi
    exit 1
fi

# ── Val subset + local data (pre-generated by m00c + m00d) ────────────
VAL_SUBSET="data/val_1k.json"
VAL_LOCAL="data/val_1k_local"
VAL_FLAG=""

if [[ "$MODE" != "SANITY" ]]; then
    if [[ ! -f "$VAL_SUBSET" ]]; then
        log "FATAL: $VAL_SUBSET not found. Generate on Mac first:"
        log "  python -u src/m00c_sample_subset.py --POC --n 1000 --seed 99 --output $VAL_SUBSET"
        exit 1
    fi
    VAL_FLAG="--val-subset $VAL_SUBSET"
    if [[ -d "$VAL_LOCAL" && -f "$VAL_LOCAL/manifest.json" ]]; then
        VAL_FLAG="$VAL_FLAG --val-local-data $VAL_LOCAL"
        log "Val data: $VAL_SUBSET + $VAL_LOCAL"
    else
        log "WARNING: $VAL_LOCAL not found. Val clips will load from training --local-data (may miss some)."
        log "  Fix: python -u src/m00d_download_subset.py --FULL --subset $VAL_SUBSET  (on Mac, ~57 min)"
        log "Val data: $VAL_SUBSET (no local shards)"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════════

log "Ch10 pipeline starting (mode=${MODE})"
log "Config: ${CONFIG}"
log "Master log: ${MASTER_LOG}"
echo "" | tee -a "$MASTER_LOG"
log "=== PRE-FLIGHT ==="

python -u src/utils/output_guard.py preflight_gpu pretrain "$CONFIG" "$OUT_DIR" 2>&1 | tee -a "$MASTER_LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log "FATAL: GPU/package pre-flight failed."
    exit 1
fi

log "Pre-flight complete."

# ── Output preflight: check all inputs/outputs before GPU work ────────
log "=== OUTPUT PREFLIGHT ==="
python -u src/utils/output_guard.py preflight_pretrain "$OUT_DIR" "$CONFIG" 2>&1 | tee -a "$MASTER_LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log "FATAL: Output preflight failed or aborted."
    exit 1
fi

log "Starting pipeline..."
echo "" | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: TRAINING (winner lambda from ablation_winner.json)
# ═══════════════════════════════════════════════════════════════════════

# Read epochs from YAML config (mode-aware)
EPOCH_KEY="optimization.max_epochs.${MODE,,}"  # sanity, poc, or full
WINNER_EPOCHS=$(python -u src/utils/config.py get-yaml "$CONFIG" "$EPOCH_KEY")

# Read winner lambda from POC ablation (selected by lowest best_val_loss)
# Search: outputs/full/ first (copied), then outputs/poc/ (original from ablation)
WINNER_JSON=""
for _dir in "${OUT_DIR}" "outputs/poc"; do
    if [[ -f "${_dir}/ablation_winner.json" ]]; then
        WINNER_JSON="${_dir}/ablation_winner.json"
        break
    fi
done

if [[ -n "$WINNER_JSON" ]]; then
    WINNER_LAMBDA=$(python -u src/utils/config.py get-json "$WINNER_JSON" winner_lambda)
    WINNER_DIR=$(python -u src/utils/config.py get-json "$WINNER_JSON" winner_dir)
    log "Winner from ablation: lambda=${WINNER_LAMBDA} (${WINNER_DIR}) [from ${WINNER_JSON}]"
else
    log "FATAL: ablation_winner.json not found in ${OUT_DIR}/ or outputs/poc/"
    log "  Run POC lambda ablation first (Ch10-2) to select winner by best_val_loss."
    exit 1
fi
WINNER_OUT="${OUT_DIR}/m09_${WINNER_DIR}"

log "Training: lambda=${WINNER_LAMBDA} (${WINNER_DIR}), ${WINNER_EPOCHS} epochs"

# Check if already trained with sufficient epochs
if [[ -f "${WINNER_OUT}/student_encoder.pt" && -f "${WINNER_OUT}/training_summary.json" ]]; then
    EXISTING_EPOCHS=$(python -u src/utils/config.py get-json "${WINNER_OUT}/training_summary.json" epochs)
    if python -c "exit(0 if float('${EXISTING_EPOCHS}') >= float('${WINNER_EPOCHS}') else 1)"; then
        log "Student already trained for ${EXISTING_EPOCHS} epochs (>= ${WINNER_EPOCHS}). Skipping."
    else
        log "Student has ${EXISTING_EPOCHS} epochs, need ${WINNER_EPOCHS}. Re-training."
        rm -f "${WINNER_OUT}/student_encoder.pt"
    fi
fi

run_step "train" "m09 pretrain (lambda=${WINNER_LAMBDA}, ${WINNER_EPOCHS} epochs)" "$T_M09" \
    "$LOGDIR/m09_${MODE,,}_${WINNER_DIR}.log" \
    src/m09_pretrain.py --config "$CONFIG" --lambda-reg "$WINNER_LAMBDA" \
        --max-epochs "$WINNER_EPOCHS" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG \
    || { log "FATAL: m09 training failed."; exit 1; }

verify "Training" "
from utils.output_guard import verify_training_artifacts
verify_training_artifacts('${WINNER_OUT}')
"

# Re-embed with trained encoder
WINNER_MODEL="${WINNER_OUT}/student_encoder.pt"
WINNER_ENCODER="vjepa_${WINNER_DIR}"
run_step "winner-embed" "m05 re-embed (winner lambda=${WINNER_LAMBDA})" "$T_M05_RE" \
    "$LOGDIR/m05_${MODE,,}_winner.log" \
    src/m05_vjepa_embed.py --model "$WINNER_MODEL" --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
    || { log "FATAL: Winner re-embed failed."; exit 1; }

# Metrics on deep-trained embeddings
run_step "winner-metrics" "m06 metrics (winner lambda=${WINNER_LAMBDA})" "$T_M06_RE" \
    "$LOGDIR/m06_${MODE,,}_winner.log" \
    src/m06_faiss_metrics.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner metrics failed."; exit 1; }

log "Phase 2 complete: train + re-embed + metrics for lambda=${WINNER_LAMBDA}"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: FULL EVALUATION (m06b + shuffled + m07 UMAP + m08 plots + m08b)
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== PHASE 3: FULL EVALUATION ==="

# m06b: Temporal correlation for adapted encoder (motion features from Ch9 m04d)
run_step "winner-temporal" "m06b temporal corr (winner)" "~2 min" \
    "$LOGDIR/m06b_${MODE,,}_winner.log" \
    src/m06b_temporal_corr.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner temporal correlation failed."; exit 1; }

# m05+m06 shuffled adapted: temporal ablation (~1h 50min)
SHUFFLED_ENCODER="${WINNER_ENCODER}_shuffled"
run_step "winner-shuffled" "m05 shuffled adapted (temporal ablation)" "$T_M05_RE" \
    "$LOGDIR/m05_${MODE,,}_winner_shuffled.log" \
    src/m05_vjepa_embed.py --model "$WINNER_MODEL" --encoder "$SHUFFLED_ENCODER" --shuffle \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
    || { log "FATAL: Shuffled adapted embed failed."; exit 1; }

run_step "winner-shuffled-metrics" "m06 metrics (shuffled adapted)" "$T_M06_RE" \
    "$LOGDIR/m06_${MODE,,}_winner_shuffled.log" \
    src/m06_faiss_metrics.py --encoder "$SHUFFLED_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Shuffled adapted metrics failed."; exit 1; }

# m07: UMAP for adapted encoder
run_step "winner-umap" "m07 UMAP (winner)" "~5 min" \
    "$LOGDIR/m07_${MODE,,}_winner.log" \
    src/m07_umap.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner UMAP failed."; exit 1; }

# m08: Plots for adapted encoder
run_step "winner-plots" "m08 plots (winner)" "~2 min" \
    "$LOGDIR/m08_${MODE,,}_winner.log" \
    src/m08_plot.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner plots failed."; exit 1; }

# m08b: Compare all encoders (frozen + baselines + adapted)
# Build encoder list: all Ch9 encoders + adapted winner
COMPARE_LIST="vjepa,random,dinov2,clip,vjepa_shuffled,${WINNER_ENCODER},${SHUFFLED_ENCODER}"
run_step "final-compare" "m08b comparison (frozen + baselines + adapted)" "~30 sec" \
    "$LOGDIR/m08b_${MODE,,}_ch10.log" \
    src/m08b_compare.py --encoders "$COMPARE_LIST" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Final comparison failed."; exit 1; }

log "Phase 3 complete: UMAP + plots + comparison for ${WINNER_ENCODER}"

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== FINAL VERIFICATION: Ch10 Ablation Outputs ==="

python -u src/utils/output_guard.py verify_pretrain_final "$OUT_DIR" "$CONFIG" 2>&1 | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo "" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  Ch10 PIPELINE COMPLETE" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  Mode:       ${MODE}" | tee -a "$MASTER_LOG"
echo "  Config:     ${CONFIG}" | tee -a "$MASTER_LOG"
echo "  Total:      ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s" | tee -a "$MASTER_LOG"
echo "  Steps:      ${STEP_PASS} passed, ${STEP_FAIL} failed, ${STEP_COUNT} total" | tee -a "$MASTER_LOG"
echo "  Outputs:    ${OUT_DIR}/" | tee -a "$MASTER_LOG"
echo "  Master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

# Wait for final background upload
if [[ -n "$UPLOAD_PID" ]]; then
    log "Waiting for final HF upload..."
    wait "$UPLOAD_PID" 2>/dev/null
    log "Final HF upload complete"
fi

# Kill watchdog
if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null
    log "GPU watchdog stopped"
fi

if [[ $STEP_FAIL -gt 0 ]]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "  FATAL: ${STEP_FAIL} step(s) failed. Check individual logs above." | tee -a "$MASTER_LOG"
    exit 1
fi

exit 0
