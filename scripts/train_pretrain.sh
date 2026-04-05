#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch10: V-JEPA Continual Pretraining — TRAINING ONLY
# Pipeline: m09(ablation + winner training) → student_encoder.pt
# Next: ./scripts/run_embed.sh → ./scripts/run_eval.sh
#
# USAGE:
#   ./scripts/train_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log  # ~5 min
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

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

# ── Shared helpers (log, banner, run_step, verify, bg_upload) ────────
source "$(dirname "$0")/lib/common.sh"
start_watchdog

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

# ── Auto-detect optimal batch size ────────────────────────────────────
# Priority: 1) profiler JSON, 2) pipeline.yaml, 3) run profiler
PROFILE_JSON="outputs/profile/training/profile_data.json"
BATCH_FLAG=""
OPTIMAL_BS=""

# 1) Try profiler JSON
if [[ -f "$PROFILE_JSON" ]]; then
    OPTIMAL_BS=$(python -c "import json; d=json.load(open('$PROFILE_JSON')); print(d.get('training',{}).get('optimal_bs', 0))")
    [[ "$OPTIMAL_BS" == "0" ]] && OPTIMAL_BS=""
fi

# 2) Fallback to pipeline.yaml (no 20-min profiler wait)
if [[ -z "$OPTIMAL_BS" ]]; then
    OPTIMAL_BS=$(python -c "import yaml; c=yaml.safe_load(open('configs/pipeline.yaml')); print(c['gpu'].get('training_batch_size', 0))")
    [[ "$OPTIMAL_BS" == "0" ]] && OPTIMAL_BS=""
fi

# 3) Last resort: run profiler
if [[ -z "$OPTIMAL_BS" ]]; then
    log "No batch size in profiler or pipeline.yaml. Running VRAM profiler (~15 min)..."
    python -u src/utils/profile_vram.py --training 2>&1 | tee "$LOGDIR/profile_vram.log" | tee -a "$MASTER_LOG"
    if [[ -f "$PROFILE_JSON" ]]; then
        OPTIMAL_BS=$(python -c "import json; d=json.load(open('$PROFILE_JSON')); print(d.get('training',{}).get('optimal_bs', 32))")
    else
        log "FATAL: Profiler failed. Cannot determine optimal batch size."
        exit 1
    fi
fi

BATCH_FLAG="--batch-size $OPTIMAL_BS"
log "Auto batch size: ${OPTIMAL_BS}"

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

# Read winner lambda from ablation (single source: ablation/ subdir)
WINNER_JSON=""
for _dir in "${OUT_DIR}/ablation" "outputs/sanity/ablation"; do
    if [[ -f "${_dir}/ablation_winner.json" ]]; then
        WINNER_JSON="${_dir}/ablation_winner.json"
        break
    fi
done

if [[ -n "$WINNER_JSON" ]]; then
    WINNER_LAMBDA=$(python -u src/utils/config.py get-json "$WINNER_JSON" winner_lambda)
    WINNER_DIR=$(python -u src/utils/config.py get-json "$WINNER_JSON" winner_dir)
    log "Winner from ablation: lambda=${WINNER_LAMBDA} (${WINNER_DIR}) [from ${WINNER_JSON}]"
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
else
    # No winner — m09 auto-ablates all lambdas, selects winner, then trains
    log "No ablation_winner.json found. m09 will auto-ablate + select winner + train."

    run_step "train" "m09 pretrain (auto-ablation + winner)" "$T_M09" \
        "$LOGDIR/m09_${MODE,,}_auto_ablation.log" \
        src/m09_pretrain.py --config "$CONFIG" \
            --max-epochs "$WINNER_EPOCHS" \
            $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG \
        || { log "FATAL: m09 training failed."; exit 1; }

    # Read winner created by m09's auto-ablation
    WINNER_JSON="${OUT_DIR}/ablation_winner.json"
    if [[ ! -f "$WINNER_JSON" ]]; then
        log "FATAL: ablation_winner.json not found after m09 auto-ablation at ${WINNER_JSON}"
        exit 1
    fi
    WINNER_LAMBDA=$(python -u src/utils/config.py get-json "$WINNER_JSON" winner_lambda)
    WINNER_DIR=$(python -u src/utils/config.py get-json "$WINNER_JSON" winner_dir)
    WINNER_OUT="${OUT_DIR}/m09_${WINNER_DIR}"
    log "Auto-ablation winner: lambda=${WINNER_LAMBDA} (${WINNER_DIR})"
fi

verify "Training" "
import sys; sys.path.insert(0, 'src')
from utils.output_guard import verify_training_artifacts
verify_training_artifacts('${WINNER_OUT}')
"

# ── Embedding + Evaluation: handled by run_embed.sh + run_eval.sh
log "Training complete. Next steps:"
log "  ./scripts/run_embed.sh --FULL --local-data data/full_local"
log "  ./scripts/run_eval.sh --FULL"

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== FINAL VERIFICATION: Ch10 Ablation Outputs ==="

python -u src/utils/output_guard.py verify_pretrain_final "$OUT_DIR" "$CONFIG" 2>&1 | tee -a "$MASTER_LOG"

finalize "Ch10"
