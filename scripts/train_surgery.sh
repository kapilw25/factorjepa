#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch11: Factor Surgery — SAM3 segmentation + progressive prefix unfreezing
# on V-JEPA 2.1 (2B, 1664-dim). THE paper novelty experiment.
#
# Pipeline: m10(SAM3) → m11(factor datasets) → m09(surgery training) → m05(re-embed) → m06(eval)
#
# USAGE:
#   ./scripts/train_surgery.sh --SANITY 2>&1 | tee logs/surgery_sanity.log
#   ./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_poc.log
#   ./scripts/train_surgery.sh --FULL 2>&1 | tee logs/surgery_full.log
#
# POC MODE (Week 1):
#   - 100 clips, 2 factors only (D_L + D_A, skip D_I)
#   - 2 stages: Stage 1 (layout), Stage 2 (agents + 10% layout replay)
#   - ~3h GPU total (30 min SAM3 + 2.5h training)
#
# PREREQUISITES:
#   1. ./setup_env_uv.sh --gpu --from-wheels
#   2. checkpoints/vjepa2_1_vitG_384.pt (~8 GB)
#   3. data/subset_10k_local/ OR data/full_local/
#   4. data/val_1k_local/ (for validation)
#   5. tmux new -s surgery
#
# NOTE: m10_sam_segment.py and m11_factor_datasets.py are NOT YET BUILT.
# This script provides the orchestration skeleton. Build Python scripts first.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL"
    echo ""
    echo "  --SANITY   Quick validation (~5 min, 20 clips)"
    echo "  --POC      100 clips, 2 factors (D_L + D_A), 2 stages (~3h GPU)"
    echo "  --FULL     10K clips, 3 factors (D_L + D_A + D_I), 3 stages (~24h GPU)"
    exit 1
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --SANITY)
            MODE="SANITY"; MODE_FLAG="--SANITY"; SUBSET_FLAG=""
            OUT_DIR="outputs/sanity"; shift ;;
        --POC)
            MODE="POC"; MODE_FLAG="--FULL"
            SUBSET_FLAG="--subset data/subset_10k.json"
            OUT_DIR="outputs/poc"; shift ;;
        --FULL)
            MODE="FULL"; MODE_FLAG="--FULL"; SUBSET_FLAG=""
            OUT_DIR="outputs/full"; shift ;;
        *) usage ;;
    esac
done

[[ -z "$MODE" ]] && usage

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

LOGDIR="logs"
mkdir -p "$LOGDIR" "$OUT_DIR"
MASTER_LOG="$LOGDIR/surgery_${MODE}_$(date +%Y%m%d_%H%M%S).log"

if [[ -d "venv_walkindia" ]]; then
    source venv_walkindia/bin/activate
else
    echo "FATAL: venv_walkindia not found. Run: ./setup_env_uv.sh --gpu"
    exit 1
fi

source "$(dirname "$0")/lib/common.sh"
start_watchdog

# Cleanup on interrupt (stop watchdog, preserve checkpoint)
cleanup_on_exit() {
    log "INTERRUPTED — preserving checkpoint for resume. Re-run same command."
    stop_watchdog
    exit 130
}
trap cleanup_on_exit INT TERM

# ── Config ────────────────────────────────────────────────────────────
MODEL_CONFIG="configs/model/vjepa2_1.yaml"
TRAIN_CONFIG="configs/train/ch11_surgery.yaml"

CKPT="checkpoints/vjepa2_1_vitG_384.pt"
if [[ ! -f "$CKPT" ]]; then
    log "FATAL: V-JEPA 2.1 checkpoint not found: $CKPT"
    log "Download: wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -P checkpoints/"
    exit 1
fi

# ── Local data (FATAL if missing for non-SANITY) ─────────────────────
LOCAL_FLAG=""
VAL_FLAG=""
if [[ "$MODE" == "SANITY" ]]; then
    LOCAL_FLAG=""
else
    if [[ -n "$SUBSET_FLAG" ]]; then
        if [[ ! -d "data/val_1k_local" ]] || [[ ! -f "data/val_1k_local/manifest.json" ]]; then
            log "FATAL: data/val_1k_local/ missing or no manifest.json"
            log "  Download: python -u src/utils/hf_outputs.py download-data"
            exit 1
        fi
        LOCAL_FLAG="--local-data data/val_1k_local"
    else
        if [[ ! -d "data/full_local" ]] || [[ ! -f "data/full_local/manifest.json" ]]; then
            log "FATAL: data/full_local/ missing or no manifest.json"
            exit 1
        fi
        LOCAL_FLAG="--local-data data/full_local"
    fi
    if [[ ! -f "data/val_1k.json" ]] || [[ ! -d "data/val_1k_local" ]]; then
        log "FATAL: val data missing (data/val_1k.json + data/val_1k_local/)"
        exit 1
    fi
    VAL_FLAG="--val-subset data/val_1k.json --val-local-data data/val_1k_local"
fi

# ── Auto batch size detection ─────────────────────────────────────────
BATCH_FLAG=""
PROFILE_JSON="outputs/profile/training/profile_data.json"
if [[ -f "$PROFILE_JSON" ]]; then
    BS=$(python -u src/utils/gpu_batch.py optimal-bs --profile-json "$PROFILE_JSON" 2>/dev/null || echo "")
    if [[ -n "$BS" ]]; then
        BATCH_FLAG="--batch-size $BS"
        log "Batch size: $BS (from profiler)"
    fi
fi
if [[ -z "$BATCH_FLAG" ]]; then
    BS=$(python -u src/utils/config.py get-yaml "$TRAIN_CONFIG" optimization.batch_size 2>/dev/null || echo "32")
    BATCH_FLAG="--batch-size $BS"
    log "Batch size: $BS (from YAML / default)"
fi

# ── GPU pre-flight ────────────────────────────────────────────────────
log "Pre-flight: checking GPU packages..."
python -u src/utils/output_guard.py preflight_gpu_packages "surgery" "$TRAIN_CONFIG" "$OUT_DIR" \
    2>&1 | tee -a "$MASTER_LOG" || { log "FATAL: GPU pre-flight failed"; exit 1; }

# ── Time estimates ────────────────────────────────────────────────────
if [[ "$MODE" == "SANITY" ]]; then
    T_SAM3="~1 min"; T_FACTORS="~30 sec"; T_SURGERY="~3 min"
    T_EMBED="~30 sec"; T_EVAL="~10 sec"
elif [[ "$MODE" == "POC" ]]; then
    T_SAM3="~30 min (100 clips)"; T_FACTORS="~5 min"; T_SURGERY="~2.5h (2 stages)"
    T_EMBED="~2h"; T_EVAL="~1 min"
else
    T_SAM3="~5h (10K clips)"; T_FACTORS="~30 min"; T_SURGERY="~15h (3 stages)"
    T_EMBED="~20h"; T_EVAL="~5 min"
fi

log ""
log "═══════════════════════════════════════════════════════"
log " Ch11 Factor Surgery — V-JEPA 2.1 (2B) on WalkIndia-200K"
log " Mode: $MODE | Model: $MODEL_CONFIG | Train: $TRAIN_CONFIG"
log "═══════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# Step 0 (GPU): SAM3 segmentation → instance masks → tracklets
# NOTE: m10_sam_segment.py NOT YET BUILT
# ═══════════════════════════════════════════════════════════════════════
FACTOR_DIR="${OUT_DIR}/m10_sam_segment"
mkdir -p "$FACTOR_DIR"

run_step "0-sam3" "m10 SAM3 segmentation → tracklets" \
    "$T_SAM3" "$LOGDIR/m10_sam3_${MODE,,}.log" \
    src/m10_sam_segment.py \
        --output-dir "$FACTOR_DIR" \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

# ═══════════════════════════════════════════════════════════════════════
# Step 1 (CPU): Generate factor datasets (D_L, D_A, optionally D_I)
# NOTE: m11_factor_datasets.py NOT YET BUILT
# ═══════════════════════════════════════════════════════════════════════
run_step "1-factors" "m11 factor datasets (D_L, D_A, D_I)" \
    "$T_FACTORS" "$LOGDIR/m11_factors_${MODE,,}.log" \
    src/m11_factor_datasets.py \
        --input-dir "$FACTOR_DIR" \
        --output-dir "$FACTOR_DIR" \
        $MODE_FLAG $SUBSET_FLAG --no-wandb

# ═══════════════════════════════════════════════════════════════════════
# Step 2 (GPU): Progressive prefix unfreezing on frozen V-JEPA 2.1
# ═══════════════════════════════════════════════════════════════════════
SURGERY_DIR="${OUT_DIR}/m09_pretrain/surgery"

run_step "2-surgery" "m09 factor surgery (progressive unfreezing)" \
    "$T_SURGERY" "$LOGDIR/m09_surgery_${MODE,,}.log" \
    src/m09_pretrain.py \
        --model-config "$MODEL_CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --surgery --factor-dir "$FACTOR_DIR" \
        --output-dir "$SURGERY_DIR" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG --no-wandb

STUDENT_PT="${SURGERY_DIR}/student_encoder.pt"
if [[ ! -f "$STUDENT_PT" ]]; then
    log "FATAL: student_encoder.pt not found at $STUDENT_PT"
    exit 1
fi
log "Surgery training complete: $STUDENT_PT"

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Re-embed with surgical model
# ═══════════════════════════════════════════════════════════════════════
run_step "3-embed" "m05 re-embed surgical adapted" \
    "$T_EMBED" "$LOGDIR/m05_surgery_${MODE,,}.log" \
    src/m05_vjepa_embed.py \
        --model-config "$MODEL_CONFIG" \
        --model "$STUDENT_PT" \
        --encoder vjepa_2_1_surgical \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Evaluate
# ═══════════════════════════════════════════════════════════════════════
run_step "4-eval" "m06 metrics (surgery vs frozen vs ExPLoRA)" \
    "$T_EVAL" "$LOGDIR/m06_surgery_${MODE,,}.log" \
    src/m06_faiss_metrics.py \
        --encoder vjepa_2_1_surgical \
        $MODE_FLAG $SUBSET_FLAG --no-wandb

finalize "Surgery"
