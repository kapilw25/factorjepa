#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ExPLoRA: LoRA + unfreeze 1-2 blocks + JEPA self-supervised pretraining
# on V-JEPA 2.1 (2B, 1664-dim). Step 1b baseline adaptation.
#
# Pipeline: m09(ExPLoRA training) → m05(re-embed) → m06(metrics)
#
# USAGE:
#   ./scripts/train_explora.sh --SANITY 2>&1 | tee logs/explora_sanity.log
#   ./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_poc.log
#   ./scripts/train_explora.sh --FULL 2>&1 | tee logs/explora_full.log
#
# PREREQUISITES:
#   1. ./setup_env_uv.sh --gpu --from-wheels
#   2. checkpoints/vjepa2_1_vitG_384.pt (~8 GB, download from Meta)
#   3. data/subset_10k_local/ OR data/full_local/ (from m00d or rsync)
#   4. data/val_1k_local/ (for validation)
#   5. tmux new -s explora
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL"
    echo ""
    echo "  --SANITY   Quick validation (~5 min, 64 clips)"
    echo "  --POC      10K subset, 5 epochs (~3h GPU)"
    echo "  --FULL     115K full corpus, 5 epochs (~15h GPU)"
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
MASTER_LOG="$LOGDIR/explora_${MODE}_$(date +%Y%m%d_%H%M%S).log"

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
TRAIN_CONFIG="configs/train/explora.yaml"

# Checkpoint check
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
    # POC: use val_1k_local (1K clips for fast signal)
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
            log "  Download: python -u src/m00d_download_subset.py --FULL --no-wandb"
            exit 1
        fi
        LOCAL_FLAG="--local-data data/full_local"
    fi
    # Val data (FATAL if missing)
    if [[ ! -f "data/val_1k.json" ]]; then
        log "FATAL: data/val_1k.json not found"
        exit 1
    fi
    if [[ ! -d "data/val_1k_local" ]]; then
        log "FATAL: data/val_1k_local/ not found"
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
python -u src/utils/output_guard.py preflight_gpu_packages "explora" "$TRAIN_CONFIG" "$OUT_DIR" \
    2>&1 | tee -a "$MASTER_LOG" || { log "FATAL: GPU pre-flight failed"; exit 1; }

# ── Time estimates ────────────────────────────────────────────────────
if [[ "$MODE" == "SANITY" ]]; then
    T_TRAIN="~3 min"; T_EMBED="~30 sec"; T_EVAL="~10 sec"
elif [[ "$MODE" == "POC" ]]; then
    T_TRAIN="~3h (5 epochs, 10K clips)"; T_EMBED="~2h"; T_EVAL="~1 min"
else
    T_TRAIN="~15h (5 epochs, 115K clips)"; T_EMBED="~20h"; T_EVAL="~5 min"
fi

# ── Print banner ──────────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════"
log " ExPLoRA Training — V-JEPA 2.1 (2B) on WalkIndia-200K"
log " Mode: $MODE | Model: $MODEL_CONFIG | Train: $TRAIN_CONFIG"
log "═══════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════
# Step 0: Generate frozen V-JEPA 2.1 baseline embeddings (one-time)
# These are needed to compare: frozen 2.1 vs ExPLoRA 2.1 (same model, fair comparison)
# ═══════════════════════════════════════════════════════════════════════
FROZEN_EMB="${OUT_DIR}/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy"
if [[ -f "$FROZEN_EMB" ]]; then
    log "Frozen 2.1 embeddings exist: $FROZEN_EMB (skipping)"
else
    run_step "0-frozen-embed" "m05 frozen V-JEPA 2.1 embeddings" \
        "$T_EMBED" "$LOGDIR/m05_vjepa_2_1_frozen_${MODE,,}.log" \
        src/m05_vjepa_embed.py \
            --model-config "$MODEL_CONFIG" \
            --encoder vjepa_2_1_frozen \
            $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

    run_step "0-frozen-eval" "m06 frozen V-JEPA 2.1 metrics" \
        "$T_EVAL" "$LOGDIR/m06_vjepa_2_1_frozen_${MODE,,}.log" \
        src/m06_faiss_metrics.py \
            --encoder vjepa_2_1_frozen \
            $MODE_FLAG $SUBSET_FLAG --no-wandb
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 1: ExPLoRA training
# ═══════════════════════════════════════════════════════════════════════
EXPLORA_DIR="${OUT_DIR}/m09_pretrain/explora"

run_step "1-train" "m09 ExPLoRA (LoRA rank=16, unfreeze=2 blocks)" \
    "$T_TRAIN" "$LOGDIR/m09_explora_${MODE,,}.log" \
    src/m09_pretrain.py \
        --model-config "$MODEL_CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --explora \
        --output-dir "$EXPLORA_DIR" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG --no-wandb

# Verify student_encoder.pt exists
STUDENT_PT="${EXPLORA_DIR}/student_encoder.pt"
if [[ ! -f "$STUDENT_PT" ]]; then
    log "FATAL: student_encoder.pt not found at $STUDENT_PT"
    exit 1
fi
log "ExPLoRA training complete: $STUDENT_PT"

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Re-embed with adapted model
# ═══════════════════════════════════════════════════════════════════════
run_step "2-embed" "m05 re-embed ExPLoRA adapted" \
    "$T_EMBED" "$LOGDIR/m05_explora_${MODE,,}.log" \
    src/m05_vjepa_embed.py \
        --model-config "$MODEL_CONFIG" \
        --model "$STUDENT_PT" \
        --encoder vjepa_2_1_explora \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Evaluate
# ═══════════════════════════════════════════════════════════════════════
run_step "3-eval" "m06 metrics (ExPLoRA vs frozen)" \
    "$T_EVAL" "$LOGDIR/m06_explora_${MODE,,}.log" \
    src/m06_faiss_metrics.py \
        --encoder vjepa_2_1_explora \
        $MODE_FLAG $SUBSET_FLAG --no-wandb

# ═══════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════
finalize "ExPLoRA"
