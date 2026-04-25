#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Standalone Embedding Extraction (reusable across Ch9, Ch10, Ch11)
# Runs: m05 (V-JEPA) + m05b (baselines: random, dinov2, clip, shuffled)
#
# Auto-detects adapted models from outputs/<mode>/m09_*/student_encoder.pt
#
# USAGE:
#   ./scripts/run_embed.sh --SANITY                                     # sanity
#   ./scripts/run_embed.sh --FULL --local-data data/full_local          # full 115K
#   ./scripts/run_embed.sh --FULL --subset data/subset_10k.json \
#       --local-data data/subset_10k_local                              # 10K fast signal
#   ./scripts/run_embed.sh --FULL --local-data data/full_local \
#       --encoders vjepa_lambda0_001                                    # specific encoder only
#
# PREREQUISITES:
#   - data/{full,subset_10k,val_1k}_local/ exist
#   - For adapted encoders: student_encoder.pt from train_pretrain.sh
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
MODE_FLAG=""
SUBSET_FLAG=""
LOCAL_DATA=""
ENCODERS=""

usage() {
    echo "Usage: $0 --SANITY | --FULL [--local-data DIR] [--subset JSON] [--encoders enc1,enc2]"
    echo ""
    echo "  --SANITY       Embed sanity clips"
    echo "  --FULL         Embed full corpus (or subset if --subset provided)"
    echo "  --local-data   Path to local TAR directory"
    echo "  --subset       Subset JSON (e.g., data/subset_10k.json) for fast signal"
    echo "  --encoders     Comma-separated list (default: frozen + all adapted)"
    echo ""
    echo "Examples:"
    echo "  $0 --FULL --local-data data/full_local                    # all encoders, 115K"
    echo "  $0 --FULL --subset data/subset_10k.json --local-data data/subset_10k_local  # 10K fast"
    echo "  $0 --FULL --local-data data/full_local --encoders vjepa_lambda0_001         # one encoder"
    exit 1
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --SANITY) MODE="SANITY"; MODE_FLAG="--SANITY"; OUT_DIR="outputs/sanity"; shift ;;
        --FULL)   MODE="FULL";   MODE_FLAG="--FULL";   OUT_DIR="outputs/full";   shift ;;
        --local-data) LOCAL_DATA="$2"; shift 2 ;;
        --subset) SUBSET_FLAG="--subset $2"; shift 2 ;;
        --encoders) ENCODERS="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$MODE" ]] && usage

LOCAL_FLAG=""
[[ -n "$LOCAL_DATA" ]] && LOCAL_FLAG="--local-data $LOCAL_DATA"

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ -d "venv_walkindia" ]]; then
    source venv_walkindia/bin/activate
else
    echo "ERROR: venv_walkindia not found. Run: ./setup_env_uv.sh --gpu"
    exit 1
fi

source "$(dirname "$0")/lib/common.sh"

LOGDIR="logs"
MASTER_LOG="$LOGDIR/embed_${MODE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGDIR" "$OUT_DIR"

start_watchdog

# ── Build encoder list ────────────────────────────────────────────────
if [[ -z "$ENCODERS" ]]; then
    # Auto-detect: frozen baselines + all adapted models
    ENCODER_LIST="vjepa,random,dinov2,clip,vjepa_shuffled"

    # Find adapted models from m09a/b/c outputs (split from monolith 2026-04-15, #49).
    # Back-compat: also glob old m09_pretrain/ paths for pre-split checkpoints.
    for model_dir in "$OUT_DIR"/m09a_pretrain/lambda*/ "$OUT_DIR"/m09a_pretrain/ablation/lambda*/ \
                     "$OUT_DIR"/m09b_explora/ "$OUT_DIR"/m09c_surgery/ \
                     "$OUT_DIR"/m09_pretrain/lambda*/ "$OUT_DIR"/m09_lambda*/; do
        if [[ -f "${model_dir}student_encoder.pt" ]]; then
            # Encoder name: strip output base dirs + any "m09*_" prefix, prepend vjepa_
            raw=$(basename "$model_dir")
            enc_name="vjepa_$(echo "$raw" | sed 's/^m09[abc]*_//')"
            ENCODER_LIST="${ENCODER_LIST},${enc_name}"
        fi
    done
    log "Auto-detected encoders: ${ENCODER_LIST}"
else
    ENCODER_LIST="$ENCODERS"
    log "Using specified encoders: ${ENCODER_LIST}"
fi

log "=== EMBEDDING PIPELINE (mode=${MODE}) ==="
log "Output dir: ${OUT_DIR}"
log "Encoders:   ${ENCODER_LIST}"
log "Local data: ${LOCAL_DATA:-HF streaming}"
log "Subset:     ${SUBSET_FLAG:-full corpus}"

# ── Embed each encoder ────────────────────────────────────────────────
IFS=',' read -ra ENC_ARRAY <<< "$ENCODER_LIST"

for enc in "${ENC_ARRAY[@]}"; do
    # Determine if frozen or adapted
    MODEL_FLAG=""
    ENCODER_FLAG="--encoder $enc"

    if [[ "$enc" == "vjepa" ]]; then
        # Frozen V-JEPA (HF model)
        run_step "embed-${enc}" "m05 V-JEPA frozen" "~varies" \
            "$LOGDIR/m05_${MODE,,}_${enc}.log" \
            src/m05_vjepa_embed.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
            || { log "WARNING: m05 ${enc} failed."; continue; }

    elif [[ "$enc" == "random" || "$enc" == "dinov2" || "$enc" == "clip" || "$enc" == "vjepa_shuffled" ]]; then
        # Baseline encoders
        run_step "embed-${enc}" "m05b ${enc}" "~varies" \
            "$LOGDIR/m05b_${MODE,,}_${enc}.log" \
            src/m05b_baselines.py --encoder "$enc" $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
            || { log "WARNING: m05b ${enc} failed."; continue; }

    elif [[ "$enc" == vjepa_lambda* ]]; then
        # Adapted model — find student_encoder.pt
        lambda_dir=$(echo "$enc" | sed 's/vjepa_/m09_/')
        model_path="${OUT_DIR}/${lambda_dir}/student_encoder.pt"
        if [[ ! -f "$model_path" ]]; then
            log "WARNING: ${model_path} not found. Skip ${enc}."
            continue
        fi

        # Check if shuffled variant
        if [[ "$enc" == *_shuffled ]]; then
            base_enc="${enc%_shuffled}"
            base_dir=$(echo "$base_enc" | sed 's/vjepa_/m09_/')
            model_path="${OUT_DIR}/${base_dir}/student_encoder.pt"
            run_step "embed-${enc}" "m05 ${enc} (shuffled adapted)" "~varies" \
                "$LOGDIR/m05_${MODE,,}_${enc}.log" \
                src/m05_vjepa_embed.py --model "$model_path" --encoder "$enc" --shuffle \
                    $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
                || { log "WARNING: m05 ${enc} failed."; continue; }
        else
            run_step "embed-${enc}" "m05 ${enc} (adapted)" "~varies" \
                "$LOGDIR/m05_${MODE,,}_${enc}.log" \
                src/m05_vjepa_embed.py --model "$model_path" --encoder "$enc" \
                    $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
                || { log "WARNING: m05 ${enc} failed."; continue; }
        fi
    else
        log "WARNING: Unknown encoder ${enc}. Skipping."
        continue
    fi

    bg_upload
done

# ── Summary ───────────────────────────────────────────────────────────
finalize "Embed"
