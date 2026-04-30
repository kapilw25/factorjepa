#!/usr/bin/env bash
# iter13 m06d orchestrator — 3 modules × {V-JEPA, DINOv2} × stages.
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# and "DELETE PROTECTION — shells stay THIN, .py owns the cache-policy prompt".
#
# Pipeline (priority 1 — frozen V-JEPA vs frozen DINOv2 on Indian action probe
# applied to data/eval_10k_local):
#   STAGE 1   m06d_action_probe.py --stage labels        (CPU, ~1 min)
#   STAGE 2   m06d_action_probe.py --stage features      (GPU × 2 encoders, ~1 h)
#   STAGE 3   m06d_action_probe.py --stage train         (GPU × 2 encoders, ~30 min)
#   STAGE 4   m06d_action_probe.py --stage paired_delta  (CPU, ~5 min)  ← P1 GATE
#   STAGE 5   m06d_motion_cos.py   --stage features      (CPU mean-pool × 2 enc)
#   STAGE 6   m06d_motion_cos.py   --stage cosine        (CPU × 2 enc)
#   STAGE 7   m06d_motion_cos.py   --stage paired_delta  (CPU)
#   STAGE 8   m06d_future_mse.py   --stage forward       (GPU, V-JEPA only, ~30 min)
#   STAGE 9   m06d_future_mse.py   --stage paired_per_variant   (CPU)
#
# REFERENCES
#   plan_code_dev.md  — per-module specs + LoC budget
#   runbook.md        — full set of pre/post-flight one-liners
#
# USAGE
#   tmux new -s m06d
#   ./scripts/run_m06d.sh                     # full pipeline, prompts for cache-policy
#   CACHE_POLICY_ALL=1 ./scripts/run_m06d.sh  # bypass prompts, keep all caches
#   CACHE_POLICY_ALL=2 ./scripts/run_m06d.sh  # bypass prompts, recompute everything
#
#   # Skip a stage (resume after failure):
#   SKIP_STAGES="1,2" ./scripts/run_m06d.sh
#   # Run only one variant (debug):
#   ENCODERS="vjepa_2_1_frozen" ./scripts/run_m06d.sh

# NO -e: a single stage failure must NOT abort the chain (overnight-chain rule).
set -uo pipefail

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

# ── Configurables (env-overridable) ─────────────────────────────────────
EVAL_SUBSET="${EVAL_SUBSET:-data/eval_10k.json}"
LOCAL_DATA="${LOCAL_DATA:-data/eval_10k_local}"
TAGS_JSON="${TAGS_JSON:-${LOCAL_DATA}/tags.json}"
ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/vjepa2_1_vitG_384.pt}"
OUTPUT_ACTION="${OUTPUT_ACTION:-outputs/full/m06d_action_probe}"
OUTPUT_COS="${OUTPUT_COS:-outputs/full/m06d_motion_cos}"
OUTPUT_MSE="${OUTPUT_MSE:-outputs/full/m06d_future_mse}"
ENCODERS="${ENCODERS:-vjepa_2_1_frozen dinov2}"
SKIP_STAGES="${SKIP_STAGES:-}"
NUM_FRAMES="${NUM_FRAMES:-16}"

T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }
should_skip() {
    local stage="$1"
    [[ ",${SKIP_STAGES}," == *",${stage},"* ]]
}

# ── Pre-flight ──────────────────────────────────────────────────────────
stamp "PRE-FLIGHT"

for f in "$EVAL_SUBSET" "$ENCODER_CKPT"; do
    if [ ! -e "$f" ]; then
        echo "FATAL: required path missing: $f"
        exit 3
    fi
done
if [ ! -d "$LOCAL_DATA" ]; then
    echo "FATAL: --local-data dir missing: $LOCAL_DATA"
    exit 3
fi
if [ ! -f "$TAGS_JSON" ]; then
    echo "  WARN: $TAGS_JSON not found — --enable-monument-class will not work (3-class default OK)"
fi
echo "  ✓ eval_subset:     $EVAL_SUBSET"
echo "  ✓ local_data:      $LOCAL_DATA"
echo "  ✓ encoder_ckpt:    $ENCODER_CKPT  ($(du -h "$ENCODER_CKPT" 2>/dev/null | awk '{print $1}'))"
echo "  ✓ encoders:        $ENCODERS"
echo "  ✓ skip_stages:     ${SKIP_STAGES:-<none>}"
echo "  ✓ cache_policy:    ${CACHE_POLICY_ALL:-prompt-per-call}"

# ── STAGE 1 — labels ────────────────────────────────────────────────────
if ! should_skip 1; then
    stamp "STAGE 1 · action_probe labels (CPU, ~1 min)"
    python -u src/m06d_action_probe.py --FULL \
        --stage labels \
        --eval-subset "$EVAL_SUBSET" \
        --tags-json "$TAGS_JSON" \
        --output-root "$OUTPUT_ACTION" \
        2>&1 | tee logs/m06d_action_probe_labels.log
fi

# ── STAGE 2 — features (action_probe) per encoder ──────────────────────
if ! should_skip 2; then
    stamp "STAGE 2 · action_probe features (GPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        EXTRA_CKPT=""
        if [[ "$ENC" == vjepa* ]]; then
            EXTRA_CKPT="--encoder-ckpt $ENCODER_CKPT"
        fi
        python -u src/m06d_action_probe.py --FULL \
            --stage features \
            --encoder "$ENC" \
            $EXTRA_CKPT \
            --eval-subset "$EVAL_SUBSET" \
            --local-data "$LOCAL_DATA" \
            --output-root "$OUTPUT_ACTION" \
            --num-frames "$NUM_FRAMES" \
            2>&1 | tee "logs/m06d_action_probe_features_${ENC}.log"
    done
fi

# ── STAGE 3 — train (action_probe) per encoder ─────────────────────────
if ! should_skip 3; then
    stamp "STAGE 3 · action_probe train (GPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/m06d_action_probe.py --FULL \
            --stage train \
            --encoder "$ENC" \
            --output-root "$OUTPUT_ACTION" \
            2>&1 | tee "logs/m06d_action_probe_train_${ENC}.log"
    done
fi

# ── STAGE 4 — paired Δ (action_probe) ←── 🔥 PRIORITY 1 GATE ──────────
if ! should_skip 4; then
    stamp "STAGE 4 · action_probe paired_delta (🔥 P1 GATE — CPU, ~5 min, BCa 10K)"
    python -u src/m06d_action_probe.py --FULL \
        --stage paired_delta \
        --output-root "$OUTPUT_ACTION" \
        2>&1 | tee logs/m06d_action_probe_paired_delta.log
fi

# ── STAGE 5 — features (motion_cos) per encoder ────────────────────────
if ! should_skip 5; then
    stamp "STAGE 5 · motion_cos features (CPU mean-pool from action_probe × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/m06d_motion_cos.py --FULL \
            --stage features \
            --encoder "$ENC" \
            --action-probe-root "$OUTPUT_ACTION" \
            --output-root "$OUTPUT_COS" \
            --share-features \
            2>&1 | tee "logs/m06d_motion_cos_features_${ENC}.log"
    done
fi

# ── STAGE 6 — cosine (motion_cos) per encoder ──────────────────────────
if ! should_skip 6; then
    stamp "STAGE 6 · motion_cos cosine (CPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/m06d_motion_cos.py --FULL \
            --stage cosine \
            --encoder "$ENC" \
            --action-probe-root "$OUTPUT_ACTION" \
            --output-root "$OUTPUT_COS" \
            2>&1 | tee "logs/m06d_motion_cos_cosine_${ENC}.log"
    done
fi

# ── STAGE 7 — paired Δ (motion_cos) ────────────────────────────────────
if ! should_skip 7; then
    stamp "STAGE 7 · motion_cos paired_delta (CPU)"
    python -u src/m06d_motion_cos.py --FULL \
        --stage paired_delta \
        --output-root "$OUTPUT_COS" \
        2>&1 | tee logs/m06d_motion_cos_paired_delta.log
fi

# ── STAGE 8 — forward (future_mse) — V-JEPA frozen only ────────────────
if ! should_skip 8; then
    stamp "STAGE 8 · future_mse forward (GPU, V-JEPA frozen only, ~30 min)"
    python -u src/m06d_future_mse.py --FULL \
        --stage forward \
        --variant vjepa_2_1_frozen \
        --encoder-ckpt "$ENCODER_CKPT" \
        --action-probe-root "$OUTPUT_ACTION" \
        --local-data "$LOCAL_DATA" \
        --output-root "$OUTPUT_MSE" \
        --num-frames "$NUM_FRAMES" \
        2>&1 | tee logs/m06d_future_mse_forward_vjepa.log
fi

# ── STAGE 9 — paired Δ (future_mse, per-variant) ───────────────────────
if ! should_skip 9; then
    stamp "STAGE 9 · future_mse paired_per_variant (CPU)"
    python -u src/m06d_future_mse.py --FULL \
        --stage paired_per_variant \
        --output-root "$OUTPUT_MSE" \
        2>&1 | tee logs/m06d_future_mse_paired.log
fi

# ── Final summary ──────────────────────────────────────────────────────
stamp "DONE · total wall = $(( ($(date +%s) - T0) / 60 )) min"
echo "Artifacts:"
echo "  🔥 P1 GATE     $OUTPUT_ACTION/m06d_paired_delta.json"
echo "  motion_cos     $OUTPUT_COS/m06d_motion_cos_paired.json"
echo "  future_mse     $OUTPUT_MSE/m06d_future_mse_per_variant.json"
echo "Per-encoder probe ckpts:"
for ENC in $ENCODERS; do
    echo "  $OUTPUT_ACTION/$ENC/probe.pt"
done
