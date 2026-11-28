#!/bin/bash
# Video held-out test (FactorJEPA workshop drafts, owner TODO, Fri Nov 27 2026): Future-frame L1 (m12d), Forward L1
# and Mask-ratio slope (m12e), Motion cosine (m12b) for saved 1B arms on a video-disjoint clip set
# (src/utils/video_heldout_split.py, cut by src/utils/hf_shard_subset.py, labels by m04e --probe-split test-all).
# Same per-script settings as scripts/run_eval.sh (--POC, 16 frames, motion_aux_head passed when present,
# probe.predictor_temporal.batch_size). Paired intervals are computed afterwards (src/utils/paired_ladder.py).
# Shared-box rules: each arm's checkpoints are leased from HF one at a time (src/utils/hf_ckpt_lease.py, deletes via
# guarded_delete), and every eval process runs with FJ_GPU_MEM_GB=<cap>, which the box venv's sitecustomize turns into
# torch.cuda.set_per_process_memory_fraction in the main process only.
# USAGE (on the box, in tmux):
#   bash scripts/heldout_video_eval.sh <gpu_cap_gb> <hf_repo> <hf_train_root> <local_data> <probe_root> <out_root> \
#       <lease_dir> <model_cfg> "<variant>|<hf_arm_dir>|<ckpt_best_name>" [...] \
#       2>&1 | tee logs/heldout_video_eval_$(date +%Y%m%d_%H%M%S).log
set -o pipefail
if [ $# -lt 9 ]; then
    echo "FATAL: usage: $0 <gpu_cap_gb> <hf_repo> <hf_train_root> <local_data> <probe_root> <out_root> <lease_dir> <model_cfg> <arm spec>..."
    exit 2
fi
CAP=$1; REPO=$2; TRAIN_ROOT=$3; LOCAL=$4; PROBE=$5; OUT=$6; LEASE=$7; MCFG=$8; shift 8
PT_BATCH="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.predictor_temporal.batch_size)" || exit 2
# Shared decoded-frame cache, same switch and keys as scripts/run_eval.sh: each clip is decoded once
# and every later stage and arm reads the cached frames (decode is deterministic, so outputs are
# unchanged). EVAL_FRAME_CACHE_DIR may be pre-set (e.g. a RAM disk when the box disk is tight).
FRAME_CACHE_ENABLED="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.eval_frame_cache.enabled)" || exit 2
case "$FRAME_CACHE_ENABLED" in
    false|False|0|"")
        echo "  eval frame cache: disabled (probe.eval_frame_cache.enabled=$FRAME_CACHE_ENABLED)" ;;
    *)
        FRAME_CACHE_SUBDIR="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.eval_frame_cache.subdir)" || exit 2
        export EVAL_FRAME_CACHE_DIR="${EVAL_FRAME_CACHE_DIR:-${LOCAL}/${FRAME_CACHE_SUBDIR}}"
        EVAL_FRAME_CACHE_MIN_FREE_GB="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.eval_frame_cache.min_free_gb)" || exit 2
        EVAL_FRAME_CACHE_MAX_GB="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.eval_frame_cache.max_cache_gb)" || exit 2
        export EVAL_FRAME_CACHE_MIN_FREE_GB EVAL_FRAME_CACHE_MAX_GB
        mkdir -p "$EVAL_FRAME_CACHE_DIR"
        echo "  eval frame cache: $EVAL_FRAME_CACHE_DIR (max ${EVAL_FRAME_CACHE_MAX_GB} GB, LRU-evict)" ;;
esac
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUT"
for SPEC in "$@"; do
    IFS='|' read -r V DIR CKPT <<< "$SPEC"
    echo "== $(date '+%F %T') arm $V ($DIR/$CKPT)"
    for F in "$CKPT" motion_aux_head.pt; do
        python -u src/utils/hf_ckpt_lease.py fetch --repo "$REPO" --path-in-repo "$TRAIN_ROOT/$DIR/$F" --dest-dir "$LEASE" || exit 3
    done
    BEST="$LEASE/$TRAIN_ROOT/$DIR/$CKPT"; HEAD="$LEASE/$TRAIN_ROOT/$DIR/motion_aux_head.pt"
    FJ_GPU_MEM_GB=$CAP python -u src/m12d_future_mse.py --POC --stage forward --variant "$V" --encoder-ckpt "$BEST" \
        --model-config "$MCFG" --motion-aux-head "$HEAD" --action-probe-root "$PROBE" --local-data "$LOCAL" \
        --output-root "$OUT/probe_future_mse" --num-frames 16 --cache-policy 1 --no-wandb \
        2>&1 | tee "logs/heldout_m12d_${V}_${TS}.log" || exit 4
    for M in causal maskratio; do
        FJ_GPU_MEM_GB=$CAP python -u src/m12e_predictor_temporal.py --POC --stage forward --metric "$M" --variant "$V" \
            --encoder-ckpt "$BEST" --model-config "$MCFG" --motion-aux-head "$HEAD" --action-probe-root "$PROBE" \
            --local-data "$LOCAL" --output-root "$OUT/predictor_temporal" --num-frames 16 --batch-size "$PT_BATCH" \
            --cache-policy 1 --no-wandb 2>&1 | tee "logs/heldout_m12e_${M}_${V}_${TS}.log" || exit 4
    done
    python -u src/utils/hf_ckpt_lease.py release --dest-dir "$LEASE" --path-in-repo "$TRAIN_ROOT/$DIR/$CKPT" --cache-policy 2 || exit 5
    python -u src/utils/hf_ckpt_lease.py fetch --repo "$REPO" --path-in-repo "$TRAIN_ROOT/$DIR/student_encoder.pt" --dest-dir "$LEASE" || exit 3
    FJ_GPU_MEM_GB=$CAP python -u src/m12b_motion_cos.py --POC --stage features --encoder "$V" \
        --encoder-ckpt "$LEASE/$TRAIN_ROOT/$DIR/student_encoder.pt" --motion-aux-head "$HEAD" --action-probe-root "$PROBE" \
        --local-data "$LOCAL" --output-root "$OUT/probe_motion_cos" --num-frames 16 --no-share-features \
        --cache-policy 1 --no-wandb 2>&1 | tee "logs/heldout_m12b_features_${V}_${TS}.log" || exit 4
    python -u src/m12b_motion_cos.py --POC --stage cosine --encoder "$V" --action-probe-root "$PROBE" \
        --output-root "$OUT/probe_motion_cos" --cache-policy 1 --no-wandb \
        2>&1 | tee "logs/heldout_m12b_cosine_${V}_${TS}.log" || exit 4
    python -u src/utils/hf_ckpt_lease.py release --dest-dir "$LEASE" --path-in-repo "$TRAIN_ROOT/$DIR/student_encoder.pt" --cache-policy 2 || exit 5
    echo "== $(date '+%F %T') arm $V done"
done
echo "== $(date '+%F %T') all arms done"
