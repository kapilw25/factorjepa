#!/bin/bash
# Run the video held-out test (scripts/heldout_video_eval.sh) for several 1B arms AT ONCE on one GPU.
# Why (measured Sat Nov 28 2026 on the RTX PRO 6000, one arm at a time): Mask-ratio slope is ~18 of
# each arm's ~35 min at 98-100% SM busy on 6.4 GB with batch 4, and pipeline.yaml records that batch
# 4 -> 32 gave no gain (per-clip sequential predictor rollout). The GPU is "busy" with small kernels,
# so the room left is concurrency: CUDA MPS lets kernels from the arms' processes run side by side
# instead of time-slicing contexts (https://docs.nvidia.com/deploy/mps/). The decoded-frame cache is
# pre-warmed once with every core (src/utils/eval_frame_cache.py), so the arms read frames, not mp4.
# Outputs are the same files heldout_video_eval.sh writes; each process keeps FJ_GPU_MEM_GB=<cap>.
# USAGE (on the box, in tmux; cap x n_arms must stay under the card's memory):
#   bash scripts/heldout_video_parallel.sh <gpu_cap_gb_per_arm> <hf_repo> <hf_train_root> <local_data> \
#       <probe_root> <out_root> <lease_dir> <model_cfg> "<variant>|<hf_arm_dir>|<ckpt_best_name>" [...] \
#       2>&1 | tee logs/heldout_video_parallel_$(date +%Y%m%d_%H%M%S).log
set -o pipefail
if [ $# -lt 9 ]; then
    echo "FATAL: usage: $0 <gpu_cap_gb_per_arm> <hf_repo> <hf_train_root> <local_data> <probe_root> <out_root> <lease_dir> <model_cfg> <arm spec>..."
    exit 2
fi
CAP=$1; REPO=$2; TRAIN_ROOT=$3; LOCAL=$4; PROBE=$5; OUT=$6; LEASE=$7; MCFG=$8; shift 8
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUT"

# Frame cache location: same yaml keys as heldout_video_eval.sh (which re-reads them per arm).
FRAME_CACHE_SUBDIR="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.eval_frame_cache.subdir)" || exit 2
export EVAL_FRAME_CACHE_DIR="${EVAL_FRAME_CACHE_DIR:-${LOCAL}/${FRAME_CACHE_SUBDIR}}"
MIN_FREE="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.eval_frame_cache.min_free_gb)" || exit 2
NUM_FRAMES="$(python scripts/lib/yaml_extract.py configs/pipeline.yaml probe.num_frames)" || exit 2

# GPU utilisation log for the whole run (sm %, memory %, every 5 s).
nvidia-smi dmon -s um -d 5 -o T > "logs/gpu_dmon_heldout_parallel_${TS}.log" &
DMON=$!

echo "== $(date '+%F %T') pre-warm frame cache -> $EVAL_FRAME_CACHE_DIR ($(nproc) workers)"
PYTHONPATH=src python -u src/utils/eval_frame_cache.py --keys "$PROBE/action_labels.json" --local-data "$LOCAL" \
    --cache-dir "$EVAL_FRAME_CACHE_DIR" --num-frames "$NUM_FRAMES" --min-free-gb "$MIN_FREE" --workers "$(nproc)" \
    || { kill "$DMON"; exit 3; }

export CUDA_MPS_PIPE_DIRECTORY="$OUT/_mps/pipe" CUDA_MPS_LOG_DIRECTORY="$OUT/_mps/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || { kill "$DMON"; echo "FATAL: MPS daemon did not start"; exit 4; }
echo "== $(date '+%F %T') MPS daemon up ($CUDA_MPS_PIPE_DIRECTORY); launching $# arms, ${CAP} GB cap each"

PIDS=(); NAMES=()
for SPEC in "$@"; do
    IFS='|' read -r V _ _ <<< "$SPEC"
    bash scripts/heldout_video_eval.sh "$CAP" "$REPO" "$TRAIN_ROOT" "$LOCAL" "$PROBE" "$OUT" "$LEASE" "$MCFG" "$SPEC" \
        > "logs/heldout_video_eval_${V}_${TS}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$V")
done
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "== $(date '+%F %T') arm ${NAMES[$i]} OK"
    else
        echo "== $(date '+%F %T') arm ${NAMES[$i]} FAILED (logs/heldout_video_eval_${NAMES[$i]}_${TS}.log)"
        FAIL=1
    fi
done
echo quit | nvidia-cuda-mps-control
kill "$DMON"
echo "== $(date '+%F %T') all arms finished (fail=$FAIL); GPU log logs/gpu_dmon_heldout_parallel_${TS}.log"
exit "$FAIL"
