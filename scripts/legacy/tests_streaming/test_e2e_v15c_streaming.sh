#!/usr/bin/env bash
# iter10 D_I streaming — end-to-end SANITY integration test.
#
# Runs the full 3-stage pipeline (m10 cached → m11 --streaming → m09c 3-stage)
# on the SANITY subset (20 clips) with v15c config (interaction_mining.enabled=true)
# and asserts:
#   1. No D_I .npy tubes written to outputs/sanity/m11_factor_datasets/D_I/ for
#      non-verify clips (only 100-verify may have them; at N=20 no verify either).
#   2. m09c completes all 3 stages (doesn't crash when D_I tubes are streamed).
#   3. student_encoder.pt is exported.
#
# Runs locally on venv_walkindia with cached m10 masks. ~5 min wall-time.
#
#   ./scripts/tests_streaming/test_e2e_v15c_streaming.sh 2>&1 | tee logs/test_e2e_v15c_streaming.log
set -euo pipefail

cd "$(dirname "$0")/../.."
source venv_walkindia/bin/activate

LOG_TAG="e2e_v15c_stream"
OUTPUT_ROOT="outputs/sanity"
SUBSET="data/sanity_100_dense.json"
LOCAL_DATA="data/val_1k_local"

step() { echo -e "\n═══ $(date +%H:%M:%S) · $1 ═══"; }

step "[1/5] Preflight — sanity subset + cached m10 masks"
test -f "$SUBSET" || { echo "FATAL: $SUBSET missing"; exit 1; }
# m10 SANITY cache — regenerate if absent (CPU-free path for this test)
if [ ! -f "$OUTPUT_ROOT/m10_sam_segment/segments.json" ]; then
    echo "  m10 SANITY cache missing — running m10 SANITY first (~10 min GPU)"
    python -u src/m10_sam_segment.py --SANITY \
        --subset "$SUBSET" --local-data "$LOCAL_DATA" --no-wandb \
        2>&1 | tee "logs/${LOG_TAG}_m10.log"
fi

step "[2/5] m10 --interactions-only (populate n_interactions for has_D_I gate)"
python -u src/m10_sam_segment.py --SANITY \
    --subset "$SUBSET" --local-data "$LOCAL_DATA" \
    --train-config configs/train/ch11_surgery_v15c.yaml \
    --interactions-only --no-wandb \
    2>&1 | tee "logs/${LOG_TAG}_m10_mine.log"

step "[3/5] m11 --streaming (should NOT write D_I .npy tubes)"
# iter11: no shell-level deletion — m11 --streaming overwrites its manifest atomically.
# To force a fresh re-build, pass --cache-policy 2 at the prompt.
python -u src/m11_factor_datasets.py --SANITY --streaming \
    --subset "$SUBSET" --local-data "$LOCAL_DATA" \
    --train-config configs/train/ch11_surgery_v15c.yaml \
    --no-wandb \
    2>&1 | tee "logs/${LOG_TAG}_m11.log"

# Assert: D_I dir should be EMPTY (or contain only verify clips, which at SANITY N=20 = 0)
D_I_COUNT=$(find "$OUTPUT_ROOT/m11_factor_datasets/D_I/" -name "*.npy" 2>/dev/null | wc -l)
step "[4/5] Assert D_I .npy count == 0 (streaming skipped writes)"
echo "  D_I tube .npy files on disk: $D_I_COUNT"
if [ "$D_I_COUNT" -gt 0 ]; then
    echo "  ❌ FAIL: streaming path wrote $D_I_COUNT D_I tubes to disk — should be 0"
    exit 2
fi
echo "  ✅ PASS: D_I/ directory empty (streaming handled tubes on-demand)"

step "[5/5] m09c SANITY 3-stage run (assert Stage 3 D_I streams cleanly)"
# iter11: no shell-level deletion — m09c uses atomic os.replace for ckpt writes and
# its --cache-policy gate owns intermediate cleanup. Pass --cache-policy 2 for fresh.
python -u src/m09c_surgery.py --SANITY \
    --subset "$SUBSET" \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery_v15c.yaml \
    --factor-dir "$OUTPUT_ROOT/m11_factor_datasets/" \
    --local-data "$LOCAL_DATA" \
    --probe-subset "$SUBSET" \
    --probe-local-data "$LOCAL_DATA" \
    --probe-tags "$LOCAL_DATA/tags.json" \
    --no-wandb \
    2>&1 | tee "logs/${LOG_TAG}_m09c.log"

# Assert student_encoder.pt exists
STUDENT_CKPT="$OUTPUT_ROOT/m09c_surgery/student_encoder.pt"
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "  ❌ FAIL: $STUDENT_CKPT not exported — m09c likely crashed mid-stage"
    exit 3
fi
echo "  ✅ PASS: $STUDENT_CKPT exported ($(du -h $STUDENT_CKPT | cut -f1))"

step "🎉 D_I streaming e2e SANITY GREEN — safe to relaunch v15c"
