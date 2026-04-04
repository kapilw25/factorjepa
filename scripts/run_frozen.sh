#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch9: Frozen Encoder Evaluation (FactorJEPA)
# Pipeline: m04(tags) → m05(vjepa) → m05b(×4 baselines) → m04d(motion)
# Evaluation: ./scripts/run_eval.sh --FULL (m06→m06b→m07→m08→m08b, all encoders)
#
# USAGE:
#   ./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log   # ~15 min
#   ./scripts/run_evaluate.sh --POC 2>&1 | tee logs/ch9_sanity.log   # ~<time ??> min
#   ./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log       # ~77h (115K clips)
#
# CACHE CONTROL (output_guard skips completed steps automatically):
#   Use all cached:    ./scripts/run_evaluate.sh --FULL
#   Re-run everything: rm -rf outputs/full && ./scripts/run_evaluate.sh --FULL
#   Re-run one step:   rm outputs/full/embeddings.npy && ./scripts/run_evaluate.sh --FULL
#   Re-run baselines:  rm outputs/full/embeddings_{random,dinov2,clip,vjepa_shuffled}.npy && ...
#
# PREREQUISITES:
#   1. ./setup_env_uv.sh --gpu --from-wheels
#   2. rsync -avhP data/ <gpu-host>:/workspace/factorjepa/data/    # for "--poc" & "val" dataset   # from Mac (~17 min)
#   3. python -u src/m00d_download_subset.py --FULL --no-wandb     # 115K clips (~24 min)
#   4. tmux new -s ch9   # run inside tmux for long runs
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# Reduce CUDA memory fragmentation (prevents batch size death spiral on long runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
SUBSET_FLAG=""

USE_VLLM=false

usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL [--vllm]"
    echo ""
    echo "  --SANITY   Quick validation: 5-20 clips per step (~15 min total)"
    echo "  --POC      10K subset: full eval pipeline (~8-12h)"
    echo "  --FULL     115K full corpus: no subset, dataset size from manifest (~48h)"
    echo ""
    echo "  --vllm     Use vLLM for m04 tagging (3-5x faster, requires venv_vllm)"
    echo "             Without --vllm: uses transformers (slower but always works)"
    exit 1
}

[[ $# -eq 0 ]] && usage

case "$1" in
    --SANITY)
        MODE="SANITY"
        MODE_FLAG="--SANITY"
        SUBSET_FLAG=""
        OUT_DIR="outputs/sanity"
        TAGS_FILE="outputs/sanity/tags_sanity_qwen.json"
        BATCH_M04=""
        ;;
    --POC)
        MODE="POC"
        MODE_FLAG="--FULL"
        SUBSET_FLAG="--subset data/subset_10k.json"
        OUT_DIR="outputs/poc"
        TAGS_FILE="outputs/poc/tags.json"
        BATCH_M04=""
        ;;
    --FULL)
        MODE="FULL"
        MODE_FLAG="--FULL"
        SUBSET_FLAG=""
        OUT_DIR="outputs/full"
        TAGS_FILE="outputs/full/tags.json"
        BATCH_M04=""
        ;;
    *)
        usage
        ;;
esac

# Check for --vllm flag (can appear as $2)
if [[ "${2:-}" == "--vllm" ]]; then
    USE_VLLM=true
fi

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Read MIN_CLIPS from configs/pipeline.yaml
PIPELINE_YAML="configs/pipeline.yaml"
if [[ "$MODE" == "SANITY" ]]; then
    MIN_CLIPS=$(python -c "import yaml; print(yaml.safe_load(open('${PIPELINE_YAML}'))['verify']['sanity_min_clips'])")
else
    MIN_CLIPS=$(python -c "import yaml; print(yaml.safe_load(open('${PIPELINE_YAML}'))['verify']['full_min_clips'])")
fi

# ── Data pre-check (FAIL LOUD if missing) ────────────────────────────
if [[ "$MODE" == "FULL" ]]; then
    if [[ ! -d "data/full_local" || ! -f "data/full_local/manifest.json" ]]; then
        echo "FATAL: data/full_local/ (115K clips, 130GB) not found. Fix with:"
        echo "  python -u src/m00d_download_subset.py --FULL --no-wandb   # full 115K from walkindia-200k (~24 min)"
        exit 1
    fi
elif [[ "$MODE" == "POC" ]]; then
    if [[ ! -d "data/subset_10k_local" || ! -f "data/subset_10k_local/manifest.json" ]]; then
        echo "FATAL: data/subset_10k_local/ (10K clips, 10.5GB) not found. Fix with ONE of:"
        echo "  python -u src/utils/hf_outputs.py download-data                                       # poc 10K + val 1K from factorjepa-outputs (~3 min, measured)"
        echo "  rsync -avhP data/ <gpu-host>:/workspace/factorjepa/data/                              # poc 10K + val 1K from Mac (~17 min)"
        echo "  python -u src/m00d_download_subset.py --FULL --subset data/subset_10k.json --no-wandb  # poc 10K from walkindia-200k (~50 min, downloads all 116 TARs)"
        exit 1
    fi
fi

LOGDIR="logs"
mkdir -p "$LOGDIR" "$OUT_DIR"

MASTER_LOG="$LOGDIR/ch9_${MODE}_$(date +%Y%m%d_%H%M%S).log"

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

# 5 frozen encoders
ENCODERS=("vjepa" "random" "dinov2" "clip" "vjepa_shuffled")

# ── Time estimates ────────────────────────────────────────────────────
# Time estimates based on actual RTX PRO 6000 96GB runs (10K POC, Mar 2026)
if [[ "$MODE" == "SANITY" ]]; then
    T_M00D="~39 min (one-time, skipped if exists)"
    T_M04="~15 sec (cached: ~2h first run)"
    T_M05="~15 sec (cached: ~80 min first run)"
    T_M05B="~15 sec (cached: ~99 min first run)"
    T_M05C="~15 sec (cached: ~5 min first run)"
    T_M04D="~1 min"
    T_M06="~10 sec each"
    T_M06B="~5 sec each"
    T_M07="~5 sec each"
    T_M08="~10 sec each"
    T_M08B="~5 sec"
else
    T_M00D="~39 min (one-time, skipped if exists)"
    T_M04="~2h (1.3 clips/s, Qwen3-VL-8B)"
    T_M05="~80 min (V-JEPA 2 ViT-G, producer-consumer)"
    T_M05B="~99 min (4 baselines sequential)"
    T_M05C="~4.7h (augmented V-JEPA, 30K subsample + torch.compile)"
    T_M04D="~70 min (PyAV parallel decode + RAFT batch=8, 2.4 clips/s)"
    T_M06="~30 sec each (FAISS-GPU + bootstrap CI)"
    T_M06B="~10 sec each (CPU temporal correlation)"
    T_M07="~5 sec each (cuML GPU UMAP)"
    T_M08="~30 sec each (matplotlib CPU)"
    T_M08B="~30 sec"
fi

# ═══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT: Fail fast if GPU/packages broken
# (mirrors plan_execution.md "Verify setup" section)
# ═══════════════════════════════════════════════════════════════════════

log "Ch9 pipeline starting (mode=${MODE})"
log "Master log: ${MASTER_LOG}"
echo "" | tee -a "$MASTER_LOG"
log "=== PRE-FLIGHT: GPU + Package Verification ==="

python -c "
import sys
errors = []

# GPU
try:
    import torch
    if not torch.cuda.is_available():
        errors.append('CUDA not available')
    else:
        props = torch.cuda.get_device_properties(0)
        print(f'GPU:            {torch.cuda.get_device_name(0)}, VRAM: {props.total_memory/1e9:.0f}GB')
        print(f'PyTorch:        {torch.__version__}')
        print(f'CUDA:           {torch.version.cuda}')
except ImportError:
    errors.append('torch not installed')

# FAISS-GPU
try:
    import faiss
    ngpu = faiss.get_num_gpus()
    if ngpu == 0:
        errors.append('FAISS-GPU: 0 GPUs detected (need >= 1)')
    else:
        print(f'FAISS GPUs:     {ngpu}')
except ImportError:
    errors.append('faiss not installed')

# Flash-Attention 2
try:
    import flash_attn
    print(f'Flash-Attn:     {flash_attn.__version__}')
except ImportError:
    errors.append('flash_attn not installed (V-JEPA requires FA2)')

# transformers
try:
    import transformers
    print(f'Transformers:   {transformers.__version__}')
except ImportError:
    errors.append('transformers not installed')

# cuML
try:
    import cuml
    print(f'cuML:           {cuml.__version__}')
except ImportError:
    errors.append('cuml not installed (m07 UMAP needs cuML)')

# wandb (optional but expected)
try:
    import wandb
    print(f'wandb:          {wandb.__version__}')
except ImportError:
    print('wandb:          NOT INSTALLED (ok, using --no-wandb)')

# HF Token
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    t = os.getenv('HF_TOKEN')
    if t:
        print(f'HF_TOKEN:       {t[:10]}...')
    else:
        errors.append('HF_TOKEN not found in .env (private dataset needs auth)')
except ImportError:
    errors.append('python-dotenv not installed')

if errors:
    print(f'\nFATAL: {len(errors)} pre-flight check(s) failed:')
    for e in errors:
        print(f'  - {e}')
    print('\nFix issues or run: ./setup_env_uv.sh --gpu --from-wheels')
    sys.exit(1)
else:
    print('\nPre-flight: ALL PASSED')
" 2>&1 | tee -a "$MASTER_LOG"

# Subset file check (FULL mode only)
if [[ "$MODE" == "FULL" ]]; then
    python -c "
import json, sys
try:
    d = json.load(open('data/subset_10k.json'))
    print(f'Subset:         {d[\"n\"]} clips from {d[\"num_videos\"]} videos')
except FileNotFoundError:
    print('FATAL: data/subset_10k.json not found')
    sys.exit(1)
" 2>&1 | tee -a "$MASTER_LOG"
fi

log "Pre-flight complete."

# ── Output preflight: check all inputs/outputs before GPU work ────────
log "=== OUTPUT PREFLIGHT ==="
if [[ "$MODE" == "FULL" ]]; then
    LOCAL_DATA="data/full_local"
else
    LOCAL_DATA="data/subset_10k_local"
fi
python -u src/utils/output_guard.py preflight_evaluate "$OUT_DIR" "$TAGS_FILE" "$LOCAL_DATA" 2>&1 | tee -a "$MASTER_LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log "FATAL: Output preflight failed or aborted."
    exit 1
fi

log "Starting pipeline..."
echo "" | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════

# ── Step 0: Pre-download to local disk ────────────────────────────────
if [[ "$MODE" == "FULL" ]]; then
    LOCAL_DATA="data/full_local"
    DOWNLOAD_CMD="src/m00d_download_subset.py --FULL --no-wandb"
else
    LOCAL_DATA="data/subset_10k_local"
    DOWNLOAD_CMD="src/m00d_download_subset.py --FULL --subset data/subset_10k.json --no-wandb"
fi
LOCAL_FLAG=""

if [[ -d "$LOCAL_DATA" && -f "$LOCAL_DATA/manifest.json" ]]; then
    log "Local data already exists: $LOCAL_DATA (skipping download)"
    LOCAL_FLAG="--local-data $LOCAL_DATA"
else
    run_step 0 "m00d pre-download" "$T_M00D" \
        "$LOGDIR/m00d_download.log" \
        $DOWNLOAD_CMD \
        || { log "FATAL: m00d failed. Cannot proceed without local data."; exit 1; }

    if [[ -d "$LOCAL_DATA" && -f "$LOCAL_DATA/manifest.json" ]]; then
        LOCAL_FLAG="--local-data $LOCAL_DATA"
        log "Local subset ready: $LOCAL_DATA"
    else
        log "WARNING: m00d did not produce expected output. Falling back to HF streaming."
    fi
fi

# ── Step 1: VLM tagging (Qwen) ───────────────────────────────────────
# --vllm: use venv_vllm/bin/python + m04_vlm_tag_vllm.py (3-5x faster)
# default: use venv_walkindia python + m04_vlm_tag.py (always works)
if [[ "$USE_VLLM" == true ]]; then
    VLLM_PYTHON="venv_vllm/bin/python"
    if [[ ! -x "$VLLM_PYTHON" ]]; then
        log "FATAL: --vllm requires venv_vllm. Run: ./setup_env_uv.sh --gpu"
        exit 1
    fi
    STEP_COUNT=$((STEP_COUNT + 1))
    banner 1 "m04 VLM tagging (Qwen via vLLM)" "$T_M04"
    log "CMD: $VLLM_PYTHON -u src/m04_vlm_tag_vllm.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb"
    VLLM_START=$(date +%s)
    if $VLLM_PYTHON -u src/m04_vlm_tag_vllm.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
        2>&1 | tee "$LOGDIR/m04_vllm_${MODE,,}.log" | tee -a "$MASTER_LOG"; then
        VLLM_END=$(date +%s)
        VLLM_ELAPSED=$(( VLLM_END - VLLM_START ))
        log "PASSED: m04 VLM tagging via vLLM ($((VLLM_ELAPSED/60))m $((VLLM_ELAPSED%60))s)"
        STEP_PASS=$((STEP_PASS + 1))
    else
        VLLM_END=$(date +%s)
        VLLM_ELAPSED=$(( VLLM_END - VLLM_START ))
        log "FAILED: m04 vLLM ($((VLLM_ELAPSED/60))m $((VLLM_ELAPSED%60))s). Falling back to transformers..."
        STEP_FAIL=$((STEP_FAIL + 1))
        USE_VLLM=false
    fi
fi
if [[ "$USE_VLLM" == false ]]; then
    run_step 1 "m04 VLM tagging (Qwen via transformers)" "$T_M04" \
        "$LOGDIR/m04_${MODE,,}_qwen.log" \
        src/m04_vlm_tag.py --model qwen $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $BATCH_M04 --no-wandb \
        || { log "WARNING: m04 failed. Step 5 (m06 metrics) will fail without tags."; }
fi
    # Non-fatal: embeddings (Steps 2-4) don't need tags. Only m06 does.

verify "Step 1 tags" "
import json, sys
try:
    t = json.load(open('${TAGS_FILE}'))
    n = len(t)
    fields = len(t[0].keys()) if t else 0
    print(f'  tags: {n} clips, {fields} fields')
    if n < ${MIN_CLIPS}: print(f'  WARN: only {n} clips (expected >= ${MIN_CLIPS})')
except Exception as e:
    print(f'  tags: NOT FOUND or corrupt ({e})')
"

# Symlink tags file so downstream steps (m06, m08) find it at the canonical path
CANONICAL_TAGS="${OUT_DIR}/tags.json"
if [[ -f "$TAGS_FILE" && "$TAGS_FILE" != "$CANONICAL_TAGS" ]]; then
    ln -sf "$(cd "$(dirname "$TAGS_FILE")" && pwd)/$(basename "$TAGS_FILE")" "$CANONICAL_TAGS"
    log "Symlinked: $(basename "$TAGS_FILE") → tags.json"
fi

# ── Step 2: V-JEPA embeddings ────────────────────────────────────────
# No dedup: using V-JEPA's own similarity to filter eval set is circular.
# Hard mode ±30s exclusion in m06 handles true temporal duplicates.
run_step 2 "m05 V-JEPA embeddings" "$T_M05" \
    "$LOGDIR/m05_${MODE,,}.log" \
    src/m05_vjepa_embed.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
    || { log "FATAL: m05 failed. Steps 3-7 depend on V-JEPA embeddings."; exit 1; }

verify "Step 2 embeddings" "
import numpy as np
emb = np.load('${OUT_DIR}/embeddings.npy')
paths = np.load('${OUT_DIR}/embeddings.paths.npy', allow_pickle=True)
print(f'  embeddings.npy:       {emb.shape}')
print(f'  embeddings.paths.npy: {len(paths)} keys')
print(f'  shape match: {emb.shape[0] == len(paths)}')
"

# ── Step 3: Baseline embeddings (per-encoder fresh process to avoid CUDA fragmentation)
BASELINE_ENCODERS=("random" "dinov2" "clip" "vjepa_shuffled")
for benc in "${BASELINE_ENCODERS[@]}"; do
    run_step "3-${benc}" "m05b ${benc} embeddings" "$T_M05B" \
        "$LOGDIR/m05b_${MODE,,}_${benc}.log" \
        src/m05b_baselines.py --encoder "$benc" $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
        || { log "WARNING: m05b ${benc} failed. Skipping."; }
done
    # Non-fatal: V-JEPA metrics still work without baselines

verify "Step 3 baselines" "
import numpy as np
for enc, sfx, dim in [('random','_random',1408), ('dinov2','_dinov2',1536),
                       ('clip','_clip',768), ('vjepa_shuffled','_vjepa_shuffled',1408)]:
    try:
        emb = np.load(f'${OUT_DIR}/embeddings{sfx}.npy')
        paths = np.load(f'${OUT_DIR}/embeddings{sfx}.paths.npy', allow_pickle=True)
        ok = 'OK' if emb.shape[1] == dim and emb.shape[0] == len(paths) else 'MISMATCH'
        print(f'  {ok} {enc:20s} {emb.shape}')
    except FileNotFoundError:
        print(f'  MISSING {enc:20s}')
"

# ── Step 4: SKIPPED — m05c True Overlap
# Reason: m05c only produces True Overlap for frozen V-JEPA. Other encoders (DINOv2,
# CLIP, random, shuffled, adapted) would still use dim-split. Mixing two different
# metrics under the same "Overlap@K" label on the radar is misleading. Using dim-split
# consistently for ALL encoders (renamed to DimConsist@K) is scientifically cleaner.
# To re-enable: uncomment and ensure m05c runs for ALL encoders, not just frozen V-JEPA.
log "SKIPPED: m05c True Overlap (using DimConsist@K consistently for all encoders)"

# ── Step 4.5: Motion features (GPU-RAFT optical flow) ─────────────────
run_step "4.5" "m04d GPU-RAFT motion features (13D per clip)" "$T_M04D" \
    "$LOGDIR/m04d_${MODE,,}.log" \
    src/m04d_motion_features.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
    || { log "WARNING: m04d failed. m06b metrics 1-3 will be skipped (4-5 still work)."; }

verify "Step 4.5 motion features" "
import numpy as np, os
mf = '${OUT_DIR}/motion_features.npy'
if os.path.exists(mf):
    m = np.load(mf)
    print(f'  motion_features.npy: {m.shape} (expect ~N x 13)')
else:
    print(f'  motion_features.npy: NOT FOUND (m06b will skip motion-based metrics)')
"

# ── Evaluation: handled by run_eval.sh (standalone, reusable across Ch9/Ch10/Ch11)
log "Embeddings complete. Run evaluation separately:"
log "  ./scripts/run_eval.sh --${MODE} 2>&1 | tee logs/eval_${MODE,,}.log"

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== FINAL VERIFICATION: Ch9 Outputs (tags + embeddings + motion) ==="

python -c "
import json, numpy as np, os

out = '${OUT_DIR}'
tags_file = '${TAGS_FILE}'
ok_count = 0
fail_count = 0

def check(label, path, validator=None):
    global ok_count, fail_count
    if os.path.exists(path):
        detail = validator(path) if validator else ''
        print(f'  OK   {label:40s} {detail}')
        ok_count += 1
    else:
        print(f'  MISS {label:40s} {path}')
        fail_count += 1

# Tags
check('tags.json', tags_file,
      lambda p: f'{len(json.load(open(p)))} clips, {len(json.load(open(p))[0].keys())} fields')

# Embeddings (5 frozen encoders)
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    check(f'{enc} embeddings', f'{out}/embeddings{sfx}.npy',
          lambda p: str(np.load(p).shape))

# Motion features
check('motion_features.npy', f'{out}/motion_features.npy',
      lambda p: str(np.load(p).shape))

print(f'\n=== TOTAL: {ok_count} OK, {fail_count} MISSING ===')
print(f'Next: ./scripts/run_eval.sh --${MODE}')
" 2>&1 | tee -a "$MASTER_LOG"

finalize "Ch9"
