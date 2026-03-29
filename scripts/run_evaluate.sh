#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Evaluate Frozen V-JEPA on Indian Urban Walking Clips (FactorJEPA Ch9)
#
# Pipeline: m04 → m05 → m05b → m05c → m04d → m06 → m06b → m07 → m08 → m08b
# Spatial (9 metrics) + Temporal (5 metrics) evaluation, 5 encoders, 95% CI
#
# USAGE:
#   ./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log   # Quick validation (~15 min)
#   ./scripts/run_evaluate.sh --POC 2>&1 | tee logs/ch9_poc.log         # 10K subset (~8h)
#   ./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log       # 115K corpus (~100h+)
#
# Features: pre-flight checks, checkpoint/resume, error handling, verification
# All steps SEQUENTIAL on single GPU. Skips completed steps automatically.
# Prerequisites: ./setup_env_uv.sh --gpu [--from-wheels]
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
SUBSET_FLAG=""

usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL"
    echo ""
    echo "  --SANITY   Quick validation: 5-20 clips per step (~15 min total)"
    echo "  --POC      10K subset: full eval pipeline (~8-12h)"
    echo "  --FULL     115K full corpus: no subset, dataset size from manifest (~48h)"
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

# ── Helper functions ──────────────────────────────────────────────────

STEP_COUNT=0
STEP_PASS=0
STEP_FAIL=0
PIPELINE_START=$(date +%s)

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg" | tee -a "$MASTER_LOG"
}

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

    if python -u "${cmd[@]}" 2>&1 | tee "$log_file" | tee -a "$MASTER_LOG"; then
        local step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        local mins=$(( elapsed / 60 ))
        local secs=$(( elapsed % 60 ))
        log "PASSED: ${step_name} (${mins}m ${secs}s)"
        STEP_PASS=$((STEP_PASS + 1))
        return 0
    else
        local step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        local mins=$(( elapsed / 60 ))
        local secs=$(( elapsed % 60 ))
        log "FAILED: ${step_name} (${mins}m ${secs}s) — exit code $?"
        STEP_FAIL=$((STEP_FAIL + 1))
        return 1
    fi
}

verify() {
    # Usage: verify "description" "python_code"
    local desc="$1"
    local code="$2"
    if python -c "$code" 2>&1 | tee -a "$MASTER_LOG"; then
        log "VERIFY OK: $desc"
    else
        log "VERIFY FAIL: $desc (non-fatal, continuing)"
    fi
}

# 5 encoders for m06/m07 loops
ENCODERS=("vjepa" "random" "dinov2" "clip" "vjepa_shuffled")

# ── Time estimates ────────────────────────────────────────────────────
# Time estimates based on actual RTX PRO 6000 96GB runs (10K POC, Mar 2026)
if [[ "$MODE" == "SANITY" ]]; then
    T_M00D="~39 min (one-time, skipped if exists)"
    T_M04="~15 sec (cached: ~2h first run)"
    T_M05="~15 sec (cached: ~80 min first run)"
    T_M05B="~15 sec (cached: ~99 min first run)"
    T_M05C="~15 sec (cached: ~93 min first run)"
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
    T_M05C="~93 min (augmented V-JEPA, deduped clips)"
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
run_step 1 "m04 VLM tagging (Qwen)" "$T_M04" \
    "$LOGDIR/m04_${MODE,,}_qwen.log" \
    src/m04_vlm_tag.py --model qwen $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $BATCH_M04 --no-wandb \
    || { log "WARNING: m04 failed. Step 5 (m06 metrics) will fail without tags."; }
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

# ── Step 3: Baseline embeddings (all 4) ──────────────────────────────
run_step 3 "m05b baselines (all 4 encoders)" "$T_M05B" \
    "$LOGDIR/m05b_${MODE,,}_all.log" \
    src/m05b_baselines.py --encoder all $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
    || { log "WARNING: m05b failed. Some baseline metrics will be missing."; }
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

# ── Step 4: True Overlap augmented embeddings ─────────────────────────
run_step 4 "m05c True Overlap augmented embeddings" "$T_M05C" \
    "$LOGDIR/m05c_${MODE,,}.log" \
    src/m05c_true_overlap.py $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
    || { log "FATAL: m05c failed."; exit 1; }

verify "Step 4 augmented embeddings" "
import numpy as np
try:
    a = np.load('${OUT_DIR}/overlap_augA.npy')
    b = np.load('${OUT_DIR}/overlap_augB.npy')
    k = np.load('${OUT_DIR}/overlap_keys.npy', allow_pickle=True)
    print(f'  overlap_augA.npy: {a.shape}')
    print(f'  overlap_augB.npy: {b.shape}')
    print(f'  overlap_keys.npy: {len(k)} keys')
    print(f'  shape match: {a.shape == b.shape and a.shape[0] == len(k)}')
except FileNotFoundError as e:
    print(f'  MISSING: {e}')
"

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

# ── Step 5: FAISS metrics (5 encoders) ────────────────────────────────
for enc in "${ENCODERS[@]}"; do
    EXTRA_FLAGS=""
    if [[ "$enc" == "vjepa" ]]; then
        EXTRA_FLAGS="--true-overlap"
    fi
    run_step "5-${enc}" "m06 FAISS metrics (${enc})" "$T_M06" \
        "$LOGDIR/m06_${MODE,,}_${enc}.log" \
        src/m06_faiss_metrics.py --encoder "$enc" $EXTRA_FLAGS $MODE_FLAG $SUBSET_FLAG --no-wandb \
        || { log "WARNING: m06 ${enc} failed. Skipping."; }
done

verify "Step 5 metrics (all encoders)" "
import json
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    try:
        m = json.load(open(f'${OUT_DIR}/m06_metrics{sfx}.json'))
        ov = m['easy'].get('overlap_method', 'dim_split')
        print(f'  OK {enc:20s} Prec@K={m[\"easy\"][\"prec_at_k\"]:5.1f}%  mAP={m[\"easy\"][\"map_at_k\"]:.3f}  overlap={ov}')
    except FileNotFoundError:
        print(f'  MISSING {enc:20s} m06_metrics{sfx}.json')
"

# ── Step 5.5: Temporal correlation (5 encoders, CPU) ──────────────────
for enc in "${ENCODERS[@]}"; do
    run_step "5.5-${enc}" "m06b temporal correlation (${enc})" "$T_M06B" \
        "$LOGDIR/m06b_${MODE,,}_${enc}.log" \
        src/m06b_temporal_corr.py --encoder "$enc" $MODE_FLAG $SUBSET_FLAG --no-wandb \
        || { log "WARNING: m06b ${enc} failed. Skipping."; }
done

verify "Step 5.5 temporal metrics (all encoders)" "
import json, os
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    f = f'${OUT_DIR}/m06b_temporal_corr{sfx}.json'
    if os.path.exists(f):
        m = json.load(open(f))
        rho = m.get('spearman_rho', 'N/A')
        loc = m.get('temporal_locality', {}).get('ratio', 'N/A')
        print(f'  OK {enc:20s} rho={rho}  locality={loc}')
    else:
        print(f'  MISS {enc:20s}')
"

# ── Step 6: UMAP (5 encoders) ────────────────────────────────────────
for enc in "${ENCODERS[@]}"; do
    run_step "6-${enc}" "m07 UMAP (${enc})" "$T_M07" \
        "$LOGDIR/m07_${MODE,,}_${enc}.log" \
        src/m07_umap.py --encoder "$enc" $MODE_FLAG $SUBSET_FLAG --no-wandb \
        || { log "WARNING: m07 ${enc} UMAP failed. Skipping."; }
done

verify "Step 6 UMAP (all encoders)" "
import numpy as np
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    try:
        u = np.load(f'${OUT_DIR}/umap_2d{sfx}.npy')
        print(f'  OK {enc:20s} umap_2d{sfx}.npy: {u.shape}')
    except FileNotFoundError:
        print(f'  MISSING {enc:20s} umap_2d{sfx}.npy')
"

# ── Step 7a: Per-encoder plots (loop × 5 encoders) ───────────────────
for enc in "${ENCODERS[@]}"; do
    run_step "7a-${enc}" "m08 plots (${enc})" "$T_M08" \
        "$LOGDIR/m08_${MODE,,}_${enc}.log" \
        src/m08_plot.py --encoder "$enc" $MODE_FLAG $SUBSET_FLAG --no-wandb \
        || { log "WARNING: m08 ${enc} plots failed."; }
done

# ── Step 7b: Multi-encoder comparison ─────────────────────────────────
run_step "7b" "m08b multi-encoder comparison" "$T_M08B" \
    "$LOGDIR/m08b_${MODE,,}.log" \
    src/m08b_compare.py $MODE_FLAG $SUBSET_FLAG --no-wandb \
    || { log "WARNING: m08b comparison failed."; }

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION (mirrors plan_execution.md "Final Verification")
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== FINAL VERIFICATION: All Ch9 Outputs ==="

python -c "
import json, numpy as np, os

out = '${OUT_DIR}'
tags_file = '${TAGS_FILE}'
ok_count = 0
fail_count = 0

def check(label, path, validator=None):
    global ok_count, fail_count
    if os.path.exists(path):
        detail = ''
        if validator:
            detail = validator(path)
        print(f'  OK   {label:40s} {detail}')
        ok_count += 1
    else:
        print(f'  MISS {label:40s} {path}')
        fail_count += 1

# Tags
print('=== TAGS (v3 taxonomy) ===')
check('tags.json', tags_file,
      lambda p: f'{len(json.load(open(p)))} clips, {len(json.load(open(p))[0].keys())} fields')

# Embeddings (5 encoders)
print()
print('=== EMBEDDINGS (5 encoders) ===')
for enc, sfx, dim in [('vjepa','',1408), ('random','_random',1408), ('dinov2','_dinov2',1536),
                       ('clip','_clip',768), ('vjepa_shuffled','_vjepa_shuffled',1408)]:
    check(f'{enc} embeddings',  f'{out}/embeddings{sfx}.npy',
          lambda p: str(np.load(p).shape))
    check(f'{enc} paths',       f'{out}/embeddings{sfx}.paths.npy',
          lambda p: f'{len(np.load(p, allow_pickle=True))} keys')

# True Overlap
print()
print('=== TRUE OVERLAP ===')
check('overlap_augA.npy', f'{out}/overlap_augA.npy',
      lambda p: str(np.load(p).shape))
check('overlap_augB.npy', f'{out}/overlap_augB.npy',
      lambda p: str(np.load(p).shape))

# Metrics (5 encoders)
print()
print('=== METRICS (5 encoders) ===')
print(f'  {\"Encoder\":20s} {\"Prec@K\":>8s} {\"mAP@K\":>8s} {\"Cycle@K\":>8s} {\"nDCG@K\":>8s}')
print('  ' + '-' * 58)
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    mpath = f'{out}/m06_metrics{sfx}.json'
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        e = m['easy']
        print(f'  {enc:20s} {e[\"prec_at_k\"]:7.1f}% {e[\"map_at_k\"]:8.4f} {e[\"cycle_at_k\"]:7.1f}% {e[\"ndcg_at_k\"]:8.4f}')
        ok_count += 1
    else:
        print(f'  {enc:20s} MISSING')
        fail_count += 1

# UMAP (5 encoders)
print()
print('=== UMAP (5 encoders) ===')
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    check(f'{enc} umap', f'{out}/umap_2d{sfx}.npy',
          lambda p: str(np.load(p).shape))

# Motion features
print()
print('=== MOTION FEATURES (m04d) ===')
check('motion_features.npy', f'{out}/motion_features.npy',
      lambda p: str(np.load(p).shape))

# Temporal metrics
print()
print('=== TEMPORAL METRICS (m06b) ===')
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    check(f'{enc} temporal', f'{out}/m06b_temporal_corr{sfx}.json')

# Plots
print()
print('=== PLOTS ===')
for f in ['m08_umap.png', 'm08_confusion_matrix.png', 'm08_knn_grid.png',
          'm08b_encoder_comparison.png', 'm08b_radar.png', 'm08b_comparison_table.tex',
          'm08b_spatial_temporal_bar.png', 'm08b_tradeoff_scatter.png']:
    check(f'plot: {f}', f'{out}/{f}')

print()
print(f'=== TOTAL: {ok_count} OK, {fail_count} MISSING ===')
" 2>&1 | tee -a "$MASTER_LOG"

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
echo "  Ch9 PIPELINE COMPLETE" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  Mode:      ${MODE}" | tee -a "$MASTER_LOG"
echo "  Total:     ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s" | tee -a "$MASTER_LOG"
echo "  Steps:     ${STEP_PASS} passed, ${STEP_FAIL} failed, ${STEP_COUNT} total" | tee -a "$MASTER_LOG"
echo "  Outputs:   ${OUT_DIR}/" | tee -a "$MASTER_LOG"
echo "  Master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

if [[ $STEP_FAIL -gt 0 ]]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "  WARNING: ${STEP_FAIL} step(s) failed. Check individual logs above." | tee -a "$MASTER_LOG"
    echo "  All steps with checkpoints can be RE-RUN safely (auto-resume)." | tee -a "$MASTER_LOG"
    exit 1
fi

exit 0
