#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch10: V-JEPA continual pretraining + 4-lambda drift control ablation
# Pipeline: m09 train → m05 re-embed → m06 metrics → m08b compare
# Needs ~65 GB disk. Auto-downloads data if missing.
#
# USAGE:
#   ./setup_env_uv.sh --gpu --from-wheels   # one-time setup
#   ./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
#   tmux new -s ch10  # for --FULL (Ctrl+B,D detach / tmux attach -t ch10)
#   ./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full_10k.log
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
SUBSET_FLAG=""
CONFIG="configs/pretrain/vitg16_indian.yaml"

usage() {
    echo "Usage: $0 --SANITY | --FULL [--config CONFIG]"
    echo ""
    echo "  --SANITY   Quick validation: 50 steps, batch_size=2 (~5 min)"
    echo "  --FULL     10K POC: 2000-10000 steps on subset (~20h)"
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
            MIN_CLIPS=5
            shift
            ;;
        --FULL)
            MODE="FULL"
            MODE_FLAG="--FULL"
            SUBSET_FLAG="--subset data/subset_10k.json"
            OUT_DIR="outputs/poc"
            MIN_CLIPS=1000
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

# ── Helper functions (same as run_evaluate.sh) ────────────────────────

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
    local desc="$1"
    local code="$2"
    if python -c "$code" 2>&1 | tee -a "$MASTER_LOG"; then
        log "VERIFY OK: $desc"
    else
        log "VERIFY FAIL: $desc (non-fatal, continuing)"
    fi
}

# ── Time estimates ────────────────────────────────────────────────────
if [[ "$MODE" == "SANITY" ]]; then
    T_M09="~5 min (50 steps, batch_size=2)"
    T_M05_RE="~15 sec (cached)"
    T_M06_RE="~10 sec"
    T_M08B_RE="~5 sec"
else
    T_M09="~20h (2000-10000 steps, batch_size=32)"
    T_M05_RE="~80 min (re-embed 10K clips with adapted encoder)"
    T_M06_RE="~30 sec (FAISS-GPU metrics)"
    T_M08B_RE="~30 sec (frozen vs adapted comparison)"
fi

# ── Auto-detect optimal batch size from profiler ─────────────────────
PROFILE_JSON="outputs/profile/profile_data.json"
BATCH_FLAG=""

if [[ "$MODE" != "SANITY" && -f "$PROFILE_JSON" ]]; then
    # Read max BS at ≤90% VRAM from profiler results (grad_ckpt run)
    OPTIMAL_BS=$(python -c "
import json
d = json.load(open('${PROFILE_JSON}'))
gpu_gb = d['gpu_total_gb']
target = gpu_gb * 0.90
best = 4  # fallback
for bs, info in sorted(d.get('grad_ckpt', {}).items(), key=lambda x: int(x[0])):
    if info['peak_gb'] <= target:
        best = int(bs)
print(best)
" 2>/dev/null || echo "32")
    BATCH_FLAG="--batch-size $OPTIMAL_BS"
    log "Auto batch size: ${OPTIMAL_BS} (from profiler, <=90% VRAM)"
else
    if [[ "$MODE" != "SANITY" ]]; then
        log "No profiler data found. Using config default batch size."
        log "Run: python scripts/profile_vram.py  (3 min, finds optimal BS)"
    fi
fi

# ── Local data ────────────────────────────────────────────────────────
LOCAL_DATA="data/subset_10k_local"
LOCAL_FLAG=""

if [[ -d "$LOCAL_DATA" && -f "$LOCAL_DATA/manifest.json" ]]; then
    log "Local subset exists: $LOCAL_DATA"
    LOCAL_FLAG="--local-data $LOCAL_DATA"
else
    log "WARNING: Local data not found at $LOCAL_DATA"
    log "Training will use HF streaming (slower)."
    log "To fix: rsync data/ from Mac, or run m00d_download_subset.py first."
fi

# ═══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════════

log "Ch10 pipeline starting (mode=${MODE})"
log "Config: ${CONFIG}"
log "Master log: ${MASTER_LOG}"
echo "" | tee -a "$MASTER_LOG"
log "=== PRE-FLIGHT ==="

python -c "
import sys
errors = []

# GPU
try:
    import torch
    if not torch.cuda.is_available():
        errors.append('CUDA not available')
    else:
        print(f'GPU:        {torch.cuda.get_device_name(0)}')
        print(f'VRAM:       {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
        print(f'PyTorch:    {torch.__version__}')
except ImportError:
    errors.append('torch not installed')

# vjepa2 dependency
import os
if not os.path.exists('deps/vjepa2/src/models/vision_transformer.py'):
    errors.append('deps/vjepa2 not found. Run: git clone --depth 1 https://github.com/facebookresearch/vjepa2.git deps/vjepa2')
else:
    print('vjepa2:     deps/vjepa2/ present')

# Flash-Attention 2
try:
    import flash_attn
    print(f'Flash-Attn: {flash_attn.__version__}')
except ImportError:
    errors.append('flash_attn not installed')

# FAISS-GPU (for validation Cycle@K)
try:
    import faiss
    if faiss.get_num_gpus() == 0:
        errors.append('FAISS-GPU: 0 GPUs')
    else:
        print(f'FAISS GPUs: {faiss.get_num_gpus()}')
except ImportError:
    errors.append('faiss not installed')

# Config file
if not os.path.exists('${CONFIG}'):
    errors.append('Config not found: ${CONFIG}')
else:
    print(f'Config:     ${CONFIG}')

# Ch9 baseline (needed for comparison)
baseline = '${OUT_DIR}/m06_metrics.json'
if os.path.exists(baseline):
    import json
    m = json.load(open(baseline))
    print(f'Baseline:   Prec@K={m[\"easy\"][\"prec_at_k\"]:.1f}% (Ch9 frozen)')
else:
    print(f'WARNING:    Ch9 baseline not found ({baseline})')
    print(f'            Run run_evaluate.sh first for comparison metrics')

if errors:
    print(f'\nFATAL: {len(errors)} check(s) failed:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('\nPre-flight: ALL PASSED')
" 2>&1 | tee -a "$MASTER_LOG"

log "Pre-flight complete. Starting pipeline..."
echo "" | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════
# PIPELINE: Lambda Ablation Sweep (4 values × 3 steps each)
# ═══════════════════════════════════════════════════════════════════════

# Lambda values to sweep (dots → underscores in dir/log names)
LAMBDAS=("0" "0.001" "0.01" "0.1")
LAMBDA_DIRS=("lambda0" "lambda0_001" "lambda0_01" "lambda0_1")
COMPLETED_LAMBDAS=()

for idx in "${!LAMBDAS[@]}"; do
    LAM="${LAMBDAS[$idx]}"
    LAM_DIR="${LAMBDA_DIRS[$idx]}"
    LAM_OUT="${OUT_DIR}/m09_${LAM_DIR}"

    echo "" | tee -a "$MASTER_LOG"
    log "╔═══════════════════════════════════════════════════════════╗"
    log "║  ABLATION: lambda=${LAM} (${LAM_DIR})                    "
    log "╚═══════════════════════════════════════════════════════════╝"

    # ── Step A: Train m09 with this lambda ────────────────────────
    run_step "${LAM_DIR}-train" "m09 pretrain (lambda=${LAM})" "$T_M09" \
        "$LOGDIR/m09_${MODE,,}_${LAM_DIR}.log" \
        src/m09_pretrain.py --config "$CONFIG" --lambda-reg "$LAM" \
            $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb \
        || { log "FAILED: m09 lambda=${LAM}. Skipping to next lambda."; continue; }

    verify "Training lambda=${LAM}" "
import os
out = '${LAM_OUT}'
for f in ['m09_ckpt_final.pt', 'student_encoder.pt']:
    path = os.path.join(out, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f'  OK   {f:30s} {size_mb:.0f} MB')
    else:
        print(f'  MISS {f}')
"

    # ── Step B: Re-embed with this lambda's adapted encoder ───────
    ADAPTED_MODEL="${LAM_OUT}/student_encoder.pt"

    if [[ -f "$ADAPTED_MODEL" ]]; then
        run_step "${LAM_DIR}-embed" "m05 re-embed (lambda=${LAM})" "$T_M05_RE" \
            "$LOGDIR/m05_${MODE,,}_${LAM_DIR}.log" \
            src/m05_vjepa_embed.py --model "$ADAPTED_MODEL" \
                $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-dedupe --no-wandb \
            || { log "WARNING: m05 re-embed failed for lambda=${LAM}."; }
    else
        log "WARNING: student_encoder.pt not found for lambda=${LAM}. Skipping re-embed."
    fi

    # ── Step C: FAISS metrics on this lambda's embeddings ─────────
    run_step "${LAM_DIR}-metrics" "m06 metrics (lambda=${LAM})" "$T_M06_RE" \
        "$LOGDIR/m06_${MODE,,}_${LAM_DIR}.log" \
        src/m06_faiss_metrics.py --encoder vjepa_adapted \
            $MODE_FLAG $SUBSET_FLAG --no-wandb \
        || { log "WARNING: m06 metrics failed for lambda=${LAM}."; }

    COMPLETED_LAMBDAS+=("$LAM_DIR")
    log "Completed ablation: lambda=${LAM} (${LAM_DIR})"
done

# ── Final: Compare all lambdas + frozen baseline on one radar ─────────
if [[ ${#COMPLETED_LAMBDAS[@]} -gt 0 ]]; then
    # Build encoder list: frozen vjepa + all completed lambda variants
    ENCODER_LIST="vjepa"
    for ld in "${COMPLETED_LAMBDAS[@]}"; do
        ENCODER_LIST="${ENCODER_LIST},vjepa_${ld}"
    done

    run_step "compare" "m08b ablation comparison (${#COMPLETED_LAMBDAS[@]} lambdas + frozen)" "$T_M08B_RE" \
        "$LOGDIR/m08b_ablation_${MODE,,}.log" \
        src/m08b_compare.py --encoders "$ENCODER_LIST" \
            $MODE_FLAG $SUBSET_FLAG --no-wandb \
        || { log "WARNING: m08b ablation comparison failed."; }
fi

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== FINAL VERIFICATION: Ch10 Ablation Outputs ==="

python -c "
import json, os

out = '${OUT_DIR}'
ok = 0
fail = 0

def check(label, path):
    global ok, fail
    if os.path.exists(path):
        print(f'  OK   {label}')
        ok += 1
    else:
        print(f'  MISS {label}')
        fail += 1

lambdas = [('0', 'lambda0'), ('0.001', 'lambda0_001'),
           ('0.01', 'lambda0_01'), ('0.1', 'lambda0_1')]

print('=== TRAINING CHECKPOINTS ===')
for lam_val, lam_dir in lambdas:
    lam_out = f'{out}/m09_{lam_dir}'
    check(f'lambda={lam_val:6s} final ckpt',    f'{lam_out}/m09_ckpt_final.pt')
    check(f'lambda={lam_val:6s} student enc',    f'{lam_out}/student_encoder.pt')

print()
print('=== FROZEN vs ALL LAMBDAS ===')
frozen_path = f'{out}/m06_metrics.json'
if not os.path.exists(frozen_path):
    print(f'  Ch9 frozen baseline not found ({frozen_path})')
    print(f'  Run run_evaluate.sh --FULL first')
else:
    frozen = json.load(open(frozen_path))['easy']
    print(f'  {\"lambda\":>10s} {\"Prec@K\":>8s} {\"Cycle@K\":>8s} {\"mAP@K\":>8s} {\"Delta P\":>8s}')
    print(f'  ' + '-' * 50)
    fp = frozen.get('prec_at_k', 0)
    fc = frozen.get('cycle_at_k', 0)
    fm = frozen.get('map_at_k', 0)
    print(f'  {\"frozen\":>10s} {fp:>7.1f}% {fc:>7.1f}% {fm:>8.4f} {\"—\":>8s}')

    for lam_val, lam_dir in lambdas:
        mpath = f'{out}/m09_{lam_dir}/m06_metrics_adapted.json'
        if os.path.exists(mpath):
            m = json.load(open(mpath))['easy']
            ap = m.get('prec_at_k', 0)
            ac = m.get('cycle_at_k', 0)
            am = m.get('map_at_k', 0)
            dp = ap - fp
            sign = '+' if dp >= 0 else ''
            print(f'  {lam_val:>10s} {ap:>7.1f}% {ac:>7.1f}% {am:>8.4f} {sign}{dp:>7.1f}%')
            ok += 1
        else:
            print(f'  {lam_val:>10s} MISSING')
            fail += 1

print()
print(f'=== TOTAL: {ok} OK, {fail} MISSING ===')
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
echo "  Ch10 PIPELINE COMPLETE" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "  Mode:       ${MODE}" | tee -a "$MASTER_LOG"
echo "  Config:     ${CONFIG}" | tee -a "$MASTER_LOG"
echo "  Total:      ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s" | tee -a "$MASTER_LOG"
echo "  Steps:      ${STEP_PASS} passed, ${STEP_FAIL} failed, ${STEP_COUNT} total" | tee -a "$MASTER_LOG"
echo "  Outputs:    ${OUT_DIR}/" | tee -a "$MASTER_LOG"
echo "  Master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

if [[ $STEP_FAIL -gt 0 ]]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "  WARNING: ${STEP_FAIL} step(s) failed. Check individual logs above." | tee -a "$MASTER_LOG"
    echo "  Training has checkpoints — safe to RE-RUN (auto-resume)." | tee -a "$MASTER_LOG"
    exit 1
fi

exit 0
