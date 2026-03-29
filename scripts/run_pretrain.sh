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

# Reduce CUDA memory fragmentation (recovers ~10-14GB on long training runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
            shift
            ;;
        --FULL)
            MODE="FULL"
            MODE_FLAG="--FULL"
            SUBSET_FLAG="--subset data/subset_10k.json"
            OUT_DIR="outputs/poc"
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

# Read MIN_CLIPS from configs/pipeline.yaml (no hardcoding in shell)
PIPELINE_YAML="configs/pipeline.yaml"
if [[ "$MODE" == "SANITY" ]]; then
    MIN_CLIPS=$(python -c "import yaml; print(yaml.safe_load(open('${PIPELINE_YAML}'))['verify']['sanity_min_clips'])")
else
    MIN_CLIPS=$(python -c "import yaml; print(yaml.safe_load(open('${PIPELINE_YAML}'))['verify']['full_min_clips'])")
fi

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

    python -u "${cmd[@]}" 2>&1 | tee "$log_file" | tee -a "$MASTER_LOG"
    local exit_code=${PIPESTATUS[0]}  # capture Python's exit code (not tee's)

    local step_end=$(date +%s)
    local elapsed=$(( step_end - step_start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [[ $exit_code -eq 0 ]]; then
        log "PASSED: ${step_name} (${mins}m ${secs}s)"
        STEP_PASS=$((STEP_PASS + 1))
        return 0
    else
        log "FAILED: ${step_name} (${mins}m ${secs}s) — exit code ${exit_code}"
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
        log "FATAL: VERIFY FAIL: $desc"
        exit 1
    fi
}

# ── Time estimates ────────────────────────────────────────────────────
if [[ "$MODE" == "SANITY" ]]; then
    T_M09="~2 min (50 steps, profiler BS)"
    T_M05_RE="~15 sec (cached)"
    T_M06_RE="~10 sec"
    T_M08B_RE="~5 sec"
else
    T_M09="~5-20h (2000-10000 steps, profiler BS)"
    T_M05_RE="~80 min (re-embed 10K clips with adapted encoder)"
    T_M06_RE="~30 sec (FAISS-GPU metrics)"
    T_M08B_RE="~30 sec (frozen vs adapted comparison)"
fi

# ── Auto-detect optimal batch size from profiler ─────────────────────
PROFILE_JSON="outputs/profile/profile_data.json"
BATCH_FLAG=""

if [[ ! -f "$PROFILE_JSON" ]]; then
    log "No profiler data found. Running VRAM profiler (~5 min)..."
    python -u scripts/profile_vram.py 2>&1 | tee "$LOGDIR/profile_vram.log" | tee -a "$MASTER_LOG"
    if [[ ! -f "$PROFILE_JSON" ]]; then
        log "FATAL: Profiler failed. Cannot determine optimal batch size."
        exit 1
    fi
fi

if [[ -f "$PROFILE_JSON" ]]; then
    # Read max BS at ≤75% VRAM (conservative: stop at first violation)
    # Profiler measures isolated peak; real training adds ~10GB overhead
    # from memory fragmentation, prefetch queue, wandb, optimizer state
    OPTIMAL_BS=$(python -c "
import json
d = json.load(open('${PROFILE_JSON}'))
gpu_gb = d['gpu_total_gb']
target = gpu_gb * 0.75
best = 4
for bs, info in sorted(d.get('grad_ckpt', {}).items(), key=lambda x: int(x[0])):
    if info['peak_gb'] > target:
        break  # stop at first violation (handles non-monotonic VRAM)
    best = int(bs)
print(best)
")
    BATCH_FLAG="--batch-size $OPTIMAL_BS"
    log "Auto batch size: ${OPTIMAL_BS} (from profiler, <=75% VRAM)"
fi

# ── Local data ────────────────────────────────────────────────────────
LOCAL_DATA="data/subset_10k_local"
LOCAL_FLAG=""

if [[ -d "$LOCAL_DATA" && -f "$LOCAL_DATA/manifest.json" ]]; then
    log "Local subset exists: $LOCAL_DATA"
    LOCAL_FLAG="--local-data $LOCAL_DATA"
else
    log "FATAL: Local data not found at $LOCAL_DATA"
    log "Run: python -u src/m00d_download_subset.py --FULL --subset data/subset_10k.json"
    exit 1
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
    errors.append(f'Ch9 baseline not found ({baseline}). Run: ./scripts/run_evaluate.sh --FULL')

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

    # ── Phase 1: Train only (no m05/m06 — select winner by jepa_loss) ──
    run_step "${LAM_DIR}-train" "m09 pretrain (lambda=${LAM})" "$T_M09" \
        "$LOGDIR/m09_${MODE,,}_${LAM_DIR}.log" \
        src/m09_pretrain.py --config "$CONFIG" --lambda-reg "$LAM" \
            $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
        || { log "FATAL: m09 lambda=${LAM} failed. Stopping pipeline."; exit 1; }

    verify "Training lambda=${LAM}" "
import os, json
out = '${LAM_OUT}'
for f in ['student_encoder.pt', 'training_summary.json', 'loss_log.csv']:
    path = os.path.join(out, f)
    if os.path.exists(path):
        if f.endswith('.json'):
            s = json.load(open(path))
            print(f'  OK   {f:30s} jepa_loss={s[\"final_jepa_loss\"]:.4f}')
        elif f.endswith('.pt'):
            size_mb = os.path.getsize(path) / 1e6
            print(f'  OK   {f:30s} {size_mb:.0f} MB')
        else:
            lines = sum(1 for _ in open(path)) - 1
            print(f'  OK   {f:30s} {lines} steps')
    else:
        print(f'  MISS {f}')
"

    COMPLETED_LAMBDAS+=("$LAM_DIR")
    log "Completed training: lambda=${LAM} (${LAM_DIR})"
done

# ═══════════════════════════════════════════════════════════════════════
# WINNER SELECTION + DEEP RUN
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== SELECTING ABLATION WINNER ==="

# Read winner_epochs from YAML
WINNER_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['optimization']['max_epochs']['winner'])")

# Pick the lambda with lowest jepa_loss — reads training_summary.json (no m05/m06 needed)
WINNER_JSON="${OUT_DIR}/ablation_winner.json"
python -c "
import json, os, sys

out = '${OUT_DIR}'
lambdas = [('0', 'lambda0'), ('0.001', 'lambda0_001'),
           ('0.01', 'lambda0_01'), ('0.1', 'lambda0_1')]

best_loss = float('inf')
best_lambda = None
best_dir = None
results = {}

for lam_val, lam_dir in lambdas:
    spath = f'{out}/m09_{lam_dir}/training_summary.json'
    if not os.path.exists(spath):
        print(f'FATAL: lambda={lam_val} training_summary.json not found: {spath}')
        sys.exit(1)
    s = json.load(open(spath))
    jepa_loss = s['final_jepa_loss']
    print(f'  lambda={lam_val}: jepa_loss={jepa_loss:.4f} (steps={s[\"steps\"]}, epochs={s[\"epochs\"]})')
    results[lam_val] = jepa_loss
    if jepa_loss < best_loss:
        best_loss = jepa_loss
        best_lambda = lam_val
        best_dir = lam_dir

if best_lambda is None:
    print('FATAL: No lambda summaries found')
    sys.exit(1)

winner = {
    'winner_lambda': best_lambda,
    'winner_dir': best_dir,
    'winner_jepa_loss': best_loss,
    'selection_metric': 'final_jepa_loss (lowest)',
    'all_results': results,
}
with open('${WINNER_JSON}', 'w') as f:
    json.dump(winner, f, indent=2)
print(f'  WINNER: lambda={best_lambda} (jepa_loss={best_loss:.4f})')
print(f'  Saved: ${WINNER_JSON}')
" 2>&1 | tee -a "$MASTER_LOG"

if [[ ! -f "$WINNER_JSON" ]]; then
    log "FATAL: Winner selection failed — ${WINNER_JSON} not created."
    exit 1
fi

# Read winner from JSON (not stdout)
WINNER_LAMBDA=$(python -c "import json; print(json.load(open('${WINNER_JSON}'))['winner_lambda'])")

WINNER_DIR="lambda$(echo "$WINNER_LAMBDA" | sed 's/\./_/g')"
log "Winner: lambda=${WINNER_LAMBDA} (${WINNER_DIR}), deep run: ${WINNER_EPOCHS} epochs"

# Delete the 1-epoch student_encoder so m09 re-trains with more epochs
WINNER_OUT="${OUT_DIR}/m09_${WINNER_DIR}"
if [[ -f "${WINNER_OUT}/student_encoder.pt" ]]; then
    rm -f "${WINNER_OUT}/student_encoder.pt"
    log "Removed 1-epoch student for re-training"
fi

# Deep train the winner
run_step "winner-train" "m09 deep train (lambda=${WINNER_LAMBDA}, ${WINNER_EPOCHS} epochs)" \
    "~$((WINNER_EPOCHS * 32))min" \
    "$LOGDIR/m09_${MODE,,}_winner.log" \
    src/m09_pretrain.py --config "$CONFIG" --lambda-reg "$WINNER_LAMBDA" \
        --max-epochs "$WINNER_EPOCHS" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
    || { log "FATAL: Winner deep train failed."; exit 1; }

# Re-embed with deep-trained encoder (same encoder name, overwrites 1-epoch embeddings)
WINNER_MODEL="${WINNER_OUT}/student_encoder.pt"
WINNER_ENCODER="vjepa_${WINNER_DIR}"
run_step "winner-embed" "m05 re-embed (winner lambda=${WINNER_LAMBDA})" "$T_M05_RE" \
    "$LOGDIR/m05_${MODE,,}_winner.log" \
    src/m05_vjepa_embed.py --model "$WINNER_MODEL" --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
    || { log "FATAL: Winner re-embed failed."; exit 1; }

# Metrics on deep-trained embeddings
run_step "winner-metrics" "m06 metrics (winner lambda=${WINNER_LAMBDA})" "$T_M06_RE" \
    "$LOGDIR/m06_${MODE,,}_winner.log" \
    src/m06_faiss_metrics.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner metrics failed."; exit 1; }

log "Winner deep run complete: lambda=${WINNER_LAMBDA}, ${WINNER_EPOCHS} epochs"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: FULL EVALUATION (m07 UMAP + m08 plots + m08b comparison)
# ═══════════════════════════════════════════════════════════════════════

echo "" | tee -a "$MASTER_LOG"
log "=== PHASE 3: FULL EVALUATION ==="

# m06b: Temporal correlation for adapted encoder (motion features from Ch9 m04d)
run_step "winner-temporal" "m06b temporal corr (winner)" "~2 min" \
    "$LOGDIR/m06b_${MODE,,}_winner.log" \
    src/m06b_temporal_corr.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner temporal correlation failed."; exit 1; }

# m05+m06 shuffled adapted: DISABLED for speed (enable for full paper run)
# Uncomment below for temporal ablation (~1h 50min)
SHUFFLED_ENCODER="${WINNER_ENCODER}_shuffled"
# run_step "winner-shuffled" "m05 shuffled adapted (temporal ablation)" "$T_M05_RE" \
#     "$LOGDIR/m05_${MODE,,}_winner_shuffled.log" \
#     src/m05_vjepa_embed.py --model "$WINNER_MODEL" --encoder "$SHUFFLED_ENCODER" --shuffle \
#         $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
#     || { log "FATAL: Shuffled adapted embed failed."; exit 1; }
# run_step "winner-shuffled-metrics" "m06 metrics (shuffled adapted)" "$T_M06_RE" \
#     "$LOGDIR/m06_${MODE,,}_winner_shuffled.log" \
#     src/m06_faiss_metrics.py --encoder "$SHUFFLED_ENCODER" \
#         $MODE_FLAG $SUBSET_FLAG \
#     || { log "FATAL: Shuffled adapted metrics failed."; exit 1; }

# m07: UMAP for adapted encoder
run_step "winner-umap" "m07 UMAP (winner)" "~5 min" \
    "$LOGDIR/m07_${MODE,,}_winner.log" \
    src/m07_umap.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner UMAP failed."; exit 1; }

# m08: Plots for adapted encoder
run_step "winner-plots" "m08 plots (winner)" "~2 min" \
    "$LOGDIR/m08_${MODE,,}_winner.log" \
    src/m08_plot.py --encoder "$WINNER_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Winner plots failed."; exit 1; }

# m08b: Compare all encoders (frozen + baselines + adapted)
# Build encoder list: all Ch9 encoders + adapted winner
COMPARE_LIST="vjepa,random,dinov2,clip,vjepa_shuffled,${WINNER_ENCODER}"
# Add shuffled adapted when enabled above:
# COMPARE_LIST="${COMPARE_LIST},${SHUFFLED_ENCODER}"
run_step "final-compare" "m08b comparison (frozen + baselines + adapted)" "~30 sec" \
    "$LOGDIR/m08b_${MODE,,}_ch10.log" \
    src/m08b_compare.py --encoders "$COMPARE_LIST" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Final comparison failed."; exit 1; }

log "Phase 3 complete: UMAP + plots + comparison for ${WINNER_ENCODER}"

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

print('=== ABLATION TRAINING (Phase 1) ===')
print(f'  {\"lambda\":>10s} {\"JEPA Loss\":>12s} {\"Steps\":>8s} {\"Student\":>10s}')
print(f'  ' + '-' * 45)
for lam_val, lam_dir in lambdas:
    lam_out = f'{out}/m09_{lam_dir}'
    spath = f'{lam_out}/training_summary.json'
    if os.path.exists(spath):
        s = json.load(open(spath))
        has_student = 'OK' if os.path.exists(f'{lam_out}/student_encoder.pt') else 'MISS'
        print(f'  {lam_val:>10s} {s[\"final_jepa_loss\"]:>12.4f} {s[\"steps\"]:>8d} {has_student:>10s}')
        ok += 1
    else:
        print(f'  {lam_val:>10s} MISSING')
        fail += 1

# Winner info
winner_path = f'{out}/ablation_winner.json'
if os.path.exists(winner_path):
    w = json.load(open(winner_path))
    print(f'  WINNER: lambda={w[\"winner_lambda\"]} (jepa_loss={w[\"winner_jepa_loss\"]:.4f})')
    ok += 1

print()
print('=== WINNER DEEP RUN (Phase 2) ===')
winner_dir = w['winner_dir'] if os.path.exists(winner_path) else None
if winner_dir:
    enc = f'vjepa_{winner_dir}'
    check(f'Winner embeddings',    f'{out}/embeddings_{enc}.npy')
    check(f'Winner metrics',       f'{out}/m06_metrics_{enc}.json')
    check(f'Winner knn_indices',   f'{out}/knn_indices_{enc}.npy')

print()
print('=== FULL EVALUATION (Phase 3) ===')
if winner_dir:
    enc = f'vjepa_{winner_dir}'
    check(f'Winner UMAP',          f'{out}/umap_2d_{enc}.npy')
    check(f'Winner UMAP plot',     f'{out}/m08_umap_{enc}.png')
    check(f'Comparison radar',     f'{out}/m08b_radar.png')
    check(f'Comparison table',     f'{out}/m08b_comparison_table.tex')

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
    echo "  FATAL: ${STEP_FAIL} step(s) failed. Check individual logs above." | tee -a "$MASTER_LOG"
    exit 1
fi

exit 0
