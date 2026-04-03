#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch10: V-JEPA Continual Pretraining (FactorJEPA)
# Pipeline: m09(train) → m05(re-embed) → m06 → m06b → m05(shuffled) → m07 → m08 → m08b
#
# USAGE:
#   ./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log  # ~5 min
#   ./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log      # ~67h (115K, 1 epoch)
#
# CACHE CONTROL (output_guard skips completed steps automatically):
#   Use all cached:    ./scripts/run_pretrain.sh --FULL
#   Re-run everything: rm -rf outputs/full/m09_* && ./scripts/run_pretrain.sh --FULL
#   Re-run training:   rm outputs/full/m09_lambda*/student_encoder.pt && ...
#   Re-run re-embed:   rm outputs/full/embeddings_vjepa_lambda*.npy && ...
#
# PREREQUISITES:
#   1. ./setup_env_uv.sh --gpu --from-wheels
#   2. rsync -avhP data/ <gpu-host>:/workspace/factorjepa/data/   # from Mac (~17 min)
#   3. data/full_local/ exists (from m00d or rsync)
#   4. data/val_1k_local/ exists (for validation during training)
#   5. Ch9 complete: outputs/full/m06_metrics.json (needed for m08b comparison)
#   6. tmux new -s ch10   # run inside tmux for long runs
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# Reduce CUDA memory fragmentation (recovers ~10-14GB on long training runs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
SUBSET_FLAG=""
CONFIG="configs/pretrain/vitg16_indian.yaml"

usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL [--config CONFIG]"
    echo ""
    echo "  --SANITY   Quick validation (~5 min, 900 train clips from config)"
    echo "  --POC      10K subset: 1-epoch ablation + 5-epoch winner (~5h)"
    echo "  --FULL     115K full corpus: no subset, dataset size from manifest (~28h)"
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
        --POC)
            MODE="POC"
            MODE_FLAG="--FULL"
            SUBSET_FLAG="--subset data/subset_10k.json"
            OUT_DIR="outputs/poc"
            shift
            ;;
        --FULL)
            MODE="FULL"
            MODE_FLAG="--FULL"
            SUBSET_FLAG=""
            OUT_DIR="outputs/full"
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

# ── GPU watchdog (background, emails alert if GPU < 50% for 5 min) ───
WATCHDOG_PID=""
if [[ -f "scripts/gpu_watchdog.py" ]]; then
    python scripts/gpu_watchdog.py &
    WATCHDOG_PID=$!
    log "GPU watchdog started (PID=$WATCHDOG_PID)"
fi

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

# Background HF upload — runs between steps, zero GPU idle time
UPLOAD_PID=""
bg_upload() {
    if [[ -n "$UPLOAD_PID" ]]; then
        wait "$UPLOAD_PID" 2>/dev/null
        UPLOAD_PID=""
    fi
    python -u src/utils/hf_outputs.py upload "$OUT_DIR" >> "$LOGDIR/hf_upload.log" 2>&1 &
    UPLOAD_PID=$!
    log "HF upload started in background (PID=$UPLOAD_PID)"
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
        bg_upload
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
    T_M09="~3 min (8 steps)"
    T_M05_RE="~30 sec"
    T_M06_RE="~10 sec"
    T_M08B_RE="~5 sec"
elif [[ "$MODE" == "POC" ]]; then
    T_M09="~35 min/lambda (80 steps)"
    T_M05_RE="~1h 47min (10K clips at 1.55 clips/s)"
    T_M06_RE="~1 min"
    T_M08B_RE="~30 sec"
else  # FULL
    T_M09="~6.4h/lambda (929 steps)"
    T_M05_RE="~20h (115K clips at 1.55 clips/s)"
    T_M06_RE="~5 min"
    T_M08B_RE="~30 sec"
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

# ── Local data (mode-dependent) ──────────────────────────────────────
if [[ "$MODE" == "FULL" ]]; then
    LOCAL_DATA="data/full_local"
else
    LOCAL_DATA="data/subset_10k_local"
fi
LOCAL_FLAG=""

if [[ -d "$LOCAL_DATA" && -f "$LOCAL_DATA/manifest.json" ]]; then
    log "Local data exists: $LOCAL_DATA"
    LOCAL_FLAG="--local-data $LOCAL_DATA"
elif [[ "$MODE" == "SANITY" ]]; then
    # SANITY can stream (small clip count)
    log "No local data — SANITY will use HF streaming"
else
    log "FATAL: Local data not found at $LOCAL_DATA"
    if [[ "$MODE" == "FULL" ]]; then
        log "Run: python -u src/m00d_download_subset.py --FULL"
    else
        log "Run: python -u src/m00d_download_subset.py --FULL --subset data/subset_10k.json"
    fi
    exit 1
fi

# ── Val subset + local data (pre-generated by m00c + m00d) ────────────
VAL_SUBSET="data/val_1k.json"
VAL_LOCAL="data/val_1k_local"
VAL_FLAG=""

if [[ "$MODE" != "SANITY" ]]; then
    if [[ ! -f "$VAL_SUBSET" ]]; then
        log "FATAL: $VAL_SUBSET not found. Generate on Mac first:"
        log "  python -u src/m00c_sample_subset.py --POC --n 1000 --seed 99 --output $VAL_SUBSET"
        exit 1
    fi
    VAL_FLAG="--val-subset $VAL_SUBSET"
    if [[ -d "$VAL_LOCAL" && -f "$VAL_LOCAL/manifest.json" ]]; then
        VAL_FLAG="$VAL_FLAG --val-local-data $VAL_LOCAL"
        log "Val data: $VAL_SUBSET + $VAL_LOCAL"
    else
        log "WARNING: $VAL_LOCAL not found. Val clips will load from training --local-data (may miss some)."
        log "  Fix: python -u src/m00d_download_subset.py --FULL --subset $VAL_SUBSET  (on Mac, ~57 min)"
        log "Val data: $VAL_SUBSET (no local shards)"
    fi
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

# Ch9 baseline (needed for m08b comparison only — NOT for training)
baseline = '${OUT_DIR}/m06_metrics.json'
if os.path.exists(baseline):
    import json
    m = json.load(open(baseline))
    print(f'Baseline:   Prec@K={m[\"easy\"][\"prec_at_k\"]:.1f}% (Ch9 frozen)')
else:
    print(f'Baseline:   NOT FOUND ({baseline}) — m08b comparison will fail, training will proceed')
    print(f'  Fix after Ch9: python -u src/utils/hf_outputs.py download outputs/full')

if errors:
    print(f'\nFATAL: {len(errors)} check(s) failed:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('\nPre-flight: ALL PASSED')
" 2>&1 | tee -a "$MASTER_LOG"

log "Pre-flight complete."

# ── Output preflight: check all inputs/outputs before GPU work ────────
log "=== OUTPUT PREFLIGHT ==="
python -u src/utils/output_guard.py preflight_pretrain "$OUT_DIR" "$CONFIG" 2>&1 | tee -a "$MASTER_LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    log "FATAL: Output preflight failed or aborted."
    exit 1
fi

log "Starting pipeline..."
echo "" | tee -a "$MASTER_LOG"

# ═══════════════════════════════════════════════════════════════════════
# TRAINING (λ=0.001, POC winner — no ablation sweep)
# Lambda ablation was done on 10K POC (all 4 lambdas identical).
# For FULL, train directly with winner λ=0.001.
# ═══════════════════════════════════════════════════════════════════════

# Read epochs from YAML config (mode-aware: SANITY/POC use their key, FULL uses 'full')
if [[ "$MODE" == "SANITY" ]]; then
    WINNER_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['optimization']['max_epochs']['sanity'])")
elif [[ "$MODE" == "POC" ]]; then
    WINNER_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['optimization']['max_epochs']['poc'])")
else
    WINNER_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['optimization']['max_epochs']['full'])")
fi

# Read winner lambda from POC ablation (selected by lowest best_val_loss)
# Search: outputs/full/ first (copied), then outputs/poc/ (original from ablation)
WINNER_JSON=""
for _dir in "${OUT_DIR}" "outputs/poc"; do
    if [[ -f "${_dir}/ablation_winner.json" ]]; then
        WINNER_JSON="${_dir}/ablation_winner.json"
        break
    fi
done

if [[ -n "$WINNER_JSON" ]]; then
    WINNER_LAMBDA=$(python -c "import json; print(json.load(open('${WINNER_JSON}'))['winner_lambda'])")
    WINNER_DIR=$(python -c "import json; print(json.load(open('${WINNER_JSON}'))['winner_dir'])")
    log "Winner from ablation: lambda=${WINNER_LAMBDA} (${WINNER_DIR}) [from ${WINNER_JSON}]"
else
    log "FATAL: ablation_winner.json not found in ${OUT_DIR}/ or outputs/poc/"
    log "  Run POC lambda ablation first (Ch10-2) to select winner by best_val_loss."
    exit 1
fi
WINNER_OUT="${OUT_DIR}/m09_${WINNER_DIR}"

log "Training: lambda=${WINNER_LAMBDA} (${WINNER_DIR}), ${WINNER_EPOCHS} epochs"

# Check if already trained with sufficient epochs
if [[ -f "${WINNER_OUT}/student_encoder.pt" && -f "${WINNER_OUT}/training_summary.json" ]]; then
    EXISTING_EPOCHS=$(python -c "import json; s=json.load(open('${WINNER_OUT}/training_summary.json')); print(s['epochs'])")
    if python -c "exit(0 if float('${EXISTING_EPOCHS}') >= float('${WINNER_EPOCHS}') else 1)"; then
        log "Student already trained for ${EXISTING_EPOCHS} epochs (>= ${WINNER_EPOCHS}). Skipping."
    else
        log "Student has ${EXISTING_EPOCHS} epochs, need ${WINNER_EPOCHS}. Re-training."
        rm -f "${WINNER_OUT}/student_encoder.pt"
    fi
fi

run_step "train" "m09 pretrain (lambda=${WINNER_LAMBDA}, ${WINNER_EPOCHS} epochs)" "$T_M09" \
    "$LOGDIR/m09_${MODE,,}_${WINNER_DIR}.log" \
    src/m09_pretrain.py --config "$CONFIG" --lambda-reg "$WINNER_LAMBDA" \
        --max-epochs "$WINNER_EPOCHS" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG \
    || { log "FATAL: m09 training failed."; exit 1; }

verify "Training" "
import os, json
out = '${WINNER_OUT}'
for f in ['student_encoder.pt', 'training_summary.json', 'loss_log.csv']:
    path = os.path.join(out, f)
    if os.path.exists(path):
        if f.endswith('.json'):
            s = json.load(open(path))
            print(f'  OK   {f:30s} jepa_loss={s[\"final_jepa_loss\"]:.4f}  epochs={s[\"epochs\"]}')
        elif f.endswith('.pt'):
            size_mb = os.path.getsize(path) / 1e6
            print(f'  OK   {f:30s} {size_mb:.0f} MB')
        else:
            lines = sum(1 for _ in open(path)) - 1
            print(f'  OK   {f:30s} {lines} steps')
    else:
        print(f'  MISS {f}')
"

# Deep train the winner
run_step "winner-train" "m09 deep train (lambda=${WINNER_LAMBDA}, ${WINNER_EPOCHS} epochs)" \
    "~$((WINNER_EPOCHS * 32))min" \
    "$LOGDIR/m09_${MODE,,}_winner.log" \
    src/m09_pretrain.py --config "$CONFIG" --lambda-reg "$WINNER_LAMBDA" \
        --max-epochs "$WINNER_EPOCHS" \
        $BATCH_FLAG $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG $VAL_FLAG \
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

# m05+m06 shuffled adapted: temporal ablation (~1h 50min)
SHUFFLED_ENCODER="${WINNER_ENCODER}_shuffled"
run_step "winner-shuffled" "m05 shuffled adapted (temporal ablation)" "$T_M05_RE" \
    "$LOGDIR/m05_${MODE,,}_winner_shuffled.log" \
    src/m05_vjepa_embed.py --model "$WINNER_MODEL" --encoder "$SHUFFLED_ENCODER" --shuffle \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG \
    || { log "FATAL: Shuffled adapted embed failed."; exit 1; }

run_step "winner-shuffled-metrics" "m06 metrics (shuffled adapted)" "$T_M06_RE" \
    "$LOGDIR/m06_${MODE,,}_winner_shuffled.log" \
    src/m06_faiss_metrics.py --encoder "$SHUFFLED_ENCODER" \
        $MODE_FLAG $SUBSET_FLAG \
    || { log "FATAL: Shuffled adapted metrics failed."; exit 1; }

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
COMPARE_LIST="vjepa,random,dinov2,clip,vjepa_shuffled,${WINNER_ENCODER},${SHUFFLED_ENCODER}"
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

# Wait for final background upload
if [[ -n "$UPLOAD_PID" ]]; then
    log "Waiting for final HF upload..."
    wait "$UPLOAD_PID" 2>/dev/null
    log "Final HF upload complete"
fi

# Kill watchdog
if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null
    log "GPU watchdog stopped"
fi

if [[ $STEP_FAIL -gt 0 ]]; then
    echo "" | tee -a "$MASTER_LOG"
    echo "  FATAL: ${STEP_FAIL} step(s) failed. Check individual logs above." | tee -a "$MASTER_LOG"
    exit 1
fi

exit 0
