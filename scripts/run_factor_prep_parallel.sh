#!/usr/bin/env bash
# iter13 v13 FIX-26 (2026-05-07): parallel m10 orchestrator + chained m11 streaming.
# Mirror of run_factor_prep.sh but with N parallel m10 workers instead of 1 serial.
#
# Estimated speedup at FULL 10K on RTX Pro 6000 Blackwell:
#   2 workers: ~1.5x   |   4 workers: ~2x   |   6 workers: ~2.4x (CPU saturates at 32 cores)
#
# Each m10 worker loads its own DINO (~500 MB GPU MEM) + SAM3 (~3.5 GB) and processes
# a disjoint slice of clips into a private output dir m10_sam_segment_w{i}/. After
# all workers finish, merge step unions segments.json + masks/ into the canonical
# m10_sam_segment/ dir. Then m11 streaming runs ONCE on the merged canonical dir
# (verify-100 cap → ~1 min wall regardless of FULL scale).
#
# USAGE:
#   ./scripts/run_factor_prep_parallel.sh <factor-yaml> [n-workers] [--SANITY|--POC|--FULL]
#
# Examples:
#   ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml 2 --SANITY
#   ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml 4 --FULL
#   N_WORKERS=6 ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml
#
# Env-var overrides (same as run_factor_prep.sh):
#   LOCAL_DATA       Override hardcoded data/eval_10k_local
#   CACHE_POLICY_ALL 1=keep / 2=recompute. Per-worker scratch dirs are empty so
#                    cache-policy=2 is effectively a no-op for workers (canonical
#                    dir is NEVER wiped — its existing segments.json drives the
#                    "already-done" filter in m10_split_subset.py).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "USAGE: $0 <factor-yaml> [n-workers]" >&2
    echo "  Example: $0 configs/train/surgery_3stage_DI_encoder.yaml 4" >&2
    exit 2
fi

FACTOR_YAML="$1"
N_WORKERS="${2:-${N_WORKERS:-4}}"
MODE_FLAG="${3:---FULL}"

if [ ! -f "$FACTOR_YAML" ]; then
    echo "FATAL: factor yaml not found: $FACTOR_YAML" >&2
    exit 3
fi
if ! [[ "$N_WORKERS" =~ ^[0-9]+$ ]] || [ "$N_WORKERS" -lt 1 ]; then
    echo "FATAL: n-workers must be a positive integer (got: $N_WORKERS)" >&2
    exit 2
fi
case "$MODE_FLAG" in
    --SANITY|--sanity) MODE="SANITY" ;;
    --POC|--poc)       MODE="POC" ;;
    --FULL|--full)     MODE="FULL" ;;
    *) echo "FATAL: mode flag must be --SANITY|--POC|--FULL (got: $MODE_FLAG)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

LOCAL_DATA="${LOCAL_DATA:-data/eval_10k_local}"
CACHE_POLICY="${CACHE_POLICY_ALL:-2}"

if [ ! -d "$LOCAL_DATA" ]; then
    echo "FATAL: LOCAL_DATA=$LOCAL_DATA is not a directory." >&2
    exit 3
fi
if [ ! -f "$LOCAL_DATA/manifest.json" ]; then
    echo "FATAL: $LOCAL_DATA/manifest.json missing — needed for clip key universe." >&2
    exit 3
fi

CANONICAL_DIR="${LOCAL_DATA}/m10_sam_segment"
SUBSET_DIR="/tmp/m10_parallel_subsets_$$"
mkdir -p "$SUBSET_DIR"

VARIANT_TAG="$(basename "$FACTOR_YAML" .yaml)"
T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

echo "──────────────────────────────────────────────"
echo "factor-prep parallel · mode=${MODE} · N_WORKERS=${N_WORKERS} · variant=${VARIANT_TAG}"
echo "  LOCAL_DATA:    $LOCAL_DATA"
echo "  CANONICAL_DIR: $CANONICAL_DIR"
echo "  SUBSET_DIR:    $SUBSET_DIR (auto-cleaned at exit)"
echo "  CACHE_POLICY:  $CACHE_POLICY (per-worker scratch — does not touch canonical)"
echo "──────────────────────────────────────────────"

stamp "Step 1/5 — split clips into ${N_WORKERS} disjoint subsets"
python -u src/utils/m10_split_subset.py \
    --manifest "${LOCAL_DATA}/manifest.json" \
    --existing-segments "${CANONICAL_DIR}/segments.json" \
    --n-workers "$N_WORKERS" \
    --out-dir "$SUBSET_DIR"

stamp "Step 2/5 — spawn ${N_WORKERS} m10 workers (parallel)"
PIDS=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    SUBSET_JSON="${SUBSET_DIR}/subset_w${i}.json"
    OUTPUT_DIR="${LOCAL_DATA}/m10_sam_segment_w${i}"
    LOG="logs/factor_full_${VARIANT_TAG}_w${i}.log"

    echo "  worker $i:"
    echo "    subset:  $SUBSET_JSON"
    echo "    output:  $OUTPUT_DIR"
    echo "    log:     $LOG"

    CACHE_POLICY_ALL="$CACHE_POLICY" \
        python -u src/m10_sam_segment.py "$MODE_FLAG" \
        --train-config "$FACTOR_YAML" \
        --subset "$SUBSET_JSON" \
        --local-data "$LOCAL_DATA" \
        --output-dir "$OUTPUT_DIR" \
        --no-wandb \
        --cache-policy "$CACHE_POLICY" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done
echo "  spawned PIDs: ${PIDS[*]}"
echo "  monitor:  tail -f logs/factor_full_${VARIANT_TAG}_w*.log"
echo "  GPU watch: nvtop  (or: watch -n2 nvidia-smi)"

stamp "Step 3/5 — wait for all ${N_WORKERS} workers"
EXIT_CODES=()
FAILED_WORKERS=()
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    if wait "$pid"; then
        ec=0
    else
        ec=$?
        FAILED_WORKERS+=("$idx")
    fi
    EXIT_CODES+=("$ec")
    echo "  worker $idx (pid $pid): exit=$ec"
done
echo "  All workers done. Exit codes: ${EXIT_CODES[*]}"
if [ ${#FAILED_WORKERS[@]} -gt 0 ]; then
    echo "  WARN: ${#FAILED_WORKERS[@]} worker(s) exited non-zero: ${FAILED_WORKERS[*]}"
    echo "    (m10's quality_gate FAIL also exits non-zero — check log; "
    echo "     segments.json is saved BEFORE the gate so merge still has the data)"
fi

stamp "Step 4/5 — merge worker outputs into ${CANONICAL_DIR}"
WORKER_DIRS=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    WORKER_DIRS+=("${LOCAL_DATA}/m10_sam_segment_w${i}")
done
python -u src/utils/m10_merge.py \
    --canonical-dir "$CANONICAL_DIR" \
    --worker-dirs "${WORKER_DIRS[@]}"

rm -rf "$SUBSET_DIR"

stamp "Step 5/5 — m11 --streaming (factor materialization on merged canonical)"
M11_LOG="logs/run_factor_prep_parallel_${VARIANT_TAG}_m11.log"
echo "  log: $M11_LOG"
# m11 streaming reads merged segments.json + masks/ from canonical dir, materializes
# the verify-100 cap subset into D_L/D_A/D_I + factor_manifest. Wall ~1 min at FULL.
# Pass --cache-policy=2 explicitly so prior m11 outputs (POC v4 leftovers, .npy files)
# are wiped — keeps the bundle clean for hf_outputs.py upload-data later.
CACHE_POLICY_ALL="$CACHE_POLICY" \
    python -u src/m11_factor_datasets.py "$MODE_FLAG" --streaming \
    --train-config "$FACTOR_YAML" \
    --local-data "$LOCAL_DATA" \
    --no-wandb \
    --cache-policy "$CACHE_POLICY" \
    2>&1 | tee "$M11_LOG"

DUR=$(( $(date +%s) - T0 ))
stamp "✅ factor-prep parallel done · wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Outputs (co-located inside ${LOCAL_DATA} — raw per-clip files only;"
echo "         tar shards + HF push are produced by 'hf_outputs.py upload-data'):"
echo "  ${CANONICAL_DIR}/  (segments.json + summary.json + masks/*.npz)"
echo "  ${LOCAL_DATA}/m11_factor_datasets/  (factor_manifest.json + D_L,D_A,D_I/*.npy + verify samples)"
echo
echo "Optional follow-ups:"
echo "  · regenerate m10 plots from merged segments.json (CPU only, ~2 min):"
echo "      python -u src/m10_sam_segment.py --plot --train-config ${FACTOR_YAML} \\"
echo "          --local-data ${LOCAL_DATA} --no-wandb"
echo "  · once verified, free disk by removing worker scratch dirs:"
for wd in "${WORKER_DIRS[@]}"; do
    echo "      rm -rf ${wd}"
done
