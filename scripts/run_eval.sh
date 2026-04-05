#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Standalone Evaluation Pipeline (reusable across Ch9, Ch10, Ch11)
# Runs: m06→m06b→m07→m08 per encoder, then m08b comparison radar
#
# USAGE:
#   ./scripts/run_eval.sh --SANITY 2>&1 | tee logs/eval_sanity.log
#   ./scripts/run_eval.sh --POC 2>&1 | tee logs/eval_poc.log
#   ./scripts/run_eval.sh --FULL   2>&1 | tee logs/eval_full.log
#   ./scripts/run_eval.sh --FULL --encoders vjepa,dinov2   # specific subset
#
# PREREQUISITES:
#   - Embeddings must exist (from run_frozen.sh / run_pretrain.sh / run_surgery.sh)
#   - tags.json must exist (from Ch9 m04)
#   - motion_features.npy for m06b temporal metrics (from Ch9 m04d)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse args ────────────────────────────────────────────────────────
MODE=""
MODE_FLAG=""
SUBSET_FLAG=""
ENCODERS=""

usage() {
    echo "Usage: $0 --SANITY | --POC | --FULL [--encoders enc1,enc2,...]"
    echo ""
    echo "  --SANITY    Evaluate sanity outputs"
    echo "  --POC       Evaluate POC 10K outputs"
    echo "  --FULL      Evaluate full 115K outputs"
    echo "  --encoders  Comma-separated encoder list (default: auto-detect from embeddings)"
    echo ""
    echo "Example:"
    echo "  $0 --POC                                     # all available encoders in outputs/poc"
    echo "  $0 --FULL --encoders vjepa,vjepa_lambda0_001 # specific encoders"
    exit 1
}

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --SANITY) MODE="SANITY"; MODE_FLAG="--SANITY"; OUT_DIR="outputs/sanity"; shift ;;
        --POC)    MODE="POC";    MODE_FLAG="--POC";    OUT_DIR="outputs/poc";    shift ;;
        --FULL)   MODE="FULL";   MODE_FLAG="--FULL";   OUT_DIR="outputs/full";   shift ;;
        --encoders) ENCODERS="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$MODE" ]] && usage

# ── Setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ -d "venv_walkindia" ]]; then
    source venv_walkindia/bin/activate
else
    echo "ERROR: venv_walkindia not found. Run: ./setup_env_uv.sh --gpu"
    exit 1
fi

# ── Shared helpers ────────────────────────────────────────────────────
source "$(dirname "$0")/lib/common.sh"

LOGDIR="logs"
MASTER_LOG="$LOGDIR/eval_${MODE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOGDIR"

start_watchdog

# ── Auto-detect encoders from available embeddings ────────────────────
if [[ -z "$ENCODERS" ]]; then
    ENCODERS=$(python -c "
from pathlib import Path
out = Path('${OUT_DIR}')
found = []
for f in sorted(out.glob('embeddings*.npy')):
    if f.name.endswith('.paths.npy'):
        continue
    if f.name == 'embeddings.npy':
        found.append('vjepa')
    else:
        enc = f.name.replace('embeddings_', '').replace('.npy', '')
        found.append(enc)
print(','.join(found))
")
    if [[ -z "$ENCODERS" ]]; then
        log "FATAL: No embeddings found in ${OUT_DIR}/. Run embedding scripts first."
        exit 1
    fi
    log "Auto-detected encoders: ${ENCODERS}"
else
    log "Using specified encoders: ${ENCODERS}"
fi

# ── Pre-checks ────────────────────────────────────────────────────────
log "=== EVALUATION PIPELINE (mode=${MODE}) ==="
log "Output dir: ${OUT_DIR}"
log "Encoders:   ${ENCODERS}"
log "Master log: ${MASTER_LOG}"

# Check tags exist (needed for m06 metrics)
TAGS_FILE="${OUT_DIR}/tags.json"
if [[ ! -f "$TAGS_FILE" ]]; then
    log "FATAL: ${TAGS_FILE} not found. Run Ch9 m04 (VLM tagging) first."
    exit 1
fi
TAG_COUNT=$(python -c "import json; print(len(json.load(open('${TAGS_FILE}'))))")
log "Tags: ${TAG_COUNT} clips in ${TAGS_FILE}"

# ── Fresh vs cached ────────────────────────────────────────────────────
EVAL_FILES=$(find "${OUT_DIR}" -maxdepth 1 \( -name "m06_*" -o -name "m06b_*" -o -name "m08_*" -o -name "m08b_*" -o -name "knn_indices_*" -o -name "umap_2d_*" \) 2>/dev/null | wc -l)
if [[ $EVAL_FILES -gt 0 ]]; then
    echo ""
    echo "  Found ${EVAL_FILES} existing eval outputs in ${OUT_DIR}/"
    echo ""
    echo "  [1] Delete all eval outputs, start fresh (use after model retrain)"
    echo "  [2] Use cache, skip existing (use for resume after crash)"
    echo ""
    read -p "  Enter choice [1/2]: " EVAL_CHOICE
    if [[ "$EVAL_CHOICE" == "1" ]]; then
        rm -f "${OUT_DIR}"/m06_* "${OUT_DIR}"/m06b_* "${OUT_DIR}"/m07_* "${OUT_DIR}"/m08_* "${OUT_DIR}"/m08b_* "${OUT_DIR}"/knn_indices* "${OUT_DIR}"/umap_2d*
        log "Deleted all eval outputs. Starting fresh."
    else
        log "Using cached eval outputs where available."
    fi
fi

# ── Run eval_suite ────────────────────────────────────────────────────
log "Starting evaluation for $(echo "$ENCODERS" | tr ',' '\n' | wc -l) encoders..."

run_step "eval" "eval suite ($(echo "$ENCODERS" | tr ',' ' '))" "~30-120 min" \
    "$LOGDIR/eval_suite_${MODE,,}.log" \
    src/utils/eval_suite.py \
        --encoders "$ENCODERS" \
        --compare-encoders "$ENCODERS" \
        $MODE_FLAG \
        --log-dir "$LOGDIR" --log-prefix "eval_${MODE,,}" \
    || { log "WARNING: eval suite had failures. Check $LOGDIR/eval_suite_${MODE,,}.log"; }

bg_upload

# ── Summary + cleanup ─────────────────────────────────────────────────
finalize "Eval"
