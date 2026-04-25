#!/usr/bin/env bash
# iter11 v2 paired eval — loops over N train yamls; for each, runs m05 frozen +
# m05 adapted + m06 (×2) + m08b paired bootstrap on the eval subset from the yaml.
# Frozen m05 + m06 run ONCE before the loop (shared baseline across all variants).
# Every path is read from the yaml's data: block via scripts/lib/yaml_extract.py.
# Per CLAUDE.md "No hardcoded paths" + "Shell scripts are THIN wrappers".
#
# Reference: scripts/run_paired_eval_10k.sh (frozen-share + per-variant chain pattern).
#
# USAGE:
#   ./scripts/run_eval.sh <train-yaml1> [<train-yaml2> ...]
#
# Example:
#   tmux new -s eval
#   ./scripts/run_eval.sh \
#       configs/train/explora.yaml \
#       configs/train/surgery_2stage_noDI.yaml \
#       configs/train/surgery_2stage_loud_agent.yaml \
#       configs/train/surgery_3stage_DI.yaml \
#       2>&1 | tee logs/run_eval_iter11_v2.log

# NO -e: a single variant failure must NOT abort the chain.
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "USAGE: $0 <train-yaml1> [<train-yaml2> ...]" >&2
    exit 2
fi

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

EX="scripts/lib/yaml_extract.py"
FROZEN_ENC="vjepa_2_1_frozen"
T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

# All variants must agree on (eval_subset, eval_local_data, model_config) — read
# from the FIRST yaml and verify the rest match (paired BCa requires same eval set).
FIRST_YAML="$1"
if [ ! -f "$FIRST_YAML" ]; then
    echo "FATAL: first yaml not found: $FIRST_YAML" >&2
    exit 3
fi
EVAL_SUBSET=$("$EX" "$FIRST_YAML" data.eval_subset)
EVAL_LOCAL=$("$EX" "$FIRST_YAML" data.eval_local_data)
MODEL_CFG=$("$EX" "$FIRST_YAML" data.model_config)
for yaml in "$@"; do
    [ -f "$yaml" ] || continue
    s=$("$EX" "$yaml" data.eval_subset)
    l=$("$EX" "$yaml" data.eval_local_data)
    m=$("$EX" "$yaml" data.model_config)
    if [ "$s" != "$EVAL_SUBSET" ] || [ "$l" != "$EVAL_LOCAL" ] || [ "$m" != "$MODEL_CFG" ]; then
        echo "FATAL: $yaml diverges on eval_subset/eval_local_data/model_config" >&2
        echo "  expected: subset=$EVAL_SUBSET local=$EVAL_LOCAL model=$MODEL_CFG" >&2
        echo "  got:      subset=$s local=$l model=$m" >&2
        exit 4
    fi
done
for req in "$EVAL_SUBSET" "$EVAL_LOCAL" "$MODEL_CFG"; do
    if [ ! -e "$req" ]; then
        echo "FATAL: missing eval path: $req" >&2
        exit 3
    fi
done

stamp "Shared Frozen baseline · eval_subset=$(basename "$EVAL_SUBSET")"
echo "  model:        $MODEL_CFG"
echo "  eval_subset:  $EVAL_SUBSET"
echo "  eval_local:   $EVAL_LOCAL"

stamp "Frozen m05"
python -u src/m05_vjepa_embed.py --FULL \
    --subset "$EVAL_SUBSET" \
    --model-config "$MODEL_CFG" \
    --encoder "$FROZEN_ENC" \
    --local-data "$EVAL_LOCAL" --no-wandb \
    2>&1 | tee "logs/run_eval_frozen_m05.log"

stamp "Frozen m06"
python -u src/m06_faiss_metrics.py --FULL \
    --subset "$EVAL_SUBSET" --encoder "$FROZEN_ENC" \
    --local-data "$EVAL_LOCAL" --no-wandb \
    2>&1 | tee "logs/run_eval_frozen_m06.log"

for yaml in "$@"; do
    if [ ! -f "$yaml" ]; then
        echo "❌ skipping: yaml not found: $yaml" >&2
        continue
    fi

    VARIANT_TAG="$(basename "$yaml" .yaml)"
    stamp "Variant: ${VARIANT_TAG}"

    OUT_DIR=$("$EX" "$yaml" data.output_dir)
    ADAPTED_ENC=$("$EX" "$yaml" data.adapted_encoder)
    ADAPTED_CKPT="${OUT_DIR}/student_encoder.pt"

    if [ ! -f "$ADAPTED_CKPT" ]; then
        echo "❌ ${VARIANT_TAG}: $ADAPTED_CKPT not found — run scripts/run_train.sh first" >&2
        continue
    fi

    python -u src/m05_vjepa_embed.py --FULL \
        --subset "$EVAL_SUBSET" \
        --model-config "$MODEL_CFG" \
        --model "$ADAPTED_CKPT" \
        --encoder "$ADAPTED_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m05.log"

    python -u src/m06_faiss_metrics.py --FULL \
        --subset "$EVAL_SUBSET" --encoder "$ADAPTED_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m06.log"

    python -u src/m08b_compare.py --FULL \
        --subset "$EVAL_SUBSET" \
        --encoders "${FROZEN_ENC},${ADAPTED_ENC}" \
        --no-wandb \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m08b.log"
done

DUR=$(( $(date +%s) - T0 ))
stamp "✅ run_eval chain done · total wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Per-variant artifacts: outputs/full/<variant>/  +  outputs/full/{m05,m06,m08b}_*/"
