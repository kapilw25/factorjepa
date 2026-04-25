#!/usr/bin/env bash
# iter11 v2 trainer — loops over N train yamls; for each, dispatches m09b (ExPLoRA)
# or m09c (Surgery) based on the yaml's data.module key.
# Every path is read from the yaml's data: block via scripts/lib/yaml_extract.py.
# Per CLAUDE.md "No hardcoded paths" + "Shell scripts are THIN wrappers".
#
# Prereq: scripts/run_factor_prep.sh has been run (m10/m11 outputs present in
# outputs/full/m10_sam_segment/ and outputs/full/m11_factor_datasets/).
#
# Reference: scripts/run_paired_eval_10k.sh / scripts/run_iter9_10k.sh Step C.
#
# USAGE:
#   ./scripts/run_train.sh <train-yaml1> [<train-yaml2> ...]
#
# Example (full iter11 v2 4-variant chain in one tmux session):
#   tmux new -s train
#   ./scripts/run_train.sh \
#       configs/train/explora.yaml \
#       configs/train/surgery_2stage_noDI.yaml \
#       configs/train/surgery_2stage_loud_agent.yaml \
#       configs/train/surgery_3stage_DI.yaml \
#       2>&1 | tee logs/run_train_iter11_v2.log

# NO -e: a single variant failure must NOT abort the chain (per overnight-chain rule).
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "USAGE: $0 <train-yaml1> [<train-yaml2> ...]" >&2
    exit 2
fi

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

EX="scripts/lib/yaml_extract.py"
T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

for yaml in "$@"; do
    if [ ! -f "$yaml" ]; then
        echo "❌ skipping: yaml not found: $yaml" >&2
        continue
    fi

    VARIANT_TAG="$(basename "$yaml" .yaml)"
    stamp "Variant: ${VARIANT_TAG}"

    MODULE=$("$EX" "$yaml" data.module)
    MODEL_CFG=$("$EX" "$yaml" data.model_config)
    TRAIN_SUBSET=$("$EX" "$yaml" data.train_subset)
    TRAIN_LOCAL=$("$EX" "$yaml" data.train_local_data)
    VAL_SUBSET=$("$EX" "$yaml" data.val_subset)
    VAL_LOCAL=$("$EX" "$yaml" data.val_local_data)
    OUT_DIR=$("$EX" "$yaml" data.output_dir)

    case "$MODULE" in
        m09b|m09c) ;;
        *) echo "❌ ${VARIANT_TAG}: unsupported data.module='$MODULE' (expected m09b|m09c)" >&2; continue ;;
    esac

    paths_ok=true
    for req in "$MODEL_CFG" "$TRAIN_SUBSET" "$TRAIN_LOCAL" "$VAL_SUBSET" "$VAL_LOCAL"; do
        if [ ! -e "$req" ]; then
            echo "❌ ${VARIANT_TAG}: missing path: $req" >&2
            paths_ok=false
        fi
    done
    $paths_ok || continue

    mkdir -p "$OUT_DIR"

    case "$MODULE" in
        m09b)
            python -u src/m09b_explora.py --FULL \
                --model-config "$MODEL_CFG" \
                --train-config "$yaml" \
                --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
                --val-subset "$VAL_SUBSET" --val-local-data "$VAL_LOCAL" \
                --output-dir "$OUT_DIR" --no-wandb \
                2>&1 | tee "logs/run_train_${VARIANT_TAG}.log" ;
            ;;
        m09c)
            FACTOR_DIR=$("$EX" "$yaml" data.factor_dir)
            VAL_TAGS="${VAL_LOCAL}/tags.json"
            if [ ! -f "$VAL_TAGS" ]; then
                echo "❌ ${VARIANT_TAG}: probe tags missing: $VAL_TAGS" >&2
                continue
            fi
            python -u src/m09c_surgery.py --FULL \
                --model-config "$MODEL_CFG" \
                --train-config "$yaml" \
                --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
                --factor-dir "$FACTOR_DIR" \
                --probe-subset "$VAL_SUBSET" \
                --probe-local-data "$VAL_LOCAL" \
                --probe-tags "$VAL_TAGS" \
                --output-dir "$OUT_DIR" --no-wandb \
                2>&1 | tee "logs/run_train_${VARIANT_TAG}.log" ;
            ;;
    esac

    if [ -f "${OUT_DIR}/student_encoder.pt" ]; then
        ls -lh "${OUT_DIR}/student_encoder.pt"
    else
        echo "  ⚠️  ${VARIANT_TAG}: student_encoder.pt not produced (check logs/run_train_${VARIANT_TAG}.log)"
    fi
done

DUR=$(( $(date +%s) - T0 ))
stamp "✅ run_train chain done · total wall=$((DUR/3600))h$(((DUR%3600)/60))m"
