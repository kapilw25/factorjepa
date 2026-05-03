#!/usr/bin/env bash
# iter11 v2 trainer — loops over N train yamls; for each, dispatches m09b (ExPLoRA)
# or m09c (Surgery) based on the yaml's data.module key.
# Every path is read from the yaml's data: block via scripts/lib/yaml_extract.py.
# Per CLAUDE.md "No hardcoded paths" + "Shell scripts are THIN wrappers".
#
# Prereq: scripts/run_factor_prep.sh has been run (m10/m11 outputs present in
# outputs/full/m10_sam_segment/ and outputs/full/m11_factor_datasets/).
#
# Reference: scripts/legacy2/run_paired_eval_10k.sh / scripts/run_iter9_10k.sh Step C.
#
# USAGE:
#   ./scripts/legacy2/run_train.sh <train-yaml1> [<train-yaml2> ...]
#
# Example (full iter11 v2 4-variant chain in one tmux session):
#   tmux new -s train
#   ./scripts/legacy2/run_train.sh \
#       configs/legacy2/explora.yaml \
#       configs/legacy2/surgery_2stage_noDI.yaml \
#       configs/legacy2/surgery_2stage_loud_agent.yaml \
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

# ── Pre-flight: gather cache-policy decisions UPFRONT (one per variant) ─
# Mirrors scripts/legacy2/run_paired_eval_10k.sh pattern. Bypasses: CACHE_POLICY_ALL=1|2
# env var skips prompts; non-TTY stdin defaults to 1.
declare -A POLICY
_check_and_prompt() {                 # $1=key  $2..=candidate cache paths/globs
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        # Path may be literal OR a glob (m09b_ckpt_step*.pt). compgen -G matches both.
        local hit
        hit=$(compgen -G "$path" 2>/dev/null | head -n1)
        if [ -n "$hit" ]; then found="$hit"; break; fi
    done
    if [ -z "$found" ]; then POLICY[$key]=1; return; fi
    if [ -n "${CACHE_POLICY_ALL:-}" ]; then
        POLICY[$key]=$CACHE_POLICY_ALL
        echo "  $key: cache at $found -> policy=${POLICY[$key]} (CACHE_POLICY_ALL)"
        return
    fi
    if [ ! -t 0 ]; then
        POLICY[$key]=1
        echo "  $key: cache at $found -> policy=1 (non-TTY default)"
        return
    fi
    local ans
    read -p "  $key cache at $found [1=keep / 2=recompute] (Enter=1): " ans
    case "${ans:-1}" in
        2|recompute) POLICY[$key]=2 ;;
        *)           POLICY[$key]=1 ;;
    esac
    return 0
}

echo "──────────────────────────────────────────────"
echo "run_train cache-policy gather (one prompt per variant if checkpoint exists)"
echo "──────────────────────────────────────────────"
for yaml in "$@"; do
    [ -f "$yaml" ] || continue
    VARIANT_TAG="$(basename "$yaml" .yaml)"
    OUT_DIR=$("$EX" "$yaml" data.output_dir)
    # iter11 v3 (2026-04-26): prompt-trigger MUST match delete-target. cache-policy=2
    # invokes wipe_output_dir() which nukes the WHOLE OUT_DIR (utils/cache_policy.py).
    # Earlier this list enumerated 6 specific ckpt paths — but a partial run leaving
    # only loss_log.csv / probe_history.jsonl / val_split.json was still a valid
    # "stale state" that cache-policy=2 would wipe, yet the prompt didn't fire and
    # the script silently set POLICY=1. Single glob `${OUT_DIR}/*` matches the same
    # set of files that wipe_output_dir destroys, so the user is always asked when
    # the dir has any content (and silently skipped when truly empty / non-existent).
    _check_and_prompt "m09_${VARIANT_TAG}" "${OUT_DIR}/*"
done

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

    P_M09="${POLICY[m09_${VARIANT_TAG}]:-1}"
    case "$MODULE" in
        m09b)
            python -u src/m09b_explora.py --FULL \
                --model-config "$MODEL_CFG" \
                --train-config "$yaml" \
                --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
                --val-subset "$VAL_SUBSET" --val-local-data "$VAL_LOCAL" \
                --output-dir "$OUT_DIR" --no-wandb \
                --cache-policy "$P_M09" \
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
                --cache-policy "$P_M09" \
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
