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

# ── Pre-flight: gather cache-policy decisions UPFRONT ─────────────────
# Mirrors scripts/run_paired_eval_10k.sh. Per-call-site × per-variant prompts:
# 2 shared (m05_frozen, m06_frozen) + 3 × N variants (m05/m06/m08b adapted).
# Bypasses: CACHE_POLICY_ALL=1|2 env skips prompts; non-TTY → 1.
declare -A POLICY
_check_and_prompt() {                 # $1=key  $2..=candidate cache paths/globs
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        # compgen -G handles literal paths AND globs uniformly. First hit wins.
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

echo "──────────────────────────────────────────────"
echo "run_eval cache-policy gather (one prompt per existing cache)"
echo "──────────────────────────────────────────────"
# iter11 v3 (2026-04-26) — scoping rationale:
#   - m05 (outputs/full/m05_vjepa_embed/) and m06 (outputs/full/m06_faiss_metrics/)
#     are SHARED dirs across encoders (frozen + N adapted variants). Files are
#     namespaced per encoder (embeddings_${ENC}.npy, m06_metrics_${ENC}.json).
#     A whole-dir wipe would delete OTHER variants' caches — wrong. We keep the
#     specific per-encoder file paths so prompt-trigger == per-encoder delete-target.
#   - m08b writes to ${OUT_DIR}/eval/ (per-variant, single-owner) → safe to glob.
#     This catches partial state (plots without paired_bootstrap_results.json,
#     stale tex tables) that the previous 2-specific-paths missed.
#
# Frozen (shared dir, per-encoder check — symmetric, no asymmetry to fix):
_check_and_prompt m05_frozen \
    "outputs/full/m05_vjepa_embed/embeddings_${FROZEN_ENC}.npy" \
    "outputs/full/m05_vjepa_embed/.m05_checkpoint_${FROZEN_ENC}.npz"
_check_and_prompt m06_frozen \
    "outputs/full/m06_faiss_metrics/m06_metrics_${FROZEN_ENC}.json"

# Adapted (per variant): 3 prompts × N
for yaml in "$@"; do
    [ -f "$yaml" ] || continue
    VARIANT_TAG="$(basename "$yaml" .yaml)"
    ADAPTED_ENC=$("$EX" "$yaml" data.adapted_encoder)
    OUT_DIR=$("$EX" "$yaml" data.output_dir)
    # m05/m06 — per-encoder file paths in shared dirs (kept specific by design)
    _check_and_prompt "m05_${VARIANT_TAG}" \
        "outputs/full/m05_vjepa_embed/embeddings_${ADAPTED_ENC}.npy"
    _check_and_prompt "m06_${VARIANT_TAG}" \
        "outputs/full/m06_faiss_metrics/m06_metrics_${ADAPTED_ENC}.json"
    # m08b — per-variant ${OUT_DIR}/eval/ is single-owner → glob the whole dir
    _check_and_prompt "m08b_${VARIANT_TAG}" "${OUT_DIR}/eval/*"
done

# Dependency propagation: upstream recompute invalidates downstream.
# Use if/then (not [...] && ...) — under `set -e`, the && form exits the script
# when the test is false (its non-zero exit status trips set -e).
if [ "${POLICY[m05_frozen]:-1}" = "2" ]; then
    POLICY[m06_frozen]=2
fi
if [ "${POLICY[m05_frozen]:-1}" = "2" ] || [ "${POLICY[m06_frozen]:-1}" = "2" ]; then
    for yaml in "$@"; do
        [ -f "$yaml" ] || continue
        VARIANT_TAG="$(basename "$yaml" .yaml)"
        POLICY[m08b_${VARIANT_TAG}]=2
    done
fi
for yaml in "$@"; do
    [ -f "$yaml" ] || continue
    VARIANT_TAG="$(basename "$yaml" .yaml)"
    if [ "${POLICY[m05_${VARIANT_TAG}]:-1}" = "2" ]; then
        POLICY[m06_${VARIANT_TAG}]=2
    fi
    if [ "${POLICY[m06_${VARIANT_TAG}]:-1}" = "2" ]; then
        POLICY[m08b_${VARIANT_TAG}]=2
    fi
done

stamp "Shared Frozen baseline · eval_subset=$(basename "$EVAL_SUBSET")"
echo "  model:        $MODEL_CFG"
echo "  eval_subset:  $EVAL_SUBSET"
echo "  eval_local:   $EVAL_LOCAL"

# ── flock-based mutex on the shared frozen baseline ─────────────────────
# Multiple instances running run_eval.sh in parallel would otherwise compute
# the same frozen embed/metrics 4× (~2.3 h GPU each, all writing to the SAME
# encoder name `vjepa_2_1_frozen`). flock guarantees exactly one instance
# computes; the others block on the lock, then read the freshly-written cache.
# Lock file lives on the shared workspace disk so all instances see the same
# inode (BSD flock; works on local FS — flaky on NFS, but iter11 v3 setup is
# local-disk per-instance with shared workspace mount).
FROZEN_LOCK="outputs/full/m05_vjepa_embed/.frozen.lock"
FROZEN_NPY="outputs/full/m05_vjepa_embed/embeddings_${FROZEN_ENC}.npy"
FROZEN_M06="outputs/full/m06_faiss_metrics/m06_metrics_${FROZEN_ENC}.json"
mkdir -p "$(dirname "$FROZEN_LOCK")" "$(dirname "$FROZEN_M06")"

# Open fd 200 → lock file. Try non-blocking acquire first.
exec 200>"$FROZEN_LOCK"
if flock -n 200; then
    stamp "Frozen baseline LOCK acquired — this instance computes m05 + m06 frozen"
    python -u src/m05_vjepa_embed.py --FULL \
        --subset "$EVAL_SUBSET" \
        --model-config "$MODEL_CFG" \
        --encoder "$FROZEN_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        --cache-policy "${POLICY[m05_frozen]}" \
        2>&1 | tee "logs/run_eval_frozen_m05.log"

    python -u src/m06_faiss_metrics.py --FULL \
        --subset "$EVAL_SUBSET" --encoder "$FROZEN_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        --cache-policy "${POLICY[m06_frozen]}" \
        2>&1 | tee "logs/run_eval_frozen_m06.log"
    flock -u 200
    stamp "Frozen baseline LOCK released — other instances can now proceed"
else
    stamp "Frozen baseline being computed by another instance — waiting on lock..."
    flock 200    # blocking — wakes when first instance releases
    flock -u 200
    stamp "Frozen baseline now ready (computed by another instance) — reusing cache"
    if [ ! -f "$FROZEN_NPY" ] || [ ! -f "$FROZEN_M06" ]; then
        echo "FATAL: lock released but frozen artifacts missing:" >&2
        echo "  $FROZEN_NPY exists=$([ -f "$FROZEN_NPY" ] && echo yes || echo no)" >&2
        echo "  $FROZEN_M06 exists=$([ -f "$FROZEN_M06" ] && echo yes || echo no)" >&2
        echo "  → first instance crashed before producing them." >&2
        exit 5
    fi
fi
exec 200>&-    # close fd

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

    P_M05="${POLICY[m05_${VARIANT_TAG}]:-1}"
    P_M06="${POLICY[m06_${VARIANT_TAG}]:-1}"
    P_M08B="${POLICY[m08b_${VARIANT_TAG}]:-1}"

    python -u src/m05_vjepa_embed.py --FULL \
        --subset "$EVAL_SUBSET" \
        --model-config "$MODEL_CFG" \
        --model "$ADAPTED_CKPT" \
        --encoder "$ADAPTED_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        --cache-policy "$P_M05" \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m05.log"

    python -u src/m06_faiss_metrics.py --FULL \
        --subset "$EVAL_SUBSET" --encoder "$ADAPTED_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        --cache-policy "$P_M06" \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m06.log"

    # Per-variant m08b output dir prevents 4-way overwrite when multiple variants
    # run eval in parallel (paired_bootstrap_results.json + 8 plots + tex table).
    M08B_OUT="${OUT_DIR}/eval"
    mkdir -p "$M08B_OUT"
    python -u src/m08b_compare.py --FULL \
        --subset "$EVAL_SUBSET" \
        --encoders "${FROZEN_ENC},${ADAPTED_ENC}" \
        --output-dir "$M08B_OUT" \
        --no-wandb \
        --cache-policy "$P_M08B" \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m08b.log"
done

DUR=$(( $(date +%s) - T0 ))
stamp "✅ run_eval chain done · total wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Per-variant artifacts: outputs/full/<variant>/  +  outputs/full/{m05,m06,m08b}_*/"
