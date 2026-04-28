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
#   INCLUDE_BASELINES=1 ./scripts/run_eval.sh <yamls...>     # bound the metric axis
#       (random/oracle/dinov2/clip/vjepa_shuffled — 5 extra m05b+m06 cycles, ~30 min)
#
# Example:
#   tmux new -s eval
#   INCLUDE_BASELINES=1 ./scripts/run_eval.sh \
#       configs/train/explora.yaml \
#       configs/train/surgery_2stage_noDI.yaml \
#       configs/train/surgery_2stage_loud_agent.yaml \
#       configs/train/surgery_3stage_DI.yaml \
#       configs/train/surgery_3stage_DI_multitask.yaml \
#       2>&1 | tee logs/run_eval_iter11_v3.log

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

# iter11 v3 (2026-04-27): opt-in baseline sweep (random/oracle/dinov2/clip/vjepa_shuffled).
# Bounds the eval-set Prec@K/mAP@K/Cycle@K between chance (random) and ceiling (oracle =
# multi-hot tag vector). Off by default — set INCLUDE_BASELINES=1 to enable.
# Each baseline runs in its own process (m05b's os._exit(0) workaround for the
# torch.compile + CUDA atexit deadlock prevents `--encoder all` from looping).
INCLUDE_BASELINES="${INCLUDE_BASELINES:-0}"
BASELINE_ENCODERS=(random oracle dinov2 clip vjepa_shuffled)

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
# Frozen + adapted: only m05 cache-policy is gathered (m06 + m08b are exempt per
# their module docstring DELETE-PROTECTION POLICY blocks — both are pure functions
# of inputs that always recompute. errors_N_fixes #80, 2026-04-27).
_check_and_prompt m05_frozen \
    "outputs/full/m05_vjepa_embed/embeddings_${FROZEN_ENC}.npy" \
    "outputs/full/m05_vjepa_embed/.m05_checkpoint_${FROZEN_ENC}.npz"

# Adapted (per variant): 1 prompt × N (m05 only — m06 + m08b exempt as above).
for yaml in "$@"; do
    [ -f "$yaml" ] || continue
    VARIANT_TAG="$(basename "$yaml" .yaml)"
    ADAPTED_ENC=$("$EX" "$yaml" data.adapted_encoder)
    _check_and_prompt "m05_${VARIANT_TAG}" \
        "outputs/full/m05_vjepa_embed/embeddings_${ADAPTED_ENC}.npy"
done

# No cascade needed — m06 + m08b are always-recompute, so upstream m05 invalidation
# automatically flows downstream by reading freshly-written embeddings on each call.

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

    # m06 always recomputes (no --cache-policy — see m06 module docstring).
    python -u src/m06_faiss_metrics.py --FULL \
        --subset "$EVAL_SUBSET" --encoder "$FROZEN_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
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

# ── Optional: shared baseline sweep (random / oracle / dinov2 / clip / vjepa_shuffled) ──
# Mirrors the frozen-baseline lock pattern so multiple instances don't recompute.
# Each baseline writes per-encoder files in shared dirs (embeddings_${ENC}.npy +
# m06_metrics_${ENC}.json) — symmetric with frozen.
if [ "$INCLUDE_BASELINES" = "1" ]; then
    TAGS_JSON="${EVAL_LOCAL}/tags.json"
    if [ ! -f "$TAGS_JSON" ]; then
        echo "FATAL: --include-baselines needs ${TAGS_JSON} (oracle dependency); not found." >&2
        exit 6
    fi

    stamp "Shared baseline sweep · ${#BASELINE_ENCODERS[@]} encoders · tags=${TAGS_JSON}"

    BASELINES_LOCK="outputs/full/m05_vjepa_embed/.baselines.lock"
    exec 201>"$BASELINES_LOCK"
    if flock -n 201; then
        stamp "Baseline LOCK acquired — this instance computes ${BASELINE_ENCODERS[*]}"
        # SYNTHETIC = CPU-only, no checkpoint, m05b internally bypasses cache-policy
        # prompt for these (see m05b main()). GPU baselines keep --cache-policy 1
        # because their .m05b_*_checkpoint.npz IS expensive (~10 min model load).
        # m06 always omits --cache-policy (always-recompute, see m06 docstring).
        SYNTHETIC_BASELINES="random oracle"
        for ENC in "${BASELINE_ENCODERS[@]}"; do
            stamp "  baseline: ${ENC}"
            if [[ " $SYNTHETIC_BASELINES " == *" $ENC "* ]]; then
                python -u src/m05b_baselines.py --encoder "$ENC" --FULL \
                    --subset "$EVAL_SUBSET" \
                    --local-data "$EVAL_LOCAL" \
                    --tags-json "$TAGS_JSON" \
                    --no-wandb \
                    2>&1 | tee "logs/run_eval_baseline_${ENC}_m05b.log"
            else
                python -u src/m05b_baselines.py --encoder "$ENC" --FULL \
                    --subset "$EVAL_SUBSET" \
                    --local-data "$EVAL_LOCAL" \
                    --tags-json "$TAGS_JSON" \
                    --no-wandb --cache-policy 1 \
                    2>&1 | tee "logs/run_eval_baseline_${ENC}_m05b.log"
            fi

            python -u src/m06_faiss_metrics.py --FULL \
                --subset "$EVAL_SUBSET" --encoder "$ENC" \
                --local-data "$EVAL_LOCAL" --no-wandb \
                2>&1 | tee "logs/run_eval_baseline_${ENC}_m06.log"
        done
        flock -u 201
        stamp "Baseline LOCK released"
    else
        stamp "Baselines being computed by another instance — waiting on lock..."
        flock 201
        flock -u 201
        stamp "Baselines now ready (computed by another instance)"
    fi
    exec 201>&-
fi

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

    python -u src/m05_vjepa_embed.py --FULL \
        --subset "$EVAL_SUBSET" \
        --model-config "$MODEL_CFG" \
        --model "$ADAPTED_CKPT" \
        --encoder "$ADAPTED_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        --cache-policy "$P_M05" \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m05.log"

    # m06 always recomputes (no --cache-policy — see m06 module docstring).
    python -u src/m06_faiss_metrics.py --FULL \
        --subset "$EVAL_SUBSET" --encoder "$ADAPTED_ENC" \
        --local-data "$EVAL_LOCAL" --no-wandb \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m06.log"

    # Per-variant m08b output dir prevents 4-way overwrite when multiple variants
    # run eval in parallel (paired_bootstrap_results.json + 8 plots + tex table).
    # NO --cache-policy here on purpose — m08b always recomputes (see m08b docstring).
    M08B_OUT="${OUT_DIR}/eval"
    mkdir -p "$M08B_OUT"
    python -u src/m08b_compare.py --FULL \
        --subset "$EVAL_SUBSET" \
        --encoders "${FROZEN_ENC},${ADAPTED_ENC}" \
        --output-dir "$M08B_OUT" \
        --no-wandb \
        2>&1 | tee "logs/run_eval_${VARIANT_TAG}_m08b.log"
done

# ── Aggregate m08b: ALL encoders side-by-side (frozen + every adapted variant + baselines) ──
# Per-variant m08b above only renders 2-encoder paired plots (frozen vs ONE adapted) and skips
# the radar (n<3). This final call rebuilds the multi-encoder radar/heatmap/comparison/STbar
# from cached m05+m06 outputs (no GPU re-run needed). cache-policy=2 wipes the aggregate dir
# so the previous run's PNGs don't shadow the new encoder set.
ALL_ADAPTED=()
for yaml in "$@"; do
    [ -f "$yaml" ] || continue
    ALL_ADAPTED+=( "$("$EX" "$yaml" data.adapted_encoder)" )
done
AGG_ENCODERS="${FROZEN_ENC}"
for enc in "${ALL_ADAPTED[@]}"; do AGG_ENCODERS="${AGG_ENCODERS},${enc}"; done
if [ "$INCLUDE_BASELINES" = "1" ]; then
    for enc in "${BASELINE_ENCODERS[@]}"; do AGG_ENCODERS="${AGG_ENCODERS},${enc}"; done
fi

stamp "Aggregate m08b · encoders=${AGG_ENCODERS}"
M08B_AGG_OUT="outputs/full/m08b_aggregate"
mkdir -p "$M08B_AGG_OUT"
# NO --cache-policy here — m08b always wipes its output_dir + recomputes (see m08b docstring).
python -u src/m08b_compare.py --FULL \
    --subset "$EVAL_SUBSET" \
    --encoders "$AGG_ENCODERS" \
    --output-dir "$M08B_AGG_OUT" \
    --no-wandb \
    2>&1 | tee logs/run_eval_aggregate_m08b.log

DUR=$(( $(date +%s) - T0 ))
stamp "✅ run_eval chain done · total wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Per-variant artifacts:  outputs/full/<variant>/eval/  +  outputs/full/{m05,m06}_*/"
echo "Aggregate (all encoders): outputs/full/m08b_aggregate/"
