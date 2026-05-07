#!/usr/bin/env bash
# iter13 v12+ Task 3 (2026-05-06) — THIN wrapper around m10 → m11.
#
# Both .py modules now self-resolve their I/O dirs from --local-data
# (co-located: <local_data>/m10_sam_segment/ + <local_data>/m11_factor_datasets/).
# m10 also self-handles the end-of-run TAR-shard pack for HF upload.
# This shell wrapper does ONLY:
#   1. yaml extract → train_subset + train_local_data
#   2. SANITY/POC/FULL mode flag passthrough
#   3. cache-policy gather + propagate (m10 recompute → m11 recompute)
#   4. dispatch m10 then m11 with --local-data and --subset
#
# USAGE:
#   ./scripts/run_factor_prep.sh <factor-yaml> [--SANITY|--POC|--FULL]
#
# Examples:
#   ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --SANITY
#   ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --FULL
#
# Bypass cache prompts in tmux/CI:
#   CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh <yaml> --FULL

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "USAGE: $0 <factor-yaml> [--SANITY|--POC|--FULL]" >&2
    echo "  Default mode: --FULL." >&2
    echo "  Pass MAXIMAL surgery yaml (interaction_mining=true)." >&2
    echo "  Recommended: configs/train/surgery_3stage_DI.yaml" >&2
    exit 2
fi

FACTOR_YAML="$1"
MODE_FLAG="${2:---FULL}"
case "$MODE_FLAG" in
    --SANITY|--sanity) MODE="SANITY" ;;
    --POC|--poc)       MODE="POC" ;;
    --FULL|--full)     MODE="FULL" ;;
    *) echo "FATAL: mode flag must be --SANITY|--POC|--FULL (got: $MODE_FLAG)" >&2; exit 2 ;;
esac

if [ ! -f "$FACTOR_YAML" ]; then
    echo "FATAL: factor yaml not found: $FACTOR_YAML" >&2
    exit 3
fi

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

EX="scripts/lib/yaml_extract.py"
TRAIN_SUBSET=$("$EX" "$FACTOR_YAML" data.train_subset)
TRAIN_LOCAL=$("$EX" "$FACTOR_YAML" data.train_local_data)

for req in "$TRAIN_SUBSET" "$TRAIN_LOCAL"; do
    if [ ! -e "$req" ]; then
        echo "FATAL: missing path from $FACTOR_YAML: $req" >&2
        exit 3
    fi
done

VARIANT_TAG="$(basename "$FACTOR_YAML" .yaml)"
T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

# Each .py self-resolves its output dir from --local-data; cache-policy probes
# use the same convention to find existing caches.
M10_OUT="${TRAIN_LOCAL}/m10_sam_segment"
M11_OUT="${TRAIN_LOCAL}/m11_factor_datasets"

declare -A POLICY
_check_and_prompt() {
    local key="$1"; shift
    local found=""
    for path in "$@"; do
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
echo "factor-prep · mode=${MODE} · variant=${VARIANT_TAG}"
echo "  train_subset: $TRAIN_SUBSET"
echo "  train_local:  $TRAIN_LOCAL  (m10/m11 outputs co-located inside)"
echo "──────────────────────────────────────────────"
_check_and_prompt m10 "${M10_OUT}/*"
_check_and_prompt m11 "${M11_OUT}/*"

# Dependency propagation: m10 recompute invalidates m11.
if [ "${POLICY[m10]:-1}" = "2" ]; then
    POLICY[m11]=2
fi

stamp "Step A — m10 Grounded-SAM (mode=${MODE}; output → \$LOCAL_DATA/m10_sam_segment)"
python -u src/m10_sam_segment.py "${MODE_FLAG}" \
    --train-config "$FACTOR_YAML" \
    --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
    --no-wandb \
    --cache-policy "${POLICY[m10]}" \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_${MODE,,}_m10.log"

stamp "Step B — m11 --streaming (mode=${MODE}; output → \$LOCAL_DATA/m11_factor_datasets)"
python -u src/m11_factor_datasets.py "${MODE_FLAG}" --streaming \
    --train-config "$FACTOR_YAML" \
    --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
    --no-wandb \
    --cache-policy "${POLICY[m11]}" \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_${MODE,,}_m11.log"

DUR=$(( $(date +%s) - T0 ))
stamp "✅ factor-prep done · mode=${MODE} · wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Outputs (co-located inside ${TRAIN_LOCAL} — uploaded as one HF bundle):"
echo "  ${M10_OUT}/  (segments.json + summary.json + masks/*.npz + masks-*.tar shards)"
echo "  ${M11_OUT}/  (factor_manifest.json + verify samples)"
