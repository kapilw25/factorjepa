#!/usr/bin/env bash
# iter13 v12+ Task 3 (2026-05-06) — THIN wrapper around m10 → m11.
#
# Both .py modules self-resolve their I/O dirs from --local-data
# (co-located: <local_data>/m10_sam_segment/ + <local_data>/m11_factor_datasets/).
# m10 also self-handles the end-of-run TAR-shard pack for HF upload.
# This wrapper does ONLY:
#   1. SANITY/POC/FULL mode flag passthrough
#   2. cache-policy gather + propagate (m10 recompute → m11 recompute)
#   3. dispatch m10 then m11 with --local-data and (optional) --subset
#
# Canonical dataset path is hardcoded HERE (in the shell wrapper). CLAUDE.md's
# no-default rule targets src/m0*.py — shell wrappers ARE the layer that pins
# the canonical paths, then forward them as `--local-data`/`--subset` CLI flags
# to the .py modules (which themselves accept no defaults).
#
# Optional env-var overrides:
#   LOCAL_DATA       Override the hardcoded data/eval_10k_local (rare).
#   TRAIN_SUBSET     Subset JSON to filter clip set further. Unset → iterate
#                    all clips in $LOCAL_DATA, cap at mode's clip-limit.
#   CACHE_POLICY_ALL 1=keep cache, 2=recompute (else interactive prompt).
#
# USAGE:
#   ./scripts/run_factor_prep.sh <factor-yaml> [--SANITY|--POC|--FULL]
#
# Examples:
#   CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --SANITY
#   CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --POC
#   CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --FULL
#
# 3stage_DI factor-prep produces a strict superset (D_L + D_A + D_I) of what
# 2stage_noDI's run would produce (D_L + D_A only). The 2stage training will
# simply ignore D_I tubes. Run factor-prep ONCE with 3stage_DI to feed both.

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

# iter13 v13 FIX-20 (2026-05-07): canonical dataset hardcoded in the wrapper
# (CLAUDE.md targets src/m0*.py, not shells). Single source of truth for all
# three modes (SANITY/POC/FULL) — mode only changes the clip-count cap from
# pipeline.yaml (sanity.default=20 / poc.factor_prep=100 / FULL=all-of-manifest),
# NOT the data source. Override via env var LOCAL_DATA=<other-dir> only when
# running on a non-canonical dataset.
#
# TRAIN_SUBSET stays optional. When unset, m10/m11 iterate ALL clips in
# $TRAIN_LOCAL; consumer caps at the mode's clip-limit.
TRAIN_LOCAL="${LOCAL_DATA:-data/eval_10k_local}"
TRAIN_SUBSET="${TRAIN_SUBSET:-}"

if [ ! -d "$TRAIN_LOCAL" ]; then
    echo "FATAL: TRAIN_LOCAL=$TRAIN_LOCAL is not a directory." >&2
    echo "  Fetch the canonical dataset:" >&2
    echo "    python -u src/utils/hf_outputs.py download-data" >&2
    echo "  Or override via env var: LOCAL_DATA=<your-dir> $0 $@" >&2
    exit 3
fi
if [ -n "$TRAIN_SUBSET" ] && [ ! -e "$TRAIN_SUBSET" ]; then
    echo "FATAL: TRAIN_SUBSET=$TRAIN_SUBSET set but file not found." >&2
    echo "  Unset TRAIN_SUBSET to iterate all clips in $TRAIN_LOCAL." >&2
    exit 3
fi

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
        # iter13 v13 (2026-05-07): single-statement `local hit=$(...)` form is
        # REQUIRED. The split form (`local hit; hit=$(...)`) inherits the
        # substitution's exit code on the regular assignment → with set -e +
        # pipefail, compgen -G's exit 1 (no glob match — expected on first run
        # when m10_sam_segment/ doesn't exist yet) propagates → script aborts
        # silently (no ERR trap installed). The merged form has `local` as the
        # outer command, whose own exit status is 0, so the substitution failure
        # is swallowed.
        local hit=$(compgen -G "$path" 2>/dev/null | head -n1)
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
echo "  LOCAL_DATA:   $TRAIN_LOCAL  (single source for SANITY/POC/FULL — m10/m11 outputs co-located inside)"
if [ -n "$TRAIN_SUBSET" ]; then
    echo "  TRAIN_SUBSET: $TRAIN_SUBSET  (filtering clip set)"
else
    echo "  TRAIN_SUBSET: <unset>     (iterate all clips in LOCAL_DATA, cap at mode's clip-limit)"
fi
echo "──────────────────────────────────────────────"
_check_and_prompt m10 "${M10_OUT}/*"
_check_and_prompt m11 "${M11_OUT}/*"

# Dependency propagation: m10 recompute invalidates m11.
if [ "${POLICY[m10]:-1}" = "2" ]; then
    POLICY[m11]=2
fi

# iter13 v13 FIX-19 (2026-05-07): conditionally pass --subset only when set.
# Empty SUBSET_FLAG → m10/m11 iterate all clips in --local-data + cap at the
# mode's clip-limit (sanity.default / poc.factor_prep / FULL). Unquoted
# expansion of $SUBSET_FLAG is intentional so it expands to 2 tokens when set
# and 0 tokens when empty.
SUBSET_FLAG=""
if [ -n "$TRAIN_SUBSET" ]; then
    SUBSET_FLAG="--subset $TRAIN_SUBSET"
fi

stamp "Step A — m10 Grounded-SAM (mode=${MODE}; output → \$LOCAL_DATA/m10_sam_segment)"
python -u src/m10_sam_segment.py "${MODE_FLAG}" \
    --train-config "$FACTOR_YAML" \
    $SUBSET_FLAG --local-data "$TRAIN_LOCAL" \
    --no-wandb \
    --cache-policy "${POLICY[m10]}" \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_${MODE,,}_m10.log"

stamp "Step B — m11 --streaming (mode=${MODE}; output → \$LOCAL_DATA/m11_factor_datasets)"
python -u src/m11_factor_datasets.py "${MODE_FLAG}" --streaming \
    --train-config "$FACTOR_YAML" \
    $SUBSET_FLAG --local-data "$TRAIN_LOCAL" \
    --no-wandb \
    --cache-policy "${POLICY[m11]}" \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_${MODE,,}_m11.log"

DUR=$(( $(date +%s) - T0 ))
stamp "✅ factor-prep done · mode=${MODE} · wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Outputs (co-located inside ${TRAIN_LOCAL} — uploaded as one HF bundle):"
echo "  ${M10_OUT}/  (segments.json + summary.json + masks/*.npz + masks-*.tar shards)"
echo "  ${M11_OUT}/  (factor_manifest.json + verify samples)"
