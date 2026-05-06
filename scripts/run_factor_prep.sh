#!/usr/bin/env bash
# iter11 v2 factor prep — runs m10 (Grounded-SAM) + m11 (factor datasets) ONCE
# with the MAXIMAL factor config so all 3 surgery variants can consume the same outputs.
#
# Pass surgery_3stage_DI.yaml here: it has `interaction_mining.enabled: true`, so m10
# emits D_I interaction metadata and m11 builds D_I tubes. The two noDI variants ignore
# D_I tubes at training time (mode_mixture has I=0) — same outputs work for all 3.
#
# Reference: scripts/legacy2/run_paired_eval_10k.sh / scripts/run_iter9_10k.sh Steps A→B.
# Per CLAUDE.md "No hardcoded paths" — every path comes from the yaml's data: block.
#
# USAGE:
#   ./scripts/run_factor_prep.sh <factor-yaml>
#
# Example:
#   ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml 2>&1 | tee logs/run_factor_prep_v2.log

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "USAGE: $0 <factor-yaml>" >&2
    echo "  Pass the MAXIMAL surgery yaml (interaction_mining=true)." >&2
    echo "  Recommended: configs/train/surgery_3stage_DI.yaml" >&2
    exit 2
fi

FACTOR_YAML="$1"
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

# ── iter13 v12+ (2026-05-06): co-locate m10/m11 outputs INSIDE TRAIN_LOCAL ──
# OUTPUT_DIR = <INPUT_DIR>/m10_sam_segment/  and  <INPUT_DIR>/m11_factor_datasets/
# where <INPUT_DIR> = data.train_local_data from $FACTOR_YAML (no hardcode).
# Rationale: hf_outputs.upload_data() uploads $TRAIN_LOCAL/ as a single dir, so
# co-located outputs ride along automatically. Old layout (outputs/full/m10_*)
# required a separate hf_outputs.upload subfolder pass — fragmented the bundle.
M10_OUT="${TRAIN_LOCAL}/m10_sam_segment"
M11_OUT="${TRAIN_LOCAL}/m11_factor_datasets"
echo "  derived OUTPUT paths:"
echo "    m10: $M10_OUT"
echo "    m11: $M11_OUT (reads from m10 via --input-dir $M10_OUT)"

# ── Pre-flight: gather cache-policy decisions UPFRONT ─────────────────
# Mirrors scripts/legacy2/run_paired_eval_10k.sh pattern: prompts only fire when a cache
# exists at known paths. Missing caches default to policy=1 (keep). Bypasses:
# CACHE_POLICY_ALL=1|2 ./run_factor_prep.sh skips prompts; non-TTY stdin → 1.
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

echo "──────────────────────────────────────────────"
echo "factor-prep cache-policy gather (one prompt per existing cache)"
echo "──────────────────────────────────────────────"
# iter11 v3 (2026-04-26): prompt-trigger == delete-target. m10 + m11 each own their
# output dir entirely (single-producer; not shared across variants the way m05/m06 are),
# so any stray file in the dir is a candidate for cache-policy=2 wipe. Glob `dir/*`
# matches partial-run state (e.g. .m10_checkpoint.json without segments.json, or stale
# masks/ from a killed run) that 6-specific-paths would have missed.
_check_and_prompt m10 "${M10_OUT}/*"
_check_and_prompt m11 "${M11_OUT}/*"

# Dependency propagation: m10 recompute invalidates m11.
# Use if/then (not [...] && ...) — under `set -e`, the && form exits the script
# when the test is false (its non-zero exit status trips set -e).
if [ "${POLICY[m10]:-1}" = "2" ]; then
    POLICY[m11]=2
fi

stamp "factor-prep START · factor-yaml=${VARIANT_TAG}"
echo "  train_subset:    $TRAIN_SUBSET"
echo "  train_local:     $TRAIN_LOCAL"

stamp "Step A — m10 Grounded-SAM (output → ${M10_OUT})"
python -u src/m10_sam_segment.py --FULL \
    --train-config "$FACTOR_YAML" \
    --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
    --output-dir "$M10_OUT" --no-wandb \
    --cache-policy "${POLICY[m10]}" \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_m10.log"

stamp "Step B — m11 --streaming (output → ${M11_OUT})"
python -u src/m11_factor_datasets.py --FULL --streaming \
    --train-config "$FACTOR_YAML" \
    --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" \
    --input-dir "$M10_OUT" --output-dir "$M11_OUT" --no-wandb \
    --cache-policy "${POLICY[m11]}" \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_m11.log"

# ── HF-friendly TAR-shard: 9300+ .npz files in masks/ would exceed the 10k-file
# limit on huggingface dataset repos. Pack into 10 shards (mirrors existing
# subset-XXXXX.tar pattern) so hf_outputs.upload_data uploads the bundle.
# Idempotent: if masks-*.tar already exists + count > 0, skips. Reads
# original .npz from masks/ then leaves it in place (m11 + m09c may still need
# random-access reads; deletion is opt-in via --cache-policy=2 in tar_shard).
stamp "Step C — TAR-shard masks/ for HF upload"
python -u src/utils/tar_shard.py pack \
    --input-dir "$M10_OUT/masks" \
    --shard-template "$M10_OUT/masks-{shard:05d}.tar" \
    --n-shards 10 \
    --keep-source 2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_tar_shard.log"

DUR=$(( $(date +%s) - T0 ))
stamp "✅ factor-prep done · wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Outputs (co-located inside TRAIN_LOCAL=$TRAIN_LOCAL — uploaded as one HF bundle):"
echo "  ${M10_OUT}/  (segments.json + summary.json + masks/*.npz + masks-*.tar shards)"
echo "  ${M11_OUT}/  (factor_manifest.json + verify samples)"
