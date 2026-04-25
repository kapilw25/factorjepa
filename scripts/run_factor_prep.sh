#!/usr/bin/env bash
# iter11 v2 factor prep — runs m10 (Grounded-SAM) + m11 (factor datasets) ONCE
# with the MAXIMAL factor config so all 3 surgery variants can consume the same outputs.
#
# Pass surgery_3stage_DI.yaml here: it has `interaction_mining.enabled: true`, so m10
# emits D_I interaction metadata and m11 builds D_I tubes. The two noDI variants ignore
# D_I tubes at training time (mode_mixture has I=0) — same outputs work for all 3.
#
# Reference: scripts/run_paired_eval_10k.sh / scripts/run_iter9_10k.sh Steps A→B.
# Per CLAUDE.md "No hardcoded paths" — every path comes from the yaml's data: block.
#
# USAGE:
#   ./scripts/run_factor_prep.sh <factor-yaml>
#
# Example:
#   ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml

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

stamp "factor-prep START · factor-yaml=${VARIANT_TAG}"
echo "  train_subset:    $TRAIN_SUBSET"
echo "  train_local:     $TRAIN_LOCAL"

stamp "Step A — m10 Grounded-SAM"
python -u src/m10_sam_segment.py --FULL \
    --train-config "$FACTOR_YAML" \
    --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" --no-wandb \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_m10.log"

stamp "Step B — m11 --streaming"
python -u src/m11_factor_datasets.py --FULL --streaming \
    --train-config "$FACTOR_YAML" \
    --subset "$TRAIN_SUBSET" --local-data "$TRAIN_LOCAL" --no-wandb \
    2>&1 | tee "logs/run_factor_prep_${VARIANT_TAG}_m11.log"

DUR=$(( $(date +%s) - T0 ))
stamp "✅ factor-prep done · wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Outputs (shared by all surgery variants):"
echo "  outputs/full/m10_sam_segment/  (segments.json + per-clip masks .npz)"
echo "  outputs/full/m11_factor_datasets/  (factor_manifest.json + verify samples)"
