#!/usr/bin/env bash
# iter13 probe encoder trainer — produces vjepa_2_1_pretrain (P2) + vjepa_2_1_surgical (P3)
# checkpoints that scripts/run_probe_eval.sh consumes for the 4-encoder paired-Δ
# action-probe gate. Mirrors the run_train.sh / run_eval.sh split (iter11 v2).
#
# Subcommands:
#   pretrain          — m09a continual SSL pretrain (LR 1e-5, 5 epochs, drift L2 anchor)
#   surgery_3stage_DI — m09c factor surgery WITH interaction tubes (D_L → D_A → D_I, 40/30/30 split)
#   surgery_noDI      — m09c factor surgery WITHOUT D_I  (D_L → D_A,    50/50 split)
#
# Why two surgery variants? iter9 v15c showed Stage 3 (D_I unfreeze 75%) caused
# BWT=-0.33 (Prec@K dropped 20.50→19.83). Whether D_I helps under the iter13
# probe-accuracy metric (vs the legacy Prec@K) is genuinely open — train both
# and let probe_eval's paired-Δ decide.
#
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python" — this
# wrapper only:
#   1. Pre-flights that probe Stage 1 ran (action_labels.json must exist)
#   2. Generates eval_10k_{train,val}_split.json via src/utils/probe_train_subset.py
#   3. Dispatches m09a or m09c with the right yaml + output dir
#
# USAGE:
#   tmux new -s probe_train
#   ./scripts/run_probe_train.sh pretrain          --SANITY   # ~3 min
#   ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY   # ~10 min
#   ./scripts/run_probe_train.sh surgery_noDI      --SANITY   # ~7 min  (no Stage 3)
#   ./scripts/run_probe_train.sh pretrain          --FULL     # ~3 GPU-h
#   ./scripts/run_probe_train.sh surgery_3stage_DI --FULL     # ~6-8 GPU-h
#   ./scripts/run_probe_train.sh surgery_noDI      --FULL     # ~4-6 GPU-h
#
# Bypass the cache-policy prompt for overnight runs:
#   CACHE_POLICY_ALL=1 ./scripts/run_probe_train.sh pretrain --FULL  # keep stale outputs
#   CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --FULL  # wipe + recompute

set -euo pipefail
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_probe_train.sh aborted at line $LINENO (exit=$rc)" >&2; exit $rc' ERR

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

if [ $# -lt 2 ]; then
    echo "USAGE: $0 {pretrain|surgery_3stage_DI|surgery_noDI} {--SANITY|--POC|--FULL}" >&2
    exit 2
fi

SUBCMD="$1"; shift
MODE_FLAG="$1"; shift

case "$SUBCMD" in
    pretrain|surgery_3stage_DI|surgery_noDI) ;;
    *) echo "FATAL: subcommand must be {pretrain|surgery_3stage_DI|surgery_noDI} (got: $SUBCMD)" >&2; exit 2 ;;
esac

case "$MODE_FLAG" in
    --SANITY|--sanity) MODE="SANITY"; mode_dir="sanity" ;;
    --POC|--poc)       MODE="POC";    mode_dir="poc" ;;
    --FULL|--full)     MODE="FULL";   mode_dir="full" ;;
    *) echo "FATAL: mode flag must be --SANITY|--POC|--FULL (got: $MODE_FLAG)" >&2; exit 2 ;;
esac

# ── Pre-flight: probe Stage 1 (action_labels.json) must exist ─────────────
ACTION_LABELS="outputs/${mode_dir}/probe_action/action_labels.json"
if [ ! -f "$ACTION_LABELS" ]; then
    echo "❌ FATAL: $ACTION_LABELS not found." >&2
    echo "  Run probe Stage 1 first to generate the train/val/test split:" >&2
    echo "    SKIP_STAGES=\"2,3,4,5,6,7,8,9,10\" ./scripts/run_probe_eval.sh ${MODE_FLAG}" >&2
    exit 3
fi

# ── Pre-flight: bitsandbytes for SANITY 8-bit optim path ────────────────
if [ "$MODE" = "SANITY" ]; then
    if ! python -c "import bitsandbytes" 2>/dev/null; then
        echo "❌ FATAL: bitsandbytes not installed (required for SANITY 24 GB 8-bit AdamW)." >&2
        echo "  Run: pip install bitsandbytes" >&2
        exit 3
    fi
fi

# ── Generate train/val split JSONs from action_labels.json ────────────────
TRAIN_SPLIT="data/eval_10k_train_split.json"
VAL_SPLIT="data/eval_10k_val_split.json"
echo "═══ $(date '+%H:%M:%S') · Generating train/val split JSONs ═══"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split train --output "$TRAIN_SPLIT"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split val --output "$VAL_SPLIT"

LOCAL_DATA="data/eval_10k_local"
[ -d "$LOCAL_DATA" ] || { echo "❌ FATAL: $LOCAL_DATA missing"; exit 3; }
MODEL_CFG="configs/model/vjepa2_1.yaml"
P_M09="${CACHE_POLICY_ALL:-1}"

# ── Multi-task probe-loss labels (iter13) ────────────────────────────────
# When base_optimization.yaml `multi_task_probe.enabled` is true for this
# mode, m09a/m09c add CrossEntropy/BCE losses on 16 taxonomy dims to JEPA L1
# — needs taxonomy_labels.json from probe_taxonomy --stage labels.
#
# Auto-generate if missing so the wrapper is self-sufficient (matches the
# pattern of probe-train-subset auto-gen above). probe_taxonomy --stage
# labels is CPU-only (~30 s on FULL); the inputs (eval_subset, tags.json,
# tag_taxonomy.json) are already on disk for any working repo. If sources
# are missing, we fall back to silent-disable + a fix-it hint.
TAXONOMY_LABELS="outputs/${mode_dir}/probe_taxonomy/taxonomy_labels.json"
TAG_TAXONOMY="configs/tag_taxonomy.json"
# eval_subset path mirrors run_probe_eval.sh's mode-gated convention.
if [ "$MODE" = "SANITY" ]; then
    EVAL_SUBSET_TX="data/eval_10k_sanity.json"
else
    EVAL_SUBSET_TX="data/eval_10k.json"
fi
TAGS_JSON_TX="${LOCAL_DATA}/tags.json"
TAXONOMY_ARGS=()
if [ ! -f "$TAXONOMY_LABELS" ]; then
    if [ -f "$TAG_TAXONOMY" ] && [ -f "$EVAL_SUBSET_TX" ] && [ -f "$TAGS_JSON_TX" ]; then
        echo "  [multi-task] $TAXONOMY_LABELS missing — auto-generating via probe_taxonomy --stage labels"
        python -u src/probe_taxonomy.py "${MODE_FLAG}" \
            --stage labels \
            --eval-subset "$EVAL_SUBSET_TX" \
            --tags-json "$TAGS_JSON_TX" \
            --tag-taxonomy "$TAG_TAXONOMY" \
            --output-root "outputs/${mode_dir}/probe_taxonomy" \
            --cache-policy "$P_M09" \
            2>&1 | tee "logs/probe_taxonomy_labels_${mode_dir}.log"
    else
        echo "  [multi-task] cannot auto-generate $TAXONOMY_LABELS — sources missing:"
        [ -f "$TAG_TAXONOMY" ]   || echo "      ✗ $TAG_TAXONOMY"
        [ -f "$EVAL_SUBSET_TX" ] || echo "      ✗ $EVAL_SUBSET_TX"
        [ -f "$TAGS_JSON_TX" ]   || echo "      ✗ $TAGS_JSON_TX"
        echo "    → multi_task_probe will auto-disable for this run"
    fi
fi
if [ -f "$TAXONOMY_LABELS" ]; then
    TAXONOMY_ARGS=(--taxonomy-labels-json "$TAXONOMY_LABELS")
    echo "  [multi-task] Using taxonomy labels: $TAXONOMY_LABELS"
fi

# ── Dispatch ──────────────────────────────────────────────────────────────
case "$SUBCMD" in
    pretrain)
        OUT_DIR="outputs/${mode_dir}/probe_pretrain"
        TRAIN_CFG="configs/train/probe_pretrain.yaml"
        # Read lambda_reg from YAML so it stays the single source of truth.
        # Passing it explicitly to m09a bypasses the legacy auto-ablation gate
        # (m09a_pretrain.py:1187 enters EWC sweep when --lambda-reg is unset;
        # probe_pretrain.yaml intentionally has ablation_lambdas=[] which would
        # then trip select_ablation_winner's non-empty assertion).
        LAMBDA_REG=$(scripts/lib/yaml_extract.py "$TRAIN_CFG" drift_control.lambda_reg)
        echo "═══ $(date '+%H:%M:%S') · m09a continual SSL pretrain (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_SPLIT"
        echo "  val:       $VAL_SPLIT"
        echo "  local:     $LOCAL_DATA"
        echo "  output:    $OUT_DIR"
        echo "  lambda:    $LAMBDA_REG (read from yaml; bypasses auto-ablation gate)"
        mkdir -p "$OUT_DIR"
        python -u src/m09a_pretrain.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_SPLIT" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" --no-wandb \
            --cache-policy "$P_M09" \
            --lambda-reg "$LAMBDA_REG" \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "${LOCAL_DATA}/tags.json" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            "${TAXONOMY_ARGS[@]}" \
            2>&1 | tee "logs/probe_pretrain_${mode_dir}.log"
        ;;
    surgery_3stage_DI|surgery_noDI)
        # Map subcommand → yaml + variant tag (used in output dir + log name).
        case "$SUBCMD" in
            surgery_3stage_DI)
                TRAIN_CFG="configs/train/surgery_3stage_DI.yaml"
                VARIANT_TAG="3stage_DI"
                ;;
            surgery_noDI)
                TRAIN_CFG="configs/train/surgery_2stage_noDI.yaml"
                VARIANT_TAG="noDI"
                ;;
        esac
        OUT_DIR="outputs/${mode_dir}/probe_surgery_${VARIANT_TAG}"
        FACTOR_DIR="outputs/full/m11_factor_datasets"
        VAL_TAGS="${LOCAL_DATA}/tags.json"
        # Surgery prereq: m10/m11 factor datasets (D_L/D_A/D_I) generated by run_factor_prep.sh.
        # noDI variant only needs D_L + D_A but the factor dir holds all three; same prereq.
        if [ ! -d "$FACTOR_DIR" ]; then
            echo "❌ FATAL: $FACTOR_DIR missing — run scripts/run_factor_prep.sh first" >&2
            exit 3
        fi
        if [ ! -f "$VAL_TAGS" ]; then
            echo "❌ FATAL: $VAL_TAGS missing — surgery's mid-training probe requires VLM tags" >&2
            exit 3
        fi
        echo "═══ $(date '+%H:%M:%S') · m09c factor surgery [variant=${VARIANT_TAG}] (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_SPLIT"
        echo "  factor:    $FACTOR_DIR"
        echo "  output:    $OUT_DIR"
        mkdir -p "$OUT_DIR"
        python -u src/m09c_surgery.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_SPLIT" --local-data "$LOCAL_DATA" \
            --factor-dir "$FACTOR_DIR" \
            --probe-subset "$VAL_SPLIT" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "$VAL_TAGS" \
            --output-dir "$OUT_DIR" --no-wandb \
            --cache-policy "$P_M09" \
            "${TAXONOMY_ARGS[@]}" \
            2>&1 | tee "logs/probe_surgery_${VARIANT_TAG}_${mode_dir}.log"
        ;;
esac

# ── Verify expected outputs ──────────────────────────────────────────────
echo ""
echo "═══ $(date '+%H:%M:%S') · DONE ═══"
echo "Expected outputs (consumed by scripts/run_probe_eval.sh):"
if [ -f "${OUT_DIR}/student_encoder.pt" ]; then
    ls -lh "${OUT_DIR}/student_encoder.pt"
else
    echo "  ⚠️  ${OUT_DIR}/student_encoder.pt NOT produced (check logs/probe_${SUBCMD}_${mode_dir}.log)"
fi
case "$SUBCMD" in
    pretrain)
        FULL_CKPT="${OUT_DIR}/m09a_ckpt_best.pt"
        ;;
    surgery_3stage_DI|surgery_noDI)
        FULL_CKPT="${OUT_DIR}/m09c_ckpt_best.pt"
        ;;
esac
if [ -f "$FULL_CKPT" ]; then
    ls -lh "$FULL_CKPT"
    echo "  → Stage 8 future_mse will use this (carries 'predictor' key)."
else
    echo "  ⚠️  $FULL_CKPT NOT produced — Stage 8 future_mse will FATAL for ${SUBCMD}"
fi
