#!/usr/bin/env bash
# iter13 probe encoder trainer — produces vjepa_2_1_pretrain_encoder (P2) + vjepa_2_1_surgical (P3)
# checkpoints that scripts/run_eval.sh consumes for the 4-encoder paired-Δ
# action-probe gate. Mirrors the run_train.sh / run_eval.sh split (iter11 v2).
#
# Subcommands:
#   pretrain          — m09a continual SSL pretrain (LR 1e-5, 5 epochs, drift L2 anchor)
#   surgery_3stage_DI — m09c factor surgery WITH interaction tubes (D_L → D_A → D_I, 40/30/30 split)
#   surgery_noDI      — m09c factor surgery WITHOUT D_I  (D_L → D_A,    50/50 split)
#
# Why two surgery variants? iter9 v15c showed Stage 3 (D_I unfreeze 75%) caused
# downstream-metric BWT=-0.33 (legacy retrieval score dropped 20.50→19.83).
# Whether D_I helps under the iter13 motion-flow probe-accuracy metric is
# genuinely open — train both and let probe_eval's paired-Δ decide.
#
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python" — this
# wrapper only:
#   1. Pre-flights that probe Stage 1 ran (action_labels.json must exist)
#   2. Generates eval_10k_{train,val}_split.json via src/utils/probe_train_subset.py
#   3. Dispatches m09a or m09c with the right yaml + output dir
#
# USAGE:
#   tmux new -s probe_train
#   ./scripts/run_train.sh pretrain          --SANITY   # ~3 min
#   ./scripts/run_train.sh surgery_3stage_DI --SANITY   # ~10 min
#   ./scripts/run_train.sh surgery_noDI      --SANITY   # ~7 min  (no Stage 3)
#   ./scripts/run_train.sh pretrain          --FULL     # ~3 GPU-h
#   ./scripts/run_train.sh surgery_3stage_DI --FULL     # ~6-8 GPU-h
#   ./scripts/run_train.sh surgery_noDI      --FULL     # ~4-6 GPU-h
#
# Bypass the cache-policy prompt for overnight runs:
#   CACHE_POLICY_ALL=1 ./scripts/run_train.sh pretrain --FULL  # keep stale outputs
#   CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain --FULL  # wipe + recompute

set -euo pipefail
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_train.sh aborted at line $LINENO (exit=$rc)" >&2; exit $rc' ERR

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

if [ $# -lt 2 ]; then
    echo "USAGE: $0 {pretrain|pretrain_2X|pretrain_head|surgery_3stage_DI|surgery_noDI|surgery_3stage_DI_head|surgery_noDI_head} {--SANITY|--POC|--FULL}" >&2
    exit 2
fi

SUBCMD="$1"; shift
MODE_FLAG="$1"; shift

# iter15 Phase 4 (2026-05-14): added 3 new head-only SUBCMDs (pretrain_head,
# surgery_3stage_DI_head, surgery_noDI_head) wrapping m09a2_pretrain_head.py
# + m09c2_surgery_head.py. The *_head variants run on 24 GB Pro 4000 (no
# encoder backward → no activation storage); _encoder variants need 96 GB.
case "$SUBCMD" in
    pretrain|pretrain_2X|pretrain_head| \
    surgery_3stage_DI|surgery_noDI| \
    surgery_3stage_DI_head|surgery_noDI_head) ;;
    *) echo "FATAL: subcommand must be {pretrain|pretrain_2X|pretrain_head|surgery_3stage_DI|surgery_noDI|surgery_3stage_DI_head|surgery_noDI_head} (got: $SUBCMD)" >&2; exit 2 ;;
esac

case "$MODE_FLAG" in
    --SANITY|--sanity) MODE="SANITY"; mode_dir="sanity" ;;
    --POC|--poc)       MODE="POC";    mode_dir="poc" ;;
    --FULL|--full)     MODE="FULL";   mode_dir="full" ;;
    *) echo "FATAL: mode flag must be --SANITY|--POC|--FULL (got: $MODE_FLAG)" >&2; exit 2 ;;
esac

# ── Probe Stage 1 (action_labels.json) — auto-bootstrap before split ─────
# iter14 recipe-v3 (2026-05-09): TECH-DEBT — this shell-level bootstrap is
# REDUNDANT with src/utils/probe_labels.ensure_probe_labels_for_mode(cfg=cfg),
# which m09a/m09c already call at startup (in-process, reads cfg, no shell
# orchestration). It's kept here ONLY because probe_train_subset.py at lines
# ~93-97 below runs BEFORE m09a/m09c and depends on ACTION_LABELS existing.
# Full refactor — option 🅱 in iter/iter14_surgery_on_pretrain/plan_no_discrepancy.md
# § "Phase E (NEW)" — moves probe_train_subset.py invocation into m09a/m09c
# (via in-process split_subset() calls) so this shell block can be deleted.
# Until then: this shell calls probe_labels-equivalent logic via the same
# CLI surface (subprocess to probe_action.py + eval_subset.py) for parity.
ACTION_LABELS="outputs/${mode_dir}/probe_action/action_labels.json"
if [ ! -f "$ACTION_LABELS" ]; then
    echo "  [run_probe_train] $ACTION_LABELS missing — auto-bootstrapping via probe_action.py --stage labels (CPU, ~1 min)"
    LOCAL_DATA_BOOTSTRAP="data/eval_10k_local"
    MOTION_FEATURES_BOOTSTRAP="${LOCAL_DATA_BOOTSTRAP}/m04d_motion_features/motion_features.npy"
    if [ ! -f "$MOTION_FEATURES_BOOTSTRAP" ]; then
        echo "❌ FATAL: $MOTION_FEATURES_BOOTSTRAP not found — run m04d_motion_features.py first" >&2
        exit 3
    fi
    if [ "$MODE" = "SANITY" ]; then
        EVAL_SUBSET_BOOTSTRAP="data/eval_10k_local/eval_10k_sanity.json"
        # iter13 v13 (2026-05-07): floor=3 is the absolute minimum that
        # stratified_split's greedy allocation supports (val=1/test=1/train=1).
        MIN_CLIPS_BOOTSTRAP=3
        MIN_SPLIT_BOOTSTRAP=1
    elif [ "$MODE" = "POC" ]; then
        # iter14 v2 (2026-05-09): POC = STRATIFIED by motion class (RAFT optical
        # flow → 8 classes), guarantees POC labels file matches FULL schema (all
        # 8 classes preserved). Replaces buggy --first-n which caused iter14 D₂
        # 855/7-class label file. Per-class target = POC_TOTAL / 8 (8 surviving
        # motion classes from m04d 13-D RAFT × magnitude-quartile × direction).
        # Source: src/CLAUDE.md POC↔FULL parity rule + plan_surgery_wins.md §12.7.
        POC_SUBSET="data/eval_10k_local/eval_10k_poc.json"
        POC_TOTAL=$(scripts/lib/yaml_extract.py configs/train/base_optimization.yaml data.poc_total_clips)
        TARGET_PER_CLASS=$((POC_TOTAL / 8))
        MOTION_FEATURES_FULL="data/eval_10k_local/m04d_motion_features/motion_features.npy"
        if [ ! -f "$MOTION_FEATURES_FULL" ]; then
            echo "❌ FATAL: $MOTION_FEATURES_FULL missing — run m04d --FULL first." >&2
            exit 4
        fi
        if [ ! -f "$POC_SUBSET" ] || [ "data/eval_10k_local/eval_10k.json" -nt "$POC_SUBSET" ] || [ "$MOTION_FEATURES_FULL" -nt "$POC_SUBSET" ]; then
            python -u src/utils/eval_subset.py \
                --eval-subset data/eval_10k_local/eval_10k.json \
                --stratified-by-motion-class \
                --motion-features "$MOTION_FEATURES_FULL" \
                --target-per-class "$TARGET_PER_CLASS" \
                --output "$POC_SUBSET"
        fi
        EVAL_SUBSET_BOOTSTRAP="$POC_SUBSET"
        # POC↔FULL parity (CLAUDE.md, 2026-05-09): POC uses SAME min_clips_per_class
        # as FULL (34) so identical class set survives the post-stratification filter.
        MIN_CLIPS_BOOTSTRAP=34
        MIN_SPLIT_BOOTSTRAP=5
    else
        EVAL_SUBSET_BOOTSTRAP="data/eval_10k_local/eval_10k.json"
        MIN_CLIPS_BOOTSTRAP=34
        MIN_SPLIT_BOOTSTRAP=5
    fi
    python -u src/probe_action.py "${MODE_FLAG}" \
        --stage labels \
        --eval-subset "$EVAL_SUBSET_BOOTSTRAP" \
        --motion-features "$MOTION_FEATURES_BOOTSTRAP" \
        --min-clips-per-class "$MIN_CLIPS_BOOTSTRAP" \
        --min-per-split "$MIN_SPLIT_BOOTSTRAP" \
        --output-root "outputs/${mode_dir}/probe_action" \
        --cache-policy "${CACHE_POLICY_ALL:-1}" \
        --no-wandb \
        2>&1 | tee "logs/probe_action_labels_${mode_dir}.log"
fi

# ── Pre-flight: bitsandbytes for SANITY 8-bit optim path ────────────────
if [ "$MODE" = "SANITY" ]; then
    if ! python -c "import bitsandbytes" 2>/dev/null; then
        echo "❌ FATAL: bitsandbytes not installed (required for SANITY 24 GB 8-bit AdamW)." >&2
        echo "  Run: pip install bitsandbytes" >&2
        exit 3
    fi
fi

# ── Generate train/val/test split JSONs from action_labels.json ──────────
# action_labels.json carries 70/15/15 split (train=6964, val=1491, test=1496)
# — paper-final m06d trio uses TEST clips that m09a never sees during pretrain.
# iter13 Task #23 (2026-05-04): added test split externalisation here so eval
# stages downstream can reference data/eval_10k_local/eval_10k_test_split.json directly.
TRAIN_SPLIT="data/eval_10k_local/eval_10k_train_split.json"
VAL_SPLIT="data/eval_10k_local/eval_10k_val_split.json"
TEST_SPLIT="data/eval_10k_local/eval_10k_test_split.json"
echo "═══ $(date '+%H:%M:%S') · Generating train/val/test split JSONs ═══"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split train --output "$TRAIN_SPLIT"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split val --output "$VAL_SPLIT"
python -u src/utils/probe_train_subset.py \
    --action-labels "$ACTION_LABELS" --split test --output "$TEST_SPLIT"

LOCAL_DATA="data/eval_10k_local"
[ -d "$LOCAL_DATA" ] || { echo "❌ FATAL: $LOCAL_DATA missing"; exit 3; }
MODEL_CFG="configs/model/vjepa2_1.yaml"
P_M09="${CACHE_POLICY_ALL:-1}"

# iter14 (2026-05-08): canonical HF endpoint for surgery init. Single source of
# truth — m09c surgery requires --init-from-ckpt with this URI (FAIL LOUD per
# CLAUDE.md). Hardcoded here per "shells ARE the layer that pin canonical paths".
# HF_TOKEN must be in .env (project root); m09c calls hf_hub_download with it.
#
# Why m09a_ckpt_best.pt (14 GB) NOT student_encoder.pt (7 GB):
# m09c needs BOTH student weights (key="student", 588 dims) AND predictor weights
# (key="predictor", 300 dims). student_encoder.pt has only encoder
# (schema="student_state_dict"); m09a_ckpt_best.pt has student + predictor +
# teacher in one bundle (schema="student") — single download covers full init.
PRETRAIN_HF_URI="hf://anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep/m09a_ckpt_best.pt"

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
# eval_subset path mirrors run_eval.sh's mode-gated convention.
if [ "$MODE" = "SANITY" ]; then
    EVAL_SUBSET_TX="data/eval_10k_local/eval_10k_sanity.json"
else
    EVAL_SUBSET_TX="data/eval_10k_local/eval_10k.json"
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
            --no-wandb \
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
    pretrain|pretrain_2X)
        # iter13 v12+ (2026-05-06): renamed probe_pretrain → m09a_pretrain_encoder to
        # match the source module's name (CLAUDE.md "m*.py = each module is
        # independent"). Mirror rename in run_eval.sh + yaml output_dir.
        # iter14 (2026-05-08): pretrain_2X shares pretrain_encoder.yaml (no new
        # yamls) but writes to m09a_pretrain_2X_encoder/ + passes --max-epochs 10 for
        # FULL only (G1=🅰️ "5+5 vs 10"). Arm A = 5 ep (yaml's max_epochs.full);
        # Arm C = 10 ep via CLI override.
        if [ "$SUBCMD" = "pretrain_2X" ]; then
            OUT_DIR="outputs/${mode_dir}/m09a_pretrain_2X_encoder"
            EPOCHS_OVERRIDE_FLAG=""
            if [ "$MODE" = "FULL" ]; then
                EPOCHS_OVERRIDE_FLAG="--max-epochs 10"   # iter14 G1=🅰️
            fi
        else
            OUT_DIR="outputs/${mode_dir}/m09a_pretrain_encoder"
            EPOCHS_OVERRIDE_FLAG=""
        fi
        TRAIN_CFG="configs/train/pretrain_encoder.yaml"
        # Read lambda_reg from YAML so it stays the single source of truth.
        # Passing it explicitly to m09a bypasses the legacy auto-ablation gate
        # (m09a1_pretrain_encoder.py:1187 enters EWC sweep when --lambda-reg is unset;
        # pretrain_encoder.yaml intentionally has ablation_lambdas=[] which would
        # then trip select_ablation_winner's non-empty assertion).
        LAMBDA_REG=$(scripts/lib/yaml_extract.py "$TRAIN_CFG" drift_control.lambda_reg)
        echo "═══ $(date '+%H:%M:%S') · m09a continual SSL ${SUBCMD} (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_SPLIT"
        echo "  val:       $VAL_SPLIT"
        echo "  local:     $LOCAL_DATA"
        echo "  output:    $OUT_DIR"
        echo "  lambda:    $LAMBDA_REG (read from yaml; bypasses auto-ablation gate)"
        if [ -n "$EPOCHS_OVERRIDE_FLAG" ]; then
            echo "  epochs:    $EPOCHS_OVERRIDE_FLAG (iter14 arm C — overrides yaml's full=5)"
        fi
        mkdir -p "$OUT_DIR"
        python -u src/m09a1_pretrain_encoder.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_SPLIT" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --lambda-reg "$LAMBDA_REG" \
            $EPOCHS_OVERRIDE_FLAG \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "${LOCAL_DATA}/tags.json" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09a_${SUBCMD}_${mode_dir}.log"
        ;;
    surgery_3stage_DI|surgery_noDI)
        # Map subcommand → yaml + variant tag (used in output dir + log name).
        case "$SUBCMD" in
            surgery_3stage_DI)
                TRAIN_CFG="configs/train/surgery_3stage_DI_encoder.yaml"
                VARIANT_TAG="3stage_DI"
                ;;
            surgery_noDI)
                TRAIN_CFG="configs/train/surgery_2stage_noDI_encoder.yaml"
                VARIANT_TAG="noDI"
                ;;
        esac
        # iter13 v12+ (2026-05-06): renamed probe_surgery_* → m09c_surgery_* to
        # match m09c1_surgery_encoder.py module name. Mirror in run_eval.sh.
        OUT_DIR="outputs/${mode_dir}/m09c_surgery_${VARIANT_TAG}"
        # iter13 v12+ Task 3 (2026-05-06): m11 outputs co-located with input
        # under <--local-data>/m11_factor_datasets/. m11 derives this default
        # itself; this consumer just mirrors the same convention.
        FACTOR_DIR="${LOCAL_DATA}/m11_factor_datasets"
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
        # iter14 recipe-v2 (2026-05-09): env-var overrides for POC sweep cells.
        # Forwards as CLI flags to m09c. Both unset → m09c uses yaml defaults
        # (teacher_mode=EMA, lp_ft_stage0.enabled=false) → legacy behavior verbatim.
        # See iter/iter14_surgery_on_pretrain/runbook.md § Recipe-v2 POC sweep.
        RECIPE_V2_ARGS=()
        if [ -n "${TEACHER_MODE_OVERRIDE:-}" ]; then
            case "$TEACHER_MODE_OVERRIDE" in
                EMA|FROZEN)
                    RECIPE_V2_ARGS+=(--teacher-mode "$TEACHER_MODE_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: TEACHER_MODE_OVERRIDE must be EMA|FROZEN (got: $TEACHER_MODE_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi
        if [ -n "${LP_FT_OVERRIDE:-}" ]; then
            case "$LP_FT_OVERRIDE" in
                on|off)
                    RECIPE_V2_ARGS+=(--lp-ft-stage0 "$LP_FT_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: LP_FT_OVERRIDE must be on|off (got: $LP_FT_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi
        # iter14 recipe-v3 (2026-05-09): five additional ablation switches for
        # scripts/run_recipe_v3_sweep.sh drop-one cells. Same env-var → CLI
        # forwarding pattern as recipe-v2 above. Each unset → m09c keeps yaml
        # default (recipe-v3 defaults: subset=recipe_v3, warmup=per_stage,
        # saliency=off, spd=off, replay=off — all gated until enabled).
        if [ -n "${SUBSET_OVERRIDE:-}" ]; then
            case "$SUBSET_OVERRIDE" in
                legacy|recipe_v3)
                    RECIPE_V2_ARGS+=(--subset-mode "$SUBSET_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: SUBSET_OVERRIDE must be legacy|recipe_v3 (got: $SUBSET_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi
        if [ -n "${WARMUP_OVERRIDE:-}" ]; then
            case "$WARMUP_OVERRIDE" in
                per_stage|single)
                    RECIPE_V2_ARGS+=(--warmup-mode "$WARMUP_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: WARMUP_OVERRIDE must be per_stage|single (got: $WARMUP_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi
        if [ -n "${SALIENCY_OVERRIDE:-}" ]; then
            case "$SALIENCY_OVERRIDE" in
                on|off)
                    RECIPE_V2_ARGS+=(--saliency "$SALIENCY_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: SALIENCY_OVERRIDE must be on|off (got: $SALIENCY_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi
        if [ -n "${SPD_OVERRIDE:-}" ]; then
            case "$SPD_OVERRIDE" in
                on|off)
                    RECIPE_V2_ARGS+=(--spd "$SPD_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: SPD_OVERRIDE must be on|off (got: $SPD_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi
        if [ -n "${REPLAY_OVERRIDE:-}" ]; then
            case "$REPLAY_OVERRIDE" in
                on|off)
                    RECIPE_V2_ARGS+=(--replay "$REPLAY_OVERRIDE")
                    ;;
                *)
                    echo "❌ FATAL: REPLAY_OVERRIDE must be on|off (got: $REPLAY_OVERRIDE)" >&2
                    exit 2
                    ;;
            esac
        fi

        echo "═══ $(date '+%H:%M:%S') · m09c factor surgery [variant=${VARIANT_TAG}] (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_SPLIT"
        echo "  factor:    $FACTOR_DIR"
        echo "  output:    $OUT_DIR"
        echo "  init:      $PRETRAIN_HF_URI (iter14 — m09c hf_hub_download via HF_TOKEN from .env)"
        if [ ${#RECIPE_V2_ARGS[@]} -gt 0 ]; then
            echo "  recipe-v2/v3 overrides: ${RECIPE_V2_ARGS[*]}"
            echo "    (env vars: TEACHER_MODE_OVERRIDE / LP_FT_OVERRIDE / SUBSET_OVERRIDE /"
            echo "     WARMUP_OVERRIDE / SALIENCY_OVERRIDE / SPD_OVERRIDE / REPLAY_OVERRIDE)"
        else
            echo "  recipe-v2/v3: <yaml defaults>"
        fi
        mkdir -p "$OUT_DIR"
        # iter14 (2026-05-08): --init-from-ckpt is REQUIRED in m09c (argparse
        # required=True). Always pass the HF URI — single source per CLAUDE.md
        # FAIL LOUD. m09c downloads via hf_hub_download (cached in HF_HOME after
        # first call; subsequent surgery runs hit cache instantly).
        python -u src/m09c1_surgery_encoder.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_SPLIT" --local-data "$LOCAL_DATA" \
            --factor-dir "$FACTOR_DIR" \
            --probe-subset "$VAL_SPLIT" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "$VAL_TAGS" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --init-from-ckpt "$PRETRAIN_HF_URI" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            "${RECIPE_V2_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09c_surgery_${VARIANT_TAG}_${mode_dir}.log"
        ;;
    pretrain_head)
        # iter15 Phase 4 (2026-05-14): head-only m09a2. Frozen encoder + frozen
        # predictor; only the ~432K motion_aux head trains. 24 GB sufficient.
        OUT_DIR="outputs/${mode_dir}/m09a_pretrain_head"
        TRAIN_CFG="configs/train/pretrain_head.yaml"
        echo "═══ $(date '+%H:%M:%S') · m09a2 HEAD-ONLY continual SSL (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_SPLIT"
        echo "  val:       $VAL_SPLIT"
        echo "  local:     $LOCAL_DATA"
        echo "  output:    $OUT_DIR"
        echo "  contract:  all 48 ViT-G blocks + predictor FROZEN; trainable = motion_aux head"
        mkdir -p "$OUT_DIR"
        python -u src/m09a2_pretrain_head.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_SPLIT" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "${LOCAL_DATA}/tags.json" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09a2_pretrain_head_${mode_dir}.log"
        ;;
    surgery_3stage_DI_head|surgery_noDI_head)
        # iter15 Phase 4 (2026-05-14): head-only m09c2. Same freeze contract as
        # pretrain_head + StreamingFactorDataset for factor-aug clips. Single
        # head-only stage (no progressive unfreeze).
        case "$SUBCMD" in
            surgery_3stage_DI_head)
                TRAIN_CFG="configs/train/surgery_3stage_DI_head.yaml"
                VARIANT_TAG="3stage_DI_head"
                ;;
            surgery_noDI_head)
                TRAIN_CFG="configs/train/surgery_2stage_noDI_head.yaml"
                VARIANT_TAG="noDI_head"
                ;;
        esac
        OUT_DIR="outputs/${mode_dir}/m09c_surgery_${VARIANT_TAG}"
        FACTOR_DIR="${LOCAL_DATA}/m11_factor_datasets"
        VAL_TAGS="${LOCAL_DATA}/tags.json"
        if [ ! -d "$FACTOR_DIR" ]; then
            echo "❌ FATAL: $FACTOR_DIR missing — run scripts/run_factor_prep.sh first" >&2
            exit 3
        fi
        echo "═══ $(date '+%H:%M:%S') · m09c2 HEAD-ONLY surgery [variant=${VARIANT_TAG}] (${MODE}) ═══"
        echo "  config:    $TRAIN_CFG"
        echo "  subset:    $TRAIN_SPLIT"
        echo "  factor:    $FACTOR_DIR"
        echo "  output:    $OUT_DIR"
        echo "  contract:  all 48 ViT-G blocks + predictor FROZEN; trainable = motion_aux head"
        mkdir -p "$OUT_DIR"
        # m09c2 derives factor_manifest from --local-data internally (see m09c2.py:284
        # `manifest_path = Path(local_data) / "m11_factor_datasets" / "factor_manifest.json"`).
        # Unlike m09c1 it doesn't accept --factor-dir CLI — StreamingFactorDataset-only path.
        python -u src/m09c2_surgery_head.py "${MODE_FLAG}" \
            --model-config "$MODEL_CFG" \
            --train-config "$TRAIN_CFG" \
            --subset "$TRAIN_SPLIT" --local-data "$LOCAL_DATA" \
            --val-subset "$VAL_SPLIT" --val-local-data "$LOCAL_DATA" \
            --output-dir "$OUT_DIR" \
            --cache-policy "$P_M09" \
            --probe-subset "outputs/${mode_dir}/probe_action/action_labels.json" \
            --probe-local-data "$LOCAL_DATA" \
            --probe-tags "$VAL_TAGS" \
            --probe-action-labels "outputs/${mode_dir}/probe_action/action_labels.json" \
            --motion-features-path "${LOCAL_DATA}/m04d_motion_features/motion_features.npy" \
            "${TAXONOMY_ARGS[@]}" \
            --no-wandb \
            2>&1 | tee "logs/m09c2_surgery_${VARIANT_TAG}_${mode_dir}.log"
        ;;
esac

# ── Verify expected outputs ──────────────────────────────────────────────
echo ""
echo "═══ $(date '+%H:%M:%S') · DONE ═══"
echo "Expected outputs (consumed by scripts/run_eval.sh):"
if [ -f "${OUT_DIR}/student_encoder.pt" ]; then
    ls -lh "${OUT_DIR}/student_encoder.pt"
else
    echo "  ⚠️  ${OUT_DIR}/student_encoder.pt NOT produced (check logs/probe_${SUBCMD}_${mode_dir}.log)"
fi
case "$SUBCMD" in
    pretrain|pretrain_2X|pretrain_head)
        FULL_CKPT="${OUT_DIR}/m09a_ckpt_best.pt"
        ;;
    surgery_3stage_DI|surgery_noDI|surgery_3stage_DI_head|surgery_noDI_head)
        FULL_CKPT="${OUT_DIR}/m09c_ckpt_best.pt"
        ;;
esac
if [ -f "$FULL_CKPT" ]; then
    ls -lh "$FULL_CKPT"
    echo "  → Stage 8 future_mse will use this (carries 'predictor' key)."
else
    echo "  ⚠️  $FULL_CKPT NOT produced — Stage 8 future_mse will FATAL for ${SUBCMD}"
fi
