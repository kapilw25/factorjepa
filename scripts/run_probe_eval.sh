#!/usr/bin/env bash
# iter13 probe orchestrator — 3 modules × {V-JEPA, DINOv2} × stages.
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# and "DELETE PROTECTION — shells stay THIN, .py owns the cache-policy prompt".
#
# Pipeline (priority 1 — frozen V-JEPA vs frozen DINOv2 on Indian action probe
# applied to data/eval_10k_local):
#   STAGE 1   probe_action.py --stage labels        (CPU, ~1 min)
#   STAGE 2   probe_action.py --stage features      (GPU × 2 encoders, ~1 h)
#   STAGE 3   probe_action.py --stage train         (GPU × 2 encoders, ~30 min)
#   STAGE 4   probe_action.py --stage paired_delta  (CPU, ~5 min)  ← P1 GATE
#   STAGE 5   probe_motion_cos.py   --stage features      (CPU mean-pool × 2 enc)
#   STAGE 6   probe_motion_cos.py   --stage cosine        (CPU × 2 enc)
#   STAGE 7   probe_motion_cos.py   --stage paired_delta  (CPU)
#   STAGE 8   probe_future_mse.py   --stage forward       (GPU, V-JEPA only, ~30 min)
#   STAGE 9   probe_future_mse.py   --stage paired_per_variant   (CPU)
#   STAGE 10  probe_plot.py                                  (CPU, ~5s — plots)
#
# REFERENCES
#   plan_code_dev.md  — per-module specs + LoC budget
#   runbook.md        — full set of pre/post-flight one-liners
#
# USAGE
#   tmux new -s probe
#
#   # FULL run (default) — eval_10k (~9.9k clips), ~4h on 24GB / ~2.5h on 96GB
#   ./scripts/run_probe_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log
#
#   # SANITY smoke test — 150 stratified clips (50/class) from THE SAME eval_10k
#   # JSON, processed against the SAME eval_10k_local/ TARs. ~6-8 min on 24GB.
#   # Outputs sandboxed to outputs/sanity/. Pass --sanity OR set MODE=SANITY.
#   ./scripts/run_probe_eval.sh --sanity 2>&1 | tee logs/run_src_probe_sanity_v1.log
#
#   # Bypass prompts (overnight / non-TTY)
#   CACHE_POLICY_ALL=1 ./scripts/run_probe_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log           # keep all caches
#   CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log           # recompute everything
#   CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity 2>&1 | tee logs/run_src_probe_sanity_v1.log  # SANITY + recompute
#
#   # Skip a stage (resume after failure):
#   SKIP_STAGES="1,2" ./scripts/run_probe_eval.sh
#   # Run only one variant (debug):
#   ENCODERS="vjepa_2_1_frozen" ./scripts/run_probe_eval.sh
#   # Tune SANITY subset size (default 50 clips per class):
#   SANITY_N_PER_CLASS=20 ./scripts/run_probe_eval.sh --sanity
#
#   # Probe-training knobs (Stage 3) — mode-aware defaults:
#   #   SANITY: EPOCHS=50  WARMUP_PCT=0.10  LR_SWEEP="5e-4"   (current behavior)
#   #   FULL:   EPOCHS=20  WARMUP_PCT=0.0   LR_SWEEP="5e-4"   (Meta-faithful single-LR)
#   # Override any of them via env vars:
#   EPOCHS=20 WARMUP_PCT=0.0 ./scripts/run_probe_eval.sh --sanity   # apply Meta recipe to sanity
#
#   # Paper-faithful FULL with LR sweep (matches Meta's multihead 5-LR setup;
#   # ~4x stage-3 wall, but mirrors deps/vjepa2/configs/eval/vitg-384/ssv2.yaml):
#   LR_SWEEP="1e-4 3e-4 1e-3 3e-3" ./scripts/run_probe_eval.sh
#   # Each LR trains a probe under outputs/full/probe_action/<encoder>/lr_<LR>/;
#   # the best-val-acc LR is symlinked to the canonical <encoder>/probe.pt path so
#   # Stage 4 paired_delta reads the winner (no code change needed).

# Fail-fast: probe stages have sequential dependencies (Stage N+1 needs N's outputs),
# so any stage failure → abort chain to avoid wasting GPU time on impossible
# downstream work. Differs from run_paired_eval_10k.sh which uses `set -uo` only
# because its variants are independent (errors_N_fixes #72).
set -euo pipefail
# Clear ERR trap surfaces line + exit code BEFORE the shell exits.
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_probe_eval.sh aborted at line $LINENO (exit=$rc) — sequential dependency failed; downstream stages skipped" >&2; exit $rc' ERR

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

# ── Mode detection (CLI flag wins; falls back to MODE env-var; default FULL) ──
MODE="${MODE:-FULL}"
for arg in "$@"; do
    case "$arg" in
        --sanity|--SANITY) MODE="SANITY" ;;
        --poc|--POC)       MODE="POC" ;;
        --full|--FULL)     MODE="FULL" ;;
    esac
done
case "$MODE" in
    SANITY|POC|FULL) ;;
    *) echo "FATAL: MODE must be SANITY|POC|FULL (got: $MODE)" >&2; exit 2 ;;
esac

# ── Mode-gated defaults ─────────────────────────────────────────────────
# SANITY: same eval_10k TARs + same encoder + same NUM_FRAMES as FULL —
# we ONLY shrink the clip_keys list (50/class stratified) and sandbox outputs.
# This guarantees zero codec/path/dtype drift when scaling SANITY → FULL.
# Probe-training defaults (Stage 3) — mode-aware to match Meta's recipe at FULL.
# SANITY: 50 epochs + 10% warmup + single LR (current behavior, code correctness).
# FULL:   20 epochs + 0% warmup + optional LR sweep (matches deps/vjepa2/configs/eval/vitg-384/ssv2.yaml).
if [ "$MODE" = "SANITY" ]; then
    DEFAULT_EPOCHS=50
    DEFAULT_WARMUP_PCT=0.10
    DEFAULT_LR_SWEEP="5e-4"   # single LR
else
    DEFAULT_EPOCHS=20
    DEFAULT_WARMUP_PCT=0.0
    DEFAULT_LR_SWEEP="5e-4"   # default still single LR; set LR_SWEEP="1e-4 3e-4 1e-3 3e-3" for paper-faithful sweep
fi
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
WARMUP_PCT="${WARMUP_PCT:-$DEFAULT_WARMUP_PCT}"
LR_SWEEP="${LR_SWEEP:-$DEFAULT_LR_SWEEP}"

if [ "$MODE" = "SANITY" ]; then
    SANITY_N_PER_CLASS="${SANITY_N_PER_CLASS:-50}"
    SANITY_SUBSET="data/eval_10k_sanity.json"
    # Regenerate if missing OR older than the source eval_10k.json (idempotent + cheap).
    # Logic lives in src/utils/eval_subset.py (importable + CLI; per CLAUDE.md
    # "shell scripts are THIN wrappers — all logic in Python").
    if [ ! -f "$SANITY_SUBSET" ] || [ "data/eval_10k.json" -nt "$SANITY_SUBSET" ]; then
        python -u src/utils/eval_subset.py \
            --eval-subset data/eval_10k.json \
            --n-per-class "$SANITY_N_PER_CLASS" \
            --output "$SANITY_SUBSET"
    fi
    DEFAULT_EVAL_SUBSET="$SANITY_SUBSET"
    DEFAULT_OUTPUT_PREFIX="outputs/sanity"
else
    DEFAULT_EVAL_SUBSET="data/eval_10k.json"
    DEFAULT_OUTPUT_PREFIX="outputs/full"
fi

# ── Configurables (env-overridable; mode-gated defaults) ────────────────
EVAL_SUBSET="${EVAL_SUBSET:-$DEFAULT_EVAL_SUBSET}"
LOCAL_DATA="${LOCAL_DATA:-data/eval_10k_local}"
TAGS_JSON="${TAGS_JSON:-${LOCAL_DATA}/tags.json}"
ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/vjepa2_1_vitG_384.pt}"
OUTPUT_ACTION="${OUTPUT_ACTION:-${DEFAULT_OUTPUT_PREFIX}/probe_action}"
OUTPUT_COS="${OUTPUT_COS:-${DEFAULT_OUTPUT_PREFIX}/probe_motion_cos}"
OUTPUT_MSE="${OUTPUT_MSE:-${DEFAULT_OUTPUT_PREFIX}/probe_future_mse}"
OUTPUT_TAXONOMY="${OUTPUT_TAXONOMY:-${DEFAULT_OUTPUT_PREFIX}/probe_taxonomy}"
OUTPUT_PLOTS="${OUTPUT_PLOTS:-${DEFAULT_OUTPUT_PREFIX}/probe_plot}"
TAG_TAXONOMY="${TAG_TAXONOMY:-configs/tag_taxonomy.json}"
ENCODERS="${ENCODERS:-vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_surgical_3stage_DI vjepa_2_1_surgical_noDI}"
SKIP_STAGES="${SKIP_STAGES:-}"
NUM_FRAMES="${NUM_FRAMES:-16}"

# Per-encoder checkpoint resolvers. Two functions because Stages 2/3 (probe-head
# training) need encoder-only ckpts, but Stage 8 (future_mse) also needs the
# predictor. probe_pretrain / probe_surgery_* write BOTH artifacts:
#   - student_encoder.pt    : encoder only      (export_student_for_eval)
#   - m09{a,c}_ckpt_best.pt : encoder+predictor (save_training_checkpoint full=True)
# Surgery has TWO variants — 3stage_DI (with interaction tubes) and noDI (without)
# — to test the skepticism that D_I worsens learning. Each writes its own dir.
encoder_ckpt_for() {                                            # encoder-only — Stages 2/3
    case "$1" in
        vjepa_2_1_frozen)              echo "$ENCODER_CKPT" ;;
        vjepa_2_1_pretrain)            echo "${DEFAULT_OUTPUT_PREFIX}/probe_pretrain/student_encoder.pt" ;;
        vjepa_2_1_surgical_3stage_DI)  echo "${DEFAULT_OUTPUT_PREFIX}/probe_surgery_3stage_DI/student_encoder.pt" ;;
        vjepa_2_1_surgical_noDI)       echo "${DEFAULT_OUTPUT_PREFIX}/probe_surgery_noDI/student_encoder.pt" ;;
        *) echo "" ;;
    esac
}
encoder_predictor_ckpt_for() {                                  # encoder+predictor — Stage 8 future_mse
    case "$1" in
        vjepa_2_1_frozen)              echo "$ENCODER_CKPT" ;;
        vjepa_2_1_pretrain)            echo "${DEFAULT_OUTPUT_PREFIX}/probe_pretrain/m09a_ckpt_best.pt" ;;
        vjepa_2_1_surgical_3stage_DI)  echo "${DEFAULT_OUTPUT_PREFIX}/probe_surgery_3stage_DI/m09c_ckpt_best.pt" ;;
        vjepa_2_1_surgical_noDI)       echo "${DEFAULT_OUTPUT_PREFIX}/probe_surgery_noDI/m09c_ckpt_best.pt" ;;
        *) echo "" ;;
    esac
}

T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }
should_skip() {
    local stage="$1"
    [[ ",${SKIP_STAGES}," == *",${stage},"* ]]
}

# ── Pre-flight ──────────────────────────────────────────────────────────
stamp "PRE-FLIGHT"

# V-JEPA 2.1 ViT-G ckpt sanity hint (~28 GB, comes from setup_env_uv.sh aria2c).
# Friendly ✓/✗ before the generic FATAL loop below — points user to the fix.
ls -lh "$ENCODER_CKPT" && echo "✓ ckpt present" || echo "✗ run setup_env_uv.sh first"

for f in "$EVAL_SUBSET" "$ENCODER_CKPT"; do
    if [ ! -e "$f" ]; then
        echo "FATAL: required path missing: $f"
        exit 3
    fi
done
if [ ! -d "$LOCAL_DATA" ]; then
    echo "FATAL: --local-data dir missing: $LOCAL_DATA"
    exit 3
fi
if [ ! -f "$TAGS_JSON" ]; then
    echo "  WARN: $TAGS_JSON not found — --enable-monument-class will not work (3-class default OK)"
fi
echo "  ✓ mode:            $MODE"
echo "  ✓ eval_subset:     $EVAL_SUBSET"
echo "  ✓ local_data:      $LOCAL_DATA"
echo "  ✓ encoder_ckpt:    $ENCODER_CKPT  ($(du -h "$ENCODER_CKPT" 2>/dev/null | awk '{print $1}'))"
echo "  ✓ encoders:        $ENCODERS"
echo "  ✓ skip_stages:     ${SKIP_STAGES:-<none>}"
echo "  ✓ cache_policy:    ${CACHE_POLICY_ALL:-prompt-per-module}"
echo "  ✓ output_prefix:   $DEFAULT_OUTPUT_PREFIX"

# ── Cache-policy gather (UPFRONT — per scripts/delete_protection_for_each_variant.md) ──
# MUST: collect ALL decisions BEFORE compute starts. Mirrors run_train.sh:43-89.
# 3 prompts (one per output dir) instead of 9 .py-level input() prompts.
# Bypass: CACHE_POLICY_ALL=1|2 env. Non-TTY: silent default to 1.
declare -A POLICY
_check_and_prompt() {                  # $1=key  $2..=candidate cache paths/globs
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        # Use bash nullglob to expand $path; non-matching globs become empty
        # array, avoiding `compgen -G`'s non-zero exit which would trip set -e.
        shopt -s nullglob
        local matches=( $path )
        shopt -u nullglob
        if [ ${#matches[@]} -gt 0 ]; then
            found="${matches[0]}"
            break
        fi
    done
    if [ -z "$found" ]; then POLICY[$key]=1; return; fi
    if [ -n "${CACHE_POLICY_ALL:-}" ]; then
        POLICY[$key]=$CACHE_POLICY_ALL
        echo "  $key: cache at $found → policy=${POLICY[$key]} (CACHE_POLICY_ALL)"
        return
    fi
    if [ ! -t 0 ]; then
        POLICY[$key]=1
        echo "  $key: cache at $found → policy=1 (non-TTY default)"
        return
    fi
    local ans
    read -p "  $key cache at $found [1=keep / 2=recompute] (Enter=1): " ans
    case "${ans:-1}" in
        2|recompute) POLICY[$key]=2 ;;
        *)           POLICY[$key]=1 ;;
    esac
}

echo ""
echo "──────────────────────────────────────────────"
echo "run_probe cache-policy gather (UPFRONT — one prompt per module)"
echo "──────────────────────────────────────────────"
_check_and_prompt "probe_action" "${OUTPUT_ACTION}/*"
_check_and_prompt "probe_cos"    "${OUTPUT_COS}/*"
_check_and_prompt "probe_mse"    "${OUTPUT_MSE}/*"
P_ACTION="${POLICY[probe_action]:-1}"
P_COS="${POLICY[probe_cos]:-1}"
P_MSE="${POLICY[probe_mse]:-1}"
echo "  → ACTION=$P_ACTION  COS=$P_COS  MSE=$P_MSE"

# ── Pre-flight: drop P2/P3 encoders whose trainer outputs aren't on disk ─
# probe eval is the consumer; P2 (m09a continual SSL) and P3 (m09c surgery) are
# producers run separately via scripts/run_probe_train.sh. If a producer hasn't
# run yet, silently dropping the encoder lets the rest of the pipeline complete
# (frozen + dinov2 always work — they're external ckpts).
echo ""
echo "──────────────────────────────────────────────"
echo "P2/P3 trainer-output pre-flight"
echo "──────────────────────────────────────────────"
NEW_ENCODERS=""
for ENC in $ENCODERS; do
    case "$ENC" in
        vjepa_2_1_pretrain|vjepa_2_1_surgical_3stage_DI|vjepa_2_1_surgical_noDI)
            CKPT="$(encoder_ckpt_for "$ENC")"
            if [ ! -e "$CKPT" ]; then
                echo "  ⚠️  $ENC: $CKPT not found — train via:"
                case "$ENC" in
                    vjepa_2_1_pretrain)            echo "       ./scripts/run_probe_train.sh pretrain          --$MODE" ;;
                    vjepa_2_1_surgical_3stage_DI)  echo "       ./scripts/run_probe_train.sh surgery_3stage_DI --$MODE" ;;
                    vjepa_2_1_surgical_noDI)       echo "       ./scripts/run_probe_train.sh surgery_noDI      --$MODE" ;;
                esac
                echo "  → dropping $ENC from this run; pipeline continues with remaining encoders"
                continue
            fi
            echo "  ✓ $ENC: $CKPT"
            ;;
        *)
            echo "  ✓ $ENC: external (no trainer needed)"
            ;;
    esac
    NEW_ENCODERS="$NEW_ENCODERS $ENC"
done
ENCODERS="$(echo "$NEW_ENCODERS" | xargs)"
if [ -z "$ENCODERS" ]; then
    echo "FATAL: ENCODERS list is empty after pre-flight — nothing to evaluate"
    exit 3
fi
echo "  → final ENCODERS: $ENCODERS"

# ── Pre-flight: Stage 8 needs predictor-bearing ckpt (NOT just encoder) ─
# Stages 2-7 use student_encoder.pt (encoder only); Stage 8 future_mse calls
# probe_future_mse._load_predictor_2_1 which requires the "predictor" key —
# present only in m09{a,c}_ckpt_best.pt (save_training_checkpoint full=True).
# m09a's _best.pt was historically saved with full=False, so vjepa_2_1_pretrain
# may have student_encoder.pt but NOT m09a_ckpt_best.pt → Stage 8 in-loop
# FATAL'd in run_src_probe_sanity_v2.log line 778. Now: build a separate
# STAGE8_ENCODERS subset, drop variants missing the predictor ckpt with a
# clear warning. Stages 2-7 still include them (encoder-only is enough).
echo ""
echo "──────────────────────────────────────────────"
echo "Stage 8 predictor-ckpt pre-flight"
echo "──────────────────────────────────────────────"
STAGE8_NEW=""
for ENC in $ENCODERS; do
    [[ "$ENC" == vjepa* ]] || continue                         # DINOv2 has no predictor — skip
    PCKPT="$(encoder_predictor_ckpt_for "$ENC")"
    if [ -e "$PCKPT" ]; then
        echo "  ✓ $ENC: $PCKPT"
        STAGE8_NEW="$STAGE8_NEW $ENC"
    else
        echo "  ⚠️  $ENC: $PCKPT not found — Stage 8 will SKIP this encoder"
        case "$ENC" in
            vjepa_2_1_frozen)
                echo "       (frozen variant uses Meta's ckpt which always carries the predictor —"
                echo "        if this is missing, ENCODER_CKPT itself is wrong)" ;;
            vjepa_2_1_pretrain)
                echo "       Re-train (m09a_ckpt_best.pt is written via save_training_checkpoint full=True):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --$MODE" ;;
            vjepa_2_1_surgical_*)
                echo "       Re-train (m09c writes m09c_ckpt_best.pt at end of surgery):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh ${ENC#vjepa_2_1_surgical_} --$MODE" ;;
        esac
    fi
done
STAGE8_ENCODERS="$(echo "$STAGE8_NEW" | xargs)"
if [ -z "$STAGE8_ENCODERS" ]; then
    echo "  → no V-JEPA variants have predictor ckpt; Stage 8/9 will be auto-skipped"
else
    echo "  → Stage 8 ENCODERS: $STAGE8_ENCODERS"
fi

# ── STAGE 1 — labels (action_probe + taxonomy, CPU, ~1-2 min) ──────────
# Two label artifacts emitted side-by-side from the same EVAL_SUBSET +
# TAGS_JSON so downstream stages and run_probe_train.sh both find them.
#   - probe_action/action_labels.json   : 3-class action (P1 gate)
#   - probe_taxonomy/taxonomy_labels.json : 16 dims (action + 15 from
#                                           tag_taxonomy.json) — used by
#                                           m09a/m09c multi-task probe loss
#                                           when multi_task_probe.enabled=true
# Re-uses cache-policy P_ACTION (cheap CPU work; kept simple).
if ! should_skip 1; then
    stamp "STAGE 1 · action_probe + taxonomy labels (CPU, ~1-2 min)"
    python -u src/probe_action.py "--$MODE" \
        --stage labels \
        --eval-subset "$EVAL_SUBSET" \
        --tags-json "$TAGS_JSON" \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        2>&1 | tee logs/probe_action_labels.log
    if [ -f "$TAG_TAXONOMY" ]; then
        python -u src/probe_taxonomy.py "--$MODE" \
            --stage labels \
            --eval-subset "$EVAL_SUBSET" \
            --tags-json "$TAGS_JSON" \
            --tag-taxonomy "$TAG_TAXONOMY" \
            --output-root "$OUTPUT_TAXONOMY" \
            --cache-policy "$P_ACTION" \
            2>&1 | tee logs/probe_taxonomy_labels.log
    else
        echo "  WARN: $TAG_TAXONOMY missing — skipping probe_taxonomy labels."
        echo "    multi_task_probe in m09a/m09c will auto-disable for this run."
    fi
fi

# ── STAGE 2 — features (action_probe) per encoder ──────────────────────
if ! should_skip 2; then
    stamp "STAGE 2 · action_probe features (GPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        EXTRA_CKPT=""
        if [[ "$ENC" == vjepa* ]]; then
            CKPT="$(encoder_ckpt_for "$ENC")"
            [ -e "$CKPT" ] || { echo "FATAL: encoder ckpt missing for $ENC: $CKPT"; exit 3; }
            EXTRA_CKPT="--encoder-ckpt $CKPT"
        fi
        python -u src/probe_action.py "--$MODE" \
            --stage features \
            --encoder "$ENC" \
            $EXTRA_CKPT \
            --eval-subset "$EVAL_SUBSET" \
            --local-data "$LOCAL_DATA" \
            --output-root "$OUTPUT_ACTION" \
            --num-frames "$NUM_FRAMES" \
            --cache-policy "$P_ACTION" \
            2>&1 | tee "logs/probe_action_features_${ENC}.log"
    done
fi

# ── STAGE 3 — train (action_probe) per encoder × LR ────────────────────
# LR_SWEEP="5e-4" → single probe per encoder (default — written to canonical
#   <encoder>/probe.pt for Stage 4 to read).
# LR_SWEEP="1e-4 3e-4 1e-3 3e-3" → multihead-style sweep; each LR writes to
#   <encoder>/lr_<LR>/, then the best-val-acc LR is symlinked to the canonical
#   path so Stage 4 reads the winner. Matches the spirit of Meta's
#   deps/vjepa2/configs/eval/vitg-384/ssv2.yaml multihead sweep.
# epochs + warmup_pct mode-aware (SANITY: 50/0.10, FULL: 20/0.0).
if ! should_skip 3; then
    n_lrs=$(echo "$LR_SWEEP" | wc -w)
    stamp "STAGE 3 · action_probe train (GPU × ${ENCODERS//[^[:space:]]/x} enc × ${n_lrs} lr; epochs=$EPOCHS warmup_pct=$WARMUP_PCT)"
    for ENC in $ENCODERS; do
        for LR in $LR_SWEEP; do
            if [ "$n_lrs" -gt 1 ]; then
                SUBDIR_FLAG="--output-subdir lr_${LR}"
                LOG_SUFFIX="_lr${LR}"
            else
                SUBDIR_FLAG=""
                LOG_SUFFIX=""
            fi
            python -u src/probe_action.py "--$MODE" \
                --stage train \
                --encoder "$ENC" \
                --output-root "$OUTPUT_ACTION" \
                --epochs "$EPOCHS" \
                --probe-lr "$LR" \
                --warmup-pct "$WARMUP_PCT" \
                $SUBDIR_FLAG \
                --cache-policy "$P_ACTION" \
                2>&1 | tee "logs/probe_action_train_${ENC}${LOG_SUFFIX}.log"
        done
        # Multi-LR sweep: pick best-val-acc LR + symlink its outputs to canonical
        # paths so Stage 4 (paired_delta) reads the winner. Logic lives in
        # probe_action.py::run_select_best_lr_stage (per CLAUDE.md "shell
        # scripts are THIN wrappers — all logic in Python"; idempotent so re-runs
        # are safe).
        if [ "$n_lrs" -gt 1 ]; then
            python -u src/probe_action.py "--$MODE" \
                --stage select_best_lr \
                --encoder "$ENC" \
                --output-root "$OUTPUT_ACTION" \
                --cache-policy "$P_ACTION" \
                2>&1 | tee "logs/probe_action_select_best_lr_${ENC}.log"
        fi
    done
fi

# ── STAGE 4 — paired Δ (action_probe) ←── 🔥 PRIORITY 1 GATE ──────────
if ! should_skip 4; then
    stamp "STAGE 4 · action_probe paired_delta (🔥 P1 GATE — CPU, ~5 min, BCa 10K)"
    python -u src/probe_action.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        2>&1 | tee logs/probe_action_paired_delta.log
fi

# ── STAGE 5 — features (motion_cos) per encoder ────────────────────────
if ! should_skip 5; then
    stamp "STAGE 5 · motion_cos features (CPU mean-pool from action_probe × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/probe_motion_cos.py "--$MODE" \
            --stage features \
            --encoder "$ENC" \
            --action-probe-root "$OUTPUT_ACTION" \
            --output-root "$OUTPUT_COS" \
            --share-features \
            --cache-policy "$P_COS" \
            2>&1 | tee "logs/probe_motion_cos_features_${ENC}.log"
    done
fi

# ── STAGE 6 — cosine (motion_cos) per encoder ──────────────────────────
if ! should_skip 6; then
    stamp "STAGE 6 · motion_cos cosine (CPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/probe_motion_cos.py "--$MODE" \
            --stage cosine \
            --encoder "$ENC" \
            --action-probe-root "$OUTPUT_ACTION" \
            --output-root "$OUTPUT_COS" \
            --cache-policy "$P_COS" \
            2>&1 | tee "logs/probe_motion_cos_cosine_${ENC}.log"
    done
fi

# ── STAGE 7 — paired Δ (motion_cos) ────────────────────────────────────
if ! should_skip 7; then
    stamp "STAGE 7 · motion_cos paired_delta (CPU)"
    python -u src/probe_motion_cos.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_COS" \
        --cache-policy "$P_COS" \
        2>&1 | tee logs/probe_motion_cos_paired_delta.log
fi

# ── STAGE 8 — forward (future_mse) — loop V-JEPA variants ─────────────
# DINOv2 has no future-frame predictor → loop only V-JEPA variants.
# Uses STAGE8_ENCODERS (computed in pre-flight at line ~289) which is the
# subset of $ENCODERS whose predictor-bearing ckpt (m09{a,c}_ckpt_best.pt)
# exists on disk. probe_future_mse._load_predictor_2_1 requires a 'predictor'
# key in the .pt — present only in full checkpoints (save_training_checkpoint
# full=True), not student_encoder.pt. See plan_code_dev.md §"R8".
#
# Defense-in-depth: even with pre-flight, defensively re-check inside the
# loop and WARN+continue (NOT FATAL) so SKIP_STAGES bypassing the pre-flight
# can't crash the rest of the pipeline. Stage 9 reads whatever per_clip_mse.npy
# files exist on disk and naturally excludes any variant whose forward was skipped.
if ! should_skip 8; then
    stamp "STAGE 8 · future_mse forward (GPU, V-JEPA variants only)"
    for ENC in $STAGE8_ENCODERS; do
        CKPT="$(encoder_predictor_ckpt_for "$ENC")"
        if [ ! -e "$CKPT" ]; then
            echo "  ⚠️  Stage 8 SKIP $ENC: predictor-bearing ckpt missing ($CKPT)"
            continue
        fi
        python -u src/probe_future_mse.py "--$MODE" \
            --stage forward \
            --variant "$ENC" \
            --encoder-ckpt "$CKPT" \
            --action-probe-root "$OUTPUT_ACTION" \
            --local-data "$LOCAL_DATA" \
            --output-root "$OUTPUT_MSE" \
            --num-frames "$NUM_FRAMES" \
            --cache-policy "$P_MSE" \
            2>&1 | tee "logs/probe_future_mse_forward_${ENC}.log"
    done
fi

# ── STAGE 9 — paired Δ (future_mse, per-variant) ───────────────────────
if ! should_skip 9; then
    stamp "STAGE 9 · future_mse paired_per_variant (CPU)"
    python -u src/probe_future_mse.py "--$MODE" \
        --stage paired_per_variant \
        --output-root "$OUTPUT_MSE" \
        --cache-policy "$P_MSE" \
        2>&1 | tee logs/probe_future_mse_paired.log
fi

# ── STAGE 10 — plots (m08d, CPU, always-recompute) ─────────────────────
# Pure visualization — no cache_policy. Wipes its own output_dir on entry
# (single-owner, m08b-style). Reads JSONs + train_log.jsonl from prior stages.
if ! should_skip 10; then
    stamp "STAGE 10 · m08d plot probe (CPU, ~5s)"
    python -u src/probe_plot.py "--$MODE" \
        --action-probe-root "$OUTPUT_ACTION" \
        --motion-cos-root   "$OUTPUT_COS" \
        --future-mse-root   "$OUTPUT_MSE" \
        --output-dir        "$OUTPUT_PLOTS" \
        2>&1 | tee logs/probe_plot.log
fi

# ── Final summary ──────────────────────────────────────────────────────
stamp "DONE · total wall = $(( ($(date +%s) - T0) / 60 )) min"
echo "Artifacts:"
echo "  🔥 P1 GATE     $OUTPUT_ACTION/probe_paired_delta.json"
echo "  motion_cos     $OUTPUT_COS/probe_motion_cos_paired.json"
echo "  future_mse     $OUTPUT_MSE/probe_future_mse_per_variant.json"
echo "  taxonomy lbls  $OUTPUT_TAXONOMY/taxonomy_labels.json (consumed by run_probe_train.sh multi-task path)"
echo "  plots          $OUTPUT_PLOTS/{probe_action_loss,probe_action_acc,probe_encoder_comparison}.{png,pdf}"
echo "Per-encoder probe ckpts:"
for ENC in $ENCODERS; do
    echo "  $OUTPUT_ACTION/$ENC/probe.pt"
done
