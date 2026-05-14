#!/usr/bin/env bash
# iter13 probe orchestrator — 3 modules × {V-JEPA, DINOv2} × stages.
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# and "DELETE PROTECTION — shells stay THIN, .py owns the cache-policy prompt".
#
# Pipeline (priority 1 — frozen V-JEPA vs frozen DINOv2 on Indian action probe
# applied to data/eval_10k_local):
#   STAGE 1   probe_action.py    --stage labels        (CPU, ~1 min)
#   STAGE 2   probe_action.py    --stage features      (GPU × 2 encoders, ~1 h)
#   STAGE 3   probe_action.py    --stage train         (GPU × 2 encoders, ~30 min)
#   STAGE 11  probe_taxonomy.py  --stage train         (GPU × 16 dims × N enc, ~30 min) [iter13]
#   STAGE 4   probe_action.py    --stage paired_delta  (CPU, ~5 min)  ← P1 GATE
#   STAGE 12  probe_taxonomy.py  --stage paired_delta  (CPU, ~5 min, BCa across 16 dims) [iter13]
#   STAGE 13  probe_taxonomy.py  --stage plot          (CPU, ~5s) [iter13]
#   STAGE 5   probe_motion_cos.py  --stage features      (CPU mean-pool × 2 enc)
#   STAGE 6   probe_motion_cos.py  --stage cosine        (CPU × 2 enc)
#   STAGE 7   probe_motion_cos.py  --stage paired_delta  (CPU)
#   STAGE 8   probe_future_mse.py  --stage forward       (GPU, V-JEPA only, ~30 min)
#   STAGE 9   probe_future_mse.py  --stage paired_per_variant   (CPU)
#   STAGE 10  probe_plot.py                                  (CPU, ~5s — plots)
#
# iter13 (2026-05-05): stages 11/12/13 added so the 16-dim taxonomy gets a
# proper paper-final eval pass (was previously train-time-only as multi-task aux).
#
# REFERENCES
#   plan_code_dev.md  — per-module specs + LoC budget
#   runbook.md        — full set of pre/post-flight one-liners
#
# USAGE
#   tmux new -s probe
#
#   # FULL run (default) — eval_10k (~9.9k clips), ~4h on 24GB / ~2.5h on 96GB
#   ./scripts/run_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log
#
#   # SANITY smoke test — 150 stratified clips (50/class) from THE SAME eval_10k
#   # JSON, processed against the SAME eval_10k_local/ TARs. ~6-8 min on 24GB.
#   # Outputs sandboxed to outputs/sanity/. Pass --sanity OR set MODE=SANITY.
#   ./scripts/run_eval.sh --sanity 2>&1 | tee logs/run_src_probe_sanity_v1.log
#
#   # Bypass prompts (overnight / non-TTY)
#   CACHE_POLICY_ALL=1 ./scripts/run_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log           # keep all caches
#   CACHE_POLICY_ALL=2 ./scripts/run_eval.sh 2>&1 | tee logs/run_src_probe_full_v1.log           # recompute everything
#   CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --sanity 2>&1 | tee logs/run_src_probe_sanity_v1.log  # SANITY + recompute
#
#   # Skip a stage (resume after failure):
#   SKIP_STAGES="1,2" ./scripts/run_eval.sh
#   # Run only one variant (debug):
#   ENCODERS="vjepa_2_1_frozen" ./scripts/run_eval.sh
#   # Tune SANITY subset size (default 50 clips per class):
#   SANITY_N_PER_CLASS=20 ./scripts/run_eval.sh --sanity
#
#   # Probe-training knobs (Stage 3) — mode-aware defaults:
#   #   SANITY: EPOCHS=50  WARMUP_PCT=0.10  LR_SWEEP="5e-4"   (current behavior)
#   #   FULL:   EPOCHS=20  WARMUP_PCT=0.0   LR_SWEEP="5e-4"   (Meta-faithful single-LR)
#   # Override any of them via env vars:
#   EPOCHS=20 WARMUP_PCT=0.0 ./scripts/run_eval.sh --sanity   # apply Meta recipe to sanity
#
#   # Paper-faithful FULL with LR sweep (matches Meta's multihead 5-LR setup;
#   # ~4x stage-3 wall, but mirrors deps/vjepa2/configs/eval/vitg-384/ssv2.yaml):
#   LR_SWEEP="1e-4 3e-4 1e-3 3e-3" ./scripts/run_eval.sh
#   # Each LR trains a probe under outputs/full/probe_action/<encoder>/lr_<LR>/;
#   # the best-val-acc LR is symlinked to the canonical <encoder>/probe.pt path so
#   # Stage 4 paired_delta reads the winner (no code change needed).

# Fail-fast: probe stages have sequential dependencies (Stage N+1 needs N's outputs),
# so any stage failure → abort chain to avoid wasting GPU time on impossible
# downstream work. Differs from run_paired_eval_10k.sh which uses `set -uo` only
# because its variants are independent (errors_N_fixes #72).
set -euo pipefail
# Clear ERR trap surfaces line + exit code BEFORE the shell exits.
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_eval.sh aborted at line $LINENO (exit=$rc) — sequential dependency failed; downstream stages skipped" >&2; exit $rc' ERR

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

# ── Mode-gated defaults — read from configs/pipeline.yaml ──────────────
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# + "no hardcoded values" — probe-head training defaults (epochs, warmup_pct,
# lr_sweep) live in configs/pipeline.yaml under the probe_head_train block.
# Env vars EPOCHS / WARMUP_PCT / LR_SWEEP still override (overnight scripts +
# ad-hoc sweeps). Mode key is lower-case ('sanity'/'poc'/'full') matching the
# yaml structure. Single-element lr_sweep arrays render as "5.0e-4" → space-
# joined into a normal LR_SWEEP string for the for-loop below.
PIPELINE_YAML="configs/pipeline.yaml"
EX="scripts/lib/yaml_extract.py"
mode_key=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
DEFAULT_EPOCHS=$(python "$EX" "$PIPELINE_YAML" "probe_head_train.${mode_key}.epochs")
DEFAULT_WARMUP_PCT=$(python "$EX" "$PIPELINE_YAML" "probe_head_train.${mode_key}.warmup_pct")
DEFAULT_LR_SWEEP=$(python "$EX" "$PIPELINE_YAML" "probe_head_train.${mode_key}.lr_sweep")
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
WARMUP_PCT="${WARMUP_PCT:-$DEFAULT_WARMUP_PCT}"
LR_SWEEP="${LR_SWEEP:-$DEFAULT_LR_SWEEP}"

if [ "$MODE" = "SANITY" ]; then
    # iter13 v12 (2026-05-06): bumped 50 → 200 because motion-flow class scheme
    # uses 16 classes (vs old 3-class action). 50/action × 3 = 150 clips → ~9/
    # motion-class avg → stratified_split crashes even at MIN_PER_SPLIT=1.
    # 200/action × 3 = 600 clips → ~37/motion-class avg → splits cleanly.
    SANITY_N_PER_CLASS="${SANITY_N_PER_CLASS:-200}"
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
elif [ "$MODE" = "POC" ]; then
    # iter14 (2026-05-08): POC mode — first N keys of eval_10k.json (N from yaml,
    # default 500), then probe_action.py --stage labels applies 70:15:15
    # stratified_split → ~350/75/75 train/val/test. Same subset that
    # run_train.sh generates so train→eval reads consistent splits.
    POC_SUBSET="data/eval_10k_poc.json"
    POC_TOTAL=$(python "$EX" configs/train/base_optimization.yaml data.poc_total_clips)
    if [ ! -f "$POC_SUBSET" ] || [ "data/eval_10k.json" -nt "$POC_SUBSET" ]; then
        python -u src/utils/eval_subset.py \
            --eval-subset data/eval_10k.json \
            --first-n "$POC_TOTAL" \
            --output "$POC_SUBSET"
    fi
    DEFAULT_EVAL_SUBSET="$POC_SUBSET"
    DEFAULT_OUTPUT_PREFIX="outputs/poc"
else                                   # FULL
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
ENCODERS="${ENCODERS:-vjepa_2_1_frozen vjepa_2_1_pretrain vjepa_2_1_pretrain_2X vjepa_2_1_surgical_3stage_DI vjepa_2_1_surgical_noDI}"
SKIP_STAGES="${SKIP_STAGES:-}"
NUM_FRAMES="${NUM_FRAMES:-16}"

# iter13 v12 (2026-05-05): MOTION-flow probe class derivation knobs.
# `motion_features.npy` is m04d's RAFT optical-flow output (13D × N_clips),
# durably stored at <local_data>/motion_features.npy and binned into
# magnitude×direction = 16 motion classes by probe_action --stage labels.
MOTION_FEATURES="${MOTION_FEATURES:-${LOCAL_DATA}/motion_features.npy}"
# Mode-aware filter floors: FULL/POC use paper-grade thresholds (34 clips/class
# → ≥5 per split at 70/15/15). SANITY relaxes them since 150-clip stratified
# subsets give ~9 clips per motion-flow class (would crash stratified_split).
if [ "$MODE" = "SANITY" ]; then
    # iter13 v13 (2026-05-07): floor=3 with greedy split (utils/action_labels.py
    # stratified_split). Keeps every motion-flow class with n≥3 → harder probe →
    # better signal for the paper goal `surgery > pretrain > frozen`. Mirror in
    # run_train.sh.
    DEFAULT_MIN_CLIPS_PER_CLASS=3
    DEFAULT_MIN_PER_SPLIT=1
elif [ "$MODE" = "POC" ]; then
    # iter14 (2026-05-08): POC ~500 clips ÷ 8 motion classes ≈ 60/class. Floor=10
    # tolerates rare-class drops while keeping 6+ classes for probe statistics.
    DEFAULT_MIN_CLIPS_PER_CLASS=10
    DEFAULT_MIN_PER_SPLIT=2
else
    DEFAULT_MIN_CLIPS_PER_CLASS=34
    DEFAULT_MIN_PER_SPLIT=5
fi
MIN_CLIPS_PER_CLASS="${MIN_CLIPS_PER_CLASS:-$DEFAULT_MIN_CLIPS_PER_CLASS}"
MIN_PER_SPLIT="${MIN_PER_SPLIT:-$DEFAULT_MIN_PER_SPLIT}"

# Per-encoder checkpoint resolvers. Two functions because Stages 2/3 (probe-head
# training) need encoder-only ckpts, but Stage 8 (future_mse) also needs the
# predictor. m09a_pretrain / m09c_surgery_* write BOTH artifacts:
#   - student_encoder.pt    : encoder only      (export_student_for_eval)
#   - m09{a,c}_ckpt_best.pt : encoder+predictor (save_training_checkpoint full=True)
# Surgery has TWO variants — 3stage_DI (with interaction tubes) and noDI (without)
# — to test whether D_I helps; each writes its own dir.
# iter13 v13 T2-rename (2026-05-07): probe_pretrain → m09a_pretrain,
# probe_surgery_* → m09c_surgery_* (matches source-module naming + run_train.sh).
encoder_ckpt_for() {                                            # encoder-only — Stages 2/3
    case "$1" in
        vjepa_2_1_frozen)              echo "$ENCODER_CKPT" ;;
        vjepa_2_1_pretrain)            echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain/student_encoder.pt" ;;
        vjepa_2_1_pretrain_2X)       echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_2X/student_encoder.pt" ;;     # iter14 arm C
        vjepa_2_1_surgical_3stage_DI)  echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI/student_encoder.pt" ;;
        vjepa_2_1_surgical_noDI)       echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI/student_encoder.pt" ;;
        *) echo "" ;;
    esac
}
encoder_predictor_ckpt_for() {                                  # encoder+predictor — Stage 8 future_mse
    case "$1" in
        vjepa_2_1_frozen)              echo "$ENCODER_CKPT" ;;
        vjepa_2_1_pretrain)            echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain/m09a_ckpt_best.pt" ;;
        vjepa_2_1_pretrain_2X)       echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_2X/m09a_ckpt_best.pt" ;;       # iter14 arm C
        vjepa_2_1_surgical_3stage_DI)  echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI/m09c_ckpt_best.pt" ;;
        vjepa_2_1_surgical_noDI)       echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI/m09c_ckpt_best.pt" ;;
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
    echo "  WARN: $TAGS_JSON not found — probe_taxonomy stages (1/3.5/12/13) will be auto-skipped"
fi
# iter13 v12 (2026-05-05): preflight m04d motion_features.npy. Required for
# Stage 1 (probe_action --stage labels) which derives motion-flow classes
# from RAFT optical-flow features. Run m04d once per local dataset (durable
# artifact stored in <local_data>/, auto-uploaded by hf_outputs.py upload-data).
if [ ! -f "$MOTION_FEATURES" ]; then
    echo "❌ FATAL: motion_features.npy not found at: $MOTION_FEATURES" >&2
    echo "   Run m04d once for this dataset (durable artifact, ~30-60 min GPU):" >&2
    echo "     python -u src/m04d_motion_features.py --$MODE \\" >&2
    echo "         --subset $EVAL_SUBSET --local-data $LOCAL_DATA \\" >&2
    echo "         --features-out $MOTION_FEATURES" >&2
    echo "   Or download via:  python -u src/utils/hf_outputs.py download-data" >&2
    exit 3
fi
MOTION_FEATURES_PATHS="${MOTION_FEATURES%.npy}.paths.npy"
if [ ! -f "$MOTION_FEATURES_PATHS" ]; then
    echo "❌ FATAL: motion_features.paths.npy not found at: $MOTION_FEATURES_PATHS (must be next to .npy)" >&2
    exit 3
fi
echo "  ✓ mode:                 $MODE"
echo "  ✓ eval_subset:          $EVAL_SUBSET"
echo "  ✓ local_data:           $LOCAL_DATA"
echo "  ✓ encoder_ckpt:         $ENCODER_CKPT  ($(du -h "$ENCODER_CKPT" 2>/dev/null | awk '{print $1}'))"
echo "  ✓ motion_features:      $MOTION_FEATURES  ($(du -h "$MOTION_FEATURES" 2>/dev/null | awk '{print $1}'))"
echo "  ✓ min_clips_per_class:  $MIN_CLIPS_PER_CLASS"
echo "  ✓ min_per_split:        $MIN_PER_SPLIT"
echo "  ✓ encoders:             $ENCODERS"
echo "  ✓ skip_stages:          ${SKIP_STAGES:-<none>}"
echo "  ✓ cache_policy:         ${CACHE_POLICY_ALL:-prompt-per-module}"
echo "  ✓ output_prefix:        $DEFAULT_OUTPUT_PREFIX"

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
# producers run separately via scripts/run_train.sh. If a producer hasn't
# run yet, silently dropping the encoder lets the rest of the pipeline complete
# (frozen + dinov2 always work — they're external ckpts).
echo ""
echo "──────────────────────────────────────────────"
echo "P2/P3 trainer-output pre-flight"
echo "──────────────────────────────────────────────"
NEW_ENCODERS=""
for ENC in $ENCODERS; do
    case "$ENC" in
        vjepa_2_1_pretrain|vjepa_2_1_pretrain_2X|vjepa_2_1_surgical_3stage_DI|vjepa_2_1_surgical_noDI)
            CKPT="$(encoder_ckpt_for "$ENC")"
            if [ ! -e "$CKPT" ]; then
                echo "  ⚠️  $ENC: $CKPT not found — train via:"
                case "$ENC" in
                    vjepa_2_1_pretrain)            echo "       ./scripts/run_train.sh pretrain          --$MODE" ;;
                    vjepa_2_1_pretrain_2X)       echo "       ./scripts/run_train.sh pretrain_2X     --$MODE" ;;
                    vjepa_2_1_surgical_3stage_DI)  echo "       ./scripts/run_train.sh surgery_3stage_DI --$MODE" ;;
                    vjepa_2_1_surgical_noDI)       echo "       ./scripts/run_train.sh surgery_noDI      --$MODE" ;;
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
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain --$MODE" ;;
            vjepa_2_1_pretrain_2X)
                echo "       Re-train iter14 arm C (10 ep, ~20 GPU-h on FULL):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_2X --$MODE" ;;
            vjepa_2_1_surgical_*)
                echo "       Re-train (m09c writes m09c_ckpt_best.pt at end of surgery):"
                echo "         CACHE_POLICY_ALL=2 ./scripts/run_train.sh ${ENC#vjepa_2_1_surgical_} --$MODE" ;;
        esac
    fi
done
STAGE8_ENCODERS="$(echo "$STAGE8_NEW" | xargs)"
if [ -z "$STAGE8_ENCODERS" ]; then
    echo "  → no V-JEPA variants have predictor ckpt; Stage 8/9 will be auto-skipped"
else
    echo "  → Stage 8 ENCODERS: $STAGE8_ENCODERS"
fi

# ── Pre-eval pretrain-cleanup (iter13, 2026-05-05) ──────────────────────
# `m09a_ckpt_latest.pt` (~29 GB on ViT-G full=True) is a RESUME anchor for
# pretrain — it's only useful if pretrain crashes mid-run and needs to
# continue from the last step. Once pretrain has completed (proxy:
# student_encoder.pt is exported), latest.pt is dead weight on disk.
# Same for m09c_ckpt_latest.pt for surgery encoders.
# Deleting it here reclaims ~29 GB per pretrained encoder BEFORE eval starts —
# critical because Stage 3's transient train cache (~73 GB at fp16 / ~146 GB
# at fp32) would otherwise overflow our 199 GB disk budget.
# Opt-out: set EVAL_KEEP_LATEST=1 to skip this cleanup (e.g. during dev
# when you might want to resume a partial pretrain). Default: ON.
if [ "${EVAL_KEEP_LATEST:-0}" != "1" ]; then
    echo ""
    echo "──────────────────────────────────────────────"
    echo "Pre-eval pretrain-cleanup (drop _latest.pt resume anchors)"
    echo "──────────────────────────────────────────────"
    pretrain_cleanup_get_latest() {
        # Map encoder → its m09{a,c}_ckpt_latest.pt path (or empty if external).
        case "$1" in
            vjepa_2_1_pretrain)            echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain/m09a_ckpt_latest.pt" ;;
            vjepa_2_1_pretrain_2X)       echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_2X/m09a_ckpt_latest.pt" ;;     # iter14 arm C
            vjepa_2_1_surgical_3stage_DI)  echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI/m09c_ckpt_latest.pt" ;;
            vjepa_2_1_surgical_noDI)       echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI/m09c_ckpt_latest.pt" ;;
            *) echo "" ;;
        esac
    }
    for ENC in $ENCODERS; do
        STUDENT_ENC="$(encoder_ckpt_for "$ENC")"
        # Only delete if pretrain/surgery has truly completed (student_encoder.pt
        # exists on the path encoder_ckpt_for resolves to). External (frozen)
        # encoders are skipped — their ckpt is shared and immutable.
        case "$ENC" in
            vjepa_2_1_frozen) continue ;;
        esac
        LATEST="$(pretrain_cleanup_get_latest "$ENC")"
        if [ -z "$LATEST" ] || [ ! -f "$LATEST" ]; then
            continue
        fi
        if [ ! -e "$STUDENT_ENC" ]; then
            echo "  [keep-latest] $ENC: student_encoder.pt missing → preserving $LATEST (pretrain not complete)"
            continue
        fi
        sz=$(du -h "$LATEST" 2>/dev/null | awk '{print $1}')
        rm -f "$LATEST"
        echo "  [pretrain-cleanup] removed $LATEST ($sz) — pretrain complete, latest.pt was resume-only"

        # iter13 (2026-05-05): also drop step ckpts (m09{a,c}_ckpt_step*.pt).
        # These are the keep_last_n=2 rotation buffers from training. Once
        # student_encoder.pt is exported and best.pt is preserved, the step
        # ckpts are dead weight — same student weights, no predictor (full=False).
        # Each step ckpt is ~7 GB at ViT-G; 2 of them = ~14 GB freed per encoder.
        # Opt-out via the same EVAL_KEEP_LATEST=1 env var as the latest.pt path.
        STEP_DIR="$(dirname "$LATEST")"
        for step_pt in "$STEP_DIR"/*ckpt_step*.pt; do
            if [ -f "$step_pt" ]; then
                step_sz=$(du -h "$step_pt" 2>/dev/null | awk '{print $1}')
                rm -f "$step_pt"
                echo "  [pretrain-cleanup] removed $(basename "$step_pt") ($step_sz) — step rotation buffer, training complete"
            fi
        done
    done
fi

# ── STAGE 1 — labels (action_probe + taxonomy, CPU, ~1-2 min) ──────────
# Two label artifacts emitted side-by-side from the same EVAL_SUBSET +
# TAGS_JSON so downstream stages and run_train.sh both find them.
#   - probe_action/action_labels.json   : 3-class action (P1 gate)
#   - probe_taxonomy/taxonomy_labels.json : 16 dims (action + 15 from
#                                           tag_taxonomy.json) — used by
#                                           m09a/m09c multi-task probe loss
#                                           when multi_task_probe.enabled=true
# Re-uses cache-policy P_ACTION (cheap CPU work; kept simple).
if ! should_skip 1; then
    stamp "STAGE 1 · action_probe + taxonomy labels (CPU, ~1-2 min)"
    # iter13 v12 (2026-05-05): probe_action labels derive MOTION-flow classes
    # from m04d's motion_features.npy (16 classes of magnitude×direction).
    # --tags-json is NOT passed any more — action labels no longer use VLM tags.
    python -u src/probe_action.py "--$MODE" \
        --stage labels \
        --eval-subset "$EVAL_SUBSET" \
        --motion-features "$MOTION_FEATURES" \
        --min-clips-per-class "$MIN_CLIPS_PER_CLASS" \
        --min-per-split "$MIN_PER_SPLIT" \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_action_labels.log
    if [ -f "$TAG_TAXONOMY" ]; then
        python -u src/probe_taxonomy.py "--$MODE" \
            --stage labels \
            --eval-subset "$EVAL_SUBSET" \
            --tags-json "$TAGS_JSON" \
            --tag-taxonomy "$TAG_TAXONOMY" \
            --output-root "$OUTPUT_TAXONOMY" \
            --cache-policy "$P_ACTION" \
            --no-wandb \
            2>&1 | tee logs/probe_taxonomy_labels.log
    else
        echo "  WARN: $TAG_TAXONOMY missing — skipping probe_taxonomy labels."
        echo "    multi_task_probe in m09a/m09c will auto-disable for this run."
    fi
fi

# ── PER-ENCODER SEQUENTIAL PIPELINE (iter13, 2026-05-05) ───────────────
# Stages 2 + 3 + 5 + 6 + 8 fused into one per-encoder loop. The disk pressure
# during eval is dominated by `features_test.npy` (~31 GB at fp32, ~16 GB at
# fp16 per encoder) and Stage 3's transient train+val resume caches (~73 GB
# at fp16 per encoder). Running stages encoder-by-encoder lets Stage 5 free
# `features_test.npy` BEFORE the next encoder writes its own, so peak disk is
# bounded by ONE encoder's footprint instead of N_encoders × footprint.
# Stages 4/7/9/10 stay AFTER this loop — they aggregate small per-encoder
# result files (test_predictions.npy ~6 KB, per_clip_motion_cos.npy ~6 KB,
# per_clip_mse.npy ~6 KB) that comfortably fit on disk.
#
# Skip semantics: each stage's `should_skip N` is honoured INSIDE the per-
# encoder loop, so SKIP_STAGES="2,3" still works as expected.
#
# To restore the legacy 3-split layout (probe_taxonomy --stage train ad-hoc),
# pass `FEATURES_SPLITS="train val test"` env var.
FEATURES_SPLITS="${FEATURES_SPLITS:-test}"
# iter13 (2026-05-05): adaptive token pool. Default 16 (V-JEPA paper §4 attentive
# probe regime; 290× smaller .npy + 290× less probe attention compute). Set
# POOL_TOKENS=0 to disable (legacy 4608-token storage; will hit OOM at probe BS=64).
POOL_TOKENS="${POOL_TOKENS:-16}"
# Stream-and-discard mode (Stage 3 only). Skips ALL feature persistence; per-batch
# encoder→pool→head→loss→backward. Use STREAM_TRAIN=1 for >1M-clip datasets where
# even pooled features don't fit RAM. Single-LR only — overrides LR_SWEEP if set.
STREAM_TRAIN="${STREAM_TRAIN:-0}"
if [ "$STREAM_TRAIN" = "1" ]; then
    STREAM_FLAG="--stream-train"
else
    STREAM_FLAG=""
fi
n_lrs=$(echo "$LR_SWEEP" | wc -w)

PER_ENC_ANY=0
for s in 2 3 5 6 8; do should_skip "$s" || PER_ENC_ANY=1; done
if [ "$PER_ENC_ANY" -eq 1 ]; then
    stamp "PER-ENCODER pipeline (Stages 2/3/5/6/8) — ${ENCODERS//[^[:space:]]/x} encoders sequentially"
    for ENC in $ENCODERS; do
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "ENCODER: $ENC  (Stages 2/3/5/6/8)"
        echo "════════════════════════════════════════════════════════════════"
        EXTRA_CKPT=""
        if [[ "$ENC" == vjepa* ]]; then
            CKPT="$(encoder_ckpt_for "$ENC")"
            [ -e "$CKPT" ] || { echo "FATAL: encoder ckpt missing for $ENC: $CKPT"; exit 3; }
            EXTRA_CKPT="--encoder-ckpt $CKPT"
        fi

        # ─── Stage 2: features for this encoder ──────────────────────────
        if ! should_skip 2; then
            stamp "  STAGE 2 · features for $ENC (splits=${FEATURES_SPLITS}, dtype=fp16)"
            python -u src/probe_action.py "--$MODE" \
                --stage features \
                --encoder "$ENC" \
                $EXTRA_CKPT \
                --eval-subset "$EVAL_SUBSET" \
                --local-data "$LOCAL_DATA" \
                --output-root "$OUTPUT_ACTION" \
                --num-frames "$NUM_FRAMES" \
                --features-splits $FEATURES_SPLITS \
                --pool-tokens "$POOL_TOKENS" \
                --cache-policy "$P_ACTION" \
                --no-wandb \
                2>&1 | tee "logs/probe_action_features_${ENC}.log"
        fi

        # ─── Stage 3: probe head training (per-LR + select_best_lr) ──────
        # LR_SWEEP="5e-4" → single probe (canonical <enc>/probe.pt for Stage 4).
        # LR_SWEEP="1e-4 3e-4 1e-3 3e-3" → multihead sweep; best-val-acc LR
        #   is symlinked to canonical path. Matches Meta's eval/vitg-384/ssv2.yaml.
        # Lazy-extract: Stage 3 extracts train+val into transient .probe_features_
        # <split>_ckpt.npz cache; cache is reused across LR runs and cleaned at
        # end of run_select_best_lr_stage (multi-LR) or end of run_train_stage
        # (single-LR).
        if ! should_skip 3; then
            stamp "  STAGE 3 · train probe ($n_lrs LR(s); epochs=$EPOCHS warmup=$WARMUP_PCT) for $ENC"
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
                    $EXTRA_CKPT \
                    --eval-subset "$EVAL_SUBSET" \
                    --local-data "$LOCAL_DATA" \
                    --num-frames "$NUM_FRAMES" \
                    --output-root "$OUTPUT_ACTION" \
                    --epochs "$EPOCHS" \
                    --probe-lr "$LR" \
                    --warmup-pct "$WARMUP_PCT" \
                    --pool-tokens "$POOL_TOKENS" \
                    $STREAM_FLAG \
                    $SUBDIR_FLAG \
                    --cache-policy "$P_ACTION" \
                    --no-wandb \
                    2>&1 | tee "logs/probe_action_train_${ENC}${LOG_SUFFIX}.log"
            done
            if [ "$n_lrs" -gt 1 ]; then
                python -u src/probe_action.py "--$MODE" \
                    --stage select_best_lr \
                    --encoder "$ENC" \
                    --output-root "$OUTPUT_ACTION" \
                    --cache-policy "$P_ACTION" \
                    --no-wandb \
                    2>&1 | tee "logs/probe_action_select_best_lr_${ENC}.log"
            fi
        fi

        # ─── Stage 3.5: probe_taxonomy --stage train (16-dim per-encoder) ─
        # iter13 (2026-05-05): the 16-dim taxonomy must be evaluated end-to-
        # end (train + val + test), not just used as multi-task aux loss
        # during m09a/m09c training. probe_taxonomy.run_train_stage trains
        # 16 attentive heads (1 single-label "action" + 15 multi-label tag
        # dims) on the SAME features that probe_action --stage train just
        # consumed. Lazy-extract logic in probe_taxonomy reuses the
        # .probe_features_<split>_ckpt.npz resume cache from probe_action's
        # extraction → near-zero re-extraction cost. Disabled by SKIP_STAGES
        # entry 11 (new stage number) or by --features-splits set excluding
        # train/val (taxonomy can't run without those).
        if ! should_skip 11 && [ -f "$TAG_TAXONOMY" ]; then
            stamp "  STAGE 3.5 · taxonomy probe (16 dims) for $ENC"
            python -u src/probe_taxonomy.py "--$MODE" \
                --stage train \
                --encoder "$ENC" \
                $EXTRA_CKPT \
                --eval-subset "$EVAL_SUBSET" \
                --local-data "$LOCAL_DATA" \
                --num-frames "$NUM_FRAMES" \
                --features-root "$OUTPUT_ACTION" \
                --output-root "$OUTPUT_TAXONOMY" \
                --epochs "$EPOCHS" \
                --probe-lr 5e-4 \
                --warmup-pct "$WARMUP_PCT" \
                --pool-tokens "$POOL_TOKENS" \
                --cache-policy "$P_ACTION" \
                --no-wandb \
                2>&1 | tee "logs/probe_taxonomy_train_${ENC}.log"
        fi

        # ─── Stage 5: motion_cos features (mean-pool features_test.npy) ──
        # Stage 5 is the LAST consumer of features_test.npy. Right after this
        # call, the file is dead weight — delete inline so the NEXT encoder's
        # Stage 2 doesn't have to share disk with this encoder's features_test.
        if ! should_skip 5; then
            stamp "  STAGE 5 · motion_cos features for $ENC"
            python -u src/probe_motion_cos.py "--$MODE" \
                --stage features \
                --encoder "$ENC" \
                --action-probe-root "$OUTPUT_ACTION" \
                --output-root "$OUTPUT_COS" \
                --pool-tokens "$POOL_TOKENS" \
                --share-features \
                --cache-policy "$P_COS" \
                --no-wandb \
                2>&1 | tee "logs/probe_motion_cos_features_${ENC}.log"

            FEATS_TEST="${OUTPUT_ACTION}/${ENC}/features_test.npy"
            KEYS_TEST="${OUTPUT_ACTION}/${ENC}/clip_keys_test.npy"
            CKPT_TEST="${OUTPUT_ACTION}/${ENC}/.probe_features_test_ckpt.npz"
            for f in "$FEATS_TEST" "$KEYS_TEST" "$CKPT_TEST"; do
                if [ -f "$f" ]; then
                    sz=$(du -h "$f" 2>/dev/null | awk '{print $1}')
                    rm -f "$f"
                    echo "  [stage5-cleanup] removed $f ($sz)"
                fi
            done
        fi

        # ─── Stage 6: motion_cos cosine ───────────────────────────────────
        # Reads pooled_features_test.npy (~10 MB) emitted by Stage 5.
        # features_test.npy is ALREADY gone by here — verified by the cleanup
        # above. Stage 6's cosine math runs on the mean-pooled tensor.
        if ! should_skip 6; then
            stamp "  STAGE 6 · motion_cos cosine for $ENC"
            python -u src/probe_motion_cos.py "--$MODE" \
                --stage cosine \
                --encoder "$ENC" \
                --action-probe-root "$OUTPUT_ACTION" \
                --output-root "$OUTPUT_COS" \
                --cache-policy "$P_COS" \
                --no-wandb \
                2>&1 | tee "logs/probe_motion_cos_cosine_${ENC}.log"
        fi

        # ─── Stage 8: future_mse forward (V-JEPA + predictor only) ───────
        # Skipped for DINOv2 (no predictor) and for any V-JEPA variant whose
        # predictor-bearing ckpt (m09{a,c}_ckpt_best.pt) wasn't found in the
        # Stage 8 preflight (line ~298). Defense-in-depth: re-check inside the
        # loop and SKIP+continue (NOT FATAL) so SKIP_STAGES bypassing the
        # preflight can't crash the rest of the per-encoder pipeline.
        if ! should_skip 8 && [[ "$ENC" == vjepa* ]]; then
            if [[ " $STAGE8_ENCODERS " == *" $ENC "* ]]; then
                PCKPT="$(encoder_predictor_ckpt_for "$ENC")"
                if [ -e "$PCKPT" ]; then
                    stamp "  STAGE 8 · future_mse forward for $ENC"
                    python -u src/probe_future_mse.py "--$MODE" \
                        --stage forward \
                        --variant "$ENC" \
                        --encoder-ckpt "$PCKPT" \
                        --action-probe-root "$OUTPUT_ACTION" \
                        --local-data "$LOCAL_DATA" \
                        --output-root "$OUTPUT_MSE" \
                        --num-frames "$NUM_FRAMES" \
                        --cache-policy "$P_MSE" \
                        --no-wandb \
                        2>&1 | tee "logs/probe_future_mse_forward_${ENC}.log"
                else
                    echo "  ⚠️  Stage 8 SKIP $ENC: predictor-bearing ckpt missing ($PCKPT)"
                fi
            else
                echo "  Stage 8 SKIP $ENC (not in STAGE8_ENCODERS preflight set)"
            fi
        fi
    done
fi

# ── STAGE 4 — paired Δ (action_probe) ←── 🔥 PRIORITY 1 GATE ──────────
# Reads test_predictions.npy + test_metrics.json from EACH encoder's
# canonical path. All encoders' Stage 3 outputs were written above.
if ! should_skip 4; then
    stamp "STAGE 4 · action_probe paired_delta (🔥 P1 GATE — CPU, ~5 min, BCa 10K)"
    python -u src/probe_action.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_action_paired_delta.log
fi

# ── STAGE 12 — paired Δ (probe_taxonomy, per-dim across 16 dims) ──────
# Aggregate stage — reads per-dim test predictions from each encoder's
# OUTPUT_TAXONOMY/<encoder>/ subdirs (written by Stage 3.5 above).
# Skipped when tag_taxonomy.json is absent (Stage 1 also skipped) or via
# SKIP_STAGES=12.
if ! should_skip 12 && [ -f "$TAG_TAXONOMY" ]; then
    stamp "STAGE 12 · probe_taxonomy paired_delta (CPU, 16 dims × encoders, BCa 10K each)"
    python -u src/probe_taxonomy.py "--$MODE" \
        --stage paired_delta \
        --features-root "$OUTPUT_ACTION" \
        --output-root "$OUTPUT_TAXONOMY" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_taxonomy_paired_delta.log
fi

# ── STAGE 13 — plot (probe_taxonomy per-dim metric across encoders) ───
if ! should_skip 13 && [ -f "$TAG_TAXONOMY" ]; then
    stamp "STAGE 13 · probe_taxonomy plot (CPU, ~5s)"
    python -u src/probe_taxonomy.py "--$MODE" \
        --stage plot \
        --features-root "$OUTPUT_ACTION" \
        --output-root "$OUTPUT_TAXONOMY" \
        --cache-policy "$P_ACTION" \
        --no-wandb \
        2>&1 | tee logs/probe_taxonomy_plot.log
fi

# ── STAGE 7 — paired Δ (motion_cos) ────────────────────────────────────
if ! should_skip 7; then
    stamp "STAGE 7 · motion_cos paired_delta (CPU)"
    python -u src/probe_motion_cos.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_COS" \
        --cache-policy "$P_COS" \
        --no-wandb \
        2>&1 | tee logs/probe_motion_cos_paired_delta.log
fi

# ── STAGE 8 (forward) is run inside the per-encoder loop above ─────────
# (see "PER-ENCODER SEQUENTIAL PIPELINE" block). Kept here as a marker for
# anyone grepping for the stage; the actual `python probe_future_mse --stage
# forward` invocation lives in the per-encoder loop so its V-JEPA forward
# pass runs while features_test.npy for that encoder is still small/freed.

# ── STAGE 9 — paired Δ (future_mse, per-variant) ───────────────────────
if ! should_skip 9; then
    stamp "STAGE 9 · future_mse paired_per_variant (CPU)"
    python -u src/probe_future_mse.py "--$MODE" \
        --stage paired_per_variant \
        --output-root "$OUTPUT_MSE" \
        --cache-policy "$P_MSE" \
        --no-wandb \
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
        --no-wandb \
        2>&1 | tee logs/probe_plot.log
fi

# ── Final summary ──────────────────────────────────────────────────────
stamp "DONE · total wall = $(( ($(date +%s) - T0) / 60 )) min"
echo "Artifacts:"
echo "  🔥 P1 GATE     $OUTPUT_ACTION/probe_paired_delta.json"
echo "  motion_cos     $OUTPUT_COS/probe_motion_cos_paired.json"
echo "  future_mse     $OUTPUT_MSE/probe_future_mse_per_variant.json"
echo "  taxonomy lbls  $OUTPUT_TAXONOMY/taxonomy_labels.json (consumed by run_train.sh multi-task path)"
echo "  plots          $OUTPUT_PLOTS/{probe_action_loss,probe_action_acc,probe_encoder_comparison}.{png,pdf}"
echo "Per-encoder probe ckpts:"
for ENC in $ENCODERS; do
    echo "  $OUTPUT_ACTION/$ENC/probe.pt"
done
