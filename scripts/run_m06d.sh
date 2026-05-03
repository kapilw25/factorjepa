#!/usr/bin/env bash
# iter13 m06d orchestrator — 3 modules × {V-JEPA, DINOv2} × stages.
# Per CLAUDE.md "shell scripts are THIN wrappers — all logic in Python"
# and "DELETE PROTECTION — shells stay THIN, .py owns the cache-policy prompt".
#
# Pipeline (priority 1 — frozen V-JEPA vs frozen DINOv2 on Indian action probe
# applied to data/eval_10k_local):
#   STAGE 1   m06d_action_probe.py --stage labels        (CPU, ~1 min)
#   STAGE 2   m06d_action_probe.py --stage features      (GPU × 2 encoders, ~1 h)
#   STAGE 3   m06d_action_probe.py --stage train         (GPU × 2 encoders, ~30 min)
#   STAGE 4   m06d_action_probe.py --stage paired_delta  (CPU, ~5 min)  ← P1 GATE
#   STAGE 5   m06d_motion_cos.py   --stage features      (CPU mean-pool × 2 enc)
#   STAGE 6   m06d_motion_cos.py   --stage cosine        (CPU × 2 enc)
#   STAGE 7   m06d_motion_cos.py   --stage paired_delta  (CPU)
#   STAGE 8   m06d_future_mse.py   --stage forward       (GPU, V-JEPA only, ~30 min)
#   STAGE 9   m06d_future_mse.py   --stage paired_per_variant   (CPU)
#   STAGE 10  m08d_plot_m06d.py                                  (CPU, ~5s — plots)
#
# REFERENCES
#   plan_code_dev.md  — per-module specs + LoC budget
#   runbook.md        — full set of pre/post-flight one-liners
#
# USAGE
#   tmux new -s m06d
#
#   # FULL run (default) — eval_10k (~9.9k clips), ~4h on 24GB / ~2.5h on 96GB
#   ./scripts/run_m06d.sh 2>&1 | tee logs/run_src_m06d_v1.log
#
#   # SANITY smoke test — 150 stratified clips (50/class) from THE SAME eval_10k
#   # JSON, processed against the SAME eval_10k_local/ TARs. ~6-8 min on 24GB.
#   # Outputs sandboxed to outputs/sanity/. Pass --sanity OR set MODE=SANITY.
#   ./scripts/run_m06d.sh --sanity 2>&1 | tee logs/run_src_m06d_sanity_v1.log
#
#   # Bypass prompts (overnight / non-TTY)
#   CACHE_POLICY_ALL=1 ./scripts/run_m06d.sh           # keep all caches
#   CACHE_POLICY_ALL=2 ./scripts/run_m06d.sh           # recompute everything
#   CACHE_POLICY_ALL=2 ./scripts/run_m06d.sh --sanity  # SANITY + recompute
#
#   # Skip a stage (resume after failure):
#   SKIP_STAGES="1,2" ./scripts/run_m06d.sh
#   # Run only one variant (debug):
#   ENCODERS="vjepa_2_1_frozen" ./scripts/run_m06d.sh
#   # Tune SANITY subset size (default 50 clips per class):
#   SANITY_N_PER_CLASS=20 ./scripts/run_m06d.sh --sanity
#
#   # Probe-training knobs (Stage 3) — mode-aware defaults:
#   #   SANITY: EPOCHS=50  WARMUP_PCT=0.10  LR_SWEEP="5e-4"   (current behavior)
#   #   FULL:   EPOCHS=20  WARMUP_PCT=0.0   LR_SWEEP="5e-4"   (Meta-faithful single-LR)
#   # Override any of them via env vars:
#   EPOCHS=20 WARMUP_PCT=0.0 ./scripts/run_m06d.sh --sanity   # apply Meta recipe to sanity
#
#   # Paper-faithful FULL with LR sweep (matches Meta's multihead 5-LR setup;
#   # ~4x stage-3 wall, but mirrors deps/vjepa2/configs/eval/vitg-384/ssv2.yaml):
#   LR_SWEEP="1e-4 3e-4 1e-3 3e-3" ./scripts/run_m06d.sh
#   # Each LR trains a probe under outputs/full/m06d_action_probe/<encoder>/lr_<LR>/;
#   # the best-val-acc LR is symlinked to the canonical <encoder>/probe.pt path so
#   # Stage 4 paired_delta reads the winner (no code change needed).

# Fail-fast: m06d stages have sequential dependencies (Stage N+1 needs N's outputs),
# so any stage failure → abort chain to avoid wasting GPU time on impossible
# downstream work. Differs from run_paired_eval_10k.sh which uses `set -uo` only
# because its variants are independent (errors_N_fixes #72).
set -euo pipefail
# Clear ERR trap surfaces line + exit code BEFORE the shell exits.
trap 'rc=$?; echo "" >&2; echo "❌ FATAL: run_m06d.sh aborted at line $LINENO (exit=$rc) — sequential dependency failed; downstream stages skipped" >&2; exit $rc' ERR

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
OUTPUT_ACTION="${OUTPUT_ACTION:-${DEFAULT_OUTPUT_PREFIX}/m06d_action_probe}"
OUTPUT_COS="${OUTPUT_COS:-${DEFAULT_OUTPUT_PREFIX}/m06d_motion_cos}"
OUTPUT_MSE="${OUTPUT_MSE:-${DEFAULT_OUTPUT_PREFIX}/m06d_future_mse}"
OUTPUT_PLOTS="${OUTPUT_PLOTS:-${DEFAULT_OUTPUT_PREFIX}/m08d_plot_m06d}"
ENCODERS="${ENCODERS:-vjepa_2_1_frozen dinov2}"
SKIP_STAGES="${SKIP_STAGES:-}"
NUM_FRAMES="${NUM_FRAMES:-16}"

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
echo "run_m06d cache-policy gather (UPFRONT — one prompt per module)"
echo "──────────────────────────────────────────────"
_check_and_prompt "m06d_action" "${OUTPUT_ACTION}/*"
_check_and_prompt "m06d_cos"    "${OUTPUT_COS}/*"
_check_and_prompt "m06d_mse"    "${OUTPUT_MSE}/*"
P_ACTION="${POLICY[m06d_action]:-1}"
P_COS="${POLICY[m06d_cos]:-1}"
P_MSE="${POLICY[m06d_mse]:-1}"
echo "  → ACTION=$P_ACTION  COS=$P_COS  MSE=$P_MSE"

# ── STAGE 1 — labels ────────────────────────────────────────────────────
if ! should_skip 1; then
    stamp "STAGE 1 · action_probe labels (CPU, ~1 min)"
    python -u src/m06d_action_probe.py "--$MODE" \
        --stage labels \
        --eval-subset "$EVAL_SUBSET" \
        --tags-json "$TAGS_JSON" \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        2>&1 | tee logs/m06d_action_probe_labels.log
fi

# ── STAGE 2 — features (action_probe) per encoder ──────────────────────
if ! should_skip 2; then
    stamp "STAGE 2 · action_probe features (GPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        EXTRA_CKPT=""
        if [[ "$ENC" == vjepa* ]]; then
            EXTRA_CKPT="--encoder-ckpt $ENCODER_CKPT"
        fi
        python -u src/m06d_action_probe.py "--$MODE" \
            --stage features \
            --encoder "$ENC" \
            $EXTRA_CKPT \
            --eval-subset "$EVAL_SUBSET" \
            --local-data "$LOCAL_DATA" \
            --output-root "$OUTPUT_ACTION" \
            --num-frames "$NUM_FRAMES" \
            --cache-policy "$P_ACTION" \
            2>&1 | tee "logs/m06d_action_probe_features_${ENC}.log"
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
            python -u src/m06d_action_probe.py "--$MODE" \
                --stage train \
                --encoder "$ENC" \
                --output-root "$OUTPUT_ACTION" \
                --epochs "$EPOCHS" \
                --probe-lr "$LR" \
                --warmup-pct "$WARMUP_PCT" \
                $SUBDIR_FLAG \
                --cache-policy "$P_ACTION" \
                2>&1 | tee "logs/m06d_action_probe_train_${ENC}${LOG_SUFFIX}.log"
        done
        # Multi-LR sweep: pick best-val-acc LR + symlink its outputs to canonical
        # paths so Stage 4 (paired_delta) reads the winner. Logic lives in
        # m06d_action_probe.py::run_select_best_lr_stage (per CLAUDE.md "shell
        # scripts are THIN wrappers — all logic in Python"; idempotent so re-runs
        # are safe).
        if [ "$n_lrs" -gt 1 ]; then
            python -u src/m06d_action_probe.py "--$MODE" \
                --stage select_best_lr \
                --encoder "$ENC" \
                --output-root "$OUTPUT_ACTION" \
                --cache-policy "$P_ACTION" \
                2>&1 | tee "logs/m06d_action_probe_select_best_lr_${ENC}.log"
        fi
    done
fi

# ── STAGE 4 — paired Δ (action_probe) ←── 🔥 PRIORITY 1 GATE ──────────
if ! should_skip 4; then
    stamp "STAGE 4 · action_probe paired_delta (🔥 P1 GATE — CPU, ~5 min, BCa 10K)"
    python -u src/m06d_action_probe.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_ACTION" \
        --cache-policy "$P_ACTION" \
        2>&1 | tee logs/m06d_action_probe_paired_delta.log
fi

# ── STAGE 5 — features (motion_cos) per encoder ────────────────────────
if ! should_skip 5; then
    stamp "STAGE 5 · motion_cos features (CPU mean-pool from action_probe × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/m06d_motion_cos.py "--$MODE" \
            --stage features \
            --encoder "$ENC" \
            --action-probe-root "$OUTPUT_ACTION" \
            --output-root "$OUTPUT_COS" \
            --share-features \
            --cache-policy "$P_COS" \
            2>&1 | tee "logs/m06d_motion_cos_features_${ENC}.log"
    done
fi

# ── STAGE 6 — cosine (motion_cos) per encoder ──────────────────────────
if ! should_skip 6; then
    stamp "STAGE 6 · motion_cos cosine (CPU × ${ENCODERS//[^[:space:]]/x} encoders)"
    for ENC in $ENCODERS; do
        python -u src/m06d_motion_cos.py "--$MODE" \
            --stage cosine \
            --encoder "$ENC" \
            --action-probe-root "$OUTPUT_ACTION" \
            --output-root "$OUTPUT_COS" \
            --cache-policy "$P_COS" \
            2>&1 | tee "logs/m06d_motion_cos_cosine_${ENC}.log"
    done
fi

# ── STAGE 7 — paired Δ (motion_cos) ────────────────────────────────────
if ! should_skip 7; then
    stamp "STAGE 7 · motion_cos paired_delta (CPU)"
    python -u src/m06d_motion_cos.py "--$MODE" \
        --stage paired_delta \
        --output-root "$OUTPUT_COS" \
        --cache-policy "$P_COS" \
        2>&1 | tee logs/m06d_motion_cos_paired_delta.log
fi

# ── STAGE 8 — forward (future_mse) — V-JEPA frozen only ────────────────
if ! should_skip 8; then
    stamp "STAGE 8 · future_mse forward (GPU, V-JEPA frozen only, ~30 min)"
    python -u src/m06d_future_mse.py "--$MODE" \
        --stage forward \
        --variant vjepa_2_1_frozen \
        --encoder-ckpt "$ENCODER_CKPT" \
        --action-probe-root "$OUTPUT_ACTION" \
        --local-data "$LOCAL_DATA" \
        --output-root "$OUTPUT_MSE" \
        --num-frames "$NUM_FRAMES" \
        --cache-policy "$P_MSE" \
        2>&1 | tee logs/m06d_future_mse_forward_vjepa.log
fi

# ── STAGE 9 — paired Δ (future_mse, per-variant) ───────────────────────
if ! should_skip 9; then
    stamp "STAGE 9 · future_mse paired_per_variant (CPU)"
    python -u src/m06d_future_mse.py "--$MODE" \
        --stage paired_per_variant \
        --output-root "$OUTPUT_MSE" \
        --cache-policy "$P_MSE" \
        2>&1 | tee logs/m06d_future_mse_paired.log
fi

# ── STAGE 10 — plots (m08d, CPU, always-recompute) ─────────────────────
# Pure visualization — no cache_policy. Wipes its own output_dir on entry
# (single-owner, m08b-style). Reads JSONs + train_log.jsonl from prior stages.
if ! should_skip 10; then
    stamp "STAGE 10 · m08d plot m06d (CPU, ~5s)"
    python -u src/m08d_plot_m06d.py "--$MODE" \
        --action-probe-root "$OUTPUT_ACTION" \
        --motion-cos-root   "$OUTPUT_COS" \
        --future-mse-root   "$OUTPUT_MSE" \
        --output-dir        "$OUTPUT_PLOTS" \
        2>&1 | tee logs/m08d_plot_m06d.log
fi

# ── Final summary ──────────────────────────────────────────────────────
stamp "DONE · total wall = $(( ($(date +%s) - T0) / 60 )) min"
echo "Artifacts:"
echo "  🔥 P1 GATE     $OUTPUT_ACTION/m06d_paired_delta.json"
echo "  motion_cos     $OUTPUT_COS/m06d_motion_cos_paired.json"
echo "  future_mse     $OUTPUT_MSE/m06d_future_mse_per_variant.json"
echo "  plots          $OUTPUT_PLOTS/{m06d_action_probe_loss,m06d_action_probe_acc,m06d_encoder_comparison}.{png,pdf}"
echo "Per-encoder probe ckpts:"
for ENC in $ENCODERS; do
    echo "  $OUTPUT_ACTION/$ENC/probe.pt"
done
