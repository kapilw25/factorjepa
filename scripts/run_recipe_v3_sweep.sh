#!/usr/bin/env bash
# scripts/run_recipe_v3_sweep.sh — iter14 recipe-v3 7-cell drop-one ablation.
#
# Loops over R0–R6 cells (drop-one ablation). Each cell sets the seven
# recipe-v2/v3 env-var overrides → run_probe_train.sh forwards them as CLI
# flags to m09c → m09c merges into cfg → bit-identical pre-iter14 paths
# active when SUBSET=legacy & WARMUP=per_stage & {SALIENCY,SPD,REPLAY}=off.
#
# Sweep design (paper-grade drop-one ablation):
#   R0 = baseline (= prior Cell A)
#   R1 = Recipe-v3 (all 5 stacked + audit A2/A4)
#   R2 = R1 minus #1 (Frozen teacher)
#   R3 = R1 minus #2 (LP-FT)
#   R4 = R1 minus #3 (Surgical subset 4/8 → restored to legacy 12/24)
#   R5 = R1 minus #4 (SPD)
#   R6 = R1 minus #5 (CLEAR replay)
#
# Yields per-intervention contribution = R1 − R(i): the standard
# "if you drop X, you lose Y pp" ablation table for the paper.
#
# Wall: ~17 min/cell on RTX Pro 6000 Blackwell · 7 cells ≈ 2 h · ~$1.60 GPU.
# Idempotent: skips cells whose output dir + log already exist.
#
# USAGE:
#   ./scripts/run_recipe_v3_sweep.sh                 # all 7 cells, sequential
#   ./scripts/run_recipe_v3_sweep.sh R1              # just the full recipe-v3 cell
#   ./scripts/run_recipe_v3_sweep.sh R1 R5           # specific subset
#
# Precondition: T1 POC sampler fix landed; outputs/poc/probe_action/
# action_labels.json has 8 motion classes (verify via plan_surgery_wins.md §12.7).

set -euo pipefail

# ─── Sweep matrix ────────────────────────────────────────────────────────
# Format: name TEACHER LPFT SUBSET WARMUP SALIENCY SPD REPLAY
CELLS=(
  "R0_baseline             EMA     off   legacy     per_stage   off   off   off"
  "R1_recipe_v3            FROZEN  on    recipe_v3  single      on    on    on"
  "R2_minus_frozen         EMA     on    recipe_v3  single      on    on    on"
  "R3_minus_lpft           FROZEN  off   recipe_v3  single      on    on    on"
  "R4_minus_subset         FROZEN  on    legacy     single      on    on    on"
  "R5_minus_spd            FROZEN  on    recipe_v3  single      on    off   on"
  "R6_minus_replay         FROZEN  on    recipe_v3  single      on    on    off"
)

VARIANT=surgery_3stage_DI
LOG_DIR=logs
OUT_BASE=outputs/poc/m09c_surgery_3stage_DI

# ─── Optional cell filter (cmdline args = subset of cell names) ─────────
declare -a SELECTED=()
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        SELECTED+=("$arg")
    done
fi

cell_selected() {
    local name="$1"
    if [ "${#SELECTED[@]}" -eq 0 ]; then
        return 0
    fi
    for s in "${SELECTED[@]}"; do
        if [[ "$name" == "$s"* ]]; then
            return 0
        fi
    done
    return 1
}

# Note: POC label bootstrap is handled IN-PROCESS by m09c at startup
# (src/utils/probe_labels.ensure_probe_labels_for_mode reads cfg, generates
# the stratified-by-motion-class POC subset + action_labels.json if missing).
# Shells stay thin per CLAUDE.md — no orchestration here.

# ─── Sweep loop ─────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TOTAL_CELLS=${#CELLS[@]}
RUN_COUNT=0
for line in "${CELLS[@]}"; do
    read -r name TEACHER LPFT SUBSET WARMUP SALIENCY SPD REPLAY <<<"$line"
    if ! cell_selected "$name"; then
        echo "── SKIP $name (not in command-line filter) ──"
        continue
    fi
    RUN_COUNT=$((RUN_COUNT + 1))
    LOG="${LOG_DIR}/iter14_poc_recipe_v3_${name}.log"
    OUT_DIR_NAMED="${OUT_BASE}__${name}"

    if [ -d "$OUT_DIR_NAMED" ] && [ -f "$LOG" ]; then
        echo "── ${name}: output + log already exist, SKIPPING (delete to re-run) ──"
        continue
    fi

    echo "════════════════════════════════════════════════════════════"
    echo "🔬 Cell ${RUN_COUNT}/${TOTAL_CELLS}: ${name}"
    echo "    teacher=$TEACHER  lpft=$LPFT  subset=$SUBSET"
    echo "    warmup=$WARMUP  saliency=$SALIENCY  spd=$SPD  replay=$REPLAY"
    echo "    log:   $LOG"
    echo "    out:   $OUT_DIR_NAMED"
    echo "    start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    CACHE_POLICY_ALL=2 \
    TEACHER_MODE_OVERRIDE="$TEACHER" \
    LP_FT_OVERRIDE="$LPFT" \
    SUBSET_OVERRIDE="$SUBSET" \
    WARMUP_OVERRIDE="$WARMUP" \
    SALIENCY_OVERRIDE="$SALIENCY" \
    SPD_OVERRIDE="$SPD" \
    REPLAY_OVERRIDE="$REPLAY" \
        ./scripts/run_probe_train.sh "$VARIANT" --POC 2>&1 | tee "$LOG"

    # Move outputs to a cell-specific dir so subsequent cells start clean.
    if [ -d "$OUT_BASE" ]; then
        mv "$OUT_BASE" "$OUT_DIR_NAMED"
    fi

    echo "    end:   $(date '+%Y-%m-%d %H:%M:%S')  (output: $OUT_DIR_NAMED)"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🏁 Recipe-v3 sweep complete. Runs landed:"
for line in "${CELLS[@]}"; do
    read -r name _ <<<"$line"
    OUT_DIR_NAMED="${OUT_BASE}__${name}"
    if [ -d "$OUT_DIR_NAMED" ]; then
        echo "    ✅ $name → $OUT_DIR_NAMED"
    fi
done
echo "════════════════════════════════════════════════════════════"
echo "Next: aggregate trio top-1 / motion_cos / future_l1 across cells →"
echo "      apply plan_surgery_wins.md §7.5 decision tree (🟢/🟡/🔴)."
