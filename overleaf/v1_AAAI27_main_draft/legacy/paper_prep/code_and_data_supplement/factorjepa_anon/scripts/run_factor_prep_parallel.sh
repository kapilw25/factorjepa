#!/usr/bin/env bash
# iter13 v13 FIX-26 (2026-05-07): parallel m10 orchestrator + chained m11 streaming.
# Mirror of run_factor_prep.sh but with N parallel m10 workers instead of 1 serial.
#
# Estimated speedup at FULL 10K on RTX Pro 6000 Blackwell:
#   2 workers: ~1.5x   |   4 workers: ~2x   |   6 workers: ~2.4x (CPU saturates at 32 cores)
#
# Each m10 worker loads its own DINO (~500 MB GPU MEM) + SAM3 (~3.5 GB) and processes
# a disjoint slice of clips into a private output dir m10_sam_segment_w{i}/. After
# all workers finish, merge step unions segments.json + masks/ into the canonical
# m10_sam_segment/ dir. Then m11 streaming runs ONCE on the merged canonical dir
# (verify-100 cap → ~1 min wall regardless of FULL scale).
#
# USAGE:
#   ./scripts/run_factor_prep_parallel.sh <factor-yaml> [n-workers] [--SANITY|--POC|--FULL]
#
# Examples:
#   ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml 2 --SANITY
#   ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml 4 --FULL
#   N_WORKERS=6 ./scripts/run_factor_prep_parallel.sh configs/train/surgery_3stage_DI_encoder.yaml
#
# Env-var overrides (full parity with run_factor_prep.sh):
#   LOCAL_DATA       Override the active data dir (default: yaml-keyed
#                    cfg.data.local_data_dir from configs/pipeline.yaml).
#   TRAIN_SUBSET     Optional subset JSON to filter clip set further. Unset →
#                    use ${LOCAL_DATA}/manifest.json (all clips). When set,
#                    must point at a JSON with "clip_keys" list (e.g.
#                    train_split.json, val_split.json, eval_10k.json).
#                    m10_split_subset.py accepts both "saved_keys" + "clip_keys".
#   CACHE_POLICY_ALL 1=keep / 2=recompute. Applied to worker scratch dirs
#                    (uniform across workers; heterogeneous would yield
#                    inconsistent merge state). m11 inherits worker policy
#                    via dep-prop (m10 recompute → m11 recompute) per serial
#                    wrapper semantics. If UNSET on a TTY, prompts per-key
#                    for each existing cache; on non-TTY, defaults to 1=keep
#                    (safe-resume).
#
# ⚠️ Resume semantics — IMPORTANT for partial-run recovery:
#   • First run, empty disk      → policy=1 effectively (no caches exist).
#   • Re-run after killed partial:
#       CACHE_POLICY_ALL=1 (or interactive Enter) → workers RESUME from
#         each worker's own .m10_checkpoint_*_<fp>.npz (saves ~25-30 min).
#       CACHE_POLICY_ALL=2 → workers WIPE scratch + restart from clip 0 of
#         their slice. Use only when you want a fresh restart.
#   • Canonical segments.json is NEVER wiped by this script. To force a
#     fully fresh m10 run, manually `rm canonical_dir/segments.json` and
#     run with CACHE_POLICY_ALL=2.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "USAGE: $0 <factor-yaml> [n-workers]" >&2
    # iter18 H7: example path composed from pipeline.yaml (no inline literal).
    echo "  Example: $0 $(scripts/lib/yaml_extract.py configs/pipeline.yaml arm_train_configs.surgery_3stage_DI_encoder) 4" >&2
    exit 2
fi

FACTOR_YAML="$1"
N_WORKERS="${2:-${N_WORKERS:-4}}"
MODE_FLAG="${3:---FULL}"

if [ ! -f "$FACTOR_YAML" ]; then
    echo "FATAL: factor yaml not found: $FACTOR_YAML" >&2
    exit 3
fi
if ! [[ "$N_WORKERS" =~ ^[0-9]+$ ]] || [ "$N_WORKERS" -lt 1 ]; then
    echo "FATAL: n-workers must be a positive integer (got: $N_WORKERS)" >&2
    exit 2
fi
case "$MODE_FLAG" in
    --SANITY|--sanity) MODE="SANITY" ;;
    --POC|--poc)       MODE="POC" ;;
    --FULL|--full)     MODE="FULL" ;;
    *) echo "FATAL: mode flag must be --SANITY|--POC|--FULL (got: $MODE_FLAG)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
source venv_denseworld/bin/activate
mkdir -p logs

# iter16 M9 (2026-05-21): default from yaml-keyed data.local_data_dir. Env var
# LOCAL_DATA still overrides for ad-hoc runs on alternate corpora.
_LOCAL_DATA_M9=$(scripts/lib/yaml_extract.py configs/pipeline.yaml data.local_data_dir)
LOCAL_DATA="${LOCAL_DATA:-$_LOCAL_DATA_M9}"
# iter15 plan gap #1 fix (2026-05-21): TRAIN_SUBSET env var parity with serial.
TRAIN_SUBSET="${TRAIN_SUBSET:-}"

if [ ! -d "$LOCAL_DATA" ]; then
    echo "FATAL: LOCAL_DATA=$LOCAL_DATA is not a directory." >&2
    exit 3
fi
if [ ! -f "$LOCAL_DATA/manifest.json" ]; then
    echo "FATAL: $LOCAL_DATA/manifest.json missing — needed for clip key universe." >&2
    exit 3
fi
if [ -n "$TRAIN_SUBSET" ] && [ ! -e "$TRAIN_SUBSET" ]; then
    echo "FATAL: TRAIN_SUBSET=$TRAIN_SUBSET set but file not found." >&2
    echo "  Unset TRAIN_SUBSET to iterate all clips in $LOCAL_DATA/manifest.json." >&2
    exit 3
fi

CANONICAL_DIR="${LOCAL_DATA}/m10_sam_segment"
SUBSET_DIR="/tmp/m10_parallel_subsets_$$"
mkdir -p "$SUBSET_DIR"

VARIANT_TAG="$(basename "$FACTOR_YAML" .yaml)"
T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

# ── iter15 plan gap #3 fix (2026-05-21) — interactive cache-policy prompt ──
# Mirror of serial's _check_and_prompt from scripts/run_factor_prep.sh:103-136.
# Probes each worker scratch dir + canonical for existing caches; if any
# found, prompts (TTY) or applies CACHE_POLICY_ALL (env) or defaults to
# 1=keep (non-TTY). Replaces the prior silent CACHE_POLICY="${CACHE_POLICY_ALL:-2}"
# default which silently wiped 25-30 min of worker scratch progress on
# resume-after-kill.
declare -A POLICY
_check_and_prompt() {
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        # NOTE: single-statement `local hit=$(...)` is required — split form
        # propagates compgen's exit 1 under set -e + pipefail (see serial
        # wrapper comment at run_factor_prep.sh:107-114).
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
echo "factor-prep parallel · mode=${MODE} · N_WORKERS=${N_WORKERS} · variant=${VARIANT_TAG}"
echo "  LOCAL_DATA:    $LOCAL_DATA"
if [ -n "$TRAIN_SUBSET" ]; then
    echo "  TRAIN_SUBSET:  $TRAIN_SUBSET  (filtering clip set)"
else
    echo "  TRAIN_SUBSET:  <unset>  (using \$LOCAL_DATA/manifest.json — all clips)"
fi
echo "  CANONICAL_DIR: $CANONICAL_DIR"
echo "  SUBSET_DIR:    $SUBSET_DIR (auto-cleaned at exit)"
echo "──────────────────────────────────────────────"

# Probe each worker scratch dir + canonical + m11 (resume-anchor heuristic).
echo
echo "Cache-policy probe (set CACHE_POLICY_ALL=1|2 to bypass prompts):"
for i in $(seq 0 $((N_WORKERS - 1))); do
    _check_and_prompt "worker_$i" "${LOCAL_DATA}/m10_sam_segment_w${i}/*"
done
_check_and_prompt "canonical" "${CANONICAL_DIR}/segments.json"
_check_and_prompt "m11"       "${LOCAL_DATA}/m11_factor_datasets/*"

# Workers share a uniform policy (heterogeneous across slices would yield
# inconsistent merge state). m11 gets its own probed policy, then dep-prop:
# m10 (workers) recompute MUST force m11 recompute (m11 reads m10's segments
# + masks). Matches serial wrapper's m10→m11 dep prop semantics (gap fix
# 2026-05-21, was uniform-policy-only).
WORKER_POLICY="${POLICY[worker_0]:-1}"
M11_POLICY="${POLICY[m11]:-1}"
if [ "$WORKER_POLICY" = "2" ]; then
    if [ "$M11_POLICY" != "2" ]; then
        echo "  ⚠️  dep-prop: m10 workers recompute → forcing m11 recompute too"
    fi
    M11_POLICY=2
fi
echo "  → workers CACHE_POLICY=$WORKER_POLICY · m11 CACHE_POLICY=$M11_POLICY"
echo "──────────────────────────────────────────────"

stamp "Step 1/5 — split clips into ${N_WORKERS} disjoint subsets"
# iter15 plan gap #1 fix (2026-05-21): TRAIN_SUBSET overrides manifest.json
# as the input clip universe. m10_split_subset.py accepts both "saved_keys"
# (manifest.json) and "clip_keys" (subset JSON) schemas.
SPLITTER_INPUT="${TRAIN_SUBSET:-${LOCAL_DATA}/manifest.json}"
python -u src/utils/m10_split_subset.py \
    --manifest "$SPLITTER_INPUT" \
    --existing-segments "${CANONICAL_DIR}/segments.json" \
    --n-workers "$N_WORKERS" \
    --out-dir "$SUBSET_DIR"

stamp "Step 2/5 — spawn ${N_WORKERS} m10 workers (parallel)"
# iter15 plan cosmetic fix (2026-05-21): log filename includes ${MODE,,} so
# SANITY/POC/FULL logs land in distinct files (was hardcoded "factor_full_*"
# regardless of mode — misleading at SANITY/POC).
PIDS=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    SUBSET_JSON="${SUBSET_DIR}/subset_w${i}.json"
    OUTPUT_DIR="${LOCAL_DATA}/m10_sam_segment_w${i}"
    LOG="logs/factor_${MODE,,}_${VARIANT_TAG}_w${i}.log"

    echo "  worker $i:"
    echo "    subset:  $SUBSET_JSON"
    echo "    output:  $OUTPUT_DIR"
    echo "    log:     $LOG"

    CACHE_POLICY_ALL="$WORKER_POLICY" \
        python -u src/m10_sam_segment.py "$MODE_FLAG" \
        --train-config "$FACTOR_YAML" \
        --subset "$SUBSET_JSON" \
        --local-data "$LOCAL_DATA" \
        --output-dir "$OUTPUT_DIR" \
        --no-wandb \
        --cache-policy "$WORKER_POLICY" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done
echo "  spawned PIDs: ${PIDS[*]}"
echo "  monitor:  tail -f logs/factor_${MODE,,}_${VARIANT_TAG}_w*.log"
echo "  GPU watch: nvtop  (or: watch -n2 nvidia-smi)"

stamp "Step 3/5 — wait for all ${N_WORKERS} workers"

# iter16 (2026-05-22): heartbeat loop — every 60s, scrape latest tqdm progress
# from each worker's log and print a consolidated summary to the orchestrator
# log. Without this, the orchestrator goes silent for ~4-6 hr on a FULL run
# (just "Step 3/5 — wait for all workers" with no updates until the first
# worker exits). The operator had to tail 6 separate worker logs to see any
# progress — terrible UX for tmux-detached overnight runs.
HEARTBEAT_INTERVAL=60
_heartbeat() {
    while sleep "$HEARTBEAT_INTERVAL"; do
        local alive=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
        if [ "$alive" -eq 0 ]; then
            return
        fi
        echo
        echo "─── $(date '+%H:%M:%S') · heartbeat ($alive/$N_WORKERS workers alive) ───"
        for i in $(seq 0 $((N_WORKERS - 1))); do
            local log="logs/factor_${MODE,,}_${VARIANT_TAG}_w${i}.log"
            if [ ! -f "$log" ]; then
                echo "  w$i: (log not yet created)"
                continue
            fi
            # Extract latest tqdm progress line. Fallback to last non-blank
            # line if no tqdm progress visible yet (e.g., still in warmup).
            local prog
            prog=$(grep -aoE '[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ ?\[[^]]+\]' "$log" 2>/dev/null | tail -1)
            if [ -n "$prog" ]; then
                echo "  w$i: $prog"
            else
                local last
                last=$(tail -n 20 "$log" 2>/dev/null | grep -avE '^[[:space:]]*$' | tail -1 | head -c 180)
                echo "  w$i: (warmup) $last"
            fi
        done
    done
}
_heartbeat &
HEARTBEAT_PID=$!

EXIT_CODES=()
FAILED_WORKERS=()
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    if wait "$pid"; then
        ec=0
    else
        ec=$?
        FAILED_WORKERS+=("$idx")
    fi
    EXIT_CODES+=("$ec")
    echo "  worker $idx (pid $pid): exit=$ec"
done

# Stop heartbeat once all workers have exited. kill/wait may return non-zero
# if the heartbeat already exited (race) — set +e/-e to allow that, matching
# the idiom at setup_env_uv.sh:595-597.
set +e
kill "$HEARTBEAT_PID" 2>/dev/null
wait "$HEARTBEAT_PID" 2>/dev/null
set -e
echo "  All workers done. Exit codes: ${EXIT_CODES[*]}"
if [ ${#FAILED_WORKERS[@]} -gt 0 ]; then
    echo "  WARN: ${#FAILED_WORKERS[@]} worker(s) exited non-zero: ${FAILED_WORKERS[*]}"
    echo "    (m10's quality_gate FAIL also exits non-zero — check log; "
    echo "     segments.json is saved BEFORE the gate so merge still has the data)"
fi

stamp "Step 4/5 — merge worker outputs into ${CANONICAL_DIR}"
WORKER_DIRS=()
for i in $(seq 0 $((N_WORKERS - 1))); do
    WORKER_DIRS+=("${LOCAL_DATA}/m10_sam_segment_w${i}")
done
python -u src/utils/m10_merge.py \
    --canonical-dir "$CANONICAL_DIR" \
    --worker-dirs "${WORKER_DIRS[@]}"

rm -rf "$SUBSET_DIR"

stamp "Step 5/5 — m11 --streaming (factor materialization on merged canonical)"
# iter15 plan cosmetic fix (2026-05-21): m11 log filename includes ${MODE,,}
# matching the serial wrapper's `logs/run_factor_prep_*_${MODE,,}_m11.log`
# convention so SANITY/POC/FULL runs land in distinct files.
M11_LOG="logs/run_factor_prep_parallel_${VARIANT_TAG}_${MODE,,}_m11.log"
echo "  log: $M11_LOG"
# m11 streaming reads merged segments.json + masks/ from canonical dir, materializes
# the verify-100 cap subset into D_L/D_A/D_I + factor_manifest. Wall ~1 min at FULL.
# iter15 plan functional fix (2026-05-21): pass --subset $TRAIN_SUBSET to m11
# when set, so m11 filters to the same clip universe the workers operated on.
# Without this, m11 over-processes any stale prior-run data in canonical.
# Matches serial wrapper's $SUBSET_FLAG threading.
SUBSET_FLAG=""
if [ -n "$TRAIN_SUBSET" ]; then
    SUBSET_FLAG="--subset $TRAIN_SUBSET"
fi
# Unquoted $SUBSET_FLAG expansion is intentional — expands to 2 tokens when
# set, 0 tokens when empty (same pattern as serial run_factor_prep.sh:162).
CACHE_POLICY_ALL="$M11_POLICY" \
    python -u src/m11_factor_datasets.py "$MODE_FLAG" --streaming \
    --train-config "$FACTOR_YAML" \
    $SUBSET_FLAG --local-data "$LOCAL_DATA" \
    --no-wandb \
    --cache-policy "$M11_POLICY" \
    2>&1 | tee "$M11_LOG"

DUR=$(( $(date +%s) - T0 ))
stamp "✅ factor-prep parallel done · wall=$((DUR/3600))h$(((DUR%3600)/60))m"
echo "Outputs (co-located inside ${LOCAL_DATA} — raw per-clip files only;"
echo "         tar shards + HF push are produced by 'hf_outputs.py upload-data'):"
echo "  ${CANONICAL_DIR}/  (segments.json + summary.json + masks/*.npz)"
echo "  ${LOCAL_DATA}/m11_factor_datasets/  (factor_manifest.json + D_L,D_A,D_I/*.npy + verify samples)"
echo
echo "Optional follow-ups:"
echo "  · regenerate m10 plots from merged segments.json (CPU only, ~2 min):"
echo "      python -u src/m10_sam_segment.py --plot --train-config ${FACTOR_YAML} \\"
echo "          --local-data ${LOCAL_DATA} --no-wandb"
echo "  · once verified, free disk by removing worker scratch dirs:"
for wd in "${WORKER_DIRS[@]}"; do
    echo "      rm -rf ${wd}"
done
