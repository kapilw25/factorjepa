#!/usr/bin/env bash
# One-shot setup for a fresh rented NVIDIA box: git clone -> venv -> data/outputs from HF -> verify.
#
# Steps (every step prints first; --dry-run prints them all and runs none):
#   1 preflight  Linux, nvidia-smi + compute capability, .env has an HF_TOKEN line (never printed),
#                pipeline.yaml data dir matches the profile
#   2 env        ./setup_env_uv.sh --gpu (adds --from-wheels on compute capability 12.x, the sm_120
#                prebuilt flash-attention wheel), then source venv_walkindia/bin/activate
#   3 disk       free disk vs the profile's download + working space; FATAL when short
#   4 download   whole HF subtrees via src/utils/hf_outputs.py download-data (never --ext / --exclude)
#   5 heldout    eval-heldout only: cut the labelled held-out clips from the HF corpus shards
#   6 verify     outputs trees: hf_outputs.py verify; data trees: required files on disk; base checkpoint
#   7 summary    one table + the next command to run
#
# Profiles (sizes measured from the HF tree listing of anonymousML123/factorjepa-outputs, 2026-11-28):
#   eval-heldout  ~3 GB    video held-out test, 4 saved 1B 10k models (checkpoints leased at run time)
#   train-1b-10k  ~21 GB   train 1B arms on eval_10k (run_train.sh / ngpu_run.py --mode POC)
#   eval-10k      ~244 GB  eval_10k data + every saved 1B 10k arm and eval (run_eval.sh --POC)
#   full          ~319 GB  115k corpus + the 1B FULL outputs (ngpu_run.py --mode FULL, iter19)
#
# USAGE (repo root, inside tmux):
#   bash scripts/setup_new_box.sh --profile eval-heldout --dry-run
#   set -o pipefail; mkdir -p logs; bash scripts/setup_new_box.sh --profile eval-heldout \
#       2>&1 | tee logs/setup_new_box_$(date +%Y%m%d_%H%M%S).log
#   # re-run after a fix, keeping the venv that is already built:
#   bash scripts/setup_new_box.sh --profile eval-10k --skip-env 2>&1 | tee logs/setup_new_box_$(date +%Y%m%d_%H%M%S).log
#
# Bash 3.2 compatible (the Mac /bin/bash) so --dry-run also works on a laptop.

set -euo pipefail
trap 'rc=$?; echo "" >&2; echo "FATAL: setup_new_box.sh aborted at line $LINENO (exit=$rc)" >&2; exit $rc' ERR

cd "$(dirname "$0")/.."
mkdir -p logs

usage() {
    echo "USAGE: bash scripts/setup_new_box.sh --profile {eval-heldout|train-1b-10k|eval-10k|full} [--dry-run] [--skip-env]"
    echo "  --dry-run   print every step and command, run none (safe on a laptop)"
    echo "  --skip-env  do not run setup_env_uv.sh (venv_walkindia must already exist)"
}

PROFILE=""
DRY_RUN=0
SKIP_ENV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)  [ $# -ge 2 ] || { usage >&2; exit 2; }; PROFILE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --skip-env) SKIP_ENV=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "FATAL: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$PROFILE" ] || { echo "FATAL: --profile is required" >&2; usage >&2; exit 2; }

# ── Profile table ──────────────────────────────────────────────────────────────────────────
# DL_GB = what lands on disk (HF tree listing, 2026-11-28). WORK_GB = working space the runs need on
# top of that; it also covers the transient tar-unpack peak of the data trees (m10 masks + m11
# factors: 10.8 GB for eval_10k_local, 83 GB for full_local). Every profile uses the 1B backbone.
# NEED_DATA_DIR / NEED_MANIFEST = what configs/pipeline.yaml data.local_data_dir / master_manifest_name
# must say for the run scripts (run_train.sh and the python config loader read them directly).
BACKBONE="vjepa_2_1_vitg"
NEED_DATA_DIR=""
NEED_MANIFEST=""
case "$PROFILE" in
    eval-heldout)
        # outputs/heldout_video 0.04 GB + full_local.json 0.01 GB + ~1,702 cut clips (~1.13 MB each).
        # WORK: frame cache 1,702 clips x ~19.7 MB = ~34 GB, 4 parallel checkpoint leases x 8.3 GB = ~33 GB.
        DL_GB=3;   WORK_GB=75
        WHAT="video held-out test (heldout_video_eval.sh / heldout_video_parallel.sh), 4 saved 1B 10k models" ;;
    train-1b-10k)
        # data/eval_10k_local 20.9 GB. WORK: ngpu_run.py's own POC gate for --cache 2 (250 GB).
        DL_GB=21;  WORK_GB=250
        NEED_DATA_DIR="data/eval_10k_local"; NEED_MANIFEST="eval_10k.json"
        WHAT="train 1B arms on eval_10k (run_train.sh --POC, ngpu_run.py --mode POC)" ;;
    eval-10k)
        # data/eval_10k_local 20.9 GB + outputs/poc/vjepa_2_1_vitg_1B 222.8 GB (train 122.5 + eval 100.3).
        # WORK: frame cache of the ~2k eval_10k test clips (~39 GB) + new eval outputs + margin.
        DL_GB=244; WORK_GB=100
        NEED_DATA_DIR="data/eval_10k_local"; NEED_MANIFEST="eval_10k.json"
        WHAT="evaluate the saved 1B 10k arms on eval_10k (run_eval.sh --POC); training also possible" ;;
    full)
        # data/full_local 213.7 GB + outputs/full 104.7 GB. WORK: set after the venv is up, from
        # probe.eval_frame_cache.max_cache_gb + 40, the same rule ngpu_run.py's disk gate applies.
        DL_GB=319; WORK_GB=""
        NEED_DATA_DIR="data/full_local"; NEED_MANIFEST="full_local.json"
        WHAT="FULL 115k run, 1B (ngpu_run.py --mode FULL, run_eval.sh --FULL), iter19" ;;
    *) echo "FATAL: unknown profile '$PROFILE'" >&2; usage >&2; exit 2 ;;
esac

# ── Helpers ───────────────────────────────────────────────────────────────────────────────
SUMMARY=()
add_row() { SUMMARY+=("$1|$2|$3"); }

step() { echo ""; echo "════ [$1] $2"; }

# run <log_name> <cmd...>: every long command is tee'd to logs/<log_name>_<timestamp>.log.
run() {
    local name="$1"; shift
    local log
    log="logs/${name}_$(date +%Y%m%d_%H%M%S).log"
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [dry-run] $(printf '%q ' "$@")2>&1 | tee $log"
        return 0
    fi
    echo "  [run] $(printf '%q ' "$@")2>&1 | tee $log"
    "$@" 2>&1 | tee "$log"
}

# yq <dotted.key>: one value from configs/pipeline.yaml via scripts/lib/yaml_extract.py (needs the venv).
yq() {
    if [ "$HAVE_VENV" = 1 ]; then
        python scripts/lib/yaml_extract.py configs/pipeline.yaml "$1"
    else
        echo "<pipeline.yaml:$1>"
    fi
}

# hf_fetch_file <repo> <path-in-repo>: one named file (not a filtered tree) with the HF CLI.
# HF_TOKEN comes from .env inside a subshell; the token is never echoed.
hf_fetch_file() {
    (
        set -a
        # shellcheck disable=SC1091
        . ./.env
        set +a
        hf download "$1" "$2" --repo-type dataset --local-dir .
    )
}

# make_label_subset <action_labels.json> <out.json>: {"clip_keys": [...]} from the label keys,
# the subset schema src/utils/hf_shard_subset.py reads.
make_label_subset() {
    mkdir -p "$(dirname "$2")"
    jq '{clip_keys: keys}' "$1" > "$2"
    echo "  wrote $2 with $(jq '.clip_keys | length' "$2") clip keys"
}

fail_or_note() {   # in --dry-run a failed check is reported and the plan continues
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [dry-run] WOULD FAIL: $1"
    else
        echo "FATAL: $1" >&2
        exit 1
    fi
}

echo "════════════════════════════════════════════════════════════"
echo " setup_new_box.sh  profile=$PROFILE  dry_run=$DRY_RUN  skip_env=$SKIP_ENV"
echo " goal: $WHAT"
echo " download ~${DL_GB} GB, working space ~${WORK_GB:-<frame-cache cap + 40>} GB"
echo " setup_env_uv.sh --gpu adds about 50 GB (1B base checkpoint 16.9 GB measured;"
echo " venv, SAM 3 and Grounding DINO estimated)"
echo "════════════════════════════════════════════════════════════"

# ── 1 preflight ───────────────────────────────────────────────────────────────────────────
step 1 "preflight"
if [ "$(uname -s)" = "Linux" ]; then
    echo "  OS: Linux"
else
    fail_or_note "this box is $(uname -s); setup_env_uv.sh --gpu needs Linux + an NVIDIA GPU"
fi

FROM_WHEELS=""
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1)"
    N_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    CC="$(echo "$GPU_LINE" | awk -F', ' '{print $2}')"
    CC_MAJOR="${CC%%.*}"
    echo "  GPU: $GPU_LINE  (x$N_GPU)"
    if [ "$CC_MAJOR" = "12" ]; then
        FROM_WHEELS="--from-wheels"
        echo "  compute capability $CC (sm_120 class) -> setup_env_uv.sh --gpu --from-wheels"
    else
        echo "  compute capability $CC -> setup_env_uv.sh --gpu (arch-matched wheels)"
    fi
    add_row "GPU" "OK" "${N_GPU}x $(echo "$GPU_LINE" | awk -F', ' '{print $1}'), cc $CC"
else
    fail_or_note "nvidia-smi not found; an NVIDIA GPU is required (no CPU fallback)"
    echo "  [dry-run] on the box: compute capability 12.x adds --from-wheels, anything else does not"
    add_row "GPU" "NOT FOUND" "dry-run only; the real run stops here"
fi

if [ -f .env ]; then
    if N_TOKEN="$(grep -cE '^(export )?HF_TOKEN=.+' .env)"; then
        echo "  .env: $N_TOKEN HF_TOKEN line(s) (value not printed)"
        add_row ".env HF_TOKEN" "OK" "$N_TOKEN line(s), value not printed"
    else
        fail_or_note ".env has no HF_TOKEN=... line; add one (huggingface.co/settings/tokens)"
        add_row ".env HF_TOKEN" "MISSING" "add HF_TOKEN=hf_... to .env"
    fi
else
    fail_or_note ".env not found in the repo root; create it with one line HF_TOKEN=hf_... (never commit it)"
    add_row ".env HF_TOKEN" "MISSING" "create .env"
fi

# The run scripts read data.local_data_dir / master_manifest_name straight from pipeline.yaml, so a
# mismatch would train or score the wrong corpus. Fast grep here (same check the iter19 runbook uses);
# step 2 re-checks with yaml_extract.py.
FLIP_CMD=""
if [ -n "$NEED_DATA_DIR" ]; then
    FLIP_CMD="sed -i -e 's|local_data_dir:.*|local_data_dir: \"$NEED_DATA_DIR\"|' -e 's|master_manifest_name:.*|master_manifest_name: \"$NEED_MANIFEST\"|' configs/pipeline.yaml"
    if grep -qE "^\s+local_data_dir: \"?$NEED_DATA_DIR\"?\s*$" configs/pipeline.yaml \
        && grep -qE "^\s+master_manifest_name: \"?$NEED_MANIFEST\"?\s*$" configs/pipeline.yaml; then
        echo "  pipeline.yaml data dir: $NEED_DATA_DIR / $NEED_MANIFEST (matches profile)"
    else
        echo "  pipeline.yaml data dir does not match profile $PROFILE ($NEED_DATA_DIR / $NEED_MANIFEST). Run:"
        echo "    $FLIP_CMD"
        fail_or_note "configs/pipeline.yaml data dir must be $NEED_DATA_DIR for profile $PROFILE"
    fi
fi

# ── 2 env ─────────────────────────────────────────────────────────────────────────────────
step 2 "environment (setup_env_uv.sh)"
if [ "$SKIP_ENV" = 1 ]; then
    echo "  --skip-env: not running setup_env_uv.sh"
    add_row "venv" "SKIPPED" "--skip-env, reusing venv_walkindia"
else
    run setup_env_gpu bash ./setup_env_uv.sh --gpu $FROM_WHEELS
    add_row "venv" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo DONE)" "setup_env_uv.sh --gpu $FROM_WHEELS"
fi
HAVE_VENV=0
if [ -f venv_walkindia/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv_walkindia/bin/activate
    HAVE_VENV=1
    echo "  venv: $(command -v python)"
else
    fail_or_note "venv_walkindia/bin/activate missing (run without --skip-env)"
    echo "  [dry-run] yaml values below are shown as <pipeline.yaml:key>"
fi
if [ "$PROFILE" = "eval-heldout" ]; then
    command -v jq > /dev/null 2>&1 || fail_or_note "jq not found (setup_env_uv.sh installs it with apt)"
fi

# Everything below reads configs/pipeline.yaml (single source); nothing is retyped here.
HF_REPO="$(yq hf_repos.outputs)"
BASE_CKPT="$(yq "backbone_checkpoints.${BACKBONE}")"
MODEL_CFG="$(yq "backbone_model_configs.${BACKBONE}")"
SHARD_SUBDIR="$(yq data.shard_subdir)"
SHARD_GLOB="$(yq artifacts.video_shard_glob)"
MASKS_SUBDIR="$(yq data.masks_subdir)"
MASKS_INNER="$(yq data.masks_inner_dirname)"
FACTOR_SUBDIR="$(yq data.factor_subdir)"
FACTOR_MANIFEST="$(yq data.factor_manifest_name)"
CACHE_CAP_GB="$(yq probe.eval_frame_cache.max_cache_gb)"
if [ "$HAVE_VENV" = 1 ]; then
    POC_TRAIN_ROOT="$(python src/utils/output_paths.py train-dir poc "$BACKBONE")"
else
    POC_TRAIN_ROOT="<output_paths.py train-dir poc $BACKBONE>"
fi
POC_BB_ROOT="$(dirname "$POC_TRAIN_ROOT")"
FULL_ROOT="$(yq output_roots.full)"

if [ -n "$NEED_DATA_DIR" ] && [ "$HAVE_VENV" = 1 ]; then
    Y_DIR="$(yq data.local_data_dir)"; Y_MAN="$(yq data.master_manifest_name)"
    if [ "$Y_DIR" = "$NEED_DATA_DIR" ] && [ "$Y_MAN" = "$NEED_MANIFEST" ]; then
        add_row "pipeline.yaml data dir" "OK" "$Y_DIR / $Y_MAN"
    else
        add_row "pipeline.yaml data dir" "WRONG" "$Y_DIR / $Y_MAN, needs $NEED_DATA_DIR / $NEED_MANIFEST"
        fail_or_note "pipeline.yaml says $Y_DIR / $Y_MAN; profile $PROFILE needs $NEED_DATA_DIR / $NEED_MANIFEST. Run: $FLIP_CMD"
    fi
fi

# Trees per profile. DATA_TREES entries are "<dir>:<corpus manifest file>".
DATA_TREES=()
OUT_TREES=()
HELDOUT_LABELS="outputs/heldout_video/probe_action/action_labels.json"
HELDOUT_SUBSET="data/heldout_video/heldout_video_labelled.json"
HELDOUT_LOCAL="data/heldout_video_local"
CORPUS_DIR="data/full_local"
CORPUS_MANIFEST="${CORPUS_DIR}/full_local.json"
case "$PROFILE" in
    eval-heldout) OUT_TREES=(outputs/heldout_video) ;;
    train-1b-10k) DATA_TREES=("data/eval_10k_local:eval_10k.json") ;;
    eval-10k)     DATA_TREES=("data/eval_10k_local:eval_10k.json"); OUT_TREES=("$POC_BB_ROOT") ;;
    full)         DATA_TREES=("${CORPUS_DIR}:full_local.json"); OUT_TREES=("$FULL_ROOT")
                  if [ "$HAVE_VENV" = 1 ]; then WORK_GB=$((CACHE_CAP_GB + 40)); else WORK_GB=640; fi ;;
esac

# ── 3 disk ────────────────────────────────────────────────────────────────────────────────
step 3 "disk space"
FREE_GB="$(df -Pk . | awk 'NR==2 {printf "%d", $4 / 1000000}')"
NEED_GB=$((DL_GB + WORK_GB))
echo "  free on $(pwd): ${FREE_GB} GB | needed: ${NEED_GB} GB (download ${DL_GB} + working ${WORK_GB})"
if [ "$FREE_GB" -ge "$NEED_GB" ]; then
    add_row "disk" "OK" "${FREE_GB} GB free >= ${NEED_GB} GB needed"
else
    add_row "disk" "SHORT" "${FREE_GB} GB free < ${NEED_GB} GB needed"
    fail_or_note "not enough disk: ${FREE_GB} GB free, profile $PROFILE needs ${NEED_GB} GB. Rent a bigger disk."
fi

# ── 4 download ────────────────────────────────────────────────────────────────────────────
step 4 "download from HF ($HF_REPO), whole subtrees only"
MARK_DIR="data/.setup_new_box"   # done markers for data trees (their tar shards are deleted after unpack)
for ENTRY in ${DATA_TREES[@]+"${DATA_TREES[@]}"}; do
    TREE="${ENTRY%%:*}"
    MARK="${MARK_DIR}/$(echo "$TREE" | tr '/' '_').done"
    if [ -f "$MARK" ]; then
        echo "  $TREE: already downloaded and unpacked ($MARK), not re-fetching the deleted tar shards"
        add_row "download $TREE" "CACHED" "$MARK"
    else
        run "download_$(basename "$TREE")" python -u src/utils/hf_outputs.py download-data "$TREE"
        add_row "download $TREE" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo DONE)" "hf_outputs.py download-data"
    fi
done
for TREE in ${OUT_TREES[@]+"${OUT_TREES[@]}"}; do
    run "download_$(echo "$TREE" | tr '/' '_')" python -u src/utils/hf_outputs.py download-data "$TREE"
    add_row "download $TREE" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo DONE)" "hf_outputs.py download-data"
done

# ── 5 heldout clips (eval-heldout only) ───────────────────────────────────────────────────
if [ "$PROFILE" = "eval-heldout" ]; then
    step 5 "held-out clips (labelled clips only, streamed from the corpus shards)"
    # The shards hold the whole 115k corpus (130 GB). hf_shard_subset.py streams only the shards that
    # hold wanted clips and keeps those clips, so this needs the corpus manifest (one named file), not
    # the corpus tree. Wanted clips = the keys of the shipped held-out labels, i.e. exactly the clips
    # the eval scripts score.
    run fetch_full_local_manifest hf_fetch_file "$HF_REPO" "$CORPUS_MANIFEST"
    run heldout_label_subset make_label_subset "$HELDOUT_LABELS" "$HELDOUT_SUBSET"
    if [ -f "${HELDOUT_LOCAL}/manifest.json" ]; then
        echo "  ${HELDOUT_LOCAL}/manifest.json exists, not re-cutting"
        add_row "held-out clips" "CACHED" "$HELDOUT_LOCAL"
    else
        # --workers 8: network-bound parallel streams, the value in the script's own USAGE block.
        run hf_shard_subset python -u src/utils/hf_shard_subset.py --repo "$HF_REPO" \
            --shard-prefix "${CORPUS_DIR}/${SHARD_SUBDIR}/" --corpus "$CORPUS_MANIFEST" \
            --subset "$HELDOUT_SUBSET" --output-dir "$HELDOUT_LOCAL" --workers 8
        add_row "held-out clips" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo DONE)" "$HELDOUT_LOCAL"
    fi
fi

# ── 6 verify ──────────────────────────────────────────────────────────────────────────────
step 6 "verify"
need_file() {   # need_file <path> <what>
    if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] check file: $1 ($2)"; return 0; fi
    [ -s "$1" ] || { echo "FATAL: missing $2: $1" >&2; exit 1; }
    echo "  ok  $1"
}
need_glob() {   # need_glob <dir> <glob> <what>
    if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] check files: $1/$2 ($3)"; return 0; fi
    local n
    n="$(find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' ')"
    [ "$n" -gt 0 ] || { echo "FATAL: no $3 in $1 ($2)" >&2; exit 1; }
    echo "  ok  $1/$2 ($n files)"
}
for ENTRY in ${DATA_TREES[@]+"${DATA_TREES[@]}"}; do
    TREE="${ENTRY%%:*}"; MAN="${ENTRY##*:}"
    # hf_outputs.py verify cannot be used on data trees: download-data deletes the mask/factor tar
    # shards after unpacking them, so verify would report them missing. Check what the runs read.
    need_file "$TREE/$MAN" "corpus manifest"
    need_file "$TREE/tags.json" "VLM tags"
    need_file "$TREE/m04d_motion_features/motion_features.npy" "motion features (run_train.sh reads this path)"
    need_glob "$TREE/$SHARD_SUBDIR" "$SHARD_GLOB" "video shards"
    need_glob "$TREE/$MASKS_SUBDIR/$MASKS_INNER" "*" "unpacked SAM masks"
    need_file "$TREE/$FACTOR_SUBDIR/$FACTOR_MANIFEST" "factor manifest"
    if [ "$DRY_RUN" = 0 ]; then
        mkdir -p "$MARK_DIR"
        date +%Y%m%d_%H%M%S > "${MARK_DIR}/$(echo "$TREE" | tr '/' '_').done"
    fi
    add_row "verify $TREE" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo PASS)" "manifest, tags, motion, shards, masks, factors"
done
for TREE in ${OUT_TREES[@]+"${OUT_TREES[@]}"}; do
    run "verify_$(echo "$TREE" | tr '/' '_')" python -u src/utils/hf_outputs.py verify "$TREE"
    add_row "verify $TREE" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo PASS)" "hf_outputs.py verify (disk vs HF)"
done
if [ "$PROFILE" = "eval-heldout" ]; then
    need_file "$HELDOUT_LABELS" "held-out action labels"
    need_file "${HELDOUT_LOCAL}/manifest.json" "held-out clip manifest"
    need_glob "${HELDOUT_LOCAL}/${SHARD_SUBDIR}" "$SHARD_GLOB" "held-out clip shards"
    add_row "verify held-out clips" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo PASS)" "labels, manifest, shards"
fi
need_file "$BASE_CKPT" "1B base checkpoint (setup_env_uv.sh downloads it)"
add_row "verify $BASE_CKPT" "$([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo PASS)" "frozen 1B V-JEPA 2.1"

# ── 7 summary ─────────────────────────────────────────────────────────────────────────────
step 7 "summary"
printf '┌%s┬%s┬%s┐\n' "$(printf '─%.0s' $(seq 1 40))" "$(printf '─%.0s' $(seq 1 11))" "$(printf '─%.0s' $(seq 1 58))"
printf '│ %-38s │ %-9s │ %-56s │\n' "step" "result" "detail"
printf '├%s┼%s┼%s┤\n' "$(printf '─%.0s' $(seq 1 40))" "$(printf '─%.0s' $(seq 1 11))" "$(printf '─%.0s' $(seq 1 58))"
for ROW in "${SUMMARY[@]}"; do
    IFS='|' read -r C1 C2 C3 <<< "$ROW"
    printf '│ %-38.38s │ %-9.9s │ %-56.56s │\n' "$C1" "$C2" "$C3"
done
printf '└%s┴%s┴%s┘\n' "$(printf '─%.0s' $(seq 1 40))" "$(printf '─%.0s' $(seq 1 11))" "$(printf '─%.0s' $(seq 1 58))"

echo ""
echo "Next (see README.md, Setup):"
echo "  source venv_walkindia/bin/activate && set -o pipefail"
case "$PROFILE" in
    eval-heldout)
        echo "  set -a; source .env; set +a   # hf_ckpt_lease.py reads HF_TOKEN from the environment"
        echo "  bash scripts/heldout_video_parallel.sh 20 $HF_REPO $POC_TRAIN_ROOT $HELDOUT_LOCAL \\"
        echo "      outputs/heldout_video/probe_action outputs/heldout_video_rerun ckpt_lease $MODEL_CFG \\"
        echo "      \"${BACKBONE}_pretrain_encoder|m09a_pretrain_encoder|m09a_ckpt_best.pt\" \\"
        echo "      \"${BACKBONE}_surgery_raw_encoder|m09c_surgery_raw_encoder|m09c_ckpt_best.pt\" \\"
        echo "      \"${BACKBONE}_surgical_3stage_DI_diheavy_encoder|m09c_surgery_3stage_DI_diheavy_encoder|m09c_ckpt_best.pt\" \\"
        echo "      \"${BACKBONE}_peft_lora_encoder|m09b_peft_lora_encoder|m09c_ckpt_best.pt\" \\"
        echo "      2>&1 | tee logs/heldout_video_parallel_\$(date +%Y%m%d_%H%M%S).log" ;;
    train-1b-10k)
        echo "  BACKBONE=$BACKBONE ./scripts/run_train.sh pretrain_encoder --POC 2>&1 | tee logs/run_train_pretrain_\$(date +%Y%m%d_%H%M%S).log" ;;
    eval-10k)
        echo "  ITER18_BACKBONE=$BACKBONE EVAL_CORPUS=eval_10k CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --POC \\"
        echo "      --encoders \"${BACKBONE}_frozen ${BACKBONE}_surgical_3stage_DI_diheavy_encoder\" \\"
        echo "      2>&1 | tee logs/run_eval_poc_\$(date +%Y%m%d_%H%M%S).log" ;;
    full)
        echo "  export ITER18_BACKBONE=$BACKBONE EVAL_CORPUS=full EMIT_PERFRAME=1"
        echo "  python -u scripts/ngpu_run.py --mode FULL --gpus <N> --cache 1 2>&1 | tee logs/ngpu_full_\$(date +%Y%m%d_%H%M%S).log" ;;
esac
if [ "$DRY_RUN" = 1 ]; then
    echo ""
    echo "dry-run: nothing was installed, downloaded or written."
fi
