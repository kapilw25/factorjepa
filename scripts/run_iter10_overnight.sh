#!/usr/bin/env bash
# iter10: sequential v15a → v15b → v15c overnight orchestrator.
# Assumes v14's outputs/full holds cached m10/m11; v14 m09c/m05/m06/m08b outputs
# must be archived BEFORE launch (see pre-flight below).
#
# USAGE:
#   # pre-flight (one-time, before launch):
#   mkdir -p outputs_versioned
#   for d in m09c_surgery m05_vjepa_embed m06_faiss_metrics m08b_compare; do
#       [ -d outputs/full/$d ] && mv outputs/full/$d outputs_versioned/v14_${d}
#   done
#   ls outputs/full/m10_sam_segment/segments.json \
#      outputs/full/m11_factor_datasets/factor_manifest.json   # sanity-check cache
#
#   # launch unmonitored:
#   tmux new -s terminal1
#   ./scripts/run_iter10_overnight.sh 2>&1 | tee logs/iter10_v2.log
#   # Ctrl-B d to detach → tmux attach -t iter10 to reattach
#   # wake-up: cat logs/iter10_status.txt
set -euo pipefail

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs outputs_versioned

STATUS=logs/iter10_status.txt
ERR_LOG=logs/iter10_error.log
: > "$STATUS"
trap 'ec=$?; echo "[$(date +%H:%M:%S)] FATAL at line $LINENO: $BASH_COMMAND (exit=$ec)" | tee -a "$STATUS" | tee -a "$ERR_LOG"; exit $ec' ERR

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

FAILED_VARIANTS=()

archive_run() {  # $1 = version tag (v15a, v15b, v15c, or <tag>_FAILED)
    local tag="$1"
    for d in m09c_surgery m05_vjepa_embed m06_faiss_metrics m08b_compare; do
        if [ -d "outputs/full/$d" ]; then
            # iter11: no shell-level rm — use merge-cp + mv. If destination archive
            # already exists, cp -rf merges into it (overwriting same-named files,
            # preserving orphans); then mv relocates the source.
            local dst="outputs_versioned/${tag}_${d}"
            if [ -d "$dst" ]; then
                mkdir -p "$dst"
                cp -rf "outputs/full/$d/." "$dst/"
                log "  merged outputs/full/$d → $dst (existing archive preserved)"
            else
                mv "outputs/full/$d" "$dst"
                log "  archived outputs/full/$d → $dst"
            fi
        fi
    done
}

# run_variant: invokes run_iter9_10k.sh with isolated errexit.
# If training fails (GPU OOM, checkpoint corruption, data error), we log it,
# archive partial outputs under <tag>_FAILED_*, register the failure in
# FAILED_VARIANTS, and RETURN 0 so the orchestrator continues to the next variant.
# PIPESTATUS[0] captures the training exit code (not tee's), working even under pipefail.
run_variant() {  # $1=tag  $2=yaml_path  $3=label
    local tag="$1"
    local yaml="$2"
    local label="$3"
    log "──────────────────────────────────────────────"
    log "$tag $label START"
    log "──────────────────────────────────────────────"
    set +e
    bash scripts/run_iter9_10k.sh --train-config "$yaml" 2>&1 | tee "logs/iter10_${tag}.log"
    local rc="${PIPESTATUS[0]}"
    set -e
    if [ "$rc" -ne 0 ]; then
        log "❌ $tag FAILED (exit=$rc) — archiving partial outputs as ${tag}_FAILED_*, continuing"
        archive_run "${tag}_FAILED"
        FAILED_VARIANTS+=("$tag")
        return 0
    fi
    archive_run "$tag"
    log "✅ $tag DONE (elapsed: $(( ($(date +%s) - T0) / 60 )) min total)"
    return 0
}

# run_prep: same pattern for the v15c D_I prep steps. If prep fails, caller
# decides whether to attempt v15c training (which depends on D_I tubes).
run_prep() {  # $1=label  rest=command
    local label="$1"
    shift
    log "$label"
    set +e
    "$@"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        log "❌ prep failed (exit=$rc): $label"
        return "$rc"
    fi
    return 0
}

# ── Pre-flight sanity ───────────────────────────────────────────────
log "iter10 overnight START"
log "pre-flight checks:"
if [ ! -f outputs/full/m10_sam_segment/segments.json ]; then
    log "  ❌ outputs/full/m10_sam_segment/segments.json missing — m10 cache gone; ABORTING"
    exit 3
fi
if [ ! -f outputs/full/m11_factor_datasets/factor_manifest.json ]; then
    log "  ❌ outputs/full/m11_factor_datasets/factor_manifest.json missing — m11 cache gone; ABORTING"
    exit 3
fi
if [ -d outputs/full/m09c_surgery ]; then
    log "  ⚠️  outputs/full/m09c_surgery EXISTS — archive it first (see header comment)"
    exit 3
fi
log "  ✅ m10/m11 caches present; outputs/full/m09c_surgery clean"

T0=$(date +%s)

# ── v15a: more-laps (max_epochs 1→3) ───────────────────────────────
run_variant v15a configs/train/ch11_surgery_v15a.yaml "more-laps (max_epochs=3)"

# ── v15b: louder-agent (S2 A:0.7→0.85) ─────────────────────────────
run_variant v15b configs/train/ch11_surgery_v15b.yaml "louder-agent (S2 A=0.85, L=0.15)"

# ── v15c prep: D_I regeneration from cached m10 masks (~20 min CPU) ─
log "──────────────────────────────────────────────"
log "v15c prep: mining interactions + rebuilding D_I tubes"
log "──────────────────────────────────────────────"
v15c_prep_ok=1
if ! run_prep "m10 --interactions-only (cached masks, no SAM3)" \
        bash -c "python -u src/m10_sam_segment.py --FULL \
            --subset data/subset_10k.json --local-data data/subset_10k_local \
            --train-config configs/train/ch11_surgery_v15c.yaml \
            --interactions-only --no-wandb \
            2>&1 | tee logs/iter10_v15c_m10_mining.log; exit \${PIPESTATUS[0]}"; then
    v15c_prep_ok=0
fi
if [ "$v15c_prep_ok" -eq 1 ]; then
    if ! run_prep "m11 --regen-D_I (rebuild tubes only; D_L/D_A untouched)" \
            bash -c "python -u src/m11_factor_datasets.py --FULL \
                --subset data/subset_10k.json --local-data data/subset_10k_local \
                --train-config configs/train/ch11_surgery_v15c.yaml \
                --regen-D_I --no-wandb \
                2>&1 | tee logs/iter10_v15c_m11.log; exit \${PIPESTATUS[0]}"; then
        v15c_prep_ok=0
    fi
fi

# ── v15c: safer-interactions (3 stages, unfreeze_below 0.50) ───────
if [ "$v15c_prep_ok" -eq 1 ]; then
    run_variant v15c configs/train/ch11_surgery_v15c.yaml "safer-interactions (3 stages, S3 unfreeze=0.50, D_I re-enabled)"
else
    log "❌ v15c SKIPPED — D_I prep failed (see logs/iter10_v15c_m10_mining.log + logs/iter10_v15c_m11.log)"
    FAILED_VARIANTS+=("v15c (prep)")
fi

# ── Final: HF push for backup (only if at least one variant succeeded) ──
log "──────────────────────────────────────────────"
if [ "${#FAILED_VARIANTS[@]}" -eq 3 ]; then
    log "⚠️ All 3 variants FAILED — skipping git_push to avoid polluting HF with broken artifacts"
    log "   Review: logs/iter10_error.log, outputs_versioned/v15*_FAILED_*"
else
    log "Final: ./git_push.sh 'iter10 v15a+b+c complete'"
    log "──────────────────────────────────────────────"
    ./git_push.sh "iter10 v15a+b+c complete" 2>&1 | tee logs/iter10_push.log || \
        log "  ⚠️ git_push.sh failed — run manually on wake-up"
fi

# ── Final summary ──────────────────────────────────────────────────
DUR=$(( $(date +%s) - T0 ))
log "──────────────────────────────────────────────"
if [ "${#FAILED_VARIANTS[@]}" -eq 0 ]; then
    log "🎉 ALL 3 VARIANTS DONE — total wall: $(( DUR / 3600 ))h $(( (DUR % 3600) / 60 ))m"
else
    log "⚠️ FINISHED with ${#FAILED_VARIANTS[@]} failed variant(s): ${FAILED_VARIANTS[*]}"
    log "   Total wall: $(( DUR / 3600 ))h $(( (DUR % 3600) / 60 ))m"
    log "   Failed outputs archived under outputs_versioned/*_FAILED_*"
fi
log "Summary artifacts:"
log "  outputs_versioned/v15{a,b,c}_{m09c_surgery,m05_vjepa_embed,m06_faiss_metrics,m08b_compare}/"
log "  outputs_versioned/v15*_FAILED_* (if any variant crashed)"
log "  logs/iter10_{status,v15a,v15b,v15c,v15c_m10_mining,v15c_m11,push,error}.log"
