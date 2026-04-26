#!/usr/bin/env bash
# iter10 Option C: evaluate ALL iter9 + iter10 surgical variants on eval_10k
# with paired-bootstrap BCa (CI_half ~0.5 pp vs test_500's ±2.4 pp).
#
# PRODUCES PER VARIANT:
#   outputs_versioned/<tag>_eval10k/
#     ├── per_clip_vjepa_2_1_{frozen,surgical}_{easy,hard}.npz
#     ├── m06_metrics_vjepa_2_1_{frozen,surgical}.json
#     └── paired_bootstrap_results.json   (Δ mean + BCa CI + p-value vs 0)
#
# USAGE:
#   tmux new -s paired_eval
#   ./scripts/run_paired_eval_10k.sh 2>&1 | tee logs/run_paired_eval_10k_v2.log
#   # Ctrl-B d to detach · tmux attach -t paired_eval
#
# PREREQS (script checks, fails loud if missing):
#   - data/eval_10k.json + data/eval_10k_local/ + data/eval_10k_local/tags.json  (run scripts/prep_eval_10k.sh first)
#   - outputs_versioned/v{10,13,14,15a,15b,15c}_m09c_surgery/student_encoder.pt
#     (v10, v13 are auto-staged from iter/iter9/v9{a,b}_*/ into outputs_versioned/ if missing)
#
# COST: ~2.3 h GPU × 7 encoders (Frozen + 6 surgical) = ~16 h · ~$12.80 on RTX Pro 6000
# FAIL-HARD: set -euo pipefail everywhere; no || true; no bare except.
set -euo pipefail

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs outputs_versioned

STATUS=logs/paired_eval_status.txt
: > "$STATUS"
trap 'ec=$?; echo "[$(date +%H:%M:%S)] FATAL at line $LINENO: $BASH_COMMAND (exit=$ec)" | tee -a "$STATUS"; exit $ec' ERR

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# ── Pre-flight: eval_10k ready? ────────────────────────────────────
log "paired_eval_10k START"
for req in "data/eval_10k.json" "data/eval_10k_local" "data/eval_10k_local/tags.json"; do
    if [ ! -e "$req" ]; then
        log "❌ missing $req — run ./scripts/prep_eval_10k.sh first"
        exit 3
    fi
done
log "  ✅ eval_10k artifacts present"

# ── Pre-flight: stage v10/v13 from iter9/ archive if not in outputs_versioned/ ──
stage_from_iter9() {  # $1=tag  $2=source_subdir
    local tag="$1" src="$2"
    local dst="outputs_versioned/${tag}_m09c_surgery"
    if [ ! -f "$dst/student_encoder.pt" ]; then
        local src_ckpt="iter/iter9/${src}/full/m09c_surgery/student_encoder.pt"
        if [ ! -f "$src_ckpt" ]; then
            log "❌ $tag: neither $dst nor $src_ckpt exists — cannot proceed"
            exit 3
        fi
        mkdir -p "$dst"
        cp "$src_ckpt" "$dst/student_encoder.pt"
        log "  staged $tag student_encoder.pt from iter/iter9/${src}/"
    fi
}
stage_from_iter9 v10 v9a_10k_LR_1e6
stage_from_iter9 v13 v9b_10k_LR_1e5_Dino_Up

# ── Pre-flight: which variants have checkpoints? ────────────────────
# iter11 (step3): `explora` inserted BETWEEN v14 and v15a so the diagnostic baseline
# runs BEFORE the remaining Surgery variants — v10/v13/v14 have all shown Δ≈0 vs
# Frozen at N=9297, so getting ExPLoRA's Δ Prec@K early tells us whether the bottleneck
# is the Surgery RECIPE (ExPLoRA wins → recipe bug) or the TASK itself (both cluster
# near 0 → iter11 must pivot to hard-slice eval). No stage_from_iter9 for explora
# needed: m09b_explora.py writes directly to outputs_versioned/explora_m09c_surgery/
# (see src/m09b_explora.py docstring's POST-TRAINING EVAL block).
VARIANTS=()
for v in v10 v13 v14 explora v15a v15b v15c; do
    if [ -f "outputs_versioned/${v}_m09c_surgery/student_encoder.pt" ]; then
        VARIANTS+=("$v")
    else
        log "  ⚠️  $v: checkpoint missing — skipping (retry after iter10 completes for v15c; run m09b_explora.py --FULL then stage ckpt per docstring for 'explora')"
    fi
done
log "  ✅ variants to evaluate: ${VARIANTS[*]} (N=${#VARIANTS[@]})"

# ── iter11 CACHE-POLICY PROMPT GATHER — one prompt per EXISTING cached artifact ──
# Detects caches at TWO levels per combo:
#   1. Final archived artifact (e.g. per_clip_*_surgical_easy.npz in $ARCHIVE) — completed
#   2. Mid-flight .m05_checkpoint_*.npz — in-progress resume point (v13-at-8448/10000 case)
# Prompts only fire when ANY such cache exists. Missing caches silently default to
# policy=1 (keep). Overrides: CACHE_POLICY_ALL=1|2 ./script.sh skips prompts; non-TTY
# stdin silently defaults to 1.
declare -A POLICY

# Mirror m05's src/m05_vjepa_embed.py::_checkpoint_fingerprint so we can pre-compute
# each variant's surgical ckpt filename and detect mid-flight progress.
_compute_m05_fp() {                          # $1=model path (student_encoder.pt)
    python3 -c "
import hashlib, os, sys
from pathlib import Path
try:
    p = Path('$1').resolve()
    st = p.stat()
    print('_' + hashlib.sha256(f'{p}|{st.st_size}|{int(st.st_mtime)}'.encode()).hexdigest()[:8])
except Exception:
    print('_nockpt')
" 2>/dev/null
}

_check_and_prompt_any() {                    # $1=combo-key  $2..=candidate paths (any-existence)
    local key="$1"; shift
    local found=""
    for path in "$@"; do
        if [ -e "$path" ]; then found="$path"; break; fi
    done
    if [ -z "$found" ]; then POLICY[$key]=1; return; fi
    if [ -n "${CACHE_POLICY_ALL:-}" ]; then
        POLICY[$key]=$CACHE_POLICY_ALL
        log "  $key: cache at $found -> policy=${POLICY[$key]} (CACHE_POLICY_ALL)"
        return
    fi
    if [ ! -t 0 ]; then
        POLICY[$key]=1
        log "  $key: cache at $found -> policy=1 (non-TTY default)"
        return
    fi
    local ans
    read -p "  $key cache at $found [1=keep / 2=recompute] (Enter=1): " ans
    POLICY[$key]=${ans:-1}
    [ "${POLICY[$key]}" != "2" ] && POLICY[$key]=1
}

log "──────────────────────────────────────────────"
log "iter11 cache-policy gather (prompts for existing archived OR mid-flight caches)"
log "──────────────────────────────────────────────"

# Frozen m05: completed .npy in archive OR frozen ckpt mid-flight in outputs/full/
_check_and_prompt_any m05_frozen \
    "outputs_versioned/frozen_eval10k/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy" \
    "outputs/full/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_frozen.npz"

# Frozen m06: metrics JSON in archive
_check_and_prompt_any m06_frozen \
    "outputs_versioned/frozen_eval10k/m06_metrics_vjepa_2_1_frozen.json"

# Surgical combos — compute each variant's fingerprinted ckpt path so we catch
# in-progress runs (e.g. v13 at 8448/10000 has a .m05_checkpoint_..._<hash>.npz)
for v in "${VARIANTS[@]}"; do
    _fp=$(_compute_m05_fp "outputs_versioned/${v}_m09c_surgery/student_encoder.pt")
    _check_and_prompt_any "m05_${v}" \
        "outputs_versioned/${v}_eval10k/per_clip_vjepa_2_1_surgical_easy.npz" \
        "outputs/full/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_surgical${_fp}.npz"
    _check_and_prompt_any "m06_${v}" \
        "outputs_versioned/${v}_eval10k/m06_metrics_vjepa_2_1_surgical.json"
    _check_and_prompt_any "m08b_${v}" \
        "outputs_versioned/${v}_eval10k/paired_bootstrap_results.json"
done

# Dependency propagation: upstream recompute invalidates downstream
[ "${POLICY[m05_frozen]:-1}" = "2" ] && POLICY[m06_frozen]=2
if [ "${POLICY[m05_frozen]:-1}" = "2" ] || [ "${POLICY[m06_frozen]:-1}" = "2" ]; then
    for v in "${VARIANTS[@]}"; do POLICY[m08b_$v]=2; done
fi
for v in "${VARIANTS[@]}"; do
    [ "${POLICY[m05_$v]:-1}" = "2" ] && POLICY[m06_$v]=2
    [ "${POLICY[m06_$v]:-1}" = "2" ] && POLICY[m08b_$v]=2
done


# ── Frozen baseline: embed + metrics ONCE (shared across all paired comparisons) ─
# Stage A: m05 frozen embed on eval_10k
log "──────────────────────────────────────────────"
log "Frozen baseline: m05 embed on eval_10k (~2.3 h)"
log "──────────────────────────────────────────────"
# iter11: no shell-level rm. All deletion authority moved into .py files behind
# --cache-policy (default=1/keep). m05's output_guard either skips (if .npy+.paths
# already present and valid) or recomputes; the v1→v10 cycle came from EXACTLY this
# block wiping .npy before m05 ran → removed. Use --cache-policy 2 at the prompt
# to force a fresh embed.
mkdir -p outputs/full/m05_vjepa_embed

# iter11 frozen-archive restore hook — break the v1→v10 2.3 h re-embed cycle.
# Prior to this, each run's cleanup wiped outputs/full/m05_vjepa_embed/ and m05
# auto-deleted its own ckpt after saving .npy, leaving NO durable state in the
# working dir between invocations. The .npy was preserved in the outputs_versioned/
# frozen_eval10k/m05_vjepa_embed/ archive, but m05's output_guard never looked there.
# This hook cp -f's the archived .npy + .paths.npy back so output_guard short-circuits.
_FA_FROZEN="outputs_versioned/frozen_eval10k/m05_vjepa_embed"
if [ ! -f outputs/full/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy ] \
   && [ -f "$_FA_FROZEN/embeddings_vjepa_2_1_frozen.npy" ]; then
    cp -f "$_FA_FROZEN/embeddings_vjepa_2_1_frozen.npy" outputs/full/m05_vjepa_embed/
    if [ -f "$_FA_FROZEN/embeddings_vjepa_2_1_frozen.paths.npy" ]; then
        cp -f "$_FA_FROZEN/embeddings_vjepa_2_1_frozen.paths.npy" outputs/full/m05_vjepa_embed/
    fi
    log "  🔁 Restored frozen .npy from archive → m05 output_guard will skip re-embed (2.3 h saved)"
fi

if [ -f outputs/full/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_frozen.npz ]; then
    n_done=$(python3 -c "import numpy as np; d=np.load('outputs/full/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_frozen.npz', allow_pickle=True); print(len(d['embeddings']))" 2>/dev/null) || n_done="?"
    log "  🔁 Frozen checkpoint found → m05 will resume from ${n_done:-?}/10,000 clips"
fi
python -u src/m05_vjepa_embed.py --FULL \
    --subset data/eval_10k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/eval_10k_local --no-wandb \
    --cache-policy "${POLICY[m05_frozen]:-1}" \
    2>&1 | tee logs/paired_eval_frozen_m05.log

# Schema-drift gate: abort if checkpoint load reported missing params > 0.
# Pattern "missing: [1-9]" matches only NON-zero — safe when m05 omits the line.
if miss_line=$(grep -oE "missing: [1-9][0-9]*" logs/paired_eval_frozen_m05.log | head -1); then
    log "❌ Frozen m05 schema drift: $miss_line — V-JEPA weights don't match architecture; aborting"
    exit 5
fi
log "  ✅ Frozen m05 schema intact (no missing>0 params)"

# Stage B: m06 frozen metrics (saves per_clip_vjepa_2_1_frozen_{easy,hard}.npz)
log "Frozen baseline: m06 faiss metrics + per-clip save"
# iter11: no shell-level rm — m06's output_guard skips if outputs exist, or
# overwrites atomically via np.savez_compressed + json.dump. Pass --cache-policy 2
# at the prompt to force fresh compute.

# iter11 frozen-archive restore hook (m06 side) — same pattern as m05 restore above.
# If the m06 archive has frozen metrics, cp them back into outputs/full/ so m06's
# output_guard short-circuits. Saves ~7 min per re-run.
_FA_M06="outputs_versioned/frozen_eval10k"
mkdir -p outputs/full/m06_faiss_metrics
_restored_m06=0
for _f in m06_metrics_vjepa_2_1_frozen.json \
          per_clip_vjepa_2_1_frozen_easy.npz \
          per_clip_vjepa_2_1_frozen_hard.npz \
          knn_indices_vjepa_2_1_frozen.npy; do
    if [ ! -f "outputs/full/m06_faiss_metrics/$_f" ] && [ -f "$_FA_M06/$_f" ]; then
        cp -f "$_FA_M06/$_f" "outputs/full/m06_faiss_metrics/"
        _restored_m06=$((_restored_m06 + 1))
    fi
done
if [ "$_restored_m06" -gt 0 ]; then
    log "  🔁 Restored $_restored_m06 m06 frozen file(s) from archive → output_guard will skip re-compute (~7 min saved)"
fi

python -u src/m06_faiss_metrics.py --FULL \
    --subset data/eval_10k.json --encoder vjepa_2_1_frozen \
    --local-data data/eval_10k_local --no-wandb \
    --cache-policy "${POLICY[m06_frozen]:-1}" \
    2>&1 | tee logs/paired_eval_frozen_m06.log

# Verify Frozen per-clip npz materialized (Option C gate)
for m in easy hard; do
    f="outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_${m}.npz"
    if [ ! -f "$f" ]; then
        log "❌ Frozen per_clip ${m}.npz missing — paired bootstrap impossible; aborting"
        exit 4
    fi
done
log "  ✅ Frozen per_clip .npz confirmed (both modes)"

# ── Archive Frozen NOW (before variant loop wipes outputs/full/m05_vjepa_embed) ─
# Per-variant `rm -rf outputs/full/m05_vjepa_embed` at line ~113 would otherwise
# delete Frozen's embeddings .npy; by the end of the loop the dir holds the last
# variant's SURGICAL embeddings, not Frozen. We `cp` the per_clip npz files into
# the archive while leaving the originals in outputs/full/m06_faiss_metrics/
# (m08b paired bootstrap reads them once per variant).
FROZEN_ARCHIVE="outputs_versioned/frozen_eval10k"
# iter11: no shell-level rm — use merge-cp (cp -f overwrites same-named files).
mkdir -p "$FROZEN_ARCHIVE" "$FROZEN_ARCHIVE/m05_vjepa_embed"
# Merge m05 dir contents (cp /. idiom copies contents, not the dir itself, and
# overwrites existing files without touching orphan files in the archive).
cp -rf outputs/full/m05_vjepa_embed/. "$FROZEN_ARCHIVE/m05_vjepa_embed/"
cp -f outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_frozen.json "$FROZEN_ARCHIVE/"
cp -f outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_easy.npz "$FROZEN_ARCHIVE/"
cp -f outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_hard.npz "$FROZEN_ARCHIVE/"
if [ -f outputs/full/m06_faiss_metrics/knn_indices_vjepa_2_1_frozen.npy ]; then
    cp -f outputs/full/m06_faiss_metrics/knn_indices_vjepa_2_1_frozen.npy "$FROZEN_ARCHIVE/"
fi
log "  ✅ Frozen archived early to $FROZEN_ARCHIVE (preserves correct embeddings)"

# ── Auto-stage leftover surgical outputs → $ARCHIVE (resume after mid-m08b crash) ──
# If a prior run crashed AFTER surgical m05+m06 but BEFORE archive mv (e.g., v9→v10
# FATAL clip_keys mismatch at m08b), surgical files sit in outputs/full/m06_faiss_metrics/
# with the in-flight variant recorded in .in_flight_surgical_variant.txt. Stage them into
# $ARCHIVE now so G2 fires below and we skip ~2.5 h of m05+m06 re-computation.
# Safe by construction: trace is WRITTEN only after m06 succeeds and CLEARED only after
# archive mv succeeds, so its presence uniquely identifies the variant whose surgical
# outputs are currently sitting in outputs/full/.
trace_file="outputs/full/m06_faiss_metrics/.in_flight_surgical_variant.txt"
if [ -f "$trace_file" ]; then
    in_flight_v=$(cat "$trace_file" 2>/dev/null | tr -d '[:space:]')
    stage_src="outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_surgical_easy.npz"
    if [ -n "$in_flight_v" ] && [ -f "$stage_src" ]; then
        AUTO_ARCHIVE="outputs_versioned/${in_flight_v}_eval10k"
        if [ ! -f "$AUTO_ARCHIVE/paired_bootstrap_results.json" ] \
           && [ ! -f "$AUTO_ARCHIVE/per_clip_vjepa_2_1_surgical_easy.npz" ]; then
            mkdir -p "$AUTO_ARCHIVE"
            for f in per_clip_vjepa_2_1_surgical_easy.npz \
                     per_clip_vjepa_2_1_surgical_hard.npz \
                     m06_metrics_vjepa_2_1_surgical.json \
                     knn_indices_vjepa_2_1_surgical.npy; do
                if [ -f "outputs/full/m06_faiss_metrics/$f" ]; then
                    cp "outputs/full/m06_faiss_metrics/$f" "$AUTO_ARCHIVE/"
                fi
            done
            log "  🔁 auto-stage: leftover surgical outputs (in-flight=$in_flight_v) → $AUTO_ARCHIVE (G2 will fire for $in_flight_v)"
        fi
    fi
fi

# ── Per-variant: stage checkpoint → m05 surgical → m06 surgical → m08b paired → archive ──
# Three orthogonal idempotency markers checked in order (iter10 #78 fix):
#   (G1) $ARCHIVE/paired_bootstrap_results.json     → variant fully done   → skip
#   (G2) $ARCHIVE/{per_clip_*_surgical_*,m06_metrics_*_surgical.json}  → m05+m06 done, m08b missing → skip m05+m06
#   (G3) .m05_checkpoint_vjepa_2_1_surgical_<hash>.npz (fingerprint from fix #75) → m05 mid-flight → resume
# These compose: G1 subsumes G2 subsumes G3. Only G3 requires per-variant wipe that preserves ckpts.
for v in "${VARIANTS[@]}"; do
    ARCHIVE="outputs_versioned/${v}_eval10k"

    # ── G1: variant fully done? paired_bootstrap_results.json only lands via archive mv AFTER all 4 stages succeed.
    # iter11: also require POLICY[m08b_$v]=1 (keep); policy=2 bypasses G1 to re-run.
    if [ -f "$ARCHIVE/paired_bootstrap_results.json" ] && [ "${POLICY[m08b_$v]:-1}" = "1" ]; then
        log "  ⏩ $v: paired_bootstrap_results.json already archived + policy=keep — skipping (G1)"
        continue
    fi

    log "──────────────────────────────────────────────"
    log "$v: paired eval on eval_10k (~2.3 h GPU + ~7 min metrics + paired)"
    log "──────────────────────────────────────────────"

    # ── G2: m05+m06 already archived, only m08b needs re-run?
    # iter11: also require POLICY[m05_$v]=POLICY[m06_$v]=1 (keep); either policy=2 bypasses G2.
    need_m05_m06=true
    if [ -f "$ARCHIVE/per_clip_vjepa_2_1_surgical_easy.npz" ] \
       && [ -f "$ARCHIVE/per_clip_vjepa_2_1_surgical_hard.npz" ] \
       && [ -f "$ARCHIVE/m06_metrics_vjepa_2_1_surgical.json" ] \
       && [ "${POLICY[m05_$v]:-1}" = "1" ] \
       && [ "${POLICY[m06_$v]:-1}" = "1" ]; then
        log "  🔁 $v (G2): surgical m05+m06 archived + policy=keep; restoring per_clip to outputs/full/ and skipping to m08b"
        mkdir -p outputs/full/m06_faiss_metrics
        cp "$ARCHIVE/per_clip_vjepa_2_1_surgical_easy.npz" outputs/full/m06_faiss_metrics/
        cp "$ARCHIVE/per_clip_vjepa_2_1_surgical_hard.npz" outputs/full/m06_faiss_metrics/
        cp "$ARCHIVE/m06_metrics_vjepa_2_1_surgical.json" outputs/full/m06_faiss_metrics/
        need_m05_m06=false
    fi

    if $need_m05_m06; then
        # ── G3: iter11 — no shell-level wipe. m05 itself (cache-policy gated) owns
        # any destructive reset. Fingerprint filename (fix #75) makes cross-variant
        # collision impossible, so stale .npy from a previous variant is ignored by
        # m05's output_guard + fingerprinted ckpt path.
        mkdir -p outputs/full/m05_vjepa_embed
        surg_n=$(find outputs/full/m05_vjepa_embed -maxdepth 1 -name '.m05_checkpoint_vjepa_2_1_surgical_*.npz' 2>/dev/null | wc -l)
        if [ "$surg_n" -gt 0 ]; then
            log "  🔁 $v (G3): $surg_n fingerprinted surgical checkpoint(s) on disk — m05 resumes if hash matches this variant"
        fi

        python -u src/m05_vjepa_embed.py --FULL \
            --subset data/eval_10k.json \
            --model-config configs/model/vjepa2_1.yaml \
            --model "outputs_versioned/${v}_m09c_surgery/student_encoder.pt" \
            --encoder vjepa_2_1_surgical \
            --local-data data/eval_10k_local --no-wandb \
            --cache-policy "${POLICY[m05_$v]:-1}" \
            2>&1 | tee "logs/paired_eval_${v}_m05.log"

        # Schema-drift gate: skip this variant if surgical ckpt has missing params.
        if miss_line=$(grep -oE "missing: [1-9][0-9]*" "logs/paired_eval_${v}_m05.log" | head -1); then
            log "❌ $v surgical m05 schema drift: $miss_line — iter9/iter10 ckpt mismatch; skipping variant"
            continue
        fi
        log "  ✅ $v surgical m05 schema intact (no missing>0 params)"

        # Stage B: m06 surgical metrics — no shell rm; m06 overwrites atomically
        # and its output_guard skips if surgical outputs already valid.
        python -u src/m06_faiss_metrics.py --FULL \
            --subset data/eval_10k.json --encoder vjepa_2_1_surgical \
            --local-data data/eval_10k_local --no-wandb \
            --cache-policy "${POLICY[m06_$v]:-1}" \
            2>&1 | tee "logs/paired_eval_${v}_m06.log"

        # Verify surgical per-clip npz landed
        for m in easy hard; do
            f="outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_surgical_${m}.npz"
            if [ ! -f "$f" ]; then
                log "❌ $v: surgical per_clip ${m}.npz missing — skipping this variant's paired bootstrap"
                continue 2
            fi
        done

        # Record which variant produced the surgical outputs now sitting in outputs/full/.
        # Auto-stage hook at script start uses this to rescue work after a mid-m08b crash
        # (paired with the `rm -f $trace_file` after successful archive mv below).
        echo "$v" > outputs/full/m06_faiss_metrics/.in_flight_surgical_variant.txt
    fi

    # Frozen per_clip must be in outputs/full/ for m08b — restore from archive if missing
    # (happens when a fresh script launch hits G2 before frozen stages re-populate outputs/full/).
    for m in easy hard; do
        f="outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_${m}.npz"
        if [ ! -f "$f" ] && [ -f "$FROZEN_ARCHIVE/per_clip_vjepa_2_1_frozen_${m}.npz" ]; then
            cp "$FROZEN_ARCHIVE/per_clip_vjepa_2_1_frozen_${m}.npz" "$f"
            log "  🔁 restored frozen per_clip_${m}.npz from $FROZEN_ARCHIVE"
        fi
    done

    # Stage C: m08b paired bootstrap (reads both per_clip.npz files, runs paired_bca).
    # No shell rm; m08b overwrites plots atomically, and its cache-policy gate
    # protects the stale-radar purge (L1224-1228).
    python -u src/m08b_compare.py --FULL \
        --subset data/eval_10k.json \
        --encoders vjepa_2_1_frozen,vjepa_2_1_surgical \
        --no-wandb \
        --cache-policy "${POLICY[m08b_$v]:-1}" \
        2>&1 | tee "logs/paired_eval_${v}_m08b.log"

    # Stage D: archive variant's results under outputs_versioned/<v>_eval10k/
    # Idempotent: mkdir -p + mv (doesn't fail if src is the one we cp'd in via G2 — mv
    # overwrites destination when src != dst, and src == dst for G2 path since we
    # cp'd INTO outputs/full/m06_faiss_metrics/ from $ARCHIVE, so mv back is a no-op
    # that we guard with existence checks).
    mkdir -p "$ARCHIVE"
    for f in m06_metrics_vjepa_2_1_surgical.json per_clip_vjepa_2_1_surgical_easy.npz \
             per_clip_vjepa_2_1_surgical_hard.npz knn_indices_vjepa_2_1_surgical.npy; do
        if [ -f "outputs/full/m06_faiss_metrics/$f" ]; then
            mv "outputs/full/m06_faiss_metrics/$f" "$ARCHIVE/"
        fi
    done
    if [ -f outputs/full/m08b_compare/paired_bootstrap_results.json ]; then
        mv outputs/full/m08b_compare/paired_bootstrap_results.json "$ARCHIVE/"
    fi
    if [ -d outputs/full/m08b_compare ]; then
        # iter11: merge-cp (no rm). cp -rf copies contents, overwriting same-named
        # files without removing orphans in the archive.
        mkdir -p "$ARCHIVE/m08b_compare"
        cp -rf outputs/full/m08b_compare/. "$ARCHIVE/m08b_compare/"
    fi

    # Clear in-flight trace — truncate (no rm) so auto-stage hook sees empty file.
    : > outputs/full/m06_faiss_metrics/.in_flight_surgical_variant.txt

    log "  ✅ $v archived to $ARCHIVE"
done

# ── Verify Frozen archive still intact (was populated pre-variant-loop) ────
# Frozen was archived BEFORE variant loop so it didn't get wiped by per-variant
# `rm -rf outputs/full/m05_vjepa_embed` (which holds the LAST variant's SURGICAL
# embeddings at this point — not Frozen). Just verify the pre-looped archive.
for required in \
    "$FROZEN_ARCHIVE/m06_metrics_vjepa_2_1_frozen.json" \
    "$FROZEN_ARCHIVE/per_clip_vjepa_2_1_frozen_easy.npz" \
    "$FROZEN_ARCHIVE/per_clip_vjepa_2_1_frozen_hard.npz"; do
    if [ ! -f "$required" ]; then
        log "⚠️ Frozen archive missing $required — was rm'd or never created"
    fi
done
log "  ✅ Frozen archive at $FROZEN_ARCHIVE (preserved pre-variant-loop)"

# ── Final summary table: extract paired Δ from each variant's archive ──
log "──────────────────────────────────────────────"
log "🏆 PAIRED EVAL SUMMARY (eval_10k, BCa 95% CI)"
log "──────────────────────────────────────────────"
python -c "
import json
from pathlib import Path
rows = []
for v in ['v10', 'v13', 'v14', 'explora', 'v15a', 'v15b', 'v15c']:
    p = Path(f'outputs_versioned/{v}_eval10k/paired_bootstrap_results.json')
    if not p.exists():
        rows.append((v, '—', '—', '—', '—', '—'))
        continue
    d = json.load(open(p))
    if 'easy' not in d.get('modes', {}):
        rows.append((v, 'no easy results', '', '', '', ''))
        continue
    pk = d['modes']['easy']['metrics']['prec_at_k']
    ci_sig = '✅' if (pk['delta_ci_lo'] > 0 or pk['delta_ci_hi'] < 0) else '🟡'
    rows.append((v, f'{pk[\"frozen_mean\"]:.4f}', f'{pk[\"surgical_mean\"]:.4f}',
                 f'{pk[\"delta_mean\"]:+.4f}', f'±{pk[\"delta_ci_half\"]:.4f}',
                 f'p={pk[\"p_value_vs_zero\"]:.4f} {ci_sig}'))
hdr = ('Variant', 'Frozen', 'Surgical', 'Δ', 'CI_half', 'p-value')
print(' | '.join(f'{h:>12s}' for h in hdr))
print('-' * (13 * 6 + 15))
for r in rows:
    print(' | '.join(f'{c:>12s}' for c in r))
print()
print('Goal: Δ ≥ +0.03 (+3 pp Prec@K) with CI_lo > 0 for publishable claim.')
"

log "🎉 paired_eval_10k ALL DONE"
log "Artifacts:"
log "  outputs_versioned/frozen_eval10k/       (Frozen baseline — shared)"
log "  outputs_versioned/v{10,13,14,15a,15b,15c}_eval10k/  (per-variant paired results)"
log "  logs/paired_eval_{frozen,v10,v13,v14,explora,v15a,v15b,v15c}_{m05,m06,m08b}.log"
