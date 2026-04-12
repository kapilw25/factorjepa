# Code Development Plan — Fix Shell Script Crashers + Safety Infrastructure

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **Blocker: Both train_explora.sh and train_surgery.sh crash on EVERY run at cleanup.**

---

## Crash Bugs (fix before ANY GPU run)

### Bug 1: `finalize` called with no argument [CRITICAL — both scripts]

Under `set -u`, `finalize` in common.sh reads `$1` which is unbound → crash at cleanup AFTER all training completes. No summary printed, watchdog orphaned.

**File:** `scripts/train_explora.sh` line 182, `scripts/train_surgery.sh` line 187

```bash
# WRONG:
finalize

# FIX:
finalize "ExPLoRA"   # or "Surgery"
```

### Bug 2: `${CONFIG}` undefined in train_pretrain.sh [HIGH]

Line 212: `log "Config: ${CONFIG}"` — `CONFIG` was deleted when we moved to `MODEL_CONFIG` + `TRAIN_CONFIG`. Under `set -u` this crashes at pre-flight.

**File:** `scripts/train_pretrain.sh` line 212

```bash
# WRONG:
log "Config: ${CONFIG}"

# FIX:
log "Model config: ${MODEL_CONFIG}"
log "Train config: ${TRAIN_CONFIG}"
```

---

## Safety Infrastructure (missing from train_explora.sh + train_surgery.sh)

All features below exist in `train_pretrain.sh` and must be ported to both scripts.

### Fix 3: Auto batch size detection [CRITICAL]

Without this, m09 uses YAML default (32) which may OOM or waste VRAM.

**Add after venv activation, before training steps:**

```bash
# Auto-detect optimal batch size (same pattern as train_pretrain.sh lines 129-158)
BATCH_FLAG=""
PROFILE_JSON="outputs/profile/training/profile_data.json"
if [[ -f "$PROFILE_JSON" ]]; then
    BS=$(python -u src/utils/gpu_batch.py optimal-bs --profile-json "$PROFILE_JSON")
    BATCH_FLAG="--batch-size $BS"
    log "Batch size: $BS (from profiler)"
elif command -v python &>/dev/null; then
    BS=$(python -u src/utils/config.py get-yaml "$TRAIN_CONFIG" optimization.batch_size 2>/dev/null || echo "32")
    BATCH_FLAG="--batch-size $BS"
    log "Batch size: $BS (from YAML config)"
fi
```

Then pass `$BATCH_FLAG` to every `run_step` that calls m09 or m05.

### Fix 4: Local data FATAL enforcement [CRITICAL]

Silent fallback to HF streaming makes POC take 10x longer.

```bash
# FATAL if local data missing for non-SANITY modes
if [[ "$MODE" != "SANITY" ]]; then
    if [[ -n "$SUBSET_FLAG" ]]; then
        if [[ ! -d "data/val_1k_local" ]] || [[ ! -f "data/val_1k_local/manifest.json" ]]; then
            log "FATAL: data/val_1k_local/ missing or no manifest.json"
            log "  Download: python -u src/utils/hf_outputs.py download-data"
            exit 1
        fi
        LOCAL_FLAG="--local-data data/val_1k_local"
    elif [[ ! -d "data/full_local" ]] || [[ ! -f "data/full_local/manifest.json" ]]; then
        log "FATAL: data/full_local/ missing or no manifest.json"
        log "  Download: python -u src/m00d_download_subset.py --FULL --no-wandb"
        exit 1
    fi
fi
```

### Fix 5: GPU/package pre-flight [CRITICAL]

Catch missing torch/FAISS/flash-attn BEFORE hours of GPU work.

```bash
# Pre-flight: verify GPU packages
log "Pre-flight: checking GPU packages..."
python -u src/utils/output_guard.py preflight_gpu_packages "explora" "$TRAIN_CONFIG" "$OUT_DIR" \
    2>&1 | tee -a "$MASTER_LOG" || { log "FATAL: GPU pre-flight failed"; exit 1; }
```

### Fix 6: Val data enforcement [HIGH]

```bash
# Val data: FATAL if missing for non-SANITY
VAL_FLAG=""
if [[ "$MODE" != "SANITY" ]]; then
    if [[ ! -f "data/val_1k.json" ]]; then
        log "FATAL: data/val_1k.json not found (validation subset)"
        exit 1
    fi
    if [[ ! -d "data/val_1k_local" ]]; then
        log "FATAL: data/val_1k_local/ not found (validation data)"
        exit 1
    fi
    VAL_FLAG="--val-subset data/val_1k.json --val-local-data data/val_1k_local"
fi
```

### Fix 7: Signal trap for cleanup [HIGH — all scripts]

```bash
# Add after set -euo pipefail:
cleanup_on_exit() {
    log "INTERRUPTED — cleaning up..."
    stop_watchdog 2>/dev/null || true
    log "Checkpoint preserved for resume. Re-run same command."
    exit 130
}
trap cleanup_on_exit INT TERM
```

### Fix 8: train_surgery.sh — FATAL guard for unimplemented m09 surgery [CRITICAL]

m09 has NO surgery mode yet. The script should fail immediately with a clear message.

```bash
# At top of train_surgery.sh, after source common.sh:
log "FATAL: m09 surgery mode (progressive prefix unfreezing + factor loading) NOT YET IMPLEMENTED"
log "  Surgery training requires:"
log "    1. m09 --surgery flag with stage iteration logic"
log "    2. m09 --factor-dir to load D_L/D_A/D_I from m11"
log "    3. Per-stage optimizer rebuild with expanding trainable prefix"
log "  See iter/iter8/plan_code_development.md for design"
exit 1
```

Remove this guard ONLY when m09 surgery mode is implemented.

---

## TODO List

| # | Fix | File(s) | Est. |
|---|---|---|---|
| 1 | `finalize "ExPLoRA"` / `finalize "Surgery"` | train_explora.sh, train_surgery.sh | 1 min |
| 2 | Fix `${CONFIG}` → `${TRAIN_CONFIG}` | train_pretrain.sh | 1 min |
| 3 | Auto batch size detection | train_explora.sh, train_surgery.sh | 10 min |
| 4 | Local data FATAL enforcement + manifest check | train_explora.sh, train_surgery.sh | 10 min |
| 5 | GPU/package pre-flight | train_explora.sh, train_surgery.sh | 5 min |
| 6 | Val data FATAL enforcement | train_explora.sh, train_surgery.sh | 5 min |
| 7 | Signal trap (INT/TERM) | train_explora.sh, train_surgery.sh, common.sh | 5 min |
| 8 | train_surgery.sh FATAL guard (m09 surgery unimplemented) | train_surgery.sh | 2 min |
| 9 | Pass `$BATCH_FLAG` to all m09/m05 run_step calls | train_explora.sh, train_surgery.sh | 5 min |
| 10 | `bash -n` syntax check on all 3 scripts | — | 2 min |
| **Total** | | | **~45 min** |

---

## Verification

```bash
# Syntax check (Mac)
bash -n scripts/train_explora.sh && echo "OK: explora"
bash -n scripts/train_surgery.sh && echo "OK: surgery"
bash -n scripts/train_pretrain.sh && echo "OK: pretrain"

# Dry run (GPU — verify pre-flight works)
./scripts/train_explora.sh --SANITY 2>&1 | tail -20
# Should see: pre-flight, batch size detection, venv check, then training
# Should NOT crash at finalize
```
