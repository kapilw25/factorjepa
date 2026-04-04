#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ch11: Surgical Fine-tuning Pipeline (placeholder)
#
# USAGE:
#   ./scripts/run_surgery.sh --SANITY 2>&1 | tee logs/ch11_sanity.log   # TBD
#   ./scripts/run_surgery.sh --FULL 2>&1 | tee logs/ch11_full.log       # TBD
#
# CACHE CONTROL:
#   Use all cached:    ./scripts/run_surgery.sh --FULL
#   Re-run everything: rm -rf outputs/full/m09_surgical* && ./scripts/run_surgery.sh --FULL
#
# PREREQUISITES: Ch9 + Ch10 complete
#
# Common infrastructure shared across all *.sh scripts in scripts/:
#
# ── run_evaluate.sh ──────────────────────────────────────────────────
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (prevents VRAM fragmentation death spiral)
#   - GPU watchdog: scripts/gpu_watchdog.py (emails kapilw25@gmail.com if GPU <50% for 5min)
#   - Background HF upload: bg_upload() after each run_step (zero GPU idle between steps)
#   - Preflight: output_guard.py checks all inputs/outputs before GPU work
#   - Temp cleanup: cleanup_temp() in each src/m*.py main() clears /tmp/hf_* and /tmp/m0*
#   - set -euo pipefail (fail hard on any error)
#   - venv_walkindia activation check
#   - tee logging to timestamped master log
#   - Step timing with banner/run_step helpers
#   - Final verification of all outputs
#
# ── run_pretrain.sh ──────────────────────────────────────────────────
#   - Same infra as run_evaluate.sh (watchdog, bg_upload, preflight, etc.)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   - Auto batch size from profiler (scripts/profile_vram.py → 75% VRAM threshold)
#   - Val subset + local data validation
#   - Mode-aware epoch count from configs/pretrain/vitg16_indian.yaml
#
# ── Common patterns ──────────────────────────────────────────────────
#
#   run_step():
#     - Prints banner with step name, mode, time estimate
#     - Runs python -u with tee to step log + master log
#     - Captures exit code via PIPESTATUS[0]
#     - On success: bg_upload() backgrounds HF upload, GPU starts next step
#     - On failure: logs FAILED with elapsed time
#
#   bg_upload():
#     - Waits for any previous upload to finish (wait $UPLOAD_PID)
#     - Backgrounds: python -u src/utils/hf_outputs.py upload $OUT_DIR
#     - CPU + network only, zero GPU idle time
#     - Final wait before pipeline summary
#
#   GPU watchdog:
#     - scripts/gpu_watchdog.py runs as background process
#     - Checks nvidia-smi every 30s
#     - Emails alert if GPU <50% for 5+ consecutive minutes
#     - 30 min cooldown between repeat alerts
#     - Killed at pipeline end
#
#   Temp cleanup:
#     - cleanup_temp() from src/utils/gpu_batch.py
#     - Called at start of every GPU script main() AND worker_main()
#     - Clears /tmp/hf_* and /tmp/m0* (stale temp from prior steps)
#
#   Output guard:
#     - verify_or_skip() checks if outputs exist before loading model
#     - On MISS: attempts HF download from anonymousML123/factorjepa-outputs
#     - On success: skips step (cached)
#
#   AdaptiveBatchSizer:
#     - VRAM-aware sub-batch sizing with geometric OOM backoff
#     - OOM_COOLDOWN=50: resets _oom_count after 50 consecutive successes
#     - expandable_segments:True prevents CUDA fragmentation death spiral
#
#   Parallel TAR reader (image encoders):
#     - iter_clips_parallel() in src/utils/data_download.py
#     - 8 threads read 8 TARs concurrently (I/O-bound, GIL-safe)
#     - Shared queue feeds decoder + processor → GPU consumer
#     - Prevents GPU starvation on fast models (CLIP/DINOv2)
#
# ── Comparison plot (Ch10 vs Ch11 on same axes) ──────────────────────
#
#   Both Ch10 (continual pretraining) and Ch11 (surgical fine-tuning) use
#   the same V-JEPA latent-prediction loss (L1 on masked tokens).
#   Same y-axis → directly comparable on one plot.
#
#   from utils.plots import plot_training_curves
#   plot_training_curves(
#       runs=[
#           {"csv_path": "outputs/full/m09_lambda0_001/loss_log.csv",
#            "label": "Continual Pretraining (Ch10)", "color": "blue"},
#           {"csv_path": "outputs/full/m09_surgical/loss_log.csv",
#            "label": "Surgical Fine-tuning (Ch11)", "color": "red"},
#       ],
#       output_dir="outputs/full/comparison",
#       title_prefix="115K Clips, ",
#   )
#
# ── Paper novelty ────────────────────────────────────────────────────
#
#   The novelty is NOT pretraining or surgical individually — both are
#   known. The novelty is:
#
#   1. Factor-decomposed inputs — isolating layout/agents/interactions
#      as separate video "factors" (hence "FactorJEPA") fed to the
#      surgical pipeline
#   2. Comparing full continual pretraining (Ch10) vs. structured
#      curriculum surgical fine-tuning (Ch11) using the SAME
#      self-supervised V-JEPA loss on the SAME Indian urban dataset
#   3. Testing whether structured depth control + factor curriculum
#      can match brute-force pretraining with fewer trainable params
#
#   The comparison plot is the key result figure — it directly shows
#   whether surgical fine-tuning achieves the same loss reduction as
#   full pretraining with less compute.
#
# ── Token budget (same for Ch10 and Ch11) ────────────────────────────
#
#   From profile_data.json at BS=112, total tokens per clip = 4608:
#     Mask gen 1 (8 small blocks): ~1024 visible = 22% of 4608
#     Mask gen 2 (2 large blocks): ~112 visible  = 2.4% of 4608
#
#   Student processes CONTEXT (visible) tokens; teacher sees ALL tokens.
#
#   Ch10 (continual pretraining): student sees ~22% — same V-JEPA 2
#   masking ratio, unchanged.
#
#   Ch11 (surgical fine-tuning): SAME masking, SAME loss, SAME ~22%
#   visible tokens. The difference is NOT token count:
#     1. What's trainable: progressive prefix unfreezing (layers 0→n_s)
#     2. What's fed: factor-patched clips (layout/agent/interaction)
#
#   Surgical efficiency = fewer trainable params (prefix only), not
#   fewer tokens. Comparison plot is valid: identical loss function,
#   masking, and token count — only input distribution and trainable
#   depth differ.
#
# ═══════════════════════════════════════════════════════════════════════
