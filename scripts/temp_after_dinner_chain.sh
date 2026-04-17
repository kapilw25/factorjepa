#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# After-dinner auto-chain — fires when m09c D.2 Surgery POC completes.
# Runs: m05 (surgical re-embed) → m06 (Prec@K metrics) → m09b SANITY.
# Leaves E.2 ExPLoRA POC for user review after decision gate.
#
# Usage (inside a new tmux session for SSH-resilience):
#   tmux new -s afterparty
#   bash src/temp_after_dinner_chain.sh
#   # Ctrl-b d to detach; `tmux attach -t afterparty` to reattach.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Always operate from project root regardless of where this is invoked
cd "$(dirname "$0")/.."

# Activate venv if not already
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv_walkindia/bin/activate
fi

mkdir -p logs
AFTER_LOG="logs/after_dinner.log"

echo "=== after-dinner chain queued at $(date) ===" | tee -a "$AFTER_LOG"

# 1. Poll for m09c D.2 completion (checks any m09c_dense100_surgery*.log — robust
#    against filename variants like _v1, _v2, or no suffix at all)
echo "[1/4] Waiting for m09c D.2 to finish..." | tee -a "$AFTER_LOG"
until grep -q "SURGERY COMPLETE" logs/m09c_dense100_surgery*.log 2>/dev/null; do
    sleep 30
done
echo "=== m09c done at $(date), chain starting ===" | tee -a "$AFTER_LOG"

# 2. m05 re-embed on surgical student (~10 min on 96GB)
echo "[2/4] m05 re-embed — surgical student" | tee -a "$AFTER_LOG"
python -u src/m05_vjepa_embed.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --model outputs/poc/m09c_surgery/student_encoder.pt \
    --encoder vjepa_2_1_surgical \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_surgical_poc.log

# 3. m06 metrics — Prec@K for frozen + surgical (decision gate input).
#    Pattern copied from scripts/train_surgery.sh:225-229 — m06 takes
#    --encoder + --subset, NOT --local-data (that arg is m05/m09-only).
echo "[3a/4] m06 Prec@K — frozen baseline" | tee -a "$AFTER_LOG"
python -u src/m06_faiss_metrics.py --POC \
    --subset data/sanity_100_dense.json \
    --encoder vjepa_2_1_frozen \
    --no-wandb \
    2>&1 | tee logs/m06_frozen_poc.log

echo "[3b/4] m06 Prec@K — surgical" | tee -a "$AFTER_LOG"
python -u src/m06_faiss_metrics.py --POC \
    --subset data/sanity_100_dense.json \
    --encoder vjepa_2_1_surgical \
    --no-wandb \
    2>&1 | tee logs/m06_surgical_poc.log

# 4. m09b ExPLoRA SANITY (20 clips, ~10 min — smoke test only)
echo "[4/4] m09b ExPLoRA SANITY" | tee -a "$AFTER_LOG"
rm -rf outputs/sanity/m09b_explora/
python -u src/m09b_explora.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09b_sanity_explora.log

echo "=== AFTER-DINNER CHAIN COMPLETE at $(date) ===" | tee -a "$AFTER_LOG"
echo "Review m06 Prec@K → decide whether to run E.2 m09b POC manually." | tee -a "$AFTER_LOG"
