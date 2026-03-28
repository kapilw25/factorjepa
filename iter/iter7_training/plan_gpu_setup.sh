#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# GPU Instance Setup + VRAM Profile + Sanity Check (Ch10 Pre-flight)
#
# Run this on a FRESH RTX PRO 6000 (96GB) instance.
# Validates: env setup, vjepa2 dependency, outputs/ consolidation,
# VRAM budget for ViT-g training, and Ch9 pipeline integrity.
#
# USAGE (on GPU instance):
#   # 1. Clone repo
#   git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
#
#   # 2. Transfer local data from Mac (skip HF streaming, saves 50+ min)
#   #    Run this ON YOUR MAC (not on GPU instance):
#   #    rsync -avhP data/ <gpu-user>@<gpu-ip>:~/factorjepa/data/
#   #    e.g. -
#   #    rsync -avhP ~/Downloads/research_projects/LLM_asAgent_3D_SR/data/ vast_RTXpro6000_96GB:/workspace/factorjepa/data/
#   #    --- OR ---
#   #    scp -r data/ <gpu-user>@<gpu-ip>:~/factorjepa/data/
#   #
#   #    Transfers: subset_10k.json (592KB) + subset_10k_local/ (9.7GB)
#   #    Time: ~2-3 min at 50MB/s (vs 50+ min HF streaming)
#
#   # 3. Run this script
#   chmod +x iter/iter7_training/plan_gpu_setup.sh
#   ./iter/iter7_training/plan_gpu_setup.sh 2>&1 | tee logs/gpu_setup.log
#
# Prerequisites:
#   - Nvidia GPU with CUDA
#   - Internet access (for pip packages + vjepa2 clone)
#   - data/ folder transferred from Mac (optional but recommended)
#
# Total time: ~20-25 min (setup: 15 min, profile: 3 min, sanity: 5 min)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."  # Navigate to project root

mkdir -p logs

echo "═══════════════════════════════════════════════════════════════"
echo "  Ch10 Pre-flight: GPU Setup + VRAM Profile + Sanity Check"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "GPU:     $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "VRAM:    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Project: $(pwd)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Environment setup (PyTorch, FA2, FAISS, cuML, vjepa2)
# ═══════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "  Step 1/4: Environment Setup (~15 min)"
echo "════════════════════════════════════════════════════════════"

if [ -d "venv_walkindia" ] && [ -f "venv_walkindia/bin/activate" ]; then
    echo "venv_walkindia already exists. Skipping setup."
    source venv_walkindia/bin/activate
else
    ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
    source venv_walkindia/bin/activate
fi

# Quick verify
python -c "
import torch, faiss
assert torch.cuda.is_available(), 'CUDA not available'
assert faiss.get_num_gpus() > 0, 'FAISS GPU not available'
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  FAISS: {faiss.get_num_gpus()} GPU(s)')
"

# Verify vjepa2 dependency
if [ ! -d "deps/vjepa2/src" ]; then
    echo "ERROR: deps/vjepa2 not cloned. setup_env_uv.sh should have done this."
    echo "Manual fix: git clone --depth 1 https://github.com/facebookresearch/vjepa2.git deps/vjepa2"
    exit 1
fi
echo "  vjepa2: deps/vjepa2/ present"
echo ""
echo "Step 1 DONE."
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Check local data (skip HF streaming)
# ═══════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "  Step 2/4: Verify Local Data"
echo "════════════════════════════════════════════════════════════"

if [ -f "data/subset_10k_local/manifest.json" ]; then
    N_TARS=$(find data/subset_10k_local -name "*.tar" 2>/dev/null | wc -l)
    SIZE=$(du -sh data/subset_10k_local/ | cut -f1)
    echo "  Local data: data/subset_10k_local/ (${N_TARS} TARs, ${SIZE})"
    echo "  SANITY + FULL will use --local-data (no HF streaming)"
else
    echo "  WARNING: data/subset_10k_local/ not found."
    echo "  Transfer from Mac:  rsync -avhP data/ <gpu-user>@<gpu-ip>:~/factorjepa/data/"
    echo "  Without it, SANITY will stream from HF (slow, 50+ min)."
    echo ""
    read -p "  Continue without local data? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Transfer data first, then re-run."
        exit 1
    fi
fi

if [ -f "data/subset_10k.json" ]; then
    N_CLIPS=$(python -c "import json; print(len(json.load(open('data/subset_10k.json'))['clip_keys']))")
    echo "  Subset: data/subset_10k.json (${N_CLIPS} clips)"
else
    echo "  WARNING: data/subset_10k.json not found."
fi
echo ""
echo "Step 2 DONE."
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 3: VRAM Profiler (ViT-g batch size sweep)
# ═══════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "  Step 3/4: VRAM Profile — ViT-g 1B (~3 min)"
echo "════════════════════════════════════════════════════════════"

if [ -f "outputs/profile/profile_data.json" ]; then
    echo "  Profile already exists: outputs/profile/profile_data.json"
    echo "  Skipping. Delete to re-run."
else
    python scripts/profile_vram.py 2>&1 | tee logs/profile_vram.log
fi
echo ""
echo "Step 3 DONE."
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Ch9 SANITY (verify outputs/ consolidation works end-to-end)
# ═══════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "  Step 4/4: Ch9 SANITY — Verify Pipeline (~5 min)"
echo "════════════════════════════════════════════════════════════"

# Only run if sanity outputs don't exist yet
if [ -f "outputs/sanity/m06_metrics.json" ]; then
    echo "  Sanity outputs already exist: outputs/sanity/m06_metrics.json"
    echo "  Skipping. Delete outputs/sanity/ to re-run."
else
    ./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log
fi
echo ""
echo "Step 4 DONE."
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL STEPS COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Outputs:"
echo "    outputs/profile/   — VRAM profiler results + 5 plots"
echo "    outputs/sanity/    — Ch9 sanity check results"
echo ""
echo "  Next steps:"
echo "    1. Review outputs/profile/plot1_batch_scaling.png"
echo "    2. Set batch_size in configs/pretrain/vitg16_indian.yaml"
echo "    3. Push results to Mac:  ./git_push.sh \"Add VRAM profile + sanity results\""
echo "    4. Implement m09_pretrain.py (Ch10)"
echo ""
