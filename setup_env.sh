#!/bin/bash
# ============================================================================
# Environment Setup Script - WalkIndia-50 POC
# ============================================================================
# Usage:
#   chmod +x setup_env.sh
#
#   # M1 Mac (CPU-based: download, scene detect, UMAP)
#   ./setup_env.sh --mac
#
#   # GPU Server (V-JEPA, Qwen-VL, FAISS) - Nvidia GPU ONLY
#   ./setup_env.sh --gpu
# ============================================================================

set -e  # Exit on error

# Show usage if no flag provided
if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env.sh --mac   # M1 Mac (CPU-based)"
    echo "  ./setup_env.sh --gpu   # GPU Server (Nvidia ONLY)"
    exit 1
fi

# Detect OS
OS="$(uname -s)"

# ============================================================================
# Common setup (both --mac and --gpu)
# ============================================================================
setup_base() {
    echo "============================================"
    echo "WalkIndia-50 POC Environment Setup"
    echo "============================================"
    echo "Detected OS: $OS"
    echo ""

    # Create virtual environment
    if [ ! -d "venv_walkindia" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv_walkindia
    else
        echo "Virtual environment already exists"
    fi

    # Activate virtual environment
    source venv_walkindia/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install base requirements
    echo "Installing base requirements..."
    pip install -r requirements.txt

    # Create directories
    echo "Creating directories..."
    mkdir -p src/data/videos src/data/clips src/outputs logs

    echo ""
    echo "Base setup complete."
    echo ""
}

# ============================================================================
# --mac: M1 Mac setup (CPU-based)
# ============================================================================
if [ "$1" = "--mac" ]; then
    setup_base

    echo "============================================"
    echo "M1 Mac Setup Complete!"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_walkindia/bin/activate"
    echo ""
    echo "Available modules (CPU-based):"
    echo "  python -u src/m01_download.py --SANITY 2>&1 | tee logs/m01_download_sanity.log"
    echo "  python -u src/m02_scene_detect.py --SANITY 2>&1 | tee logs/m02_scene_detect_sanity.log"
    echo "  python -u src/m06_umap_plot.py --SANITY 2>&1 | tee logs/m06_umap_plot_sanity.log"
    echo ""
    echo "Note: m03, m04, m05 require Nvidia GPU (NO CPU fallback)"
    echo ""
    exit 0
fi

# ============================================================================
# --gpu: GPU Server setup (V-JEPA + Qwen-VL + FAISS) - Nvidia ONLY
# ============================================================================
if [ "$1" = "--gpu" ]; then
    if [ "$OS" != "Linux" ]; then
        echo "Error: --gpu requires Linux. Detected: $OS"
        exit 1
    fi

    setup_base

    echo "============================================"
    echo "GPU Setup (Linux + Nvidia ONLY)"
    echo "============================================"

    # 1. Install PyTorch 2.5.1 with CUDA 12.4
    echo ""
    echo "[1/4] Installing PyTorch 2.5.1+cu124..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/4] Verifying PyTorch + CUDA..."
    python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    exit(1)
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"

    # 3. Install GPU requirements
    echo ""
    echo "[3/4] Installing GPU requirements..."
    pip install -r requirements_gpu.txt

    # 4. Install FAISS-GPU (Nvidia ONLY)
    echo ""
    echo "[4/4] Installing FAISS-GPU (Nvidia ONLY)..."
    pip install faiss-gpu

    # Final verification
    echo ""
    echo "Verifying GPU setup..."
    python -c "
import torch
import faiss

if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)

if faiss.get_num_gpus() == 0:
    print('ERROR: FAISS GPU not available')
    exit(1)

print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.version.cuda}')
print(f'GPU:       {torch.cuda.get_device_name(0)}')
print(f'FAISS GPU: {faiss.get_num_gpus()} GPU(s) available')
print('')
print('SUCCESS: All GPU components verified')
"

    echo ""
    echo "============================================"
    echo "GPU Setup Complete!"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_walkindia/bin/activate"
    echo ""
    echo "GPU modules (Nvidia ONLY):"
    echo "  python -u src/m03_vjepa_embed.py --SANITY 2>&1 | tee logs/m03_vjepa_embed_sanity.log"
    echo "  python -u src/m04_qwen_tag.py --SANITY 2>&1 | tee logs/m04_qwen_tag_sanity.log"
    echo "  python -u src/m05_faiss_metrics.py --SANITY 2>&1 | tee logs/m05_faiss_metrics_sanity.log"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env.sh --mac   # M1 Mac (CPU-based)"
echo "  ./setup_env.sh --gpu   # GPU Server (Nvidia ONLY)"
exit 1
