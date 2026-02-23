#!/bin/bash
# ============================================================================
# Environment Setup Script - WalkIndia-200K (UV-based)
# ============================================================================
# Usage:
#   chmod +x setup_env_uv.sh
#
#   # M1 Mac (CPU-based: download, scene detect, UMAP)
#   ./setup_env_uv.sh --mac
#
#   # GPU Server (V-JEPA, Qwen-VL, FAISS) - Nvidia GPU ONLY
#   ./setup_env_uv.sh --gpu
#
#   # Activate
#   # source venv_walkindia/bin/activate
# ============================================================================

set -e  # Exit on error

# Show usage if no flag provided
if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env_uv.sh --mac   # M1 Mac (CPU-based)"
    echo "  ./setup_env_uv.sh --gpu   # GPU Server (Nvidia ONLY)"
    exit 1
fi

# Detect OS
OS="$(uname -s)"

# ============================================================================
# Common setup (both --mac and --gpu)
# ============================================================================
setup_base() {
    echo "============================================"
    echo "WalkIndia-200K Environment Setup (UV)"
    echo "============================================"
    echo "Detected OS: $OS"
    echo ""

    # Install system dependencies
    if [ "$OS" = "Linux" ]; then
        NEED_APT_UPDATE=false
        APT_PACKAGES=""

        if ! command -v tree &> /dev/null; then
            APT_PACKAGES="$APT_PACKAGES tree"
            NEED_APT_UPDATE=true
        fi

        if ! command -v python3.12 &> /dev/null; then
            echo "Adding deadsnakes PPA for Python 3.12..."
            apt-get update
            apt-get install -y software-properties-common
            add-apt-repository -y ppa:deadsnakes/ppa
            APT_PACKAGES="$APT_PACKAGES python3.12 python3.12-venv python3.12-dev"
            NEED_APT_UPDATE=true
        fi

        if [ "$NEED_APT_UPDATE" = true ]; then
            echo "Installing: $APT_PACKAGES"
            apt-get update
            apt-get install -y $APT_PACKAGES
        fi
    elif [ "$OS" = "Darwin" ]; then
        command -v tree &> /dev/null || brew install tree
        command -v python3.12 &> /dev/null || brew install python@3.12
    fi
    echo "Python 3.12: $(python3.12 --version)"

    # Install UV if not available
    if ! command -v uv &> /dev/null; then
        echo ""
        echo "Installing UV package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add to PATH for current session
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "UV: $(uv --version)"

    # Create virtual environment with UV
    if [ ! -d "venv_walkindia" ]; then
        echo ""
        echo "Creating virtual environment (Python 3.12)..."
        uv venv --python 3.12 venv_walkindia
    else
        echo "Virtual environment already exists (venv_walkindia)"
    fi

    # Activate virtual environment
    source venv_walkindia/bin/activate

    # Install base requirements with UV
    echo ""
    echo "Installing base requirements (UV - fast)..."
    uv pip install -r requirements.txt

    # Create directories
    echo ""
    echo "Creating directories..."
    mkdir -p src/data/videos src/data/clips src/data/shards src/data/bakeoff src/outputs src/outputs_poc data logs

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
    echo "M1 Mac Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_walkindia/bin/activate"
    echo ""
    echo "See each script's docstring for usage (python src/m*.py --help)"
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
    echo "[1/5] Installing PyTorch 2.5.1+cu124..."
    uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/5] Verifying PyTorch + CUDA..."
    python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    exit(1)
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"

    # 3. Install GPU requirements
    echo ""
    echo "[3/5] Installing GPU requirements (UV - fast)..."
    uv pip install -r requirements_gpu.txt

    # 4. Install Flash-Attention 2.8.3 (pre-built wheel for CUDA 12 + PyTorch 2.5)
    echo ""
    echo "[4/5] Installing Flash-Attention 2.8.3..."
    WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

    # Clean any existing/partial wheel files before download
    rm -f flash_attn*.whl
    # Use aria2 with 16 parallel connections to bypass GitHub CDN throttling
    if command -v aria2c &> /dev/null; then
        aria2c -x 16 -s 16 -o "$WHEEL_NAME" "$WHEEL_URL"
    else
        echo "aria2c not found, using wget..."
        wget -O "$WHEEL_NAME" "$WHEEL_URL"
    fi
    uv pip install "$WHEEL_NAME"
    rm -f "$WHEEL_NAME"

    # 5. Install FAISS-GPU (CUDA 12)
    echo ""
    echo "[5/5] Installing FAISS-GPU (CUDA 12)..."
    uv pip install faiss-gpu-cu12

    # Final verification
    echo ""
    echo "Verifying GPU setup..."
    python -c "
import torch
import faiss
import flash_attn

if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)

if faiss.get_num_gpus() == 0:
    print('ERROR: FAISS GPU not available. No CPU fallback.')
    exit(1)

import transformers
from datasets import load_dataset

print(f'PyTorch:        {torch.__version__}')
print(f'CUDA:           {torch.version.cuda}')
print(f'GPU:            {torch.cuda.get_device_name(0)}')
print(f'VRAM:           {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB')
print(f'FAISS GPU:      {faiss.get_num_gpus()} GPU(s) available')
print(f'Flash-Attn:     {flash_attn.__version__}')
print(f'Transformers:   {transformers.__version__}')
print(f'Datasets:       OK')
print('')
print('SUCCESS: All GPU components verified')
"

    echo ""
    echo "============================================"
    echo "GPU Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_walkindia/bin/activate"
    echo ""
    echo "See each script's docstring for usage (python src/m*.py --help)"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env_uv.sh --mac   # M1 Mac (CPU-based)"
echo "  ./setup_env_uv.sh --gpu   # GPU Server (Nvidia ONLY)"
exit 1
