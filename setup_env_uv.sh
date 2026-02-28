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

        for pkg in jq htop tmux wget curl; do
            if ! command -v "$pkg" &> /dev/null; then
                APT_PACKAGES="$APT_PACKAGES $pkg"
                NEED_APT_UPDATE=true
            fi
        done

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

    # 1. Install PyTorch (auto-detect Blackwell vs Ampere/Hopper)
    echo ""
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    echo "Detected GPU: ${GPU_NAME:-unknown}"
    if echo "$GPU_NAME" | grep -qiE "blackwell|rtx.*pro.*(4000|6000)|rtx.*5090|rtx.*5080|rtx.*5070"; then
        echo "[1/7] Installing PyTorch (CUDA 12.8 — Blackwell)..."
        echo "NOTE: Blackwell requires PyTorch nightly with cu128"
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        echo "[1/7] Installing PyTorch 2.5.1+cu124..."
        uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    fi

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/7] Verifying PyTorch + CUDA..."
    python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    exit(1)
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"

    # 3. Install GPU requirements
    echo ""
    echo "[3/7] Installing GPU requirements (UV - fast)..."
    uv pip install -r requirements_gpu.txt

    # 4. Install Flash-Attention 2 (auto-detect GPU arch)
    echo ""
    echo "[4/7] Installing Flash-Attention 2..."
    GPU_ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
    echo "GPU compute capability: sm_${GPU_ARCH:-unknown}"

    if [ "$GPU_ARCH" = "80" ] || [ "$GPU_ARCH" = "86" ] || [ "$GPU_ARCH" = "89" ] || [ "$GPU_ARCH" = "90" ]; then
        # Ampere/Ada/Hopper — use prebuilt wheel (fast)
        echo "Using prebuilt FA2 wheel for sm_${GPU_ARCH}..."
        WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
        rm -f flash_attn*.whl
        if command -v aria2c &> /dev/null; then
            aria2c -x 16 -s 16 -o "$WHEEL_NAME" "$WHEEL_URL"
        else
            wget -O "$WHEEL_NAME" "$WHEEL_URL"
        fi
        uv pip install "$WHEEL_NAME"
        rm -f "$WHEEL_NAME"
    elif [ -n "$GPU_ARCH" ]; then
        # Unknown arch (e.g. sm_120 Blackwell) — build from source
        echo "WARNING: No prebuilt FA2 wheel for sm_${GPU_ARCH}. Building from source (30-90 min)..."
        if ! command -v nvcc &> /dev/null; then
            for CUDA_PATH in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12; do
                if [ -f "${CUDA_PATH}/bin/nvcc" ]; then
                    export PATH="${CUDA_PATH}/bin:$PATH"
                    export CUDA_HOME="${CUDA_PATH}"
                    break
                fi
            done
        fi
        if ! command -v nvcc &> /dev/null; then
            echo "ERROR: nvcc not found. Cannot build FA2 from source."
            echo "Install CUDA toolkit 12.8+ or set CUDA_HOME."
            echo "Scripts will fall back to PyTorch SDPA (slower but functional)."
        else
            export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"
            echo "Using nvcc: $(nvcc --version | grep release)"
            FA2_DIR="/tmp/flash-attention-build"
            rm -rf "$FA2_DIR"
            git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FA2_DIR"
            cd "$FA2_DIR" && git submodule update --init --recursive && cd -
            echo "Compiling FA2 for sm_${GPU_ARCH} (this takes 30-90 min)..."
            FLASH_ATTN_CUDA_ARCHS="${GPU_ARCH}" MAX_JOBS=4 NVCC_THREADS=1 \
                uv pip install "$FA2_DIR" --no-build-isolation 2>&1 | tee /tmp/fa2_build.log
            rm -rf "$FA2_DIR"
            echo "FlashAttention-2 built for sm_${GPU_ARCH}"
        fi
    else
        echo "WARNING: Could not detect GPU arch. Skipping FA2."
        echo "Install manually: TORCH_CUDA_ARCH_LIST='X.Y' pip install flash-attn --no-build-isolation"
    fi

    # 5. Install FAISS-GPU (CUDA 12)
    echo ""
    echo "[5/7] Installing FAISS-GPU (CUDA 12)..."
    uv pip install faiss-gpu-cu12

    # 6. Install cuML (GPU UMAP) from RAPIDS PyPI
    echo ""
    echo "[6/7] Installing cuML (GPU UMAP)..."
    uv pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com

    # 7. Install wandb (experiment tracking)
    echo ""
    echo "[7/7] Installing wandb..."
    uv pip install wandb

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
    print('ERROR: FAISS GPU not available. No CPU fallback.')
    exit(1)

cc = torch.cuda.get_device_capability()
try:
    import flash_attn
    fa_ver = flash_attn.__version__
except ImportError:
    fa_ver = 'NOT INSTALLED (will use PyTorch SDPA)'

import transformers
from datasets import load_dataset
import cuml
import wandb

print(f'PyTorch:        {torch.__version__}')
print(f'CUDA:           {torch.version.cuda}')
print(f'GPU:            {torch.cuda.get_device_name(0)}')
print(f'GPU Arch:       sm_{cc[0]}{cc[1]}')
print(f'VRAM:           {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB')
print(f'FAISS GPU:      {faiss.get_num_gpus()} GPU(s) available')
print(f'Flash-Attn:     {fa_ver}')
print(f'Transformers:   {transformers.__version__}')
print(f'cuML:           {cuml.__version__}')
print(f'wandb:          {wandb.__version__}')
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
