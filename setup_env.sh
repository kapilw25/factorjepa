#!/bin/bash
# ============================================================================
# Environment Setup Script
# ============================================================================
# Usage:
#   chmod +x setup_env.sh
#
#   # M1 Mac (API-based LLM/VLM)
#   ./setup_env.sh --mac
#
#   # GPU Server (AI2-THOR + PyTorch + Flash-Attention)
#   ./setup_env.sh --gpu
# ============================================================================

set -e  # Exit on error

# Show usage if no flag provided
if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env.sh --mac   # M1 Mac (API-based)"
    echo "  ./setup_env.sh --gpu   # GPU Server (AI2-THOR + PyTorch)"
    exit 1
fi

# Detect OS
OS="$(uname -s)"

# ============================================================================
# Common setup (both --mac and --gpu)
# ============================================================================
setup_base() {
    echo "============================================"
    echo "Environment Setup"
    echo "============================================"
    echo "Detected OS: $OS"
    echo ""

    # Create virtual environment
    if [ ! -d "venv_3Denv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv_3Denv
    else
        echo "Virtual environment already exists"
    fi

    # Activate virtual environment
    source venv_3Denv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install base requirements
    echo "Installing base requirements..."
    pip install -r requirements.txt

    echo ""
    echo "Base setup complete."
    echo ""
}

# ============================================================================
# --mac: M1 Mac setup (API-based)
# ============================================================================
if [ "$1" = "--mac" ]; then
    setup_base

    echo "============================================"
    echo "M1 Mac Setup Complete!"
    echo "============================================"
    echo ""
    echo "To activate environment:"
    echo "  source venv_3Denv/bin/activate"
    echo ""
    echo "Available modules (API-based):"
    echo "  python src/m03_evaluator.py --help        # VLM-as-Judge (OpenAI API)"
    echo "  python src/m03_evaluator.py --plot_only   # Generate plots from metrics.csv"
    echo ""
    echo "Note: m01, m02, m04 require GPU (run on A100 server)"
    echo ""
    exit 0
fi

# ============================================================================
# --gpu: GPU Server setup (AI2-THOR + PyTorch + Flash-Attention)
# ============================================================================
if [ "$1" = "--gpu" ]; then
    if [ "$OS" != "Linux" ]; then
        echo "Error: --gpu requires Linux. Detected: $OS"
        exit 1
    fi

    setup_base

    echo "============================================"
    echo "GPU Setup (Linux + Nvidia)"
    echo "============================================"

    # 0. Install system dependencies (including X server for AI2-THOR)
    echo ""
    echo "[0/6] Installing system dependencies..."
    # libgl1: Ubuntu 24.04+ replacement for deprecated libgl1-mesa-glx
    sudo apt update && sudo apt install -y \
        xvfb \
        libgl1 \
        libglu1-mesa \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-bibtex-extra \
        texlive-science \
        biber \
        tree

    # Start virtual X server for headless AI2-THOR
    echo "Starting Xvfb for AI2-THOR..."
    Xvfb :99 -screen 0 1024x768x24 &
    export DISPLAY=:99

    # 1. Install Python 3.12
    echo ""
    echo "[1/6] Installing Python 3.12..."
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev

    # 2. Install PyTorch 2.5.1 with CUDA 12.4
    echo ""
    echo "[2/6] Installing PyTorch 2.5.1+cu124..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

    # 3. Verify PyTorch
    echo ""
    echo "[3/6] Verifying PyTorch..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

    # 4. Install GPU requirements (AI2-THOR)
    echo ""
    echo "[4/6] Installing GPU requirements (AI2-THOR)..."
    pip install -r requirements_gpu.txt

    # 5. Install Flash-Attention
    echo ""
    echo "[5/6] Installing Flash-Attention 2.8.3..."
    WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

    curl -L -o "$WHEEL_NAME" "$WHEEL_URL"
    pip install "$WHEEL_NAME"
    rm -f "$WHEEL_NAME"

    # 6. Final verification
    echo ""
    echo "[6/6] Verifying GPU setup..."
    python -c "
import torch
import ai2thor
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPU:      {torch.cuda.is_available()}')
print(f'AI2-THOR: {ai2thor.__version__}')
"

    echo ""
    echo "============================================"
    echo "GPU Setup Complete!"
    echo "============================================"
    echo ""
    echo "Available modules:"
    echo "  # Full pipeline (sequential model loading)"
    echo "  python -u src/m04_pipeline_orchestrator.py --sanity 2>&1 | tee logs/m04_sanity.log"
    echo "  python -u src/m04_pipeline_orchestrator.py --full 2>&1 | tee logs/m04_full.log"
    echo ""
    echo "  # Individual modules"
    echo "  python -u src/m01_scene_understanding.py --sanity    # Qwen2.5-VL-32B (~40GB)"
    echo "  python -u src/m02_instruction_generator.py --sanity  # Llama-3.1-70B (~70GB)"
    echo "  python -u src/m03_evaluator.py --sanity              # GPT-4o API"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env.sh --mac   # M1 Mac (API-based)"
echo "  ./setup_env.sh --gpu   # GPU Server (AI2-THOR + PyTorch)"
exit 1
