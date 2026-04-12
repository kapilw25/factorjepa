#!/bin/bash
# ============================================================================
# Environment Setup Script - WalkIndia-200K (UV-based)
# ============================================================================
# Usage:
#   chmod +x setup_env_uv.sh
#
#   # M1 Mac (CPU-based: download, scene detect, UMAP)
#   mkdir -p logs && ./setup_env_uv.sh --mac 2>&1 | tee logs/setup_env_cpu.log
#
#   # GPU Server (FA2 + FAISS source build)
#   mkdir -p logs && ./setup_env_uv.sh --gpu 2>&1 | tee logs/setup_env_gpu.log
#
#   # GPU Server with prebuilt wheels (skip FA2 + FAISS source build)
#   mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
#
#   # Activate
#   # source venv_walkindia/bin/activate # replace    
# ============================================================================

set -e  # Exit on error
mkdir -p logs  # Ensure logs/ exists early (for tee piping)

# ============================================================================
# Pinned versions (Blackwell sm_120 + CUDA 13.0 + Python 3.12)
# ============================================================================
TORCH_VERSION="2.12.0.dev20260228"  # PyTorch nightly cu128 — pinned for FA2 wheel compat
RELEASE_TAG="sm120-cu128-py312"     # GitHub release tag for prebuilt FA2 + FAISS wheels

# ============================================================================
# Parse flags
# ============================================================================
FROM_WHEELS=false
for arg in "$@"; do
    if [ "$arg" = "--from-wheels" ]; then
        FROM_WHEELS=true
    fi
done

# Show usage if no flag provided
if [ -z "$1" ]; then
    echo "Error: No flag provided"
    echo ""
    echo "Usage:"
    echo "  ./setup_env_uv.sh --mac                 # M1 Mac (CPU-based)"
    echo "  ./setup_env_uv.sh --gpu                 # GPU Server (Nvidia ONLY)"
    echo "  ./setup_env_uv.sh --gpu --from-wheels   # GPU + prebuilt FA2/FAISS wheels"
    exit 1
fi

# Detect OS
OS="$(uname -s)"

# ============================================================================
# Download prebuilt sm_120 wheels from GitHub Release
# ============================================================================
download_sm120_wheels() {
    local REPO_SLUG
    REPO_SLUG=$(git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||' | sed 's|\.git$||')
    if [ -z "$REPO_SLUG" ]; then
        echo "FATAL: Cannot detect GitHub repo from git remote."
        return 1
    fi

    local API_URL="https://api.github.com/repos/${REPO_SLUG}/releases/tags/${RELEASE_TAG}"
    mkdir -p wheels
    echo "Downloading prebuilt sm_120 wheels from: github.com/${REPO_SLUG}/releases/tag/${RELEASE_TAG}"

    local URLS
    URLS=$(curl -sL "$API_URL" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for asset in data.get('assets', []):
        if asset['name'].endswith('.whl'):
            print(asset['browser_download_url'])
except: pass
" 2>/dev/null)

    if [ -z "$URLS" ]; then
        echo "WARNING: No wheels found in release '${RELEASE_TAG}'."
        echo "Upload wheels first: gh release create ${RELEASE_TAG} wheels/*.whl"
        return 1
    fi

    for url in $URLS; do
        echo "  Downloading: $(basename "$url")"
        wget -q -P wheels/ "$url"
    done
    echo "Downloaded $(ls wheels/*.whl 2>/dev/null | wc -l) wheel(s) to wheels/"
}

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
    mkdir -p src/data/videos src/data/clips src/data/shards src/data/bakeoff \
            outputs/full outputs/poc outputs/sanity outputs/data_prep outputs/profile \
            data logs wheels deps

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

    # Load .env for HF_TOKEN (SAM 3.1 is gated, checkpoint downloads need auth)
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
        echo "Loaded .env (HF_TOKEN set)"
    fi

    # Clone vjepa2 (contains BOTH 2.0 and 2.1 code)
    # 2.0: deps/vjepa2/src/models/ (base ViT, predictor)
    # 2.1: deps/vjepa2/app/vjepa_2_1/models/ (hierarchical output, deep supervision)
    if [ ! -d "deps/vjepa2/src" ]; then
        echo ""
        echo "Cloning facebookresearch/vjepa2 (contains V-JEPA 2.0 + 2.1)..."
        rm -rf deps/vjepa2
        git clone --depth 1 https://github.com/facebookresearch/vjepa2.git deps/vjepa2
        echo "vjepa2 cloned to deps/vjepa2/"
    else
        echo "deps/vjepa2 already present"
    fi

    # Download V-JEPA 2.1 ViT-G (2B) checkpoint (~8 GB)
    VJEPA_CKPT="checkpoints/vjepa2_1_vitG_384.pt"
    if [ ! -f "$VJEPA_CKPT" ]; then
        echo ""
        echo "Downloading V-JEPA 2.1 ViT-G checkpoint (~8 GB)..."
        mkdir -p checkpoints
        wget -q --show-progress https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -P checkpoints/
        echo "Checkpoint saved: $VJEPA_CKPT"
    else
        echo "V-JEPA 2.1 checkpoint already present: $VJEPA_CKPT"
    fi

    # Download prebuilt wheels if --from-wheels
    if [ "$FROM_WHEELS" = true ]; then
        echo ""
        echo "=== Downloading prebuilt wheels ==="
        download_sm120_wheels || {
            echo "Falling back to building from source..."
            FROM_WHEELS=false
        }
    fi

    # 1. Install PyTorch (auto-detect Blackwell vs Ampere/Hopper)
    echo ""
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    echo "Detected GPU: ${GPU_NAME:-unknown}"
    if echo "$GPU_NAME" | grep -qiE "blackwell|rtx.*pro.*(4000|6000)|rtx.*5090|rtx.*5080|rtx.*5070"; then
        echo "[1/7] Installing PyTorch ${TORCH_VERSION}+cu128 (Blackwell — pinned)..."
        uv pip install "torch==${TORCH_VERSION}" torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
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

    # 3. Install GPU requirements (includes hf_transfer for fast HF downloads)
    echo ""
    echo "[3/7] Installing GPU requirements (UV - fast)..."
    uv pip install -r requirements_gpu.txt
    # Enable Rust-based HF transfer (1.5-3x faster downloads per file)
    export HF_HUB_ENABLE_HF_TRANSFER=1

    # 4. Install Flash-Attention 2 (auto-detect GPU arch)
    echo ""
    echo "[4/7] Installing Flash-Attention 2..."
    GPU_ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
    echo "GPU compute capability: sm_${GPU_ARCH:-unknown}"

    if ls wheels/flash_attn*.whl &>/dev/null 2>&1; then
        # Prebuilt wheel available (from --from-wheels or previous build)
        echo "Installing FA2 from prebuilt wheel..."
        uv pip install wheels/flash_attn*.whl
    elif [ "$GPU_ARCH" = "80" ] || [ "$GPU_ARCH" = "86" ] || [ "$GPU_ARCH" = "89" ] || [ "$GPU_ARCH" = "90" ]; then
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
        # Unknown arch (e.g. sm_120 Blackwell) — check existing install, then build from source
        if python -c "import flash_attn; print(f'Flash-Attn {flash_attn.__version__} already installed')" 2>/dev/null; then
            echo "Skipping FA2 build (already installed)."
        else
        echo "WARNING: No prebuilt FA2 wheel for sm_${GPU_ARCH}. Building from source (30-90 min)..."

        # FA2 build requires nvcc matching PyTorch's CUDA version exactly
        PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
        echo "PyTorch compiled with CUDA: ${PYTORCH_CUDA}"

        # Search for version-matched CUDA toolkit (versioned paths first)
        FA2_CUDA_HOME=""
        for CUDA_PATH in "/usr/local/cuda-${PYTORCH_CUDA}" /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
            if [ -f "${CUDA_PATH}/bin/nvcc" ]; then
                NVCC_VER=$("${CUDA_PATH}/bin/nvcc" --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
                if [ "${NVCC_VER}" = "${PYTORCH_CUDA}" ]; then
                    FA2_CUDA_HOME="${CUDA_PATH}"
                    echo "Found matching CUDA ${NVCC_VER} toolkit at ${CUDA_PATH}"
                    break
                fi
            fi
        done

        # If no match found, install matching CUDA toolkit via apt
        if [ -z "$FA2_CUDA_HOME" ]; then
            CUDA_PKG="cuda-toolkit-$(echo "${PYTORCH_CUDA}" | tr '.' '-')"
            echo "No CUDA ${PYTORCH_CUDA} toolkit found. Installing ${CUDA_PKG} via apt..."
            apt-get update -qq && apt-get install -y -qq "${CUDA_PKG}" > /dev/null 2>&1 || true
            if [ -f "/usr/local/cuda-${PYTORCH_CUDA}/bin/nvcc" ]; then
                FA2_CUDA_HOME="/usr/local/cuda-${PYTORCH_CUDA}"
                echo "Installed CUDA ${PYTORCH_CUDA} at ${FA2_CUDA_HOME}"
            fi
        fi

        if [ -z "$FA2_CUDA_HOME" ]; then
            echo "FATAL: Could not find or install CUDA toolkit ${PYTORCH_CUDA}."
            echo "System nvcc: $(nvcc --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p') (needs ${PYTORCH_CUDA})"
            echo "Install manually: apt-get install cuda-toolkit-$(echo "${PYTORCH_CUDA}" | tr '.' '-')"
        else
            export CUDA_HOME="${FA2_CUDA_HOME}"
            export PATH="${FA2_CUDA_HOME}/bin:$PATH"
            echo "Using nvcc: $(nvcc --version | grep release)"
            FA2_DIR="/tmp/flash-attention-build"
            rm -rf "$FA2_DIR"
            git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FA2_DIR"
            cd "$FA2_DIR" && git submodule update --init --recursive && cd -
            # Build wheel (saved to wheels/ for caching) then install
            echo "Building FA2 wheel for sm_${GPU_ARCH} (this takes 30-90 min)..."
            mkdir -p wheels
            uv pip install pip 2>/dev/null || true
            FLASH_ATTN_CUDA_ARCHS="${GPU_ARCH}" MAX_JOBS=4 NVCC_THREADS=1 \
                pip wheel "$FA2_DIR" --no-build-isolation --no-deps --wheel-dir wheels/ 2>&1 | tee /tmp/fa2_build.log
            uv pip install wheels/flash_attn*.whl
            rm -rf "$FA2_DIR"
            echo "FlashAttention-2 built for sm_${GPU_ARCH}"
            echo "Wheel saved: $(ls wheels/flash_attn*.whl 2>/dev/null | head -1)"
        fi
        fi  # end of "already installed" check
    else
        echo "WARNING: Could not detect GPU arch. Skipping FA2."
        echo "Install manually: TORCH_CUDA_ARCH_LIST='X.Y' pip install flash-attn --no-build-isolation"
    fi

    # 5. Install FAISS-GPU (CUDA 12)
    echo ""
    echo "[5/7] Installing FAISS-GPU (CUDA 12)..."
    # FAISS source-built wheel needs libopenblas at runtime
    if ! dpkg -s libopenblas-dev &>/dev/null 2>&1; then
        echo "Installing libopenblas-dev (FAISS runtime dependency)..."
        apt-get update -qq && apt-get install -y -qq libopenblas-dev > /dev/null 2>&1
    fi

    if ls wheels/faiss*.whl &>/dev/null 2>&1; then
        # Prebuilt wheel available (from --from-wheels or previous build)
        echo "Installing FAISS-GPU from prebuilt wheel..."
        uv pip uninstall -y faiss-gpu faiss-gpu-cu12 faiss-cpu faiss 2>/dev/null || true
        uv pip install wheels/faiss*.whl
        # Fix RPATH so _swigfaiss.so finds libfaiss.so in the same directory
        FAISS_PKG="$(python -c 'import sysconfig,os; print(os.path.join(sysconfig.get_path("purelib"),"faiss"))' 2>/dev/null || echo '')"
        if [ -n "$FAISS_PKG" ] && [ -d "$FAISS_PKG" ]; then
            command -v patchelf &>/dev/null || apt-get install -y -qq patchelf > /dev/null 2>&1
            for so in "$FAISS_PKG"/_swigfaiss*.so "$FAISS_PKG"/libfaiss_python_callbacks.so "$FAISS_PKG"/_faiss_example_external_module.so; do
                [ -f "$so" ] && patchelf --set-rpath '$ORIGIN' "$so"
            done
            echo "Fixed RPATH for FAISS .so files in $FAISS_PKG"
        fi
    elif [ "$GPU_ARCH" = "120" ]; then
        if [ -f "/tmp/faiss_build/build/faiss/python/setup.py" ]; then
            echo "Blackwell (sm_120) — reinstalling FAISS-GPU from cached build artifacts..."
            ./build_faiss_sm120.sh --install 2>&1 | tee logs/build_faiss_sm120.log
        else
            echo "Blackwell (sm_120) — building FAISS-GPU from source (~10 min, pip wheel lacks sm_120 kernels)..."
            ./build_faiss_sm120.sh 2>&1 | tee logs/build_faiss_sm120.log
        fi
    else
        uv pip install faiss-gpu-cu12
    fi

    # 6. Install cuML (GPU UMAP) from RAPIDS PyPI
    echo ""
    echo "[6/7] Installing cuML (GPU UMAP)..."
    uv pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com

    # 7. Install wandb (experiment tracking)
    echo ""
    echo "[7/8] Installing wandb..."
    uv pip install wandb

    # 8. Install SAM 3.1 (gated model — user must accept access at hf.co/facebook/sam3.1)
    echo ""
    echo "[8/8] Installing SAM 3.1 from source..."
    if python -c "import sam3" 2>/dev/null; then
        echo "SAM 3.1 already installed"
    else
        pip install git+https://github.com/facebookresearch/sam3.git
    fi

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
print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
print(f'FAISS:          {faiss.__version__} ({faiss.get_num_gpus()} GPU(s) available)')
print(f'Flash-Attn:     {fa_ver}')
print(f'Transformers:   {transformers.__version__}')
print(f'cuML:           {cuml.__version__}')
print(f'wandb:          {wandb.__version__}')
print(f'Datasets:       OK')
print('')
print('SUCCESS: All GPU components verified')
"

    # vLLM setup SKIPPED — transformers is 2.5x faster for offline batch tagging.
    # See iter/utils/vLLM_plan_Blackwell.md for 14 root causes found + fixed.
    echo ""
    echo "vLLM: SKIPPED (transformers 2.5x faster for offline batch, see vLLM_plan_Blackwell.md)"

    echo ""
    echo "============================================"
    echo "GPU Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "Two venvs:"
    echo "  source venv_walkindia/bin/activate   # m04-m09 pipeline (FAISS, cuML, V-JEPA)"
    echo "  source venv_vllm/bin/activate        # m04 vLLM tagging (optional, 3-5x faster)"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_evaluate.sh --FULL            # transformers (always works)"
    echo "  ./scripts/run_evaluate.sh --FULL --vllm     # vLLM (faster, if available)"
    echo ""
    exit 0
fi

# Unknown flag
echo "Error: Unknown flag '$1'"
echo ""
echo "Usage:"
echo "  ./setup_env_uv.sh --mac                 # M1 Mac (CPU-based)"
echo "  ./setup_env_uv.sh --gpu                 # GPU Server (Nvidia ONLY)"
echo "  ./setup_env_uv.sh --gpu --from-wheels   # GPU + prebuilt FA2/FAISS wheels"
exit 1
