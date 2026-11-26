#!/bin/bash
# ============================================================================
# Environment Setup Script - DENSEWORLD (UV-based)
# ============================================================================
# Usage:
#   chmod +x setup_env_uv.sh
#
#   # M1 Mac (CPU-based: download, scene detect, UMAP)
#   mkdir -p logs && ./setup_env_uv.sh --mac 2>&1 | tee logs/setup_env_cpu_$(date +%Y%m%d_%H%M%S).log
#
#   # GPU Server (FA2 + FAISS source build)
#   mkdir -p logs && ./setup_env_uv.sh --gpu 2>&1 | tee logs/setup_env_gpu_$(date +%Y%m%d_%H%M%S).log
#
#   # GPU Server with prebuilt wheels (skip FA2 + FAISS source build)
#   mkdir -p logs && bash setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu_$(date +%Y%m%d_%H%M%S).log
#
#   # Activate
#   # source venv/bin/activate # replace    
# ============================================================================

set -e  # Exit on error
mkdir -p logs  # Ensure logs/ exists early (for tee piping)

# ============================================================================
# Redirect UV scratch space to /dev/shm (RAM-tmpfs, typically 26-31 GB).
# Why: PyTorch nightly + CUDA wheel set explodes to ~8.5 GB during extraction.
# On vast.ai, both / (overlay) and /workspace (XFS prjquota, often host-shared
# and near-full) can OOM the install. /dev/shm is per-container RAM and is
# always sized at ~half-RAM, so it has the headroom and clears itself after
# the install finishes.
# Evidence: logs/setup_env_gpu_20260519_182245.log:335-340 (/tmp ENOSPC) +
#          logs/setup_env_gpu_20260519_183654.log:195-199 (/workspace ENOSPC).
# ============================================================================
if [ -d "/dev/shm" ]; then
    export TMPDIR="${TMPDIR:-/dev/shm/uv_tmp}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/uv_cache}"
    mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
fi

# uv's default HTTP timeout is 30s — too low for the 250-630 MB CUDA wheels that torch pulls
# (cusolver 255M, cublas 567M, cudnn 627M). A single transient stall trips `set -e` and aborts the
# WHOLE setup before torch installs (logs/setup_env_gpu_20260530_231230.log: cusolver timeout). 10 min.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

# ============================================================================
# Pinned versions (Blackwell sm_120 + CUDA 13.0 + Python 3.12)
# ============================================================================
TORCH_VERSION="2.12.0.dev20260408"  # PyTorch nightly cu128 — pinned for FA2 wheel compat.
                                    # 2026-04-30 bump #1: dev20260228 was rotated off the nightly CDN
                                    # (logs/setup_env_gpu_v1.log:277-279).
                                    # 2026-04-30 bump #2: dev20260408 has no paired torchvision build
                                    # — latest torchvision (0.27.0.dev20260407) only declares compat
                                    # with torch 20260406/20260407, so we drop one day to dev20260407
                                    # which DOES have a paired torchvision (logs/setup_env_gpu_v2.log:414).
                                    # 2026-06-12 bump #3: dev20260407 rotated off the CDN too
                                    # (logs/setup_env_gpu_20260612_061338.log:650 "no version of
                                    # torch==2.12.0.dev20260407"). dev20260408 is now the ONLY 2.12
                                    # nightly left on whl/nightly/cu128, and the only surviving
                                    # torchvision (0.27.0.dev20260407) hard-pins the ROTATED torch
                                    # (its wheel METADATA: "Requires-Dist: torch (==2.12.0.dev20260407)",
                                    # verified 2026-06-12) → NO resolver-clean pair exists. Fix:
                                    # install torch==dev20260408, then torchvision --no-deps (below).
                                    # ABI-safe: same 2.12 branch, one day apart. If dev20260408 also
                                    # rotates, re-derive: curl the torch/ + torchvision/ index pages,
                                    # pick the newest surviving pair, update BOTH pins here.
                                    # NOTE: the prebuilt FA2 wheel in wheels/ was built against the
                                    # 20260228 ABI. With this pin the auto-detect at line ~343 will
                                    # take the source-rebuild path (~30-90 min) unless a fresh
                                    # FA2 wheel matching dev20260408 is uploaded to GitHub release
                                    # `sm120-cu128-py312` first. Same applies to FAISS-GPU.
TORCHVISION_VERSION="0.27.0.dev20260407"  # last torchvision on nightly/cu128 (2026-06-12); its
                                    # metadata pins the rotated torch dev20260407 → installed with
                                    # --no-deps against torch dev20260408 (same-branch ABI).
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
    echo "DENSEWORLD Environment Setup (UV)"
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
    if [ ! -d "venv" ]; then
        echo ""
        echo "Creating virtual environment (Python 3.12)..."
        uv venv --python 3.12 venv
    else
        echo "Virtual environment already exists (venv)"
    fi

    # Activate virtual environment
    source venv/bin/activate

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
    echo "  source venv/bin/activate"
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

    # Idempotent patch: cast RoPE-rotated Q/K back to V.dtype before SDPA.
    # V-JEPA 2.1's RoPE keeps Q/K in fp32 for numerical stability, but the model is
    # loaded in bf16 → torch.compile's FakeTensor check rejects the Q/K/V dtype
    # mismatch at F.scaled_dot_product_attention. Lossless downcast (rotation math
    # stays fp32, only final Q/K result is cast). Must re-apply on every fresh clone.
    # errors_N_fixes.md #44. Idempotent: skips if already present.
    python3 << 'PYEOF'
path = 'deps/vjepa2/app/vjepa_2_1/models/utils/modules.py'
src = open(path).read()
if 'q = q.to(v.dtype)' in src:
    print("vjepa2 RoPE dtype patch already present (#44)")
else:
    anchor = "            k = torch.cat([kd, kh, kw], dim=-1)\n\n        if self.use_sdpa:"
    replacement = (
        "            k = torch.cat([kd, kh, kw], dim=-1)\n\n"
        "        # RoPE outputs fp32 Q/K; cast back to V.dtype before SDPA so\n"
        "        # torch.compile FakeTensor check passes. errors_N_fixes.md #44.\n"
        "        q = q.to(v.dtype)\n"
        "        k = k.to(v.dtype)\n\n"
        "        if self.use_sdpa:"
    )
    if anchor not in src:
        raise SystemExit(f"FATAL: vjepa2 upstream refactored {path} — #44 anchor not found. Re-derive patch.")
    open(path, 'w').write(src.replace(anchor, replacement))
    print("vjepa2 RoPE dtype patch applied (#44)")
PYEOF

    # Download ONLY the backbones the scheduler TRAINS — registry group 'trainable' (iter18: 2B ViT-G only, ~16 GB) — so the
    # torch / GPU-deps install below does NOT wait behind the ~59 GB of eval-only §G baseline ckpts
    # (iter16 vitG champion + image/version frozen baselines). Those live under group 'eval_baselines'
    # in configs/checkpoints_download.yaml and are fetched ON DEMAND, before (re)evaluating the frozen
    # row, by run_eval's downstream consumers:
    #   python -u src/utils/download_checkpoints.py --registry configs/checkpoints_download.yaml \
    #       --group eval_baselines --dest checkpoints
    # The registry is the SINGLE SOURCE (catalog: canonical filename→url ; groups: editable membership —
    # add a name to `trainable` to make setup_env fetch it). Idempotent + resumable (aria2c -c). Canonical
    # filenames are what run_eval.sh frozen_ckpt_for + configs/eval/probe_encoders.yaml expect.
    mkdir -p checkpoints
    command -v aria2c &> /dev/null || apt-get install -y -qq aria2 > /dev/null 2>&1
    python -u src/utils/download_checkpoints.py \
        --registry configs/checkpoints_download.yaml --group trainable --dest checkpoints

    # Detect GPU EARLY — the prebuilt wheels in RELEASE_TAG are sm_120/cu128/torch2.12 (Blackwell-only).
    echo ""
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    echo "Detected GPU: ${GPU_NAME:-unknown}"
    IS_BLACKWELL=false
    if echo "$GPU_NAME" | grep -qiE "blackwell|rtx.*pro.*(4000|6000)|rtx.*5090|rtx.*5080|rtx.*5070"; then
        IS_BLACKWELL=true
    fi

    # Download prebuilt wheels if --from-wheels — ONLY on Blackwell. Those wheels need sm_120 + CUDA 13
    # (libcudart.so.13) + the torch-2.12 ABI; on Ampere/Ada (e.g. RTX 3070 → torch 2.5/cu124) they
    # ImportError, so a non-Blackwell box must fetch the arch-matched FA2 + faiss-gpu-cu12 instead. 2026-07-01.
    if [ "$FROM_WHEELS" = true ] && [ "$IS_BLACKWELL" = true ]; then
        echo ""
        echo "=== Downloading prebuilt wheels ==="
        download_sm120_wheels || {
            echo "Falling back to building from source..."
            FROM_WHEELS=false
        }
    elif [ "$FROM_WHEELS" = true ]; then
        echo "--from-wheels IGNORED: '${GPU_NAME}' is not Blackwell (sm_120) — using arch-matched FA2 + faiss-gpu-cu12."
        FROM_WHEELS=false
    fi

    # 1. Install PyTorch (auto-detect Blackwell vs Ampere/Hopper)
    echo ""
    if [ "$IS_BLACKWELL" = true ]; then
        echo "[1/10] Installing PyTorch ${TORCH_VERSION}+cu128 (Blackwell — pinned)..."
        # Two-step install (bump #3, 2026-06-12): torchvision dev20260407's metadata pins the
        # CDN-rotated torch dev20260407, so a combined resolve is UNSATISFIABLE. Install torch
        # alone, then torchvision --no-deps (same 2.12 branch, 1 day apart → ABI-compatible;
        # the [2/10] verify imports torchvision so an ABI break still FAILS LOUD right here).
        uv pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/nightly/cu128
        uv pip install --no-deps "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        echo "[1/10] Installing PyTorch 2.5.1+cu124..."
        uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    fi

    # 2. Verify PyTorch + CUDA
    echo ""
    echo "[2/10] Verifying PyTorch + CUDA..."
    python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. Nvidia GPU required.')
    exit(1)
# torchvision is installed --no-deps (bump #3) — importing its compiled ops here
# FAILS LOUD on a torch ABI mismatch instead of mid-pipeline.
import torchvision
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
print(f'torchvision: {torchvision.__version__} (installed --no-deps, ABI import OK)')
"

    # 3. Install GPU requirements (hf-xet ships with huggingface_hub >=1.x for fast HF transfers)
    echo ""
    echo "[3/10] Installing GPU requirements (UV - fast)..."
    uv pip install -r requirements_gpu.txt
    # hf_transfer was REMOVED in huggingface_hub 1.x — HF_HUB_ENABLE_HF_TRANSFER is a silent no-op.
    # The supported fast path is hf-xet high-performance mode (saturates bandwidth + cores).
    export HF_XET_HIGH_PERFORMANCE=1

    # 4. Install Flash-Attention 2 (auto-detect GPU arch)
    echo ""
    echo "[4/10] Installing Flash-Attention 2..."
    GPU_ARCH=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
    echo "GPU compute capability: sm_${GPU_ARCH:-unknown}"

    if [ "$IS_BLACKWELL" = true ] && ls wheels/flash_attn*.whl &>/dev/null 2>&1; then
        # Prebuilt sm_120 wheel — ONLY on Blackwell. A leftover Blackwell wheel in wheels/ must NOT be used
        # on Ampere/Ada (wrong torch/CUDA ABI → import fails); non-Blackwell falls to the arch wheel below.
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
    echo "[5/10] Installing FAISS-GPU (CUDA 12)..."
    # FAISS source-built wheel needs libopenblas at runtime
    if ! dpkg -s libopenblas-dev &>/dev/null 2>&1; then
        echo "Installing libopenblas-dev (FAISS runtime dependency)..."
        apt-get update -qq && apt-get install -y -qq libopenblas-dev > /dev/null 2>&1
    fi

    if [ "$IS_BLACKWELL" = true ] && ls wheels/faiss*.whl &>/dev/null 2>&1; then
        # Prebuilt sm_120/cu13 wheel — ONLY on Blackwell. A leftover here on Ampere/Ada needs
        # libcudart.so.13 (torch cu124 ships .so.12) → the exact ImportError this guard fixes.
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
        # a prior --from-wheels run may have left the sm_120 `faiss` wheel installed (broken libcudart.so.13
        # import) — uninstall it so the cu12 PyPI wheel replaces it cleanly. set +e: uninstalling an absent
        # package is non-fatal (mirrors the sam3.1 block below), and must not trip the outer set -e.
        set +e
        uv pip uninstall -y faiss-gpu faiss-gpu-cu12 faiss-cpu faiss 2>/dev/null
        set -e
        uv pip install faiss-gpu-cu12
    fi

    # 6. Install cuML (GPU UMAP) from RAPIDS PyPI
    echo ""
    echo "[6/10] Installing cuML (GPU UMAP)..."
    uv pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com

    # 7. Install wandb (experiment tracking)
    echo ""
    echo "[7/10] Installing wandb..."
    uv pip install wandb

    # 8. Install SAM 3.1 (gated model — user must accept access at hf.co/facebook/sam3.1)
    #    --no-deps: SAM3 metadata pins numpy<2.0 which breaks cuML/CuPy (requires numpy>=2.3).
    #    SAM3 works fine with numpy 2.x — the pin is overly conservative.
    #    All real deps (einops, torch, PIL, etc.) already installed via requirements_gpu.txt.
    echo ""
    echo "[8/10] Installing SAM 3.1 from source (--no-deps to preserve numpy>=2.3)..."
    if python -c "import sam3" 2>/dev/null; then
        echo "SAM 3.1 already installed"
    else
        uv pip install --no-deps git+https://github.com/facebookresearch/sam3.git
    fi

    # 9. Pre-download Grounding DINO weights for m10 Grounded-SAM (DINO + SAM 3.1).
    #    Done OOB via `hf download` (parallel, resumable) to avoid the 10s HEAD-request
    #    timeouts that AutoProcessor.from_pretrained() hits when HF CDN is flaky.
    #    Reference: errors_N_fixes.md #22 (HF timeout) + #27 (Path D text+boxes hybrid).
    echo ""
    echo "[9/10] Pre-downloading Grounding DINO (IDEA-Research/grounding-dino-base, ~1.8 GB)..."
    : "${HF_HOME:=/workspace/volume/hf_cache}"
    DINO_CACHE="${HF_HOME}/hub/models--IDEA-Research--grounding-dino-base"
    DINO_SNAP=$(ls "${DINO_CACHE}/snapshots" 2>/dev/null | head -1 || echo "")
    if [ -n "$DINO_SNAP" ] \
       && [ -f "${DINO_CACHE}/snapshots/${DINO_SNAP}/model.safetensors" ] \
       && [ -f "${DINO_CACHE}/snapshots/${DINO_SNAP}/tokenizer.json" ]; then
        echo "Grounding DINO already cached: ${DINO_CACHE}"
    else
        mkdir -p "${HF_HOME}"
        hf download IDEA-Research/grounding-dino-base
        echo "Grounding DINO cached at ${DINO_CACHE}"
    fi

    # 10. Pre-download HF SAM 3 weights — PRIMARY model loaded by m10_sam_segment.py.
    #     `facebook/sam3` hosts the full Transformers integration (config.json +
    #     model.safetensors): Sam3TrackerVideoModel (P-5a, our primary) and
    #     Sam3VideoModel (P-3a text-only probe). ~12 GB total. HF_TRANSFER (set above)
    #     parallel-streams the download in Rust, 1.5-3× faster than wget per file.
    #     Gated model: user must accept access at hf.co/facebook/sam3 once (HF_TOKEN in .env).
    echo ""
    echo "[10/11] Pre-downloading HF SAM 3 (facebook/sam3, ~12 GB, HF_TRANSFER parallel)..."
    SAM3_HF_CACHE="${HF_HOME}/hub/models--facebook--sam3"
    SAM3_HF_SNAP=$(ls "${SAM3_HF_CACHE}/snapshots" 2>/dev/null | head -1 || echo "")
    if [ -n "$SAM3_HF_SNAP" ] \
       && [ -f "${SAM3_HF_CACHE}/snapshots/${SAM3_HF_SNAP}/config.json" ] \
       && ls "${SAM3_HF_CACHE}/snapshots/${SAM3_HF_SNAP}"/*.safetensors >/dev/null 2>&1; then
        echo "HF SAM 3 already cached: ${SAM3_HF_CACHE}"
    else
        mkdir -p "${HF_HOME}"
        # xet chunk-parallel download (high-performance mode saturates bandwidth).
        HF_XET_HIGH_PERFORMANCE=1 hf download facebook/sam3 \
            --exclude "*.bin" \
            || { echo "FATAL: facebook/sam3 download failed. Accept access at hf.co/facebook/sam3 with your HF_TOKEN, then re-run."; exit 1; }
        echo "HF SAM 3 cached at ${SAM3_HF_CACHE}"
    fi

    # 11. Pre-download SAM 3.1 checkpoint — FUTURE UPGRADE PATH.
    #     iter16 (2026-05-22): SAM 3.1 (Object Multiplex, joint multi-object tracking
    #     ~7× faster on H100 + ½ VRAM, Meta release 2026-03-27, RELEASE_SAM3p1.md) is
    #     ready as a standalone checkpoint, but Meta has NOT yet shipped the HF
    #     Transformers config files for it — `facebook/sam3.1` HF repo hosts ONLY the
    #     `.pt` checkpoint (no config.json / model.safetensors with HF naming).
    #     `Sam3TrackerVideoModel.from_pretrained("facebook/sam3.1")` fails OSError
    #     "does not appear to have a file named pytorch_model.bin or model.safetensors".
    #     Until transformers ships sam3.1 integration, the pipeline runs sam3 (via
    #     surgery_base.yaml `sam_hf_model: facebook/sam3`). This step pre-caches the
    #     sam3.1 checkpoint so the upgrade is a YAML-flip away once transformers
    #     supports it — no re-download wait at upgrade time. Non-fatal if the download
    #     fails (e.g., gating not accepted) — sam3 is the active path.
    echo ""
    echo "[11/11] Pre-downloading SAM 3.1 checkpoint (facebook/sam3.1, future-upgrade path, non-fatal)..."
    SAM3P1_HF_CACHE="${HF_HOME}/hub/models--facebook--sam3.1"
    SAM3P1_HF_SNAP=$(ls "${SAM3P1_HF_CACHE}/snapshots" 2>/dev/null | head -1 || echo "")
    if [ -n "$SAM3P1_HF_SNAP" ] \
       && ls "${SAM3P1_HF_CACHE}/snapshots/${SAM3P1_HF_SNAP}"/*.pt >/dev/null 2>&1; then
        echo "SAM 3.1 already cached: ${SAM3P1_HF_CACHE}"
    else
        mkdir -p "${HF_HOME}"
        # `hf download` is non-fatal here — sam3.1 may not be HF-readable yet on every
        # account (gated). Pipeline still works on sam3 if this fails. Explicit `set +e`
        # / `set -e` to avoid the outer `set -e` killing the whole setup on a non-blocking
        # cache miss.
        set +e
        HF_XET_HIGH_PERFORMANCE=1 hf download facebook/sam3.1
        SAM3P1_RC=$?
        set -e
        if [ "$SAM3P1_RC" -eq 0 ]; then
            echo "SAM 3.1 cached at ${SAM3P1_HF_CACHE}"
        else
            echo "WARN: facebook/sam3.1 fetch failed (rc=${SAM3P1_RC}). Non-fatal —"
            echo "      pipeline runs on facebook/sam3. Accept access at hf.co/facebook/sam3.1"
            echo "      and re-run to pre-cache the future-upgrade checkpoint."
        fi
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

# Assert Grounding DINO is loadable offline (catches missing tokenizer.json etc upfront).
import os
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor
_dino_proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
dino_status = f'cached (fast tokenizer={_dino_proc.tokenizer.is_fast})'

# Assert HF SAM 3 is loadable offline (both Sam3TrackerVideoModel for P-5a + Sam3VideoModel for P-3a).
try:
    from transformers import Sam3TrackerVideoProcessor, Sam3VideoProcessor
    _sam3tv = Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
    _sam3v  = Sam3VideoProcessor.from_pretrained('facebook/sam3')
    sam3_hf_status = 'cached (TrackerVideo + Video processors loadable)'
except Exception as e:
    sam3_hf_status = f'FAIL: {e}'

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
print(f'Grounding DINO: {dino_status}')
print(f'HF SAM 3:       {sam3_hf_status}')
print('')
print('SUCCESS: All GPU components verified')
"

    # Dependency health check: fail hard on new mismatches, allowlist 4 known-OK ones.
    # See iter/iter8/errors_N_fixes.md #5 (sam3 --no-deps) + #10 (torchcodec/decord).
    echo ""
    echo "[Dependency health] uv pip check..."
    set +e
    CHECK_OUT=$(uv pip check 2>&1)
    set -e
    # Known-OK incompatibilities (grep -E patterns) — each verified non-fatal at runtime:
    #   sam3.numpy          : SAM3 pin <2 obsolete; numpy 2.x works (m04-m11 ran weeks on it)
    #   sam3.ftfy           : 6.1.1 → 6.3.1 patch, fix_text() API unchanged
    #   torch.cuda-bindings : cuML/torch patch ping-pong, ABI-stable
    #   torch.cuda-toolkit  : cuML downgrades cuda-toolkit 13.0.2→12.9.2.0; runtime imports +
    #                         GPU verify passed on 24GB (06-12) and 4x96GB (06-12) boxes
    #   decord.platform     : dormant (m04 tags.json already produced, PyAV is active decoder)
    #   torch.nvidia-*-cu12 : the STABLE torch 2.5/cu124 (non-Blackwell path) hard-pins nvidia-*-cu12
    #                         ==12.4.x; cuML upgrades them to 12.9.x — ABI-compatible within cu12.x (CUDA
    #                         minor-version compat), and the GPU verify above PASSED using them (RTX 3070,
    #                         07-01). Scoped to cu12 minors ONLY: a cu13/major bump (nvidia-*-cu13) is NOT
    #                         matched → still FATAL. The Blackwell torch-2.12 nightly doesn't pin these.
    EXPECTED='sam3.*numpy|sam3.*ftfy|torch.*cuda-bindings|torch.*cuda-toolkit|torch.*nvidia-.*-cu12|decord.*platform'
    UNEXPECTED=$(echo "$CHECK_OUT" | grep -E '^The package' | grep -vE "$EXPECTED" || echo "")
    if [ -n "$UNEXPECTED" ]; then
        echo "FATAL: uv pip check found UNEXPECTED incompatibilities:"
        echo "$UNEXPECTED"
        echo "Investigate before running pipeline. See errors_N_fixes.md for resolution patterns."
        exit 1
    fi
    EXPECTED_COUNT=$(echo "$CHECK_OUT" | grep -cE '^The package' || echo 0)
    echo "OK — ${EXPECTED_COUNT} known-OK incompatibilities (allowlist: numpy/ftfy/cuda-bindings/cuda-toolkit/decord), 0 unexpected"

    # vLLM setup SKIPPED — transformers is 2.5x faster for offline batch tagging.
    # See iter/utils/vLLM_plan_Blackwell.md for 14 root causes found + fixed.
    echo ""
    echo "vLLM: SKIPPED (transformers 2.5x faster for offline batch, see vLLM_plan_Blackwell.md)"

    echo ""
    echo "============================================"
    echo "GPU Setup Complete! (UV)"
    echo "============================================"
    echo ""
    echo "Venv:"
    echo "  source venv/bin/activate   # m00-m11 pipeline (FAISS, cuML, V-JEPA, SAM 3.1)"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_evaluate.sh --FULL            # transformers VLM tagging (m04)"
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
