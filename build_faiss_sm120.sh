#!/usr/bin/env bash
# Build FAISS-GPU from source for Blackwell (sm_120).
# RTX PRO 4000 Blackwell needs sm_120 kernels; pip faiss-gpu-cu12 only ships sm_70+sm_80.
#
# Usage:
#   chmod +x build_faiss_sm120.sh
#   ./build_faiss_sm120.sh           # full build from scratch
#   ./build_faiss_sm120.sh --install # skip build, just install from existing build artifacts
#
# Prerequisites: CUDA toolkit, cmake, python venv activated
# Time: ~10 min on 96-core machine

set -euo pipefail

VENV_DIR="${VENV_DIR:-/workspace/LLM_asAgent_3D_SR/venv_walkindia}"
FAISS_SRC="/tmp/faiss_build"
CUDA_ARCH="120"
JOBS="$(nproc)"
INSTALL_ONLY=false

if [[ "${1:-}" == "--install" ]]; then
    INSTALL_ONLY=true
fi

echo "=== FAISS-GPU Build for sm_${CUDA_ARCH} (Blackwell) ==="
echo "VENV:        ${VENV_DIR}"
echo "Source dir:   ${FAISS_SRC}"
echo "CUDA arch:    sm_${CUDA_ARCH}"
echo "Install only: ${INSTALL_ONLY}"
echo ""

# ── 1. Activate venv ─────────────────────────────────────────────────────
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "FATAL: venv not found at ${VENV_DIR}"
    exit 1
fi
source "${VENV_DIR}/bin/activate"
PYTHON="${VENV_DIR}/bin/python"
PIP="uv pip"
export VIRTUAL_ENV="${VENV_DIR}"
echo "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "Pip:    ${PIP}"

# ── 2. Remove old faiss packages from venv ────────────────────────────────
echo ""
echo "=== Removing old faiss packages ==="
${PIP} uninstall -y faiss-gpu faiss-gpu-cu12 faiss-gpu-cu11 faiss-cpu faiss 2>/dev/null || true

${PYTHON} -c "import faiss" 2>/dev/null && {
    echo "WARNING: faiss still importable after uninstall — removing manually..."
    FAISS_PKG_DIR=$(${PYTHON} -c "import faiss, os; print(os.path.dirname(faiss.__file__))")
    rm -rf "${FAISS_PKG_DIR}"
    rm -rf "${FAISS_PKG_DIR}/../faiss"* 2>/dev/null || true
    echo "Removed: ${FAISS_PKG_DIR}"
} || echo "faiss cleanly uninstalled"

# ── Skip to install if --install flag ─────────────────────────────────────
if [ "$INSTALL_ONLY" = true ]; then
    if [ ! -f "${FAISS_SRC}/build/faiss/python/setup.py" ]; then
        echo "FATAL: No build artifacts found at ${FAISS_SRC}/build/faiss/python/"
        echo "Run without --install first to build from source."
        exit 1
    fi
    echo "Skipping build, installing from existing artifacts..."
else

# ── 3. Install build deps ────────────────────────────────────────────────
echo ""
echo "=== Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq swig libopenblas-dev > /dev/null 2>&1 || {
    echo "WARNING: apt-get failed, trying pip install swig..."
    pip install swig
    echo "WARNING: libopenblas-dev must be installed via apt for FAISS build"
}
pip install -q numpy cmake

echo "swig:  $(swig -version 2>&1 | grep SWIG | head -1)"
echo "cmake: $(cmake --version | head -1)"
echo "nvcc:  $(nvcc --version 2>&1 | tail -1)"

# ── 4. Clone FAISS source (skip if already exists) ────────────────────────
echo ""
if [ -d "${FAISS_SRC}/.git" ]; then
    echo "=== FAISS source already exists, reusing ==="
    cd "${FAISS_SRC}"
else
    echo "=== Cloning FAISS source ==="
    rm -rf "${FAISS_SRC}"
    git clone --depth 1 https://github.com/facebookresearch/faiss.git "${FAISS_SRC}"
    cd "${FAISS_SRC}"
fi
echo "FAISS source: $(git log --oneline -1)"

# ── 5. CMake configure ───────────────────────────────────────────────────
echo ""
echo "=== CMake configure (sm_${CUDA_ARCH}) ==="
cmake -B build \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DFAISS_ENABLE_C_API=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DPython_EXECUTABLE="${PYTHON}" \
    -DCMAKE_BUILD_TYPE=Release

# ── 6. Build ─────────────────────────────────────────────────────────────
echo ""
echo "=== Building FAISS (${JOBS} jobs) ==="
START_TIME=$(date +%s)
cmake --build build --config Release -j "${JOBS}"
END_TIME=$(date +%s)
echo "Build time: $(( (END_TIME - START_TIME) / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"

fi  # end of build block

# ── 7. Build wheel + install Python bindings into venv ────────────────────
WHEELS_DIR="$(cd "$(dirname "$0")" && pwd)/wheels"
mkdir -p "${WHEELS_DIR}"
echo ""
echo "=== Building wheel + installing Python bindings ==="
cd "${FAISS_SRC}/build/faiss/python"
# Build wheel for caching (saved to wheels/ for GitHub release upload)
${PIP} install pip 2>/dev/null || true
pip wheel . --no-deps --wheel-dir "${WHEELS_DIR}/" 2>/dev/null && {
    echo "Wheel saved: $(ls "${WHEELS_DIR}"/faiss*.whl 2>/dev/null | head -1)"
} || echo "Wheel export skipped (non-fatal, installing directly)"
# Install into venv
${PIP} install .
echo "Installed faiss to: $(${PIP} show faiss 2>/dev/null | grep Location || echo 'unknown')"

# ── 8. Verify ────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
cd /tmp
${PYTHON} -c "
import faiss
print(f'faiss version: {faiss.__version__}')
print(f'GPU count:     {faiss.get_num_gpus()}')

# Quick GPU smoke test
d = 64
n = 100
import numpy as np
xb = np.random.random((n, d)).astype('float32')
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatL2(d)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index_gpu.add(xb)
D, I = index_gpu.search(xb[:5], 4)
print(f'GPU search:    OK (searched {n} vectors, dim={d})')
print(f'Top-1 self:    {(I[:, 0] == np.arange(5)).all()} (expect True)')
print()
print('=== FAISS-GPU sm_${CUDA_ARCH} BUILD SUCCESSFUL ===')
"

# ── 9. Cleanup ───────────────────────────────────────────────────────────
echo ""
echo "Build artifacts kept at ${FAISS_SRC} (rerun with --install to reinstall)"
echo "To delete: rm -rf ${FAISS_SRC}"
echo "Done."
