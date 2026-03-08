#!/usr/bin/env bash
# Build FAISS-GPU from source for Blackwell (sm_120).
# RTX PRO 4000 Blackwell needs sm_120 kernels; pip faiss-gpu-cu12 only ships sm_70+sm_80.
#
# Usage:
#   chmod +x build_faiss_sm120.sh
#   ./build_faiss_sm120.sh 2>&1 | tee logs/build_faiss_sm120.log           # full build from scratch
#   ./build_faiss_sm120.sh --install 2>&1 | tee logs/install_faiss_sm120.log # skip build, just install from existing build artifacts
#
# Prerequisites: CUDA toolkit, cmake, python venv activated
# Time: ~10 min on 96-core machine

set -euo pipefail

# Resolve paths BEFORE any cd (BASH_SOURCE is relative, must resolve now)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheels"

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
# WHEELS_DIR was resolved at top of script (before any cd)
mkdir -p "${WHEELS_DIR}"
echo ""
echo "=== Building platform wheel + installing Python bindings ==="
echo "Wheels dir: ${WHEELS_DIR}"

FAISS_PY_DIR="${FAISS_SRC}/build/faiss/python"
cd "${FAISS_PY_DIR}"

# Collect all .so files that must be in the wheel
echo "Locating native .so files from build tree..."
SO_MANIFEST="/tmp/_faiss_so_manifest"
rm -f "${SO_MANIFEST}"

for so_file in \
    "${FAISS_SRC}/build/faiss/libfaiss.so" \
    "${FAISS_SRC}/build/faiss/libfaiss_avx2.so" \
    "${FAISS_SRC}/build/faiss/gpu/libfaiss_gpu.so" \
; do
    [ -f "$so_file" ] && echo "$so_file" >> "${SO_MANIFEST}"
done

# SWIG extensions (in python build dir, may have versioned names)
for so_file in "${FAISS_PY_DIR}"/_swigfaiss*.so "${FAISS_PY_DIR}"/_swigfaiss_gpu*.so; do
    [ -f "$so_file" ] && echo "$so_file" >> "${SO_MANIFEST}"
done

# Callbacks library
[ -f "${FAISS_PY_DIR}/libfaiss_python_callbacks.so" ] && \
    echo "${FAISS_PY_DIR}/libfaiss_python_callbacks.so" >> "${SO_MANIFEST}"

SO_COUNT=$(wc -l < "${SO_MANIFEST}" 2>/dev/null || echo 0)
echo "Found ${SO_COUNT} .so file(s) to include:"
cat "${SO_MANIFEST}" | while read f; do ls -lh "$f"; done

if [ "${SO_COUNT}" -eq 0 ]; then
    echo "FATAL: No .so files found in build tree!"
    find "${FAISS_SRC}/build" -name "*.so" -type f
    exit 1
fi

# Remove any previous wheel
rm -f "${WHEELS_DIR}"/faiss*.whl

# Step 1: Let pip build its base wheel (Python files + whatever its build
# system includes — typically misses libfaiss.so)
${PIP} install pip wheel setuptools 2>/dev/null || true
echo ""
echo "Building base wheel..."
pip wheel . --no-deps --no-build-isolation --wheel-dir "${WHEELS_DIR}/"
WHEEL_FILE=$(ls "${WHEELS_DIR}"/faiss*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "WARNING: pip wheel failed. Installing directly into venv..."
    ${PIP} install --no-build-isolation .
    echo "Installed faiss to: $(${PIP} show faiss 2>/dev/null | grep Location || echo 'unknown')"
else
    # Step 2: Inject ALL required .so files into the wheel zip and fix
    # the platform tag (both internal WHEEL metadata and filename).
    # This avoids fighting pip's package_data/MANIFEST.in — we directly
    # control what goes into the zip.
    echo ""
    echo "Injecting .so files into wheel and fixing platform tag..."
    FAISS_SO_MANIFEST="${SO_MANIFEST}" FAISS_WHEEL="${WHEEL_FILE}" \
    ${PYTHON} << 'INJECT_SCRIPT'
import zipfile, os, hashlib, base64

wheel_path = os.environ['FAISS_WHEEL']
manifest_path = os.environ['FAISS_SO_MANIFEST']

with open(manifest_path) as f:
    so_files = [line.strip() for line in f if line.strip()]

print(f"Wheel: {wheel_path}")
print(f"Processing {len(so_files)} .so file(s)")

with zipfile.ZipFile(wheel_path, 'r') as zin:
    existing_names = set(zin.namelist())

    # Find .dist-info directory
    dist_info = [n.split('/')[0] for n in existing_names if '.dist-info/' in n][0]
    wheel_meta_path = f"{dist_info}/WHEEL"
    record_path = f"{dist_info}/RECORD"

    tmp_path = wheel_path + '.tmp'
    with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        # Copy existing files (except WHEEL and RECORD — we rewrite those)
        for name in existing_names:
            if name in (wheel_meta_path, record_path):
                continue
            zout.writestr(name, zin.read(name))

        # Inject .so files under faiss/ in the wheel
        for so_path in so_files:
            so_basename = os.path.basename(so_path)
            wheel_entry = f"faiss/{so_basename}"
            if wheel_entry in existing_names:
                print(f"  Already in wheel: {wheel_entry}")
                continue
            with open(so_path, 'rb') as f:
                data = f.read()
            zout.writestr(wheel_entry, data)
            print(f"  Injected: {wheel_entry} ({len(data) / 1024 / 1024:.1f} MB)")

        # Fix WHEEL metadata: correct platform tag
        wheel_meta = zin.read(wheel_meta_path).decode()
        wheel_meta = wheel_meta.replace(
            'Tag: py3-none-any', 'Tag: cp312-cp312-linux_x86_64')
        zout.writestr(wheel_meta_path, wheel_meta)

        # Rebuild RECORD with correct hashes
        record_entries = []
        for name in zout.namelist():
            if name == record_path:
                continue
            data = zout.read(name)
            digest = hashlib.sha256(data).digest()
            h = "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
            record_entries.append(f"{name},{h},{len(data)}")
        record_entries.append(f"{record_path},,")
        zout.writestr(record_path, '\n'.join(record_entries) + '\n')

os.replace(tmp_path, wheel_path)

# Fix filename to match the platform tag
dirname = os.path.dirname(wheel_path)
basename = os.path.basename(wheel_path)
if 'none-any' in basename:
    new_basename = basename.replace('py3-none-any', 'cp312-cp312-linux_x86_64')
    new_path = os.path.join(dirname, new_basename)
    os.rename(wheel_path, new_path)
    wheel_path = new_path
    print(f"Renamed: {new_basename}")

# Write final path for shell to read back
with open('/tmp/_faiss_wheel_path', 'w') as f:
    f.write(wheel_path)

print("Done.")
INJECT_SCRIPT

    WHEEL_FILE=$(cat /tmp/_faiss_wheel_path 2>/dev/null || echo "")
    rm -f /tmp/_faiss_wheel_path "${SO_MANIFEST}"

    if [ -n "$WHEEL_FILE" ] && [ -f "$WHEEL_FILE" ]; then
        echo ""
        echo "Wheel saved: ${WHEEL_FILE}"
        echo ""
        echo "=== Wheel contents (.so files) ==="
        ${PYTHON} -c "
import zipfile, os
z = zipfile.ZipFile('${WHEEL_FILE}')
so_files = [n for n in z.namelist() if n.endswith('.so')]
for f in so_files:
    info = z.getinfo(f)
    print(f'  {f} ({info.file_size / 1024 / 1024:.1f} MB)')
whl_mb = os.path.getsize('${WHEEL_FILE}') / 1024 / 1024
print(f'Total: {len(so_files)} .so file(s), wheel size: {whl_mb:.1f} MB')
if not any('libfaiss.so' in f for f in so_files):
    print('FATAL: libfaiss.so missing from wheel — it will not work!')
    exit(1)
if not any('_swigfaiss' in f for f in so_files):
    print('FATAL: _swigfaiss.so missing from wheel — it will not work!')
    exit(1)
print('OK: wheel contains required native libraries')
"
        if [ $? -ne 0 ]; then
            echo "Deleting broken wheel."
            rm -f "${WHEEL_FILE}"
        fi
    else
        echo "WARNING: Wheel post-processing failed."
    fi
fi

# Install into venv (direct install from build tree — always works)
${PIP} install --no-build-isolation .
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
