#!/bin/bash
# ============================================================================
# Build + upload prebuilt wheels for Blackwell sm_120
# ============================================================================
# Usage:
#   # Full build (FA2 from source) + upload to GitHub Release
#   ./build_wheels_sm120.sh 2>&1 | tee logs/build_wheels_sm120.log
#
#   # Upload-only (wheels already in wheels/, skip FA2 build)
#   ./build_wheels_sm120.sh --upload-only 2>&1 | tee logs/build_wheels_sm120.log
#
#   # Build-only (no upload — e.g. gh CLI or token unavailable)
#   ./build_wheels_sm120.sh --build-only 2>&1 | tee logs/build_wheels_sm120.log
#
#   # Verify wheels exist (no build, no upload)
#   ./build_wheels_sm120.sh --verify-only
#
# Prerequisites:
#   - FAISS wheel already in wheels/ (built by build_faiss_sm120.sh)
#   - CUDA 12.8 toolkit installed (for FA2 build)
#   - GITHUB_TOKEN in .env (for upload — NOT hardcoded, sourced at runtime)
#
# Credentials:
#   Reads GITHUB_TOKEN from .env file (same dir as this script).
#   Format:  GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   gh CLI is auto-installed if missing. Token auth is non-interactive.
#
# Time: ~40 min (FA2 build) + seconds (upload)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/venv_walkindia}"
RELEASE_TAG="sm120-cu128-py312"
FA2_DIR="/tmp/flash-attention-build"

# ── Parse flags ──────────────────────────────────────────────────────────────
UPLOAD_ONLY=false
BUILD_ONLY=false
VERIFY_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --upload-only)  UPLOAD_ONLY=true ;;
        --build-only)   BUILD_ONLY=true ;;
        --verify-only)  VERIFY_ONLY=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage:"
            echo "  ./build_wheels_sm120.sh                # full build + upload"
            echo "  ./build_wheels_sm120.sh --upload-only   # upload existing wheels"
            echo "  ./build_wheels_sm120.sh --build-only    # build FA2, no upload"
            echo "  ./build_wheels_sm120.sh --verify-only   # check wheels exist"
            exit 1
            ;;
    esac
done

# ── Load credentials from .env ───────────────────────────────────────────────
load_github_token() {
    local ENV_FILE="${SCRIPT_DIR}/.env"
    if [ ! -f "$ENV_FILE" ]; then
        echo "FATAL: .env file not found at ${ENV_FILE}"
        echo "Add:   GITHUB_TOKEN=ghp_xxx   to .env"
        return 1
    fi
    # Extract GITHUB_TOKEN (handles spaces around =, inline comments)
    GITHUB_TOKEN=$(grep -E '^GITHUB_TOKEN=' "$ENV_FILE" | head -1 | sed 's/^GITHUB_TOKEN=//' | sed 's/#.*//' | xargs)
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "FATAL: GITHUB_TOKEN not found in ${ENV_FILE}"
        echo "Add:   GITHUB_TOKEN=ghp_xxx   to .env"
        return 1
    fi
    echo "GitHub token loaded from .env (${#GITHUB_TOKEN} chars)"
}

# ── Install gh CLI if missing ────────────────────────────────────────────────
ensure_gh_cli() {
    if command -v gh &>/dev/null; then
        echo "gh CLI: $(gh --version | head -1)"
        return 0
    fi
    echo "Installing gh CLI..."
    if [ "$(uname -s)" = "Linux" ]; then
        # Official GitHub method (works on Debian/Ubuntu without adding repo)
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
            | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
            | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        apt-get update -qq && apt-get install -y -qq gh > /dev/null 2>&1
    elif [ "$(uname -s)" = "Darwin" ]; then
        brew install gh
    fi
    if ! command -v gh &>/dev/null; then
        echo "FATAL: Failed to install gh CLI"
        return 1
    fi
    echo "gh CLI installed: $(gh --version | head -1)"
}

# ── Authenticate gh with token (non-interactive) ────────────────────────────
auth_gh() {
    echo "$GITHUB_TOKEN" | gh auth login --with-token 2>/dev/null
    echo "gh authenticated as: $(gh api user --jq .login 2>/dev/null || echo 'unknown')"
}

# ── Verify wheels ────────────────────────────────────────────────────────────
verify_wheels() {
    echo ""
    echo "=== Wheels in wheels/ ==="
    if ! ls wheels/*.whl &>/dev/null 2>&1; then
        echo "FATAL: No wheels found in wheels/"
        exit 1
    fi
    ls -lh wheels/*.whl

    FA2_COUNT=$(find wheels/ -maxdepth 1 -name "flash_attn*.whl" 2>/dev/null | wc -l)
    FAISS_COUNT=$(find wheels/ -maxdepth 1 -name "faiss*.whl" 2>/dev/null | wc -l)

    if [ "$FA2_COUNT" -eq 0 ]; then
        echo "FATAL: No FA2 wheel found in wheels/"
        exit 1
    fi
    if [ "$FAISS_COUNT" -eq 0 ]; then
        echo "WARNING: No FAISS wheel found. Run build_faiss_sm120.sh first."
    fi
    echo "FA2 wheels: ${FA2_COUNT}, FAISS wheels: ${FAISS_COUNT}"
}

# ── Upload wheels to GitHub Release ──────────────────────────────────────────
upload_wheels() {
    load_github_token
    ensure_gh_cli
    auth_gh

    local REPO_SLUG
    REPO_SLUG=$(git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||' | sed 's|\.git$||')

    echo ""
    echo "=== Uploading to GitHub Release: ${RELEASE_TAG} ==="

    # Build release notes from actual wheel filenames
    RELEASE_NOTES="$(cat <<EOF
Pin: torch==2.12.0.dev20260228+cu128, Python 3.12.12
$(ls wheels/*.whl 2>/dev/null | while read w; do echo "- $(basename "$w")"; done)
Built for sm_120 Blackwell, CUDA 12.8.
Rebuild if PyTorch version changes.
EOF
)"

    # Check if release already exists
    if gh release view "${RELEASE_TAG}" &>/dev/null 2>&1; then
        echo "Release '${RELEASE_TAG}' already exists. Uploading assets (overwrite if exists)..."
        gh release upload "${RELEASE_TAG}" wheels/*.whl --clobber
        # Update release notes to match current wheel versions
        gh release edit "${RELEASE_TAG}" --notes "$RELEASE_NOTES"
        echo "Release notes updated."
    else
        echo "Creating release '${RELEASE_TAG}'..."
        gh release create "${RELEASE_TAG}" wheels/*.whl \
            --title "Prebuilt wheels: sm_120 + CUDA 12.8 + Python 3.12" \
            --notes "$RELEASE_NOTES"
    fi

    echo ""
    echo "Release: https://github.com/${REPO_SLUG}/releases/tag/${RELEASE_TAG}"
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

# --verify-only: just check wheels exist
if [ "$VERIFY_ONLY" = true ]; then
    verify_wheels
    echo ""
    echo "=== Verify complete ==="
    exit 0
fi

# --upload-only: skip build, go straight to verify + upload
if [ "$UPLOAD_ONLY" = true ]; then
    verify_wheels
    upload_wheels
    echo ""
    echo "=== Upload complete ==="
    echo "On new machines, install with:"
    echo "  ./setup_env_uv.sh --gpu --from-wheels"
    exit 0
fi

# ── 0. Activate venv (needed for FA2 build) ─────────────────────────────────
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "FATAL: venv not found at ${VENV_DIR}"
    exit 1
fi
source "${VENV_DIR}/bin/activate"
export VIRTUAL_ENV="${VENV_DIR}"

# ── 1. Set CUDA 12.8 (must match PyTorch cu128) ─────────────────────────────
if [ ! -f /usr/local/cuda-12.8/bin/nvcc ]; then
    echo "FATAL: CUDA 12.8 toolkit not found at /usr/local/cuda-12.8"
    echo "Install: apt-get install cuda-toolkit-12-8"
    exit 1
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
echo "nvcc: $(nvcc --version | grep release)"

# ── 2. Build FA2 wheel (~38 min) ────────────────────────────────────────────
mkdir -p wheels logs

if ls wheels/flash_attn*.whl &>/dev/null 2>&1; then
    echo "FA2 wheel already exists: $(ls wheels/flash_attn*.whl)"
    echo "Delete it to rebuild: rm wheels/flash_attn*.whl"
else
    echo "=== Building Flash-Attention 2 wheel for sm_120 ==="
    uv pip install pip 2>/dev/null || true
    rm -rf "$FA2_DIR"
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FA2_DIR"
    cd "$FA2_DIR" && git submodule update --init --recursive && cd -

    START_TIME=$(date +%s)
    FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 NVCC_THREADS=1 \
        pip wheel "$FA2_DIR" --no-build-isolation --no-deps --wheel-dir wheels/
    END_TIME=$(date +%s)
    echo "FA2 build time: $(( (END_TIME - START_TIME) / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"

    rm -rf "$FA2_DIR"
fi

# ── 3. Verify both wheels ───────────────────────────────────────────────────
verify_wheels

# ── 4. Upload (unless --build-only) ─────────────────────────────────────────
if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "=== Build complete (--build-only, skipping upload) ==="
    echo "To upload later: ./build_wheels_sm120.sh --upload-only"
    exit 0
fi

upload_wheels

echo ""
echo "=== Done ==="
echo "On new machines, install with:"
echo "  ./setup_env_uv.sh --gpu --from-wheels"
