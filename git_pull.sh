#!/bin/bash
: '
=============================================================================
Git Pull + HF Download — sync code + compute outputs from remote
=============================================================================

Usage:
    ./git_pull.sh              # pull code + download outputs from HF
    ./git_pull.sh --code-only  # pull code only (no HF download)
    ./git_pull.sh --force      # discard ALL local changes + download outputs

Examples:
    # GPU2 initial setup (after git clone):
    ./git_pull.sh

    # GPU1 wants Ch10 results from GPU2:
    ./git_pull.sh

    # CPU wants both Ch9 + Ch10 for m08b compare:
    ./git_pull.sh

    # Nuclear option (throws away ALL local changes):
    ./git_pull.sh --force

=============================================================================
'

set -euo pipefail

FORCE=false
CODE_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --code-only) CODE_ONLY=true ;;
        *) echo "Unknown arg: $arg"; echo "Usage: ./git_pull.sh [--force] [--code-only]"; exit 1 ;;
    esac
done

echo "=== Git Pull ==="

if [[ "$FORCE" == true ]]; then
    echo "WARNING: --force will discard ALL local uncommitted changes."
    read -p "Are you sure? [y/N]: " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
    git fetch origin
    git reset --hard origin/main
    echo "Reset to origin/main (all local changes discarded)"
else
    git pull --ff-only origin main
    echo "Code updated (fast-forward only)"
fi

if [[ "$CODE_ONLY" == true ]]; then
    echo "Done (--code-only, skipping HF download)"
    exit 0
fi

echo ""
echo "=== HF Download: Compute Outputs ==="

# Activate venv for HF SDK
if [[ -d "venv_walkindia" ]]; then
    source venv_walkindia/bin/activate
fi

mkdir -p logs

# Download all outputs (HF dedup = only downloads new/changed files)
python -u src/utils/hf_outputs.py download outputs 2>&1 | tee logs/hf_download_outputs.log

echo ""
echo "=== Sync Complete ==="
echo "  Code: $(git log --oneline -1)"
echo "  Outputs: outputs/ (check logs/hf_download_outputs.log)"
