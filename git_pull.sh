#!/bin/bash
: '
=============================================================================
Git Pull + HF Download — sync code + compute outputs from remote
=============================================================================

Usage:

On Linux 
    ./git_pull.sh 2>&1 | tee logs/git_pull.log # sync code + download outputs + data from HF

On Mac (preserves .gitignored files like src/data/videos/, src/data/clips/):
    ./git_pull.sh --code-only 2>&1 | tee logs/git_pull.log   # safe — git clean -fd skips .gitignored files

=============================================================================
'

set -euo pipefail

CODE_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --code-only) CODE_ONLY=true ;;
        *) echo "Unknown arg: $arg"; echo "Usage: ./git_pull.sh [--code-only]"; exit 1 ;;
    esac
done

echo "=== Git Pull ==="

# Hard reset to remote (exact mirror, no stale files)
# git clean -fd removes untracked files but PRESERVES .gitignored files (src/data/videos/, clips/)
git fetch origin
git reset --hard origin/main
git clean -fd
echo "Synced to origin/main (exact mirror)"

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
echo "=== HF Download: Data (val_1k, subset_10k) ==="
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/hf_download_data.log

echo ""
echo "=== Sync Complete ==="
echo "  Code: $(git log --oneline -1)"
echo "  Outputs: outputs/ (check logs/hf_download_outputs.log)"
echo "  Data: data/ (check logs/hf_download_data.log)"
