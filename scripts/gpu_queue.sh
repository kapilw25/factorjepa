#!/usr/bin/env bash
# Thin wrapper for src/utils/gpu_queue.py (logic and docs there). Run inside tmux on the GPU box:
#   tmux new -d -s fj_queue "bash scripts/gpu_queue.sh <jobs.txt> <env_file or ''> 2>&1 | tee -a logs/gpu_queue_$(date +%Y%m%d_%H%M%S).log"
# Position and finished-job files sit next to the jobs file: <jobs>.pos and <jobs dir>/gpu_queue_done.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS="$1"; ENV_FILE="$2"
source venv_walkindia/bin/activate
python -u src/utils/gpu_queue.py --jobs "$JOBS" --pos "${JOBS}.pos" \
    --done "$(dirname "$JOBS")/gpu_queue_done.txt" --venv venv_walkindia --env-file "$ENV_FILE"
