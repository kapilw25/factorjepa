#!/usr/bin/env bash
# iter10 Option C: curate + download a 10K eval pool disjoint from training + val.
# Enables paired-bootstrap evaluation at N=10K where CI_half ~0.54 pp (vs ±2.4 pp at N=500).
#
# USAGE (foreground, ~25 min CPU + network):
#   ./scripts/prep_eval_10k.sh 2>&1 | tee logs/prep_eval_10k_v2.log
#
# USAGE (detachable — recommended while v15b/v15c training is live):
#   tmux new -s eval10k './scripts/prep_eval_10k.sh'
#   # Ctrl-B d to detach · tmux attach -t eval10k to reattach
#
# OUTPUTS:
#   data/eval_10k.json                      (10K disjoint clip_keys, seed=99)
#   data/eval_10k_local/*.tar + manifest    (~10 GB TARs from HF)
#   data/eval_10k_local/tags.json           (symlink to full_local/tags.json, 100% coverage)
set -euo pipefail

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

# ── Step 1: m00c sample 10K disjoint clip_keys (seed=99 ≠ training seed=42) ──
stamp "m00c sample 10K disjoint (seed=99, exclude subset_10k + val_1k)"
python -u src/m00c_sample_subset.py --n 10000 --seed 99 \
    --exclude data/subset_10k.json,data/val_1k.json \
    --output data/eval_10k.json \
    2>&1 | tee logs/m00c_eval_10k.log

# ── Step 2: m00d download MP4 TAR shards (CPU-only, 8 parallel HF workers) ──
stamp "m00d download eval_10k MP4s from HF (~25 min)"
python -u src/m00d_download_subset.py --FULL --subset data/eval_10k.json --no-wandb \
    2>&1 | tee logs/m00d_eval_10k.log

# ── Step 3: symlink tags.json (full_local/tags.json has 100% coverage of 115K) ──
stamp "symlink tags.json (avoids m04 VLM tagging — re-use cached labels)"
ln -sf /workspace/factorjepa/data/full_local/tags.json \
    /workspace/factorjepa/data/eval_10k_local/tags.json

# ── Step 4: verify ────────────────────────────────────────────────────────
stamp "verify eval_10k artifacts"
ls -la /workspace/factorjepa/data/eval_10k_local/tags.json
du -sh /workspace/factorjepa/data/eval_10k_local/
python -c "
import json
d = json.load(open('data/eval_10k.json'))
print(f'eval_10k: {d[\"n\"]} clips · seed={d[\"seed\"]} · exclude_cfg={d.get(\"exclude\", \"see log\")}')
"

stamp "✅ prep_eval_10k complete — ready for m05 + m06 + m08b paired bootstrap"
