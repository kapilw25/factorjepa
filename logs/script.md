# HF re-upload (push new layout)

# Pushes subset_10k_local/{subset_10k.json} + val_1k_local/{val_1k,sanity_100_dense,
# val_500,test_500}.json to HF under their new paths.
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload-data \
2>&1 | tee logs/upload_data_phase_B_$(date +%Y%m%d_%H%M%S).log
```

# OPTIONAL — drop the OLD root-path orphans on HF (one-time cleanup):
```bash
python -u <<'PYEOF'
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
api.delete_files(
    repo_id='anonymousML123/factorjepa-outputs',
    repo_type='dataset',
    delete_patterns=[
        'data/eval_10k.json',
        'data/eval_10k_sanity.json',
        'data/eval_10k_poc.json',
        'data/eval_10k_train_split.json',
        'data/eval_10k_val_split.json',
        'data/eval_10k_test_split.json',
        'data/subset_10k.json',
        'data/val_1k.json',
        'data/sanity_100_dense.json',
        'data/val_500.json',
        'data/test_500.json',
    ],
    commit_message='iter15 Phase A+B: drop legacy root-path JSONs (now under data/*_local/ subdirs)',
)
print('Cleaned 11 orphan files from HF root')
PYEOF
```

# Verify on HF after cleanup
```bash
python -u <<'PYEOF'
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
remote = api.list_repo_tree(
    'anonymousML123/factorjepa-outputs', path_in_repo='data',
    repo_type='dataset', recursive=False,
)
for item in remote:
    print(f"  {item.path}")
PYEOF
```