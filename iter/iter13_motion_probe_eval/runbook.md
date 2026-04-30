# 🚀 iter13 m06d Runbook — terminal commands only

> All logic lives in `scripts/run_m06d.sh` (9-stage thin orchestrator). This runbook = bootstrap, pre-flight `ls`, launch, post-flight `cat`.

---

## -1 · Fresh GPU bootstrap (skip if `venv_walkindia/` already exists)

```bash
cd /workspace
git clone https://github.com/kapilw25/factorjepa
cd factorjepa

# Env: torch nightly + FA2 + FAISS + cuML + SAM 3.1 + Grounding DINO + V-JEPA 2.1 ckpt
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
```

> ⚠️ **Torch-nightly rotation gotcha** — if `[1/9] Installing PyTorch` fails with `× No solution found when resolving dependencies`, the pinned nightly was rotated off `download.pytorch.org/whl/nightly/cu128`. Fix:
> ```bash
> source venv_walkindia/bin/activate                       # use whatever torch was already partially-installed
> uv pip install --dry-run --pre torch \
>     --index-url https://download.pytorch.org/whl/nightly/cu128 \
>     2>&1 | grep -oE 'torch==[0-9.dev+a-z]+' | head -1
> # Edit setup_env_uv.sh:27 → TORCH_VERSION="<dev2026MMDD from above>"
> # IMPORTANT: torchvision lags torch by 1-2 days — pick a torch nightly that has a paired torchvision
> # (logs/setup_env_gpu_v2.log:414 case study).  Re-run setup.
> ```

```bash
source venv_walkindia/bin/activate

# Data: 10K Indian clips (10 TARs) + tags.json + manifest.json (~7 GB total) — pulls from HF
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_data.log

# Quick sanity: required inputs all present + correct sizes
ls -lh data/eval_10k.json \
       data/eval_10k_local/manifest.json \
       data/eval_10k_local/tags.json \
       data/eval_10k_local/subset-00000.tar \
       checkpoints/vjepa2_1_vitG_384.pt
```

> 🟡 **3-class default**: `--enable-monument-class` is unviable on this 10K subset (only 49 monument clips, fails the ≥5/test-split BCa floor). Default 3-class (walking/driving/drone) gives ≥200 test clips per class — robust statistics. See `plan_training.md:74-79`.

---

## 0 · Pre-flight (verify INPUTS)

```bash
cd /workspace/factorjepa
source venv_walkindia/bin/activate

# (a) Inputs exist
ls -lh data/eval_10k.json \
       data/eval_10k_local/manifest.json \
       data/eval_10k_local/tags.json \
       data/eval_10k_local/subset-00000.tar \
       checkpoints/vjepa2_1_vitG_384.pt \
       deps/vjepa2/src/models/attentive_pooler.py \
       scripts/run_m06d.sh

# (b) Imports + AttentiveClassifier shim wired
python -c "
import sys; sys.path.insert(0, 'src')
from utils.action_labels import load_subset_with_labels
from utils.frozen_features import ENCODERS, load_vjepa_2_1_frozen, load_dinov2_frozen, extract_features_for_keys
from utils.vjepa2_imports import get_attentive_classifier
import m06d_action_probe, m06d_motion_cos, m06d_future_mse
print('✓ all m06d imports + utils.frozen_features OK')
"

# (c) Bash syntax + executable
bash -n scripts/run_m06d.sh && [ -x scripts/run_m06d.sh ] && echo '✓ run_m06d.sh syntax OK + executable'
```

If any line fails → fix before launching. Otherwise →

---

## 1 · SANITY (Stage 1 only — labels, ~1 min CPU)

```bash
SKIP_STAGES="2,3,4,5,6,7,8,9" CACHE_POLICY_ALL=2 \
    OUTPUT_ACTION=outputs/sanity/m06d_action_probe \
    ./scripts/run_m06d.sh 2>&1 | tee logs/run_m06d_sanity.log

# Verify Stage 1 output
cat outputs/sanity/m06d_action_probe/class_counts.json
# Expect: {"walking":{"train":3894,...}, "driving":{...}, "drone":{...}} — 3 classes, all >5 per split
```

---

## 2 · FULL launch (~2.5 GPU-h end-to-end)

```bash
tmux new -s m06d
CACHE_POLICY_ALL=1 ./scripts/run_m06d.sh 2>&1 | tee logs/run_m06d_v1.log
# Ctrl-B d to detach · tmux attach -t m06d
```

> `CACHE_POLICY_ALL=1` reuses caches between re-runs (fast resume). Use `=2` for fresh recompute.
> Resume from a specific stage after a crash: `SKIP_STAGES="1,2,3" ./scripts/run_m06d.sh ...`.

---

## 3 · Post-flight (verify OUTPUTS, ✅ when each present)

```bash
# Stage 1 — labels
ls -lh outputs/full/m06d_action_probe/action_labels.json \
       outputs/full/m06d_action_probe/class_counts.json

# Stage 2 — features × 2 encoders (shape: (N, n_tokens, D) fp32)
python -c "
import numpy as np
for enc in ['vjepa_2_1_frozen', 'dinov2']:
    for sp in ['train', 'val', 'test']:
        a = np.load(f'outputs/full/m06d_action_probe/{enc}/features_{sp}.npy', mmap_mode='r')
        print(f'  {enc:18s} {sp:5s}: shape={a.shape} dtype={a.dtype}')
"

# Stage 3 — probe ckpts + test metrics × 2 encoders
for enc in vjepa_2_1_frozen dinov2; do
    ls -lh outputs/full/m06d_action_probe/$enc/probe.pt
    cat outputs/full/m06d_action_probe/$enc/test_metrics.json | python -m json.tool
done

# 🔥 Stage 4 — P1 GATE verdict
cat outputs/full/m06d_action_probe/m06d_paired_delta.json | python -m json.tool
# PASS criteria: ci_lo_pp > 0  AND  delta_pp > 0  AND  p_value < 0.05  AND  gate_pass = true

# Stage 5–7 — motion_cos
ls -lh outputs/full/m06d_motion_cos/{vjepa_2_1_frozen,dinov2}/pooled_features_test.npy \
       outputs/full/m06d_motion_cos/{vjepa_2_1_frozen,dinov2}/per_clip_motion_cos.npy
cat outputs/full/m06d_motion_cos/m06d_motion_cos_paired.json | python -m json.tool

# Stage 8–9 — future_mse (V-JEPA only)
ls -lh outputs/full/m06d_future_mse/vjepa_2_1_frozen/per_clip_mse.npy
cat outputs/full/m06d_future_mse/m06d_future_mse_per_variant.json | python -m json.tool
```

---

## 4 · One-shot gate verdict

```bash
python -c "
import json
g = json.load(open('outputs/full/m06d_action_probe/m06d_paired_delta.json'))
m = json.load(open('outputs/full/m06d_motion_cos/m06d_motion_cos_paired.json'))
f = json.load(open('outputs/full/m06d_future_mse/m06d_future_mse_per_variant.json'))
print()
print(f'🥇 P1 ACTION-PROBE GATE  Δ={g[\"delta_pp\"]:+.2f} pp  CI [{g[\"ci_lo_pp\"]:+.2f}, {g[\"ci_hi_pp\"]:+.2f}]  p={g[\"p_value\"]:.4f}  → {\"✅ PASS\" if g[\"gate_pass\"] else \"❌ FAIL\"}')
print(f'   motion_cos Δ={m[\"delta_mean\"]:+.4f}  CI_half=±{m[\"delta_ci_half\"]:.4f}  p={m[\"p_value\"]:.4f}  → {\"✅\" if m[\"gate_pass\"] else \"🟡\"}')
print(f'   future_mse vjepa_frozen mse_mean={f[\"by_variant\"][\"vjepa_2_1_frozen\"][\"mse_mean\"]:.4f}   (DINOv2={f[\"by_variant\"][\"dinov2\"]})')
"
```

---

## 5 · Cleanup / re-run

```bash
# Re-run stage N only (e.g., re-train probe with different lr)
SKIP_STAGES=\"1,2,5,6,7,8,9\" CACHE_POLICY_ALL=2 ./scripts/run_m06d.sh

# Single-encoder debug (V-JEPA only, all stages)
ENCODERS=\"vjepa_2_1_frozen\" SKIP_STAGES=\"4,8,9\" ./scripts/run_m06d.sh
```

---

## 🚦 Decision matrix on P1 gate output

| `m06d_paired_delta.json` | Reading | Next |
|:--|:--|:--|
| `gate_pass: true`, `p_value < 0.05` | V-JEPA frozen > DINOv2 on Indian motion-centric probe | proceed to **Priority 2** (`vjepa_explora` vs `vjepa_frozen`) |
| `gate_pass: false`, `delta_pp > 0`, CI overlap | underpowered or noise floor | re-run with `NUM_FRAMES=64` (~4× slower); add `--enable-monument-class` for harder 4-class task |
| `delta_pp ≤ 0` | DINOv2 ties or beats V-JEPA on our domain | 🛑 cancel P2/P3, diff `m05` vs `vjepa2_demo.ipynb`, open critical bug |
