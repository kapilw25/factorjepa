# Ch10 Coding Plan: `m09_pretrain.py` + `run_pretrain.sh`

> Implements proposal Sec 10.1–10.7. Ref: `plan_training.md`, `FactorJEPA.md`

---

## 0. Architecture Decision: How to Get Training-Ready V-JEPA

### Problem

Our m05 pipeline uses HuggingFace `AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384")` for inference. This gives us the encoder with `skip_predictor=True`, but JEPA *training* requires:

1. **Mask-aware encoding** — student encoder processes only VISIBLE tokens (not full forward pass)
2. **Predictor network** — narrow transformer mapping student features → teacher feature space
3. **Spatiotemporal mask sampling** — block masking on the token grid
4. **EMA teacher update** — parameter-level exponential moving average

The HF model's `forward()` processes ALL tokens (designed for inference, not training). No public `VJepa2ForPreTraining` class exists in `transformers`.

### Decision: Install `facebookresearch/vjepa2` as training dependency

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **A: Use vjepa2 repo directly** | Battle-tested training code, correct mask-aware encoding, predictor + mask utilities included | Additional dependency, weight format differs from HF | **CHOSEN** |
| B: HF model + full forward (no masking) | No new dependency | 15-30% compute overhead (processes masked tokens too), predictor not available, must reimplement mask handling | Rejected — wastes GPU at scale |
| C: Vendor ~1500 lines of vjepa2 | No dependency | Maintenance burden, high bug risk reimplementing ViT + predictor + masks | Rejected — over-engineering |

**Installation** (add to `requirements_gpu.txt` + `setup_env_uv.sh`):
```bash
# In requirements_gpu.txt:
vjepa2 @ git+https://github.com/facebookresearch/vjepa2.git

# OR clone + install editable (preferred for debugging):
git clone https://github.com/facebookresearch/vjepa2.git deps/vjepa2
pip install -e deps/vjepa2
```

**What we import from vjepa2:**
```python
from vjepa2.src.models.vision_transformer import vit_giant       # Encoder factory
from vjepa2.src.models.predictor import VisionTransformerPredictor # Predictor
from vjepa2.src.masks.multiseq_multiblock3d import MaskCollator   # Block masking
from vjepa2.src.masks.utils import apply_masks                    # Mask application
```

**What we build ourselves** (our codebase patterns):
- Data loading → WebDataset streaming (reuse `_create_stream()` from m05)
- Training loop → follows proposal Sec 10.3-10.5
- Drift control → `λ·‖θ-θ₀‖²` regularizer (Sec 10.4)
- Validation → Cycle@K hard mode (reuse `per_clip_cycle_at_k` from bootstrap.py)
- Checkpoint → our atomic `.npz` / `.pt` pattern
- WandB → `utils/wandb_utils.py`
- Config → YAML loaded via PyYAML

### Weight Loading: vjepa2 checkpoint (not HF)

**Why**: vjepa2's VisionTransformer expects weights in their format (`blocks.0.attn.qkv.weight`), not HF format (`encoder.layer.0.attention.self.query.weight`). Converting between formats is error-prone.

**Load pretrained weights** via `torch.hub` or direct download:
```python
# Option 1: torch.hub (auto-downloads)
encoder = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vitg", pretrained=True)

# Option 2: Direct checkpoint load
ckpt = torch.load("checkpoints/vjepa2_vitg.pt", map_location="cpu")
encoder.load_state_dict(ckpt["target_encoder"])  # Use EMA teacher weights as starting point
```

**After training**: Export adapted student encoder to HF format for m05/m06 evaluation (key conversion function needed).

---

## 1. Dependencies

### New dependencies (add to `requirements_gpu.txt`):
```
pyyaml>=6.0                    # YAML config loading
# vjepa2 installed separately (git clone + pip install -e)
```

### Existing dependencies reused:
```
torch, torchvision             # Model + transforms
transformers                   # AutoVideoProcessor (frame preprocessing)
datasets                       # HF WebDataset streaming (data loading)
flash-attn                     # Memory-efficient attention (ViT-g/ViT-G)
faiss-gpu                      # Validation: kNN for Cycle@K
scipy                          # Bootstrap CI
tqdm                           # Progress bars
wandb                          # Logging
```

### vjepa2 repo dependencies:
Check `vjepa2/pyproject.toml` for conflicts. Key risks:
- `torchvision` version pinning
- `timm` dependency (vjepa2 uses timm for some utilities)
- `xformers` (optional, for memory-efficient attention)

**Mitigation**: Install vjepa2 with `--no-deps` and manually verify compatibility:
```bash
pip install -e deps/vjepa2 --no-deps
```

---

## 2. Config YAML Structure

**File**: `configs/pretrain/vitg16_indian.yaml`

```yaml
# ═══════════════════════════════════════════════════════════════
# V-JEPA 2 ViT-g Continual Pretraining on Indian Urban Clips
# Ref: proposal Sec 10.3-10.5, plan_training.md
# ═══════════════════════════════════════════════════════════════

# ── Data ──────────────────────────────────────────────────────
data:
  hf_dataset: "anonymousML123/walkindia-200k"   # HF WebDataset repo
  local_data: null                               # Override with --local-data
  subset: null                                   # Override with --subset
  num_frames: 16                                 # T (frames per clip). 16 for POC, 64 for full
  crop_size: 384                                 # Spatial resolution (match pretrained)
  patch_size: 16                                 # ViT patch size
  tubelet_size: 2                                # Temporal tubelet size
  val_fraction: 0.1                              # 10% of videos held out for validation
  val_seed: 42                                   # Deterministic video-level split

# ── Model ─────────────────────────────────────────────────────
model:
  name: "vit_giant"                              # vjepa2 factory function
  checkpoint: "vjepa2_vitg"                      # torch.hub model name
  embed_dim: 1408                                # ViT-g hidden dimension
  depth: 40                                      # Number of transformer blocks

  # Predictor
  pred_depth: 12                                 # Predictor transformer layers
  pred_embed_dim: 384                            # Predictor hidden dimension
  pred_num_heads: 12                             # Predictor attention heads

# ── Masking ───────────────────────────────────────────────────
mask:
  # Small blocks (8 blocks × ~15% each)
  - aspect_ratio: [0.75, 1.5]
    num_blocks: 8
    spatial_scale: [0.15, 0.15]
    temporal_scale: [1.0, 1.0]
  # Large blocks (2 blocks × ~70% each)
  - aspect_ratio: [0.75, 1.5]
    num_blocks: 2
    spatial_scale: [0.7, 0.7]
    temporal_scale: [1.0, 1.0]

# ── Augmentation ──────────────────────────────────────────────
augmentation:
  random_resize_scale: [0.3, 1.0]               # RandomResizedCrop scale range
  random_resize_ratio: [0.75, 1.35]              # Aspect ratio range
  horizontal_flip: 0.5                           # Flip probability
  color_jitter: 0.4                              # Color jitter strength (0=off)

# ── Optimization ──────────────────────────────────────────────
optimization:
  # Learning rate
  lr: 1.0e-5                                     # Adaptation-scale LR (100x smaller than from-scratch)
  pred_lr_multiplier: 10.0                       # Predictor LR = lr × multiplier = 1e-4
  warmup_steps: 500                              # Linear warmup
  min_lr: 1.0e-7                                 # Minimum LR after cosine decay

  # Optimizer
  weight_decay: 0.04
  betas: [0.9, 0.999]
  eps: 1.0e-8
  grad_clip: 1.0                                 # Max gradient norm

  # EMA
  ema_momentum: 0.99925                          # Fixed (V-JEPA 2 default)

  # Training budget
  max_steps: 10000                               # Total training steps (POC: 2000, Full: 10000)
  batch_size: 4                                  # Per-GPU batch size (ViT-g is large)

  # Loss
  loss_exp: 1.0                                  # L1 loss (1.0) or MSE (2.0)

# ── Drift Control (Sec 10.4) ─────────────────────────────────
drift_control:
  enabled: true
  lambda_reg: 0.01                               # λ in λ·‖θ-θ₀‖²
  # Tune via ablation: [0, 0.001, 0.01, 0.1]

# ── Validation ────────────────────────────────────────────────
validation:
  interval_steps: 2000                           # Validate every N steps
  cycle_k: [1, 10]                               # Cycle@K values to compute
  faiss_k: 6                                     # kNN neighbors (includes self)

# ── Checkpointing ────────────────────────────────────────────
checkpoint:
  save_every_steps: 2000                         # Checkpoint frequency
  keep_last_n: 5                                 # Keep N most recent checkpoints
  output_dir: null                               # Auto-set from --SANITY/--FULL/--subset

# ── Mixed Precision ──────────────────────────────────────────
mixed_precision:
  enabled: true
  dtype: "bfloat16"                              # bfloat16 for training (not float16 — gradient underflow)
```

---

## 3. `m09_pretrain.py` — Section-by-Section Specification

### 3.0 File Header

```python
"""
V-JEPA 2 continual pretraining on Indian urban clips (Ch10).
Student-teacher JEPA with EMA, L1 latent prediction, drift control.
GPU-only (requires vjepa2 package + CUDA).

USAGE:
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --SANITY 2>&1 | tee logs/m09_pretrain_sanity.log
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
        2>&1 | tee logs/m09_pretrain_poc.log
"""
```

**Estimated file size**: ~600-800 lines (comparable to m05_vjepa_embed.py at 712 lines)

### 3.1 Imports & Constants

```python
import argparse
import copy
import gc
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    HF_DATASET_REPO, check_gpu,
    add_subset_arg, add_local_data_arg, get_output_dir, load_subset,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb,
)
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg

import torch
import torch.nn.functional as F

# vjepa2 imports (installed via deps/vjepa2)
from src.models.vision_transformer import vit_giant as build_encoder
from src.models.predictor import VisionTransformerPredictor
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks

# Reuse data loading from m05
from m05_vjepa_embed import (
    get_clip_key, _create_stream, decode_video_bytes,
    DECODE_WORKERS, MAX_STREAM_RETRIES, PREFETCH_QUEUE_SIZE,
)

# Constants
DEFAULT_CONFIG = "configs/pretrain/vitg16_indian.yaml"
CHECKPOINT_PREFIX = "m09_ckpt"
```

**Import verification**: If vjepa2 import paths differ, adjust. The actual import paths depend on how vjepa2 is installed:
- If `pip install -e deps/vjepa2`: imports are `from src.models...` (relative to vjepa2 package)
- If vendored: imports are `from utils.vjepa2_vendor...`
- Run `python -c "from vjepa2.src.models.vision_transformer import vit_giant; print('OK')"` to verify

### 3.2 YAML Config Loading

```python
def load_config(config_path: str) -> dict:
    """Load YAML config and return as nested dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def merge_config_with_args(cfg: dict, args) -> dict:
    """Override YAML config with CLI args where provided."""
    if args.subset:
        cfg["data"]["subset"] = args.subset
    if getattr(args, "local_data", None):
        cfg["data"]["local_data"] = args.local_data
    if args.SANITY:
        cfg["optimization"]["max_steps"] = 50          # Quick validation
        cfg["optimization"]["batch_size"] = 2
        cfg["validation"]["interval_steps"] = 25
        cfg["checkpoint"]["save_every_steps"] = 25
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    # Output dir
    cfg["checkpoint"]["output_dir"] = str(get_output_dir(args.subset, sanity=args.SANITY))
    return cfg
```

### 3.3 Data Loading (WebDataset → Video Tensors)

**Pattern**: Reuse m05's `_create_stream()` + `decode_video_bytes()` for HF streaming. Add video-consistent augmentation + train/val split by video_id.

```python
def split_by_video_id(clip_keys: list, val_fraction: float, seed: int) -> tuple:
    """Split clip keys into train/val by video_id (no leakage).

    Returns (train_keys: set, val_keys: set).
    """
    rng = np.random.RandomState(seed)
    video_ids = sorted(set(k.split("/")[-2] for k in clip_keys))  # Extract video_id from key
    rng.shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * val_fraction))
    val_video_ids = set(video_ids[:n_val])
    train_keys = set(k for k in clip_keys if k.split("/")[-2] not in val_video_ids)
    val_keys = set(k for k in clip_keys if k.split("/")[-2] in val_video_ids)
    return train_keys, val_keys


def augment_clip_consistent(video_tensor: torch.Tensor, cfg_aug: dict) -> torch.Tensor:
    """Video-consistent augmentation: one random crop applied to ALL T frames.

    Matches m05c_true_overlap.py `_augment_clip_consistent()` pattern.
    """
    T_frames, C, H, W = video_tensor.shape
    crop_size = cfg_aug.get("crop_size", 384)
    scale = cfg_aug.get("random_resize_scale", [0.3, 1.0])
    ratio = cfg_aug.get("random_resize_ratio", [0.75, 1.35])

    import torchvision.transforms as TT
    i, j, h, w = TT.RandomResizedCrop.get_params(
        video_tensor[0], scale=scale, ratio=ratio)

    # Apply SAME crop to all frames (video-consistent)
    video = video_tensor.float() / 255.0
    video = video[:, :, i:i+h, j:j+w]
    video = F.interpolate(video, size=(crop_size, crop_size),
                          mode='bilinear', align_corners=False)

    # Random horizontal flip (same for all frames)
    if torch.rand(1).item() < cfg_aug.get("horizontal_flip", 0.5):
        video = video.flip(-1)

    return video  # (T, C, crop_size, crop_size)


def producer_thread(cfg: dict, q: queue.Queue, stop_event: threading.Event,
                    train_keys: set, processed_steps: int):
    """Stream WebDataset, decode videos, augment, enqueue batches.

    Follows m05 _producer_thread() pattern:
    - ThreadPoolExecutor for parallel decode (GIL released during I/O)
    - Enqueue preprocessed batches for GPU training
    - Auto-retry on stream errors
    """
    batch_size = cfg["optimization"]["batch_size"]
    num_frames = cfg["data"]["num_frames"]
    local_data = cfg["data"].get("local_data")
    subset_keys = train_keys  # Only train clips

    # ... follows m05 producer pattern with augmentation added ...
    # Key difference: augment_clip_consistent() applied to each decoded clip
    # Two views per clip: context view + target view (same crop, different color jitter)
```

**Train/Val split**: By `video_id` (not by clip). Extract video_id from clip key format `section/video_id/source_file`.

**Same clip to both student and teacher** (verified from V-JEPA 2 source): V-JEPA 2 does NOT use two separately augmented views. The **same augmented clip** is passed to both student and teacher. The asymmetry comes from **masking**: student processes only visible tokens (via `masks_enc`), teacher processes all tokens. Masks are applied post-forward on teacher output to extract targets at masked positions for loss computation.

### 3.4 Model Setup (Student + Teacher + Predictor)

```python
def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder, teacher encoder (EMA), and predictor.

    Returns dict with keys: "student", "teacher", "predictor", "init_params"
    """
    model_cfg = cfg["model"]

    # ── Student encoder ──────────────────────────────────────
    student = build_encoder(
        img_size=cfg["data"]["crop_size"],
        patch_size=model_cfg.get("patch_size", cfg["data"]["patch_size"]),
        tubelet_size=cfg["data"]["tubelet_size"],
    )

    # Load pretrained weights
    ckpt = torch.hub.load("facebookresearch/vjepa2",
                          model_cfg["checkpoint"], pretrained=True)
    # ckpt is the encoder state_dict (or full model)
    # Handle different checkpoint formats:
    if isinstance(ckpt, dict) and "target_encoder" in ckpt:
        # Full training checkpoint: use target_encoder (EMA = best quality)
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                      for k, v in ckpt["target_encoder"].items()}
    elif isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    else:
        state_dict = ckpt

    student.load_state_dict(state_dict, strict=False)
    student = student.to(device)
    print(f"Student encoder loaded: {sum(p.numel() for p in student.parameters()):,} params")

    # ── Teacher encoder (EMA copy, frozen) ───────────────────
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher encoder created (deepcopy of student)")

    # ── Predictor ────────────────────────────────────────────
    predictor = VisionTransformerPredictor(
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["pred_depth"],
        predictor_embed_dim=model_cfg["pred_embed_dim"],
        num_heads=model_cfg["pred_num_heads"],
    )
    predictor = predictor.to(device)
    print(f"Predictor loaded: {sum(p.numel() for p in predictor.parameters()):,} params")

    # ── Save initial params for drift control ────────────────
    init_params = {name: p.clone().detach()
                   for name, p in student.named_parameters()}
    print(f"Saved initial params for drift control ({len(init_params)} tensors)")

    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "init_params": init_params,
    }
```

**Critical: Weight loading path**. Must verify that `torch.hub.load("facebookresearch/vjepa2", "vjepa2_vitg", pretrained=True)` works and returns the correct model/weights. If torch.hub format differs, adapt:

```python
# Fallback: direct checkpoint download
import urllib.request
CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vitg16.pt"  # Verify actual URL
ckpt_path = Path("checkpoints/vjepa2_vitg16.pt")
if not ckpt_path.exists():
    urllib.request.urlretrieve(CKPT_URL, ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
```

### 3.5 Mask Generation

```python
def build_mask_collator(cfg: dict) -> MaskCollator:
    """Build spatiotemporal block mask sampler from config.

    Uses vjepa2's MaskCollator which generates:
    - masks_enc: visible token indices for student (context)
    - masks_pred: masked token indices for predictor (target)
    """
    mask_cfgs = cfg["mask"]
    crop_size = cfg["data"]["crop_size"]
    patch_size = cfg["data"]["patch_size"]
    num_frames = cfg["data"]["num_frames"]
    tubelet_size = cfg["data"]["tubelet_size"]

    # Spatial grid: crop_size / patch_size per dimension
    height = width = crop_size // patch_size  # 384/16 = 24
    duration = num_frames // tubelet_size      # 16/2 = 8 (or 64/2 = 32)

    collator = MaskCollator(
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        cfgs_mask=mask_cfgs,
        # Additional params from MaskCollator signature
    )
    return collator
```

**Token grid dimensions** (ViT-g @ 384px, 16f):
- Spatial: 384/16 = 24×24 = 576 patches per frame
- Temporal: 16/2 = 8 tubelets
- Total tokens: 576 × 8 = 4,608 tokens per clip

**Mask coverage**: 8 small blocks (15% each) + 2 large blocks (70% each) → ~75-90% masked → student sees ~10-25% of tokens.

### 3.6 JEPA Loss + Drift Control

```python
def compute_jepa_loss(pred_features: list, target_features: list,
                      masks_pred: list, loss_exp: float = 1.0) -> torch.Tensor:
    """L1 latent prediction loss on masked tokens (V-JEPA 2 convention).

    Args:
        pred_features: Predictor outputs (list of tensors per mask set)
        target_features: Teacher outputs (list of tensors per layer)
        masks_pred: Target mask indices (which tokens to predict)
        loss_exp: 1.0 = L1, 2.0 = MSE

    Follows vjepa2 `loss_fn`:
        loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
    """
    # Apply masks to teacher features to get targets at masked positions
    h_masked = [apply_masks(hi, mi, concat=False)
                for hi, mi in zip(target_features, masks_pred)]

    loss = torch.tensor(0.0, device=pred_features[0][0].device)
    n = 0
    for zi, hi in zip(pred_features, h_masked):
        for zij, hij in zip(zi, hi):
            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    loss /= max(n, 1)
    return loss


def compute_drift_loss(student: torch.nn.Module,
                       init_params: dict,
                       lambda_reg: float) -> torch.Tensor:
    """L2 drift control: λ·‖θ - θ₀‖² (Sec 10.4).

    Anchors student to pretrained initialization to prevent catastrophic forgetting.
    Equivalent to EWC with uniform importance weights.
    """
    drift = torch.tensor(0.0, device=next(student.parameters()).device)
    for name, param in student.named_parameters():
        if name in init_params:
            drift += torch.sum((param - init_params[name]) ** 2)
    return lambda_reg * drift
```

### 3.7 EMA Teacher Update

```python
@torch.no_grad()
def update_teacher_ema(student: torch.nn.Module,
                       teacher: torch.nn.Module,
                       momentum: float):
    """EMA update: θ̄ ← τ·θ̄ + (1-τ)·θ

    Uses torch._foreach_mul_ / _foreach_add_ for batched update (V-JEPA 2 convention).
    """
    params_student = list(student.parameters())
    params_teacher = list(teacher.parameters())
    torch._foreach_mul_(params_teacher, momentum)
    torch._foreach_add_(params_teacher, params_student, alpha=1.0 - momentum)
```

### 3.8 Optimizer & Scheduler

```python
def build_optimizer(student: torch.nn.Module,
                    predictor: torch.nn.Module,
                    cfg_opt: dict) -> torch.optim.AdamW:
    """Build AdamW with separate param groups for encoder vs predictor.

    Predictor gets pred_lr_multiplier × base LR (Sec 10.4).
    Bias/norm params get weight_decay=0 (V-JEPA 2 convention).
    """
    base_lr = cfg_opt["lr"]
    pred_lr = base_lr * cfg_opt["pred_lr_multiplier"]
    wd = cfg_opt["weight_decay"]

    def split_params(module, lr):
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

    param_groups = split_params(student, base_lr) + split_params(predictor, pred_lr)
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg_opt["betas"]),
                                  eps=cfg_opt["eps"])
    return optimizer


def build_scheduler(optimizer, cfg_opt: dict):
    """Cosine decay with linear warmup.

    NOTE: V-JEPA 2 from-scratch uses warmup-constant-cooldown (NOT cosine).
    We use cosine for continual pretraining because:
    (a) our training budget is much smaller (2K-10K steps vs millions),
    (b) cosine provides a natural annealing without needing a separate cooldown phase.
    """
    warmup_steps = cfg_opt["warmup_steps"]
    max_steps = cfg_opt["max_steps"]
    min_lr = cfg_opt["min_lr"]
    base_lr = cfg_opt["lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # Linear warmup
        # Cosine decay
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(min_lr / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 3.9 Validation (Cycle@K Hard Mode)

```python
def run_validation(student: torch.nn.Module,
                   val_clips: list,
                   cfg: dict,
                   device: torch.device,
                   step: int) -> dict:
    """Run Cycle@K (hard mode) on validation clips.

    Reuses m06 FAISS-GPU kNN + bootstrap.py Cycle@K computation.
    Returns dict of metrics with 95% CI.
    """
    import faiss
    from utils.bootstrap import bootstrap_ci, per_clip_cycle_at_k

    student.eval()

    # 1. Extract embeddings for val clips (frozen student forward)
    embeddings = []
    clip_keys = []
    with torch.no_grad():
        for batch_clips, batch_keys in val_dataloader:
            batch_clips = batch_clips.to(device)
            # Full forward (no masking for evaluation)
            features = student(batch_clips)
            # Pool: mean over spatiotemporal tokens
            emb = features.mean(dim=1)  # (B, D)
            emb = F.normalize(emb, dim=-1)  # L2 normalize
            embeddings.append(emb.cpu().numpy())
            clip_keys.extend(batch_keys)

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    n, d = embeddings.shape

    # 2. Build FAISS GPU index
    faiss_k = cfg["validation"]["faiss_k"]
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    index.add(embeddings)
    distances, indices = index.search(embeddings, faiss_k)

    # 3. Compute Cycle@K with bootstrap CI
    results = {}
    for k in cfg["validation"]["cycle_k"]:
        scores = per_clip_cycle_at_k(indices, k)
        ci = bootstrap_ci(scores)
        results[f"cycle_at_{k}"] = ci
        print(f"  Step {step} | Cycle@{k}: {ci['mean']:.1f}% "
              f"[{ci['ci_lo']:.1f}, {ci['ci_hi']:.1f}]")

    student.train()
    return results
```

**Hard mode exclusion**: For POC (10K clips), skip hard mode exclusion. Add at 115K scale:
```python
# Hard mode: build exclusion mask from clip_keys (same-video ±30s)
# Reuse m06_faiss_metrics.py build_exclusion_mask() + apply_hard_filter()
```

**Overlap@K deferred**: Proposal Sec 10.6 specifies validation with both Cycle@K AND Overlap@K. Overlap@K requires generating augmented views (as in m05c) which is expensive during training. For POC, Cycle@K alone is sufficient for checkpoint selection. Add Overlap@K at 115K scale or as a post-training evaluation (reuse m05c + m06).

### 3.10 Training Loop (Core)

```python
def train(cfg: dict, args):
    """Main training loop (proposal Sec 10.5).

    One training step:
    1. Sample minibatch from WebDataset stream
    2. Decode T frames, resize to crop_size
    3. Generate context + target views (video-consistent augmentation)
    4. Sample spatiotemporal masks
    5. Student forward on context view (visible tokens only)
    6. Teacher forward on target view (all tokens, no grad)
    7. Predictor: student features → teacher feature space
    8. L1 loss on masked token predictions + drift control
    9. AdamW step on student + predictor
    10. EMA update teacher
    """
    check_gpu()
    device = torch.device("cuda")

    # ── Output-exists guard (CLAUDE.md rule #6) ──────────────
    output_dir = Path(cfg["checkpoint"]["output_dir"])
    final_ckpt = output_dir / f"{CHECKPOINT_PREFIX}_final.pt"
    if final_ckpt.exists():
        print(f"Final checkpoint already exists: {final_ckpt}")
        print("Delete to re-train, or use for evaluation.")
        return

    # ── Build model ──────────────────────────────────────────
    models = build_model(cfg, device)
    student = models["student"]
    teacher = models["teacher"]
    predictor = models["predictor"]
    init_params = models["init_params"]

    student.train()
    predictor.train()
    teacher.eval()

    # ── Build mask collator ──────────────────────────────────
    mask_collator = build_mask_collator(cfg)

    # ── Build optimizer & scheduler ──────────────────────────
    optimizer = build_optimizer(student, predictor, cfg["optimization"])
    scheduler = build_scheduler(optimizer, cfg["optimization"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["mixed_precision"]["enabled"])

    # ── Load checkpoint for resume ───────────────────────────
    start_step = 0
    best_cycle_k = 0.0
    ckpt_path = output_dir / f"{CHECKPOINT_PREFIX}_latest.pt"
    if ckpt_path.exists():
        start_step, best_cycle_k = load_training_checkpoint(
            ckpt_path, student, teacher, predictor, optimizer, scheduler, scaler)
        print(f"Resumed from step {start_step}, best Cycle@1: {best_cycle_k:.1f}%")

    # ── Data stream ──────────────────────────────────────────
    # ... producer thread setup (reuse m05 pattern) ...

    # ── WandB ────────────────────────────────────────────────
    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m09", mode, config=cfg, enabled=not args.no_wandb)

    # ── Training loop ────────────────────────────────────────
    max_steps = cfg["optimization"]["max_steps"]
    batch_size = cfg["optimization"]["batch_size"]
    ema_momentum = cfg["optimization"]["ema_momentum"]
    loss_exp = cfg["optimization"]["loss_exp"]
    drift_cfg = cfg["drift_control"]
    val_interval = cfg["validation"]["interval_steps"]
    ckpt_interval = cfg["checkpoint"]["save_every_steps"]

    dtype = getattr(torch, cfg["mixed_precision"]["dtype"])

    pbar = tqdm(total=max_steps, initial=start_step,
                desc="m09_pretrain", unit="step")

    # Windowed throughput tracking (CLAUDE.md rule #11)
    window_start = time.time()
    window_steps = 0
    running_loss = 0.0

    for step in range(start_step, max_steps):
        step_start = time.time()

        # ── Get batch from producer queue ────────────────
        try:
            msg_type, batch_clips, batch_keys = q.get(timeout=300)
        except queue.Empty:
            print(f"\nProducer timeout at step {step}. Saving checkpoint...")
            break
        if msg_type == "done":
            print(f"\nData exhausted at step {step}.")
            break

        # ── Generate masks ───────────────────────────────
        # MaskCollator returns (masks_enc, masks_pred) per sample
        collated_masks = mask_collator(batch_size)
        masks_enc, masks_pred = collated_masks

        # Move to device
        batch_clips = batch_clips.to(device)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        # ── Forward pass (mixed precision) ───────────────
        with torch.amp.autocast("cuda", dtype=dtype,
                                enabled=cfg["mixed_precision"]["enabled"]):
            # Teacher forward (no grad, all tokens)
            with torch.no_grad():
                teacher_features = teacher(batch_clips)
                # Layer norm on teacher outputs (V-JEPA 2 convention)
                teacher_features = [F.layer_norm(h, (h.size(-1),))
                                    for h in (teacher_features
                                              if isinstance(teacher_features, list)
                                              else [teacher_features])]

            # Student forward (only visible tokens via masks_enc)
            student_features = student(batch_clips, masks_enc)

            # Predictor (map student → teacher space for masked positions)
            pred_features = predictor(student_features, masks_enc, masks_pred)

            # Loss
            jepa_loss = compute_jepa_loss(pred_features, teacher_features,
                                          masks_pred, loss_exp)

            # Drift control
            if drift_cfg["enabled"] and drift_cfg["lambda_reg"] > 0:
                drift_loss = compute_drift_loss(student, init_params,
                                                drift_cfg["lambda_reg"])
                total_loss = jepa_loss + drift_loss
            else:
                drift_loss = torch.tensor(0.0)
                total_loss = jepa_loss

        # ── Backward + optimizer step ────────────────────
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(predictor.parameters()),
            cfg["optimization"]["grad_clip"])

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── EMA teacher update ───────────────────────────
        update_teacher_ema(student, teacher, ema_momentum)

        # ── Logging (windowed throughput) ────────────────
        running_loss += total_loss.item()
        window_steps += 1
        now = time.time()
        window_elapsed = now - window_start

        if window_elapsed >= 30:  # Reset window every 30s
            throughput = window_steps / window_elapsed
            pbar.set_postfix_str(
                f"loss={running_loss/window_steps:.4f} "
                f"drift={drift_loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"grad={grad_norm:.2f} "
                f"{throughput:.2f} step/s")
            log_metrics(wb_run, {
                "loss/jepa": jepa_loss.item(),
                "loss/drift": drift_loss.item(),
                "loss/total": total_loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "throughput_steps_per_s": throughput,
                "ema_momentum": ema_momentum,
            }, step=step)
            window_start = now
            window_steps = 0
            running_loss = 0.0

        pbar.update(1)

        # ── Validation ───────────────────────────────────
        if (step + 1) % val_interval == 0:
            val_metrics = run_validation(student, val_clips, cfg, device, step + 1)
            log_metrics(wb_run, {f"val/{k}": v["mean"] for k, v in val_metrics.items()},
                        step=step)

            # Best model selection by Cycle@1
            cycle_1 = val_metrics.get("cycle_at_1", {}).get("mean", 0)
            if cycle_1 > best_cycle_k:
                best_cycle_k = cycle_1
                save_training_checkpoint(
                    output_dir / f"{CHECKPOINT_PREFIX}_best.pt",
                    student, teacher, predictor, optimizer, scheduler, scaler,
                    step + 1, best_cycle_k)
                print(f"  New best: Cycle@1 = {best_cycle_k:.1f}%")

        # ── Periodic checkpoint ──────────────────────────
        if (step + 1) % ckpt_interval == 0:
            save_training_checkpoint(
                output_dir / f"{CHECKPOINT_PREFIX}_step{step+1}.pt",
                student, teacher, predictor, optimizer, scheduler, scaler,
                step + 1, best_cycle_k)
            save_training_checkpoint(
                ckpt_path,  # Also update _latest for resume
                student, teacher, predictor, optimizer, scheduler, scaler,
                step + 1, best_cycle_k)

    pbar.close()

    # ── Save final checkpoint ────────────────────────────
    save_training_checkpoint(
        final_ckpt, student, teacher, predictor, optimizer, scheduler, scaler,
        step + 1, best_cycle_k)

    # ── Export student encoder for m05/m06 evaluation ────
    export_student_for_eval(student, output_dir / "student_encoder.pt")

    finish_wandb(wb_run)
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Steps:       {step + 1}")
    print(f"Best Cycle@1: {best_cycle_k:.1f}%")
    print(f"Final ckpt:  {final_ckpt}")
    print(f"Student:     {output_dir / 'student_encoder.pt'}")
```

### 3.11 Checkpoint Management

```python
def save_training_checkpoint(path: Path, student, teacher, predictor,
                              optimizer, scheduler, scaler,
                              step: int, best_metric: float):
    """Save full training state for resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save({
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "best_metric": best_metric,
    }, tmp_path)
    os.replace(tmp_path, path)  # Atomic rename


def load_training_checkpoint(path: Path, student, teacher, predictor,
                              optimizer, scheduler, scaler) -> tuple:
    """Load training state for resume. Returns (step, best_metric)."""
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    predictor.load_state_dict(ckpt["predictor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt.get("best_metric", 0.0)


def export_student_for_eval(student, path: Path):
    """Export student encoder weights for use in m05/m06 evaluation.

    Saves in HF-compatible format so m05_vjepa_embed.py can load it:
        model = AutoModel.from_pretrained(path)
    """
    # Option 1: Save raw state_dict (for vjepa2 model loading)
    torch.save(student.state_dict(), path)

    # Option 2: Convert to HF format and save as pretrained model
    # This requires key name conversion (vjepa2 → HF transformers)
    # TODO: implement key conversion after verifying model structure
    # hf_model = convert_vjepa2_to_hf(student)
    # hf_model.save_pretrained(path.parent / "hf_adapted")
```

### 3.12 Main / Argparse

```python
def main():
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 continual pretraining on Indian urban clips (Ch10)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="YAML config path")
    parser.add_argument("--SANITY", action="store_true",
                        help="Quick validation: 50 steps, batch_size=2")
    parser.add_argument("--FULL", action="store_true",
                        help="Full training run")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = merge_config_with_args(cfg, args)

    train(cfg, args)


if __name__ == "__main__":
    main()
```

---

## 4. `scripts/run_pretrain.sh` — Structure

**Pattern**: Follows `run_evaluate.sh` conventions (pre-flight, banner, run_step, verify).

```bash
#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Continual Pretrain V-JEPA on Indian Urban Clips (Ch10)
#
# Pipeline: m09 pretrain → m05 re-embed → m06 metrics → m08 plots
# Compares frozen vs adapted encoder on the same eval suite
#
# USAGE:
#   ./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
#   ./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
#
# Prerequisites:
#   - Ch9 pipeline completed (frozen encoder eval as baseline)
#   - vjepa2 installed: pip install -e deps/vjepa2
#   - ./setup_env_uv.sh --gpu
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Parse args ────────────────────────────────────────────────
# (same pattern as run_evaluate.sh)

# ── Pre-flight checks ────────────────────────────────────────
# Check: GPU, vjepa2 package, Ch9 baseline outputs exist

# ── Step 1: Continual pretraining ─────────────────────────────
run_step 1 "m09 continual pretraining" "$T_M09" \
    "$LOGDIR/m09_${MODE,,}.log" \
    src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-wandb

# Verify: checkpoint exists, student_encoder.pt exported
verify "Step 1 pretrain" "..."

# ── Step 2: Re-embed with adapted encoder ─────────────────────
# Use the adapted student encoder to generate new embeddings
# Then run m06 + m08 to compare frozen vs adapted
run_step 2 "m05 re-embed (adapted)" "$T_M05" \
    "$LOGDIR/m05_adapted_${MODE,,}.log" \
    src/m05_vjepa_embed.py --model ${OUT_DIR}/hf_adapted \
        $MODE_FLAG $SUBSET_FLAG $LOCAL_FLAG --no-dedupe --no-wandb

# ── Step 3: Metrics on adapted embeddings ─────────────────────
run_step 3 "m06 metrics (adapted)" "$T_M06" \
    "$LOGDIR/m06_adapted_${MODE,,}.log" \
    src/m06_faiss_metrics.py --encoder vjepa_adapted \
        $MODE_FLAG $SUBSET_FLAG --no-wandb

# ── Step 4: Comparison plots (frozen vs adapted) ─────────────
# m08b_compare.py with --encoders vjepa,vjepa_adapted
run_step 4 "m08b comparison (frozen vs adapted)" "$T_M08B" \
    "$LOGDIR/m08b_adapted_${MODE,,}.log" \
    src/m08b_compare.py --encoders vjepa,vjepa_adapted \
        $MODE_FLAG $SUBSET_FLAG --no-wandb

# ── Summary ──────────────────────────────────────────────────
# Print frozen vs adapted metrics side-by-side
```

**Key difference from run_evaluate.sh**: After training (m09), we re-run the Ch9 eval pipeline (m05→m06→m08) with the adapted encoder to measure improvement.

### Encoder Registry Extension

Add `vjepa_adapted` to `utils/config.py`:
```python
ENCODER_REGISTRY = {
    ...
    "vjepa_adapted":  {"model_id": None, "dim": 1408, "type": "video", "suffix": "_adapted"},
}
```

The adapted model path is set at runtime via `--model` flag (already supported by m05).

---

## 5. File Dependencies

```
configs/pretrain/vitg16_indian.yaml     # NEW — config
src/m09_pretrain.py                      # NEW — main training script
scripts/run_pretrain.sh                  # NEW — orchestrator
src/utils/config.py                      # MODIFY — add vjepa_adapted encoder
requirements_gpu.txt                     # MODIFY — add pyyaml
setup_env_uv.sh                          # MODIFY — install vjepa2
```

### Data flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         m09_pretrain.py                              │
│                                                                     │
│  WebDataset stream   ──→  Video decode  ──→  Augment + Mask         │
│  (HF or local TARs)       (reuse m05)        (vjepa2 collator)      │
│                                                                     │
│  Student(visible tokens) ──→ Predictor ──→ L1 loss ← Teacher(all)   │
│       ↓                        ↓                        ↓           │
│  AdamW step              drift loss                 EMA update      │
│       ↓                                                             │
│  Checkpoint (student + teacher + predictor + optimizer)              │
│       ↓                                                             │
│  Validation: Cycle@K (FAISS-GPU) every N steps                      │
│       ↓                                                             │
│  Best checkpoint → export student_encoder.pt                        │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  m05 re-embed (adapted) → m06 metrics → m08 plots → comparison     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Mandatory Checklist Compliance (CLAUDE.md)

| # | Requirement | How m09 satisfies it |
|---|-------------|---------------------|
| 1 | tqdm progress bar | `tqdm(total=max_steps, desc="m09_pretrain", unit="step")` |
| 2 | Auto-resume from checkpoint | `load_training_checkpoint()` on `_latest.pt` |
| 3 | Tee logging | `python -u src/m09_pretrain.py ... 2>&1 \| tee logs/m09.log` |
| 4 | WandB integration | `init_wandb("m09", ...)`, `log_metrics()` per step |
| 5 | Windowed throughput | 30s window, `step/s` in postfix, not `total/elapsed` |
| 6 | Output-exists guard | Check `final_ckpt.exists()` before loading model |
| 7 | No CPU fallback | `check_gpu()` at top, FAISS-GPU for validation |
| 8 | Dynamic prints | All prints include step count, loss values, metrics |
| 9 | Bootstrap CI | Validation Cycle@K uses `bootstrap_ci()` |
| 10 | Gradient clipping | `clip_grad_norm_(params, 1.0)` |

---

## 7. Testing Strategy

### 7.1 M1 Mac (CPU, no GPU)

```bash
# Syntax check
source venv_walkindia/bin/activate && python3 -m py_compile src/m09_pretrain.py

# AST structural check
python3 -c "
import ast, sys
tree = ast.parse(open('src/m09_pretrain.py').read())
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
required = ['load_config', 'build_model', 'compute_jepa_loss', 'compute_drift_loss',
            'update_teacher_ema', 'train', 'run_validation', 'main']
missing = [f for f in required if f not in funcs]
assert not missing, f'Missing functions: {missing}'
print(f'OK: {len(funcs)} functions found, all required present')
"

# Help flag
python3 src/m09_pretrain.py --help
```

### 7.2 GPU Cloud (--SANITY)

```bash
# Quick validation: 50 steps, 2 clips/batch
python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
    --SANITY --no-wandb 2>&1 | tee logs/m09_sanity.log

# Verify outputs:
# - m09_ckpt_latest.pt exists
# - m09_ckpt_final.pt exists
# - student_encoder.pt exists
# - No NaN in loss
# - Cycle@K computed (even if meaningless at 5 clips)
```

### 7.3 GPU Cloud (--FULL POC)

```bash
# 10K POC: 2000 steps, batch_size=4
python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
    --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
    --no-wandb 2>&1 | tee logs/m09_pretrain_poc.log

# Expected: ~20h on RTX PRO 6000 96GB
# Validation every 2000 steps: Cycle@1 should improve from frozen baseline
```

---

## 8. Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| vjepa2 import paths differ from expected | HIGH | Test imports first; adjust `from vjepa2.src...` to actual package structure |
| Weight format mismatch (HF vs vjepa2) | HIGH | Use vjepa2's own checkpoint loading, NOT HF format |
| OOM on ViT-g training (1B params + gradients + optimizer state) | HIGH | batch_size=4 default; AMP bfloat16; AdaptiveBatchSizer; gradient checkpointing |
| Loss NaN/explosion | MEDIUM | Gradient clipping (1.0), learning rate warmup, bfloat16 (not float16) |
| Predictor convergence | LOW | 12-layer narrow transformer is well-tested in V-JEPA 2 |
| Drift control too strong/weak | LOW | Ablation over λ ∈ [0, 0.001, 0.01, 0.1] |

### VRAM Budget (ViT-g, batch=4, 16 frames, 384px, bfloat16)

| Component | VRAM (est.) |
|-----------|-------------|
| Student encoder (1B params, bf16) | ~2 GB |
| Teacher encoder (1B params, bf16, no grad) | ~2 GB |
| Predictor (~22M params) | ~0.05 GB |
| Optimizer state (Adam: 2× student + predictor params, fp32) | ~8 GB |
| Activations (batch=4, 4608 tokens × 1408 dim × 40 layers) | ~30 GB |
| Gradient checkpointing savings | −15 GB |
| **Total estimated** | **~27 GB** |

Fits comfortably on RTX PRO 6000 (96 GB). Can increase batch_size to 8-16.

---

## 9. Implementation Order

| Step | What | Est. Time | Depends On |
|------|------|-----------|------------|
| 1 | Install vjepa2, verify imports | 1h | GPU cloud access |
| 2 | Write `configs/pretrain/vitg16_indian.yaml` | 30m | — |
| 3 | Write m09 skeleton: argparse, config loading, main() | 1h | Step 2 |
| 4 | Write model loading (student, teacher, predictor) | 2h | Step 1 |
| 5 | Write data loading (reuse m05 producer, add augmentation) | 2h | Step 3 |
| 6 | Write training loop (loss, optimizer, EMA, masking) | 3h | Steps 4-5 |
| 7 | Write validation (Cycle@K) | 1h | Step 6 |
| 8 | Write checkpoint management | 1h | Step 6 |
| 9 | Write run_pretrain.sh | 1h | Step 8 |
| 10 | --SANITY test on GPU | 1h | Step 9 |
| 11 | Fix bugs from SANITY, run --FULL POC | ~20h GPU | Step 10 |

**Total dev time**: ~13h coding + ~21h GPU time ≈ 1.5 days coding + 1 day GPU

---

## 10. Open Questions (Resolve Before Coding)

1. **vjepa2 package structure**: What are the actual import paths? Run:
   ```bash
   git clone https://github.com/facebookresearch/vjepa2.git deps/vjepa2
   find deps/vjepa2 -name "*.py" | head -30
   python -c "import deps.vjepa2.src.models.vision_transformer as m; print(dir(m))"
   ```

2. **Pretrained checkpoint availability**: Does `torch.hub.load("facebookresearch/vjepa2", "vjepa2_vitg")` work? If not, what's the direct download URL?

3. **Predictor checkpoint**: Does the public checkpoint include predictor weights, or just the encoder? If no predictor weights, we initialize randomly (acceptable for continual pretraining).

4. **MaskCollator API**: What arguments does `MaskCollator.__init__()` take? What does `__call__()` return? Need to verify against actual source.

5. **HF export**: How to convert vjepa2 state_dict keys → HF transformers format for m05 re-embedding? Might need a `convert_vjepa2_to_hf()` utility.

6. **Memory**: Does ViT-g with batch_size=4 + AMP + gradient checkpointing fit in 96GB? Need to profile first with a dry run.
