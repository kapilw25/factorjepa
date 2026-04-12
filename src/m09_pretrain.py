"""V-JEPA 2.1 training: ExPLoRA (Step 1b) + Ch10 continual pretraining (legacy). GPU-only.

ExPLoRA (V-JEPA 2.1, LoRA + unfreeze 1-2 blocks):
    python -u src/m09_pretrain.py --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/explora.yaml --explora \
        --SANITY --local-data data/val_1k_local --val-local-data data/val_1k_local \
        2>&1 | tee logs/m09_explora_sanity.log
    python -u src/m09_pretrain.py --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/explora.yaml --explora \
        --POC --local-data data/val_1k_local --val-local-data data/val_1k_local \
        2>&1 | tee logs/m09_explora_poc.log
    python -u src/m09_pretrain.py --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/explora.yaml --explora \
        --FULL --local-data data/full_local --val-local-data data/val_1k_local \
        2>&1 | tee logs/m09_explora_full.log

Ch10 continual pretraining (legacy, --config single YAML):
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --SANITY 2>&1 | tee logs/m09_pretrain_sanity.log
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --POC --subset data/subset_10k.json --local-data data/subset_10k_local \
        --val-subset data/val_1k.json --val-local-data data/val_1k_local \
        2>&1 | tee logs/m09_pretrain_poc.log
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --FULL --local-data data/full_local --val-subset data/val_1k.json \
        --val-local-data data/val_1k_local 2>&1 | tee logs/m09_pretrain_full.log
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # Must be before torch import
os.environ.setdefault("MKL_NUM_THREADS", "1")   # Prevent OpenMP thread explosion in workers

import argparse
import copy
import csv
import gc
import glob
import json
import math
import queue
import random
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    check_gpu,
    add_subset_arg, add_local_data_arg, get_output_dir, load_subset,
    get_pipeline_config, load_merged_config,
    add_model_config_arg, add_train_config_arg,
)
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)

import torch
import torch.nn.functional as F

# vjepa2 imports via shim (avoids src/ namespace collision)
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
    get_mask_generator, get_apply_masks,
)

# Constants
DEFAULT_MODEL_CONFIG = "configs/model/vjepa2_1.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train/ch10_pretrain.yaml"
CHECKPOINT_PREFIX = "m09_ckpt"
_pcfg = get_pipeline_config()
PREFETCH_QUEUE_SIZE = _pcfg["streaming"]["prefetch_queue_train"]
DECODE_WORKERS = _pcfg["streaming"]["decode_workers_train"]
MAX_STREAM_RETRIES = _pcfg["streaming"]["max_retries"]

# Shared video I/O from utils (Rule 32: no cross-imports between m*.py)
from utils.video_io import get_clip_key, create_stream, decode_video_bytes

_create_stream = create_stream


# ═════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config_with_args(cfg: dict, args) -> dict:
    if args.subset:
        cfg["data"]["subset"] = args.subset
    if getattr(args, "local_data", None):
        cfg["data"]["local_data"] = args.local_data
    if args.SANITY:
        mode_key = "sanity"
    elif args.POC:
        mode_key = "poc"
    else:
        mode_key = "full"
    cfg["optimization"]["max_epochs"] = cfg["optimization"]["max_epochs"][mode_key]
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        cfg["optimization"]["max_epochs"] = args.max_epochs
    if args.lambda_reg is not None:
        cfg["drift_control"]["lambda_reg"] = args.lambda_reg
        if args.lambda_reg == 0:
            cfg["drift_control"]["enabled"] = False
    elif args.SANITY:
        # SANITY: use 0.001 as default (just testing code paths, not real training)
        cfg["drift_control"]["lambda_reg"] = 0.001

    # Output dir: explicit --output-dir, or auto from mode + lambda
    if getattr(args, "output_dir", None):
        cfg["checkpoint"]["output_dir"] = args.output_dir
        return cfg
    base_out = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
    lam = cfg["drift_control"]["lambda_reg"]
    if lam is None:
        # Auto-ablation in main() will set lambda_reg and output_dir later
        cfg["checkpoint"]["output_dir"] = str(base_out / "m09_pending_ablation")
        return cfg
    lam_str = f"{lam:g}".replace(".", "_")  # 0.0→"0", 0.01→"0_01" (no trailing .0)
    cfg["checkpoint"]["output_dir"] = str(base_out / f"m09_lambda{lam_str}")
    return cfg


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

def load_val_subset(val_subset_path: str) -> set:
    """Load val subset clip keys from JSON (generated by m00c_sample_subset.py)."""
    if not val_subset_path:
        return set()
    p = Path(val_subset_path)
    if not p.exists():
        print(f"FATAL: Val subset file not found: {p}")
        sys.exit(1)
    with open(p) as f:
        data = json.load(f)
    keys = set(data["clip_keys"])
    print(f"[VAL] Loaded val subset: {len(keys):,} clips from {p.name}")
    return keys


# ImageNet normalization — MUST match Meta's V-JEPA training pipeline.
# deps/vjepa2/app/vjepa/transforms.py line 110: _tensor_normalize_inplace(buffer, mean, std)
# Without this, the model sees [0,1] range during training but [-2,2.6] during evaluation
# (HF processor normalizes). The mismatch caused -26% Prec@K artifact.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def augment_clip_consistent(video_tensor: torch.Tensor, cfg_aug: dict,
                            crop_size: int) -> torch.Tensor:
    """Video-consistent augmentation matching Meta's V-JEPA pipeline exactly.

    Steps: RandomResizedCrop → HorizontalFlip → ImageNet normalize.
    Matches deps/vjepa2/app/vjepa/transforms.py:VideoTransform.__call__().
    """
    import torchvision.transforms as TT

    T_frames, C, H, W = video_tensor.shape
    scale = cfg_aug["random_resize_scale"]
    ratio = cfg_aug["random_resize_ratio"]

    i, j, h, w = TT.RandomResizedCrop.get_params(
        video_tensor[0], scale=scale, ratio=ratio)

    video = video_tensor.float() / 255.0
    video = video[:, :, i:i+h, j:j+w]
    video = F.interpolate(video, size=(crop_size, crop_size),
                          mode='bilinear', align_corners=False)

    if torch.rand(1).item() < cfg_aug["horizontal_flip"]:
        video = video.flip(-1)

    # ImageNet normalization — matches Meta's training exactly
    video = (video - IMAGENET_MEAN.to(video.device)) / IMAGENET_STD.to(video.device)

    return video  # (T, C, crop_size, crop_size), ImageNet-normalized


def producer_thread(cfg: dict, q: queue.Queue, stop_event: threading.Event,
                    train_keys: set, processed_steps: int):
    """Stream WebDataset, decode videos, augment, enqueue batches (multi-epoch)."""
    from concurrent.futures import ThreadPoolExecutor

    # Prevent ATen thread oversubscription (CLAUDE.md rule #12)
    torch.set_num_threads(1)

    batch_size = cfg["optimization"]["batch_size"]
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    local_data = cfg["data"]["local_data"]
    cfg_aug = cfg["augmentation"]
    retries = 0
    epoch = 0

    tmp_dir = tempfile.mkdtemp(prefix="m09_")

    def _decode_batch(pool, pending_bytes, pending_keys):
        """Decode + augment a batch using the shared thread pool."""
        futures = [
            pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
            for b, k in zip(pending_bytes, pending_keys)
        ]
        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

        batch_tensors = [t for t, k in results if t is not None]
        batch_keys = [k for t, k in results if t is not None]

        if batch_tensors:
            augmented = [augment_clip_consistent(vt, cfg_aug, crop_size)
                         for vt in batch_tensors]
            batch = torch.stack(augmented, dim=0).permute(0, 2, 1, 3, 4)
            q.put(("batch", batch, batch_keys[:]))

    try:
        # Single thread pool for entire producer lifetime (avoids thread exhaustion —
        # creating per-batch leaks threads: CPython #98467, ffmpeg spawns 16 internal
        # threads per decoder when thread_count=0)
        with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
            while not stop_event.is_set():
                try:
                    epoch += 1
                    if epoch > 1:
                        print(f"  Producer: starting epoch {epoch}")
                    pending_bytes, pending_keys = [], []

                    if local_data:
                        # Fast path: parallel TAR readers (8 threads per epoch)
                        clip_q, tar_stop, _reader = iter_clips_parallel(
                            local_data, subset_keys=train_keys or None)
                        while not stop_event.is_set():
                            item = clip_q.get(timeout=120)
                            if item is None:
                                break
                            clip_key, mp4_bytes = item
                            if not mp4_bytes:
                                continue
                            pending_bytes.append(mp4_bytes)
                            pending_keys.append(clip_key)
                            if len(pending_bytes) >= batch_size:
                                _decode_batch(pool, pending_bytes, pending_keys)
                                pending_bytes, pending_keys = [], []
                        tar_stop.set()
                    else:
                        # Fallback: sequential HF streaming
                        ds = _create_stream(0, local_data=local_data)
                        for example in ds:
                            if stop_event.is_set():
                                break
                            clip_key = get_clip_key(example)
                            if train_keys and clip_key not in train_keys:
                                continue
                            mp4_data = example.get("mp4", b"")
                            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
                            if not mp4_bytes:
                                continue
                            pending_bytes.append(mp4_bytes)
                            pending_keys.append(clip_key)
                            if len(pending_bytes) >= batch_size:
                                _decode_batch(pool, pending_bytes, pending_keys)
                                pending_bytes, pending_keys = [], []

                    # Handle final partial batch
                    if pending_bytes and not stop_event.is_set():
                        _decode_batch(pool, pending_bytes, pending_keys)

                    # Loop to next epoch (don't break — training needs multiple passes)

                except (ConnectionError, TimeoutError, OSError) as e:
                    retries += 1
                    if retries > MAX_STREAM_RETRIES:
                        print(f"  FATAL: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                        break
                    wait = min(2 ** retries, 60)
                    print(f"  Stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES} in {wait}s")
                    time.sleep(wait)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        q.put(("error" if retries > MAX_STREAM_RETRIES else "done", None, None))


# ═════════════════════════════════════════════════════════════════════════
# MODEL SETUP (Q1-Q5 corrected: no wrappers, vit_giant_xformers, RoPE)
# ═════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder, teacher encoder (EMA), and predictor."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    arch = model_cfg["arch"]

    vit_constructor = get_vit_by_arch(arch)
    vit_predictor = get_vit_predictor()

    # Student encoder — arch from model config YAML (vit_giant_xformers or vit_gigantic_xformers)
    crop_size = model_cfg["crop_size"]
    student = vit_constructor(
        img_size=(crop_size, crop_size),
        patch_size=data_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=data_cfg["tubelet_size"],
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=model_cfg["use_rope"],
        use_activation_checkpointing=model_cfg["use_activation_checkpointing"],
    )

    # Load pretrained weights — checkpoint path from model config YAML
    project_root = Path(__file__).parent.parent
    ckpt_path = project_root / model_cfg["checkpoint_path"]
    ckpt_url = model_cfg["checkpoint_url"]
    if ckpt_path.exists():
        print(f"Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        print(f"Downloading pretrained weights: {ckpt_url}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu",
                                                    model_dir=str(ckpt_path.parent))

    # Q3: Use target_encoder (EMA teacher = best quality starting point)
    if "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
    elif "encoder" in ckpt:
        state_dict = ckpt["encoder"]
    else:
        state_dict = ckpt

    # Q5: Strip DDP + wrapper prefixes
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in state_dict.items()}

    msg = student.load_state_dict(state_dict, strict=False)
    total_keys = len(list(student.state_dict().keys()))
    loaded_keys = total_keys - len(msg.missing_keys)
    load_pct = loaded_keys / max(total_keys, 1) * 100
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params "
          f"({loaded_keys}/{total_keys} keys = {load_pct:.0f}%)")
    if msg.missing_keys:
        # pos_embed is expected missing when using RoPE
        unexpected_missing = [k for k in msg.missing_keys if "pos_embed" not in k]
        if unexpected_missing:
            print(f"FATAL: {len(unexpected_missing)} unexpected missing keys in student checkpoint:")
            for k in unexpected_missing[:10]:
                print(f"    {k}")
            print("  These layers will be randomly initialized — training results invalid.")
            print("  Fix: verify checkpoint matches model architecture.")
            sys.exit(1)
    min_student_pct = model_cfg["min_student_load_pct"]
    if load_pct < min_student_pct:
        print(f"FATAL: Only {load_pct:.0f}% of student keys loaded. "
              f"Checkpoint likely incompatible.")
        print(f"  Checkpoint keys sample: {list(state_dict.keys())[:3]}")
        print(f"  Model keys sample:      {list(student.state_dict().keys())[:3]}")
        sys.exit(1)
    student = student.to(device)

    # V-JEPA 2.1: student + teacher MUST produce hierarchical output (4 * embed_dim)
    # The 2.1 predictor's predictor_embed expects 4 * 1664 = 6656 input dim.
    # Without this, predictor crashes: RuntimeError mat1 and mat2 shapes cannot be multiplied.
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = True

    # V-JEPA 2.1 requires RoPE (no pos_embed registered in model)
    if model_cfg["predict_all"] or model_cfg.get("n_output_distillation", 1) > 1:
        if not model_cfg["use_rope"]:
            print("FATAL: V-JEPA 2.1 requires use_rope=True (no pos_embed registered in model)")
            sys.exit(1)

    # Teacher (EMA copy, frozen)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("Teacher created (deepcopy of student, hierarchical output enabled)")

    # Predictor: use 2.1 version if predict_all (supports return_all_tokens + proj_context)
    pred_constructor = get_vit_predictor_2_1() if model_cfg["predict_all"] else vit_predictor
    predictor = pred_constructor(
        img_size=(data_cfg["crop_size"], data_cfg["crop_size"]),
        patch_size=data_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=data_cfg["tubelet_size"],
        embed_dim=model_cfg["embed_dim"],
        predictor_embed_dim=model_cfg["pred_embed_dim"],
        depth=model_cfg["pred_depth"],
        num_heads=model_cfg["pred_num_heads"],
        use_mask_tokens=True,
        num_mask_tokens=model_cfg["num_mask_tokens"],
        zero_init_mask_tokens=True,
        use_rope=model_cfg["use_rope"],
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=model_cfg["use_activation_checkpointing"],
        return_all_tokens=model_cfg["predict_all"],
    )

    # Q3: Load predictor weights if available
    if "predictor" in ckpt:
        pred_state = {k.replace("module.", "").replace("backbone.", ""): v
                      for k, v in ckpt["predictor"].items()}
        pred_msg = predictor.load_state_dict(pred_state, strict=False)
        pred_total = len(list(predictor.state_dict().keys()))
        pred_loaded = pred_total - len(pred_msg.missing_keys)
        pred_pct = pred_loaded / max(pred_total, 1) * 100
        print(f"Predictor loaded from checkpoint ({pred_loaded}/{pred_total} keys = {pred_pct:.0f}%)")
        min_pred_pct = model_cfg["min_predictor_load_pct"]
        if pred_pct < min_pred_pct:
            print(f"FATAL: Predictor only {pred_pct:.0f}% loaded ({pred_loaded}/{pred_total} keys).")
            print("  Near-random predictor init will produce garbage predictions → invalid loss.")
            print("  Fix: verify checkpoint has 'predictor' key with matching architecture.")
            sys.exit(1)
    else:
        print("FATAL: Checkpoint has no 'predictor' key — predictor would be randomly initialized.")
        print("  V-JEPA 2 official checkpoint includes predictor weights.")
        print("  This likely means a wrong/corrupt checkpoint file.")
        sys.exit(1)

    predictor = predictor.to(device)
    print(f"Predictor: {sum(p.numel() for p in predictor.parameters()):,} params")

    # Save initial params for drift control (on CPU to save ~4GB VRAM)
    init_params = {name: p.clone().detach().cpu()
                   for name, p in student.named_parameters()}

    del ckpt
    gc.collect()

    # ExPLoRA: LoRA injection + block freezing (after checkpoint loaded, before return)
    explora_cfg = cfg.get("explora")
    explora_enabled = explora_cfg and explora_cfg.get("enabled", False)
    if explora_enabled:
        from peft import get_peft_model, LoraConfig

        # 1. Freeze all student params
        for param in student.parameters():
            param.requires_grad = False

        # 2. Unfreeze first N blocks (ExPLoRA recipe: 1-2)
        n_unfreeze = explora_cfg["unfreeze_blocks"]
        for i in range(n_unfreeze):
            for param in student.blocks[i].parameters():
                param.requires_grad = True

        # 3. Unfreeze all norm layers (ExPLoRA requirement)
        if explora_cfg["unfreeze_norm_layers"]:
            for name, param in student.named_parameters():
                if "norm" in name or "ln" in name:
                    param.requires_grad = True

        # 4. Inject LoRA on frozen attention layers
        lora_config = LoraConfig(
            r=explora_cfg["lora_rank"],
            lora_alpha=explora_cfg["lora_alpha"],
            target_modules=explora_cfg["lora_target_modules"],
            lora_dropout=explora_cfg["lora_dropout"],
            bias="none",
        )
        student = get_peft_model(student, lora_config)

        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        print(f"  ExPLoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

        # 5. Recompute init_params (only frozen non-LoRA params for drift control)
        init_params = {name: p.clone().detach().cpu()
                       for name, p in student.named_parameters()
                       if not p.requires_grad and "lora" not in name}

    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "init_params": init_params,
        "explora_enabled": explora_enabled,
    }


# ═════════════════════════════════════════════════════════════════════════
# MASKING (Q4: Use _MaskGenerator directly, not MaskCollator)
# ═════════════════════════════════════════════════════════════════════════

def build_mask_generators(cfg: dict) -> list:
    """Build one _MaskGenerator per mask config."""
    _MaskGenerator = get_mask_generator()

    data_cfg = cfg["data"]
    generators = []
    for m_cfg in cfg["mask"]:
        mg = _MaskGenerator(
            crop_size=(data_cfg["crop_size"], data_cfg["crop_size"]),
            num_frames=data_cfg["num_frames"],
            spatial_patch_size=(data_cfg["patch_size"], data_cfg["patch_size"]),
            temporal_patch_size=data_cfg["tubelet_size"],
            spatial_pred_mask_scale=m_cfg["spatial_scale"],
            temporal_pred_mask_scale=m_cfg["temporal_scale"],
            aspect_ratio=m_cfg["aspect_ratio"],
            npred=m_cfg["num_blocks"],
        )
        generators.append(mg)
    return generators


# ═════════════════════════════════════════════════════════════════════════
# LOSS (L1 latent prediction + drift control)
# ═════════════════════════════════════════════════════════════════════════

def compute_jepa_loss(pred_features: list, pred_context: list,
                      teacher_output: torch.Tensor,
                      masks_pred: list, masks_enc: list,
                      loss_exp: float, predict_all: bool,
                      lambda_context: float = 0.5) -> tuple:
    """V-JEPA 2.1 dense loss: masked tokens + context tokens.

    When predict_all=True (V-JEPA 2.1), computes loss on ALL tokens:
    - Masked loss: predictor output vs teacher at masked positions
    - Context loss: predictor context output vs teacher at visible positions
    - Total = masked_loss + lambda_context * context_loss

    Returns (total_loss, loss_masked, loss_context).
    """
    apply_masks = get_apply_masks()

    # Masked token loss
    loss_masked = torch.tensor(0.0, device=teacher_output.device)
    n = 0
    for pred_i, mask_i in zip(pred_features, masks_pred):
        h_masked = apply_masks(teacher_output, [mask_i])
        loss_masked += torch.mean(torch.abs(pred_i - h_masked) ** loss_exp) / loss_exp
        n += 1
    loss_masked = loss_masked / max(n, 1)

    # Context token loss (V-JEPA 2.1 dense loss — doubles training signal)
    loss_context = torch.tensor(0.0, device=teacher_output.device)
    if predict_all and pred_context:
        n_ctx = 0
        for ctx_i, mask_enc_i in zip(pred_context, masks_enc):
            if ctx_i is None:
                continue
            h_context = apply_masks(teacher_output, [mask_enc_i])
            loss_context += torch.mean(torch.abs(ctx_i - h_context) ** loss_exp) / loss_exp
            n_ctx += 1
        loss_context = loss_context / max(n_ctx, 1)

    total = loss_masked + lambda_context * loss_context
    return total, loss_masked, loss_context


def compute_drift_loss(student: torch.nn.Module, init_params: dict,
                       lambda_reg: float) -> torch.Tensor:
    """L2 drift control: lambda * ||theta - theta_0||^2 (Sec 10.4)."""
    device = next(student.parameters()).device
    drift = torch.tensor(0.0, device=device)
    for name, param in student.named_parameters():
        if name in init_params:
            drift += torch.sum((param - init_params[name].to(device)) ** 2)
    return lambda_reg * drift


# ═════════════════════════════════════════════════════════════════════════
# EMA TEACHER UPDATE
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_teacher_ema(student: torch.nn.Module, teacher: torch.nn.Module,
                       momentum: float):
    """EMA update: theta_bar <- tau * theta_bar + (1-tau) * theta.

    Uses name-based matching so ExPLoRA LoRA params (which only exist in student,
    not teacher) are safely skipped.
    """
    student_dict = dict(student.named_parameters())
    for name, param_t in teacher.named_parameters():
        if name in student_dict:
            param_t.mul_(momentum).add_(student_dict[name].data, alpha=1.0 - momentum)


# ═════════════════════════════════════════════════════════════════════════
# OPTIMIZER & SCHEDULER
# ═════════════════════════════════════════════════════════════════════════

def build_optimizer(student, predictor, cfg_opt: dict):
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
    return torch.optim.AdamW(param_groups, betas=tuple(cfg_opt["betas"]),
                              eps=cfg_opt["eps"])


def build_scheduler(optimizer, cfg_opt: dict, total_steps: int):
    # Cap warmup to 10% of total steps (avoids never leaving warmup on short runs)
    warmup_cap_pct = cfg_opt["warmup_cap_pct"]
    warmup_steps = min(cfg_opt["warmup_steps"], max(1, total_steps * warmup_cap_pct // 100))
    min_lr = cfg_opt["min_lr"]
    base_lr = cfg_opt["lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def update_weight_decay(optimizer, cfg_opt: dict, step: int, total_steps: int):
    """Cosine weight decay schedule (matches Meta's CosineWDSchedule).

    Ramps WD from weight_decay to final_weight_decay over training.
    If both are equal (default: 0.04), this is a no-op.
    """
    initial_wd = cfg_opt["weight_decay"]
    final_wd = cfg_opt["final_weight_decay"]
    if initial_wd == final_wd:
        return  # no-op: fixed WD
    progress = step / max(total_steps, 1)
    new_wd = final_wd + 0.5 * (initial_wd - final_wd) * (1 + np.cos(np.pi * progress))
    for group in optimizer.param_groups:
        if group.get("weight_decay", 0) > 0:  # don't touch no_decay groups
            group["weight_decay"] = new_wd


# ═════════════════════════════════════════════════════════════════════════
# VALIDATION (JEPA L1 loss on held-out val clips)
# ═════════════════════════════════════════════════════════════════════════

def run_validation(student, teacher, predictor, mask_generators,
                   val_batches: list, cfg: dict, device: torch.device,
                   step: int) -> float:
    """Compute JEPA L1 loss on val clips. Same as training loss, different data.

    Returns mean val loss (scalar).
    """
    student.eval()
    predictor.eval()

    mp_cfg = cfg["mixed_precision"]
    dtype = getattr(torch, mp_cfg["dtype"])
    loss_exp = cfg["optimization"]["loss_exp"]
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_clips in val_batches:
            batch_clips = batch_clips.to(device)
            actual_bs = batch_clips.shape[0]

            # Generate masks (same as training)
            all_masks_enc, all_masks_pred = [], []
            for mg in mask_generators:
                m_enc, m_pred = mg(actual_bs)
                all_masks_enc.append(m_enc.to(device))
                all_masks_pred.append(m_pred.to(device))

            with torch.amp.autocast("cuda", dtype=dtype, enabled=mp_cfg["enabled"]):
                # Teacher: all tokens + per-chunk LayerNorm (deep supervision)
                embed_dim_val = cfg["model"]["embed_dim"]
                n_levels_val = cfg["model"].get("n_output_distillation", 4)
                h = teacher(batch_clips)
                if h.size(-1) == n_levels_val * embed_dim_val:
                    chunks = []
                    for lvl in range(n_levels_val):
                        chunk = h[:, :, lvl * embed_dim_val : (lvl + 1) * embed_dim_val]
                        chunks.append(F.layer_norm(chunk, (embed_dim_val,)))
                    h = torch.cat(chunks, dim=2)
                else:
                    h = F.layer_norm(h, (h.size(-1),))

                # Student + Predictor per mask generator
                predict_all_val = cfg["model"]["predict_all"]
                pred_features = []
                pred_context_val = []
                for i, (m_enc, m_pred) in enumerate(zip(all_masks_enc, all_masks_pred)):
                    z = student(batch_clips, masks=[m_enc])
                    outputs = predictor(z, [m_enc], [m_pred], mask_index=i)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        pred_features.append(outputs[0])
                        pred_context_val.append(outputs[1])
                    else:
                        pred_features.append(outputs)

                # Dense loss (same as training, no drift control)
                jepa_loss, _, _ = compute_jepa_loss(
                    pred_features, pred_context_val, h,
                    all_masks_pred, all_masks_enc,
                    loss_exp, predict_all_val, lambda_context=0.5)

            total_loss += jepa_loss.item()
            n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    print(f"  Step {step} | Val JEPA loss: {val_loss:.4f} ({len(val_batches)} batches)")

    student.train()
    predictor.train()
    return val_loss


# ═════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════

def save_training_checkpoint(path: Path, student, teacher, predictor,
                              optimizer, scheduler, scaler,
                              step: int, best_metric: float,
                              full: bool = True):
    """Save checkpoint. full=True includes optimizer (for resume), False is light (for selection)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    ckpt = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "predictor": predictor.state_dict(),
        "step": step,
        "best_metric": best_metric,
    }
    if full:
        ckpt["optimizer"] = optimizer.state_dict()
        ckpt["scheduler"] = scheduler.state_dict()
        ckpt["scaler"] = scaler.state_dict() if scaler else None
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def cleanup_old_checkpoints(output_dir: Path, keep_n: int = 2):
    """Delete oldest m09_ckpt_step*.pt files, keeping only the last keep_n."""
    pattern = str(output_dir / f"{CHECKPOINT_PREFIX}_step*.pt")
    step_files = sorted(glob.glob(pattern),
                        key=lambda f: os.path.getmtime(f))
    if len(step_files) > keep_n:
        for old_file in step_files[:-keep_n]:
            os.remove(old_file)
            print(f"  Cleaned old checkpoint: {Path(old_file).name}")


def load_training_checkpoint(path: Path, student, teacher, predictor,
                              optimizer, scheduler, scaler) -> tuple:
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    predictor.load_state_dict(ckpt["predictor"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt.get("best_metric", float("inf"))


def export_student_for_eval(student, path: Path, explora_enabled: bool = False):
    """Export student encoder weights + metadata for m05/m06 re-evaluation."""
    from utils.config import VJEPA_MODEL_ID
    path.parent.mkdir(parents=True, exist_ok=True)
    # ExPLoRA: merge LoRA weights into base model before saving
    export_model = student
    if explora_enabled and hasattr(student, "merge_and_unload"):
        print("  Merging LoRA weights into base model for export...")
        export_model = student.merge_and_unload()
    torch.save({
        "student_state_dict": export_model.state_dict(),
        "model_id": VJEPA_MODEL_ID,
        "type": "vjepa2_adapted",
    }, path)
    print(f"Exported student encoder: {path}")


# ═════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════

def train(cfg: dict, args):
    """Epoch-based training loop (proposal Sec 10.5)."""
    check_gpu()
    device = torch.device("cuda")

    # Reproducibility seeds (matches Meta's train.py lines 45-48, 152-154)
    seed = cfg["data"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Disable auto GC, collect manually every 50 steps (matches Meta's sync_gc pattern)
    gc.disable()
    gc.collect()

    # Output-exists guard
    from utils.output_guard import verify_training_output
    output_dir = Path(cfg["checkpoint"]["output_dir"])
    student_path = output_dir / "student_encoder.pt"
    if verify_training_output(output_dir, min_epochs=cfg["optimization"]["max_epochs"]):
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    print("\n=== Building Model ===")
    models = build_model(cfg, device)
    student = models["student"]
    teacher = models["teacher"]
    predictor = models["predictor"]
    init_params = models["init_params"]

    student.train()
    predictor.train()

    # Build mask generators
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)} "
          f"(blocks: {[m['num_blocks'] for m in cfg['mask']]})")

    # Compute epoch geometry
    # Train: --subset keys minus val keys. Val: --val-subset keys (pre-sampled via m00c).
    subset_keys = load_subset(args.subset) if args.subset else set()
    val_key_set = load_val_subset(args.val_subset)

    # Exclude val keys from training (no leakage)
    if subset_keys and val_key_set:
        overlap = subset_keys & val_key_set
        if overlap:
            print(f"Excluding {len(overlap)} val clips from training set")
            train_keys = subset_keys - val_key_set
        else:
            train_keys = subset_keys
    elif subset_keys:
        train_keys = subset_keys
    else:
        train_keys = set()

    # SANITY: cap train and val to small clip counts from config
    if args.SANITY:
        sanity_train = cfg["data"]["sanity_train_clips"]
        sanity_val = cfg["data"]["sanity_val_clips"]
        if train_keys:
            train_keys = set(list(train_keys)[:sanity_train])
        else:
            local_data = cfg["data"]["local_data"]
            ds = _create_stream(0, local_data=local_data)
            collected = []
            for example in ds:
                collected.append(get_clip_key(example))
                if len(collected) >= sanity_train + sanity_val:
                    break
            train_keys = set(collected[:sanity_train])
            val_key_set = set(collected[sanity_train:sanity_train + sanity_val])
        if val_key_set:
            val_key_set = set(list(val_key_set)[:sanity_val])

    # Discover n_train: from subset, local manifest, or fail
    if train_keys:
        n_train = len(train_keys)
    else:
        # Full dataset: read clip count from local manifest
        local_data = cfg["data"]["local_data"]
        manifest_path = Path(local_data) / "manifest.json" if local_data else None
        if manifest_path and manifest_path.exists():
            manifest = json.load(open(manifest_path))
            n_total = manifest.get("n") or manifest.get("n_clips") or manifest.get("total_clips")
            n_train = n_total - len(val_key_set)
            print(f"Dataset size from manifest: {n_total:,} total, {n_train:,} train "
                  f"({len(val_key_set)} val excluded)")
        else:
            print("FATAL: Cannot determine dataset size.")
            print("Provide --subset <file> or --local-data <dir> with manifest.json")
            sys.exit(1)
    batch_size = cfg["optimization"]["batch_size"]
    max_epochs = cfg["optimization"]["max_epochs"]
    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = steps_per_epoch * max_epochs
    saves_per_epoch = cfg["checkpoint"]["saves_per_epoch"]
    ckpt_interval = max(1, steps_per_epoch // saves_per_epoch)
    keep_last_n = cfg["checkpoint"]["keep_last_n"]
    evals_per_epoch = cfg["validation"]["evals_per_epoch"]
    val_interval = max(1, steps_per_epoch // evals_per_epoch)

    print(f"Train clips: {n_train:,} | Val clips: {len(val_key_set):,}")
    print(f"Epochs: {max_epochs} | Steps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(f"Checkpoint every {ckpt_interval} steps ({saves_per_epoch}x/epoch, keep last {keep_last_n})")
    print(f"Validation every {val_interval} steps ({evals_per_epoch}x/epoch, {len(val_key_set)} val clips)")

    # Optimizer & scheduler (cosine over total_steps)
    optimizer = build_optimizer(student, predictor, cfg["optimization"])
    scheduler = build_scheduler(optimizer, cfg["optimization"], total_steps)
    mp_cfg = cfg["mixed_precision"]
    use_scaler = mp_cfg["enabled"] and mp_cfg["dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float("inf")
    ckpt_path = output_dir / f"{CHECKPOINT_PREFIX}_latest.pt"
    if ckpt_path.exists():
        start_step, best_val_loss = load_training_checkpoint(
            ckpt_path, student, teacher, predictor, optimizer, scheduler, scaler)
        print(f"Resumed from step {start_step}, best val loss: {best_val_loss:.4f}")

    # ── Collect val clips into memory (from --val-subset, once before training) ──
    val_batches = []
    val_collected_keys = []

    if val_key_set:
        print(f"\nCollecting {len(val_key_set)} val clips into memory...")
        _val_tmp = tempfile.mkdtemp(prefix="m09_val_")
        # Use --val-local-data if provided, otherwise fall back to --local-data
        val_local = getattr(args, "val_local_data", None) or cfg["data"]["local_data"]
        _val_ds = _create_stream(0, local_data=val_local)
        _val_batch_buf = []
        for _ex in _val_ds:
            _ck = get_clip_key(_ex)
            if _ck not in val_key_set:
                continue
            _mp4 = _ex.get("mp4", b"")
            _mp4b = _mp4["bytes"] if isinstance(_mp4, dict) else _mp4
            if not _mp4b:
                continue
            _vt = decode_video_bytes(_mp4b, _val_tmp, _ck, cfg["data"]["num_frames"])
            if _vt is None:
                continue
            _aug = augment_clip_consistent(_vt, cfg["augmentation"], cfg["data"]["crop_size"])
            _val_batch_buf.append(_aug)
            val_collected_keys.append(_ck)
            if len(_val_batch_buf) >= batch_size:
                _batch = torch.stack(_val_batch_buf, dim=0).permute(0, 2, 1, 3, 4)
                val_batches.append(_batch)
                _val_batch_buf = []
            if len(val_collected_keys) >= len(val_key_set):
                break
        if _val_batch_buf:
            _batch = torch.stack(_val_batch_buf, dim=0).permute(0, 2, 1, 3, 4)
            val_batches.append(_batch)
        shutil.rmtree(_val_tmp, ignore_errors=True)
        print(f"Val clips collected: {len(val_collected_keys)} in {len(val_batches)} batches")
        if len(val_collected_keys) < len(val_key_set):
            pct = len(val_collected_keys) / len(val_key_set) * 100
            print(f"WARNING: Only {len(val_collected_keys)}/{len(val_key_set)} val clips ({pct:.0f}%). Auto-downloading...")
            import subprocess
            subprocess.run([sys.executable, "-u", "src/m00d_download_subset.py",
                           "--FULL", "--subset", args.val_subset, "--no-wandb"], check=True)
            # Retry collection with fresh data
            val_batches = []
            val_collected_keys = []
            _val_tmp = tempfile.mkdtemp(prefix="m09_val_retry_")
            _val_ds = _create_stream(0, local_data=val_local)
            _val_batch_buf = []
            for _ex in _val_ds:
                _ck = get_clip_key(_ex)
                if _ck not in val_key_set:
                    continue
                _mp4 = _ex.get("mp4", b"")
                _mp4b = _mp4["bytes"] if isinstance(_mp4, dict) else _mp4
                if not _mp4b:
                    continue
                _vt = decode_video_bytes(_mp4b, _val_tmp, _ck, cfg["data"]["num_frames"])
                if _vt is None:
                    continue
                _aug = augment_clip_consistent(_vt, cfg["augmentation"], cfg["data"]["crop_size"])
                _val_batch_buf.append(_aug)
                val_collected_keys.append(_ck)
                if len(_val_batch_buf) >= batch_size:
                    _batch = torch.stack(_val_batch_buf, dim=0).permute(0, 2, 1, 3, 4)
                    val_batches.append(_batch)
                    _val_batch_buf = []
                if len(val_collected_keys) >= len(val_key_set):
                    break
            if _val_batch_buf:
                _batch = torch.stack(_val_batch_buf, dim=0).permute(0, 2, 1, 3, 4)
                val_batches.append(_batch)
            shutil.rmtree(_val_tmp, ignore_errors=True)
            print(f"Val clips collected (retry): {len(val_collected_keys)} in {len(val_batches)} batches")
    else:
        if not getattr(args, "SANITY", False):
            print("No --val-subset provided. Attempting auto-download of val data...")
            try:
                from utils.hf_outputs import download_data
                download_data()
                # Check if val data now exists
                val_path = Path("data/val_1k.json")
                val_local_path = Path("data/val_1k_local")
                if val_path.exists() and val_local_path.exists():
                    print(f"Auto-downloaded val data. Loading {val_path}...")
                    val_key_set = load_val_subset(str(val_path))
                    args.val_local_data = str(val_local_path)
                    # Re-run val collection with downloaded data
                    _val_tmp2 = tempfile.mkdtemp(prefix="m09_val_")
                    _val_ds2 = _create_stream(0, local_data=str(val_local_path))
                    _val_batch_buf2 = []
                    for _ex in _val_ds2:
                        _ck = get_clip_key(_ex)
                        if _ck not in val_key_set:
                            continue
                        _mp4 = _ex.get("mp4", b"")
                        _mp4b = _mp4["bytes"] if isinstance(_mp4, dict) else _mp4
                        if not _mp4b:
                            continue
                        _vt = decode_video_bytes(_mp4b, _val_tmp2, _ck, cfg["data"]["num_frames"])
                        if _vt is None:
                            continue
                        _aug = augment_clip_consistent(_vt, cfg["augmentation"], cfg["data"]["crop_size"])
                        _val_batch_buf2.append(_aug)
                        val_collected_keys.append(_ck)
                        if len(_val_batch_buf2) >= batch_size:
                            _batch = torch.stack(_val_batch_buf2, dim=0).permute(0, 2, 1, 3, 4)
                            val_batches.append(_batch)
                            _val_batch_buf2 = []
                        if len(val_collected_keys) >= len(val_key_set):
                            break
                    if _val_batch_buf2:
                        _batch = torch.stack(_val_batch_buf2, dim=0).permute(0, 2, 1, 3, 4)
                        val_batches.append(_batch)
                    shutil.rmtree(_val_tmp2, ignore_errors=True)
                    print(f"Val clips collected: {len(val_collected_keys)} in {len(val_batches)} batches")
                else:
                    print("FATAL: Auto-download failed — val data still missing.")
                    print("  Fix: python -u src/utils/hf_outputs.py download-data")
                    sys.exit(1)
            except Exception as e:
                print(f"FATAL: Auto-download failed ({e}). Val data required for non-SANITY runs.")
                print("  Fix: python -u src/utils/hf_outputs.py download-data")
                sys.exit(1)
        else:
            # SANITY mode: still validate with first 50 clips (tests validation code path)
            print("SANITY mode — auto-downloading val data for validation code path test...")
            try:
                from utils.hf_outputs import download_data
                download_data()
                val_path = Path("data/val_1k.json")
                val_local_path = Path("data/val_1k_local")
                if val_path.exists():
                    val_key_set = load_val_subset(str(val_path))
                    # Use only first 50 keys for SANITY speed
                    sanity_val_cap = cfg["data"]["sanity_val_clips"]
                    val_key_set = set(list(val_key_set)[:sanity_val_cap])
                    val_local = str(val_local_path) if val_local_path.exists() else cfg["data"]["local_data"]
                    _val_tmp3 = tempfile.mkdtemp(prefix="m09_val_")
                    _val_ds3 = _create_stream(0, local_data=val_local)
                    _val_batch_buf3 = []
                    for _ex in _val_ds3:
                        _ck = get_clip_key(_ex)
                        if _ck not in val_key_set:
                            continue
                        _mp4 = _ex.get("mp4", b"")
                        _mp4b = _mp4["bytes"] if isinstance(_mp4, dict) else _mp4
                        if not _mp4b:
                            continue
                        _vt = decode_video_bytes(_mp4b, _val_tmp3, _ck, cfg["data"]["num_frames"])
                        if _vt is None:
                            continue
                        _aug = augment_clip_consistent(_vt, cfg["augmentation"], cfg["data"]["crop_size"])
                        _val_batch_buf3.append(_aug)
                        val_collected_keys.append(_ck)
                        if len(_val_batch_buf3) >= batch_size:
                            _batch = torch.stack(_val_batch_buf3, dim=0).permute(0, 2, 1, 3, 4)
                            val_batches.append(_batch)
                            _val_batch_buf3 = []
                        if len(val_collected_keys) >= sanity_val_cap:
                            break
                    if _val_batch_buf3:
                        _batch = torch.stack(_val_batch_buf3, dim=0).permute(0, 2, 1, 3, 4)
                        val_batches.append(_batch)
                    shutil.rmtree(_val_tmp3, ignore_errors=True)
                    print(f"SANITY val clips: {len(val_collected_keys)} in {len(val_batches)} batches")
                else:
                    print("FATAL: SANITY val data download failed — val_1k.json not found after download.")
                    print("  Fix: python -u src/utils/hf_outputs.py download-data")
                    sys.exit(1)
            except Exception as e:
                print(f"FATAL: SANITY val data auto-download failed ({e})")
                print("  Fix: python -u src/utils/hf_outputs.py download-data")
                sys.exit(1)

    # Data stream (producer loops over epochs automatically)
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()
    prod = threading.Thread(
        target=producer_thread,
        args=(cfg, q, stop_event, train_keys, start_step),
        daemon=True,
    )
    prod.start()

    # WandB — include lambda in run name for ablation comparison
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    lam_val = cfg["drift_control"]["lambda_reg"]
    lam_tag = f"_lambda{f'{lam_val:g}'.replace('.', '_')}"
    wb_run = init_wandb("m09", f"{mode}{lam_tag}", config=cfg, enabled=not args.no_wandb)

    # Training config
    ema_momentum = cfg["optimization"]["ema_momentum"]
    loss_exp = cfg["optimization"]["loss_exp"]
    drift_cfg = cfg["drift_control"]
    dtype = getattr(torch, mp_cfg["dtype"])

    pbar = tqdm(total=total_steps, initial=start_step,
                desc="m09_pretrain", unit="step")

    # JSONL loss log — crash-safe (fsync after every write, survives OOM/SIGKILL)
    # Each line is a self-contained JSON record. Partial last line = only that step lost.
    jsonl_path = output_dir / "loss_log.jsonl"
    jsonl_file = open(jsonl_path, "a")

    def _log_step(record: dict):
        """Write one JSON record + flush + fsync (Detectron2 pattern)."""
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()
        os.fsync(jsonl_file.fileno())

    # Also keep CSV for backward compat (plots, wandb upload)
    csv_path = output_dir / "loss_log.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["step", "epoch", "loss_jepa", "loss_drift", "loss_total",
                             "lr", "grad_norm", "throughput", "val_loss"])
        csv_file.flush()

    # Windowed throughput
    window_start = time.time()
    window_steps = 0
    running_loss = 0.0

    print(f"\n=== Training: {start_step} → {total_steps} steps ({max_epochs} epochs) ===")
    print(f"Batch size: {batch_size}")
    print(f"Grad checkpointing: {cfg['model']['use_activation_checkpointing']}")
    print(f"Mixed precision: {mp_cfg['dtype']}")
    print(f"EMA momentum: {ema_momentum}")
    print(f"Drift control: lambda={drift_cfg['lambda_reg'] if drift_cfg['enabled'] else 0}")
    print(f"Loss log: {csv_path}")

    step = start_step
    try:
        for step in range(start_step, total_steps):
            epoch = step // steps_per_epoch
            epoch_step = step % steps_per_epoch

            # Get batch from producer
            try:
                msg_type, batch_clips, batch_keys = q.get(timeout=600)
            except queue.Empty:
                print(f"Producer timeout at step {step}/{total_steps}. "
                      f"GPU idle for 10 min — saving checkpoint and stopping gracefully.")
                break
            if msg_type == "error":
                print(f"FATAL: Producer stream failed at step {step}/{total_steps}. "
                      f"Data source unreachable after {MAX_STREAM_RETRIES} retries.")
                print("  Checkpoint saved. Check network/disk and resume.")
                sys.exit(1)
            if msg_type == "done":
                pct_done = (step + 1) / total_steps * 100
                if pct_done < 95:
                    print(f"\n  Data exhausted at step {step}/{total_steps} ({pct_done:.0f}%). "
                          f"Training {100-pct_done:.0f}% incomplete — finishing with available data.")
                else:
                    print(f"\nData exhausted at step {step}/{total_steps} ({pct_done:.0f}%).")
                break

            batch_clips = batch_clips.to(device)
            actual_bs = batch_clips.shape[0]

            # Generate masks
            all_masks_enc, all_masks_pred = [], []
            for mg in mask_generators:
                m_enc, m_pred = mg(actual_bs)
                all_masks_enc.append(m_enc.to(device))
                all_masks_pred.append(m_pred.to(device))

            # Forward pass (zero_grad after step, matching Meta's pattern)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=mp_cfg["enabled"]):
                # Teacher: all tokens, no grad
                # V-JEPA 2.1 deep supervision: teacher returns (B, N, 4*D) hierarchical
                # Per-chunk LayerNorm (Meta's train.py lines 597-607)
                embed_dim = cfg["model"]["embed_dim"]
                n_levels = cfg["model"].get("n_output_distillation", 4)
                with torch.no_grad():
                    h = teacher(batch_clips)
                    if h.size(-1) == n_levels * embed_dim:
                        chunks = []
                        for lvl in range(n_levels):
                            chunk = h[:, :, lvl * embed_dim : (lvl + 1) * embed_dim]
                            chunks.append(F.layer_norm(chunk, (embed_dim,)))
                        h = torch.cat(chunks, dim=2)
                    else:
                        h = F.layer_norm(h, (h.size(-1),))

                predict_all = cfg["model"]["predict_all"]
                pred_features = []
                pred_context = []
                for i, (m_enc, m_pred) in enumerate(zip(all_masks_enc, all_masks_pred)):
                    z = student(batch_clips, masks=[m_enc])
                    outputs = predictor(z, [m_enc], [m_pred], mask_index=i)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        pred_features.append(outputs[0])
                        pred_context.append(outputs[1])
                    else:
                        pred_features.append(outputs)

                jepa_loss, loss_masked, loss_context = compute_jepa_loss(
                    pred_features, pred_context, h,
                    all_masks_pred, all_masks_enc,
                    loss_exp, predict_all, lambda_context=0.5)

                if drift_cfg["enabled"] and drift_cfg["lambda_reg"] > 0:
                    drift_loss = compute_drift_loss(student, init_params,
                                                    drift_cfg["lambda_reg"])
                    total_loss = jepa_loss + drift_loss
                else:
                    drift_loss = torch.tensor(0.0)
                    total_loss = jepa_loss

            # Backward + optimizer step
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(predictor.parameters()),
                cfg["optimization"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            update_weight_decay(optimizer, cfg["optimization"], step, total_steps)

            # EMA teacher update
            update_teacher_ema(student, teacher, ema_momentum)

            # Periodic GC (every 50 steps, matches Meta's sync_gc pattern)
            if step % cfg["optimization"]["gc_interval"] == 0:
                gc.collect()

            # Per-step logging + NaN/Inf guard
            jepa_val = jepa_loss.item()
            masked_val = loss_masked.item()
            context_val = loss_context.item()
            drift_val = drift_loss.item()
            total_val = total_loss.item()
            lr_val = scheduler.get_last_lr()[0]
            gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            if math.isnan(jepa_val) or math.isinf(jepa_val):
                nan_strikes = getattr(main, '_nan_strikes', 0) + 1
                main._nan_strikes = nan_strikes
                print(f"  NaN/Inf loss at step {step} (strike {nan_strikes}/3). "
                      f"GradScaler will auto-adjust.")
                if nan_strikes >= cfg["optimization"]["nan_tolerance"]:
                    print("FATAL: 3 consecutive NaN/Inf losses. Model diverged.")
                    print("  Debug: check loss_log.csv for divergence point.")
                    sys.exit(1)
            else:
                main._nan_strikes = 0  # reset on valid loss

            if math.isnan(gn_val) or math.isinf(gn_val):
                print(f"  Gradient norm {gn_val} at step {step} — GradScaler auto-skipped this step.")

            running_loss += total_val
            window_steps += 1
            now = time.time()
            window_elapsed = now - window_start
            throughput = window_steps / window_elapsed if window_elapsed > 0 else 0

            step_record = {
                "step": step, "epoch": epoch,
                "loss_jepa": round(jepa_val, 6),
                "loss_masked": round(masked_val, 6),
                "loss_context": round(context_val, 6),
                "loss_drift": round(drift_val, 6),
                "loss_total": round(total_val, 6), "lr": lr_val,
                "grad_norm": round(gn_val, 4), "throughput": round(throughput, 2),
            }
            _log_step(step_record)  # JSONL: crash-safe (fsync per write)

            csv_writer.writerow([step, epoch, f"{jepa_val:.6f}", f"{drift_val:.6f}",
                                 f"{total_val:.6f}", f"{lr_val:.2e}",
                                 f"{gn_val:.4f}", f"{throughput:.2f}", ""])
            if step % 10 == 0:
                csv_file.flush()

            log_metrics(wb_run, {
                "loss/jepa": jepa_val,
                "loss/masked": masked_val,
                "loss/context": context_val,
                "loss/drift": drift_val,
                "loss/total": total_val,
                "lr": lr_val,
                "grad_norm": gn_val,
                "epoch": epoch,
                "throughput_steps_per_s": throughput,
            }, step=step)

            if window_elapsed >= 30:
                pbar.set_postfix_str(
                    f"E{epoch} loss={running_loss/window_steps:.4f} "
                    f"drift={drift_val:.4f} "
                    f"lr={lr_val:.2e} "
                    f"grad={gn_val:.2f} "
                    f"{throughput:.2f} step/s")
                window_start = now
                window_steps = 0
                running_loss = 0.0

            pbar.update(1)

            # Periodic checkpoint (every ckpt_interval steps)
            if (step + 1) % ckpt_interval == 0:
                pct = (epoch_step + 1) / steps_per_epoch * 100
                print(f"\n--- Checkpoint at {pct:.0f}% of epoch {epoch + 1} (step {step + 1}) ---")
                save_training_checkpoint(
                    output_dir / f"{CHECKPOINT_PREFIX}_step{step+1}.pt",
                    student, teacher, predictor, optimizer, scheduler, scaler,
                    step + 1, best_val_loss, full=False)
                save_training_checkpoint(
                    ckpt_path, student, teacher, predictor, optimizer, scheduler,
                    scaler, step + 1, best_val_loss, full=True)
                cleanup_old_checkpoints(output_dir, keep_n=keep_last_n)

            # Periodic validation (every val_interval steps)
            if (step + 1) % val_interval == 0 and val_batches:
                pct = (epoch_step + 1) / steps_per_epoch * 100
                val_loss = run_validation(
                    student, teacher, predictor, mask_generators,
                    val_batches, cfg, device, step + 1)
                log_metrics(wb_run, {
                    "val/jepa_loss": val_loss,
                    "val/epoch_pct": pct,
                }, step=step)

                # Write val_loss to JSONL (crash-safe) + CSV
                _log_step({"step": step, "epoch": epoch, "val_loss": round(val_loss, 6),
                           "epoch_pct": round(pct, 1)})
                csv_writer.writerow([step, epoch, "", "", "",
                                     "", "", "", f"{val_loss:.6f}"])
                csv_file.flush()

                # Live training plots (regenerate on each validation)
                try:
                    from utils.plots import plot_training_curves
                    plot_training_curves(
                        runs=[{"csv_path": str(csv_path),
                               "label": f"V-JEPA 2.0 (\u03bb={drift_cfg['lambda_reg']})",
                               "color": "blue",
                               "batch_size": batch_size}],
                        output_dir=str(output_dir),
                        title_prefix=f"{n_train:,} clips, ",
                    )
                except Exception:
                    pass  # non-fatal: plot failure must never stop training

                # Best model selection by lowest val loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_training_checkpoint(
                        output_dir / f"{CHECKPOINT_PREFIX}_best.pt",
                        student, teacher, predictor, optimizer, scheduler,
                        scaler, step + 1, best_val_loss, full=False)
                    print(f"  New best val loss: {best_val_loss:.4f}")

            # End-of-epoch logging
            if epoch_step == steps_per_epoch - 1:
                print(f"\n--- Epoch {epoch + 1}/{max_epochs} complete (step {step + 1}) ---")
                log_metrics(wb_run, {"epoch_complete": epoch + 1}, step=step)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint (model NOT exported — training incomplete).")
        pbar.close()
        csv_file.close()
        jsonl_file.close()
        stop_event.set()
        gc.enable()
        save_training_checkpoint(
            ckpt_path, student, teacher, predictor, optimizer, scheduler,
            scaler, step + 1, best_val_loss, full=True)
        print(f"  Checkpoint saved at step {step + 1}/{total_steps}. Resume with same command.")
        sys.exit(0)  # user-initiated, not an error
    finally:
        pbar.close()
        csv_file.close()
        jsonl_file.close()
        stop_event.set()
        gc.enable()

    # Cooldown phase: switch to 64f, linear LR decay (V-JEPA 2.1 recipe)
    cooldown_cfg = cfg.get("cooldown", {})
    if cooldown_cfg.get("enabled") and not args.SANITY:
        print("\n=== COOLDOWN PHASE ===")
        print(f"  Frames: {cfg['data']['num_frames']} → {cooldown_cfg['num_frames']}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e} → {cooldown_cfg['final_lr']}")
        print(f"  Epochs: {cooldown_cfg['epochs']}")
        print(f"  Warmup: {cooldown_cfg['warmup_steps']} steps")
        # Store current LR for linear decay
        current_lr = scheduler.get_last_lr()[0]
        cooldown_final_lr = cooldown_cfg["final_lr"]
        cooldown_steps = int(steps_per_epoch * cooldown_cfg["epochs"])
        # Update num_frames in config for producer
        cfg["data"]["num_frames"] = cooldown_cfg["num_frames"]
        # Linear decay: LR drops from current to final over cooldown_steps
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        for cd_step in range(cooldown_steps):
            frac = cd_step / max(cooldown_steps - 1, 1)
            cd_lr = current_lr + frac * (cooldown_final_lr - current_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cd_lr
            # Note: full cooldown training loop (data loading, forward, backward)
            # would require restarting the producer thread with 64f.
            # For now, log the schedule; full implementation needs producer restart.
        print(f"  Cooldown LR schedule: {current_lr:.2e} → {cooldown_final_lr:.2e} "
              f"over {cooldown_steps} steps (linear)")
        print("  NOTE: Full cooldown with 64f data loading requires producer restart.")
        print("  Cooldown is a paper-quality enhancement. POC results valid without it.")

    # Export student encoder (the only deliverable — only reached if training completed)
    explora_en = cfg.get("explora", {}).get("enabled", False)
    export_student_for_eval(student, student_path, explora_enabled=explora_en)

    # Cleanup ALL checkpoints — student_encoder.pt is the deliverable
    for ckpt_file in output_dir.glob(f"{CHECKPOINT_PREFIX}_*.pt"):
        ckpt_file.unlink()
        print(f"  Cleaned: {ckpt_file.name}")

    # Save training summary as JSON (for winner selection, not stdout parsing)
    total_epochs_done = (step + 1) / max(steps_per_epoch, 1)
    summary = {
        "steps": step + 1,
        "epochs": round(total_epochs_done, 2),
        "clips_seen": (step + 1) * batch_size,
        "final_jepa_loss": jepa_val,
        "final_drift_loss": drift_val,
        "final_total_loss": total_val,
        "best_val_loss": best_val_loss,
        "final_lr": lr_val,
        "final_grad_norm": gn_val,
        "lambda_reg": drift_cfg["lambda_reg"],
        "batch_size": batch_size,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    finish_wandb(wb_run)
    print("\n=== TRAINING COMPLETE ===")
    print(f"Steps:        {step + 1} ({total_epochs_done:.1f} epochs)")
    print(f"Clips seen:   {(step + 1) * batch_size:,}")
    print(f"Student:      {student_path}")
    print(f"Loss log:     {csv_path}")
    print(f"Summary:      {summary_path}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 continual pretraining on Indian urban clips (Ch10)")
    parser.add_argument("--config", type=str, default=None,
                        help="Legacy single YAML config (backward compat with train_pretrain.sh)")
    add_model_config_arg(parser)
    add_train_config_arg(parser)
    parser.add_argument("--explora", action="store_true",
                        help="Enable ExPLoRA mode (LoRA + block freeze)")
    parser.add_argument("--SANITY", action="store_true",
                        help="Quick validation: 50 steps, batch_size=2")
    parser.add_argument("--POC", action="store_true",
                        help="POC subset (~10K clips, 5 epochs)")
    parser.add_argument("--FULL", action="store_true",
                        help="Full training run")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Override drift control lambda (ablation: 0, 0.001, 0.01, 0.1)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max epochs (SANITY=1, --POC=5, --FULL=1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (used by ablation to write to ablation/ subdir)")
    parser.add_argument("--val-subset", type=str, default=None,
                        help="Path to val subset JSON (e.g., data/val_1k.json). "
                             "These clips are excluded from training and used for periodic val loss.")
    parser.add_argument("--val-local-data", type=str, default=None,
                        help="Local WebDataset dir for val clips (e.g., data/val_1k_local). "
                             "If omitted, val clips are loaded from --local-data.")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)

    # Load config: --model-config + --train-config (new) or --config (legacy)
    if args.config:
        cfg = load_config(args.config)
    elif args.train_config:
        cfg = load_merged_config(args.model_config, args.train_config)
    else:
        cfg = load_merged_config(DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG)
    cfg = merge_config_with_args(cfg, args)
    if args.explora:
        cfg.setdefault("explora", {})["enabled"] = True

    # Auto-ablation: if no --lambda-reg specified, find or run ablation
    if args.lambda_reg is None:
        out_dir = Path(cfg["checkpoint"]["output_dir"]).parent
        ablation_dir = out_dir / "ablation"
        winner_json = ablation_dir / "ablation_winner.json"

        if winner_json.exists():
            w = json.load(open(winner_json))
            cfg["drift_control"]["lambda_reg"] = float(w["winner_lambda"])
            lam_str = f"{float(w['winner_lambda']):g}".replace(".", "_")
            cfg["checkpoint"]["output_dir"] = str(out_dir / f"m09_lambda{lam_str}")
            print(f"Using ablation winner: lambda={w['winner_lambda']} "
                  f"(best_val_loss={w.get('winner_val_loss', '?')}) from {winner_json}")
        else:
            ablation_dir = out_dir / "ablation"
            print(f"\n{'='*60}")
            print(f"  AUTO-ABLATION: No {winner_json} found.")
            print("  Running 4 lambda sweep on data/subset_10k_local (~2h)")
            print(f"  Ablation (10K) → {ablation_dir}/m09_lambda*/")
            print(f"  Full training  → {out_dir}/m09_lambda<winner>/")
            print(f"{'='*60}\n")

            drift_cfg = cfg["drift_control"]
            ablation_lambdas = drift_cfg["ablation_lambdas"]

            for lam in ablation_lambdas:
                lam_str = f"{lam:g}".replace(".", "_")
                lam_out = ablation_dir / f"m09_lambda{lam_str}"

                # Skip if already trained with valid val_loss
                summary_path = lam_out / "training_summary.json"
                if summary_path.exists():
                    s = json.load(open(summary_path))
                    if s.get("best_val_loss") is not None and s["best_val_loss"] != float("inf"):
                        print(f"  lambda={lam}: already trained (val_loss={s['best_val_loss']:.4f}), skipping")
                        continue

                print(f"\n--- Ablation: lambda={lam} ---")
                import subprocess
                mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
                config_flags = (["--config", args.config] if args.config
                                else ["--model-config", args.model_config,
                                      "--train-config", args.train_config or DEFAULT_TRAIN_CONFIG])
                ablation_cmd = [
                    sys.executable, "-u", os.path.abspath(__file__),
                    *config_flags,
                    mode_flag,
                    "--local-data", drift_cfg["ablation_local_data"],
                    "--val-subset", drift_cfg["ablation_val_subset"],
                    "--val-local-data", drift_cfg["ablation_val_local_data"],
                    "--lambda-reg", str(lam),
                    "--max-epochs", str(drift_cfg["ablation_epochs"]),
                    "--output-dir", str(lam_out),
                ]
                if args.batch_size is not None:
                    ablation_cmd += ["--batch-size", str(args.batch_size)]
                result = subprocess.run(ablation_cmd)
                if result.returncode != 0:
                    print(f"FATAL: Ablation failed for lambda={lam}")
                    sys.exit(1)

            select_ablation_winner(str(ablation_dir), [str(l) for l in ablation_lambdas])

            if not winner_json.exists():
                print("FATAL: ablation_winner.json not created after ablation")
                sys.exit(1)
            w = json.load(open(winner_json))
            cfg["drift_control"]["lambda_reg"] = float(w["winner_lambda"])
            lam_str = f"{float(w['winner_lambda']):g}".replace(".", "_")
            cfg["checkpoint"]["output_dir"] = str(out_dir / f"m09_lambda{lam_str}")
            print(f"\nProceeding with winner: lambda={w['winner_lambda']} "
                  f"(best_val_loss={w['winner_val_loss']:.4f})")

    train(cfg, args)


def select_ablation_winner(output_dir: str, lambdas=None):
    """Compare best_val_loss across lambda ablation runs → write ablation_winner.json.

    USAGE:
        python -u src/m09_pretrain.py --select-winner outputs/poc 2>&1 | tee logs/m09_select_winner.log
    """
    if lambdas is None:
        cfg = load_merged_config(DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG)
        lambdas = [str(l) for l in cfg["drift_control"]["ablation_lambdas"]]

    out = Path(output_dir)
    results = {}

    print("\n=== Lambda Ablation Winner Selection ===")
    print(f"Output dir: {out}")
    print("Selection metric: best_val_loss (lowest wins)\n")

    for lam in lambdas:
        lam_dir = "lambda" + lam.replace(".", "_")
        summary_path = out / f"m09_{lam_dir}" / "training_summary.json"
        if not summary_path.exists():
            print(f"  lambda={lam}: MISSING ({summary_path})")
            continue
        s = json.load(open(summary_path))
        val_loss = s.get("best_val_loss")
        if val_loss is None or val_loss == float("inf"):
            print(f"  lambda={lam}: NO VALID val_loss (was validation run?)")
            continue
        results[lam] = {"val_loss": val_loss, "dir": lam_dir,
                        "jepa_loss": s.get("final_jepa_loss"),
                        "epochs": s.get("epochs"), "steps": s.get("steps")}
        print(f"  lambda={lam}: best_val_loss={val_loss:.4f} "
              f"(jepa_loss={s.get('final_jepa_loss', '?'):.4f}, "
              f"{s.get('epochs', '?')} epochs, {s.get('steps', '?')} steps)")

    if not results:
        print("\nFATAL: No valid ablation results found. Run POC ablation first.")
        sys.exit(1)

    winner = min(results, key=lambda k: results[k]["val_loss"])
    winner_info = {
        "winner_lambda": winner,
        "winner_dir": results[winner]["dir"],
        "winner_val_loss": results[winner]["val_loss"],
        "selection_metric": "best_val_loss (lowest)",
        "all_results": {k: v["val_loss"] for k, v in results.items()},
    }

    winner_path = out / "ablation_winner.json"
    with open(winner_path, "w") as f:
        json.dump(winner_info, f, indent=2)

    print(f"\n  Winner: lambda={winner} (best_val_loss={results[winner]['val_loss']:.4f})")
    print(f"  Saved: {winner_path}")
    print("  run_pretrain.sh --FULL will read this automatically.")

    # Plot val_loss curves for all lambdas (publication quality)
    try:
        from utils.plots import plot_val_loss_curves
        plot_val_loss_curves(out, lambdas, winner)
    except Exception as e:
        print(f"  WARN: ablation plot failed ({e}), continuing")

    # TODO: V-JEPA 2.1 comparison — after training both versions, generate comparison plots:
    # from utils.plots import plot_training_curves
    # plot_training_curves(
    #     runs=[
    #         {"csv_path": "outputs/full/m09_lambda0_001/loss_log.csv",
    #          "label": "V-JEPA 2.0", "color": "blue"},
    #         {"csv_path": "outputs/full_v21/m09_lambda0_001/loss_log.csv",
    #          "label": "V-JEPA 2.1", "color": "red"},
    #     ],
    #     output_dir="outputs/full/comparison",
    #     title_prefix="115K Clips, ",
    # )


if __name__ == "__main__":
    # Check for --select-winner subcommand before argparse
    if len(sys.argv) >= 3 and sys.argv[1] == "--select-winner":
        select_ablation_winner(sys.argv[2])
    else:
        main()

    # Force exit: CUDA atexit cleanup can deadlock on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
