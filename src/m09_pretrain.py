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

ABLATION (drift control lambda sweep — run sequentially or on 4 GPUs):
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
        --lambda-reg 0 --no-wandb 2>&1 | tee logs/m09_lambda0.log
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
        --lambda-reg 0.001 --no-wandb 2>&1 | tee logs/m09_lambda0_001.log
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
        --lambda-reg 0.01 --no-wandb 2>&1 | tee logs/m09_lambda0_01.log
    python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
        --FULL --subset data/subset_10k.json --local-data data/subset_10k_local \
        --lambda-reg 0.1 --no-wandb 2>&1 | tee logs/m09_lambda0_1.log
"""
import argparse
import copy
import gc
import json
import os
import queue
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
    HF_DATASET_REPO, check_gpu,
    add_subset_arg, add_local_data_arg, get_output_dir, load_subset,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)

import torch
import torch.nn.functional as F

# Add vjepa2 to path (Q1: sys.path, not pip install)
VJEPA2_DIR = Path(__file__).parent.parent / "deps" / "vjepa2"
sys.path.insert(0, str(VJEPA2_DIR))

# Constants
DEFAULT_CONFIG = "configs/pretrain/vitg16_indian.yaml"
CHECKPOINT_PREFIX = "m09_ckpt"
PREFETCH_QUEUE_SIZE = 2
DECODE_WORKERS = 4
MAX_STREAM_RETRIES = 5

# Reuse data loading from m05
from m05_vjepa_embed import (
    get_clip_key, _create_stream, decode_video_bytes,
)


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
        cfg["optimization"]["max_steps"] = 50
        cfg["optimization"]["batch_size"] = 2
        cfg["validation"]["interval_steps"] = 25
        cfg["checkpoint"]["save_every_steps"] = 25
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    if args.max_steps is not None:
        cfg["optimization"]["max_steps"] = args.max_steps
    if args.lambda_reg is not None:
        cfg["drift_control"]["lambda_reg"] = args.lambda_reg
        if args.lambda_reg == 0:
            cfg["drift_control"]["enabled"] = False

    # Output dir: base/m09_lambda{value}/ for ablation isolation
    base_out = get_output_dir(args.subset, sanity=args.SANITY)
    lam = cfg["drift_control"]["lambda_reg"]
    lam_str = str(lam).replace(".", "_")  # 0.01 → 0_01 (avoid dots in dir names)
    cfg["checkpoint"]["output_dir"] = str(base_out / f"m09_lambda{lam_str}")
    return cfg


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

def split_by_video_id(clip_keys: list, val_fraction: float, seed: int) -> tuple:
    """Split clip keys into train/val by video_id (no leakage)."""
    rng = np.random.RandomState(seed)
    # Extract video_id: key format is "section/video_id/source_file"
    video_ids = sorted(set(k.rsplit("/", 1)[0].rsplit("/", 1)[-1] for k in clip_keys))
    rng.shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * val_fraction))
    val_video_ids = set(video_ids[:n_val])
    train_keys = set(k for k in clip_keys if k.rsplit("/", 1)[0].rsplit("/", 1)[-1] not in val_video_ids)
    val_keys = set(k for k in clip_keys if k.rsplit("/", 1)[0].rsplit("/", 1)[-1] in val_video_ids)
    return train_keys, val_keys


def augment_clip_consistent(video_tensor: torch.Tensor, cfg_aug: dict,
                            crop_size: int) -> torch.Tensor:
    """Video-consistent augmentation: same random crop applied to ALL T frames."""
    import torchvision.transforms as TT

    T_frames, C, H, W = video_tensor.shape
    scale = cfg_aug.get("random_resize_scale", [0.3, 1.0])
    ratio = cfg_aug.get("random_resize_ratio", [0.75, 1.35])

    i, j, h, w = TT.RandomResizedCrop.get_params(
        video_tensor[0], scale=scale, ratio=ratio)

    video = video_tensor.float() / 255.0
    video = video[:, :, i:i+h, j:j+w]
    video = F.interpolate(video, size=(crop_size, crop_size),
                          mode='bilinear', align_corners=False)

    if torch.rand(1).item() < cfg_aug.get("horizontal_flip", 0.5):
        video = video.flip(-1)

    return video  # (T, C, crop_size, crop_size)


def producer_thread(cfg: dict, q: queue.Queue, stop_event: threading.Event,
                    train_keys: set, processed_steps: int):
    """Stream WebDataset, decode videos, augment, enqueue batches."""
    from concurrent.futures import ThreadPoolExecutor

    batch_size = cfg["optimization"]["batch_size"]
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]
    local_data = cfg["data"].get("local_data")
    cfg_aug = cfg.get("augmentation", {})
    retries = 0

    tmp_dir = tempfile.mkdtemp(prefix="m09_")

    try:
        while not stop_event.is_set():
            try:
                ds = _create_stream(0, local_data=local_data)
                pending_bytes, pending_keys = [], []

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
                        with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                            futures = [
                                pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                                for b, k in zip(pending_bytes, pending_keys)
                            ]
                            results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                        batch_tensors = [t for t, k in results if t is not None]
                        batch_keys = [k for t, k in results if t is not None]

                        if batch_tensors:
                            # Video-consistent augmentation
                            augmented = []
                            for vt in batch_tensors:
                                aug = augment_clip_consistent(vt, cfg_aug, crop_size)
                                augmented.append(aug)

                            # Stack: (B, C, T, H, W) — V-JEPA expects channel-first video
                            batch = torch.stack(augmented, dim=0)  # (B, T, C, H, W)
                            batch = batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                            q.put(("batch", batch, batch_keys[:]))

                        pending_bytes, pending_keys = [], []
                        retries = 0

                # Handle final partial batch
                if pending_bytes and not stop_event.is_set():
                    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
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

                break  # stream exhausted

            except (ConnectionError, TimeoutError, OSError) as e:
                retries += 1
                if retries > MAX_STREAM_RETRIES:
                    print(f"  ERROR: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                    break
                wait = min(2 ** retries, 60)
                print(f"  WARN: stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES} in {wait}s")
                time.sleep(wait)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        q.put(("done", None, None))


# ═════════════════════════════════════════════════════════════════════════
# MODEL SETUP (Q1-Q5 corrected: no wrappers, vit_giant_xformers, RoPE)
# ═════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder, teacher encoder (EMA), and predictor."""
    from src.models.vision_transformer import vit_giant_xformers
    from src.models.predictor import vit_predictor

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # Student encoder (Q1: use vit_giant_xformers, Q5: use_rope=True)
    student = vit_giant_xformers(
        img_size=(data_cfg["crop_size"], data_cfg["crop_size"]),
        patch_size=data_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=data_cfg["tubelet_size"],
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=model_cfg.get("use_rope", True),
        use_activation_checkpointing=model_cfg.get("use_activation_checkpointing", True),
    )

    # Load pretrained weights (Q3: target_encoder key, Q5: strip prefixes)
    ckpt_path = Path("checkpoints/vjepa2_vitg384.pt")
    if ckpt_path.exists():
        print(f"Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        print("Downloading pretrained weights via torch.hub...")
        # Q2: URL is broken in source, use direct download
        url = "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt"
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu",
                                                    model_dir="checkpoints")

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
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params")
    if msg.missing_keys:
        print(f"  Missing keys (expected for RoPE): {msg.missing_keys[:5]}")
    student = student.to(device)

    # Teacher (EMA copy, frozen)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"Teacher created (deepcopy of student)")

    # Predictor (Q4: num_mask_tokens, Q1: use_rope)
    predictor = vit_predictor(
        img_size=(data_cfg["crop_size"], data_cfg["crop_size"]),
        patch_size=data_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=data_cfg["tubelet_size"],
        embed_dim=model_cfg["embed_dim"],
        predictor_embed_dim=model_cfg["pred_embed_dim"],
        depth=model_cfg["pred_depth"],
        num_heads=model_cfg["pred_num_heads"],
        use_mask_tokens=True,
        num_mask_tokens=model_cfg.get("num_mask_tokens", 2),
        zero_init_mask_tokens=True,
        use_rope=model_cfg.get("use_rope", True),
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=model_cfg.get("use_activation_checkpointing", True),
    )

    # Q3: Load predictor weights if available
    if "predictor" in ckpt:
        pred_state = {k.replace("module.", "").replace("backbone.", ""): v
                      for k, v in ckpt["predictor"].items()}
        pred_msg = predictor.load_state_dict(pred_state, strict=False)
        print(f"Predictor loaded from checkpoint (missing: {len(pred_msg.missing_keys)} keys)")
    else:
        print("Predictor initialized randomly (no pretrained weights)")

    predictor = predictor.to(device)
    print(f"Predictor: {sum(p.numel() for p in predictor.parameters()):,} params")

    # Save initial params for drift control
    init_params = {name: p.clone().detach()
                   for name, p in student.named_parameters()}

    del ckpt
    gc.collect()

    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "init_params": init_params,
    }


# ═════════════════════════════════════════════════════════════════════════
# MASKING (Q4: Use _MaskGenerator directly, not MaskCollator)
# ═════════════════════════════════════════════════════════════════════════

def build_mask_generators(cfg: dict) -> list:
    """Build one _MaskGenerator per mask config."""
    from src.masks.multiseq_multiblock3d import _MaskGenerator

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

def compute_jepa_loss(pred_features: list, teacher_output: torch.Tensor,
                      masks_pred: list, loss_exp: float = 1.0) -> torch.Tensor:
    """L1 latent prediction loss on masked tokens (V-JEPA 2 convention).

    pred_features: list of predictor outputs, one per mask generator
    teacher_output: teacher forward on full clip (B, N_total, D)
    masks_pred: list of target mask tensors, one per mask generator
    """
    from src.masks.utils import apply_masks

    loss = torch.tensor(0.0, device=teacher_output.device)
    n = 0
    for pred_i, mask_i in zip(pred_features, masks_pred):
        # Extract teacher features at masked positions
        h_masked = apply_masks(teacher_output, [mask_i])  # (B, N_masked, D)
        loss += torch.mean(torch.abs(pred_i - h_masked) ** loss_exp) / loss_exp
        n += 1
    return loss / max(n, 1)


def compute_drift_loss(student: torch.nn.Module, init_params: dict,
                       lambda_reg: float) -> torch.Tensor:
    """L2 drift control: lambda * ||theta - theta_0||^2 (Sec 10.4)."""
    drift = torch.tensor(0.0, device=next(student.parameters()).device)
    for name, param in student.named_parameters():
        if name in init_params:
            drift += torch.sum((param - init_params[name]) ** 2)
    return lambda_reg * drift


# ═════════════════════════════════════════════════════════════════════════
# EMA TEACHER UPDATE
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def update_teacher_ema(student: torch.nn.Module, teacher: torch.nn.Module,
                       momentum: float):
    """EMA update: theta_bar <- tau * theta_bar + (1-tau) * theta"""
    params_s = list(student.parameters())
    params_t = list(teacher.parameters())
    torch._foreach_mul_(params_t, momentum)
    torch._foreach_add_(params_t, params_s, alpha=1.0 - momentum)


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


def build_scheduler(optimizer, cfg_opt: dict):
    warmup_steps = cfg_opt["warmup_steps"]
    max_steps = cfg_opt["max_steps"]
    min_lr = cfg_opt["min_lr"]
    base_lr = cfg_opt["lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(min_lr / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════════════
# VALIDATION (Cycle@K hard mode)
# ═════════════════════════════════════════════════════════════════════════

def run_validation(student, val_clips: list, val_keys: list,
                   cfg: dict, device: torch.device, step: int) -> dict:
    """Extract embeddings for val clips, compute Cycle@K with bootstrap CI."""
    import faiss
    from utils.bootstrap import bootstrap_ci, per_clip_cycle_at_k

    student.eval()
    embeddings = []

    with torch.no_grad():
        for clip_batch in val_clips:
            clip_batch = clip_batch.to(device)
            features = student(clip_batch)  # (B, N, D) — full forward, no masks
            emb = features.mean(dim=1)  # (B, D) — mean pool
            emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    n, d = embeddings.shape

    faiss_k = cfg["validation"]["faiss_k"]
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    index.add(embeddings)
    _, indices = index.search(embeddings, faiss_k)

    results = {}
    for k in cfg["validation"]["cycle_k"]:
        scores = per_clip_cycle_at_k(indices, k)
        ci = bootstrap_ci(scores)
        results[f"cycle_at_{k}"] = ci
        print(f"  Step {step} | Cycle@{k}: {ci['mean']:.1f}% "
              f"[{ci['ci_lo']:.1f}, {ci['ci_hi']:.1f}]")

    student.train()
    return results


# ═════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═════════════════════════════════════════════════════════════════════════

def save_training_checkpoint(path: Path, student, teacher, predictor,
                              optimizer, scheduler, scaler,
                              step: int, best_metric: float):
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
    os.replace(tmp_path, path)


def load_training_checkpoint(path: Path, student, teacher, predictor,
                              optimizer, scheduler, scaler) -> tuple:
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
    """Export student encoder weights + metadata for m05/m06 re-evaluation."""
    from utils.config import VJEPA_MODEL_ID
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "student_state_dict": student.state_dict(),
        "model_id": VJEPA_MODEL_ID,  # base architecture for instantiation
        "type": "vjepa2_adapted",
    }, path)
    print(f"Exported student encoder: {path}")


# ═════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════

def train(cfg: dict, args):
    """Main training loop (proposal Sec 10.5)."""
    check_gpu()
    device = torch.device("cuda")

    # Output-exists guard (CLAUDE.md rule #6)
    output_dir = Path(cfg["checkpoint"]["output_dir"])
    final_ckpt = output_dir / f"{CHECKPOINT_PREFIX}_final.pt"
    if final_ckpt.exists():
        print(f"Final checkpoint already exists: {final_ckpt}")
        print("Delete to re-train, or use for evaluation.")
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

    # Build mask generators (Q4: _MaskGenerator, not MaskCollator)
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)} "
          f"(blocks: {[m['num_blocks'] for m in cfg['mask']]})")

    # Optimizer & scheduler
    optimizer = build_optimizer(student, predictor, cfg["optimization"])
    scheduler = build_scheduler(optimizer, cfg["optimization"])
    mp_cfg = cfg["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=mp_cfg["enabled"])

    # Resume from checkpoint
    start_step = 0
    best_cycle_k = 0.0
    ckpt_path = output_dir / f"{CHECKPOINT_PREFIX}_latest.pt"
    if ckpt_path.exists():
        start_step, best_cycle_k = load_training_checkpoint(
            ckpt_path, student, teacher, predictor, optimizer, scheduler, scaler)
        print(f"Resumed from step {start_step}, best Cycle@1: {best_cycle_k:.1f}%")

    # Data stream
    subset_keys = load_subset(args.subset) if args.subset else set()
    if subset_keys:
        train_keys, val_key_set = split_by_video_id(
            list(subset_keys), cfg["data"]["val_fraction"], cfg["data"]["val_seed"])
        print(f"Train/val split: {len(train_keys):,} / {len(val_key_set):,} clips")
    else:
        train_keys = set()
        val_key_set = set()

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()
    prod = threading.Thread(
        target=producer_thread,
        args=(cfg, q, stop_event, train_keys, start_step),
        daemon=True,
    )
    prod.start()

    # WandB
    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m09", mode, config=cfg, enabled=not args.no_wandb)

    # Training config
    max_steps = cfg["optimization"]["max_steps"]
    ema_momentum = cfg["optimization"]["ema_momentum"]
    loss_exp = cfg["optimization"]["loss_exp"]
    drift_cfg = cfg["drift_control"]
    val_interval = cfg["validation"]["interval_steps"]
    ckpt_interval = cfg["checkpoint"]["save_every_steps"]
    dtype = getattr(torch, mp_cfg["dtype"])

    pbar = tqdm(total=max_steps, initial=start_step,
                desc="m09_pretrain", unit="step")

    # Windowed throughput (CLAUDE.md rule #11)
    window_start = time.time()
    window_steps = 0
    running_loss = 0.0

    print(f"\n=== Training: {start_step} → {max_steps} steps ===")
    print(f"Batch size: {cfg['optimization']['batch_size']}")
    print(f"Grad checkpointing: {cfg['model'].get('use_activation_checkpointing', False)}")
    print(f"Mixed precision: {mp_cfg['dtype']}")
    print(f"EMA momentum: {ema_momentum}")
    print(f"Drift control: lambda={drift_cfg['lambda_reg'] if drift_cfg['enabled'] else 0}")

    step = start_step
    try:
        for step in range(start_step, max_steps):
            # Get batch from producer
            try:
                msg_type, batch_clips, batch_keys = q.get(timeout=600)
            except queue.Empty:
                print(f"\nProducer timeout at step {step}. Saving checkpoint...")
                break
            if msg_type == "done":
                print(f"\nData exhausted at step {step}.")
                break

            batch_clips = batch_clips.to(device)
            actual_bs = batch_clips.shape[0]

            # Generate masks (Q4: _MaskGenerator directly)
            all_masks_enc, all_masks_pred = [], []
            for mg in mask_generators:
                m_enc, m_pred = mg(actual_bs)
                all_masks_enc.append(m_enc.to(device))
                all_masks_pred.append(m_pred.to(device))

            # Forward pass (mixed precision)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=dtype, enabled=mp_cfg["enabled"]):
                # Teacher: all tokens, no grad
                with torch.no_grad():
                    h = teacher(batch_clips)  # (B, N_total, D)
                    h = F.layer_norm(h, (h.size(-1),))

                # Student + Predictor per mask generator
                pred_features = []
                for i, (m_enc, m_pred) in enumerate(zip(all_masks_enc, all_masks_pred)):
                    z = student(batch_clips, masks=[m_enc])  # visible tokens only
                    p = predictor(z, [m_enc], [m_pred], mask_index=i)
                    pred_features.append(p)

                # Loss
                jepa_loss = compute_jepa_loss(pred_features, h, all_masks_pred, loss_exp)

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
            scheduler.step()

            # EMA teacher update
            update_teacher_ema(student, teacher, ema_momentum)

            # Logging (windowed throughput)
            running_loss += total_loss.item()
            window_steps += 1
            now = time.time()
            window_elapsed = now - window_start

            if window_elapsed >= 30:
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
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "throughput_steps_per_s": throughput,
                }, step=step)
                window_start = now
                window_steps = 0
                running_loss = 0.0

            pbar.update(1)

            # Validation
            if (step + 1) % val_interval == 0 and val_key_set:
                print(f"\n--- Validation at step {step + 1} ---")
                # Note: full val requires pre-collected val clips
                # For now, log loss-based metrics; full Cycle@K in post-training eval
                log_metrics(wb_run, {"val/step": step + 1}, step=step)

                cycle_1 = 0.0  # Placeholder until val clips are collected
                if cycle_1 > best_cycle_k:
                    best_cycle_k = cycle_1
                    save_training_checkpoint(
                        output_dir / f"{CHECKPOINT_PREFIX}_best.pt",
                        student, teacher, predictor, optimizer, scheduler, scaler,
                        step + 1, best_cycle_k)

            # Periodic checkpoint
            if (step + 1) % ckpt_interval == 0:
                save_training_checkpoint(
                    output_dir / f"{CHECKPOINT_PREFIX}_step{step+1}.pt",
                    student, teacher, predictor, optimizer, scheduler, scaler,
                    step + 1, best_cycle_k)
                save_training_checkpoint(
                    ckpt_path, student, teacher, predictor, optimizer, scheduler,
                    scaler, step + 1, best_cycle_k)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
    finally:
        pbar.close()
        stop_event.set()

    # Save final checkpoint
    save_training_checkpoint(
        final_ckpt, student, teacher, predictor, optimizer, scheduler,
        scaler, step + 1, best_cycle_k)

    # Export student for m05/m06 re-evaluation
    export_student_for_eval(student, output_dir / "student_encoder.pt")

    finish_wandb(wb_run)
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Steps:        {step + 1}")
    print(f"Best Cycle@1: {best_cycle_k:.1f}%")
    print(f"Final ckpt:   {final_ckpt}")
    print(f"Student:      {output_dir / 'student_encoder.pt'}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

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
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Override drift control lambda (ablation: 0, 0.001, 0.01, 0.1)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps from config")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
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
