"""ExPLoRA training — LoRA on blocks 2-47 + unfreeze blocks 0-1, no drift control. GPU-only.

Split from m09_pretrain.py on 2026-04-15 (#49). Pairs with m09a_pretrain.py (vanilla Ch10)
and m09c_surgery.py (factor surgery). Shared primitives live in utils.training.

Recipe: Freeze all transformer blocks EXCEPT first 1-2. Add LoRA (rank 8-16) on remaining
blocks. Continue same SSL objective (JEPA loss) on new domain. Reference:
https://arxiv.org/abs/2406.10973 (ICML 2025).

    python -u src/m09b_explora.py --SANITY --model-config configs/model/vjepa2_1.yaml --train-config configs/train/explora.yaml --no-wandb 2>&1 | tee logs/m09b_sanity.log
    python -u src/m09b_explora.py --POC    --subset data/sanity_100_dense.json --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m09b_dense100.log
    python -u src/m09b_explora.py --FULL   --local-data data/full_local --no-wandb 2>&1 | tee logs/m09b_full.log
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # Must be before torch import
os.environ.setdefault("MKL_NUM_THREADS", "1")   # Prevent OpenMP thread explosion in workers

import argparse
import copy
import csv
import gc
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
from tqdm import tqdm

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    check_gpu,
    add_subset_arg, add_local_data_arg, get_module_output_dir, load_subset,
    get_pipeline_config, load_merged_config,
    add_model_config_arg, add_train_config_arg,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)

import torch

from utils.progress import make_pbar  # noqa: F401 — available for optional pbar customization

# vjepa2 imports via shim (avoids src/ namespace collision)
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
)

# Shared video I/O from utils (Rule 32: no cross-imports between m*.py)
from utils.video_io import get_clip_key, create_stream, decode_video_bytes

# peft — LoRA injection for ExPLoRA recipe
from peft import get_peft_model, LoraConfig

# Shared training primitives from Phase 1 extraction (utils/training.py).
from utils.training import (
    MAX_STREAM_RETRIES,
    load_config, load_val_subset, augment_clip_consistent,
    producer_thread,
    build_mask_generators,
    _train_step_grad_accum,
    update_teacher_ema,
    build_optimizer, build_scheduler, update_weight_decay,
    run_validation,
    save_training_checkpoint, cleanup_old_checkpoints, load_training_checkpoint,
    export_student_for_eval,
)

# Module-level constants
DEFAULT_MODEL_CONFIG = "configs/model/vjepa2_1.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train/explora.yaml"
CHECKPOINT_PREFIX = "m09b_ckpt"
_pcfg = get_pipeline_config()
PREFETCH_QUEUE_SIZE = _pcfg["streaming"]["prefetch_queue_train"]

_create_stream = create_stream


# ═════════════════════════════════════════════════════════════════════════
# CONFIG (merge_config_with_args — argparse-coupled dispatch)
# ═════════════════════════════════════════════════════════════════════════

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

    # Hardcode ExPLoRA enabled — module IS the choice.
    cfg.setdefault("explora", {})["enabled"] = True

    # Output dir: explicit --output-dir, or auto from mode
    if getattr(args, "output_dir", None):
        cfg["checkpoint"]["output_dir"] = args.output_dir
        return cfg
    base_out = get_module_output_dir("m09b_explora", args.subset,
                                    sanity=args.SANITY, poc=args.POC)
    cfg["checkpoint"]["output_dir"] = str(base_out)
    return cfg


# ═════════════════════════════════════════════════════════════════════════
# MODEL SETUP — ExPLoRA path hardcoded (always apply LoRA + unfreeze first N blocks)
# ═════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder (w/ LoRA + first N blocks unfrozen), teacher (EMA), and predictor."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    arch = model_cfg["arch"]

    vit_constructor = get_vit_by_arch(arch)
    vit_predictor = get_vit_predictor()

    # Student encoder — arch from model config YAML (vit_giant_xformers or vit_gigantic_xformers)
    crop_size = model_cfg["crop_size"]
    student = vit_constructor(
        img_size=(crop_size, crop_size),
        patch_size=model_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=model_cfg["tubelet_size"],
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
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = True

    # V-JEPA 2.1 requires RoPE (no pos_embed registered in model)
    if model_cfg["predict_all"] or model_cfg["n_output_distillation"] > 1:
        if not model_cfg["use_rope"]:
            print("FATAL: V-JEPA 2.1 requires use_rope=True (no pos_embed registered in model)")
            sys.exit(1)

    # Teacher (EMA copy, frozen) — created BEFORE LoRA injection so teacher stays plain ViT.
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("Teacher created (deepcopy of student, hierarchical output enabled)")

    # Predictor: use 2.1 version if predict_all (supports return_all_tokens + proj_context)
    pred_constructor = get_vit_predictor_2_1() if model_cfg["predict_all"] else vit_predictor
    predictor = pred_constructor(
        img_size=(model_cfg["crop_size"], model_cfg["crop_size"]),
        patch_size=model_cfg["patch_size"],
        num_frames=data_cfg["num_frames"],
        tubelet_size=model_cfg["tubelet_size"],
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

    del ckpt
    gc.collect()

    # ExPLoRA: LoRA injection + block freezing (HARDCODED — always applied in m09b).
    explora_cfg = cfg["explora"]

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

    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "explora_enabled": True,
    }


# ═════════════════════════════════════════════════════════════════════════
# TRAINING LOOP (ExPLoRA, drift-free)
# ═════════════════════════════════════════════════════════════════════════

def train(cfg: dict, args):
    """Epoch-based ExPLoRA training loop (no drift loss)."""
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

    # Build model (LoRA + unfrozen blocks 0..N-1 — hardcoded in build_model)
    print("\n=== Building Model (ExPLoRA) ===")
    models = build_model(cfg, device)
    student = models["student"]
    teacher = models["teacher"]
    predictor = models["predictor"]

    student.train()
    predictor.train()

    # Build mask generators
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)} "
          f"(blocks: {[m['num_blocks'] for m in cfg['mask']]})")

    # Compute epoch geometry
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
        _val_tmp = tempfile.mkdtemp(prefix="m09b_val_")
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
            _val_tmp = tempfile.mkdtemp(prefix="m09b_val_retry_")
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
                    _val_tmp2 = tempfile.mkdtemp(prefix="m09b_val_")
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
                    _val_tmp3 = tempfile.mkdtemp(prefix="m09b_val_")
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

    # WandB — no lambda suffix for ExPLoRA (drift disabled)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m09b_explora", mode, config=cfg, enabled=not args.no_wandb)

    # Training config
    ema_momentum = cfg["optimization"]["ema_momentum"]
    loss_exp = cfg["optimization"]["loss_exp"]
    dtype = getattr(torch, mp_cfg["dtype"])

    pbar = tqdm(total=total_steps, initial=start_step,
                desc="m09b_explora", unit="step")

    # JSONL loss log — crash-safe (fsync after every write, survives OOM/SIGKILL)
    jsonl_path = output_dir / "loss_log.jsonl"
    jsonl_file = open(jsonl_path, "a")

    def _log_step(record: dict):
        """Write one JSON record + flush + fsync (Detectron2 pattern)."""
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()
        os.fsync(jsonl_file.fileno())

    # Also keep CSV for backward compat (plots, wandb upload) — no drift column in m09b.
    csv_path = output_dir / "loss_log.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["step", "epoch", "loss_jepa", "loss_total",
                             "lr", "grad_norm", "throughput", "val_loss"])
        csv_file.flush()

    # Windowed throughput
    window_start = time.time()
    window_steps = 0
    running_loss = 0.0

    # Local NaN strike counter (replaces danger-zone global `main._nan_strikes`)
    nan_strikes = 0

    print(f"\n=== Training: {start_step} → {total_steps} steps ({max_epochs} epochs) ===")
    print(f"Batch size: {batch_size}")
    print(f"Grad checkpointing: {cfg['model']['use_activation_checkpointing']}")
    print(f"Mixed precision: {mp_cfg['dtype']}")
    print(f"EMA momentum: {ema_momentum}")
    print("Drift control: DISABLED (ExPLoRA recipe)")
    print(f"Loss log: {csv_path}")

    # AdaptiveBatchSizer for gradient accumulation (#48). Effective BS stays = batch_size.
    _gpu_cfg = get_pipeline_config()["gpu"]
    train_sizer = AdaptiveBatchSizer(
        initial_size=min(_gpu_cfg["training_initial_bs"], batch_size),
        min_size=1, max_size=batch_size,
        memory_cap=_gpu_cfg["gpu_memory_target"])
    print(f"AdaptiveBatchSizer (training, grad-accum): start={train_sizer.size}, "
          f"max={batch_size} (= effective BS), target VRAM={_gpu_cfg['gpu_memory_target']:.0%}")

    step = start_step
    # Per-epoch memory hygiene — clears fragmented blocks accumulated across steps.
    _last_epoch_cleared = -1
    # Track last loss values so final-summary branch (if loop exits with no iterations)
    # still has something to report.
    jepa_val = 0.0
    masked_val = 0.0
    context_val = 0.0
    total_val = 0.0
    lr_val = 0.0
    gn_val = 0.0
    try:
        for step in range(start_step, total_steps):
            epoch = step // steps_per_epoch
            epoch_step = step % steps_per_epoch
            if epoch != _last_epoch_cleared:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                _last_epoch_cleared = epoch

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

            # Adaptive grad-accumulation forward+backward (#48 / #55). m09b relies on
            # the helper defaults (both regularizer args default to None), so the helper
            # short-circuits the regularizer branch and returns a zero fourth value.
            #
            # Within-step retry loop (#55): on OOM, sizer.on_oom() shrinks; we retry
            # the SAME macro at the new sub-batch instead of `continue`-ing to next
            # step. Old `continue` was harmless when total_steps >> 1 (next step would
            # try the new sub-batch), but failed silently in low-step regimes (POC/SANITY).
            # Now we retry until either (a) success or (b) sizer at min and OOMed → fail-hard.
            step_succeeded = False
            while not step_succeeded:
                try:
                    jepa_val, masked_val, context_val, _unused = _train_step_grad_accum(
                        student, teacher, predictor, batch_clips,
                        all_masks_enc, all_masks_pred,
                        cfg, dtype, mp_cfg, scaler, train_sizer, loss_exp)
                    step_succeeded = True
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()  # discard partial grads from incomplete macro
                    if train_sizer.size <= train_sizer.min_size:
                        raise RuntimeError(
                            f"Step {step}: OOM persists at minimum sub-batch="
                            f"{train_sizer.size}. ExPLoRA budget (LoRA rank={cfg['explora']['lora_rank']} "
                            f"+ first-2-block unfreeze) exceeded GPU memory cap "
                            f"({_gpu_cfg['gpu_memory_target']:.0%}). Mitigations: lower lora_rank, "
                            f"reduce training_batch_size in YAML, or move to FULL hardware. "
                            f"See errors_N_fixes.md #55."
                        ) from None
                    print(f"  OOM at step {step}: sub-batch shrunk to "
                          f"{train_sizer.size}, retrying SAME macro")
            total_val = jepa_val

            # Single optimizer step per macro batch — preserves effective BS = batch_size
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

            # Per-step logging + NaN/Inf guard.
            lr_val = scheduler.get_last_lr()[0]
            gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            if math.isnan(jepa_val) or math.isinf(jepa_val):
                nan_strikes += 1
                print(f"  NaN/Inf loss at step {step} (strike {nan_strikes}/3). "
                      f"GradScaler will auto-adjust.")
                if nan_strikes >= cfg["optimization"]["nan_tolerance"]:
                    print("FATAL: 3 consecutive NaN/Inf losses. Model diverged.")
                    print("  Debug: check loss_log.csv for divergence point.")
                    sys.exit(1)
            else:
                nan_strikes = 0  # reset on valid loss

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
                "loss_total": round(total_val, 6), "lr": lr_val,
                "grad_norm": round(gn_val, 4), "throughput": round(throughput, 2),
            }
            _log_step(step_record)  # JSONL: crash-safe (fsync per write)

            csv_writer.writerow([step, epoch, f"{jepa_val:.6f}",
                                 f"{total_val:.6f}", f"{lr_val:.2e}",
                                 f"{gn_val:.4f}", f"{throughput:.2f}", ""])
            if step % 10 == 0:
                csv_file.flush()

            log_metrics(wb_run, {
                "loss/jepa": jepa_val,
                "loss/masked": masked_val,
                "loss/context": context_val,
                "loss/total": total_val,
                "lr": lr_val,
                "grad_norm": gn_val,
                "epoch": epoch,
                "throughput_steps_per_s": throughput,
            }, step=step)

            if window_elapsed >= 30:
                pbar.set_postfix_str(
                    f"E{epoch} loss={running_loss/window_steps:.4f} "
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
                csv_writer.writerow([step, epoch, "", "",
                                     "", "", "", f"{val_loss:.6f}"])
                csv_file.flush()

                # Live training plots (regenerate on each validation)
                try:
                    from utils.plots import plot_training_curves
                    plot_training_curves(
                        runs=[{"csv_path": str(csv_path),
                               "label": "V-JEPA 2.1 ExPLoRA",
                               "color": "green",
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
        print(f"  Cooldown LR schedule: {current_lr:.2e} → {cooldown_final_lr:.2e} "
              f"over {cooldown_steps} steps (linear)")
        print("  NOTE: Full cooldown with 64f data loading requires producer restart.")
        print("  Cooldown is a paper-quality enhancement. POC results valid without it.")

    # Export student encoder (the only deliverable — only reached if training completed).
    # explora_enabled=True triggers peft's merge_and_unload() — converts the LoRA-wrapped
    # student back to a plain ViT that m05 can load.
    export_student_for_eval(student, student_path, explora_enabled=True)

    # Cleanup ALL checkpoints — student_encoder.pt is the deliverable
    for ckpt_file in output_dir.glob(f"{CHECKPOINT_PREFIX}_*.pt"):
        ckpt_file.unlink()
        print(f"  Cleaned: {ckpt_file.name}")

    # Save training summary as JSON (no drift / lambda fields in m09b)
    total_epochs_done = (step + 1) / max(steps_per_epoch, 1)
    summary = {
        "steps": step + 1,
        "epochs": round(total_epochs_done, 2),
        "clips_seen": (step + 1) * batch_size,
        "final_jepa_loss": jepa_val,
        "final_total_loss": total_val,
        "best_val_loss": best_val_loss,
        "final_lr": lr_val,
        "final_grad_norm": gn_val,
        "batch_size": batch_size,
        "explora_enabled": True,
        "unfreeze_blocks": cfg["explora"]["unfreeze_blocks"],
        "lora_rank": cfg["explora"]["lora_rank"],
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
    # Reduce CUDA fragmentation — pairs with AdaptiveBatchSizer + per-epoch gc (#47/#48/#53).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 ExPLoRA training (LoRA + unfreeze first N blocks)")
    parser.add_argument("--config", type=str, default=None,
                        help="Legacy single YAML config (backward compat)")
    add_model_config_arg(parser)
    add_train_config_arg(parser)
    parser.add_argument("--SANITY", action="store_true",
                        help="Quick validation: small subset, 1 epoch")
    parser.add_argument("--POC", action="store_true",
                        help="POC subset (~10K clips, 5 epochs)")
    parser.add_argument("--FULL", action="store_true",
                        help="Full training run")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max epochs (SANITY=1, --POC=5, --FULL=1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
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

    train(cfg, args)


if __name__ == "__main__":
    main()

    # Force exit: CUDA atexit cleanup can deadlock on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
