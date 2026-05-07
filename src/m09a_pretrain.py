"""Vanilla Ch10 Continual Pretraining (V-JEPA 2.1 with drift control + lambda ablation sweep). GPU-only.

Split from m09_pretrain.py on 2026-04-15 (#49). Pairs with m09b_explora.py (LoRA variant)
and m09c_surgery.py (factor surgery). Shared primitives live in utils.training.

USAGE (every path arg required — CLAUDE.md no-default rule):
    python -u src/m09a_pretrain.py --SANITY \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/legacy2/ch10_pretrain.yaml \
        --subset data/sanity_100_dense.json --local-data data/val_1k_local \
        --val-subset data/val_1k.json --val-local-data data/val_1k_local \
        --no-wandb 2>&1 | tee logs/m09a_sanity.log
    python -u src/m09a_pretrain.py --POC \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/legacy2/ch10_pretrain.yaml \
        --subset data/sanity_100_dense.json --local-data data/val_1k_local \
        --val-subset data/val_1k.json --val-local-data data/val_1k_local \
        --no-wandb 2>&1 | tee logs/m09a_poc.log
    python -u src/m09a_pretrain.py --FULL \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/legacy2/ch10_pretrain.yaml \
        --subset data/subset_10k.json --local-data data/subset_10k_local \
        --val-subset data/val_1k.json --val-local-data data/val_1k_local \
        --no-wandb 2>&1 | tee logs/m09a_full.log
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
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
# iter11 live-debug: SIGUSR1/SIGUSR2 stack dump so stuck GPU runs can be
# inspected without CAP_SYS_PTRACE (py-spy / gdb / strace are blocked in
# the training container). See src/utils/live_debug.py for usage.
from utils.live_debug import install_debug_handlers
install_debug_handlers()

from utils.config import (
    check_gpu,
    add_subset_arg, add_local_data_arg, get_module_output_dir, load_subset,
    get_pipeline_config, load_merged_config,
    add_model_config_arg, add_train_config_arg,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer
from utils.progress import make_pbar
from utils.plots import (
    plot_training_curves, plot_val_loss_curves, plot_combined_losses,
    plot_probe_trajectory_trio, plot_val_loss_with_kill_switch_overlay,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.cache_policy import (
    add_cache_policy_arg, resolve_cache_policy_interactive,
    guarded_delete, wipe_output_dir,
)

import torch

# vjepa2 imports via shim (avoids src/ namespace collision)
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
)

# Shared video I/O from utils (Rule 32: no cross-imports between m*.py)
from utils.video_io import get_clip_key, create_stream, decode_video_bytes

_create_stream = create_stream

# Shared training primitives — moved to utils/training.py in Phase 1 of iter8 split.
# Note: `cleanup_old_checkpoints` is re-implemented inline below because the
# utils version hardcodes the m09 legacy prefix; m09a uses `m09a_ckpt`.
from utils.training import (
    MAX_STREAM_RETRIES,
    load_config, load_val_subset, augment_clip_consistent,
    producer_thread,
    build_mask_generators,
    _train_step_grad_accum,
    update_teacher_ema,
    build_optimizer, build_scheduler, update_weight_decay,
    run_validation,
    save_training_checkpoint, load_training_checkpoint,
    enable_gradient_checkpointing,
    build_probe_clips,
    cleanup_old_checkpoints,
    run_trio_at_val, track_block_drift_at_val, update_best_state_on_score,
    export_student_for_eval, finalize_training,
)
from utils.action_labels import load_action_labels
from utils.multi_task_loss import (
    merge_multi_task_config, build_multi_task_head_from_cfg,
    attach_head_to_optimizer, run_multi_task_step,
)
# iter13 v12 (2026-05-06): motion_aux loss — joint K-class CE + 13-D MSE on
# RAFT optical-flow targets. REPLACES multi_task_probe (15 retrieval tag dims)
# as the sole supervised aux loss for v12. See plan_v12_motion_aux.md.
from utils.motion_aux_loss import (
    merge_motion_aux_config, build_motion_aux_head_from_cfg,
    attach_motion_aux_to_optimizer, run_motion_aux_step,
)
from utils.probe_labels import ensure_probe_labels_for_mode

# Constants — paths come from CLI args only (CLAUDE.md no-default rule)
CHECKPOINT_PREFIX = "m09a_ckpt"
_pcfg = get_pipeline_config()
PREFETCH_QUEUE_SIZE = _pcfg["streaming"]["prefetch_queue_train"]


# iter13 v13 C2-fix (2026-05-07): retired local _cleanup_old_checkpoints wrapper
# — was a one-line delegate to utils.training.cleanup_old_checkpoints. Calls
# now invoke the shared util directly with `prefix=CHECKPOINT_PREFIX`, matching
# m09c's pattern. Eliminates the discrepancy without behavioural change.


# iter13 v13 C3-fix (2026-05-07): _render_m09a_probe_plots was retired.
# val_jepa overlay moved to utils.plots.plot_val_loss_with_kill_switch_overlay
# (shared with m09c). Trio trajectory was always via plot_probe_trajectory_trio
# in utils.plots — no change there. Caller (line ~1260) now invokes both shared
# utils directly.
def _render_m09a_probe_plots(probe_history: list, output_dir: Path,
                              best_state: dict, kill_state: dict, drift_cfg: dict) -> None:
    """Thin compat wrapper — calls shared utils. Kept so existing call sites
    don't need touching. May be removed in a future R6 utils-refactor pass."""
    plot_val_loss_with_kill_switch_overlay(
        probe_history, output_dir,
        best_state=best_state, kill_state=kill_state,
        file_prefix="m09a",
        title_prefix=f"m09a · λ={drift_cfg['lambda_reg']} · ",
    )
    plot_probe_trajectory_trio(
        probe_history, output_dir,
        title_prefix=f"m09a · λ={drift_cfg['lambda_reg']} · ",
        file_prefix="m09a",
    )


# ═════════════════════════════════════════════════════════════════════════
# CONFIG (merge_config_with_args stays here — argparse-coupled dispatch)
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
    # Per-mode memory-saver flatten (probe_pretrain.yaml uses dicts; ch10_pretrain
    # uses scalars). Mirror m09c_surgery.py:130-139 pattern. SANITY (24GB) → True
    # for all three; FULL (96GB) → False for clean fp32 paper-quality runs.
    # Membership check (NOT .get()) — CLAUDE.md fail-loud rule on YAML subscripts;
    # paged_optim isn't in base_optimization.yaml so it may be absent under ch10.
    for k in ("use_8bit_optim", "gradient_checkpointing", "paged_optim"):
        if k in cfg["optimization"] and isinstance(cfg["optimization"][k], dict):
            cfg["optimization"][k] = cfg["optimization"][k][mode_key]
    # Mid-training probe block flatten (mirror m09c_surgery.py:145-170). The
    # `probe:` schema lives in base_optimization.yaml:249-283 and is shared
    # with m09c surgery — same field names. Per-mode flatten so the train loop
    # reads scalars directly. CLI overrides win over yaml. SANITY mode disables
    # probe by default (n=21 too small for stable BCa CI).
    if "probe" in cfg:
        probe_cfg = cfg["probe"]
        # CLI overrides for paths (mirror m09c:152-157).
        if getattr(args, "probe_subset", None):
            probe_cfg["subset"] = args.probe_subset
        if getattr(args, "probe_local_data", None):
            probe_cfg["local_data"] = args.probe_local_data
        if getattr(args, "probe_tags", None):
            probe_cfg["tags_path"] = args.probe_tags
        # Per-mode flatten of all gate-style booleans (mirror m09c:160-170).
        for k in ("enabled", "best_ckpt_enabled", "kill_switch_enabled",
                  "plateau_enabled", "prec_plateau_enabled",
                  "bwt_trigger_enabled", "use_permanent_val"):
            if k in probe_cfg and isinstance(probe_cfg[k], dict):
                probe_cfg[k] = probe_cfg[k][mode_key]
        # --no-probe CLI flag overrides yaml (force-disable).
        if getattr(args, "no_probe", False):
            probe_cfg["enabled"] = False
    # Multi-task probe block — per-mode flatten + CLI overrides (utils helper).
    merge_multi_task_config(cfg, args, mode_key)
    # iter13 v12: motion_aux block — same per-mode flatten + CLI override pattern.
    merge_motion_aux_config(cfg, args, mode_key)
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
    base_out = get_module_output_dir("m09a_pretrain", args.subset,
                                    sanity=args.SANITY, poc=args.POC)
    lam = cfg["drift_control"]["lambda_reg"]
    if lam is None:
        cfg["checkpoint"]["output_dir"] = str(base_out / "pending_ablation")
        return cfg
    lam_str = f"{lam:g}".replace(".", "_")
    cfg["checkpoint"]["output_dir"] = str(base_out / f"lambda{lam_str}")
    return cfg


# ═════════════════════════════════════════════════════════════════════════
# MODEL SETUP (Q1-Q5 corrected: no wrappers, vit_giant_xformers, RoPE)
# ═════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder, teacher encoder (EMA), and predictor. Vanilla (no LoRA)."""
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
    # The 2.1 predictor's predictor_embed expects 4 * 1664 = 6656 input dim.
    # Without this, predictor crashes: RuntimeError mat1 and mat2 shapes cannot be multiplied.
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = True

    # V-JEPA 2.1 requires RoPE (no pos_embed registered in model)
    if model_cfg["predict_all"] or model_cfg["n_output_distillation"] > 1:
        if not model_cfg["use_rope"]:
            print("FATAL: V-JEPA 2.1 requires use_rope=True (no pos_embed registered in model)")
            sys.exit(1)

    # iter13 Fix #2 (2026-05-04): wire `layer_freeze` from yaml into m09a.
    # The yaml directive (probe_pretrain.yaml:layer_freeze) was previously a SILENT
    # NO-OP — m09a never read it, so all 48 blocks were nominally trainable. With
    # the LR bump from 1e-5 → 1e-4 (Fix #1), an actual freeze of early blocks
    # anchors low-level visual features against catastrophic forgetting. Mirrors
    # the LayerLock convention (arXiv:2509.10156): ViT layers converge in depth
    # order, so the bottom is most stable and benefits least from continual
    # training. Freeze [0, freeze_below); train [freeze_below, n_blocks).
    # Norms (LayerNorm/LN) stay trainable everywhere — Meta convention.
    layer_freeze_cfg = cfg["layer_freeze"]
    if layer_freeze_cfg["enabled"]:
        freeze_below = layer_freeze_cfg["freeze_below"]
        n_blocks = len(student.blocks)
        for i, blk in enumerate(student.blocks):
            req = i >= freeze_below
            for param in blk.parameters():
                param.requires_grad = req
        # Norms are always trainable (Meta convention)
        for name, param in student.named_parameters():
            if "norm" in name or "ln" in name:
                param.requires_grad = True
        # patch_embed stays trainable iff freeze_below == 0 (consistent with
        # "freeze nothing" mode); otherwise frozen alongside early blocks.
        if freeze_below > 0:
            for name, param in student.named_parameters():
                if name.startswith("patch_embed"):
                    param.requires_grad = False
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        print(f"  layer_freeze: blocks [0, {freeze_below}) frozen + "
              f"[{freeze_below}, {n_blocks}) trainable + norms always trainable. "
              f"Trainable params: {trainable:,}/{total:,} = {100*trainable/total:.1f}%")
    else:
        print("  layer_freeze: disabled (all blocks trainable)")

    # Teacher (EMA copy, frozen)
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

    # Save initial params for drift control (on CPU to save ~4GB VRAM)
    init_params = {name: p.clone().detach().cpu()
                   for name, p in student.named_parameters()}

    del ckpt
    gc.collect()

    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "init_params": init_params,
        "explora_enabled": False,
    }


# ═════════════════════════════════════════════════════════════════════════
# TRAINING LOOP (Ch10 vanilla pretrain)
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

    # Auto-bootstrap probe labels (action_labels.json + taxonomy_labels.json) if
    # missing. Lets `python -u src/m09a_pretrain.py ...` run end-to-end without
    # the shell having pre-run run_probe_eval.sh Stage 1. No-op when both files
    # already exist (~1ms stat()s). Mirrors run_probe_eval.sh:342-358 exactly.
    mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
    ensure_probe_labels_for_mode(
        mode_flag=mode_flag,
        project_root=Path(__file__).parent.parent,
        cache_policy=args.cache_policy,
    )

    output_dir = Path(cfg["checkpoint"]["output_dir"])
    student_path = output_dir / "student_encoder.pt"
    # iter11 v3 (2026-04-26): cache-policy=2 nukes the WHOLE output_dir at startup
    # so load_checkpoint() finds nothing → fresh step-0 run.
    wipe_output_dir(output_dir, args.cache_policy, label=f"output_dir ({output_dir.name})")
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

    # Gradient checkpointing — enabled at SANITY 24GB to fit ViT-G; disabled at
    # FULL 96GB for max throughput. Mirrors m09c_surgery.py:418-420. Flag value
    # was flattened from per-mode dict by merge_config_with_args() above.
    # Direct subscript (NOT .get()) — base_optimization.yaml provides scalar
    # default `gradient_checkpointing: false` so the key is always present.
    if cfg["optimization"]["gradient_checkpointing"]:
        enable_gradient_checkpointing(student)

    # Multi-task probe head — adds CrossEntropy/BCE loss on 16 taxonomy dims
    # to JEPA L1 → direct gradient signal toward eval metric. See
    # utils/multi_task_loss.py + base_optimization.yaml `multi_task_probe`.
    mt_head, mt_labels_by_clip, mt_dims_spec, mt_cfg = build_multi_task_head_from_cfg(cfg, device)

    # iter13 v12 (2026-05-06): motion_aux head — JOINT K-class CE + 13-D MSE on
    # RAFT optical-flow targets from m04d. Sole supervised aux loss in v12 (replaces
    # multi_task_probe which was disabled in probe_pretrain.yaml; v11 confirmed
    # 15 retrieval tag dims gave flat motion-flow top1).
    ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)

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
    # iter11 v3 (2026-04-26): val + ckpt share the SAME cadence
    # (`checkpoint.saves_per_epoch`). Removed redundant `validation.evals_per_epoch`.
    val_interval = ckpt_interval

    print(f"Train clips: {n_train:,} | Val clips: {len(val_key_set):,}")
    print(f"Epochs: {max_epochs} | Steps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(f"Checkpoint every {ckpt_interval} steps ({saves_per_epoch}x/epoch, keep last {keep_last_n})")
    print(f"Validation every {val_interval} steps ({saves_per_epoch}x/epoch, {len(val_key_set)} val clips)")

    # Optimizer & scheduler (cosine over total_steps)
    optimizer = build_optimizer(student, predictor, cfg["optimization"])
    # Multi-task head gets its own param group at base_lr × head_lr_multiplier.
    # Must run AFTER build_optimizer so 8-bit/paged AdamW machinery is intact.
    attach_head_to_optimizer(optimizer, mt_head, mt_cfg, cfg["optimization"]["lr"])
    # iter13 v12: motion_aux head gets its own param group too (independent
    # head_lr_multiplier from multi_task — both default to 10×).
    attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, cfg["optimization"]["lr"])
    scheduler = build_scheduler(optimizer, cfg["optimization"], total_steps)
    mp_cfg = cfg["mixed_precision"]
    use_scaler = mp_cfg["enabled"] and mp_cfg["dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Resume from checkpoint
    start_step = 0
    # iter13 v11 (2026-05-05): track best by probe_top1 (higher=better), not val_loss.
    # v10 demonstrated val_loss keeps dropping while encoder reverts to init under L2
    # anchor pullback — picking the lowest-val_loss ckpt picks a degraded encoder.
    # probe_top1 reflects actual representation quality at every val cycle.
    best_probe_top1 = -1.0
    ckpt_path = output_dir / f"{CHECKPOINT_PREFIX}_latest.pt"
    if ckpt_path.exists():
        # `best_metric` slot in the ckpt now stores best_probe_top1 (was best_val_loss in v10).
        start_step, best_probe_top1 = load_training_checkpoint(
            ckpt_path, student, teacher, predictor, optimizer, scheduler, scaler)
        print(f"Resumed from step {start_step}, best probe_top1: {best_probe_top1:.4f}")

    # ── Collect val clips into memory (from --val-subset, once before training) ──
    val_batches = []
    val_collected_keys = []

    if val_key_set:
        print(f"\nCollecting {len(val_key_set)} val clips into memory...")
        _val_tmp = tempfile.mkdtemp(prefix="m09a_val_")
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
            # SANITY validates code paths only — partial val coverage is fine. Skip
            # the auto-download (which requires --master-tags that this caller
            # doesn't have) and proceed with whatever val clips were collected.
            if args.SANITY:
                print(f"  [SANITY] Only {len(val_collected_keys)}/{len(val_key_set)} val clips ({pct:.0f}%) — proceeding (SANITY = code-path validation, not metric quality)")
            elif len(val_collected_keys) >= 5:
                print(f"  WARN: Only {len(val_collected_keys)}/{len(val_key_set)} val clips ({pct:.0f}%) — proceeding with partial coverage (>=5 clips ok for val_jepa)")
            else:
                print(f"WARNING: Only {len(val_collected_keys)}/{len(val_key_set)} val clips ({pct:.0f}%). Auto-downloading...")
                subprocess.run([sys.executable, "-u", "src/m00d_download_subset.py",
                               "--FULL", "--subset", args.val_subset, "--no-wandb"], check=True)
                # Retry collection with fresh data — ONLY after auto-download (which
                # may have fetched the missing clips). Skipped when SANITY bypass or
                # >=5-clips bypass fired above (those already have usable val_batches).
                val_batches = []
                val_collected_keys = []
                _val_tmp = tempfile.mkdtemp(prefix="m09a_val_retry_")
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
        # --val-subset is required=True (CLAUDE.md no-default rule), so val_key_set
        # being empty means the JSON file itself was empty — degenerate, FAIL LOUD.
        print(f"FATAL: --val-subset {args.val_subset} loaded but val_key_set is empty.")
        sys.exit(1)

    # ── Mid-training probe init (iter13 backport from m09c) ─────────────
    # Pre-decode probe clips ONCE so subsequent run_probe_acc_eval calls only
    # do GPU forwards (~10-30 s/eval). Skipped when cfg.probe.enabled=False
    # (SANITY mode auto-disables — n=21 too small for stable BCa CI).
    probe_clips, probe_labels = None, None
    probe_history = []
    probe_history_path = output_dir / "probe_history.jsonl"
    # iter13 Task #19: per-validation block-drift diagnostic. Each entry is
    # {"step", "rel_l2_per_block": [n_blocks], "freeze_below": int}. Plot
    # rendered every val cycle to m09_block_drift.{png,pdf} → catches the
    # uniform-noise pathology (all 48 blocks at ~1e-5) in 1 min instead of 5 h.
    block_drift_history = []

    # iter13 (2026-05-05): per-STEP block_drift_mean uses the SAME metric as
    # per-val block_drift_mean — `mean_i(‖Δθ_block_i‖ / ‖θ₀_block_i‖)` over the
    # 48 transformer blocks. Calls utils.plots.compute_block_drift each step.
    # Earlier (same-day) we used a cheap global rel_l2 proxy
    # (`sqrt(loss_drift / lambda / theta_init_l2_sq)`) — that worked but gave
    # NUMERICALLY DIFFERENT values from the val-cycle metric, causing user
    # confusion when drift_table.py merged both sources. Now both per-step and
    # per-val records carry identical-semantics block_drift_mean. Cost: ~100 ms
    # per step (588 named_parameters × .cpu() + norm + reduction). Total run
    # overhead: ~100ms × 1085 ≈ 1.8 min on a 7-h run = 0.4%.
    from utils.plots import compute_block_drift  # local import: avoid circular
    plateau_state = {"recent_val_losses": []}      # plateau detector state
    best_state = {"val_loss": float("inf"), "step": -1, "probe_acc": -1.0}
    kill_state = {"triggered": False, "reason": None}
    if cfg.get("probe", {}).get("enabled"):
        probe_cfg = cfg["probe"]
        action_labels_path = (args.probe_action_labels or
                              str(Path(probe_cfg["subset"]).parent / "action_labels.json"))
        if not Path(action_labels_path).exists():
            print(f"  [probe] WARN: action_labels.json not found at {action_labels_path} — "
                  f"falling back to val_jepa-only probe (no top-1 acc)", flush=True)
            probe_labels = {}
        else:
            probe_labels = load_action_labels(Path(action_labels_path))
        try:
            print(f"  [probe] decoding clips from {probe_cfg['subset']} ...", flush=True)
            probe_clips = build_probe_clips(
                probe_subset_path=probe_cfg["subset"],
                probe_local_data=probe_cfg["local_data"],
                probe_tags_path=probe_cfg["tags_path"],
                num_frames=cfg["data"]["num_frames"],
                crop_size=cfg["model"]["crop_size"],
                max_clips=cfg["monitoring"]["knn_probe_clips"],   # cap N to avoid /dev/shm overflow
            )
            print(f"  [probe] decoded {len(probe_clips)} clips ({len(probe_labels)} have action labels)", flush=True)
        except Exception as _probe_build_err:
            # iter13 (2026-05-05): per CLAUDE.md FAIL HARD — no WARN-without-exit.
            # Mid-train probe is research signal (top1 trajectory across val cycles),
            # not optional telemetry. Re-raise so the run aborts and the bug gets fixed.
            print(f"  [probe] FATAL: build_probe_clips failed: {_probe_build_err}", flush=True)
            print("  [probe] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
            raise

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
    wb_run = init_wandb("m09a", f"{mode}{lam_tag}", config=cfg, enabled=not args.no_wandb)

    # Training config
    ema_momentum = cfg["optimization"]["ema_momentum"]
    loss_exp = cfg["optimization"]["loss_exp"]
    drift_cfg = cfg["drift_control"]
    dtype = getattr(torch, mp_cfg["dtype"])

    pbar = make_pbar(total=total_steps, initial=start_step,
                     desc="m09a_pretrain", unit="step")

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

    # AdaptiveBatchSizer for gradient accumulation (#48). Effective BS stays = batch_size
    # (preserves optimizer dynamics: Adam momentum, LR scaling, weight decay scheduling),
    # but each forward+backward runs on a micro-batch sized by VRAM availability. Sub-batches
    # accumulate gradients before a single optimizer step. All 3 sizer params from yaml.
    _gpu_cfg = get_pipeline_config()["gpu"]
    train_sizer = AdaptiveBatchSizer(
        initial_size=min(_gpu_cfg["training_initial_bs"], batch_size),
        min_size=1, max_size=batch_size,
        memory_cap=_gpu_cfg["gpu_memory_target"])
    print(f"AdaptiveBatchSizer (training, grad-accum): start={train_sizer.size}, "
          f"max={batch_size} (= effective BS), target VRAM={_gpu_cfg['gpu_memory_target']:.0%}")

    # Danger zone #1 fix: local NaN-strike counter (was main._nan_strikes in m09_pretrain.py).
    nan_strikes = 0

    # Pre-init loss locals — they'd otherwise be UnboundLocalError if 0
    # successful steps complete before the train loop exits (SANITY OOM
    # retry path can do this; see probe_pretrain_sanity_v3.log crash).
    jepa_val = drift_val = total_val = masked_val = context_val = 0.0
    lr_val = cfg["optimization"]["lr"]
    gn_val = 0.0
    mt_loss_val = 0.0

    step = start_step
    # Per-epoch memory hygiene — clears fragmented blocks accumulated across steps.
    # Paired with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True env at shell level (#47).
    _last_epoch_cleared = -1
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

            # Adaptive grad-accumulation forward+backward (#48). Replaces the inline
            # forward/backward block — same semantics (loss-scaled by micro/macro ratio so
            # accumulated gradient is bit-equivalent to a single full-batch step), but the
            # micro-batch is sized by AdaptiveBatchSizer to track VRAM target.
            #
            # Bug B fix (iter13, mirrors m09c #55): retry the SAME macro on OOM
            # rather than `continue` to the next step. The old `continue` advanced
            # the for-loop iterator → SANITY total_steps=1 exited the loop with 0
            # successful steps → exported untrained Meta weights silently
            # (probe_pretrain_sanity_v3.log:60). The while-loop here halves the
            # sub-batch via sizer.on_oom() (fired inside the helper) and retries
            # at the new size until success OR sub-batch hits min — at which point
            # we fail-hard with the standard #55 mitigation hint.
            step_succeeded = False
            while not step_succeeded:
                try:
                    (jepa_val, masked_val, context_val, drift_val,
                     _infonce_val, _tcc_val,
                     _uw_w_jepa, _uw_w_infonce, _uw_w_tcc) = _train_step_grad_accum(
                        student, teacher, predictor, batch_clips,
                        all_masks_enc, all_masks_pred,
                        cfg, dtype, mp_cfg, scaler, train_sizer, loss_exp,
                        init_params=init_params, drift_cfg=drift_cfg,
                        loss_cfg=cfg["optimization"]["loss"], uw=None)
                    step_succeeded = True
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()  # discard partial grads from incomplete macro
                    # Force release of fragmented memory between retries (probe_pretrain_sanity_v4.log
                    # bug: orphan tensors from the failed forward stayed allocated, so each
                    # successive sub-batch shrink started with LESS free VRAM than the previous
                    # → eventually OOM at sub-batch=1 even though sub-batch=1 forward should fit).
                    # gc.collect() releases Python references; empty_cache() returns blocks to
                    # the CUDA driver. Without these, ~3 OOMs of orphan memory accumulated.
                    gc.collect()
                    torch.cuda.empty_cache()
                    if train_sizer.size <= train_sizer.min_size:
                        raise RuntimeError(
                            f"m09a step {step}: OOM persists at minimum sub-batch="
                            f"{train_sizer.size}. GPU memory budget cannot fit V-JEPA "
                            f"ViT-G + AdamW state. Gold-standard mitigations: (1) gradient "
                            f"checkpointing on transformer blocks, (2) bitsandbytes AdamW8bit, "
                            f"(3) move pretrain to FULL hardware (96GB). See errors_N_fixes.md #55."
                        ) from None
                    print(f"  OOM at step {step}: sub-batch shrunk to "
                          f"{train_sizer.size}, retrying SAME macro")
            total_val = jepa_val + drift_val

            # Multi-task forward+backward (no-op when mt_head is None).
            # Accumulates onto the same param.grad buffer as the JEPA grads
            # → single optimizer.step() below consumes both losses.
            try:
                mt_loss_val, mt_per_dim = run_multi_task_step(
                    student, mt_head, mt_cfg, mt_labels_by_clip, mt_dims_spec,
                    batch_clips, batch_keys, scaler, mp_cfg, dtype, device)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad()
                print(f"  OOM at step {step} (multi-task forward): macro discarded, retrying")
                torch.cuda.empty_cache()
                continue

            # iter13 v12: motion_aux forward+backward (no-op when ma_head is None).
            # Same param.grad accumulation pattern → optimizer.step() consumes
            # JEPA + multi_task + motion_aux gradients in one update.
            try:
                ma_loss_val, ma_per_branch = run_motion_aux_step(
                    student, ma_head, ma_cfg, ma_lookup,
                    batch_clips, batch_keys, scaler, mp_cfg, dtype, device)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad()
                print(f"  OOM at step {step} (motion_aux forward): macro discarded, retrying")
                torch.cuda.empty_cache()
                continue
            if mt_head is not None:
                total_val = jepa_val + drift_val + mt_loss_val * float(mt_cfg["weight_probe"])

            # Single optimizer step per macro batch — preserves effective BS = batch_size
            scaler.unscale_(optimizer)
            _clip_params = list(student.parameters()) + list(predictor.parameters())
            if mt_head is not None:
                _clip_params += list(mt_head.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                _clip_params, cfg["optimization"]["grad_clip"])
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

            # Per-step logging + NaN/Inf guard. jepa_val/masked_val/context_val/drift_val/
            # total_val already populated by _train_step_grad_accum (#48) — they are the
            # macro-batch mean losses (weighted sum across micro-batches).
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

            # iter13 (2026-05-05): per-step block_drift_mean — IDENTICAL metric
            # to per-val (utils.training.track_block_drift_at_val). Both compute
            # `mean_i(‖Δθ_block_i‖ / ‖θ₀_block_i‖)`. Earlier same-day we used a
            # cheap global rel_l2 proxy — fast but numerically different, causing
            # drift_table.py to mix two metrics under one column. Now unified.
            drift_per_block = compute_block_drift(student, init_params)
            block_drift_mean = (float(sum(drift_per_block) / len(drift_per_block))
                                if drift_per_block else 0.0)
            step_record = {
                "step": step, "epoch": epoch,
                "loss_jepa": round(jepa_val, 6),
                "loss_masked": round(masked_val, 6),
                "loss_context": round(context_val, 6),
                "loss_drift": round(drift_val, 6),
                "loss_total": round(total_val, 6), "lr": lr_val,
                "grad_norm": round(gn_val, 4), "throughput": round(throughput, 2),
                "block_drift_mean": round(block_drift_mean, 8),
            }
            if mt_head is not None:
                step_record["loss_multi_task"] = round(mt_loss_val, 6)
                step_record["loss_multi_task_per_dim"] = {
                    d: round(v, 6) for d, v in mt_per_dim.items()}
            # iter13 v12: motion_aux per-step logging (CE + MSE branches, n_kept).
            if ma_head is not None:
                step_record["loss_motion_aux"]     = round(ma_loss_val, 6)
                step_record["loss_motion_aux_ce"]  = round(ma_per_branch.get("ce", 0.0), 6)
                step_record["loss_motion_aux_mse"] = round(ma_per_branch.get("mse", 0.0), 6)
                step_record["motion_aux_n_kept"]   = ma_per_branch.get("n_kept", 0)
            _log_step(step_record)  # JSONL: crash-safe (fsync per write)

            csv_writer.writerow([step, epoch, f"{jepa_val:.6f}", f"{drift_val:.6f}",
                                 f"{total_val:.6f}", f"{lr_val:.2e}",
                                 f"{gn_val:.4f}", f"{throughput:.2f}", ""])
            if step % 10 == 0:
                csv_file.flush()

            wb_metrics = {
                "loss/jepa": jepa_val,
                "loss/masked": masked_val,
                "loss/context": context_val,
                "loss/drift": drift_val,
                "loss/total": total_val,
                "lr": lr_val,
                "grad_norm": gn_val,
                "epoch": epoch,
                "throughput_steps_per_s": throughput,
            }
            if mt_head is not None:
                wb_metrics["loss/multi_task"] = mt_loss_val
                for d, v in mt_per_dim.items():
                    wb_metrics[f"loss/multi_task/{d}"] = v
            # iter13 v12: motion_aux wandb metrics.
            if ma_head is not None:
                wb_metrics["loss/motion_aux"]     = ma_loss_val
                wb_metrics["loss/motion_aux/ce"]  = ma_per_branch.get("ce", 0.0)
                wb_metrics["loss/motion_aux/mse"] = ma_per_branch.get("mse", 0.0)
            log_metrics(wb_run, wb_metrics, step=step)

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
                    step + 1, best_probe_top1, full=False)
                save_training_checkpoint(
                    ckpt_path, student, teacher, predictor, optimizer, scheduler,
                    scaler, step + 1, best_probe_top1, full=True)
                cleanup_old_checkpoints(output_dir, prefix=CHECKPOINT_PREFIX,
                                        keep_n=keep_last_n)

            # Periodic validation (every val_interval steps)
            if (step + 1) % val_interval == 0 and val_batches:
                pct = (epoch_step + 1) / steps_per_epoch * 100
                # iter13 (2026-05-05): run_validation now returns dict with
                # all 4 losses (jepa + multi_task + drift + total). Backward-
                # compat: pull `val_loss = result["jepa"]` for downstream
                # kill-switch / best-ckpt / plateau (which expect a scalar).
                # New fields land in probe_record (see end-of-val block).
                val_result = run_validation(
                    student, teacher, predictor, mask_generators,
                    val_batches, cfg, device, step + 1,
                    val_keys=val_collected_keys,
                    mt_head=mt_head, mt_dims_spec=mt_dims_spec,
                    mt_labels_by_clip=mt_labels_by_clip, mt_cfg=mt_cfg,
                    init_params=init_params, drift_cfg=drift_cfg)
                val_loss = val_result["jepa"]
                log_metrics(wb_run, {
                    "val/jepa_loss":       val_result["jepa"],
                    "val/multi_task_loss": val_result["multi_task"],
                    "val/drift_loss":      val_result["drift"],
                    "val/total_loss":      val_result["total"],
                    "val/epoch_pct":       pct,
                }, step=step)

                # Write all 4 val losses to JSONL (crash-safe) + CSV (legacy: jepa only).
                _log_step({"step": step, "epoch": epoch,
                           "val_loss":            round(val_result["jepa"], 6),
                           "val_multi_task_loss": round(val_result["multi_task"], 6),
                           "val_drift_loss":      round(val_result["drift"], 8),
                           "val_total_loss":      round(val_result["total"], 6),
                           "epoch_pct": round(pct, 1)})
                csv_writer.writerow([step, epoch, "", "", "",
                                     "", "", "", f"{val_loss:.6f}"])
                csv_file.flush()

                # Live training plots (regenerate on each validation)
                try:
                    plot_training_curves(
                        runs=[{"csv_path": str(csv_path),
                               "label": f"V-JEPA 2.0 (\u03bb={drift_cfg['lambda_reg']})",
                               "color": "blue",
                               "batch_size": batch_size}],
                        output_dir=str(output_dir),
                        title_prefix=f"{n_train:,} clips × {max_epochs} ep × BS={batch_size} × LR={cfg['optimization']['lr']:.1e} ({total_steps:,} steps)\n",
                        file_prefix="m09a",
                    )
                except Exception as e:
                    # iter13 (2026-05-05): per CLAUDE.md FAIL HARD — no silent
                    # except: pass. Plot failures often hide real bugs (mismatched
                    # CSV schema, missing column). Surface and abort.
                    print(f"  [plot] FATAL: training_curves render failed: {e}", flush=True)
                    print("  [plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                    raise

                # iter13: 4-loss decomposition (jepa | drift | multi_task | total) on
                # one image — total_loss BOLD shows which component drives the optimizer.
                # Critical for diagnosing the "multi-task dominates" pattern visible in
                # FULL training where jepa-L1 saturates but mt-loss drives total drop.
                try:
                    plot_combined_losses(
                        jsonl_path=output_dir / "loss_log.jsonl",
                        output_dir=output_dir,
                        title_prefix=f"m09a · LR={cfg['optimization']['lr']:.1e} · ",
                        file_prefix="m09a",
                    )
                except Exception as e:
                    # iter13 (2026-05-05): per CLAUDE.md FAIL HARD.
                    print(f"  [plot] FATAL: combined-loss render failed: {e}", flush=True)
                    print("  [plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                    raise

                # ── Mid-training m06d trio (top-1 + motion-cos + future-L1) ──
                # iter13 Task #11: replaces the iter11 kNN-centroid action probe.
                # Single shared encoder forward emits all 3 metrics (~25% cheaper
                # than running probe_action+probe_motion_cos+probe_future_mse
                # separately). Trajectory uses VAL split; paper-final m06d uses
                # TEST split (different N, different absolute numbers — by design).
                probe_record = {"step": step, "epoch": epoch,
                                "val_jepa_loss":       round(val_result["jepa"], 6),
                                "val_multi_task_loss": round(val_result["multi_task"], 6),
                                "val_drift_loss":      round(val_result["drift"], 8),
                                "val_total_loss":      round(val_result["total"], 6),
                                "epoch_pct": round(pct, 1)}
                if probe_clips is not None and probe_labels:
                    run_trio_at_val(
                        student, predictor, probe_clips, probe_labels,
                        mask_gen=mask_generators[0], cfg=cfg, device=device,
                        step=step, wb_run=wb_run, probe_record=probe_record)

                # iter13 Task #19: per-block drift diagnostic. Catches the
                # v5+v6+v7 "uniform ~1e-5 across all 48 blocks" pathology in
                # 1 min instead of 5 h of waiting on downstream metrics.
                _freeze_below = (cfg["layer_freeze"]["freeze_below"]
                                 if cfg["layer_freeze"]["enabled"] else 0)
                track_block_drift_at_val(
                    student, init_params, freeze_below=_freeze_below,
                    block_drift_history=block_drift_history,
                    output_dir=output_dir, step=step,
                    probe_record=probe_record,
                    title_prefix=f"m09a step={step} · ",
                    file_prefix="m09a")

                probe_history.append(probe_record)
                with open(probe_history_path, "a") as ph_f:
                    ph_f.write(json.dumps(probe_record) + "\n")
                    ph_f.flush()
                    os.fsync(ph_f.fileno())

                # ── Kill-switch ensemble: catastrophic + plateau on val_jepa ──
                # Both opt-in via cfg.probe.{kill_switch_enabled, plateau_enabled}
                # (per-mode-flattened). Same spread-window pattern as
                # m09c_surgery.py:892-905, but no stage gate (m09a is single-phase).
                probe_cfg_local = cfg.get("probe", {})
                # iter13 Task #25: use shared update_best_state_on_score helper
                # so m09a + m09c share identical best-ckpt selection plumbing.
                # m09a tracks val_loss (lower=better); the actual best.pt save
                # is gated separately at line ~1170 (different code path —
                # save_callback=None here).
                # iter13 v11 (2026-05-05): track best_state by probe_top1 (downstream
                # metric), NOT val_loss. v10 showed val_loss keeps dropping while
                # encoder reverts to init under L2 anchor pullback — picking
                # lowest-val_loss ckpt picks a degraded encoder. probe_top1 reflects
                # actual representation quality at every val cycle.
                probe_top1_for_best = probe_record.get("probe_top1", -1.0)
                update_best_state_on_score(
                    best_state, probe_top1_for_best, score_key="probe_top1",
                    higher_is_better=True, step=step)
                # Mirror val_loss onto best_state for the plot helper (line 161-163)
                # which annotates the "val_loss AT best-top1 step" for context.
                best_state["val_loss_at_best"] = val_loss
                # Kill-switch tracks lowest-val_loss-EVER independently (different
                # purpose than best.pt selection — detect catastrophic INCREASE in
                # val_loss regardless of probe_top1 trajectory).
                if val_loss < best_state.get("min_val_loss_seen", float("inf")):
                    best_state["min_val_loss_seen"] = val_loss
                if probe_cfg_local.get("kill_switch_enabled"):
                    max_inc_pct = probe_cfg_local["forgetting_threshold_pct"]
                    min_val_seen = best_state.get("min_val_loss_seen", float("inf"))
                    bound = min_val_seen * (1.0 + max_inc_pct / 100.0)
                    if val_loss > bound:
                        kill_state["triggered"] = True
                        kill_state["reason"] = "catastrophic_val_jepa_increase"
                        print(f"  [kill] val_jepa {val_loss:.4f} > min_seen {min_val_seen:.4f} "
                              f"x (1 + {max_inc_pct}%) = {bound:.4f} -> catastrophic forgetting")
                if probe_cfg_local.get("plateau_enabled"):
                    plateau_state["recent_val_losses"].append(val_loss)
                    patience = probe_cfg_local["plateau_patience"]
                    min_delta = probe_cfg_local["plateau_min_delta"]
                    window = plateau_state["recent_val_losses"][-(patience + 1):]
                    if len(window) >= patience + 1:
                        spread = max(window) - min(window)
                        if spread < min_delta:
                            kill_state["triggered"] = True
                            kill_state["reason"] = "val_jepa_plateau"
                            print(f"  [plateau] val_jepa range over last {patience+1} probes = "
                                  f"{spread:.5f} < {min_delta} -> early-stop (plateau)")

                # m09a-specific live plots: val_loss curve + probe_acc trajectory.
                try:
                    _render_m09a_probe_plots(probe_history, output_dir,
                                             best_state, kill_state, drift_cfg)
                except Exception as e:
                    # iter13 (2026-05-05): per CLAUDE.md FAIL HARD.
                    print(f"  [plot] FATAL: live probe-plot render failed: {e}", flush=True)
                    print("  [plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                    raise

                # Best model selection by lowest val loss.
                # full=True (Bug A fix, iter13): include predictor + teacher + opt
                # state so probe_future_mse Stage 8 can load the predictor key.
                # Pre-iter13 used full=False which saved encoder-only → Stage 8
                # FATAL on KeyError 'predictor' (run_src_probe_sanity_v2.log:778).
                # File grows ~7GB → ~15GB; acceptable on 200GB workspace and
                # symmetric with surgery_base.yaml convention.
                # iter13 v11 (2026-05-05): best.pt save gate uses probe_top1 (higher=better),
                # not val_loss. v10 demonstrated val_loss keeps dropping while encoder
                # reverts to init — picking lowest-val_loss ckpt picks a degraded encoder.
                probe_top1_now = probe_record.get("probe_top1", -1.0)
                if probe_top1_now > best_probe_top1:
                    best_probe_top1 = probe_top1_now
                    # iter13 disk-budget fix (2026-05-04, after v6 ENOSPC at step 744):
                    # include_optimizer=False → best.pt drops 16 GB optimizer state.
                    # Downstream (m05 re-embed, Stage 8 future_mse) needs only
                    # student+teacher+predictor; optimizer is dead weight here.
                    # latest.pt still saves include_optimizer=True for resume.
                    save_training_checkpoint(
                        output_dir / f"{CHECKPOINT_PREFIX}_best.pt",
                        student, teacher, predictor, optimizer, scheduler,
                        scaler, step + 1, best_probe_top1,
                        full=True, include_optimizer=False)
                    print(f"  🎯 New best probe_top1: {best_probe_top1:.4f} "
                          f"(val_jepa at this step: {val_loss:.4f})")

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
            scaler, step + 1, best_probe_top1, full=True)
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

    # Bug B fix (iter13, mirrors m09c #55): fail-hard if zero successful training
    # steps completed. Without this, m09a would silently export the *initial*
    # (frozen Meta) student weights and report "TRAINING COMPLETE" — passing CI
    # but producing untrained encoders downstream (probe_pretrain_sanity_v3.log
    # was the canonical case: OOM at step 0, sizer shrunk, then for-loop ended
    # because total_steps=1). Per CLAUDE.md "Silent failures = garbage metrics".
    if step + 1 == start_step:
        raise RuntimeError(
            f"M09A FAILED: 0 successful training steps (start_step={start_step}, "
            f"step+1={step + 1}). The exported student would be identical to the input "
            f"frozen V-JEPA weights — refusing to write a misleading checkpoint. "
            f"See errors_N_fixes.md #55 for memory-budget mitigations. Likely cause: "
            f"OOM-retry exhausted sub-batch shrink budget in SANITY's total_steps=1 run."
        )

    # iter13 v13 R4 (2026-05-07): end-of-train via shared utils.
    # Step 1: export student encoder (m09a is vanilla → explora_enabled=False).
    # Step 2-4: shared finalize_training (assert_diverged + mt_head + ma_head).
    export_student_for_eval(student, student_path, explora_enabled=False)
    init_ckpt_path = Path(__file__).parent.parent / cfg["model"]["checkpoint_path"]
    finalize_training(
        student=student, mt_head=mt_head, mt_dims_spec=mt_dims_spec,
        ma_head=ma_head, output_dir=output_dir,
        init_ckpt_path=init_ckpt_path,
        embed_dim=cfg["model"]["embed_dim"],
        label="m09a pretrain encoder",
    )

    # iter11 META-fix: gate post-training checkpoint cleanup through --cache-policy.
    # student_encoder.pt is the deliverable, but intermediate step*.pt files are
    # resume points — user must explicitly authorize their destruction.
    # Bug A complement (iter13): preserve `_best.pt`. Bug A fixed content (full=True)
    # so it carries the predictor; this glob would still wipe the file. Stage 8
    # future_mse requires it as input; without this skip, downstream FATAL.
    # Mirrors m09c R8 pattern (cleanup glob `_stage*.pt` excludes `_best.pt`).
    BEST_NAME = f"{CHECKPOINT_PREFIX}_best.pt"
    for ckpt_file in output_dir.glob(f"{CHECKPOINT_PREFIX}_*.pt"):
        if ckpt_file.name == BEST_NAME:
            continue
        guarded_delete(ckpt_file, args.cache_policy,
                       label=f"m09a checkpoint {ckpt_file.name}")

    # Save training summary as JSON (for winner selection, not stdout parsing)
    total_epochs_done = (step + 1) / max(steps_per_epoch, 1)
    summary = {
        "steps": step + 1,
        "epochs": round(total_epochs_done, 2),
        "clips_seen": (step + 1) * batch_size,
        "final_jepa_loss": jepa_val,
        "final_drift_loss": drift_val,
        "final_total_loss": total_val,
        # iter13 v11: best_state now tracked by probe_top1; keep "best_val_loss"
        # field name in training_summary.json for backward compat with downstream
        # consumers (lambda-ablation winner picker), but value is now NaN-equivalent
        # (-1.0) when probe_top1 path is active.
        "best_probe_top1": best_probe_top1,
        "best_val_loss": -1.0,    # legacy field — see above
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
    # Reduce CUDA fragmentation — pairs with AdaptiveBatchSizer + per-epoch gc (#47/#48/#53).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 continual pretraining on Indian urban clips (Ch10 vanilla)")
    parser.add_argument("--config", type=str, default=None,
                        help="Legacy single YAML config (backward compat with train_pretrain.sh)")
    add_model_config_arg(parser)
    add_train_config_arg(parser)
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
    parser.add_argument("--val-subset", required=True,
                        help="Path to val subset JSON (e.g., data/val_1k.json). "
                             "These clips are excluded from training and used for periodic val loss.")
    parser.add_argument("--val-local-data", required=True,
                        help="Local WebDataset dir for val clips (e.g., data/val_1k_local).")
    # Mid-training probe (iter13 backport from m09c). When --probe-subset is given,
    # m09a runs run_probe_acc_eval (top-1 action accuracy) every probe interval +
    # writes to probe_history.jsonl + applies plateau/catastrophic kill-switch.
    # SANITY mode auto-disables (n=21 too small for stable BCa CI).
    parser.add_argument("--probe-subset", type=str, default=None,
                        help="Path to probe-eval subset JSON (overrides cfg.probe.subset).")
    parser.add_argument("--probe-local-data", type=str, default=None,
                        help="Local WebDataset dir for probe clips (overrides cfg.probe.local_data).")
    parser.add_argument("--probe-tags", type=str, default=None,
                        help="Path to tags.json for probe clips (overrides cfg.probe.tags_path).")
    parser.add_argument("--probe-action-labels", type=str, default=None,
                        help="Path to action_labels.json (REQUIRED if --probe-subset given). "
                             "Provides clip_key → class_id mapping for top-1 accuracy compute.")
    parser.add_argument("--no-probe", action="store_true",
                        help="Force-disable mid-training probe (overrides cfg.probe.enabled).")
    # Multi-task probe head supervision (iter13). When enabled, adds CrossEntropy /
    # BCEWithLogits losses on 16 taxonomy dims (action + 15 taxonomy) on top of
    # JEPA L1 → direct gradient signal toward the eval metric. See
    # utils/multi_task_loss.py + base_optimization.yaml `multi_task_probe:` block.
    parser.add_argument("--taxonomy-labels-json", type=str, default=None,
                        help="Path to taxonomy_labels.json (overrides cfg.multi_task_probe.labels_path). "
                             "Produced by `probe_taxonomy.py --stage labels`.")
    parser.add_argument("--no-multi-task", action="store_true",
                        help="Force-disable multi-task probe-head supervision "
                             "(overrides cfg.multi_task_probe.enabled).")
    # iter13 v12 (2026-05-06): motion_aux loss — joint K-class CE + 13-D MSE
    # on m04d's RAFT optical-flow targets. See utils/motion_aux_loss.py +
    # configs/train/probe_pretrain.yaml `motion_aux:` block.
    parser.add_argument("--motion-features-path", type=Path, default=None,
                        help="Path to m04d motion_features.npy (overrides "
                             "cfg.motion_aux.motion_features_path). Companion "
                             ".paths.npy must exist alongside.")
    parser.add_argument("--no-motion-aux", action="store_true",
                        help="Force-disable motion_aux supervised loss "
                             "(overrides cfg.motion_aux.enabled).")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    # Cache-policy gate (iter11): every destructive delete in this module must route
    # through utils.cache_policy.guarded_delete(path, args.cache_policy, ...).
    # --cache-policy defaults to 1 (keep) so overnight re-runs never destroy cache.
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    # Cache-policy prompt — shells stay thin (CLAUDE.md DELETE PROTECTION).
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)

    # Load config: --model-config + --train-config (new) or --config (legacy).
    # Both are required=True via add_model_config_arg/add_train_config_arg, so
    # args.train_config is guaranteed unless --config (legacy) is used.
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_merged_config(args.model_config, args.train_config)
    cfg = merge_config_with_args(cfg, args)

    # Auto-ablation: if no --lambda-reg specified, find or run ablation
    if args.lambda_reg is None:
        out_dir = Path(cfg["checkpoint"]["output_dir"]).parent
        ablation_dir = out_dir / "ablation"
        winner_json = ablation_dir / "ablation_winner.json"

        if winner_json.exists():
            w = json.load(open(winner_json))
            cfg["drift_control"]["lambda_reg"] = float(w["winner_lambda"])
            lam_str = f"{float(w['winner_lambda']):g}".replace(".", "_")
            cfg["checkpoint"]["output_dir"] = str(out_dir / f"lambda{lam_str}")
            print(f"Using ablation winner: lambda={w['winner_lambda']} "
                  f"(best_val_loss={w.get('winner_val_loss', '?')}) from {winner_json}")
        else:
            ablation_dir = out_dir / "ablation"
            print(f"\n{'='*60}")
            print(f"  AUTO-ABLATION: No {winner_json} found.")
            print("  Running 4 lambda sweep on data/subset_10k_local (~2h)")
            print(f"  Ablation (10K) → {ablation_dir}/lambda*/")
            print(f"  Full training  → {out_dir}/lambda<winner>/")
            print(f"{'='*60}\n")

            drift_cfg = cfg["drift_control"]
            ablation_lambdas = drift_cfg["ablation_lambdas"]

            for lam in ablation_lambdas:
                lam_str = f"{lam:g}".replace(".", "_")
                lam_out = ablation_dir / f"lambda{lam_str}"

                # Skip if already trained with valid val_loss
                summary_path = lam_out / "training_summary.json"
                if summary_path.exists():
                    s = json.load(open(summary_path))
                    if s.get("best_val_loss") is not None and s["best_val_loss"] != float("inf"):
                        print(f"  lambda={lam}: already trained (val_loss={s['best_val_loss']:.4f}), skipping")
                        continue

                print(f"\n--- Ablation: lambda={lam} ---")
                mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
                config_flags = (["--config", args.config] if args.config
                                else ["--model-config", args.model_config,
                                      "--train-config", args.train_config])
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
                # Pass parent's resolved cache-policy so ablation worker doesn't re-prompt.
                ablation_cmd += ["--cache-policy", args.cache_policy]
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
            cfg["checkpoint"]["output_dir"] = str(out_dir / f"lambda{lam_str}")
            print(f"\nProceeding with winner: lambda={w['winner_lambda']} "
                  f"(best_val_loss={w['winner_val_loss']:.4f})")

    # Dispatch: vanilla pretrain only (m09b/m09c are separate modules)
    train(cfg, args)


def select_ablation_winner(output_dir: str, lambdas: list):
    """Compare best_val_loss across lambda ablation runs → write ablation_winner.json.

    USAGE (lambdas + configs all required — NO DEFAULT per CLAUDE.md):
        python -u src/m09a_pretrain.py --select-winner outputs/poc \\
            configs/model/vjepa2_1.yaml configs/legacy2/ch10_pretrain.yaml \\
            2>&1 | tee logs/m09a_select_winner.log
    """
    # Bug fix 2026-04-27: removed `if lambdas is None` fallback that referenced
    # DEFAULT_MODEL_CONFIG / DEFAULT_TRAIN_CONFIG (undefined names — F821 crash).
    # Per CLAUDE.md NO DEFAULT: caller MUST pass `lambdas` explicitly.
    assert isinstance(lambdas, list) and lambdas, \
        "select_ablation_winner: `lambdas` must be a non-empty list"

    out = Path(output_dir)
    results = {}

    print("\n=== Lambda Ablation Winner Selection ===")
    print(f"Output dir: {out}")
    print("Selection metric: best_val_loss (lowest wins)\n")

    for lam in lambdas:
        lam_dir = "lambda" + lam.replace(".", "_")
        summary_path = out / lam_dir / "training_summary.json"
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
        plot_val_loss_curves(out, lambdas, winner)
    except Exception as e:
        # iter13 (2026-05-05): per CLAUDE.md FAIL HARD.
        print(f"  FATAL: ablation plot failed: {e}", flush=True)
        raise


if __name__ == "__main__":
    import traceback as _traceback
    try:
        # --select-winner subcommand: positional args = output_dir, model_config, train_config.
        # All required — derive lambdas from cfg["drift_control"]["ablation_lambdas"]
        # (NO DEFAULT, per CLAUDE.md "no hardcoded paths" rule).
        if len(sys.argv) >= 2 and sys.argv[1] == "--select-winner":
            if len(sys.argv) != 5:
                print("FATAL: --select-winner requires 3 positional args:")
                print("  python -u src/m09a_pretrain.py --select-winner "
                      "<output_dir> <model_config.yaml> <train_config.yaml>")
                sys.exit(2)
            _out_dir, _model_cfg_path, _train_cfg_path = sys.argv[2], sys.argv[3], sys.argv[4]
            _cfg = load_merged_config(_model_cfg_path, _train_cfg_path)
            _lambdas = [str(l) for l in _cfg["drift_control"]["ablation_lambdas"]]
            select_ablation_winner(_out_dir, _lambdas)
        else:
            main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)   # force exit: CUDA atexit + producer threads can deadlock at exit
    except SystemExit:
        raise         # honor explicit sys.exit() codes
    except BaseException as _exc:
        # Fail-loud + force-kill non-daemon threads (producer TAR readers, multi_task
        # forwards) that otherwise keep the process alive after the main thread raised.
        # 2026-05-03 incident: disk-full at step 86 left PID 294680 alive 1h+ after
        # torch.save raised — log captured the traceback but tmux session never freed.
        # Mirrors m10_sam_segment.py:1127-1133 pattern.
        print(f"\nFATAL (unhandled m09a exception): {type(_exc).__name__}: {_exc}",
              file=sys.stderr)
        _traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
