"""Ch11 Factor Surgery — 3-stage progressive unfreezing with D_L/D_A/D_I factor datasets. GPU-only.

Split from m09_pretrain.py on 2026-04-15 (#49). Pairs with m09a_pretrain.py (vanilla Ch10)
and m09b_explora.py (LoRA variant). Shared primitives live in utils.training.

Pipeline: m10 (Grounded-SAM) → m11 (factor datasets) → m09c (surgery training).
The paper novelty — factor-disentangled surgery on a frozen V-JEPA 2.1 encoder.

    python -u src/m09c_surgery.py --SANITY --model-config configs/model/vjepa2_1.yaml --train-config configs/train/ch11_surgery.yaml --factor-dir outputs/sanity/m11_factor_datasets/ --no-wandb 2>&1 | tee logs/m09c_sanity.log
    python -u src/m09c_surgery.py --POC --subset data/sanity_100_dense.json --factor-dir outputs/poc/m11_factor_datasets/ --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m09c_dense100.log
    python -u src/m09c_surgery.py --FULL --factor-dir outputs/full/m11_factor_datasets/ --local-data data/full_local --no-wandb 2>&1 | tee logs/m09c_full.log
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # Must be before torch import
os.environ.setdefault("MKL_NUM_THREADS", "1")   # Prevent OpenMP thread explosion in workers

import argparse
import copy
import csv
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm  # noqa: F401 — retained for parity; make_pbar is preferred

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    check_gpu,
    add_subset_arg, add_local_data_arg, get_output_dir, get_module_output_dir, load_subset,  # noqa: F401
    get_pipeline_config, load_merged_config,
    add_model_config_arg, add_train_config_arg,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup  # noqa: F401 — wired via utils.training
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)

import torch
import torch.nn.functional as F

from utils.progress import make_pbar

# vjepa2 imports via shim (avoids src/ namespace collision)
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
    get_mask_generator, get_apply_masks,  # noqa: F401 — consumed via utils.training helpers
)

# Constants
DEFAULT_MODEL_CONFIG = "configs/model/vjepa2_1.yaml"
DEFAULT_TRAIN_CONFIG = "configs/train/ch11_surgery.yaml"
CHECKPOINT_PREFIX = "m09c_ckpt"

# Shared training primitives — utils/training.py (Phase 1 of iter8 split).
from utils.training import (
    load_config,
    build_mask_generators,
    compute_jepa_loss, _train_step_grad_accum,  # noqa: F401 — _train_step_grad_accum kept for future grad-accum wiring
    update_teacher_ema,
    build_optimizer, build_scheduler, update_weight_decay,  # noqa: F401 — build_scheduler/update_weight_decay kept for future stage schedulers
    save_training_checkpoint, cleanup_old_checkpoints, load_training_checkpoint,  # noqa: F401 — cleanup_old_checkpoints/load_training_checkpoint kept for resume
    export_student_for_eval,
    set_trainable_prefix, enable_gradient_checkpointing,
    FactorSampler, build_factor_index, load_factor_clip,
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
    # Mode-gated memory-saver flags (#57): flatten the per-mode dicts in
    # ch11_surgery.yaml into scalars that build_optimizer + enable_gradient_checkpointing
    # read directly. SANITY (24GB) → both True; POC/FULL (96GB) → both False, so
    # research-quality runs use the published V-JEPA fp32 AdamW recipe without
    # the 8-bit extrapolation-risk confound. See errors_N_fixes.md #57.
    cfg["optimization"]["use_8bit_optim"] = \
        cfg["optimization"]["use_8bit_optim"][mode_key]
    cfg["optimization"]["gradient_checkpointing"] = \
        cfg["optimization"]["gradient_checkpointing"][mode_key]
    cfg["optimization"]["paged_optim"] = \
        cfg["optimization"]["paged_optim"][mode_key]
    if args.batch_size is not None:
        cfg["optimization"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        cfg["optimization"]["max_epochs"] = args.max_epochs

    # Output dir: explicit --output-dir, or auto from module + mode
    if getattr(args, "output_dir", None):
        cfg["checkpoint"]["output_dir"] = args.output_dir
        return cfg
    base_out = get_module_output_dir("m09c_surgery", args.subset,
                                    sanity=args.SANITY, poc=args.POC)
    cfg["checkpoint"]["output_dir"] = str(base_out)
    return cfg


# ═════════════════════════════════════════════════════════════════════════
# MODEL SETUP (vanilla ViT — progressive unfreezing handled in train_surgery)
# ═════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder, teacher encoder (EMA), and predictor.

    m09c uses vanilla ViT. set_trainable_prefix() (utils.training) drives the
    3-stage progressive unfreezing inside train_surgery — NO LoRA injection here.
    """
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

    del ckpt
    gc.collect()

    # m09c uses vanilla ViT — NO ExPLoRA LoRA injection. Progressive unfreezing
    # via set_trainable_prefix() is driven per-stage by train_surgery().
    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "explora_enabled": False,
    }


# ═════════════════════════════════════════════════════════════════════════
# SURGERY TRAINING (Ch11 — 3-stage progressive prefix unfreezing)
# ═════════════════════════════════════════════════════════════════════════

def train_surgery(cfg: dict, args):
    """3-stage progressive prefix unfreezing with factor datasets (Ch11 proposal Sec 11.5-11.6)."""
    check_gpu()
    device = torch.device("cuda")

    seed = cfg["data"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    gc.disable()
    gc.collect()

    # Local NaN-strike counter (replaces the main._nan_strikes global from m09).
    nan_strikes = 0  # noqa: F841 — wired into future dense NaN guard; kept for parity with m09a/b

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else get_module_output_dir(
        "m09c_surgery", args.subset, sanity=args.SANITY, poc=args.POC)
    output_dir.mkdir(parents=True, exist_ok=True)
    student_path = output_dir / "student_encoder.pt"

    from utils.output_guard import verify_or_skip
    if verify_or_skip(output_dir, {"student": student_path}, label="m09c surgery"):
        return

    mode_key = "sanity" if args.SANITY else ("poc" if args.POC else "full")
    mp_cfg = cfg["mixed_precision"]
    dtype = getattr(torch, mp_cfg["dtype"])
    scaler = torch.amp.GradScaler("cuda", enabled=(mp_cfg["dtype"] == "float16"))
    loss_exp = cfg["optimization"]["loss_exp"]
    ema_momentum = cfg["optimization"]["ema_momentum"]

    wb_run = init_wandb("m09c_surgery", mode_key.upper(), config=vars(args),
                        enabled=not args.no_wandb)

    # Build model (V-JEPA 2.1 with deep supervision + dense loss)
    models = build_model(cfg, device)
    student = models["student"]
    teacher = models["teacher"]
    predictor = models["predictor"]

    # Gradient checkpointing (#56): only on student (teacher runs under torch.no_grad
    # so no activations are stored — checkpointing would be wasted CPU overhead).
    # Enabled via configs/train/ch11_surgery.yaml:optimization.gradient_checkpointing.
    if cfg["optimization"]["gradient_checkpointing"]:
        enable_gradient_checkpointing(student)

    mask_generators = build_mask_generators(cfg)

    # Load factor datasets
    factor_dir = Path(args.factor_dir)
    if not factor_dir.exists():
        print(f"FATAL: factor_dir not found: {factor_dir}")
        sys.exit(1)
    manifest_file = factor_dir / "factor_manifest.json"
    if not manifest_file.exists():
        print(f"FATAL: factor_manifest.json not found in {factor_dir}")
        sys.exit(1)

    # Quality gate: check m10 summary before training (Rule 33: logic in Python)
    summary_file = factor_dir / "summary.json"
    if summary_file.exists():
        m10_summary = json.load(open(summary_file))
        if m10_summary.get("quality_gate") == "FAIL":
            print(f"FATAL: m10 quality gate FAILED (concept_recall={m10_summary['mean_concept_recall']:.2f})")
            print("  SAM 3.1 did not detect enough objects. Fix m10 before training.")
            sys.exit(1)
        print(f"  m10 quality: concept_recall={m10_summary['mean_concept_recall']:.2f} (PASS)")

    manifest = json.load(open(manifest_file))
    factor_index = build_factor_index(manifest,
                                       factor_dir / "D_L",
                                       factor_dir / "D_A",
                                       factor_dir / "D_I")

    # Surgery config
    surgery_cfg = cfg["surgery"]
    stages = surgery_cfg["stages"]
    depth = cfg["model"]["depth"]
    embed_dim = cfg["model"]["embed_dim"]
    n_levels = cfg["model"]["n_output_distillation"]
    predict_all = cfg["model"]["predict_all"]
    crop_size = cfg["model"]["crop_size"]
    num_frames = cfg["data"]["num_frames"]
    batch_size = args.batch_size if args.batch_size else cfg["optimization"]["batch_size"]

    # merge_config_with_args (line 88) already flattened the per-mode dict → int. Don't re-subscript (#52).
    max_epochs = cfg["optimization"]["max_epochs"]
    total_clips = len(factor_index)
    steps_per_epoch = max(total_clips // batch_size, 1)
    total_steps = steps_per_epoch * max_epochs

    print(f"\n{'='*60}")
    print(f"SURGERY TRAINING — {len(stages)} stages on {total_clips} factor clips")
    print(f"Model: {cfg['model']['arch']} ({depth} blocks, {embed_dim}-dim)")
    print(f"Epochs: {max_epochs} | Steps: {total_steps} | BS: {batch_size}")
    print(f"Dense loss: predict_all={predict_all} | Deep supervision: {n_levels} levels")
    print(f"{'='*60}")

    # AdaptiveBatchSizer for gradient accumulation (#48 / #53). Effective BS stays =
    # batch_size (preserves Adam momentum + LR scaling + WD schedule); each forward+backward
    # runs on a micro-batch sized by VRAM availability. Sub-batches accumulate gradients
    # before a single optimizer step. All 3 sizer params from yaml — same pattern as m09a/m09b.
    _gpu_cfg = get_pipeline_config()["gpu"]
    train_sizer = AdaptiveBatchSizer(
        initial_size=min(_gpu_cfg["training_initial_bs"], batch_size),
        min_size=1, max_size=batch_size,
        memory_cap=_gpu_cfg["gpu_memory_target"])
    print(f"AdaptiveBatchSizer (surgery, grad-accum): start={train_sizer.size}, "
          f"max={batch_size} (= effective BS), target VRAM={_gpu_cfg['gpu_memory_target']:.0%}")

    # Danger zone #1 fix: local NaN-strike counter (was main._nan_strikes in m09_pretrain.py).
    nan_strikes = 0  # noqa: F841 — wired into future dense NaN guard for surgery loop

    global_step = 0
    csv_path = output_dir / "loss_log.csv"
    jsonl_path = output_dir / "loss_log.jsonl"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "stage", "loss_jepa", "loss_masked", "loss_context", "lr", "grad_norm"])
    jsonl_file = open(jsonl_path, "w")

    def _log_step(record):
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()
        os.fsync(jsonl_file.fileno())

    try:
        for stage_idx, stage_cfg in enumerate(stages):
            stage_name = stage_cfg["name"]
            n_trainable = int(depth * stage_cfg["unfreeze_below"])
            stage_pct = stage_cfg["max_epochs_pct"]
            stage_steps = max(int(total_steps * stage_pct), 1)
            warmup_steps = stage_cfg["warmup_steps"]
            mode_mixture = stage_cfg["mode_mixture"]

            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage_name}")
            print(f"  Layers 0-{n_trainable} trainable | {stage_steps} steps | warmup {warmup_steps}")
            print(f"  Mixture: {mode_mixture}")
            print(f"{'='*60}")

            # Inter-stage cleanup (#58): explicitly release prior stage's optimizer/scheduler
            # /sampler BEFORE allocating the new stage's optimizer. Without this, Python's
            # assignment order (`optimizer = build_optimizer(...)`) evaluates RHS first,
            # briefly holding BOTH old and new optimizers on GPU → peak ~2× optimizer state
            # (~16 GB for Stage 3 vs steady-state 8.5 GB). On v6 this double-hold pushed
            # Stage 3 past 24 GB even with 8-bit + checkpointing. `ipc_collect` flushes
            # inter-process CUDA cache entries too (relevant for torch.compile'd models).
            if stage_idx > 0:
                del optimizer, scheduler, sampler
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                free, total = torch.cuda.mem_get_info(0)
                print(f"  Inter-stage cleanup: {(total - free) / 1e9:.1f} GB used / "
                      f"{total / 1e9:.1f} GB total after releasing Stage {stage_idx} state")

            # Set trainable prefix
            set_trainable_prefix(student, n_trainable)

            # Rebuild optimizer (only trainable params)
            optimizer = build_optimizer(student, predictor, cfg["optimization"])

            # Per-stage LR scheduler (warmup then constant)
            def lr_lambda(step, ws=warmup_steps):
                if step < ws:
                    return (step + 1) / max(ws, 1)
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            # Factor sampler for this stage
            sampler = FactorSampler(factor_index, mode_mixture)

            pbar = make_pbar(total=stage_steps, desc=f"surgery:{stage_name}", unit="step")

            # Track last loss values so end-of-stage summary still reports something
            # when every step in this stage early-`continue`s on OOM (matches m09b
            # pattern lines 683-688). Reset each stage so the summary reflects THIS
            # stage's last successful step, not the prior stage's.
            jepa_val = 0.0
            masked_val = 0.0
            context_val = 0.0
            lr_val = 0.0
            gn_val = 0.0

            for local_step in range(stage_steps):
                # Build batch from factor clips
                batch_tensors = []
                for _ in range(batch_size):
                    _, _, clip_path = sampler.sample()
                    clip_tensor = load_factor_clip(clip_path, num_frames, crop_size)
                    batch_tensors.append(clip_tensor)
                batch_clips = torch.stack(batch_tensors).to(device)
                batch_clips = batch_clips.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

                actual_bs = batch_clips.shape[0]

                # Generate masks
                all_masks_enc, all_masks_pred = [], []
                for mg in mask_generators:
                    m_enc, m_pred = mg(actual_bs)
                    all_masks_enc.append(m_enc.to(device))
                    all_masks_pred.append(m_pred.to(device))

                # Adaptive grad-accumulation forward+backward (#48 / #53 / #55). Replaces
                # the inline forward/backward block — same semantics (loss-scaled by
                # micro/macro ratio so accumulated gradient is bit-equivalent to a single
                # full-batch step), micro-batch sized by AdaptiveBatchSizer to track VRAM
                # target. Surgery: no drift control (init_params=None, drift_cfg=None).
                #
                # Within-step retry loop (#55): on OOM, sizer.on_oom() shrinks; we retry
                # the SAME macro at the new sub-batch instead of continuing to the next
                # step. With stage_steps=1 (SANITY) the old `continue` skipped to a non-
                # existent next step → 0 successful steps → silent success. Now we retry
                # until either (a) success, (b) sizer at min and OOMed → fail-hard.
                step_succeeded = False
                while not step_succeeded:
                    try:
                        jepa_val, masked_val, context_val, _drift_val = _train_step_grad_accum(
                            student, teacher, predictor, batch_clips,
                            all_masks_enc, all_masks_pred,
                            cfg, dtype, mp_cfg, scaler, train_sizer, loss_exp,
                            init_params=None, drift_cfg=None)
                        step_succeeded = True
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad()  # discard partial grads from incomplete macro
                        # sizer.on_oom() ran inside helper. If at min and still OOMing, fail hard.
                        if train_sizer.size <= train_sizer.min_size:
                            raise RuntimeError(
                                f"Stage {stage_name} step {global_step}: OOM persists at "
                                f"minimum sub-batch={train_sizer.size}. GPU memory budget "
                                f"({_gpu_cfg['gpu_memory_target']:.0%} cap on this device) "
                                f"cannot fit V-JEPA ViT-G + AdamW state for "
                                f"{int(depth * stage_cfg['unfreeze_below'])}/{depth} trainable "
                                f"blocks. Gold-standard mitigations: (1) gradient checkpointing "
                                f"on transformer blocks (~25% throughput hit), (2) bitsandbytes "
                                f"AdamW8bit (4× optimizer-state reduction), (3) move surgery to "
                                f"FULL hardware (96GB). See errors_N_fixes.md #55."
                            ) from None
                        print(f"  OOM at step {global_step}: sub-batch shrunk to "
                              f"{train_sizer.size}, retrying SAME macro")

                # Single optimizer step per macro batch — preserves effective BS = batch_size
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(predictor.parameters()),
                    cfg["optimization"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # EMA teacher update
                update_teacher_ema(student, teacher, ema_momentum)

                # Logging — values from _train_step_grad_accum are macro-batch means
                # (weighted sum of micro-batch values), preserving the per-step diagnostics.
                lr_val = scheduler.get_last_lr()[0]
                gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                step_record = {
                    "step": global_step, "stage": stage_name,
                    "loss_jepa": round(jepa_val, 6),
                    "loss_masked": round(masked_val, 6),
                    "loss_context": round(context_val, 6),
                    "lr": lr_val, "grad_norm": round(gn_val, 4),
                }
                _log_step(step_record)
                csv_writer.writerow([global_step, stage_name, f"{jepa_val:.6f}",
                                     f"{masked_val:.6f}", f"{context_val:.6f}",
                                     f"{lr_val:.2e}", f"{gn_val:.4f}"])

                log_metrics(wb_run, {
                    "loss/jepa": jepa_val, "loss/masked": masked_val,
                    "loss/context": context_val, "lr": lr_val,
                    "grad_norm": gn_val, "stage": stage_idx,
                }, step=global_step)

                global_step += 1
                pbar.update(1)

                if global_step % cfg["optimization"]["gc_interval"] == 0:
                    gc.collect()

            pbar.close()
            save_training_checkpoint(output_dir / f"{CHECKPOINT_PREFIX}_stage{stage_idx}.pt",
                                     student, teacher, predictor, optimizer, scheduler,
                                     scaler, global_step, 0.0, full=False)
            print(f"  Stage {stage_name} complete: {stage_steps} steps, loss={jepa_val:.4f}")

    finally:
        csv_file.close()
        jsonl_file.close()
        gc.enable()

    # Fail-hard if zero successful training steps across all 3 stages (#55).
    # Without this guard, the script would silently export the *initial* (frozen)
    # student weights and report "SURGERY COMPLETE" — passing CI but producing
    # garbage for downstream m05 eval. Per CLAUDE.md "Silent failures = garbage metrics".
    if global_step == 0:
        raise RuntimeError(
            "SURGERY FAILED: 0 successful training steps across all 3 stages. "
            "Every step OOMed at minimum sub-batch=1 OR helper raised before any "
            "macro completed. The exported student would be identical to the input "
            "frozen V-JEPA weights — refusing to write a misleading checkpoint. "
            "See errors_N_fixes.md #55 for memory-budget mitigations."
        )

    # Export student encoder (vanilla ViT — no LoRA merge)
    export_student_for_eval(student, student_path, explora_enabled=False)

    # Training summary
    summary = {
        "steps": global_step,
        "stages": [s["name"] for s in stages],
        "total_factor_clips": len(factor_index),
        "batch_size": batch_size,
        "final_loss": jepa_val,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Training curves (reuse utils/plots.py)
    from utils.plots import plot_training_curves
    plot_training_curves(
        [{"csv_path": str(csv_path), "label": "Surgery", "batch_size": batch_size}],
        str(output_dir), title_prefix="Surgery: ")

    finish_wandb(wb_run)
    print(f"\nSURGERY COMPLETE: {global_step} steps across {len(stages)} stages")
    print(f"  Exported: {student_path}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    # Reduce CUDA fragmentation — lets allocator grow existing segments rather than
    # reserving fresh blocks. Critical for grad-accum: small micro-batches + repeated
    # alloc/free patterns fragment the heap fast. Idempotent (env only read at first
    # CUDA init). Pairs with AdaptiveBatchSizer + per-epoch gc (#47/#48/#53).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="Ch11 factor surgery: 3-stage progressive unfreeze + D_L/D_A/D_I factor datasets")
    parser.add_argument("--config", type=str, default=None,
                        help="Legacy single YAML config (backward compat)")
    add_model_config_arg(parser)
    add_train_config_arg(parser)
    parser.add_argument("--factor-dir", type=str, default=None,
                        help="Factor dataset dir from m11 (contains D_L/, D_A/, D_I/, factor_manifest.json)")
    parser.add_argument("--SANITY", action="store_true",
                        help="Quick validation: 50 steps, batch_size=2")
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
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    if not args.factor_dir:
        print("FATAL: m09c_surgery requires --factor-dir (path to m11 factor datasets)")
        print("  Pipeline: m10 (Grounded-SAM) → m11 (factor datasets) → m09c (surgery)")
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

    # Dispatch: surgery (only mode in this module)
    train_surgery(cfg, args)


if __name__ == "__main__":
    main()

    # Force exit: CUDA atexit cleanup can deadlock on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
