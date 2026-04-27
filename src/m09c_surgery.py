"""Ch11 Factor Surgery — 3-stage progressive unfreezing with D_L/D_A/D_I factor datasets. GPU-only.

Split from m09_pretrain.py on 2026-04-15 (#49). Pairs with m09a_pretrain.py (vanilla Ch10)
and m09b_explora.py (LoRA variant). Shared primitives live in utils.training.

Pipeline: m10 (Grounded-SAM) → m11 (factor datasets) → m09c (surgery training).
The paper novelty — factor-disentangled surgery on a frozen V-JEPA 2.1 encoder.

USAGE (every path arg required — CLAUDE.md no-default rule; --probe-* fall back to yaml):
    python -u src/m09c_surgery.py --SANITY \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/surgery_2stage_noDI.yaml \
        --subset data/sanity_100_dense.json --local-data data/val_1k_local \
        --factor-dir outputs/sanity/m11_factor_datasets/ \
        --probe-subset data/val_1k.json --probe-local-data data/val_1k_local \
        --probe-tags data/val_1k_local/tags.json \
        --no-wandb 2>&1 | tee logs/m09c_sanity.log
    python -u src/m09c_surgery.py --POC \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/surgery_2stage_noDI.yaml \
        --subset data/sanity_100_dense.json --local-data data/val_1k_local \
        --factor-dir outputs/poc/m11_factor_datasets/ \
        --no-wandb 2>&1 | tee logs/m09c_dense100.log
    python -u src/m09c_surgery.py --FULL \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/surgery_3stage_DI.yaml \
        --subset data/ultra_hard_3066_train.json \
        --local-data data/ultra_hard_3066_local \
        --factor-dir outputs/full/m11_factor_datasets/ \
        --probe-subset data/ultra_hard_3066_val.json \
        --probe-local-data data/ultra_hard_3066_local \
        --probe-tags data/ultra_hard_3066_local/tags.json \
        --output-dir outputs/full/surgery_3stage_DI \
        --no-wandb 2>&1 | tee logs/m09c_full.log
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
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede any pyplot import chain (utils.plots)

import numpy as np
from tqdm import tqdm  # noqa: F401 — retained for parity; make_pbar is preferred

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
# iter11 live-debug: SIGUSR1/SIGUSR2 stack dump so stuck GPU runs can be
# inspected without CAP_SYS_PTRACE (py-spy / gdb / strace are blocked in
# the training container). See src/utils/live_debug.py for usage.
from utils.live_debug import install_debug_handlers
install_debug_handlers()

from utils.config import (
    check_gpu,
    add_subset_arg, add_local_data_arg, get_output_dir, get_module_output_dir, load_subset,  # noqa: F401
    get_pipeline_config, load_merged_config,
    add_model_config_arg, add_train_config_arg,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup  # noqa: F401 — wired via utils.training
from utils.plots import plot_training_curves
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.cache_policy import (
    add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir,
)

import torch

from utils.progress import make_pbar

# vjepa2 imports via shim (avoids src/ namespace collision)
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
    get_mask_generator, get_apply_masks,  # noqa: F401 — consumed via utils.training helpers
)

# Constants — paths come from CLI args only (CLAUDE.md no-default rule).
# --model-config and --train-config are both required=True via the helpers in utils.config.
CHECKPOINT_PREFIX = "m09c_ckpt"

# Shared training primitives — utils/training.py (Phase 1 of iter8 split).
from utils.training import (
    load_config,
    build_mask_generators,
    compute_jepa_loss, _train_step_grad_accum,  # noqa: F401 — _train_step_grad_accum kept for future grad-accum wiring
    update_teacher_ema,
    build_optimizer, build_scheduler, update_weight_decay,  # noqa: F401 — build_scheduler/update_weight_decay kept for future stage schedulers
    save_training_checkpoint, cleanup_old_checkpoints, cleanup_stage_checkpoints, load_training_checkpoint,  # noqa: F401 — cleanup_old_checkpoints/load_training_checkpoint kept for resume
    export_student_for_eval,
    set_trainable_prefix, enable_gradient_checkpointing,
    FactorSampler, build_factor_index, load_factor_clip, create_train_val_split,
    StreamingFactorDataset, build_streaming_indices, _streaming_worker_init,
    build_probe_clips, run_probe_eval, run_probe_val_loss, compute_trajectory_stats,
    render_training_plots,
)
from torch.utils.data import DataLoader


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

    # Mid-training probe (Prec@K/mAP@K/Cycle@K with BCa 95% CI at stage
    # boundaries — companion to the D.4 decision gate). Mode-gated: SANITY off
    # (N=20 too small for stable CI), POC/FULL on. CLI overrides win over yaml.
    # Fail-loud: ch11_surgery.yaml MUST have a `probe:` block with all keys;
    # missing/typo → KeyError at config merge (no .get default, per CLAUDE.md).
    probe_cfg = cfg["probe"]
    probe_enabled = probe_cfg["enabled"][mode_key]
    if getattr(args, "probe_subset", None):
        probe_cfg["subset"] = args.probe_subset
    if getattr(args, "probe_local_data", None):
        probe_cfg["local_data"] = args.probe_local_data
    if getattr(args, "probe_tags", None):
        probe_cfg["tags_path"] = args.probe_tags
    if getattr(args, "no_probe", False):
        probe_enabled = False
    cfg["probe"]["enabled"] = probe_enabled

    # Best-ckpt + kill-switch + plateau + BWT trigger + permanent-val flag:
    # mode-gated (off for SANITY, on for POC/FULL). Flatten per-mode dicts → scalars
    # so training loop reads booleans directly.
    cfg["probe"]["best_ckpt_enabled"] = probe_cfg["best_ckpt_enabled"][mode_key]
    cfg["probe"]["kill_switch_enabled"] = probe_cfg["kill_switch_enabled"][mode_key]
    cfg["probe"]["plateau_enabled"] = probe_cfg["plateau_enabled"][mode_key]
    cfg["probe"]["prec_plateau_enabled"] = probe_cfg["prec_plateau_enabled"][mode_key]
    cfg["probe"]["bwt_trigger_enabled"] = probe_cfg["bwt_trigger_enabled"][mode_key]
    cfg["probe"]["use_permanent_val"] = probe_cfg["use_permanent_val"][mode_key]

    # 90/10 train/val split (see data.train_val_split in ch11_surgery.yaml). Capped at
    # 1000 val clips to prevent FULL (115K × 10% = 11.5K) from over-sizing the val-set.
    # FULL mode intentionally has NO train_val_split key (probe.use_permanent_val.full=true
    # overrides → external val_subset is used). FAIL LOUD only if the override is also off.
    _tvs = cfg["data"]["train_val_split"]
    if mode_key in _tvs:
        cfg["data"]["train_val_split"] = _tvs[mode_key]
    elif cfg["probe"]["use_permanent_val"]:
        cfg["data"]["train_val_split"] = None        # dormant — external val_subset overrides
    else:
        raise KeyError(
            f"data.train_val_split.{mode_key} missing AND probe.use_permanent_val.{mode_key} "
            f"is false — must specify one (split-internal) or the other (external val_subset)."
        )

    # Streaming factor generation (eliminates m11 D_L/D_A .npy disk writes).
    # Flatten mode-gated enabled + num_workers into scalars. CLI override
    # (--factor-streaming / --no-factor-streaming) wins over yaml mode gate.
    # See iter/iter9/plan_code_dev.md for architecture + parity verification.
    fs_cfg = cfg["factor_streaming"]
    fs_enabled = fs_cfg[mode_key]
    fs_override = getattr(args, "factor_streaming_override", None)
    if fs_override is not None:
        fs_enabled = fs_override
    cfg["factor_streaming"]["enabled"] = fs_enabled
    cfg["factor_streaming"]["num_workers"] = fs_cfg["num_workers"][mode_key]

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
    # iter11 v3 (2026-04-26): cache-policy=2 nukes the WHOLE output_dir at startup
    # so load_checkpoint() finds nothing → fresh step-0 run.
    wipe_output_dir(output_dir, args.cache_policy, label=f"output_dir ({output_dir.name})")
    output_dir.mkdir(parents=True, exist_ok=True)
    student_path = output_dir / "student_encoder.pt"

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
        if m10_summary["quality_gate"] == "FAIL":
            print(f"FATAL: m10 quality gate FAILED (concept_recall={m10_summary['mean_concept_recall']:.2f})")
            print("  SAM 3.1 did not detect enough objects. Fix m10 before training.")
            sys.exit(1)
        print(f"  m10 quality: concept_recall={m10_summary['mean_concept_recall']:.2f} (PASS)")

    manifest = json.load(open(manifest_file))

    # Two held-out val strategies:
    # (A) Permanent val_1k (probe.use_permanent_val=True, default for FULL/10K+):
    #     val_keys come from probe.subset JSON (data/val_1k.json). Training uses
    #     ALL factor_manifest clips. Requires val_1k ∩ training_manifest = ∅
    #     (verified fail-loud below — m00c constructs disjoint subsets by seed).
    # (B) Internal 90/10 split (probe.use_permanent_val=False, used at POC 1K
    #     where training is val_1k itself): deterministic seeded shuffle via
    #     create_train_val_split(), val capped at 1000 clips.
    # Persisted val_split.json so Step D/E/F read the SAME held-out set.
    split_ratio = cfg["data"]["train_val_split"]
    use_permanent_val = cfg["probe"]["use_permanent_val"]
    all_keys = list(manifest.keys())
    if use_permanent_val:
        # Load permanent val-set keys from probe.subset (e.g. data/val_1k.json).
        permanent_val_path = cfg["probe"]["subset"]
        val_keys = json.load(open(permanent_val_path))["clip_keys"]
        train_keys = all_keys  # ALL manifest clips used for training
        # FAIL HARD on overlap — any val_1k clip in training manifest = test leakage.
        overlap = set(val_keys) & set(train_keys)
        if overlap:
            print(f"FATAL: permanent val_1k overlaps training manifest by "
                  f"{len(overlap)} clips. Training set must be disjoint from val_1k "
                  f"(m00c normally guarantees this via seed). First 5 overlaps: "
                  f"{list(overlap)[:5]}")
            sys.exit(1)
        split_source = f"PERMANENT {permanent_val_path}"
    else:
        train_keys, val_keys = create_train_val_split(
            all_keys, split_ratio, seed, max_val_clips=1000)
        split_source = f"INTERNAL seed={seed} ratio={split_ratio:.2f}"
    # max_val_clips only applies to internal_split (where the cap gates
    # create_train_val_split); under permanent val it's inapplicable — val comes
    # verbatim from probe.subset JSON (e.g., data/val_500.json).
    applied_max_val_clips = None if use_permanent_val else 1000
    train_manifest = {k: manifest[k] for k in train_keys}
    val_split_path = output_dir / "val_split.json"
    with open(val_split_path, "w") as f:
        json.dump({"n": len(val_keys), "seed": seed,
                   "source": str(args.subset) if args.subset else str(cfg["data"]["subset"]),
                   "split_strategy": "permanent" if use_permanent_val else "internal_split",
                   "split_ratio": split_ratio,
                   "max_val_clips": applied_max_val_clips,
                   "clip_keys": val_keys}, f, indent=2)
    print(f"  train/val split: {len(train_keys)} train / {len(val_keys)} val "
          f"({split_source}) → {val_split_path}")

    factor_index = build_factor_index(train_manifest,
                                       factor_dir / "D_L",
                                       factor_dir / "D_A",
                                       factor_dir / "D_I")

    # Streaming mode (iter9+): build (mp4_index, mask_index) from local TARs +
    # m10 masks instead of .npy factor paths. `factor_cfg_streaming` renames
    # yaml keys to the post-rename form that make_layout_only/make_agent_only
    # expect (matches m11:799-805). D_I keeps its legacy tube paths.
    streaming_enabled = cfg["factor_streaming"]["enabled"]
    mp4_index = mask_index = streaming_manifest = factor_cfg_streaming = None
    di_legacy_index = None
    if streaming_enabled:
        # get_module_output_dir already imported at module top (line 33).
        # A local re-import here would shadow the module-level name for the
        # ENTIRE function via Python scoping rules → UnboundLocalError at any
        # earlier call site (e.g., line 342). Removed 2026-04-20 (errors_N_fixes #68).
        m10_out = get_module_output_dir(
            "m10_sam_segment", args.subset,
            sanity=args.SANITY, poc=args.POC)
        masks_dir = Path(m10_out) / "masks"
        mp4_index, mask_index, streaming_manifest = build_streaming_indices(
            manifest_path=factor_dir / "factor_manifest.json",
            masks_dir=masks_dir,
            local_data=str(Path(cfg["data"]["local_data"])),
        )
        mp4_index = {k: v for k, v in mp4_index.items() if k in train_manifest}
        mask_index = {k: v for k, v in mask_index.items() if k in train_manifest}
        fcy = cfg["factor_datasets"]
        factor_cfg_streaming = {
            "layout_method": fcy["layout_patch_method"],
            "agent_method":  fcy["agent_patch_method"],
            "matte_factor":  fcy["soft_matte_factor"],
            "blur_sigma":    fcy["blur_sigma"],
            "feather_sigma": fcy["feather_sigma"],
        }
        di_legacy_index = {
            k: entry["D_I"]
            for k, entry in factor_index.items() if "D_I" in entry
        }

    # Surgery config
    surgery_cfg = cfg["surgery"]
    stages = surgery_cfg["stages"]

    # Fail-loud pre-flight: cross-check every stage's non-zero mode_mixture keys
    # against available factor data BEFORE the stage loop starts. Catches the iter10
    # v15c class of silent bug (yaml requests I=0.70 but n_interactions=0 in m10
    # segments.json → StreamingFactorDataset would silently renormalize to L/A-only).
    # CLAUDE.md §5 bans silent defaults; this surfaces the mismatch up-front rather
    # than waiting for stage-boundary dataset build (which would still crash under
    # the training.py fail-loud, but later in wall-time).
    _factor_map = {"L": "D_L", "A": "D_A", "I": "D_I"}
    if streaming_enabled:
        _avail = {
            "D_L": sum(1 for e in streaming_manifest.values() if e["has_D_L"]),
            "D_A": sum(1 for e in streaming_manifest.values() if e["has_D_A"]),
            "D_I": sum(1 for e in streaming_manifest.values() if e["has_D_I"]),
        }
    else:
        _avail = {
            "D_L": sum(1 for e in factor_index.values() if "D_L" in e),
            "D_A": sum(1 for e in factor_index.values() if "D_A" in e),
            "D_I": sum(1 for e in factor_index.values() if "D_I" in e),
        }
    for _si, _sc in enumerate(stages):
        for _mk, _w in _sc["mode_mixture"].items():
            if _w > 0 and _avail[_factor_map[_mk]] == 0:
                raise RuntimeError(
                    f"m09c pre-flight: stage {_si + 1}/{len(stages)} '{_sc['name']}' "
                    f"requests {_mk}={_w} in mode_mixture, but manifest has 0 clips "
                    f"with {_factor_map[_mk]} (per-factor counts={_avail}). This would "
                    f"silently renormalize away from the yaml mixture at stage build "
                    f"time. Fix upstream (e.g., m10 --interactions-only for D_I; "
                    f"m11 to populate D_A) OR set {_mk}=0.0 in yaml."
                )
    print(f"  [pre-flight] mode_mixture vs manifest availability OK: {_avail}")

    depth = cfg["model"]["depth"]
    embed_dim = cfg["model"]["embed_dim"]
    n_levels = cfg["model"]["n_output_distillation"]
    predict_all = cfg["model"]["predict_all"]
    crop_size = cfg["model"]["crop_size"]
    num_frames = cfg["data"]["num_frames"]
    batch_size = args.batch_size if args.batch_size else cfg["optimization"]["batch_size"]

    # merge_config_with_args (line 88) already flattened the per-mode dict → int. Don't re-subscript (#52).
    max_epochs = cfg["optimization"]["max_epochs"]
    # Step-count source-of-truth. Two data paths coexist post-iter9:
    #   - streaming (FULL research path): iterate full mp4_index (e.g., 9566 clips at 10K)
    #   - legacy (iter8 SANITY/POC bitwise-reproducibility path): FactorSampler over
    #     ALL materialized .npy in factor_index (~1000 at POC, ~20-100 at SANITY)
    # v9 10K collapsed 9566-clip training to 3 steps because this branch used
    # len(factor_index) under streaming — factor_index only holds the ~100 verify
    # clips materialized by `m11 --streaming`. errors_N_fixes #71.
    if streaming_enabled:
        total_clips = len(mp4_index)
    elif args.SANITY or args.POC:
        total_clips = len(factor_index)
    else:
        # FULL + streaming_enabled=False is a config error — fail loud, never silent
        # fall back to legacy factor_index (would collapse to ~3 steps at 10K).
        raise RuntimeError(
            f"FULL scale with streaming_enabled=False — check yaml factor_streaming.full "
            f"or --no-factor-streaming override. factor_index has {len(factor_index)} "
            f"materialized .npy (streaming short-circuits all non-verify clips), which "
            f"would collapse training to {len(factor_index) // batch_size} steps/epoch."
        )
    # Fail-loud: below this threshold training silently becomes a no-op. Caught v9's
    # total_clips=100 collapse before burning ~25 min of GPU on garbage weights.
    if total_clips < batch_size:
        raise RuntimeError(
            f"total_clips={total_clips} < batch_size={batch_size} → 0 steps/epoch → "
            f"training collapses to a no-op. streaming_enabled={streaming_enabled}, "
            f"len(factor_index)={len(factor_index)}, "
            f"len(mp4_index)={len(mp4_index) if mp4_index else 'None'}. "
            f"See errors_N_fixes.md #71."
        )
    steps_per_epoch = total_clips // batch_size
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

    # Mid-training probe runs on held-out val-split keys (from train/val split above).
    # Required yaml keys accessed directly — missing key = KeyError = fail loud
    # (CLAUDE.md "no .get default").
    #
    # Cadence (ch11_surgery.yaml probe.cadence):
    #   "every_n_steps"    → distributes `n_points` probes uniformly across total_steps.
    #   "saves_per_epoch"  → reuses checkpoint.saves_per_epoch (base_optimization.yaml)
    #                        to fire that many probes per training epoch.
    #   "stage_boundary"   → 3 probes only (legacy, low-resolution trajectory).
    probe_cfg = cfg["probe"]
    probe_clips = None
    probe_history = []          # list of {stage_idx, stage_name, global_step, prec@k, map@k, cycle@k, val_loss_*}
    probe_jsonl_path = output_dir / "probe_history.jsonl"
    probe_cadence = probe_cfg["cadence"]
    probe_every = None
    probe_compute_val_loss = probe_cfg["compute_val_loss"]
    if probe_cfg["enabled"]:
        # Required yaml keys. Direct access — missing key → KeyError with file+key name.
        subset_path = probe_cfg["subset"]
        local_data_path = probe_cfg["local_data"]
        tags_path = probe_cfg["tags_path"]
        if probe_cadence == "every_n_steps":
            n_points = probe_cfg["n_points"]
            probe_every = max(1, total_steps // n_points)
            print(f"\n{'='*60}\n[probe] N={len(val_keys)} held-out val clips, cadence=every_n_steps, "
                  f"n_points={n_points} (probe_every={probe_every}), val_loss={probe_compute_val_loss}\n{'='*60}")
        elif probe_cadence == "saves_per_epoch":
            saves_per_epoch = cfg["checkpoint"]["saves_per_epoch"]
            probe_every = max(1, steps_per_epoch // saves_per_epoch)
            n_probes_total = max_epochs * saves_per_epoch
            print(f"\n{'='*60}\n[probe] N={len(val_keys)} held-out val clips, cadence=saves_per_epoch, "
                  f"saves_per_epoch={saves_per_epoch} (probe_every={probe_every} steps, "
                  f"~{n_probes_total} total probes), val_loss={probe_compute_val_loss}\n{'='*60}")
        elif probe_cadence == "stage_boundary":
            print(f"\n{'='*60}\n[probe] N={len(val_keys)} held-out val clips, cadence=stage_boundary "
                  f"(3 probes only), val_loss={probe_compute_val_loss}\n{'='*60}")
        else:
            print(f"FATAL: probe.cadence='{probe_cadence}' — must be 'every_n_steps', 'saves_per_epoch', or 'stage_boundary'")
            sys.exit(1)
        # Use held-out val_keys from train/val split (in-memory override) instead of
        # reading probe.subset JSON — eliminates train/test overlap.
        probe_clips = build_probe_clips(
            probe_subset_path=subset_path,
            probe_local_data=local_data_path,
            probe_tags_path=tags_path,
            num_frames=num_frames, crop_size=crop_size,
            subset_keys_override=set(val_keys) if val_keys else None,
        )
        probe_jsonl_file = open(probe_jsonl_path, "w")
    else:
        print("[probe] disabled (SANITY mode or --no-probe) — skipping probe + val-loss eval")
        probe_jsonl_file = None

    # Best-ckpt tracker + catastrophic-forgetting kill-switch state.
    # Both driven by prec_at_k.mean (gate-aligned metric). best_state exports
    # student_best.pt on each new running-max; post-training it is promoted to
    # student_encoder.pt. kill_state accumulates strikes when current < max - threshold.
    best_state = {"prec_at_k": -1.0, "global_step": -1, "stage_name": "", "probe_record": None}
    kill_state = {"strikes": 0, "triggered": False, "reason": None}
    # Plateau + BWT trigger state (companion early-stop triggers).
    # v13 bug fix (#79, 2026-04-21): plateau buffers now track `last_stage_idx` and
    # reset when a new stage begins. Without this, a flat Stage-1 window killed
    # training at the FIRST Stage-2 probe (v13 logged prec_plateau kill after only
    # 1 S2 step). Reset semantics: buffer represents *within-current-stage* behaviour,
    # so Stage-1 plateau can't prematurely terminate Stage 2. BWT state is NOT reset
    # (it's cumulative-from-first-probe by design — that span-stages semantics is
    # intentional, not bugged).
    plateau_state = {"recent_val_losses": [], "patience_counter": 0, "last_stage_idx": -1}
    prec_plateau_state = {"recent_prec_at_k": [], "last_stage_idx": -1}
    bwt_state = {"first_prec_at_k": None, "patience_counter": 0}
    best_ckpt_enabled = probe_cfg["best_ckpt_enabled"]
    kill_switch_enabled = probe_cfg["kill_switch_enabled"]
    plateau_enabled = probe_cfg["plateau_enabled"]
    prec_plateau_enabled = probe_cfg["prec_plateau_enabled"]
    bwt_trigger_enabled = probe_cfg["bwt_trigger_enabled"]
    forgetting_threshold_pct = probe_cfg["forgetting_threshold_pct"]
    forgetting_patience = probe_cfg["forgetting_patience"]
    plateau_min_delta = probe_cfg["plateau_min_delta"]
    plateau_patience = probe_cfg["plateau_patience"]
    prec_plateau_min_delta = probe_cfg["prec_plateau_min_delta"]
    prec_plateau_patience = probe_cfg["prec_plateau_patience"]
    bwt_tolerance_pct = probe_cfg["bwt_tolerance_pct"]     # legacy, displayed on plot only
    bwt_ci_fraction = probe_cfg["bwt_ci_fraction"]         # Option C-adapted (#73)
    bwt_absolute_floor = probe_cfg["bwt_absolute_floor"]   # Option C-adapted (#73)
    bwt_patience = probe_cfg["bwt_patience"]
    best_ckpt_path = output_dir / "student_best.pt"

    def _render_live_plots(verbose: bool = False):
        """iter11: delegates to shared utils.training.render_training_plots.
        Local closure of probe config + state passed explicitly as kwargs."""
        render_training_plots(
            probe_history=probe_history,
            output_dir=output_dir,
            forgetting_threshold_pct=forgetting_threshold_pct,
            forgetting_patience=forgetting_patience,
            bwt_trigger_enabled=bwt_trigger_enabled,
            bwt_ci_fraction=bwt_ci_fraction,
            bwt_absolute_floor=bwt_absolute_floor,
            bwt_patience=bwt_patience,
            kill_state=kill_state,
            best_state=best_state,
            probe_compute_val_loss=probe_compute_val_loss,
            verbose=verbose,
            n_train_clips=total_clips,
            n_epochs=max_epochs,
            total_steps=total_steps,
            batch_size=batch_size,
            lr=cfg["optimization"]["lr"],
        )

    def _run_probe_at_step(stage_idx_, stage_name_, global_step_):
        """Run probe + optional val-loss, append to history, log + fsync.
        Also updates best-ckpt tracker + kill-switch state (both driven by
        prec_at_k.mean). Silent success on probe failure (try/except) so a
        bad probe doesn't kill training — kill-switch acts only on successful probes.

        SINGLE-WRITER POLICY (Fix1 #76): probe_history.jsonl is written ONLY by
        this function. Any external post-hoc backfill or analysis script that
        also appends to probe_history.jsonl will produce torn records (two
        dicts concatenated on one line without newline — observed v10, 2026-04-20).
        Post-hoc consumers MUST read `training_summary.json.probe_history` via
        `src/utils/probe_history.py:read_probe_history_robust()` instead —
        that field is written once, atomically, at end-of-training.
        """
        if probe_clips is None:
            return
        try:
            pr = run_probe_eval(student, probe_clips, cfg, device,
                                k=probe_cfg["k"], bootstrap_iter=probe_cfg["bootstrap_iter"])
            if probe_compute_val_loss:
                vl = run_probe_val_loss(student, teacher, predictor, probe_clips,
                                        mask_generators, cfg, device)
                pr["val_jepa_loss"] = vl["jepa_loss"]
                pr["val_masked_loss"] = vl["masked_loss"]
                pr["val_context_loss"] = vl["context_loss"]
            pr["stage_idx"] = stage_idx_
            pr["stage_name"] = stage_name_
            pr["global_step"] = global_step_
            # BWT = Prec@K[current] − Prec@K[first_probe]. Mirrors the bwt-kill
            # formula at line ~881. Persisted to jsonl + plotted in m09_forgetting
            # so users can see drift in real time (errors_N_fixes #73).
            first_prec = probe_history[0]["prec_at_k"]["mean"] if probe_history else pr["prec_at_k"]["mean"]
            pr["bwt"] = pr["prec_at_k"]["mean"] - first_prec
            probe_history.append(pr)
            probe_jsonl_file.write(json.dumps(pr) + "\n")
            probe_jsonl_file.flush()
            os.fsync(probe_jsonl_file.fileno())
            pk, mk, ck = pr["prec_at_k"], pr["map_at_k"], pr["cycle_at_k"]
            vl_msg = f" val_jepa={pr['val_jepa_loss']:.4f}" if probe_compute_val_loss else ""
            print(f"  [probe] step={global_step_} stage={stage_name_} N={pr['num_clips']} "
                  f"Prec@K={pk['mean']:.2f}±{pk['ci_half']:.2f} "
                  f"mAP@K={mk['mean']:.2f}±{mk['ci_half']:.2f} "
                  f"Cycle@K={ck['mean']:.2f}±{ck['ci_half']:.2f}{vl_msg}")
            log_metrics(wb_run, {
                f"probe/{stage_name_}/prec_at_k": pk["mean"],
                f"probe/{stage_name_}/prec_at_k_ci_half": pk["ci_half"],
                f"probe/{stage_name_}/map_at_k": mk["mean"],
                f"probe/{stage_name_}/cycle_at_k": ck["mean"],
                **({f"probe/{stage_name_}/val_jepa_loss": pr["val_jepa_loss"]}
                   if probe_compute_val_loss else {}),
            }, step=global_step_)

            # Best-ckpt tracker: export student_best.pt on each new running-max prec_at_k.
            current_prec = pk["mean"]
            if best_ckpt_enabled and current_prec > best_state["prec_at_k"]:
                best_state.update({
                    "prec_at_k": current_prec,
                    "global_step": global_step_,
                    "stage_name": stage_name_,
                    "probe_record": pr,
                })
                export_student_for_eval(student, best_ckpt_path, explora_enabled=False)
                print(f"  [best] new max Prec@K={current_prec:.2f} "
                      f"(step {global_step_}) → saved student_best.pt")

            # Early-stop #1: catastrophic-forgetting kill-switch — drop > forgetting_threshold_pct
            # from running max for forgetting_patience consecutive probes → abort training.
            if kill_switch_enabled and best_state["prec_at_k"] > 0:
                running_max = best_state["prec_at_k"]
                if current_prec < running_max - forgetting_threshold_pct:
                    kill_state["strikes"] += 1
                    print(f"  [forgetting-kill] strike {kill_state['strikes']}/{forgetting_patience}: "
                          f"Prec@K {current_prec:.2f} < max {running_max:.2f} - {forgetting_threshold_pct:.1f}")
                    if kill_state["strikes"] >= forgetting_patience:
                        kill_state["triggered"] = True
                        kill_state["reason"] = "catastrophic_forgetting"
                else:
                    kill_state["strikes"] = 0

            # #79 per-stage plateau semantics (2026-04-21, v13 bug fix):
            #   (1) buffers reset when entering a new stage → each window represents
            #       *within-current-stage* dynamics only, not cross-stage;
            #   (2) kill only triggers in the FINAL stage — intermediate stages are
            #       SUPPOSED to plateau (that's the whole point of having a next stage:
            #       different unfreeze depth + input distribution may unlock new signal).
            # Without (1)+(2), v13's flat Stage-1 trajectory killed at Stage-2 entry,
            # preventing D_A from training. BWT state is NOT reset (cumulative semantics
            # are intentional for backward-transfer).
            if plateau_state["last_stage_idx"] != stage_idx_:
                plateau_state["recent_val_losses"] = []
                prec_plateau_state["recent_prec_at_k"] = []
                plateau_state["last_stage_idx"] = stage_idx_
                prec_plateau_state["last_stage_idx"] = stage_idx_
            is_final_stage = (stage_idx_ == len(stages) - 1)

            # Early-stop #2: val-loss plateau detector — FINAL stage only.
            # Tracks last plateau_patience+1 val_jepa_loss values IN-STAGE; if max-min
            # over that window < plateau_min_delta AND we're in the last stage → halt.
            if plateau_enabled and probe_compute_val_loss and "val_jepa_loss" in pr:
                plateau_state["recent_val_losses"].append(pr["val_jepa_loss"])
                window = plateau_state["recent_val_losses"][-(plateau_patience + 1):]
                if is_final_stage and len(window) >= plateau_patience + 1:
                    spread = max(window) - min(window)
                    if spread < plateau_min_delta:
                        print(f"  [plateau-kill] val_jepa range over last "
                              f"{plateau_patience + 1} in-stage probes = {spread:.5f} < "
                              f"{plateau_min_delta} → plateau in final stage {stage_idx_}")
                        kill_state["triggered"] = True
                        kill_state["reason"] = "val_loss_plateau"

            # Early-stop #2b (H5 #76): Prec@K plateau on the gate metric itself.
            # val_jepa plateau can miss "representation decoupling" cases (v10: val_jepa
            # kept ↓ while Prec@K was flat from step 58 onward across 9 further probes).
            # Trigger halts when Prec@K spread over last prec_plateau_patience+1 probes
            # < prec_plateau_min_delta. min_delta sized << N=500 CI_half (2.35 pp) so
            # only true stagnation fires. Saves ~2 h compute per v10 analysis.
            if prec_plateau_enabled:
                prec_plateau_state["recent_prec_at_k"].append(current_prec)
                pwin = prec_plateau_state["recent_prec_at_k"][-(prec_plateau_patience + 1):]
                # #79: only fire in the FINAL stage (Stage 1's flat Prec@K is expected
                # under progressive-unfreeze recipe; next stage's D_A may unlock signal).
                if is_final_stage and len(pwin) >= prec_plateau_patience + 1:
                    pspread = max(pwin) - min(pwin)
                    if pspread < prec_plateau_min_delta:
                        print(f"  [prec-plateau-kill] Prec@K range over last "
                              f"{prec_plateau_patience + 1} in-stage probes = {pspread:.3f} pp "
                              f"< {prec_plateau_min_delta} pp → plateau in final stage {stage_idx_}")
                        kill_state["triggered"] = True
                        kill_state["reason"] = "prec_at_k_plateau"

            # Early-stop #3: cumulative negative-BWT trigger.
            # BWT_t = Prec@K[t] − Prec@K[first_probe]. If BWT < -bwt_tolerance_pct for
            # bwt_patience consecutive probes → cumulative soft forgetting, halt.
            # Complements kill-switch (which checks per-probe drops vs running-max);
            # this one catches slow drift that the 5pp threshold misses.
            if bwt_trigger_enabled:
                if bwt_state["first_prec_at_k"] is None:
                    bwt_state["first_prec_at_k"] = current_prec
                bwt_now = current_prec - bwt_state["first_prec_at_k"]
                # Option C-adapted (#73): compound noise-aware trigger. Old flat
                # -0.5pp threshold was never reachable at N=500 (CI ±2.4pp = noise
                # floor alone). New rule fires when BOTH:
                #   (1) BWT < -bwt_ci_fraction × current_ci_half  (noise-scaled)
                #   (2) BWT < -bwt_absolute_floor                 (absolute safety)
                # hold for bwt_patience consecutive probes.
                ci_half_now = pk["ci_half"]
                ci_threshold = -(bwt_ci_fraction * ci_half_now)
                abs_threshold = -bwt_absolute_floor
                fires_ci = bwt_now < ci_threshold
                fires_abs = bwt_now < abs_threshold
                if fires_ci and fires_abs:
                    bwt_state["patience_counter"] += 1
                    print(f"  [bwt-kill] strike {bwt_state['patience_counter']}/{bwt_patience}: "
                          f"BWT {bwt_now:+.3f} pp < ci_thr {ci_threshold:+.3f} "
                          f"(−{bwt_ci_fraction:.2f}×CI_half={ci_half_now:.2f}) "
                          f"AND abs_thr {abs_threshold:+.2f}")
                    if bwt_state["patience_counter"] >= bwt_patience:
                        kill_state["triggered"] = True
                        kill_state["reason"] = "negative_bwt"
                else:
                    bwt_state["patience_counter"] = 0

            # Live plots (trajectory + forgetting + val_loss) refreshed on every
            # probe so mid-run progress is visible via `ls outputs/poc/m09c_surgery/*.png`.
            # Silent (verbose=False); wrapped in try/except so plot render failure
            # never kills training. Skipped when probe_history has <2 entries.
            _render_live_plots(verbose=False)
            # Live training-loss curve: regenerate m09_train_loss.{png,pdf} from the
            # fsync'd CSV so the loss trajectory is visible mid-run (not only at end
            # of training). Matches m09a/m09b post-validation plot pattern. ~0.5 s
            # per probe; failure is loud-in-log (never crashes training, but also
            # never silent). errors_N_fixes #72.
            try:
                plot_training_curves(
                    [{"csv_path": str(csv_path), "label": "Surgery", "batch_size": batch_size}],
                    str(output_dir),
                    title_prefix=f"Surgery · {total_clips:,} clips × {max_epochs} ep × BS={batch_size} × LR={cfg['optimization']['lr']:.1e} ({total_steps:,} steps)\n",
                    x_axis_mode="steps")
            except Exception as _e:
                print(f"  [live-plot] WARN: train-curve render failed at step "
                      f"{global_step_}: {_e}")
        except Exception as e:
            print(f"  [probe] WARN: eval failed at step {global_step_} stage {stage_name_}: {e}")

    try:
        for stage_idx, stage_cfg in enumerate(stages):
            stage_name = stage_cfg["name"]
            n_trainable = int(depth * stage_cfg["unfreeze_below"])
            stage_pct = stage_cfg["max_epochs_pct"]
            stage_steps = max(int(total_steps * stage_pct), 1)
            # Warmup = warmup_pct * stage_steps (auto-scales with run length).
            # Replaces hardcoded warmup_steps=200 which broke POC where
            # stage_steps (~99) < 200 → LR never reached target. `max(1, ...)`
            # guards SANITY 1-step case where 20% floor-divides to 0.
            warmup_steps = max(1, int(stage_steps * surgery_cfg["warmup_pct"]))
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
                # Drop references so Python GC releases underlying objects; then
                # explicitly return CUDA cache blocks to the pool BEFORE
                # build_optimizer allocates the new stage's state.
                optimizer = scheduler = sampler = None
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

            # Factor source for this stage — streaming DataLoader (iter9+) or
            # legacy synchronous FactorSampler (iter8 1K POC). Exactly one is
            # active per stage; the other stays None.
            sampler = None
            stream_loader = None
            stream_iter = None
            if streaming_enabled:
                ds = StreamingFactorDataset(
                    mp4_index=mp4_index,
                    mask_index=mask_index,
                    factor_manifest=streaming_manifest,
                    factor_cfg=factor_cfg_streaming,
                    mode_mixture=mode_mixture,
                    num_frames=num_frames,
                    crop_size=crop_size,
                    di_legacy_index=di_legacy_index,
                    base_seed=seed + stage_idx,
                    steps_per_epoch=stage_steps * batch_size,
                    interaction_cfg=cfg["interaction_mining"],
                )
                fs_cfg = cfg["factor_streaming"]
                stream_loader = DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=fs_cfg["num_workers"],
                    prefetch_factor=fs_cfg["prefetch_factor"] if fs_cfg["num_workers"] > 0 else None,
                    persistent_workers=fs_cfg["persistent_workers"] if fs_cfg["num_workers"] > 0 else False,
                    pin_memory=fs_cfg["pin_memory"],
                    worker_init_fn=_streaming_worker_init if fs_cfg["num_workers"] > 0 else None,
                )
                stream_iter = iter(stream_loader)
            else:
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

            # Windowed rolling mean for tqdm postfix (matches m09a:652 / m09b:669 pattern).
            # 30-s cadence; resets each stage so mid-stage mean is meaningful (stages
            # have different unfreeze depth + mode mixture → mixing their losses would
            # blur diagnostic value at the Stage 1→2 transition).
            window_start = time.time()
            window_steps = 0
            running_loss = 0.0

            for local_step in range(stage_steps):
                # Build batch from factor clips — streaming DataLoader or legacy sampler.
                if stream_iter is not None:
                    batch = next(stream_iter)
                    batch_clips = batch["tensor"].to(device)              # (B, T, C, H, W)
                    batch_clips = batch_clips.permute(0, 2, 1, 3, 4)      # (B, C, T, H, W)
                else:
                    batch_tensors = []
                    for _ in range(batch_size):
                        _, _, clip_path = sampler.sample()
                        clip_tensor = load_factor_clip(clip_path, num_frames, crop_size)
                        batch_tensors.append(clip_tensor)
                    batch_clips = torch.stack(batch_tensors).to(device)
                    batch_clips = batch_clips.permute(0, 2, 1, 3, 4)      # (B, C, T, H, W)

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

                # tqdm postfix — windowed rolling mean (matches m09a:814-823 pattern).
                # At surgery's ~60 s/step wall, 30-s window usually fires every step →
                # effectively a per-step loss display. Shown fields: stage · rolling-mean
                # jepa loss · LR · grad norm · throughput (step/s ≈ 1/s-per-step).
                running_loss += jepa_val
                window_steps += 1
                now = time.time()
                window_elapsed = now - window_start
                throughput = window_steps / window_elapsed if window_elapsed > 0 else 0.0
                if window_elapsed >= 30:
                    pbar.set_postfix_str(
                        f"S{stage_idx + 1} loss={running_loss / window_steps:.4f} "
                        f"lr={lr_val:.2e} "
                        f"grad={gn_val:.2f} "
                        f"{throughput:.2f} step/s")
                    window_start = now
                    window_steps = 0
                    running_loss = 0.0

                global_step += 1
                pbar.update(1)

                if global_step % cfg["optimization"]["gc_interval"] == 0:
                    gc.collect()

                # Mid-stage probe: fire every `probe_every` steps under "every_n_steps"
                # cadence. Excludes local_step == stage_steps - 1 (last step of stage) —
                # that gets a forced probe below for the BWT anchor guarantee.
                is_last_step_of_stage = (local_step == stage_steps - 1)
                if (probe_clips is not None
                        and probe_every is not None
                        and not is_last_step_of_stage
                        and global_step % probe_every == 0):
                    _run_probe_at_step(stage_idx, stage_name, global_step)

                # Early-stop: abort if ANY of kill-switch / plateau / BWT triggered.
                # Break inner step loop AND flag outer stage loop via same kill_state.
                if kill_state["triggered"]:
                    reason = kill_state["reason"]
                    icon = {"catastrophic_forgetting": "⚠️", "val_loss_plateau": "🟰",
                            "prec_at_k_plateau": "📊", "negative_bwt": "📉"}[reason]
                    print(f"\n{icon}  EARLY-STOP [{reason}] — aborting training at step {global_step}")
                    print(f"     Best Prec@K={best_state['prec_at_k']:.2f} saved at "
                          f"step {best_state['global_step']} (stage {best_state['stage_name']})")
                    break

            pbar.close()
            if kill_state["triggered"]:
                # Stage ckpt = resume/rollback anchor → full=True so optimizer +
                # scheduler + scaler restore correctly. Bug fix 2026-04-27 — same
                # root cause as m09b lines 1026/1175 (silent re-warm on resume).
                save_training_checkpoint(output_dir / f"{CHECKPOINT_PREFIX}_stage{stage_idx}.pt",
                                         student, teacher, predictor, optimizer, scheduler,
                                         scaler, global_step, best_state["prec_at_k"], full=True)
                cleanup_stage_checkpoints(output_dir, CHECKPOINT_PREFIX, keep_n=1, cache_policy=args.cache_policy)
                _run_probe_at_step(stage_idx, stage_name, global_step)
                break
            save_training_checkpoint(output_dir / f"{CHECKPOINT_PREFIX}_stage{stage_idx}.pt",
                                     student, teacher, predictor, optimizer, scheduler,
                                     scaler, global_step, 0.0, full=True)
            # Per-stage rotation: keep only the newest stage checkpoint on disk. Without
            # this, the run accumulates 3 × ~15 GB = ~45 GB of redundant rollback points
            # (cause of the 2026-04-19 disk-full halt on 199 GB /workspace). `keep_n=1`
            # preserves one resume anchor for mid-stage crash recovery. Final cleanup
            # (keep_n=0) happens after export_student_for_eval below.
            cleanup_stage_checkpoints(output_dir, CHECKPOINT_PREFIX, keep_n=1, cache_policy=args.cache_policy)
            print(f"  Stage {stage_name} complete: {stage_steps} steps, loss={jepa_val:.4f}")

            # Forced stage-boundary probe (BWT anchor) — fires regardless of cadence
            # so the 3 anchors are always present in probe_history for BWT computation.
            _run_probe_at_step(stage_idx, stage_name, global_step)

    finally:
        csv_file.close()
        jsonl_file.close()
        if probe_jsonl_file is not None:
            probe_jsonl_file.close()
        gc.enable()

    # Fail-hard if zero successful training steps across all stages (#55).
    # Without this guard, the script would silently export the *initial* (frozen)
    # student weights and report "SURGERY COMPLETE" — passing CI but producing
    # garbage for downstream m05 eval. Per CLAUDE.md "Silent failures = garbage metrics".
    if global_step == 0:
        raise RuntimeError(
            f"SURGERY FAILED: 0 successful training steps across all {len(stages)} stages. "
            "Every step OOMed at minimum sub-batch=1 OR helper raised before any "
            "macro completed. The exported student would be identical to the input "
            "frozen V-JEPA weights — refusing to write a misleading checkpoint. "
            "See errors_N_fixes.md #55 for memory-budget mitigations."
        )

    # Best-ckpt promotion: if best_ckpt_enabled fired and student_best.pt exists,
    # promote it to student_encoder.pt (the downstream artifact). The "best" student
    # is the one that maxed Prec@K on the held-out val split during training —
    # eliminates the final-step-not-always-best problem. If no best was recorded
    # (probe disabled, or all probes failed), export current student weights.
    if best_ckpt_enabled and best_ckpt_path.exists():
        shutil.move(str(best_ckpt_path), str(student_path))
        print(f"  [best] Promoted student_best.pt (Prec@K={best_state['prec_at_k']:.2f} "
              f"at step {best_state['global_step']}, stage {best_state['stage_name']}) "
              f"→ student_encoder.pt")
    else:
        export_student_for_eval(student, student_path, explora_enabled=False)

    # Final checkpoint cleanup: `student_encoder.pt` is the only downstream artifact
    # (consumed by m05 surgical re-embed + m06 Prec@K). Stage rollback ckpts are
    # disposable once the run completes cleanly. Per CLAUDE.md "Clean all
    # intermediates after training." Saves ~15 GB per run at 2B model scale.
    cleanup_stage_checkpoints(output_dir, CHECKPOINT_PREFIX, keep_n=0, cache_policy=args.cache_policy)

    # Trajectory stats across stage boundaries. Single-probe-set regime so BWT
    # degenerates to net Prec@K improvement (R[-1]-R[0]). Non-zero max_drop
    # flags a stage transition that hurt Prec@K despite replay — paper's
    # "replay prevents forgetting" claim fails on this run if so.
    traj_stats = compute_trajectory_stats(probe_history) if probe_history else {}
    # compute_trajectory_stats returns either full stats dict (≥2 entries) or
    # a dict with None-valued fields (0-1 entries). Distinguish via trajectory length.
    if traj_stats and traj_stats["trajectory"]:
        print(f"\n{'='*60}\n[probe] Trajectory across {len(probe_history)} stages:")
        print(f"  Prec@K: {traj_stats['trajectory']}")
        print(f"  ΔPrec@K (BWT-proxy): {traj_stats['bwt_prec_at_k']:+.2f}")
        if not traj_stats["monotonic"]:
            print(f"  ⚠  max_drop = {traj_stats['max_drop_prec_at_k']:.2f} "
                  f"— some stage hurt Prec@K despite replay")
        else:
            print("  ✓  monotonic improvement across stages")
        print("=" * 60)
        log_metrics(wb_run, {
            "probe/trajectory/bwt_prec_at_k": traj_stats["bwt_prec_at_k"],
            "probe/trajectory/max_drop_prec_at_k": traj_stats["max_drop_prec_at_k"],
            "probe/trajectory/monotonic": int(traj_stats["monotonic"]),
        }, step=global_step)

        # Final pass at end-of-training — same helper used for live mid-run renders,
        # verbose=True so Saved: lines appear in the log for end-of-run confirmation.
        _render_live_plots(verbose=True)

    # Training summary
    summary = {
        "steps": global_step,
        "stages": [s["name"] for s in stages],
        "total_factor_clips": total_clips,    # matches the authoritative count used for step math (#71)
        "batch_size": batch_size,
        "final_loss": jepa_val,
        "probe_history": probe_history,
        "probe_trajectory_stats": traj_stats,
        "train_val_split": {
            "train": len(train_keys), "val": len(val_keys),
            "split_ratio": split_ratio, "seed": seed,
            "split_strategy": "permanent" if use_permanent_val else "internal_split",
            "max_val_clips": applied_max_val_clips,
            "val_split_path": str(val_split_path),
        },
        "best_ckpt": {
            "prec_at_k": best_state["prec_at_k"],
            "global_step": best_state["global_step"],
            "stage_name": best_state["stage_name"],
            "probe_record": best_state["probe_record"],
        } if best_ckpt_enabled else None,
        "early_stop": {
            "triggered": kill_state["triggered"],
            "reason": kill_state["reason"],
            "catastrophic_forgetting": {
                "enabled": kill_switch_enabled,
                "strikes_at_end": kill_state["strikes"],
                "threshold_pct": forgetting_threshold_pct,
                "patience": forgetting_patience,
            },
            "val_loss_plateau": {
                "enabled": plateau_enabled,
                "min_delta": plateau_min_delta,
                "patience": plateau_patience,
            },
            "prec_at_k_plateau": {                           # H5 #76 companion trigger
                "enabled": prec_plateau_enabled,
                "min_delta_pp": prec_plateau_min_delta,
                "patience": prec_plateau_patience,
                "recent_values": list(prec_plateau_state["recent_prec_at_k"]),
            },
            "negative_bwt": {
                "enabled": bwt_trigger_enabled,
                "strikes_at_end": bwt_state["patience_counter"],
                "first_prec_at_k": bwt_state["first_prec_at_k"],
                "ci_fraction": bwt_ci_fraction,              # #73 active threshold (noise-scaled)
                "absolute_floor_pp": bwt_absolute_floor,     # #73 active threshold (absolute)
                "tolerance_pct_legacy": bwt_tolerance_pct,   # #73 deprecated — present only for continuity
                "patience": bwt_patience,
            },
        },
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Training curves (reuse utils/plots.py). Pass x_axis_mode="steps" because
    # FactorSampler samples with REPLACEMENT from a small pool (e.g., 100 unique
    # clips on POC) — step × batch_size would over-report by the replacement
    # factor. "Optimizer Steps" is the only honest x-axis for surgery.
    # Match the authoritative total_clips used for step math (#71).
    n_unique_clips = total_clips
    plot_training_curves(
        [{"csv_path": str(csv_path), "label": "Surgery", "batch_size": batch_size}],
        str(output_dir),
        title_prefix=f"Surgery · {n_unique_clips:,} clips × {max_epochs} ep × BS={batch_size} × LR={cfg['optimization']['lr']:.1e} ({total_steps:,} steps)\n",
        x_axis_mode="steps")

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
    # Mid-training probe (Prec@K/mAP@K/Cycle@K with BCa 95% CI at stage boundaries).
    # Defaults resolved from configs/train/ch11_surgery.yaml probe: block; CLI wins.
    parser.add_argument("--probe-subset", type=str, default=None,
                        help="Probe clip subset JSON (default: cfg.probe.subset = val_1k)")
    parser.add_argument("--probe-local-data", type=str, default=None,
                        help="Probe WebDataset TAR dir (default: cfg.probe.local_data)")
    parser.add_argument("--probe-tags", type=str, default=None,
                        help="Probe tags.json path (default: cfg.probe.tags_path)")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip mid-training probe regardless of yaml setting")
    fs_group = parser.add_mutually_exclusive_group()
    fs_group.add_argument("--factor-streaming", dest="factor_streaming_override",
                          action="store_true", default=None,
                          help="Force streaming factor generation (overrides yaml mode gate)")
    fs_group.add_argument("--no-factor-streaming", dest="factor_streaming_override",
                          action="store_false",
                          help="Force legacy .npy factor path (overrides yaml mode gate)")
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

    if not args.factor_dir:
        print("FATAL: m09c_surgery requires --factor-dir (path to m11 factor datasets)")
        print("  Pipeline: m10 (Grounded-SAM) → m11 (factor datasets) → m09c (surgery)")
        sys.exit(1)

    ensure_local_data(args)

    # Load config: --model-config + --train-config (new) or --config (legacy).
    # Both required=True via add_*_config_arg, so args.train_config is guaranteed
    # unless --config (legacy) is used.
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_merged_config(args.model_config, args.train_config)
    cfg = merge_config_with_args(cfg, args)

    # Dispatch: surgery (only mode in this module)
    train_surgery(cfg, args)


if __name__ == "__main__":
    main()

    # Force exit: CUDA atexit cleanup can deadlock on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
