r"""ExPLoRA training — LoRA on blocks 2-47 + unfreeze blocks 0-1 + all LayerNorms, no drift control. GPU-only.

Pairs with m09a_pretrain.py (vanilla Ch10) and m09c_surgery.py (factor surgery). Shared primitives live in utils.training.

Recipe (arxiv 2406.10973, ICML 2025):
- Unfreeze first `unfreeze_blocks` transformer blocks (FULLY trainable).
- LoRA (rank 16) on remaining FROZEN blocks — attention qkv + proj linear layers.
- Unfreeze all LayerNorms across the whole ViT.
- Predictor fully trainable (separate from ExPLoRA — JEPA-specific head).
- Continue same SSL objective (JEPA loss) on new domain. No drift control (LoRA params auto-excluded from init_params).
- iter11 probe infra (port from m09c): mid-training Prec@K trajectory + best-ckpt-by-Prec@K + BWT / plateau / forgetting early-stop. See iter/iter11/plan_code_dev.md.

USAGE (all 4 args valid in every mode; defaults apply when omitted):
    # SANITY — 1-step smoke run, caps clips via data.sanity_train_clips (typically 20)
    python -u src/m09b_explora.py --SANITY \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/legacy2/explora.yaml \
        --subset data/sanity_100_dense.json \
        --local-data data/val_1k_local \
        --no-wandb 2>&1 | tee logs/m09b_sanity_v1.log

    # POC — short-training end-to-end (typically 5 ep × ~30 steps/ep = 155 steps @ 1K subset)
python -u src/m09b_explora.py --POC \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/legacy2/explora.yaml \
    --subset data/val_1k.json \
    --local-data data/val_1k_local \
    --val-subset data/sanity_100_dense.json \
    --val-local-data data/val_1k_local \
    --no-wandb 2>&1 | tee logs/m09b_poc_v2.log

    # FULL — target-scale 10K diagnostic baseline vs Surgery m09c.
    # Uses 3 MUTUALLY DISJOINT splits (verified zero-overlap on 2026-04-23):
    #   🏋️  TRAIN  data/subset_10k.json + data/subset_10k_local/   N=9,566 (after m10 filter)
    #   🧪  VAL    data/val_1k.json     + data/val_1k_local/       N=1,000 (probed 10×/epoch:
    #                                                              JEPA val-loss + Prec@K/mAP@K/
    #                                                              Cycle@K + early-stop triggers)
    #   🎯  EVAL   data/eval_10k.json   + data/eval_10k_local/     N=10,000 (POST-training only,
    #                                                              via scripts/legacy2/run_paired_eval_10k.sh
    #                                                              — NOT this script)
python -u src/m09b_explora.py --FULL \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/legacy2/explora.yaml \
    --subset data/subset_10k.json \
    --local-data data/subset_10k_local \
    --val-subset data/val_1k.json \
    --val-local-data data/val_1k_local \
    --no-wandb 2>&1 | tee logs/m09b_full_v1.log

    # POST-TRAINING EVAL (3-step pipeline — m09b is training-only; eval runs in orchestrator):
    # 1. Stage the trained ckpt into the paired_eval archive layout so run_paired_eval_10k.sh sees it:
    mkdir -p outputs_versioned/explora_m09c_surgery && cp outputs/full/m09b_explora/student_encoder.pt outputs_versioned/explora_m09c_surgery/
    # 2. Add 'explora' to VARIANTS list in scripts/legacy2/run_paired_eval_10k.sh:65 (same pattern as v10/v13/v14).
    # 3. Launch paired_eval — produces outputs_versioned/explora_eval10k/paired_bootstrap_results.json:
    ./scripts/legacy2/run_paired_eval_10k.sh 2>&1 | tee logs/paired_eval_explora.log

CLI DEFAULTS (when arg omitted):
    --model-config  → configs/model/vjepa2_1.yaml  (V-JEPA 2.1 ViT-G 2B)
    --train-config  → configs/legacy2/explora.yaml   (lora_rank=16, unfreeze_blocks=2)
    --subset        → stream full HF dataset (no client-side subset filter)
    --local-data    → stream from HF (slower than local TAR shards)
    --cache-policy  → auto-resolved from yaml[mode]: sanity=1(keep), poc/full=2(recompute)
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
from tqdm import tqdm

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
from utils.plots import plot_training_curves
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.cache_policy import (
    add_cache_policy_arg, resolve_cache_policy_interactive,
    guarded_delete, wipe_output_dir,
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
    # iter11 probe infra (ported from m09c_surgery) — mid-training Prec@K trajectory +
    # best-ckpt-by-Prec@K + BWT/plateau/forgetting early-stop. Technique-agnostic helpers;
    # ExPLoRA uses them to produce an apples-apples baseline vs Surgery's m09c outputs.
    build_probe_clips, run_probe_eval, run_probe_val_loss, compute_trajectory_stats,
    render_training_plots,
)

# Module-level constants — paths come from CLI args only (CLAUDE.md no-default rule)
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

    # iter11 probe-block mode-gated flatten — port from m09c pattern (src/m09c_surgery.py:94-129).
    # ExPLoRA is single-stage so every flag collapses from {sanity/poc/full} dict → scalar per
    # mode_key. Uses .get("probe", {}) for migration compat: pre-patch yaml without a probe
    # block → empty dict → probe.enabled=False downstream → silent no-op.
    probe_cfg = cfg.get("probe", {})  # noqa: B6-migration-compat
    for key in ("enabled", "best_ckpt_enabled", "kill_switch_enabled",
                "plateau_enabled", "prec_plateau_enabled",
                "bwt_trigger_enabled", "use_permanent_val"):
        if key in probe_cfg and isinstance(probe_cfg[key], dict):
            probe_cfg[key] = probe_cfg[key][mode_key]

    # iter11 auto disk-hygiene — resolve mode-gated cache_policy default from yaml
    # when user did NOT explicitly pass --cache-policy on CLI. POC/FULL need
    # cleanup_old_checkpoints to fire (keep_n=2) or ~25 × 15 GB periodic ckpts fill
    # the disk mid-training. SANITY keeps all ckpts (1-step run, trivial footprint).
    # Detection of "no CLI override" is done via sys.argv scan because argparse's
    # default collapses to "1" either way (can't distinguish default-vs-explicit).
    cp_yaml = cfg.get("optimization", {}).get("cache_policy")
    if isinstance(cp_yaml, dict):
        import sys as _sys
        _cli_explicit = any(a == "--cache-policy" or a.startswith("--cache-policy=")
                            for a in _sys.argv)
        if not _cli_explicit and mode_key in cp_yaml:
            args.cache_policy = str(cp_yaml[mode_key])
            print(f"  [cache-policy] resolved from yaml[{mode_key}] "
                  f"→ --cache-policy={args.cache_policy} "
                  f"(no CLI override; pass --cache-policy 1|2 to force)")
        # Flatten so downstream reads of cfg['optimization']['cache_policy'] get a scalar
        cfg["optimization"]["cache_policy"] = str(cp_yaml.get(mode_key, "1"))

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
    # iter11 bug-fix: PEFT's get_peft_model() calls _mark_only_adapters_as_trainable()
    # which freezes ALL non-adapter params, including the first-N blocks and LayerNorms
    # we'd pre-unfrozen. So the unfreeze MUST happen AFTER get_peft_model(). Before this
    # fix, m09b was effectively running standard LoRA (7.7M adapter-only, 0.4% trainable)
    # instead of the paper's ExPLoRA recipe (~84M = 2 blocks + norms + adapters, ~4.5%).
    explora_cfg = cfg["explora"]

    # 1. Freeze all student params (pre-LoRA)
    for param in student.parameters():
        param.requires_grad = False

    # 2. Inject LoRA on target attention layers (this also re-freezes everything)
    lora_config = LoraConfig(
        r=explora_cfg["lora_rank"],
        lora_alpha=explora_cfg["lora_alpha"],
        target_modules=explora_cfg["lora_target_modules"],
        lora_dropout=explora_cfg["lora_dropout"],
        bias="none",
    )
    student = get_peft_model(student, lora_config)

    # 3. Re-unfreeze first N blocks (ExPLoRA recipe: 1-2). PEFT wraps the base model
    # inside student.base_model.model; iterate there for the original block access.
    base_vit = student.base_model.model
    n_unfreeze = explora_cfg["unfreeze_blocks"]
    unfrozen_blocks = 0
    for i in range(n_unfreeze):
        for param in base_vit.blocks[i].parameters():
            param.requires_grad = True
        unfrozen_blocks += 1

    # 4. Re-unfreeze all norm layers (ExPLoRA requirement per arxiv 2406.10973 §3).
    unfrozen_norms = 0
    if explora_cfg["unfreeze_norm_layers"]:
        for name, param in student.named_parameters():
            if ("norm" in name.lower() or ".ln" in name.lower() or "layernorm" in name.lower()) \
               and not param.requires_grad:
                param.requires_grad = True
                unfrozen_norms += 1

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    # Detailed breakdown so regressions surface immediately:
    lora_params = sum(p.numel() for n, p in student.named_parameters()
                      if p.requires_grad and "lora_" in n)
    block_params = sum(p.numel() for i in range(n_unfreeze)
                       for p in base_vit.blocks[i].parameters() if p.requires_grad)
    norm_params = sum(p.numel() for n, p in student.named_parameters()
                      if p.requires_grad and ("norm" in n.lower() or "layernorm" in n.lower())
                      and "lora_" not in n)
    print(f"  ExPLoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
    print(f"    ├─ LoRA adapters: {lora_params:,} ({100*lora_params/total:.2f}%)")
    print(f"    ├─ First {unfrozen_blocks} blocks (full-unfrozen): {block_params:,} "
          f"({100*block_params/total:.2f}%)")
    print(f"    └─ LayerNorms (all blocks + final): {norm_params:,} tensors, "
          f"{unfrozen_norms} unfrozen")

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

    output_dir = Path(cfg["checkpoint"]["output_dir"])
    student_path = output_dir / "student_encoder.pt"
    # iter11 v3 (2026-04-26): cache-policy=2 nukes the WHOLE output_dir at startup
    # so load_checkpoint() finds nothing → fresh step-0 run. Old "recompute" path
    # only enforced keep_last_n eviction during training (which still resumed).
    wipe_output_dir(output_dir, args.cache_policy, label=f"output_dir ({output_dir.name})")
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

    # iter11: val_split.json (matches m09c:446-453 artifact set for downstream analysis)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "val_split.json", "w") as _vsf:
        json.dump({
            "n": len(val_key_set),
            "seed": None,
            "source": str(args.val_subset),
            "split_strategy": "permanent",
            "split_ratio": None,
            "max_val_clips": None,
            "clip_keys": sorted(val_key_set),
        }, _vsf, indent=2)

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
            # Fail-loud (CLAUDE.md §5): prior code used `.get("n") or .get("n_clips") or
            # .get("total_clips")` which silently returned None on any schema drift, then
            # `None - len(val_key_set)` → TypeError deep in training. Enumerate expected
            # keys explicitly; if none match, raise KeyError with the failing path.
            n_total = None
            for _k in ("n", "n_clips", "total_clips"):
                if _k in manifest:
                    n_total = manifest[_k]
                    break
            if n_total is None:
                raise KeyError(
                    f"{manifest_path}: expected one of ('n','n_clips','total_clips'), "
                    f"got keys={list(manifest.keys())}"
                )
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
    # iter11 v3 (2026-04-26): val + probe + ckpt all share the SAME cadence
    # (`checkpoint.saves_per_epoch`). Removed redundant `validation.evals_per_epoch`
    # — it was 2× saves_per_epoch and made probe fire 10x/epoch instead of 5x/epoch
    # (15 h wasted/run; m09c was already correct via probe.cadence=saves_per_epoch).
    val_interval = ckpt_interval

    print(f"Train clips: {n_train:,} | Val clips: {len(val_key_set):,}")
    print(f"Epochs: {max_epochs} | Steps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(f"Checkpoint every {ckpt_interval} steps ({saves_per_epoch}x/epoch, keep last {keep_last_n})")
    print(f"Validation+probe every {val_interval} steps ({saves_per_epoch}x/epoch, {len(val_key_set)} val clips)")

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

    # ── iter11 probe clips: separate from val_batches; deterministic no-aug preprocessing
    # (canonical 64-frame center crop via build_probe_clips). Used by run_probe_eval to
    # compute Prec@K/mAP@K/Cycle@K on a held-out set DIFFERENT from val_batches (which
    # compute JEPA val-loss). Loads once before training; probe runs at val_interval
    # cadence during training (see per-cadence block below). Only if probe.enabled.
    probe_clips = None
    probe_cfg_flat = cfg.get("probe", {})
    if probe_cfg_flat.get("enabled", False):
        probe_clips = build_probe_clips(
            probe_subset_path=probe_cfg_flat["subset"],
            probe_local_data=probe_cfg_flat["local_data"],
            probe_tags_path=probe_cfg_flat["tags_path"],
            num_frames=cfg["data"]["num_frames"],
            crop_size=cfg["data"]["crop_size"],
            subset_keys_override=None,
        )
        print(f"Probe clips pre-decoded: {len(probe_clips)}")

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
        # --val-subset is required=True (CLAUDE.md no-default rule), so val_key_set
        # being empty means the JSON file itself was empty — degenerate, FAIL LOUD.
        print(f"FATAL: --val-subset {args.val_subset} loaded but val_key_set is empty.")
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

    # ── iter11 probe infra state (port from m09c) ──────────────────────────────
    # best_prec_at_k: running max of probe.prec_at_k.mean, used to promote
    #   student_best_prec.pt on each new max (parallel to best_val_loss).
    # probe_history: per-probe records (step, prec_at_k, map_at_k, cycle_at_k,
    #   optionally val_jepa_loss), appended during training, used by
    #   compute_trajectory_stats post-training for BWT/max-drop/monotonic.
    # kill_state: latch for early-stop triggers (forgetting, val-plateau,
    #   prec-plateau, BWT compound). Mutated by _check_kill_triggers.
    # *_state dicts: running windows for each trigger's patience/delta logic.
    best_prec_at_k = float("-inf")
    best_state = {"global_step": -1, "prec_at_k": 0.0}  # for render_training_plots
    probe_history = []
    kill_state = {"triggered": False, "reason": None}
    forgetting_state = {"strikes": 0, "running_max": float("-inf")}
    plateau_state = {"recent_val_losses": []}
    prec_plateau_state = {"recent_prec_at_k": []}
    bwt_state = {"first_prec_at_k": None, "ci_strikes": 0}

    # probe_history.jsonl — crash-safe, independent from loss_log.jsonl so downstream
    # analysis (plots, best-ckpt selection verification) can read a clean series.
    probe_jsonl_path = output_dir / "probe_history.jsonl"
    probe_jsonl_file = open(probe_jsonl_path, "a") if probe_clips is not None else None

    def _log_probe(record: dict):
        """Append one probe record to probe_history.jsonl with fsync."""
        if probe_jsonl_file is None:
            return
        probe_jsonl_file.write(json.dumps(record) + "\n")
        probe_jsonl_file.flush()
        os.fsync(probe_jsonl_file.fileno())

    def _check_kill_triggers(probe_record, probe_cfg):
        """Evaluate 4 early-stop triggers against state dicts and mutate kill_state.

        Port of m09c_surgery._run_probe_at_step kill logic (src/m09c_surgery.py:984-1079),
        simplified for m09b's single-stage invariant: `is_final_stage=True` always, no
        per-stage state resets (only one continuous probe series exists).

        Mutates: forgetting_state, plateau_state, prec_plateau_state, bwt_state, kill_state.
        """
        current_prec = probe_record["prec_at_k"]["mean"]
        ci_half_now = probe_record["prec_at_k"]["ci_half"]

        # #1 catastrophic-forgetting — drop > threshold_pct from running max for `patience`
        if probe_cfg.get("kill_switch_enabled"):
            rm = forgetting_state["running_max"]
            if current_prec > rm:
                forgetting_state["running_max"] = current_prec
                forgetting_state["strikes"] = 0
            elif rm > 0 and current_prec < rm - probe_cfg["forgetting_threshold_pct"]:
                forgetting_state["strikes"] += 1
                print(f"  [forgetting-kill] strike {forgetting_state['strikes']}/"
                      f"{probe_cfg['forgetting_patience']}: Prec@K {current_prec:.2f} < "
                      f"max {rm:.2f} - {probe_cfg['forgetting_threshold_pct']:.1f}")
                if forgetting_state["strikes"] >= probe_cfg["forgetting_patience"]:
                    kill_state["triggered"] = True
                    kill_state["reason"] = "catastrophic_forgetting"
            else:
                forgetting_state["strikes"] = 0

        # #2 val-loss plateau (fires when max-min over window < min_delta)
        if (probe_cfg.get("plateau_enabled") and probe_cfg.get("compute_val_loss")
                and "val_jepa_loss" in probe_record):
            plateau_state["recent_val_losses"].append(probe_record["val_jepa_loss"])
            win = plateau_state["recent_val_losses"][-(probe_cfg["plateau_patience"] + 1):]
            if len(win) >= probe_cfg["plateau_patience"] + 1:
                spread = max(win) - min(win)
                if spread < probe_cfg["plateau_min_delta"]:
                    print(f"  [plateau-kill] val_jepa range over last "
                          f"{probe_cfg['plateau_patience'] + 1} probes = {spread:.5f} "
                          f"< {probe_cfg['plateau_min_delta']}")
                    kill_state["triggered"] = True
                    kill_state["reason"] = "val_loss_plateau"

        # #2b Prec@K plateau on the gate metric itself
        if probe_cfg.get("prec_plateau_enabled"):
            prec_plateau_state["recent_prec_at_k"].append(current_prec)
            pwin = prec_plateau_state["recent_prec_at_k"][-(probe_cfg["prec_plateau_patience"] + 1):]
            if len(pwin) >= probe_cfg["prec_plateau_patience"] + 1:
                pspread = max(pwin) - min(pwin)
                if pspread < probe_cfg["prec_plateau_min_delta"]:
                    print(f"  [prec-plateau-kill] Prec@K range over last "
                          f"{probe_cfg['prec_plateau_patience'] + 1} probes = {pspread:.3f} pp "
                          f"< {probe_cfg['prec_plateau_min_delta']} pp")
                    kill_state["triggered"] = True
                    kill_state["reason"] = "prec_at_k_plateau"

        # #3 negative-BWT compound trigger (noise-scaled + absolute floor)
        if probe_cfg.get("bwt_trigger_enabled"):
            if bwt_state["first_prec_at_k"] is None:
                bwt_state["first_prec_at_k"] = current_prec
            bwt_now = current_prec - bwt_state["first_prec_at_k"]
            ci_threshold = -(probe_cfg["bwt_ci_fraction"] * ci_half_now)
            abs_threshold = -probe_cfg["bwt_absolute_floor"]
            if bwt_now < ci_threshold and bwt_now < abs_threshold:
                bwt_state["ci_strikes"] += 1
                print(f"  [bwt-kill] strike {bwt_state['ci_strikes']}/"
                      f"{probe_cfg['bwt_patience']}: BWT {bwt_now:+.3f} pp < "
                      f"ci_thr {ci_threshold:+.3f} AND abs_thr {abs_threshold:+.2f}")
                if bwt_state["ci_strikes"] >= probe_cfg["bwt_patience"]:
                    kill_state["triggered"] = True
                    kill_state["reason"] = "negative_bwt"
            else:
                bwt_state["ci_strikes"] = 0

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
                    (jepa_val, masked_val, context_val, _unused,
                     _infonce_val, _tcc_val,
                     _uw_w_jepa, _uw_w_infonce, _uw_w_tcc) = _train_step_grad_accum(
                        student, teacher, predictor, batch_clips,
                        all_masks_enc, all_masks_pred,
                        cfg, dtype, mp_cfg, scaler, train_sizer, loss_exp,
                        init_params=None, drift_cfg=None,
                        loss_cfg=cfg["optimization"]["loss"], uw=None)
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
                # CKPT_PATH (= _latest.pt) is the resume anchor — must be full=True
                # so load_training_checkpoint can restore optimizer + scheduler +
                # scaler state. Bug fix 2026-04-27 after v10 resume re-fired the LR
                # warmup (lr=5.85e-7 at step 286) because v9's _latest.pt was
                # full=False → no scheduler state → fresh cosine from step 0.
                save_training_checkpoint(
                    ckpt_path, student, teacher, predictor, optimizer, scheduler,
                    scaler, step + 1, best_val_loss, full=True)
                cleanup_old_checkpoints(output_dir, prefix=CHECKPOINT_PREFIX,
                                        keep_n=keep_last_n,
                                        cache_policy=args.cache_policy)

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
                    plot_training_curves(
                        runs=[{"csv_path": str(csv_path),
                               "label": "V-JEPA 2.1 ExPLoRA",
                               "color": "green",
                               "batch_size": batch_size}],
                        output_dir=str(output_dir),
                        file_prefix="m09b",
                        title_prefix=f"ExPLoRA · {n_train:,} clips × {max_epochs} ep × BS={batch_size} × LR={cfg['optimization']['lr']:.1e} ({total_steps:,} steps)\n",
                    )
                except (OSError, IOError, ValueError, RuntimeError) as _plot_e:
                    # Plot failure must never stop training, but swallow is also banned
                    # per CLAUDE.md §5 → narrow to known non-fatal errors (disk-write
                    # OSError, broken matplotlib state ValueError/RuntimeError, IO
                    # hiccups) AND log so operator sees it. Unexpected exception types
                    # (e.g., KeyboardInterrupt, MemoryError, BaseException) still propagate.
                    print(f"  [plot] WARN: non-fatal plotting error ({type(_plot_e).__name__}): {_plot_e}")

                # iter11 v3 (2026-04-26): val_loss tracker — print only, no ckpt save.
                # Previously wrote `m09b_ckpt_best.pt` but it was never consumed
                # downstream (export uses student_best.pt = best-Prec@K, not best-val-loss
                # — see `iter/utils/experiment_log.md` v10 entry for the val_jepa↔Prec@K
                # decoupling evidence). Dropped to align filenames with m09c.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  New best val loss: {best_val_loss:.4f}")

            # ── iter11 Probe: mid-training Prec@K trajectory (parallel to val-loss above) ──
            # Cadence mirrors val_interval. Fires only if probe.enabled (post-flatten).
            # Augments (does NOT replace) the JEPA-val-loss path — both series go into
            # training_summary.json and feed the best-ckpt comparison at end-of-training.
            if (step + 1) % val_interval == 0 and probe_clips is not None:
                eval_result = run_probe_eval(
                    student, probe_clips, cfg, device,
                    k=probe_cfg_flat["k"],
                    bootstrap_iter=probe_cfg_flat["bootstrap_iter"])
                # iter11 v3 (2026-04-26): mirror m09c's pattern — start from eval_result
                # (which carries num_clips) so render_training_plots can title the panels
                # with N=. Filter-copy was dropping num_clips → silent KeyError suppressed
                # probe_trajectory.png + m09_forgetting.png generation.
                probe_record = dict(eval_result)
                probe_record.update({
                    "step":        step + 1,
                    "global_step": step + 1,
                    "stage_idx":   0,                  # single-stage invariant
                    "stage_name":  "stage1_explora",
                })
                if probe_cfg_flat.get("compute_val_loss"):
                    val_result = run_probe_val_loss(
                        student, teacher, predictor, probe_clips,
                        mask_generators, cfg, device)
                    probe_record.update({
                        "val_jepa_loss": val_result["jepa_loss"],
                        "masked_loss":   val_result["masked_loss"],
                        "context_loss":  val_result["context_loss"],
                    })
                # bwt = Prec@K[current] − Prec@K[first_probe] (matches m09c:783)
                first_prec = (probe_history[0]["prec_at_k"]["mean"]
                              if probe_history else probe_record["prec_at_k"]["mean"])
                probe_record["bwt"] = probe_record["prec_at_k"]["mean"] - first_prec
                probe_history.append(probe_record)
                _log_probe(probe_record)

                pk = eval_result["prec_at_k"]
                mk = eval_result["map_at_k"]
                ck = eval_result["cycle_at_k"]
                vl_msg = f" val_jepa={probe_record['val_jepa_loss']:.4f}" \
                    if "val_jepa_loss" in probe_record else ""
                print(f"  [probe step {step + 1}] Prec@K={pk['mean']:.2f}±{pk['ci_half']:.2f} "
                      f"mAP@K={mk['mean']:.2f}±{mk['ci_half']:.2f} "
                      f"Cycle@K={ck['mean']:.2f}±{ck['ci_half']:.2f}{vl_msg}")
                log_metrics(wb_run, {
                    "probe/prec_at_k": pk["mean"],
                    "probe/prec_at_k_ci_half": pk["ci_half"],
                    "probe/map_at_k": mk["mean"],
                    "probe/cycle_at_k": ck["mean"],
                    **({"probe/val_jepa_loss": probe_record["val_jepa_loss"]}
                       if "val_jepa_loss" in probe_record else {}),
                }, step=step)

                # 🏆 Best-ckpt by Prec@K (parallel to best_val_loss above) — iter11
                # iter11 v3 (2026-04-26): rename m09b_ckpt_best_prec.pt → student_best.pt
                # for filename parity with m09c. Same artifact, same role (running max
                # Prec@K ckpt, promoted to student_encoder.pt at end-of-training).
                if probe_cfg_flat.get("best_ckpt_enabled") and pk["mean"] > best_prec_at_k:
                    best_prec_at_k = pk["mean"]
                    best_state["global_step"] = step + 1
                    best_state["prec_at_k"] = best_prec_at_k
                    save_training_checkpoint(
                        output_dir / "student_best.pt",
                        student, teacher, predictor, optimizer, scheduler,
                        scaler, step + 1, best_prec_at_k, full=False)
                    print(f"  🏆 New best Prec@K: {best_prec_at_k:.2f} (step {step + 1}) → saved student_best.pt")

                # Live plots (trajectory + forgetting + val_loss) — same cadence as m09c.
                # Shared util; silent on <2 probes. Wrapped in try/except in the util itself.
                render_training_plots(
                    probe_history=probe_history,
                    output_dir=output_dir,
                    forgetting_threshold_pct=probe_cfg_flat["forgetting_threshold_pct"],
                    forgetting_patience=probe_cfg_flat["forgetting_patience"],
                    bwt_trigger_enabled=probe_cfg_flat.get("bwt_trigger_enabled", False),
                    bwt_ci_fraction=probe_cfg_flat.get("bwt_ci_fraction", 0.5),
                    bwt_absolute_floor=probe_cfg_flat.get("bwt_absolute_floor", 0.3),
                    bwt_patience=probe_cfg_flat.get("bwt_patience", 3),
                    kill_state=kill_state,
                    best_state=best_state,
                    probe_compute_val_loss=probe_cfg_flat.get("compute_val_loss", True),
                    verbose=False,
                    n_train_clips=n_train,
                    n_epochs=max_epochs,
                    total_steps=total_steps,
                    batch_size=batch_size,
                    lr=cfg["optimization"]["lr"],
                    file_prefix="m09b",
                )

                # 🛑 Early-stop triggers — mutates kill_state on fire
                _check_kill_triggers(probe_record, probe_cfg_flat)
                if kill_state["triggered"]:
                    print(f"[early-stop] {kill_state['reason']} → halting training "
                          f"at step {step + 1}")
                    # Save a final ckpt before breaking — full=True so post-training
                    # export AND any future resume-from-early-stop both work.
                    save_training_checkpoint(
                        ckpt_path, student, teacher, predictor,
                        optimizer, scheduler, scaler, step + 1, best_val_loss, full=True)
                    break   # exit inner for-loop; post-loop code handles export

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
        # SANITY: student-only (matches periodic-save exception above — no need for a
        # full resume ckpt on 1-step smoke runs; avoids torch.save disk-full regression).
        save_training_checkpoint(
            ckpt_path, student, teacher, predictor, optimizer, scheduler,
            scaler, step + 1, best_val_loss, full=not args.SANITY)
        print(f"  Checkpoint saved at step {step + 1}/{total_steps}. Resume with same command.")
        sys.exit(0)  # user-initiated, not an error
    finally:
        pbar.close()
        csv_file.close()
        jsonl_file.close()
        stop_event.set()
        gc.enable()

    # Cooldown phase: switch to 64f, linear LR decay (V-JEPA 2.1 recipe)
    # Fail-loud (CLAUDE.md §5): yaml MUST declare the `cooldown` section with an explicit
    # `enabled: true|false`. Prior silent `.get("cooldown", {}).get("enabled")` path
    # would quietly skip cooldown on schema drift → wrong LR schedule → contaminated Δ.
    cooldown_cfg = cfg["cooldown"]
    if cooldown_cfg["enabled"] and not args.SANITY:
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

    # ── iter11: Promote best-Prec@K ckpt over best-val-loss ckpt for export ────
    # v10 evidence (experiment_log.md:L93-95) shows val_jepa DECOUPLES from Prec@K
    # (val kept ↓ while Prec@K peaked mid-training, then ↓). Exporting the best-val-loss
    # ckpt loses the Prec@K-peak state. If probe.best_ckpt_enabled, reload the saved
    # best-Prec@K weights into `student` before export so student_encoder.pt carries
    # the best-Prec@K state. Falls back to current (end-of-training) student otherwise.
    best_prec_ckpt = output_dir / "student_best.pt"
    if probe_cfg_flat.get("best_ckpt_enabled") and best_prec_ckpt.exists():
        load_training_checkpoint(best_prec_ckpt, student, teacher, predictor,
                                 optimizer, scheduler, scaler)
        print(f"  Promoted {best_prec_ckpt.name} (Prec@K={best_prec_at_k:.2f}) → export")

    # Export student encoder (the only deliverable — only reached if training completed).
    # explora_enabled=True triggers peft's merge_and_unload() — converts the LoRA-wrapped
    # student back to a plain ViT that m05 can load.
    export_student_for_eval(student, student_path, explora_enabled=True)

    # iter11 META-fix: gate post-training checkpoint cleanup through --cache-policy.
    for ckpt_file in output_dir.glob(f"{CHECKPOINT_PREFIX}_*.pt"):
        guarded_delete(ckpt_file, args.cache_policy,
                       label=f"m09b checkpoint {ckpt_file.name}")

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

    # ── iter11 probe-trajectory stats (port from m09c) ─────────────────────────
    # compute_trajectory_stats produces BWT (Prec@K[last] - Prec@K[first]), max-drop
    # from running peak, and monotonic flag for post-run analysis. Only populated if
    # probe was enabled AND collected ≥1 record.
    if probe_history:
        traj = compute_trajectory_stats(probe_history)
        summary["probe_trajectory_stats"] = {
            "bwt_prec_at_k":      traj.get("bwt_prec_at_k"),
            "max_drop_prec_at_k": traj.get("max_drop_prec_at_k"),
            "monotonic":          traj.get("monotonic"),
        }
        summary["n_probes"] = len(probe_history)
        # Final end-of-training render (verbose=True prints "Saved: ..." to log).
        render_training_plots(
            probe_history=probe_history,
            output_dir=output_dir,
            forgetting_threshold_pct=probe_cfg_flat["forgetting_threshold_pct"],
            forgetting_patience=probe_cfg_flat["forgetting_patience"],
            bwt_trigger_enabled=probe_cfg_flat.get("bwt_trigger_enabled", False),
            bwt_ci_fraction=probe_cfg_flat.get("bwt_ci_fraction", 0.5),
            bwt_absolute_floor=probe_cfg_flat.get("bwt_absolute_floor", 0.3),
            bwt_patience=probe_cfg_flat.get("bwt_patience", 3),
            kill_state=kill_state,
            best_state=best_state,
            probe_compute_val_loss=probe_cfg_flat.get("compute_val_loss", True),
            verbose=True,
            n_train_clips=n_train,
            n_epochs=max_epochs,
            total_steps=total_steps,
            batch_size=batch_size,
            lr=cfg["optimization"]["lr"],
            file_prefix="m09b",
        )
        summary["best_prec_at_k"] = (best_prec_at_k
                                     if best_prec_at_k > float("-inf") else None)
        summary["best_ckpt"] = {
            "source": ("student_best.pt" if probe_cfg_flat.get("best_ckpt_enabled")
                       and best_prec_ckpt.exists()
                       else "student_encoder.pt"),
            "criterion": ("prec_at_k" if probe_cfg_flat.get("best_ckpt_enabled")
                          and best_prec_ckpt.exists() else "end_of_training"),
        }
        if kill_state["triggered"]:
            summary["early_stop"] = {
                "triggered": True,
                "reason":    kill_state["reason"],
                "step":      step + 1,
            }

    if probe_jsonl_file is not None:
        probe_jsonl_file.close()

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
    parser.add_argument("--val-subset", required=True,
                        help="Path to val subset JSON (e.g., data/val_1k.json). "
                             "These clips are excluded from training and used for periodic val loss.")
    parser.add_argument("--val-local-data", required=True,
                        help="Local WebDataset dir for val clips (e.g., data/val_1k_local).")
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
    # Both required=True via add_*_config_arg, so args.train_config is guaranteed
    # unless --config (legacy) is used.
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_merged_config(args.model_config, args.train_config)
    cfg = merge_config_with_args(cfg, args)

    train(cfg, args)


if __name__ == "__main__":
    main()

    # Force exit: CUDA atexit cleanup can deadlock on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
