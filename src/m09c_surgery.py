"""Ch11 Factor Surgery — 3-stage progressive unfreezing with D_L/D_A/D_I factor datasets. GPU-only.
Gold standard #1 (training loop primitives): https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py
Gold standard #2 (mask-conditioned video SSL paradigm): https://github.com/MCG-NJU/MGMAE
Gold standard #3 (foundational video masked SSL): https://github.com/MCG-NJU/VideoMAE
Claude Code: re-WebSearch all 3 URLs on every read of this file (each verified live 2026-05-09).

Split from m09_pretrain.py on 2026-04-15 (#49). Pairs with m09a_pretrain.py (vanilla Ch10)
and m09b_explora.py (LoRA variant). Shared primitives live in utils.training.

Pipeline: m10 (Grounded-SAM) → m11 (factor datasets) → m09c (surgery training).
The paper novelty — factor-disentangled surgery on a frozen V-JEPA 2.1 encoder.

USAGE (every path arg required — CLAUDE.md no-default rule; --probe-* fall back to yaml):
    python -u src/m09c_surgery.py --SANITY \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/legacy2/surgery_2stage_noDI.yaml \
        --subset data/sanity_100_dense.json --local-data data/val_1k_local \
        --factor-dir outputs/sanity/m11_factor_datasets/ \
        --probe-subset data/val_1k.json --probe-local-data data/val_1k_local \
        --probe-tags data/val_1k_local/tags.json \
        --no-wandb 2>&1 | tee logs/m09c_sanity.log
    python -u src/m09c_surgery.py --POC \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/legacy2/surgery_2stage_noDI.yaml \
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
    check_gpu, get_output_dir, get_module_output_dir, load_subset,  # noqa: F401
    get_pipeline_config, load_merged_config,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup  # noqa: F401 — wired via utils.training
from utils.plots import (plot_training_curves, plot_combined_losses,
                          plot_probe_trajectory_trio,
                          plot_val_loss_with_kill_switch_overlay,
                          compute_block_drift)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.cache_policy import (
    resolve_cache_policy_interactive, wipe_output_dir,
)
# iter13 v13 R1+R2+R5 (2026-05-07): shared CLI + config-merge + probe-pipeline
# helpers — replaces ~80 LoC of inline boilerplate that was duplicated with
# m09a_pretrain.py. Technique-specific args (--factor-dir, --factor-streaming)
# are still added INLINE in main() after add_m09_common_args(parser) fires.
from utils.m09_common import (
    add_m09_common_args, merge_m09_common_config, setup_probe_pipeline,
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
    compute_jepa_loss, _train_step_grad_accum, compute_total_loss, UncertaintyWeights,  # noqa: F401 — _train_step_grad_accum kept for future grad-accum wiring
    update_teacher_ema,
    build_optimizer, build_scheduler, update_weight_decay,  # noqa: F401 — build_scheduler/update_weight_decay kept for future stage schedulers
    save_training_checkpoint, cleanup_old_checkpoints, cleanup_stage_checkpoints, load_training_checkpoint,  # noqa: F401 — cleanup_old_checkpoints/load_training_checkpoint kept for resume
    export_student_for_eval,
    set_trainable_prefix, enable_gradient_checkpointing,
    FactorSampler, build_factor_index, load_factor_clip,
    StreamingFactorDataset, build_streaming_indices, _streaming_worker_init,
    run_probe_val_loss, compute_trajectory_stats,
    run_trio_at_val, track_block_drift_at_val,
    apply_val_cycle_triggers, finalize_training,
)
from utils.multi_task_loss import (
    build_multi_task_head_from_cfg,
    attach_head_to_optimizer, run_multi_task_step,
)
# iter13 v12+ (Phase 4, 2026-05-06): motion_aux loss — joint K-class CE + 13-D MSE
# on m04d's RAFT optical-flow targets. Surgery rebuilds the optimizer per stage
# so attach_motion_aux_to_optimizer must fire inside each stage's build_optimizer
# block (mirrors attach_head_to_optimizer above). See plan_v12_motion_aux.md.
from utils.motion_aux_loss import (
    build_motion_aux_head_from_cfg,
    attach_motion_aux_to_optimizer, run_motion_aux_step,
)
from utils.probe_labels import ensure_probe_labels_for_mode
from torch.utils.data import DataLoader


# ═════════════════════════════════════════════════════════════════════════
# CONFIG (merge_config_with_args stays here — argparse-coupled dispatch)
# ═════════════════════════════════════════════════════════════════════════

def merge_config_with_args(cfg: dict, args) -> dict:
    if args.SANITY:
        mode_key = "sanity"
    elif args.POC:
        mode_key = "poc"
    else:
        mode_key = "full"
    # iter13 v13 R2 (2026-05-07): shared CLI overrides + per-mode flatten + multi_task
    # / motion_aux delegation now lives in utils.m09_common. Replaces ~80 LoC of
    # boilerplate that mirrored m09a's body 1:1 (D3-D8 + D4-FIX + D7-FIX + D8-FIX
    # all consolidated into the helper). Technique-specific blocks stay below.
    merge_m09_common_config(cfg, args, mode_key)

    # iter14 (2026-05-08): pass --init-from-ckpt through to cfg so build_model can
    # dispatch on it. None when not specified (build_model falls back to Meta's
    # frozen V-JEPA URL = legacy iter13 behavior).
    cfg["init_from_ckpt"] = args.init_from_ckpt

    # iter14 recipe-v2 (2026-05-09): CLI --teacher-mode overrides yaml surgery.teacher_mode.
    # POC sweep cell selector: {EMA, FROZEN} × {LP-FT off, on}. None → keep yaml default.
    if getattr(args, "teacher_mode", None) is not None:
        cfg["surgery"]["teacher_mode"] = args.teacher_mode
    # iter14 recipe-v2 (2026-05-09): CLI --lp-ft-stage0 overrides yaml surgery.lp_ft_stage0.enabled.
    if getattr(args, "lp_ft_stage0", None) is not None:
        cfg["surgery"]["lp_ft_stage0"]["enabled"] = (args.lp_ft_stage0 == "on")

    # iter14 recipe-v3 (2026-05-09) — five additional ablation switches for the
    # drop-one POC sweep (T7). Each respects None = "keep yaml value", so legacy
    # callers without these flags get the yaml-level recipe-v3 defaults.

    # #3 surgical subset: legacy = 12/24/24 blocks (unfreeze_below 0.25/0.50/0.50);
    # recipe_v3 = 4/8/8 (yaml-level default after T2). When --subset-mode=legacy,
    # rewrite the per-stage unfreeze_below back to the iter10 v15c values.
    if getattr(args, "subset_mode", None) == "legacy":
        legacy_unfreeze = {
            "stage1_layout": 0.25,
            "stage2_agent": 0.50,
            "stage3_interaction": 0.50,
        }
        for s in cfg["surgery"]["stages"]:
            if s["name"] in legacy_unfreeze:
                s["unfreeze_below"] = legacy_unfreeze[s["name"]]
        print("  [recipe-v3 override] subset_mode=legacy → unfreeze_below restored "
              "to 12/24/24 blocks (was 4/8/8 per yaml).")

    # A4 single warmup: yaml default is "per_stage". --warmup-mode={per_stage,single}.
    if getattr(args, "warmup_mode", None) is not None:
        cfg["surgery"]["warmup_mode"] = args.warmup_mode

    # A2 saliency-weighted JEPA loss: yaml default false. --saliency={off,on}.
    if getattr(args, "saliency", None) is not None:
        cfg["optimization"]["loss"]["saliency_weighting"] = (args.saliency == "on")

    # #4 SPD optimizer: yaml default disabled. --spd={off,on}.
    if getattr(args, "spd", None) is not None:
        cfg["optimization"]["spd"]["enabled"] = (args.spd == "on")

    # #5 CLEAR raw replay: yaml default 0.0. --replay={off,on} where on=0.5.
    if getattr(args, "replay", None) is not None:
        cfg["replay"]["raw_pretrain_pct"] = 0.5 if args.replay == "on" else 0.0

    # iter14 (2026-05-08): retired the legacy `data.train_val_split` + per-mode
    # `probe.use_permanent_val` resolution. m09c now mirrors m09a's gold-standard
    # pattern (m09a:466-479): single yaml/CLI-driven external val pool via
    # cfg["probe"]["subset"], train pool subtracts val to prevent leakage. No
    # internal 90:10 fallback. See val pool block at line ~530+ for the pattern.

    # m09c-specific: streaming factor generation (eliminates m11 D_L/D_A .npy disk
    # writes). Flatten mode-gated enabled + num_workers into scalars. CLI override
    # (--factor-streaming / --no-factor-streaming) wins over yaml mode gate.
    # See iter/iter9/plan_code_dev.md for architecture + parity verification.
    fs_cfg = cfg["factor_streaming"]
    fs_enabled = fs_cfg[mode_key]
    fs_override = getattr(args, "factor_streaming_override", None)
    if fs_override is not None:
        fs_enabled = fs_override
    cfg["factor_streaming"]["enabled"] = fs_enabled
    cfg["factor_streaming"]["num_workers"] = fs_cfg["num_workers"][mode_key]

    # iter14 recipe-v3 audit (2026-05-10): FAIL LOUD when REPLAY=on but the
    # data loader cannot honor it. raw-replay branch lives ONLY in
    # StreamingFactorDataset.__iter__ (training.py:1838); legacy FactorSampler
    # path silently ignores raw_pretrain_pct. Without this guard, SANITY mode
    # (factor_streaming.sanity=false) accepts --replay on but never mixes raw
    # mp4 — orchestrator banner says "replay=on" while wiring is no-op.
    if cfg["replay"]["raw_pretrain_pct"] > 0.0 and not fs_enabled:
        sys.exit(
            f"❌ FATAL: replay.raw_pretrain_pct={cfg['replay']['raw_pretrain_pct']} > 0 "
            f"requires factor_streaming.enabled=true (raw-replay branch lives in "
            f"StreamingFactorDataset only). Got factor_streaming.{mode_key}={fs_enabled} "
            f"→ legacy FactorSampler path would silently drop replay. Either:\n"
            f"  (a) flip configs/train/surgery_base.yaml factor_streaming.{mode_key}=true\n"
            f"  (b) drop --replay on (use --replay off for {mode_key} mode)\n"
            f"  (c) pass --factor-streaming on the CLI to override yaml gate.")
    print(f"  [recipe-v3 receipts] saliency_weighting={cfg['optimization']['loss']['saliency_weighting']} "
          f"· spd.enabled={cfg['optimization']['spd']['enabled']} "
          f"· raw_pretrain_pct={cfg['replay']['raw_pretrain_pct']} "
          f"· factor_streaming.enabled={fs_enabled}")

    # iter13 v13 FIX-4 (2026-05-07): output dir resolution — two-tier priority:
    #   1. explicit --output-dir CLI flag (highest; shell wrapper uses this)
    #   2. derive variant tag from --train-config filename + use mode subdir from
    #      args. The yaml's data.output_dir field (e.g. "outputs/full/...") is
    #      INFORMATIONAL only — it hardcodes "full", so honoring it when running
    #      --SANITY would write SANITY artifacts to the FULL dir. The filename
    #      derivation matches the shell wrapper's VARIANT_TAG convention exactly:
    #        surgery_3stage_DI.yaml  → m09c_surgery_3stage_DI
    #        surgery_2stage_noDI.yaml → m09c_surgery_noDI  (yaml says "noDI"; see surgery_2stage_noDI.yaml:30)
    # Without this, two standalone m09c runs with different variants silently
    # overwrite the same outputs/<mode>/m09c_surgery/ dir → variant collision +
    # broken downstream eval.
    if getattr(args, "output_dir", None):
        cfg["checkpoint"]["output_dir"] = args.output_dir
        return cfg
    train_cfg_path = getattr(args, "train_config", None) or getattr(args, "config", None)
    if train_cfg_path:
        stem = Path(train_cfg_path).stem  # e.g. "surgery_3stage_DI"
        # Match shell wrapper convention: 2stage_noDI yaml → noDI tag (the "2stage_"
        # is run-recipe info, not a directory tag). Strip leading "surgery_" then
        # strip leading "2stage_" / "3stage_" if present so the tag is just the variant
        # discriminator (DI / noDI / loud_agent / ...).
        if stem.startswith("surgery_"):
            stem = stem[len("surgery_"):]
        for prefix in ("2stage_", "3stage_"):
            if stem.startswith(prefix):
                # 3stage_DI → DI is too short / collides with multi-stage variants;
                # keep the stage prefix for 3stage_DI (preserves yaml→shell mapping).
                # Drop ONLY for 2stage_noDI → noDI (matches shell VARIANT_TAG=noDI).
                if prefix == "2stage_":
                    stem = stem[len(prefix):]
                break
        module_name = f"m09c_surgery_{stem}"
    else:
        module_name = "m09c_surgery"
    base_out = get_module_output_dir(module_name, args.subset,
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

    # Load pretrained weights — iter14 sequential SSL: HF model-repo download via
    # hf_hub_download (HF_TOKEN from .env). SINGLE source, SINGLE schema, FAIL LOUD
    # on any mismatch (CLAUDE.md "no DEFAULT, no hardcoded paths"). The cfg key
    # init_from_ckpt is set from --init-from-ckpt CLI (required=True in argparse).
    project_root = Path(__file__).parent.parent
    init_from = cfg["init_from_ckpt"]   # always set — argparse required=True

    if not init_from.startswith("hf://"):
        print("FATAL: --init-from-ckpt must be hf:// URI.")
        print(f"  Got:      {init_from}")
        print("  Expected: hf://<owner>/<repo>/<filename>")
        print("  Example:  hf://anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep/m09a_ckpt_best.pt")
        sys.exit(1)

    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    load_dotenv(project_root / ".env")
    uri = init_from[len("hf://"):]
    parts = uri.split("/", 2)            # owner / repo / filename
    if len(parts) < 3:
        print(f"FATAL: bad --init-from-ckpt URI: {init_from}")
        print("  Expected: hf://<owner>/<repo>/<filename>")
        sys.exit(1)
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2]
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("FATAL: HF_TOKEN missing in .env — required for HF model-repo download.")
        print(f"  Repo: {repo_id}")
        print("  Fix: add HF_TOKEN=hf_... to .env (project root)")
        sys.exit(1)

    print(f"  [iter14] HF download: {repo_id}/{filename}")
    init_path = Path(hf_hub_download(
        repo_id=repo_id, filename=filename, token=hf_token))
    print(f"  [iter14] HF cached at: {init_path}")
    print(f"  [iter14] Loading init from prior-run ckpt: {init_path}")
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)

    # SINGLE schema — FAIL LOUD on mismatch. iter14 expects the FULL ckpt
    # (m09a_ckpt_best.pt) which carries key='student' (588 dims) + key='predictor'
    # (300 dims). The 'predictor' key is consumed downstream at the predictor-load
    # block below. NOT 'student_state_dict' — that's the encoder-only schema from
    # student_encoder.pt which lacks the predictor needed for JEPA L1 loss.
    # Verified at HF endpoint anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep.
    if not (isinstance(ckpt, dict) and "student" in ckpt
            and isinstance(ckpt["student"], dict)
            and "predictor" in ckpt
            and isinstance(ckpt["predictor"], dict)):
        print(f"FATAL: HF ckpt missing 'student' + 'predictor' schema: {init_path}")
        top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
        print(f"  Top-level: {top_keys}")
        print("  iter14 accepts ONLY full m09a_ckpt_best.pt schema "
              "(NOT student_encoder.pt's 'student_state_dict' — lacks predictor).")
        sys.exit(1)
    state_dict = ckpt["student"]
    print(f"  [iter14] Schema: student ({len(state_dict)} keys) + "
          f"predictor ({len(ckpt['predictor'])} keys)")

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
    # iter14 recipe-v2 (2026-05-09): teacher_mode determines whether the deepcopy
    # is later EMA-updated (legacy) or held FROZEN at the init checkpoint (SALT).
    # Read here for visibility; gating happens at the update_teacher_ema call site.
    _teacher_mode = cfg["surgery"]["teacher_mode"]
    if _teacher_mode == "FROZEN":
        print("Teacher created (deepcopy of student) — mode=FROZEN (SALT) "
              "→ EMA updates will be SKIPPED; teacher == init checkpoint forever")
    else:
        print(f"Teacher created (deepcopy of student) — mode=EMA "
              f"(legacy; τ={cfg['optimization']['ema_momentum']}, hierarchical output enabled)")

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
    # via set_trainable_prefix() is driven per-stage by train().
    return {
        "student": student,
        "teacher": teacher,
        "predictor": predictor,
        "explora_enabled": False,
    }


# ═════════════════════════════════════════════════════════════════════════
# SURGERY TRAINING (Ch11 — 3-stage progressive prefix unfreezing)
# ═════════════════════════════════════════════════════════════════════════

def train(cfg: dict, args):
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

    # Auto-bootstrap probe labels (action_labels.json + taxonomy_labels.json) if
    # missing. Lets `python -u src/m09c_surgery.py ...` run end-to-end without
    # the shell having pre-run run_probe_eval.sh Stage 1. No-op when both files
    # already exist (~1ms stat()s). Mirrors run_probe_eval.sh:342-358 exactly.
    mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
    # iter14 recipe-v3 (2026-05-09): pass cfg so probe_labels reads ALL paths +
    # numbers from yaml (CLAUDE.md "no hardcoded values"). For POC mode, the
    # bootstrap also generates the stratified-by-motion-class subset in-process.
    ensure_probe_labels_for_mode(
        mode_flag=mode_flag,
        project_root=Path(__file__).parent.parent,
        cache_policy=args.cache_policy,
        cfg=cfg,
    )

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

    # iter13 Task #19: snapshot of student weights at surgery start (CPU clones).
    # Surgery doesn't use drift control, but we still want per-block drift
    # diagnostic at every val checkpoint (catches stuck-encoder pathology).
    # Reference is "weights at surgery start" — independent of stage transitions
    # (each stage's progressive unfreeze is visible as bands shifting bottom-up).
    init_params = {name: p.clone().detach().cpu()
                   for name, p in student.named_parameters()}

    # Gradient checkpointing (#56): only on student (teacher runs under torch.no_grad
    # so no activations are stored — checkpointing would be wasted CPU overhead).
    # Enabled via configs/train/ch11_surgery.yaml:optimization.gradient_checkpointing.
    if cfg["optimization"]["gradient_checkpointing"]:
        enable_gradient_checkpointing(student)

    # Multi-task probe head — adds CrossEntropy/BCE loss on 16 taxonomy dims
    # to JEPA L1. Surgery's factor data path (StreamingFactorDataset OR
    # FactorSampler) already provides per-clip clip_key — we thread it
    # through into run_multi_task_step.
    mt_head, mt_labels_by_clip, mt_dims_spec, mt_cfg = build_multi_task_head_from_cfg(cfg, device)

    # iter13 v12+ (Phase 4): motion_aux head — JOINT K-class CE + 13-D MSE on
    # m04d RAFT-flow vec13d. Builds head + clip_key→(class_id, vec13d) lookup;
    # ma_head=None when motion_aux disabled (graceful no-op throughout).
    ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)

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

    # iter14 (2026-05-08): mirror m09a's gold-standard val pool pattern
    # (m09a:466-479). Single yaml/CLI-driven val pool via cfg["probe"]["subset"]
    # — at POC: data/eval_10k_val_split.json (125 stratified POC val clips); at
    # FULL: same path, larger contents (~1388 clips auto-generated by
    # run_probe_train.sh). Train manifest = full m11 manifest minus val (no
    # leakage). Surgery still trains via factor_index (intersection of
    # train_manifest with on-disk D_L/D_A/D_I .npy files); train_keys here only
    # sub-selects manifest entries to feed build_factor_index.
    val_path = cfg["probe"]["subset"]
    val_keys = json.load(open(val_path))["clip_keys"]
    val_set = set(val_keys)
    all_keys = list(manifest.keys())
    n_overlap = sum(1 for k in all_keys if k in val_set)
    if n_overlap:
        print(f"  Excluding {n_overlap} val clips from training manifest "
              f"(mirrors m09a:472-477 leakage guard)")
    train_keys = [k for k in all_keys if k not in val_set]
    split_source = f"EXTERNAL {val_path}"

    # iter14 (2026-05-08): POC mode pool capping happens UPSTREAM via the shell
    # generating data/eval_10k_poc.json (first N keys, N from
    # base_optimization.yaml:data.poc_total_clips). By the time m09c reaches
    # this split point, val_path resolves to data/eval_10k_val_split.json which
    # was already stratified 70:15:15 from action_labels (which itself was
    # filtered to motion-flow-eligible clips within the POC pool). No in-script
    # POC cap needed — single source of truth = poc_total_clips yaml key.
    train_manifest = {k: manifest[k] for k in train_keys}
    # iter14 (2026-05-08): val_split.json artifact RETIRED. Was a m09c-only
    # output that violated cross-encoder file-list parity with m09a (which
    # holds val_keys in-memory only). val_keys are still computed and used
    # in-process; persisting them to disk added no downstream value (no
    # script actually loaded the file — only training_summary.json kept a
    # path string). For audit reproducibility, val metadata is now embedded
    # in training_summary.json (see val_split block below at summary write).
    print(f"  train/val split: {len(train_keys)} train / {len(val_keys)} val "
          f"({split_source})")

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

    # iter14 recipe-v2 (2026-05-09): LP-FT Stage 0 — prepend head-only warmup stage.
    # When enabled, encoder is fully frozen (set_trainable_prefix(0) zeros all block
    # gradients; norm/ln layers still update). Predictor + motion_aux head are NOT
    # touched by set_trainable_prefix (separate modules) → they receive gradients,
    # giving the head a chance to "find" the factor-data manifold before the encoder
    # is allowed to move. Fixes step-1 feature distortion (Kumar ICLR'22 LP-FT).
    # See plan_surgery_wins.md §0 row 2️⃣ + §11.6 A4.
    if surgery_cfg["lp_ft_stage0"]["enabled"]:
        s0_cfg = surgery_cfg["lp_ft_stage0"]
        s0 = {
            "name": s0_cfg["name"],
            "unfreeze_below": 0.0,                # encoder frozen — head-only step
            "max_epochs_pct": s0_cfg["max_epochs_pct"],
            "mode_mixture": s0_cfg["mode_mixture"],
        }
        stages = [s0] + list(stages)
        print(f"  [LP-FT Stage 0 ENABLED] prepended head-only warmup stage: "
              f"{s0_cfg['max_epochs_pct']:.0%} of total_steps, mixture={s0['mode_mixture']}, "
              f"encoder FROZEN (predictor + motion_aux head trainable only)")

    # Multi-task loss + Uncertainty Weighting (Kendall et al. CVPR 2018) — enabled
    # when cfg["optimization"]["loss"]["uncertainty_weighting"]=true. nn.Module
    # holds N log-variance parameters (one per active task) — added to every
    # rebuilt per-stage optimizer below via add_param_group(). State persists across
    # stages (UW gradients are smooth, no Adam-moment reset needed).
    # errors_N_fixes #81: tcc_enabled drives UW shape (3-task with TCC, 2-task
    # without). compute_multitask_loss MUST be called with the matching task list.
    loss_cfg_top = cfg["optimization"]["loss"]
    uw_module = None
    if loss_cfg_top["uncertainty_weighting"]:
        uw_task_names = ["jepa", "infonce", "tcc"] if loss_cfg_top["tcc_enabled"] \
                        else ["jepa", "infonce"]
        uw_module = UncertaintyWeights(task_names=uw_task_names).to(device)
        print(f"[loss] Uncertainty Weighting ENABLED — α/β/γ ignored, "
              f"learning log-vars for {uw_task_names} "
              f"(init s=0 → all weights=1, will adapt during training); "
              f"tcc_scale={loss_cfg_top['tcc_scale']}")
    elif loss_cfg_top["beta_infonce"] > 0.0 or loss_cfg_top["gamma_tcc"] > 0.0:
        print(f"[loss] Multi-task FIXED weights: α={loss_cfg_top['alpha_jepa']} "
              f"β={loss_cfg_top['beta_infonce']} γ={loss_cfg_top['gamma_tcc']} "
              f"tcc_enabled={loss_cfg_top['tcc_enabled']} "
              f"tcc_scale={loss_cfg_top['tcc_scale']}")
    else:
        print("[loss] Pure JEPA L1 (α=1, β=γ=0) — default scaffold path, no multi-task active")

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
    csv_writer.writerow(["step", "stage", "loss_jepa", "loss_masked", "loss_context",
                         "loss_infonce", "loss_tcc",
                         "uw_w_jepa", "uw_w_infonce", "uw_w_tcc",
                         "lr", "grad_norm"])
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
    # iter13 Task #19: per-validation block-drift diagnostic — same as m09a.
    # Heatmap rendered every probe call to m09_block_drift.{png,pdf}.
    block_drift_history = []
    probe_cadence = probe_cfg["cadence"]
    probe_every = None
    probe_compute_val_loss = probe_cfg["compute_val_loss"]
    if probe_cfg["enabled"]:
        # iter13 v13 R5 (2026-05-07): subset/local_data/tags resolution moved to
        # setup_probe_pipeline below — this block now only sets cadence + jsonl.
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
        # iter13 v13 R5 (2026-05-07): probe-pipeline (build_probe_clips +
        # load_action_labels with --probe-action-labels CLI override + derived
        # fallback) factored to utils.m09_common.setup_probe_pipeline. Replaces
        # ~20 LoC of inline boilerplate. m09c-specific: feed held-out val_keys
        # via subset_keys_override to eliminate train/test overlap (m09a uses
        # external val_1k → no override needed).
        probe_clips, probe_labels = setup_probe_pipeline(
            cfg, args, output_dir,
            subset_keys_override=set(val_keys) if val_keys else None,
        )
        probe_jsonl_file = open(probe_jsonl_path, "w")
    else:
        print("[probe] disabled (SANITY mode or --no-probe) — skipping probe + val-loss eval")
        probe_jsonl_file = None
        probe_labels = None

    # iter13 v13 (2026-05-07): best-ckpt selected on probe_top1 (motion-flow gate).
    # Reason: paper-final probe-trio reports top-1, motion-cos, future-L1.
    # Legacy retrieval (Prec@K / mAP@K / Cycle@K) entirely retired — kill-switch /
    # plateau / BWT triggers all key on probe_top1 too (was prec_at_k pre-v13).
    best_state = {
        "trio_score": -float("inf"),
        "top1":       0.0,
        "motion_cos": 0.0,
        "future_l1":  float("inf"),
        "global_step": -1,
        "stage_name":  "",
        "probe_record": None,
    }
    kill_state = {"strikes": 0, "triggered": False, "reason": None}
    # Plateau + BWT trigger state (companion early-stop triggers).
    # v13 bug fix (#79, 2026-04-21): plateau buffers now track `last_stage_idx` and
    # reset when a new stage begins. Without this, a flat Stage-1 window killed
    # training at the FIRST Stage-2 probe (v13 logged prec_plateau kill after only
    # 1 S2 step). Reset semantics: buffer represents *within-current-stage* behaviour,
    # so Stage-1 plateau can't prematurely terminate Stage 2. BWT state is NOT reset
    # (it's cumulative-from-first-probe by design — that span-stages semantics is
    # intentional, not bugged).
    # iter14 (2026-05-08): top@1 plateau is the ONLY trigger. val_jepa
    # plateau_state + BWT bwt_state retired alongside their detectors.
    top1_plateau_state = {"recent_top1": [], "last_stage_idx": -1}
    # iter14 (2026-05-08): early-stop reduced to top@1 plateau ONLY.
    # val_jepa kill_switch + val_jepa plateau + BWT regression triggers were
    # removed from yaml because they don't correlate with downstream motion-flow
    # probe top@1 (the paper-grade gate metric).
    best_ckpt_enabled = probe_cfg["best_ckpt_enabled"]
    prec_plateau_enabled = probe_cfg["prec_plateau_enabled"]
    prec_plateau_min_delta = probe_cfg["prec_plateau_min_delta"]
    prec_plateau_patience = probe_cfg["prec_plateau_patience"]
    best_ckpt_path = output_dir / "student_best.pt"

    def _render_live_plots(verbose: bool = False):
        """iter14 (2026-05-08): renders cross-encoder-identical plot set by
        calling the same two shared utils m09a uses (plots.py) — gold-standard
        parity. The legacy render_training_plots (utils.training.py) is now a
        no-op stub kept only for legacy/m09b_explora.py import compatibility."""
        plot_probe_trajectory_trio(
            probe_history, output_dir,
            title_prefix=f"m09c · λ={cfg['drift_control']['lambda_reg']} · ",
            file_prefix="m09c",
        )
        plot_val_loss_with_kill_switch_overlay(
            probe_history, output_dir,
            best_state=best_state, kill_state=kill_state,
            file_prefix="m09c",
            title_prefix=f"m09c · λ={cfg['drift_control']['lambda_reg']} · ",
        )

    def _run_probe_at_step(stage_idx_, stage_name_, global_step_):
        """Run probe + optional val-loss, append to history, log + fsync.
        Also updates best-ckpt tracker + kill-switch state (both driven by
        probe_top1 — motion-flow gate metric). Silent success on probe failure
        (try/except) so a bad probe doesn't kill training — kill-switch acts
        only on successful probes.

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
            # iter13 v13 (2026-05-07): legacy retrieval probe (run_probe_eval →
            # Prec@K/mAP@K/Cycle@K) retired. Probe trio (top-1 + motion-cos +
            # future-L1) is the iter13 paper gate. Bare-init `pr` then have
            # run_trio_at_val populate probe_top1/motion_cos/future_l1.
            pr = {
                "num_clips":   len(probe_clips),
                "stage_idx":   stage_idx_,
                "stage_name":  stage_name_,
                "step":        global_step_,    # iter14 (2026-05-08): parity with m09a's probe_record schema
                                                # — plot_probe_trajectory_trio (plots.py:929) reads r["step"]
                                                # unconditionally. m09c's old "global_step" alias kept below
                                                # for backward compat with downstream consumers.
                "global_step": global_step_,
            }
            if probe_compute_val_loss:
                # iter13 v13 D6-FIX (2026-05-07, post-audit): full 4-loss schema
                # NUMERICAL parity with m09a — was previously symbolic (multi_task
                # + drift hardcoded to 0.0 regardless of config). run_probe_val_loss
                # now accepts the same mt_* + drift_* kwargs as run_validation. When
                # surgery configs disable both knobs (current default) values are
                # genuinely 0.0; if a future config re-enables them the val plots
                # will reflect the real numbers without any code change.
                # iter13 v13 FIX-5 (2026-05-07): added ma_head + ma_lookup + ma_cfg
                # kwargs so val_motion_aux_loss is genuinely computed (motion_aux is
                # surgery's primary aux loss per Phase 4 lever swap; previously
                # invisible at val time → NaN'd head only surfaced in train records).
                vl = run_probe_val_loss(
                    student, teacher, predictor, probe_clips,
                    mask_generators, cfg, device,
                    mt_head=mt_head, mt_dims_spec=mt_dims_spec,
                    mt_labels_by_clip=mt_labels_by_clip, mt_cfg=mt_cfg,
                    init_params=init_params, drift_cfg=cfg.get("drift_control"),
                    ma_head=ma_head, ma_lookup=ma_lookup, ma_cfg=ma_cfg,
                )
                pr["val_jepa_loss"] = vl["jepa_loss"]
                pr["val_masked_loss"] = vl["masked_loss"]
                pr["val_context_loss"] = vl["context_loss"]
                pr["val_multi_task_loss"] = vl["multi_task_loss"]
                pr["val_motion_aux_loss"] = vl["motion_aux_loss"]
                pr["val_drift_loss"] = vl["drift_loss"]
                pr["val_total_loss"] = vl["total_loss"]

            # Trio (top-1 + motion-cos + future-L1) — the iter13 paper metrics.
            # Adds `probe_top1`, `motion_cos`, `future_l1` to `pr`.
            if probe_labels:
                run_trio_at_val(
                    student, predictor, probe_clips, probe_labels,
                    mask_gen=mask_generators[0], cfg=cfg, device=device,
                    step=global_step_, wb_run=wb_run, probe_record=pr)

            # BWT = probe_top1[current] − probe_top1[first_probe].
            # Persisted to jsonl + plotted so users can see drift in real time.
            cur_top1 = pr.get("probe_top1", 0.0)
            first_top1 = (probe_history[0].get("probe_top1", cur_top1)
                          if probe_history else cur_top1)
            pr["bwt"] = cur_top1 - first_top1
            probe_history.append(pr)
            probe_jsonl_file.write(json.dumps(pr) + "\n")
            probe_jsonl_file.flush()
            os.fsync(probe_jsonl_file.fileno())
            vl_msg = f" val_jepa={pr['val_jepa_loss']:.4f}" if probe_compute_val_loss else ""
            mc_msg = f" motion_cos={pr['motion_cos']:.4f}" if "motion_cos" in pr else ""
            fl_msg = f" future_l1={pr['future_l1']:.4f}" if "future_l1" in pr else ""
            print(f"  [probe] step={global_step_} stage={stage_name_} N={pr['num_clips']} "
                  f"top-1={cur_top1:.4f}{mc_msg}{fl_msg}{vl_msg}")

            # iter13 Task #19: per-block drift diagnostic. Same pathology hunt
            # as m09a — catches uniform-noise across all blocks (= stuck encoder).
            _freeze_below_m09c = 0   # m09c uses progressive prefix-unfreezing
                                     # via set_trainable_prefix; no static freeze
                                     # below a single threshold. Pass 0 → plot
                                     # treats all blocks as "trainable" band.
            track_block_drift_at_val(
                student, init_params,
                freeze_below=_freeze_below_m09c,
                block_drift_history=block_drift_history,
                output_dir=output_dir, step=global_step_,
                probe_record=pr,
                title_prefix=f"m09c {stage_name_} step={global_step_} · ",
                file_prefix="m09c")

            # Best-ckpt tracker — iter13 v13 (2026-05-07): trio_score = top-1
            # from compute_metric_trio. Aligns best.pt selection with paper-final
            # probe-trio panel. Legacy retrieval prec_at_k removed entirely.
            current_top1 = pr.get("probe_top1")
            if best_ckpt_enabled and current_top1 is not None:
                def _save_best():
                    best_state.update({
                        "trio_score": current_top1,
                        "top1":       current_top1,
                        "motion_cos": pr.get("motion_cos", 0.0),
                        "future_l1":  pr.get("future_l1", float("inf")),
                        "global_step": global_step_,
                        "stage_name":  stage_name_,
                        "probe_record": pr,
                    })
                    export_student_for_eval(student, best_ckpt_path, explora_enabled=False)
                    print(f"  [best] new max trio_top1={current_top1:.4f} "
                          f"(step {global_step_}) → saved student_best.pt")
            # iter13 v13 R3 (2026-05-07): best-ckpt update + 4 early-stop triggers
            # (forgetting / val-loss-plateau / top1-plateau / negative-BWT) +
            # per-stage plateau resets (#79) factored to apply_val_cycle_triggers.
            # Replaces ~95 LoC of inline state-coupled bookkeeping. Helper mutates
            # best_state / kill_state / plateau_state / top1_plateau_state /
            # bwt_state in place. _save_best closure (defined above) is the
            # save-callback fired when probe_top1 sets a new max.
            apply_val_cycle_triggers(
                pr,
                probe_history=probe_history,
                best_state=best_state, kill_state=kill_state,
                top1_plateau_state=top1_plateau_state,
                best_ckpt_enabled=best_ckpt_enabled,
                save_best_callback=(_save_best if (best_ckpt_enabled and current_top1 is not None) else None),
                prec_plateau_enabled=prec_plateau_enabled,
                prec_plateau_min_delta=prec_plateau_min_delta,
                prec_plateau_patience=prec_plateau_patience,
                stage_idx=stage_idx_,
                global_step=global_step_,
                is_final_stage=(stage_idx_ == len(stages) - 1),
                probe_compute_val_loss=probe_compute_val_loss,
            )

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
                    x_axis_mode="steps",
                    file_prefix="m09c")
            except Exception as _e:
                # iter13 (2026-05-05): per CLAUDE.md FAIL HARD.
                print(f"  [live-plot] FATAL: train-curve render failed at step "
                      f"{global_step_}: {_e}", flush=True)
                print("  [live-plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                raise
            # iter13: 4-loss decomposition (jepa | drift | multi_task | total) on
            # one image — total_loss BOLD shows which component drives the optimizer.
            # Mirrors m09a's call. See utils.plots.plot_combined_losses.
            try:
                plot_combined_losses(
                    jsonl_path=output_dir / "loss_log.jsonl",
                    output_dir=output_dir,
                    title_prefix=f"m09c {stage_name_} · LR={cfg['optimization']['lr']:.1e} · ",
                    file_prefix="m09c",
                )
            except Exception as _e:
                # iter13 (2026-05-05): per CLAUDE.md FAIL HARD.
                print(f"  [live-plot] FATAL: combined-loss render failed: {_e}", flush=True)
                print("  [live-plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                raise
        except Exception as e:
            # iter13 (2026-05-05): per CLAUDE.md FAIL HARD — probe is research signal.
            print(f"  [probe] FATAL: eval failed at step {global_step_} stage {stage_name_}: {e}", flush=True)
            print("  [probe] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
            raise

    try:
        for stage_idx, stage_cfg in enumerate(stages):
            stage_name = stage_cfg["name"]
            n_trainable = int(depth * stage_cfg["unfreeze_below"])
            stage_pct = stage_cfg["max_epochs_pct"]
            stage_steps = max(int(total_steps * stage_pct), 1)
            # iter14 recipe-v3 audit A4 (2026-05-09): warmup mode dispatch.
            #   per_stage (legacy) = warmup_pct × stage_steps repeated every stage.
            #     BUG at POC: 1-step stages × warmup_pct=0.20 → warmup_steps=1
            #     → encoder ALWAYS in warmup, never sees configured base_lr.
            #   single (recipe-v3) = front-loaded warmup at stage 0 only;
            #     stages 1+ skip warmup, run at full base_lr immediately.
            warmup_mode = surgery_cfg["warmup_mode"]
            if warmup_mode == "per_stage":
                # Legacy: warmup_pct × stage_steps every stage. max(1, ...) guards
                # SANITY 1-step case where 20% floor-divides to 0.
                warmup_steps = max(1, int(stage_steps * surgery_cfg["warmup_pct"]))
            elif warmup_mode == "single":
                # Recipe-v3: single front-loaded warmup. Stage 0 absorbs the full
                # total_warmup_pct × total_steps budget; stages 1+ get 0 warmup
                # (lr_lambda returns 1.0 immediately).
                if stage_idx == 0:
                    warmup_steps = max(1, int(total_steps * surgery_cfg["total_warmup_pct"]))
                else:
                    warmup_steps = 0
            else:
                sys.exit(f"❌ FATAL: surgery.warmup_mode='{warmup_mode}' invalid. "
                         f"Expected 'per_stage' or 'single'.")
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
            # iter14 recipe-v3 (2026-05-09): pass init_params so SPDAdamW can
            # build its anchor mapping when cfg["optimization"]["spd"]["enabled"]
            # is True. Legacy AdamW path ignores init_params.
            optimizer = build_optimizer(student, predictor, cfg["optimization"],
                                         init_params=init_params)
            # Re-attach UW log-variance params to the freshly-built optimizer
            # (build_optimizer re-creates fresh Adam moments per stage; UW params
            # need their own param_group every time). Same base LR as encoder;
            # weight_decay=0 (don't regularize log-vars).
            if uw_module is not None:
                optimizer.add_param_group({
                    "params": list(uw_module.parameters()),
                    "lr": cfg["optimization"]["lr"],
                    "weight_decay": 0.0,
                })
            # Re-attach multi-task probe head every stage — fresh optimizer
            # per stage drops the head's param group, same rationale as UW above.
            attach_head_to_optimizer(optimizer, mt_head, mt_cfg, cfg["optimization"]["lr"])

            # iter13 v12+ (Phase 4): re-attach motion_aux head every stage too
            # (same rationale as multi-task above — fresh optimizer drops it).
            attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, cfg["optimization"]["lr"])

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
                # iter14 recipe-v3 #5 (2026-05-09): CLEAR raw replay. When
                # cfg["replay"]["raw_pretrain_pct"] > 0, the dataset interleaves
                # raw mp4 clips (no factor view) at the configured probability.
                # raw_clip_keys defaults to ALL keys in mp4_index (i.e., the
                # streaming manifest's pool). To anchor against pretrain
                # specifically, pass a different list (future enhancement).
                replay_pct = cfg["replay"]["raw_pretrain_pct"]
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
                    raw_replay_pct=replay_pct,
                    raw_clip_keys=None,  # default: full mp4_index pool
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
                # batch_keys threaded for multi-task probe loss (iter13). Streaming
                # yields {"tensor", "factor_type", "clip_key"} (utils/training.py:1363);
                # collator turns clip_key into a list. Sampler returns
                # (factor, clip_key, path) (utils/training.py:1071) — capture the key.
                batch_keys = []
                if stream_iter is not None:
                    batch = next(stream_iter)
                    batch_clips = batch["tensor"].to(device)              # (B, T, C, H, W)
                    batch_clips = batch_clips.permute(0, 2, 1, 3, 4)      # (B, C, T, H, W)
                    if "clip_key" in batch:
                        _ck = batch["clip_key"]
                        batch_keys = list(_ck) if not isinstance(_ck, list) else _ck
                else:
                    batch_tensors = []
                    for _ in range(batch_size):
                        _, clip_key, clip_path = sampler.sample()
                        clip_tensor = load_factor_clip(clip_path, num_frames, crop_size)
                        batch_tensors.append(clip_tensor)
                        batch_keys.append(clip_key)
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
                # target. iter14 (2026-05-08): wired init_params + drift_cfg through (was
                # None/None pre-iter14). compute_drift_loss applies `λ * Σ ‖θ - θ_init‖²`
                # when drift_control.enabled + lambda_reg > 0. With --init-from-ckpt
                # loading pretrain weights from HF, init_params (line 442) captures
                # pretrain → the L2 anchor target IS pretrain (iter14 design).
                #
                # Within-step retry loop (#55): on OOM, sizer.on_oom() shrinks; we retry
                # the SAME macro at the new sub-batch instead of continuing to the next
                # step. With stage_steps=1 (SANITY) the old `continue` skipped to a non-
                # existent next step → 0 successful steps → silent success. Now we retry
                # until either (a) success, (b) sizer at min and OOMed → fail-hard.
                step_succeeded = False
                while not step_succeeded:
                    try:
                        # iter14 (2026-05-08): drift_val is the L2 anchor-to-pretrain loss
                        # value (computed in _train_step_grad_accum when drift_control is
                        # enabled). Was discarded as `_drift_val` pre-iter14 because surgery
                        # had drift disabled; now logged in step_record for plot visibility.
                        (jepa_val, masked_val, context_val, drift_val,
                         infonce_val, tcc_val,
                         uw_w_jepa, uw_w_infonce, uw_w_tcc) = _train_step_grad_accum(
                            student, teacher, predictor, batch_clips,
                            all_masks_enc, all_masks_pred,
                            cfg, dtype, mp_cfg, scaler, train_sizer, loss_exp,
                            init_params=init_params, drift_cfg=cfg["drift_control"],
                            loss_cfg=cfg["optimization"]["loss"],
                            uw=uw_module)
                        step_succeeded = True
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad()  # discard partial grads from incomplete macro
                        # Force release of fragmented GPU memory between retries (mirrors
                        # m09a fix from probe_pretrain_sanity_v4.log: orphan tensors from
                        # the failed forward stayed allocated, so each successive sub-batch
                        # shrink started with LESS free VRAM → eventual OOM at sub-batch=1
                        # even though sub-batch=1 forward should fit). gc.collect() releases
                        # Python references; empty_cache() returns blocks to the CUDA driver.
                        gc.collect()
                        torch.cuda.empty_cache()
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

                # Multi-task forward+backward (no-op when mt_head is None or
                # batch_keys empty). Accumulates onto the same param.grad
                # buffer as the JEPA grads → single optimizer.step() consumes both.
                try:
                    mt_loss_val, mt_per_dim = run_multi_task_step(
                        student, mt_head, mt_cfg, mt_labels_by_clip, mt_dims_spec,
                        batch_clips, batch_keys, scaler, mp_cfg, dtype, device)
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    print(f"  OOM at step {global_step} (multi-task forward): "
                          f"discarding step, continuing")
                    continue

                # iter13 v12+ (Phase 4): motion_aux forward+backward (no-op when
                # ma_head is None). Accumulates grads onto the same buffers as
                # JEPA + multi_task → single optimizer.step() consumes all three.
                try:
                    ma_loss_val, ma_per_branch = run_motion_aux_step(
                        student, ma_head, ma_cfg, ma_lookup,
                        batch_clips, batch_keys, scaler, mp_cfg, dtype, device)
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    print(f"  OOM at step {global_step} (motion_aux forward): "
                          f"discarding step, continuing")
                    continue

                # Single optimizer step per macro batch — preserves effective BS = batch_size
                scaler.unscale_(optimizer)
                _clip_params = list(student.parameters()) + list(predictor.parameters())
                if mt_head is not None:
                    _clip_params += list(mt_head.parameters())
                if ma_head is not None:
                    _clip_params += list(ma_head.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    _clip_params,
                    cfg["optimization"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # EMA teacher update — iter14 recipe-v2 (2026-05-09): gated on
                # surgery.teacher_mode. FROZEN (SALT) skips this so the teacher
                # stays at the init checkpoint and JEPA loss targets don't decay
                # alongside a regressing student. See plan_surgery_wins.md §0 row 1️⃣.
                if cfg["surgery"]["teacher_mode"] == "EMA":
                    update_teacher_ema(student, teacher, ema_momentum)

                # Logging — values from _train_step_grad_accum are macro-batch means
                # (weighted sum of micro-batch values), preserving the per-step diagnostics.
                lr_val = scheduler.get_last_lr()[0]
                gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                # iter14 (2026-05-08): canonical total via shared utils.training.compute_total_loss
                # — required by plot_combined_losses (utils/plots.py:668) which reads loss_total
                # unconditionally. POC's 1st surgery_3stage_DI run hit KeyError here on the 2nd
                # row. Same source-of-truth as m09a step_record + val_total at training.py:939.
                total_val = compute_total_loss(
                    jepa=jepa_val, drift=drift_val,
                    mt=mt_loss_val if mt_head is not None else 0.0,
                    mt_cfg=mt_cfg if mt_head is not None else None,
                    ma=ma_loss_val if ma_head is not None else 0.0,
                    ma_cfg=ma_cfg if ma_head is not None else None)

                # iter14 (2026-05-08): per-step block_drift_mean — parity with m09a:965-967.
                # IDENTICAL metric to per-val (track_block_drift_at_val): both compute
                # mean_i(‖Δθ_block_i‖ / ‖θ₀_block_i‖). Surgery has λ=0.005 → drift signal
                # matters; logging per-step lets drift_table.py mix m09a + m09c columns
                # under one unified diagnostic.
                drift_per_block = compute_block_drift(student, init_params)
                block_drift_mean = (float(sum(drift_per_block) / len(drift_per_block))
                                    if drift_per_block else 0.0)

                step_record = {
                    "step": global_step, "stage": stage_name,
                    "loss_jepa": round(jepa_val, 6),
                    "loss_total": round(total_val, 6),     # iter14: canonical α·jepa + β·mt + γ·ma + λ·drift (matches training.py:939 val_total)
                    "loss_drift": round(drift_val, 6),     # iter14: L2 anchor-to-pretrain (λ=0.005, 0 when disabled)
                    "block_drift_mean": round(block_drift_mean, 8),     # iter14: parity with m09a:965 — drift_table.py needs unified column
                    "loss_masked": round(masked_val, 6),
                    "loss_context": round(context_val, 6),
                    "loss_infonce": round(infonce_val, 6),
                    "loss_tcc": round(tcc_val, 6),
                    "uw_w_jepa": round(uw_w_jepa, 6),
                    "uw_w_infonce": round(uw_w_infonce, 6),
                    "uw_w_tcc": round(uw_w_tcc, 6),
                    "lr": lr_val, "grad_norm": round(gn_val, 4),
                }
                if mt_head is not None:
                    step_record["loss_multi_task"] = round(mt_loss_val, 6)
                    step_record["loss_multi_task_per_dim"] = {
                        d: round(v, 6) for d, v in mt_per_dim.items()}
                if ma_head is not None:
                    step_record["loss_motion_aux"]     = round(ma_loss_val, 6)
                    step_record["loss_motion_aux_ce"]  = round(ma_per_branch.get("ce", 0.0), 6)
                    step_record["loss_motion_aux_mse"] = round(ma_per_branch.get("mse", 0.0), 6)
                    step_record["motion_aux_n_kept"]   = ma_per_branch.get("n_kept", 0)
                _log_step(step_record)
                csv_writer.writerow([global_step, stage_name, f"{jepa_val:.6f}",
                                     f"{masked_val:.6f}", f"{context_val:.6f}",
                                     f"{infonce_val:.6f}", f"{tcc_val:.6f}",
                                     f"{uw_w_jepa:.6f}", f"{uw_w_infonce:.6f}", f"{uw_w_tcc:.6f}",
                                     f"{lr_val:.2e}", f"{gn_val:.4f}"])

                wb_metrics = {
                    "loss/jepa": jepa_val, "loss/masked": masked_val,
                    "loss/context": context_val,
                    "loss/infonce": infonce_val, "loss/tcc": tcc_val,
                    "loss/uw_w_jepa": uw_w_jepa,
                    "loss/uw_w_infonce": uw_w_infonce,
                    "loss/uw_w_tcc": uw_w_tcc,
                    "lr": lr_val,
                    "grad_norm": gn_val, "stage": stage_idx,
                }
                if mt_head is not None:
                    wb_metrics["loss/multi_task"] = mt_loss_val
                    for d, v in mt_per_dim.items():
                        wb_metrics[f"loss/multi_task/{d}"] = v
                if ma_head is not None:
                    wb_metrics["loss/motion_aux"]     = ma_loss_val
                    wb_metrics["loss/motion_aux/ce"]  = ma_per_branch.get("ce", 0.0)
                    wb_metrics["loss/motion_aux/mse"] = ma_per_branch.get("mse", 0.0)
                log_metrics(wb_run, wb_metrics, step=global_step)

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
                            "top1_plateau": "📊", "negative_bwt": "📉"}[reason]
                    print(f"\n{icon}  EARLY-STOP [{reason}] — aborting training at step {global_step}")
                    print(f"     Best top1={best_state.get('top1', 0.0):.4f} saved at "
                          f"step {best_state['global_step']} (stage {best_state['stage_name']})")
                    break

            pbar.close()
            if kill_state["triggered"]:
                # Stage ckpt = resume/rollback anchor → full=True so optimizer +
                # scheduler + scaler restore correctly.
                save_training_checkpoint(output_dir / f"{CHECKPOINT_PREFIX}_stage{stage_idx}.pt",
                                         student, teacher, predictor, optimizer, scheduler,
                                         scaler, global_step, best_state.get("top1", 0.0), full=True,
                                         uw=uw_module)
                cleanup_stage_checkpoints(output_dir, CHECKPOINT_PREFIX, keep_n=1, cache_policy=args.cache_policy)
                _run_probe_at_step(stage_idx, stage_name, global_step)
                break
            save_training_checkpoint(output_dir / f"{CHECKPOINT_PREFIX}_stage{stage_idx}.pt",
                                     student, teacher, predictor, optimizer, scheduler,
                                     scaler, global_step, 0.0, full=True,
                                     uw=uw_module)
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
    # is the one that maxed motion-flow probe top-1 on the held-out val split —
    # eliminates the final-step-not-always-best problem. If no best was recorded
    # (probe disabled, or all probes failed), export current student weights.
    # Best-ckpt promotion: if best_ckpt_enabled fired and student_best.pt exists,
    # promote it to student_encoder.pt. Otherwise export current weights.
    if best_ckpt_enabled and best_ckpt_path.exists():
        shutil.move(str(best_ckpt_path), str(student_path))
        print(f"  [best] Promoted student_best.pt (top1={best_state.get('top1', 0.0):.4f} "
              f"at step {best_state['global_step']}, stage {best_state['stage_name']}) "
              f"→ student_encoder.pt")
    else:
        export_student_for_eval(student, student_path, explora_enabled=False)

    # iter13 v13 R4 (2026-05-07): post-export shared finalize_training
    # (assert_diverged + export_multi_task_head + export_motion_aux_head).
    init_ckpt_path = Path(__file__).parent.parent / cfg["model"]["checkpoint_path"]
    finalize_training(
        student=student, mt_head=mt_head, mt_dims_spec=mt_dims_spec,
        ma_head=ma_head, output_dir=output_dir,
        init_ckpt_path=init_ckpt_path,
        embed_dim=cfg["model"]["embed_dim"],
        label="m09c surgical encoder",
        # iter13 v13 (2026-05-07): SANITY: 1-step warmup → encoder ≡ init.
        skip_diverged_check=args.SANITY,
    )

    # Bug R8 fix (iter13): write m09c_ckpt_best.pt carrying the predictor so
    # Stage 8 probe_future_mse can load it. Without this, m09c writes
    # student_best.pt (full=False, encoder-only) which gets promoted to
    # student_encoder.pt and the only full ckpts are stage rollbacks
    # that get nuked by cleanup_stage_checkpoints(keep_n=0) below. The new
    # filename is OUTSIDE the cleanup pattern ({prefix}_stage*.pt) so survives.
    # iter13 disk-budget fix (2026-05-04, mirroring m09a fix): include_optimizer=False
    # drops 16 GB optimizer state; saves student+teacher+predictor only (~8 GB).
    # Downstream (m05 re-embed, Stage 8 future_mse) needs only those — optimizer
    # is dead weight. uw_module dropped along with optimizer (Kendall UW is part
    # of the optimization state). run_probe_eval.sh:158 expects this exact filename.
    save_training_checkpoint(
        output_dir / f"{CHECKPOINT_PREFIX}_best.pt",
        student, teacher, predictor, optimizer, scheduler, scaler,
        global_step, best_state.get("top1", 0.0),
        full=True, uw=uw_module, include_optimizer=False)
    print(f"Exported predictor-bearing best ckpt: "
          f"{output_dir / f'{CHECKPOINT_PREFIX}_best.pt'}")

    # Final checkpoint cleanup: `student_encoder.pt` + m09c_ckpt_best.pt are
    # the downstream artifacts (consumed by m05 surgical re-embed + m06 Prec@K
    # and probe_future_mse Stage 8 respectively). Stage rollback ckpts are
    # disposable once the run completes cleanly. Per CLAUDE.md "Clean all
    # intermediates after training." Saves ~15 GB per run at 2B model scale.
    # The keep_n=0 pattern only matches `{prefix}_stage*.pt`, NOT `_best.pt`
    # (different glob), so the predictor-bearing ckpt survives this call.
    cleanup_stage_checkpoints(output_dir, CHECKPOINT_PREFIX, keep_n=0, cache_policy=args.cache_policy)

    # Trajectory stats across stage boundaries. Single-probe-set regime so BWT
    # degenerates to net top-1 improvement (R[-1]-R[0]). Non-zero max_drop
    # flags a stage transition that hurt top-1 despite replay — paper's
    # "replay prevents forgetting" claim fails on this run if so.
    # iter13 v13 (2026-05-07): trajectory stats now keyed on probe_top1.
    traj_stats = compute_trajectory_stats(probe_history) if probe_history else {}
    if traj_stats and traj_stats.get("trajectory"):
        print(f"\n{'='*60}\n[probe] Trajectory across {len(probe_history)} stages:")
        print(f"  top1: {traj_stats['trajectory']}")
        print(f"  Δtop1 (BWT-proxy): {traj_stats['bwt_top1']:+.4f}")
        if not traj_stats["monotonic"]:
            print(f"  ⚠  max_drop = {traj_stats['max_drop_top1']:.4f} "
                  f"— some stage hurt top-1 despite replay")
        else:
            print("  ✓  monotonic improvement across stages")
        print("=" * 60)
        log_metrics(wb_run, {
            "probe/trajectory/bwt_top1": traj_stats["bwt_top1"],
            "probe/trajectory/max_drop_top1": traj_stats["max_drop_top1"],
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
            "seed": seed,
            "split_strategy": "external",     # iter14: only path; mirrors m09a
            "val_clip_keys": val_keys,        # iter14: embed inline (val_split.json retired)
            "source": split_source,
        },
        "best_ckpt": {
            "top1": best_state.get("top1", 0.0),
            "motion_cos": best_state.get("motion_cos", 0.0),
            "future_l1": best_state.get("future_l1", float("inf")),
            "global_step": best_state["global_step"],
            "stage_name": best_state["stage_name"],
            "probe_record": best_state["probe_record"],
        } if best_ckpt_enabled else None,
        "early_stop": {
            # iter14 (2026-05-08): top@1 plateau is the ONLY active trigger.
            # val_jepa kill_switch + val_jepa plateau + BWT regression removed —
            # see configs/train/base_optimization.yaml probe block.
            "triggered": kill_state["triggered"],
            "reason": kill_state["reason"],
            "top1_plateau": {
                "enabled": prec_plateau_enabled,
                "min_delta": prec_plateau_min_delta,
                "patience": prec_plateau_patience,
                "recent_values": list(top1_plateau_state["recent_top1"]),
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
        x_axis_mode="steps",
        file_prefix="m09c")

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
    # iter13 v13 R1 (2026-05-07): 16 shared CLI args (mode flags, --batch-size,
    # --max-epochs, --output-dir, --lambda-reg, --val-subset, --val-local-data,
    # --probe-* + --probe-action-labels + --no-probe, --taxonomy-labels-json,
    # --no-multi-task, --motion-features-path, --no-motion-aux, --subset,
    # --local-data, --cache-policy) live in utils.m09_common. Replaces ~80 LoC
    # of inline boilerplate. Technique-specific args (--factor-dir + factor-streaming
    # mutex group) added below.
    add_m09_common_args(parser)
    # iter14 (2026-05-08): m09c surgery REQUIRES init from a prior m09a pretrain
    # export hosted on HF. CLAUDE.md "no DEFAULT, FAIL LOUD" — required=True with
    # NO default, single source = HF model repo, single schema = student_state_dict.
    parser.add_argument("--init-from-ckpt", type=str, required=True,
                        help="iter14 REQUIRED: HF model-repo URI for prior pretrain student_encoder.pt. "
                             "Format: hf://<owner>/<repo>/<filename>. "
                             "Example: hf://anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep/m09a_ckpt_best.pt. "
                             "Schema = student_state_dict (FAIL LOUD on mismatch).")
    # iter14 recipe-v2 (2026-05-09): POC sweep axis #1. Overrides cfg["surgery"]["teacher_mode"]
    # from yaml. EMA = legacy deepcopy+EMA-update. FROZEN = SALT (Apple arXiv:2509.24317):
    # teacher = init from --init-from-ckpt, never updated. See plan_surgery_wins.md §0 row 1️⃣.
    parser.add_argument("--teacher-mode", type=str, choices=["EMA", "FROZEN"], default=None,
                        help="iter14 recipe-v2: override surgery.teacher_mode in yaml. "
                             "FROZEN = SALT (no EMA decay). Default None → use yaml value.")
    # iter14 recipe-v2 (2026-05-09): POC sweep axis #2. Overrides cfg["surgery"]["lp_ft_stage0"]["enabled"]
    # from yaml. LP-FT (Kumar ICLR'22): head-only warmup before backbone unfreeze. See plan_surgery_wins.md §0 row 2️⃣.
    parser.add_argument("--lp-ft-stage0", type=str, choices=["on", "off"], default=None,
                        help="iter14 recipe-v2: override surgery.lp_ft_stage0.enabled in yaml. "
                             "'on' prepends a head-only warmup stage (encoder frozen, predictor + "
                             "motion_aux head only). Default None → use yaml value.")
    # iter14 recipe-v3 (2026-05-09): five drop-one ablation switches (T7).
    # Each respects None = "use yaml". The shell wrapper forwards these from
    # SUBSET_OVERRIDE / WARMUP_OVERRIDE / SALIENCY_OVERRIDE / SPD_OVERRIDE /
    # REPLAY_OVERRIDE env-vars, used by scripts/run_recipe_v3_sweep.sh.
    parser.add_argument("--subset-mode", type=str, choices=["legacy", "recipe_v3"], default=None,
                        help="iter14 recipe-v3 #3: override surgery.stages[*].unfreeze_below. "
                             "legacy = 12/24/24 blocks (iter10 v15c). recipe_v3 = 4/8/8 (Lee ICLR'23 prescription). "
                             "Default None → use yaml value (recipe_v3 after T2).")
    parser.add_argument("--warmup-mode", type=str, choices=["per_stage", "single"], default=None,
                        help="iter14 recipe-v3 A4: override surgery.warmup_mode. "
                             "per_stage = legacy (every stage repeats warmup, BUG at POC). "
                             "single = front-loaded warmup at stage 0; subsequent stages skip warmup. "
                             "Default None → use yaml value.")
    parser.add_argument("--saliency", type=str, choices=["off", "on"], default=None,
                        help="iter14 recipe-v3 A2: override optimization.loss.saliency_weighting. "
                             "on = MGMAE-style per-token weighting by teacher-norm saliency. "
                             "Default None → use yaml value (off).")
    parser.add_argument("--spd", type=str, choices=["off", "on"], default=None,
                        help="iter14 recipe-v3 #4: override optimization.spd.enabled. "
                             "on = SPDAdamW (selective projection decay vs uniform L2 anchor). "
                             "Default None → use yaml value (off).")
    parser.add_argument("--replay", type=str, choices=["off", "on"], default=None,
                        help="iter14 recipe-v3 #5: override replay.raw_pretrain_pct. "
                             "on = 0.5 (50%% raw + 50%% factor batches, CLEAR Rolnick'18). "
                             "off = 0.0 (legacy factor-only). Default None → use yaml value.")
    parser.add_argument("--factor-dir", type=str, default=None,
                        help="Factor dataset dir from m11 (contains D_L/, D_A/, D_I/, factor_manifest.json)")
    fs_group = parser.add_mutually_exclusive_group()
    fs_group.add_argument("--factor-streaming", dest="factor_streaming_override",
                          action="store_true", default=None,
                          help="Force streaming factor generation (overrides yaml mode gate)")
    fs_group.add_argument("--no-factor-streaming", dest="factor_streaming_override",
                          action="store_false",
                          help="Force legacy .npy factor path (overrides yaml mode gate)")
    add_wandb_args(parser)
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
    train(cfg, args)


if __name__ == "__main__":
    import traceback as _traceback
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)   # force exit: CUDA atexit + producer threads can deadlock at exit
    except SystemExit:
        raise         # honor explicit sys.exit() codes
    except BaseException as _exc:
        # Fail-loud + force-kill non-daemon threads on unhandled exception.
        # Without os._exit, producer TAR readers + factor-streaming workers keep
        # the process alive indefinitely (mirrors m09a fix; same root cause:
        # 2026-05-03 disk-full incident). See m10_sam_segment.py:1127-1133.
        print(f"\nFATAL (unhandled m09c exception): {type(_exc).__name__}: {_exc}",
              file=sys.stderr)
        _traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
