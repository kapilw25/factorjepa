"""V-JEPA 2.1 HEAD-ONLY surgery: factor-aug data + FROZEN encoder + FROZEN predictor. GPU-only.
Gold standard: https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py
Claude Code: re-WebSearch this URL on every read of this file.

iter15 Phase 2 (2026-05-14): head-only sibling of m09c1_surgery_encoder.py. All 48 ViT-G
blocks + predictor FROZEN; only the motion_aux head trains. Differs from m09a2 by the
DATA path — uses StreamingFactorDataset for D_L/D_A/D_I factor-augmented clips per the
recipe-v3 mode_mixture. Single training stage (vs m09c1's 2-3 progressive unfreeze stages)
since the encoder is frozen always. Rule 32: zero cross-imports from m09a1/m09a2/m09c1;
shared primitives via utils/training.py + utils/motion_aux_loss.py + utils/factor_streaming.py.

USAGE (every path arg required — CLAUDE.md no-default rule):
    python -u src/m09c2_surgery_head.py --SANITY \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/surgery_3stage_DI_head.yaml \
        --subset data/val_1k_local/sanity_100_dense.json --local-data data/val_1k_local \
        --val-subset data/val_1k_local/val_1k.json --val-local-data data/val_1k_local \
        --no-wandb 2>&1 | tee logs/m09c2_3stage_DI_head_sanity_$(date +%Y%m%d_%H%M%S).log
    python -u src/m09c2_surgery_head.py --POC \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/surgery_2stage_noDI_head.yaml \
        --subset data/eval_10k_local/eval_10k_train_split.json --local-data data/eval_10k_local \
        --val-subset data/eval_10k_local/eval_10k_val_split.json --val-local-data data/eval_10k_local \
        --no-wandb 2>&1 | tee logs/m09c2_noDI_head_poc_$(date +%Y%m%d_%H%M%S).log
    python -u src/m09c2_surgery_head.py --FULL \
        --model-config configs/model/vjepa2_1.yaml \
        --train-config configs/train/surgery_3stage_DI_head.yaml \
        --subset data/eval_10k_local/eval_10k_train_split.json --local-data data/eval_10k_local \
        --val-subset data/eval_10k_local/eval_10k_val_split.json --val-local-data data/eval_10k_local \
        --no-wandb 2>&1 | tee logs/m09c2_3stage_DI_head_full_$(date +%Y%m%d_%H%M%S).log
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils.live_debug import install_debug_handlers
install_debug_handlers()

from utils.config import (
    check_gpu, get_module_output_dir, load_subset,
    get_pipeline_config, load_merged_config,
)
from utils.data_download import ensure_local_data
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.progress import make_pbar
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.cache_policy import resolve_cache_policy_interactive, wipe_output_dir
from utils.m09_common import add_m09_common_args, merge_m09_common_config
from utils.vjepa2_imports import (
    get_vit_by_arch, get_vit_predictor, get_vit_predictor_2_1,
)
from utils.training import (
    load_config,
    build_optimizer, build_scheduler,
    assert_encoder_frozen, set_trainable_prefix,
    export_student_for_eval,
    augment_clip_consistent,
    StreamingFactorDataset, build_streaming_indices, _streaming_worker_init,
    # iter15 Phase 6 C1+C2 (2026-05-16): trio + head-drift wiring for head cells.
    build_probe_clips, run_trio_at_val, track_head_drift_at_val,
    build_mask_generators,
)
from utils.action_labels import load_action_labels
from utils.motion_aux_loss import (
    build_motion_aux_head_from_cfg,
    attach_motion_aux_to_optimizer, run_motion_aux_step,
    export_motion_aux_head,
)
from utils.probe_labels import ensure_probe_labels_for_mode
# iter15 Phase 6 audit (2026-05-16): emit-parity with m09c1 — same loss_log
# schema + plot calls so outputs/{poc,full}/m09c_surgery_*_head/ matches the
# encoder-cell layout.
from utils.plots import (
    plot_training_curves, plot_combined_losses,
    plot_probe_trajectory_trio, plot_val_loss_with_kill_switch_overlay,
)

CHECKPOINT_PREFIX = "m09c_ckpt"  # output filename preserved for downstream eval compat
_pcfg = get_pipeline_config()


def merge_config_with_args(cfg: dict, args) -> dict:
    """Mode-gated config merge: delegates to utils.m09_common (shared with m09a/c)."""
    if args.SANITY:
        mode_key = "sanity"
    elif args.POC:
        mode_key = "poc"
    else:
        mode_key = "full"
    merge_m09_common_config(cfg, args, mode_key)

    # === Force head-only contract regardless of yaml/CLI ===
    # iter15 Phase 5 V0 preflight fix (2026-05-14): surgery configs use per-stage
    # `unfreeze_below` (via set_trainable_prefix in build_model), NOT the global
    # `layer_freeze` switch — so we DON'T touch cfg["layer_freeze"] here (the
    # key doesn't exist in surgery_base.yaml inheritance chain). Head-only
    # encoder freeze is enforced at runtime in build_model() via:
    #   1. set_trainable_prefix(student, 0)  → all blocks frozen
    #   2. assert_encoder_frozen(student)    → fail-loud guard
    # The per-stage validation below also requires stages[0].unfreeze_below=0.0.
    cfg["drift_control"]["enabled"] = False
    cfg["loss"]["weight_jepa"] = 0.0
    cfg["loss"]["weight_motion_aux"] = 1.0

    # === Force single head-only stage ===
    # The yaml SHOULD already declare exactly one stage with unfreeze_below=0.0
    # (per pretrain_head + surgery_*_head.yaml). Fail loud if it doesn't.
    stages = cfg["surgery"]["stages"]
    if len(stages) != 1:
        print(f"FATAL [m09c2]: head-only mode requires exactly 1 surgery stage in yaml; "
              f"found {len(stages)}. Use configs/train/surgery_*_head.yaml not _encoder.yaml.")
        sys.exit(1)
    if stages[0]["unfreeze_below"] != 0.0:
        print(f"FATAL [m09c2]: head-only stage 0 must set unfreeze_below=0.0; "
              f"got {stages[0]['unfreeze_below']}.")
        sys.exit(1)

    # === Force factor streaming on (raw replay mixture lives only in streaming path) ===
    cfg["factor_streaming"]["enabled"] = True
    fs_cfg = cfg["factor_streaming"]
    cfg["factor_streaming"]["num_workers"] = fs_cfg["num_workers"][mode_key]

    # === Output dir: explicit --output-dir, or auto from mode ===
    if args.output_dir is not None:
        cfg["checkpoint"]["output_dir"] = args.output_dir
    else:
        base_out = get_module_output_dir(
            "m09c2_surgery_head", args.subset,
            sanity=args.SANITY, poc=args.POC)
        # Append variant tag from yaml's adapted_encoder so 3stage_DI_head + noDI_head
        # write to DIFFERENT subdirs (downstream eval scans by encoder name).
        variant_tag = cfg["data"]["adapted_encoder"].replace("vjepa_2_1_surgical_", "")
        cfg["checkpoint"]["output_dir"] = str(base_out / variant_tag)
    return cfg


def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder + predictor, BOTH FROZEN. assert_encoder_frozen() validates."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    arch = model_cfg["arch"]

    vit_constructor = get_vit_by_arch(arch)
    vit_predictor = get_vit_predictor()

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

    project_root = Path(__file__).parent.parent
    ckpt_path = project_root / model_cfg["checkpoint_path"]
    ckpt_url = model_cfg["checkpoint_url"]
    if ckpt_path.exists():
        print(f"Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        print(f"Downloading pretrained weights: {ckpt_url}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = torch.hub.load_state_dict_from_url(
            ckpt_url, map_location="cpu", model_dir=str(ckpt_path.parent))

    if "target_encoder" in ckpt:
        state_dict = ckpt["target_encoder"]
    elif "encoder" in ckpt:
        state_dict = ckpt["encoder"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in state_dict.items()}
    msg = student.load_state_dict(state_dict, strict=False)
    total_keys = len(list(student.state_dict().keys()))
    loaded_keys = total_keys - len(msg.missing_keys)
    load_pct = loaded_keys / max(total_keys, 1) * 100
    print(f"Student loaded: {sum(p.numel() for p in student.parameters()):,} params "
          f"({loaded_keys}/{total_keys} keys = {load_pct:.0f}%)")
    if msg.missing_keys:
        unexpected_missing = [k for k in msg.missing_keys if "pos_embed" not in k]
        if unexpected_missing:
            print(f"FATAL: {len(unexpected_missing)} unexpected missing keys in student ckpt")
            for k in unexpected_missing[:10]:
                print(f"    {k}")
            sys.exit(1)
    if load_pct < model_cfg["min_student_load_pct"]:
        print(f"FATAL: only {load_pct:.0f}% of student keys loaded. Ckpt incompatible.")
        sys.exit(1)
    student = student.to(device)
    if hasattr(student, "return_hierarchical"):
        student.return_hierarchical = True

    # === FREEZE encoder via set_trainable_prefix(0) — mirrors m09c1 stage logic ===
    # set_trainable_prefix(student, 0) sets all blocks frozen; norms remain trainable
    # by Meta convention. We immediately re-freeze norms here to ensure STRICT zero
    # encoder gradient (head-only contract).
    set_trainable_prefix(student, 0)
    for p in student.parameters():
        p.requires_grad = False
    assert_encoder_frozen(student)
    student.eval()
    print("[m09c2 STRICT HEAD-ONLY] encoder FROZEN: 0 trainable block params (asserted)")

    # === Predictor: load Meta weights, FROZEN ===
    pred_constructor = get_vit_predictor_2_1() if model_cfg["predict_all"] else vit_predictor
    predictor = pred_constructor(
        img_size=(crop_size, crop_size),
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
    if "predictor" not in ckpt:
        print("FATAL: ckpt has no 'predictor' key — V-JEPA 2.1 distribution must include it")
        sys.exit(1)
    pred_state = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in ckpt["predictor"].items()}
    pred_msg = predictor.load_state_dict(pred_state, strict=False)
    pred_total = len(list(predictor.state_dict().keys()))
    pred_loaded = pred_total - len(pred_msg.missing_keys)
    pred_pct = pred_loaded / max(pred_total, 1) * 100
    print(f"Predictor loaded: {pred_loaded}/{pred_total} keys = {pred_pct:.0f}%")
    if pred_pct < model_cfg["min_predictor_load_pct"]:
        print(f"FATAL: predictor only {pred_pct:.0f}% loaded")
        sys.exit(1)
    for p in predictor.parameters():
        p.requires_grad = False
    predictor = predictor.to(device)
    predictor.eval()
    print(f"Predictor: {sum(p.numel() for p in predictor.parameters()):,} params (FROZEN)")

    init_ckpt_path = str(ckpt_path)

    del ckpt
    gc.collect()

    return {
        "student": student,
        "predictor": predictor,
        "init_ckpt_path": init_ckpt_path,
        "explora_enabled": False,
    }


def _build_factor_loader(cfg: dict, train_keys: list, mode_mixture: dict,
                         stage_steps: int, base_seed: int) -> DataLoader:
    """Construct StreamingFactorDataset + DataLoader for factor-aug clips.

    Mirrors m09c1's setup (src/m09c1_surgery_encoder.py:1237-1262) — the
    StreamingFactorDataset generates D_L / D_A / D_I tubes on demand from
    (raw_mp4, mask.npz) pairs per the recipe-v3 mode_mixture.
    """
    data_cfg = cfg["data"]
    num_frames = data_cfg["num_frames"]
    crop_size = cfg["model"]["crop_size"]
    batch_size = cfg["optimization"]["batch_size"]
    local_data = data_cfg["local_data"]
    manifest_path = Path(local_data) / "m11_factor_datasets" / "factor_manifest.json"
    masks_dir = Path(local_data) / "m10_sam_segment" / "masks"
    if not manifest_path.exists():
        print(f"FATAL: factor_manifest.json missing at {manifest_path}. "
              f"Run scripts/run_factor_prep.sh to generate.")
        sys.exit(1)
    # build_streaming_indices returns 3-tuple (m09c1:659 baseline):
    # (mp4_index, mask_index, streaming_manifest). The manifest re-read below is
    # redundant — fn already loads + validates it during the scan.
    mp4_index, mask_index, streaming_manifest = build_streaming_indices(
        manifest_path=manifest_path,
        masks_dir=masks_dir,
        local_data=local_data,
    )
    # Build factor_cfg by remapping yaml keys → stream_factor's expected names.
    # Mirrors m09c1:667-673 baseline. FAIL LOUD per CLAUDE.md: cfg["factor_datasets"]
    # crashes here if the yaml chain is broken (no .get() fallback to silent {}).
    fcy = cfg["factor_datasets"]
    factor_cfg_streaming = {
        "layout_method": fcy["layout_patch_method"],
        "agent_method":  fcy["agent_patch_method"],
        "matte_factor":  fcy["soft_matte_factor"],
        "blur_sigma":    fcy["blur_sigma"],
        "feather_sigma": fcy["feather_sigma"],
    }
    replay_pct = cfg["replay"]["raw_pretrain_pct"]
    ds = StreamingFactorDataset(
        mp4_index=mp4_index,
        mask_index=mask_index,
        factor_manifest=streaming_manifest,
        factor_cfg=factor_cfg_streaming,
        mode_mixture=mode_mixture,
        num_frames=num_frames,
        crop_size=crop_size,
        di_legacy_index=None,
        base_seed=base_seed,
        steps_per_epoch=stage_steps * batch_size,
        interaction_cfg=cfg["interaction_mining"],
        raw_replay_pct=replay_pct,
        raw_clip_keys=None,
    )
    fs_cfg = cfg["factor_streaming"]
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=fs_cfg["num_workers"],
        prefetch_factor=fs_cfg["prefetch_factor"] if fs_cfg["num_workers"] > 0 else None,
        persistent_workers=fs_cfg["persistent_workers"] if fs_cfg["num_workers"] > 0 else False,
        pin_memory=fs_cfg["pin_memory"],
        worker_init_fn=_streaming_worker_init if fs_cfg["num_workers"] > 0 else None,
    )
    return loader


def _compute_val_motion_aux_loss(student, ma_head, ma_cfg, ma_lookup,
                                  val_keys, val_local_data, cfg, device,
                                  dtype) -> float:
    """One-pass val motion_aux loss over val_keys (synchronous, raw clips).

    Val intentionally uses RAW clips (not factor-aug) — the eval-time motion-class
    taxonomy is defined on raw videos, so val loss should measure head accuracy on
    raw clip features, not on artificially augmented factor tubes.

    iter15 Phase 5 V2 fix (2026-05-15): mirrors m09a1:583-611 val pipeline —
    decode_video_bytes(mp4, tmp_dir, key, num_frames) returns variable-shape;
    augment_clip_consistent then resizes to fixed crop_size. _val_tmp dir is
    scoped to one function call via try/finally cleanup.
    """
    from utils.video_io import create_stream, decode_video_bytes, get_clip_key
    ma_head.eval()
    student.eval()
    total_loss = 0.0
    total_n = 0
    val_keys_set = set(val_keys)
    ds = create_stream(local_data=val_local_data)
    batch_clips, batch_keys = [], []
    batch_size = cfg["optimization"]["batch_size"]
    num_frames = cfg["data"]["num_frames"]
    mp_cfg = cfg["mixed_precision"]
    _val_tmp = tempfile.mkdtemp(prefix="m09c2_val_")
    try:
        for example in ds:
            k = get_clip_key(example)
            if k not in val_keys_set:
                continue
            mp4 = example.get("mp4", b"")
            mp4_bytes = mp4["bytes"] if isinstance(mp4, dict) else mp4
            if isinstance(mp4_bytes, str):
                mp4_bytes = mp4_bytes.encode()
            if not mp4_bytes or len(mp4_bytes) < 1000:
                continue
            clip = decode_video_bytes(mp4_bytes, _val_tmp, k, num_frames)
            if clip is None:
                continue
            clip = augment_clip_consistent(clip, cfg["augmentation"],
                                            cfg["data"]["crop_size"])
            batch_clips.append(clip)
            batch_keys.append(k)
            if len(batch_clips) >= batch_size:
                # (T, C, H, W) per clip → stack (B, T, C, H, W) → permute (B, C, T, H, W)
                # matches utils/training.py:160 producer + m09a1:603/609 val pattern.
                clips_tensor = torch.stack(batch_clips).permute(0, 2, 1, 3, 4).to(device)
                with torch.no_grad():
                    loss_val, _ = run_motion_aux_step(
                        student, ma_head, ma_cfg, ma_lookup,
                        clips_tensor, batch_keys, scaler=None,
                        mp_cfg=mp_cfg, dtype=dtype, device=device,
                    )
                total_loss += float(loss_val) * len(batch_keys)
                total_n += len(batch_keys)
                batch_clips, batch_keys = [], []
        if batch_clips:
            clips_tensor = torch.stack(batch_clips).permute(0, 2, 1, 3, 4).to(device)
            with torch.no_grad():
                loss_val, _ = run_motion_aux_step(
                    student, ma_head, ma_cfg, ma_lookup,
                    clips_tensor, batch_keys, scaler=None,
                    mp_cfg=mp_cfg, dtype=dtype, device=device,
                )
            total_loss += float(loss_val) * len(batch_keys)
            total_n += len(batch_keys)
    finally:
        shutil.rmtree(_val_tmp, ignore_errors=True)
    ma_head.train()
    if total_n == 0:
        print("FATAL: val cycle saw 0 clips — val_subset / val_local_data mismatch?")
        sys.exit(1)
    return total_loss / total_n


def train(cfg: dict, args) -> None:
    """Head-only surgery: frozen encoder + predictor; only motion_aux head trains.

    Data path: factor-augmented clips via StreamingFactorDataset (mode_mixture from
    yaml; e.g. 3stage_DI = {L:0.15, A:0.15, I:0.70}, noDI = {L:0.50, A:0.50, I:0.00}).
    """
    check_gpu()
    print_cgroup_header(prefix="[m09c2]")
    start_oom_watchdog(prefix="[m09c2]-oom-watchdog")
    device = torch.device("cuda")

    seed = cfg["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(cfg["checkpoint"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    train_keys = load_subset(args.subset)
    val_keys = load_subset(args.val_subset)
    print(f"Train: {len(train_keys):,} keys · Val: {len(val_keys):,} keys")

    mode_key = "sanity" if args.SANITY else ("poc" if args.POC else "full")
    mode_flag = "--SANITY" if args.SANITY else ("--POC" if args.POC else "--FULL")
    ensure_probe_labels_for_mode(
        mode_flag=mode_flag,
        project_root=Path(__file__).parent.parent,
        cache_policy=args.cache_policy,
        cfg=cfg,
    )

    # === Build model (frozen encoder + frozen predictor) ===
    model_d = build_model(cfg, device)
    student = model_d["student"]
    predictor = model_d["predictor"]
    init_ckpt_path = model_d["init_ckpt_path"]

    # === Build motion_aux head ===
    ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)
    if ma_head is None:
        print("FATAL [m09c2]: motion_aux head REQUIRED — sole training signal in head-only mode")
        sys.exit(1)

    # === Optimizer over head params only ===
    opt_cfg = cfg["optimization"]
    optimizer = build_optimizer(student, predictor, opt_cfg, init_params=None)
    n_enc_pred_params = sum(p.numel() for grp in optimizer.param_groups for p in grp["params"])
    if n_enc_pred_params > 0:
        print(f"FATAL [m09c2]: build_optimizer returned {n_enc_pred_params:,} encoder/predictor "
              f"trainable params — expected 0. Check requires_grad freeze in build_model().")
        sys.exit(1)
    attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, base_lr=opt_cfg["lr"])
    head_params = sum(p.numel() for p in ma_head.parameters() if p.requires_grad)
    print(f"Trainable params: motion_aux head = {head_params:,} (~432K expected)")

    # iter15 Phase 6 C1+C2 (2026-05-16): snapshot motion_aux head init + build
    # probe_clips for in-training trio. Same wiring as m09a2 (mirrors m09c1
    # encoder-side flow). probe is universal-true post yaml-parity audit.
    # variant_tag hoisted from build_model scope for in-training plot titles.
    _variant_tag = cfg["data"]["adapted_encoder"].replace("vjepa_2_1_surgical_", "")
    head_init_params = {n: p.detach().cpu().clone() for n, p in ma_head.named_parameters()}
    head_drift_history = []
    probe_history = []
    # iter15 D15 (2026-05-16): probe-trio encoder cache. Encoder + predictor are
    # FROZEN for the entire run → encoder forward + future_l1 outputs are time-
    # invariant across val checkpoints. First call populates this dict; subsequent
    # calls reuse the cached pooled features + per-clip L1 → MA head re-runs each
    # time (it IS the trained component). Saves ~10 min × (N_val - 1) val cycles.
    # compute_metric_trio FAILS LOUD if encoder signature changes mid-run.
    encoder_cache: dict = {}
    probe_history_path = output_dir / "probe_history.jsonl"
    mask_generators = build_mask_generators(cfg)
    print(f"Mask generators: {len(mask_generators)} (for in-training trio future_l1)")
    probe_clips, probe_labels = None, None
    _probe_block = cfg["probe"] if "probe" in cfg else None
    if _probe_block is not None and _probe_block.get("enabled", False):
        action_labels_path = (args.probe_action_labels or
                              str(Path(_probe_block["subset"]).parent / "action_labels.json"))
        if not Path(action_labels_path).exists():
            print(f"❌ FATAL [probe]: action_labels.json not found at {action_labels_path}", file=sys.stderr)
            sys.exit(3)
        probe_labels = load_action_labels(Path(action_labels_path))
        print(f"  [probe] decoding clips from {_probe_block['subset']} ...", flush=True)
        probe_clips = build_probe_clips(
            probe_subset_path=_probe_block["subset"],
            probe_local_data=_probe_block["local_data"],
            probe_tags_path=_probe_block["tags_path"],
            num_frames=cfg["data"]["num_frames"],
            crop_size=cfg["model"]["crop_size"],
            max_clips=cfg["monitoring"]["knn_probe_clips"],
        )
        print(f"  [probe] decoded {len(probe_clips)} clips ({len(probe_labels)} have action labels)", flush=True)

    # === Single head-only stage (validated in merge_config_with_args) ===
    surgery_cfg = cfg["surgery"]
    stage_cfg = surgery_cfg["stages"][0]
    stage_name = stage_cfg["name"]
    mode_mixture = stage_cfg["mode_mixture"]
    max_epochs_pct = stage_cfg["max_epochs_pct"]
    if abs(max_epochs_pct - 1.0) > 1e-6:
        print(f"FATAL [m09c2]: head-only mode requires max_epochs_pct=1.0 in the single stage; "
              f"got {max_epochs_pct}.")
        sys.exit(1)

    max_epochs = opt_cfg["max_epochs"]
    batch_size = opt_cfg["batch_size"]
    n_train = len(train_keys)
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = max_epochs * steps_per_epoch
    # iter15 D13 (2026-05-16): mirror m09a1:538 cadence — val/probe-trio every
    # val_interval steps (= steps_per_epoch // saves_per_epoch). Previously head
    # cells validated ONLY at end-of-epoch (1×/ep) while encoder cells used 2×/ep
    # → asymmetric trajectory plots. Aligns observability with m09a2 + m09a1/c1.
    saves_per_epoch = cfg["checkpoint"]["saves_per_epoch"]
    val_interval = max(1, steps_per_epoch // saves_per_epoch)
    print(f"Stage: {stage_name} · mixture: {mode_mixture} · mode: {mode_key} · "
          f"epochs: {max_epochs} · batch: {batch_size} · steps/epoch: {steps_per_epoch}")
    print(f"Validation every {val_interval} steps ({saves_per_epoch}x/epoch — D13 cadence parity)")

    scheduler = build_scheduler(optimizer, opt_cfg, total_steps)

    mp_cfg = cfg["mixed_precision"]
    dtype = torch.bfloat16 if mp_cfg["dtype"] == "bfloat16" else torch.float16
    # iter15 Phase 5 V2 fix (2026-05-15): GradScaler required for motion_aux
    # backward (motion_aux_loss.py:413). No-op for bfloat16 default but the
    # object must exist. Mirrors m09a1:560-561.
    use_scaler = mp_cfg["enabled"] and mp_cfg["dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    sizer = AdaptiveBatchSizer(
        initial_size=batch_size,
        min_size=1,
        max_size=batch_size,
        memory_cap=_pcfg["gpu"]["gpu_memory_target"],
    )
    print(f"AdaptiveBatchSizer: {sizer}")

    mode_label = mode_key.upper()
    wb_run = init_wandb("m09c2", mode_label, config=vars(args),
                          enabled=not args.no_wandb)

    # === Build factor-aug DataLoader (replaces m09a2's producer_thread) ===
    loader = _build_factor_loader(
        cfg=cfg,
        train_keys=train_keys,
        mode_mixture=mode_mixture,
        stage_steps=total_steps,
        base_seed=seed,
    )

    # iter15 Phase 6 audit (2026-05-16): unified emit-set with m09c1 — renamed
    # training_log.jsonl → loss_log.jsonl + added loss_log.csv (m09c1 schema).
    # Frozen-encoder fields (loss_jepa/loss_drift/loss_multi_task) emit NaN/0.
    jsonl_path = output_dir / "loss_log.jsonl"
    summary_path = output_dir / "training_summary.json"
    train_log_f = jsonl_path.open("a", buffering=1)
    csv_path = output_dir / "loss_log.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["step", "epoch", "loss_jepa", "loss_drift", "loss_total",
                             "loss_multi_task", "loss_motion_aux",
                             "lr", "grad_norm", "throughput", "val_loss", "stage"])
        csv_file.flush()

    best_val_loss = float("inf")
    best_epoch = -1
    pbar = make_pbar(total=total_steps, desc=f"m09c2 head-only [{stage_name}]", unit="step")
    step = 0
    t_start = time.time()

    try:
        loader_iter = iter(loader)
        for epoch in range(max_epochs):
            ma_head.train()
            student.eval()
            epoch_train_losses = []
            epoch_started = time.time()

            for _ in range(steps_per_epoch):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    # DataLoader exhausted before stage budget — re-iterate.
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                batch_clips = batch["tensor"].to(device)            # (B, T, C, H, W)
                batch_clips = batch_clips.permute(0, 2, 1, 3, 4)    # (B, C, T, H, W)
                _ck = batch["clip_key"]
                batch_keys = list(_ck) if not isinstance(_ck, list) else _ck

                try:
                    optimizer.zero_grad(set_to_none=True)
                    loss_val, per_branch = run_motion_aux_step(
                        student, ma_head, ma_cfg, ma_lookup,
                        batch_clips, batch_keys, scaler=scaler,
                        mp_cfg=mp_cfg, dtype=dtype, device=device,
                    )
                    optimizer.step()
                    scheduler.step()
                    sizer.after_batch_success()
                except torch.cuda.OutOfMemoryError:
                    print(f"[m09c2] OOM at step {step}, sub-batch {sizer.size}")
                    cuda_cleanup()
                    if not sizer.on_oom():
                        print("FATAL [m09c2]: OOM at min sub-batch — cannot continue")
                        sys.exit(1)
                    continue

                epoch_train_losses.append(float(loss_val))
                step += 1
                pbar.update(1)
                if step % 20 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    row = {
                        "step": step, "epoch": epoch,
                        "loss_jepa": float("nan"),       # encoder frozen → no JEPA gradient
                        "loss_drift": 0.0,                # encoder frozen → drift ≡ 0
                        "loss_multi_task": float("nan"),  # multi_task off in head-only
                        "loss_motion_aux": float(loss_val),
                        "loss_total": float(loss_val),    # head-only total = motion_aux
                        "lr": cur_lr,
                        "branch": per_branch,
                        "stage": stage_name,
                    }
                    train_log_f.write(json.dumps(row) + "\n")
                    train_log_f.flush()
                    os.fsync(train_log_f.fileno())
                    csv_writer.writerow([step, epoch, "", "0", f"{float(loss_val):.6f}",
                                          "", f"{float(loss_val):.6f}",
                                          f"{cur_lr:.6e}", "", "", "", stage_name])
                    csv_file.flush()
                    log_metrics(wb_run, {"train/loss": float(loss_val),
                                         "train/lr": cur_lr},
                                step=step)

                # iter15 D13 (2026-05-16) + D16 fix (2026-05-16): val cycle every
                # val_interval steps. Mirrors m09a1/c1 cadence (saves_per_epoch=2 →
                # 2 probes per epoch). D16: removed the `step % steps_per_epoch == 0`
                # OR-clause that fired EXTRA val cycles when steps_per_epoch was not
                # evenly divisible by saves_per_epoch (e.g. POC m09c2: 35 steps/ep,
                # val_interval=17 → step 35 fired 1 step after step 34, wasting an
                # 11-min probe-trio cycle). The val_interval cadence alone gives
                # exactly saves_per_epoch fires per epoch (last fire near, but not
                # exactly at, end-of-epoch when there's a remainder).
                if step > 0 and step % val_interval == 0:
                    mean_train = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
                    val_loss = _compute_val_motion_aux_loss(
                        student, ma_head, ma_cfg, ma_lookup,
                        val_keys, args.val_local_data, cfg, device, dtype,
                    )
                    elapsed = time.time() - epoch_started
                    is_end_of_epoch = (step % steps_per_epoch == 0)
                    tag = "epoch-end" if is_end_of_epoch else "mid-epoch"
                    print(f"\n[step {step} · epoch {epoch}/{max_epochs} · {tag}] "
                          f"train_loss={mean_train:.4f}  val_loss={val_loss:.4f}  "
                          f"wall={elapsed:.0f}s  stage={stage_name}")
                    row = {"epoch": epoch, "train_loss": mean_train, "val_loss": val_loss,
                           "wall_sec": elapsed, "step": step, "stage": stage_name}
                    train_log_f.write(json.dumps(row) + "\n")
                    train_log_f.flush()
                    os.fsync(train_log_f.fileno())
                    csv_writer.writerow([step, epoch, "", "", "", "", "",
                                          "", "", "", f"{val_loss:.6f}", stage_name])
                    csv_file.flush()
                    log_metrics(wb_run, {"val/loss": val_loss,
                                         "train/epoch_mean_loss": mean_train,
                                         "epoch": epoch}, step=step)

                    val_total = val_loss * float(ma_cfg["weight_motion"])
                    probe_record = {
                        "step": step, "epoch": epoch,
                        "val_jepa_loss":       val_total,
                        "val_motion_aux_loss": val_loss,
                        "val_total_loss":      val_total,
                        "val_drift_loss":      0.0,
                        "val_multi_task_loss": 0.0,
                        "stage":               stage_name,
                        "epoch_pct": round((epoch + 1) / max_epochs * 100, 1),
                    }
                    if probe_clips is not None and probe_labels:
                        run_trio_at_val(
                            student, predictor, probe_clips, probe_labels,
                            mask_gen=mask_generators[0], cfg=cfg, device=device,
                            step=step, wb_run=wb_run, probe_record=probe_record,
                            motion_aux_head=ma_head,
                            encoder_cache=encoder_cache,   # iter15 D15
                            encoder_frozen=True,           # iter15 D15: head cell, encoder+predictor FROZEN
                        )
                    track_head_drift_at_val(
                        ma_head, head_init_params, head_drift_history,
                        output_dir=output_dir, step=step,
                        probe_record=probe_record,
                        title_prefix=f"m09c head [{_variant_tag}] step={step} · ",
                        file_prefix="m09c2",
                    )
                    probe_history.append(probe_record)
                    with open(probe_history_path, "a") as _ph:
                        _ph.write(json.dumps(probe_record) + "\n")
                        _ph.flush()
                        os.fsync(_ph.fileno())

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        export_motion_aux_head(ma_head, output_dir / "motion_aux_head.pt")
                        print(f"  ✅ new best val_loss={best_val_loss:.4f} (step {step}, epoch {epoch})")

                    # iter15 D17 (2026-05-16): symmetric per-val plot rendering with
                    # m09c1 encoder cell (which renders 5 plots per val: train_loss,
                    # loss_decomposition, block_drift, val_loss_jepa, probe_trajectory
                    # _trio). Head cells previously rendered ONLY block_drift per val
                    # (via track_head_drift_at_val) → 4 plots missing mid-training.
                    # Asymmetric observability between encoder + head sibling modules.
                    # End-of-training plots stay (idempotent, regenerates final view).
                    # FAIL LOUD on render exceptions per CLAUDE.md.
                    best_state_live = {
                        "step": step if best_epoch == epoch else -1,
                        "val_loss_at_best": float(best_val_loss),
                        "probe_top1": float(probe_record.get("probe_top1", -1.0)),
                    }
                    kill_state_live = {"triggered": False, "reason": None}
                    try:
                        plot_training_curves(
                            runs=[{"csv_path": str(csv_path),
                                   "label": f"m09c2 head-only [{_variant_tag}]",
                                   "color": "blue",
                                   "batch_size": batch_size}],
                            output_dir=str(output_dir),
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · {len(train_keys):,} train × "
                                         f"{max_epochs} ep × BS={batch_size} · step={step}\n",
                            file_prefix="m09c2",
                        )
                        plot_combined_losses(
                            jsonl_path=jsonl_path,
                            output_dir=output_dir,
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · LR={opt_cfg['lr']:.1e} · step={step} · ",
                            file_prefix="m09c2",
                        )
                        plot_val_loss_with_kill_switch_overlay(
                            probe_history, output_dir,
                            best_state=best_state_live, kill_state=kill_state_live,
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · val_total (motion_aux × weight) · step={step}\n",
                            file_prefix="m09c2",
                        )
                        plot_probe_trajectory_trio(
                            probe_history, output_dir,
                            title_prefix=f"m09c2 head-only [{_variant_tag}] · step={step} · ",
                            file_prefix="m09c2",
                        )
                    except Exception as _e:
                        print(f"  [plot] FATAL: per-val plot render failed at step {step}: {_e}", flush=True)
                        print("  [plot] traceback follows; aborting per CLAUDE.md FAIL HARD:", flush=True)
                        raise
    finally:
        train_log_f.close()
        csv_file.close()

    pbar.close()

    # iter15 Phase 6 audit (2026-05-16): emit-parity plots — match m09c1.
    # _variant_tag already hoisted above (used by in-training trio plots too).
    try:
        plot_training_curves(
            runs=[{"csv_path": str(csv_path),
                   "label": f"m09c2 head-only [{_variant_tag}]",
                   "color": "blue",
                   "batch_size": batch_size}],
            output_dir=str(output_dir),
            title_prefix=f"m09c2 head-only [{_variant_tag}] · {len(train_keys):,} train × "
                         f"{max_epochs} ep × BS={batch_size}\n",
            file_prefix="m09c2",
        )
    except Exception as e:
        print(f"  [plot] WARN train_loss render skipped: {type(e).__name__}: {e}", flush=True)
    try:
        plot_combined_losses(
            jsonl_path=jsonl_path,
            output_dir=output_dir,
            title_prefix=f"m09c2 head-only [{_variant_tag}] · ",
            file_prefix="m09c2",
        )
    except Exception as e:
        print(f"  [plot] WARN loss_decomposition render skipped: {type(e).__name__}: {e}", flush=True)
    # iter15 Phase 6 C1+C3 (2026-05-16): probe_trajectory_trio + val_loss_jepa
    # plots from probe_history. track_head_drift_at_val (C2) already emits the
    # m09c_block_drift.{png,pdf} + m09c_block_drift_history.json per val cycle.
    # probe.enabled is universal-true (yaml audit 2026-05-16) → probe_history is
    # always non-empty by the time we reach this line.
    plot_probe_trajectory_trio(
        probe_history, output_dir,
        title_prefix=f"m09c2 head-only [{_variant_tag}] · ",
        file_prefix="m09c2",
    )
    kill_state = {"triggered": False, "reason": None}
    # D4 fix (2026-05-16): plot reads "val_loss_at_best" (plots.py:857), not
    # "val_jepa_loss_at_best". Use the schema the plot expects.
    best_state = {"step": -1, "val_loss_at_best": float("inf"), "probe_top1": -1.0}
    plot_val_loss_with_kill_switch_overlay(
        probe_history, output_dir,
        best_state=best_state, kill_state=kill_state,
        title_prefix=f"m09c2 head-only [{_variant_tag}] · val_total\n",
        file_prefix="m09c2",
    )

    # === Finalization ===
    student_export = output_dir / "student_encoder.pt"
    export_student_for_eval(student, student_export, explora_enabled=False)

    combined_ckpt = output_dir / f"{CHECKPOINT_PREFIX}_best.pt"
    torch.save({
        "student_state_dict": student.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "motion_aux_head_state_dict": ma_head.state_dict(),
        "n_motion_classes": ma_head.n_motion_classes,
        "n_motion_dims": ma_head.n_motion_dims,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stage_name": stage_name,
        "mode_mixture": mode_mixture,
        "type": "m09c2_head_only_surgery",
    }, combined_ckpt)
    print(f"Saved: {combined_ckpt}")

    summary = {
        "mode": mode_key,
        "adapted_encoder": cfg["data"]["adapted_encoder"],
        "stage_name": stage_name,
        "mode_mixture": mode_mixture,
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "total_steps": step,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "wall_sec": round(time.time() - t_start, 1),
        "head_params": head_params,
        "init_ckpt_path": init_ckpt_path,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {summary_path}")

    finish_wandb(wb_run)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 HEAD-ONLY surgery (m09c2 — iter15 Phase 2).")
    add_m09_common_args(parser, require_val_data=True)
    add_wandb_args(parser)
    args = parser.parse_args()

    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_merged_config(args.model_config, args.train_config)
    cfg = merge_config_with_args(cfg, args)

    if args.cache_policy == "2":
        wipe_output_dir(cfg["checkpoint"]["output_dir"], args.cache_policy,
                          label="m09c2 head-only surgery output dir")

    train(cfg, args)


if __name__ == "__main__":
    main()
