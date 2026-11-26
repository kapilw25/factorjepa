"""V-JEPA 2.1 HEAD-ONLY continual SSL: train motion_aux head on FROZEN encoder + predictor. GPU-only.
Gold standard: https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py
Claude Code: re-WebSearch this URL on every read of this file.

iter15 Phase 2 (2026-05-14): head-only sibling of m09a1_pretrain_encoder.py. All 48 ViT-G
blocks + predictor frozen → no backward through 1.84 B-param encoder → no activation
storage → ViT-G fits 24 GB RTX Pro 4000 at $0.20/hr. ONLY the ~432 K-param motion_aux
head trains (joint K-class CE + 23-D MSE on m04d optical-flow targets). Rule 32: zero
cross-imports from m09a1; shared primitives via utils/training.py + utils/motion_aux_loss.py.

USAGE — FULL arg set (run_train.sh is the canonical caller; it wires EVERY arg below.
Eyeball to confirm nothing is silently cfg-defaulted. <LD> = pipeline.yaml
data.local_data_dir; <M> = sanity|poc|full; swap --FULL→--SANITY/--POC for other tiers):
    python -u src/m09a2_pretrain_head.py --FULL \
        --model-config         configs/model/vjepa2_1.yaml \
        --train-config         configs/train/pretrain_head.yaml \
        --subset               <LD>/train_pool.json \
        --local-data           <LD> \
        --val-subset           <LD>/val_split.json \
        --val-local-data       <LD> \
        --output-dir           outputs/<M>/m09a_pretrain_head \
        --cache-policy         <1=keep|2=recompute> \
        --probe-subset         <LD>/val_split.json \
        --probe-local-data     <LD> \
        --probe-tags           <LD>/tags.json \
        --probe-action-labels  outputs/<M>/probe_action/action_labels.json \
        --motion-features-path <LD>/m04d_motion_features/motion_features.npy \
        --taxonomy-labels-json outputs/<M>/probe_taxonomy/taxonomy_labels.json \
        --no-wandb 2>&1 | tee logs/m09a2_pretrain_head_full.log
    # HEAD-ONLY: encoder+predictor FROZEN (no --lambda-reg/--init-from-ckpt). --subset is
    #   the leakage-safe pool (clip_splits, iter17). Optional: --no-probe / --no-multi-task.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gc
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

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
from utils.cache_policy import resolve_cache_policy_interactive, wipe_output_dir, is_recompute
from utils.m09_common import add_m09_common_args, merge_m09_common_config
# iter17 DRY #31: ViT/predictor constructors now live behind utils.training.build_student_predictor
from utils.training import (
    load_config, producer_thread,
    build_optimizer, build_scheduler,
    assert_encoder_frozen, finalize_outputs,   # iter17 DRY #34: export+ckpt+summary
    build_student_predictor,       # iter17 DRY #31: shared student+predictor construction
    TrainLogWriter,                # iter17 DRY #32: crash-safe jsonl+csv loss log
    compute_val_motion_aux_loss,   # iter17 DRY: shared with m09c2 (was local _compute_…)
    # iter15 Phase 6 C1+C2 (2026-05-16): trio + head-drift wiring for head cells.
    build_probe_clips, run_trio_at_val, track_head_drift_at_val,
    build_mask_generators,
    TimedSaveGate, atomic_torch_save,   # iter18 (2026-06-06): wall-clock resume anchor
)
from utils.action_labels import load_action_labels
# iter15 Phase 6 audit (2026-05-16): emit-parity with m09a1 — add plot calls so
# outputs/{poc,full}/m09a_pretrain_head/ has same file layout as encoder cells.
from utils.plots import render_val_plots   # iter17 DRY #33: shared 4-plot per-val render
from utils.motion_aux_loss import (
    build_motion_aux_head_from_cfg,
    attach_motion_aux_to_optimizer, run_motion_aux_step,
    export_motion_aux_head,
)
from utils.probe_labels import ensure_probe_labels_for_mode
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

CHECKPOINT_PREFIX = "m09a_ckpt"  # output filename preserved for downstream eval compat
_pcfg = get_pipeline_config()
PREFETCH_QUEUE_SIZE = _pcfg["streaming"]["prefetch_queue_train"]


def merge_config_with_args(cfg: dict, args) -> dict:
    """Mode-gated config merge: delegates to utils.m09_common (shared with m09a1/c1)."""
    if args.SANITY:
        mode_key = "sanity"
    elif args.POC:
        mode_key = "poc"
    else:
        mode_key = "full"
    merge_m09_common_config(cfg, args, mode_key)

    # Head-only contract values (layer_freeze.freeze_below, drift_control.enabled,
    # loss.weight_jepa=0.0, loss.weight_motion_aux=1.0) are DECLARED in
    # configs/train/pretrain_head.yaml — the single source of truth. We do NOT re-set
    # or assert them against literals here (CLAUDE.md: no hardcoded values, no fallback;
    # an assert-vs-literal just relocates the hardcode). The contract is ENFORCED at
    # runtime: build_model sets requires_grad=False on every encoder+predictor param +
    # assert_encoder_frozen(), and train() FAILs LOUD if the optimizer sees any
    # encoder/predictor trainable param (n_enc_pred_params > 0). Any consumer of a
    # missing yaml key raises KeyError (fail loud) — no silent default.

    # Output dir: explicit --output-dir, or auto from mode (no lambda — drift off).
    if args.output_dir is not None:
        cfg["checkpoint"]["output_dir"] = args.output_dir
    else:
        base_out = get_module_output_dir(
            "m09a2_pretrain_head", args.subset,
            sanity=args.SANITY, poc=args.POC)
        cfg["checkpoint"]["output_dir"] = str(base_out)
    return cfg


def build_model(cfg: dict, device: torch.device) -> dict:
    """Build student encoder + predictor, BOTH FROZEN. Returns dict; head built in train()."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    # iter17 DRY #31: shared student+predictor construction via utils.training (kwargs
    # identical across m09a1/a2/c1/c2). Predictor is built here too — it was constructed
    # lower; behaviour-neutral (construction is deterministic + independent of ckpt-load).
    # This cell still owns the load/freeze logic below.
    student, predictor = build_student_predictor(model_cfg, data_cfg)

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

    # === FREEZE encoder (head-only contract) — every block AND every param ===
    # iter15 Phase 2: assert_encoder_frozen() validates blocks specifically;
    # we additionally freeze norms+patch_embed so ZERO encoder params receive grad.
    for p in student.parameters():
        p.requires_grad = False
    assert_encoder_frozen(student)
    student.eval()  # disable dropout during the frozen forward
    print("[m09a2 STRICT HEAD-ONLY] encoder FROZEN: 0 trainable block params (asserted)")

    # === Predictor: load Meta weights, FROZEN === (constructed above via build_student_predictor)
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

    # Store init ckpt path for later student_encoder.pt COPY (encoder is bit-identical to init).
    init_ckpt_path = str(ckpt_path)

    del ckpt
    gc.collect()

    return {
        "student": student,
        "predictor": predictor,
        "init_ckpt_path": init_ckpt_path,
        "explora_enabled": False,
    }


def train(cfg: dict, args) -> None:
    """Head-only training loop. Frozen encoder + predictor; only motion_aux head moves."""
    check_gpu()
    print_cgroup_header(prefix="[m09a2]")
    start_oom_watchdog(prefix="[m09a2]-oom-watchdog")
    device = torch.device("cuda")

    seed = cfg["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(cfg["checkpoint"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    train_keys = load_subset(args.subset)
    val_keys = load_subset(args.val_subset)
    # iter17 (2026-05-26): clip-count per mode is governed by clip_pool_ratio[mode] (pipeline.yaml)
    # applied in clip_splits.py → the --subset train_pool is already ratio-scaled (SANITY/POC/FULL).
    # No in-trainer cap (the fixed sanity_train_clips cap was removed — it couldn't scale to 115k).
    print(f"Train: {len(train_keys):,} keys · Val: {len(val_keys):,} keys")

    # Mode-gated action_labels.json must exist (motion_aux's CE branch needs it).
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

    # === Build motion_aux head — the SOLE trainable component ===
    ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)
    if ma_head is None:
        print("FATAL [m09a2]: motion_aux head is REQUIRED — it is the sole training signal "
              "for head-only mode. Check yaml: motion_aux.enabled.{mode} must be true.")
        sys.exit(1)

    # === Optimizer over head params only (encoder/predictor naturally excluded by requires_grad=False) ===
    opt_cfg = cfg["optimization"]
    optimizer = build_optimizer(student, predictor, opt_cfg, init_params=None)
    n_enc_pred_params = sum(p.numel() for grp in optimizer.param_groups for p in grp["params"])
    if n_enc_pred_params > 0:
        print(f"FATAL [m09a2]: build_optimizer returned {n_enc_pred_params:,} encoder/predictor "
              f"trainable params — expected 0. Check requires_grad freeze in build_model().")
        sys.exit(1)
    attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, base_lr=opt_cfg["lr"])
    head_params = sum(p.numel() for p in ma_head.parameters() if p.requires_grad)
    print(f"Trainable params: motion_aux head = {head_params:,} (~432K expected)")

    # iter15 Phase 6 C1+C2 (2026-05-16): snapshot motion_aux head init for
    # head-drift diagnostic + build probe_clips for in-training trio. Encoder
    # is frozen → encoder-drift = 0 by contract; HEAD drift is the meaningful
    # one. Mirrors m09a1:670-733 plumbing for run_trio_at_val.
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
    # probe block is declared in the yaml chain (single source) — read directly.
    _probe_block = cfg["probe"]
    if _probe_block["enabled"]:
        action_labels_path = (args.probe_action_labels or
                              str(Path(_probe_block["subset"]).parent / artifact("action_labels")))
        if not Path(action_labels_path).exists():
            print(f"❌ FATAL [probe]: action_labels.json not found at {action_labels_path}", file=sys.stderr)
            sys.exit(3)
        probe_labels = load_action_labels(Path(action_labels_path))
        # iter18 (2026-06-06) probe-leak fix: probe the HELD-OUT val split (same
        # pool as the m09c-family), not _probe_block["subset"] — the old path
        # probed action_labels.json which overlaps the train pool. train_pool_keys
        # arms the shared [probe-leak guard].
        print(f"  [probe] decoding {len(val_keys)} held-out val clips "
              f"({args.val_subset}) ...", flush=True)
        probe_clips = build_probe_clips(
            probe_subset_path=_probe_block["subset"],
            probe_local_data=_probe_block["local_data"],
            probe_tags_path=_probe_block["tags_path"],
            num_frames=cfg["data"]["num_frames"],
            crop_size=cfg["model"]["crop_size"],
            subset_keys_override=set(val_keys),
            max_clips=cfg["monitoring"]["knn_probe_clips"],
            train_pool_keys=set(train_keys),
        )
        print(f"  [probe] decoded {len(probe_clips)} clips ({len(probe_labels)} have action labels)", flush=True)

    max_epochs = opt_cfg["max_epochs"]
    batch_size = opt_cfg["batch_size"]
    n_train = len(train_keys)
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = max_epochs * steps_per_epoch
    # iter15 D13 (2026-05-16): mirror m09a1:538 cadence — read saves_per_epoch from
    # cfg and fire val/probe-trio every val_interval steps. Previously head cells
    # validated ONLY at end-of-epoch (1×/ep) while encoder cells validated 2×/ep,
    # producing asymmetric trajectory plots (2 vs 4 points). Aligns observability.
    saves_per_epoch = cfg["checkpoint"]["saves_per_epoch"]
    val_interval = max(1, steps_per_epoch // saves_per_epoch)
    print(f"Mode: {mode_key} · epochs: {max_epochs} · batch: {batch_size} · "
          f"steps/epoch: {steps_per_epoch} · total steps: {total_steps}")
    print(f"Validation every {val_interval} steps ({saves_per_epoch}x/epoch — D13 cadence parity with m09a1)")

    scheduler = build_scheduler(optimizer, opt_cfg, total_steps)

    # === Mixed-precision + AdaptiveBatchSizer (OOM safety on 24 GB) ===
    mp_cfg = cfg["mixed_precision"]
    # Supported mixed-precision dtypes (whitelist mirrors base_optimization.yaml
    # mixed_precision.dtype). KeyError = fail loud on a bad/typo'd dtype string —
    # was a ternary that SILENTLY defaulted to float16 on any non-bfloat16 value.
    # Unified across m09a1/a2/c1/c2.
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[mp_cfg["dtype"]]
    # iter15 Phase 5 V2 fix (2026-05-15): GradScaler is required for the
    # motion_aux backward in run_motion_aux_step (scaler.scale(loss).backward()
    # at motion_aux_loss.py:413 — unconditional once requires_grad=True).
    # `enabled=use_scaler` makes it a no-op for bfloat16 (the default in
    # base_optimization.yaml), but the object must exist. Mirrors m09a1:560-561.
    use_scaler = mp_cfg["enabled"] and mp_cfg["dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    sizer = AdaptiveBatchSizer(
        initial_size=batch_size,
        min_size=1,
        max_size=batch_size,
        memory_cap=_pcfg["gpu"]["gpu_memory_target"],
    )
    print(f"AdaptiveBatchSizer: {sizer}")

    # === Wandb ===
    mode_label = mode_key.upper()
    wb_run = init_wandb("m09a2", mode_label, config=vars(args),
                          enabled=not args.no_wandb)

    # iter18 (2026-06-06): wall-clock resume anchor (head-only) — mirrors m09c2
    # (sibling parity). Saved every checkpointing.timed_save_interval_s in the
    # step loop below; loaded here under --cache-policy 1. Must run BEFORE the
    # producer thread starts so processed_steps gets the resumed step (the
    # producer then enqueues exactly the remaining-step budget).
    timed_gate = TimedSaveGate(_pcfg["checkpointing"]["timed_save_interval_s"])
    resume_path = output_dir / f"{CHECKPOINT_PREFIX}{artifact('ckpt_resume_suffix')}"
    resume_step = 0
    best_val_loss = float("inf")
    best_epoch = -1
    if resume_path.exists() and not is_recompute(args.cache_policy):
        _ra = torch.load(resume_path, map_location="cpu", weights_only=False)
        ma_head.load_state_dict(_ra["ma_head"])
        optimizer.load_state_dict(_ra["optimizer"])
        scheduler.load_state_dict(_ra["scheduler"])
        scaler.load_state_dict(_ra["scaler"])
        resume_step = int(_ra["step"])
        best_val_loss = float(_ra["best_val_loss"])
        best_epoch = int(_ra["best_epoch"])
        del _ra
        print(f"Resumed from step {resume_step} ({resume_path.name}) — "
              f"best val_loss={best_val_loss:.4f} @ epoch {best_epoch}")

    # === Producer-consumer for train clips (CPU decode → GPU forward) ===
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()
    producer = threading.Thread(
        target=producer_thread,
        args=(cfg, q, stop_event, set(train_keys), resume_step),
        daemon=True,
    )
    producer.start()

    # === Train log files (crash-safe JSONL with fsync + CSV mirror) ===
    # iter15 Phase 6 audit (2026-05-16): unified emit-set with m09a1 (encoder
    # sibling). Renamed training_log.jsonl → loss_log.jsonl + added loss_log.csv
    # with the m09a1 column schema so plot_training_curves / plot_combined_losses
    # render correctly for head cells too. Encoder-frozen fields (loss_jepa,
    # loss_drift, loss_multi_task) emit NaN/0 — plot functions handle gracefully.
    jsonl_path = output_dir / "loss_log.jsonl"   # kept: render_val_plots reads this path
    csv_path = output_dir / artifact("loss_log_csv")        # kept: render_val_plots reads this path
    logw = TrainLogWriter(output_dir, columns=[   # iter17 DRY #32 (jsonl+csv mechanics)
        "step", "epoch", "loss_jepa", "loss_drift", "loss_total",
        "loss_multi_task", "loss_motion_aux",
        "lr", "grad_norm", "throughput", "val_loss"])

    # best_val_loss / best_epoch initialized (and possibly restored) by the
    # resume block above — do NOT re-init here (would clobber the anchor).
    pbar = make_pbar(total=total_steps, initial=resume_step,
                     desc="m09a2 head-only", unit="step")
    step = resume_step
    t_start = time.time()

    try:
        for epoch in range(max_epochs):
            if (epoch + 1) * steps_per_epoch <= resume_step:
                continue   # iter18 resume: epoch fully covered by the anchor
            ma_head.train()
            student.eval()  # always eval — encoder forward only, never trains
            epoch_train_losses = []
            epoch_started = time.time()

            # iter18 resume: the first resumed epoch runs only its remaining steps.
            for _ in range(steps_per_epoch - max(0, resume_step - epoch * steps_per_epoch)):
                try:
                    item = q.get(timeout=600)  # 10-min stall = fatal
                except queue.Empty:
                    print(f"FATAL [m09a2]: producer stalled for 10 min at epoch={epoch} step={step}")
                    sys.exit(1)
                if item is None:
                    # Producer exhausted the stream early — break to val cycle.
                    break
                kind = item[0]
                if kind == "done":
                    break
                if kind != "batch":
                    continue
                _, batch_clips, batch_keys = item[0], item[1], item[2]
                batch_clips = batch_clips.to(device, non_blocking=True)

                try:
                    optimizer.zero_grad(set_to_none=True)
                    # motion_aux: encoder forward (no_grad, frozen) → pooled feats → head → CE+MSE
                    loss_val, per_branch = run_motion_aux_step(
                        student, ma_head, ma_cfg, ma_lookup,
                        batch_clips, batch_keys, scaler=scaler,
                        mp_cfg=mp_cfg, dtype=dtype, device=device,
                    )
                    optimizer.step()
                    scheduler.step()
                    sizer.after_batch_success()
                except torch.cuda.OutOfMemoryError:
                    print(f"[m09a2] OOM at step {step}, sub-batch {sizer.size}")
                    cuda_cleanup()
                    if not sizer.on_oom():
                        print("FATAL [m09a2]: OOM at min sub-batch — cannot continue")
                        sys.exit(1)
                    continue

                epoch_train_losses.append(float(loss_val))
                step += 1
                pbar.update(1)

                # iter18 (2026-06-06): wall-clock resume anchor — head + optimizer
                # state only (~MBs; encoder frozen). Atomic tmp + os.replace.
                if timed_gate.due():
                    atomic_torch_save(resume_path, {
                        "ma_head": ma_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                    })
                    timed_gate.mark()
                    print(f"  [timed-ckpt] {resume_path.name} @ step {step} (epoch {epoch}) "
                          f"— wall-clock anchor", flush=True)
                if step % 20 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # JSONL: m09a1-compatible schema (encoder-frozen → loss_jepa/
                    # loss_drift/loss_multi_task are NaN/0; loss_total = loss_motion_aux).
                    row = {
                        "step": step, "epoch": epoch,
                        "loss_jepa": float("nan"),      # encoder frozen → no JEPA gradient
                        "loss_drift": 0.0,               # encoder frozen → drift identically 0
                        "loss_multi_task": float("nan"), # multi_task off in head-only
                        "loss_motion_aux": float(loss_val),
                        "loss_total": float(loss_val),   # head-only total = motion_aux
                        "lr": cur_lr,
                        "branch": per_branch,
                    }
                    logw.log_jsonl(row)
                    logw.log_csv([step, epoch, "", "0", f"{float(loss_val):.6f}",
                                  "", f"{float(loss_val):.6f}",
                                  f"{cur_lr:.6e}", "", "", ""])
                    log_metrics(wb_run, {"train/loss": float(loss_val),
                                         "train/lr": cur_lr},
                                step=step)

                # iter15 D13 (2026-05-16) + D16 fix (2026-05-16): val cycle every
                # val_interval steps. Mirrors m09a1 cadence. D16: removed the
                # `step % steps_per_epoch == 0` OR-clause that fired EXTRA val cycles
                # when steps_per_epoch was not evenly divisible by saves_per_epoch
                # (e.g. POC m09c2: 35 steps/ep, val_interval=17 → step 35 fired
                # 1 step after step 34, wasting an 11-min probe-trio cycle). The
                # val_interval cadence alone gives exactly saves_per_epoch fires
                # per epoch. For m09a2 POC (steps_per_epoch=34 evenly divisible by
                # saves_per_epoch=2), the OR-clause was redundant; for m09c2 POC
                # (35, not divisible), it caused asymmetric over-firing.
                if step > 0 and step % val_interval == 0:
                    mean_train = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
                    val_loss = compute_val_motion_aux_loss(
                        student, ma_head, ma_cfg, ma_lookup,
                        val_keys, args.val_local_data, cfg, device, dtype,
                        tmp_prefix="m09a2_val_",
                    )
                    elapsed = time.time() - epoch_started
                    is_end_of_epoch = (step % steps_per_epoch == 0)
                    tag = "epoch-end" if is_end_of_epoch else "mid-epoch"
                    print(f"\n[step {step} · epoch {epoch}/{max_epochs} · {tag}] "
                          f"train_loss={mean_train:.4f}  val_loss={val_loss:.4f}  "
                          f"wall={elapsed:.0f}s")
                    row = {"epoch": epoch, "train_loss": mean_train, "val_loss": val_loss,
                           "wall_sec": elapsed, "step": step}
                    logw.log_jsonl(row)
                    logw.log_csv([step, epoch, "", "", "", "", "",
                                  "", "", "", f"{val_loss:.6f}"])
                    log_metrics(wb_run, {"val/loss": val_loss,
                                         "train/epoch_mean_loss": mean_train,
                                         "epoch": epoch}, step=step)

                    # iter15 Phase 6 C1+C2+C3 (2026-05-16): probe_record + trio + drift +
                    # probe_history.jsonl writes for file-parity with m09a1 encoder cell.
                    val_total = val_loss * float(ma_cfg["weight_motion"])
                    probe_record = {
                        "step": step, "epoch": epoch,
                        "val_jepa_loss":       val_total,
                        "val_motion_aux_loss": val_loss,
                        "val_total_loss":      val_total,
                        "val_drift_loss":      0.0,
                        "val_multi_task_loss": 0.0,
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
                        title_prefix=f"m09a head step={step} · ",
                        file_prefix="m09a2",
                    )
                    probe_history.append(probe_record)
                    with open(probe_history_path, "a") as _ph:
                        _ph.write(json.dumps(probe_record) + "\n")
                        _ph.flush()
                        os.fsync(_ph.fileno())

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        export_motion_aux_head(ma_head, output_dir / artifact("motion_aux_head"))
                        print(f"  ✅ new best val_loss={best_val_loss:.4f} (step {step}, epoch {epoch})")

                    # iter15 D17 (2026-05-16): symmetric per-val plot rendering with
                    # m09a1 encoder cell (which renders 5 plots per val: train_loss,
                    # loss_decomposition, block_drift, val_loss_jepa, probe_trajectory
                    # _trio). Head cells previously rendered ONLY block_drift per val
                    # (via track_head_drift_at_val) → 4 plots missing mid-training.
                    # Asymmetric observability between encoder + head sibling modules.
                    # End-of-training plots stay (idempotent, regenerates final view).
                    # FAIL LOUD on render exceptions per CLAUDE.md.
                    best_state_live = {
                        "step": step if best_epoch == epoch else -1,
                        "val_loss_at_best": float(best_val_loss),
                        "top1": float(probe_record.get("probe_top1", -1.0)),   # iter18 2026-06-06: unified best_state key (plots.py:879 strict)
                    }
                    kill_state_live = {"triggered": False, "reason": None}
                    render_val_plots(
                        csv_path=csv_path, jsonl_path=jsonl_path,
                        probe_history=probe_history, output_dir=output_dir,
                        file_prefix="m09a2", label="m09a2 head-only (motion_aux)",
                        color="blue", batch_size=batch_size,
                        curves_title=f"m09a2 head-only · {len(train_keys):,} train × {max_epochs} ep × "
                                     f"BS={batch_size} · step={step}\n",
                        combined_title=f"m09a2 head-only · LR={cfg['optimization']['lr']:.1e} · step={step} · ",
                        kill_title=f"m09a2 head-only · val_total (motion_aux × weight) · step={step}\n",
                        trio_title=f"m09a2 head-only · step={step} · ",
                        best_state=best_state_live, kill_state=kill_state_live,
                    )

        stop_event.set()
        producer.join(timeout=10)
    finally:
        logw.close()

    pbar.close()

    # === Finalization FIRST (iter17 DRY #34 + #33 reorder) — save the ckpt BEFORE the
    # end-of-train plots so a plot-render failure (now FAIL-LOUD via render_val_plots)
    # can't lose a trained model. student_encoder.pt is bit-identical to the Meta init by
    # contract (requires_grad False end-to-end). Divergent payload/summary keys built locally.
    finalize_outputs(
        student=student,
        output_dir=output_dir,
        ckpt_prefix=CHECKPOINT_PREFIX,
        ckpt_payload={
            "student_state_dict": student.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "motion_aux_head_state_dict": ma_head.state_dict(),
            "n_motion_classes": ma_head.n_motion_classes,
            "n_motion_dims": ma_head.n_motion_dims,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "type": "m09a2_head_only",
        },
        summary={
            "mode": mode_key,
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
        },
    )

    # End-of-train plots (iter17 DRY #33: shared render_val_plots, FAIL-LOUD now safe — ckpt
    # already saved above). Encoder-frozen rows have NaN loss_jepa → plot_combined_losses
    # skips them gracefully. probe ALWAYS runs (POC↔FULL parity) → probe_history non-empty.
    # track_head_drift_at_val already emits m09a_block_drift.{png,pdf} per val cycle.
    render_val_plots(
        csv_path=csv_path, jsonl_path=jsonl_path,
        probe_history=probe_history, output_dir=output_dir,
        file_prefix="m09a2", label="m09a2 head-only (motion_aux)",
        color="blue", batch_size=batch_size,
        curves_title=f"m09a2 head-only · {len(train_keys):,} train × {max_epochs} ep × "
                     f"BS={batch_size} · POC:FULL=2:5\n",
        combined_title=f"m09a2 head-only · LR={cfg['optimization']['lr']:.1e} · ",
        kill_title="m09a2 head-only · val_total (motion_aux × weight)\n",
        trio_title="m09a2 head-only · ",
        best_state={"step": -1, "val_loss_at_best": float("inf"), "top1": -1.0},
        kill_state={"triggered": False, "reason": None},
    )

    finish_wandb(wb_run)

    # Force exit: torch.compile + CUDA atexit can deadlock on futex_wait_queue.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 HEAD-ONLY continual SSL (m09a2 — iter15 Phase 2).")
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

    # iter11 DELETE PROTECTION: cache-policy=2 wipes the entire output_dir
    # (gives a clean slate for re-runs); cache-policy=1 keeps it.
    if args.cache_policy == "2":
        wipe_output_dir(cfg["checkpoint"]["output_dir"], args.cache_policy,
                          label="m09a2 head-only output dir")

    train(cfg, args)


if __name__ == "__main__":
    main()
