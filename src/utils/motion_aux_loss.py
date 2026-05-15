"""Motion-features auxiliary loss for m09a pretraining (Phase 3, iter13 v12).

Adds a JOINT K-class CE + n_motion_dims MSE supervised gradient on top of V-JEPA's
self-supervised JEPA L1, so the encoder learns motion structure directly.
Targets come from m04d's RAFT optical-flow features:
  - K-class motion class_id (from action_labels.json — same scheme probe_action eval uses)
  - n_motion_dims motion vector (Phase 0: 23-D — global 13 + FG 10) — z-normalized

K is runtime-derived from action_labels.json (typically 8 on walkindia after the
≥34-clip filter; design tolerates 4-16 without code change). Per-dim mean+std
for MSE z-norm are stored as buffers (computed once at init from FULL
motion_features.npy distribution).

Mirrors src/utils/multi_task_loss.py architectural pattern (5 helpers
parallel to merge/build/attach/run/export there).

iter13 v12 (2026-05-06): REPLACES multi_task_probe (15 retrieval tag dims) with
motion_aux as the sole supervised aux loss. multi_task_probe is disabled in
configs/train/pretrain_encoder.yaml — v11 empirically showed retrieval gradients
gave flat motion-flow top1.

Usage in m09a (5 call sites, ~3 LoC each — same pattern as multi_task_probe):
    from utils.motion_aux_loss import (
        merge_motion_aux_config, build_motion_aux_head_from_cfg,
        attach_motion_aux_to_optimizer, run_motion_aux_step,
        export_motion_aux_head,
    )

    # cfg parse:
    merge_motion_aux_config(cfg, args, mode_key)

    # build:
    ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)

    # optim:
    attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg,
                                    base_lr=cfg["optimization"]["lr"])

    # train step (after JEPA loss but BEFORE optimizer.step):
    ma_loss_val, ma_per_branch = run_motion_aux_step(
        student, ma_head, ma_cfg, ma_lookup,
        batch_clips, batch_keys, scaler, mp_cfg, dtype, device)

    # end-of-train:
    export_motion_aux_head(ma_head, output_dir / "motion_aux_head.pt")
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Architecture ─────────────────────────────────────────────────────

class MotionAuxHead(nn.Module):
    """Joint K-class CE + n_motion_dims MSE head on pooled V-JEPA features.

    Architecture:
      pooled (B, D=1664) → trunk (D → hidden_dim, GELU, LayerNorm, dropout)
                         → CE branch (hidden_dim → K_classes)
                         → MSE branch (hidden_dim → n_motion_dims)

    Both branches share the trunk so encoder gradient is consolidated. ~430K params
    at hidden_dim=256, K=8.

    Per-dim mean+std for MSE z-norm are stored as buffers (computed once at init
    from the FULL motion_features.npy distribution).
    """

    def __init__(self, d_encoder: int, n_motion_classes: int, n_motion_dims: int,
                 hidden_dim: int = 256, dropout: float = 0.1,
                 vec_mean: torch.Tensor = None, vec_std: torch.Tensor = None):
        super().__init__()
        self.d_encoder = d_encoder
        self.n_motion_classes = n_motion_classes
        self.n_motion_dims = n_motion_dims
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(d_encoder, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.ce_head = nn.Linear(hidden_dim, n_motion_classes)
        self.mse_head = nn.Linear(hidden_dim, n_motion_dims)
        # Per-dim normalization buffers (z-score on raw RAFT vector).
        if vec_mean is None:
            vec_mean = torch.zeros(n_motion_dims)
        if vec_std is None:
            vec_std = torch.ones(n_motion_dims)
        self.register_buffer("vec_mean", vec_mean.float())
        self.register_buffer("vec_std",  vec_std.float().clamp_min(1e-6))

    def forward(self, pooled_feats: torch.Tensor) -> tuple:
        """pooled_feats: (B, D) → (ce_logits[B, K], mse_pred[B, n_motion_dims])."""
        trunk_out = self.trunk(pooled_feats)
        return self.ce_head(trunk_out), self.mse_head(trunk_out)

    def normalize_target(self, vec_motion: torch.Tensor) -> torch.Tensor:
        """Z-normalize raw RAFT motion vector using stored stats. (B, n_motion_dims) → (B, n_motion_dims). Phase 0: 23-D."""
        return (vec_motion - self.vec_mean) / self.vec_std


# ── Label loader ─────────────────────────────────────────────────────

def load_motion_targets_for_training(motion_features_path: Path,
                                      action_labels_path: Path) -> tuple:
    """Build {clip_key: (class_id, vec_motion)} lookup + per-dim stats.

    Returns (lookup, n_motion_classes, vec_mean, vec_std):
      lookup:           {clip_key: {"class_id": int, "vec_motion": np.ndarray(n_motion_dims,)}}
      n_motion_classes: int — derived from max(class_id) + 1 (typically 8 on walkindia)
      vec_mean:         torch.Tensor(n_motion_dims,) — per-dim mean over ALL clips in motion_features.npy
      vec_std:          torch.Tensor(n_motion_dims,) — per-dim std (+ clamp_min(1e-6))
    """
    motion_features_path = Path(motion_features_path)
    action_labels_path = Path(action_labels_path)
    if not motion_features_path.exists():
        raise FileNotFoundError(
            f"motion_features.npy not found at {motion_features_path}. "
            f"Run `python -u src/m04d_motion_features.py --FULL --local-data <local_data>` "
            f"first (writes to <local_data>/m04d_motion_features/ by default).")
    if not action_labels_path.exists():
        raise FileNotFoundError(
            f"action_labels.json not found at {action_labels_path}. "
            f"Run `probe_action.py --stage labels` first.")

    flow_features = np.load(motion_features_path)            # (N, 23) post-Phase-0
    flow_paths    = np.load(motion_features_path.with_name(
        motion_features_path.stem + ".paths.npy"), allow_pickle=True)
    if flow_features.shape[0] != flow_paths.shape[0]:
        raise ValueError(
            f"motion_features rows ({flow_features.shape[0]}) != "
            f"paths rows ({flow_paths.shape[0]})")
    vec_by_key = {str(k): flow_features[i] for i, k in enumerate(flow_paths)}

    # Per-dim stats over the FULL motion_features distribution (not just the
    # train split — keeps z-norm consistent at val/test/eval time as well).
    vec_mean = torch.from_numpy(flow_features.mean(axis=0))
    vec_std  = torch.from_numpy(flow_features.std(axis=0))

    action_labels = json.loads(action_labels_path.read_text())
    n_motion_classes = max((info["class_id"] for info in action_labels.values()),
                           default=-1) + 1
    if n_motion_classes < 2:
        raise ValueError(
            f"action_labels.json has {n_motion_classes} class(es); need >= 2")

    lookup = {}
    n_no_motion = 0
    for k, info in action_labels.items():
        if k in vec_by_key:
            lookup[k] = {"class_id": int(info["class_id"]),
                         "vec_motion":   vec_by_key[k]}
        else:
            n_no_motion += 1
    if n_no_motion > 0:
        # iter14 recipe-v2 (2026-05-09): FAIL LOUD per CLAUDE.md "WARNING-without-exit
        # banned". Convert silent-disable + WARN to threshold-gated FATAL: small
        # drift (<5%) is OK (m04d may genuinely fail to decode a few clips), but
        # >5% indicates schema mismatch (e.g., m04d ran on a different subset than
        # action_labels.json was generated from).
        drop_pct = n_no_motion / len(action_labels)
        if drop_pct > 0.05:
            print(f"❌ FATAL [motion_aux]: {n_no_motion}/{len(action_labels)} "
                  f"({drop_pct:.1%}) action_labels clips have NO motion_features record — "
                  f"exceeds 5% threshold (likely schema mismatch).", file=sys.stderr)
            print(f"   motion_features: {motion_features_path}", file=sys.stderr)
            print(f"   action_labels:   {action_labels_path}", file=sys.stderr)
            print("   Either re-run m04d on this subset, or regenerate action_labels "
                  "from the same subset m04d processed.", file=sys.stderr)
            sys.exit(3)
        print(f"  [motion_aux] {n_no_motion}/{len(action_labels)} ({drop_pct:.1%}) "
              f"clips dropped (under 5% threshold — accepted as m04d decode failures)")
    if not lookup:
        # Should be unreachable after the threshold check above, but keep as
        # belt-and-suspenders against zero-action-labels edge case.
        print(f"❌ FATAL [motion_aux]: No clip_keys in common between {motion_features_path} "
              f"and {action_labels_path}", file=sys.stderr)
        sys.exit(3)
    return lookup, n_motion_classes, vec_mean, vec_std


# ── Loss computer ────────────────────────────────────────────────────

def compute_motion_aux_loss(pooled_feats: torch.Tensor,
                             head: MotionAuxHead,
                             clip_keys: list,
                             motion_lookup: dict,
                             weight_ce: float = 1.0,
                             weight_mse: float = 1.0,
                             device: torch.device = None) -> tuple:
    """Compute joint CE + MSE loss for motion_aux.

    Args:
        pooled_feats:   (B, D) — encoder features mean-pooled over patch tokens
        head:           MotionAuxHead
        clip_keys:      list[str] — len(clip_keys) == B; clip identifiers in same order as features
        motion_lookup:  {clip_key: {"class_id": int, "vec_motion": np.ndarray(n_motion_dims,)}}
        weight_ce, weight_mse: scalar branch weights (combined inside this fn)
        device:         torch device for label tensors

    Returns: (total_weighted_loss, per_branch_loss_dict)
        total_weighted_loss is a scalar tensor (with grad).
        per_branch_loss_dict is {"ce": float, "mse": float, "n_kept": int} for logging.

    Skips clips without a motion record. If ALL clips in batch lack records,
    returns (zero-tensor, {"ce": 0.0, "mse": 0.0, "n_kept": 0}).
    """
    if device is None:
        device = pooled_feats.device
    B = pooled_feats.shape[0]
    if B != len(clip_keys):
        raise ValueError(
            f"pooled_feats batch dim ({B}) != len(clip_keys) ({len(clip_keys)}); "
            f"producer thread must yield clip_keys aligned with clips.")

    # Mask to clips that have a motion record.
    keep_idx, class_ids, vec_motion_list = [], [], []
    for i, k in enumerate(clip_keys):
        if k in motion_lookup:
            keep_idx.append(i)
            class_ids.append(motion_lookup[k]["class_id"])
            vec_motion_list.append(motion_lookup[k]["vec_motion"])

    if not keep_idx:
        return (torch.zeros((), device=device, dtype=pooled_feats.dtype),
                {"ce": 0.0, "mse": 0.0, "n_kept": 0})

    keep_idx_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
    sub_feats = pooled_feats.index_select(0, keep_idx_t)            # (n_kept, D)
    ce_logits, mse_pred = head(sub_feats)                           # (n_kept, K), (n_kept, 13)

    y_class = torch.tensor(class_ids, dtype=torch.long, device=device)
    y_vec   = torch.from_numpy(np.stack(vec_motion_list)).to(device).to(pooled_feats.dtype)
    y_vec_norm = head.normalize_target(y_vec)                       # (n_kept, 13)

    loss_ce  = F.cross_entropy(ce_logits, y_class)
    loss_mse = F.mse_loss(mse_pred, y_vec_norm)

    total = weight_ce * loss_ce + weight_mse * loss_mse
    return total, {
        "ce":     float(loss_ce.detach().item()),
        "mse":    float(loss_mse.detach().item()),
        "n_kept": int(len(keep_idx)),
    }


# ── 5 helpers (mirror multi_task_loss.py merge/build/attach/run/export) ────

def merge_motion_aux_config(cfg: dict, args, mode_key: str) -> None:
    """Flatten cfg['motion_aux'] per-mode + apply CLI overrides. In-place.

    Schema (from configs/train/pretrain_encoder.yaml):
        motion_aux:
          enabled: {sanity:bool, poc:bool, full:bool}
          motion_features_path: <str>
          action_labels_path:   <str>
          weight_motion: <float>
          weight_ce:     <float>
          weight_mse:    <float>
          head: {hidden_dim:int, dropout:float}
          head_lr_multiplier: <float>

    CLI args (REQUIRED — argparse `required=True` per CLAUDE.md FAIL LOUD):
      --motion-features-path <path>  → cfg['motion_aux']['motion_features_path']
      --probe-action-labels <path>   → cfg['motion_aux']['action_labels_path']

    CLI override flags (optional):
      --no-motion-aux                → cfg['motion_aux']['enabled'] = False

    No-op when cfg has no `motion_aux` block.

    iter15 Phase 5 V2 fix (2026-05-15): both path args are now required-by-
    argparse and yaml hardcoded values were removed (pretrain_encoder.yaml +
    surgery_base.yaml). run_train.sh:276/277/429/430/458/459/503/504 wires
    the mode-gated values. Direct args.X access — no getattr() fallback per
    CLAUDE.md "No `getattr(args, key, default)`".
    """
    if "motion_aux" not in cfg:
        return
    ma_cfg = cfg["motion_aux"]
    if isinstance(ma_cfg.get("enabled"), dict):
        ma_cfg["enabled"] = ma_cfg["enabled"][mode_key]
    ma_cfg["motion_features_path"] = str(args.motion_features_path)
    ma_cfg["action_labels_path"] = str(args.probe_action_labels)
    if getattr(args, "no_motion_aux", False):
        ma_cfg["enabled"] = False


def build_motion_aux_head_from_cfg(cfg: dict, device) -> tuple:
    """Construct MotionAuxHead + load motion targets.

    Returns (ma_head, ma_lookup, ma_cfg) — all None (except ma_cfg) when disabled
    OR motion_features.npy / action_labels.json missing. Caller checks
    `ma_head is not None`.

    Side effect: prints a one-line status. Mutates cfg['motion_aux']['enabled']
    to False if dependencies missing — same graceful-disable pattern as
    multi_task_loss.build_multi_task_head_from_cfg.
    """
    # iter14 recipe-v2 (2026-05-09): FAIL LOUD per CLAUDE.md "no DEFAULT, no
    # silent disable — silent error → research paper rejection". Previous WARN
    # +disable swallowed the rm-rf-induced motion_aux loss in Cell D v1, making
    # an apples-to-oranges comparison. If user genuinely wants motion_aux off,
    # set yaml `motion_aux.enabled: false` explicitly. cfg["motion_aux"] (not
    # cfg.get) so a missing yaml block crashes instead of returning silently.
    ma_cfg = cfg["motion_aux"] if "motion_aux" in cfg else None
    if ma_cfg is None or not ma_cfg["enabled"]:
        return None, None, ma_cfg or {}
    motion_features_path = Path(ma_cfg["motion_features_path"])
    action_labels_path   = Path(ma_cfg["action_labels_path"])
    if not motion_features_path.exists() or not action_labels_path.exists():
        print("❌ FATAL [motion_aux]: required prereq files missing", file=sys.stderr)
        print(f"   motion_features_path: {motion_features_path}  exists={motion_features_path.exists()}", file=sys.stderr)
        print(f"   action_labels_path:   {action_labels_path}  exists={action_labels_path.exists()}", file=sys.stderr)
        print("   To proceed without motion_aux, set yaml `motion_aux.enabled: false` explicitly.", file=sys.stderr)
        print("   To regenerate labels: python -u src/probe_action.py --<MODE> --stage labels ...", file=sys.stderr)
        sys.exit(3)

    lookup, n_classes, vec_mean, vec_std = load_motion_targets_for_training(
        motion_features_path, action_labels_path)
    d_encoder = cfg["model"]["embed_dim"]
    # iter15 Phase 0 (2026-05-14): n_motion_dims bumped 13 → 23 to match m04d's
    # extended FG (camera-subtracted) feature vector. vec_mean / vec_std auto-
    # resize via flow_features.mean(axis=0) in load_motion_targets_for_training
    # (already shape-agnostic). 23-D MSE head adds 23×256 = 5.9 K extra params.
    n_motion_dims = int(vec_mean.numel())
    if n_motion_dims < 23:
        sys.exit(
            f"FATAL [motion_aux]: motion_features at {motion_features_path} is "
            f"{n_motion_dims}-D; Phase 0 requires 23-D (adds FG fg_mean_mag at "
            f"vec[13]). Rerun: CACHE_POLICY_ALL=2 python -u "
            f"src/m04d_motion_features.py --FULL --subset <subset.json> "
            f"--local-data <local_data> (writes to "
            f"<local_data>/m04d_motion_features/ by default)"
        )
    ma_head = MotionAuxHead(
        d_encoder=d_encoder, n_motion_classes=n_classes, n_motion_dims=n_motion_dims,
        hidden_dim=ma_cfg["head"]["hidden_dim"],
        dropout=ma_cfg["head"]["dropout"],
        vec_mean=vec_mean, vec_std=vec_std,
    ).to(device)
    ma_head.train()
    n_params = sum(p.numel() for p in ma_head.parameters())
    print(f"  [motion_aux] enabled: {n_classes} classes, {n_motion_dims}-D vec, "
          f"{len(lookup):,} clips with targets, {n_params:,} head params, "
          f"weight_motion={ma_cfg['weight_motion']}, "
          f"weight_ce={ma_cfg['weight_ce']}, weight_mse={ma_cfg['weight_mse']}")
    return ma_head, lookup, ma_cfg


def attach_motion_aux_to_optimizer(optimizer, ma_head: "MotionAuxHead | None",
                                    ma_cfg: dict, base_lr: float) -> None:
    """Add ma_head's params as a separate param group on `optimizer`.

    No-op when ma_head is None. Mirrors multi_task_loss.attach_head_to_optimizer
    (separate group lets head LR multiplier kick in independently of encoder LR).
    """
    if ma_head is None:
        return
    optimizer.add_param_group({
        "params": list(ma_head.parameters()),
        "lr":     base_lr * ma_cfg["head_lr_multiplier"],
        "weight_decay": 0.0,
        "name":   "motion_aux_head",
    })


def run_motion_aux_step(student, ma_head: "MotionAuxHead | None",
                         ma_cfg: dict, ma_lookup,
                         batch_clips: torch.Tensor, batch_keys: list,
                         scaler, mp_cfg: dict, dtype, device) -> tuple:
    """One motion_aux forward+backward over the macro-batch.

    Mirrors run_multi_task_step: toggles return_hierarchical OFF for this
    forward only (V-JEPA emits hierarchical output by default), runs head,
    computes loss, scales by weight_motion, backward into the same param.grad
    buffer as JEPA grads (single optimizer.step consumes both).

    Returns (ma_loss_val: float, ma_per_branch: dict).
      - (0.0, {}) when ma_head is None OR batch_keys is empty.
      - Re-raises torch.cuda.OutOfMemoryError so the caller's per-step
        OOM handler (m09a's `continue`, m09c's sub-batch shrink) wins.
    """
    if ma_head is None or not batch_keys:
        return 0.0, {}
    # V-JEPA 2.1 ViT has return_hierarchical=True at training time (m09a1_pretrain_encoder.py:351),
    # so student(x) returns (B, N, 4*D) — 4 deep-supervision layers concatenated along
    # the feature dim. The motion_aux head expects (B, D), so we toggle hierarchical OFF
    # for this forward only. Mirrors the pattern in multi_task_loss.run_multi_task_step
    # (321-323). try/finally guarantees restoration even on OOM.
    had_hier = getattr(student, "return_hierarchical", None)
    if had_hier is True:
        student.return_hierarchical = False
    try:
        with torch.amp.autocast("cuda", enabled=mp_cfg["enabled"], dtype=dtype):
            feats = student(batch_clips)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            pooled = feats.mean(dim=1)                       # (B, D)
            ma_loss, ma_per_branch = compute_motion_aux_loss(
                pooled, ma_head, batch_keys, ma_lookup,
                weight_ce=ma_cfg["weight_ce"],
                weight_mse=ma_cfg["weight_mse"],
                device=device)
            ma_loss_scaled = ma_loss * float(ma_cfg["weight_motion"])
        if ma_loss_scaled.requires_grad and float(ma_loss_scaled.detach().item()) > 0.0:
            scaler.scale(ma_loss_scaled).backward()
            return float(ma_loss.detach().item()), ma_per_branch
        return 0.0, ma_per_branch
    finally:
        if had_hier is not None:
            student.return_hierarchical = had_hier


def export_motion_aux_head(ma_head: "MotionAuxHead | None", path: Path) -> None:
    """Write motion_aux_head.pt next to student_encoder.pt. No-op when None.

    Stored fields are sufficient for downstream eval to either re-use the head
    for warm-start or ignore it: state_dict + n_motion_classes + n_motion_dims
    + d_encoder + hidden_dim.
    """
    if ma_head is None:
        return
    torch.save({
        "state_dict":       ma_head.state_dict(),
        "n_motion_classes": ma_head.n_motion_classes,
        "n_motion_dims":    ma_head.n_motion_dims,
        "d_encoder":        ma_head.d_encoder,
        "hidden_dim":       ma_head.hidden_dim,
    }, path)
    print(f"Exported motion_aux head: {path}")
