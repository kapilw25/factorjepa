"""Multi-task probe head + loss for m09a/m09c training.

Adds a supervised classification gradient to the SSL training objective so
that the encoder gets DIRECT signal toward the eval metric (top-1 acc on
14 single-label dims + sample-F1 on 2 multi-label dims). Without this,
pure JEPA L1 has no mechanism to outperform frozen on probe metrics
(per the user's iter11/12 critique — earlier retrieval gates were the
wrong target, but multi-task with the RIGHT eval metric is the legitimate
lever).

Usage in m09a/m09c:
    from utils.multi_task_loss import (
        MultiTaskProbeHead, load_taxonomy_labels_for_training,
        compute_multi_task_probe_loss,
    )

    # At build_model:
    if cfg["multi_task_probe"]["enabled"]:
        labels_by_clip, dims_spec = load_taxonomy_labels_for_training(
            cfg["multi_task_probe"]["labels_path"])
        mt_head = MultiTaskProbeHead(d_encoder=embed_dim, dims_spec=dims_spec).to(device)

    # At each train step (after encoder forward):
    if mt_head is not None:
        pooled_feats = student_features.mean(dim=1)        # (B, D)
        mt_loss, mt_per_dim = compute_multi_task_probe_loss(
            pooled_feats, mt_head, clip_keys,
            labels_by_clip, dims_spec,
            weight_per_dim=cfg["multi_task_probe"]["weight_per_dim"])
        total_loss = α·jepa_loss + β·mt_loss + drift_loss
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Probe head architecture ──────────────────────────────────────────

class MultiTaskProbeHead(nn.Module):
    """One Linear head per dim, sharing a pooled encoder feature.

    Why Linear (not AttentiveClassifier)? AttentiveClassifier expects token
    sequences (B, N, D). At training time we work with mean-pooled features
    (B, D) for cheap loss computation — no second attention block needed.
    The eval-time probe (probe_taxonomy.py) uses AttentiveClassifier on
    the same features for the FINAL paper number; this train-time head is
    a lightweight signal for gradient flow only.

    For each single-label dim: nn.Linear(D, n_classes) → CrossEntropy
    For each multi-label dim:  nn.Linear(D, n_classes) → BCEWithLogits
    """

    def __init__(self, d_encoder: int, dims_spec: dict):
        super().__init__()
        self.d_encoder = d_encoder
        self.dims_spec = dims_spec
        # nn.ModuleDict keyed by dim_name. Order preserved across runs.
        self.heads = nn.ModuleDict({
            dim_name: nn.Linear(d_encoder, spec["n_classes"], bias=True)
            for dim_name, spec in dims_spec.items()
        })

    def forward(self, pooled_feats: torch.Tensor) -> dict:
        """pooled_feats: (B, D) → {dim_name: logits[B, n_classes_d]}"""
        return {dim_name: head(pooled_feats) for dim_name, head in self.heads.items()}


# ── Label loader ─────────────────────────────────────────────────────

def load_taxonomy_labels_for_training(labels_path: Path) -> tuple:
    """Load taxonomy_labels.json (produced by probe_taxonomy --stage labels).

    Returns (labels_by_clip, dims_spec):
      labels_by_clip: {clip_key: {dim_name: int (single) | list[int] (multi-hot)}}
      dims_spec:      {dim_name: {type, values, n_classes, default}}
    """
    p = Path(labels_path)
    if not p.exists():
        raise FileNotFoundError(
            f"taxonomy_labels.json not found at {p}. Run "
            f"`probe_taxonomy.py --stage labels` first.")
    data = json.loads(p.read_text())
    return data["labels"], data["dims"]


# ── Loss computer ────────────────────────────────────────────────────

def compute_multi_task_probe_loss(pooled_feats: torch.Tensor,
                                    head: MultiTaskProbeHead,
                                    clip_keys: list,
                                    labels_by_clip: dict,
                                    dims_spec: dict,
                                    weight_per_dim=None,
                                    device: torch.device = None) -> tuple:
    """Compute weighted-sum multi-task loss over all dims.

    Args:
        pooled_feats: (B, D) — encoder features mean-pooled over patch tokens
        head: MultiTaskProbeHead
        clip_keys: list[str] — len(clip_keys) == B; clip identifiers in same order as features
        labels_by_clip: from load_taxonomy_labels_for_training
        dims_spec:    from load_taxonomy_labels_for_training
        weight_per_dim: None = equal (1/n_dims) or dict {dim_name: float}
        device: torch device for label tensors

    Returns: (total_weighted_loss, per_dim_loss_dict)
        total_weighted_loss is a scalar tensor (with grad).
        per_dim_loss_dict is {dim_name: float (detached scalar)} for logging.

    iter14 recipe-v2 (2026-05-09): FAIL LOUD on label-file mismatch per CLAUDE.md.
    Per-clip sparseness (clip in labels_by_clip but missing some dim_names) is
    legitimate multi-label sparsity. But per-clip ABSENCE (clip_key not in
    labels_by_clip at all) is a label-file mismatch — previously silently
    skipped, now FATAL.
    """
    if device is None:
        device = pooled_feats.device
    B = pooled_feats.shape[0]
    if B != len(clip_keys):
        raise ValueError(
            f"pooled_feats batch dim ({B}) != len(clip_keys) ({len(clip_keys)}); "
            f"producer thread must yield clip_keys aligned with clips.")

    # iter14 FAIL LOUD: clip-key absence from labels_by_clip is a config bug
    # (labels file generated from different subset than current train pool).
    unlabeled = [k for k in clip_keys if k not in labels_by_clip]
    if unlabeled:
        print(f"❌ FATAL [multi-task]: {len(unlabeled)}/{B} clips missing from labels_by_clip "
              f"(label-file mismatch, not multi-label sparseness)", file=sys.stderr)
        print(f"   Sample missing keys: {unlabeled[:3]}", file=sys.stderr)
        print("   Regenerate labels: python -u src/probe_taxonomy.py --<MODE> --stage labels ...", file=sys.stderr)
        sys.exit(3)

    logits_dict = head(pooled_feats)               # {dim_name: (B, n_classes_d)}
    n_dims = len(dims_spec)
    # weight_per_dim resolution (matches YAML schema in base_optimization.yaml:
    # multi_task_probe.weight_per_dim):
    #   None / null    → equal split: each dim gets 1/n_dims
    #   int / float    → scalar broadcast: same weight for every dim
    #   dict           → per-dim override (missing dims fall back to 1/n_dims)
    #   "kendall_uw"   → learnable log-variance per task (Kendall CVPR 2018);
    #                    requires nn.Parameter + optimizer wiring; NOT yet
    #                    implemented (would need a separate Module that owns
    #                    log_sigma_d; out of scope for v1 — see iter12 attempt
    #                    + roadmap in plan_training.md if/when revisited).
    if weight_per_dim is None:
        weight_per_dim = {d: 1.0 / n_dims for d in dims_spec}
    elif isinstance(weight_per_dim, (int, float)):
        weight_per_dim = {d: float(weight_per_dim) for d in dims_spec}
    elif isinstance(weight_per_dim, str):
        if weight_per_dim == "kendall_uw":
            raise NotImplementedError(
                "weight_per_dim='kendall_uw' not yet implemented. Use null (equal "
                "weighting) or a per-dim dict for now. Kendall UW requires learnable "
                "log_sigma_d per dim — separate Module + optimizer integration. "
                "iter12 v3 attempted UncertaintyWeights for InfoNCE/TCC mix; "
                "see utils.training.UncertaintyWeights for prior art.")
        raise ValueError(f"weight_per_dim string {weight_per_dim!r} unrecognized "
                         f"(only 'kendall_uw' supported, currently NotImplemented)")
    elif isinstance(weight_per_dim, dict):
        # Validate dict keys match dims_spec; warn on unknowns.
        unknown = set(weight_per_dim.keys()) - set(dims_spec.keys())
        if unknown:
            print(f"  [multi_task_loss] WARN: weight_per_dim has unknown dim(s) "
                  f"{sorted(unknown)} — ignored")
        # Missing dims fall back to equal split (handled by .get() below).
    else:
        raise TypeError(f"weight_per_dim type {type(weight_per_dim).__name__} unsupported "
                        f"(expected None, int/float, dict, or 'kendall_uw' string)")

    total = torch.zeros((), device=device, dtype=pooled_feats.dtype)
    per_dim_loss = {}
    for dim_name, spec in dims_spec.items():
        # Build per-dim label tensor from clip_keys, masking out unlabeled clips.
        # iter14 (2026-05-09): clip-key absence already FATAL'd above; only need
        # to handle per-dim sparseness here (clip in dict but missing this dim).
        keep_idx, label_list = [], []
        for i, k in enumerate(clip_keys):
            if dim_name in labels_by_clip[k]:
                keep_idx.append(i)
                label_list.append(labels_by_clip[k][dim_name])
        if not keep_idx:
            per_dim_loss[dim_name] = 0.0
            continue

        keep_idx_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
        sub_logits = logits_dict[dim_name].index_select(0, keep_idx_t)   # (n_kept, n_classes)

        if spec["type"] == "single":
            y = torch.tensor(label_list, dtype=torch.long, device=device)
            loss = F.cross_entropy(sub_logits, y)
        else:                                                               # multi-label
            y = torch.tensor(label_list, dtype=sub_logits.dtype, device=device)
            loss = F.binary_cross_entropy_with_logits(sub_logits, y)

        w = weight_per_dim.get(dim_name, 1.0 / n_dims)
        total = total + w * loss
        per_dim_loss[dim_name] = float(loss.detach().item())

    return total, per_dim_loss


# ── Optimizer-param helper (heads need their own param group) ────────

def get_probe_head_param_groups(head: MultiTaskProbeHead, base_lr: float,
                                 lr_multiplier: float = 10.0,
                                 weight_decay: float = 0.0) -> list:
    """Return AdamW param groups for the multi-task head.

    Heads are tiny (~1M params total for 16 dims × ~3-15 classes × 1664-D).
    Higher LR (10× base) is standard for probe heads — the ENCODER is the
    expensive surface; the heads should converge quickly. weight_decay=0
    avoids penalizing the linear classifier weights (standard practice).
    """
    return [{
        "params": list(head.parameters()),
        "lr":     base_lr * lr_multiplier,
        "weight_decay": weight_decay,
        "name":   "multi_task_probe_head",
    }]


# ── m09a/m09c integration helpers ─────────────────────────────────────
# Five helpers below replace the duplicated wiring that landed in #62-65
# (m09a_pretrain.py + m09c_surgery.py). Each m09 call site shrinks from
# ~22 LoC to ~3 LoC. All helpers are technique-agnostic — they don't care
# whether they're called from m09a (vanilla pretrain) or m09c (surgery).

def merge_multi_task_config(cfg: dict, args, mode_key: str) -> None:
    """Flatten cfg['multi_task_probe'] per-mode + apply CLI overrides. In-place.

    YAML schema:
        multi_task_probe:
          enabled: {sanity: bool, poc: bool, full: bool}
          labels_path: <str>
          weight_jepa: <float>
          weight_probe: <float>
          weight_per_dim: null | <float> | <dict> | "kendall_uw"
          head_lr_multiplier: <float>

    Only `enabled` is per-mode; other fields are scalars. CLI overrides:
      --taxonomy-labels-json <path>  → cfg['multi_task_probe']['labels_path']
      --no-multi-task                → cfg['multi_task_probe']['enabled'] = False

    No-op when cfg has no `multi_task_probe` block (ch10/legacy configs).
    """
    if "multi_task_probe" not in cfg:
        return
    mt_cfg = cfg["multi_task_probe"]
    if isinstance(mt_cfg.get("enabled"), dict):
        mt_cfg["enabled"] = mt_cfg["enabled"][mode_key]
    if getattr(args, "taxonomy_labels_json", None):
        mt_cfg["labels_path"] = args.taxonomy_labels_json
    if getattr(args, "no_multi_task", False):
        mt_cfg["enabled"] = False


def build_multi_task_head_from_cfg(cfg: dict, device) -> tuple:
    """Construct MultiTaskProbeHead from cfg['multi_task_probe'] + load labels.

    Returns (mt_head, mt_labels_by_clip, mt_dims_spec, mt_cfg) — all None
    (except mt_cfg which is {} or the cfg block) when multi-task is disabled
    OR labels file is missing. Caller checks `mt_head is not None`.

    Side effect: prints a one-line status. Mutates cfg['multi_task_probe']['enabled']
    to False if the labels file isn't on disk (silent disable + WARN — same
    semantics as before refactor, so the train loop's `if mt_head is not None`
    gate still gives correct skip behavior).
    """
    # iter14 recipe-v2 (2026-05-09): FAIL LOUD per CLAUDE.md (mirrors
    # build_motion_aux_head_from_cfg). Silent WARN+disable was the bug class
    # behind iter14 Cell D v1's apples-to-oranges comparison.
    mt_cfg = cfg["multi_task_probe"] if "multi_task_probe" in cfg else None
    if mt_cfg is None or not mt_cfg["enabled"]:
        return None, None, None, mt_cfg or {}
    mt_labels_path = Path(mt_cfg["labels_path"])
    if not mt_labels_path.exists():
        print(f"❌ FATAL [multi-task]: labels file missing: {mt_labels_path}", file=sys.stderr)
        print("   To proceed without multi-task probe, set yaml `multi_task_probe.enabled: false` explicitly.", file=sys.stderr)
        print("   To regenerate: python -u src/probe_taxonomy.py --<MODE> --stage labels ...", file=sys.stderr)
        sys.exit(3)
    mt_labels_by_clip, mt_dims_spec = load_taxonomy_labels_for_training(mt_labels_path)
    d_encoder = cfg["model"]["embed_dim"]
    mt_head = MultiTaskProbeHead(d_encoder=d_encoder, dims_spec=mt_dims_spec).to(device)
    mt_head.train()
    n_head_params = sum(p.numel() for p in mt_head.parameters())
    print(f"  [multi-task] enabled: {len(mt_dims_spec)} dims, "
          f"{len(mt_labels_by_clip):,} clips labeled, "
          f"{n_head_params:,} head params, "
          f"weight_jepa={mt_cfg['weight_jepa']}, weight_probe={mt_cfg['weight_probe']}")
    return mt_head, mt_labels_by_clip, mt_dims_spec, mt_cfg


def attach_head_to_optimizer(optimizer, mt_head: "MultiTaskProbeHead | None",
                              mt_cfg: dict, base_lr: float) -> None:
    """Add mt_head's params as a separate param group on `optimizer`.

    No-op when mt_head is None. Used by both m09a (once after build_optimizer)
    and m09c (after every per-stage build_optimizer rebuild — surgery's
    progressive unfreezing recreates the optimizer each stage, so the head
    params must be re-attached every time).
    """
    if mt_head is None:
        return
    head_groups = get_probe_head_param_groups(
        mt_head, base_lr=base_lr,
        lr_multiplier=mt_cfg["head_lr_multiplier"],
        weight_decay=0.0)
    for g in head_groups:
        optimizer.add_param_group(g)


def run_multi_task_step(student, mt_head: "MultiTaskProbeHead | None",
                         mt_cfg: dict, mt_labels_by_clip, mt_dims_spec,
                         batch_clips: torch.Tensor, batch_keys: list,
                         scaler, mp_cfg: dict, dtype, device) -> tuple:
    """One multi-task forward+backward pass over the macro-batch.

    Runs ONCE per macro-batch on the full batch_clips (mean-pooled patch
    tokens, no masks — clean encoder feature). Backward accumulates onto
    the same param.grad buffer as the JEPA grads, so a single
    optimizer.step() consumes both losses.

    Returns (mt_loss_val: float, mt_per_dim: dict).
      - (0.0, {}) when mt_head is None OR batch_keys is empty.
      - Re-raises torch.cuda.OutOfMemoryError so the caller's per-step
        OOM handler (m09a's `continue`, m09c's sub-batch shrink) wins.
    """
    if mt_head is None or not batch_keys:
        return 0.0, {}
    # V-JEPA 2.1 ViT has return_hierarchical=True at training time (m09a_pretrain.py:351),
    # so student(x) returns (B, N, 4*D) — 4 deep-supervision layers concatenated along the
    # feature dim. The multi-task head expects (B, D), so we toggle hierarchical OFF for
    # this forward only. Mirrors the toggle-and-restore pattern used 3× in
    # utils/training.py (1529-1562, 1639-1667, 1738-1810). try/finally guarantees
    # restoration even on OOM. errors_N_fixes.md (m09a multi-task hierarchical-shape).
    had_hier = getattr(student, "return_hierarchical", None)
    if had_hier is True:
        student.return_hierarchical = False
    try:
        with torch.amp.autocast("cuda", enabled=mp_cfg["enabled"], dtype=dtype):
            feats = student(batch_clips)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            pooled = feats.mean(dim=1)              # (B, D)
            mt_loss, mt_per_dim = compute_multi_task_probe_loss(
                pooled, mt_head, batch_keys,
                mt_labels_by_clip, mt_dims_spec,
                weight_per_dim=mt_cfg["weight_per_dim"],
                device=device)
            mt_loss_scaled = mt_loss * float(mt_cfg["weight_probe"])
        if mt_loss_scaled.requires_grad and float(mt_loss_scaled.detach().item()) > 0.0:
            scaler.scale(mt_loss_scaled).backward()
            return float(mt_loss.detach().item()), mt_per_dim
        return 0.0, mt_per_dim
    finally:
        if had_hier is not None:
            student.return_hierarchical = had_hier


def export_multi_task_head(mt_head: "MultiTaskProbeHead | None",
                            mt_dims_spec: dict, d_encoder: int, path: Path) -> None:
    """Write multi_task_head.pt next to student_encoder.pt. No-op when head is None.

    Stored fields are sufficient for downstream eval to either re-use the
    head for warm-start or ignore it: state_dict + dims_spec + d_encoder.
    """
    if mt_head is None:
        return
    torch.save({
        "state_dict": mt_head.state_dict(),
        "dims_spec":  mt_dims_spec,
        "d_encoder":  d_encoder,
    }, path)
    print(f"Exported multi-task probe head: {path}")
