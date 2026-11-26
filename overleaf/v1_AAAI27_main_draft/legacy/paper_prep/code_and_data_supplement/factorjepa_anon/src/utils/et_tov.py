"""Temporal Order Verification / VCOP encoder-temporal metric (Misra ECCV16 ; Xu CVPR19).

Mechanism: N-way discriminative head over (ordered + N−1 frame-level permutations) of each
clip. Tiny linear head on mean-pooled encoder features for each permutation variant; test-set
top-1 accuracy = per-clip score. surgery≫pretrain hypothesis: TOV-acc higher = encoder retains
temporal-order information.

The orchestrator (m12f) supplies the N permutation indices (deterministic, seeded) so the
permutation set is part of the run config — NOT hardcoded here. Permutation 0 = identity
(orchestrator generates `n_permutations` of shape (T,) and labels 0..n_permutations-1).

Outputs (m12f, --metric tov):
  per_clip_tov.npy   ← per-test-clip top-1 over all permutation variants (avg of N examples)
  aggregate_tov.json ← mean / std / BCa 95% CI / n_test / n_permutations

Gold:
  - Misra et al. "Shuffle and Learn: Unsupervised Learning Using Temporal Order Verification"
    ECCV 2016
  - Xu et al. "Self-supervised Spatiotemporal Learning via Video Clip Order Prediction" CVPR 2019
    github.com/xudejing/video-clip-order-prediction

GPU-VALIDATION REQUIRED post-eval. CPU-checked (py_compile + ruff F,E9) only.
"""
import numpy as np
import torch
import torch.nn as nn

from utils.per_frame_features import forward_per_frame, permute_frames

# iter18 W7 (PLR2004): contracts.
_FEATS_RANK = 2
_MIN_PERMS = 2


@torch.no_grad()
def compute_features(encoder, batch, num_frames, tubelet_size, permutations, *, skip_first=False):
    """Encoder features per permutation in `permutations` (list of (T,) LongTensor;
    permutations[0] MUST be identity = torch.arange(T) — orchestrator enforces). Returns
    stack (n_perm, B, D) — mean-pooled features per permutation variant.
    iter18 #2 (variant-batching): the per-permutation loop is replaced by ONE stacked
    (K·B) forward — same math (encoder is per-sample independent), far better GPU util at
    small B; OOM safety = m12f's halving backoff.
    skip_first=True (iter18 #6 share-features): identity features are served from
    probe_action's cache by the orchestrator → only permutations[1:] are forwarded and the
    returned stack has n_perm−1 rows (caller prepends the cached identity row)."""
    if len(permutations) < _MIN_PERMS:
        raise RuntimeError(
            f"compute_features: need >=2 permutations (identity + >=1 shuffled); got {len(permutations)}")
    T = batch.shape[1]
    expected_identity = torch.arange(T)
    if not torch.equal(permutations[0].cpu().long(), expected_identity):
        raise RuntimeError(
            f"permutations[0] must be identity arange({T}); got {permutations[0].tolist()}")
    todo = permutations[1:] if skip_first else permutations
    B = batch.shape[0]
    x_all = torch.cat([permute_frames(batch, p) for p in todo], dim=0)    # (K·B, T, C, H, W)
    f_all = forward_per_frame(encoder, x_all, num_frames, tubelet_size).mean(dim=1).float().cpu()
    return f_all.view(len(todo), B, -1)  # (n_perm[−1], B, D) — cat order is perm-major


def train_head(train_feats, train_labels, *, embed_dim, n_classes, lr, epochs, weight_decay,
               batch_size, device, seed):
    """Train a linear head (D → n_classes) for permutation-id classification."""
    if train_feats.ndim != _FEATS_RANK or train_feats.shape[1] != embed_dim:
        raise RuntimeError(
            f"train_feats shape {tuple(train_feats.shape)} mismatches embed_dim={embed_dim}")
    if train_labels.ndim != 1 or train_labels.numel() != train_feats.shape[0]:
        raise RuntimeError(
            f"train_labels shape {tuple(train_labels.shape)} != feats N={train_feats.shape[0]}")
    if train_feats.shape[0] == 0:
        raise RuntimeError("train_head: 0 training examples — pipeline failure")
    seen = torch.unique(train_labels)
    if seen.numel() < n_classes:
        raise RuntimeError(
            f"train_head: only {seen.numel()} classes in labels ({seen.tolist()}); expected {n_classes}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    head = nn.Linear(embed_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = train_feats.shape[0]
    feats_d, labels_d = train_feats.to(device), train_labels.to(device)
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(feats_d[idx]), labels_d[idx])
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def eval_head(head, test_feats, test_labels, *, device, batch_size):
    """Per-example top-1 correctness. Orchestrator averages over the n_permutations rows
    belonging to the same clip before writing per_clip_tov.npy."""
    if test_feats.shape[0] != test_labels.shape[0]:
        raise RuntimeError(
            f"test_feats N={test_feats.shape[0]} != test_labels N={test_labels.shape[0]}")
    out = np.empty(test_feats.shape[0], dtype=np.float32)
    feats_d, labels_d = test_feats.to(device), test_labels.to(device)
    for s in range(0, feats_d.shape[0], batch_size):
        pred = head(feats_d[s:s + batch_size]).argmax(dim=1)
        out[s:s + batch_size] = (pred == labels_d[s:s + batch_size]).float().cpu().numpy()
    return out


__all__ = ["compute_features", "train_head", "eval_head"]
