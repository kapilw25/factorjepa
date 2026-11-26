"""Pace / playback-speed encoder-temporal metric (Wang et al. ECCV20).

Mechanism: N-way classifier over playback rates. Source clips are decoded with OVERSAMPLE
frames (T_src = num_frames × max_stride) so every requested stride fits without cyclic wrap;
each stride yields one (B, D) mean-pooled feature vector → tiny linear head classifies rate.
Per-clip score on test = top-1 over the N stride variants (avg of N examples per clip).

surgery≫pretrain hypothesis: Pace-acc higher = encoder is sensitive to temporal scale (rather
than treating sped-up vs slow-mo as the same content). RSPNet / SpeedNet confirm this signal
discriminates between motion-trained vs appearance-trained encoders.

Outputs (m12f, --metric pace):
  per_clip_pace.npy   ← per-test-clip top-1 (avg over strides)
  aggregate_pace.json ← mean / std / BCa 95% CI / n_test / strides

Gold:
  - Wang et al. "Self-supervised Video Representation Learning by Pace Prediction" ECCV 2020
    github.com/laura-wang/video-pace
  - Benaim et al. "SpeedNet" CVPR 2020
  - Chen et al. "RSPNet: Relative Speed Perception for Unsupervised Video Representation Learning"
    AAAI 2021

GPU-VALIDATION REQUIRED post-eval. The decode-oversample path requires the orchestrator to
pass T_src-frame source clips (not the default num_frames-decoded clips) — this is configured
via --pace-source-frames in m12f at run time. CPU-checked (py_compile + ruff F,E9) only.
"""
import numpy as np
import torch
import torch.nn as nn

from utils.per_frame_features import forward_per_frame, stride_indices

# iter18 W7 (PLR2004): contracts.
_VIDEO_RANK = 5
_FEATS_RANK = 2
_MIN_STRIDES = 2


@torch.no_grad()
def compute_features(encoder, source_batch, num_frames, tubelet_size, strides):
    """source_batch shape (B, T_src, C, H, W) where T_src = num_frames × max(strides) ideally
    (orchestrator decodes oversample). Returns stack (n_strides, B, D) — mean-pooled features
    per stride. strides MUST be a sorted positive int list/tensor (orchestrator enforces)."""
    if source_batch.ndim != _VIDEO_RANK:
        raise RuntimeError(
            f"source_batch shape must be 5-D (B,T_src,C,H,W); got {tuple(source_batch.shape)}")
    if len(strides) < _MIN_STRIDES:
        raise RuntimeError(f"strides must have >=2 distinct rates; got {list(strides)}")
    T_src = source_batch.shape[1]
    # iter18 #2 (variant-batching): one stacked (K·B) forward replaces the per-stride loop —
    # same math (per-sample independent), better GPU util at small B; OOM = m12f halving backoff.
    B = source_batch.shape[0]
    x_all = torch.cat(
        [source_batch.index_select(1, stride_indices(num_frames, int(s), T_src)
                                   .to(source_batch.device)) for s in strides],
        dim=0).contiguous()                                               # (K·B, T, C, H, W)
    f_all = forward_per_frame(encoder, x_all, num_frames, tubelet_size).mean(dim=1).float().cpu()
    return f_all.view(len(strides), B, -1)  # (n_strides, B, D) — cat order is stride-major


def train_head(train_feats, train_labels, *, embed_dim, n_classes, lr, epochs, weight_decay,
               batch_size, device, seed):
    """Linear head (D → n_strides) for playback-rate classification."""
    if train_feats.ndim != _FEATS_RANK or train_feats.shape[1] != embed_dim:
        raise RuntimeError(
            f"train_feats shape {tuple(train_feats.shape)} mismatches embed_dim={embed_dim}")
    if train_labels.numel() != train_feats.shape[0]:
        raise RuntimeError(
            f"train_labels N={train_labels.numel()} != feats N={train_feats.shape[0]}")
    if train_feats.shape[0] == 0:
        raise RuntimeError("train_head: 0 training examples — pipeline failure")
    seen = torch.unique(train_labels)
    if seen.numel() < n_classes:
        raise RuntimeError(
            f"train_head: only {seen.numel()} stride classes present ({seen.tolist()}); expected {n_classes}")
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
    """Per-example top-1 correctness over stride classes."""
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
