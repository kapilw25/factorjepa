"""On-device row gather — the fix for the CPU-gather-then-transfer anti-pattern (iter19 2026-07-09).

THE ANTI-PATTERN (what hung m12f's TCC pairwise): a chunk loop that does

    for s in range(0, n, chunk):
        out = compute(big[idx_a[s:e]].to("cuda"), big[idx_b[s:e]].to("cuda"))   # ✗

`big[idx[s:e]]` runs an ADVANCED-INDEX gather ON THE HOST first — materializing a
(chunk, ...) tensor in CPU RAM — and only THEN copies it host→device. At m12f the chunk was
70547 rows of (8, 6656) fp32 ≈ 15 GB PER SIDE (30 GB/chunk); single-threaded, 494 chunks, no
progress bar → GPU0 idled at 0 %, one core pinned at 100 %, RSS climbed to 25 GB, and it made no
visible progress for over an hour. /gpu-bottleneck 2026-07-09.

THE FIX (websearch: pytorch forums/issues #116555, #15245 — "gathering is faster on the GPU; the
CPU-gather-then-transfer two-stage handoff is the bottleneck / redundant memory copies over PCIe"):
upload the big source to the device ONCE, then gather each chunk ON-DEVICE with `index_select`
(coalesced, no host allocation, no per-chunk host→device copy). Values are byte-identical —
`index_select(0, idx)` IS advanced indexing `source[idx]`, just executed on `source.device`.

    big = big.to("cuda")                              # upload ONCE, before the loop
    idx_a = torch.as_tensor(idx_a, dtype=torch.long, device=big.device)
    for s in range(0, n, chunk):
        out = compute(device_gather(big, idx_a[s:e]), ...)                      # ✓

USAGE (shared by m12f's TCC pairwise; import where a chunked gather of a large on-device source
recurs — NOT a plain producer→GPU batch transfer, which is `x.to(device)` and needs no gather):
    from utils.gpu_gather import device_gather
"""
import torch


def device_gather(source: torch.Tensor, indices) -> torch.Tensor:
    """Return `source[indices]` along dim 0, gathered ON `source.device` via `index_select`.

    Contract: `source` must ALREADY be on the target device (upload it ONCE before a gather loop) —
    that is the whole point; a CPU `source` re-does the host gather + a per-call host→device copy,
    which is the anti-pattern this exists to kill. `indices` may be a python/numpy sequence or a
    tensor; it is coerced to a long tensor on `source.device` (a no-op if already so). Byte-identical
    to `source[torch.as_tensor(indices)]`."""
    if not torch.is_tensor(source):
        raise TypeError(f"device_gather: source must be a Tensor, got {type(source).__name__}")
    if torch.is_tensor(indices) and indices.dtype == torch.long and indices.device == source.device:
        idx = indices
    else:
        idx = torch.as_tensor(indices, dtype=torch.long, device=source.device)
    if idx.ndim != 1:
        raise ValueError(f"device_gather: indices must be 1-D; got shape {tuple(idx.shape)}")
    return source.index_select(0, idx)


__all__ = ["device_gather"]
