"""On-demand factor generation primitives — streaming variant of m11 disk-write path.

Generates D_L / D_A factors from (raw_mp4_bytes, m10_mask_npz) pairs at training
time instead of materializing .npy files to disk. Unlocks 10K → 50K → 115K scale
ladder on a single 500 GB vast.ai instance (vs 500 GB → 3 TB → 5 TB without).

Bitwise-parity contract with m11: `stream_factor(mp4_bytes, mask_npz_path, 'D_L',
cfg, ...)` returns the SAME (T, H, W, C) uint8 array that
`np.load(m11_outputs/D_L/<clip>.npy)` yields for the same inputs. Tested in
scripts/tests_streaming/test_parity.py.

This module is pure: no globals, no side effects, no RNG. Deterministic given
(mp4_bytes, mask_npz_path, factor_type, factor_cfg).
"""
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))
from m11_factor_datasets import (
    make_layout_only,
    make_agent_only,
    make_interaction_tubes_from_bboxes,
    make_interaction_tubes_from_centroids,
)
from utils.video_io import decode_video_bytes


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _align_mask(
    mask: np.ndarray,
    target_T: int,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Temporal + spatial align mask to frame grid — mirrors m11:652-664 EXACTLY.

    Args:
        mask: (T_mask, H_mask, W_mask) bool
        target_T: desired temporal length (matches frames.shape[0])
        target_hw: (H, W) desired spatial shape (matches frames.shape[1:3])
    Returns: (target_T, H, W) bool — same dtype, same alignment logic as m11.
    """
    T_mask = mask.shape[0]
    if T_mask != target_T:
        idx = np.linspace(0, T_mask - 1, target_T, dtype=int)
        mask = mask[idx]
    if mask.shape[1:] != target_hw:
        H, W = target_hw
        aligned = np.zeros((target_T, H, W), dtype=bool)
        for t in range(target_T):
            aligned[t] = np.array(
                PILImage.fromarray(mask[t]).resize((W, H), PILImage.NEAREST)
            )
        mask = aligned
    return mask


def stream_factor(
    mp4_bytes: bytes,
    mask_npz_path: Path,
    factor_type: str,
    factor_cfg: dict,
    num_frames: int,
    tmp_dir: str,
    clip_key: str,
) -> np.ndarray:
    """Generate D_L or D_A factor array on-demand. Returns (T, H, W, C) uint8.

    Args:
        mp4_bytes: raw MP4 bytes (same source as m11 reads via iter_clips_parallel)
        mask_npz_path: path to m10 .npz with 'agent_mask', 'layout_mask' keys
        factor_type: "D_L" (layout-only, agents blurred) or "D_A" (agent-only, BG matte)
        factor_cfg: flattened m11 config dict with keys:
            layout_method, blur_sigma, feather_sigma      # for D_L
            agent_method, matte_factor, feather_sigma      # for D_A
        num_frames: temporal count (default 16, matches m10/m11)
        tmp_dir: per-worker scratch dir for MP4 decode
        clip_key: for error context + decode_video_bytes tmp name

    Output shape/dtype is bitwise-identical to what m11 writes at
    src/m11_factor_datasets.py:675 (D_L) or :681 (D_A) given the same inputs.
    """
    if factor_type not in ("D_L", "D_A"):
        # D_I is handled by the separate `stream_interaction_tubes()` entrypoint
        # below (returns List[np.ndarray]) — not routed through this scalar function.
        raise ValueError(
            f"stream_factor: factor_type must be 'D_L' or 'D_A', got {factor_type!r}. "
            f"For D_I, call stream_interaction_tubes() instead.")

    frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames)
    if frames_tensor is None:
        raise RuntimeError(
            f"stream_factor: decode_video_bytes returned None for clip_key={clip_key!r}. "
            f"Upstream MP4 is corrupt or unreadable.")
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    data = np.load(mask_npz_path)
    if factor_type == "D_L":
        agent_mask = data["agent_mask"]
        agent_mask = _align_mask(
            agent_mask,
            target_T=frames_np.shape[0],
            target_hw=(frames_np.shape[1], frames_np.shape[2]),
        )
        return make_layout_only(
            frames_np, agent_mask,
            method=factor_cfg["layout_method"],
            blur_sigma=factor_cfg["blur_sigma"],
            feather_sigma=factor_cfg["feather_sigma"],
        )

    layout_mask = data["layout_mask"]
    layout_mask = _align_mask(
        layout_mask,
        target_T=frames_np.shape[0],
        target_hw=(frames_np.shape[1], frames_np.shape[2]),
    )
    return make_agent_only(
        frames_np, layout_mask,
        method=factor_cfg["agent_method"],
        matte_factor=factor_cfg["matte_factor"],
        feather_sigma=factor_cfg["feather_sigma"],
    )


# ─────────────────────────────────────────────────────────────────────────
# D_I interaction-tube READ-THROUGH DISK CACHE (iter18 2026-06-15)
# ─────────────────────────────────────────────────────────────────────────
# stream_interaction_tubes() is a PURE deterministic fn of (mp4_bytes, mask.npz,
# cfg, num_frames) — its docstring contract. So its (decode MP4 + mine tubes)
# output can be memoized to ONE .npz per (clip_key, cfg-hash): the 2nd+ epoch reads
# the .npz instead of re-decoding + re-mining. RESULT-SAFETY: the cache returns the
# SAME tube LIST → the caller's random pick (training.py:2235) is byte-identical →
# training is bit-equivalent. The cache key includes every input that changes the
# tubes (cfg fields + num_frames) so any change MISSES (no stale reuse). Atomic
# write (temp + os.replace) survives concurrent DataLoader fork-workers computing
# the same clip. Bypassed entirely when tube_cache absent / enabled:false.

def _tube_cache_key(clip_key: str, num_frames: int, interaction_cfg: dict) -> str:
    """Short stable hash of EVERYTHING the tube list depends on. A change to any
    field below → different key → cache MISS → recompute (never a stale tube).

    Inputs hashed: clip_key (identifies mp4_bytes + mask.npz pair), num_frames
    (decode_video_bytes arg → changes frame sampling), and the 3 cfg fields the
    compute reads: enabled, tube_margin_pct, category_pair_blacklist. mp4_bytes +
    mask.npz are NOT hashed directly (immutable per clip_key in this corpus); the
    clip_key stands in for them. blacklist is order-canonicalized (sorted pairs)
    so [[a,b]] and [[b,a]] map to the same key (matches the compute's filter)."""
    blacklist = sorted(tuple(sorted(pair))
                       for pair in interaction_cfg["category_pair_blacklist"])
    payload = json.dumps({
        "clip_key": clip_key,
        "num_frames": int(num_frames),
        "enabled": bool(interaction_cfg["enabled"]),
        "tube_margin_pct": interaction_cfg["tube_margin_pct"],
        "category_pair_blacklist": blacklist,
        "v": 1,  # cache schema version — bump to invalidate all entries
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _tube_cache_path(cache_dir: str, clip_key: str, key: str) -> Path:
    """<cache_dir>/<safe_clip_key>__<cfg-hash>.npz (slashes → __ like masks)."""
    safe_key = clip_key.replace("/", "__")
    return Path(cache_dir) / f"{safe_key}__{key}.npz"


def _load_tube_cache(path: Path) -> List[np.ndarray]:
    """Read a cached tube list. Returns the list (possibly empty when n=0).
    Raises on a malformed file (fail loud — a corrupt cache must not be silently
    treated as a miss-then-overwrite that could mask an upstream write bug)."""
    d = np.load(path, allow_pickle=False)
    n = int(d["n"])
    return [d[f"t{i}"] for i in range(n)]


# ── Bounded LRU cap for the D_I tube cache (iter19 2026-07-06) ───────────────────
# The surgery mines interaction tubes and caches one .npz per (clip, cfg-hash) with NO size
# limit — on FULL it reached 314G and was draining the disk ~20G/hr (uncapped). Same fix as the
# frame cache: EVAL_TUBE_CACHE_MAX_GB is a HARD ceiling on the tube-cache dir's total .npz bytes;
# a store that would exceed it LRU-evicts oldest entries (least-recently-STORED) so the cache can
# NEVER fill the disk. Serialized across fork-workers by an flock; a running .tube_bytes counter
# keeps the common path O(1) and each eviction recomputes truth → self-heals. A miss just recomputes.
_TUBE_CACHE_MAX_ENV = "EVAL_TUBE_CACHE_MAX_GB"
_TUBE_EVICT_LOW_WATER = 0.85


def _tube_dir_bytes(cache_dir: str) -> int:
    """True total bytes of the cached .npz tube files (scandir = source of truth)."""
    total = 0
    with contextlib.suppress(OSError):
        with os.scandir(cache_dir) as it:
            for e in it:
                if e.name.endswith(".npz"):
                    with contextlib.suppress(OSError):
                        total += e.stat().st_size
    return total


def _tube_evict_to(cache_dir: str, target_bytes: float) -> int:
    """Delete oldest-STORED .npz tubes until the dir is <= target_bytes; return the true
    post-eviction total. Caller holds the flock so racing fork-workers never double-evict."""
    entries = []
    with contextlib.suppress(OSError):
        with os.scandir(cache_dir) as it:
            for e in it:
                if not e.name.endswith(".npz"):
                    continue
                with contextlib.suppress(OSError):
                    st = e.stat()
                    entries.append((st.st_mtime, st.st_size, e.path))
    total = sum(sz for _, sz, _ in entries)
    entries.sort(key=lambda t: t[0])   # oldest store-time first = least-recently-stored
    for _mt, sz, ep in entries:
        if total <= target_bytes:
            break
        with contextlib.suppress(OSError):
            os.unlink(ep)
            total -= sz
    return total


def _save_tube_cache_atomic(path: Path, tubes: List[np.ndarray]) -> None:
    """Atomically write the whole tube list to `path` under a HARD size cap
    (EVAL_TUBE_CACHE_MAX_GB) with LRU eviction. Write to a temp file in the
    SAME dir then os.replace (atomic on POSIX) so a concurrent reader never sees a
    half-written file and two fork-workers racing the same clip don't corrupt it
    (last writer wins; both wrote identical bytes — the fn is deterministic).
    Empty list → n=0, no t* arrays (handled symmetrically by _load_tube_cache)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(path.parent)
    _max_gb = os.environ.get(_TUBE_CACHE_MAX_ENV)
    if _max_gb is not None:
        # HARD CAP + LRU — bound the tube cache's own footprint so it can NEVER fill the disk.
        _max_bytes = float(_max_gb) * (1024 ** 3)
        _new_bytes = int(sum(int(t.nbytes) for t in tubes)) + 4096  # +header estimate
        if _new_bytes <= _max_bytes:
            _counter = os.path.join(cache_dir, ".tube_bytes")
            with open(os.path.join(cache_dir, ".tube.lock"), "w") as _lf:
                fcntl.flock(_lf, fcntl.LOCK_EX)
                try:
                    try:
                        with open(_counter) as _cf:
                            _total = int(_cf.read().strip())
                    except (OSError, ValueError):
                        _total = _tube_dir_bytes(cache_dir)   # missing/corrupt → recompute truth
                    if _total + _new_bytes > _max_bytes:
                        _total = _tube_evict_to(cache_dir, _TUBE_EVICT_LOW_WATER * _max_bytes - _new_bytes)
                    with contextlib.suppress(OSError):
                        with open(_counter, "w") as _cf:
                            _cf.write(str(_total + _new_bytes))
                finally:
                    fcntl.flock(_lf, fcntl.LOCK_UN)
    arrays = {"n": np.int64(len(tubes))}
    for i, t in enumerate(tubes):
        arrays[f"t{i}"] = t
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=f".{path.stem}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez(tmp_name, **arrays)
        # np.savez appends .npz to a name without that suffix; normalize.
        written = tmp_name if os.path.exists(tmp_name) else tmp_name + ".npz"
        os.replace(written, path)
    finally:
        for leftover in (tmp_name, tmp_name + ".npz"):
            if os.path.exists(leftover):
                os.remove(leftover)


def _compute_interaction_tubes(
    mp4_bytes: bytes,
    mask_npz_path: Path,
    interaction_cfg: dict,
    num_frames: int,
    tmp_dir: str,
    clip_key: str,
) -> List[np.ndarray]:
    """PURE compute — the original (pre-cache) stream_interaction_tubes body.

    Bitwise-parity contract with m11's legacy disk path at
    src/m11_factor_datasets.py:_process_one_clip lines 643-691 (the regen_di_only
    branch). 4-step pipeline:
      1. np.load(mask_npz_path) → interactions_json + per_object_bboxes_json +
         centroids_json + obj_id_to_cat_json
      2. decode_video_bytes(mp4_bytes, ...) → frames_np (T, H, W, C) uint8
      3. Apply category_pair_blacklist filter (order-insensitive, per #77)
      4. bboxes → make_interaction_tubes_from_bboxes / centroids → _from_centroids

    Deterministic given (mp4_bytes, mask_npz, cfg, num_frames). No globals, no RNG.
    """
    # iter18 H3: strict — "interaction_cfg must contain enabled" is the declared
    # contract; a missing yaml key silently disabling D_I mining is exactly the
    # masked-failure class CLAUDE.md bans.
    if not interaction_cfg["enabled"]:
        return []

    data = np.load(mask_npz_path, allow_pickle=False)
    interactions = (json.loads(str(data["interactions_json"]))
                    if "interactions_json" in data.files else [])
    if not interactions:
        return []

    centroids = (json.loads(str(data["centroids_json"]))
                 if "centroids_json" in data.files else {})
    per_object_bboxes = (json.loads(str(data["per_object_bboxes_json"]))
                         if "per_object_bboxes_json" in data.files else {})
    obj_id_to_cat = (json.loads(str(data["obj_id_to_cat_json"]))
                     if "obj_id_to_cat_json" in data.files else {})

    # Blacklist filter: match m11:661-670 EXACTLY for parity.
    blacklist = {tuple(sorted(pair))
                 for pair in interaction_cfg["category_pair_blacklist"]}
    filtered: list = []
    for ev in interactions:
        ca = ev.get("cat_a") or obj_id_to_cat.get(str(ev["obj_a"]))
        cb = ev.get("cat_b") or obj_id_to_cat.get(str(ev["obj_b"]))
        if ca is not None and cb is not None and tuple(sorted((ca, cb))) in blacklist:
            continue
        ev = dict(ev)
        ev["cat_a"], ev["cat_b"] = ca, cb
        filtered.append(ev)

    if not filtered:
        return []

    frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames)
    if frames_tensor is None:
        raise RuntimeError(
            f"stream_interaction_tubes: decode_video_bytes returned None for "
            f"clip_key={clip_key!r}. Upstream MP4 is corrupt or unreadable.")
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    frames_np = ((frames_np * 255).astype(np.uint8)
                 if frames_np.max() <= 1.0 else frames_np.astype(np.uint8))

    tube_margin = interaction_cfg["tube_margin_pct"]
    if per_object_bboxes:
        return make_interaction_tubes_from_bboxes(
            frames_np, filtered, per_object_bboxes, tube_margin)
    if centroids:
        return make_interaction_tubes_from_centroids(
            frames_np, filtered, centroids, tube_margin)
    return []


def stream_interaction_tubes(
    mp4_bytes: bytes,
    mask_npz_path: Path,
    interaction_cfg: dict,
    num_frames: int,  # iter18 H6: caller passes (cfg data.num_frames)
    tmp_dir: str = None,
    clip_key: str = "",
) -> List[np.ndarray]:
    """Generate D_I interaction tubes on-demand from (raw MP4 bytes + m10 mask.npz).

    Returns list of (T_tube, H, W, C) uint8 arrays — one per non-filtered
    interaction event with ≥4 valid frames. Empty list when all filtered out or
    no interactions present in mask.npz.

    interaction_cfg must contain:
      - `tube_margin_pct`: float, box expansion margin (e.g. 0.15)
      - `category_pair_blacklist`: list of [cat_a, cat_b] pairs to drop
      - `enabled`: bool (when False returns empty list — matches m11 behavior)
      - OPTIONAL `tube_cache`: {`enabled`: bool, `dir`: str} — when enabled+dir set,
        this fn is a READ-THROUGH cache: on call, if a .npz for (clip_key, cfg-hash)
        exists, load+return it (skips the MP4 decode + tube mining); else compute the
        tubes (via _compute_interaction_tubes), atomically save, return. Absent /
        disabled → exact pre-cache behavior (pure compute every call).

    RESULT-SAFETY: the cache returns the SAME tube LIST as the compute path, so the
    caller's random tube-pick (StreamingFactorDataset) is byte-identical → cached and
    uncached training are bit-equivalent. The cache key includes every tube-affecting
    input (clip_key, num_frames, enabled, tube_margin_pct, category_pair_blacklist) so
    any change MISSES → recompute. Determinism verified in
    scripts/legacy/tests_streaming/test_tube_cache.py.

    Deterministic given (mp4_bytes, mask_npz, cfg, num_frames). The only side effect
    (when the cache is enabled) is the atomic .npz write — pure read-through otherwise.
    Called per-step from StreamingFactorDataset; the caller picks a random tube via
    its seeded RNG.
    """
    tube_cache = interaction_cfg.get("tube_cache")
    cache_on = bool(tube_cache and tube_cache.get("enabled") and tube_cache.get("dir"))

    if not cache_on:
        return _compute_interaction_tubes(
            mp4_bytes, mask_npz_path, interaction_cfg, num_frames, tmp_dir, clip_key)

    key = _tube_cache_key(clip_key, num_frames, interaction_cfg)
    cache_path = _tube_cache_path(tube_cache["dir"], clip_key, key)
    if cache_path.exists():
        return _load_tube_cache(cache_path)

    tubes = _compute_interaction_tubes(
        mp4_bytes, mask_npz_path, interaction_cfg, num_frames, tmp_dir, clip_key)
    _save_tube_cache_atomic(cache_path, tubes)
    return tubes


def tensor_from_factor_array(
    frames_uint8: np.ndarray,
    num_frames: int,
    crop_size: int,
) -> torch.Tensor:
    """(T, H, W, C) uint8 → (T, C, H, W) float32 ImageNet-normalized.

    Shared normalization path between legacy disk loader (load_factor_clip) and
    streaming dataloader — guarantees bitwise parity across paths.

    Mirrors src/utils/training.py:load_factor_clip body (lines 1154-1172),
    minus the `np.load(path)` disk read.
    """
    frames = frames_uint8
    if frames.shape[1] != crop_size or frames.shape[2] != crop_size:
        resized = []
        for t in range(frames.shape[0]):
            img = PILImage.fromarray(frames[t])
            img = img.resize((crop_size, crop_size), PILImage.BILINEAR)
            resized.append(np.array(img))
        frames = np.stack(resized)
    if frames.shape[0] > num_frames:
        indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
        frames = frames[indices]
    elif frames.shape[0] < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor
