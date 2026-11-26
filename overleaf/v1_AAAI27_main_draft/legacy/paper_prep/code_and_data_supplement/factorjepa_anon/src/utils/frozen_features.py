"""Shared frozen-encoder feature extraction for probe_* modules. GPU-only.

Lives in `utils/` because both probe_action.py and probe_motion_cos.py need
the SAME forward pass over local TARs. Factored here per CLAUDE.md rule 32
("no cross-imports between m*.py files").

PUBLIC API
    ENCODERS                       — encoder catalog dict (kind, arch, embed_dim)
    load_vjepa_2_1_frozen(ckpt_path, num_frames) -> (model, crop, embed_dim)
    load_dinov2_frozen()                          -> (model, processor, crop, embed_dim)
    decode_to_tensor(mp4_bytes, ...)              -> (T, 3, crop, crop) fp32
    forward_vjepa(model, batch)                   -> (B, n_tokens, D) fp32 cpu
    forward_dinov2(model, batch, num_frames)      -> (B, T*n_spatial, D) fp32 cpu
    extract_features_for_keys(args, model, kind, crop, embed_dim,
                              keys, output_dir, *, label)
        -> (features (N, n_tokens, D), ordered_keys list[str])

Mirrors m05_vjepa_embed.py:580-700 (V-JEPA loader, RoPE/SDPA quirks, bf16) and
m05b_baselines.py:368-388 (DINOv2 loader, fp16 + FA2). bit-identical preprocessing
matters for paired-Δ to be honest — same crop + same ImageNet mean/std for both.
"""
import contextlib
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.checkpoint import save_array_checkpoint
from utils.config import get_pipeline_config, get_model_config
from utils.data_download import iter_clips_parallel
from utils.decode_feeder import BatchFeeder
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup
from utils.progress import make_pbar
from utils.video_io import decode_video_bytes
from utils.vjepa2_imports import get_vit_by_arch
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# iter18 W7 (PLR2004): semantic named constants.
_PENDING_FLUSH_N = 32    # decoded-tensor flush batch (perf, not research)
_TOKENS_RANK, _POOLED_RANK = 3, 2   # (N,tokens,D) vs (N,D) feature ranks


# ── Constants ─────────────────────────────────────────────────────────

_PCFG = get_pipeline_config()

# iter16 (2026-05-20): PATCH_SIZE + TUBELET_SIZE moved to configs/model/vjepa2_1.yaml
# (already present there). frozen_features is V-JEPA-coupled (builds ViT-G predictor)
# so it reads from the primary model config.
_MODEL_CFG    = get_model_config(None)["model"]   # None → default vjepa2_1.yaml
PATCH_SIZE    = _MODEL_CFG["patch_size"]            # was 16 literal
TUBELET_SIZE  = _MODEL_CFG["tubelet_size"]          # was 2 literal

# Mid-extract flush cadence. Probe extraction is a ~30-min job — at the m04/m05
# default of 500, that's ~13 flushes during one FULL train-split extraction.
# Probe-specific override to `streaming.probe_features_checkpoint_every` lets us
# cut to ~1-2 flushes without affecting m04/m05 (which keep their finer cadence
# for hour-long extractions).
# iter16 (2026-05-20): the prior `.get(..., fallback)` form violated CLAUDE.md
# "No `.get(key, default)` on YAML — use cfg[key] so missing keys crash".
# probe_features_checkpoint_every is REQUIRED in pipeline.yaml (line 62).
CHECKPOINT_EVERY = _PCFG["streaming"]["probe_features_checkpoint_every"]
# iter17 T1 (2026-05-29): parallel decode-worker count for the eval feature-extraction
# consumer. Was serial main-thread decode → GPU starved to ~0% util (decode-bound). Reuses
# the m05/embedding decode knob; same per-clip PyAV linspace decode (bit-identical frames),
# only parallelized — mirrors training.py's DECODE_WORKERS pool.
DECODE_WORKERS_EMBED = _PCFG["streaming"]["decode_workers_embed"]

# ImageNet normalization — both V-JEPA 2.1 + DINOv2 expect this.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _load_encoders_registry() -> dict:
    """Load the probe encoder registry from configs/eval/probe_encoders.yaml.

    Single source of truth — adding a new V-JEPA variant means editing the YAML,
    not this Python file. Schema documented in that YAML's header.

    iter15 Phase 8 (2026-05-17): moved configs/probe_encoders.yaml →
    configs/eval/probe_encoders.yaml to mirror configs/{model,train,legacy2}/
    subdir layout for clarity.
    """
    import yaml
    # iter18 H1: registry pointer from pipeline.yaml config_paths.
    cfg_path = (Path(__file__).resolve().parents[2]
                / _PCFG["config_paths"]["probe_encoders_registry"])
    if not cfg_path.exists():
        sys.exit(f"FATAL: encoder registry config missing: {cfg_path}")
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    if "encoders" not in data:
        sys.exit(f"FATAL: {cfg_path} missing top-level 'encoders' key")
    return data["encoders"]


ENCODERS = _load_encoders_registry()


# ── Encoder loaders ───────────────────────────────────────────────────

def resolve_encoder_state_dict(ckpt: dict) -> dict:
    """Pick the encoder state_dict from a V-JEPA-style checkpoint.

    Single source of truth for ckpt-key dispatch across all eval-side loaders
    (probe_action features, probe_future_mse forward). Recognized schemas:
      - "target_encoder" — Meta's V-JEPA 2.1 frozen ckpt (EMA teacher)
      - "encoder"        — older Meta convention
      - "student_state_dict" — written by utils.training.export_student_for_eval
                               (student_encoder.pt — the m09a/m09c export artifact)
      - "student"        — written by utils.training.save_training_checkpoint(full=True)
                           (m09{a,c}_ckpt_best.pt — full periodic ckpt)
      - raw dict         — last-resort fallback (state_dict already at top level)

    Without this, an m09a-exported student_encoder.pt would fall through to
    the raw-dict path and report 0/588 missing keys (the wrapper dict's
    {"student_state_dict", "model_id", "type"} get treated as state).
    """
    for key in ("target_encoder", "encoder", "student_state_dict", "student"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def resolve_predictor_state_dict(ckpt: dict) -> dict:
    """Pick the predictor state_dict from a V-JEPA-style checkpoint.

    Single source of truth for predictor ckpt-key dispatch — mirrors
    resolve_encoder_state_dict above. Recognized schemas:
      - "predictor"             — Meta's V-JEPA 2.1 frozen ckpt
      - "predictor_state_dict"  — written by utils.training.save_training_checkpoint
                                  (m09{a,c}_ckpt_best.pt — incl. iter15 head-only
                                  exports which carry predictor+motion_aux head)

    FAIL LOUD on neither: returns nothing, caller asserts.
    iter15 Phase 5 V6 fix (2026-05-15): added because probe_future_mse Stage 8
    failed on m09a2/c2 head-only ckpts (which use _state_dict suffix).
    """
    for key in ("predictor", "predictor_state_dict"):
        if key in ckpt:
            return ckpt[key]
    return None


def load_vjepa_frozen(ckpt_path: Path, num_frames: int, encoder_name: str):
    """Load ANY native V-JEPA frozen encoder by registry name (arch/crop/embed_dim from
    ENCODERS[encoder_name]). iter17 §B2: generalized from the ViT-G-hardcoded loader so
    cross-arch frozen baselines (2.0 g/L, v1 L/H, 2 vitL_256) build the right ViT.
    Returns (model, crop, embed_dim).
    VERIFY-FIRST: the xformers ctor kwargs below (use_silu/wide_silu/use_rope) are the
    2.0/2.1 family signature; V-JEPA-1 vit_large/vit_huge may need different kwargs."""
    if not ckpt_path.exists():
        sys.exit(f"FATAL: encoder ckpt not found: {ckpt_path}")
    enc = ENCODERS[encoder_name]
    crop = enc["crop"]
    print(f"Loading V-JEPA frozen {encoder_name} ({enc['arch']}, crop={crop}, T={num_frames}) ...")
    # iter17 (2026-05-27): mmap=True — memory-map the (up to 14 GB) ckpt instead of
    # reading it all into anon CPU RAM. The eval feature-extraction producer is tiny
    # (inline decode, 1 clip at a time), so the per-encoder ckpt LOAD is what drives
    # host-RAM to the 92-93% cgroup ceiling; mmap-backed storages are reclaimable page
    # cache, not anon → cuts the per-encoder spike with zero throughput cost (pages are
    # read on-demand during load_state_dict). Zip-serialized ckpts (torch.save default).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = resolve_encoder_state_dict(ckpt)
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

    vit_constructor = get_vit_by_arch(enc["arch"])
    model = vit_constructor(
        img_size=(crop, crop), patch_size=PATCH_SIZE, num_frames=num_frames,
        tubelet_size=TUBELET_SIZE, use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    msg = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(msg.unexpected_keys)
    total = len(list(model.state_dict().keys()))
    print(f"  Loaded {loaded}/{total} params  (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
    if loaded < total * 0.9:
        sys.exit(f"FATAL: only {loaded}/{total} V-JEPA params loaded — key mismatch")

    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    # SDPA monkey-patch for torch.compile (PyTorch issue #130098).
    torch.backends.cuda.sdp_kernel = contextlib.nullcontext
    return model, crop, enc["embed_dim"]


def load_vjepa_2_1_frozen(ckpt_path: Path, num_frames: int):
    """Back-compat alias (iter17 §B2) → the ViT-G frozen registry row. Callers that know the
    encoder name should use load_vjepa_frozen(ckpt, nf, name) so cross-arch backbones load right."""
    return load_vjepa_frozen(ckpt_path, num_frames, "vjepa_2_1_frozen")


def load_dinov2_frozen():
    """DINOv2 ViT-G/14 with registers, fp16, FA2. Mirrors m05b_baselines.py:368-388.
    Returns (model, processor, crop=224, embed_dim=1536).
    """
    from transformers import AutoImageProcessor, AutoModel
    enc = ENCODERS["dinov2"]
    print(f"Loading DINOv2 frozen ({enc['model_id']}, crop={enc['crop']}) ...")
    processor = AutoImageProcessor.from_pretrained(enc["model_id"])
    model = AutoModel.from_pretrained(
        enc["model_id"], dtype=torch.float16,
        device_map="cuda", attn_implementation="flash_attention_2",
    ).eval()
    return model, processor, enc["crop"], enc["embed_dim"]


def load_encoder_by_kind(encoder_name: str, ckpt_path, num_frames: int):
    """iter17 §B2/C2: single dispatch — load ANY registry encoder by its `kind`.
    Returns (model, crop, embed_dim). kind=vjepa needs a local --encoder-ckpt; ijepa/dinov2
    load from the registry HF model_id (ckpt_path ignored). Replaces the per-module
    if/elif vjepa/dinov2 blocks in m12a/m12b/m12c so a new kind is wired in ONE place."""
    kind = ENCODERS[encoder_name]["kind"]
    if kind == "vjepa":
        if ckpt_path is None:
            sys.exit(f"FATAL: {encoder_name} (kind=vjepa) requires --encoder-ckpt")
        return load_vjepa_frozen(ckpt_path, num_frames, encoder_name)
    if kind == "ijepa":
        from utils.ijepa_features import load_ijepa_frozen   # lazy: avoids circular import
        model, _proc, crop, embed_dim = load_ijepa_frozen(encoder_name)
        return model, crop, embed_dim
    if kind == "hf_vjepa2":
        from utils.hf_vjepa2_features import load_hf_vjepa2_frozen   # lazy: avoids circular import
        return load_hf_vjepa2_frozen(encoder_name)
    if kind == "lejepa":
        from utils.lejepa_features import load_lejepa_frozen   # lazy: avoids circular import
        return load_lejepa_frozen(encoder_name)
    if kind == "dinov2":
        model, _proc, crop, embed_dim = load_dinov2_frozen()
        return model, crop, embed_dim
    sys.exit(f"FATAL: unknown encoder kind '{kind}' for {encoder_name}")


# ── Frame preprocessing ───────────────────────────────────────────────

def resize_and_normalize(frames_np: np.ndarray, crop: int) -> torch.Tensor:
    """frames_np: (T, H, W, 3) uint8 → (T, 3, crop, crop) fp32 ImageNet-normalized.
    Center-crop after resize-shorter-side. Eval recipe (NOT the train-time random crop).
    """
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    _, _, H, W = t.shape
    short = min(H, W)
    new_H = int(round(H * crop / short))
    new_W = int(round(W * crop / short))
    t = F.interpolate(t, size=(new_H, new_W), mode="bilinear", align_corners=False, antialias=True)
    top  = (new_H - crop) // 2
    left = (new_W - crop) // 2
    t = t[:, :, top:top + crop, left:left + crop]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def decode_to_tensor(mp4_bytes: bytes, tmp_dir: str, clip_key: str,
                    num_frames: int, crop: int, cache_frames: bool = True):
    """Decode MP4 bytes → preprocessed (T, 3, crop, crop) fp32 tensor.
    Returns None on decode failure (caller skips).

    cache_frames=False skips storing the decoded frames in the eval frame cache — for the probe
    feature extraction, which reads each clip exactly ONCE (caching it only churns the shared cache
    and evicts the metrics' re-read TEST set → the 4x eval blow-up). The METRICS call this with the
    default True so their re-read test frames DO cache (decode once → hit for the other 9 metrics × 6
    encoders); the cache then holds ONLY the ~454G test set, not the 2.25TB full working set.
    """
    frames_t = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames,
                                  cache_frames=cache_frames)
    if frames_t is None:
        return None
    f = frames_t.permute(0, 2, 3, 1).numpy()
    f = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
    return resize_and_normalize(f, crop)


# ── Forward passes ────────────────────────────────────────────────────

@torch.no_grad()
def forward_vjepa(model, batch: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """V-JEPA 2.1: input (B, T, 3, H, W) → (B, n_st_tokens, D) fp32 on CPU.
    Constructor expects (B, C, T, H, W) so we permute. Models with deep supervision
    return list/tuple of features; we use the FINAL layer (matches Meta's eval recipe).
    """
    pixel = batch.to("cuda", dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()
    out = model(pixel)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.float().cpu()


@torch.no_grad()
def forward_dinov2(model, batch: torch.Tensor, num_frames: int) -> torch.Tensor:
    """DINOv2 ViT-G/14: process T frames per clip independently, concat token sequences
    over time → (B, T*n_spatial, D) fp32 on CPU. V-JEPA 2 paper §4.1 'tile + temporal pool'.
    """
    B, T, C, H, W = batch.shape
    assert T == num_frames, f"DINOv2 batch frame count mismatch: {T} vs {num_frames}"
    flat = batch.view(B * T, C, H, W).to("cuda", dtype=torch.float16)
    last_hidden = model(pixel_values=flat).last_hidden_state          # (B*T, n_tokens, D)
    _, n_tokens_per_frame, D = last_hidden.shape
    return last_hidden.view(B, T * n_tokens_per_frame, D).float().cpu()


# ── Extraction loop (producer-consumer, OOM-safe, resumable) ─────────

def _save_features_ckpt(ckpt_file: Path, feats_acc, keys_acc) -> None:
    """Atomic .npz checkpoint of features-so-far.
    Suffix is .tmp.npz (not .npz.tmp) — np.savez auto-appends .npz when the
    path doesn't already end in .npz, which would silently rename our tmp
    file behind our back and break the os.replace. See errors_N_fixes.md #82.

    iter13 disk-saving (2026-05-05): features stored as float16 instead of
    float32 — saves 50% disk per cache (~146 GB train cache → 73 GB on FULL
    eval). Information-equivalent: V-JEPA's forward returns bf16 internally,
    upcast to fp32 in forward_vjepa via .float() — saving back as fp16 has no
    information loss vs the bf16 source. PyTorch consumers (probe head, motion
    cos) all do `.float()` on load, NumPy consumers do explicit `astype(float32)`.
    """
    if not feats_acc:
        return
    tmp = ckpt_file.with_suffix(".tmp.npz")
    # iter13 (2026-05-05): drop OLD ckpt BEFORE writing NEW tmp so peak transient
    # is 1× cache (73 GB at fp16) instead of 2× (146 GB). Required to fit eval on
    # 199 GB disk — otherwise mid-extract checkpoint flushes ENOSPC. Trade-off:
    # if killed mid-write, no resume cache (re-extract from clip 0, ~30 min loss);
    # without this, ENOSPC during write is ~80% per flush. Strict improvement.
    if ckpt_file.exists():
        ckpt_file.unlink()
    # iter13 (2026-05-05): plain np.savez (not _compressed). Tried compression
    # earlier — turned out to cost ~3 hours on Stage 3 train extraction because
    # each flush re-compresses CUMULATIVE features (O(N²) work on 13 flushes
    # × growing 5→68 GB). Uncompressed write is ~5 sec per flush at SSD speed.
    # Disk savings from compression (~50 GB peak) weren't worth 3 h wall-cost
    # given fp16 + unlink-before-write already brought peak from 261 GB → 73 GB
    # transient (fits 199 GB budget with 11 GB margin).
    np.savez(tmp,
             features=np.stack(feats_acc).astype(np.float16),
             keys=np.array(keys_acc, dtype=object))
    # fsync the tmp file to ensure data hits the disk BEFORE the atomic rename.
    # Without this, a crash between np.savez return and os.replace could leave
    # tmp.npz with not-yet-flushed pages — recovery via the orphan-tmp cleanup
    # is fine, but fsync makes the safe-completion path durable.
    with open(tmp, "rb") as _fp:
        os.fsync(_fp.fileno())
    os.replace(tmp, ckpt_file)


def _pool_tokens(feats: torch.Tensor, n_out: int) -> torch.Tensor:
    """(N, T, D) → (N, n_out, D) via adaptive_avg_pool1d on token axis.

    Encoder-agnostic block-mean pool over contiguous token chunks. Works for
    V-JEPA's (T_temp×H×W=4608) layout AND DINOv2's (T_frames×n_spatial=4112)
    layout — both collapse to `n_out` tokens with no encoder-specific reshape.
    Pooling is information-equivalent to mean-pool over the same blocks; the
    AttentiveClassifier's cross-attention then attends over n_out queries × n_out
    keys (e.g. 16² = 256 vs the unpooled 4608² = 21M) — ~290× attention compute
    savings at the head with negligible accuracy impact (V-JEPA paper §4 evals
    use 16-32 tokens at the attentive probe input).
    """
    if feats.shape[1] == n_out:
        return feats
    # adaptive_avg_pool1d expects (..., C, L); we have (N, T, D) → transpose to
    # (N, D, T) → pool L axis to n_out → transpose back → (N, n_out, D).
    pooled = F.adaptive_avg_pool1d(feats.transpose(1, 2).contiguous(), n_out)
    return pooled.transpose(1, 2).contiguous()


def _flush_batch(batch, pending_keys, model, encoder_kind, num_frames,
                 sizer, feats_acc, keys_acc, pbar, pool_tokens=None,
                 perframe_acc=None, pf_tubelet=None):
    """Sub-batch and run encoder. AdaptiveBatchSizer halves on OOM and retries.

    iter13 (2026-05-05): if `pool_tokens` is set, apply token-axis adaptive
    avg-pool BEFORE the .numpy() materialisation so feats_acc accumulates the
    pooled (N, pool_tokens, D) shape. Disk-saving is HUGE for ViT-G FULL evals:
    4608 → 16 tokens = 288× per-clip reduction (~21 MB → ~73 KB at fp16).

    iter19 2026-07-09 rungs 2+3: `batch` arrives pre-STACKED + pinned from BatchFeeder and is
    uploaded ONCE here — the sizer's sub-batches become cuda views and forward_*'s .to("cuda")
    a no-op move. Values byte-identical (stack + slice = pure data movement).

    iter19 2026-07-09 tcc-cache (OPT-IN, vjepa only): when `perframe_acc` is given, the SAME encoder
    forward (per_frame_features.forward_tokens) feeds BOTH the adaptive-pool probe features
    (tokens.cpu() == forward_vjepa's tensor output) AND a per-tubelet spatial-mean pooled ON CUDA →
    fp32 perframe_acc — so a downstream tcc run reads it byte-for-byte instead of re-encoding. The
    pool MUST stay on CUDA (a CPU reduction of the same tokens differs in the last ULP). Default
    perframe_acc=None → the probe path below is byte-identical to before (forward_vjepa)."""
    batch = batch.to("cuda", non_blocking=True)
    n_total = batch.shape[0]
    _emit_pf = perframe_acc is not None and encoder_kind == "vjepa"
    i = 0
    while i < n_total:
        sub = batch[i:i + sizer.size]
        pf_np = None
        try:
            if _emit_pf:
                from utils.per_frame_features import forward_tokens, pool_per_frame   # lazy: avoid circular
                tokens = forward_tokens(model, sub)                       # ONE shared forward (cuda fp32)
                pf_np = pool_per_frame(tokens, num_frames, pf_tubelet).cpu().numpy()   # CUDA pool → tcc-identical
                feats = tokens.cpu()                                      # == forward_vjepa's tensor output
            elif encoder_kind == "vjepa":
                feats = forward_vjepa(model, sub)
            elif encoder_kind == "ijepa":
                from utils.ijepa_features import forward_ijepa   # lazy: avoids ijepa↔frozen circular import
                feats = forward_ijepa(model, sub, num_frames)
            elif encoder_kind == "hf_vjepa2":
                from utils.hf_vjepa2_features import forward_hf_vjepa2   # lazy
                feats = forward_hf_vjepa2(model, sub, num_frames)
            elif encoder_kind == "lejepa":
                from utils.lejepa_features import forward_lejepa   # lazy
                feats = forward_lejepa(model, sub, num_frames)
            else:
                feats = forward_dinov2(model, sub, num_frames)
        except torch.cuda.OutOfMemoryError:
            cuda_cleanup()
            if not sizer.on_oom():
                raise
            continue
        if pool_tokens is not None:
            feats = _pool_tokens(feats, pool_tokens)
        feats_np = feats.numpy()
        for r in range(feats_np.shape[0]):
            feats_acc.append(feats_np[r])
            if _emit_pf:
                perframe_acc.append(pf_np[r].astype(np.float32))
            keys_acc.append(pending_keys[i + r])
            pbar.update(1)
        sizer.after_batch_success()
        i += sub.shape[0]


def extract_features_for_keys(args, model, encoder_kind: str, crop: int, embed_dim: int,
                              keys, output_dir: Path, *, label: str = "feat",
                              pool_tokens: int = None, cache_frames: bool = True,
                              emit_perframe_path: "Path" = None, pf_tubelet: int = None):
    """Producer-consumer feature extraction over local TARs.

    Args:
        args:           Parsed argparse Namespace; reads .local_data, .cache_policy, .num_frames.
        model:          Loaded encoder (V-JEPA or DINOv2).
        encoder_kind:   "vjepa" | "dinov2" (selects forward_*).
        crop:           Encoder's expected square input size.
        embed_dim:      Encoder's hidden dim D (used only when no clips processed).
        keys:           Iterable of clip_keys to extract.
        output_dir:     Where to write the resume checkpoint (.probe_<label>_ckpt.npz).
        label:          Prefix for the resume checkpoint filename.
        pool_tokens:    If int, adaptive-avg-pool the token axis from N_raw → pool_tokens
                        BEFORE storage. Default None = no pool (legacy path; (N, 4608, D)
                        for V-JEPA ViT-G). Set to 16 for V-JEPA-paper-style attentive probe
                        with 290× less downstream compute + 290× smaller .npy artifacts.

    Returns:
        (features (N, n_tokens, D) **fp16**, ordered_keys list[str]) — aligned by row.
        n_tokens = pool_tokens if set, else encoder-native token count.
        Float16 storage (iter13, 2026-05-05) saves 50% disk + RAM. Consumers
        upcast via `.float()` (PyTorch) or `astype(np.float32)` (NumPy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = output_dir / f".probe_{label}{artifact('probe_ckpt_suffix')}"

    # iter13 (2026-05-05): orphan-tmp cleanup. The 2026-05-04 v2 ENOSPC crash
    # left a 35 GB `.probe_features_train_ckpt.tmp.npz` behind because torch.save
    # didn't reach `os.replace`. Sweep stale .tmp.npz BEFORE we start so the
    # disk pressure doesn't snowball across reruns. Only safe to delete because:
    #   (a) tmp files are write-only intermediates; readers ALWAYS open the
    #       atomic-renamed final ckpt_file (no consumer reads .tmp.npz);
    #   (b) any in-flight writer in another process would have the file locked
    #       on POSIX, and we don't kill in-flight processes — only sweep dead orphans.
    tmp_npz = ckpt_file.with_suffix(".tmp.npz")
    if tmp_npz.exists():
        try:
            sz_gb = tmp_npz.stat().st_size / 1e9
            tmp_npz.unlink()
            print(f"  [orphan-cleanup] removed stale {tmp_npz.name} ({sz_gb:.1f} GB) "
                  f"— prior crash left it behind")
        except OSError as e:
            print(f"  WARN: could not remove stale {tmp_npz}: {e}")

    feats_acc, keys_acc, processed = [], [], set()
    if ckpt_file.exists():
        try:
            data = np.load(ckpt_file, allow_pickle=True)
            cached_features = data["features"]
            # iter13 (2026-05-05): if pool_tokens differs from cached shape, the
            # cache is from a prior run with a different pooling regime — drop it.
            # Otherwise the resume path would mix raw+pooled rows and the np.stack
            # would explode at flush time.
            if pool_tokens is not None and cached_features.shape[1] != pool_tokens:
                print(f"  resume ckpt token mismatch (cached={cached_features.shape[1]}, "
                      f"requested pool_tokens={pool_tokens}) — dropping stale cache, "
                      f"re-extracting from scratch")
                ckpt_file.unlink()
            else:
                feats_acc = list(cached_features)
                keys_acc = list(data["keys"])
                processed = set(keys_acc)
                print(f"  Resume: {len(processed)} clips already cached")
        except Exception as e:
            # iter13 (2026-05-05): per CLAUDE.md FAIL HARD. Corrupt resume cache
            # is a real bug (crash mid-flush, schema mismatch) — surface it instead
            # of silently re-extracting and burning ~30 min of GPU on something
            # the user probably wants to investigate. Re-run with cache-policy 2
            # to wipe the cache deliberately.
            print(f"  [resume] FATAL: resume ckpt corrupt ({e})", flush=True)
            print(f"  [resume] delete {ckpt_file} and re-run if you want a fresh extract", flush=True)
            raise

    # iter19 opt-in tcc-cache: emit the per-tubelet fp32 cache ONLY on a CLEAN (non-resumed) extract —
    # a resumed run's perframe_acc would miss the already-cached clips, so skip and let tcc re-encode
    # byte-identically rather than trust a partial cache. vjepa only (tcc is vjepa-only).
    _do_pf = emit_perframe_path is not None and encoder_kind == "vjepa" and not processed
    perframe_acc = [] if _do_pf else None
    if emit_perframe_path is not None and not _do_pf:
        print(f"  [perframe] skip emit ({label}): resumed/non-vjepa (processed={len(processed)}) — "
              f"tcc re-encodes this arm byte-identically")
    keys = list(keys)
    target = set(keys) - processed
    if not target:
        # iter13: float16 storage matches _save_features_ckpt; saves 50% RAM at
        # the resume short-circuit (when the ckpt already covers all target keys).
        feats = (np.stack(feats_acc).astype(np.float16)
                 if feats_acc else np.empty((0, 0, embed_dim), dtype=np.float16))
        return feats, keys_acc

    # iter13 (2026-05-05): batch envelope read from pipeline.yaml gpu block —
    # was hardcoded `initial=8, max=32` (legacy pre-Blackwell). On 96 GB
    # Blackwell, ViT-G fwd-only feature extraction can run at BS=44 (matches
    # m05 inference profile at 64-frame; here we use 16-frame so headroom is
    # even larger). AdaptiveBatchSizer ramps from initial → max based on actual
    # VRAM utilisation. DINOv2 (300M params) tolerates BS=96.
    if encoder_kind in ("vjepa", "hf_vjepa2"):   # iter17: hf_vjepa2 is a heavy video ViT-g → vjepa BS envelope
        # iter13: probe-specific initial BS — required in pipeline.yaml > gpu.
        # iter16 (2026-05-20): replaced `.get(key, fallback)` form per CLAUDE.md
        # "No .get(key, default) on YAML — use cfg[key] so missing keys crash".
        # Both yaml keys exist (line ~127 + 132); fall-back idiom was unnecessary.
        initial_bs = _PCFG["gpu"]["inference_vjepa_probe_initial_bs"]
        max_bs     = _PCFG["gpu"]["inference_adapted_probe_bs"]   # 44
    else:
        initial_bs = _PCFG["gpu"]["inference_dinov2_initial_bs"]  # 16
        max_bs     = _PCFG["gpu"]["inference_dinov2_bs"]          # 96
    sizer = AdaptiveBatchSizer(
        initial_size=initial_bs, min_size=1, max_size=max_bs,
        memory_cap=_PCFG["gpu"]["gpu_memory_target"],
    )
    print(f"  AdaptiveBatchSizer({label}): start={sizer.size}, max={max_bs}, target_vram={_PCFG['gpu']['gpu_memory_target']:.0%}")

    pbar = make_pbar(total=len(keys), desc=f"probe_{label}", unit="clip", initial=len(processed))

    tmp_base = output_dir / f"tmp_decode_{label}"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    clip_q, tar_stop, _reader = iter_clips_parallel(
        local_data=args.local_data, subset_keys=target, processed_keys=processed)

    # iter17 T1 (2026-05-29): PARALLEL decode pool. The TAR readers above only prefetch mp4
    # BYTES; previously the MAIN thread decoded one clip at a time (serial) → GPU starved to
    # ~0% util (spiky), eval decode-bound. Now DECODE_WORKERS_EMBED threads pull bytes →
    # decode_to_tensor (per-clip tmp file → thread-safe) → decoded_q; the main thread only
    # batches → GPU. Bit-identical frames (same PyAV linspace decode), just parallelized.
    # set_num_threads(1) avoids ATen oversubscription across the N workers' resize_and_normalize
    # (CLAUDE.md threading rule). Clips arrive out-of-order but every feature is tagged with its
    # clip_key (keys_acc), so downstream (keyed by clip_key) is order-independent.
    torch.set_num_threads(1)
    decoded_q: "queue.Queue" = queue.Queue(maxsize=max(64, DECODE_WORKERS_EMBED * 4))
    decode_stop = threading.Event()

    def _decode_worker():
        while not decode_stop.is_set():
            try:
                item = clip_q.get(timeout=300)
            except queue.Empty:
                break
            if item is None:
                clip_q.put(None)            # re-broadcast sentinel to sibling workers
                break
            ck, mp4_bytes = item
            try:
                # cache_frames threads from the caller: the TEST extraction caches (test is re-read by the
                # 10 metrics × 6 encoders → caching here WARMS the shared cache fast, at decode speed ~8/s,
                # instead of the slow compute-bound metrics warming it ~1.3/s). TRAIN/VAL pass False (read
                # ONCE → caching them only churns/evicts the 454G test set → thrash). So the cache holds
                # ONLY the test set, and it's warm before the metrics need it.
                dt = decode_to_tensor(mp4_bytes, tmp_dir, ck, args.num_frames, crop, cache_frames=cache_frames)
            except Exception as e:          # per-clip FAIL-SOFT (matches serial path's None-skip)
                print(f"  SKIP (decode error {ck}): {e}")
                dt = None
            decoded_q.put((ck, dt))

    decode_workers = [threading.Thread(target=_decode_worker, daemon=True)
                      for _ in range(DECODE_WORKERS_EMBED)]
    for w in decode_workers:
        w.start()

    def _decode_joiner():                   # signal end-of-stream once all decoders drain
        for w in decode_workers:
            w.join()
        decoded_q.put(None)
    threading.Thread(target=_decode_joiner, daemon=True).start()
    print(f"  decode pool: {DECODE_WORKERS_EMBED} workers (T1 parallel decode), "
          f"decoded_q maxsize={decoded_q.maxsize}")

    # iter19 2026-07-09 rung 2: adapt decoded_q into an iterator so BatchFeeder's builder
    # thread stacks+pins the next batch while the current one forwards (double-buffer, same
    # util as m12d/m12f); the old pending-list accumulation + post-loop tail flush both fold
    # into it (tail batch published as-is). Same decoded tensors, same order semantics.
    def _decoded_iter():
        while True:
            try:
                item = decoded_q.get(timeout=300)
            except queue.Empty:
                print("  WARN: decoded queue timeout (5 min) — draining out")
                return
            if item is None:
                return
            ck, dt = item
            if dt is None:
                print(f"  SKIP (decode fail): {ck}")
                continue
            yield ck, dt

    batches = BatchFeeder(_decoded_iter(), batch_size=_PENDING_FLUSH_N, depth=2, pin_memory=True)
    n_since_ckpt = 0
    t0 = time.time()
    try:
        for pending_keys, batch_t in batches:
            _flush_batch(batch_t, pending_keys,
                         model, encoder_kind, args.num_frames,
                         sizer, feats_acc, keys_acc, pbar,
                         pool_tokens=pool_tokens,
                         perframe_acc=perframe_acc, pf_tubelet=pf_tubelet)
            n_since_ckpt += len(pending_keys)
            if n_since_ckpt >= CHECKPOINT_EVERY:
                _save_features_ckpt(ckpt_file, feats_acc, keys_acc)
                n_since_ckpt = 0
    finally:
        decode_stop.set()
        tar_stop.set()
        pbar.close()

    elapsed = time.time() - t0
    print(f"  extract_features_for_keys({label}): {len(keys_acc)}/{len(keys)} clips in {elapsed:.0f}s")

    # iter15 D9 fix (2026-05-16): unconditional final ckpt save. Without this, at
    # POC scale (~1096 train clips < CHECKPOINT_EVERY=5000) the resume ckpt is
    # NEVER written → downstream callers (probe_taxonomy Stage 3.5 reusing
    # probe_action's lazy extraction) can't short-circuit and re-extract from
    # scratch, wasting ~7 min × N encoders of GPU. At FULL scale this overwrites
    # the last intermediate save with identical content (one extra np.savez ≈
    # ~5 sec @ ~24 GB; cleaned up by run_eval.sh cleanup_lazy after Stage 3.5).
    # Pairs with D8's --keep-lazy-cache defer logic in probe_action.
    if feats_acc:
        _save_features_ckpt(ckpt_file, feats_acc, keys_acc)
        print(f"  [D9] final resume ckpt saved: {ckpt_file.name} "
              f"({len(keys_acc)} clips) — Stage 3.5 will resume from this")

    # iter19 opt-in tcc-cache: atomically publish the fp32 per-tubelet cache (only on a clean extract).
    if _do_pf and perframe_acc and len(perframe_acc) == len(keys_acc):
        pf_arr = np.stack(perframe_acc).astype(np.float32)
        _pf_tmp = Path(emit_perframe_path).with_suffix(".tmp.npy")
        np.save(_pf_tmp, pf_arr)
        os.replace(_pf_tmp, emit_perframe_path)
        print(f"  [perframe] saved fp32 per-tubelet cache: {Path(emit_perframe_path).name} "
              f"{pf_arr.shape} — a tcc run reads this instead of re-encoding")
    elif _do_pf:
        print(f"  [perframe] NOT saved ({label}): count mismatch "
              f"(perframe={len(perframe_acc) if perframe_acc else 0} != feats={len(keys_acc)})")

    # iter13: float16 storage. Matches _save_features_ckpt. Downstream consumers
    # (probe head training in probe_action.py:380, motion-cos in
    # probe_motion_cos.py:81-87) all upcast to fp32 explicitly via .float() or
    # .astype(np.float32) — fp16 storage is information-equivalent (V-JEPA
    # forward is bf16 internally; .float() in forward_vjepa already loses bf16
    # precision; round-tripping through fp16 storage adds zero new loss).
    feats = (np.stack(feats_acc).astype(np.float16)
             if feats_acc else np.empty((0, 0, embed_dim), dtype=np.float16))
    return feats, keys_acc


# ── iter15 Phase 6 audit (2026-05-16): post-hoc head augmenter ─────────
# See iter/iter15_trainHead_freezeEncoder/planCODE_head_eval.md Step 2.
# Cleaner than threading motion_aux_head through _flush_batch — keeps
# extract_features_for_keys schema unchanged. Probes call extract_features
# (encoder-only path, bit-identical to iter14 baseline), then optionally
# call this augmenter to compute motion_aux head outputs from the already-
# extracted features. Output saved separately as
# motion_aux_concat_<label>.npy; probes load both at train/eval time.

def apply_motion_aux_head_to_features(features, motion_aux_head, batch_size: int,
                                      align_to: int = None):  # iter18 H6: None → pipeline.yaml eval.motion_aux_align_to
    """Run motion_aux head on mean-pooled features → (N, K + n_dims [+ pad]) concat tensor.

    Args:
        features:        (N, n_tokens, D) float16 OR (N, D) float16 — from
                          extract_features_for_keys output. If 3-D, mean-pool
                          across the token axis to (N, D) before head forward.
        motion_aux_head: MotionAuxHead in eval mode (call load_motion_aux_head).
                          Must have d_encoder matching features' D dim — FAIL
                          LOUD on mismatch.
        batch_size:      Per-batch forward size for the head. Default 64 fits
                          ~36-dim output × N_clips trivially on CPU; bump for
                          GPU. Head is ~430K params so memory is not the issue.
        align_to:        D10 (2026-05-16): pad output dim with zeros to next
                          multiple of `align_to` so the downstream
                          AttentiveClassifier(num_heads=16) reshape
                          `qkv.reshape(B, N, 3, num_heads, C//num_heads)`
                          succeeds. With K+n_dims=36 and D=1664 (already 16-aligned),
                          unpadded total is 1664+36=1700 = 16×106.25 → reshape fails.
                          Padding to 48 yields 1664+48=1712 = 16×107. Zero pad is
                          information-preserving (qkv linear maps zero columns to
                          zero contribution; downstream head learns to ignore them).

    Returns:
        np.ndarray (N, K + n_dims + pad) float32 — [ce_logits ‖ mse_pred ‖ zeros].
        Probes save this alongside features_<label>.npy as
        motion_aux_concat_<label>.npy and concat at probe input.
    """
    import torch
    import numpy as np
    if features.ndim == _TOKENS_RANK:
        # (N, n_tokens, D) → mean over token axis → (N, D)
        feats_pooled = features.astype(np.float32).mean(axis=1)
    elif features.ndim == _POOLED_RANK:
        feats_pooled = features.astype(np.float32)
    else:
        raise ValueError(f"apply_motion_aux_head_to_features: features must be 2-D or 3-D, "
                         f"got shape {features.shape}")
    if feats_pooled.shape[1] != motion_aux_head.d_encoder:
        raise RuntimeError(
            f"feature dim mismatch: features D={feats_pooled.shape[1]} vs "
            f"motion_aux_head.d_encoder={motion_aux_head.d_encoder}. "
            f"Wrong encoder/head pairing — check run_eval.sh motion_aux_head_for resolver.")
    device = next(motion_aux_head.parameters()).device
    # iter18 H6: None → pipeline.yaml eval.motion_aux_align_to.
    if align_to is None:
        align_to = _PCFG["eval"]["motion_aux_align_to"]
    n = feats_pooled.shape[0]
    out_dim = motion_aux_head.n_motion_classes + motion_aux_head.n_motion_dims
    # D10 (2026-05-16): pad K+n_dims to next multiple of `align_to` so downstream
    # AttentiveClassifier qkv reshape (B, N, 3, num_heads, C//num_heads) succeeds.
    pad = (-out_dim) % align_to                         # 0 if already aligned
    out_dim_padded = out_dim + pad
    out = np.zeros((n, out_dim_padded), dtype=np.float32)  # zeros-init covers the pad cols
    motion_aux_head.eval()                              # belt + suspenders
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = torch.from_numpy(feats_pooled[i:i + batch_size]).to(device)
            ce_logits, mse_pred = motion_aux_head(chunk)
            out[i:i + batch_size, :motion_aux_head.n_motion_classes] = \
                ce_logits.detach().cpu().numpy()
            out[i:i + batch_size, motion_aux_head.n_motion_classes:out_dim] = \
                mse_pred.detach().cpu().numpy()
            # cols [out_dim : out_dim_padded] stay zero (D10 pad)
    return out


__all__ = [
    "ENCODERS",
    "load_vjepa_frozen", "load_vjepa_2_1_frozen", "load_dinov2_frozen", "load_encoder_by_kind",
    "decode_to_tensor", "resize_and_normalize",
    "forward_vjepa", "forward_dinov2",
    "extract_features_for_keys",
    "apply_motion_aux_head_to_features",   # iter15 Phase 6 audit
    "save_array_checkpoint",   # re-export so callers don't both-import from utils.checkpoint
]
