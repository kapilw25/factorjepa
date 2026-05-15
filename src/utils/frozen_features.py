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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.checkpoint import save_array_checkpoint
from utils.config import get_pipeline_config
from utils.data_download import iter_clips_parallel
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup
from utils.progress import make_pbar
from utils.video_io import decode_video_bytes
from utils.vjepa2_imports import get_vit_by_arch


# ── Constants ─────────────────────────────────────────────────────────

_PCFG = get_pipeline_config()
PATCH_SIZE = 16
TUBELET_SIZE = 2
# Mid-extract flush cadence. Lower = finer-grained crash resume but more I/O;
# higher = less I/O but bigger re-extract on crash. Probe extraction is a
# ~30-min job — at the legacy m04/m05 default of 500, that's ~13 flushes
# during a single FULL train-split extraction. Probe-specific override to
# `streaming.probe_features_checkpoint_every` lets us cut to ~1-2 flushes
# without affecting m04/m05 (which keep their finer cadence for hour-long
# extractions). Falls back to the global value if the probe key is absent.
CHECKPOINT_EVERY = _PCFG["streaming"].get(
    "probe_features_checkpoint_every",
    _PCFG["streaming"]["checkpoint_every"],
)

# ImageNet normalization — both V-JEPA 2.1 + DINOv2 expect this.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _load_encoders_registry() -> dict:
    """Load the probe encoder registry from configs/probe_encoders.yaml.

    Single source of truth — adding a new V-JEPA variant means editing the YAML,
    not this Python file. Schema documented in that YAML's header.
    """
    import yaml
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "probe_encoders.yaml"
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


def load_vjepa_2_1_frozen(ckpt_path: Path, num_frames: int):
    """V-JEPA 2.1 ViT-G frozen. Mirrors m05_vjepa_embed.py:629-670.
    Returns (model, crop=384, embed_dim=1664).
    """
    if not ckpt_path.exists():
        sys.exit(f"FATAL: encoder ckpt not found: {ckpt_path}")
    enc = ENCODERS["vjepa_2_1_frozen"]
    crop = enc["crop"]
    print(f"Loading V-JEPA 2.1 ViT-G ({enc['arch']}, crop={crop}, T={num_frames}) ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
                    num_frames: int, crop: int):
    """Decode MP4 bytes → preprocessed (T, 3, crop, crop) fp32 tensor.
    Returns None on decode failure (caller skips).
    """
    frames_t = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames)
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


def _flush_batch(pending_tensors, pending_keys, model, encoder_kind, num_frames,
                 sizer, feats_acc, keys_acc, pbar, pool_tokens=None):
    """Sub-batch and run encoder. AdaptiveBatchSizer halves on OOM and retries.

    iter13 (2026-05-05): if `pool_tokens` is set, apply token-axis adaptive
    avg-pool BEFORE the .numpy() materialisation so feats_acc accumulates the
    pooled (N, pool_tokens, D) shape. Disk-saving is HUGE for ViT-G FULL evals:
    4608 → 16 tokens = 288× per-clip reduction (~21 MB → ~73 KB at fp16).
    """
    batch = torch.stack(pending_tensors)
    n_total = batch.shape[0]
    i = 0
    while i < n_total:
        sub = batch[i:i + sizer.size]
        try:
            if encoder_kind == "vjepa":
                feats = forward_vjepa(model, sub)
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
            keys_acc.append(pending_keys[i + r])
            pbar.update(1)
        sizer.after_batch_success()
        i += sub.shape[0]


def extract_features_for_keys(args, model, encoder_kind: str, crop: int, embed_dim: int,
                              keys, output_dir: Path, *, label: str = "feat",
                              pool_tokens: int = None):
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
    ckpt_file = output_dir / f".probe_{label}_ckpt.npz"

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
    if encoder_kind == "vjepa":
        # iter13: probe-specific initial BS (yaml key inference_vjepa_probe_initial_bs).
        # Falls back to the m05 64-frame default (8) if probe key is absent.
        initial_bs = _PCFG["gpu"].get(
            "inference_vjepa_probe_initial_bs",
            _PCFG["gpu"]["inference_vjepa_initial_bs"],
        )
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

    pending_tensors, pending_keys = [], []
    n_since_ckpt = 0
    t0 = time.time()
    try:
        while True:
            try:
                item = clip_q.get(timeout=300)
            except queue.Empty:
                print("  WARN: clip queue timeout (5 min) — flushing pending and exiting loop")
                break
            if item is None:
                break
            clip_key, mp4_bytes = item
            t = decode_to_tensor(mp4_bytes, tmp_dir, clip_key, args.num_frames, crop)
            if t is None:
                print(f"  SKIP (decode fail): {clip_key}")
                continue
            pending_tensors.append(t)
            pending_keys.append(clip_key)
            if len(pending_tensors) >= 32:
                _flush_batch(pending_tensors, pending_keys,
                             model, encoder_kind, args.num_frames,
                             sizer, feats_acc, keys_acc, pbar,
                             pool_tokens=pool_tokens)
                n_since_ckpt += len(pending_keys)
                pending_tensors, pending_keys = [], []
                if n_since_ckpt >= CHECKPOINT_EVERY:
                    _save_features_ckpt(ckpt_file, feats_acc, keys_acc)
                    n_since_ckpt = 0
        if pending_tensors:
            _flush_batch(pending_tensors, pending_keys,
                         model, encoder_kind, args.num_frames,
                         sizer, feats_acc, keys_acc, pbar,
                         pool_tokens=pool_tokens)
    finally:
        tar_stop.set()
        pbar.close()

    elapsed = time.time() - t0
    print(f"  extract_features_for_keys({label}): {len(keys_acc)}/{len(keys)} clips in {elapsed:.0f}s")

    # iter13: float16 storage. Matches _save_features_ckpt. Downstream consumers
    # (probe head training in probe_action.py:380, motion-cos in
    # probe_motion_cos.py:81-87) all upcast to fp32 explicitly via .float() or
    # .astype(np.float32) — fp16 storage is information-equivalent (V-JEPA
    # forward is bf16 internally; .float() in forward_vjepa already loses bf16
    # precision; round-tripping through fp16 storage adds zero new loss).
    feats = (np.stack(feats_acc).astype(np.float16)
             if feats_acc else np.empty((0, 0, embed_dim), dtype=np.float16))
    return feats, keys_acc


__all__ = [
    "ENCODERS",
    "load_vjepa_2_1_frozen", "load_dinov2_frozen",
    "decode_to_tensor", "resize_and_normalize",
    "forward_vjepa", "forward_dinov2",
    "extract_features_for_keys",
    "save_array_checkpoint",   # re-export so callers don't both-import from utils.checkpoint
]
