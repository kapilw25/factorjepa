#!/usr/bin/env python3
"""
load_factorjepa.py — self-contained loader for a FactorJEPA (V-JEPA 2.1 ViT-G, 2B) HF repo.

This repo ships EVERYTHING needed to load and run the model — no external code, no private
package. The encoder/predictor architecture is the vendored Meta V-JEPA 2 source
(`vjepa2_src/`, MIT, github.com/facebookresearch/vjepa2 @ 204698b); the weights are this
repo's `student_encoder.pt` (encoder) and `m09c_ckpt_best.pt` (encoder + predictor).

The weights are a state_dict for the NATIVE V-JEPA 2.1 ViT — NOT the Hugging Face
`transformers.VJEPA2Model` (different key names + no 2.1 deep-supervision head). Load with the
constructors below, not `AutoModel.from_pretrained`. No `xformers` is required — attention runs
on `torch.nn.functional.scaled_dot_product_attention` (`use_sdpa=True`).

USAGE
  # 1. encoder only (feature extraction) — runs a random-input smoke test, no video needed:
  python load_factorjepa.py --encoder student_encoder.pt

  # 2. encoder + predictor (for next-frame / JEPA prediction work):
  python load_factorjepa.py --encoder student_encoder.pt --predictor m09c_ckpt_best.pt

  # 3. in your own code:
  from load_factorjepa import load_encoder, preprocess_frames, extract_features
  enc = load_encoder("student_encoder.pt", device="cuda")          # bf16 on cuda, fp32 on cpu
  clip = preprocess_frames(frames_uint8)[None]                     # frames: (T,H,W,3) uint8 -> (1,16,3,384,384)
  feats = extract_features(enc, clip)                              # (1, 4608, 1664) token features
  pooled = feats.mean(1)                                           # (1, 1664) clip vector

Tested against the original eval pipeline: same constructor kwargs, same state_dict keys,
same preprocessing (verified file:line against the training/eval code that produced the weights).
"""
import argparse
import os
import sys

import torch

# ── make the vendored Meta V-JEPA 2 architecture importable ───────────────────────────────
# vjepa2_src/ contains Meta's `src/` and `app/` packages verbatim. We prepend it to sys.path so
# `from app.vjepa_2_1...` / `from src...` resolve to the vendored copies. Run this script from the
# downloaded repo dir. (If you import it INTO a project that already has top-level `src`/`app`
# packages, import load_factorjepa FIRST, or rename those — `src`/`app` are Meta's package roots.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vjepa2_src")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers  # noqa: E402
from app.vjepa_2_1.models.predictor import vit_predictor                   # noqa: E402

# ── fixed architecture config (V-JEPA 2.1 ViT-G 2B; depth/heads/dim are baked into the ctor) ──
CROP, PATCH, TUBELET, NUM_FRAMES = 384, 16, 2, 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pick(device, dtype):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return device, dtype


def _strip(sd):
    """Drop the `module.` / `backbone.` prefixes the training wrapper adds (matches the eval loader)."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def load_encoder(ckpt_path, device=None, dtype=None, min_load_frac=0.90):
    """Build the 2B ViT-G encoder and load `student_encoder.pt`. Returns an eval()-mode model.

    student_encoder.pt is `{"student_state_dict": <state_dict>, "model_id": ..., "type": ...}`
    (NOT a bare state_dict). We also accept Meta's raw `target_encoder` key and a bare dict.
    """
    device, dtype = _pick(device, dtype)
    enc = vit_gigantic_xformers(
        img_size=(CROP, CROP), patch_size=PATCH, num_frames=NUM_FRAMES, tubelet_size=TUBELET,
        use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False, use_rope=True,
        use_activation_checkpointing=False,   # inference; n_output_distillation defaults to 4
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("student_state_dict") or ck.get("encoder") or ck.get("target_encoder") or ck
    sd = _strip(sd)
    msg = enc.load_state_dict(sd, strict=False)
    loaded = len(enc.state_dict()) - len(msg.missing_keys)
    total = len(enc.state_dict())
    if loaded < min_load_frac * total:
        raise RuntimeError(
            f"FATAL: only {loaded}/{total} encoder params loaded ({loaded / total:.0%}) — key mismatch. "
            f"missing(5)={msg.missing_keys[:5]} unexpected(5)={msg.unexpected_keys[:5]}")
    print(f"  encoder: loaded {loaded}/{total} params ({loaded / total:.1%}) · "
          f"{sum(p.numel() for p in enc.parameters()) / 1e9:.2f}B · device={device} dtype={dtype}")
    return enc.to(device, dtype).eval()


def load_predictor(ckpt_path, device=None, dtype=None, min_load_frac=0.50):
    """Build the 2.1 predictor and load it from `m09c_ckpt_best.pt` (key `predictor`).

    Needed for next-frame / JEPA prediction (encode context -> predict masked-future latents).
    NOT needed for plain feature extraction (use load_encoder for that).
    """
    device, dtype = _pick(device, dtype)
    pred = vit_predictor(
        img_size=(CROP, CROP), patch_size=PATCH, num_frames=NUM_FRAMES, tubelet_size=TUBELET,
        embed_dim=1664, predictor_embed_dim=384, depth=24, num_heads=12,
        use_mask_tokens=True, num_mask_tokens=2, zero_init_mask_tokens=True,
        use_rope=True, uniform_power=False, use_sdpa=True, use_silu=False, wide_silu=True,
        use_activation_checkpointing=False, return_all_tokens=True,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("predictor") or ck.get("predictor_state_dict")
    if sd is None:
        raise RuntimeError(f"FATAL: no `predictor` key in {ckpt_path}. top-level keys: {list(ck.keys())[:8]}")
    sd = _strip(sd)
    msg = pred.load_state_dict(sd, strict=False)
    loaded = len(pred.state_dict()) - len(msg.missing_keys)
    total = len(pred.state_dict())
    if loaded < min_load_frac * total:
        raise RuntimeError(f"FATAL: predictor only {loaded}/{total} loaded — random init = garbage.")
    print(f"  predictor: loaded {loaded}/{total} params ({loaded / total:.1%}) · device={device} dtype={dtype}")
    return pred.to(device, dtype).eval()


def preprocess_frames(frames_uint8):
    """(T, H, W, 3) uint8 -> (NUM_FRAMES, 3, 384, 384) float32, the EVAL recipe:
    uniform-sample NUM_FRAMES frames -> resize shorter side to 384 -> center-crop 384 ->
    /255 -> ImageNet normalize. (Decode your own video to (T,H,W,3) uint8 first.)"""
    import numpy as np
    import torch.nn.functional as F
    f = np.asarray(frames_uint8)
    idx = np.linspace(0, len(f) - 1, NUM_FRAMES).astype(int)
    x = torch.from_numpy(f[idx]).float().permute(0, 3, 1, 2) / 255.0   # (T,3,H,W)
    h, w = x.shape[-2:]
    s = CROP / min(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
    h, w = x.shape[-2:]
    top, left = (h - CROP) // 2, (w - CROP) // 2
    x = x[..., top:top + CROP, left:left + CROP]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_features(encoder, clip, hierarchical=False):
    """clip: (B, T, 3, 384, 384) float -> encoder features.
    hierarchical=False -> (B, 4608, 1664) final-layer tokens (mean over dim=1 for a clip vector).
    hierarchical=True  -> (B, 4608, 6656) the 4-layer deep-supervision concat."""
    p = next(encoder.parameters())
    pixel = clip.to(p.device, p.dtype).permute(0, 2, 1, 3, 4).contiguous()   # (B,3,T,H,W)
    if hierarchical:
        encoder.return_hierarchical = True
    try:
        out = encoder(pixel)
    finally:
        if hierarchical:
            encoder.return_hierarchical = False
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.float().cpu()


def _main():
    ap = argparse.ArgumentParser(description="Load + smoke-test a FactorJEPA V-JEPA 2.1 ViT-G checkpoint.")
    ap.add_argument("--encoder", required=True, help="path to student_encoder.pt (or a repo_id to auto-download)")
    ap.add_argument("--predictor", default=None, help="path to m09c_ckpt_best.pt (optional; for prediction)")
    ap.add_argument("--repo-id", default=None, help="HF repo to hf_hub_download the files from (public, no token)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    enc_path, pred_path = args.encoder, args.predictor
    if args.repo_id:
        from huggingface_hub import hf_hub_download
        enc_path = hf_hub_download(args.repo_id, args.encoder)
        if args.predictor:
            pred_path = hf_hub_download(args.repo_id, args.predictor)

    print("Building V-JEPA 2.1 ViT-G (2B) encoder + loading weights ...")
    enc = load_encoder(enc_path, device=args.device)
    # smoke test on RANDOM input (no video file needed) — proves the forward pass runs end-to-end
    dummy = torch.randn(1, NUM_FRAMES, 3, CROP, CROP)
    feats = extract_features(enc, dummy)
    print(f"  forward OK · token features {tuple(feats.shape)} · clip vector {tuple(feats.mean(1).shape)}")
    hier = extract_features(enc, dummy, hierarchical=True)
    print(f"  hierarchical (deep-sup concat) {tuple(hier.shape)}")
    if pred_path:
        print("Building predictor + loading weights ...")
        load_predictor(pred_path, device=args.device)
    print("\n✅ self-contained load OK — no external code needed.")


if __name__ == "__main__":
    _main()
