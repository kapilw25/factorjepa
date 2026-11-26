"""LeJEPA frozen IMAGE-encoder loader for the eval registry (kind=lejepa). GPU-only.

LeJEPA (Balestriero & LeCun, arXiv 2511.08544) trains a DINOv3-style ViT with SIGReg — no
video predictor → FROZEN-eval only (Stages 2/3/3.5/5/6 + m12 action/motion_cos/taxonomy);
SKIPs Stage 8 future_mse + the m12e predictor suite (run_eval gates `kind == vjepa`).

Unlike kind=ijepa/dinov2 (HF AutoModel.from_pretrained), the published LeJEPA-L weight is a RAW
state_dict (keys `encoder.model.*` over a DINOv3-style backbone: storage/register tokens, RoPE,
layerscale, fused qkv+bias_mask) — no HF model card. It loads via timm's DINOv3 port
(`vit_large_patch16_dinov3_qkvb`) + `timm.models.eva.checkpoint_filter_fn`, the timm-verified
Meta→timm key remap (storage_tokens→reg_token, ls{1,2}.gamma→gamma_{1,2}, qkv.bias→q_bias/v_bias,
RoPE recomputed as a non-persistent buffer). Verified: 0 missing / 0 unexpected (strict load),
forward_features → (B, 201, 1024) = 1 cls + 4 reg + 196 patches, all finite.

Per-frame encode → token sequence over time, mirroring frozen_features.forward_dinov2 (the frame
batch is pre-normalized by frozen_features.resize_and_normalize with ImageNet stats — same as the
other image baselines, so the cross-baseline comparison is apples-to-apples).
Gold: LeJEPA arXiv 2511.08544 · DINOv3 arXiv 2508.10104 · timm.models.eva.
"""
import sys

import torch

from utils.frozen_features import ENCODERS


def load_lejepa_frozen(enc_name: str):
    """Load a LeJEPA image encoder by registry name. fp16, eval.
    Reads arch (timm name) / crop / embed_dim / ckpt (local .pt) from ENCODERS[enc_name].
    Returns (model, crop, embed_dim). Mirrors load_dinov2_frozen's (model, crop, dim) shape."""
    import timm
    from timm.models.eva import checkpoint_filter_fn
    enc = ENCODERS[enc_name]
    ckpt_path = enc["ckpt"]
    print(f"Loading LeJEPA frozen ({enc['arch']} timm, ckpt={ckpt_path}, crop={enc['crop']}) ...")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k.replace("encoder.model.", "", 1): v for k, v in raw.items() if k.startswith("encoder.model.")}
    if not sd:
        sys.exit(f"FATAL: {enc_name} ckpt has no 'encoder.model.*' keys (format changed?): {ckpt_path}")
    model = timm.create_model(enc["arch"], pretrained=False, num_classes=0, img_size=enc["crop"])
    model.load_state_dict(checkpoint_filter_fn(sd, model), strict=True)   # 0 miss/0 unexpected (verified)
    model = model.eval().to("cuda", dtype=torch.float16)
    return model, enc["crop"], enc["embed_dim"]


@torch.no_grad()
def forward_lejepa(model, batch: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Process T pre-normalized frames per clip independently → concat token sequences over time
    → (B, T*n_tokens, D) fp32 on CPU. Exact mirror of forward_dinov2 (timm forward_features returns
    the post-norm token sequence incl. cls+reg+patches; _pool_tokens block-pools it downstream)."""
    B, T, C, H, W = batch.shape
    assert T == num_frames, f"LeJEPA batch frame count mismatch: {T} vs {num_frames}"
    flat = batch.view(B * T, C, H, W).to("cuda", dtype=torch.float16)
    tokens = model.forward_features(flat)                 # (B*T, n_tokens, D)
    _, n_tokens_per_frame, D = tokens.shape
    return tokens.view(B, T * n_tokens_per_frame, D).float().cpu()


__all__ = ["load_lejepa_frozen", "forward_lejepa"]
