"""HF-transformers V-JEPA 2 frozen loader (kind=hf_vjepa2). GPU-only.

For V-JEPA-2 models published ONLY as HF transformers VJEPA2Model (no native .pt) — e.g. the
SSv2-supervised-FT encoder facebook/vjepa2-vitg-fpc64-384-ssv2 (the supervised-action baseline).
VJEPA2Model.forward(pixel_values_videos, context_mask, target_mask, skip_predictor, ...) →
last_hidden_state; skip_predictor=True returns the ENCODER token sequence only, mirroring
frozen_features.forward_vjepa's output contract (B, n_tokens, D) fp32 cpu.

VERIFY-on-GPU: (1) pixel_values_videos shape — HF expects (B, T, C, H, W); frozen_features
feeds exactly that. (2) frame count — pretrained fpc64; T=16 eval relies on tubelet flexibility
(same assumption as the native loader). If either errors, the GPU smoke shows it immediately.
Gold: https://huggingface.co/docs/transformers/model_doc/vjepa2
"""
import torch

from utils.frozen_features import ENCODERS


def load_hf_vjepa2_frozen(enc_name: str):
    """Load an HF VJEPA2Model encoder by registry name. fp16, eval.
    Returns (model, crop, embed_dim). Mirrors load_dinov2_frozen's (model, crop, dim) shape."""
    from transformers import VJEPA2Model
    enc = ENCODERS[enc_name]
    print(f"Loading HF V-JEPA2 frozen ({enc['model_id']}, crop={enc['crop']}) ...")
    model = VJEPA2Model.from_pretrained(
        enc["model_id"], dtype=torch.float16, device_map="cuda",
    ).eval()
    return model, enc["crop"], enc["embed_dim"]


@torch.no_grad()
def forward_hf_vjepa2(model, batch: torch.Tensor, num_frames: int) -> torch.Tensor:
    """batch: (B, T, C, H, W) pre-normalized → (B, n_tokens, D) fp32 cpu (encoder-only)."""
    B, T, C, H, W = batch.shape
    assert T == num_frames, f"hf_vjepa2 frame count mismatch: {T} vs {num_frames}"
    vid = batch.to("cuda", dtype=torch.float16)
    out = model(pixel_values_videos=vid, skip_predictor=True)
    return out.last_hidden_state.float().cpu()          # (B, n_patches, D)


__all__ = ["load_hf_vjepa2_frozen", "forward_hf_vjepa2"]
