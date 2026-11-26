"""I-JEPA / LeJEPA frozen IMAGE-encoder loader for the eval registry (kind=ijepa). GPU-only.

Image JEPAs have no video predictor → FROZEN-eval only (Stages 2/3/3.5/5/6 + m12 action/
motion_cos/taxonomy). They SKIP Stage 8 future_mse + the m12e predictor-temporal suite
(no predictor — CLAUDE.md TRUE-IMPOSSIBILITY carve-out; callers guard `if kind != "ijepa"`).

Per-frame encode → token sequence over time, mirroring frozen_features.load_dinov2_frozen +
forward_dinov2 (the frame batch is pre-normalized by frozen_features.resize_and_normalize, so
forward feeds the tensor straight to the HF model — no re-processing).

Registry rows (added in plan_model.md D4, DEFERRED post-eval) e.g.:
  ijepa_vitH14:  {kind: ijepa, model_id: facebook/ijepa_vith14_1k,  crop: 224, embed_dim: 1280}
  ijepa_vitG16:  {kind: ijepa, model_id: facebook/ijepa_vitg16_22k, crop: 224, embed_dim: 1408}
  lejepa_vitH14: {kind: ijepa, model_id: <HF asset>,                crop: 224, embed_dim: 1280}

GPU-VALIDATION REQUIRED post-eval — written + py_compile/ruff-clean; not yet import/GPU-smoked
(host RAM ~97% from the live eval → no torch import now). Gold: I-JEPA arXiv 2301.08243.
"""
import torch

from utils.frozen_features import ENCODERS


def load_ijepa_frozen(enc_name: str):
    """Load an I-JEPA/LeJEPA image encoder by registry name. fp16, eval.
    Returns (model, processor, crop, embed_dim). Mirrors load_dinov2_frozen.
    """
    from transformers import AutoImageProcessor, AutoModel
    enc = ENCODERS[enc_name]
    print(f"Loading I-JEPA frozen ({enc['model_id']}, crop={enc['crop']}) ...")
    processor = AutoImageProcessor.from_pretrained(enc["model_id"])
    # Default attention (no FA2): I-JEPA's HF impl may not support flash_attention_2 — keep safe.
    model = AutoModel.from_pretrained(
        enc["model_id"], dtype=torch.float16, device_map="cuda",
    ).eval()
    return model, processor, enc["crop"], enc["embed_dim"]


@torch.no_grad()
def forward_ijepa(model, batch: torch.Tensor, num_frames: int) -> torch.Tensor:
    """I-JEPA: process T pre-normalized frames per clip independently, concat token sequences
    over time → (B, T*n_patch, D) fp32 on CPU. Exact mirror of forward_dinov2 (I-JEPA returns
    patch tokens via last_hidden_state, no CLS)."""
    B, T, C, H, W = batch.shape
    assert T == num_frames, f"I-JEPA batch frame count mismatch: {T} vs {num_frames}"
    flat = batch.view(B * T, C, H, W).to("cuda", dtype=torch.float16)
    last_hidden = model(pixel_values=flat).last_hidden_state          # (B*T, n_patch, D)
    _, n_tokens_per_frame, D = last_hidden.shape
    return last_hidden.view(B, T * n_tokens_per_frame, D).float().cpu()


__all__ = ["load_ijepa_frozen", "forward_ijepa"]
