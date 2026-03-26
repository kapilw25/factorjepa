# Open Questions Q1-Q5: Resolved (from vjepa2 source)

> Investigated from `deps/vjepa2` (shallow clone, commit latest on main).
> All findings verified by reading actual source code, not documentation.

---

## Q1: Actual Import Paths

### plan_code_dev.md assumed:
```python
from src.models.vision_transformer import vit_giant
from src.models.predictor import VisionTransformerPredictor
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
```

### Actual (verified from source):

The repo uses **intra-package `src.*` imports** (e.g., `from src.models.utils.modules import Block`). These work when running from the repo root, NOT when installed as a package.

**Problem**: `setup.py` doesn't define `packages=` or `package_dir=` — it's a minimal stub. `pip install -e deps/vjepa2` will NOT make `from src.models...` importable as a proper package.

**Solution**: Add `deps/vjepa2` to `sys.path` at runtime:

```python
import sys
sys.path.insert(0, "deps/vjepa2")

# Then these work exactly as the repo uses them internally:
from src.models.vision_transformer import vit_giant          # ✅ Encoder factory
from src.models.predictor import VisionTransformerPredictor   # ✅ Predictor class
from src.models.predictor import vit_predictor                # ✅ Predictor factory (preferred)
from src.masks.multiseq_multiblock3d import MaskCollator      # ✅ Block mask sampler
from src.masks.utils import apply_masks                       # ✅ Mask application
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper  # ✅ Multi-FPC wrappers
```

### CRITICAL: Training uses wrapper classes

The actual training code (`app/vjepa/utils.py:init_video_model`) wraps models:

```python
encoder = video_vit.__dict__[model_name](...)   # Raw VisionTransformer
encoder = MultiSeqWrapper(encoder)               # Wraps with .backbone attribute

predictor = vit_pred.__dict__["vit_predictor"](...)
predictor = PredictorMultiSeqWrapper(predictor)  # Wraps with .backbone attribute
```

**Impact on m09**: We have two options:
1. **Use wrappers** (match Meta's training code exactly) — checkpoint keys will be `backbone.blocks.0...`
2. **Skip wrappers** (simpler, since we have only 1 FPC) — checkpoint keys will be `blocks.0...`

### Option 1 vs Option 2: Full Comparison

#### How the list-of-lists data structure works (Option 1 complexity)

MaskCollator groups samples by FPC and returns nested lists:
```
sample = [
    (collated_batch_fpc0, [masks_enc_mg0, masks_enc_mg1], [masks_pred_mg0, masks_pred_mg1]),
    (collated_batch_fpc1, [masks_enc_mg0, masks_enc_mg1], [masks_pred_mg0, masks_pred_mg1]),
]
```
Outer = FPC groups. Inner = mask generators (8-small + 2-large = 2 generators).

MultiSeqWrapper iterates both levels:
```python
outs = [[] for _ in x]
for i, (xi, mi) in enumerate(zip(x, masks)):     # outer: FPC groups
    for mij in mi:                                 # inner: mask generators
        outs[i] += [self.backbone(xi, masks=mij)]  # actual ViT forward call
return outs  # list[list[Tensor]]
```

PredictorMultiSeqWrapper does the same, additionally passing `mask_index=i` (selects learned mask token).

The loss function then double-iterates:
```python
for zi, hi in zip(z, h):           # outer: FPC groups
    for zij, hij in zip(zi, hi):   # inner: mask generators
        loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
```

With a **single FPC** (our case), the outer loop has exactly 1 iteration — the wrappers become pure overhead.

#### Comparison table

| Criterion | Option 1: Use wrappers | Option 2: Skip wrappers |
|-----------|----------------------|------------------------|
| **Checkpoint compatibility** | Keys match Meta's training checkpoints exactly (`backbone.blocks.0...`) | Must strip `module.backbone.` prefix on load (one-liner via `_clean_backbone_key`) |
| **Code complexity** | list-of-lists everywhere: data loading, forward, loss, logging | Standard single-tensor pipeline |
| **Data loading** | Must match MaskCollator's collate_fn API (expects DataLoader samples, not bare tensors) | Use `_MaskGenerator(batch_size)` directly — returns `(masks_enc, masks_pred)` tensors |
| **Forward pass** | `encoder(clips_list, masks_list_of_lists)` → `list[list[Tensor]]` | `encoder(clip_tensor, masks_list)` → `Tensor` |
| **Predictor** | `predictor(z_nested, masks_enc_nested, masks_pred_nested)` → `list[list[Tensor]]` | `predictor(z, masks_enc, masks_pred, mask_index=i)` → `Tensor` |
| **Loss computation** | Double-nested zip loop | Single loop over mask generators |
| **Gradient flow** | Identical (wrappers add zero parameters) | Identical |
| **Debugging** | Hard to inspect intermediate shapes in nested lists | Standard tensor shapes, easy to debug |
| **Meta precedent** | Used by `app/vjepa/train.py` (from-scratch pretraining) | Used by `app/vjepa_droid/train.py` (continual pretraining), `notebooks/vjepa2_demo.py`, ALL evaluation code |
| **HF export** | Must strip `backbone.` prefix to save HF-compatible weights | Keys already HF-compatible |
| **Multi-FPC future** | Ready if we ever use [16, 64] frame clips | Would need refactoring |
| **Risk of bugs** | Higher — nesting errors are silent (wrong dim, wrong index) | Lower — standard PyTorch patterns |

#### Evidence from Meta's own codebase

**Meta skips wrappers in 3 out of 4 use cases:**

1. **`app/vjepa_droid/`** (continual pretraining for robotics): Raw ViT, no wrappers. Successfully trains on custom data. Strips `backbone.` prefix when loading pretrained weights.

2. **`notebooks/vjepa2_demo.py`**: Raw ViT, no wrappers. Strips `module.` + `backbone.` prefix.

3. **`evals/video_classification_frozen/`**: Raw ViT, no wrappers. Strips both prefixes.

4. **`app/vjepa/train.py`**: Uses wrappers — but only because from-scratch pretraining uses multiple FPCs (e.g., [16, 64] frame clips) and needs the multi-sequence batching.

#### GitHub issue confirmation

Issue #124: A user got `mask_index` mismatch with fewer mask generators than pretrained `num_mask_tokens=10`. Maintainer confirmed: "you can use all the tokens enabled, during training it uses only `len(masks)`." The `mask_index % num_mask_tokens` modular indexing handles mismatches.

Issue #140: A user successfully started V-JEPA 2.1 continual pretraining on Sentinel-2 satellite data (single H100). No mention of wrappers — likely used raw models.

### Decision: Option 2 (Skip wrappers) — CONFIRMED

**Recommendation**: Skip wrappers for m09 (we only use 1 FPC = 16 frames). Use raw `VisionTransformer` + `VisionTransformerPredictor`. This follows Meta's own continual pretraining precedent (`app/vjepa_droid/`), avoids the list-of-lists complexity, produces HF-compatible keys, and has identical gradient flow. When loading Meta's pretrained weights, strip `module.backbone.` prefix (one-liner via `_clean_backbone_key`).

**Loading pattern** (from Meta's own demo):
```python
ckpt = torch.load("vitg-384.pt", map_location="cpu")
state_dict = ckpt["target_encoder"]
state_dict = {k.replace("module.", "").replace("backbone.", ""): v
              for k, v in state_dict.items()}
encoder.load_state_dict(state_dict, strict=False)  # strict=False for RoPE/pos_embed mismatch
```

### Model factory: `vit_giant` vs `vit_giant_xformers`

The training config uses `model_name: vit_giant_xformers` (22 heads), but our Ch9 inference model (`facebook/vjepa2-vitg-fpc64-384`) maps to `vit_giant_xformers` in the hub:

```python
# From backbones.py ARCH_NAME_MAP:
"vit_giant_384": ("vit_giant_xformers", "vitg-384")
```

The difference: `vit_giant` = 16 heads, `vit_giant_xformers` = 22 heads. **Use `vit_giant_xformers` to match the pretrained checkpoint.**

---

## Q2: torch.hub Entry Points & Checkpoint URLs

### Hub functions (from `hubconf.py`):

```python
torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384", pretrained=True)
# Returns: (encoder, predictor) tuple!  ← NOT just encoder
```

Available entry points:
| Function | Model | Resolution | Returns |
|----------|-------|-----------|---------|
| `vjepa2_vit_large` | ViT-L | 256 | (encoder, predictor) |
| `vjepa2_vit_huge` | ViT-H | 256 | (encoder, predictor) |
| `vjepa2_vit_giant` | ViT-g | 256 | (encoder, predictor) |
| `vjepa2_vit_giant_384` | ViT-g | **384** | (encoder, predictor) |
| `vjepa2_ac_vit_giant` | ViT-g + AC | 256 | (encoder, predictor) |
| `vjepa2_1_vit_base_384` | ViT-B 2.1 | 384 | (encoder, predictor) |
| `vjepa2_1_vit_large_384` | ViT-L 2.1 | 384 | (encoder, predictor) |
| `vjepa2_1_vit_giant_384` | ViT-g 2.1 | 384 | (encoder, predictor) |
| `vjepa2_1_vit_gigantic_384` | ViT-G 2.1 | 384 | (encoder, predictor) |

### Checkpoint URL format:

```python
# From backbones.py:
VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"  # (commented out in code, see below)
# ACTUAL current value (for testing): "http://localhost:8300"
```

**BLOCKER**: The URL is currently set to `http://localhost:8300` (local test server). The production URL `https://dl.fbaipublicfiles.com/vjepa2` is **commented out**. This means `torch.hub.load(..., pretrained=True)` will FAIL out of the box.

**Workaround options**:
1. Patch `backbones.py` to uncomment the production URL
2. Download checkpoints manually from `https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt`
3. Use HF model (`facebook/vjepa2-vitg-fpc64-384`) for weight loading, then convert keys

**For our model** (`vjepa2_vit_giant_384`):
- Checkpoint file: `vitg-384.pt`
- Full URL: `https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt`
- `checkpoint_key`: `"target_encoder"` (EMA teacher weights used as starting point)

### Hub model construction details (for `vjepa2_vit_giant_384`):

```python
# Encoder kwargs (from _make_vjepa2_model):
vit_encoder_kwargs = dict(
    patch_size=16,
    img_size=(384, 384),
    num_frames=64,         # NOTE: 64 frames default, not 16
    tubelet_size=2,
    use_sdpa=True,
    use_SiLU=False,
    wide_SiLU=True,
    uniform_power=False,
    use_rope=True,         # RoPE, NOT sincos positional embeddings
)
arch_name = "vit_giant_xformers"  # 22 heads, not 16

# Predictor kwargs:
vit_predictor_kwargs = dict(
    img_size=(384, 384),
    patch_size=16,
    use_mask_tokens=True,
    embed_dim=1408,            # encoder.embed_dim
    predictor_embed_dim=384,
    num_frames=64,
    tubelet_size=2,
    depth=12,
    num_heads=12,
    num_mask_tokens=10,        # 10 mask tokens (one per mask generator: 8 small + 2 large)
    use_rope=True,
    uniform_power=False,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
)
```

---

## Q3: Predictor Weights in Public Checkpoint

**YES — predictor weights are included.**

### Checkpoint structure (from `backbones.py:_make_vjepa2_model` and `app/vjepa/train.py:save_checkpoint`):

```python
# Training checkpoint keys:
save_dict = {
    "encoder": encoder.state_dict(),         # Student (DDP wrapped)
    "predictor": predictor.state_dict(),     # Predictor (DDP wrapped)
    "target_encoder": target_encoder.state_dict(),  # Teacher/EMA
    "opt": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": epoch,
    "loss": loss_meter.avg,
    "batch_size": batch_size,
    "world_size": world_size,
    "lr": lr,
}
```

### Hub loading code confirms both are loaded:

```python
# From _make_vjepa2_model:
encoder_state_dict = _clean_backbone_key(state_dict["target_encoder"])  # Uses EMA teacher
encoder.load_state_dict(encoder_state_dict, strict=False)

predictor_state_dict = _clean_backbone_key(state_dict["predictor"])
predictor.load_state_dict(predictor_state_dict, strict=False)  # ✅ Predictor loaded!
```

**Key detail**: The hub loads `target_encoder` (EMA/teacher) into the encoder, NOT `encoder` (student). This is because the EMA teacher generally has better quality weights than the student for inference.

**For Ch10 continual pretraining**: We should load `target_encoder` weights into BOTH student AND teacher (via deepcopy), then train the student. The predictor can be loaded from `predictor` key or initialized fresh.

### `strict=False` reasoning:

Both loads use `strict=False` because the models use **RoPE** (no learned positional embeddings), but the checkpoint may contain `pos_embed` weights from sincos initialization. The `strict=False` silently ignores the mismatch.

---

## Q4: MaskCollator API

### Constructor: `MaskCollator.__init__`

```python
class MaskCollator(object):
    def __init__(
        self,
        cfgs_mask,         # list[dict] — mask configs (from YAML "mask" key)
        dataset_fpcs,      # list[int] — frames-per-clip values, e.g., [16] or [16, 64]
        crop_size=(224, 224),   # tuple or int
        patch_size=(16, 16),    # tuple or int
        tubelet_size=2,         # temporal patch size
    ):
```

Each dict in `cfgs_mask` maps to a `_MaskGenerator` with keys:
```python
{
    "spatial_scale": [0.15, 0.15],     # spatial fraction of tokens per block
    "temporal_scale": [1.0, 1.0],      # temporal fraction (1.0 = full time span)
    "aspect_ratio": [0.75, 1.5],       # block aspect ratio range
    "num_blocks": 8,                   # blocks per sample
    "max_temporal_keep": 1.0,          # context temporal limit (optional)
    "max_keep": null,                  # max context tokens (optional)
    "full_complement": false,          # pred = complement of enc (optional)
    "pred_full_complement": false,     # enc = complement of pred (optional)
    "inv_block": false,                # predict context from block (optional)
}
```

### `__call__` — THIS IS NOT WHAT plan_code_dev.md ASSUMED

**plan_code_dev.md assumed**:
```python
collated_masks = mask_collator(batch_size)     # ← WRONG
masks_enc, masks_pred = collated_masks         # ← WRONG
```

**Actual signature**:
```python
def __call__(self, batch):
    # batch = list of (buffer, label, clip_indices) samples from DataLoader
    # NOT a batch_size integer!
```

**MaskCollator is a DataLoader COLLATE FUNCTION**, not a standalone mask generator.

It works as:
1. Receives a list of samples from DataLoader
2. Groups samples by FPC (frames-per-clip)
3. For each FPC group, calls `_MaskGenerator(batch_size)` per mask config
4. Returns: `list[(collated_batch, collated_masks_enc, collated_masks_pred)]`

### `_MaskGenerator.__call__` — THIS is the actual mask generator

```python
def __call__(self, batch_size) -> tuple[Tensor, Tensor]:
    # Returns: (masks_enc, masks_pred)
    # masks_enc: Tensor of shape [batch_size, N_visible] — indices of visible tokens
    # masks_pred: Tensor of shape [batch_size, N_masked] — indices of masked tokens
```

### For m09: Use `_MaskGenerator` directly (not MaskCollator)

Since we bypass their DataLoader pipeline (we use WebDataset), we should call `_MaskGenerator` directly:

```python
from src.masks.multiseq_multiblock3d import _MaskGenerator

# Create one generator per mask config
mask_generators = []
for m_cfg in cfg["mask"]:
    mg = _MaskGenerator(
        crop_size=(384, 384),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=m_cfg["spatial_scale"],
        temporal_pred_mask_scale=m_cfg["temporal_scale"],
        aspect_ratio=m_cfg["aspect_ratio"],
        npred=m_cfg["num_blocks"],
    )
    mask_generators.append(mg)

# Per batch: generate masks from each generator, combine
all_masks_enc, all_masks_pred = [], []
for mg in mask_generators:
    masks_enc, masks_pred = mg(batch_size)  # Each: [B, N_tokens]
    all_masks_enc.append(masks_enc)
    all_masks_pred.append(masks_pred)
```

### Training code flow (from `app/vjepa/train.py:train_step`):

```python
# clips = list of tensors (one per FPC)
# masks_enc = list[list[Tensor]] — outer: FPC, inner: mask configs
# masks_pred = list[list[Tensor]] — same structure

h = forward_target(clips)           # Teacher: h = [target_encoder(c) for c in clips]
                                    # Each h[i] is a tensor (no masks applied)
z = forward_context(clips)          # Student: encoder(c, masks_enc) → predictor(z, masks_enc, masks_pred)
loss = loss_fn(z, h)                # L1 on masked positions
```

---

## Q5: vjepa2 vs HF Weight Key Names

### vjepa2 key format (raw VisionTransformer):

```
patch_embed.proj.weight            # Conv3d(3, 1408, kernel_size=(2,16,16))
patch_embed.proj.bias
blocks.0.norm1.weight              # LayerNorm
blocks.0.norm1.bias
blocks.0.attn.qkv.weight          # Combined Q/K/V linear
blocks.0.attn.qkv.bias
blocks.0.attn.proj.weight          # Output projection
blocks.0.attn.proj.bias
blocks.0.norm2.weight
blocks.0.norm2.bias
blocks.0.mlp.fc1.weight
blocks.0.mlp.fc1.bias
blocks.0.mlp.fc2.weight
blocks.0.mlp.fc2.bias
...                                # blocks.1 through blocks.39
norm.weight                        # Final LayerNorm
norm.bias
```

### With MultiSeqWrapper (training checkpoints):
All keys get `backbone.` prefix → `backbone.blocks.0.attn.qkv.weight`

With DDP: `module.backbone.blocks.0.attn.qkv.weight`

### HF transformers format (from `facebook/vjepa2-vitg-fpc64-384`):

The HF model uses the same key names as raw VisionTransformer (no wrapper). This is because `_clean_backbone_key()` strips `module.` and `backbone.` prefixes.

**Key finding**: vjepa2 raw keys and HF keys are **already compatible** after `_clean_backbone_key()` cleanup. The checkpoint stores DDP+wrapper keys, but the hub loading function strips prefixes before loading.

### For m09 export (student → m05 re-evaluation):

```python
# After training, student is a raw VisionTransformer (no wrapper)
# Its state_dict keys: blocks.0.attn.qkv.weight, etc.
# These are already compatible with HF model loading.
# No conversion needed IF we skip wrappers in m09.
```

**If we use wrappers**: Need to strip `backbone.` prefix before saving for HF:
```python
student_dict = {k.replace("backbone.", ""): v for k, v in student.state_dict().items()}
```

---

## Summary of Corrections to plan_code_dev.md

| # | plan_code_dev.md says | Actual | Impact |
|---|----------------------|--------|--------|
| 1 | `from src.models.vision_transformer import vit_giant` | Use `vit_giant_xformers` (22 heads, matches checkpoint) | **Model architecture mismatch if wrong** |
| 2 | `torch.hub.load(...)` works out of box | URL is `localhost:8300` (test). Must patch or download manually | **Pretrained loading broken without fix** |
| 3 | Predictor weights may not exist | **YES**, `state_dict["predictor"]` exists in checkpoint | Good news — can warm-start predictor |
| 4 | `mask_collator(batch_size)` returns `(masks_enc, masks_pred)` | MaskCollator is a DataLoader collate fn. Use `_MaskGenerator(batch_size)` directly | **API mismatch — must use _MaskGenerator** |
| 5 | Need key conversion utility for HF export | Keys are already compatible after stripping `module.backbone.` | **Simpler than expected** |
| 6 | No mention of RoPE | Model uses `use_rope=True` (no learned pos_embed) | Must set `use_rope=True` in model construction |
| 7 | No mention of `use_mask_tokens` | Predictor uses `use_mask_tokens=True, num_mask_tokens=10` | Must configure predictor correctly |
| 8 | `num_frames=64` default | Hub creates model with 64 frames. We use 16 for training | Must override `num_frames=16` |
| 9 | EMA momentum is fixed | Config: `ema: [0.99925, 0.99925]` (same start/end = fixed) but code uses a **linear ramp** generator between ema[0] and ema[1] | With same start/end it IS fixed, confirming plan |
| 10 | Encoder+predictor same LR | Training code uses **single LR** for both encoder and predictor (same param groups) | Plan's `pred_lr_multiplier` is our addition, not Meta's |
